"""第三部分第二轮：F7/G0 强基线公平训练。

本脚本不修改也不覆盖既有 PatchTST、其它深度基线或 F7/G0 产物。它仅复用
F7 的 Keras 结构定义，并在统一强基线协议下从随机初始化开始单阶段训练：

* batch_size=256，epochs=80，validation_split=0.15；
* Adam(learning_rate=5e-4, clipnorm=1.0)，Huber(delta=1.0)；
* forecast/corrected candidate 双监督权重 1.0/0.5；
* EarlyStopping patience=10，ReduceLROnPlateau patience=4；
* checkpoint、早停和学习率调度均监控 val_forecast_power_loss；
* seed=2026，并请求 TensorFlow 确定性算子。

重要：此处明确禁止载入旧 B2/F7 权重。这样可与 ``wind_dl_model_train.py``
及 ``wind_dl_other_models_train.py`` 的单阶段随机初始化基线采用同一优化协议。
模型结构仍是 F7：Persistence + 轻量 causal residual + P+H+D 显式工况编码器
+ 逐样本逐 horizon G0 门控。

正式产物写入：
``wind_results/part3_new_module_supplement/02_strong_baseline_f7_g0_fair_training``。
子集/冒烟运行自动写入 ``partial_runs``，不会污染正式五场站 bundle。
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras

import wind_RegimeEncoder_PatchTST_feature_screen_train as feature_train
import wind_RegimeEncoder_PatchTST_train as regime_train
from wind_FeTS_PatchTST_train import (
    NonFiniteTrainingGuard,
    ensure_finite_training_history,
)
from wind_dl_model_train import (
    DATA_DIR,
    FORECAST_LEN,
    HISTORY_LEN,
    TARGET_COL,
    TIME_FREQ,
    make_window_dataset,
    set_global_seed,
)


MODEL_FAMILY = "part3_round2_f7_g0_strong_baseline"
MODEL_NAME = "part3_round2_f7_g0_strong_baseline"
VARIANT_ID = "sb_f7_g0_bs256"
SOURCE_STRUCTURE_VARIANT = "f7"
ARCHITECTURE_VERSION = "part3_round2_f7_g0_from_scratch_v1"
PROTOCOL_VERSION = "part3_round2_fair_training_bs256_v1"
ARTIFACT_SCHEMA_VERSION = 1
TRAINING_MODE = "single_stage_from_scratch_fair_protocol"

RESULT_ROOT = os.path.join(
    ".",
    "wind_results",
    "part3_new_module_supplement",
    "02_strong_baseline_f7_g0_fair_training",
)
TRAINING_SUMMARY_NAME = "part3_round2_strong_baseline_training_metrics.csv"
VALIDATION_SUMMARY_NAME = "part3_round2_strong_baseline_validation_summary.csv"
PROTOCOL_MANIFEST_NAME = "part3_round2_fair_training_protocol.json"
PROTOCOL_COMPARISON_NAME = "part3_round2_training_protocol_comparison.csv"
COMPLETION_MARKER_NAME = (
    "part3_round2_strong_baseline_training_bundle_complete.json"
)

RANDOM_SEED = 2026
BATCH_SIZE = 256
EPOCHS = 80
VALIDATION_SPLIT = 0.15
LEARNING_RATE = 5e-4
CANDIDATE_LOSS_WEIGHT = 0.5
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6
CLIPNORM = 1.0
EXPECTED_FARM_COUNT = 5
EXPECTED_TOTAL_PARAMS = 20_969
EXPECTED_TRAINABLE_PARAMS = 20_969

SUBDIR_NAMES = (
    "models",
    "weights",
    "preprocess",
    "history",
    "tensorboard",
    "tails",
    "validation_diagnostics",
    "visualizations",
    "manifests",
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path):
    path = os.path.abspath(os.fspath(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return {
        "path": path,
        "sha256": _file_sha256(path),
        "size_bytes": int(os.path.getsize(path)),
    }


def _atomic_json_dump(payload, path):
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)
    return os.path.abspath(path)


def _atomic_csv(frame, path):
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)
    return os.path.abspath(path)


def _atomic_joblib_dump(payload, path):
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    joblib.dump(payload, temporary)
    os.replace(temporary, path)
    return os.path.abspath(path)


def configure_reproducibility():
    set_global_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def result_dirs(result_root, create=True):
    paths = {"root": os.path.abspath(result_root)}
    paths.update(
        {
            name: os.path.join(paths["root"], name)
            for name in SUBDIR_NAMES
        }
    )
    if create:
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
    return paths


def artifact_paths(dirs, farm_id):
    stem = f"{MODEL_NAME}_farm_{farm_id}"
    return {
        "model_path": os.path.join(dirs["models"], f"{stem}.keras"),
        "best_weights_path": os.path.join(
            dirs["weights"], f"{stem}_best.weights.h5"
        ),
        "artifact_path": os.path.join(
            dirs["preprocess"], f"{stem}_preprocess.pkl"
        ),
        "history_path": os.path.join(dirs["history"], f"{stem}_history.csv"),
        "history_plot_path": os.path.join(
            dirs["history"], f"{stem}_history.png"
        ),
        "tail_path": os.path.join(
            dirs["tails"], f"{MODEL_NAME}_tail_farm_{farm_id}.csv"
        ),
        "horizon_metrics_path": os.path.join(
            dirs["validation_diagnostics"],
            f"{stem}_validation_horizon_metrics.csv",
        ),
        "regime_metrics_path": os.path.join(
            dirs["validation_diagnostics"],
            f"{stem}_validation_regime_metrics.csv",
        ),
        "gate_metrics_path": os.path.join(
            dirs["validation_diagnostics"],
            f"{stem}_validation_gate_by_horizon.csv",
        ),
        "validation_plot_path": os.path.join(
            dirs["visualizations"],
            f"{stem}_validation_horizon_metrics.png",
        ),
    }


def _compile_fair_model(model):
    """覆盖旧F7 compile配置，锁定与强基线一致的优化器协议。"""
    for layer in model.layers:
        layer.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=CLIPNORM,
        ),
        loss={
            "forecast_power": keras.losses.Huber(delta=1.0),
            "candidate_forecast": keras.losses.Huber(delta=1.0),
        },
        loss_weights={
            "forecast_power": 1.0,
            "candidate_forecast": CANDIDATE_LOSS_WEIGHT,
        },
        metrics={
            "forecast_power": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
            "candidate_forecast": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
        },
    )
    return model


def build_model(prepared):
    """构建F7结构；不调用任何旧权重迁移函数。"""
    if tuple(feature_train.VARIANT_SPECS[SOURCE_STRUCTURE_VARIANT]["groups"]) != (
        "P",
        "H",
        "D",
    ):
        raise ValueError("上游F7已不再对应P+H+D特征组")
    if len(feature_train.selected_feature_names(SOURCE_STRUCTURE_VARIANT)) != 36:
        raise ValueError("上游F7显式工况特征维数不再是36")
    configure_reproducibility()
    model = feature_train.build_feature_screen_model(
        SOURCE_STRUCTURE_VARIANT,
        len(prepared["input_cols"]),
        int(prepared["target_index"]),
        float(prepared["power_scale_ratio"]),
        float(prepared["power_scale_offset"]),
        prepared["regime_feature_config"],
    )
    model = _compile_fair_model(model)
    total_params = int(model.count_params())
    trainable_params = int(
        sum(int(np.prod(value.shape)) for value in model.trainable_weights)
    )
    if total_params != EXPECTED_TOTAL_PARAMS:
        raise ValueError(
            f"F7总参数量漂移: {total_params:,} != {EXPECTED_TOTAL_PARAMS:,}"
        )
    if trainable_params != EXPECTED_TRAINABLE_PARAMS:
        raise ValueError(
            "F7必须全模型可训练；"
            f"实际{trainable_params:,} != {EXPECTED_TRAINABLE_PARAMS:,}"
        )
    return model


def _attach_targets(dataset):
    def map_targets(batch_x, batch_y):
        return batch_x, {
            "forecast_power": batch_y,
            "candidate_forecast": batch_y,
        }

    return dataset.map(
        map_targets,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    ).prefetch(tf.data.AUTOTUNE)


def make_datasets(prepared, batch_size=BATCH_SIZE):
    train_ds, val_ds, train_samples, total_samples = make_window_dataset(
        prepared["features"],
        prepared["target"],
        HISTORY_LEN,
        FORECAST_LEN,
        int(batch_size),
        VALIDATION_SPLIT,
    )
    return (
        _attach_targets(train_ds),
        _attach_targets(val_ds),
        int(train_samples),
        int(total_samples),
    )


def _history_metric_pair(frame, train_candidates, val_candidates):
    train_name = next((name for name in train_candidates if name in frame), None)
    val_name = next((name for name in val_candidates if name in frame), None)
    return train_name, val_name


def save_history(history, paths, farm_id):
    frame = pd.DataFrame(history.history)
    if frame.empty:
        raise ValueError(f"{farm_id}训练history为空")
    metric_columns = [
        name
        for name in frame.columns
        if name not in {"lr", "learning_rate"}
    ]
    invalid = [
        name
        for name in metric_columns
        if not np.isfinite(pd.to_numeric(frame[name], errors="coerce")).all()
    ]
    if invalid:
        raise ValueError(f"{farm_id}训练history含非有限指标: {invalid}")
    output = frame.copy()
    output.insert(0, "epoch", np.arange(1, len(output) + 1))
    _atomic_csv(output, paths["history_path"])

    os.environ.setdefault(
        "MPLCONFIGDIR",
        os.path.join(os.path.dirname(paths["history_plot_path"]), ".mpl_cache"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [
        (
            "Loss",
            *_history_metric_pair(frame, ["loss"], ["val_loss"]),
        ),
        (
            "Forecast MAE (scaled)",
            *_history_metric_pair(
                frame,
                ["forecast_power_mae"],
                ["val_forecast_power_mae"],
            ),
        ),
        (
            "Forecast RMSE (scaled)",
            *_history_metric_pair(
                frame,
                ["forecast_power_rmse"],
                ["val_forecast_power_rmse"],
            ),
        ),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    x = np.arange(1, len(frame) + 1)
    for ax, (title, train_name, val_name) in zip(axes, pairs):
        if train_name:
            ax.plot(x, frame[train_name], label="train")
        if val_name:
            ax.plot(x, frame[val_name], label="validation")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if train_name or val_name:
            ax.legend()
    axes[-1].set_xlabel("Epoch")
    fig.suptitle(f"F7/G0 strong-baseline training - Farm {farm_id}")
    fig.tight_layout()
    fig.savefig(paths["history_plot_path"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return frame


def _inverse_power(scaler_y, values):
    values = np.asarray(values, dtype=float)
    return scaler_y.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def _metric_row(y_true, y_pred, capacity):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        return {"mae": np.nan, "rmse": np.nan, "nmae": np.nan, "nrmse": np.nan}
    error = y_pred[valid] - y_true[valid]
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    return {
        "mae": mae,
        "rmse": rmse,
        "nmae": mae / float(capacity),
        "nrmse": rmse / float(capacity),
    }


def collect_validation_outputs(model, val_ds, prepared):
    diagnostic = keras.Model(
        model.inputs,
        [
            model.get_layer("forecast_power").output,
            model.get_layer("persistence_forecast_candidate").output,
            model.get_layer("corrected_forecast_candidate").output,
            model.get_layer("correction_gate").output,
        ],
    )
    truth_parts = []
    prediction_parts = [[], [], [], []]
    for batch_x, batch_targets in val_ds:
        batch_y = batch_targets["forecast_power"]
        values = diagnostic(batch_x, training=False)
        truth_parts.append(np.asarray(batch_y))
        for bucket, value in zip(prediction_parts, values):
            bucket.append(np.asarray(value))
    del diagnostic
    if not truth_parts:
        raise ValueError("验证集为空")
    y_scaled = np.concatenate(truth_parts, axis=0)
    forecast_scaled, persistence_scaled, corrected_scaled, gate = [
        np.concatenate(parts, axis=0) for parts in prediction_parts
    ]
    arrays = [y_scaled, forecast_scaled, persistence_scaled, corrected_scaled, gate]
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("验证集预测含非有限值")
    return {
        "truth": _inverse_power(prepared["scaler_y"], y_scaled),
        "forecast": _inverse_power(prepared["scaler_y"], forecast_scaled),
        "persistence": _inverse_power(prepared["scaler_y"], persistence_scaled),
        "corrected": _inverse_power(prepared["scaler_y"], corrected_scaled),
        "gate": np.asarray(gate, dtype=float),
    }


def save_horizon_diagnostics(outputs, prepared, paths, farm_id):
    rows = []
    for horizon in range(FORECAST_LEN):
        row = {
            "model_name": MODEL_NAME,
            "variant_id": VARIANT_ID,
            "farm_id": str(farm_id),
            "horizon": horizon + 1,
            "lead_minutes": (horizon + 1) * 15,
            "gate_mean": float(np.mean(outputs["gate"][:, horizon])),
            "gate_std": float(np.std(outputs["gate"][:, horizon])),
        }
        for role in ("forecast", "corrected", "persistence"):
            metrics = _metric_row(
                outputs["truth"][:, horizon],
                outputs[role][:, horizon],
                prepared["capacity"],
            )
            row.update({f"{role}_{key}": value for key, value in metrics.items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    _atomic_csv(frame, paths["horizon_metrics_path"])

    os.environ.setdefault(
        "MPLCONFIGDIR",
        os.path.join(os.path.dirname(paths["validation_plot_path"]), ".mpl_cache"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for role, label in (
        ("forecast", "F7/G0 fused"),
        ("corrected", "corrected candidate"),
        ("persistence", "persistence"),
    ):
        axes[0].plot(frame["horizon"], frame[f"{role}_nrmse"], label=label)
        axes[1].plot(frame["horizon"], frame[f"{role}_nmae"], label=label)
    axes[0].set_ylabel("NRMSE")
    axes[1].set_ylabel("NMAE")
    axes[1].set_xlabel("Forecast horizon")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Validation horizon diagnostics - Farm {farm_id}")
    fig.tight_layout()
    fig.savefig(paths["validation_plot_path"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return frame


def _atomic_save_model(model, model_path):
    model_path = os.fspath(model_path)
    stem, extension = os.path.splitext(model_path)
    temporary = f"{stem}.tmp{extension}"
    if os.path.isdir(temporary):
        shutil.rmtree(temporary)
    elif os.path.exists(temporary):
        os.remove(temporary)
    try:
        model.save(temporary)
        os.replace(temporary, model_path)
    finally:
        if os.path.isdir(temporary):
            shutil.rmtree(temporary)
        elif os.path.exists(temporary):
            os.remove(temporary)


def verify_saved_model(model, model_path, val_ds):
    sample_x, _ = next(iter(val_ds))
    sample_x = sample_x[:2]
    layer_names = (
        "forecast_power",
        "persistence_forecast_candidate",
        "corrected_forecast_candidate",
        "correction_gate",
    )
    source_diag = keras.Model(
        model.inputs,
        [model.get_layer(name).output for name in layer_names],
    )
    expected = source_diag(sample_x, training=False)
    restored = keras.models.load_model(
        model_path,
        custom_objects=feature_train.get_feature_screen_custom_objects(),
        compile=False,
    )
    restored_diag = keras.Model(
        restored.inputs,
        [restored.get_layer(name).output for name in layer_names],
    )
    actual = restored_diag(sample_x, training=False)
    maximum_error = 0.0
    for name, left, right in zip(layer_names, expected, actual):
        error = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
        maximum_error = max(maximum_error, error)
        if not np.allclose(left, right, rtol=1e-6, atol=1e-6):
            raise ValueError(f"完整.keras重载后{name}不一致: max_abs={error}")
    if len(restored.inputs) != 1 or tuple(restored.input_shape[1:]) != tuple(model.input_shape[1:]):
        raise ValueError("重载模型历史输入接口漂移")
    del source_diag, restored_diag, restored
    return maximum_error


def _callbacks(dirs, paths, farm_id):
    tensorboard_log_dir = os.path.join(
        dirs["tensorboard"],
        f"farm_{farm_id}",
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    guard = NonFiniteTrainingGuard()
    callbacks = [
        guard,
        keras.callbacks.TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=0,
            write_graph=True,
            update_freq="epoch",
            profile_batch=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_forecast_power_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_forecast_power_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            paths["best_weights_path"],
            monitor="val_forecast_power_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]
    return callbacks, guard, tensorboard_log_dir


def train_one_farm(train_file, dirs, epochs=EPOCHS, batch_size=BATCH_SIZE):
    farm_id = str(regime_train.get_farm_id(train_file))
    print(
        f"\n===== Part-3 Round-2 F7/G0 strong baseline / farm={farm_id} "
        f"/ seed={RANDOM_SEED} / batch={batch_size} ====="
    )
    keras.backend.clear_session()
    configure_reproducibility()
    prepared = regime_train._prepare_farm(train_file)
    if str(prepared["farm_id"]) != farm_id:
        raise ValueError("预处理场站身份漂移")
    train_ds, val_ds, train_samples, total_samples = make_datasets(
        prepared, batch_size=batch_size
    )
    model = build_model(prepared)
    paths = artifact_paths(dirs, farm_id)
    callbacks, guard, tensorboard_log_dir = _callbacks(
        dirs, paths, farm_id
    )
    start = time.monotonic()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(epochs),
        callbacks=callbacks,
        verbose=1,
    )
    ensure_finite_training_history(history, guard)
    history_frame = save_history(history, paths, farm_id)
    if not os.path.isfile(paths["best_weights_path"]):
        raise FileNotFoundError(
            f"未产生最佳checkpoint: {paths['best_weights_path']}"
        )
    model.load_weights(paths["best_weights_path"])

    outputs = collect_validation_outputs(model, val_ds, prepared)
    overall = {
        f"val_{key}": value
        for key, value in _metric_row(
            outputs["truth"], outputs["forecast"], prepared["capacity"]
        ).items()
    }
    candidate = {
        f"val_candidate_{key}": value
        for key, value in _metric_row(
            outputs["truth"], outputs["corrected"], prepared["capacity"]
        ).items()
    }
    persistence = {
        f"val_persistence_{key}": value
        for key, value in _metric_row(
            outputs["truth"], outputs["persistence"], prepared["capacity"]
        ).items()
    }
    save_horizon_diagnostics(outputs, prepared, paths, farm_id)
    legacy_diagnostics = feature_train._collect_validation_diagnostics(
        model, val_ds, prepared, SOURCE_STRUCTURE_VARIANT
    )
    _atomic_csv(
        pd.DataFrame(legacy_diagnostics["regime_rows"]),
        paths["regime_metrics_path"],
    )
    _atomic_csv(
        pd.DataFrame(legacy_diagnostics["gate_rows"]),
        paths["gate_metrics_path"],
    )

    prepared["train_df"].iloc[-HISTORY_LEN:].to_csv(
        paths["tail_path"], index=True, encoding="utf-8-sig"
    )
    _atomic_save_model(model, paths["model_path"])
    reload_max_abs_error = verify_saved_model(
        model, paths["model_path"], val_ds
    )
    elapsed = float(time.monotonic() - start)
    steps_per_epoch = int(np.ceil(train_samples / float(batch_size)))
    optimizer_updates = int(model.optimizer.iterations.numpy())
    total_params = int(model.count_params())
    trainable_params = int(
        sum(int(np.prod(value.shape)) for value in model.trainable_weights)
    )
    best_epoch = int(
        np.argmin(np.asarray(history.history["val_forecast_power_loss"])) + 1
    )
    diagnostic_layers = {
        "forecast": "forecast_power",
        "gate": "correction_gate",
        "persistence_candidate": "persistence_forecast_candidate",
        "corrected_candidate": "corrected_forecast_candidate",
    }
    output_semantics = {
        "forecast_power": (
            "primary F7/G0 fused forecast: persistence*(1-gate) + corrected*gate"
        ),
        "candidate_forecast": "auxiliary supervised corrected residual candidate",
        "gate": "history-only per-sample per-horizon corrected-candidate weight",
        "persistence_candidate": "last observed scaled power repeated for 16 horizons",
        "corrected_candidate": "persistence plus lightweight causal residual",
    }
    source_code_path = os.path.abspath(__file__)
    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "model_name": MODEL_NAME,
        "variant_id": VARIANT_ID,
        "architecture_version": ARCHITECTURE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "experiment_part": 3,
        "experiment_round": 2,
        "experiment_role": "strong_baseline_fair_training",
        "farm_id": farm_id,
        "train_file": os.path.abspath(train_file),
        "feature_cols": list(prepared["feature_cols"]),
        "input_cols": list(prepared["input_cols"]),
        "target_col": TARGET_COL,
        "target_index": int(prepared["target_index"]),
        "scaler_x": prepared["scaler_x"],
        "scaler_y": prepared["scaler_y"],
        "capacity": float(prepared["capacity"]),
        "history_len": HISTORY_LEN,
        "forecast_len": FORECAST_LEN,
        "time_freq": TIME_FREQ,
        "power_scale_ratio": float(prepared["power_scale_ratio"]),
        "power_scale_offset": float(prepared["power_scale_offset"]),
        "regime_feature_config": prepared["regime_feature_config"],
        "selected_regime_feature_groups": ["P", "H", "D"],
        "selected_regime_feature_names": list(
            feature_train.selected_feature_names(SOURCE_STRUCTURE_VARIANT)
        ),
        "requires_keras_model": True,
        "model_kind": "keras_network",
        "gate_type": "sample_horizon_sigmoid",
        "encoder_type": "explicit_wind_regime_statistics_PHD",
        "expert_names": ["persistence", "corrected"],
        "model_output_names": list(model.output_names),
        "diagnostic_layers": diagnostic_layers,
        "output_semantics": output_semantics,
        "primary_prediction_output": "forecast_power",
        "training_mode": TRAINING_MODE,
        "initialization_mode": "from_scratch",
        "warm_start": False,
        "loaded_pretrained_weights": False,
        "source_weight_artifact": None,
        "all_model_layers_trainable": True,
        "random_seed": RANDOM_SEED,
        "deterministic_ops_requested": True,
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "epochs_ran": int(len(history_frame)),
        "best_epoch": best_epoch,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_updates": optimizer_updates,
        "validation_split": VALIDATION_SPLIT,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "clipnorm": CLIPNORM,
        "loss": "Huber(delta=1.0)",
        "forecast_loss_weight": 1.0,
        "candidate_supervision_loss_weight": CANDIDATE_LOSS_WEIGHT,
        "early_stopping_monitor": "val_forecast_power_loss",
        "checkpoint_monitor": "val_forecast_power_loss",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "reduce_lr_patience": REDUCE_LR_PATIENCE,
        "reduce_lr_factor": REDUCE_LR_FACTOR,
        "min_learning_rate": MIN_LEARNING_RATE,
        "minimum_learning_rate": MIN_LEARNING_RATE,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_bytes": int(os.path.getsize(paths["model_path"])),
        "training_elapsed_seconds": elapsed,
        "train_samples": train_samples,
        "val_samples": total_samples - train_samples,
        "model_path": os.path.abspath(paths["model_path"]),
        "model_sha256": _file_sha256(paths["model_path"]),
        "best_weights_path": os.path.abspath(paths["best_weights_path"]),
        "best_weights_sha256": _file_sha256(paths["best_weights_path"]),
        "artifact_path": os.path.abspath(paths["artifact_path"]),
        "history_path": os.path.abspath(paths["history_path"]),
        "history_plot_path": os.path.abspath(paths["history_plot_path"]),
        "tensorboard_log_dir": os.path.abspath(tensorboard_log_dir),
        "tail_path": os.path.abspath(paths["tail_path"]),
        "validation_horizon_metrics_path": os.path.abspath(
            paths["horizon_metrics_path"]
        ),
        "validation_regime_metrics_path": os.path.abspath(
            paths["regime_metrics_path"]
        ),
        "validation_gate_diagnostics_path": os.path.abspath(
            paths["gate_metrics_path"]
        ),
        "validation_plot_path": os.path.abspath(paths["validation_plot_path"]),
        "saved_model_reload_verified": True,
        "saved_model_reload_max_abs_error": reload_max_abs_error,
        "training_code_path": source_code_path,
        "training_code_sha256": _file_sha256(source_code_path),
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(keras, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        **overall,
        **candidate,
        **persistence,
        **legacy_diagnostics["gate_fields"],
    }
    _atomic_joblib_dump(artifact, paths["artifact_path"])

    row = {
        "model_family": MODEL_FAMILY,
        "model_name": MODEL_NAME,
        "variant_id": VARIANT_ID,
        "farm_id": farm_id,
        "protocol_version": PROTOCOL_VERSION,
        "training_mode": TRAINING_MODE,
        "initialization_mode": "from_scratch",
        "warm_start": False,
        "random_seed": RANDOM_SEED,
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "epochs_ran": int(len(history_frame)),
        "best_epoch": best_epoch,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_updates": optimizer_updates,
        "learning_rate": LEARNING_RATE,
        "train_samples": train_samples,
        "val_samples": total_samples - train_samples,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "training_elapsed_seconds": elapsed,
        **overall,
        **candidate,
        **persistence,
        **legacy_diagnostics["gate_fields"],
        "model_path": os.path.abspath(paths["model_path"]),
        "model_sha256": _file_sha256(paths["model_path"]),
        "best_weights_path": os.path.abspath(paths["best_weights_path"]),
        "best_weights_sha256": _file_sha256(paths["best_weights_path"]),
        "artifact_path": os.path.abspath(paths["artifact_path"]),
        "artifact_sha256": _file_sha256(paths["artifact_path"]),
        "history_path": os.path.abspath(paths["history_path"]),
        "history_plot_path": os.path.abspath(paths["history_plot_path"]),
        "tail_path": os.path.abspath(paths["tail_path"]),
        "validation_horizon_metrics_path": os.path.abspath(
            paths["horizon_metrics_path"]
        ),
        "validation_regime_metrics_path": os.path.abspath(
            paths["regime_metrics_path"]
        ),
        "validation_gate_diagnostics_path": os.path.abspath(
            paths["gate_metrics_path"]
        ),
        "validation_plot_path": os.path.abspath(paths["validation_plot_path"]),
    }
    print(
        f"{MODEL_NAME}/{farm_id}: val NRMSE={overall['val_nrmse']:.6f}, "
        f"NMAE={overall['val_nmae']:.6f}, best_epoch={best_epoch}, "
        f"params={total_params:,}"
    )
    del model
    keras.backend.clear_session()
    return row


def write_protocol_files(dirs):
    legacy_f7 = {
        "batch_size": 192,
        "epochs": 60,
        "learning_rate": 1e-4,
        "early_stopping_patience": 8,
        "reduce_lr_patience": 3,
        "initialization": "Stage-1 B2 warm-start",
    }
    fair = {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "reduce_lr_patience": REDUCE_LR_PATIENCE,
        "initialization": "single-stage random initialization",
    }
    protocol = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": _utc_now(),
        "model_name": MODEL_NAME,
        "variant_id": VARIANT_ID,
        "result_root": dirs["root"],
        "objective": (
            "Align final F7/G0 optimization hyperparameters with native PatchTST "
            "and other deep baselines without modifying their source artifacts."
        ),
        "architecture": (
            "Persistence + lightweight causal residual + P+H+D explicit regime "
            "encoder + sample-horizon G0 fusion gate"
        ),
        "fair_training_protocol": fair,
        "legacy_f7_protocol": legacy_f7,
        "fixed_common_fields": {
            "validation_split": VALIDATION_SPLIT,
            "random_seed": RANDOM_SEED,
            "optimizer": "Adam",
            "clipnorm": CLIPNORM,
            "huber_delta": 1.0,
            "reduce_lr_factor": REDUCE_LR_FACTOR,
            "min_learning_rate": MIN_LEARNING_RATE,
        },
        "necessary_architecture_specific_fields": {
            "forecast_loss_weight": 1.0,
            "candidate_loss_weight": CANDIDATE_LOSS_WEIGHT,
            "monitor": "val_forecast_power_loss",
            "reason": (
                "F7/G0 has an auxiliary corrected-candidate output; the monitor is "
                "the primary forecast loss, semantically equivalent to val_loss "
                "for a single-output baseline."
            ),
        },
        "forbidden_initialization": [
            "Stage-1 B2 checkpoint",
            "legacy F7 checkpoint",
            "Stage-A checkpoint",
        ],
        "test_split_used_during_training": False,
        "selection_split": "test_after_all_models_are_frozen",
    }
    protocol_path = os.path.join(dirs["manifests"], PROTOCOL_MANIFEST_NAME)
    _atomic_json_dump(protocol, protocol_path)

    rows = [
        {
            "configuration": "native_patchtst",
            "source_code": "wind_dl_model_train.py",
            "batch_size": 256,
            "epochs": 80,
            "learning_rate": 5e-4,
            "early_stopping_patience": 10,
            "reduce_lr_patience": 4,
            "initialization": "random",
            "forecast_loss": "Huber",
        },
        {
            "configuration": "other_deep_baselines_default",
            "source_code": "wind_dl_other_models_train.py",
            "batch_size": 256,
            "epochs": 60,
            "learning_rate": 5e-4,
            "early_stopping_patience": 10,
            "reduce_lr_patience": 4,
            "initialization": "random",
            "forecast_loss": "Huber",
        },
        {
            "configuration": "legacy_f7",
            "source_code": "wind_RegimeEncoder_PatchTST_feature_screen_train.py",
            **legacy_f7,
            "forecast_loss": "Huber + candidate Huber",
        },
        {
            "configuration": VARIANT_ID,
            "source_code": os.path.basename(__file__),
            **fair,
            "forecast_loss": "Huber + candidate Huber",
        },
    ]
    comparison_path = os.path.join(
        dirs["manifests"], PROTOCOL_COMPARISON_NAME
    )
    _atomic_csv(pd.DataFrame(rows), comparison_path)
    return os.path.abspath(protocol_path), os.path.abspath(comparison_path)


def discover_train_files(farms=None):
    files = sorted(
        glob.glob(os.path.join(DATA_DIR, regime_train.TRAIN_FILE_PATTERN))
    )
    if farms:
        requested = {str(value) for value in farms}
        files = [
            path
            for path in files
            if str(regime_train.get_farm_id(path)) in requested
        ]
        found = {str(regime_train.get_farm_id(path)) for path in files}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"请求场站没有训练文件: {missing}")
    if not files:
        raise FileNotFoundError(f"未在{DATA_DIR}找到训练文件")
    return files


def _row_from_existing_artifact(path, expected_farm_id):
    artifact = joblib.load(path)
    required = {
        "model_name": MODEL_NAME,
        "variant_id": VARIANT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "training_mode": TRAINING_MODE,
        "initialization_mode": "from_scratch",
        "warm_start": False,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "candidate_supervision_loss_weight": CANDIDATE_LOSS_WEIGHT,
        "checkpoint_monitor": "val_forecast_power_loss",
        "total_params": EXPECTED_TOTAL_PARAMS,
        "trainable_params": EXPECTED_TRAINABLE_PARAMS,
        "all_model_layers_trainable": True,
    }
    failed = [key for key, value in required.items() if artifact.get(key) != value]
    if artifact.get("expert_names") != ["persistence", "corrected"]:
        failed.append("expert_names")
    if failed:
        raise ValueError(f"既有artifact身份不兼容: {path}; fields={failed}")
    if str(artifact.get("farm_id")) != str(expected_farm_id):
        raise ValueError(
            "既有artifact场站身份与文件槽位不一致: "
            f"{artifact.get('farm_id')} != {expected_farm_id}; {path}"
        )
    for key in (
        "model_path",
        "best_weights_path",
        "history_path",
        "history_plot_path",
        "tail_path",
        "validation_horizon_metrics_path",
        "validation_regime_metrics_path",
        "validation_gate_diagnostics_path",
        "validation_plot_path",
    ):
        if not os.path.exists(artifact[key]):
            raise FileNotFoundError(f"既有artifact缺少{key}: {artifact[key]}")
    hash_checks = {
        "model_path": "model_sha256",
        "best_weights_path": "best_weights_sha256",
    }
    for file_field, hash_field in hash_checks.items():
        actual = _file_sha256(artifact[file_field])
        if actual != artifact.get(hash_field):
            raise ValueError(
                f"既有artifact的{file_field}哈希漂移: "
                f"{actual} != {artifact.get(hash_field)}"
            )
    if not bool(artifact.get("saved_model_reload_verified", False)):
        raise ValueError("既有artifact未声明完整.keras重载验收")
    fields = {
        key: artifact.get(key)
        for key in (
            "val_mae",
            "val_rmse",
            "val_nmae",
            "val_nrmse",
            "val_candidate_mae",
            "val_candidate_rmse",
            "val_candidate_nmae",
            "val_candidate_nrmse",
            "val_persistence_mae",
            "val_persistence_rmse",
            "val_persistence_nmae",
            "val_persistence_nrmse",
            "gate_mean",
            "gate_std",
            "gate_min",
            "gate_max",
        )
    }
    return {
        "model_family": MODEL_FAMILY,
        "model_name": MODEL_NAME,
        "variant_id": VARIANT_ID,
        "farm_id": str(artifact["farm_id"]),
        "protocol_version": PROTOCOL_VERSION,
        "training_mode": TRAINING_MODE,
        "initialization_mode": "from_scratch",
        "warm_start": False,
        "random_seed": RANDOM_SEED,
        "batch_size": int(artifact["batch_size"]),
        "epochs": int(artifact["epochs"]),
        "epochs_ran": int(artifact["epochs_ran"]),
        "best_epoch": int(artifact["best_epoch"]),
        "steps_per_epoch": int(artifact["steps_per_epoch"]),
        "optimizer_updates": int(artifact["optimizer_updates"]),
        "learning_rate": float(artifact["learning_rate"]),
        "train_samples": int(artifact["train_samples"]),
        "val_samples": int(artifact["val_samples"]),
        "total_params": int(artifact["total_params"]),
        "trainable_params": int(artifact["trainable_params"]),
        "training_elapsed_seconds": float(artifact["training_elapsed_seconds"]),
        **fields,
        "model_path": artifact["model_path"],
        "model_sha256": artifact["model_sha256"],
        "best_weights_path": artifact["best_weights_path"],
        "best_weights_sha256": artifact["best_weights_sha256"],
        "artifact_path": os.path.abspath(path),
        "artifact_sha256": _file_sha256(path),
        "history_path": artifact["history_path"],
        "history_plot_path": artifact["history_plot_path"],
        "tail_path": artifact["tail_path"],
        "validation_horizon_metrics_path": artifact[
            "validation_horizon_metrics_path"
        ],
        "validation_regime_metrics_path": artifact[
            "validation_regime_metrics_path"
        ],
        "validation_gate_diagnostics_path": artifact[
            "validation_gate_diagnostics_path"
        ],
        "validation_plot_path": artifact["validation_plot_path"],
    }


def validation_summary(metrics):
    numeric = (
        "val_nrmse",
        "val_nmae",
        "val_candidate_nrmse",
        "val_candidate_nmae",
        "val_persistence_nrmse",
        "val_persistence_nmae",
        "training_elapsed_seconds",
    )
    row = {
        "model_name": MODEL_NAME,
        "variant_id": VARIANT_ID,
        "farm_count": int(metrics["farm_id"].astype(str).nunique()),
        "selection_split": "test_not_validation",
        "validation_is_descriptive_only": True,
    }
    for key in numeric:
        values = pd.to_numeric(metrics[key], errors="coerce")
        row[f"macro_{key}"] = float(values.mean())
        row[f"std_{key}"] = float(values.std(ddof=0))
    return pd.DataFrame([row])


def publish_complete_marker(dirs, metrics_path, validation_path, protocol_paths, rows):
    farm_ids = sorted(str(row["farm_id"]) for row in rows)
    if len(farm_ids) != EXPECTED_FARM_COUNT or len(set(farm_ids)) != EXPECTED_FARM_COUNT:
        raise ValueError(f"完整bundle应为{EXPECTED_FARM_COUNT}场站，实际{farm_ids}")
    files = {
        "training_summary": _file_record(metrics_path),
        "validation_summary": _file_record(validation_path),
        "protocol_manifest": _file_record(protocol_paths[0]),
        "protocol_comparison": _file_record(protocol_paths[1]),
    }
    for row in rows:
        farm_id = str(row["farm_id"])
        for field in (
            "model_path",
            "best_weights_path",
            "artifact_path",
            "history_path",
            "history_plot_path",
            "tail_path",
            "validation_horizon_metrics_path",
            "validation_regime_metrics_path",
            "validation_gate_diagnostics_path",
            "validation_plot_path",
        ):
            files[f"farm_{farm_id}_{field}"] = _file_record(row[field])
    marker = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "model_name": MODEL_NAME,
        "variant_id": VARIANT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "expected_farm_ids": farm_ids,
        "farm_count": len(farm_ids),
        "selection_split": "test",
        "validation_selection_forbidden": True,
        "training_used_test_data": False,
        "single_stage_from_scratch": True,
        "training_mode": TRAINING_MODE,
        "initialization_mode": "from_scratch",
        "warm_start": False,
        "files": files,
        "training_code": _file_record(os.path.abspath(__file__)),
    }
    path = os.path.join(dirs["root"], COMPLETION_MARKER_NAME)
    return _atomic_json_dump(marker, path)


def validate_complete_marker(marker_path):
    """只读验收既有正式bundle；失败时保留marker及所有归档。"""
    with open(marker_path, "r", encoding="utf-8") as handle:
        marker = json.load(handle)
    expected_identity = {
        "status": "complete",
        "model_name": MODEL_NAME,
        "variant_id": VARIANT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "farm_count": EXPECTED_FARM_COUNT,
        "training_mode": TRAINING_MODE,
        "initialization_mode": "from_scratch",
        "warm_start": False,
    }
    drifted = [
        key
        for key, expected in expected_identity.items()
        if marker.get(key) != expected
    ]
    if drifted:
        raise ValueError(
            f"既有complete marker身份漂移: {drifted}; {marker_path}"
        )
    records = dict(marker.get("files") or {})
    records["training_code"] = marker.get("training_code")
    if not records or records["training_code"] is None:
        raise ValueError(f"既有complete marker缺少文件记录: {marker_path}")
    for name, expected in records.items():
        if not isinstance(expected, dict) or "path" not in expected:
            raise ValueError(f"marker文件记录无效: {name}")
        actual = _file_record(expected["path"])
        if (
            actual["sha256"] != expected.get("sha256")
            or actual["size_bytes"] != int(expected.get("size_bytes", -1))
        ):
            raise ValueError(
                f"既有正式bundle文件发生变化: {name}={expected['path']}"
            )
    return marker


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Part-3 Round-2 F7/G0 strong-baseline fair training"
    )
    parser.add_argument(
        "--farms",
        nargs="*",
        help="仅用于局部/冒烟运行；正式bundle默认自动发现全部5场站",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--result-root", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configure_reproducibility()
    train_files = discover_train_files(args.farms)
    all_files = discover_train_files()
    all_farm_ids = sorted(str(regime_train.get_farm_id(path)) for path in all_files)
    if len(all_farm_ids) != EXPECTED_FARM_COUNT:
        raise ValueError(
            f"正式协议预期{EXPECTED_FARM_COUNT}场站，发现{all_farm_ids}"
        )

    is_formal = (
        not args.smoke
        and not args.farms
        and args.result_root is None
        and len(train_files) == EXPECTED_FARM_COUNT
    )
    if args.result_root:
        result_root = os.path.abspath(args.result_root)
        if os.path.realpath(result_root) == os.path.realpath(RESULT_ROOT):
            raise ValueError("自定义--result-root不能指向正式RESULT_ROOT")
    elif is_formal:
        result_root = RESULT_ROOT
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "smoke" if args.smoke else "partial"
        result_root = os.path.join(RESULT_ROOT, "partial_runs", f"{stamp}_{tag}")
    dirs = result_dirs(result_root)
    marker_path = os.path.join(dirs["root"], COMPLETION_MARKER_NAME)
    if is_formal and os.path.exists(marker_path):
        if not args.resume:
            raise FileExistsError(
                "正式训练bundle已经完成；如需逐文件hash复验，请使用--resume: "
                f"{marker_path}"
            )
        validate_complete_marker(marker_path)
        print(f"正式训练bundle逐文件hash复验通过，无需重写: {marker_path}")
        return

    existing_artifacts = [
        artifact_paths(
            dirs, str(regime_train.get_farm_id(train_file))
        )["artifact_path"]
        for train_file in train_files
    ]
    existing_artifacts = [
        path for path in existing_artifacts if os.path.isfile(path)
    ]
    if existing_artifacts and not args.resume:
        raise FileExistsError(
            "目标目录已有场站artifact，未改写任何manifest；"
            "请使用--resume复验复用或更换--result-root: "
            + ", ".join(existing_artifacts)
        )
    protocol_paths = write_protocol_files(dirs)

    epochs = 1 if args.smoke else EPOCHS
    rows = []
    for train_file in train_files:
        farm_id = str(regime_train.get_farm_id(train_file))
        paths = artifact_paths(dirs, farm_id)
        if args.resume and os.path.isfile(paths["artifact_path"]):
            print(f"复用已验收artifact: {paths['artifact_path']}")
            row = _row_from_existing_artifact(
                paths["artifact_path"],
                expected_farm_id=farm_id,
            )
        else:
            if os.path.exists(paths["artifact_path"]):
                raise FileExistsError(
                    f"已有场站产物，使用--resume复用或更换result-root: "
                    f"{paths['artifact_path']}"
                )
            row = train_one_farm(
                train_file,
                dirs,
                epochs=epochs,
                batch_size=BATCH_SIZE,
            )
        rows.append(row)
        progress_path = os.path.join(
            dirs["root"], "part3_round2_strong_baseline_training_progress.csv"
        )
        _atomic_csv(pd.DataFrame(rows), progress_path)

    metrics = pd.DataFrame(rows).sort_values("farm_id").reset_index(drop=True)
    suffix = "" if is_formal else "_partial"
    metrics_path = os.path.join(
        dirs["root"],
        TRAINING_SUMMARY_NAME.replace(".csv", f"{suffix}.csv"),
    )
    validation_path = os.path.join(
        dirs["root"],
        VALIDATION_SUMMARY_NAME.replace(".csv", f"{suffix}.csv"),
    )
    _atomic_csv(metrics, metrics_path)
    _atomic_csv(validation_summary(metrics), validation_path)
    print(f"训练汇总: {metrics_path}")
    print(f"验证描述汇总: {validation_path}")
    if is_formal:
        marker = publish_complete_marker(
            dirs,
            metrics_path,
            validation_path,
            protocol_paths,
            rows,
        )
        print(f"正式训练bundle完成标志: {marker}")
    else:
        print("当前为partial/smoke运行，不发布正式complete marker。")
    print("最终模型必须由预测入口按测试集统一CSV比较，禁止按验证集定型。")


if __name__ == "__main__":
    main()
