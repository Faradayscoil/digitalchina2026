"""Stage 5A：轻量 fine/mid/coarse corrected-candidate 表示预筛训练。

正式矩阵固定为五个变体：

* X0：只读引用 Stage-4B D0/G0/F7，不训练、不复制模型；
* X1-F：fine 历史表示，causal patch=4、right-aligned stride=2；
* X1-M：mid 历史表示，causal patch=8、right-aligned stride=4；
* X1-C：coarse 历史表示，causal patch=16、right-aligned stride=8；
* X1：三个独立尺度编码到共同 latent 后作静态融合，不包含 token 交互。

X1-F/M/C/X1 都冻结 F7 residual、显式工况 context 与原 G0 gate，只训练
零初始化的新增 candidate residual。训练 checkpoint 只依据 validation candidate
NRMSE，冻结 G0 融合结果仅用于收益转化诊断。最终模型选择由配套预测脚本在
测试集完成。任何 smoke、变体/场站/epoch override 均写入 partial_runs，不会
覆盖正式 4×5 新训练 bundle。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

import wind_RegimeEncoder_PatchTST_feature_screen_train as feature_train
import wind_RegimeEncoder_PatchTST_train as regime_train
import wind_controlled_gate_cali_train as gate_train
import wind_dl_model_train as common_train
import wind_time_freq_model_stage4b_train as stage4b_train
import wind_time_freq_model_train as time_freq_train
from wind_dl_model_train import (
    DATA_DIR,
    FORECAST_LEN,
    HISTORY_LEN,
    TARGET_COL,
    TIME_FREQ,
    make_window_dataset,
    set_global_seed,
)


MODEL_FAMILY = "multiscale_correc_cand"
ARCHITECTURE_VERSION = "stage5a_multiscale_candidate_x0_x1_v1"
ARTIFACT_SCHEMA_VERSION = 1
PROTOCOL_VERSION = "multiscale_corrected_candidate_test_selected_v1"
RESULT_ROOT = os.path.join("./wind_results", MODEL_FAMILY)
SOURCE_VARIANT = "f7"
SOURCE_FEATURE_GROUPS = "P+H+D"
SOURCE_FEATURE_COUNT = 36
RANDOM_SEED = 2026

BATCH_SIZE = int(os.getenv("WIND_MULTISCALE_BATCH_SIZE", "192"))
VALIDATION_SPLIT = float(os.getenv("WIND_MULTISCALE_VALIDATION_SPLIT", "0.15"))
CANDIDATE_EPOCHS = int(os.getenv("WIND_MULTISCALE_CANDIDATE_EPOCHS", "30"))
CANDIDATE_LEARNING_RATE = float(
    os.getenv("WIND_MULTISCALE_CANDIDATE_LEARNING_RATE", "0.0001")
)
EARLY_STOPPING_PATIENCE = int(os.getenv("WIND_MULTISCALE_PATIENCE", "6"))
PARAMETER_LIMIT = 30000
COMMON_LATENT_DIM = 16
PATCH_FILTERS = 8
FUSION_HIDDEN_DIM = 32
CORRECTION_KERNEL_L2 = regime_train.CORRECTION_KERNEL_L2
SOURCE_RECONSTRUCTION_MAX_ABS_TOL = 1e-7

SCALE_SPECS = {
    "fine": {"patch": 4, "stride": 2, "seed": 2511},
    "mid": {"patch": 8, "stride": 4, "seed": 2521},
    "coarse": {"patch": 16, "stride": 8, "seed": 2531},
}

VARIANT_SPECS = {
    "x0": {
        "label": "X0 D0/G0/F7 direct reference",
        "requires_training": False,
        "scales": (),
        "fusion": "none",
        "description": "只读引用Stage-4B D0/F7，不训练、不复制、不重新推理",
    },
    "x1_f": {
        "label": "X1-F fine causal patch representation",
        "requires_training": True,
        "scales": ("fine",),
        "fusion": "single_scale_static",
        "description": "patch=4/stride=2的fine历史表示",
    },
    "x1_m": {
        "label": "X1-M mid causal patch representation",
        "requires_training": True,
        "scales": ("mid",),
        "fusion": "single_scale_static",
        "description": "patch=8/stride=4的mid历史表示",
    },
    "x1_c": {
        "label": "X1-C coarse causal patch representation",
        "requires_training": True,
        "scales": ("coarse",),
        "fusion": "single_scale_static",
        "description": "patch=16/stride=8的coarse历史表示",
    },
    "x1": {
        "label": "X1 independent fine/mid/coarse static fusion",
        "requires_training": True,
        "scales": ("fine", "mid", "coarse"),
        "fusion": "concat_common_latent_then_static_dense",
        "description": "三尺度独立编码到共同latent后静态融合；无token交互",
    },
}
TRAINABLE_VARIANTS = ("x1_f", "x1_m", "x1_c", "x1")
REFERENCE_VARIANTS = ("x0",)

SOURCE_WEIGHTED_LAYER_NAMES = tuple(gate_train.COMMON_WEIGHTED_LAYER_NAMES) + (
    "correction_gate",
)
SCALE_WEIGHTED_LAYER_NAMES = {
    scale: (
        f"ms_{scale}_causal_patch_projection",
        f"ms_{scale}_token_norm",
        f"ms_{scale}_latent_projection",
    )
    for scale in SCALE_SPECS
}
ADAPTER_WEIGHTED_LAYER_NAMES = {
    "x1_f": SCALE_WEIGHTED_LAYER_NAMES["fine"]
    + ("ms_single_static_hidden", "ms_residual_delta"),
    "x1_m": SCALE_WEIGHTED_LAYER_NAMES["mid"]
    + ("ms_single_static_hidden", "ms_residual_delta"),
    "x1_c": SCALE_WEIGHTED_LAYER_NAMES["coarse"]
    + ("ms_single_static_hidden", "ms_residual_delta"),
    "x1": SCALE_WEIGHTED_LAYER_NAMES["fine"]
    + SCALE_WEIGHTED_LAYER_NAMES["mid"]
    + SCALE_WEIGHTED_LAYER_NAMES["coarse"]
    + ("ms_multiscale_static_fusion", "ms_residual_delta"),
}

# 这些值在文件末的真实TensorFlow构图自检中再次强制验证。
EXPECTED_TOTAL_PARAMS = {
    "x0": 20969,
    "x1_f": 22369,
    "x1_m": 22401,
    "x1_c": 22465,
    "x1": 24177,
}
EXPECTED_ADAPTER_TRAINABLE_PARAMS = {
    "x1_f": 1400,
    "x1_m": 1432,
    "x1_c": 1496,
    "x1": 3208,
}

TRAINING_SUMMARY_NAME = "multiscale_correc_cand_training_metrics.csv"
MANIFEST_NAME = "multiscale_correc_cand_experiment_manifest.csv"
TRAINING_MARKER_NAME = "multiscale_correc_cand_training_bundle_complete.json"
RUNNING_MARKER_NAME = "multiscale_correc_cand_training_bundle_running.json"
PREDICTION_MARKER_RELATIVE_PATH = os.path.join(
    "testdata_predict_output", "multiscale_correc_cand_test_bundle_complete.json"
)


def configure_reproducibility():
    os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_SEED))
    set_global_seed(RANDOM_SEED)
    keras.utils.set_random_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


@keras.utils.register_keras_serializable(package="WindMultiscaleCandidate")
class TargetHistoryChannel(layers.Layer):
    """Select the scaled historical target and retain a one-channel axis."""

    def __init__(self, target_channel_index, **kwargs):
        super().__init__(**kwargs)
        self.target_channel_index = int(target_channel_index)
        if self.target_channel_index < 0:
            raise ValueError("target_channel_index必须非负")

    def call(self, inputs):
        index = self.target_channel_index
        return inputs[:, -HISTORY_LEN:, index : index + 1]

    def compute_output_shape(self, input_shape):
        return input_shape[0], HISTORY_LEN, 1

    def get_config(self):
        config = super().get_config()
        config.update({"target_channel_index": self.target_channel_index})
        return config


@keras.utils.register_keras_serializable(package="WindMultiscaleCandidate")
class RightAlignedTokenSubsample(layers.Layer):
    """Take right-aligned tokens so every scale's final token contains t=95.

    The preceding convolution always uses stride=1 and causal padding.  For a
    stride ``s``, indices ``s-1, 2s-1, ...`` are retained. HISTORY_LEN=96 is
    divisible by every registered stride, therefore the last retained token is
    always index 95 and no latest observation is silently discarded.
    """

    def __init__(self, stride, **kwargs):
        super().__init__(**kwargs)
        self.stride = int(stride)
        if self.stride <= 0 or HISTORY_LEN % self.stride != 0:
            raise ValueError("stride必须为HISTORY_LEN的正因子")

    def call(self, inputs):
        return inputs[:, self.stride - 1 :: self.stride, :]

    def compute_output_shape(self, input_shape):
        return input_shape[0], HISTORY_LEN // self.stride, input_shape[-1]

    def get_config(self):
        config = super().get_config()
        config.update({"stride": self.stride})
        return config


def get_multiscale_custom_objects():
    objects = dict(feature_train.get_feature_screen_custom_objects())
    for cls in (TargetHistoryChannel, RightAlignedTokenSubsample):
        objects[cls.__name__] = cls
        objects[f"WindMultiscaleCandidate>{cls.__name__}"] = cls
    return objects


def get_time_freq_custom_objects():
    """Compatibility alias for prediction helpers that expect this name."""
    return get_multiscale_custom_objects()


def _sha256(path):
    return time_freq_train._sha256(path)


def _array_sha256(values):
    return time_freq_train._array_sha256(values)


def _atomic_to_csv(frame, path):
    return time_freq_train._atomic_to_csv(frame, path)


def _atomic_write_json(value, path):
    return time_freq_train._atomic_write_json(value, path)


def _atomic_joblib_dump(value, path):
    return time_freq_train._atomic_joblib_dump(value, path)


def _file_record(path):
    return time_freq_train._file_record(path)


def _save_model_atomic(model, path):
    return time_freq_train._save_model_atomic(model, path)


def expected_farm_ids():
    farms = tuple(str(value) for value in feature_train.expected_training_farm_ids())
    if len(farms) != 5:
        raise ValueError(f"正式来源场站数不是5: {farms}")
    return farms


def variant_model_name(variant_id):
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"未知Stage-5A变体: {variant_id}")
    return f"{MODEL_FAMILY}_{variant_id}"


def variant_dirs(variant_id, create=True, result_root=None):
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"未知Stage-5A变体: {variant_id}")
    root = os.path.join(RESULT_ROOT if result_root is None else result_root, variant_id)
    values = {
        "root": root,
        "models": os.path.join(root, "models"),
        "weights": os.path.join(root, "weights"),
        "preprocess": os.path.join(root, "preprocess"),
        "history": os.path.join(root, "history"),
        "tensorboard": os.path.join(root, "tensorboard"),
        "validation_diagnostics": os.path.join(root, "validation_diagnostics"),
        "candidate_diagnostics": os.path.join(root, "candidate_diagnostics"),
        "tails": os.path.join(root, "tails"),
        "records": os.path.join(root, "records"),
    }
    if create:
        for path in values.values():
            os.makedirs(path, exist_ok=True)
    return values


def _paths(dirs, variant_id, farm_id):
    prefix = f"{variant_model_name(variant_id)}_farm_{farm_id}"
    return {
        "model_path": os.path.join(dirs["models"], f"{prefix}.keras"),
        "weights_path": os.path.join(
            dirs["weights"], f"{prefix}_candidate_best.weights.h5"
        ),
        "artifact_path": os.path.join(
            dirs["preprocess"], f"{prefix}_preprocess.pkl"
        ),
        "history_path": os.path.join(
            dirs["history"], f"{prefix}_candidate_history.csv"
        ),
        "history_figure_path": os.path.join(
            dirs["history"], f"{prefix}_candidate_history.png"
        ),
        "validation_path": os.path.join(
            dirs["validation_diagnostics"], f"{prefix}_validation.csv"
        ),
        "checkpoint_path": os.path.join(
            dirs["validation_diagnostics"], f"{prefix}_checkpoint_trace.csv"
        ),
        "provenance_path": os.path.join(
            dirs["candidate_diagnostics"], f"{prefix}_candidate_provenance.csv"
        ),
        "tail_path": os.path.join(dirs["tails"], f"{prefix}_tail.csv"),
        "record_path": os.path.join(dirs["records"], f"{prefix}_record.json"),
    }


def _weighted_snapshot(model, layer_names):
    values = []
    for name in layer_names:
        layer = model.get_layer(name)
        for index, value in enumerate(layer.get_weights()):
            values.append((f"{name}:{index}", value))
    return values


def _copy_source_weights(source_model, model):
    copied = []
    for name in SOURCE_WEIGHTED_LAYER_NAMES:
        source_values = source_model.get_layer(name).get_weights()
        target = model.get_layer(name)
        if [value.shape for value in source_values] != [
            value.shape for value in target.get_weights()
        ]:
            raise ValueError(f"F7->{model.name}层{name}权重形状不一致")
        target.set_weights(source_values)
        if any(
            not np.array_equal(left, right)
            for left, right in zip(source_values, target.get_weights())
        ):
            raise ValueError(f"F7->{model.name}层{name}未精确复制")
        copied.append(name)
    return copied


def _scale_encoder(signal, scale):
    spec = SCALE_SPECS[scale]
    seed = int(spec["seed"])
    tokens = layers.Conv1D(
        PATCH_FILTERS,
        kernel_size=int(spec["patch"]),
        strides=1,
        padding="causal",
        activation="gelu",
        kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
        bias_initializer="zeros",
        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
        name=f"ms_{scale}_causal_patch_projection",
    )(signal)
    tokens = RightAlignedTokenSubsample(
        int(spec["stride"]), name=f"ms_{scale}_right_aligned_subsample"
    )(tokens)
    tokens = layers.LayerNormalization(
        epsilon=1e-5, name=f"ms_{scale}_token_norm"
    )(tokens)
    average = layers.GlobalAveragePooling1D(
        name=f"ms_{scale}_global_average"
    )(tokens)
    maximum = layers.GlobalMaxPooling1D(name=f"ms_{scale}_global_max")(tokens)
    summary = layers.Concatenate(name=f"ms_{scale}_static_summary")(
        [average, maximum]
    )
    return layers.Dense(
        COMMON_LATENT_DIM,
        activation="gelu",
        kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 1),
        bias_initializer="zeros",
        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
        name=f"ms_{scale}_latent_projection",
    )(summary)


def _adapter_delta(variant_id, template, source_artifact):
    signal = TargetHistoryChannel(
        int(source_artifact["target_index"]), name="ms_target_history"
    )(template.inputs[0])
    scales = tuple(VARIANT_SPECS[variant_id]["scales"])
    latent = [_scale_encoder(signal, scale) for scale in scales]
    if len(latent) == 1:
        hidden = layers.Dense(
            FUSION_HIDDEN_DIM,
            activation="gelu",
            kernel_initializer=keras.initializers.GlorotUniform(seed=2541),
            bias_initializer="zeros",
            kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
            name="ms_single_static_hidden",
        )(latent[0])
    else:
        merged = layers.Concatenate(name="ms_multiscale_latent_concat")(latent)
        hidden = layers.Dense(
            FUSION_HIDDEN_DIM,
            activation="gelu",
            kernel_initializer=keras.initializers.GlorotUniform(seed=2551),
            bias_initializer="zeros",
            kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
            name="ms_multiscale_static_fusion",
        )(merged)
    return layers.Dense(
        FORECAST_LEN,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
        name="ms_residual_delta",
    )(hidden)


def build_multiscale_model(variant_id, source_artifact):
    if variant_id not in TRAINABLE_VARIANTS:
        raise ValueError(f"禁止构建引用变体{variant_id}")
    configure_reproducibility()
    template = feature_train.build_feature_screen_model_from_artifact(source_artifact)
    persistence = template.get_layer("persistence_forecast_candidate").output
    base_corrected = template.get_layer("corrected_forecast_candidate").output
    source_gate = template.get_layer("correction_gate").output
    delta = _adapter_delta(variant_id, template, source_artifact)
    corrected = layers.Add(name="ms_corrected_add")([base_corrected, delta])
    corrected = layers.Activation(
        "linear", name="multiscale_corrected_candidate"
    )(corrected)
    gate = layers.Activation("linear", name="frozen_g0_gate")(source_gate)
    forecast = regime_train.TwoCandidateGateFusion(name="forecast_power")(
        [persistence, corrected, gate]
    )
    candidate = layers.Activation("linear", name="candidate_forecast")(corrected)
    model = keras.Model(
        template.inputs,
        {"forecast_power": forecast, "candidate_forecast": candidate},
        name=f"WindMultiscaleCandidate_{variant_id.upper()}",
    )
    total = int(model.count_params())
    expected = EXPECTED_TOTAL_PARAMS[variant_id]
    if total != expected:
        raise ValueError(f"{variant_id}参数量{total} != 预注册值{expected}")
    if total >= PARAMETER_LIMIT:
        raise ValueError(f"{variant_id}参数量{total}不满足<{PARAMETER_LIMIT}")
    return model


def diagnostic_model(model):
    return keras.Model(
        model.inputs,
        {
            "forecast": model.get_layer("forecast_power").output,
            "persistence": model.get_layer("persistence_forecast_candidate").output,
            "base_corrected": model.get_layer("corrected_forecast_candidate").output,
            "corrected": model.get_layer("multiscale_corrected_candidate").output,
            "gate": model.get_layer("frozen_g0_gate").output,
            "delta": model.get_layer("ms_residual_delta").output,
        },
    )


def _source_diagnostic_model(model):
    return keras.Model(
        model.inputs,
        {
            "forecast": model.get_layer("forecast_power").output,
            "persistence": model.get_layer("persistence_forecast_candidate").output,
            "corrected": model.get_layer("corrected_forecast_candidate").output,
            "gate": model.get_layer("correction_gate").output,
        },
    )


def _plain_datasets(prepared):
    return make_window_dataset(
        prepared["features"],
        prepared["target"],
        HISTORY_LEN,
        FORECAST_LEN,
        BATCH_SIZE,
        VALIDATION_SPLIT,
    )


def _attach_training_targets(dataset):
    def attach(batch_x, batch_y):
        return batch_x, {
            "forecast_power": batch_y,
            "candidate_forecast": batch_y,
        }

    return dataset.map(
        attach, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    ).prefetch(tf.data.AUTOTUNE)


def _inverse_scaled(values, prepared):
    shape = np.asarray(values).shape
    values = (
        prepared["scaler_y"]
        .inverse_transform(np.asarray(values).reshape(-1, 1))
        .reshape(shape)
    )
    return np.clip(values, 0.0, float(prepared["capacity"]))


def validation_diagnostics(model, dataset, prepared, variant_id):
    diagnostic = diagnostic_model(model)
    truths = []
    outputs = {name: [] for name in ("forecast", "persistence", "corrected", "gate", "delta")}
    for batch_x, batch_y in dataset:
        result = diagnostic(batch_x, training=False)
        truths.append(np.asarray(batch_y))
        for name in outputs:
            outputs[name].append(np.asarray(result[name]))
    if not truths:
        raise ValueError("验证集为空")
    truth = _inverse_scaled(np.concatenate(truths), prepared)
    values = {name: np.concatenate(parts) for name, parts in outputs.items()}
    physical = {
        name: _inverse_scaled(values[name], prepared)
        for name in ("forecast", "persistence", "corrected")
    }
    capacity = float(prepared["capacity"])
    row = {
        "variant_id": variant_id,
        "farm_id": str(prepared["farm_id"]),
        "valid_count": int(truth.size),
        "candidate_mae": float(np.mean(np.abs(physical["corrected"] - truth))),
        "candidate_rmse": float(np.sqrt(np.mean(np.square(physical["corrected"] - truth)))),
        "candidate_nmae": float(np.mean(np.abs(physical["corrected"] - truth)) / capacity),
        "candidate_nrmse": float(np.sqrt(np.mean(np.square(physical["corrected"] - truth))) / capacity),
        "frozen_g0_fused_mae": float(np.mean(np.abs(physical["forecast"] - truth))),
        "frozen_g0_fused_rmse": float(np.sqrt(np.mean(np.square(physical["forecast"] - truth)))),
        "frozen_g0_fused_nmae": float(np.mean(np.abs(physical["forecast"] - truth)) / capacity),
        "frozen_g0_fused_nrmse": float(np.sqrt(np.mean(np.square(physical["forecast"] - truth))) / capacity),
        "persistence_nrmse": float(np.sqrt(np.mean(np.square(physical["persistence"] - truth))) / capacity),
        "gate_mean": float(np.mean(values["gate"])),
        "gate_std": float(np.std(values["gate"])),
        "scaled_delta_mean_abs": float(np.mean(np.abs(values["delta"]))),
        "scaled_delta_max_abs": float(np.max(np.abs(values["delta"]))),
        "diagnostic_scope": "validation_checkpoint_only_not_test_selection",
    }
    if not all(
        np.isfinite(value) for value in row.values() if isinstance(value, float)
    ):
        raise FloatingPointError("验证诊断包含非有限值")
    return pd.DataFrame([row])


class CandidateValidationCheckpoint(keras.callbacks.Callback):
    """Checkpoint full model strictly by physical candidate validation NRMSE."""

    def __init__(self, full_model, path, validation_dataset, prepared, variant_id):
        super().__init__()
        self.full_model = full_model
        self.path = path
        self.validation_dataset = validation_dataset
        self.prepared = prepared
        self.variant_id = variant_id
        self.best = np.inf
        self.records = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = validation_diagnostics(
            self.full_model,
            self.validation_dataset,
            self.prepared,
            self.variant_id,
        ).iloc[0].to_dict()
        nrmse = float(row["candidate_nrmse"])
        updated = nrmse < self.best - 1e-12
        if updated:
            self.best = nrmse
            self.full_model.save_weights(self.path)
        logs["selection_val_candidate_nrmse"] = nrmse
        self.records.append(
            {
                "epoch": int(epoch),
                "candidate_validation_nrmse": nrmse,
                "frozen_g0_fused_validation_nrmse": float(
                    row["frozen_g0_fused_nrmse"]
                ),
                "checkpoint_updated": bool(updated),
            }
        )


def _set_candidate_phase(model, variant_id):
    for layer in model.layers:
        layer.trainable = False
    for name in ADAPTER_WEIGHTED_LAYER_NAMES[variant_id]:
        model.get_layer(name).trainable = True
    model.get_layer("residual_dropout").rate = 0.0
    model.get_layer("regime_context_dropout").rate = 0.0
    count = int(sum(int(np.prod(weight.shape)) for weight in model.trainable_weights))
    expected = EXPECTED_ADAPTER_TRAINABLE_PARAMS[variant_id]
    if count != expected:
        raise ValueError(f"{variant_id}可训练参数{count} != {expected}")
    return count


def _compile_candidate(model):
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=CANDIDATE_LEARNING_RATE, clipnorm=1.0
        ),
        loss={
            "forecast_power": keras.losses.Huber(delta=1.0),
            "candidate_forecast": keras.losses.Huber(delta=1.0),
        },
        loss_weights={"forecast_power": 0.0, "candidate_forecast": 1.0},
        metrics={
            "candidate_forecast": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ]
        },
    )


def _history_frame(history, checkpoint):
    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", np.arange(len(frame), dtype=int))
    trace = pd.DataFrame(checkpoint.records)
    if len(frame) != len(trace):
        raise ValueError("history与candidate checkpoint轨迹长度不一致")
    frame["selection_val_candidate_nrmse"] = trace[
        "candidate_validation_nrmse"
    ].to_numpy()
    frame["selection_val_frozen_g0_fused_nrmse"] = trace[
        "frozen_g0_fused_validation_nrmse"
    ].to_numpy()
    return frame, trace


def _metric_column(frame, suffix, validation=False):
    prefix = "val_" if validation else ""
    preferred = f"{prefix}candidate_forecast_{suffix}"
    if preferred in frame:
        return preferred
    values = [
        name
        for name in frame.columns
        if name.startswith(prefix) and name.endswith(suffix)
    ]
    return values[0] if values else None


def _plot_history(frame, path, title):
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    specs = (("loss", "Loss"), ("mae", "MAE"), ("rmse", "RMSE"))
    for axis, (suffix, label) in zip(axes, specs):
        train_col = _metric_column(frame, suffix, validation=False)
        val_col = _metric_column(frame, suffix, validation=True)
        if train_col:
            axis.plot(
                frame["epoch"], frame[train_col], marker="o", markersize=3,
                label="train",
            )
        if val_col:
            axis.plot(
                frame["epoch"], frame[val_col], marker="o", markersize=3,
                label="validation",
            )
        axis.set_title(label)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(title)
    figure.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _canonical_probe(prepared, count=4):
    features = np.asarray(prepared["features"], dtype=np.float32)
    if len(features) < HISTORY_LEN + count:
        raise ValueError("训练数据不足以生成固定probe")
    return np.stack(
        [features[index : index + HISTORY_LEN] for index in range(count)], axis=0
    )


def _assert_source_compatible(prepared, artifact):
    gate_train._validate_prepared_against_source(prepared, artifact)
    if list(artifact.get("selected_regime_feature_groups", ())) != ["P", "H", "D"]:
        raise ValueError("Stage-5A来源F7不是P+H+D")


def _source_state(variant_id, prepared):
    farm_id = str(prepared["farm_id"])
    source_model, artifact, artifact_path, model_path = gate_train.load_source_f7(
        farm_id
    )
    _assert_source_compatible(prepared, artifact)
    model = build_multiscale_model(variant_id, artifact)
    copied = _copy_source_weights(source_model, model)
    probe = _canonical_probe(prepared)
    source_output = _source_diagnostic_model(source_model)(probe, training=False)
    target_output = diagnostic_model(model)(probe, training=False)
    differences = {}
    mapping = {
        "persistence": "persistence",
        "corrected": "corrected",
        "gate": "gate",
        "forecast": "forecast",
    }
    for source_name, target_name in mapping.items():
        difference = float(
            np.max(
                np.abs(
                    np.asarray(source_output[source_name])
                    - np.asarray(target_output[target_name])
                )
            )
        )
        differences[target_name] = difference
        if difference > SOURCE_RECONSTRUCTION_MAX_ABS_TOL:
            raise ValueError(
                f"{variant_id}零初始化未复现F7 {target_name}: {difference}"
            )
    if float(np.max(np.abs(np.asarray(target_output["delta"])))) != 0.0:
        raise ValueError(f"{variant_id} residual delta未零初始化")
    source_snapshot = _array_sha256(
        _weighted_snapshot(source_model, SOURCE_WEIGHTED_LAYER_NAMES)
    )
    target_snapshot = _array_sha256(
        _weighted_snapshot(model, SOURCE_WEIGHTED_LAYER_NAMES)
    )
    if source_snapshot != target_snapshot:
        raise ValueError("F7 residual/context/gate权重快照复制hash不一致")
    scale_initial_snapshots = {
        scale: _array_sha256(
            _weighted_snapshot(model, SCALE_WEIGHTED_LAYER_NAMES[scale])
        )
        for scale in VARIANT_SPECS[variant_id]["scales"]
    }
    return {
        "source_model": source_model,
        "source_artifact": artifact,
        "source_artifact_path": os.path.abspath(artifact_path),
        "source_model_path": os.path.abspath(model_path),
        "model": model,
        "probe": probe,
        "copied_layers": copied,
        "initial_outputs": {
            name: np.asarray(value) for name, value in target_output.items()
        },
        "source_snapshot_sha256": source_snapshot,
        "initial_snapshot_sha256": target_snapshot,
        "scale_initial_snapshot_sha256": scale_initial_snapshots,
        "zero_initialization_max_abs": differences,
    }


def dependency_code_records():
    modules = {
        "feature_screen_train": feature_train,
        "regime_encoder_train": regime_train,
        "controlled_gate_train": gate_train,
        "stage4b_train": stage4b_train,
        "time_freq_train": time_freq_train,
        "common_dl_train": common_train,
    }
    return {
        name: _file_record(os.path.realpath(module.__file__))
        for name, module in modules.items()
    }


def validate_dependency_code_records(records, role="artifact"):
    """Validate every dependency path/hash against the current code checkout."""
    if not isinstance(records, dict) or not records:
        raise ValueError(f"{role}缺少dependency_code_records")
    current = dependency_code_records()
    if set(records) != set(current):
        raise ValueError(f"{role}依赖代码集合漂移")
    for name, expected in current.items():
        recorded = records[name]
        if (
            os.path.realpath(str(recorded.get("path", "")))
            != os.path.realpath(expected["path"])
            or recorded.get("sha256") != expected["sha256"]
            or _sha256(recorded.get("path")) != recorded.get("sha256")
        ):
            raise ValueError(f"{role}依赖代码漂移: {name}")
    return True


def _dependency_code_records():
    """Backward-compatible private alias."""
    return dependency_code_records()


def train_variant_for_farm(
    variant_id,
    prepared,
    result_root=None,
    candidate_epochs=CANDIDATE_EPOCHS,
):
    if variant_id not in TRAINABLE_VARIANTS:
        raise ValueError(f"禁止训练{variant_id}")
    keras.backend.clear_session()
    configure_reproducibility()
    farm_id = str(prepared["farm_id"])
    state = _source_state(variant_id, prepared)
    model = state["model"]
    source_model = state["source_model"]
    plain_train, plain_val, train_samples, total_samples = _plain_datasets(prepared)
    train_ds = _attach_training_targets(plain_train)
    val_ds = _attach_training_targets(plain_val)
    dirs = variant_dirs(variant_id, result_root=result_root)
    paths = _paths(dirs, variant_id, farm_id)
    if os.path.exists(paths["weights_path"]):
        os.remove(paths["weights_path"])

    source_snapshot_before = _array_sha256(
        _weighted_snapshot(model, SOURCE_WEIGHTED_LAYER_NAMES)
    )
    source_f7_snapshot_before = _array_sha256(
        _weighted_snapshot(model, gate_train.COMMON_WEIGHTED_LAYER_NAMES)
    )
    source_g0_snapshot_before = _array_sha256(
        _weighted_snapshot(model, ("correction_gate",))
    )
    source_probe_before = {
        name: np.asarray(value)
        for name, value in diagnostic_model(model)(state["probe"], training=False).items()
        if name in {"persistence", "base_corrected", "gate"}
    }
    source_f7_probe_before = _array_sha256(
        [
            ("persistence", source_probe_before["persistence"]),
            ("base_corrected", source_probe_before["base_corrected"]),
        ]
    )
    source_g0_probe_before = _array_sha256(
        [("gate", source_probe_before["gate"])]
    )
    candidate_before = np.asarray(state["initial_outputs"]["corrected"])
    adapter_snapshot_before = _array_sha256(
        _weighted_snapshot(model, ADAPTER_WEIGHTED_LAYER_NAMES[variant_id])
    )
    trainable_params = _set_candidate_phase(model, variant_id)
    _compile_candidate(model)
    checkpoint = CandidateValidationCheckpoint(
        model, paths["weights_path"], plain_val, prepared, variant_id
    )
    finite_guard = feature_train.NonFiniteTrainingGuard()
    start = time.monotonic()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(candidate_epochs),
        callbacks=[
            finite_guard,
            keras.callbacks.TerminateOnNaN(),
            checkpoint,
            keras.callbacks.EarlyStopping(
                monitor="selection_val_candidate_nrmse",
                mode="min",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=False,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="selection_val_candidate_nrmse",
                mode="min",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(
                    dirs["tensorboard"],
                    f"farm_{farm_id}",
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                ),
                histogram_freq=0,
                profile_batch=0,
            ),
        ],
        verbose=1,
    )
    elapsed = float(time.monotonic() - start)
    feature_train.ensure_finite_training_history(history, finite_guard)
    if not os.path.isfile(paths["weights_path"]):
        raise FileNotFoundError("candidate checkpoint未生成")
    model.load_weights(paths["weights_path"])
    history_frame, checkpoint_trace = _history_frame(history, checkpoint)
    _atomic_to_csv(history_frame, paths["history_path"])
    _atomic_to_csv(checkpoint_trace, paths["checkpoint_path"])
    _plot_history(
        history_frame,
        paths["history_figure_path"],
        f"{VARIANT_SPECS[variant_id]['label']} | farm={farm_id}",
    )

    source_snapshot_after = _array_sha256(
        _weighted_snapshot(model, SOURCE_WEIGHTED_LAYER_NAMES)
    )
    source_f7_snapshot_after = _array_sha256(
        _weighted_snapshot(model, gate_train.COMMON_WEIGHTED_LAYER_NAMES)
    )
    source_g0_snapshot_after = _array_sha256(
        _weighted_snapshot(model, ("correction_gate",))
    )
    if source_f7_snapshot_after != source_f7_snapshot_before:
        raise ValueError("candidate训练改变了冻结F7 residual/context")
    if source_g0_snapshot_after != source_g0_snapshot_before:
        raise ValueError("candidate训练改变了冻结G0 gate")
    if source_snapshot_after != source_snapshot_before:
        raise ValueError("candidate训练改变了冻结F7 residual/context/G0 gate")
    final_probe = diagnostic_model(model)(state["probe"], training=False)
    frozen_probe_drifts = {
        name: float(
            np.max(np.abs(np.asarray(final_probe[name]) - source_probe_before[name]))
        )
        for name in source_probe_before
    }
    source_f7_probe_after = _array_sha256(
        [
            ("persistence", np.asarray(final_probe["persistence"])),
            ("base_corrected", np.asarray(final_probe["base_corrected"])),
        ]
    )
    source_g0_probe_after = _array_sha256(
        [("gate", np.asarray(final_probe["gate"]))]
    )
    if source_f7_probe_after != source_f7_probe_before:
        raise ValueError("candidate训练改变了冻结F7 persistence/base candidate probe")
    if source_g0_probe_after != source_g0_probe_before:
        raise ValueError("candidate训练改变了冻结G0 gate probe")
    if any(value != 0.0 for value in frozen_probe_drifts.values()):
        raise ValueError(f"冻结Persistence/G0 gate输出漂移: {frozen_probe_drifts}")
    candidate_after = np.asarray(final_probe["corrected"])
    candidate_probe_change = float(np.max(np.abs(candidate_after - candidate_before)))
    adapter_snapshot_after = _array_sha256(
        _weighted_snapshot(model, ADAPTER_WEIGHTED_LAYER_NAMES[variant_id])
    )

    validation = validation_diagnostics(model, plain_val, prepared, variant_id)
    _atomic_to_csv(validation, paths["validation_path"])
    provenance = pd.DataFrame(
        [
            {
                "variant_id": variant_id,
                "farm_id": farm_id,
                "source_f7_model_path": state["source_model_path"],
                "source_f7_model_sha256": _sha256(state["source_model_path"]),
                "source_snapshot_before_sha256": source_snapshot_before,
                "source_snapshot_after_sha256": source_snapshot_after,
                "adapter_snapshot_before_sha256": adapter_snapshot_before,
                "adapter_snapshot_after_sha256": adapter_snapshot_after,
                "persistence_probe_max_abs_drift": frozen_probe_drifts["persistence"],
                "g0_gate_probe_max_abs_drift": frozen_probe_drifts["gate"],
                "candidate_probe_max_abs_change": candidate_probe_change,
                "candidate_training_only": True,
                "test_used": False,
            }
        ]
    )
    _atomic_to_csv(provenance, paths["provenance_path"])
    prepared["train_df"].iloc[-HISTORY_LEN:].to_csv(paths["tail_path"], index=True)
    _save_model_atomic(model, paths["model_path"])
    restored = keras.models.load_model(
        paths["model_path"], custom_objects=get_multiscale_custom_objects(), compile=False
    )
    expected_output = diagnostic_model(model)(state["probe"], training=False)
    actual_output = diagnostic_model(restored)(state["probe"], training=False)
    for name in expected_output:
        if not np.allclose(expected_output[name], actual_output[name], rtol=1e-7, atol=1e-7):
            raise ValueError(f"保存/重载{name}输出不一致")

    total_params = int(model.count_params())
    if total_params != EXPECTED_TOTAL_PARAMS[variant_id] or total_params >= PARAMETER_LIMIT:
        raise ValueError(f"{variant_id}最终参数量异常: {total_params}")
    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "model_family": MODEL_FAMILY,
        "variant_id": variant_id,
        "variant_spec": dict(VARIANT_SPECS[variant_id]),
        "farm_id": farm_id,
        "random_seed": RANDOM_SEED,
        "history_len": HISTORY_LEN,
        "forecast_len": FORECAST_LEN,
        "time_freq": TIME_FREQ,
        "target_col": TARGET_COL,
        "train_file": os.path.abspath(prepared["train_file"]),
        "train_file_sha256": _sha256(prepared["train_file"]),
        "feature_cols": list(prepared["feature_cols"]),
        "input_cols": list(prepared["input_cols"]),
        "target_index": int(prepared["target_index"]),
        "scaler_x": prepared["scaler_x"],
        "scaler_y": prepared["scaler_y"],
        "capacity": float(prepared["capacity"]),
        "power_scale_ratio": float(prepared["power_scale_ratio"]),
        "power_scale_offset": float(prepared["power_scale_offset"]),
        "regime_feature_config": prepared["regime_feature_config"],
        "selected_regime_feature_groups": ["P", "H", "D"],
        "selected_regime_feature_names": list(
            feature_train.selected_feature_names(SOURCE_VARIANT)
        ),
        "selected_regime_feature_count": SOURCE_FEATURE_COUNT,
        "multiscale_definition": {
            "scales": list(VARIANT_SPECS[variant_id]["scales"]),
            "scale_specs": {
                key: dict(SCALE_SPECS[key])
                for key in VARIANT_SPECS[variant_id]["scales"]
            },
            "patch_projection": "stride1_causal_conv",
            "token_subsampling": "right_aligned_stride_minus_1_to_t95",
            "common_latent_dim": COMMON_LATENT_DIM,
            "patch_filters": PATCH_FILTERS,
            "static_fusion": VARIANT_SPECS[variant_id]["fusion"],
            "token_interaction": False,
            "same_scale_seed_shared_across_variants": True,
        },
        "candidate_training": {
            "epochs_requested": int(candidate_epochs),
            "epochs_actual": int(len(history_frame)),
            "learning_rate": CANDIDATE_LEARNING_RATE,
            "checkpoint_metric": "validation_candidate_physical_nrmse",
            "forecast_power_loss_weight": 0.0,
            "candidate_forecast_loss_weight": 1.0,
            "zero_initialized_delta_head": True,
            "f7_residual_context_g0_gate_frozen": True,
            "test_used": False,
        },
        "batch_size": BATCH_SIZE,
        "validation_split": VALIDATION_SPLIT,
        "selection_split": "test_in_prediction_script",
        "test_used_for_training": False,
        "test_is_final_blind_evaluation": False,
        "legacy_bidirectional_weather_imputation": True,
        "scaler_fit_scope": "full_train_file_including_validation",
        "validation_target_overlap_steps": FORECAST_LEN - 1,
        "source_f7_model_path": state["source_model_path"],
        "source_f7_model_sha256": _sha256(state["source_model_path"]),
        "source_f7_artifact_path": state["source_artifact_path"],
        "source_f7_artifact_sha256": _sha256(state["source_artifact_path"]),
        "copied_source_layer_names": state["copied_layers"],
        "source_snapshot_before_sha256": source_snapshot_before,
        "source_snapshot_after_sha256": source_snapshot_after,
        "source_snapshot_frozen_verified": True,
        "source_f7_snapshot_before_training_sha256": source_f7_snapshot_before,
        "source_f7_snapshot_after_training_sha256": source_f7_snapshot_after,
        "source_g0_snapshot_before_training_sha256": source_g0_snapshot_before,
        "source_g0_snapshot_after_training_sha256": source_g0_snapshot_after,
        "source_f7_probe_before_training_sha256": source_f7_probe_before,
        "source_f7_probe_after_training_sha256": source_f7_probe_after,
        "source_g0_probe_before_training_sha256": source_g0_probe_before,
        "source_g0_probe_after_training_sha256": source_g0_probe_after,
        "persistence_probe_max_abs_drift": frozen_probe_drifts["persistence"],
        "g0_gate_probe_max_abs_drift": frozen_probe_drifts["gate"],
        "candidate_probe_max_abs_change": candidate_probe_change,
        "candidate_probe_before_sha256": _array_sha256([("corrected", candidate_before)]),
        "candidate_probe_after_sha256": _array_sha256([("corrected", candidate_after)]),
        "adapter_snapshot_before_sha256": adapter_snapshot_before,
        "adapter_snapshot_after_sha256": adapter_snapshot_after,
        "scale_initial_snapshot_sha256": dict(
            state["scale_initial_snapshot_sha256"]
        ),
        "total_params": total_params,
        "adapter_trainable_params": trainable_params,
        "multiscale_trainable_parameter_count": trainable_params,
        "multiscale_added_parameter_count": total_params - EXPECTED_TOTAL_PARAMS["x0"],
        "parameter_limit": PARAMETER_LIMIT,
        "training_elapsed_seconds": elapsed,
        "train_samples": int(train_samples),
        "validation_samples": int(total_samples - train_samples),
        "best_validation_candidate_nrmse": float(checkpoint.best),
        "model_path": os.path.abspath(paths["model_path"]),
        "model_sha256": _sha256(paths["model_path"]),
        "best_weights_path": os.path.abspath(paths["weights_path"]),
        "best_weights_sha256": _sha256(paths["weights_path"]),
        "artifact_path": os.path.abspath(paths["artifact_path"]),
        "history_path": os.path.abspath(paths["history_path"]),
        "history_sha256": _sha256(paths["history_path"]),
        "history_figure_path": os.path.abspath(paths["history_figure_path"]),
        "history_figure_sha256": _sha256(paths["history_figure_path"]),
        "validation_path": os.path.abspath(paths["validation_path"]),
        "validation_sha256": _sha256(paths["validation_path"]),
        "checkpoint_path": os.path.abspath(paths["checkpoint_path"]),
        "checkpoint_sha256": _sha256(paths["checkpoint_path"]),
        "candidate_provenance_path": os.path.abspath(paths["provenance_path"]),
        "candidate_provenance_sha256": _sha256(paths["provenance_path"]),
        "tail_path": os.path.abspath(paths["tail_path"]),
        "tail_sha256": _sha256(paths["tail_path"]),
        "training_code_path": os.path.abspath(__file__),
        "training_code_sha256": _sha256(__file__),
        "dependency_code": dependency_code_records(),
        "dependency_code_records": dependency_code_records(),
    }
    _atomic_joblib_dump(artifact, paths["artifact_path"])
    row = validation.iloc[0].to_dict()
    row.update(
        {
            "model_family": MODEL_FAMILY,
            "variant_id": variant_id,
            "variant_label": VARIANT_SPECS[variant_id]["label"],
            "farm_id": farm_id,
            "reference_only": False,
            "requires_training": True,
            "random_seed": RANDOM_SEED,
            "feature_groups": SOURCE_FEATURE_GROUPS,
            "feature_count": SOURCE_FEATURE_COUNT,
            "scales": "+".join(VARIANT_SPECS[variant_id]["scales"]),
            "fusion": VARIANT_SPECS[variant_id]["fusion"],
            "token_interaction": False,
            "parameter_count": total_params,
            "adapter_trainable_parameter_count": trainable_params,
            "training_elapsed_seconds": elapsed,
            "actual_epoch_count": int(len(history_frame)),
            "best_validation_candidate_nrmse": float(checkpoint.best),
            "source_variant": "d0/g0/f7",
            "source_model_path": state["source_model_path"],
            "source_model_sha256": _sha256(state["source_model_path"]),
            "source_artifact_path": state["source_artifact_path"],
            "source_artifact_sha256": _sha256(state["source_artifact_path"]),
            "candidate_probe_max_abs_change": candidate_probe_change,
            "fine_initial_snapshot_sha256": state[
                "scale_initial_snapshot_sha256"
            ].get("fine"),
            "mid_initial_snapshot_sha256": state[
                "scale_initial_snapshot_sha256"
            ].get("mid"),
            "coarse_initial_snapshot_sha256": state[
                "scale_initial_snapshot_sha256"
            ].get("coarse"),
            "persistence_probe_max_abs_drift": frozen_probe_drifts["persistence"],
            "g0_gate_probe_max_abs_drift": frozen_probe_drifts["gate"],
            "model_path": os.path.abspath(paths["model_path"]),
            "model_sha256": _sha256(paths["model_path"]),
            "best_weights_path": os.path.abspath(paths["weights_path"]),
            "best_weights_sha256": _sha256(paths["weights_path"]),
            "artifact_path": os.path.abspath(paths["artifact_path"]),
            "artifact_sha256": _sha256(paths["artifact_path"]),
            "history_path": os.path.abspath(paths["history_path"]),
            "history_sha256": _sha256(paths["history_path"]),
            "history_figure_path": os.path.abspath(paths["history_figure_path"]),
            "history_figure_sha256": _sha256(paths["history_figure_path"]),
            "validation_path": os.path.abspath(paths["validation_path"]),
            "validation_sha256": _sha256(paths["validation_path"]),
            "checkpoint_path": os.path.abspath(paths["checkpoint_path"]),
            "checkpoint_sha256": _sha256(paths["checkpoint_path"]),
            "candidate_provenance_path": os.path.abspath(paths["provenance_path"]),
            "candidate_provenance_sha256": _sha256(paths["provenance_path"]),
            "tail_path": os.path.abspath(paths["tail_path"]),
            "tail_sha256": _sha256(paths["tail_path"]),
            "record_path": os.path.abspath(paths["record_path"]),
            "training_code_path": os.path.abspath(__file__),
            "training_code_sha256": _sha256(__file__),
            "result_source": "new_stage5a_candidate_training",
            "selection_split": "test_in_prediction_script",
        }
    )
    _atomic_write_json(row, paths["record_path"])
    del restored, source_model, model
    keras.backend.clear_session()
    return row


def build_x0_reference_rows(farm_ids):
    summary_path = os.path.join(stage4b_train.RESULT_ROOT, stage4b_train.TRAINING_SUMMARY_NAME)
    frame = pd.read_csv(summary_path, dtype={"farm_id": str})
    frame = frame[
        (frame["variant_id"].astype(str) == "d0")
        & frame["farm_id"].astype(str).isin([str(value) for value in farm_ids])
    ].copy()
    if len(frame) != len(farm_ids) or frame["farm_id"].nunique() != len(farm_ids):
        raise ValueError("X0/D0训练引用没有唯一覆盖请求场站")
    rows = []
    for _, source in frame.iterrows():
        model_path = source["source_model_path"]
        artifact_path = source["source_artifact_path"]
        if _sha256(model_path) != source["source_model_sha256"]:
            raise ValueError(f"X0来源模型hash漂移: {source['farm_id']}")
        if _sha256(artifact_path) != source["source_artifact_sha256"]:
            raise ValueError(f"X0来源artifact hash漂移: {source['farm_id']}")
        rows.append(
            {
                "model_family": MODEL_FAMILY,
                "variant_id": "x0",
                "variant_label": VARIANT_SPECS["x0"]["label"],
                "farm_id": str(source["farm_id"]),
                "reference_only": True,
                "requires_training": False,
                "random_seed": RANDOM_SEED,
                "feature_groups": SOURCE_FEATURE_GROUPS,
                "feature_count": SOURCE_FEATURE_COUNT,
                "scales": "none",
                "fusion": "existing_d0",
                "token_interaction": False,
                "parameter_count": EXPECTED_TOTAL_PARAMS["x0"],
                "adapter_trainable_parameter_count": 0,
                "source_variant": "stage4b_d0/g0/f7",
                "source_model_path": os.path.abspath(model_path),
                "source_model_sha256": source["source_model_sha256"],
                "source_artifact_path": os.path.abspath(artifact_path),
                "source_artifact_sha256": source["source_artifact_sha256"],
                "source_summary_path": os.path.abspath(summary_path),
                "source_summary_sha256": _sha256(summary_path),
                "result_source": "direct_reference_existing_stage4b_d0_no_training_no_copy",
                "selection_split": "test_in_prediction_script",
            }
        )
    return rows


def _validate_completed_record(record_path, variant_id, farm_id):
    if not os.path.isfile(record_path):
        return None
    with open(record_path, "r", encoding="utf-8") as file:
        row = json.load(file)
    if row.get("variant_id") != variant_id or str(row.get("farm_id")) != str(farm_id):
        raise ValueError(f"resume record身份不一致: {record_path}")
    for path_key, hash_key in (
        ("model_path", "model_sha256"),
        ("best_weights_path", "best_weights_sha256"),
        ("artifact_path", "artifact_sha256"),
        ("history_path", "history_sha256"),
        ("history_figure_path", "history_figure_sha256"),
        ("validation_path", "validation_sha256"),
        ("checkpoint_path", "checkpoint_sha256"),
        ("candidate_provenance_path", "candidate_provenance_sha256"),
        ("tail_path", "tail_sha256"),
    ):
        if _sha256(row.get(path_key)) != row.get(hash_key):
            raise ValueError(f"resume文件hash不一致: {path_key}")
    artifact = joblib.load(row["artifact_path"])
    if (
        artifact.get("protocol_version") != PROTOCOL_VERSION
        or artifact.get("architecture_version") != ARCHITECTURE_VERSION
        or artifact.get("model_sha256") != row.get("model_sha256")
        or artifact.get("source_snapshot_before_sha256")
        != artifact.get("source_snapshot_after_sha256")
    ):
        raise ValueError("resume artifact协议/模型身份/冻结证据不一致")
    current_code_hash = _sha256(__file__)
    if (
        row.get("training_code_sha256") != current_code_hash
        or artifact.get("training_code_sha256") != current_code_hash
    ):
        raise ValueError("resume由不同训练代码生成；请使用--force重训")
    validate_dependency_code_records(
        artifact.get("dependency_code_records"), role="resume artifact"
    )
    for path_key, hash_key in (
        ("source_f7_model_path", "source_f7_model_sha256"),
        ("source_f7_artifact_path", "source_f7_artifact_sha256"),
        ("train_file", "train_file_sha256"),
    ):
        if _sha256(artifact.get(path_key)) != artifact.get(hash_key):
            raise ValueError(f"resume上游来源hash漂移: {path_key}")
    if (
        int(artifact.get("total_params", -1))
        != EXPECTED_TOTAL_PARAMS[variant_id]
        or int(artifact.get("adapter_trainable_params", -1))
        != EXPECTED_ADAPTER_TRAINABLE_PARAMS[variant_id]
    ):
        raise ValueError("resume artifact参数协议漂移")
    return row


def write_manifest(result_root=RESULT_ROOT, run_scope="formal"):
    rows = []
    for order, (variant_id, spec) in enumerate(VARIANT_SPECS.items()):
        rows.append(
            {
                "variant_order": order,
                "variant_id": variant_id,
                "label": spec["label"],
                "requires_training": spec["requires_training"],
                "scales": "+".join(spec["scales"]) or "none",
                "patch_stride": ";".join(
                    f"{name}:{SCALE_SPECS[name]['patch']}/{SCALE_SPECS[name]['stride']}"
                    for name in spec["scales"]
                ) or "none",
                "common_latent_dim": COMMON_LATENT_DIM if spec["scales"] else None,
                "fusion": spec["fusion"],
                "token_interaction": False,
                "right_aligned_last_token_index": 95 if spec["scales"] else None,
                "same_scale_initializer_across_variants": True,
                "description": spec["description"],
                "source_candidate": "f7_persistence_plus_light_residual",
                "source_residual_context_gate_frozen": variant_id != "x0",
                "zero_initialized_candidate_delta": variant_id != "x0",
                "forecast_power_loss_weight": 0.0 if variant_id != "x0" else None,
                "candidate_forecast_loss_weight": 1.0 if variant_id != "x0" else None,
                "checkpoint_metric": "validation_candidate_nrmse" if variant_id != "x0" else None,
                "expected_total_params": EXPECTED_TOTAL_PARAMS[variant_id],
                "expected_adapter_trainable_params": EXPECTED_ADAPTER_TRAINABLE_PARAMS.get(variant_id, 0),
                "parameter_limit_exclusive": PARAMETER_LIMIT,
                "random_seed": RANDOM_SEED,
                "batch_size": BATCH_SIZE,
                "selection_split": "test",
                "test_used_for_selection": True,
                "test_is_final_blind_evaluation": False,
                "protocol_version": PROTOCOL_VERSION,
                "run_scope": run_scope,
            }
        )
    return _atomic_to_csv(pd.DataFrame(rows), os.path.join(result_root, MANIFEST_NAME))


def _validate_marker(path, expected_protocol, critical_keys):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少上游complete marker: {path}")
    with open(path, "r", encoding="utf-8") as file:
        marker = json.load(file)
    if marker.get("status") != "complete":
        raise ValueError(f"上游marker不是complete: {path}")
    if expected_protocol and marker.get("protocol_version") != expected_protocol:
        raise ValueError(f"上游marker协议漂移: {path}")
    records = marker.get("files", {})
    for key in critical_keys:
        record = records.get(key)
        if not isinstance(record, dict) or _sha256(record.get("path")) != record.get("sha256"):
            raise ValueError(f"上游marker关键文件hash漂移: {key}")
    for key, record in records.items():
        if not isinstance(record, dict) or _sha256(record.get("path")) != record.get("sha256"):
            raise ValueError(f"上游marker文件hash漂移: {key}")
    return marker


def validate_required_source_bundles():
    """Validate only the Stage-4B dependencies consumed during training.

    Stage-5A training rebuilds the frozen F7 candidate from the Stage-4B
    training bundle.  It does not read Stage-4B test predictions.  A formal
    Stage-4B training rerun deliberately invalidates/removes its downstream
    prediction marker, so requiring that marker here would incorrectly block
    otherwise valid Stage-5A training.  The Stage-5A prediction entry point
    remains responsible for validating the rebuilt Stage-4B test bundle.
    """
    training_path = os.path.join(stage4b_train.RESULT_ROOT, stage4b_train.TRAINING_MARKER_NAME)
    prediction_path = os.path.join(stage4b_train.RESULT_ROOT, stage4b_train.PREDICTION_MARKER_RELATIVE_PATH)
    _validate_marker(
        training_path,
        stage4b_train.PROTOCOL_VERSION,
        ("training_summary", "experiment_manifest", "source_stage4_training_marker"),
    )
    source_identity = {
        "stage4b_training_marker_path": os.path.abspath(training_path),
        "stage4b_training_marker_sha256": _sha256(training_path),
        "stage4b_prediction_marker_expected_path": os.path.abspath(prediction_path),
        "stage4b_prediction_marker_path": None,
        "stage4b_prediction_marker_sha256": None,
        "stage4b_prediction_bundle_status_at_training": (
            "missing_not_required_for_training_rebuild_before_prediction"
        ),
        "stage4b_prediction_bundle_required_for_training": False,
        "stage4b_prediction_bundle_required_for_test_selection": True,
    }
    if not os.path.isfile(prediction_path):
        print(
            "提示: Stage-4B预测complete marker当前不存在；这不影响Stage-5A训练。"
            "正式运行Stage-5A预测前，请先重新运行 "
            "wind_time_freq_model_stage4b_predict.py 发布与当前训练marker匹配的"
            "预测bundle。"
        )
        return source_identity

    prediction = _validate_marker(
        prediction_path,
        stage4b_train.PROTOCOL_VERSION,
        ("training_marker", "formal.summary", "formal.candidate", "formal.final_selection"),
    )
    record = prediction["files"]["training_marker"]
    if (
        os.path.realpath(record["path"]) != os.path.realpath(training_path)
        or record["sha256"] != _sha256(training_path)
    ):
        raise ValueError("Stage-4B预测marker没有锁定当前训练marker")
    source_identity.update(
        {
            "stage4b_prediction_marker_path": os.path.abspath(prediction_path),
            "stage4b_prediction_marker_sha256": _sha256(prediction_path),
            "stage4b_prediction_bundle_status_at_training": (
                "validated_and_locked_to_current_training_marker"
            ),
        }
    )
    return source_identity


def _validate_same_scale_initialization(summary):
    """Prove each single-scale arm starts from the same encoder as X1."""
    mapping = {
        "fine": "x1_f",
        "mid": "x1_m",
        "coarse": "x1_c",
    }
    for farm_id in expected_farm_ids():
        farm = summary[summary["farm_id"].astype(str) == str(farm_id)]
        combined = farm[farm["variant_id"] == "x1"]
        if len(combined) != 1:
            raise ValueError(f"{farm_id}缺少唯一X1初始化记录")
        combined = combined.iloc[0]
        for scale, single_variant in mapping.items():
            single = farm[farm["variant_id"] == single_variant]
            field = f"{scale}_initial_snapshot_sha256"
            if len(single) != 1:
                raise ValueError(f"{farm_id}缺少唯一{single_variant}初始化记录")
            single_hash = single.iloc[0].get(field)
            combined_hash = combined.get(field)
            if (
                not isinstance(single_hash, str)
                or not single_hash
                or single_hash != combined_hash
            ):
                raise ValueError(
                    f"{farm_id} {single_variant}与X1的{scale}初始化不一致"
                )
    return True


def publish_training_marker(summary_path, manifest_path, summary, source_identity):
    new_rows = summary[summary["variant_id"].isin(TRAINABLE_VARIANTS)]
    if len(new_rows) != len(TRAINABLE_VARIANTS) * len(expected_farm_ids()):
        raise ValueError("正式新训练矩阵不是4×5")
    _validate_same_scale_initialization(summary)
    files = {
        "training_summary": _file_record(summary_path),
        "experiment_manifest": _file_record(manifest_path),
        "training_code": _file_record(__file__),
        "source_stage4b_training_marker": _file_record(source_identity["stage4b_training_marker_path"]),
    }
    prediction_marker_path = source_identity.get("stage4b_prediction_marker_path")
    if prediction_marker_path:
        files["source_stage4b_prediction_marker"] = _file_record(
            prediction_marker_path
        )
    for name, record in dependency_code_records().items():
        files[f"dependency.{name}"] = record
    for _, row in new_rows.iterrows():
        prefix = f"{row['variant_id']}.{row['farm_id']}"
        for key in (
            "model_path", "best_weights_path", "artifact_path", "history_path",
            "history_figure_path", "validation_path", "checkpoint_path",
            "candidate_provenance_path", "tail_path", "record_path",
        ):
            files[f"{prefix}.{key}"] = _file_record(row[key])
    marker = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "expected_farm_ids": list(expected_farm_ids()),
        "variants": list(VARIANT_SPECS),
        "new_training_variants": list(TRAINABLE_VARIANTS),
        "new_model_count": int(len(new_rows)),
        "x0_reused_model_count": len(expected_farm_ids()),
        "x0_retraining_forbidden": True,
        "stage4b_prediction_bundle_required_for_training": False,
        "stage4b_prediction_bundle_required_for_test_selection": True,
        "stage4b_prediction_bundle_status_at_training": source_identity[
            "stage4b_prediction_bundle_status_at_training"
        ],
        "stage4b_prediction_marker_expected_path": source_identity[
            "stage4b_prediction_marker_expected_path"
        ],
        "source_f7_residual_context_g0_gate_frozen_verified": bool(
            (new_rows["persistence_probe_max_abs_drift"] == 0.0).all()
            and (new_rows["g0_gate_probe_max_abs_drift"] == 0.0).all()
        ),
        "same_scale_initialization_single_vs_x1_verified": True,
        "token_interaction_forbidden": True,
        "parameter_limit_exclusive": PARAMETER_LIMIT,
        "files": files,
    }
    return _atomic_write_json(marker, os.path.join(RESULT_ROOT, TRAINING_MARKER_NAME))


def _discover_train_files(requested_farms=None):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "wind_train_*.csv")))
    if requested_farms:
        farm_set = {str(value) for value in requested_farms}
        files = [
            path
            for path in files
            if re.search(r"wind_train_(\d+)\.csv$", os.path.basename(path)).group(1)
            in farm_set
        ]
    return files


def _parse_csv(value):
    return [item.strip().lower().replace("-", "_") for item in value.split(",") if item.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default=os.getenv("WIND_MULTISCALE_VARIANTS", ",".join(VARIANT_SPECS)),
        help="逗号分隔: x0,x1-f,x1-m,x1-c,x1",
    )
    parser.add_argument(
        "--farms", default=os.getenv("WIND_MULTISCALE_FARMS", ""), help="逗号分隔场站ID"
    )
    parser.add_argument("--candidate-epochs", type=int, default=CANDIDATE_EPOCHS)
    parser.add_argument("--epochs", type=int, default=None, help="调试epoch覆盖；自动进入partial")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _formal_protocol(args, variants, farm_ids):
    return (
        not args.smoke_test
        and args.epochs is None
        and set(variants) == set(VARIANT_SPECS)
        and set(farm_ids) == set(expected_farm_ids())
        and args.candidate_epochs == CANDIDATE_EPOCHS == 30
        and BATCH_SIZE == 192
        and np.isclose(VALIDATION_SPLIT, 0.15, rtol=0.0, atol=1e-12)
        and np.isclose(CANDIDATE_LEARNING_RATE, 1e-4, rtol=0.0, atol=1e-12)
        and EARLY_STOPPING_PATIENCE == 6
    )


def main(argv=None):
    args = parse_args(argv)
    configure_reproducibility()
    source_identity = validate_required_source_bundles()
    variants = list(dict.fromkeys(_parse_csv(args.variants)))
    invalid = sorted(set(variants) - set(VARIANT_SPECS))
    if invalid:
        raise ValueError(f"未知变体{invalid}; 可选{list(VARIANT_SPECS)}")
    farms = _parse_csv(args.farms) if args.farms else []
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs必须为正")
        args.candidate_epochs = args.epochs
    if args.smoke_test:
        variants = ["x1" if "x1" in variants else next((v for v in variants if v in TRAINABLE_VARIANTS), "x1")]
        farms = farms[:1] if farms else [expected_farm_ids()[0]]
        args.candidate_epochs = 1
    if BATCH_SIZE <= 0 or not 0.0 < VALIDATION_SPLIT < 1.0:
        raise ValueError("batch_size/validation_split无效")
    if args.candidate_epochs <= 0 or CANDIDATE_LEARNING_RATE <= 0:
        raise ValueError("candidate epochs/learning rate必须为正")
    train_files = _discover_train_files(farms)
    if not train_files:
        raise FileNotFoundError("没有匹配的训练文件")
    farm_ids = [regime_train.get_farm_id(path) for path in train_files]
    formal = _formal_protocol(args, variants, farm_ids)
    if formal:
        run_root = RESULT_ROOT
        run_scope = "formal"
        # 保留上一份complete训练marker，直到新bundle原子发布完成；running
        # marker阻止预测脚本在正式训练未完成时消费新旧混合文件。
        downstream = os.path.join(RESULT_ROOT, PREDICTION_MARKER_RELATIVE_PATH)
        if os.path.exists(downstream):
            os.remove(downstream)
        _atomic_write_json(
            {
                "status": "running",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "variants": variants,
                "farm_ids": farm_ids,
            },
            os.path.join(RESULT_ROOT, RUNNING_MARKER_NAME),
        )
    else:
        tag = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_root = os.path.join(RESULT_ROOT, "partial_runs", tag)
        run_scope = "smoke_or_partial_or_protocol_override"
    manifest_path = write_manifest(run_root, run_scope)
    print(
        f"Stage-5A场站={farm_ids}; 变体={variants}; 输出={run_root}; "
        f"formal={formal}; seed={RANDOM_SEED}; batch={BATCH_SIZE}"
    )
    print(f"上游Stage-4B bundle已验证: {source_identity}")
    rows = []
    if "x0" in variants:
        rows.extend(build_x0_reference_rows(farm_ids))
    trainable = [variant for variant in variants if variant in TRAINABLE_VARIANTS]
    for train_file in train_files:
        prepared = regime_train._prepare_farm(train_file)
        for variant_id in trainable:
            dirs = variant_dirs(variant_id, result_root=run_root)
            record_path = _paths(dirs, variant_id, str(prepared["farm_id"]))["record_path"]
            completed = None if args.force else _validate_completed_record(record_path, variant_id, prepared["farm_id"])
            if completed is not None:
                print(f"跳过已验证完成模型: {variant_id}/{prepared['farm_id']}")
                rows.append(completed)
                continue
            print(f"\n===== {VARIANT_SPECS[variant_id]['label']} / farm={prepared['farm_id']} =====")
            rows.append(
                train_variant_for_farm(
                    variant_id,
                    prepared,
                    result_root=run_root,
                    candidate_epochs=args.candidate_epochs,
                )
            )
    summary = pd.DataFrame(rows)
    if summary.empty or summary.duplicated(["variant_id", "farm_id"]).any():
        raise ValueError("训练/引用summary为空或存在重复键")
    summary_path = _atomic_to_csv(summary, os.path.join(run_root, TRAINING_SUMMARY_NAME))
    print(f"训练汇总: {summary_path}")
    if formal:
        expected_rows = len(VARIANT_SPECS) * len(expected_farm_ids())
        if len(summary) != expected_rows:
            raise ValueError(f"正式summary应为{expected_rows}行，实际{len(summary)}")
        marker_path = publish_training_marker(summary_path, manifest_path, summary, source_identity)
        running_path = os.path.join(RESULT_ROOT, RUNNING_MARKER_NAME)
        if os.path.exists(running_path):
            os.remove(running_path)
        print(f"正式训练bundle完成: {marker_path}")
        print("X0只读引用Stage-4B D0/F7，新增训练模型数=20")
    else:
        print("partial/smoke运行不覆盖正式summary，不发布complete marker")


if __name__ == "__main__":
    main()
