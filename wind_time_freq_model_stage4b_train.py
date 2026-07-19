"""Stage 4B：T1 candidate 到门控收益的受控闭环训练。

正式矩阵固定为五个变体：

* D0：只读引用 T0/G0/F7，不训练；
* D0R：冻结既有 T1 candidate，重新训练非因子化 direct gate；
* D1：冻结 F7 candidate，训练非因子化 calibrated-safe gate；
* D2：冻结与 D0R/D3 完全相同的 T1 candidate，训练 D1 同构门控；
* D3：冻结同一 T1 candidate，训练因子化 calibrated-dynamic-safe gate。

D0R 的 fixed-G0-on-T1 只作为门控训练前的零训练诊断，并不是第六个模型。
D1/D2/D3 从各自冻结 candidate 的 train-only 窗口生成 soft oracle 与逐
horizon |C-P| Q90。任何 partial/smoke/epoch override 都写入 partial_runs，
不会覆盖正式结果或发布 complete marker。
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
from tensorflow.keras import layers

import wind_RegimeEncoder_PatchTST_feature_screen_train as feature_train
import wind_RegimeEncoder_PatchTST_train as regime_train
import wind_controlled_gate_cali_train as gate_train
import wind_dl_model_train as common_train
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


MODEL_FAMILY = "time_freq_stage4b_gate_closure"
ARCHITECTURE_VERSION = "stage4b_gate_closure_d0_d3_v1"
ARTIFACT_SCHEMA_VERSION = 1
PROTOCOL_VERSION = "stage4b_gate_closure_test_selected_v1"
RESULT_ROOT = os.path.join(
    "./wind_results", "time_freq_model", "supplement_round2_stage4b_gate_closure"
)
SOURCE_FEATURE_GROUPS = "P+H+D"
SOURCE_FEATURE_COUNT = 36
RANDOM_SEED = 2026

BATCH_SIZE = int(os.getenv("WIND_STAGE4B_BATCH_SIZE", "192"))
VALIDATION_SPLIT = float(os.getenv("WIND_STAGE4B_VALIDATION_SPLIT", "0.15"))
GATE_ONLY_EPOCHS = int(os.getenv("WIND_STAGE4B_GATE_ONLY_EPOCHS", "3"))
CONTEXT_EPOCHS = int(os.getenv("WIND_STAGE4B_CONTEXT_EPOCHS", "5"))
OBJECTIVE_EPOCHS = int(os.getenv("WIND_STAGE4B_OBJECTIVE_EPOCHS", "30"))
INITIAL_LR = float(os.getenv("WIND_STAGE4B_INITIAL_LR", "0.0001"))
OBJECTIVE_LR = float(os.getenv("WIND_STAGE4B_OBJECTIVE_LR", "0.00005"))
PATIENCE = int(os.getenv("WIND_STAGE4B_PATIENCE", "6"))
PARAMETER_LIMIT = 30000
SOURCE_RECONSTRUCTION_MAX_ABS_TOL = 1e-4
CONTEXT_RECONSTRUCTION_MAX_ABS_TOL = 1e-6

CALIBRATION_WEIGHT = gate_train.CALIBRATION_WEIGHT
DYNAMIC_WEIGHT = gate_train.DYNAMIC_WEIGHT
SAFETY_WEIGHT = gate_train.SAFETY_WEIGHT
SOFT_ORACLE_TEMPERATURE = gate_train.SOFT_ORACLE_TEMPERATURE
CALIBRATION_DIFFERENCE_QUANTILE = gate_train.CALIBRATION_DIFFERENCE_QUANTILE

VARIANT_SPECS = {
    "d0": {
        "label": "D0 T0/G0/F7 direct reference",
        "requires_training": False,
        "candidate_source": "f7",
        "factorized_gate": False,
        "calibration_weight": 0.0,
        "dynamic_weight": 0.0,
        "safety_weight": 0.0,
        "initial_gate": "existing_g0",
        "contrast": "formal reference",
    },
    "d0r": {
        "label": "D0R frozen T1 + retrained non-factorized direct gate",
        "requires_training": True,
        "candidate_source": "t1",
        "factorized_gate": False,
        "calibration_weight": 0.0,
        "dynamic_weight": 0.0,
        "safety_weight": 0.0,
        "initial_gate": "0.95",
        "contrast": "D0R - fixed replay = gate retraining effect",
    },
    "d1": {
        "label": "D1 frozen F7 + non-factorized calibrated-safe gate",
        "requires_training": True,
        "candidate_source": "f7",
        "factorized_gate": False,
        "calibration_weight": CALIBRATION_WEIGHT,
        "dynamic_weight": 0.0,
        "safety_weight": SAFETY_WEIGHT,
        "initial_gate": "own_train_soft_oracle_clipped_mean",
        "contrast": "D2 - D1 = candidate effect",
    },
    "d2": {
        "label": "D2 frozen T1 + non-factorized calibrated-safe gate",
        "requires_training": True,
        "candidate_source": "t1",
        "factorized_gate": False,
        "calibration_weight": CALIBRATION_WEIGHT,
        "dynamic_weight": 0.0,
        "safety_weight": SAFETY_WEIGHT,
        "initial_gate": "own_train_soft_oracle_clipped_mean",
        "contrast": "D2 - D0R = auxiliary objective effect",
    },
    "d3": {
        "label": "D3 frozen T1 + factorized calibrated-dynamic-safe gate",
        "requires_training": True,
        "candidate_source": "t1",
        "factorized_gate": True,
        "calibration_weight": CALIBRATION_WEIGHT,
        "dynamic_weight": DYNAMIC_WEIGHT,
        "safety_weight": SAFETY_WEIGHT,
        "initial_gate": "factorized_0.45",
        "contrast": "D3 - D2 = factorization + dynamic auxiliary joint effect",
    },
}
TRAINABLE_VARIANTS = ("d0r", "d1", "d2", "d3")
REFERENCE_VARIANTS = ("d0",)
T1_CANDIDATE_VARIANTS = ("d0r", "d2", "d3")
CALIBRATED_VARIANTS = ("d1", "d2", "d3")

EXPECTED_TOTAL_PARAMS = {"d0": 20969, "d0r": 24121, "d1": 20969, "d2": 24121, "d3": 23561}
EXPECTED_ADAPTER_TRAINABLE_PARAMS = {key: 0 for key in TRAINABLE_VARIANTS}
EXPECTED_CANDIDATE_STRUCTURAL_PARAMS = {"d0": 18416, "d0r": 21568, "d1": 18416, "d2": 21568, "d3": 21568}
EXPECTED_GATE_TRAINABLE_PARAMS = {
    "d0r": {"gate_only": 993, "context": 2553, "objective": 2553},
    "d1": {"gate_only": 993, "context": 2553, "objective": 2553},
    "d2": {"gate_only": 993, "context": 2553, "objective": 2553},
    "d3": {"gate_only": 433, "context": 1993, "objective": 1993},
}

TRAINING_SUMMARY_NAME = "stage4b_gate_closure_training_metrics.csv"
MANIFEST_NAME = "stage4b_gate_closure_experiment_manifest.csv"
TRAINING_MARKER_NAME = "stage4b_gate_closure_training_bundle_complete.json"
RUNNING_MARKER_NAME = "stage4b_gate_closure_training_bundle_running.json"
PREDICTION_MARKER_RELATIVE_PATH = os.path.join(
    "testdata_predict_output", "stage4b_gate_closure_test_bundle_complete.json"
)


def configure_reproducibility():
    os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_SEED))
    set_global_seed(RANDOM_SEED)
    keras.utils.set_random_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def expected_farm_ids():
    farms = tuple(str(value) for value in feature_train.expected_training_farm_ids())
    if len(farms) != 5:
        raise ValueError(f"正式来源场站数不是5: {farms}")
    return farms


def variant_model_name(variant_id):
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"未知Stage-4B变体: {variant_id}")
    return f"{MODEL_FAMILY}_{variant_id}"


def variant_dirs(variant_id, create=True, result_root=None):
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"未知Stage-4B变体: {variant_id}")
    root = os.path.join(RESULT_ROOT if result_root is None else result_root, variant_id)
    values = {
        "root": root,
        "models": os.path.join(root, "models"),
        "weights": os.path.join(root, "weights"),
        "preprocess": os.path.join(root, "preprocess"),
        "history": os.path.join(root, "history"),
        "tensorboard": os.path.join(root, "tensorboard"),
        "tails": os.path.join(root, "tails"),
        "validation_diagnostics": os.path.join(root, "validation_diagnostics"),
        "pretrain_diagnostics": os.path.join(root, "pretrain_diagnostics"),
        "records": os.path.join(root, "records"),
    }
    if create:
        for path in values.values():
            os.makedirs(path, exist_ok=True)
    return values


def get_stage4b_custom_objects():
    return dict(time_freq_train.get_time_freq_custom_objects())


def get_time_freq_custom_objects():
    """Compatibility alias used by the shared Stage-4 prediction helpers."""
    return get_stage4b_custom_objects()


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


def dependency_code_records():
    """Return the exact implementation files that define a Stage-4B model."""
    modules = {
        "feature_screen_train": feature_train,
        "regime_encoder_train": regime_train,
        "controlled_gate_train": gate_train,
        "time_freq_train": time_freq_train,
        "common_dl_train": common_train,
    }
    return {name: _file_record(module.__file__) for name, module in modules.items()}


def validate_dependency_code_records(records, role="Stage-4B artifact"):
    expected = dependency_code_records()
    if not isinstance(records, dict) or set(records) != set(expected):
        raise ValueError(f"{role}依赖代码集合不完整")
    for name, current in expected.items():
        recorded = records[name]
        if (
            not isinstance(recorded, dict)
            or os.path.realpath(str(recorded.get("path", "")))
            != os.path.realpath(current["path"])
            or recorded.get("sha256") != current["sha256"]
        ):
            raise ValueError(f"{role}依赖代码漂移: {name}")
    return True


def _weighted_snapshot(model, layer_names):
    return time_freq_train._weighted_snapshot(model, layer_names)


def _copy_layers(source, target, names, role):
    copied = []
    for name in names:
        source_values = source.get_layer(name).get_weights()
        target_layer = target.get_layer(name)
        if [x.shape for x in source_values] != [x.shape for x in target_layer.get_weights()]:
            raise ValueError(f"{role}层{name}形状不一致")
        target_layer.set_weights(source_values)
        if any(not np.array_equal(a, b) for a, b in zip(source_values, target_layer.get_weights())):
            raise ValueError(f"{role}层{name}未精确复制")
        copied.append(name)
    return copied


def _plain_datasets(prepared):
    return make_window_dataset(
        prepared["features"], prepared["target"], HISTORY_LEN, FORECAST_LEN,
        BATCH_SIZE, VALIDATION_SPLIT,
    )


def _attach_targets(dataset):
    def attach(x, y):
        return x, {"forecast_power": y, "candidate_forecast": y, "control_packet": y}
    return dataset.map(attach, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True).prefetch(tf.data.AUTOTUNE)


def diagnostic_model(model):
    packet = model.get_layer("control_packet").output
    h = FORECAST_LEN
    return keras.Model(model.inputs, {
        "forecast": model.get_layer("forecast_power").output,
        "persistence": packet[:, h:2*h],
        "corrected": packet[:, 2*h:3*h],
        "gate": packet[:, :h],
        "q": packet[:, 4*h:5*h],
        "s": packet[:, 5*h:6*h],
    })


def _candidate_layer_names(source_kind):
    names = list(regime_train.B2_WEIGHTED_LAYER_NAMES)
    if source_kind == "t1":
        names.extend(time_freq_train.ADAPTER_WEIGHTED_LAYER_NAMES["t1"])
    return tuple(names)


def _canonical_probe(prepared, count=2):
    features = np.asarray(prepared["features"], dtype=np.float32)
    if len(features) < HISTORY_LEN + count:
        raise ValueError("训练数据不足以生成固定candidate probe")
    return np.stack([features[i:i + HISTORY_LEN] for i in range(count)], axis=0)


def _load_t1_source(farm_id):
    summary_path = os.path.join(time_freq_train.RESULT_ROOT, time_freq_train.TRAINING_SUMMARY_NAME)
    frame = pd.read_csv(summary_path, dtype={"farm_id": str})
    row = frame[(frame["variant_id"] == "t1") & (frame["farm_id"] == str(farm_id))]
    if len(row) != 1:
        raise ValueError(f"T1/{farm_id}正式训练summary不是唯一一行")
    row = row.iloc[0]
    for path_key, hash_key in (("model_path", "model_sha256"), ("artifact_path", "artifact_sha256")):
        if _sha256(row[path_key]) != row[hash_key]:
            raise ValueError(f"T1/{farm_id}来源{path_key} hash漂移")
    artifact = joblib.load(row["artifact_path"])
    if artifact.get("variant_id") != "t1" or artifact.get("protocol_version") != time_freq_train.PROTOCOL_VERSION:
        raise ValueError(f"T1/{farm_id} artifact协议不兼容")
    if (
        os.path.realpath(str(artifact.get("model_path", "")))
        != os.path.realpath(str(row["model_path"]))
        or artifact.get("model_sha256") != row["model_sha256"]
        or artifact.get("candidate_snapshot_before_gate_sha256")
        != artifact.get("candidate_snapshot_after_gate_sha256")
        or artifact.get("candidate_output_before_gate_sha256")
        != artifact.get("candidate_output_after_gate_sha256")
        or int(artifact.get("total_params", -1))
        != time_freq_train.EXPECTED_TOTAL_PARAMS["t1"]
    ):
        raise ValueError(f"T1/{farm_id} artifact未锁定冻结candidate或正式模型")
    model = keras.models.load_model(row["model_path"], custom_objects=time_freq_train.get_time_freq_custom_objects(), compile=False)
    if int(model.count_params()) != time_freq_train.EXPECTED_TOTAL_PARAMS["t1"]:
        raise ValueError(f"T1/{farm_id}模型参数量漂移")
    return model, artifact, os.path.abspath(row["artifact_path"]), os.path.abspath(row["model_path"])


def _source_candidate_output(model, source_kind):
    name = "time_freq_corrected_candidate" if source_kind == "t1" else "corrected_forecast_candidate"
    return model.get_layer(name).output


def build_stage4b_model(variant_id, source_artifact, initial_gate_weight):
    if variant_id not in TRAINABLE_VARIANTS:
        raise ValueError(f"禁止构建引用变体{variant_id}")
    configure_reproducibility()
    source_kind = VARIANT_SPECS[variant_id]["candidate_source"]
    if source_kind == "t1":
        template = time_freq_train.build_time_freq_model("t1", source_artifact)
        corrected = template.get_layer("time_freq_corrected_candidate").output
    else:
        template = feature_train.build_feature_screen_model_from_artifact(source_artifact)
        corrected = template.get_layer("corrected_forecast_candidate").output
    persistence = template.get_layer("persistence_forecast_candidate").output
    context = template.get_layer("regime_context").output
    if VARIANT_SPECS[variant_id]["factorized_gate"]:
        gate, q_by_horizon, horizon_prior = gate_train._build_factorized_gate(context)
    else:
        gate = regime_train.SampleHorizonCorrectionGate(
            forecast_len=FORECAST_LEN,
            hidden_dim=feature_train.GATE_HIDDEN_DIM,
            horizon_embedding_dim=feature_train.HORIZON_EMBEDDING_DIM,
            dropout=feature_train.GATE_DROPOUT,
            initial_weight=float(initial_gate_weight),
            name="controlled_gate",
        )(context)
        q = gate_train.HorizonGateMean(name="sample_dynamic_probability")(gate)
        q_by_horizon = gate_train.BroadcastScalarToHorizon(
            FORECAST_LEN, name="sample_dynamic_probability_by_horizon"
        )(q)
        horizon_prior = gate_train.OnesHorizonPrior(name="horizon_gate_prior")(gate)
    forecast = regime_train.TwoCandidateGateFusion(name="forecast_power")([persistence, corrected, gate])
    candidate = layers.Activation("linear", name="candidate_forecast")(corrected)
    packet = layers.Concatenate(name="control_packet")([
        gate, persistence, corrected, forecast, q_by_horizon, horizon_prior
    ])
    model = keras.Model(template.inputs, {
        "forecast_power": forecast,
        "candidate_forecast": candidate,
        "control_packet": packet,
    }, name=f"Stage4BGateClosure_{variant_id.upper()}")
    expected = EXPECTED_TOTAL_PARAMS[variant_id]
    if int(model.count_params()) != expected:
        raise ValueError(f"{variant_id}参数量{model.count_params()} != {expected}")
    return model


def _initialize_candidate_and_context(model, source_kind, f7_model, t1_model=None):
    """Use F7 B2/context for every variant and only T1 adapter for T1 variants."""
    copied = {
        "f7_b2": _copy_layers(
            f7_model, model, regime_train.B2_WEIGHTED_LAYER_NAMES, "F7 B2"
        ),
        "f7_context": _copy_layers(
            f7_model, model, gate_train.CONTEXT_WEIGHTED_LAYER_NAMES, "F7 context"
        ),
        "t1_adapter": [],
    }
    if source_kind == "t1":
        if t1_model is None:
            raise ValueError("T1 candidate缺少来源模型")
        copied["t1_adapter"] = _copy_layers(
            t1_model,
            model,
            time_freq_train.ADAPTER_WEIGHTED_LAYER_NAMES["t1"],
            "T1 adapter",
        )
    return copied


def _candidate_snapshot(model, source_kind):
    return _array_sha256(_weighted_snapshot(model, _candidate_layer_names(source_kind)))


def _context_snapshot(model):
    return _array_sha256(
        _weighted_snapshot(model, gate_train.CONTEXT_WEIGHTED_LAYER_NAMES)
    )


def _probe_outputs(model, probe):
    values = diagnostic_model(model)(probe, training=False)
    return {name: np.asarray(value) for name, value in values.items()}


def _inverse_scaled(values, prepared):
    shape = np.asarray(values).shape
    physical = (
        prepared["scaler_y"]
        .inverse_transform(np.asarray(values).reshape(-1, 1))
        .reshape(shape)
    )
    return np.clip(physical, 0.0, float(prepared["capacity"]))


def _ece(probability, truth, bins=10):
    probability = np.asarray(probability, dtype=float)
    truth = np.asarray(truth, dtype=float)
    ids = np.minimum((np.clip(probability, 0.0, 1.0) * bins).astype(int), bins - 1)
    value = 0.0
    for bin_id in range(bins):
        mask = ids == bin_id
        if mask.any():
            value += float(mask.mean()) * abs(
                float(probability[mask].mean()) - float(truth[mask].mean())
            )
    return float(value)


def validation_diagnostics(model, dataset, prepared, variant_id):
    diagnostic = diagnostic_model(model)
    outputs = {key: [] for key in ("forecast", "persistence", "corrected", "gate", "q", "s")}
    truths = []
    for batch_x, batch_y in dataset:
        result = diagnostic(batch_x, training=False)
        truths.append(np.asarray(batch_y))
        for key in outputs:
            outputs[key].append(np.asarray(result[key]))
    if not truths:
        raise ValueError("验证集为空")
    truth = _inverse_scaled(np.concatenate(truths), prepared)
    values = {key: np.concatenate(parts) for key, parts in outputs.items()}
    forecast = _inverse_scaled(values["forecast"], prepared)
    persistence = _inverse_scaled(values["persistence"], prepared)
    corrected = _inverse_scaled(values["corrected"], prepared)
    gate = values["gate"]
    valid = (
        np.isfinite(truth) & np.isfinite(forecast) & np.isfinite(persistence)
        & np.isfinite(corrected) & np.isfinite(gate)
    )
    if not valid.any():
        raise FloatingPointError("验证输出没有有限元素")
    capacity = float(prepared["capacity"])
    error = forecast[valid] - truth[valid]
    corrected_error = corrected[valid] - truth[valid]
    p_abs = np.abs(persistence[valid] - truth[valid]) / capacity
    f_abs = np.abs(error) / capacity
    oracle = np.abs(corrected_error) < np.abs(persistence[valid] - truth[valid])
    regret = np.maximum(0.0, f_abs - p_abs)
    row = {
        "variant_id": variant_id,
        "farm_id": str(prepared["farm_id"]),
        "valid_count": int(valid.sum()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "capacity_normalized_mae": float(np.mean(np.abs(error)) / capacity),
        "capacity_normalized_rmse": float(np.sqrt(np.mean(np.square(error))) / capacity),
        "corrected_capacity_normalized_rmse": float(
            np.sqrt(np.mean(np.square(corrected_error))) / capacity
        ),
        "gate_mean": float(np.mean(gate)),
        "gate_std": float(np.std(gate)),
        "gate_low_saturation_rate": float(np.mean(gate < 0.05)),
        "gate_high_saturation_rate": float(np.mean(gate > 0.95)),
        "q_mean": float(np.mean(values["q"])),
        "s_mean": float(np.mean(values["s"])),
        "positive_regret_mean": float(np.mean(regret)),
        "harm_rate_0_005": float(np.mean((f_abs - p_abs) > 0.005)),
        "oracle_brier": float(np.mean(np.square(gate[valid] - oracle.astype(float)))),
        "ece_10bin": _ece(gate[valid], oracle),
        "diagnostic_scope": "validation_checkpoint_only_not_test_selection",
    }
    return pd.DataFrame([row])


def estimate_train_only_calibration(model, train_ds, prepared):
    diagnostic = diagnostic_model(model)
    oracle_sum = 0.0
    count = 0
    differences = []
    for batch_x, batch_y in train_ds:
        output = diagnostic(batch_x, training=False)
        truth = gate_train._scaled_to_capacity_fraction(batch_y.numpy(), prepared)
        persistence = gate_train._scaled_to_capacity_fraction(
            output["persistence"].numpy(), prepared
        )
        corrected = gate_train._scaled_to_capacity_fraction(
            output["corrected"].numpy(), prepared
        )
        e_p = np.abs(truth - persistence)
        e_c = np.abs(truth - corrected)
        advantage = (e_p - e_c) / (e_p + e_c + 1e-8)
        oracle = 1.0 / (1.0 + np.exp(-advantage / SOFT_ORACLE_TEMPERATURE))
        oracle_sum += float(oracle.sum())
        count += int(oracle.size)
        differences.append(np.abs(corrected - persistence))
    if not differences or count == 0:
        raise ValueError("训练窗口为空，无法生成soft oracle/Q90")
    difference = np.concatenate(differences, axis=0)
    q90 = np.quantile(
        difference, CALIBRATION_DIFFERENCE_QUANTILE, axis=0
    ).astype(np.float32)
    if q90.shape != (FORECAST_LEN,) or not np.isfinite(q90).all():
        raise FloatingPointError("train-only Q90无效")
    raw_mean = float(oracle_sum / count)
    return {
        "applicable": True,
        "soft_oracle_mean": raw_mean,
        "soft_oracle_mean_clipped": float(np.clip(raw_mean, 0.05, 0.95)),
        "candidate_difference_q90": q90,
        "candidate_difference_q90_sha256": _array_sha256([("q90", q90)]),
        "quantile": CALIBRATION_DIFFERENCE_QUANTILE,
        "sample_count": int(difference.shape[0]),
        "element_count": int(count),
        "scope": "per_farm_per_horizon_train_frozen_candidate",
        "future_truth_role": "train_target_only",
    }


def _not_applicable_calibration():
    return {
        "applicable": False,
        "reason": "D0R direct gate has calibration/dynamic/safety weights all zero",
        "soft_oracle_mean": None,
        "soft_oracle_mean_clipped": None,
        "candidate_difference_q90": [],
        "candidate_difference_q90_sha256": None,
        "quantile": None,
        "sample_count": None,
        "element_count": None,
        "scope": "not_applicable",
        "future_truth_role": "not_used_by_direct_gate_objective",
    }


class GateCheckpoint(keras.callbacks.Callback):
    """Choose validation checkpoint by NRMSE, then regret/Brier within 0.1%."""

    def __init__(self, path, val_ds, prepared, variant_id):
        super().__init__()
        self.path = path
        self.val_ds = val_ds
        self.prepared = prepared
        self.variant_id = variant_id
        self.phase = None
        self.records = []
        self.snapshots = []
        self.best = np.inf
        self.best_regret = np.inf
        self.best_brier = np.inf
        self.best_phase = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = validation_diagnostics(
            self.model, self.val_ds, self.prepared, self.variant_id
        ).iloc[0].to_dict()
        logs["selection_val_nrmse"] = row["capacity_normalized_rmse"]
        logs["selection_val_positive_regret"] = row["positive_regret_mean"]
        logs["selection_val_brier"] = row["oracle_brier"]
        self.records.append({
            "global_epoch": len(self.records), "phase": self.phase,
            "phase_epoch": int(epoch), **row,
        })
        # ``model.get_weights()`` can be reordered when layers move between the
        # trainable/non-trainable groups across phases.  Snapshot by stable
        # top-level layer identity so a gate-only checkpoint can be restored
        # safely while the model is currently in the objective phase.
        self.snapshots.append(
            {
                layer.name: [
                    np.array(value, copy=True) for value in layer.get_weights()
                ]
                for layer in self.model.layers
                if layer.weights
            }
        )

    def finalize(self):
        frame = pd.DataFrame(self.records)
        if frame.empty or len(frame) != len(self.snapshots):
            raise ValueError("门控checkpoint轨迹不完整")
        minimum = float(frame["capacity_normalized_rmse"].min())
        eligible = frame[frame["capacity_normalized_rmse"] <= minimum * 1.001]
        index = int(eligible.sort_values(
            ["positive_regret_mean", "oracle_brier", "capacity_normalized_rmse", "global_epoch"],
            kind="stable",
        ).index[0])
        selected = frame.loc[index]
        snapshot = self.snapshots[index]
        current_names = {layer.name for layer in self.model.layers if layer.weights}
        if set(snapshot) != current_names:
            raise ValueError("跨phase checkpoint层集合发生漂移")
        for layer in self.model.layers:
            if layer.name in snapshot:
                layer.set_weights(snapshot[layer.name])
        self.model.save_weights(self.path)
        self.best = float(selected["capacity_normalized_rmse"])
        self.best_regret = float(selected["positive_regret_mean"])
        self.best_brier = float(selected["oracle_brier"])
        self.best_phase = str(selected["phase"])
        frame["selected_checkpoint"] = frame.index == index
        return frame


def _set_gate_phase(model, variant_id, phase):
    if phase not in {"gate_only", "context", "objective"}:
        raise ValueError(f"未知phase: {phase}")
    for layer in model.layers:
        layer.trainable = False
    for name in (
        "controlled_gate", "sample_dynamic_hidden", "sample_dynamic_dropout",
        "sample_dynamic_probability", "sample_dynamic_probability_by_horizon",
        "horizon_gate_prior",
    ):
        try:
            model.get_layer(name).trainable = True
        except ValueError:
            pass
    if phase in {"context", "objective"}:
        for name in gate_train.CONTEXT_WEIGHTED_LAYER_NAMES:
            model.get_layer(name).trainable = True
    model.get_layer("residual_dropout").rate = 0.0
    model.get_layer("regime_context_dropout").rate = (
        0.0 if phase == "gate_only" else float(feature_train.GATE_DROPOUT)
    )
    count = int(sum(int(np.prod(weight.shape)) for weight in model.trainable_weights))
    expected = EXPECTED_GATE_TRAINABLE_PARAMS[variant_id][phase]
    if count != expected:
        raise ValueError(f"{variant_id}/{phase}可训练参数{count} != {expected}")
    return count


def _compile(model, variant_id, prepared, q90, learning_rate):
    spec = VARIANT_SPECS[variant_id]
    q90 = (
        np.zeros(FORECAST_LEN, dtype=np.float32)
        if q90 is None or np.asarray(q90).size == 0
        else q90
    )
    auxiliary = gate_train.ControlledGateAuxiliaryLoss(
        forecast_len=FORECAST_LEN,
        target_mean=float(prepared["scaler_y"].mean_[0]),
        target_scale=float(prepared["scaler_y"].scale_[0]),
        capacity=float(prepared["capacity"]),
        calibration_weight=spec["calibration_weight"],
        dynamic_weight=spec["dynamic_weight"],
        safety_weight=spec["safety_weight"],
        candidate_difference_q90=q90,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss={
            "forecast_power": keras.losses.Huber(delta=1.0),
            "candidate_forecast": keras.losses.Huber(delta=1.0),
            "control_packet": auxiliary,
        },
        loss_weights={
            "forecast_power": 1.0,
            "candidate_forecast": 0.0,
            "control_packet": 1.0,
        },
        metrics={"forecast_power": [
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ]},
    )


def _paths(dirs, variant_id, farm_id):
    prefix = f"{variant_model_name(variant_id)}_farm_{farm_id}"
    return {
        "model_path": os.path.join(dirs["models"], f"{prefix}.keras"),
        "weights_path": os.path.join(dirs["weights"], f"{prefix}_best.weights.h5"),
        "artifact_path": os.path.join(dirs["preprocess"], f"{prefix}_preprocess.pkl"),
        "history_path": os.path.join(dirs["history"], f"{prefix}_gate_history.csv"),
        "history_figure_path": os.path.join(dirs["history"], f"{prefix}_gate_history.png"),
        "validation_path": os.path.join(dirs["validation_diagnostics"], f"{prefix}_validation.csv"),
        "checkpoint_path": os.path.join(dirs["validation_diagnostics"], f"{prefix}_checkpoint_trace.csv"),
        "pretrain_path": os.path.join(dirs["pretrain_diagnostics"], f"{prefix}_fixed_g0_replay.csv"),
        "candidate_provenance_path": os.path.join(dirs["pretrain_diagnostics"], f"{prefix}_candidate_provenance.csv"),
        "tail_path": os.path.join(dirs["tails"], f"{prefix}_tail.csv"),
        "record_path": os.path.join(dirs["records"], f"{prefix}_record.json"),
    }


def _history_frame(histories):
    rows = []
    global_epoch = 0
    for phase, history in histories:
        count = len(next(iter(history.history.values()))) if history.history else 0
        for index in range(count):
            row = {"global_epoch": global_epoch, "phase": phase, "phase_epoch": index}
            row.update({key: values[index] for key, values in history.history.items()})
            rows.append(row)
            global_epoch += 1
    return pd.DataFrame(rows)


def _metric_column(frame, suffix, validation=False):
    prefix = "val_" if validation else ""
    preferred = f"{prefix}forecast_power_{suffix}"
    if preferred in frame:
        return preferred
    matches = [c for c in frame if c.startswith(prefix) and c.endswith(suffix)]
    return matches[0] if matches else None


def _plot_history(frame, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    panels = (("loss", "Loss"), ("mae", "MAE"), ("rmse", "RMSE"))
    x = frame["global_epoch"] + 1
    for axis, (metric, label) in zip(axes, panels):
        train_col = metric if metric == "loss" else _metric_column(frame, metric)
        val_col = f"val_{metric}" if metric == "loss" else _metric_column(frame, metric, True)
        if train_col in frame:
            axis.plot(x, frame[train_col], label="train", linewidth=1.8)
        if val_col in frame:
            axis.plot(x, frame[val_col], label="validation", linewidth=1.8)
        for boundary in frame.groupby("phase", sort=False)["global_epoch"].max().iloc[:-1]:
            axis.axvline(float(boundary) + 1.5, color="grey", linestyle="--", alpha=0.35)
        axis.set_title(label)
        axis.set_xlabel("Global epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_model_atomic(model, path):
    return time_freq_train._save_model_atomic(model, path)


def _source_state(variant_id, prepared, train_ds):
    farm_id = str(prepared["farm_id"])
    f7_model, f7_artifact, f7_artifact_path, f7_model_path = gate_train.load_source_f7(farm_id)
    gate_train._validate_prepared_against_source(prepared, f7_artifact)
    source_kind = VARIANT_SPECS[variant_id]["candidate_source"]
    t1_model = t1_artifact = t1_artifact_path = t1_model_path = None
    if source_kind == "t1":
        t1_model, t1_artifact, t1_artifact_path, t1_model_path = _load_t1_source(farm_id)
        gate_train._validate_prepared_against_source(prepared, t1_artifact)
        if (
            t1_artifact.get("source_f7_model_sha256") != _sha256(f7_model_path)
            or t1_artifact.get("source_f7_artifact_sha256")
            != _sha256(f7_artifact_path)
        ):
            raise ValueError(
                f"T1/{farm_id} adapter来源没有锁定当前F7 candidate/context"
            )
    # 架构模板始终由兼容的F7 artifact重建；T1 artifact只作来源验证，
    # 且只复制adapter权重，绝不继承T1最终门控训练后的context。
    build_artifact = f7_artifact

    initial_gate = 0.95 if variant_id == "d0r" else gate_train.FACTORIZED_INITIAL_GATE
    model = build_stage4b_model(variant_id, build_artifact, initial_gate)
    copied = _initialize_candidate_and_context(model, source_kind, f7_model, t1_model)

    probe = _canonical_probe(prepared)
    target_probe = _probe_outputs(model, probe)
    candidate_source_model = t1_model if source_kind == "t1" else f7_model
    source_candidate = keras.Model(
        candidate_source_model.inputs,
        _source_candidate_output(candidate_source_model, source_kind),
    )(probe, training=False)
    source_persistence = keras.Model(
        candidate_source_model.inputs,
        candidate_source_model.get_layer("persistence_forecast_candidate").output,
    )(probe, training=False)
    candidate_difference = float(np.max(np.abs(target_probe["corrected"] - source_candidate)))
    persistence_difference = float(np.max(np.abs(target_probe["persistence"] - source_persistence)))
    if (
        candidate_difference > SOURCE_RECONSTRUCTION_MAX_ABS_TOL
        or persistence_difference > SOURCE_RECONSTRUCTION_MAX_ABS_TOL
    ):
        raise ValueError(
            f"{variant_id}未复现冻结{source_kind} candidate: "
            f"corrected={candidate_difference}, persistence={persistence_difference}"
        )
    source_context = keras.Model(
        f7_model.inputs, f7_model.get_layer("regime_context").output
    )(probe, training=False)
    target_context = keras.Model(
        model.inputs, model.get_layer("regime_context").output
    )(probe, training=False)
    context_difference = float(np.max(np.abs(source_context - target_context)))
    if context_difference > CONTEXT_RECONSTRUCTION_MAX_ABS_TOL:
        raise ValueError(f"{variant_id}初始context未精确复现F7: {context_difference}")

    calibration = (
        estimate_train_only_calibration(model, train_ds, prepared)
        if variant_id in CALIBRATED_VARIANTS
        else _not_applicable_calibration()
    )
    # D1/D2的非因子化gate必须由各自candidate的train-only oracle mean初始化。
    if variant_id in {"d1", "d2"}:
        model = build_stage4b_model(
            variant_id, build_artifact, calibration["soft_oracle_mean_clipped"]
        )
        copied = _initialize_candidate_and_context(model, source_kind, f7_model, t1_model)
        rebuilt_probe = _probe_outputs(model, probe)
        if not np.array_equal(rebuilt_probe["corrected"], target_probe["corrected"]):
            raise ValueError(f"{variant_id}按oracle mean重建后candidate发生漂移")
        target_probe = rebuilt_probe

    candidate_snapshot = _candidate_snapshot(model, source_kind)
    context_snapshot = _context_snapshot(model)
    adapter_names = time_freq_train.ADAPTER_WEIGHTED_LAYER_NAMES["t1"]
    adapter_snapshot = (
        _array_sha256(_weighted_snapshot(model, adapter_names))
        if source_kind == "t1" else None
    )
    calibration.update(
        {
            "candidate_source": source_kind,
            "candidate_snapshot_sha256": candidate_snapshot,
            "candidate_probe_sha256": _array_sha256(
                [("corrected", target_probe["corrected"])]
            ),
        }
    )
    return {
        "model": model,
        "f7_model": f7_model,
        "f7_artifact": f7_artifact,
        "f7_artifact_path": f7_artifact_path,
        "f7_model_path": f7_model_path,
        "t1_model": t1_model,
        "t1_artifact": t1_artifact,
        "t1_artifact_path": t1_artifact_path,
        "t1_model_path": t1_model_path,
        "source_kind": source_kind,
        "build_artifact": build_artifact,
        "copied": copied,
        "probe": probe,
        "probe_outputs": target_probe,
        "candidate_snapshot_sha256": candidate_snapshot,
        "candidate_probe_sha256": _array_sha256([("corrected", target_probe["corrected"])]),
        "f7_context_snapshot_sha256": context_snapshot,
        "t1_adapter_snapshot_sha256": adapter_snapshot,
        "calibration": calibration,
        "candidate_source_probe_max_abs_difference": candidate_difference,
        "persistence_source_probe_max_abs_difference": persistence_difference,
        "f7_context_probe_max_abs_difference": context_difference,
    }


def _fixed_g0_on_t1_replay(model, f7_model, val_ds, prepared, variant_id):
    """Evaluate frozen original G0 gate on T1 candidate before D0R retraining."""
    if variant_id != "d0r":
        return pd.DataFrame([{
            "variant_id": variant_id,
            "farm_id": str(prepared["farm_id"]),
            "applicable": False,
            "diagnostic_role": "not_applicable_not_a_sixth_model",
        }])
    current = diagnostic_model(model)
    g0 = gate_train._source_diagnostic_model(f7_model)
    truth_parts, p_parts, c_parts, gate_parts = [], [], [], []
    for batch_x, batch_y in val_ds:
        output = current(batch_x, training=False)
        source = g0(batch_x, training=False)
        truth_parts.append(np.asarray(batch_y))
        p_parts.append(np.asarray(output["persistence"]))
        c_parts.append(np.asarray(output["corrected"]))
        gate_parts.append(np.asarray(source["gate"]))
    truth = _inverse_scaled(np.concatenate(truth_parts), prepared)
    persistence = _inverse_scaled(np.concatenate(p_parts), prepared)
    corrected = _inverse_scaled(np.concatenate(c_parts), prepared)
    gate = np.concatenate(gate_parts)
    fused = persistence + gate * (corrected - persistence)
    error = fused - truth
    capacity = float(prepared["capacity"])
    return pd.DataFrame([{
        "variant_id": variant_id,
        "farm_id": str(prepared["farm_id"]),
        "applicable": True,
        "diagnostic_role": "fixed_original_g0_gate_on_frozen_t1_candidate_pretrain_only_not_variant",
        "capacity_normalized_mae": float(np.mean(np.abs(error)) / capacity),
        "capacity_normalized_rmse": float(np.sqrt(np.mean(np.square(error))) / capacity),
        "gate_mean": float(np.mean(gate)),
        "gate_std": float(np.std(gate)),
        "selection_eligible": False,
    }])


def train_variant_for_farm(
    variant_id,
    prepared,
    result_root=None,
    gate_only_epochs=GATE_ONLY_EPOCHS,
    context_epochs=CONTEXT_EPOCHS,
    objective_epochs=OBJECTIVE_EPOCHS,
):
    if variant_id not in TRAINABLE_VARIANTS:
        raise ValueError(f"禁止训练{variant_id}")
    keras.backend.clear_session()
    configure_reproducibility()
    farm_id = str(prepared["farm_id"])
    plain_train, plain_val, train_samples, total_samples = _plain_datasets(prepared)
    state = _source_state(variant_id, prepared, plain_train)
    model = state["model"]
    dirs = variant_dirs(variant_id, result_root=result_root)
    paths = _paths(dirs, variant_id, farm_id)
    if os.path.exists(paths["weights_path"]):
        os.remove(paths["weights_path"])

    before_candidate_snapshot = _candidate_snapshot(model, state["source_kind"])
    before_context_snapshot = _context_snapshot(model)
    before_probe = _probe_outputs(model, state["probe"])
    calibration = state["calibration"]
    if calibration["applicable"] and calibration["sample_count"] != int(train_samples):
        raise ValueError(f"{variant_id}/{farm_id} Q90不是由完整train-only窗口生成")

    replay = _fixed_g0_on_t1_replay(
        model, state["f7_model"], plain_val, prepared, variant_id
    )
    _atomic_to_csv(replay, paths["pretrain_path"])
    provenance = pd.DataFrame([{
        "variant_id": variant_id,
        "farm_id": farm_id,
        "candidate_source": state["source_kind"],
        "candidate_frozen": True,
        "candidate_snapshot_sha256": before_candidate_snapshot,
        "candidate_probe_sha256": state["candidate_probe_sha256"],
        "t1_adapter_snapshot_sha256": state["t1_adapter_snapshot_sha256"],
        "f7_context_snapshot_sha256": before_context_snapshot,
        "source_t1_model_path": state["t1_model_path"],
        "source_t1_model_sha256": _sha256(state["t1_model_path"]),
        "source_f7_model_path": state["f7_model_path"],
        "source_f7_model_sha256": _sha256(state["f7_model_path"]),
        "soft_oracle_q90_applicable": calibration["applicable"],
        "candidate_difference_q90_sha256": calibration["candidate_difference_q90_sha256"],
    }])
    _atomic_to_csv(provenance, paths["candidate_provenance_path"])

    train_ds = _attach_targets(plain_train)
    val_ds = _attach_targets(plain_val)
    checkpoint = GateCheckpoint(
        paths["weights_path"], plain_val, prepared, variant_id
    )
    histories = []
    phase_trainable = {}
    phase_specs = (
        ("gate_only", int(gate_only_epochs), INITIAL_LR),
        ("context", int(context_epochs), INITIAL_LR),
        ("objective", int(objective_epochs), OBJECTIVE_LR),
    )
    started = time.monotonic()
    for phase, epochs, learning_rate in phase_specs:
        if epochs <= 0:
            continue
        phase_trainable[phase] = _set_gate_phase(model, variant_id, phase)
        _compile(
            model,
            variant_id,
            prepared,
            calibration["candidate_difference_q90"],
            learning_rate,
        )
        checkpoint.phase = phase
        finite_guard = feature_train.NonFiniteTrainingGuard()
        callbacks = [
            finite_guard,
            keras.callbacks.TerminateOnNaN(),
            checkpoint,
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(
                    dirs["tensorboard"], f"farm_{farm_id}",
                    datetime.now().strftime("%Y%m%d-%H%M%S"), phase,
                ),
                histogram_freq=0,
                profile_batch=0,
            ),
        ]
        if phase == "objective":
            callbacks.extend([
                keras.callbacks.EarlyStopping(
                    monitor="selection_val_nrmse", mode="min", patience=PATIENCE,
                    restore_best_weights=False, verbose=1,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="selection_val_nrmse", mode="min", factor=0.5,
                    patience=3, min_lr=1e-6, verbose=1,
                ),
            ])
        history = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs,
            callbacks=callbacks, verbose=1,
        )
        feature_train.ensure_finite_training_history(history, finite_guard)
        histories.append((phase, history))
    checkpoint_trace = checkpoint.finalize()
    model.load_weights(paths["weights_path"])
    elapsed = float(time.monotonic() - started)
    history_frame = _history_frame(histories)
    if len(history_frame) != len(checkpoint_trace):
        raise ValueError("history和checkpoint轨迹长度不一致")
    _atomic_to_csv(history_frame, paths["history_path"])
    _atomic_to_csv(checkpoint_trace, paths["checkpoint_path"])
    _plot_history(
        history_frame, paths["history_figure_path"],
        f"Stage 4B {variant_id.upper()} farm {farm_id}",
    )

    after_candidate_snapshot = _candidate_snapshot(model, state["source_kind"])
    if after_candidate_snapshot != before_candidate_snapshot:
        raise ValueError(f"{variant_id}门控训练改变了冻结candidate权重")
    after_probe = _probe_outputs(model, state["probe"])
    candidate_probe_drift = float(
        np.max(np.abs(after_probe["corrected"] - before_probe["corrected"]))
    )
    persistence_probe_drift = float(
        np.max(np.abs(after_probe["persistence"] - before_probe["persistence"]))
    )
    if candidate_probe_drift != 0.0 or persistence_probe_drift != 0.0:
        raise ValueError(
            f"{variant_id}门控训练改变candidate输出: "
            f"corrected={candidate_probe_drift}, persistence={persistence_probe_drift}"
        )

    validation = validation_diagnostics(model, plain_val, prepared, variant_id)
    _atomic_to_csv(validation, paths["validation_path"])
    prepared["train_df"].iloc[-HISTORY_LEN:].to_csv(paths["tail_path"], index=True)
    _save_model_atomic(model, paths["model_path"])
    restored = keras.models.load_model(
        paths["model_path"], custom_objects=get_stage4b_custom_objects(), compile=False
    )
    restored_probe = _probe_outputs(restored, state["probe"])
    for key, expected in after_probe.items():
        if not np.allclose(expected, restored_probe[key], rtol=1e-7, atol=1e-7):
            raise ValueError(f"{variant_id}保存/重载{key}不一致")
    total_params = int(model.count_params())
    if total_params != EXPECTED_TOTAL_PARAMS[variant_id] or total_params >= PARAMETER_LIMIT:
        raise ValueError(f"{variant_id}最终参数量异常: {total_params}")

    calibration_artifact = dict(calibration)
    if isinstance(calibration_artifact.get("candidate_difference_q90"), np.ndarray):
        calibration_artifact["candidate_difference_q90"] = calibration_artifact[
            "candidate_difference_q90"
        ].tolist()
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
        "feature_cols": prepared["feature_cols"],
        "input_cols": prepared["input_cols"],
        "target_index": prepared["target_index"],
        "scaler_x": prepared["scaler_x"],
        "scaler_y": prepared["scaler_y"],
        "capacity": float(prepared["capacity"]),
        "power_scale_ratio": prepared["power_scale_ratio"],
        "power_scale_offset": prepared["power_scale_offset"],
        "regime_feature_config": prepared["regime_feature_config"],
        "selected_regime_feature_groups": ["P", "H", "D"],
        "selected_regime_feature_names": list(feature_train.selected_feature_names("f7")),
        "selected_regime_feature_count": SOURCE_FEATURE_COUNT,
        "candidate_source": state["source_kind"],
        "candidate_frozen_all_phases": True,
        "candidate_snapshot_before_gate_sha256": before_candidate_snapshot,
        "candidate_snapshot_after_gate_sha256": after_candidate_snapshot,
        "candidate_output_before_gate_sha256": _array_sha256([("corrected", before_probe["corrected"])]),
        "candidate_output_after_gate_sha256": _array_sha256([("corrected", after_probe["corrected"])]),
        "candidate_gate_max_abs_drift": candidate_probe_drift,
        "candidate_gate_calibration_max_abs_drift": candidate_probe_drift,
        "persistence_gate_max_abs_drift": persistence_probe_drift,
        "source_t1_adapter_snapshot_sha256": state["t1_adapter_snapshot_sha256"],
        "source_f7_context_snapshot_sha256": before_context_snapshot,
        "source_f7_context_initial_only": True,
        "source_t1_final_context_inherited": False,
        "candidate_calibration": calibration_artifact,
        "gate_training": {
            "topology": "factorized_pi=q_i*s_h" if VARIANT_SPECS[variant_id]["factorized_gate"] else "nonfactorized_sample_horizon",
            "candidate_frozen_all_phases": True,
            "calibration_weight": VARIANT_SPECS[variant_id]["calibration_weight"],
            "dynamic_weight": VARIANT_SPECS[variant_id]["dynamic_weight"],
            "safety_weight": VARIANT_SPECS[variant_id]["safety_weight"],
            "phases": [
                {"phase": p, "epochs": e, "learning_rate": lr, "trainable_parameter_count": phase_trainable.get(p)}
                for p, e, lr in phase_specs
            ],
        },
        "fixed_g0_replay": {
            "applicable": variant_id == "d0r",
            "selection_eligible": False,
            "role": "pretrain_diagnostic_not_sixth_model",
            "path": os.path.abspath(paths["pretrain_path"]),
            "sha256": _sha256(paths["pretrain_path"]),
        },
        "batch_size": BATCH_SIZE,
        "validation_split": VALIDATION_SPLIT,
        "selection_split": "test_in_prediction_script",
        "test_used_for_training": False,
        "test_is_final_blind_evaluation": False,
        "legacy_bidirectional_weather_imputation": True,
        "scaler_fit_scope": "full_train_file_including_validation",
        "validation_target_overlap_steps": FORECAST_LEN - 1,
        "source_f7_model_path": os.path.abspath(state["f7_model_path"]),
        "source_f7_model_sha256": _sha256(state["f7_model_path"]),
        "source_f7_artifact_path": os.path.abspath(state["f7_artifact_path"]),
        "source_f7_artifact_sha256": _sha256(state["f7_artifact_path"]),
        "source_t1_model_path": os.path.abspath(state["t1_model_path"]) if state["t1_model_path"] else None,
        "source_t1_model_sha256": _sha256(state["t1_model_path"]),
        "source_t1_artifact_path": os.path.abspath(state["t1_artifact_path"]) if state["t1_artifact_path"] else None,
        "source_t1_artifact_sha256": _sha256(state["t1_artifact_path"]),
        "training_code_path": os.path.abspath(__file__),
        "training_code_sha256": _sha256(__file__),
        "dependency_code_records": dependency_code_records(),
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
        "candidate_provenance_path": os.path.abspath(paths["candidate_provenance_path"]),
        "candidate_provenance_sha256": _sha256(paths["candidate_provenance_path"]),
        "tail_path": os.path.abspath(paths["tail_path"]),
        "tail_sha256": _sha256(paths["tail_path"]),
        "total_params": total_params,
        "parameter_limit": PARAMETER_LIMIT,
        "train_samples": int(train_samples),
        "validation_samples": int(total_samples - train_samples),
        "training_elapsed_seconds": elapsed,
        "gate_training_elapsed_seconds": elapsed,
        "best_validation_nrmse": checkpoint.best,
        "best_validation_positive_regret": checkpoint.best_regret,
        "best_validation_brier": checkpoint.best_brier,
        "best_phase": checkpoint.best_phase,
    }
    _atomic_joblib_dump(artifact, paths["artifact_path"])
    row = validation.iloc[0].to_dict()
    row.update({
        "model_family": MODEL_FAMILY,
        "variant_id": variant_id,
        "variant_label": VARIANT_SPECS[variant_id]["label"],
        "farm_id": farm_id,
        "feature_groups": SOURCE_FEATURE_GROUPS,
        "feature_count": SOURCE_FEATURE_COUNT,
        "reference_only": False,
        "requires_training": True,
        "candidate_source": state["source_kind"],
        "candidate_snapshot_sha256": before_candidate_snapshot,
        "candidate_probe_sha256": state["candidate_probe_sha256"],
        "t1_adapter_snapshot_sha256": state["t1_adapter_snapshot_sha256"],
        "f7_context_snapshot_sha256": before_context_snapshot,
        "candidate_difference_q90_sha256": calibration["candidate_difference_q90_sha256"],
        "soft_oracle_q90_applicable": calibration["applicable"],
        "soft_oracle_mean": calibration["soft_oracle_mean"],
        "soft_oracle_mean_clipped": calibration["soft_oracle_mean_clipped"],
        "soft_oracle_sample_count": calibration["sample_count"],
        "soft_oracle_element_count": calibration["element_count"],
        "parameter_count": total_params,
        "random_seed": RANDOM_SEED,
        "training_elapsed_seconds": elapsed,
        "best_validation_nrmse": checkpoint.best,
        "best_phase": checkpoint.best_phase,
        "model_path": artifact["model_path"],
        "model_sha256": artifact["model_sha256"],
        "best_weights_path": artifact["best_weights_path"],
        "best_weights_sha256": artifact["best_weights_sha256"],
        "artifact_path": artifact["artifact_path"],
        "artifact_sha256": _sha256(paths["artifact_path"]),
        "history_path": artifact["history_path"],
        "history_figure_path": artifact["history_figure_path"],
        "validation_path": artifact["validation_path"],
        "checkpoint_path": artifact["checkpoint_path"],
        "pretrain_diagnostic_path": artifact["fixed_g0_replay"]["path"],
        "candidate_provenance_path": artifact["candidate_provenance_path"],
        "tail_path": artifact["tail_path"],
        "record_path": os.path.abspath(paths["record_path"]),
        "training_code_path": os.path.abspath(__file__),
        "training_code_sha256": _sha256(__file__),
        "result_source": "new_stage4b_training",
        "selection_split": "test_in_prediction_script",
    })
    _atomic_write_json(row, paths["record_path"])
    del restored, model
    keras.backend.clear_session()
    return row


def build_d0_reference_rows(farm_ids):
    rows = time_freq_train.build_t0_reference_rows(farm_ids)
    result = []
    for source in rows:
        row = dict(source)
        row.update({
            "model_family": MODEL_FAMILY,
            "variant_id": "d0",
            "variant_label": VARIANT_SPECS["d0"]["label"],
            "candidate_source": "f7",
            "reference_only": True,
            "requires_training": False,
            "parameter_count": EXPECTED_TOTAL_PARAMS["d0"],
            "result_source": "direct_reference_existing_t0_g0_f7_no_training_no_copy",
            "source_stage4_variant": "t0",
            "selection_split": "test_in_prediction_script",
        })
        result.append(row)
    return result


def _validate_completed_record(path, variant_id, farm_id):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        row = json.load(file)
    if row.get("variant_id") != variant_id or str(row.get("farm_id")) != str(farm_id):
        raise ValueError(f"resume记录身份不一致: {path}")
    if row.get("training_code_sha256") != _sha256(__file__):
        raise ValueError("resume记录由不同Stage-4B训练代码生成；请使用--force")
    for path_key, hash_key in (
        ("model_path", "model_sha256"),
        ("best_weights_path", "best_weights_sha256"),
        ("artifact_path", "artifact_sha256"),
    ):
        if _sha256(row.get(path_key)) != row.get(hash_key):
            raise ValueError(f"resume文件hash漂移: {path_key}")
    artifact = joblib.load(row["artifact_path"])
    if (
        artifact.get("protocol_version") != PROTOCOL_VERSION
        or artifact.get("architecture_version") != ARCHITECTURE_VERSION
        or artifact.get("variant_id") != variant_id
    ):
        raise ValueError("resume artifact协议不兼容")
    if artifact.get("candidate_snapshot_before_gate_sha256") != artifact.get(
        "candidate_snapshot_after_gate_sha256"
    ):
        raise ValueError("resume artifact未证明candidate冻结")
    if (
        artifact.get("training_code_sha256") != _sha256(__file__)
        or os.path.realpath(str(artifact.get("training_code_path", "")))
        != os.path.realpath(__file__)
    ):
        raise ValueError("resume artifact训练代码漂移；请使用--force")
    validate_dependency_code_records(
        artifact.get("dependency_code_records"), role="resume artifact"
    )

    # A record is resumable only when every artifact member is intact.  This
    # prevents a truncated plot/CSV from being silently re-published under a
    # fresh complete marker.
    artifact_members = (
        ("model_path", "model_sha256"),
        ("best_weights_path", "best_weights_sha256"),
        ("history_path", "history_sha256"),
        ("history_figure_path", "history_figure_sha256"),
        ("validation_path", "validation_sha256"),
        ("checkpoint_path", "checkpoint_sha256"),
        ("candidate_provenance_path", "candidate_provenance_sha256"),
        ("tail_path", "tail_sha256"),
    )
    for path_key, hash_key in artifact_members:
        member_path = artifact.get(path_key)
        if _sha256(member_path) != artifact.get(hash_key):
            raise ValueError(f"resume artifact成员hash漂移: {path_key}")
        if path_key in row and os.path.realpath(str(row[path_key])) != os.path.realpath(
            str(member_path)
        ):
            raise ValueError(f"resume record/artifact路径不一致: {path_key}")
    replay = artifact.get("fixed_g0_replay", {})
    if _sha256(replay.get("path")) != replay.get("sha256"):
        raise ValueError("resume fixed-G0诊断文件hash漂移")
    if os.path.realpath(str(row.get("pretrain_diagnostic_path", ""))) != os.path.realpath(
        str(replay.get("path", ""))
    ):
        raise ValueError("resume fixed-G0诊断路径不一致")

    # Lock the exact train data and frozen upstream candidate bundle used by
    # this completed run.  Main() has already validated the Stage-4 markers;
    # these direct hashes additionally protect per-farm resume.
    source_members = (
        ("train_file", "train_file_sha256", True),
        ("source_f7_model_path", "source_f7_model_sha256", True),
        ("source_f7_artifact_path", "source_f7_artifact_sha256", True),
        (
            "source_t1_model_path",
            "source_t1_model_sha256",
            VARIANT_SPECS[variant_id]["candidate_source"] == "t1",
        ),
        (
            "source_t1_artifact_path",
            "source_t1_artifact_sha256",
            VARIANT_SPECS[variant_id]["candidate_source"] == "t1",
        ),
    )
    for path_key, hash_key, required in source_members:
        source_path = artifact.get(path_key)
        source_hash = artifact.get(hash_key)
        if required and (not source_path or not source_hash):
            raise ValueError(f"resume artifact缺少来源: {path_key}")
        if source_path and _sha256(source_path) != source_hash:
            raise ValueError(f"resume上游来源hash漂移: {path_key}")
    return row


def _validate_stage4_source_bundle():
    earlier = time_freq_train.validate_required_source_bundles()
    training_marker_path = os.path.join(
        time_freq_train.RESULT_ROOT, time_freq_train.TRAINING_MARKER_NAME
    )
    training_marker = time_freq_train._validate_marker_file_records(
        training_marker_path,
        expected_protocol=time_freq_train.PROTOCOL_VERSION,
        critical_keys=("training_summary", "experiment_manifest"),
    )
    prediction_marker_path = os.path.join(
        time_freq_train.RESULT_ROOT,
        time_freq_train.PREDICTION_MARKER_RELATIVE_PATH,
    )
    prediction_marker = time_freq_train._validate_marker_file_records(
        prediction_marker_path,
        expected_protocol=time_freq_train.PROTOCOL_VERSION,
        critical_keys=("training_marker", "formal.summary", "formal.final_selection"),
    )
    locked = prediction_marker["files"]["training_marker"]
    if (
        os.path.realpath(locked["path"]) != os.path.realpath(training_marker_path)
        or locked["sha256"] != _sha256(training_marker_path)
    ):
        raise ValueError("Stage-4预测marker未锁定当前训练marker")
    return {
        **earlier,
        "stage4_training_marker_path": os.path.abspath(training_marker_path),
        "stage4_training_marker_sha256": _sha256(training_marker_path),
        "stage4_prediction_marker_path": os.path.abspath(prediction_marker_path),
        "stage4_prediction_marker_sha256": _sha256(prediction_marker_path),
        "stage4_training_status": training_marker.get("status"),
        "stage4_prediction_status": prediction_marker.get("status"),
    }


def write_manifest(result_root=RESULT_ROOT, run_scope="formal"):
    rows = []
    for order, (variant_id, spec) in enumerate(VARIANT_SPECS.items()):
        rows.append({
            "variant_order": order,
            "variant_id": variant_id,
            "label": spec["label"],
            "requires_training": spec["requires_training"],
            "candidate_source": spec["candidate_source"],
            "candidate_frozen": variant_id != "d0",
            "factorized_gate": spec["factorized_gate"],
            "calibration_weight": spec["calibration_weight"],
            "dynamic_weight": spec["dynamic_weight"],
            "safety_weight": spec["safety_weight"],
            "initial_gate": spec["initial_gate"],
            "controlled_contrast": spec["contrast"],
            "d3_minus_d2_is_pure_topology_effect": False,
            "d3_minus_d2_interpretation": (
                "factorization_plus_dynamic_auxiliary_joint_effect"
                if variant_id == "d3" else "not_applicable"
            ),
            "fixed_g0_replay_internal_only": variant_id == "d0r",
            "fixed_g0_replay_selection_eligible": False,
            "soft_oracle_q90_recomputed_train_only": variant_id in CALIBRATED_VARIANTS,
            "soft_oracle_q90_applicable": variant_id in CALIBRATED_VARIANTS,
            "source_context": "F7_initial_context_not_T1_final_context",
            "source_t1_adapter_only": spec["candidate_source"] == "t1",
            "expected_total_params": EXPECTED_TOTAL_PARAMS[variant_id],
            "expected_candidate_structural_params": EXPECTED_CANDIDATE_STRUCTURAL_PARAMS[variant_id],
            "parameter_limit_exclusive": PARAMETER_LIMIT,
            "random_seed": RANDOM_SEED,
            "batch_size": BATCH_SIZE,
            "selection_split": "test",
            "test_used_for_selection": True,
            "test_is_final_blind_evaluation": False,
            "protocol_version": PROTOCOL_VERSION,
            "run_scope": run_scope,
        })
    return _atomic_to_csv(pd.DataFrame(rows), os.path.join(result_root, MANIFEST_NAME))


def _validate_formal_candidate_identity(summary):
    for farm_id in expected_farm_ids():
        farm = summary[summary["farm_id"].astype(str) == str(farm_id)]
        t1 = farm[farm["variant_id"].isin(T1_CANDIDATE_VARIANTS)]
        if len(t1) != 3:
            raise ValueError(f"{farm_id}未覆盖D0R/D2/D3 candidate身份检查")
        for field in (
            "candidate_snapshot_sha256", "candidate_probe_sha256",
            "t1_adapter_snapshot_sha256",
        ):
            if t1[field].nunique(dropna=False) != 1:
                raise ValueError(f"{farm_id} D0R/D2/D3 {field}不一致")
        calibrated_t1 = farm[farm["variant_id"].isin(("d2", "d3"))]
        if calibrated_t1["candidate_difference_q90_sha256"].nunique(dropna=False) != 1:
            raise ValueError(f"{farm_id} D2/D3同candidate却生成不同Q90")
        for field in ("soft_oracle_sample_count", "soft_oracle_element_count"):
            if calibrated_t1[field].nunique(dropna=False) != 1:
                raise ValueError(f"{farm_id} D2/D3同candidate却生成不同{field}")
        oracle_means = calibrated_t1["soft_oracle_mean"].to_numpy(dtype=float)
        if len(oracle_means) != 2 or not np.isclose(
            oracle_means[0], oracle_means[1], rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"{farm_id} D2/D3同candidate却生成不同soft oracle mean")
        all_new = farm[farm["variant_id"].isin(TRAINABLE_VARIANTS)]
        if all_new["f7_context_snapshot_sha256"].nunique(dropna=False) != 1:
            raise ValueError(f"{farm_id}四个新模型没有锁定同一F7初始context")


def publish_training_marker(summary_path, manifest_path, summary, source_identity):
    new_rows = summary[summary["variant_id"].isin(TRAINABLE_VARIANTS)].copy()
    if len(new_rows) != 4 * len(expected_farm_ids()):
        raise ValueError("Stage-4B正式新训练矩阵不是4×5")
    _validate_formal_candidate_identity(summary)
    files = {
        "training_summary": _file_record(summary_path),
        "experiment_manifest": _file_record(manifest_path),
        "training_code": _file_record(__file__),
        "source_stage4_training_marker": _file_record(
            source_identity["stage4_training_marker_path"]
        ),
        "source_stage4_prediction_marker": _file_record(
            source_identity["stage4_prediction_marker_path"]
        ),
    }
    files.update({
        f"dependency.{name}": record
        for name, record in dependency_code_records().items()
    })
    for _, row in new_rows.iterrows():
        prefix = f"{row['variant_id']}.{row['farm_id']}"
        for key in (
            "model_path", "best_weights_path", "artifact_path", "history_path",
            "history_figure_path", "validation_path", "checkpoint_path",
            "pretrain_diagnostic_path", "candidate_provenance_path", "tail_path",
            "record_path",
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
        "d0_reused_model_count": 5,
        "d0_retraining_forbidden": True,
        "fixed_g0_replay_is_internal_diagnostic_not_variant": True,
        "candidate_identity_d0r_d2_d3_verified": True,
        "q90_identity_d2_d3_verified": True,
        "soft_oracle_statistics_identity_d2_d3_verified": True,
        "source_context_f7_not_t1_final_verified": True,
        "files": files,
    }
    marker_path = _atomic_write_json(
        marker, os.path.join(RESULT_ROOT, TRAINING_MARKER_NAME)
    )
    running_path = os.path.join(RESULT_ROOT, RUNNING_MARKER_NAME)
    if os.path.exists(running_path):
        os.remove(running_path)
    return marker_path


def _discover_train_files(requested_farms=None):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "wind_train_*.csv")))
    if requested_farms:
        wanted = {str(value) for value in requested_farms}
        files = [
            path for path in files
            if re.search(r"wind_train_(\d+)\.csv$", os.path.basename(path)).group(1)
            in wanted
        ]
    return files


def _parse_csv(value):
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default=os.getenv("WIND_STAGE4B_VARIANTS", ",".join(VARIANT_SPECS)),
        help="逗号分隔: d0,d0r,d1,d2,d3",
    )
    parser.add_argument(
        "--farms", default=os.getenv("WIND_STAGE4B_FARMS", ""),
        help="逗号分隔场站ID；空值为全部正式场站",
    )
    parser.add_argument("--gate-only-epochs", type=int, default=GATE_ONLY_EPOCHS)
    parser.add_argument("--context-epochs", type=int, default=CONTEXT_EPOCHS)
    parser.add_argument("--objective-epochs", type=int, default=OBJECTIVE_EPOCHS)
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="调试覆盖三个phase epoch；自动进入partial_runs",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _formal_protocol(args, variants, farm_ids):
    return (
        not args.smoke_test
        and args.epochs is None
        and set(variants) == set(VARIANT_SPECS)
        and set(farm_ids) == set(expected_farm_ids())
        and args.gate_only_epochs == GATE_ONLY_EPOCHS == 3
        and args.context_epochs == CONTEXT_EPOCHS == 5
        and args.objective_epochs == OBJECTIVE_EPOCHS == 30
        and BATCH_SIZE == 192
        and np.isclose(VALIDATION_SPLIT, 0.15, rtol=0.0, atol=1e-12)
        and np.isclose(INITIAL_LR, 1e-4, rtol=0.0, atol=1e-12)
        and np.isclose(OBJECTIVE_LR, 5e-5, rtol=0.0, atol=1e-12)
        and PATIENCE == 6
    )


def main(argv=None):
    args = parse_args(argv)
    configure_reproducibility()
    source_identity = _validate_stage4_source_bundle()
    variants = list(dict.fromkeys(_parse_csv(args.variants)))
    invalid = sorted(set(variants) - set(VARIANT_SPECS))
    if invalid:
        raise ValueError(f"未知变体{invalid}; 可选{list(VARIANT_SPECS)}")
    farms = _parse_csv(args.farms) if args.farms else []
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs必须为正")
        args.gate_only_epochs = args.context_epochs = args.objective_epochs = args.epochs
    if args.smoke_test:
        variants = ["d3" if "d3" in variants else next(
            (item for item in variants if item in TRAINABLE_VARIANTS), "d3"
        )]
        farms = farms[:1] if farms else [expected_farm_ids()[0]]
        args.gate_only_epochs = args.context_epochs = args.objective_epochs = 1
    if any(value < 0 for value in (
        args.gate_only_epochs, args.context_epochs, args.objective_epochs
    )):
        raise ValueError("epoch不得为负")
    if set(variants).intersection(TRAINABLE_VARIANTS) and (
        args.gate_only_epochs + args.context_epochs + args.objective_epochs <= 0
    ):
        raise ValueError("可训练变体至少需要1个门控epoch")
    train_files = _discover_train_files(farms)
    if not train_files:
        raise FileNotFoundError("没有匹配的训练文件")
    farm_ids = [regime_train.get_farm_id(path) for path in train_files]
    formal = _formal_protocol(args, variants, farm_ids)
    if formal:
        run_root = RESULT_ROOT
        run_scope = "formal"
        # Keep the previous complete marker until a newly validated bundle can
        # atomically replace it.  The running marker makes predictors refuse a
        # mixed old/new bundle if this process is interrupted midway.
        _atomic_write_json(
            {
                "status": "running",
                "protocol_version": PROTOCOL_VERSION,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "training_code": _file_record(__file__),
                "force": bool(args.force),
            },
            os.path.join(RESULT_ROOT, RUNNING_MARKER_NAME),
        )
        downstream_marker = os.path.join(
            RESULT_ROOT, PREDICTION_MARKER_RELATIVE_PATH
        )
        if os.path.exists(downstream_marker):
            os.remove(downstream_marker)
    else:
        tag = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_root = os.path.join(RESULT_ROOT, "partial_runs", tag)
        run_scope = "smoke_or_partial_or_protocol_override"
    manifest_path = write_manifest(run_root, run_scope)
    print(
        f"Stage-4B farms={farm_ids}; variants={variants}; output={run_root}; "
        f"formal={formal}; seed={RANDOM_SEED}; batch={BATCH_SIZE}"
    )
    rows = []
    if "d0" in variants:
        rows.extend(build_d0_reference_rows(farm_ids))
    trainable = [item for item in variants if item in TRAINABLE_VARIANTS]
    for train_file in train_files:
        prepared = regime_train._prepare_farm(train_file)
        for variant_id in trainable:
            dirs = variant_dirs(variant_id, result_root=run_root)
            record_path = _paths(dirs, variant_id, prepared["farm_id"])["record_path"]
            completed = None if args.force else _validate_completed_record(
                record_path, variant_id, prepared["farm_id"]
            )
            if completed is not None:
                print(f"跳过已验证完成模型: {variant_id}/{prepared['farm_id']}")
                rows.append(completed)
                continue
            print(f"\n===== {VARIANT_SPECS[variant_id]['label']} / farm={prepared['farm_id']} =====")
            rows.append(train_variant_for_farm(
                variant_id,
                prepared,
                result_root=run_root,
                gate_only_epochs=args.gate_only_epochs,
                context_epochs=args.context_epochs,
                objective_epochs=args.objective_epochs,
            ))
    summary = pd.DataFrame(rows)
    if summary.empty or summary.duplicated(["variant_id", "farm_id"]).any():
        raise ValueError("Stage-4B summary为空或存在重复键")
    summary_path = _atomic_to_csv(
        summary, os.path.join(run_root, TRAINING_SUMMARY_NAME)
    )
    print(f"训练汇总: {summary_path}")
    if formal:
        if len(summary) != len(VARIANT_SPECS) * len(expected_farm_ids()):
            raise ValueError("正式summary不是5变体×5场站")
        marker = publish_training_marker(
            summary_path, manifest_path, summary, source_identity
        )
        print(f"Stage-4B正式训练bundle完成: {marker}")
        print("D0只读引用；D0R/D1/D2/D3新增训练模型数=20")
    else:
        print("partial/smoke已隔离，不发布formal complete marker")


if __name__ == "__main__":
    main()
