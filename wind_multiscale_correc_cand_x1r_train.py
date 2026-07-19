"""Stage 5A-X1R：冻结完整 X1 candidate 的门控收益转化闭环训练。

本轮只训练一个新变体（固定五场站各一个模型）：

* X1R：复用 Stage-5A 已训练 X1 的 Persistence 与 fine/mid/coarse
  corrected candidate，冻结全部 candidate 权重；
* 用 X1 自身的 train-only 窗口重新生成 soft oracle 与逐 horizon
  ``|C-P|`` Q90；
* 以 Stage-4B D2 同构的非因子化 sample×horizon calibrated-safe gate
  替换旧 G0 gate，仅训练新 gate 与显式工况 context。

训练脚本不读取任何测试集或 Stage-5A 测试预测。完整正式结果写入
``wind_results/multiscale_correc_cand/x1r_gate_closure``；smoke、场站子集或
epoch override 均隔离到 ``partial_runs``，不会发布 complete marker。
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
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import wind_RegimeEncoder_PatchTST_feature_screen_train as feature_train
import wind_RegimeEncoder_PatchTST_train as regime_train
import wind_controlled_gate_cali_train as gate_train
import wind_dl_model_train as common_train
import wind_multiscale_correc_cand_train as multiscale_train
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


MODEL_FAMILY = "multiscale_correc_cand_x1r_gate_closure"
ARCHITECTURE_VERSION = "stage5a_x1r_frozen_x1_calibrated_safe_gate_v1"
ARTIFACT_SCHEMA_VERSION = 1
PROTOCOL_VERSION = "stage5a_x1r_gate_closure_test_selected_v1"
RESULT_ROOT = os.path.join(
    "./wind_results", "multiscale_correc_cand", "x1r_gate_closure"
)
SOURCE_FEATURE_GROUPS = "P+H+D"
SOURCE_FEATURE_COUNT = 36
SOURCE_VARIANT = "x1"
RANDOM_SEED = 2026

BATCH_SIZE = int(os.getenv("WIND_X1R_BATCH_SIZE", "192"))
VALIDATION_SPLIT = float(os.getenv("WIND_X1R_VALIDATION_SPLIT", "0.15"))
GATE_ONLY_EPOCHS = int(os.getenv("WIND_X1R_GATE_ONLY_EPOCHS", "3"))
CONTEXT_EPOCHS = int(os.getenv("WIND_X1R_CONTEXT_EPOCHS", "5"))
OBJECTIVE_EPOCHS = int(os.getenv("WIND_X1R_OBJECTIVE_EPOCHS", "30"))
INITIAL_LR = float(os.getenv("WIND_X1R_INITIAL_LR", "0.0001"))
OBJECTIVE_LR = float(os.getenv("WIND_X1R_OBJECTIVE_LR", "0.00005"))
PATIENCE = int(os.getenv("WIND_X1R_PATIENCE", "6"))
PARAMETER_LIMIT = 30_000

CALIBRATION_WEIGHT = gate_train.CALIBRATION_WEIGHT
DYNAMIC_WEIGHT = 0.0
SAFETY_WEIGHT = gate_train.SAFETY_WEIGHT
SOFT_ORACLE_TEMPERATURE = gate_train.SOFT_ORACLE_TEMPERATURE
CALIBRATION_DIFFERENCE_QUANTILE = gate_train.CALIBRATION_DIFFERENCE_QUANTILE

VARIANT_SPECS = {
    "x1r": {
        "label": "X1R frozen X1 + non-factorized calibrated-safe gate",
        "requires_training": True,
        "candidate_source": "x1",
        "factorized_gate": False,
        "calibration_weight": CALIBRATION_WEIGHT,
        "dynamic_weight": DYNAMIC_WEIGHT,
        "safety_weight": SAFETY_WEIGHT,
        "initial_gate": "own_train_soft_oracle_clipped_mean",
        "selection_role": "deployment_candidate",
    }
}
TRAINABLE_VARIANTS = ("x1r",)
REFERENCE_VARIANTS = ("x0", "x1_fixed_g0")
EXPECTED_TOTAL_PARAMS = {"x1r": 24_177}
EXPECTED_ADAPTER_TRAINABLE_PARAMS = {"x1r": 0}
EXPECTED_GATE_TRAINABLE_PARAMS = {
    "x1r": {"gate_only": 993, "context": 2553, "objective": 2553}
}

# Candidate身份只包含 Persistence/F7 residual 与完整X1多尺度adapter；显式工况
# context在第二、三phase允许训练，因此故意不纳入candidate冻结快照。
CANDIDATE_WEIGHTED_LAYER_NAMES = (
    tuple(regime_train.B2_WEIGHTED_LAYER_NAMES)
    + tuple(multiscale_train.ADAPTER_WEIGHTED_LAYER_NAMES["x1"])
)
CONTEXT_WEIGHTED_LAYER_NAMES = tuple(gate_train.CONTEXT_WEIGHTED_LAYER_NAMES)

TRAINING_SUMMARY_NAME = "x1r_gate_closure_training_metrics.csv"
MANIFEST_NAME = "x1r_gate_closure_experiment_manifest.csv"
TRAINING_MARKER_NAME = "x1r_gate_closure_training_bundle_complete.json"
RUNNING_MARKER_NAME = "x1r_gate_closure_training_bundle_running.json"
PREDICTION_MARKER_RELATIVE_PATH = os.path.join(
    "testdata_predict_output", "x1r_gate_closure_test_bundle_complete.json"
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
    farms = tuple(str(value) for value in multiscale_train.expected_farm_ids())
    if len(farms) != 5:
        raise ValueError(f"X1R正式来源场站数不是5: {farms}")
    return farms


def variant_model_name(variant_id):
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"未知X1R变体: {variant_id}")
    return f"{MODEL_FAMILY}_{variant_id}"


def result_dirs(create=True, result_root=None):
    root = RESULT_ROOT if result_root is None else result_root
    values = {
        "root": root,
        "models": os.path.join(root, "models"),
        "weights": os.path.join(root, "weights"),
        "preprocess": os.path.join(root, "preprocess"),
        "history": os.path.join(root, "history"),
        "tensorboard": os.path.join(root, "tensorboard"),
        "soft_oracle": os.path.join(root, "soft_oracle"),
        "q90_diagnostics": os.path.join(root, "q90_diagnostics"),
        "calibration": os.path.join(root, "calibration"),
        "safety_diagnostics": os.path.join(root, "safety_diagnostics"),
        "validation_diagnostics": os.path.join(root, "validation_diagnostics"),
        "candidate_diagnostics": os.path.join(root, "candidate_diagnostics"),
        "tails": os.path.join(root, "tails"),
        "records": os.path.join(root, "records"),
    }
    if create:
        for path in values.values():
            os.makedirs(path, exist_ok=True)
    return values


def variant_dirs(variant_id, create=True, result_root=None):
    """Compatibility interface for shared prediction helpers."""
    if variant_id != "x1r":
        raise ValueError(f"未知X1R变体: {variant_id}")
    return result_dirs(create=create, result_root=result_root)


def _paths(dirs, farm_id):
    prefix = f"{variant_model_name('x1r')}_farm_{farm_id}"
    return {
        "model_path": os.path.join(dirs["models"], f"{prefix}.keras"),
        "weights_path": os.path.join(dirs["weights"], f"{prefix}_best.weights.h5"),
        "artifact_path": os.path.join(dirs["preprocess"], f"{prefix}_preprocess.pkl"),
        "history_path": os.path.join(dirs["history"], f"{prefix}_gate_history.csv"),
        "history_figure_path": os.path.join(dirs["history"], f"{prefix}_gate_history.png"),
        "checkpoint_path": os.path.join(dirs["validation_diagnostics"], f"{prefix}_checkpoint_trace.csv"),
        "validation_path": os.path.join(dirs["validation_diagnostics"], f"{prefix}_validation.csv"),
        "soft_oracle_path": os.path.join(dirs["soft_oracle"], f"{prefix}_train_only_soft_oracle.json"),
        "q90_path": os.path.join(dirs["q90_diagnostics"], f"{prefix}_train_only_q90.csv"),
        "provenance_path": os.path.join(dirs["candidate_diagnostics"], f"{prefix}_candidate_provenance.csv"),
        "replay_path": os.path.join(dirs["candidate_diagnostics"], f"{prefix}_fixed_g0_validation_replay.csv"),
        "tail_path": os.path.join(dirs["tails"], f"{prefix}_tail.csv"),
        "record_path": os.path.join(dirs["records"], f"{prefix}_record.json"),
    }


def get_x1r_custom_objects():
    objects = dict(multiscale_train.get_multiscale_custom_objects())
    objects.update(stage4b_train.get_stage4b_custom_objects())
    return objects


def get_time_freq_custom_objects():
    """Compatibility alias used by shared prediction helpers."""
    return get_x1r_custom_objects()


_sha256 = time_freq_train._sha256
_array_sha256 = time_freq_train._array_sha256
_atomic_to_csv = time_freq_train._atomic_to_csv
_atomic_write_json = time_freq_train._atomic_write_json
_atomic_joblib_dump = time_freq_train._atomic_joblib_dump
_file_record = time_freq_train._file_record
_save_model_atomic = time_freq_train._save_model_atomic
_weighted_snapshot = time_freq_train._weighted_snapshot


def dependency_code_records():
    modules = {
        "multiscale_train": multiscale_train,
        "stage4b_gate_train": stage4b_train,
        "controlled_gate_train": gate_train,
        "feature_screen_train": feature_train,
        "regime_encoder_train": regime_train,
        "time_freq_train": time_freq_train,
        "common_dl_train": common_train,
    }
    return {name: _file_record(module.__file__) for name, module in modules.items()}


def validate_dependency_code_records(records, role="X1R artifact"):
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


def _validate_file_record(label, record):
    if not isinstance(record, dict):
        raise ValueError(f"{label}不是文件记录")
    path = str(record.get("path", ""))
    if not os.path.isfile(path) or _sha256(path) != record.get("sha256"):
        raise ValueError(f"{label}不存在或hash漂移: {path}")
    return path


def validate_source_bundle():
    """只验证Stage-5A训练bundle；本函数禁止读取测试预测marker。"""
    running = os.path.join(multiscale_train.RESULT_ROOT, multiscale_train.RUNNING_MARKER_NAME)
    if os.path.isfile(running):
        raise RuntimeError(f"Stage-5A训练仍在运行或未完整收尾: {running}")
    marker_path = os.path.join(
        multiscale_train.RESULT_ROOT, multiscale_train.TRAINING_MARKER_NAME
    )
    if not os.path.isfile(marker_path):
        raise FileNotFoundError(f"缺少Stage-5A训练complete marker: {marker_path}")
    with open(marker_path, "r", encoding="utf-8") as file:
        marker = json.load(file)
    checks = {
        "status": marker.get("status") == "complete",
        "protocol": marker.get("protocol_version") == multiscale_train.PROTOCOL_VERSION,
        "architecture": marker.get("architecture_version") == multiscale_train.ARCHITECTURE_VERSION,
        "farms": set(map(str, marker.get("expected_farm_ids", ()))) == set(expected_farm_ids()),
        "x1_trained": "x1" in set(marker.get("new_training_variants", ())),
        "no_token_interaction": bool(marker.get("token_interaction_forbidden")),
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError(f"Stage-5A训练marker不兼容: {failed}")
    files = marker.get("files", {})
    summary_path = _validate_file_record(
        "Stage-5A files.training_summary", files.get("training_summary")
    )
    _validate_file_record(
        "Stage-5A files.experiment_manifest", files.get("experiment_manifest")
    )
    source_training_code = _validate_file_record(
        "Stage-5A files.training_code", files.get("training_code")
    )
    if os.path.realpath(source_training_code) != os.path.realpath(
        multiscale_train.__file__
    ):
        raise ValueError("Stage-5A marker训练代码路径不是当前multiscale_train")
    for key, record in files.items():
        if key.startswith("dependency."):
            _validate_file_record(f"Stage-5A files.{key}", record)
    # 只验证本轮实际消费的X1模型、artifact与record；不触碰Stage-5A预测产物。
    for farm_id in expected_farm_ids():
        for key in ("model_path", "artifact_path", "record_path"):
            _validate_file_record(
                f"Stage-5A x1/{farm_id}/{key}", files.get(f"x1.{farm_id}.{key}")
            )
    summary = pd.read_csv(summary_path, dtype={"farm_id": str})
    x1 = summary[summary["variant_id"].astype(str) == "x1"].copy()
    if (
        len(x1) != len(expected_farm_ids())
        or set(x1["farm_id"].astype(str)) != set(expected_farm_ids())
        or x1.duplicated(["farm_id"]).any()
    ):
        raise ValueError("Stage-5A训练summary没有唯一完整的5场站X1")
    return {
        "marker_path": os.path.abspath(marker_path),
        "marker_sha256": _sha256(marker_path),
        "marker": marker,
        "summary_path": os.path.abspath(summary_path),
        "summary_sha256": _sha256(summary_path),
        "summary": x1,
        "test_prediction_read": False,
    }


def load_source_x1(farm_id, source_identity=None):
    source_identity = source_identity or validate_source_bundle()
    farm_id = str(farm_id)
    row = source_identity["summary"]
    row = row[row["farm_id"].astype(str) == farm_id]
    if len(row) != 1:
        raise ValueError(f"X1/{farm_id}来源summary不是唯一一行")
    row = row.iloc[0]
    files = source_identity["marker"]["files"]
    model_path = _validate_file_record(
        f"X1/{farm_id}/model", files[f"x1.{farm_id}.model_path"]
    )
    artifact_path = _validate_file_record(
        f"X1/{farm_id}/artifact", files[f"x1.{farm_id}.artifact_path"]
    )
    record_path = _validate_file_record(
        f"X1/{farm_id}/record", files[f"x1.{farm_id}.record_path"]
    )
    if (
        os.path.realpath(model_path) != os.path.realpath(str(row["model_path"]))
        or os.path.realpath(artifact_path) != os.path.realpath(str(row["artifact_path"]))
        or _sha256(model_path) != row["model_sha256"]
        or _sha256(artifact_path) != row["artifact_sha256"]
    ):
        raise ValueError(f"X1/{farm_id} marker与summary来源身份不一致")
    artifact = joblib.load(artifact_path)
    source_code_record = files["training_code"]
    artifact_checks = {
        "variant": artifact.get("variant_id") == "x1",
        "farm": str(artifact.get("farm_id")) == farm_id,
        "protocol": artifact.get("protocol_version") == multiscale_train.PROTOCOL_VERSION,
        "architecture": artifact.get("architecture_version") == multiscale_train.ARCHITECTURE_VERSION,
        "params": int(artifact.get("total_params", -1))
        == multiscale_train.EXPECTED_TOTAL_PARAMS["x1"],
        "model_path": os.path.realpath(str(artifact.get("model_path", "")))
        == os.path.realpath(model_path),
        "model_hash": artifact.get("model_sha256") == _sha256(model_path),
        "training_code_path": os.path.realpath(
            str(artifact.get("training_code_path", ""))
        )
        == os.path.realpath(source_code_record["path"]),
        "training_code_hash": artifact.get("training_code_sha256")
        == source_code_record["sha256"],
        "candidate_trained": artifact.get("candidate_probe_before_sha256")
        != artifact.get("candidate_probe_after_sha256"),
        "token_interaction_false": not bool(
            artifact.get("multiscale_definition", {}).get("token_interaction", True)
        ),
    }
    failed = [key for key, value in artifact_checks.items() if not value]
    if failed:
        raise ValueError(f"X1/{farm_id} artifact不兼容: {failed}")
    multiscale_train.validate_dependency_code_records(
        artifact.get("dependency_code_records"), role=f"X1/{farm_id} artifact"
    )
    model = keras.models.load_model(
        model_path,
        custom_objects=multiscale_train.get_multiscale_custom_objects(),
        compile=False,
    )
    if int(model.count_params()) != multiscale_train.EXPECTED_TOTAL_PARAMS["x1"]:
        raise ValueError(f"X1/{farm_id}加载后参数量漂移")
    return model, artifact, os.path.abspath(model_path), os.path.abspath(artifact_path), os.path.abspath(record_path)


def build_x1r_model(source_model, initial_gate_weight):
    """Replace the pruned old G0 output with a fresh D2-isomorphic gate."""
    configure_reproducibility()
    persistence = source_model.get_layer("persistence_forecast_candidate").output
    corrected = source_model.get_layer("multiscale_corrected_candidate").output
    context = source_model.get_layer("regime_context").output
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
    forecast = regime_train.TwoCandidateGateFusion(name="forecast_power")(
        [persistence, corrected, gate]
    )
    candidate = layers.Activation("linear", name="candidate_forecast")(corrected)
    packet = layers.Concatenate(name="control_packet")(
        [gate, persistence, corrected, forecast, q_by_horizon, horizon_prior]
    )
    model = keras.Model(
        source_model.inputs,
        {
            "forecast_power": forecast,
            "candidate_forecast": candidate,
            "control_packet": packet,
        },
        name="WindMultiscaleX1RGateClosure",
    )
    if any(layer.name in {"correction_gate", "frozen_g0_gate"} for layer in model.layers):
        raise ValueError("X1R新图错误保留了旧G0 gate")
    if int(model.count_params()) != EXPECTED_TOTAL_PARAMS["x1r"]:
        raise ValueError(
            f"X1R参数量{model.count_params()} != {EXPECTED_TOTAL_PARAMS['x1r']}"
        )
    return model


def diagnostic_model(model):
    packet = model.get_layer("control_packet").output
    h = FORECAST_LEN
    return keras.Model(
        model.inputs,
        {
            "forecast": model.get_layer("forecast_power").output,
            "persistence": packet[:, h : 2 * h],
            "corrected": packet[:, 2 * h : 3 * h],
            "gate": packet[:, :h],
            "q": packet[:, 4 * h : 5 * h],
            "s": packet[:, 5 * h : 6 * h],
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


def _attach_targets(dataset):
    def attach(x, y):
        return x, {
            "forecast_power": y,
            "candidate_forecast": y,
            "control_packet": y,
        }

    return dataset.map(
        attach, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    ).prefetch(tf.data.AUTOTUNE)


def _candidate_snapshot(model):
    return _array_sha256(_weighted_snapshot(model, CANDIDATE_WEIGHTED_LAYER_NAMES))


def _context_snapshot(model):
    return _array_sha256(_weighted_snapshot(model, CONTEXT_WEIGHTED_LAYER_NAMES))


def _canonical_probe(prepared, count=2):
    features = np.asarray(prepared["features"], dtype=np.float32)
    if len(features) < HISTORY_LEN + count:
        raise ValueError("训练数据不足以生成固定X1 candidate probe")
    return np.stack([features[i : i + HISTORY_LEN] for i in range(count)], axis=0)


def _probe_outputs(model, probe):
    return {
        key: np.asarray(value)
        for key, value in diagnostic_model(model)(probe, training=False).items()
    }


def _estimate_train_only_calibration(model, train_ds, prepared):
    result = stage4b_train.estimate_train_only_calibration(model, train_ds, prepared)
    q90 = np.asarray(result["candidate_difference_q90"], dtype=np.float32)
    if (
        q90.shape != (FORECAST_LEN,)
        or not np.isfinite(q90).all()
        or np.any(q90 < 0.0)
        or int(result["element_count"]) != int(result["sample_count"]) * FORECAST_LEN
    ):
        raise ValueError("X1 train-only soft oracle/Q90统计不完整")
    result["candidate_source"] = "x1"
    return result


def _set_phase(model, phase):
    if phase not in {"gate_only", "context", "objective"}:
        raise ValueError(f"未知X1R phase: {phase}")
    for layer in model.layers:
        layer.trainable = False
    for name in (
        "controlled_gate",
        "sample_dynamic_probability",
        "sample_dynamic_probability_by_horizon",
        "horizon_gate_prior",
    ):
        try:
            model.get_layer(name).trainable = True
        except ValueError:
            pass
    if phase in {"context", "objective"}:
        for name in CONTEXT_WEIGHTED_LAYER_NAMES:
            model.get_layer(name).trainable = True
    model.get_layer("residual_dropout").rate = 0.0
    model.get_layer("regime_context_dropout").rate = (
        0.0 if phase == "gate_only" else float(feature_train.GATE_DROPOUT)
    )
    count = int(sum(int(np.prod(weight.shape)) for weight in model.trainable_weights))
    expected = EXPECTED_GATE_TRAINABLE_PARAMS["x1r"][phase]
    if count != expected:
        raise ValueError(f"X1R/{phase}可训练参数{count} != {expected}")
    return count


def _compile(model, prepared, q90, learning_rate):
    auxiliary = gate_train.ControlledGateAuxiliaryLoss(
        forecast_len=FORECAST_LEN,
        target_mean=float(prepared["scaler_y"].mean_[0]),
        target_scale=float(prepared["scaler_y"].scale_[0]),
        capacity=float(prepared["capacity"]),
        calibration_weight=CALIBRATION_WEIGHT,
        dynamic_weight=DYNAMIC_WEIGHT,
        safety_weight=SAFETY_WEIGHT,
        candidate_difference_q90=np.asarray(q90, dtype=np.float32),
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
        metrics={
            "forecast_power": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ]
        },
    )


def _source_fixed_g0_validation(source_model, val_ds, prepared):
    source_diag = multiscale_train.diagnostic_model(source_model)
    truth_parts, forecast_parts, gate_parts = [], [], []
    for batch_x, batch_y in val_ds:
        output = source_diag(batch_x, training=False)
        truth_parts.append(np.asarray(batch_y))
        forecast_parts.append(np.asarray(output["forecast"]))
        gate_parts.append(np.asarray(output["gate"]))
    truth = stage4b_train._inverse_scaled(np.concatenate(truth_parts), prepared)
    forecast = stage4b_train._inverse_scaled(np.concatenate(forecast_parts), prepared)
    gate = np.concatenate(gate_parts)
    error = forecast - truth
    capacity = float(prepared["capacity"])
    return pd.DataFrame(
        [
            {
                "variant_id": "x1_fixed_g0",
                "farm_id": str(prepared["farm_id"]),
                "diagnostic_scope": "validation_pretrain_only_not_selection_variant",
                "capacity_normalized_mae": float(np.mean(np.abs(error)) / capacity),
                "capacity_normalized_rmse": float(
                    np.sqrt(np.mean(np.square(error))) / capacity
                ),
                "gate_mean": float(np.mean(gate)),
                "gate_std": float(np.std(gate)),
                "selection_eligible": False,
            }
        ]
    )


def _source_state(prepared, train_ds, source_identity):
    farm_id = str(prepared["farm_id"])
    source_model, source_artifact, source_model_path, source_artifact_path, source_record_path = load_source_x1(
        farm_id, source_identity
    )
    gate_train._validate_prepared_against_source(prepared, source_artifact)
    if (
        os.path.realpath(str(prepared["train_file"]))
        != os.path.realpath(str(source_artifact.get("train_file", "")))
        or _sha256(prepared["train_file"])
        != source_artifact.get("train_file_sha256")
    ):
        raise ValueError(
            f"X1R/{farm_id}当前训练CSV不是X1 artifact锁定的同一文件"
        )
    probe = _canonical_probe(prepared)
    provisional = build_x1r_model(source_model, 0.5)
    provisional_probe = _probe_outputs(provisional, probe)
    source_diag = multiscale_train.diagnostic_model(source_model)
    source_output = source_diag(probe, training=False)
    corrected_drift = float(
        np.max(np.abs(provisional_probe["corrected"] - np.asarray(source_output["corrected"])))
    )
    persistence_drift = float(
        np.max(np.abs(provisional_probe["persistence"] - np.asarray(source_output["persistence"])))
    )
    if corrected_drift != 0.0 or persistence_drift != 0.0:
        raise ValueError(
            f"X1R未精确共享X1 candidate: corrected={corrected_drift}, "
            f"persistence={persistence_drift}"
        )
    calibration = _estimate_train_only_calibration(provisional, train_ds, prepared)
    model = build_x1r_model(source_model, calibration["soft_oracle_mean_clipped"])
    probe_outputs = _probe_outputs(model, probe)
    if not np.array_equal(probe_outputs["corrected"], provisional_probe["corrected"]):
        raise ValueError("按soft-oracle mean重建后X1 candidate发生漂移")
    calibration.update(
        {
            "candidate_snapshot_sha256": _candidate_snapshot(model),
            "candidate_probe_sha256": _array_sha256(
                [("corrected", probe_outputs["corrected"])]
            ),
            "persistence_probe_sha256": _array_sha256(
                [("persistence", probe_outputs["persistence"])]
            ),
        }
    )
    return {
        "model": model,
        "source_model": source_model,
        "source_artifact": source_artifact,
        "source_model_path": source_model_path,
        "source_artifact_path": source_artifact_path,
        "source_record_path": source_record_path,
        "probe": probe,
        "probe_outputs": probe_outputs,
        "calibration": calibration,
        "candidate_snapshot_sha256": _candidate_snapshot(model),
        "context_snapshot_sha256": _context_snapshot(model),
        "source_candidate_probe_max_abs_drift": corrected_drift,
        "source_persistence_probe_max_abs_drift": persistence_drift,
    }


def train_farm(
    prepared,
    source_identity,
    result_root=None,
    gate_only_epochs=GATE_ONLY_EPOCHS,
    context_epochs=CONTEXT_EPOCHS,
    objective_epochs=OBJECTIVE_EPOCHS,
):
    keras.backend.clear_session()
    configure_reproducibility()
    farm_id = str(prepared["farm_id"])
    plain_train, plain_val, train_samples, total_samples = _plain_datasets(prepared)
    state = _source_state(prepared, plain_train, source_identity)
    model = state["model"]
    dirs = result_dirs(result_root=result_root)
    paths = _paths(dirs, farm_id)
    if os.path.exists(paths["weights_path"]):
        os.remove(paths["weights_path"])

    before_snapshot = _candidate_snapshot(model)
    before_context = _context_snapshot(model)
    before_probe = _probe_outputs(model, state["probe"])
    calibration = state["calibration"]
    if int(calibration["sample_count"]) != int(train_samples):
        raise ValueError(f"X1R/{farm_id} Q90不是由完整train-only窗口生成")

    replay = _source_fixed_g0_validation(
        state["source_model"], plain_val, prepared
    )
    _atomic_to_csv(replay, paths["replay_path"])
    q90 = np.asarray(calibration["candidate_difference_q90"], dtype=np.float32)
    _atomic_to_csv(
        pd.DataFrame(
            {
                "farm_id": farm_id,
                "horizon_step": np.arange(1, FORECAST_LEN + 1),
                "candidate_difference_q90": q90,
                "quantile": CALIBRATION_DIFFERENCE_QUANTILE,
                "scope": "train_only_frozen_x1_candidate",
            }
        ),
        paths["q90_path"],
    )
    soft_oracle_json = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in calibration.items()
        if key != "candidate_difference_q90"
    }
    soft_oracle_json["candidate_difference_q90_path"] = os.path.abspath(
        paths["q90_path"]
    )
    soft_oracle_json["candidate_difference_q90_file_sha256"] = _sha256(
        paths["q90_path"]
    )
    _atomic_write_json(soft_oracle_json, paths["soft_oracle_path"])
    provenance = pd.DataFrame(
        [
            {
                "variant_id": "x1r",
                "farm_id": farm_id,
                "candidate_source": "x1",
                "candidate_frozen_all_phases": True,
                "candidate_snapshot_sha256": before_snapshot,
                "candidate_probe_sha256": calibration["candidate_probe_sha256"],
                "source_x1_model_path": state["source_model_path"],
                "source_x1_model_sha256": _sha256(state["source_model_path"]),
                "source_x1_artifact_path": state["source_artifact_path"],
                "source_x1_artifact_sha256": _sha256(state["source_artifact_path"]),
                "source_x1_record_path": state["source_record_path"],
                "source_x1_record_sha256": _sha256(state["source_record_path"]),
                "source_x1_train_file_path": os.path.abspath(prepared["train_file"]),
                "source_x1_train_file_sha256": _sha256(prepared["train_file"]),
                "train_only_soft_oracle": True,
                "q90_per_horizon": True,
                "candidate_difference_q90_sha256": calibration[
                    "candidate_difference_q90_sha256"
                ],
            }
        ]
    )
    _atomic_to_csv(provenance, paths["provenance_path"])

    train_ds, val_ds = _attach_targets(plain_train), _attach_targets(plain_val)
    checkpoint = stage4b_train.GateCheckpoint(
        paths["weights_path"], plain_val, prepared, "x1r"
    )
    histories, phase_trainable = [], {}
    phase_specs = (
        ("gate_only", int(gate_only_epochs), INITIAL_LR),
        ("context", int(context_epochs), INITIAL_LR),
        ("objective", int(objective_epochs), OBJECTIVE_LR),
    )
    started = time.monotonic()
    for phase, epochs, learning_rate in phase_specs:
        if epochs <= 0:
            continue
        phase_trainable[phase] = _set_phase(model, phase)
        _compile(model, prepared, q90, learning_rate)
        checkpoint.phase = phase
        finite_guard = feature_train.NonFiniteTrainingGuard()
        callbacks = [
            finite_guard,
            keras.callbacks.TerminateOnNaN(),
            checkpoint,
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(
                    dirs["tensorboard"],
                    f"farm_{farm_id}",
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    phase,
                ),
                histogram_freq=0,
                profile_batch=0,
            ),
        ]
        if phase == "objective":
            callbacks.extend(
                [
                    keras.callbacks.EarlyStopping(
                        monitor="selection_val_nrmse",
                        mode="min",
                        patience=PATIENCE,
                        restore_best_weights=False,
                        verbose=1,
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor="selection_val_nrmse",
                        mode="min",
                        factor=0.5,
                        patience=3,
                        min_lr=1e-6,
                        verbose=1,
                    ),
                ]
            )
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        feature_train.ensure_finite_training_history(history, finite_guard)
        histories.append((phase, history))

    checkpoint_trace = checkpoint.finalize()
    model.load_weights(paths["weights_path"])
    elapsed = float(time.monotonic() - started)
    history_frame = stage4b_train._history_frame(histories)
    if len(history_frame) != len(checkpoint_trace):
        raise ValueError("X1R history与checkpoint轨迹长度不一致")
    _atomic_to_csv(history_frame, paths["history_path"])
    _atomic_to_csv(checkpoint_trace, paths["checkpoint_path"])
    stage4b_train._plot_history(
        history_frame,
        paths["history_figure_path"],
        f"X1R gate closure farm {farm_id}",
    )

    after_snapshot = _candidate_snapshot(model)
    after_probe = _probe_outputs(model, state["probe"])
    corrected_drift = float(
        np.max(np.abs(after_probe["corrected"] - before_probe["corrected"]))
    )
    persistence_drift = float(
        np.max(np.abs(after_probe["persistence"] - before_probe["persistence"]))
    )
    if after_snapshot != before_snapshot or corrected_drift != 0.0 or persistence_drift != 0.0:
        raise ValueError(
            "X1R门控训练改变了冻结candidate: "
            f"snapshot={after_snapshot != before_snapshot}, C={corrected_drift}, P={persistence_drift}"
        )
    validation = stage4b_train.validation_diagnostics(
        model, plain_val, prepared, "x1r"
    )
    _atomic_to_csv(validation, paths["validation_path"])
    prepared["train_df"].iloc[-HISTORY_LEN:].to_csv(paths["tail_path"], index=True)
    _save_model_atomic(model, paths["model_path"])
    restored = keras.models.load_model(
        paths["model_path"], custom_objects=get_x1r_custom_objects(), compile=False
    )
    restored_probe = _probe_outputs(restored, state["probe"])
    for key, expected in after_probe.items():
        if not np.allclose(expected, restored_probe[key], rtol=1e-7, atol=1e-7):
            raise ValueError(f"X1R保存/重载{key}不一致")
    total_params = int(model.count_params())
    if total_params != EXPECTED_TOTAL_PARAMS["x1r"] or total_params >= PARAMETER_LIMIT:
        raise ValueError(f"X1R最终参数量异常: {total_params}")

    calibration_artifact = dict(calibration)
    calibration_artifact["candidate_difference_q90"] = q90.tolist()
    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "model_family": MODEL_FAMILY,
        "variant_id": "x1r",
        "variant_spec": dict(VARIANT_SPECS["x1r"]),
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
            state["source_artifact"]["selected_regime_feature_names"]
        ),
        "selected_regime_feature_count": SOURCE_FEATURE_COUNT,
        "candidate_source": "x1",
        "candidate_frozen_all_phases": True,
        "candidate_snapshot_before_gate_sha256": before_snapshot,
        "candidate_snapshot_after_gate_sha256": after_snapshot,
        "candidate_output_before_gate_sha256": _array_sha256(
            [("corrected", before_probe["corrected"])]
        ),
        "candidate_output_after_gate_sha256": _array_sha256(
            [("corrected", after_probe["corrected"])]
        ),
        "candidate_gate_max_abs_drift": corrected_drift,
        "candidate_gate_calibration_max_abs_drift": corrected_drift,
        "persistence_gate_max_abs_drift": persistence_drift,
        "candidate_calibration": calibration_artifact,
        "gate_training": {
            "topology": "nonfactorized_sample_horizon",
            "candidate_frozen_all_phases": True,
            "calibration_weight": CALIBRATION_WEIGHT,
            "dynamic_weight": DYNAMIC_WEIGHT,
            "safety_weight": SAFETY_WEIGHT,
            "phases": [
                {
                    "phase": phase,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "trainable_parameter_count": phase_trainable.get(phase),
                }
                for phase, epochs, learning_rate in phase_specs
            ],
            "checkpoint_rule": "validation_nrmse_then_regret_brier_within_0.1pct",
        },
        "old_g0_gate_in_new_graph": False,
        "source_x1_context_initialization": True,
        "source_x1_context_trainable_after_gate_only": True,
        "batch_size": BATCH_SIZE,
        "validation_split": VALIDATION_SPLIT,
        "selection_split": "test_in_prediction_script",
        "test_used_for_training": False,
        "test_prediction_read_during_training": False,
        "test_is_final_blind_evaluation": False,
        "legacy_bidirectional_weather_imputation": True,
        "scaler_fit_scope": "full_train_file_including_validation",
        "validation_target_overlap_steps": FORECAST_LEN - 1,
        "source_x1_model_path": state["source_model_path"],
        "source_x1_model_sha256": _sha256(state["source_model_path"]),
        "source_x1_artifact_path": state["source_artifact_path"],
        "source_x1_artifact_sha256": _sha256(state["source_artifact_path"]),
        "source_x1_record_path": state["source_record_path"],
        "source_x1_record_sha256": _sha256(state["source_record_path"]),
        "source_stage5a_training_marker_path": source_identity["marker_path"],
        "source_stage5a_training_marker_sha256": source_identity["marker_sha256"],
        "candidate_context_snapshot_before_sha256": before_context,
        "candidate_context_snapshot_after_sha256": _context_snapshot(model),
        "total_params": total_params,
        "parameter_limit": PARAMETER_LIMIT,
        "train_samples": int(train_samples),
        "validation_samples": int(total_samples - train_samples),
        "gate_training_elapsed_seconds": elapsed,
        "best_validation_nrmse": checkpoint.best,
        "best_validation_positive_regret": checkpoint.best_regret,
        "best_validation_brier": checkpoint.best_brier,
        "best_phase": checkpoint.best_phase,
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
        "soft_oracle_path": os.path.abspath(paths["soft_oracle_path"]),
        "soft_oracle_sha256": _sha256(paths["soft_oracle_path"]),
        "q90_path": os.path.abspath(paths["q90_path"]),
        "q90_sha256": _sha256(paths["q90_path"]),
        "candidate_provenance_path": os.path.abspath(paths["provenance_path"]),
        "candidate_provenance_sha256": _sha256(paths["provenance_path"]),
        "fixed_g0_validation_replay_path": os.path.abspath(paths["replay_path"]),
        "fixed_g0_validation_replay_sha256": _sha256(paths["replay_path"]),
        "tail_path": os.path.abspath(paths["tail_path"]),
        "tail_sha256": _sha256(paths["tail_path"]),
        "training_code_path": os.path.abspath(__file__),
        "training_code_sha256": _sha256(__file__),
        "dependency_code_records": dependency_code_records(),
    }
    _atomic_joblib_dump(artifact, paths["artifact_path"])
    row = validation.iloc[0].to_dict()
    row.update(
        {
            "model_family": MODEL_FAMILY,
            "variant_id": "x1r",
            "variant_label": VARIANT_SPECS["x1r"]["label"],
            "farm_id": farm_id,
            "feature_groups": SOURCE_FEATURE_GROUPS,
            "feature_count": SOURCE_FEATURE_COUNT,
            "reference_only": False,
            "requires_training": True,
            "candidate_source": "x1",
            "candidate_snapshot_sha256": before_snapshot,
            "candidate_snapshot_after_gate_sha256": after_snapshot,
            "candidate_probe_sha256": calibration["candidate_probe_sha256"],
            "candidate_gate_max_abs_drift": corrected_drift,
            "persistence_gate_max_abs_drift": persistence_drift,
            "candidate_difference_q90_sha256": calibration[
                "candidate_difference_q90_sha256"
            ],
            "soft_oracle_mean": calibration["soft_oracle_mean"],
            "soft_oracle_mean_clipped": calibration["soft_oracle_mean_clipped"],
            "soft_oracle_sample_count": calibration["sample_count"],
            "soft_oracle_element_count": calibration["element_count"],
            "parameter_count": total_params,
            "gate_trainable_parameter_count": EXPECTED_GATE_TRAINABLE_PARAMS["x1r"][
                "objective"
            ],
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
            "soft_oracle_path": artifact["soft_oracle_path"],
            "q90_path": artifact["q90_path"],
            "candidate_provenance_path": artifact["candidate_provenance_path"],
            "fixed_g0_validation_replay_path": artifact[
                "fixed_g0_validation_replay_path"
            ],
            "tail_path": artifact["tail_path"],
            "record_path": os.path.abspath(paths["record_path"]),
            "training_code_path": os.path.abspath(__file__),
            "training_code_sha256": _sha256(__file__),
            "result_source": "new_x1r_calibrated_safe_gate_closure_training",
            "selection_split": "test_in_prediction_script",
            "test_used_for_training": False,
        }
    )
    _atomic_write_json(row, paths["record_path"])
    del restored, model, state
    keras.backend.clear_session()
    return row


def _validate_completed_record(path, farm_id):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        row = json.load(file)
    if row.get("variant_id") != "x1r" or str(row.get("farm_id")) != str(farm_id):
        raise ValueError(f"X1R resume记录身份不一致: {path}")
    if row.get("training_code_sha256") != _sha256(__file__):
        raise ValueError("X1R resume记录由不同训练代码生成；请使用--force")
    for path_key, hash_key in (
        ("model_path", "model_sha256"),
        ("best_weights_path", "best_weights_sha256"),
        ("artifact_path", "artifact_sha256"),
    ):
        if _sha256(row.get(path_key)) != row.get(hash_key):
            raise ValueError(f"X1R resume文件hash漂移: {path_key}")
    artifact = joblib.load(row["artifact_path"])
    if (
        artifact.get("protocol_version") != PROTOCOL_VERSION
        or artifact.get("architecture_version") != ARCHITECTURE_VERSION
        or artifact.get("variant_id") != "x1r"
        or artifact.get("candidate_snapshot_before_gate_sha256")
        != artifact.get("candidate_snapshot_after_gate_sha256")
        or artifact.get("source_stage5a_training_marker_sha256")
        != _sha256(artifact.get("source_stage5a_training_marker_path"))
        or int(artifact.get("total_params", -1)) != EXPECTED_TOTAL_PARAMS["x1r"]
        or float(artifact.get("candidate_gate_max_abs_drift", np.nan)) != 0.0
        or float(artifact.get("persistence_gate_max_abs_drift", np.nan)) != 0.0
        or os.path.realpath(str(artifact.get("training_code_path", "")))
        != os.path.realpath(__file__)
        or artifact.get("training_code_sha256") != _sha256(__file__)
    ):
        raise ValueError("X1R resume artifact协议/冻结/来源不兼容")
    recorded_phases = {
        item.get("phase"): item.get("trainable_parameter_count")
        for item in artifact.get("gate_training", {}).get("phases", ())
    }
    if recorded_phases != EXPECTED_GATE_TRAINABLE_PARAMS["x1r"]:
        raise ValueError("X1R resume artifact的phase可训练参数计数漂移")
    validate_dependency_code_records(artifact.get("dependency_code_records"), "resume artifact")
    for path_key, hash_key in (
        ("model_path", "model_sha256"),
        ("best_weights_path", "best_weights_sha256"),
        ("history_path", "history_sha256"),
        ("history_figure_path", "history_figure_sha256"),
        ("validation_path", "validation_sha256"),
        ("checkpoint_path", "checkpoint_sha256"),
        ("soft_oracle_path", "soft_oracle_sha256"),
        ("q90_path", "q90_sha256"),
        ("candidate_provenance_path", "candidate_provenance_sha256"),
        ("fixed_g0_validation_replay_path", "fixed_g0_validation_replay_sha256"),
        ("tail_path", "tail_sha256"),
        ("source_x1_model_path", "source_x1_model_sha256"),
        ("source_x1_artifact_path", "source_x1_artifact_sha256"),
        ("source_x1_record_path", "source_x1_record_sha256"),
    ):
        if _sha256(artifact.get(path_key)) != artifact.get(hash_key):
            raise ValueError(f"X1R resume artifact成员hash漂移: {path_key}")
    return row


def write_manifest(result_root=RESULT_ROOT, run_scope="formal"):
    row = {
        "variant_id": "x1r",
        "label": VARIANT_SPECS["x1r"]["label"],
        "requires_training": True,
        "source_candidate": "stage5a_x1",
        "source_candidate_retrained": False,
        "old_g0_gate_reused": False,
        "new_gate_topology": "nonfactorized_sample_horizon",
        "calibration_weight": CALIBRATION_WEIGHT,
        "dynamic_weight": DYNAMIC_WEIGHT,
        "safety_weight": SAFETY_WEIGHT,
        "soft_oracle_scope": "per_farm_train_only",
        "q90_scope": "per_farm_per_horizon_train_only",
        "expected_total_params": EXPECTED_TOTAL_PARAMS["x1r"],
        "expected_gate_trainable_params": EXPECTED_GATE_TRAINABLE_PARAMS["x1r"][
            "objective"
        ],
        "parameter_limit_exclusive": PARAMETER_LIMIT,
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "selection_split": "test",
        "test_used_for_training": False,
        "test_is_final_blind_evaluation": False,
        "protocol_version": PROTOCOL_VERSION,
        "run_scope": run_scope,
    }
    return _atomic_to_csv(pd.DataFrame([row]), os.path.join(result_root, MANIFEST_NAME))


def publish_training_marker(summary_path, manifest_path, summary, source_identity):
    if (
        len(summary) != len(expected_farm_ids())
        or set(summary["farm_id"].astype(str)) != set(expected_farm_ids())
        or summary.duplicated(["farm_id"]).any()
    ):
        raise ValueError("X1R正式训练summary不是唯一5场站")
    required_freeze_fields = {
        "candidate_snapshot_sha256",
        "candidate_snapshot_after_gate_sha256",
        "candidate_gate_max_abs_drift",
        "persistence_gate_max_abs_drift",
    }
    if not required_freeze_fields.issubset(summary.columns):
        raise ValueError("X1R正式summary缺少candidate冻结证据字段")
    frozen_verified = bool(
        (
            summary["candidate_snapshot_sha256"].astype(str)
            == summary["candidate_snapshot_after_gate_sha256"].astype(str)
        ).all()
        and (pd.to_numeric(summary["candidate_gate_max_abs_drift"]) == 0.0).all()
        and (pd.to_numeric(summary["persistence_gate_max_abs_drift"]) == 0.0).all()
    )
    if not frozen_verified:
        raise ValueError("X1R正式summary未证明5场站candidate完全冻结")
    files = {
        "training_summary": _file_record(summary_path),
        "experiment_manifest": _file_record(manifest_path),
        "training_code": _file_record(__file__),
        "source_stage5a_training_marker": _file_record(source_identity["marker_path"]),
        "source_stage5a_training_summary": _file_record(source_identity["summary_path"]),
    }
    files.update(
        {f"dependency.{name}": record for name, record in dependency_code_records().items()}
    )
    for _, row in summary.iterrows():
        prefix = f"x1r.{row['farm_id']}"
        for key in (
            "model_path",
            "best_weights_path",
            "artifact_path",
            "history_path",
            "history_figure_path",
            "validation_path",
            "checkpoint_path",
            "soft_oracle_path",
            "q90_path",
            "candidate_provenance_path",
            "fixed_g0_validation_replay_path",
            "tail_path",
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
        "variants": ["x1r"],
        "new_training_variants": ["x1r"],
        "new_model_count": len(expected_farm_ids()),
        "source_x1_model_reused_count": len(expected_farm_ids()),
        "source_x1_candidate_retraining_forbidden": True,
        "source_x1_candidate_frozen_verified": frozen_verified,
        "old_g0_gate_pruned": True,
        "train_only_soft_oracle_q90": True,
        "test_prediction_read_during_training": False,
        "parameter_limit_exclusive": PARAMETER_LIMIT,
        "files": files,
    }
    path = _atomic_write_json(marker, os.path.join(RESULT_ROOT, TRAINING_MARKER_NAME))
    running = os.path.join(RESULT_ROOT, RUNNING_MARKER_NAME)
    if os.path.exists(running):
        os.remove(running)
    return path


def _discover_train_files(requested_farms=None):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "wind_train_*.csv")))
    if requested_farms:
        wanted = {str(value) for value in requested_farms}
        files = [
            path
            for path in files
            if re.search(r"wind_train_(\d+)\.csv$", os.path.basename(path)).group(1)
            in wanted
        ]
    return files


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--farms", default=os.getenv("WIND_X1R_FARMS", ""), help="逗号分隔场站ID"
    )
    parser.add_argument("--gate-only-epochs", type=int, default=GATE_ONLY_EPOCHS)
    parser.add_argument("--context-epochs", type=int, default=CONTEXT_EPOCHS)
    parser.add_argument("--objective-epochs", type=int, default=OBJECTIVE_EPOCHS)
    parser.add_argument(
        "--epochs", type=int, default=None, help="调试时覆盖三个phase；自动进入partial"
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _formal_protocol(args, farm_ids):
    return (
        not args.smoke_test
        and args.epochs is None
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
    source_identity = validate_source_bundle()
    farms = [value.strip() for value in args.farms.split(",") if value.strip()]
    invalid_farms = set(farms) - set(expected_farm_ids())
    if invalid_farms:
        raise ValueError(f"未知场站: {sorted(invalid_farms)}")
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs必须为正")
        args.gate_only_epochs = args.context_epochs = args.objective_epochs = args.epochs
    if args.smoke_test:
        farms = farms[:1] if farms else [expected_farm_ids()[0]]
        args.gate_only_epochs = args.context_epochs = args.objective_epochs = 1
    if any(
        value < 0
        for value in (
            args.gate_only_epochs,
            args.context_epochs,
            args.objective_epochs,
        )
    ):
        raise ValueError("epoch不得为负")
    if args.gate_only_epochs + args.context_epochs + args.objective_epochs <= 0:
        raise ValueError("X1R至少需要1个门控epoch")
    train_files = _discover_train_files(farms)
    if not train_files:
        raise FileNotFoundError("没有匹配的训练文件")
    farm_ids = [regime_train.get_farm_id(path) for path in train_files]
    formal = _formal_protocol(args, farm_ids)
    if formal:
        run_root, run_scope = RESULT_ROOT, "formal"
        _atomic_write_json(
            {
                "status": "running",
                "protocol_version": PROTOCOL_VERSION,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "training_code": _file_record(__file__),
                "farm_ids": farm_ids,
                "force": bool(args.force),
            },
            os.path.join(RESULT_ROOT, RUNNING_MARKER_NAME),
        )
        downstream = os.path.join(RESULT_ROOT, PREDICTION_MARKER_RELATIVE_PATH)
        if os.path.exists(downstream):
            os.remove(downstream)
    else:
        run_root = os.path.join(
            RESULT_ROOT,
            "partial_runs",
            datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
        )
        run_scope = "smoke_or_partial_or_protocol_override"
    manifest_path = write_manifest(run_root, run_scope)
    print(
        f"X1R farms={farm_ids}; output={run_root}; formal={formal}; "
        f"seed={RANDOM_SEED}; batch={BATCH_SIZE}"
    )
    rows = []
    for train_file in train_files:
        prepared = regime_train._prepare_farm(train_file)
        record_path = _paths(result_dirs(result_root=run_root), prepared["farm_id"])[
            "record_path"
        ]
        completed = None if args.force else _validate_completed_record(
            record_path, prepared["farm_id"]
        )
        if completed is not None:
            print(f"跳过已验证完成模型: x1r/{prepared['farm_id']}")
            rows.append(completed)
            continue
        print(f"\n===== X1R frozen-X1 gate closure / farm={prepared['farm_id']} =====")
        rows.append(
            train_farm(
                prepared,
                source_identity,
                result_root=run_root,
                gate_only_epochs=args.gate_only_epochs,
                context_epochs=args.context_epochs,
                objective_epochs=args.objective_epochs,
            )
        )
    summary = pd.DataFrame(rows)
    if summary.empty or summary.duplicated(["farm_id"]).any():
        raise ValueError("X1R训练summary为空或场站重复")
    summary_path = _atomic_to_csv(
        summary, os.path.join(run_root, TRAINING_SUMMARY_NAME)
    )
    print(f"X1R训练汇总: {summary_path}")
    if formal:
        marker = publish_training_marker(
            summary_path, manifest_path, summary, source_identity
        )
        print(f"X1R正式训练bundle完成: {marker}")
        print("新增训练模型数=5；Stage-5A X1 candidate未重复训练")
    else:
        print("partial/smoke已隔离，不发布formal complete marker")


if __name__ == "__main__":
    main()
