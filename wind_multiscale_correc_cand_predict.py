"""Stage-5A X0/X1-F/X1-M/X1-C/X1 多尺度 corrected candidate 测试与选型。

本轮只回答一个问题：轻量 fine/mid/coarse 历史表示能否改善 corrected
candidate。本文件因此把 candidate 指标作为正式主指标；冻结的既有 G0 门控
回放只作为闭环可转化性诊断，不参与 Stage-5A 晋级判定。

X0 通过 Stage-4B complete marker 逐文件 hash 校验后直接引用 D0/F7 结果，
不加载模型、不 forward、不复制模型、artifact、预测 CSV 或 candidate archive。
X1-F/X1-M/X1-C/X1 各场站只执行一次测试前向。

默认执行正式五变体、五场站矩阵::

    python wind_multiscale_correc_cand_predict.py

``--smoke``、子集运行、``--max-samples`` 或 ``--skip-plots`` 的结果写入
``partial_runs``，不会发布正式 complete marker。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import wind_controlled_gate_cali_predict as gate_predict
import wind_dl_model_predict as common_predict
import wind_multiscale_correc_cand_train as multiscale_train
import wind_time_freq_model_predict as stage4_predict


OUTPUT_SUBDIR = "testdata_predict_output"
TEST_REUSE_STATUS = "legacy_seen_test_selected"
FORMAL_MARKER_NAME = "multiscale_correc_cand_test_bundle_complete.json"
RUNNING_MARKER_NAME = "multiscale_correc_cand_test_bundle_running.json"
ALL_VARIANTS = tuple(multiscale_train.VARIANT_SPECS)
NEW_VARIANTS = tuple(multiscale_train.TRAINABLE_VARIANTS)

STAGE4B_ROOT = os.path.join(
    "./wind_results",
    "time_freq_model",
    "supplement_round2_stage4b_gate_closure",
)
STAGE4B_OUTPUT = os.path.join(STAGE4B_ROOT, OUTPUT_SUBDIR)
STAGE4B_MARKER = os.path.join(
    STAGE4B_OUTPUT, "stage4b_gate_closure_test_bundle_complete.json"
)
STAGE4B_FILES = {
    "summary": "stage4b_gate_closure_test_summary.csv",
    "candidate": "stage4b_gate_closure_test_candidate.csv",
    "regime": "stage4b_gate_closure_test_regime.csv",
    "assignments": "stage4b_gate_closure_test_assignments.csv",
}

# 预声明的 Stage-5A candidate 晋级门槛。X0 是安全 fallback。
REQUIRED_MACRO_IMPROVEMENT = 0.003
FARM_NONDEGRADE_ATOL = 1e-12
MIN_NONDEGRADED_FARMS = 4
MIN_STRICTLY_IMPROVED_FARMS = 3
REGIME_RELATIVE_DEGRADATION_TOL = 0.005
PARAMETER_LIMIT = 30_000
NRMSE_TIE_TOL = 0.001

# 同窗、Persistence 和冻结 G0 回放不变量容差。
PERSISTENCE_MAX_NORM_TOL = 1e-6
PERSISTENCE_MEAN_NORM_TOL = 1e-7
G0_GATE_MAX_ABS_TOL = 1e-4
G0_GATE_MEAN_ABS_TOL = 1e-6
FUSED_REPLAY_MAX_NORM_TOL = 1e-4
FUSED_REPLAY_MEAN_NORM_TOL = 1e-6
# X0 archive来自较早TensorFlow运行；当前环境对全部5场站用同一F7权重
# 全测试集重建后，实测最坏max≈2.11e-4、mean≈1.99e-5（容量归一化）。
# 权重hash仍须严格一致，输出仅使用显式跨运行时容量容差，避免把数值内核
# 差异误判为模型漂移。
BASE_CORRECTED_MAX_NORM_TOL = 2.5e-4
BASE_CORRECTED_MEAN_NORM_TOL = 2.5e-5

# 复用已经验证过的原子保存与指标原语，但不改动任何既有磁盘产物。
_sha256 = stage4_predict._sha256
_file_record = stage4_predict._file_record
_atomic_csv = stage4_predict._atomic_csv
_atomic_json = stage4_predict._atomic_json
_atomic_text = stage4_predict._atomic_text
_atomic_npz = stage4_predict._atomic_npz
_validate_record = stage4_predict._validate_record

# gate_predict 的低层 payload/工况函数在运行时查询训练模块全局变量。
# 显式绑定本轮模块，确保 family、variant 和 forecast_len 均属于 Stage-5A。
gate_predict.gate_train = multiscale_train


def _variant_label(variant: str) -> str:
    spec = multiscale_train.VARIANT_SPECS[variant]
    return str(spec.get("label", variant.upper()))


def _model_name(variant: str) -> str:
    if hasattr(multiscale_train, "variant_model_name"):
        return multiscale_train.variant_model_name(variant)
    return f"{multiscale_train.MODEL_FAMILY}_{variant}"


def _expected_farms() -> list[str]:
    return [str(value) for value in multiscale_train.expected_farm_ids()]


def _canonical_horizon(value) -> str:
    text = str(value).strip().lower()
    return "all" if text == "all" else str(int(float(text)))


def _normalize_variant(value: str) -> str:
    value = value.strip().lower().replace("-", "_")
    aliases = {"x1f": "x1_f", "x1m": "x1_m", "x1c": "x1_c"}
    return aliases.get(value, value)


def _parse_list(raw: str, allowed, label: str, variant=False) -> list[str]:
    normalize = _normalize_variant if variant else lambda value: value.strip()
    values = list(dict.fromkeys(normalize(item) for item in raw.split(",") if item.strip()))
    invalid = set(values) - set(map(str, allowed))
    if invalid or not values:
        raise ValueError(f"{label}非法: {sorted(invalid)}")
    return values


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _marker_file(marker: dict, key: str, expected_path: str | None = None) -> str:
    record = marker.get("files", {}).get(key)
    if record is None:
        raise KeyError(f"complete marker缺少files.{key}")
    path = _validate_record(key, record)
    if expected_path and os.path.realpath(path) != os.path.realpath(expected_path):
        raise ValueError(f"files.{key}路径漂移: {path} != {expected_path}")
    return path


def _relabel_x0(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["source_model_family"] = frame.get("model_family", "")
    frame["source_model_variant"] = frame.get("model_variant", "")
    frame["model_family"] = multiscale_train.MODEL_FAMILY
    frame["model_variant"] = "x0"
    if "model_name" in frame:
        frame["model_name"] = _model_name("x0")
    return frame


def validate_stage4b_x0_source():
    """Hash 校验 Stage-4B bundle，并构造只读 X0/F7 candidate 表。"""
    if not os.path.isfile(STAGE4B_MARKER):
        raise FileNotFoundError(f"缺少Stage-4B complete marker: {STAGE4B_MARKER}")
    marker = _read_json(STAGE4B_MARKER)
    if marker.get("status") != "complete":
        raise ValueError("Stage-4B source marker不是complete")
    expected = set(_expected_farms())
    if set(map(str, marker.get("expected_farm_ids", ()))) != expected:
        raise ValueError("Stage-4B source marker未锁定固定5场站")
    if set(map(str, marker.get("test_files", {}))) != expected:
        raise ValueError("Stage-4B source marker test_files未覆盖固定5场站")
    for key, record in marker.get("files", {}).items():
        _validate_record(f"Stage-4B files.{key}", record)
    for farm_id, record in marker.get("test_files", {}).items():
        _validate_record(f"Stage-4B test_files.{farm_id}", record)

    raw, paths = {}, {}
    for key, filename in STAGE4B_FILES.items():
        expected_path = os.path.join(STAGE4B_OUTPUT, filename)
        path = _marker_file(marker, f"formal.{key}", expected_path)
        raw[key] = pd.read_csv(path, dtype={"farm_id": str})
        paths[key] = path

    source_summary = raw["summary"][
        raw["summary"]["model_variant"].astype(str) == "d0"
    ].copy()
    source_candidate = raw["candidate"][
        raw["candidate"]["model_variant"].astype(str) == "d0"
    ].copy()
    source_regime = raw["regime"][
        raw["regime"]["model_variant"].astype(str) == "d0"
    ].copy()
    source_assignments = raw["assignments"][
        raw["assignments"]["model_variant"].astype(str) == "d0"
    ].copy()
    for name, frame in {
        "summary": source_summary,
        "candidate": source_candidate,
        "regime": source_regime,
        "assignments": source_assignments,
    }.items():
        if set(frame["farm_id"].astype(str)) != expected:
            raise ValueError(f"Stage-4B D0 {name}未覆盖固定5场站")

    source_candidate["candidate"] = source_candidate["candidate"].replace(
        {"fused": "frozen_g0_replay"}
    )
    source_regime["candidate"] = source_regime["candidate"].replace(
        {"fused": "frozen_g0_replay"}
    )
    candidate = _relabel_x0(source_candidate)
    regime = _relabel_x0(source_regime)
    assignments = _relabel_x0(source_assignments)

    horizon = candidate[
        candidate["candidate"].astype(str) == "corrected"
    ].copy()
    horizon["metric_role"] = "primary_corrected_candidate"
    horizon["formal_metric_source"] = (
        "hash_validated_stage4b_d0_f7_corrected_candidate_direct_reference"
    )

    primary = horizon[horizon["horizon_step"].map(_canonical_horizon) == "all"].copy()
    if len(primary) != len(expected):
        raise ValueError("X0 corrected candidate overall指标不是5场站")
    source_meta = source_summary.set_index("farm_id")
    replay_all = candidate[
        (candidate["candidate"] == "frozen_g0_replay")
        & (candidate["horizon_step"].map(_canonical_horizon) == "all")
    ].set_index("farm_id")
    persistence_all = candidate[
        (candidate["candidate"] == "persistence")
        & (candidate["horizon_step"].map(_canonical_horizon) == "all")
    ].set_index("farm_id")
    rows = []
    for _, metric in primary.iterrows():
        farm_id = str(metric["farm_id"])
        meta = source_meta.loc[farm_id]
        row = metric.to_dict()
        row.update(
            {
                "model_family": multiscale_train.MODEL_FAMILY,
                "model_variant": "x0",
                "variant_label": _variant_label("x0"),
                "primary_metric_candidate": "corrected",
                "selection_metric_scope": "corrected_candidate",
                "corrected_candidate_nrmse": float(metric["capacity_normalized_rmse"]),
                "corrected_candidate_nmae": float(metric["capacity_normalized_mae"]),
                "frozen_g0_replay_nrmse": float(
                    replay_all.loc[farm_id, "capacity_normalized_rmse"]
                ),
                "frozen_g0_replay_nmae": float(
                    replay_all.loc[farm_id, "capacity_normalized_mae"]
                ),
                "persistence_nrmse": float(
                    persistence_all.loc[farm_id, "capacity_normalized_rmse"]
                ),
                "parameter_count": int(meta["parameter_count"]),
                "trainable_parameter_count": 0,
                "multiscale_added_parameter_count": 0,
                "training_elapsed_seconds": 0.0,
                "inference_elapsed_seconds": 0.0,
                "inference_milliseconds_per_sample": 0.0,
                "reference_only": True,
                "selection_eligible": True,
                "result_source": (
                    "hash_validated_stage4b_d0_f7_direct_reference_no_training_"
                    "no_forward_no_copy"
                ),
                "diagnostic_source": "existing_stage4b_d0_frozen_g0_fused",
                "selection_split": "test",
                "test_used_for_selection": True,
                "test_is_final_blind_evaluation": False,
                "test_reuse_status": TEST_REUSE_STATUS,
                "random_seed": multiscale_train.RANDOM_SEED,
                "model_path": meta.get("model_path"),
                "model_sha256": meta.get("model_sha256"),
                "artifact_path": meta.get("artifact_path"),
                "artifact_sha256": meta.get("artifact_sha256"),
                "prediction_path": meta.get("prediction_path"),
                "candidate_archive_path": meta.get("candidate_archive_path"),
                "candidate_archive_sha256": meta.get("candidate_archive_sha256"),
                "single_window_figure_path": meta.get("single_window_figure_path"),
                "weighted_curve_figure_path": meta.get("weighted_curve_figure_path"),
            }
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    return marker, {
        "summary": summary,
        "horizon": horizon,
        "candidate": candidate,
        "regime": regime,
        "assignments": assignments,
    }, paths, source_summary


def validate_training_bundle(required_variants):
    if not required_variants:
        return None, None
    running_name = getattr(
        multiscale_train,
        "RUNNING_MARKER_NAME",
        "multiscale_correc_cand_training_running.json",
    )
    running = os.path.join(multiscale_train.RESULT_ROOT, running_name)
    if os.path.isfile(running):
        raise RuntimeError(f"Stage-5A训练仍在运行或未完整收尾: {running}")
    path = os.path.join(
        multiscale_train.RESULT_ROOT, multiscale_train.TRAINING_MARKER_NAME
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少Stage-5A训练complete marker: {path}")
    marker = _read_json(path)
    if marker.get("status") != "complete":
        raise ValueError("Stage-5A训练marker不是complete")
    if marker.get("protocol_version") != multiscale_train.PROTOCOL_VERSION:
        raise ValueError("Stage-5A训练marker协议不匹配")
    architecture = getattr(multiscale_train, "ARCHITECTURE_VERSION", None)
    if architecture and marker.get("architecture_version") != architecture:
        raise ValueError("Stage-5A训练marker架构版本不匹配")
    if set(map(str, marker.get("expected_farm_ids", ()))) != set(_expected_farms()):
        raise ValueError("Stage-5A训练marker未锁定固定5场站")
    if not bool(marker.get("same_scale_initialization_single_vs_x1_verified")):
        raise ValueError("Stage-5A训练marker未证明single与X1同尺度同初始化")
    if not bool(marker.get("token_interaction_forbidden")):
        raise ValueError("Stage-5A训练marker没有锁定无token交互协议")
    if not bool(marker.get("x0_retraining_forbidden")):
        raise ValueError("Stage-5A训练marker没有锁定X0只读引用协议")
    marker_variants = set(marker.get("trained_variants", marker.get("variants", ())))
    if not set(required_variants).issubset(marker_variants):
        raise ValueError("Stage-5A训练marker未覆盖请求的新变体")
    for key, record in marker.get("files", {}).items():
        _validate_record(f"Stage-5A training files.{key}", record)
    return path, marker


def _resolve_training_record(marker, variant: str, farm_id: str, kind: str) -> str:
    files = marker.get("files", {})
    direct_keys = (
        f"{variant}.{farm_id}.{kind}_path",
        f"{variant}.{farm_id}.{kind}",
        f"{variant}/{farm_id}/{kind}",
    )
    for key in direct_keys:
        if key in files:
            return _validate_record(f"training {key}", files[key])
    suffixes = {"model": (".keras",), "artifact": (".pkl", ".joblib")}[kind]
    candidates = []
    for key, record in files.items():
        path = str(record.get("path", ""))
        lower = path.lower()
        if (
            variant in key.lower()
            and str(farm_id) in key
            and lower.endswith(suffixes)
            and (kind in key.lower() or kind in os.path.basename(lower))
        ):
            candidates.append((key, record))
    if len(candidates) != 1:
        raise KeyError(
            f"训练marker无法唯一解析{variant}/{farm_id}/{kind}: "
            f"{[key for key, _ in candidates]}"
        )
    key, record = candidates[0]
    return _validate_record(f"training {key}", record)


def _load_model(variant: str, farm_id: str, marker: dict):
    artifact_path = _resolve_training_record(marker, variant, farm_id, "artifact")
    model_path = _resolve_training_record(marker, variant, farm_id, "model")
    artifact = joblib.load(artifact_path)
    source_model_path = str(artifact.get("source_f7_model_path", ""))
    source_artifact_path = str(artifact.get("source_f7_artifact_path", ""))
    checks = {
        "variant": str(artifact.get("variant_id")) == variant,
        "farm": str(artifact.get("farm_id")) == str(farm_id),
        "family": artifact.get("model_family") == multiscale_train.MODEL_FAMILY,
        "protocol": artifact.get("protocol_version") == multiscale_train.PROTOCOL_VERSION,
        "seed": int(artifact.get("random_seed", -1)) == multiscale_train.RANDOM_SEED,
        "history_len": int(artifact.get("history_len", -1))
        == multiscale_train.HISTORY_LEN,
        "forecast_len": int(artifact.get("forecast_len", -1))
        == multiscale_train.FORECAST_LEN,
        "model_path": os.path.realpath(str(artifact.get("model_path", "")))
        == os.path.realpath(model_path),
        "model_hash": artifact.get("model_sha256") == _sha256(model_path),
        "source_snapshot_frozen": bool(
            artifact.get("source_snapshot_frozen_verified")
        )
        and bool(artifact.get("source_snapshot_before_sha256"))
        and artifact.get("source_snapshot_before_sha256")
        == artifact.get("source_snapshot_after_sha256"),
        "persistence_probe_frozen": float(
            artifact.get("persistence_probe_max_abs_drift", np.nan)
        )
        == 0.0,
        "g0_gate_probe_frozen": float(
            artifact.get("g0_gate_probe_max_abs_drift", np.nan)
        )
        == 0.0,
        "source_f7_model_file": os.path.isfile(source_model_path)
        and _sha256(source_model_path) == artifact.get("source_f7_model_sha256"),
        "source_f7_artifact_file": os.path.isfile(source_artifact_path)
        and _sha256(source_artifact_path)
        == artifact.get("source_f7_artifact_sha256"),
        "parameter_count": int(artifact.get("total_params", -1))
        == multiscale_train.EXPECTED_TOTAL_PARAMS[variant],
        "adapter_parameter_count": int(
            artifact.get("multiscale_trainable_parameter_count", -1)
        )
        == multiscale_train.EXPECTED_ADAPTER_TRAINABLE_PARAMS[variant],
        "scale_definition": tuple(
            artifact.get("multiscale_definition", {}).get("scales", ())
        )
        == tuple(multiscale_train.VARIANT_SPECS[variant]["scales"]),
        "no_token_interaction": artifact.get("multiscale_definition", {}).get(
            "token_interaction"
        )
        is False,
        "candidate_only_training": bool(
            artifact.get("candidate_training", {}).get(
                "f7_residual_context_g0_gate_frozen"
            )
        )
        and float(
            artifact.get("candidate_training", {}).get(
                "forecast_power_loss_weight", np.nan
            )
        )
        == 0.0
        and float(
            artifact.get("candidate_training", {}).get(
                "candidate_forecast_loss_weight", np.nan
            )
        )
        == 1.0,
    }
    architecture = getattr(multiscale_train, "ARCHITECTURE_VERSION", None)
    if architecture:
        checks["architecture"] = artifact.get("architecture_version") == architecture
    if hasattr(multiscale_train, "ARTIFACT_SCHEMA_VERSION"):
        checks["artifact_schema"] = int(artifact.get("artifact_schema_version", -1)) == int(
            multiscale_train.ARTIFACT_SCHEMA_VERSION
        )
    # 训练端必须证明 F7/G0 冻结公共分支未因 candidate 训练漂移。
    for before, after, label in (
        (
            "source_f7_snapshot_before_training_sha256",
            "source_f7_snapshot_after_training_sha256",
            "f7_snapshot_frozen",
        ),
        (
            "source_g0_snapshot_before_training_sha256",
            "source_g0_snapshot_after_training_sha256",
            "g0_snapshot_frozen",
        ),
        (
            "source_f7_probe_before_training_sha256",
            "source_f7_probe_after_training_sha256",
            "f7_probe_frozen",
        ),
        (
            "source_g0_probe_before_training_sha256",
            "source_g0_probe_after_training_sha256",
            "g0_probe_frozen",
        ),
    ):
        if before in artifact or after in artifact:
            checks[label] = bool(artifact.get(before)) and artifact.get(before) == artifact.get(after)
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{variant}/{farm_id} artifact校验失败: {failed}")
    if hasattr(multiscale_train, "validate_dependency_code_records"):
        multiscale_train.validate_dependency_code_records(
            artifact.get("dependency_code_records"),
            role=f"{variant}/{farm_id} prediction artifact",
        )
    else:
        dependency_records = artifact.get("dependency_code", {})
        if not isinstance(dependency_records, dict) or not dependency_records:
            raise ValueError(f"{variant}/{farm_id} artifact缺少dependency_code锁定")
        for key, record in dependency_records.items():
            _validate_record(f"{variant}/{farm_id} dependency.{key}", record)
    model = keras.models.load_model(
        model_path,
        custom_objects=multiscale_train.get_multiscale_custom_objects(),
        compile=False,
    )
    expected_params = artifact.get("total_params")
    if expected_params is not None and int(model.count_params()) != int(expected_params):
        raise ValueError(f"{variant}/{farm_id}模型参数量与artifact不一致")
    if int(model.count_params()) >= PARAMETER_LIMIT:
        raise ValueError(f"{variant}/{farm_id}参数量{model.count_params()}不小于30k")
    return artifact, artifact_path, model, model_path


def _normal_output(value, shape, label):
    value = np.asarray(value, dtype=np.float64)
    if value.shape == (shape[0], 1):
        value = np.repeat(value, shape[1], axis=1)
    elif value.shape == (shape[1],):
        value = np.repeat(value[None, :], shape[0], axis=0)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{label} shape/finite异常: {value.shape} != {shape}")
    return value


def _pick_output(outputs: dict, names, label: str):
    for name in names:
        if name in outputs:
            return outputs[name]
    raise KeyError(f"diagnostic输出缺少{label}，尝试键={tuple(names)}")


def predict_variant(variant, test_file, marker, max_samples=None):
    farm_id = str(common_predict.get_farm_id(test_file))
    artifact, artifact_path, model, model_path = _load_model(variant, farm_id, marker)
    df, features, actual_power, capacity = common_predict.prepare_prediction_arrays(
        test_file, artifact
    )
    history_len = int(artifact["history_len"])
    forecast_len = int(artifact["forecast_len"])
    if max_samples is not None:
        keep = history_len + forecast_len + int(max_samples) - 1
        df = df.iloc[:keep]
        features = features[:keep]
        actual_power = actual_power[:keep]
    dataset, n_samples = common_predict.make_prediction_dataset(
        features, history_len, forecast_len
    )
    diagnostic = multiscale_train.diagnostic_model(model)
    started = time.perf_counter()
    outputs = diagnostic.predict(dataset, verbose=common_predict.PREDICT_VERBOSE)
    elapsed = float(time.perf_counter() - started)
    if not isinstance(outputs, dict):
        raise TypeError(f"{variant}/{farm_id} diagnostic_model必须返回dict")
    shape = (n_samples, forecast_len)
    corrected = _normal_output(
        _pick_output(outputs, ("corrected", "candidate_forecast"), "candidate"),
        shape,
        f"{variant}/{farm_id}/corrected",
    )
    base_corrected = _normal_output(
        _pick_output(
            outputs,
            ("base_corrected", "source_f7_corrected", "base_candidate_forecast"),
            "frozen source F7 corrected candidate",
        ),
        shape,
        f"{variant}/{farm_id}/base_corrected",
    )
    fused = _normal_output(
        _pick_output(outputs, ("forecast", "forecast_power"), "frozen G0 replay"),
        shape,
        f"{variant}/{farm_id}/forecast",
    )
    persistence = _normal_output(
        _pick_output(outputs, ("persistence", "persistence_forecast_candidate"), "persistence"),
        shape,
        f"{variant}/{farm_id}/persistence",
    )
    gate = _normal_output(
        _pick_output(outputs, ("gate", "frozen_g0_gate"), "frozen G0 gate"),
        shape,
        f"{variant}/{farm_id}/gate",
    )
    q = np.repeat(np.mean(gate, axis=1, keepdims=True), forecast_len, axis=1)
    s = np.ones_like(gate)
    y_true = common_predict.build_truth_windows(
        actual_power, n_samples, history_len, forecast_len
    )
    payload = gate_predict._build_payload(
        variant,
        farm_id,
        df,
        artifact,
        {
            "forecast": fused,
            "persistence": persistence,
            "corrected": corrected,
            "gate": gate,
            "q": q,
            "s": s,
        },
        y_true,
        capacity,
        history_len,
    )
    payload["base_corrected_scaled"] = base_corrected
    payload["base_corrected"] = gate_predict._inverse_scaled(
        artifact, base_corrected, capacity
    )
    payload.update(
        {
            "model_path": model_path,
            "model_sha256": _sha256(model_path),
            "artifact_path": artifact_path,
            "artifact_sha256": _sha256(artifact_path),
            "parameter_count": int(model.count_params()),
            "trainable_parameter_count": int(
                artifact.get(
                    "multiscale_trainable_parameter_count",
                    artifact.get(
                        "trainable_parameter_count",
                        sum(int(np.prod(weight.shape)) for weight in model.trainable_weights),
                    ),
                )
            ),
            "multiscale_added_parameter_count": int(
                artifact.get(
                    "multiscale_added_parameter_count",
                    artifact.get("adapter_parameter_count", 0),
                )
            ),
            "training_elapsed_seconds": float(
                artifact.get(
                    "training_elapsed_seconds",
                    artifact.get("candidate_training_elapsed_seconds", np.nan),
                )
            ),
            "reference_only": False,
            "result_source": "stage5a_single_formal_test_forward",
            "diagnostic_source": "same_forward_frozen_g0_replay_not_selection_target",
            "inference_elapsed_seconds": elapsed,
            "inference_milliseconds_per_sample": 1000.0 * elapsed / n_samples,
        }
    )
    return payload


def prediction_dirs(variant: str, output_root: str) -> dict[str, str]:
    if hasattr(multiscale_train, "variant_dirs"):
        try:
            formal_root = multiscale_train.variant_dirs(variant)["root"]
            if os.path.realpath(output_root) == os.path.realpath(multiscale_train.RESULT_ROOT):
                root = os.path.join(formal_root, OUTPUT_SUBDIR)
            else:
                root = os.path.join(output_root, variant, OUTPUT_SUBDIR)
        except (KeyError, TypeError):
            root = os.path.join(output_root, variant, OUTPUT_SUBDIR)
    else:
        root = os.path.join(output_root, variant, OUTPUT_SUBDIR)
    dirs = {
        "root": root,
        "predictions": os.path.join(root, "predictions"),
        "candidate_metrics": os.path.join(root, "candidate_metrics"),
        "regime_metrics": os.path.join(root, "regime_metrics"),
        "regime_assignments": os.path.join(root, "regime_assignments"),
        "candidate_archives": os.path.join(root, "candidate_archives"),
        "safety_diagnostics": os.path.join(root, "frozen_g0_replay_diagnostics"),
        "calibration": os.path.join(root, "frozen_g0_replay_calibration"),
        "gate_diagnostics": os.path.join(root, "frozen_g0_gate_points"),
        "single_windows": os.path.join(root, "single_window_comparisons"),
        "weighted_curves": os.path.join(root, "weighted_curves"),
        "figures": os.path.join(root, "figures"),
        "matplotlib_cache": os.path.join(root, "matplotlib_cache"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def _candidate_metrics(payload):
    frames = []
    for candidate, values in (
        ("corrected", payload["corrected"]),
        ("persistence", payload["persistence"]),
        ("frozen_g0_replay", payload["fused"]),
    ):
        frame = common_predict.metrics_by_horizon(
            _model_name(payload["variant_id"]),
            payload["farm_id"],
            payload["y_true"],
            values,
            payload["capacity"],
            payload["forecast_len"],
        )
        frame["model_family"] = multiscale_train.MODEL_FAMILY
        frame["model_variant"] = payload["variant_id"]
        frame["candidate"] = candidate
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _regime_metrics(payload):
    frame = gate_predict._regime_metrics(payload).copy()
    frame["candidate"] = frame["candidate"].replace(
        {"fused": "frozen_g0_replay"}
    )
    frame["model_family"] = multiscale_train.MODEL_FAMILY
    frame["model_variant"] = payload["variant_id"]
    return frame


def save_payload(payload, output_root, skip_plots=False):
    variant, farm_id = payload["variant_id"], payload["farm_id"]
    name, dirs = _model_name(variant), prediction_dirs(variant, output_root)
    prediction = common_predict.build_prediction_frame(
        name,
        payload["df"],
        farm_id,
        payload["corrected"],
        payload["y_true"],
        payload["history_len"],
        payload["forecast_len"],
    )
    prediction["prediction_role"] = "primary_corrected_candidate"
    prediction["persistence_power"] = payload["persistence"].T.reshape(-1)
    prediction["frozen_g0_replay_power"] = payload["fused"].T.reshape(-1)
    prediction_path = _atomic_csv(
        prediction,
        os.path.join(dirs["predictions"], f"{name}_predictions_farm_{farm_id}.csv"),
    )
    candidates = _candidate_metrics(payload)
    candidate_path = _atomic_csv(
        candidates,
        os.path.join(
            dirs["candidate_metrics"], f"{name}_candidate_metrics_farm_{farm_id}.csv"
        ),
    )
    horizon = candidates[candidates["candidate"] == "corrected"].copy()
    horizon["metric_role"] = "primary_corrected_candidate"
    horizon_path = _atomic_csv(
        horizon,
        os.path.join(dirs["root"], f"{name}_metrics_by_horizon_farm_{farm_id}.csv"),
    )
    regimes = _regime_metrics(payload)
    regime_path = _atomic_csv(
        regimes,
        os.path.join(dirs["regime_metrics"], f"{name}_regime_metrics_farm_{farm_id}.csv"),
    )
    assignments = gate_predict._assignment_frame(payload)
    assignments["model_family"] = multiscale_train.MODEL_FAMILY
    assignments["model_variant"] = variant
    assignment_path = _atomic_csv(
        assignments,
        os.path.join(
            dirs["regime_assignments"], f"{name}_regime_assignments_farm_{farm_id}.csv"
        ),
    )
    safety = gate_predict.build_safety_scope_frame(payload)
    safety["model_family"] = multiscale_train.MODEL_FAMILY
    safety["model_variant"] = variant
    safety["diagnostic_only"] = True
    safety_path = _atomic_csv(
        safety,
        os.path.join(
            dirs["safety_diagnostics"], f"{name}_frozen_g0_safety_farm_{farm_id}.csv"
        ),
    )
    calibration = gate_predict.build_reliability_frame(payload)
    calibration["model_family"] = multiscale_train.MODEL_FAMILY
    calibration["model_variant"] = variant
    calibration["diagnostic_only"] = True
    calibration_path = _atomic_csv(
        calibration,
        os.path.join(
            dirs["calibration"], f"{name}_frozen_g0_reliability_farm_{farm_id}.csv"
        ),
    )
    gate_points = gate_predict.build_point_gate_frame(payload)
    gate_points["model_variant"] = variant
    gate_path = _atomic_csv(
        gate_points,
        os.path.join(
            dirs["gate_diagnostics"], f"{name}_frozen_g0_points_farm_{farm_id}.csv"
        ),
    )
    archive_path = _atomic_npz(
        os.path.join(
            dirs["candidate_archives"], f"{name}_candidate_archive_farm_{farm_id}.npz"
        ),
        schema_version=np.asarray("multiscale_correc_cand_archive_v1"),
        model_variant=np.asarray(variant),
        farm_id=np.asarray(farm_id),
        sample_id=payload["sample_id"],
        horizon_step=payload["horizon_step"],
        forecast_origin_time=payload["forecast_origin_time"],
        capacity=np.asarray(payload["capacity"]),
        y_true=payload["y_true"],
        y=payload["y_true"],
        persistence=payload["persistence"],
        P=payload["persistence"],
        corrected=payload["corrected"],
        C=payload["corrected"],
        frozen_g0_replay=payload["fused"],
        F=payload["fused"],
        persistence_scaled=payload["persistence_scaled"],
        corrected_scaled=payload["corrected_scaled"],
        base_corrected_scaled=payload["base_corrected_scaled"],
        base_corrected=payload["base_corrected"],
        frozen_g0_replay_scaled=payload["fused_scaled"],
        frozen_g0_gate=payload["applied_gate"],
        q=payload["q"],
        s=payload["s"],
    )
    single_path = single_figure = weighted_path = weighted_figure = None
    replay_single_path = replay_single_figure = None
    replay_weighted_path = replay_weighted_figure = None
    weighted_metrics = {}
    if not skip_plots:
        single_path, single_figure = common_predict.save_single_window_plot(
            prediction, name, farm_id, dirs, payload["forecast_len"]
        )
        weighted_path, weighted_figure, weighted_metrics = (
            common_predict.save_weighted_full_test_plot(
                prediction, name, farm_id, dirs, payload["capacity"]
            )
        )
        replay_prediction = common_predict.build_prediction_frame(
            f"{name}_frozen_g0_replay",
            payload["df"],
            farm_id,
            payload["fused"],
            payload["y_true"],
            payload["history_len"],
            payload["forecast_len"],
        )
        replay_single_path, replay_single_figure = common_predict.save_single_window_plot(
            replay_prediction,
            f"{name}_frozen_g0_replay",
            farm_id,
            dirs,
            payload["forecast_len"],
        )
        replay_weighted_path, replay_weighted_figure, _ = (
            common_predict.save_weighted_full_test_plot(
                replay_prediction,
                f"{name}_frozen_g0_replay",
                farm_id,
                dirs,
                payload["capacity"],
            )
        )
    overall = horizon[horizon["horizon_step"].map(_canonical_horizon) == "all"].iloc[0]
    replay = candidates[
        (candidates["candidate"] == "frozen_g0_replay")
        & (candidates["horizon_step"].map(_canonical_horizon) == "all")
    ].iloc[0]
    persistence = candidates[
        (candidates["candidate"] == "persistence")
        & (candidates["horizon_step"].map(_canonical_horizon) == "all")
    ].iloc[0]
    summary = {
        **overall.to_dict(),
        "model_family": multiscale_train.MODEL_FAMILY,
        "model_variant": variant,
        "variant_label": _variant_label(variant),
        "primary_metric_candidate": "corrected",
        "selection_metric_scope": "corrected_candidate",
        "corrected_candidate_nrmse": float(overall["capacity_normalized_rmse"]),
        "corrected_candidate_nmae": float(overall["capacity_normalized_mae"]),
        "frozen_g0_replay_nrmse": float(replay["capacity_normalized_rmse"]),
        "frozen_g0_replay_nmae": float(replay["capacity_normalized_mae"]),
        "persistence_nrmse": float(persistence["capacity_normalized_rmse"]),
        "parameter_count": payload["parameter_count"],
        "trainable_parameter_count": payload["trainable_parameter_count"],
        "multiscale_added_parameter_count": payload["multiscale_added_parameter_count"],
        "training_elapsed_seconds": payload["training_elapsed_seconds"],
        "inference_elapsed_seconds": payload["inference_elapsed_seconds"],
        "inference_milliseconds_per_sample": payload[
            "inference_milliseconds_per_sample"
        ],
        "reference_only": False,
        "selection_eligible": True,
        "result_source": payload["result_source"],
        "diagnostic_source": payload["diagnostic_source"],
        "selection_split": "test",
        "test_used_for_selection": True,
        "test_is_final_blind_evaluation": False,
        "test_reuse_status": TEST_REUSE_STATUS,
        "random_seed": multiscale_train.RANDOM_SEED,
        "model_path": payload["model_path"],
        "model_sha256": payload["model_sha256"],
        "artifact_path": payload["artifact_path"],
        "artifact_sha256": payload["artifact_sha256"],
        "prediction_path": prediction_path,
        "prediction_sha256": _sha256(prediction_path),
        "candidate_archive_path": archive_path,
        "candidate_archive_sha256": _sha256(archive_path),
        "horizon_metric_path": horizon_path,
        "candidate_metric_path": candidate_path,
        "regime_metric_path": regime_path,
        "regime_assignment_path": assignment_path,
        "frozen_g0_safety_path": safety_path,
        "frozen_g0_calibration_path": calibration_path,
        "frozen_g0_points_path": gate_path,
        "single_window_path": single_path,
        "single_window_figure_path": single_figure,
        "weighted_curve_path": weighted_path,
        "weighted_curve_figure_path": weighted_figure,
        "replay_single_window_path": replay_single_path,
        "replay_single_window_figure_path": replay_single_figure,
        "replay_weighted_curve_path": replay_weighted_path,
        "replay_weighted_curve_figure_path": replay_weighted_figure,
        "fusion_reconstruction_max_abs_error_scaled": payload[
            "fusion_reconstruction_max_abs_error_scaled"
        ],
        **{f"weighted_{key}": value for key, value in weighted_metrics.items()},
    }
    paths = {
        "prediction": prediction_path,
        "horizon": horizon_path,
        "candidate": candidate_path,
        "regime": regime_path,
        "assignment": assignment_path,
        "safety": safety_path,
        "calibration": calibration_path,
        "gate_points": gate_path,
        "archive": archive_path,
        "single_window": single_path,
        "single_figure": single_figure,
        "weighted_curve": weighted_path,
        "weighted_figure": weighted_figure,
        "replay_single_window": replay_single_path,
        "replay_single_figure": replay_single_figure,
        "replay_weighted_curve": replay_weighted_path,
        "replay_weighted_figure": replay_weighted_figure,
    }
    return {
        "summary": pd.DataFrame([summary]),
        "horizon": horizon,
        "candidate": candidates,
        "regime": regimes,
        "assignments": assignments,
        "safety": safety,
        "calibration": calibration,
        "gate_points": gate_points,
        "paths": paths,
        "payload": payload,
    }


def _read_archive(row, label):
    path = row.get("candidate_archive_path")
    if not isinstance(path, str) or not os.path.isfile(path):
        raise FileNotFoundError(f"{label}缺少candidate archive: {path}")
    if _sha256(path) != row.get("candidate_archive_sha256"):
        raise ValueError(f"{label} candidate archive hash漂移")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def validate_candidate_invariants(x0_summary, results):
    sources = {
        str(row["farm_id"]): _read_archive(row, f"X0/{row['farm_id']}")
        for _, row in x0_summary.iterrows()
    }
    rows = []
    for result in results:
        payload = result["payload"]
        variant, farm_id = payload["variant_id"], str(payload["farm_id"])
        source = sources[farm_id]
        for key in ("sample_id", "horizon_step", "forecast_origin_time"):
            if not np.array_equal(np.asarray(source[key]), np.asarray(payload[key])):
                raise ValueError(f"{variant}/{farm_id} {key}与X0不一致")
        source_y = np.asarray(source.get("y_true", source.get("y")), dtype=float)
        source_p = np.asarray(source.get("persistence", source.get("P")), dtype=float)
        source_c = np.asarray(source.get("corrected", source.get("C")), dtype=float)
        source_gate = np.asarray(
            source.get("applied_gate", source.get("raw_gate")), dtype=float
        )
        if not np.array_equal(source_y, payload["y_true"], equal_nan=True):
            raise ValueError(f"{variant}/{farm_id}测试真值窗口与X0不一致")
        capacity = float(payload["capacity"])
        p_drift = np.abs(source_p - payload["persistence"]) / capacity
        base_c_drift = np.abs(source_c - payload["base_corrected"]) / capacity
        gate_drift = np.abs(source_gate - payload["applied_gate"])
        expected_fused_scaled = payload["persistence_scaled"] + source_gate * (
            payload["corrected_scaled"] - payload["persistence_scaled"]
        )
        expected_fused = gate_predict._inverse_scaled(
            payload["artifact"], expected_fused_scaled, capacity
        )
        fused_drift = np.abs(expected_fused - payload["fused"]) / capacity
        artifact = payload["artifact"]
        x0_row = x0_summary.set_index("farm_id").loc[farm_id]
        row = {
            "model_variant": variant,
            "farm_id": farm_id,
            "x0_archive_path": x0_summary.set_index("farm_id").loc[
                farm_id, "candidate_archive_path"
            ],
            "x0_archive_sha256": x0_summary.set_index("farm_id").loc[
                farm_id, "candidate_archive_sha256"
            ],
            "sample_count": len(payload["sample_id"]),
            "truth_exact": True,
            "sample_keys_exact": True,
            "persistence_capacity_normalized_max_abs_drift": float(np.max(p_drift)),
            "persistence_capacity_normalized_mean_abs_drift": float(np.mean(p_drift)),
            "base_corrected_capacity_normalized_max_abs_drift": float(
                np.max(base_c_drift)
            ),
            "base_corrected_capacity_normalized_mean_abs_drift": float(
                np.mean(base_c_drift)
            ),
            "base_corrected_capacity_normalized_p999_abs_drift": float(
                np.quantile(base_c_drift, 0.999)
            ),
            "base_corrected_cross_runtime_max_tolerance": (
                BASE_CORRECTED_MAX_NORM_TOL
            ),
            "base_corrected_cross_runtime_mean_tolerance": (
                BASE_CORRECTED_MEAN_NORM_TOL
            ),
            "base_corrected_comparison_mode": (
                "same_f7_weight_hash_cross_tensorflow_runtime_capacity_tolerance"
            ),
            "frozen_g0_gate_max_abs_drift": float(np.max(gate_drift)),
            "frozen_g0_gate_mean_abs_drift": float(np.mean(gate_drift)),
            "frozen_g0_replay_capacity_normalized_max_abs_drift": float(
                np.max(fused_drift)
            ),
            "frozen_g0_replay_capacity_normalized_mean_abs_drift": float(
                np.mean(fused_drift)
            ),
            "source_f7_snapshot_before": artifact.get(
                "source_snapshot_before_sha256"
            ),
            "source_f7_snapshot_after": artifact.get(
                "source_snapshot_after_sha256"
            ),
            "source_f7_model_sha256": artifact.get("source_f7_model_sha256"),
            "x0_f7_model_sha256": x0_row.get("model_sha256"),
            "source_f7_artifact_sha256": artifact.get(
                "source_f7_artifact_sha256"
            ),
            "x0_f7_artifact_sha256": x0_row.get("artifact_sha256"),
            "persistence_probe_max_abs_drift": artifact.get(
                "persistence_probe_max_abs_drift"
            ),
            "g0_gate_probe_max_abs_drift": artifact.get(
                "g0_gate_probe_max_abs_drift"
            ),
        }
        row["persistence_control_pass"] = bool(
            row["persistence_capacity_normalized_max_abs_drift"]
            <= PERSISTENCE_MAX_NORM_TOL
            and row["persistence_capacity_normalized_mean_abs_drift"]
            <= PERSISTENCE_MEAN_NORM_TOL
        )
        row["frozen_g0_gate_control_pass"] = bool(
            row["frozen_g0_gate_max_abs_drift"] <= G0_GATE_MAX_ABS_TOL
            and row["frozen_g0_gate_mean_abs_drift"] <= G0_GATE_MEAN_ABS_TOL
        )
        row["base_corrected_control_pass"] = bool(
            row["base_corrected_capacity_normalized_max_abs_drift"]
            <= BASE_CORRECTED_MAX_NORM_TOL
            and row["base_corrected_capacity_normalized_mean_abs_drift"]
            <= BASE_CORRECTED_MEAN_NORM_TOL
        )
        row["frozen_g0_replay_control_pass"] = bool(
            row["frozen_g0_replay_capacity_normalized_max_abs_drift"]
            <= FUSED_REPLAY_MAX_NORM_TOL
            and row["frozen_g0_replay_capacity_normalized_mean_abs_drift"]
            <= FUSED_REPLAY_MEAN_NORM_TOL
        )
        row["source_f7_snapshot_control_pass"] = bool(
            not row["source_f7_snapshot_before"]
            or row["source_f7_snapshot_before"] == row["source_f7_snapshot_after"]
        )
        row["source_f7_identity_matches_x0"] = bool(
            row["source_f7_model_sha256"] == row["x0_f7_model_sha256"]
            and row["source_f7_artifact_sha256"]
            == row["x0_f7_artifact_sha256"]
        )
        row["source_g0_probe_control_pass"] = bool(
            float(row["persistence_probe_max_abs_drift"]) == 0.0
            and float(row["g0_gate_probe_max_abs_drift"]) == 0.0
        )
        row["all_invariants_pass"] = all(
            bool(row[key])
            for key in (
                "truth_exact",
                "sample_keys_exact",
                "persistence_control_pass",
                "base_corrected_control_pass",
                "frozen_g0_gate_control_pass",
                "frozen_g0_replay_control_pass",
                "source_f7_snapshot_control_pass",
                "source_f7_identity_matches_x0",
                "source_g0_probe_control_pass",
            )
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    expected_rows = len(NEW_VARIANTS) * len(_expected_farms())
    if len(frame) != expected_rows or not frame["all_invariants_pass"].all():
        raise ValueError("Stage-5A candidate/source不变量未全部通过")
    return frame, {
        "status": "pass",
        "rows": len(frame),
        "all_invariants_pass": True,
    }


def _exact_five(frame, label, required_columns=()):
    frame = frame.copy()
    if len(frame) != len(_expected_farms()):
        raise ValueError(f"{label}不是恰好5个场站: {len(frame)}")
    if set(frame["farm_id"].astype(str)) != set(_expected_farms()):
        raise ValueError(f"{label}场站集合不完整")
    for column in required_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"{label}/{column}包含非有限值")
    return frame


def _macro_regime(regime, variant, group, candidate="corrected"):
    part = regime[
        (regime["model_variant"].astype(str) == variant)
        & (regime["candidate"].astype(str) == candidate)
        & (regime["regime_group"].astype(str) == group)
        & (regime["horizon_step"].map(_canonical_horizon) == "all")
    ]
    part = _exact_five(part, f"{variant}/{candidate}/{group}", ("capacity_normalized_rmse",))
    return float(part["capacity_normalized_rmse"].mean())


def build_comparison(summary, regime):
    rows = []
    for variant in ALL_VARIANTS:
        frame = _exact_five(
            summary[summary["model_variant"].astype(str) == variant],
            variant,
            (
                "corrected_candidate_nrmse",
                "corrected_candidate_nmae",
                "frozen_g0_replay_nrmse",
                "parameter_count",
            ),
        )
        row = {
            "model_variant": variant,
            "variant_label": _variant_label(variant),
            "macro_corrected_candidate_test_nrmse": float(
                frame["corrected_candidate_nrmse"].mean()
            ),
            "macro_corrected_candidate_test_nmae": float(
                frame["corrected_candidate_nmae"].mean()
            ),
            "macro_frozen_g0_replay_test_nrmse": float(
                frame["frozen_g0_replay_nrmse"].mean()
            ),
            "parameter_count_max": int(frame["parameter_count"].max()),
            "multiscale_added_parameter_count_max": int(
                pd.to_numeric(frame["multiscale_added_parameter_count"]).max()
            ),
            "macro_inference_milliseconds_per_sample": float(
                pd.to_numeric(frame["inference_milliseconds_per_sample"]).mean()
            ),
        }
        for group in ("dynamic", "ramp_up", "ramp_down"):
            row[f"corrected_{group}_nrmse"] = _macro_regime(
                regime, variant, group, "corrected"
            )
            row[f"frozen_g0_replay_{group}_nrmse"] = _macro_regime(
                regime, variant, group, "frozen_g0_replay"
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    base = result[result["model_variant"] == "x0"].iloc[0]
    result["relative_candidate_nrmse_vs_x0"] = (
        result["macro_corrected_candidate_test_nrmse"]
        / float(base["macro_corrected_candidate_test_nrmse"])
        - 1.0
    )
    result["actual_candidate_improvement_vs_x0"] = -result[
        "relative_candidate_nrmse_vs_x0"
    ]
    result["selection_metric_scope"] = "corrected_candidate"
    farm_base = summary[summary["model_variant"] == "x0"].set_index("farm_id")[
        "corrected_candidate_nrmse"
    ].astype(float)
    guard_rows = []
    for _, row in result.iterrows():
        variant = row["model_variant"]
        target = summary[summary["model_variant"] == variant].set_index("farm_id")[
            "corrected_candidate_nrmse"
        ].astype(float).reindex(farm_base.index)
        nondegraded = int((target <= farm_base + FARM_NONDEGRADE_ATOL).sum())
        improved = int((target < farm_base - FARM_NONDEGRADE_ATOL).sum())
        benefit = farm_base - target
        best_benefit_farm = str(benefit.idxmax())
        retained = farm_base.index != best_benefit_farm
        base_without = float(farm_base[retained].mean())
        target_without = float(target[retained].mean())
        leave_one_improvement = 1.0 - target_without / base_without
        macro = bool(
            row["macro_corrected_candidate_test_nrmse"]
            <= float(base["macro_corrected_candidate_test_nrmse"])
            * (1.0 - REQUIRED_MACRO_IMPROVEMENT)
        )
        regime_guard = all(
            row[f"corrected_{group}_nrmse"]
            <= float(base[f"corrected_{group}_nrmse"])
            * (1.0 + REGIME_RELATIVE_DEGRADATION_TOL)
            for group in ("dynamic", "ramp_up", "ramp_down")
        )
        parameter = bool(row["parameter_count_max"] < PARAMETER_LIMIT)
        leave_one = bool(leave_one_improvement > 0.0)
        guard = bool(
            macro
            and nondegraded >= MIN_NONDEGRADED_FARMS
            and improved >= MIN_STRICTLY_IMPROVED_FARMS
            and regime_guard
            and leave_one
            and parameter
        )
        if variant == "x0":
            guard = True
        guard_rows.append(
            {
                "model_variant": variant,
                "macro_candidate_improves_at_least_0_3pct": macro,
                "farms_nondegraded_vs_x0": nondegraded,
                "at_least_4_farms_nondegraded": nondegraded
                >= MIN_NONDEGRADED_FARMS,
                "farms_strictly_improved_vs_x0": improved,
                "at_least_3_farms_strictly_improved": improved
                >= MIN_STRICTLY_IMPROVED_FARMS,
                "dynamic_ramp_guard_pass": bool(regime_guard),
                "best_benefit_farm_removed": best_benefit_farm,
                "leave_best_benefit_farm_out_improvement": leave_one_improvement,
                "leave_best_benefit_farm_out_positive": leave_one,
                "parameter_under_30k": parameter,
                "selection_guard_pass": guard,
            }
        )
    return result.merge(pd.DataFrame(guard_rows), on="model_variant", validate="one_to_one")


def select_model(comparison):
    comparison = comparison.copy()
    lowest = comparison.sort_values(
        "macro_corrected_candidate_test_nrmse", kind="stable"
    ).iloc[0]
    qualified = comparison[
        (comparison["model_variant"] != "x0")
        & comparison["selection_guard_pass"].astype(bool)
    ]
    if qualified.empty:
        selected = comparison[comparison["model_variant"] == "x0"].iloc[0]
        status = "fallback_x0_no_new_multiscale_candidate_passed_all_guards"
    else:
        best = float(qualified["macro_corrected_candidate_test_nrmse"].min())
        near = qualified[
            qualified["macro_corrected_candidate_test_nrmse"]
            <= best * (1.0 + NRMSE_TIE_TOL)
        ]
        selected = near.sort_values(
            [
                "parameter_count_max",
                "macro_inference_milliseconds_per_sample",
                "macro_corrected_candidate_test_nmae",
                "macro_corrected_candidate_test_nrmse",
            ],
            kind="stable",
        ).iloc[0]
        status = "qualified_candidate_nrmse_0_1pct_tie_then_complexity_latency_nmae"
    comparison["numerically_lowest_candidate_nrmse"] = (
        comparison["model_variant"] == lowest["model_variant"]
    )
    comparison["selected"] = comparison["model_variant"] == selected["model_variant"]
    comparison["selection_status"] = status
    return comparison[comparison["selected"]].iloc[0], comparison


def build_complexity(summary):
    frame = (
        summary.groupby("model_variant", as_index=False)
        .agg(
            parameter_count_max=("parameter_count", "max"),
            trainable_parameter_count_max=("trainable_parameter_count", "max"),
            multiscale_added_parameter_count_max=(
                "multiscale_added_parameter_count",
                "max",
            ),
            training_elapsed_seconds_macro=("training_elapsed_seconds", "mean"),
            inference_ms_per_sample_macro=(
                "inference_milliseconds_per_sample",
                "mean",
            ),
        )
        .copy()
    )
    x0_params = int(
        frame.loc[frame["model_variant"] == "x0", "parameter_count_max"].iloc[0]
    )
    frame["parameter_delta_vs_x0"] = frame["parameter_count_max"] - x0_params
    frame["parameter_under_30k"] = frame["parameter_count_max"] < PARAMETER_LIMIT
    frame["random_seed"] = multiscale_train.RANDOM_SEED
    frame["seed_count"] = 1
    frame["stability_scope"] = "single_seed_2026_no_multiseed_claim"
    return frame


def validate_complete_matrix(frames):
    expected_variants, expected_farms = set(ALL_VARIANTS), set(_expected_farms())
    natural_keys = {
        "summary": ("model_variant", "farm_id"),
        "horizon": ("model_variant", "farm_id", "horizon_step"),
        "candidate": ("model_variant", "farm_id", "candidate", "horizon_step"),
        "regime": (
            "model_variant",
            "farm_id",
            "regime_group",
            "candidate",
            "horizon_step",
        ),
        "assignments": ("model_variant", "farm_id", "sample_id"),
    }
    if set(frames) != set(natural_keys):
        raise ValueError("Stage-5A正式表集不完整")
    for name, frame in frames.items():
        keys = natural_keys[name]
        if set(keys) - set(frame.columns):
            raise KeyError(f"{name}缺少自然键")
        if set(frame["model_variant"].astype(str)) != expected_variants:
            raise ValueError(f"{name}未覆盖五个变体")
        if set(frame["farm_id"].astype(str)) != expected_farms:
            raise ValueError(f"{name}未覆盖固定5场站")
        normalized = frame.copy()
        for key in keys:
            normalized[key] = normalized[key].map(
                lambda value: "<NA>" if pd.isna(value) else str(value)
            )
        if normalized.duplicated(list(keys)).any():
            raise ValueError(f"{name}自然键重复")
        suffix = keys[2:]
        for farm_id in expected_farms:
            baseline = normalized[
                (normalized["model_variant"] == "x0")
                & (normalized["farm_id"] == farm_id)
            ]
            for variant in ALL_VARIANTS:
                target = normalized[
                    (normalized["model_variant"] == variant)
                    & (normalized["farm_id"] == farm_id)
                ]
                if len(target) != len(baseline):
                    raise ValueError(f"{name}/{variant}/{farm_id}行数与X0不同")
                if suffix and set(target[list(suffix)].itertuples(index=False, name=None)) != set(
                    baseline[list(suffix)].itertuples(index=False, name=None)
                ):
                    raise ValueError(f"{name}/{variant}/{farm_id}自然键集与X0不同")
    if len(frames["summary"]) != len(ALL_VARIANTS) * len(expected_farms):
        raise ValueError("summary不是5变体x5场站")


def save_aggregate_figures(comparison, summary, horizon, regime, output_dir):
    figure_dir = os.path.join(output_dir, "figures")
    cache_dir = os.path.join(output_dir, "matplotlib_cache")
    os.makedirs(figure_dir, exist_ok=True)
    plt = common_predict.setup_matplotlib({"matplotlib_cache": cache_dir})
    paths = {}

    ordered = comparison.sort_values("macro_corrected_candidate_test_nrmse")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = ["#d62728" if flag else "#4c78a8" for flag in ordered["selected"]]
    ax.bar(ordered["model_variant"].map(_variant_label), ordered["macro_corrected_candidate_test_nrmse"], color=colors)
    ax.set_ylabel("Five-farm macro candidate NRMSE")
    ax.set_title("Stage-5A guarded corrected-candidate ranking")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths["candidate_rank_figure"] = os.path.join(
        figure_dir, "multiscale_candidate_test_nrmse_rank.png"
    )
    fig.savefig(paths["candidate_rank_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    matrix = summary.pivot(
        index="model_variant", columns="farm_id", values="corrected_candidate_nrmse"
    ).reindex(index=ALL_VARIANTS, columns=_expected_farms())
    if matrix.isna().any().any():
        raise ValueError("场站candidate NRMSE热力图矩阵不完整")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    image = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(matrix)), labels=[_variant_label(v) for v in matrix.index])
    ax.set_xticks(range(len(matrix.columns)), labels=[str(v)[-4:] for v in matrix.columns])
    ax.set_xlabel("Farm ID (last 4 digits)")
    ax.set_title("Corrected candidate NRMSE by farm")
    fig.colorbar(image, ax=ax, label="NRMSE")
    fig.tight_layout()
    paths["farm_heatmap_figure"] = os.path.join(
        figure_dir, "multiscale_candidate_test_farm_heatmap.png"
    )
    fig.savefig(paths["farm_heatmap_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    numeric = horizon[horizon["horizon_step"].map(_canonical_horizon) != "all"].copy()
    numeric["h"] = pd.to_numeric(numeric["horizon_step"], errors="raise")
    macro = numeric.groupby(["model_variant", "h"], as_index=False)[
        "capacity_normalized_rmse"
    ].mean()
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for variant in ALL_VARIANTS:
        part = macro[macro["model_variant"] == variant].sort_values("h")
        if len(part) != multiscale_train.FORECAST_LEN:
            raise ValueError(f"{variant}逐horizon指标不完整")
        ax.plot(part["h"], part["capacity_normalized_rmse"], marker="o", ms=3, label=_variant_label(variant))
    ax.set(xlabel="Forecast horizon (15-min steps)", ylabel="Five-farm macro candidate NRMSE")
    ax.set_title("Corrected-candidate horizon-wise test error")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    paths["horizon_figure"] = os.path.join(
        figure_dir, "multiscale_candidate_test_horizon_nrmse.png"
    )
    fig.savefig(paths["horizon_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    compare = comparison.set_index("model_variant").reindex(ALL_VARIANTS)
    x = np.arange(len(compare))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.3, 5.0))
    ax.bar(x - width / 2, compare["macro_corrected_candidate_test_nrmse"], width, label="corrected candidate")
    ax.bar(x + width / 2, compare["macro_frozen_g0_replay_test_nrmse"], width, label="frozen G0 replay")
    ax.set_xticks(x, labels=[_variant_label(v) for v in compare.index])
    ax.set_ylabel("Five-farm macro NRMSE")
    ax.set_title("Candidate quality and frozen-G0 conversion diagnostic")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    paths["candidate_replay_figure"] = os.path.join(
        figure_dir, "multiscale_candidate_vs_frozen_g0_replay.png"
    )
    fig.savefig(paths["candidate_replay_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    groups = ("dynamic", "ramp_up", "ramp_down")
    x = np.arange(len(groups))
    width = 0.16
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for index, variant in enumerate(ALL_VARIANTS):
        values = [
            _macro_regime(regime, variant, group, "corrected") for group in groups
        ]
        ax.bar(x + (index - 2) * width, values, width, label=_variant_label(variant))
    ax.set_xticks(x, labels=groups)
    ax.set_ylabel("Five-farm macro candidate NRMSE")
    ax.set_title("Dynamic and ramp candidate performance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    paths["regime_figure"] = os.path.join(
        figure_dir, "multiscale_candidate_dynamic_ramp.png"
    )
    fig.savefig(paths["regime_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for _, row in comparison.iterrows():
        ax.scatter(row["parameter_count_max"], row["macro_corrected_candidate_test_nrmse"], s=70)
        ax.annotate(_variant_label(row["model_variant"]), (row["parameter_count_max"], row["macro_corrected_candidate_test_nrmse"]), xytext=(4, 4), textcoords="offset points")
    ax.axvline(PARAMETER_LIMIT, color="red", linestyle="--", label="30k limit")
    ax.set(xlabel="Parameter count", ylabel="Five-farm macro candidate NRMSE")
    ax.set_title("Accuracy-complexity trade-off")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    paths["complexity_figure"] = os.path.join(
        figure_dir, "multiscale_candidate_accuracy_complexity.png"
    )
    fig.savefig(paths["complexity_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def write_report(comparison, selected, invariants, output_dir):
    compact = [
        "model_variant",
        "variant_label",
        "macro_corrected_candidate_test_nrmse",
        "macro_corrected_candidate_test_nmae",
        "actual_candidate_improvement_vs_x0",
        "farms_nondegraded_vs_x0",
        "farms_strictly_improved_vs_x0",
        "corrected_dynamic_nrmse",
        "corrected_ramp_up_nrmse",
        "corrected_ramp_down_nrmse",
        "leave_best_benefit_farm_out_improvement",
        "parameter_count_max",
        "selection_guard_pass",
        "numerically_lowest_candidate_nrmse",
        "selected",
    ]
    selected_label = _variant_label(str(selected["model_variant"]))
    text = [
        "# Stage-5A轻量多尺度 corrected candidate：测试集最终选型",
        "",
        f"最终选中 **{selected_label}**；5场站等权宏平均 corrected-candidate "
        f"NRMSE=`{selected['macro_corrected_candidate_test_nrmse']:.9f}`。",
        "",
        "本轮按用户指定在当前测试集选型，标记为 `legacy_seen_test_selected`；"
        "不是独立最终盲测。",
        "",
        "## 五变体正式矩阵",
        "",
        comparison[compact].to_markdown(index=False),
        "",
        "## 来源与不变量",
        "",
        invariants.to_markdown(index=False),
        "",
        "## 预声明守门协议",
        "",
        "- X0只读、hash校验引用Stage-4B D0/F7；不重训练、不forward、不复制产物。",
        "- 正式主指标是corrected candidate，而不是冻结G0回放fused。",
        "- 新变体macro candidate NRMSE须至少改善0.3%。",
        "- 至少4/5场站不退化、至少3/5场站严格改善。",
        "- corrected candidate在dynamic、ramp-up、ramp-down的宏NRMSE均不得相对X0恶化超过0.5%。",
        "- 删除收益最大的单场站后，余下4场站宏改善仍须严格为正。",
        "- 参数量必须小于30k；没有新变体通过全部条件时回退X0。",
        "- 冻结G0回放只诊断candidate收益能否被既有门控转化，不用于本轮晋级。",
        "- 通过全部守门的模型若处于最优candidate NRMSE的0.1%带内，再按参数量、推理耗时、NMAE和NRMSE排序。",
        "",
    ]
    return _atomic_text(
        "\n".join(text),
        os.path.join(output_dir, "multiscale_correc_cand_test_final_selection.md"),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default=os.getenv("WIND_MULTISCALE_PREDICT_VARIANTS", ",".join(ALL_VARIANTS)),
        help="逗号分隔: x0,x1_f,x1_m,x1_c,x1（也接受x1-f等写法）",
    )
    parser.add_argument(
        "--farms",
        default=os.getenv("WIND_MULTISCALE_FARMS", ""),
        help="逗号分隔场站ID；空值为全部",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("PYTHONHASHSEED", str(multiscale_train.RANDOM_SEED))
    keras.utils.set_random_seed(multiscale_train.RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    variants = _parse_list(args.variants, ALL_VARIANTS, "variants", variant=True)
    expected_farms = _expected_farms()
    farms = (
        _parse_list(args.farms, expected_farms, "farms")
        if args.farms
        else expected_farms
    )
    if args.smoke:
        if set(variants) == set(ALL_VARIANTS):
            variants = ["x1_f"]
        if not args.farms:
            farms = expected_farms[:1]
        args.max_samples = args.max_samples or 32
    if args.max_samples is not None and not args.smoke:
        raise ValueError("--max-samples只允许与--smoke同时使用")
    full = bool(
        set(variants) == set(ALL_VARIANTS)
        and set(farms) == set(expected_farms)
        and not args.smoke
        and args.max_samples is None
        and not args.skip_plots
    )
    output_root = (
        multiscale_train.RESULT_ROOT
        if full
        else os.path.join(
            multiscale_train.RESULT_ROOT,
            "partial_runs",
            args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    )
    output_dir = os.path.join(output_root, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)
    formal_marker = os.path.join(
        multiscale_train.RESULT_ROOT, OUTPUT_SUBDIR, FORMAL_MARKER_NAME
    )
    prediction_running_marker = os.path.join(output_dir, RUNNING_MARKER_NAME)
    if full:
        # Keep the previous complete marker until the newly validated bundle is
        # atomically published.  A surviving running marker makes any
        # interrupted mixed bundle visibly non-consumable.
        _atomic_json(
            {
                "status": "running",
                "protocol_version": multiscale_train.PROTOCOL_VERSION,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "variants": variants,
                "farm_ids": farms,
            },
            prediction_running_marker,
        )

    stage4b_marker, x0_frames, source_paths, source_summary = (
        validate_stage4b_x0_source()
    )
    training_marker_path, training_marker = validate_training_bundle(
        [variant for variant in variants if variant in NEW_VARIANTS]
    )
    source_test = {
        str(farm_id): record["path"]
        for farm_id, record in stage4b_marker["test_files"].items()
    }
    results = []
    for farm_id in farms:
        for variant in variants:
            if variant == "x0":
                continue
            print(f"\n===== Stage-5A预测 variant={variant} farm={farm_id} =====")
            payload = predict_variant(
                variant, source_test[farm_id], training_marker, args.max_samples
            )
            results.append(save_payload(payload, output_root, args.skip_plots))

    if not full:
        pieces = []
        if "x0" in variants:
            pieces.append(x0_frames["summary"][x0_frames["summary"]["farm_id"].isin(farms)])
        pieces.extend(result["summary"] for result in results)
        partial = pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()
        path = _atomic_csv(
            partial,
            os.path.join(output_dir, "multiscale_correc_cand_partial_summary.csv"),
        )
        _atomic_json(
            {
                "status": "partial_not_formal",
                "variants": variants,
                "farms": farms,
                "max_samples": args.max_samples,
                "skip_plots": args.skip_plots,
                "summary": _file_record(path),
            },
            os.path.join(output_dir, "partial_run_manifest.json"),
        )
        print(f"partial/smoke结果（不参与正式选型）: {path}")
        return

    if len(results) != len(NEW_VARIANTS) * len(expected_farms):
        raise ValueError("Stage-5A正式预测必须是4个新变体x5场站")
    frames = {
        key: pd.concat(
            [x0_frames[key]] + [result[key] for result in results],
            ignore_index=True,
            sort=False,
        )
        for key in ("summary", "horizon", "candidate", "regime", "assignments")
    }
    validate_complete_matrix(frames)
    invariants, invariant_status = validate_candidate_invariants(
        x0_frames["summary"], results
    )
    comparison = build_comparison(frames["summary"], frames["regime"])
    selected, comparison = select_model(comparison)
    complexity = build_complexity(frames["summary"])

    paths = {}
    for key, frame in frames.items():
        paths[key] = _atomic_csv(
            frame,
            os.path.join(output_dir, f"multiscale_correc_cand_test_{key}.csv"),
        )
    paths["comparison"] = _atomic_csv(
        comparison,
        os.path.join(output_dir, "multiscale_correc_cand_test_variant_comparison.csv"),
    )
    paths["final_selection"] = _atomic_csv(
        comparison[comparison["selected"]],
        os.path.join(output_dir, "multiscale_correc_cand_test_final_selection.csv"),
    )
    paths["candidate_invariants"] = _atomic_csv(
        invariants,
        os.path.join(output_dir, "multiscale_correc_cand_candidate_invariants.csv"),
    )
    paths["complexity"] = _atomic_csv(
        complexity,
        os.path.join(output_dir, "multiscale_correc_cand_test_complexity.csv"),
    )
    paths.update(
        save_aggregate_figures(
            comparison,
            frames["summary"],
            frames["horizon"],
            frames["regime"],
            output_dir,
        )
    )

    source_rows = [
        {
            "source": "Stage-4B complete marker",
            "key": "marker",
            **_file_record(STAGE4B_MARKER),
            "reuse_action": "hash_validated_read_only_dependency",
        }
    ]
    for key, path in source_paths.items():
        source_rows.append(
            {
                "source": "Stage-4B formal aggregate",
                "key": key,
                **_file_record(path),
                "reuse_action": "filter_d0_relabel_x0_no_inference_no_copy",
            }
        )
    for _, row in source_summary.iterrows():
        for key in (
            "model_path",
            "artifact_path",
            "prediction_path",
            "candidate_archive_path",
            "single_window_figure_path",
            "weighted_curve_figure_path",
        ):
            path = row.get(key)
            if isinstance(path, str) and os.path.isfile(path):
                source_rows.append(
                    {
                        "source": f"Stage-4B D0/F7 farm {row['farm_id']}",
                        "key": key,
                        **_file_record(path),
                        "reuse_action": "direct_path_reference_no_copy_no_forward",
                    }
                )
    paths["source_manifest"] = _atomic_csv(
        pd.DataFrame(source_rows),
        os.path.join(output_dir, "multiscale_correc_cand_source_reuse_manifest.csv"),
    )
    paths["report"] = write_report(comparison, selected, invariants, output_dir)

    visual_candidates = []
    for key, path in paths.items():
        if isinstance(path, str) and path.lower().endswith(".png"):
            visual_candidates.append((f"aggregate.{key}", path))
    for index, result in enumerate(results):
        for key, path in result["paths"].items():
            if isinstance(path, str) and path.lower().endswith(".png"):
                visual_candidates.append((f"result{index}.{key}", path))
    for _, row in source_summary.iterrows():
        for key in ("single_window_figure_path", "weighted_curve_figure_path"):
            path = row.get(key)
            if isinstance(path, str) and path.lower().endswith(".png"):
                visual_candidates.append((f"source_x0.{row['farm_id']}.{key}", path))
    visual_rows, seen = [], set()
    for key, path in visual_candidates:
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        visual_rows.append({"key": key, **_file_record(path)})
    if not visual_rows:
        raise ValueError("正式Stage-5A bundle没有任何可视化图片")
    paths["visual_inventory"] = _atomic_csv(
        pd.DataFrame(visual_rows),
        os.path.join(output_dir, "multiscale_correc_cand_visual_inventory.csv"),
    )

    files = {
        "prediction_code": _file_record(__file__),
        "training_code": _file_record(multiscale_train.__file__),
        "dependency.stage4_prediction_helpers": _file_record(stage4_predict.__file__),
        "dependency.controlled_gate_prediction_helpers": _file_record(gate_predict.__file__),
        "dependency.common_prediction_helpers": _file_record(common_predict.__file__),
        "stage4b_source_marker": _file_record(STAGE4B_MARKER),
        "training_marker": _file_record(training_marker_path),
    }
    files.update({f"formal.{key}": _file_record(path) for key, path in paths.items()})
    for index, result in enumerate(results):
        for key, path in result["paths"].items():
            if path:
                files[f"result{index}.{key}"] = _file_record(path)
    marker = {
        "status": "complete",
        "protocol_version": multiscale_train.PROTOCOL_VERSION,
        "architecture_version": getattr(multiscale_train, "ARCHITECTURE_VERSION", None),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": multiscale_train.RANDOM_SEED,
        "selection_split": "test",
        "test_used_for_selection": True,
        "test_reuse_status": TEST_REUSE_STATUS,
        "test_is_final_blind_evaluation": False,
        "variants": list(ALL_VARIANTS),
        "expected_farm_ids": expected_farms,
        "x0_policy": "direct_stage4b_d0_f7_reference_no_training_no_forward_no_copy",
        "primary_selection_target": "corrected_candidate",
        "selection_metric_scope": "corrected_candidate",
        "frozen_g0_replay_policy": "diagnostic_only_not_stage5a_selection_target",
        "new_prediction_count": len(results),
        "visualization_count": len(visual_rows),
        "candidate_invariants": invariant_status,
        "selected_variant": str(selected["model_variant"]),
        "test_files": {
            farm_id: _file_record(source_test[farm_id]) for farm_id in expected_farms
        },
        "files": files,
    }
    marker_path = _atomic_json(marker, formal_marker)
    if os.path.exists(prediction_running_marker):
        os.remove(prediction_running_marker)
    print(
        f"\nStage-5A测试集最终选择: {selected['model_variant']} / corrected candidate "
        f"macro NRMSE={selected['macro_corrected_candidate_test_nrmse']:.9f}"
    )
    print(f"正式报告: {paths['report']}")
    print(f"正式bundle marker: {marker_path}")


if __name__ == "__main__":
    main()
