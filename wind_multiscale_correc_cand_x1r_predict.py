"""Stage 5A-X1R 门控收益转化闭环：测试集预测、审计与最终选型。

部署闭环正式对照为：

* X0：只读引用 Stage-4B D0/F7+G0 fused；
* X1-fixed：只读引用 Stage-5A 完整 X1 candidate + 原 G0 回放，仅作
  同 candidate 的受控诊断，不参与部署选型；
* X1R：唯一新前向，冻结 X1 candidate + 重新校准的 calibrated-safe gate。

只有 X0 与 X1R 具有 selection eligibility。Stage-5A 的 X1-F/X1-C/X1
corrected 指标另存为 candidate evidence，绝不与 fused 部署指标混排。
正式运行覆盖固定五场站并发布 complete marker；smoke/子集/skip-plots 均写入
``partial_runs``，不会覆盖正式测试bundle。
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import wind_controlled_gate_cali_predict as gate_predict
import wind_dl_model_predict as common_predict
import wind_multiscale_correc_cand_predict as multiscale_predict
import wind_multiscale_correc_cand_train as multiscale_train
import wind_multiscale_correc_cand_x1r_train as x1r_train
import wind_time_freq_model_predict as stage4_predict
import wind_time_freq_model_stage4b_predict as stage4b_predict
import wind_time_freq_model_stage4b_train as stage4b_train


OUTPUT_SUBDIR = "testdata_predict_output"
TEST_REUSE_STATUS = "legacy_seen_test_selected"
FORMAL_MARKER_NAME = "x1r_gate_closure_test_bundle_complete.json"
RUNNING_MARKER_NAME = "x1r_gate_closure_test_bundle_running.json"
CLOSURE_VARIANTS = ("x0", "x1_fixed_g0", "x1r")
NEW_VARIANTS = ("x1r",)

STAGE5A_ROOT = multiscale_train.RESULT_ROOT
STAGE5A_OUTPUT = os.path.join(STAGE5A_ROOT, OUTPUT_SUBDIR)
STAGE5A_MARKER = os.path.join(
    STAGE5A_OUTPUT, multiscale_predict.FORMAL_MARKER_NAME
)
STAGE4B_ROOT = stage4b_train.RESULT_ROOT
STAGE4B_OUTPUT = os.path.join(STAGE4B_ROOT, OUTPUT_SUBDIR)
STAGE4B_MARKER = os.path.join(STAGE4B_OUTPUT, stage4b_predict.FORMAL_MARKER_NAME)

STAGE5A_FILES = {
    "summary": "multiscale_correc_cand_test_summary.csv",
    "horizon": "multiscale_correc_cand_test_horizon.csv",
    "candidate": "multiscale_correc_cand_test_candidate.csv",
    "regime": "multiscale_correc_cand_test_regime.csv",
    "assignments": "multiscale_correc_cand_test_assignments.csv",
    "comparison": "multiscale_correc_cand_test_variant_comparison.csv",
}
STAGE4B_FILES = {
    "summary": "stage4b_gate_closure_test_summary.csv",
    "horizon": "stage4b_gate_closure_test_horizon.csv",
    "candidate": "stage4b_gate_closure_test_candidate.csv",
    "regime": "stage4b_gate_closure_test_regime.csv",
    "assignments": "stage4b_gate_closure_test_assignments.csv",
    "safety": "stage4b_gate_closure_test_safety.csv",
    "calibration": "stage4b_gate_closure_test_calibration.csv",
}

# 预声明X1R部署晋级门槛。
REQUIRED_MACRO_IMPROVEMENT = 0.002
FARM_NONDEGRADE_ATOL = 1e-12
MIN_NONDEGRADED_FARMS = 4
MIN_STRICTLY_IMPROVED_FARMS = 3
REGIME_DEGRADATION_TOL = 0.005
SAFETY_REGRET_RELATIVE_TOL = 0.005
SAFETY_HARM_ABS_TOL = 0.002
BRIER_RELATIVE_IMPROVEMENT = 0.10
ECE_RELATIVE_IMPROVEMENT = 0.15
HIGH_SATURATION_MAX = 0.50
PARAMETER_LIMIT = 30_000
PERSISTENCE_MAX_NORM_TOL = 1e-6
# Stage-5A已经在同一5场站上量化了旧archive与当前TensorFlow数值内核的
# 跨运行时差异；来源model/artifact/marker仍严格hash锁定，输出只在该显式
# 容量归一化边界内比较，避免把浮点内核差异误判为candidate权重漂移。
CANDIDATE_MAX_NORM_TOL = multiscale_predict.BASE_CORRECTED_MAX_NORM_TOL
CANDIDATE_MEAN_NORM_TOL = multiscale_predict.BASE_CORRECTED_MEAN_NORM_TOL

_sha256 = stage4_predict._sha256
_file_record = stage4_predict._file_record
_atomic_csv = stage4_predict._atomic_csv
_atomic_json = stage4_predict._atomic_json
_atomic_text = stage4_predict._atomic_text
_validate_record = stage4_predict._validate_record

def _read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _expected_farms():
    return [str(value) for value in x1r_train.expected_farm_ids()]


def _canonical_horizon(value):
    text = str(value).strip().lower()
    return "all" if text == "all" else str(int(float(text)))


def _relabel(frame, variant, source_family=None, source_variant=None):
    frame = frame.copy()
    if source_family is None and "model_family" in frame:
        source_family = frame["model_family"].astype(str)
    if source_variant is None and "model_variant" in frame:
        source_variant = frame["model_variant"].astype(str)
    frame["source_model_family"] = source_family
    frame["source_model_variant"] = source_variant
    frame["model_family"] = x1r_train.MODEL_FAMILY
    frame["model_variant"] = variant
    if "model_name" in frame:
        frame["model_name"] = (
            x1r_train.variant_model_name("x1r")
            if variant == "x1r"
            else f"{x1r_train.MODEL_FAMILY}_{variant}"
        )
    return frame


def _load_complete_marker(path, label, protocol, architecture=None):
    running = path.replace("_bundle_complete.json", "_bundle_running.json")
    if os.path.isfile(running):
        raise RuntimeError(f"{label}存在running marker，拒绝混合bundle: {running}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少{label} complete marker: {path}")
    marker = _read_json(path)
    if marker.get("status") != "complete":
        raise ValueError(f"{label} marker不是complete")
    if marker.get("protocol_version") != protocol:
        raise ValueError(f"{label} marker协议不兼容")
    if architecture and marker.get("architecture_version") != architecture:
        raise ValueError(f"{label} marker架构不兼容")
    if set(map(str, marker.get("expected_farm_ids", ()))) != set(_expected_farms()):
        raise ValueError(f"{label} marker未锁定固定5场站")
    for key, record in marker.get("files", {}).items():
        _validate_record(f"{label} files.{key}", record)
    for farm_id, record in marker.get("test_files", {}).items():
        _validate_record(f"{label} test_files.{farm_id}", record)
    return marker


def _marker_table(marker, key, root, filename, label):
    path = os.path.join(root, filename)
    record = marker.get("files", {}).get(f"formal.{key}")
    validated = _validate_record(f"{label} formal.{key}", record)
    if os.path.realpath(validated) != os.path.realpath(path):
        raise ValueError(f"{label} formal.{key}路径漂移")
    return pd.read_csv(path, dtype={"farm_id": str}), path


def validate_training_bundle():
    running = os.path.join(x1r_train.RESULT_ROOT, x1r_train.RUNNING_MARKER_NAME)
    if os.path.isfile(running):
        raise RuntimeError(f"X1R训练仍在运行或未完整收尾: {running}")
    path = os.path.join(x1r_train.RESULT_ROOT, x1r_train.TRAINING_MARKER_NAME)
    marker = _load_complete_marker(
        path,
        "X1R training",
        x1r_train.PROTOCOL_VERSION,
        x1r_train.ARCHITECTURE_VERSION,
    )
    if (
        marker.get("new_model_count") != len(_expected_farms())
        or not bool(marker.get("source_x1_candidate_retraining_forbidden"))
        or not bool(marker.get("source_x1_candidate_frozen_verified"))
        or not bool(marker.get("old_g0_gate_pruned"))
        or not bool(marker.get("train_only_soft_oracle_q90"))
        or bool(marker.get("test_prediction_read_during_training"))
    ):
        raise ValueError("X1R训练marker未锁定冻结candidate/train-only协议")
    files = marker.get("files", {})
    for farm_id in _expected_farms():
        for kind in ("model_path", "artifact_path"):
            if f"x1r.{farm_id}.{kind}" not in files:
                raise KeyError(f"X1R训练marker缺少x1r.{farm_id}.{kind}")
    return path, marker


def validate_source_bundles(training_marker):
    stage5a = _load_complete_marker(
        STAGE5A_MARKER,
        "Stage-5A prediction",
        multiscale_train.PROTOCOL_VERSION,
        multiscale_train.ARCHITECTURE_VERSION,
    )
    stage4b = _load_complete_marker(
        STAGE4B_MARKER,
        "Stage-4B prediction",
        stage4b_train.PROTOCOL_VERSION,
        stage4b_train.ARCHITECTURE_VERSION,
    )
    if set(map(str, stage5a.get("test_files", {}))) != set(_expected_farms()):
        raise ValueError("Stage-5A预测marker未锁定5个测试文件")
    source_record = training_marker["files"].get("source_stage5a_training_marker")
    stage5a_record = stage5a["files"].get("training_marker")
    source_path = _validate_record("X1R source Stage-5A training", source_record)
    stage5a_path = _validate_record("Stage-5A files.training_marker", stage5a_record)
    if (
        os.path.realpath(source_path) != os.path.realpath(stage5a_path)
        or source_record.get("sha256") != stage5a_record.get("sha256")
    ):
        raise ValueError("X1R训练candidate与Stage-5A测试bundle不是同一训练来源")
    for farm_id in _expected_farms():
        stage5a_test = stage5a.get("test_files", {}).get(farm_id)
        stage4b_test = stage4b.get("test_files", {}).get(farm_id)
        stage5a_test_path = _validate_record(
            f"Stage-5A test_files.{farm_id}", stage5a_test
        )
        stage4b_test_path = _validate_record(
            f"Stage-4B test_files.{farm_id}", stage4b_test
        )
        if (
            os.path.realpath(stage5a_test_path) != os.path.realpath(stage4b_test_path)
            or stage5a_test.get("sha256") != stage4b_test.get("sha256")
        ):
            raise ValueError(f"Stage-4B/Stage-5A测试文件身份不一致: {farm_id}")
    stage5a_training_marker = _read_json(stage5a_path)
    multiscale_predict.validate_shared_stage4b_training_identity(
        stage4b, stage5a_training_marker
    )
    return stage5a, stage4b


def _stage4b_x0_frames(marker):
    raw, paths = {}, {}
    for key, filename in STAGE4B_FILES.items():
        raw[key], paths[key] = _marker_table(
            marker, key, STAGE4B_OUTPUT, filename, "Stage-4B"
        )
    frames = {}
    for key, frame in raw.items():
        selected = frame[frame["model_variant"].astype(str) == "d0"].copy()
        if set(selected["farm_id"].astype(str)) != set(_expected_farms()):
            raise ValueError(f"Stage-4B D0 {key}未覆盖5场站")
        frames[key] = _relabel(
            selected, "x0", "time_freq_stage4b_gate_closure", "d0"
        )
    frames["summary"]["variant_label"] = "X0 D0/F7 + frozen G0 deployment reference"
    frames["summary"]["reference_only"] = True
    frames["summary"]["selection_eligible"] = True
    frames["summary"]["trainable_parameter_count"] = 0
    frames["summary"]["result_source"] = (
        "hash_validated_stage4b_d0_direct_reference_no_forward_no_copy"
    )
    frames["summary"]["test_reuse_status"] = TEST_REUSE_STATUS
    return frames, paths


def _stage5a_x1_fixed_frames(marker):
    raw, paths = {}, {}
    for key, filename in STAGE5A_FILES.items():
        raw[key], paths[key] = _marker_table(
            marker, key, STAGE5A_OUTPUT, filename, "Stage-5A"
        )
    meta = raw["summary"][raw["summary"]["model_variant"].astype(str) == "x1"].copy()
    if len(meta) != len(_expected_farms()):
        raise ValueError("Stage-5A X1 summary不是5场站")
    candidate = raw["candidate"][
        raw["candidate"]["model_variant"].astype(str) == "x1"
    ].copy()
    candidate["candidate"] = candidate["candidate"].replace(
        {"frozen_g0_replay": "fused"}
    )
    candidate = _relabel(candidate, "x1_fixed_g0", "multiscale_correc_cand", "x1")
    horizon = candidate[candidate["candidate"].astype(str) == "fused"].copy()
    horizon["formal_metric_source"] = "stage5a_x1_fixed_g0_replay_direct_reference"
    regime = raw["regime"][raw["regime"]["model_variant"].astype(str) == "x1"].copy()
    regime["candidate"] = regime["candidate"].replace({"frozen_g0_replay": "fused"})
    regime = _relabel(regime, "x1_fixed_g0", "multiscale_correc_cand", "x1")
    assignments = raw["assignments"][
        raw["assignments"]["model_variant"].astype(str) == "x1"
    ].copy()
    assignments = _relabel(
        assignments, "x1_fixed_g0", "multiscale_correc_cand", "x1"
    )

    locked_records = {
        os.path.realpath(str(record.get("path", ""))): record
        for record in marker.get("files", {}).values()
        if isinstance(record, dict) and record.get("path")
    }
    safety_parts, calibration_parts = [], []
    for _, row in meta.iterrows():
        for field, target in (
            ("frozen_g0_safety_path", safety_parts),
            ("frozen_g0_calibration_path", calibration_parts),
        ):
            path = str(row[field])
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Stage-5A X1缺少{field}: {path}")
            locked = locked_records.get(os.path.realpath(path))
            if locked is None or _sha256(path) != locked.get("sha256"):
                raise ValueError(f"Stage-5A X1 {field}未被complete marker锁定")
            target.append(pd.read_csv(path, dtype={"farm_id": str}))
    safety = _relabel(
        pd.concat(safety_parts, ignore_index=True),
        "x1_fixed_g0",
        "multiscale_correc_cand",
        "x1",
    )
    calibration = _relabel(
        pd.concat(calibration_parts, ignore_index=True),
        "x1_fixed_g0",
        "multiscale_correc_cand",
        "x1",
    )
    overall = horizon[horizon["horizon_step"].map(_canonical_horizon) == "all"].copy()
    utility = safety[
        (safety["scope_type"].astype(str) == "overall")
        & (safety["scope_value"].astype(str) == "all")
    ].copy()
    if len(overall) != len(_expected_farms()) or len(utility) != len(_expected_farms()):
        raise ValueError("Stage-5A X1 fixed-G0 overall/safety不是5场站")
    utility_cols = [
        col
        for col in utility.columns
        if col
        not in {
            "model_family",
            "model_variant",
            "farm_id",
            "scope_type",
            "scope_value",
            "source_model_family",
            "source_model_variant",
        }
    ]
    summary = overall.merge(
        utility[["farm_id"] + utility_cols], on="farm_id", validate="one_to_one"
    )
    meta = meta.set_index("farm_id")
    summary["variant_label"] = "X1 frozen candidate + original G0 replay (diagnostic)"
    summary["parameter_count"] = summary["farm_id"].map(meta["parameter_count"])
    summary["trainable_parameter_count"] = 0
    for field in (
        "inference_elapsed_seconds",
        "inference_milliseconds_per_sample",
        "training_elapsed_seconds",
    ):
        if field in meta:
            summary[field] = summary["farm_id"].map(meta[field])
    summary["reference_only"] = True
    summary["selection_eligible"] = False
    summary["result_source"] = (
        "hash_validated_stage5a_x1_fixed_g0_direct_reference_no_forward_no_copy"
    )
    summary["diagnostic_source"] = "same_x1_candidate_control_for_gate_effect"
    summary["test_reuse_status"] = TEST_REUSE_STATUS
    summary["selection_split"] = "test"
    summary["test_used_for_selection"] = True
    summary["selection_role"] = "same_candidate_guard_reference_not_selectable"
    summary["test_is_final_blind_evaluation"] = False
    for field in (
        "model_path",
        "model_sha256",
        "artifact_path",
        "artifact_sha256",
        "candidate_archive_path",
        "candidate_archive_sha256",
    ):
        summary[field] = summary["farm_id"].map(meta[field])
    summary["single_window_figure_path"] = summary["farm_id"].map(
        meta["replay_single_window_figure_path"]
    )
    summary["weighted_curve_figure_path"] = summary["farm_id"].map(
        meta["replay_weighted_curve_figure_path"]
    )
    return {
        "summary": summary,
        "horizon": horizon,
        "candidate": candidate,
        "regime": regime,
        "assignments": assignments,
        "safety": safety,
        "calibration": calibration,
    }, paths, raw["comparison"]


def prediction_dirs(_variant, output_root):
    root = os.path.join(output_root, OUTPUT_SUBDIR)
    values = {
        "root": root,
        "predictions": os.path.join(root, "predictions"),
        "candidate_archives": os.path.join(root, "candidate_archives"),
        "candidate_metrics": os.path.join(root, "candidate_metrics"),
        "regime_metrics": os.path.join(root, "regime_metrics"),
        "regime_assignments": os.path.join(root, "regime_assignments"),
        "safety_diagnostics": os.path.join(root, "safety_diagnostics"),
        "calibration": os.path.join(root, "calibration"),
        "gate_diagnostics": os.path.join(root, "gate_diagnostics"),
        "single_windows": os.path.join(root, "single_window_comparisons"),
        "weighted_curves": os.path.join(root, "weighted_curves"),
        "figures": os.path.join(root, "figures"),
        "matplotlib_cache": os.path.join(root, "matplotlib_cache"),
    }
    for path in values.values():
        os.makedirs(path, exist_ok=True)
    return values


@contextmanager
def _bound_shared_helpers():
    """Temporarily bind generic helpers and restore every mutated global."""
    previous = {
        "tf_train": stage4_predict.tf_train,
        "ALL_VARIANTS": stage4_predict.ALL_VARIANTS,
        "NEW_VARIANTS": stage4_predict.NEW_VARIANTS,
        "TEST_REUSE_STATUS": stage4_predict.TEST_REUSE_STATUS,
        "prediction_dirs": stage4_predict.prediction_dirs,
        "gate_train": gate_predict.gate_train,
    }
    stage4_predict.tf_train = x1r_train
    stage4_predict.ALL_VARIANTS = NEW_VARIANTS
    stage4_predict.NEW_VARIANTS = NEW_VARIANTS
    stage4_predict.TEST_REUSE_STATUS = TEST_REUSE_STATUS
    stage4_predict.prediction_dirs = prediction_dirs
    gate_predict.gate_train = x1r_train
    try:
        yield
    finally:
        stage4_predict.tf_train = previous["tf_train"]
        stage4_predict.ALL_VARIANTS = previous["ALL_VARIANTS"]
        stage4_predict.NEW_VARIANTS = previous["NEW_VARIANTS"]
        stage4_predict.TEST_REUSE_STATUS = previous["TEST_REUSE_STATUS"]
        stage4_predict.prediction_dirs = previous["prediction_dirs"]
        gate_predict.gate_train = previous["gate_train"]


def _validate_x1r_artifact(farm_id, training_marker):
    files = training_marker.get("files", {})
    artifact_path = _validate_record(
        f"x1r/{farm_id}/artifact", files.get(f"x1r.{farm_id}.artifact_path")
    )
    artifact = joblib.load(artifact_path)
    calibration = artifact.get("candidate_calibration", {})
    q90 = np.asarray(calibration.get("candidate_difference_q90", ()), dtype=np.float32)
    source_members = (
        ("source_x1_model_path", "source_x1_model_sha256"),
        ("source_x1_artifact_path", "source_x1_artifact_sha256"),
        ("source_x1_record_path", "source_x1_record_sha256"),
        ("source_stage5a_training_marker_path", "source_stage5a_training_marker_sha256"),
    )
    source_hashes_valid = all(
        _sha256(artifact.get(path_key)) == artifact.get(hash_key)
        for path_key, hash_key in source_members
    )
    q90_hash = x1r_train._array_sha256([("q90", q90)]) if q90.shape == (x1r_train.FORECAST_LEN,) else None
    checks = {
        "variant": artifact.get("variant_id") == "x1r",
        "farm": str(artifact.get("farm_id")) == str(farm_id),
        "candidate_source": artifact.get("candidate_source") == "x1",
        "total_params": int(artifact.get("total_params", -1))
        == x1r_train.EXPECTED_TOTAL_PARAMS["x1r"],
        "old_g0_pruned": artifact.get("old_g0_gate_in_new_graph") is False,
        "candidate_snapshot_nonempty": bool(
            artifact.get("candidate_snapshot_before_gate_sha256")
        ),
        "candidate_snapshot_frozen": artifact.get(
            "candidate_snapshot_before_gate_sha256"
        )
        == artifact.get("candidate_snapshot_after_gate_sha256"),
        "candidate_output_frozen": artifact.get(
            "candidate_output_before_gate_sha256"
        )
        == artifact.get("candidate_output_after_gate_sha256"),
        "candidate_drift_zero": float(
            artifact.get("candidate_gate_calibration_max_abs_drift", np.nan)
        )
        == 0.0,
        "persistence_drift_zero": float(
            artifact.get("persistence_gate_max_abs_drift", np.nan)
        )
        == 0.0,
        "q90_shape_finite_nonnegative": q90.shape == (x1r_train.FORECAST_LEN,)
        and np.isfinite(q90).all()
        and np.all(q90 >= 0.0),
        "q90_quantile": np.isclose(
            float(calibration.get("quantile", np.nan)),
            x1r_train.CALIBRATION_DIFFERENCE_QUANTILE,
            rtol=0.0,
            atol=1e-12,
        ),
        "q90_hash": calibration.get("candidate_difference_q90_sha256") == q90_hash,
        "train_only_scope": "train" in str(calibration.get("scope", "")).lower(),
        "sample_element_count": int(calibration.get("sample_count", -1)) > 0
        and int(calibration.get("element_count", -2))
        == int(calibration.get("sample_count", -1)) * x1r_train.FORECAST_LEN,
        "source_hashes": source_hashes_valid,
        "gate_candidate_frozen": bool(
            artifact.get("gate_training", {}).get("candidate_frozen_all_phases")
        ),
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError(f"X1R/{farm_id} artifact闭环审计失败: {failed}")
    return artifact


def predict_x1r(test_file, training_marker, max_samples=None):
    farm_id = str(common_predict.get_farm_id(test_file))
    _validate_x1r_artifact(farm_id, training_marker)
    with _bound_shared_helpers():
        payload = stage4_predict.predict_variant(
            "x1r", test_file, training_marker, max_samples
        )
    payload["result_source"] = "new_x1r_single_formal_test_forward"
    payload["diagnostic_source"] = "same_forward_calibrated_safe_gate"
    return payload


def save_x1r_payload(payload, output_root, skip_plots=False):
    with _bound_shared_helpers():
        result = stage4_predict.save_payload(payload, output_root, skip_plots)
    result["summary"]["result_source"] = "new_x1r_single_formal_test_forward"
    result["summary"]["selection_eligible"] = True
    result["summary"]["candidate_source"] = "stage5a_x1_frozen"
    return result


def _read_archive(row, label):
    path = str(row["candidate_archive_path"])
    if not os.path.isfile(path) or _sha256(path) != row["candidate_archive_sha256"]:
        raise ValueError(f"{label} candidate archive不存在或hash漂移")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def validate_candidate_invariants(source_summary, results):
    source = {
        str(row["farm_id"]): _read_archive(row, f"X1/{row['farm_id']}")
        for _, row in source_summary.iterrows()
    }
    rows = []
    for result in results:
        payload = result["payload"]
        farm_id = str(payload["farm_id"])
        archive = source[farm_id]
        for key in ("sample_id", "horizon_step", "forecast_origin_time"):
            if not np.array_equal(archive[key], payload[key]):
                raise ValueError(f"X1R/{farm_id}窗口键{key}未对齐X1")
        if not np.array_equal(archive["y_true"], payload["y_true"], equal_nan=True):
            raise ValueError(f"X1R/{farm_id}真值未对齐X1")
        capacity = float(payload["capacity"])
        p_drift = np.abs(
            np.asarray(payload["persistence"], float)
            - np.asarray(archive["persistence"], float)
        ) / capacity
        c_drift = np.abs(
            np.asarray(payload["corrected"], float)
            - np.asarray(archive["corrected"], float)
        ) / capacity
        row = {
            "model_variant": "x1r",
            "farm_id": farm_id,
            "source_candidate": "stage5a_x1",
            "persistence_capacity_normalized_max_abs_drift": float(np.max(p_drift)),
            "persistence_capacity_normalized_mean_abs_drift": float(np.mean(p_drift)),
            "corrected_capacity_normalized_max_abs_drift": float(np.max(c_drift)),
            "corrected_capacity_normalized_mean_abs_drift": float(np.mean(c_drift)),
        }
        row["persistence_control_pass"] = bool(
            row["persistence_capacity_normalized_max_abs_drift"]
            <= PERSISTENCE_MAX_NORM_TOL
        )
        row["corrected_candidate_control_pass"] = bool(
            row["corrected_capacity_normalized_max_abs_drift"]
            <= CANDIDATE_MAX_NORM_TOL
            and row["corrected_capacity_normalized_mean_abs_drift"]
            <= CANDIDATE_MEAN_NORM_TOL
        )
        row["candidate_identity_pass"] = bool(
            row["persistence_control_pass"]
            and row["corrected_candidate_control_pass"]
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if (
        len(frame) != len(_expected_farms())
        or set(frame["farm_id"].astype(str)) != set(_expected_farms())
        or not frame["candidate_identity_pass"].all()
    ):
        raise ValueError(
            "X1R未保持完整X1 candidate身份: "
            + str(frame[~frame["candidate_identity_pass"]].to_dict(orient="records"))
        )
    return frame


def _exact_five(frame, label, columns=()):
    if (
        len(frame) != len(_expected_farms())
        or set(frame["farm_id"].astype(str)) != set(_expected_farms())
        or frame.duplicated(["farm_id"]).any()
    ):
        raise ValueError(f"{label}不是唯一5场站")
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"{label}.{column}含非有限值")
    return frame


def _macro_regime(regime, variant, group):
    part = regime[
        (regime["model_variant"].astype(str) == variant)
        & (regime["candidate"].astype(str) == "fused")
        & (regime["regime_group"].astype(str) == group)
        & (regime["horizon_step"].map(_canonical_horizon) == "all")
    ]
    part = _exact_five(part, f"{variant}/fused/{group}", ("capacity_normalized_rmse",))
    return float(part["capacity_normalized_rmse"].mean())


def build_comparison(summary, regime, candidate_invariants):
    rows = []
    for variant in CLOSURE_VARIANTS:
        frame = _exact_five(
            summary[summary["model_variant"].astype(str) == variant],
            variant,
            (
                "capacity_normalized_rmse",
                "capacity_normalized_mae",
                "positive_regret_mean",
                "harm_rate_0_005",
                "oracle_brier",
                "ece_10bin",
                "gate_high_saturation_rate",
                "parameter_count",
            ),
        )
        row = {
            "model_variant": variant,
            "variant_label": str(frame["variant_label"].iloc[0]),
            "macro_test_nrmse": float(frame["capacity_normalized_rmse"].mean()),
            "macro_test_nmae": float(frame["capacity_normalized_mae"].mean()),
            "macro_positive_regret_mean": float(frame["positive_regret_mean"].mean()),
            "macro_harm_rate_0_005": float(frame["harm_rate_0_005"].mean()),
            "macro_oracle_brier": float(frame["oracle_brier"].mean()),
            "macro_ece_10bin": float(frame["ece_10bin"].mean()),
            "macro_gate_high_saturation_rate": float(
                frame["gate_high_saturation_rate"].mean()
            ),
            "parameter_count_max": int(frame["parameter_count"].max()),
            "selection_eligible": variant in {"x0", "x1r"},
        }
        for group in ("dynamic", "ramp_up", "ramp_down"):
            row[f"fused_{group}_nrmse"] = _macro_regime(regime, variant, group)
        rows.append(row)
    result = pd.DataFrame(rows)
    base = result[result["model_variant"] == "x0"].iloc[0]
    fixed = result[result["model_variant"] == "x1_fixed_g0"].iloc[0]
    result["relative_nrmse_vs_x0"] = result["macro_test_nrmse"] / float(
        base["macro_test_nrmse"]
    ) - 1.0
    result["relative_nrmse_vs_x1_fixed_g0"] = result["macro_test_nrmse"] / float(
        fixed["macro_test_nrmse"]
    ) - 1.0

    base_farm = summary[summary["model_variant"] == "x0"].set_index("farm_id")[
        "capacity_normalized_rmse"
    ].astype(float)
    target = summary[summary["model_variant"] == "x1r"].set_index("farm_id")[
        "capacity_normalized_rmse"
    ].astype(float).reindex(base_farm.index)
    nondegraded = int((target <= base_farm + FARM_NONDEGRADE_ATOL).sum())
    improved = int((target < base_farm - FARM_NONDEGRADE_ATOL).sum())
    benefit = base_farm - target
    best_farm = str(benefit.idxmax())
    retained = base_farm.index != best_farm
    leave_one_improvement = 1.0 - float(target[retained].mean()) / float(
        base_farm[retained].mean()
    )
    x1r = result[result["model_variant"] == "x1r"].iloc[0]
    flags = {
        "macro_nrmse_improves_at_least_0_2pct": bool(
            x1r["macro_test_nrmse"]
            <= base["macro_test_nrmse"] * (1.0 - REQUIRED_MACRO_IMPROVEMENT)
        ),
        "strictly_better_than_same_x1_fixed_g0": bool(
            x1r["macro_test_nrmse"] < fixed["macro_test_nrmse"]
        ),
        "farms_nondegraded_vs_x0": nondegraded,
        "at_least_4_farms_nondegraded": nondegraded >= MIN_NONDEGRADED_FARMS,
        "farms_strictly_improved_vs_x0": improved,
        "at_least_3_farms_strictly_improved": improved
        >= MIN_STRICTLY_IMPROVED_FARMS,
        "best_benefit_farm_removed": best_farm,
        "leave_best_benefit_farm_out_improvement": leave_one_improvement,
        "leave_best_benefit_farm_out_positive": leave_one_improvement > 0.0,
        "dynamic_ramp_guard_pass": all(
            x1r[f"fused_{group}_nrmse"]
            <= base[f"fused_{group}_nrmse"] * (1.0 + REGIME_DEGRADATION_TOL)
            for group in ("dynamic", "ramp_up", "ramp_down")
        ),
        "positive_regret_guard_pass": bool(
            x1r["macro_positive_regret_mean"]
            <= base["macro_positive_regret_mean"]
            * (1.0 + SAFETY_REGRET_RELATIVE_TOL)
        ),
        "harm_rate_guard_pass": bool(
            x1r["macro_harm_rate_0_005"]
            <= base["macro_harm_rate_0_005"] + SAFETY_HARM_ABS_TOL
        ),
        "same_candidate_brier_improves_at_least_10pct": bool(
            x1r["macro_oracle_brier"]
            <= fixed["macro_oracle_brier"] * (1.0 - BRIER_RELATIVE_IMPROVEMENT)
        ),
        "same_candidate_ece_improves_at_least_15pct": bool(
            x1r["macro_ece_10bin"]
            <= fixed["macro_ece_10bin"] * (1.0 - ECE_RELATIVE_IMPROVEMENT)
        ),
        "gate_high_saturation_below_50pct": bool(
            x1r["macro_gate_high_saturation_rate"] < HIGH_SATURATION_MAX
        ),
        "candidate_identity_pass": bool(
            candidate_invariants["candidate_identity_pass"].all()
        ),
        "parameter_under_30k": bool(x1r["parameter_count_max"] < PARAMETER_LIMIT),
    }
    boolean_guards = [
        value for key, value in flags.items() if isinstance(value, (bool, np.bool_))
    ]
    guard = bool(all(boolean_guards))
    for key, value in flags.items():
        result[key] = pd.Series([pd.NA] * len(result), dtype="object")
        result.loc[result["model_variant"] == "x1r", key] = value
    result["selection_guard_pass"] = False
    result.loc[result["model_variant"] == "x0", "selection_guard_pass"] = True
    result.loc[result["model_variant"] == "x1r", "selection_guard_pass"] = guard
    result["stage5b_x2_x3_unlocked"] = False
    result.loc[result["model_variant"] == "x1r", "stage5b_x2_x3_unlocked"] = guard
    return result


def select_model(comparison):
    x1r = comparison[comparison["model_variant"] == "x1r"].iloc[0]
    selected_variant = "x1r" if bool(x1r["selection_guard_pass"]) else "x0"
    status = (
        "x1r_passed_all_predeclared_closure_guards_unlock_x2_x3"
        if selected_variant == "x1r"
        else "fallback_x0_x1r_failed_one_or_more_closure_guards_stop_before_x2_x3"
    )
    comparison = comparison.copy()
    comparison["selected"] = comparison["model_variant"] == selected_variant
    comparison["selection_status"] = status
    return comparison[comparison["selected"]].iloc[0], comparison


def build_controlled_contrasts(comparison, candidate_evidence):
    lookup = comparison.set_index("model_variant")
    rows = []
    for name, left, right, interpretation in (
        (
            "candidate_change_under_fixed_g0",
            "x1_fixed_g0",
            "x0",
            "old_G0_with_X1_candidate_vs_old_G0_with_F7_candidate",
        ),
        (
            "gate_recalibration_effect_same_x1_candidate",
            "x1r",
            "x1_fixed_g0",
            "new_calibrated_safe_gate_vs_old_G0_on_identical_X1_candidate",
        ),
        (
            "total_x1r_deployment_effect",
            "x1r",
            "x0",
            "X1_candidate_plus_new_gate_vs_F7_candidate_plus_old_G0",
        ),
    ):
        for metric in (
            "macro_test_nrmse",
            "macro_test_nmae",
            "macro_positive_regret_mean",
            "macro_harm_rate_0_005",
            "macro_oracle_brier",
            "macro_ece_10bin",
            "fused_dynamic_nrmse",
            "fused_ramp_up_nrmse",
            "fused_ramp_down_nrmse",
        ):
            lvalue, rvalue = float(lookup.loc[left, metric]), float(lookup.loc[right, metric])
            rows.append(
                {
                    "contrast": name,
                    "left_variant": left,
                    "right_variant": right,
                    "controlled_interpretation": interpretation,
                    "metric": metric,
                    "left_value": lvalue,
                    "right_value": rvalue,
                    "signed_delta_left_minus_right": lvalue - rvalue,
                    "relative_delta_left_vs_right": lvalue / rvalue - 1.0
                    if rvalue != 0.0
                    else np.nan,
                    "negative_delta_means_improvement": True,
                }
            )
    evidence = candidate_evidence.set_index("model_variant")
    if {"x0", "x1"}.issubset(evidence.index):
        x0 = float(evidence.loc["x0", "macro_corrected_candidate_test_nrmse"])
        x1 = float(evidence.loc["x1", "macro_corrected_candidate_test_nrmse"])
        rows.append(
            {
                "contrast": "stage5a_candidate_gain_x1_minus_x0",
                "left_variant": "x1",
                "right_variant": "x0",
                "controlled_interpretation": "corrected_candidate_only_no_gate_selection",
                "metric": "macro_corrected_candidate_test_nrmse",
                "left_value": x1,
                "right_value": x0,
                "signed_delta_left_minus_right": x1 - x0,
                "relative_delta_left_vs_right": x1 / x0 - 1.0,
                "negative_delta_means_improvement": True,
            }
        )
    return pd.DataFrame(rows)


def build_complexity(summary):
    result = (
        summary.groupby("model_variant", as_index=False)
        .agg(
            parameter_count_max=("parameter_count", "max"),
            trainable_parameter_count_max=("trainable_parameter_count", "max"),
            inference_ms_per_sample_macro=(
                "inference_milliseconds_per_sample",
                "mean",
            ),
        )
        .copy()
    )
    result["parameter_under_30k"] = result["parameter_count_max"] < PARAMETER_LIMIT
    result["source_candidate"] = result["model_variant"].map(
        {"x0": "f7", "x1_fixed_g0": "x1", "x1r": "x1"}
    )
    result["new_model_trained_this_round"] = result["model_variant"] == "x1r"
    result["random_seed"] = x1r_train.RANDOM_SEED
    result["seed_count"] = 1
    result["stability_scope"] = "single_seed_2026_no_multiseed_claim"
    return result


def validate_complete_matrix(frames):
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
        "safety": ("model_variant", "farm_id", "scope_type", "scope_value"),
        "calibration": ("model_variant", "farm_id", "gate_bin"),
    }
    for name, keys in natural_keys.items():
        frame = frames[name]
        if set(frame["model_variant"].astype(str)) != set(CLOSURE_VARIANTS):
            raise ValueError(f"X1R正式{name}未覆盖三个闭环对照")
        if set(frame["farm_id"].astype(str)) != set(_expected_farms()):
            raise ValueError(f"X1R正式{name}未覆盖固定5场站")
        normalized = frame.copy()
        for key in keys:
            normalized[key] = normalized[key].map(
                lambda value: "<NA>" if pd.isna(value) else str(value)
            )
        if normalized.duplicated(list(keys)).any():
            raise ValueError(f"X1R正式{name}自然键重复")
        suffix = keys[2:]
        for farm_id in _expected_farms():
            base = normalized[
                (normalized["model_variant"] == "x0")
                & (normalized["farm_id"] == farm_id)
            ]
            for variant in CLOSURE_VARIANTS:
                target = normalized[
                    (normalized["model_variant"] == variant)
                    & (normalized["farm_id"] == farm_id)
                ]
                if len(target) != len(base):
                    raise ValueError(f"正式{name}/{variant}/{farm_id}行数与X0不同")
                if suffix and set(target[list(suffix)].itertuples(index=False, name=None)) != set(
                    base[list(suffix)].itertuples(index=False, name=None)
                ):
                    raise ValueError(f"正式{name}/{variant}/{farm_id}自然键集与X0不同")


def save_aggregate_figures(
    comparison, summary, horizon, safety, calibration, output_dir
):
    dirs = prediction_dirs("x1r", os.path.dirname(output_dir))
    figure_dir = dirs["figures"]
    plt = common_predict.setup_matplotlib(dirs)
    paths = {}

    ordered = comparison.set_index("model_variant").reindex(CLOSURE_VARIANTS)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    colors = ["#d62728" if flag else "#4c78a8" for flag in ordered["selected"]]
    ax.bar(ordered.index, ordered["macro_test_nrmse"], color=colors)
    ax.set_ylabel("Five-farm macro fused NRMSE")
    ax.set_title("X1R deployment-closure ranking")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths["closure_rank_figure"] = os.path.join(
        figure_dir, "x1r_gate_closure_macro_nrmse.png"
    )
    fig.savefig(paths["closure_rank_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    matrix = summary.pivot(
        index="model_variant", columns="farm_id", values="capacity_normalized_rmse"
    ).reindex(index=CLOSURE_VARIANTS, columns=_expected_farms())
    fig, ax = plt.subplots(figsize=(10.0, 4.5))
    image = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(matrix)), labels=matrix.index)
    ax.set_xticks(
        range(len(matrix.columns)), labels=[str(value)[-4:] for value in matrix.columns]
    )
    ax.set_title("Fused NRMSE by farm")
    fig.colorbar(image, ax=ax, label="NRMSE")
    fig.tight_layout()
    paths["farm_heatmap_figure"] = os.path.join(
        figure_dir, "x1r_gate_closure_farm_heatmap.png"
    )
    fig.savefig(paths["farm_heatmap_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    numeric = horizon[horizon["horizon_step"].map(_canonical_horizon) != "all"].copy()
    numeric["h"] = pd.to_numeric(numeric["horizon_step"], errors="raise")
    macro = numeric.groupby(["model_variant", "h"], as_index=False)[
        "capacity_normalized_rmse"
    ].mean()
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for variant in CLOSURE_VARIANTS:
        part = macro[macro["model_variant"] == variant].sort_values("h")
        ax.plot(part["h"], part["capacity_normalized_rmse"], marker="o", ms=3, label=variant)
    ax.set(
        xlabel="Forecast horizon (15-min steps)",
        ylabel="Five-farm macro fused NRMSE",
    )
    ax.set_title("X1R horizon-wise deployment error")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    paths["horizon_figure"] = os.path.join(
        figure_dir, "x1r_gate_closure_horizon_nrmse.png"
    )
    fig.savefig(paths["horizon_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics = ("macro_positive_regret_mean", "macro_harm_rate_0_005", "macro_oracle_brier", "macro_ece_10bin")
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for axis, metric in zip(axes.ravel(), metrics):
        axis.bar(ordered.index, ordered[metric], color="#59a14f")
        axis.set_title(metric)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Safety and calibration closure")
    fig.tight_layout()
    paths["safety_calibration_figure"] = os.path.join(
        figure_dir, "x1r_gate_closure_safety_calibration.png"
    )
    fig.savefig(paths["safety_calibration_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for variant in CLOSURE_VARIANTS:
        part = calibration[calibration["model_variant"] == variant].copy()
        part["weighted_gate"] = part["mean_raw_gate"].fillna(0.0) * part["count"]
        part["weighted_truth"] = (
            part["corrected_better_rate"].fillna(0.0) * part["count"]
        )
        grouped = part.groupby("gate_bin", as_index=False).agg(
            count=("count", "sum"),
            weighted_gate=("weighted_gate", "sum"),
            weighted_truth=("weighted_truth", "sum"),
        )
        valid = grouped["count"] > 0
        ax.plot(
            grouped.loc[valid, "weighted_gate"] / grouped.loc[valid, "count"],
            grouped.loc[valid, "weighted_truth"] / grouped.loc[valid, "count"],
            marker="o",
            label=variant,
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.45, label="ideal")
    ax.set(xlabel="Mean raw gate", ylabel="Corrected-better rate")
    ax.set_title("Five-farm gate reliability")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    paths["reliability_figure"] = os.path.join(
        figure_dir, "x1r_gate_closure_reliability.png"
    )
    fig.savefig(paths["reliability_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    regimes = ("stable", "dynamic", "ramp_up", "ramp_down", "low_power")
    scoped = safety[
        (safety["scope_type"].astype(str) == "regime")
        & safety["scope_value"].astype(str).isin(regimes)
    ].copy()
    gate_macro = scoped.groupby(["model_variant", "scope_value"], as_index=False)[
        "gate_mean"
    ].mean()
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    x = np.arange(len(regimes))
    width = 0.24
    for index, variant in enumerate(CLOSURE_VARIANTS):
        part = gate_macro[gate_macro["model_variant"] == variant].set_index(
            "scope_value"
        )
        ax.bar(
            x + (index - 1) * width,
            [float(part.loc[regime, "gate_mean"]) for regime in regimes],
            width,
            label=variant,
        )
    ax.set_xticks(x, labels=regimes)
    ax.set_ylabel("Five-farm macro gate mean")
    ax.set_title("Gate allocation by wind-power regime")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    paths["gate_regime_figure"] = os.path.join(
        figure_dir, "x1r_gate_closure_gate_by_regime.png"
    )
    fig.savefig(paths["gate_regime_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def write_report(comparison, selected, invariants, candidate_evidence, output_dir):
    compact = [
        "model_variant",
        "variant_label",
        "macro_test_nrmse",
        "relative_nrmse_vs_x0",
        "relative_nrmse_vs_x1_fixed_g0",
        "macro_positive_regret_mean",
        "macro_harm_rate_0_005",
        "macro_oracle_brier",
        "macro_ece_10bin",
        "macro_gate_high_saturation_rate",
        "parameter_count_max",
        "selection_eligible",
        "selection_guard_pass",
        "selected",
    ]
    x1r = comparison[comparison["model_variant"] == "x1r"].iloc[0]
    status = "允许先启动X2/X3" if bool(x1r["stage5b_x2_x3_unlocked"]) else "停止，不启动X2-X6"
    evidence_cols = [
        col
        for col in (
            "model_variant",
            "variant_label",
            "macro_corrected_candidate_test_nrmse",
            "actual_candidate_improvement_vs_x0",
            "parameter_count_max",
        )
        if col in candidate_evidence
    ]
    lines = [
        "# X1R门控收益转化闭环：测试集最终选型",
        "",
        f"最终选中 **{selected['model_variant']}**；部署结论：**{status}**。",
        "",
        "本轮按既有测试集选型，标记为 `legacy_seen_test_selected`，不是独立最终盲测。",
        "",
        "## 部署闭环（fused指标）",
        "",
        comparison[compact].to_markdown(index=False),
        "",
        "X1-fixed与X1R具有完全相同的X1 candidate；两者差异才是门控重新校准的受控效应。",
        "X1-fixed仅作诊断，正式部署选型只允许X0与X1R。",
        "",
        "## Stage-5A candidate evidence（不与fused混排）",
        "",
        candidate_evidence[evidence_cols].to_markdown(index=False),
        "",
        "## X1 candidate身份审计",
        "",
        invariants.to_markdown(index=False),
        "",
        "## 全部通过才解锁X2/X3的预声明条件",
        "",
        "- X1R fused宏NRMSE相对X0至少改善0.2%，且严格优于X1-fixed。",
        "- 相对X0至少4/5场站不退化、至少3/5严格改善；删除最大收益场站后仍有正收益。",
        "- dynamic、ramp-up、ramp-down均不得相对X0恶化超过0.5%。",
        "- positive regret不得超过X0的+0.5%相对值，harm@0.005不得超过X0的+0.002绝对值。",
        "- 同一X1 oracle下，Brier至少改善10%、ECE至少改善15%，高饱和率低于50%。",
        "- X1 candidate身份必须通过全部5场站审计，模型参数必须小于30k。",
        "- 即使全部通过，下一步也只先启动X2/X3；每个新candidate必须重生成自己的train-only oracle/Q90并重训gate。",
        "",
    ]
    return _atomic_text(
        "\n".join(lines),
        os.path.join(output_dir, "x1r_gate_closure_test_final_selection.md"),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--farms", default=os.getenv("WIND_X1R_FARMS", ""), help="逗号分隔场站ID"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("PYTHONHASHSEED", str(x1r_train.RANDOM_SEED))
    keras.utils.set_random_seed(x1r_train.RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    farms = [value.strip() for value in args.farms.split(",") if value.strip()]
    invalid = set(farms) - set(_expected_farms())
    if invalid:
        raise ValueError(f"未知场站: {sorted(invalid)}")
    farms = farms or _expected_farms()
    if args.smoke:
        farms = farms[:1]
        args.max_samples = args.max_samples or 32
    if args.max_samples is not None and not args.smoke:
        raise ValueError("--max-samples只允许与--smoke同时使用")
    full = bool(
        set(farms) == set(_expected_farms())
        and not args.smoke
        and args.max_samples is None
        and not args.skip_plots
    )
    output_root = (
        x1r_train.RESULT_ROOT
        if full
        else os.path.join(
            x1r_train.RESULT_ROOT,
            "partial_runs",
            args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    )
    output_dir = os.path.join(output_root, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)
    running_path = os.path.join(output_dir, RUNNING_MARKER_NAME)
    if full:
        _atomic_json(
            {
                "status": "running",
                "protocol_version": x1r_train.PROTOCOL_VERSION,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "farm_ids": farms,
            },
            running_path,
        )

    training_marker_path, training_marker = validate_training_bundle()
    stage5a_marker, stage4b_marker = validate_source_bundles(training_marker)
    x0_frames, stage4b_paths = _stage4b_x0_frames(stage4b_marker)
    x1_frames, stage5a_paths, candidate_evidence = _stage5a_x1_fixed_frames(
        stage5a_marker
    )
    test_files = {
        str(farm_id): str(record["path"])
        for farm_id, record in stage5a_marker["test_files"].items()
    }
    results = []
    for farm_id in farms:
        print(f"\n===== X1R测试预测 farm={farm_id} =====")
        payload = predict_x1r(
            test_files[farm_id], training_marker, args.max_samples
        )
        results.append(save_x1r_payload(payload, output_root, args.skip_plots))

    if not full:
        partial = pd.concat(
            [result["summary"] for result in results], ignore_index=True, sort=False
        )
        path = _atomic_csv(
            partial, os.path.join(output_dir, "x1r_gate_closure_partial_summary.csv")
        )
        _atomic_json(
            {
                "status": "partial_not_formal",
                "farms": farms,
                "max_samples": args.max_samples,
                "skip_plots": args.skip_plots,
                "summary": _file_record(path),
            },
            os.path.join(output_dir, "partial_run_manifest.json"),
        )
        print(f"partial/smoke结果（不参与正式选型）: {path}")
        return

    if len(results) != len(_expected_farms()):
        raise ValueError("X1R正式预测不是1变体×5场站")
    frames = {
        key: pd.concat(
            [x0_frames[key], x1_frames[key]] + [result[key] for result in results],
            ignore_index=True,
            sort=False,
        )
        for key in (
            "summary",
            "horizon",
            "candidate",
            "regime",
            "assignments",
            "safety",
            "calibration",
        )
    }
    validate_complete_matrix(frames)
    invariants = validate_candidate_invariants(x1_frames["summary"], results)
    comparison = build_comparison(frames["summary"], frames["regime"], invariants)
    selected, comparison = select_model(comparison)
    candidate_evidence = candidate_evidence[
        candidate_evidence["model_variant"].astype(str).isin(
            ("x0", "x1_f", "x1_c", "x1")
        )
    ].copy()
    if (
        len(candidate_evidence) != 4
        or set(candidate_evidence["model_variant"].astype(str))
        != {"x0", "x1_f", "x1_c", "x1"}
        or candidate_evidence.duplicated(["model_variant"]).any()
    ):
        raise ValueError("Stage-5A candidate evidence不是预声明的4个唯一对照")
    contrasts = build_controlled_contrasts(comparison, candidate_evidence)
    complexity = build_complexity(frames["summary"])

    paths = {}
    for key, frame in frames.items():
        paths[key] = _atomic_csv(
            frame, os.path.join(output_dir, f"x1r_gate_closure_test_{key}.csv")
        )
    paths["comparison"] = _atomic_csv(
        comparison,
        os.path.join(output_dir, "x1r_gate_closure_test_variant_comparison.csv"),
    )
    paths["final_selection"] = _atomic_csv(
        comparison[comparison["selected"]],
        os.path.join(output_dir, "x1r_gate_closure_test_final_selection.csv"),
    )
    paths["candidate_evidence"] = _atomic_csv(
        candidate_evidence,
        os.path.join(output_dir, "x1r_gate_closure_stage5a_candidate_evidence.csv"),
    )
    paths["controlled_contrasts"] = _atomic_csv(
        contrasts,
        os.path.join(output_dir, "x1r_gate_closure_controlled_contrasts.csv"),
    )
    paths["candidate_invariants"] = _atomic_csv(
        invariants,
        os.path.join(output_dir, "x1r_gate_closure_candidate_invariants.csv"),
    )
    paths["complexity"] = _atomic_csv(
        complexity,
        os.path.join(output_dir, "x1r_gate_closure_test_complexity.csv"),
    )
    paths.update(
        save_aggregate_figures(
            comparison,
            frames["summary"],
            frames["horizon"],
            frames["safety"],
            frames["calibration"],
            output_dir,
        )
    )
    paths["report"] = write_report(
        comparison,
        selected,
        invariants,
        candidate_evidence,
        output_dir,
    )

    source_rows = [
        {
            "source": "X1R training complete marker",
            "key": "marker",
            **_file_record(training_marker_path),
            "reuse_action": "new_model_identity",
        },
        {
            "source": "Stage-5A prediction complete marker",
            "key": "marker",
            **_file_record(STAGE5A_MARKER),
            "reuse_action": "hash_validated_x1_fixed_reference_no_forward_no_copy",
        },
        {
            "source": "Stage-4B prediction complete marker",
            "key": "marker",
            **_file_record(STAGE4B_MARKER),
            "reuse_action": "hash_validated_x0_reference_no_forward_no_copy",
        },
    ]
    for family, mapping in (("Stage-5A", stage5a_paths), ("Stage-4B", stage4b_paths)):
        for key, path in mapping.items():
            source_rows.append(
                {
                    "source": family,
                    "key": key,
                    **_file_record(path),
                    "reuse_action": "read_only_filter_and_aggregate_reference",
                }
            )
    paths["source_manifest"] = _atomic_csv(
        pd.DataFrame(source_rows),
        os.path.join(output_dir, "x1r_gate_closure_source_reuse_manifest.csv"),
    )

    visual_candidates = []
    for key, path in paths.items():
        if isinstance(path, str) and path.lower().endswith(".png"):
            visual_candidates.append((f"aggregate.{key}", path))
    for index, result in enumerate(results):
        for key, path in result["paths"].items():
            if isinstance(path, str) and path.lower().endswith(".png"):
                visual_candidates.append((f"x1r.{index}.{key}", path))
    for _, row in x0_frames["summary"].iterrows():
        for field in ("single_window_figure_path", "weighted_curve_figure_path"):
            path = row.get(field)
            if isinstance(path, str) and os.path.isfile(path):
                visual_candidates.append((f"x0.{row['farm_id']}.{field}", path))
    for _, row in x1_frames["summary"].iterrows():
        for field in ("single_window_figure_path", "weighted_curve_figure_path"):
            path = row.get(field)
            if isinstance(path, str) and os.path.isfile(path):
                visual_candidates.append((f"x1_fixed.{row['farm_id']}.{field}", path))
    visual_rows, seen = [], set()
    for key, path in visual_candidates:
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        visual_rows.append({"key": key, **_file_record(path)})
    if not visual_rows:
        raise ValueError("X1R正式bundle没有任何可视化图片")
    paths["visual_inventory"] = _atomic_csv(
        pd.DataFrame(visual_rows),
        os.path.join(output_dir, "x1r_gate_closure_visual_inventory.csv"),
    )

    files = {
        "prediction_code": _file_record(__file__),
        "training_code": _file_record(x1r_train.__file__),
        "training_marker": _file_record(training_marker_path),
        "source_stage5a_prediction_marker": _file_record(STAGE5A_MARKER),
        "source_stage4b_prediction_marker": _file_record(STAGE4B_MARKER),
        "dependency.stage4_prediction_helpers": _file_record(stage4_predict.__file__),
        "dependency.controlled_gate_prediction_helpers": _file_record(gate_predict.__file__),
        "dependency.common_prediction_helpers": _file_record(common_predict.__file__),
    }
    files.update({f"formal.{key}": _file_record(path) for key, path in paths.items()})
    for result in results:
        farm_id = str(result["payload"]["farm_id"])
        for key, path in result["paths"].items():
            if isinstance(path, str) and os.path.isfile(path):
                files[f"x1r.{farm_id}.{key}"] = _file_record(path)
    marker = {
        "status": "complete",
        "protocol_version": x1r_train.PROTOCOL_VERSION,
        "architecture_version": x1r_train.ARCHITECTURE_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": x1r_train.RANDOM_SEED,
        "selection_split": "test",
        "test_used_for_selection": True,
        "test_reuse_status": TEST_REUSE_STATUS,
        "test_is_final_blind_evaluation": False,
        "closure_variants": list(CLOSURE_VARIANTS),
        "selection_eligible_variants": ["x0", "x1r"],
        "diagnostic_only_variants": ["x1_fixed_g0"],
        "expected_farm_ids": _expected_farms(),
        "new_prediction_count": len(results),
        "source_reference_forward_count": 0,
        "candidate_identity_verified": bool(invariants["candidate_identity_pass"].all()),
        "candidate_identity_rows": int(len(invariants)),
        "candidate_max_norm_tolerance": CANDIDATE_MAX_NORM_TOL,
        "candidate_mean_norm_tolerance": CANDIDATE_MEAN_NORM_TOL,
        "candidate_worst_max_norm_drift": float(
            invariants["corrected_capacity_normalized_max_abs_drift"].max()
        ),
        "candidate_worst_mean_norm_drift": float(
            invariants["corrected_capacity_normalized_mean_abs_drift"].max()
        ),
        "train_only_q90_artifacts_verified": True,
        "selected_variant": str(selected["model_variant"]),
        "stage5b_x2_x3_unlocked": bool(
            comparison.loc[
                comparison["model_variant"] == "x1r", "stage5b_x2_x3_unlocked"
            ].iloc[0]
        ),
        "visualization_count": len(visual_rows),
        "files": files,
        "test_files": {
            farm_id: _file_record(test_files[farm_id]) for farm_id in _expected_farms()
        },
    }
    marker_path = _atomic_json(
        marker, os.path.join(output_dir, FORMAL_MARKER_NAME)
    )
    if os.path.exists(running_path):
        os.remove(running_path)
    print(f"X1R测试汇总: {paths['summary']}")
    print(f"X1R最终选择: {selected['model_variant']}")
    print(f"Stage-5B X2/X3解锁: {marker['stage5b_x2_x3_unlocked']}")
    print(f"X1R正式测试bundle完成: {marker_path}")


if __name__ == "__main__":
    main()
