"""T0/M0/T1--T3 测试集预测、归档与最终选型。

T0 严格只读引用已完成的 Stage-3 G0 bundle：不加载模型、不执行 forward、
不复制模型/权重/候选 archive。M0/T1/T2/T3 各执行一次正式测试前向，并在
``wind_results/time_freq_model`` 下保存逐场结果。最终选型按用户指定的当前测试集
进行，因此报告固定标记为 ``legacy_seen_test_selected``，不是最终盲测。

默认命令执行完整 5 变体 x 5 场站矩阵；partial/smoke 输出自动隔离，不会覆盖
正式结果，也不会发布 complete marker::

    python wind_time_freq_model_predict.py
    python wind_time_freq_model_predict.py --variants t1,t2 --farms FARM --smoke
"""

import argparse
import hashlib
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
import wind_time_freq_model_train as tf_train


OUTPUT_SUBDIR = "testdata_predict_output"
TEST_REUSE_STATUS = "legacy_seen_test_selected"
STAGE3_ROOT = os.path.join("./wind_results", "controlled_gate_cali")
STAGE3_OUTPUT = os.path.join(STAGE3_ROOT, OUTPUT_SUBDIR)
STAGE3_MARKER = os.path.join(
    STAGE3_OUTPUT, "controlled_gate_cali_test_bundle_complete.json"
)
FORMAL_MARKER_NAME = "time_freq_model_test_bundle_complete.json"
ALL_VARIANTS = tuple(tf_train.VARIANT_SPECS)
NEW_VARIANTS = tuple(tf_train.TRAINABLE_VARIANTS)

# 预声明的测试集守门阈值。T0始终是安全fallback。
MACRO_NRMSE_TOL = 0.002
FARM_NRMSE_TOL = 0.01
MIN_FARMS_WITHIN_GUARD = 4
RAMP_NRMSE_TOL = 0.005
CANDIDATE_OVERALL_TOL = 0.002
CANDIDATE_REGIME_TOL = 0.005
SAFETY_REGRET_TOL = 0.005
SAFETY_HARM_ABS_TOL = 0.002
NRMSE_TIE_TOL = 0.001

# 候选对照不变量。Persistence 是同一测试窗最后一个历史功率的直接复制，
# 所以 scaled archive 必须逐点一致；M0 corrected 会跨 TensorFlow 运行时重建，
# 沿用 Stage-3 已验证的容量归一化浮点容差，不能误设为 bitwise exact。
PERSISTENCE_CONTROL_MAX_NORM_TOL = 1e-6
M0_CORRECTED_MAX_NORM_TOL = 1e-4
M0_CORRECTED_MEAN_NORM_TOL = 1e-6

STAGE3_FORMAL_FILES = {
    "summary": "controlled_gate_cali_test_metrics_summary.csv",
    "horizon": "controlled_gate_cali_test_metrics_by_horizon.csv",
    "candidate": "controlled_gate_cali_test_candidate_metrics.csv",
    "regime": "controlled_gate_cali_test_regime_metrics.csv",
    "assignments": "controlled_gate_cali_test_regime_assignments.csv",
    "safety": "controlled_gate_cali_test_gate_safety.csv",
    "calibration": "controlled_gate_cali_test_reliability.csv",
}


def _sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _file_record(path):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"正式文件不存在: {path}")
    return {
        "path": os.path.abspath(path),
        "sha256": _sha256(path),
        "size_bytes": os.path.getsize(path),
    }


def _atomic_csv(frame, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def _atomic_json(value, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def _atomic_text(text, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            file.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def _atomic_npz(path, **arrays):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp.npz"
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def _validate_record(label, record):
    path = record.get("path")
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{label}缺失: {path}")
    if _sha256(path) != record.get("sha256"):
        raise ValueError(f"{label} hash漂移")
    if os.path.getsize(path) != int(record.get("size_bytes", -1)):
        raise ValueError(f"{label} size漂移")
    return path


def validate_stage3_bundle():
    """Validate the immutable source and return G0 reference frames/paths."""
    if not os.path.isfile(STAGE3_MARKER):
        raise FileNotFoundError(f"缺少Stage-3 complete marker: {STAGE3_MARKER}")
    with open(STAGE3_MARKER, "r", encoding="utf-8") as file:
        marker = json.load(file)
    if marker.get("status") != "complete":
        raise ValueError("Stage-3预测bundle不是complete")
    if set(map(str, marker.get("test_files", {}))) != set(tf_train.expected_farm_ids()):
        raise ValueError("Stage-3 marker未锁定完整5场站")
    # 完整验证依赖链，而非只相信marker文件名。
    for key, record in marker.get("files", {}).items():
        _validate_record(f"Stage-3 files.{key}", record)
    for farm_id, record in marker.get("test_files", {}).items():
        _validate_record(f"Stage-3 test.{farm_id}", record)

    frames, paths = {}, {}
    for key, filename in STAGE3_FORMAL_FILES.items():
        path = os.path.join(STAGE3_OUTPUT, filename)
        member = marker.get("files", {}).get(f"formal.{key}")
        if member is None:
            raise KeyError(f"Stage-3 marker缺少formal.{key}")
        if os.path.realpath(path) != os.path.realpath(
            _validate_record(f"formal.{key}", member)
        ):
            raise ValueError(f"Stage-3 formal.{key}路径不一致")
        frame = pd.read_csv(path, dtype={"farm_id": str})
        if "model_variant" not in frame:
            raise KeyError(f"Stage-3 {key}缺少model_variant")
        frame = frame[frame["model_variant"].astype(str) == "g0"].copy()
        frame["source_model_family"] = "controlled_gate_cali"
        frame["source_model_variant"] = "g0"
        frame["model_family"] = tf_train.MODEL_FAMILY
        frame["model_variant"] = "t0"
        if "model_name" in frame:
            frame["model_name"] = tf_train.variant_model_name("t0")
        frames[key], paths[key] = frame, path
    gate_predict._assert_exact_farm_metrics(
        frames["summary"], "T0 Stage-3 direct reference", ("capacity_normalized_rmse",)
    )
    frames["summary"]["result_source"] = (
        "direct_reference_stage3_g0_no_inference_no_retraining_no_archive_copy"
    )
    frames["summary"]["variant_label"] = tf_train.VARIANT_SPECS["t0"]["label"]
    frames["summary"]["adapter_trainable_parameter_count"] = 0
    frames["summary"]["candidate_training_elapsed_seconds"] = 0.0
    frames["summary"]["gate_training_elapsed_seconds"] = np.nan
    frames["summary"]["reference_only"] = True
    frames["summary"]["test_reuse_status"] = TEST_REUSE_STATUS
    return marker, frames, paths


def validate_training_bundle(required_variants):
    path = os.path.join(tf_train.RESULT_ROOT, tf_train.TRAINING_MARKER_NAME)
    if not required_variants:
        return None, None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少时频训练complete marker: {path}")
    with open(path, "r", encoding="utf-8") as file:
        marker = json.load(file)
    if marker.get("status") != "complete":
        raise ValueError("时频训练marker不是complete；partial模型不得进入正式测试")
    if marker.get("protocol_version") != tf_train.PROTOCOL_VERSION:
        raise ValueError("时频训练marker协议不匹配")
    if marker.get("architecture_version") != tf_train.ARCHITECTURE_VERSION:
        raise ValueError("时频训练marker架构版本不匹配")
    if set(map(str, marker.get("expected_farm_ids", ()))) != set(
        tf_train.expected_farm_ids()
    ):
        raise ValueError("时频训练marker场站集合不匹配")
    for key, record in marker.get("files", {}).items():
        _validate_record(f"training files.{key}", record)
    for variant in required_variants:
        for farm_id in tf_train.expected_farm_ids():
            for kind in ("model_path", "artifact_path"):
                if f"{variant}.{farm_id}.{kind}" not in marker.get("files", {}):
                    raise KeyError(f"训练marker缺少{variant}.{farm_id}.{kind}")
    return path, marker


def prediction_dirs(variant, output_root):
    root = os.path.join(output_root, variant, OUTPUT_SUBDIR)
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
    for value in values.values():
        os.makedirs(value, exist_ok=True)
    return values


def _load_model(variant, farm_id, marker):
    files = marker["files"]
    artifact_path = _validate_record(
        f"{variant}/{farm_id} artifact", files[f"{variant}.{farm_id}.artifact_path"]
    )
    artifact = joblib.load(artifact_path)
    calibration = artifact.get("candidate_calibration", {})
    q90 = np.asarray(calibration.get("candidate_difference_q90", ()), dtype=float)
    checks = {
        "variant": artifact.get("variant_id") == variant,
        "farm": str(artifact.get("farm_id")) == str(farm_id),
        "family": artifact.get("model_family") == tf_train.MODEL_FAMILY,
        "architecture": artifact.get("architecture_version")
        == tf_train.ARCHITECTURE_VERSION,
        "protocol": artifact.get("protocol_version") == tf_train.PROTOCOL_VERSION,
        "schema": int(artifact.get("artifact_schema_version", -1))
        == tf_train.ARTIFACT_SCHEMA_VERSION,
        "seed": int(artifact.get("random_seed", -1)) == tf_train.RANDOM_SEED,
        "history": int(artifact.get("history_len", -1)) == tf_train.HISTORY_LEN,
        "forecast": int(artifact.get("forecast_len", -1)) == tf_train.FORECAST_LEN,
        "params": int(artifact.get("total_params", -1))
        == tf_train.EXPECTED_TOTAL_PARAMS[variant],
        "candidate_weight_frozen": artifact.get("candidate_snapshot_before_gate_sha256")
        == artifact.get("candidate_snapshot_after_gate_sha256"),
        "candidate_output_frozen": artifact.get("candidate_output_before_gate_sha256")
        == artifact.get("candidate_output_after_gate_sha256"),
        "candidate_gate_drift_zero": float(
            artifact.get("candidate_gate_calibration_max_abs_drift", np.nan)
        )
        == 0.0,
        "train_only_soft_oracle": (
            q90.shape == (tf_train.FORECAST_LEN,)
            and np.isfinite(q90).all()
            and np.all(q90 >= 0.0)
            and "train" in str(calibration.get("scope", "")).lower()
            and int(calibration.get("sample_count", 0)) > 0
        ),
        "candidate_frozen_during_gate_calibration": bool(
            artifact.get("gate_training", {}).get("candidate_frozen_all_phases")
        ),
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError(f"{variant}/{farm_id} artifact不兼容: {failed}")
    model_path = _validate_record(
        f"{variant}/{farm_id} model", files[f"{variant}.{farm_id}.model_path"]
    )
    if os.path.realpath(model_path) != os.path.realpath(artifact.get("model_path", "")):
        raise ValueError(f"{variant}/{farm_id} model路径未被artifact锁定")
    if _sha256(model_path) != artifact.get("model_sha256"):
        raise ValueError(f"{variant}/{farm_id} model hash与artifact不一致")
    model = keras.models.load_model(
        model_path,
        custom_objects=tf_train.get_time_freq_custom_objects(),
        compile=False,
    )
    if int(model.count_params()) != tf_train.EXPECTED_TOTAL_PARAMS[variant]:
        raise ValueError(f"{variant}/{farm_id}加载后参数量漂移")
    if int(model.count_params()) >= tf_train.PARAMETER_LIMIT:
        raise ValueError(f"{variant}/{farm_id}超过30k参数上限")
    return artifact, artifact_path, model, model_path


def _normalize_output_shape(value, expected_shape, label):
    value = np.asarray(value, dtype=np.float64)
    if value.shape == (expected_shape[0], 1):
        value = np.repeat(value, expected_shape[1], axis=1)
    elif value.shape == (expected_shape[1],):
        value = np.repeat(value[None, :], expected_shape[0], axis=0)
    if value.shape != expected_shape or not np.isfinite(value).all():
        raise ValueError(f"{label} shape/finite异常: {value.shape} != {expected_shape}")
    return value


def predict_variant(variant, test_file, training_marker, max_samples=None):
    farm_id = str(common_predict.get_farm_id(test_file))
    artifact, artifact_path, model, model_path = _load_model(
        variant, farm_id, training_marker
    )
    df, features, actual_power, capacity = common_predict.prepare_prediction_arrays(
        test_file, artifact
    )
    history_len, forecast_len = (
        int(artifact["history_len"]),
        int(artifact["forecast_len"]),
    )
    if max_samples is not None:
        keep = history_len + forecast_len + int(max_samples) - 1
        df, features, actual_power = (
            df.iloc[:keep],
            features[:keep],
            actual_power[:keep],
        )
    dataset, n_samples = common_predict.make_prediction_dataset(
        features, history_len, forecast_len
    )
    diagnostic = tf_train.diagnostic_model(model)
    started = time.perf_counter()
    outputs = diagnostic.predict(dataset, verbose=common_predict.PREDICT_VERBOSE)
    elapsed = float(time.perf_counter() - started)
    if not isinstance(outputs, dict):
        raise TypeError(f"{variant}/{farm_id} diagnostic未返回dict")
    required = ("forecast", "persistence", "corrected", "gate", "q", "s")
    if any(key not in outputs for key in required):
        raise KeyError(
            f"{variant}/{farm_id} diagnostic缺少{set(required) - set(outputs)}"
        )
    expected_shape = (n_samples, forecast_len)
    outputs = {
        key: _normalize_output_shape(outputs[key], expected_shape, f"{variant}/{key}")
        for key in required
    }
    # 真值严格在模型前向完成后才构建。
    y_true = common_predict.build_truth_windows(
        actual_power, n_samples, history_len, forecast_len
    )
    payload = gate_predict._build_payload(
        variant, farm_id, df, artifact, outputs, y_true, capacity, history_len
    )
    payload.update(
        {
            "model_path": model_path,
            "model_sha256": _sha256(model_path),
            "artifact_path": artifact_path,
            "artifact_sha256": _sha256(artifact_path),
            "parameter_count": int(model.count_params()),
            "trainable_parameter_count": int(
                sum(int(np.prod(weight.shape)) for weight in model.trainable_weights)
            ),
            "reference_only": False,
            "result_source": "new_time_freq_model_single_test_forward",
            "diagnostic_source": "same_forward_as_forecast",
            "inference_elapsed_seconds": elapsed,
            "inference_milliseconds_per_sample": 1000.0 * elapsed / n_samples,
        }
    )
    return payload


def _candidate_metrics(payload):
    frames = []
    for candidate, values in (
        ("fused", payload["fused"]),
        ("persistence", payload["persistence"]),
        ("corrected", payload["corrected"]),
    ):
        frame = common_predict.metrics_by_horizon(
            tf_train.variant_model_name(payload["variant_id"]),
            payload["farm_id"],
            payload["y_true"],
            values,
            payload["capacity"],
            payload["forecast_len"],
        )
        frame["model_family"] = tf_train.MODEL_FAMILY
        frame["model_variant"] = payload["variant_id"]
        frame["candidate"] = candidate
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def save_payload(payload, output_root, skip_plots=False):
    variant, farm_id = payload["variant_id"], payload["farm_id"]
    dirs, name = (
        prediction_dirs(variant, output_root),
        tf_train.variant_model_name(variant),
    )
    prediction = common_predict.build_prediction_frame(
        name,
        payload["df"],
        farm_id,
        payload["fused"],
        payload["y_true"],
        payload["history_len"],
        payload["forecast_len"],
    )
    prediction_path = _atomic_csv(
        prediction,
        os.path.join(dirs["predictions"], f"{name}_predictions_farm_{farm_id}.csv"),
    )
    horizon = common_predict.metrics_by_horizon(
        name,
        farm_id,
        payload["y_true"],
        payload["fused"],
        payload["capacity"],
        payload["forecast_len"],
    )
    horizon["model_family"], horizon["model_variant"] = tf_train.MODEL_FAMILY, variant
    horizon_path = _atomic_csv(
        horizon,
        os.path.join(dirs["root"], f"{name}_metrics_by_horizon_farm_{farm_id}.csv"),
    )
    candidates = _candidate_metrics(payload)
    candidate_path = _atomic_csv(
        candidates,
        os.path.join(
            dirs["candidate_metrics"], f"{name}_candidate_metrics_farm_{farm_id}.csv"
        ),
    )
    regimes = gate_predict._regime_metrics(payload)
    regimes["model_family"], regimes["model_variant"] = tf_train.MODEL_FAMILY, variant
    regime_path = _atomic_csv(
        regimes,
        os.path.join(
            dirs["regime_metrics"], f"{name}_regime_metrics_farm_{farm_id}.csv"
        ),
    )
    assignments = gate_predict._assignment_frame(payload)
    assignments["model_family"], assignments["model_variant"] = (
        tf_train.MODEL_FAMILY,
        variant,
    )
    assignment_path = _atomic_csv(
        assignments,
        os.path.join(
            dirs["regime_assignments"], f"{name}_regime_assignments_farm_{farm_id}.csv"
        ),
    )
    safety = gate_predict.build_safety_scope_frame(payload)
    safety["model_family"], safety["model_variant"] = tf_train.MODEL_FAMILY, variant
    safety_path = _atomic_csv(
        safety,
        os.path.join(
            dirs["safety_diagnostics"], f"{name}_safety_by_scope_farm_{farm_id}.csv"
        ),
    )
    calibration = gate_predict.build_reliability_frame(payload)
    calibration["model_family"], calibration["model_variant"] = (
        tf_train.MODEL_FAMILY,
        variant,
    )
    calibration_path = _atomic_csv(
        calibration,
        os.path.join(dirs["calibration"], f"{name}_reliability_farm_{farm_id}.csv"),
    )
    gate_points = gate_predict.build_point_gate_frame(payload)
    gate_points["model_variant"] = variant
    gate_path = _atomic_csv(
        gate_points,
        os.path.join(
            dirs["gate_diagnostics"], f"{name}_gate_points_farm_{farm_id}.csv"
        ),
    )
    archive_path = _atomic_npz(
        os.path.join(
            dirs["candidate_archives"], f"{name}_candidate_archive_farm_{farm_id}.npz"
        ),
        schema_version=np.asarray("time_freq_candidate_archive_v1"),
        model_variant=np.asarray(variant),
        farm_id=np.asarray(farm_id),
        sample_id=payload["sample_id"],
        horizon_step=payload["horizon_step"],
        forecast_origin_time=payload["forecast_origin_time"],
        capacity=np.asarray(payload["capacity"]),
        y=payload["y_true"],
        P=payload["persistence"],
        C=payload["corrected"],
        F=payload["fused"],
        raw_gate=payload["raw_gate"],
        applied_gate=payload["applied_gate"],
        q=payload["q"],
        s=payload["s"],
        y_true=payload["y_true"],
        persistence=payload["persistence"],
        corrected=payload["corrected"],
        fused=payload["fused"],
        persistence_scaled=payload["persistence_scaled"],
        corrected_scaled=payload["corrected_scaled"],
        fused_scaled=payload["fused_scaled"],
    )
    single_path = single_figure = weighted_path = weighted_figure = None
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
    overall = horizon[horizon["horizon_step"].astype(str) == "all"].iloc[0]
    utility = safety[
        (safety["scope_type"] == "overall")
        & (safety["scope_value"].astype(str) == "all")
    ].iloc[0]
    summary = {**overall.to_dict()}
    summary.update(
        {
            key: utility[key]
            for key in utility.index
            if key
            not in {
                "model_family",
                "model_variant",
                "farm_id",
                "scope_type",
                "scope_value",
            }
        }
    )
    summary.update(
        {
            "model_family": tf_train.MODEL_FAMILY,
            "model_variant": variant,
            "variant_label": tf_train.VARIANT_SPECS[variant]["label"],
            "farm_id": farm_id,
            "feature_groups": tf_train.SOURCE_FEATURE_GROUPS,
            "feature_count": tf_train.SOURCE_FEATURE_COUNT,
            "parameter_count": payload["parameter_count"],
            "trainable_parameter_count": payload["trainable_parameter_count"],
            "adapter_trainable_parameter_count": tf_train.EXPECTED_ADAPTER_TRAINABLE_PARAMS[
                variant
            ],
            "candidate_training_elapsed_seconds": payload["artifact"].get(
                "candidate_training_elapsed_seconds"
            ),
            "gate_training_elapsed_seconds": payload["artifact"].get(
                "gate_training_elapsed_seconds"
            ),
            "inference_elapsed_seconds": payload["inference_elapsed_seconds"],
            "inference_milliseconds_per_sample": payload[
                "inference_milliseconds_per_sample"
            ],
            "reference_only": False,
            "selection_eligible": True,
            "result_source": payload["result_source"],
            "diagnostic_source": payload["diagnostic_source"],
            "test_reuse_status": TEST_REUSE_STATUS,
            "selection_split": "test",
            "test_used_for_selection": True,
            "test_is_final_blind_evaluation": False,
            "random_seed": tf_train.RANDOM_SEED,
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
            "safety_diagnostics_path": safety_path,
            "calibration_path": calibration_path,
            "gate_points_path": gate_path,
            "single_window_path": single_path,
            "single_window_figure_path": single_figure,
            "weighted_curve_path": weighted_path,
            "weighted_curve_figure_path": weighted_figure,
            "fusion_reconstruction_max_abs_error_scaled": payload[
                "fusion_reconstruction_max_abs_error_scaled"
            ],
            **{f"weighted_{key}": value for key, value in weighted_metrics.items()},
        }
    )
    paths = {
        "prediction": prediction_path,
        "horizon": horizon_path,
        "candidate": candidate_path,
        "regime": regime_path,
        "assignment": assignment_path,
        "safety": safety_path,
        "calibration": calibration_path,
        "gate": gate_path,
        "archive": archive_path,
        "single_window": single_path,
        "single_figure": single_figure,
        "weighted_curve": weighted_path,
        "weighted_figure": weighted_figure,
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


def _concat(reference, results, key):
    values = [reference[key]] + [item[key] for item in results]
    return pd.concat(values, ignore_index=True, sort=False)


def validate_complete_output_matrix(frames):
    """Reject any silently incomplete per-farm/per-diagnostic formal output."""
    expected_variants = set(ALL_VARIANTS)
    expected_farms = set(tf_train.expected_farm_ids())
    natural_keys = {
        "summary": ("model_variant", "farm_id"),
        "horizon": ("model_variant", "farm_id", "horizon_step"),
        "candidate": (
            "model_variant",
            "farm_id",
            "candidate",
            "horizon_step",
        ),
        "regime": (
            "model_variant",
            "farm_id",
            "regime_group",
            "candidate",
            "horizon_step",
        ),
        "assignments": ("model_variant", "farm_id", "sample_id"),
        "safety": (
            "model_variant",
            "farm_id",
            "scope_type",
            "scope_value",
        ),
        "calibration": ("model_variant", "farm_id", "gate_bin"),
    }

    def canonical_key(value):
        if pd.isna(value):
            return "<NA>"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        return str(value)

    for key, frame in frames.items():
        if key not in natural_keys:
            raise KeyError(f"正式输出存在未声明自然键的表: {key}")
        missing = set(natural_keys[key]) - set(frame.columns)
        if missing:
            raise KeyError(f"正式{key}缺少自然键列: {sorted(missing)}")
        if set(frame["model_variant"].astype(str)) != expected_variants:
            raise ValueError(f"正式{key}未覆盖T0/M0/T1--T3")
        if set(frame["farm_id"].astype(str)) != expected_farms:
            raise ValueError(f"正式{key}未覆盖5个固定场站")

        normalized = frame.copy()
        for column in natural_keys[key]:
            normalized[column] = normalized[column].map(canonical_key)
        if normalized.duplicated(list(natural_keys[key])).any():
            duplicates = normalized.loc[
                normalized.duplicated(list(natural_keys[key]), keep=False),
                list(natural_keys[key]),
            ].head(5)
            raise ValueError(
                f"正式{key}自然键重复: {duplicates.to_dict(orient='records')}"
            )

        baseline_counts = (
            normalized[normalized["model_variant"] == "t0"]
            .groupby("farm_id")
            .size()
            .reindex(sorted(expected_farms))
        )
        if baseline_counts.isna().any() or (baseline_counts <= 0).any():
            raise ValueError(f"正式{key} T0存在缺失场站")
        for variant in ALL_VARIANTS:
            counts = (
                normalized[normalized["model_variant"] == variant]
                .groupby("farm_id")
                .size()
                .reindex(sorted(expected_farms))
            )
            if counts.isna().any() or not np.array_equal(
                counts.to_numpy(dtype=np.int64),
                baseline_counts.to_numpy(dtype=np.int64),
            ):
                raise ValueError(f"正式{key}/{variant}逐场行数与只读T0基准不一致")

        # 行数相同仍可能是“重复一个键、遗漏另一个键”。逐场比较去掉
        # model_variant/farm_id 后的键集合，确保所有变体和 T0 是同一批窗口/分组。
        suffix = [
            column
            for column in natural_keys[key]
            if column not in ("model_variant", "farm_id")
        ]
        if suffix:
            for farm_id in sorted(expected_farms):
                baseline = normalized[
                    (normalized["model_variant"] == "t0")
                    & (normalized["farm_id"] == farm_id)
                ]
                baseline_keys = set(baseline[suffix].itertuples(index=False, name=None))
                for variant in ALL_VARIANTS:
                    target = normalized[
                        (normalized["model_variant"] == variant)
                        & (normalized["farm_id"] == farm_id)
                    ]
                    target_keys = set(target[suffix].itertuples(index=False, name=None))
                    if target_keys != baseline_keys:
                        missing_keys = list(baseline_keys - target_keys)[:5]
                        extra_keys = list(target_keys - baseline_keys)[:5]
                        raise ValueError(
                            f"正式{key}/{variant}/{farm_id}自然键集合与T0不一致; "
                            f"missing={missing_keys}, extra={extra_keys}"
                        )
    summary = frames["summary"]
    if (
        len(summary) != len(ALL_VARIANTS) * len(expected_farms)
        or summary.duplicated(["model_variant", "farm_id"]).any()
    ):
        raise ValueError("正式summary不是5变体×5场站唯一矩阵")


def _exact_five(frame, label, columns):
    gate_predict._assert_exact_farm_metrics(frame, label, columns)
    return frame


def _macro_metric(frame, variant, column, **filters):
    part = frame[frame["model_variant"].astype(str) == variant]
    for key, value in filters.items():
        part = part[part[key].astype(str) == str(value)]
    part = _exact_five(part, f"{variant}/{filters}", (column,))
    return float(pd.to_numeric(part[column]).mean())


def build_comparison(summary, candidate, regime):
    rows = []
    for variant in ALL_VARIANTS:
        frame = _exact_five(
            summary[summary["model_variant"] == variant],
            variant,
            (
                "capacity_normalized_rmse",
                "positive_regret_mean",
                "harm_rate_0_005",
                "oracle_brier",
                "ece_10bin",
                "parameter_count",
                "inference_milliseconds_per_sample",
            ),
        )
        row = {
            "model_variant": variant,
            "variant_label": tf_train.VARIANT_SPECS[variant]["label"],
            "macro_test_nrmse": float(frame["capacity_normalized_rmse"].mean()),
            "macro_test_nmae": float(frame["capacity_normalized_mae"].mean()),
            "macro_positive_regret_mean": float(frame["positive_regret_mean"].mean()),
            "macro_harm_rate_0_005": float(frame["harm_rate_0_005"].mean()),
            "macro_oracle_brier": float(frame["oracle_brier"].mean()),
            "macro_ece_10bin": float(frame["ece_10bin"].mean()),
            "parameter_count_max": int(frame["parameter_count"].max()),
            "macro_inference_milliseconds_per_sample": float(
                frame["inference_milliseconds_per_sample"].mean()
            ),
            "corrected_overall_nrmse": _macro_metric(
                candidate,
                variant,
                "capacity_normalized_rmse",
                candidate="corrected",
                horizon_step="all",
            ),
            "corrected_overall_nmae": _macro_metric(
                candidate,
                variant,
                "capacity_normalized_mae",
                candidate="corrected",
                horizon_step="all",
            ),
        }
        for group in ("dynamic", "ramp_up", "ramp_down"):
            row[f"corrected_{group}_nrmse"] = _macro_metric(
                regime,
                variant,
                "capacity_normalized_rmse",
                candidate="corrected",
                regime_group=group,
                horizon_step="all",
            )
            row[f"fused_{group}_nrmse"] = _macro_metric(
                regime,
                variant,
                "capacity_normalized_rmse",
                candidate="fused",
                regime_group=group,
                horizon_step="all",
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    base = result[result["model_variant"] == "t0"].iloc[0]
    matched_control = result[result["model_variant"] == "m0"].iloc[0]
    result["relative_macro_nrmse_vs_t0"] = (
        result["macro_test_nrmse"] / float(base["macro_test_nrmse"]) - 1.0
    )
    result["relative_macro_nrmse_vs_m0"] = (
        result["macro_test_nrmse"] / float(matched_control["macro_test_nrmse"]) - 1.0
    )
    result["relative_corrected_nrmse_vs_m0"] = (
        result["corrected_overall_nrmse"]
        / float(matched_control["corrected_overall_nrmse"])
        - 1.0
    )
    farm_base = summary[summary["model_variant"] == "t0"].set_index("farm_id")[
        "capacity_normalized_rmse"
    ]
    flags = []
    for _, row in result.iterrows():
        variant = row["model_variant"]
        farm = summary[summary["model_variant"] == variant].set_index("farm_id")[
            "capacity_normalized_rmse"
        ]
        farm = farm.reindex(farm_base.index)
        within = int((farm <= farm_base * (1.0 + FARM_NRMSE_TOL)).sum())
        accuracy = row["macro_test_nrmse"] <= base["macro_test_nrmse"] * (
            1.0 + MACRO_NRMSE_TOL
        )
        ramp = all(
            row[f"fused_{group}_nrmse"]
            <= base[f"fused_{group}_nrmse"] * (1.0 + RAMP_NRMSE_TOL)
            for group in ("ramp_up", "ramp_down")
        )
        candidate_overall = all(
            row[column] <= base[column] * (1.0 + CANDIDATE_OVERALL_TOL)
            for column in ("corrected_overall_nrmse", "corrected_overall_nmae")
        )
        candidate_regime = all(
            row[f"corrected_{group}_nrmse"]
            <= base[f"corrected_{group}_nrmse"] * (1.0 + CANDIDATE_REGIME_TOL)
            for group in ("dynamic", "ramp_up", "ramp_down")
        )
        safety = (
            row["macro_positive_regret_mean"]
            <= base["macro_positive_regret_mean"] * (1.0 + SAFETY_REGRET_TOL)
            and row["macro_harm_rate_0_005"]
            <= base["macro_harm_rate_0_005"] + SAFETY_HARM_ABS_TOL
        )
        parameter = row["parameter_count_max"] < tf_train.PARAMETER_LIMIT
        guard = bool(
            accuracy
            and within >= MIN_FARMS_WITHIN_GUARD
            and ramp
            and candidate_overall
            and candidate_regime
            and safety
            and parameter
        )
        if variant == "t0":
            guard = True
        flags.append(
            {
                "model_variant": variant,
                "macro_accuracy_guard_pass": bool(accuracy),
                "farms_within_1pct_vs_t0": within,
                "farm_guard_pass": within >= MIN_FARMS_WITHIN_GUARD,
                "ramp_guard_pass": bool(ramp),
                "candidate_overall_guard_pass": bool(candidate_overall),
                "candidate_dynamic_ramp_guard_pass": bool(candidate_regime),
                "safety_guard_pass": bool(safety),
                "parameter_under_30k": bool(parameter),
                "selection_guard_pass": guard,
            }
        )
    return result.merge(pd.DataFrame(flags), on="model_variant", validate="one_to_one")


def select_model(comparison):
    candidates = comparison[comparison["selection_guard_pass"]].copy()
    nonreference = candidates[candidates["model_variant"] != "t0"]
    status = "test_guarded_selection"
    if nonreference.empty:
        candidates = candidates[candidates["model_variant"] == "t0"]
        status = "fallback_t0_no_new_variant_passed_guards"
    best = float(candidates["macro_test_nrmse"].min())
    near = candidates[candidates["macro_test_nrmse"] <= best * (1.0 + NRMSE_TIE_TOL)]
    selected = near.sort_values(
        [
            "macro_positive_regret_mean",
            "macro_fixed_t0_oracle_brier",
            "parameter_count_max",
            "macro_inference_milliseconds_per_sample",
            "macro_test_nrmse",
        ],
        kind="stable",
    ).iloc[0]
    comparison = comparison.copy()
    comparison["selected"] = comparison["model_variant"] == selected["model_variant"]
    comparison["selection_status"] = status
    return selected, comparison


def build_joint_complementarity(candidate):
    """Descriptive M0-controlled complementarity, not a causal factorial effect.

    T0 is deliberately excluded because it uses a different, non-factorized gate.
    T3 also has a dedicated joint head rather than an exactly additive T1+T2 head,
    so this contrast must not be interpreted as an identifiable interaction.
    """
    frame = candidate[(candidate["candidate"] == "corrected")].copy()
    frame["horizon_key"] = frame["horizon_step"].astype(str)
    pivot = frame.pivot(
        index=["farm_id", "horizon_key"],
        columns="model_variant",
        values="capacity_normalized_rmse",
    )
    required = {"m0", "t1", "t2", "t3"}
    if not required.issubset(pivot.columns):
        raise ValueError("时频interaction报告缺少M0/T1/T2/T3")
    result = pivot.reset_index()
    result["joint_complementarity_contrast_nrmse"] = (
        result["t1"] + result["t2"] - result["m0"] - result["t3"]
    )
    result["positive_descriptive_complementarity"] = (
        result["joint_complementarity_contrast_nrmse"] > 0
    )
    result["control_baseline"] = "m0_same_factorized_calibrated_safe_gate"
    result["interpretation_scope"] = (
        "descriptive_joint_complementarity_not_identifiable_factorial_interaction"
    )
    return result


def build_candidate_drift(t0_summary, results):
    """Compare every new candidate pair with the immutable T0 source archive."""
    source_by_farm = {}
    for _, row in t0_summary.iterrows():
        path = row.get("candidate_archive_path")
        if not isinstance(path, str) or not os.path.isfile(path):
            raise FileNotFoundError(f"T0/{row['farm_id']}缺少Stage-3 source archive")
        if _sha256(path) != row.get("candidate_archive_sha256"):
            raise ValueError(f"T0/{row['farm_id']} source archive hash与summary不一致")
        with np.load(path, allow_pickle=False) as archive:
            source_by_farm[str(row["farm_id"])] = {
                key: np.asarray(archive[key])
                for key in (
                    "sample_id",
                    "horizon_step",
                    "forecast_origin_time",
                    "y_true",
                    "persistence_scaled",
                    "corrected_scaled",
                    "persistence",
                    "corrected",
                    "raw_gate",
                    "capacity",
                )
            }
    rows = []
    for result in results:
        target = result["payload"]
        farm_id = target["farm_id"]
        source = source_by_farm[farm_id]
        for key in ("sample_id", "horizon_step", "forecast_origin_time"):
            if not np.array_equal(source[key], target[key]):
                raise ValueError(f"T0->{target['variant_id']}/{farm_id} {key}未对齐")
        if not np.array_equal(source["y_true"], target["y_true"], equal_nan=True):
            raise ValueError(f"T0->{target['variant_id']}/{farm_id}真值未对齐")
        source_capacity = float(np.asarray(source["capacity"]).reshape(()))
        target_capacity = float(np.asarray(target["capacity"]).reshape(()))
        if (
            not np.isfinite(source_capacity)
            or source_capacity <= 0.0
            or not np.isclose(source_capacity, target_capacity, rtol=0.0, atol=1e-12)
        ):
            raise ValueError(
                f"T0->{target['variant_id']}/{farm_id}容量不一致或非法: "
                f"source={source_capacity}, target={target_capacity}"
            )

        persistence_scaled_exact = np.array_equal(
            source["persistence_scaled"],
            target["persistence_scaled"],
            equal_nan=True,
        )
        persistence_physical_exact = np.array_equal(
            source["persistence"], target["persistence"], equal_nan=True
        )
        corrected_scaled_exact = np.array_equal(
            source["corrected_scaled"],
            target["corrected_scaled"],
            equal_nan=True,
        )
        corrected_physical_exact = np.array_equal(
            source["corrected"], target["corrected"], equal_nan=True
        )
        persistence_finite_mask_exact = np.array_equal(
            np.isfinite(source["persistence"]),
            np.isfinite(target["persistence"]),
        )
        corrected_finite_mask_exact = np.array_equal(
            np.isfinite(source["corrected"]),
            np.isfinite(target["corrected"]),
        )
        scopes = [("all", np.ones_like(target["y_true"], dtype=bool))]
        for horizon in range(target["forecast_len"]):
            mask = np.zeros_like(target["y_true"], dtype=bool)
            mask[:, horizon] = True
            scopes.append((horizon + 1, mask))
        for horizon_step, valid in scopes:
            for value in (
                source["y_true"],
                source["persistence"],
                source["corrected"],
                target["persistence"],
                target["corrected"],
                target["raw_gate"],
            ):
                valid &= np.isfinite(value)
            if not valid.any():
                raise ValueError(
                    f"T0->{target['variant_id']}/{farm_id}/{horizon_step}无有效样本"
                )
            source_oracle = np.abs(
                source["corrected"][valid] - source["y_true"][valid]
            ) < np.abs(source["persistence"][valid] - source["y_true"][valid])
            target_oracle = np.abs(
                target["corrected"][valid] - target["y_true"][valid]
            ) < np.abs(target["persistence"][valid] - target["y_true"][valid])
            gate = target["raw_gate"][valid]
            c_scaled_drift = (
                target["corrected_scaled"][valid] - source["corrected_scaled"][valid]
            )
            p_scaled_drift = (
                target["persistence_scaled"][valid]
                - source["persistence_scaled"][valid]
            )
            c_physical_drift = target["corrected"][valid] - source["corrected"][valid]
            p_physical_drift = (
                target["persistence"][valid] - source["persistence"][valid]
            )
            c_normalized_abs = np.abs(c_physical_drift) / source_capacity
            p_normalized_abs = np.abs(p_physical_drift) / source_capacity
            persistence_control_pass = bool(
                persistence_scaled_exact
                and persistence_finite_mask_exact
                and np.max(p_normalized_abs) <= PERSISTENCE_CONTROL_MAX_NORM_TOL
            )
            m0_corrected_control_pass = bool(
                corrected_finite_mask_exact
                and np.max(c_normalized_abs) <= M0_CORRECTED_MAX_NORM_TOL
                and np.mean(c_normalized_abs) <= M0_CORRECTED_MEAN_NORM_TOL
            )
            rows.append(
                {
                    "baseline_variant": "t0",
                    "target_variant": target["variant_id"],
                    "farm_id": farm_id,
                    "horizon_step": horizon_step,
                    "valid_count": int(valid.sum()),
                    "candidate_pair_exact_full_archive": bool(
                        persistence_scaled_exact and corrected_scaled_exact
                    ),
                    "persistence_scaled_exact_full_archive": bool(
                        persistence_scaled_exact
                    ),
                    "persistence_physical_exact_full_archive": bool(
                        persistence_physical_exact
                    ),
                    "persistence_finite_mask_exact_full_archive": bool(
                        persistence_finite_mask_exact
                    ),
                    "corrected_scaled_exact_full_archive": bool(corrected_scaled_exact),
                    "corrected_physical_exact_full_archive": bool(
                        corrected_physical_exact
                    ),
                    "corrected_finite_mask_exact_full_archive": bool(
                        corrected_finite_mask_exact
                    ),
                    "persistence_scaled_max_abs_drift": float(
                        np.max(np.abs(p_scaled_drift))
                    ),
                    "corrected_scaled_max_abs_drift": float(
                        np.max(np.abs(c_scaled_drift))
                    ),
                    "persistence_capacity_normalized_max_abs_drift": float(
                        np.max(p_normalized_abs)
                    ),
                    "persistence_capacity_normalized_mean_abs_drift": float(
                        np.mean(p_normalized_abs)
                    ),
                    "persistence_capacity_normalized_p999_abs_drift": float(
                        np.quantile(p_normalized_abs, 0.999)
                    ),
                    "corrected_capacity_normalized_max_abs_drift": float(
                        np.max(c_normalized_abs)
                    ),
                    "corrected_capacity_normalized_mean_abs_drift": float(
                        np.mean(c_normalized_abs)
                    ),
                    "corrected_capacity_normalized_p999_abs_drift": float(
                        np.quantile(c_normalized_abs, 0.999)
                    ),
                    "corrected_normalized_rmse_drift": float(
                        np.sqrt(np.mean(np.square(c_physical_drift))) / source_capacity
                    ),
                    "persistence_control_max_norm_tolerance": (
                        PERSISTENCE_CONTROL_MAX_NORM_TOL
                    ),
                    "persistence_control_pass": persistence_control_pass,
                    "m0_corrected_control_required": (target["variant_id"] == "m0"),
                    "m0_corrected_max_norm_tolerance": M0_CORRECTED_MAX_NORM_TOL,
                    "m0_corrected_mean_norm_tolerance": M0_CORRECTED_MEAN_NORM_TOL,
                    "m0_corrected_control_pass": (
                        m0_corrected_control_pass
                        if target["variant_id"] == "m0"
                        else np.nan
                    ),
                    "baseline_oracle_prevalence": float(source_oracle.mean()),
                    "target_oracle_prevalence": float(target_oracle.mean()),
                    "oracle_label_agreement": float(
                        np.mean(source_oracle == target_oracle)
                    ),
                    "fixed_t0_oracle_brier": float(
                        np.mean(np.square(gate - source_oracle.astype(float)))
                    ),
                    "fixed_t0_oracle_ece_10bin": gate_predict._ece(
                        gate, source_oracle, bins=10, adaptive=False
                    ),
                    "calibration_comparison_scope": "fixed_t0_oracle_descriptive_candidate_drift_control",
                }
            )
    frame = pd.DataFrame(rows)
    expected = (
        len(NEW_VARIANTS)
        * len(tf_train.expected_farm_ids())
        * (tf_train.FORECAST_LEN + 1)
    )
    if (
        len(frame) != expected
        or frame.duplicated(["target_variant", "farm_id", "horizon_step"]).any()
    ):
        raise ValueError("candidate drift不是4x5x17唯一矩阵")
    return frame


def validate_candidate_control_invariants(candidate_drift):
    """Fail before selection if the immutable candidate controls have drifted."""
    overall = candidate_drift[
        candidate_drift["horizon_step"].astype(str) == "all"
    ].copy()
    expected_rows = len(NEW_VARIANTS) * len(tf_train.expected_farm_ids())
    if (
        len(overall) != expected_rows
        or overall.duplicated(["target_variant", "farm_id"]).any()
    ):
        raise ValueError("candidate drift overall不是4变体×5场站唯一矩阵")
    if set(overall["target_variant"].astype(str)) != set(NEW_VARIANTS):
        raise ValueError("candidate drift overall缺少M0/T1/T2/T3")
    if set(overall["farm_id"].astype(str)) != set(tf_train.expected_farm_ids()):
        raise ValueError("candidate drift overall缺少固定5场站")

    persistence_failures = overall[~overall["persistence_control_pass"].astype(bool)]
    if not persistence_failures.empty:
        details = persistence_failures[
            [
                "target_variant",
                "farm_id",
                "persistence_scaled_exact_full_archive",
                "persistence_capacity_normalized_max_abs_drift",
            ]
        ].to_dict(orient="records")
        raise ValueError(
            f"Persistence候选不再与只读T0逐点一致，禁止进入测试集选型: {details}"
        )

    m0 = overall[overall["target_variant"].astype(str) == "m0"]
    if len(m0) != len(tf_train.expected_farm_ids()):
        raise ValueError("M0 corrected控制校验未覆盖固定5场站")
    m0_pass = m0["m0_corrected_control_pass"].fillna(False).astype(bool)
    if not m0_pass.all():
        details = m0.loc[
            ~m0_pass,
            [
                "farm_id",
                "corrected_capacity_normalized_max_abs_drift",
                "corrected_capacity_normalized_mean_abs_drift",
                "corrected_capacity_normalized_p999_abs_drift",
            ],
        ].to_dict(orient="records")
        raise ValueError(
            "M0 corrected未在Stage-3跨运行时容量容差内复现F7，"
            f"禁止作为门控匹配控制组: {details}"
        )
    return {
        "persistence_control_pass": True,
        "m0_corrected_control_pass": True,
        "persistence_max_norm_tolerance": PERSISTENCE_CONTROL_MAX_NORM_TOL,
        "m0_corrected_max_norm_tolerance": M0_CORRECTED_MAX_NORM_TOL,
        "m0_corrected_mean_norm_tolerance": M0_CORRECTED_MEAN_NORM_TOL,
    }


def attach_fixed_t0_calibration(comparison, candidate_drift):
    overall = candidate_drift[candidate_drift["horizon_step"].astype(str) == "all"]
    rows = []
    for variant in NEW_VARIANTS:
        frame = _exact_five(
            overall[overall["target_variant"] == variant],
            f"fixed T0 oracle/{variant}",
            (
                "fixed_t0_oracle_brier",
                "fixed_t0_oracle_ece_10bin",
                "oracle_label_agreement",
            ),
        )
        rows.append(
            {
                "model_variant": variant,
                "macro_fixed_t0_oracle_brier": float(
                    frame["fixed_t0_oracle_brier"].mean()
                ),
                "macro_fixed_t0_oracle_ece_10bin": float(
                    frame["fixed_t0_oracle_ece_10bin"].mean()
                ),
                "macro_oracle_label_agreement_vs_t0": float(
                    frame["oracle_label_agreement"].mean()
                ),
            }
        )
    t0 = comparison[comparison["model_variant"] == "t0"].iloc[0]
    rows.append(
        {
            "model_variant": "t0",
            "macro_fixed_t0_oracle_brier": float(t0["macro_oracle_brier"]),
            "macro_fixed_t0_oracle_ece_10bin": float(t0["macro_ece_10bin"]),
            "macro_oracle_label_agreement_vs_t0": 1.0,
        }
    )
    return comparison.merge(
        pd.DataFrame(rows), on="model_variant", validate="one_to_one"
    )


def save_aggregate_figures(comparison, summary, horizon, complementarity, output_dir):
    """Save compact paper-facing aggregate views; metrics remain in CSV files."""
    figure_dir = os.path.join(output_dir, "figures")
    cache_dir = os.path.join(output_dir, "matplotlib_cache")
    os.makedirs(figure_dir, exist_ok=True)
    plt = common_predict.setup_matplotlib({"matplotlib_cache": cache_dir})
    paths = {}

    ordered = comparison.sort_values("macro_test_nrmse", kind="stable")
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    colors = ["#d62728" if bool(value) else "#4c78a8" for value in ordered["selected"]]
    ax.bar(ordered["model_variant"], ordered["macro_test_nrmse"], color=colors)
    ax.set_ylabel("Five-farm macro NRMSE")
    ax.set_xlabel("Variant")
    ax.set_title("T0/M0/T1--T3 test-set ranking")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths["rank_figure"] = os.path.join(
        figure_dir, "time_freq_model_test_nrmse_rank.png"
    )
    fig.savefig(paths["rank_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    farm_matrix = summary.pivot(
        index="model_variant", columns="farm_id", values="capacity_normalized_rmse"
    ).reindex(index=ALL_VARIANTS, columns=tf_train.expected_farm_ids())
    if farm_matrix.isna().any().any():
        raise ValueError("场站NRMSE热力图矩阵不完整")
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    image = ax.imshow(farm_matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(farm_matrix.index)), labels=farm_matrix.index)
    ax.set_xticks(
        np.arange(len(farm_matrix.columns)),
        labels=[str(value)[-4:] for value in farm_matrix.columns],
    )
    ax.set_xlabel("Farm ID (last 4 digits)")
    ax.set_ylabel("Variant")
    ax.set_title("Capacity-normalized RMSE by farm")
    fig.colorbar(image, ax=ax, label="NRMSE")
    fig.tight_layout()
    paths["farm_heatmap_figure"] = os.path.join(
        figure_dir, "time_freq_model_test_farm_heatmap.png"
    )
    fig.savefig(paths["farm_heatmap_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    numeric_horizon = horizon[horizon["horizon_step"].astype(str) != "all"].copy()
    numeric_horizon["horizon_step_numeric"] = pd.to_numeric(
        numeric_horizon["horizon_step"], errors="raise"
    )
    macro_horizon = (
        numeric_horizon.groupby(
            ["model_variant", "horizon_step_numeric"], as_index=False
        )["capacity_normalized_rmse"]
        .mean()
        .sort_values(["model_variant", "horizon_step_numeric"])
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for variant in ALL_VARIANTS:
        part = macro_horizon[macro_horizon["model_variant"] == variant]
        if len(part) != tf_train.FORECAST_LEN:
            raise ValueError(f"{variant}逐horizon宏平均不完整")
        ax.plot(
            part["horizon_step_numeric"],
            part["capacity_normalized_rmse"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=variant.upper(),
        )
    ax.set_xlabel("Forecast horizon (15-min steps)")
    ax.set_ylabel("Five-farm macro NRMSE")
    ax.set_title("Horizon-wise test error")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    paths["horizon_figure"] = os.path.join(
        figure_dir, "time_freq_model_test_horizon_nrmse.png"
    )
    fig.savefig(paths["horizon_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    complement = complementarity[
        complementarity["horizon_key"].astype(str) != "all"
    ].copy()
    complement["horizon_step_numeric"] = pd.to_numeric(
        complement["horizon_key"], errors="raise"
    )
    complement = complement.groupby("horizon_step_numeric", as_index=False)[
        "joint_complementarity_contrast_nrmse"
    ].mean()
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.plot(
        complement["horizon_step_numeric"],
        complement["joint_complementarity_contrast_nrmse"],
        marker="o",
        color="#59a14f",
    )
    ax.set_xlabel("Forecast horizon (15-min steps)")
    ax.set_ylabel("T1 + T2 - M0 - T3 NRMSE")
    ax.set_title("Descriptive joint complementarity (not factorial causality)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    paths["complementarity_figure"] = os.path.join(
        figure_dir, "time_freq_model_test_joint_complementarity.png"
    )
    fig.savefig(paths["complementarity_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def write_reports(comparison, selected, complementarity, output_dir):
    compact = [
        "model_variant",
        "macro_test_nrmse",
        "relative_macro_nrmse_vs_t0",
        "relative_macro_nrmse_vs_m0",
        "corrected_overall_nrmse",
        "relative_corrected_nrmse_vs_m0",
        "fused_dynamic_nrmse",
        "fused_ramp_up_nrmse",
        "fused_ramp_down_nrmse",
        "macro_fixed_t0_oracle_brier",
        "macro_oracle_label_agreement_vs_t0",
        "parameter_count_max",
        "selection_guard_pass",
        "selected",
    ]
    cn = [
        "# 最小Residual与T0--T3时频矩阵：测试集最终选型",
        "",
        f"最终选中 **{selected['model_variant']}**，5场站等权宏平均NRMSE=`{selected['macro_test_nrmse']:.9f}`。",
        "",
        "本轮按既定要求在当前测试集筛选；该测试段已参与此前实验，属于legacy-seen探索性选择，不是最终盲测。",
        "",
        "## 结果矩阵",
        "",
        comparison[compact].to_markdown(index=False),
        "",
        "## 协议说明",
        "",
        "- T0从Stage-3 complete bundle的G0聚合与逐场文件只读引用；未重新训练、未执行forward、未复制候选archive。",
        "- M0/T1/T2/T3使用同一P+H+D工况输入、同一因子化校准安全门控；每次candidate改变后均使用训练集重建soft oracle与Q90。",
        "- 选型强制恰好覆盖5场站且关键macro为有限值；先过宏精度、逐场、ramp、candidate、安全和30k参数守门，再以NRMSE为主并按后悔/Brier/复杂度打破0.1%平局。",
        "- 联合互补性对照定义为T1+T2-M0-T3（corrected NRMSE）；M0与T1--T3使用统一门控。T0门控不同，禁止用于该对照。",
        "- T3并非严格复用T1/T2两个主效应头的2×2可识别结构，因此该量仅作描述性joint complementarity，不宣称factorial causal interaction。",
        "",
        "## 时频联合互补性摘要",
        "",
        complementarity.groupby("horizon_key", as_index=False)[
            "joint_complementarity_contrast_nrmse"
        ]
        .mean()
        .to_markdown(index=False),
        "",
    ]
    en = [
        "# Minimal-residual T0--T3 time-frequency matrix: test selection",
        "",
        f"Selected **{selected['model_variant']}** with five-farm macro NRMSE `{selected['macro_test_nrmse']:.9f}`.",
        "",
        "This is a legacy-seen test-selected exploratory result, not a final blind evaluation.",
        "",
        comparison[compact].to_markdown(index=False),
        "",
        "T0 is a hash-validated read-only reference to Stage-3 G0: no retraining, inference, model copy, or archive copy was performed.",
        "The T1+T2-M0-T3 contrast uses the gate-matched M0 control and is reported only as descriptive joint complementarity, not as an identifiable factorial/causal interaction.",
        "All five farms must be present with finite selection metrics. Accuracy, farm, ramp, candidate, safety, and parameter guards precede the NRMSE ranking.",
    ]
    return {
        "report_cn": _atomic_text(
            "\n".join(cn),
            os.path.join(output_dir, "time_freq_model_test_final_selection.md"),
        ),
        "report_en": _atomic_text(
            "\n".join(en),
            os.path.join(output_dir, "time_freq_model_test_final_selection_en.md"),
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default=os.getenv("WIND_TIME_FREQ_PREDICT_VARIANTS", ",".join(ALL_VARIANTS)),
        help="逗号分隔: t0,m0,t1,t2,t3",
    )
    parser.add_argument(
        "--farms",
        default=os.getenv("WIND_TIME_FREQ_FARMS", ""),
        help="逗号分隔场站ID；空值为全部",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="隔离输出、默认1场站/32窗口，不发布正式marker",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="仅partial/smoke允许限制测试窗口"
    )
    parser.add_argument("--skip-plots", action="store_true", help="跳过逐场可视化")
    parser.add_argument("--run-id", default=None, help="partial输出标识")
    return parser.parse_args(argv)


def _parse_list(raw, allowed, label):
    values = list(
        dict.fromkeys(item.strip().lower() for item in raw.split(",") if item.strip())
    )
    invalid = set(values) - set(allowed)
    if invalid or not values:
        raise ValueError(f"{label}非法: {sorted(invalid)}")
    return values


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("PYTHONHASHSEED", str(tf_train.RANDOM_SEED))
    keras.utils.set_random_seed(tf_train.RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    variants = _parse_list(args.variants, ALL_VARIANTS, "variants")
    expected_farms = list(tf_train.expected_farm_ids())
    farms = (
        _parse_list(args.farms, expected_farms, "farms")
        if args.farms
        else expected_farms
    )
    if args.smoke:
        if args.variants == ",".join(ALL_VARIANTS):
            variants = ["t3"]
        if not args.farms:
            farms = expected_farms[:1]
        args.max_samples = args.max_samples or 32
    full = (
        set(variants) == set(ALL_VARIANTS)
        and set(farms) == set(expected_farms)
        and not args.smoke
        and args.max_samples is None
    )
    if args.max_samples is not None and not args.smoke:
        raise ValueError("--max-samples只允许与--smoke一起使用")
    output_root = (
        tf_train.RESULT_ROOT
        if full
        else os.path.join(
            tf_train.RESULT_ROOT,
            "partial_runs",
            args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    )
    output_dir = os.path.join(output_root, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)
    formal_marker = os.path.join(
        tf_train.RESULT_ROOT, OUTPUT_SUBDIR, FORMAL_MARKER_NAME
    )
    if full and os.path.exists(formal_marker):
        os.remove(formal_marker)

    stage3_marker, t0_frames, source_paths = validate_stage3_bundle()
    training_marker_path, training_marker = validate_training_bundle(
        [v for v in variants if v in NEW_VARIANTS]
    )
    source_test = {
        str(farm): record["path"]
        for farm, record in stage3_marker["test_files"].items()
    }
    test_files = [source_test[farm] for farm in farms]
    results = []
    for test_file in test_files:
        farm_id = str(common_predict.get_farm_id(test_file))
        for variant in variants:
            if variant == "t0":
                continue
            print(f"\n===== Time-Frequency预测 variant={variant} farm={farm_id} =====")
            payload = predict_variant(
                variant, test_file, training_marker, args.max_samples
            )
            results.append(save_payload(payload, output_root, args.skip_plots))

    # Partial运行只生成所请求的新结果摘要，绝不混入/覆盖正式选型。
    if not full:
        partial = (
            pd.concat([item["summary"] for item in results], ignore_index=True)
            if results
            else t0_frames["summary"][t0_frames["summary"]["farm_id"].isin(farms)]
        )
        path = _atomic_csv(
            partial, os.path.join(output_dir, "time_freq_model_partial_summary.csv")
        )
        _atomic_json(
            {
                "status": "partial_not_formal",
                "variants": variants,
                "farms": farms,
                "max_samples": args.max_samples,
                "summary": _file_record(path),
            },
            os.path.join(output_dir, "partial_run_manifest.json"),
        )
        print(f"partial/smoke结果（不参与正式选型）: {path}")
        return

    if len(results) != len(NEW_VARIANTS) * len(expected_farms):
        raise ValueError("正式测试必须包含M0/T1/T2/T3 x 5场站")
    frames = {key: _concat(t0_frames, results, key) for key in STAGE3_FORMAL_FILES}
    validate_complete_output_matrix(frames)
    comparison = build_comparison(
        frames["summary"], frames["candidate"], frames["regime"]
    )
    candidate_drift = build_candidate_drift(t0_frames["summary"], results)
    candidate_control_invariants = validate_candidate_control_invariants(
        candidate_drift
    )
    comparison = attach_fixed_t0_calibration(comparison, candidate_drift)
    selected, comparison = select_model(comparison)
    complementarity = build_joint_complementarity(frames["candidate"])
    complexity = (
        frames["summary"]
        .groupby("model_variant", as_index=False)
        .agg(
            parameter_count_max=("parameter_count", "max"),
            adapter_parameter_count_max=("adapter_trainable_parameter_count", "max"),
            inference_ms_per_sample_macro=("inference_milliseconds_per_sample", "mean"),
            candidate_training_seconds_macro=(
                "candidate_training_elapsed_seconds",
                "mean",
            ),
            gate_training_seconds_macro=("gate_training_elapsed_seconds", "mean"),
        )
    )
    t0_params = int(
        complexity.loc[complexity["model_variant"] == "t0", "parameter_count_max"].iloc[
            0
        ]
    )
    complexity["parameter_delta_vs_t0"] = complexity["parameter_count_max"] - t0_params
    complexity["random_seed"] = tf_train.RANDOM_SEED
    complexity["seed_count"] = 1
    complexity["stability_scope"] = "single_seed_2026_no_multiseed_claim"
    paths = {}
    for key, frame in frames.items():
        paths[key] = _atomic_csv(
            frame, os.path.join(output_dir, f"time_freq_model_test_{key}.csv")
        )
    paths["comparison"] = _atomic_csv(
        comparison,
        os.path.join(output_dir, "time_freq_model_test_variant_comparison.csv"),
    )
    paths["final_selection"] = _atomic_csv(
        comparison[comparison["selected"]],
        os.path.join(output_dir, "time_freq_model_test_final_selection.csv"),
    )
    paths["joint_complementarity"] = _atomic_csv(
        complementarity,
        os.path.join(output_dir, "time_freq_model_test_joint_complementarity.csv"),
    )
    paths["candidate_drift"] = _atomic_csv(
        candidate_drift,
        os.path.join(output_dir, "time_freq_model_test_candidate_drift.csv"),
    )
    paths["complexity"] = _atomic_csv(
        complexity, os.path.join(output_dir, "time_freq_model_test_complexity.csv")
    )
    paths.update(
        save_aggregate_figures(
            comparison,
            frames["summary"],
            frames["horizon"],
            complementarity,
            output_dir,
        )
    )
    source_rows = [
        {
            "source": "Stage-3 complete marker",
            "key": "marker",
            **_file_record(STAGE3_MARKER),
            "reuse_action": "hash_validated_read_only_dependency",
        }
    ]
    for key, path in source_paths.items():
        source_rows.append(
            {
                "source": "Stage-3 G0 formal aggregate",
                "key": key,
                **_file_record(path),
                "reuse_action": "filter_g0_relabel_t0_no_inference",
            }
        )
    t0_summary = t0_frames["summary"]
    for _, row in t0_summary.iterrows():
        for key in (
            "model_path",
            "artifact_path",
            "prediction_path",
            "candidate_archive_path",
        ):
            path = row.get(key)
            if isinstance(path, str) and os.path.isfile(path):
                source_rows.append(
                    {
                        "source": f"Stage-3 G0 farm {row['farm_id']}",
                        "key": key,
                        **_file_record(path),
                        "reuse_action": "direct_path_reference_no_copy",
                    }
                )
    paths["source_manifest"] = _atomic_csv(
        pd.DataFrame(source_rows),
        os.path.join(output_dir, "time_freq_model_t0_source_reuse_manifest.csv"),
    )
    paths.update(write_reports(comparison, selected, complementarity, output_dir))
    files = {
        "prediction_code": _file_record(__file__),
        "training_code": _file_record(tf_train.__file__),
        "stage3_marker": _file_record(STAGE3_MARKER),
        "training_marker": _file_record(training_marker_path),
    }
    files.update({f"formal.{key}": _file_record(path) for key, path in paths.items()})
    for index, result in enumerate(results):
        for key, path in result["paths"].items():
            if path:
                files[f"result{index}.{key}"] = _file_record(path)
    marker = {
        "status": "complete",
        "protocol_version": tf_train.PROTOCOL_VERSION,
        "architecture_version": tf_train.ARCHITECTURE_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": tf_train.RANDOM_SEED,
        "selection_split": "test",
        "test_used_for_selection": True,
        "test_reuse_status": TEST_REUSE_STATUS,
        "test_is_final_blind_evaluation": False,
        "variants": list(ALL_VARIANTS),
        "expected_farm_ids": expected_farms,
        "t0_policy": "direct_stage3_g0_reference_no_training_no_forward_no_archive_copy",
        "t0_source_reuse_farm_count": len(expected_farms),
        "candidate_control_invariants": candidate_control_invariants,
        "new_prediction_count": len(results),
        "selected_variant": str(selected["model_variant"]),
        "test_files": {
            farm: _file_record(source_test[farm]) for farm in expected_farms
        },
        "files": files,
    }
    marker_path = _atomic_json(marker, formal_marker)
    print(
        f"\n测试集最终选择: {selected['model_variant']} / macro NRMSE={selected['macro_test_nrmse']:.9f}"
    )
    print(f"正式中文报告: {paths['report_cn']}")
    print(f"正式bundle marker: {marker_path}")


if __name__ == "__main__":
    main()
