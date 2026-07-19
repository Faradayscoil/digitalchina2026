"""第三部分第一轮 Stage-A（A0--A5）测试预测、归档与最终选型。

本脚本与 ``wind_part3_round1_stage_a_timealign_qdf_train.py`` 配套：

* A0 严格从 Stage-4B formal D0 complete bundle 只读引用，不训练、不复制、
  不重新执行模型 forward；
* A1--A5 各自加载正式训练 marker 锁定的模型，执行一次测试前向；
* corrected candidate 的五场站 macro NRMSE 是正式选型主指标，同时完整报告
  Persistence、冻结旧 G0 门控回放、逐 horizon、工况、安全性和复杂度；
* candidate 改变后本轮不重校准 gate，因此 frozen-G0 fused 仅是诊断，选中的
  candidate 必须在后续 Stage-C 重新生成 train-only oracle/Q90 并闭环后才能部署；
* 当前测试集已在历史开发中被反复查看，所有正式输出固定标记为
  ``legacy_seen_test_selected``，不能解释为最终盲测；
* 子集、``--max-samples`` 或 ``--skip-plots`` 运行写入 ``partial_runs``，不会
  覆盖正式 bundle 或发布 formal complete marker。

正式运行::

    python wind_part3_round1_stage_a_timealign_qdf_predict.py

单场站小样本检查（A0 不允许与截断样本混比）::

    python wind_part3_round1_stage_a_timealign_qdf_predict.py \
      --variants a1 --farms FARM_ID --max-samples 32 --skip-plots
"""

from __future__ import annotations

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
import wind_part3_round1_stage_a_timealign_qdf_train as stage_a_train
import wind_time_freq_model_stage4b_predict as stage4b_predict
import wind_time_freq_model_stage4b_train as stage4b_train


OUTPUT_SUBDIR = "testdata_predict_output"
TEST_REUSE_STATUS = "legacy_seen_test_selected"
FORMAL_MARKER_NAME = "stage_a_test_bundle_complete.json"
RUNNING_MARKER_NAME = "stage_a_test_bundle_running.json"

ALL_VARIANTS = tuple(stage_a_train.VARIANT_SPECS)
NEW_VARIANTS = tuple(stage_a_train.TRAINABLE_VARIANTS)
A0_VARIANT = "a0"
EXPECTED_VARIANTS = ("a0", "a1", "a2", "a3", "a4", "a5")

SOURCE_ROOT = stage4b_train.RESULT_ROOT
SOURCE_OUTPUT = os.path.join(SOURCE_ROOT, OUTPUT_SUBDIR)
SOURCE_MARKER = os.path.join(
    SOURCE_OUTPUT, stage4b_predict.FORMAL_MARKER_NAME
)
SOURCE_VARIANT = "d0"
SOURCE_FORMAL_KEYS = (
    "summary",
    "horizon",
    "candidate",
    "regime",
    "assignments",
    "safety",
    "calibration",
)

# Stage-B 是后续条件启动，不等同于本轮“数值最优”标签。阈值来自路线文档，
# 这里虽然按用户要求在 test 上计算，仍必须显式标记其非盲测属性。
STAGE_B_MACRO_GAIN = 0.005
STAGE_B_REGIME_DEGRADATION_TOL = 0.0
STAGE_B_HORIZON_DEGRADATION_TOL = 0.0
SELECTION_NRMSE_TIE_TOL = 0.001
FARM_ATOL = 1e-12
ROBUST_REGIME_DEGRADATION_TOL = 0.005
ROBUST_HORIZON_DEGRADATION_TOL = 0.005
ROBUST_MIN_NONDEGRADED_FARMS = 4
ROBUST_MIN_STRICTLY_IMPROVED_FARMS = 3
ROBUST_MAX_CONSECUTIVE_DEGRADED_HORIZONS = 2


def _sha256(path, chunk_size=1024 * 1024):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"无法计算SHA256，文件不存在: {path}")
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
        raise FileNotFoundError(f"正式bundle成员不存在: {path}")
    return {
        "path": os.path.abspath(path),
        "sha256": _sha256(path),
        "size_bytes": int(os.path.getsize(path)),
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


def _atomic_text(value, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            file.write(value)
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


def _read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _validate_record(label, record, expected_path=None):
    if not isinstance(record, dict):
        raise TypeError(f"{label}不是文件记录")
    path = record.get("path")
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{label}记录文件不存在: {path}")
    if expected_path and os.path.realpath(path) != os.path.realpath(expected_path):
        raise ValueError(f"{label}路径漂移: {path} != {expected_path}")
    if _sha256(path) != record.get("sha256"):
        raise ValueError(f"{label} SHA256漂移")
    recorded_size = record.get("size_bytes")
    if recorded_size is not None and int(os.path.getsize(path)) != int(recorded_size):
        raise ValueError(f"{label} size漂移")
    return os.path.abspath(path)


def _expected_farms():
    farms = tuple(str(value) for value in stage_a_train.expected_farm_ids())
    if len(farms) != 5 or len(set(farms)) != 5:
        raise ValueError(f"Stage-A正式场站集合不是唯一5场站: {farms}")
    return farms


def _variant_label(variant):
    spec = stage_a_train.VARIANT_SPECS[variant]
    return str(spec.get("label", variant.upper()))


def _model_name(variant):
    return stage_a_train.variant_model_name(variant)


def _canonical_horizon(value):
    text = str(value).strip().lower()
    if text in {"all", "all.0", "nan"}:
        return "all"
    return str(int(float(text)))


def _parse_list(raw, allowed, label):
    allowed = tuple(str(value) for value in allowed)
    values = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    values = list(dict.fromkeys(values))
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"{label}包含非法值{invalid}; 可选={list(allowed)}")
    return values


def configure_prediction_reproducibility():
    seed = int(stage_a_train.RANDOM_SEED)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _relabel_source(frame, table):
    if "model_variant" not in frame:
        raise KeyError(f"Stage-4B formal.{table}缺少model_variant")
    result = frame[frame["model_variant"].astype(str) == SOURCE_VARIANT].copy()
    result["source_model_family"] = result.get("model_family", "")
    result["source_model_variant"] = SOURCE_VARIANT
    result["model_family"] = stage_a_train.MODEL_FAMILY
    result["model_variant"] = A0_VARIANT
    if "model_name" in result:
        result["model_name"] = _model_name(A0_VARIANT)
    if table == "summary":
        result["variant_label"] = _variant_label(A0_VARIANT)
        result["reference_only"] = True
        result["selection_eligible"] = True
        result["result_source"] = (
            "hash_validated_stage4b_formal_d0_direct_reference_"
            "no_training_no_forward_no_copy"
        )
        result["selection_metric_scope"] = "corrected_candidate_forecast"
        result["fused_role"] = "frozen_g0_diagnostic_only"
        result["candidate_specific_gate_recalibrated"] = False
        result["deployment_eligible"] = False
        result["selection_split"] = "test"
        result["test_used_for_selection"] = True
        result["test_is_final_blind_evaluation"] = False
        result["test_reuse_status"] = TEST_REUSE_STATUS
    elif table == "candidate":
        result["metric_role"] = np.where(
            result["candidate"].astype(str) == "corrected",
            "primary_corrected_candidate",
            np.where(
                result["candidate"].astype(str) == "fused",
                "frozen_g0_fused_diagnostic",
                "persistence_control",
            ),
        )
    elif table == "horizon":
        result["metric_role"] = "frozen_g0_fused_diagnostic"
    return result


def validate_a0_source_bundle():
    """Validate and read Stage-4B D0 tables without model reconstruction."""
    if not os.path.isfile(SOURCE_MARKER):
        raise FileNotFoundError(
            f"缺少A0来源Stage-4B预测complete marker: {SOURCE_MARKER}"
        )
    marker = _read_json(SOURCE_MARKER)
    checks = {
        "status": marker.get("status") == "complete",
        "protocol": marker.get("protocol_version") == stage4b_train.PROTOCOL_VERSION,
        "architecture": marker.get("architecture_version")
        == stage4b_train.ARCHITECTURE_VERSION,
        "farms": set(map(str, marker.get("expected_farm_ids", ())))
        == set(_expected_farms()),
        "test_files": set(map(str, marker.get("test_files", {})))
        == set(_expected_farms()),
        "d0_no_retraining": str(marker.get("d0_policy", "")).startswith("direct_"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"A0 Stage-4B来源marker不兼容: {failed}")
    for key, record in marker.get("files", {}).items():
        _validate_record(f"Stage-4B files.{key}", record)
    for farm_id, record in marker.get("test_files", {}).items():
        _validate_record(f"Stage-4B test_files.{farm_id}", record)

    frames, paths = {}, {}
    for table in SOURCE_FORMAL_KEYS:
        record = marker.get("files", {}).get(f"formal.{table}")
        if record is None:
            raise KeyError(f"Stage-4B marker缺少files.formal.{table}")
        path = _validate_record(f"Stage-4B formal.{table}", record)
        raw = pd.read_csv(path, dtype={"farm_id": str})
        frames[table] = _relabel_source(raw, table)
        paths[table] = path

    expected = set(_expected_farms())
    for table, frame in frames.items():
        if set(frame["farm_id"].astype(str)) != expected:
            raise ValueError(f"A0 formal.{table}未唯一覆盖5场站")
    if len(frames["summary"]) != 5:
        raise ValueError("A0 formal.summary不是恰好5行")

    candidate_all = frames["candidate"].copy()
    candidate_all["_h"] = candidate_all["horizon_step"].map(_canonical_horizon)
    candidate_all = candidate_all[candidate_all["_h"] == "all"]
    corrected = candidate_all[
        candidate_all["candidate"].astype(str) == "corrected"
    ].set_index("farm_id")
    persistence = candidate_all[
        candidate_all["candidate"].astype(str) == "persistence"
    ].set_index("farm_id")
    if len(corrected) != 5 or len(persistence) != 5:
        raise ValueError("A0 candidate表缺少五场站overall corrected/Persistence")
    summary = frames["summary"].copy()
    summary["corrected_candidate_nrmse"] = summary["farm_id"].map(
        corrected["capacity_normalized_rmse"]
    )
    summary["corrected_candidate_nmae"] = summary["farm_id"].map(
        corrected["capacity_normalized_mae"]
    )
    summary["persistence_nrmse"] = summary["farm_id"].map(
        persistence["capacity_normalized_rmse"]
    )
    summary["persistence_nmae"] = summary["farm_id"].map(
        persistence["capacity_normalized_mae"]
    )
    summary["fused_test_nrmse"] = pd.to_numeric(
        summary["capacity_normalized_rmse"], errors="raise"
    )
    summary["fused_test_nmae"] = pd.to_numeric(
        summary["capacity_normalized_mae"], errors="raise"
    )
    summary["trainable_parameter_count_current_round"] = 0
    summary["training_wrapper_parameter_count"] = 0
    summary["training_only_parameter_count"] = 0
    summary["training_wrapper_trainable_parameter_count"] = 0
    summary["training_only_trainable_parameter_count"] = 0
    summary["training_elapsed_seconds_current_round"] = 0.0
    frames["summary"] = summary
    return marker, frames, paths


def validate_training_bundle(required_variants):
    if not required_variants:
        return None, None
    running = os.path.join(stage_a_train.RESULT_ROOT, stage_a_train.RUNNING_MARKER_NAME)
    if os.path.isfile(running):
        raise RuntimeError(f"Stage-A训练仍在运行或未收尾: {running}")
    path = os.path.join(stage_a_train.RESULT_ROOT, stage_a_train.TRAINING_MARKER_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少Stage-A训练complete marker: {path}")
    marker = _read_json(path)
    checks = {
        "status": marker.get("status") == "complete",
        "protocol": marker.get("protocol_version") == stage_a_train.PROTOCOL_VERSION,
        "architecture": marker.get("architecture_version")
        == stage_a_train.ARCHITECTURE_VERSION,
        "farms": set(map(str, marker.get("expected_farm_ids", ())))
        == set(_expected_farms()),
    }
    marker_variants = set(
        marker.get("new_training_variants", marker.get("variants", ()))
    )
    checks["variants"] = set(required_variants).issubset(marker_variants)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Stage-A训练marker不兼容: {failed}")
    for key, record in marker.get("files", {}).items():
        _validate_record(f"Stage-A training files.{key}", record)
    for variant in required_variants:
        for farm_id in _expected_farms():
            for kind in ("model_path", "artifact_path"):
                key = f"{variant}.{farm_id}.{kind}"
                if key not in marker.get("files", {}):
                    raise KeyError(f"Stage-A训练marker缺少{key}")
    return os.path.abspath(path), marker


def _load_model(variant, farm_id, marker):
    if variant not in NEW_VARIANTS:
        raise ValueError(f"{variant}不是新增可预测模型")
    files = marker["files"]
    artifact_path = _validate_record(
        f"{variant}/{farm_id}/artifact",
        files[f"{variant}.{farm_id}.artifact_path"],
    )
    model_path = _validate_record(
        f"{variant}/{farm_id}/model",
        files[f"{variant}.{farm_id}.model_path"],
    )
    artifact = joblib.load(artifact_path)
    checks = {
        "schema": int(artifact.get("artifact_schema_version", -1))
        == int(stage_a_train.ARTIFACT_SCHEMA_VERSION),
        "family": artifact.get("model_family") == stage_a_train.MODEL_FAMILY,
        "architecture": artifact.get("architecture_version")
        == stage_a_train.ARCHITECTURE_VERSION,
        "protocol": artifact.get("protocol_version") == stage_a_train.PROTOCOL_VERSION,
        "variant": artifact.get("variant_id") == variant,
        "farm": str(artifact.get("farm_id")) == str(farm_id),
        "seed": int(artifact.get("random_seed", -1))
        == int(stage_a_train.RANDOM_SEED),
        "history": int(artifact.get("history_len", -1))
        == int(stage_a_train.HISTORY_LEN),
        "forecast": int(artifact.get("forecast_len", -1))
        == int(stage_a_train.FORECAST_LEN),
        "model_path": os.path.realpath(str(artifact.get("model_path", "")))
        == os.path.realpath(model_path),
        "model_hash": artifact.get("model_sha256") == _sha256(model_path),
        "teacher_train_only": artifact.get("future_target_training_only") is True,
        "teacher_removed": artifact.get("teacher_removed_at_inference") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{variant}/{farm_id} artifact不兼容: {failed}")
    if "input_cols" not in artifact or "scaler_x" not in artifact or "scaler_y" not in artifact:
        raise KeyError(f"{variant}/{farm_id} artifact缺少预测预处理对象")
    model = keras.models.load_model(
        model_path,
        custom_objects=stage_a_train.get_stagea_custom_objects(),
        compile=False,
    )
    if len(model.inputs) != 1:
        raise ValueError(f"{variant}/{farm_id}正式模型不是单历史输入")
    input_tensor = model.inputs[0]
    input_name = input_tensor.name.split(":")[0]
    input_shape = tuple(input_tensor.shape.as_list())
    expected_shape = (
        None,
        int(stage_a_train.HISTORY_LEN),
        len(artifact["input_cols"]),
    )
    if input_name != "history_features" or input_shape != expected_shape:
        raise ValueError(
            f"{variant}/{farm_id}输入契约漂移: "
            f"{input_name}/{input_shape} != history_features/{expected_shape}"
        )
    forbidden_tokens = (
        "future_teacher",
        "teacher_decoder",
        "student_projector",
        "qdf_objective",
    )
    forbidden_layers = [
        layer.name
        for layer in model.layers
        if any(token in layer.name for token in forbidden_tokens)
    ]
    if forbidden_layers:
        raise ValueError(
            f"{variant}/{farm_id}训练期teacher/QDF泄漏进推理图: {forbidden_layers}"
        )
    source_model_path = artifact.get("source_model_path")
    if (
        not source_model_path
        or not os.path.isfile(source_model_path)
        or artifact.get("source_model_sha256") != _sha256(source_model_path)
    ):
        raise ValueError(f"{variant}/{farm_id} artifact锁定的同场站F7来源已漂移")
    expected_params = artifact.get(
        "inference_parameter_count", artifact.get("total_params")
    )
    if expected_params is None:
        raise KeyError(f"{variant}/{farm_id} artifact缺少inference参数量")
    if int(model.count_params()) != int(expected_params):
        raise ValueError(f"{variant}/{farm_id}加载参数量与artifact不一致")
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


def predict_variant(variant, test_file, training_marker, max_samples=None):
    farm_id = str(common_predict.get_farm_id(test_file))
    artifact, artifact_path, model, model_path = _load_model(
        variant, farm_id, training_marker
    )
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
    diagnostic = stage_a_train.diagnostic_model(model)
    started = time.perf_counter()
    outputs = diagnostic.predict(dataset, verbose=common_predict.PREDICT_VERBOSE)
    elapsed = float(time.perf_counter() - started)
    if not isinstance(outputs, dict):
        raise TypeError(f"{variant}/{farm_id} diagnostic_model没有返回dict")
    shape = (n_samples, forecast_len)
    required = {}
    aliases = {
        "forecast": ("forecast", "forecast_power", "fused"),
        "persistence": ("persistence", "persistence_forecast_candidate"),
        "corrected": ("corrected", "candidate_forecast"),
        "gate": ("gate", "correction_gate", "frozen_g0_gate"),
    }
    for target, names in aliases.items():
        source = next((name for name in names if name in outputs), None)
        if source is None:
            raise KeyError(f"{variant}/{farm_id}诊断输出缺少{target}: {names}")
        required[target] = _normal_output(
            outputs[source], shape, f"{variant}/{farm_id}/{target}"
        )
    required["q"] = np.repeat(
        np.mean(required["gate"], axis=1, keepdims=True), forecast_len, axis=1
    )
    required["s"] = np.ones(shape, dtype=np.float64)
    y_true = common_predict.build_truth_windows(
        actual_power, n_samples, history_len, forecast_len
    )
    payload = gate_predict._build_payload(
        variant,
        farm_id,
        df,
        artifact,
        required,
        y_true,
        capacity,
        history_len,
    )
    residual = outputs.get("residual")
    if residual is None:
        residual = required["corrected"] - required["persistence"]
    payload["residual_scaled"] = _normal_output(
        residual, shape, f"{variant}/{farm_id}/residual"
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
                    "trainable_parameter_count",
                    sum(int(np.prod(weight.shape)) for weight in model.trainable_weights),
                )
            ),
            "training_wrapper_parameter_count": int(
                artifact.get("training_wrapper_parameter_count", model.count_params())
            ),
            "training_only_parameter_count": int(
                artifact.get("training_only_parameter_count", 0)
            ),
            "training_wrapper_trainable_parameter_count": int(
                artifact.get(
                    "training_wrapper_trainable_parameter_count",
                    sum(int(np.prod(weight.shape)) for weight in model.trainable_weights),
                )
            ),
            "training_only_trainable_parameter_count": int(
                artifact.get("training_only_trainable_parameter_count", 0)
            ),
            "training_elapsed_seconds": float(
                artifact.get("training_elapsed_seconds", np.nan)
            ),
            "inference_elapsed_seconds": elapsed,
            "inference_milliseconds_per_sample": 1000.0 * elapsed / n_samples,
            "reference_only": False,
            "result_source": "part3_stagea_single_formal_test_forward",
        }
    )
    del model
    keras.backend.clear_session()
    return payload


def prediction_dirs(variant, output_root):
    formal = os.path.realpath(output_root) == os.path.realpath(stage_a_train.RESULT_ROOT)
    if formal:
        root = os.path.join(
            stage_a_train.variant_dirs(variant, create=True)["root"], OUTPUT_SUBDIR
        )
    else:
        root = os.path.join(output_root, variant, OUTPUT_SUBDIR)
    values = {
        "root": root,
        "predictions": os.path.join(root, "predictions"),
        "candidate_metrics": os.path.join(root, "candidate_metrics"),
        "regime_metrics": os.path.join(root, "regime_metrics"),
        "regime_assignments": os.path.join(root, "regime_assignments"),
        "candidate_archives": os.path.join(root, "candidate_archives"),
        "safety": os.path.join(root, "safety_diagnostics"),
        "calibration": os.path.join(root, "calibration"),
        "gate_points": os.path.join(root, "gate_diagnostics"),
        "single_windows": os.path.join(root, "single_window_comparisons"),
        "weighted_curves": os.path.join(root, "weighted_curves"),
        "figures": os.path.join(root, "figures"),
        "matplotlib_cache": os.path.join(root, "matplotlib_cache"),
    }
    for path in values.values():
        os.makedirs(path, exist_ok=True)
    return values


def _candidate_metrics(payload):
    rows = []
    for candidate, values in (
        ("fused", payload["fused"]),
        ("persistence", payload["persistence"]),
        ("corrected", payload["corrected"]),
    ):
        frame = common_predict.metrics_by_horizon(
            _model_name(payload["variant_id"]),
            payload["farm_id"],
            payload["y_true"],
            values,
            payload["capacity"],
            payload["forecast_len"],
        )
        frame["model_family"] = stage_a_train.MODEL_FAMILY
        frame["model_variant"] = payload["variant_id"]
        frame["candidate"] = candidate
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


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
    prediction["corrected_candidate_power"] = payload["corrected"].T.reshape(-1)
    prediction["frozen_g0_fused_power"] = payload["fused"].T.reshape(-1)
    prediction["applied_gate"] = payload["applied_gate"].T.reshape(-1)
    prediction_path = _atomic_csv(
        prediction,
        os.path.join(dirs["predictions"], f"{name}_predictions_farm_{farm_id}.csv"),
    )
    candidates = _candidate_metrics(payload)
    candidates["metric_role"] = np.where(
        candidates["candidate"].astype(str) == "corrected",
        "primary_corrected_candidate",
        np.where(
            candidates["candidate"].astype(str) == "fused",
            "frozen_g0_fused_diagnostic",
            "persistence_control",
        ),
    )
    candidate_path = _atomic_csv(
        candidates,
        os.path.join(
            dirs["candidate_metrics"], f"{name}_candidate_metrics_farm_{farm_id}.csv"
        ),
    )
    horizon = candidates[candidates["candidate"] == "fused"].copy()
    horizon["metric_role"] = "frozen_g0_fused_diagnostic"
    horizon_path = _atomic_csv(
        horizon,
        os.path.join(dirs["root"], f"{name}_metrics_by_horizon_farm_{farm_id}.csv"),
    )
    regime = gate_predict._regime_metrics(payload).copy()
    regime["model_family"] = stage_a_train.MODEL_FAMILY
    regime["model_variant"] = variant
    regime_path = _atomic_csv(
        regime,
        os.path.join(dirs["regime_metrics"], f"{name}_regime_metrics_farm_{farm_id}.csv"),
    )
    assignments = gate_predict._assignment_frame(payload).copy()
    assignments["model_family"] = stage_a_train.MODEL_FAMILY
    assignments["model_variant"] = variant
    assignment_path = _atomic_csv(
        assignments,
        os.path.join(
            dirs["regime_assignments"], f"{name}_regime_assignments_farm_{farm_id}.csv"
        ),
    )
    safety = gate_predict.build_safety_scope_frame(payload).copy()
    safety["model_family"] = stage_a_train.MODEL_FAMILY
    safety["model_variant"] = variant
    safety_path = _atomic_csv(
        safety,
        os.path.join(dirs["safety"], f"{name}_safety_farm_{farm_id}.csv"),
    )
    calibration = gate_predict.build_reliability_frame(payload).copy()
    calibration["model_family"] = stage_a_train.MODEL_FAMILY
    calibration["model_variant"] = variant
    calibration_path = _atomic_csv(
        calibration,
        os.path.join(dirs["calibration"], f"{name}_reliability_farm_{farm_id}.csv"),
    )
    gate_points = gate_predict.build_point_gate_frame(payload).copy()
    gate_points["model_family"] = stage_a_train.MODEL_FAMILY
    gate_points["model_variant"] = variant
    gate_path = _atomic_csv(
        gate_points,
        os.path.join(dirs["gate_points"], f"{name}_gate_points_farm_{farm_id}.csv"),
    )
    archive_path = _atomic_npz(
        os.path.join(
            dirs["candidate_archives"], f"{name}_candidate_archive_farm_{farm_id}.npz"
        ),
        schema_version=np.asarray("part3_stagea_candidate_archive_v1"),
        model_variant=np.asarray(variant),
        farm_id=np.asarray(farm_id),
        sample_id=payload["sample_id"],
        horizon_step=payload["horizon_step"],
        forecast_origin_time=payload["forecast_origin_time"],
        capacity=np.asarray(payload["capacity"]),
        y_true=payload["y_true"],
        Y=payload["y_true"],
        persistence=payload["persistence"],
        P=payload["persistence"],
        corrected=payload["corrected"],
        C=payload["corrected"],
        fused=payload["fused"],
        F=payload["fused"],
        persistence_scaled=payload["persistence_scaled"],
        corrected_scaled=payload["corrected_scaled"],
        fused_scaled=payload["fused_scaled"],
        residual_scaled=payload["residual_scaled"],
        raw_gate=payload["raw_gate"],
        applied_gate=payload["applied_gate"],
        q=payload["q"],
        s=payload["s"],
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
    overall = horizon[horizon["horizon_step"].map(_canonical_horizon) == "all"].iloc[0]
    candidate_all = candidates[
        candidates["horizon_step"].map(_canonical_horizon) == "all"
    ].set_index("candidate")
    safety_overall = safety[
        (safety["scope_type"].astype(str) == "overall")
        & (safety["scope_value"].astype(str) == "all")
    ]
    utility = safety_overall.iloc[0].to_dict() if len(safety_overall) == 1 else {}
    for key in (
        "model_family",
        "model_variant",
        "farm_id",
        "scope_type",
        "scope_value",
    ):
        utility.pop(key, None)
    summary = {
        **overall.to_dict(),
        **utility,
        "model_family": stage_a_train.MODEL_FAMILY,
        "model_variant": variant,
        "variant_label": _variant_label(variant),
        "farm_id": farm_id,
        "fused_test_nrmse": float(overall["capacity_normalized_rmse"]),
        "fused_test_nmae": float(overall["capacity_normalized_mae"]),
        "corrected_candidate_nrmse": float(
            candidate_all.loc["corrected", "capacity_normalized_rmse"]
        ),
        "corrected_candidate_nmae": float(
            candidate_all.loc["corrected", "capacity_normalized_mae"]
        ),
        "persistence_nrmse": float(
            candidate_all.loc["persistence", "capacity_normalized_rmse"]
        ),
        "persistence_nmae": float(
            candidate_all.loc["persistence", "capacity_normalized_mae"]
        ),
        "parameter_count": payload["parameter_count"],
        "trainable_parameter_count_current_round": payload[
            "trainable_parameter_count"
        ],
        "training_wrapper_parameter_count": payload[
            "training_wrapper_parameter_count"
        ],
        "training_only_parameter_count": payload["training_only_parameter_count"],
        "training_wrapper_trainable_parameter_count": payload[
            "training_wrapper_trainable_parameter_count"
        ],
        "training_only_trainable_parameter_count": payload[
            "training_only_trainable_parameter_count"
        ],
        "training_elapsed_seconds_current_round": payload[
            "training_elapsed_seconds"
        ],
        "inference_elapsed_seconds": payload["inference_elapsed_seconds"],
        "inference_milliseconds_per_sample": payload[
            "inference_milliseconds_per_sample"
        ],
        "reference_only": False,
        "selection_eligible": True,
        "selection_metric_scope": "corrected_candidate_forecast",
        "fused_role": "frozen_g0_diagnostic_only",
        "candidate_specific_gate_recalibrated": False,
        "deployment_eligible": False,
        "result_source": payload["result_source"],
        "selection_split": "test",
        "test_used_for_selection": True,
        "test_is_final_blind_evaluation": False,
        "test_reuse_status": TEST_REUSE_STATUS,
        "random_seed": int(stage_a_train.RANDOM_SEED),
        "model_path": payload["model_path"],
        "model_sha256": payload["model_sha256"],
        "artifact_path": payload["artifact_path"],
        "artifact_sha256": payload["artifact_sha256"],
        "prediction_path": prediction_path,
        "prediction_sha256": _sha256(prediction_path),
        "horizon_metric_path": horizon_path,
        "candidate_metric_path": candidate_path,
        "regime_metric_path": regime_path,
        "regime_assignment_path": assignment_path,
        "safety_diagnostics_path": safety_path,
        "calibration_path": calibration_path,
        "gate_points_path": gate_path,
        "candidate_archive_path": archive_path,
        "candidate_archive_sha256": _sha256(archive_path),
        "single_window_path": single_path,
        "single_window_figure_path": single_figure,
        "weighted_curve_path": weighted_path,
        "weighted_curve_figure_path": weighted_figure,
        "fusion_reconstruction_max_abs_error_scaled": payload[
            "fusion_reconstruction_max_abs_error_scaled"
        ],
    }
    summary.update({f"weighted_{key}": value for key, value in weighted_metrics.items()})
    return {
        "summary": pd.DataFrame([summary]),
        "horizon": horizon,
        "candidate": candidates,
        "regime": regime,
        "assignments": assignments,
        "safety": safety,
        "calibration": calibration,
    }


def _exact_five(frame, label, numeric_columns=()):
    result = frame.copy()
    if len(result) != 5 or set(result["farm_id"].astype(str)) != set(_expected_farms()):
        raise ValueError(f"{label}不是固定5场站唯一矩阵")
    if result["farm_id"].astype(str).duplicated().any():
        raise ValueError(f"{label}存在重复场站")
    for column in numeric_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"{label}/{column}包含非有限值")
    return result


def validate_complete_matrix(frames):
    required = {
        "summary",
        "horizon",
        "candidate",
        "regime",
        "assignments",
        "safety",
        "calibration",
    }
    if set(frames) != required:
        raise ValueError(f"Stage-A表集不完整: {set(frames)} != {required}")
    variants, farms = set(EXPECTED_VARIANTS), set(_expected_farms())
    for name, frame in frames.items():
        if set(frame["model_variant"].astype(str)) != variants:
            raise ValueError(f"{name}未覆盖A0--A5")
        if set(frame["farm_id"].astype(str)) != farms:
            raise ValueError(f"{name}未覆盖固定5场站")
    if len(frames["summary"]) != len(variants) * len(farms):
        raise ValueError("Stage-A summary不是6变体×5场站")
    if frames["summary"].duplicated(["model_variant", "farm_id"]).any():
        raise ValueError("Stage-A summary自然键重复")

    horizon = frames["horizon"].copy()
    horizon["_h"] = horizon["horizon_step"].map(_canonical_horizon)
    candidate = frames["candidate"].copy()
    candidate["_h"] = candidate["horizon_step"].map(_canonical_horizon)
    expected_h = {"all", *(str(index) for index in range(1, 17))}
    for variant in EXPECTED_VARIANTS:
        for farm_id in _expected_farms():
            part = horizon[
                (horizon["model_variant"].astype(str) == variant)
                & (horizon["farm_id"].astype(str) == farm_id)
            ]
            if set(part["_h"]) != expected_h or len(part) != 17:
                raise ValueError(f"{variant}/{farm_id} fused逐horizon不完整")
            cpart = candidate[
                (candidate["model_variant"].astype(str) == variant)
                & (candidate["farm_id"].astype(str) == farm_id)
            ]
            if set(cpart["candidate"].astype(str)) != {
                "fused",
                "persistence",
                "corrected",
            }:
                raise ValueError(f"{variant}/{farm_id} P/C/F候选集合不完整")
            for role in ("fused", "persistence", "corrected"):
                role_part = cpart[cpart["candidate"].astype(str) == role]
                if set(role_part["_h"]) != expected_h or len(role_part) != 17:
                    raise ValueError(f"{variant}/{farm_id}/{role}逐horizon不完整")
    return True


def _macro_regime(regime, variant, group, candidate="fused"):
    part = regime[
        (regime["model_variant"].astype(str) == variant)
        & (regime["candidate"].astype(str) == candidate)
        & (regime["regime_group"].astype(str) == group)
        & (regime["horizon_step"].map(_canonical_horizon) == "all")
    ]
    part = _exact_five(
        part,
        f"{variant}/{candidate}/{group}",
        ("capacity_normalized_rmse", "capacity_normalized_mae"),
    )
    return {
        "nrmse": float(part["capacity_normalized_rmse"].mean()),
        "nmae": float(part["capacity_normalized_mae"].mean()),
    }


def _macro_horizon(horizon, variant):
    part = horizon[
        (horizon["model_variant"].astype(str) == variant)
        & (horizon["horizon_step"].map(_canonical_horizon) != "all")
    ].copy()
    part["h"] = pd.to_numeric(part["horizon_step"], errors="raise").astype(int)
    result = (
        part.groupby("h", as_index=False)[
            ["capacity_normalized_rmse", "capacity_normalized_mae"]
        ]
        .mean()
        .sort_values("h")
    )
    if len(result) != 16 or set(result["h"]) != set(range(1, 17)):
        raise ValueError(f"{variant}五场站macro逐horizon不完整")
    return result


def _macro_candidate_horizon(candidate_frame, variant, role="corrected"):
    """Five-farm macro horizon curve for the Stage-A primary candidate."""
    part = candidate_frame[
        (candidate_frame["model_variant"].astype(str) == variant)
        & (candidate_frame["candidate"].astype(str) == role)
        & (candidate_frame["horizon_step"].map(_canonical_horizon) != "all")
    ].copy()
    part["h"] = pd.to_numeric(part["horizon_step"], errors="raise").astype(int)
    result = (
        part.groupby("h", as_index=False)[
            ["capacity_normalized_rmse", "capacity_normalized_mae"]
        ]
        .mean()
        .sort_values("h")
    )
    if len(result) != 16 or set(result["h"]) != set(range(1, 17)):
        raise ValueError(f"{variant}/{role}五场站macro逐horizon不完整")
    return result


def _longest_true_run(values):
    longest = current = 0
    for value in np.asarray(values, dtype=bool).reshape(-1):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def build_pairwise_vs_a0(summary):
    base = _exact_five(
        summary[summary["model_variant"].astype(str) == A0_VARIANT],
        "A0 pairwise baseline",
        (
            "fused_test_nrmse",
            "fused_test_nmae",
            "corrected_candidate_nrmse",
            "corrected_candidate_nmae",
        ),
    ).set_index("farm_id")
    rows = []
    metric_names = (
        "fused_test_nrmse",
        "fused_test_nmae",
        "corrected_candidate_nrmse",
        "corrected_candidate_nmae",
    )
    for variant in NEW_VARIANTS:
        target = _exact_five(
            summary[summary["model_variant"].astype(str) == variant],
            f"{variant} pairwise target",
            metric_names,
        ).set_index("farm_id").reindex(base.index)
        for farm_id in base.index:
            row = {"model_variant": variant, "farm_id": str(farm_id)}
            for metric in metric_names:
                reference = float(base.loc[farm_id, metric])
                value = float(target.loc[farm_id, metric])
                row[f"a0_{metric}"] = reference
                row[metric] = value
                row[f"{metric}_delta_vs_a0"] = value - reference
                row[f"{metric}_improvement_vs_a0"] = 1.0 - value / reference
            row["fused_nrmse_strictly_improved"] = bool(
                row["fused_test_nrmse_delta_vs_a0"] < -FARM_ATOL
            )
            row["fused_nmae_nondegraded"] = bool(
                row["fused_test_nmae_delta_vs_a0"] <= FARM_ATOL
            )
            row["corrected_nrmse_strictly_improved"] = bool(
                row["corrected_candidate_nrmse_delta_vs_a0"] < -FARM_ATOL
            )
            rows.append(row)
        macro = {"model_variant": variant, "farm_id": "macro"}
        for metric in metric_names:
            reference = float(base[metric].mean())
            value = float(target[metric].mean())
            macro[f"a0_{metric}"] = reference
            macro[metric] = value
            macro[f"{metric}_delta_vs_a0"] = value - reference
            macro[f"{metric}_improvement_vs_a0"] = 1.0 - value / reference
        macro["fused_nrmse_strictly_improved"] = bool(
            macro["fused_test_nrmse_delta_vs_a0"] < -FARM_ATOL
        )
        macro["fused_nmae_nondegraded"] = bool(
            macro["fused_test_nmae_delta_vs_a0"] <= FARM_ATOL
        )
        macro["corrected_nrmse_strictly_improved"] = bool(
            macro["corrected_candidate_nrmse_delta_vs_a0"] < -FARM_ATOL
        )
        rows.append(macro)
    return pd.DataFrame(rows)


def build_comparison(summary, candidate, regime, pairwise):
    """Build candidate-primary comparisons; frozen-G0 fused stays diagnostic."""
    rows = []
    regime_groups = ("dynamic", "ramp_up", "ramp_down", "change_ge_20")
    for variant in EXPECTED_VARIANTS:
        frame = _exact_five(
            summary[summary["model_variant"].astype(str) == variant],
            f"{variant} summary",
            (
                "fused_test_nrmse",
                "fused_test_nmae",
                "corrected_candidate_nrmse",
                "corrected_candidate_nmae",
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
            "macro_frozen_g0_fused_test_nrmse": float(
                frame["fused_test_nrmse"].mean()
            ),
            "macro_frozen_g0_fused_test_nmae": float(
                frame["fused_test_nmae"].mean()
            ),
            "macro_frozen_g0_fused_test_r2": float(
                pd.to_numeric(frame["r2"], errors="coerce").mean()
            ),
            "macro_persistence_test_nrmse": float(frame["persistence_nrmse"].mean()),
            "parameter_count_max": int(frame["parameter_count"].max()),
            "training_wrapper_parameter_count_max": int(
                pd.to_numeric(
                    frame["training_wrapper_parameter_count"], errors="raise"
                ).max()
            ),
            "training_only_parameter_count_max": int(
                pd.to_numeric(
                    frame["training_only_parameter_count"], errors="raise"
                ).max()
            ),
            "trainable_parameter_count_max": int(
                pd.to_numeric(
                    frame["trainable_parameter_count_current_round"], errors="coerce"
                ).fillna(0).max()
            ),
            "macro_inference_milliseconds_per_sample": float(
                pd.to_numeric(
                    frame["inference_milliseconds_per_sample"], errors="coerce"
                ).mean()
            ),
            "macro_positive_regret_mean_frozen_gate": float(
                pd.to_numeric(frame.get("positive_regret_mean"), errors="coerce").mean()
            ),
            "macro_harm_rate_gt_0_frozen_gate": float(
                pd.to_numeric(frame.get("harm_rate_gt_0"), errors="coerce").mean()
            ),
            "macro_oracle_brier_frozen_gate": float(
                pd.to_numeric(frame.get("oracle_brier"), errors="coerce").mean()
            ),
        }
        for group in regime_groups:
            corrected_values = _macro_regime(regime, variant, group, "corrected")
            row[f"corrected_{group}_nrmse"] = corrected_values["nrmse"]
            row[f"corrected_{group}_nmae"] = corrected_values["nmae"]
            fused_values = _macro_regime(regime, variant, group, "fused")
            row[f"frozen_g0_fused_{group}_nrmse"] = fused_values["nrmse"]
            row[f"frozen_g0_fused_{group}_nmae"] = fused_values["nmae"]
        rows.append(row)
    comparison = pd.DataFrame(rows)
    base = comparison[comparison["model_variant"] == A0_VARIANT].iloc[0]
    for metric in (
        "macro_corrected_candidate_test_nrmse",
        "macro_corrected_candidate_test_nmae",
        "macro_frozen_g0_fused_test_nrmse",
        "macro_frozen_g0_fused_test_nmae",
    ):
        comparison[f"{metric}_improvement_vs_a0"] = (
            1.0 - comparison[metric] / float(base[metric])
        )

    guard_rows = []
    base_farms = summary[summary["model_variant"] == A0_VARIANT].set_index("farm_id")
    base_candidate_horizon = _macro_candidate_horizon(candidate, A0_VARIANT)
    for _, row in comparison.iterrows():
        variant = str(row["model_variant"])
        if variant == A0_VARIANT:
            guard_rows.append(
                {
                    "model_variant": variant,
                    "farms_candidate_nrmse_strictly_improved": 0,
                    "farms_candidate_nrmse_nondegraded": 5,
                    "farms_candidate_nmae_nondegraded": 5,
                    "farms_frozen_g0_fused_nrmse_strictly_improved": 0,
                    "robust_macro_candidate_nrmse_strictly_improved": False,
                    "robust_macro_candidate_nmae_nondegraded": True,
                    "robust_at_least_4of5_farms_candidate_nrmse_nondegraded": True,
                    "robust_at_least_3of5_farms_candidate_nrmse_improved": False,
                    "robust_at_least_4of5_farms_candidate_nmae_nondegraded": True,
                    "robust_candidate_regime_guard_pass": True,
                    "robust_candidate_longest_consecutive_degraded_horizons": 0,
                    "robust_candidate_no_3_consecutive_degraded_horizons": True,
                    "robust_selection_pass": True,
                    "stage_b_5of5_candidate_nrmse_improved": False,
                    "stage_b_5of5_candidate_nmae_nondegraded": True,
                    "stage_b_macro_candidate_nrmse_gain_ge_0_5pct": False,
                    "stage_b_macro_candidate_nmae_gain_ge_0_5pct": False,
                    "stage_b_candidate_regime_guard_pass": True,
                    "stage_b_candidate_horizon_guard_pass": True,
                    "stage_b_unlock_pass": False,
                }
            )
            continue
        target = summary[summary["model_variant"] == variant].set_index("farm_id")
        target = target.reindex(base_farms.index)
        candidate_nrmse = target["corrected_candidate_nrmse"].astype(float)
        base_candidate_nrmse = base_farms["corrected_candidate_nrmse"].astype(float)
        candidate_nmae = target["corrected_candidate_nmae"].astype(float)
        base_candidate_nmae = base_farms["corrected_candidate_nmae"].astype(float)
        farms_candidate_improved = int(
            (candidate_nrmse < base_candidate_nrmse - FARM_ATOL).sum()
        )
        farms_candidate_nondegraded = int(
            (candidate_nrmse <= base_candidate_nrmse + FARM_ATOL).sum()
        )
        farms_candidate_nmae_nondegraded = int(
            (candidate_nmae <= base_candidate_nmae + FARM_ATOL).sum()
        )
        farms_fused_improved = int(
            (
                target["fused_test_nrmse"].astype(float)
                < base_farms["fused_test_nrmse"].astype(float) - FARM_ATOL
            ).sum()
        )
        stage_b_regime_guard = all(
            float(row[f"corrected_{group}_nrmse"])
            <= float(base[f"corrected_{group}_nrmse"])
            * (1.0 + STAGE_B_REGIME_DEGRADATION_TOL)
            and float(row[f"corrected_{group}_nmae"])
            <= float(base[f"corrected_{group}_nmae"])
            * (1.0 + STAGE_B_REGIME_DEGRADATION_TOL)
            for group in regime_groups
        )
        target_candidate_horizon = _macro_candidate_horizon(candidate, variant)
        stage_b_horizon_guard = bool(
            np.all(
                target_candidate_horizon["capacity_normalized_rmse"].to_numpy(float)
                <= base_candidate_horizon["capacity_normalized_rmse"].to_numpy(float)
                * (1.0 + STAGE_B_HORIZON_DEGRADATION_TOL)
            )
        )
        robust_regime_guard = all(
            float(row[f"corrected_{group}_nrmse"])
            <= float(base[f"corrected_{group}_nrmse"])
            * (1.0 + ROBUST_REGIME_DEGRADATION_TOL)
            and float(row[f"corrected_{group}_nmae"])
            <= float(base[f"corrected_{group}_nmae"])
            * (1.0 + ROBUST_REGIME_DEGRADATION_TOL)
            for group in regime_groups
        )
        degraded = (
            target_candidate_horizon["capacity_normalized_rmse"].to_numpy(float)
            > base_candidate_horizon["capacity_normalized_rmse"].to_numpy(float)
            * (1.0 + ROBUST_HORIZON_DEGRADATION_TOL)
        )
        longest_degraded = _longest_true_run(degraded)
        robust_horizon_guard = bool(
            longest_degraded <= ROBUST_MAX_CONSECUTIVE_DEGRADED_HORIZONS
        )
        robust_macro_nrmse = bool(
            row["macro_corrected_candidate_test_nrmse_improvement_vs_a0"] > 0.0
        )
        robust_macro_nmae = bool(
            row["macro_corrected_candidate_test_nmae_improvement_vs_a0"] >= -FARM_ATOL
        )
        robust_selection = bool(
            robust_macro_nrmse
            and robust_macro_nmae
            and farms_candidate_nondegraded >= ROBUST_MIN_NONDEGRADED_FARMS
            and farms_candidate_improved >= ROBUST_MIN_STRICTLY_IMPROVED_FARMS
            and farms_candidate_nmae_nondegraded >= ROBUST_MIN_NONDEGRADED_FARMS
            and robust_regime_guard
            and robust_horizon_guard
        )
        macro_candidate_nrmse = bool(
            row["macro_corrected_candidate_test_nrmse_improvement_vs_a0"]
            >= STAGE_B_MACRO_GAIN
        )
        macro_candidate_nmae = bool(
            row["macro_corrected_candidate_test_nmae_improvement_vs_a0"]
            >= STAGE_B_MACRO_GAIN
        )
        unlock = bool(
            farms_candidate_improved == 5
            and farms_candidate_nmae_nondegraded == 5
            and macro_candidate_nrmse
            and macro_candidate_nmae
            and stage_b_regime_guard
            and stage_b_horizon_guard
        )
        guard_rows.append(
            {
                "model_variant": variant,
                "farms_candidate_nrmse_strictly_improved": farms_candidate_improved,
                "farms_candidate_nrmse_nondegraded": farms_candidate_nondegraded,
                "farms_candidate_nmae_nondegraded": farms_candidate_nmae_nondegraded,
                "farms_frozen_g0_fused_nrmse_strictly_improved": farms_fused_improved,
                "robust_macro_candidate_nrmse_strictly_improved": robust_macro_nrmse,
                "robust_macro_candidate_nmae_nondegraded": robust_macro_nmae,
                "robust_at_least_4of5_farms_candidate_nrmse_nondegraded": (
                    farms_candidate_nondegraded >= ROBUST_MIN_NONDEGRADED_FARMS
                ),
                "robust_at_least_3of5_farms_candidate_nrmse_improved": (
                    farms_candidate_improved >= ROBUST_MIN_STRICTLY_IMPROVED_FARMS
                ),
                "robust_at_least_4of5_farms_candidate_nmae_nondegraded": (
                    farms_candidate_nmae_nondegraded >= ROBUST_MIN_NONDEGRADED_FARMS
                ),
                "robust_candidate_regime_guard_pass": bool(robust_regime_guard),
                "robust_candidate_longest_consecutive_degraded_horizons": longest_degraded,
                "robust_candidate_no_3_consecutive_degraded_horizons": robust_horizon_guard,
                "robust_selection_pass": robust_selection,
                "stage_b_5of5_candidate_nrmse_improved": farms_candidate_improved == 5,
                "stage_b_5of5_candidate_nmae_nondegraded": (
                    farms_candidate_nmae_nondegraded == 5
                ),
                "stage_b_macro_candidate_nrmse_gain_ge_0_5pct": macro_candidate_nrmse,
                "stage_b_macro_candidate_nmae_gain_ge_0_5pct": macro_candidate_nmae,
                "stage_b_candidate_regime_guard_pass": bool(stage_b_regime_guard),
                "stage_b_candidate_horizon_guard_pass": bool(stage_b_horizon_guard),
                "stage_b_unlock_pass": unlock,
            }
        )
    return comparison.merge(
        pd.DataFrame(guard_rows), on="model_variant", validate="one_to_one"
    )


def select_model(comparison):
    numerical_best = str(
        comparison.sort_values(
            [
                "macro_corrected_candidate_test_nrmse",
                "macro_corrected_candidate_test_nmae",
            ],
            kind="stable",
        ).iloc[0]["model_variant"]
    )
    robust = comparison[
        (comparison["model_variant"] != A0_VARIANT)
        & comparison["robust_selection_pass"].astype(bool)
    ].copy()
    if robust.empty:
        selected = comparison[comparison["model_variant"] == A0_VARIANT].iloc[0]
        selection_status = "fallback_a0_no_new_variant_passed_robust_selection"
    else:
        robust_minimum = float(
            robust["macro_corrected_candidate_test_nrmse"].min()
        )
        near = robust[
            robust["macro_corrected_candidate_test_nrmse"]
            <= robust_minimum * (1.0 + SELECTION_NRMSE_TIE_TOL)
        ]
        selected = near.sort_values(
            [
                "macro_corrected_candidate_test_nmae",
                "training_wrapper_parameter_count_max",
                "parameter_count_max",
                "macro_inference_milliseconds_per_sample",
            ],
            kind="stable",
        ).iloc[0]
        selection_status = "robust_pass_then_candidate_nrmse_0_1pct_tie_breakers"
    qualified = comparison[
        (comparison["model_variant"] != A0_VARIANT)
        & comparison["stage_b_unlock_pass"].astype(bool)
    ]
    stage_b_variant = None
    if not qualified.empty:
        stage_b_variant = str(
            qualified.sort_values(
                [
                    "macro_corrected_candidate_test_nrmse",
                    "macro_corrected_candidate_test_nmae",
                    "training_wrapper_parameter_count_max",
                    "parameter_count_max",
                ],
                kind="stable",
            ).iloc[0]["model_variant"]
        )
    result = comparison.copy()
    result["numerically_lowest_candidate_nrmse"] = (
        result["model_variant"] == numerical_best
    )
    fused_best = str(
        result.sort_values(
            ["macro_frozen_g0_fused_test_nrmse", "macro_frozen_g0_fused_test_nmae"],
            kind="stable",
        ).iloc[0]["model_variant"]
    )
    result["numerically_lowest_frozen_g0_fused_nrmse"] = (
        result["model_variant"] == fused_best
    )
    result["selected"] = result["model_variant"] == selected["model_variant"]
    result["stage_b_recommended"] = (
        result["model_variant"] == stage_b_variant
        if stage_b_variant is not None
        else False
    )
    result["selection_status"] = selection_status
    result["selection_rule"] = (
        "new_variant_must_pass_robust_guards_else_fallback_a0; qualified_"
        "variants_use_candidate_nrmse_then_0.1pct_candidate_nmae_"
        "training_wrapper_params_inference_params_latency"
    )
    return result[result["selected"]].iloc[0], stage_b_variant, result


def build_complexity(summary):
    frame = (
        summary.groupby("model_variant", as_index=False)
        .agg(
            parameter_count_max=("parameter_count", "max"),
            trainable_parameter_count_max=(
                "trainable_parameter_count_current_round",
                "max",
            ),
            training_wrapper_parameter_count_max=(
                "training_wrapper_parameter_count",
                "max",
            ),
            training_only_parameter_count_max=(
                "training_only_parameter_count",
                "max",
            ),
            training_wrapper_trainable_parameter_count_max=(
                "training_wrapper_trainable_parameter_count",
                "max",
            ),
            training_only_trainable_parameter_count_max=(
                "training_only_trainable_parameter_count",
                "max",
            ),
            training_elapsed_seconds_macro=(
                "training_elapsed_seconds_current_round",
                "mean",
            ),
            inference_milliseconds_per_sample_macro=(
                "inference_milliseconds_per_sample",
                "mean",
            ),
        )
        .copy()
    )
    base = int(
        frame.loc[
            frame["model_variant"] == A0_VARIANT, "parameter_count_max"
        ].iloc[0]
    )
    frame["parameter_delta_vs_a0"] = frame["parameter_count_max"] - base
    frame["parameter_budget_policy"] = "unbounded_per_user_but_fully_reported"
    frame["random_seed"] = int(stage_a_train.RANDOM_SEED)
    frame["seed_count"] = 1
    frame["stability_scope"] = "single_seed_2026_no_multiseed_claim"
    return frame


def save_aggregate_figures(
    comparison,
    summary,
    candidate,
    regime,
    pairwise,
    complexity,
    output_dir,
):
    figure_dir = os.path.join(output_dir, "figures")
    cache_dir = os.path.join(output_dir, "matplotlib_cache")
    os.makedirs(figure_dir, exist_ok=True)
    plt = common_predict.setup_matplotlib({"matplotlib_cache": cache_dir})
    paths = {}

    ordered = comparison.sort_values(
        "macro_corrected_candidate_test_nrmse", kind="stable"
    )
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    colors = ["#d62728" if value else "#4c78a8" for value in ordered["selected"]]
    ax.bar(
        ordered["model_variant"],
        ordered["macro_corrected_candidate_test_nrmse"],
        color=colors,
    )
    ax.set_ylabel("Five-farm macro corrected-candidate NRMSE")
    ax.set_title("Part-3 Stage-A candidate test ranking")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths["rank_figure"] = os.path.join(
        figure_dir, "stage_a_candidate_nrmse_rank.png"
    )
    fig.savefig(paths["rank_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    matrix = summary.pivot(
        index="model_variant", columns="farm_id", values="corrected_candidate_nrmse"
    ).reindex(index=EXPECTED_VARIANTS, columns=_expected_farms())
    if matrix.isna().any().any():
        raise ValueError("Stage-A场站NRMSE热力图矩阵不完整")
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    image = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(matrix)), labels=matrix.index)
    ax.set_xticks(
        range(len(matrix.columns)), labels=[str(value)[-4:] for value in matrix.columns]
    )
    ax.set_xlabel("Farm ID (last four digits)")
    ax.set_title("Corrected-candidate NRMSE by farm")
    fig.colorbar(image, ax=ax, label="NRMSE")
    fig.tight_layout()
    paths["farm_heatmap_figure"] = os.path.join(
        figure_dir, "stage_a_candidate_farm_heatmap.png"
    )
    fig.savefig(paths["farm_heatmap_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    numeric = candidate[
        (candidate["candidate"].astype(str) == "corrected")
        & (candidate["horizon_step"].map(_canonical_horizon) != "all")
    ].copy()
    numeric["h"] = pd.to_numeric(numeric["horizon_step"], errors="raise")
    macro = numeric.groupby(["model_variant", "h"], as_index=False)[
        "capacity_normalized_rmse"
    ].mean()
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for variant in EXPECTED_VARIANTS:
        part = macro[macro["model_variant"] == variant].sort_values("h")
        ax.plot(
            part["h"],
            part["capacity_normalized_rmse"],
            marker="o",
            markersize=3,
            label=variant.upper(),
        )
    ax.set_xlabel("Forecast horizon (15-min steps)")
    ax.set_ylabel("Five-farm macro corrected-candidate NRMSE")
    ax.set_title("Horizon-wise candidate error")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    paths["horizon_figure"] = os.path.join(
        figure_dir, "stage_a_candidate_horizon_nrmse.png"
    )
    fig.savefig(paths["horizon_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    ordered = comparison.set_index("model_variant").reindex(EXPECTED_VARIANTS)
    x = np.arange(len(ordered))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.6, 5.1))
    ax.bar(
        x - width / 2,
        ordered["macro_frozen_g0_fused_test_nrmse"],
        width,
        label="frozen-G0 fused diagnostic",
    )
    ax.bar(
        x + width / 2,
        ordered["macro_corrected_candidate_test_nrmse"],
        width,
        label="corrected candidate",
    )
    ax.set_xticks(x, labels=[value.upper() for value in ordered.index])
    ax.set_ylabel("Five-farm macro NRMSE")
    ax.set_title("Candidate quality and frozen-gate conversion")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    paths["candidate_fused_figure"] = os.path.join(
        figure_dir, "stage_a_candidate_vs_fused.png"
    )
    fig.savefig(paths["candidate_fused_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    groups = ("dynamic", "ramp_up", "ramp_down", "change_ge_20")
    x = np.arange(len(groups))
    width = 0.13
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    for index, variant in enumerate(EXPECTED_VARIANTS):
        values = [
            _macro_regime(regime, variant, group, "corrected")["nrmse"]
            for group in groups
        ]
        ax.bar(x + (index - 2.5) * width, values, width, label=variant.upper())
    ax.set_xticks(x, labels=groups)
    ax.set_ylabel("Five-farm macro corrected-candidate NRMSE")
    ax.set_title("Candidate dynamic, ramp and large-change performance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    paths["regime_figure"] = os.path.join(
        figure_dir, "stage_a_dynamic_ramp_large_change.png"
    )
    fig.savefig(paths["regime_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    macro_pair = pairwise[pairwise["farm_id"].astype(str) == "macro"].copy()
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.bar(
        macro_pair["model_variant"],
        100.0 * macro_pair["corrected_candidate_nrmse_improvement_vs_a0"],
        color="#59a14f",
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.axhline(100.0 * STAGE_B_MACRO_GAIN, color="red", linestyle="--")
    ax.set_ylabel("Macro corrected-candidate NRMSE improvement vs A0 (%)")
    ax.set_title("Direct paired gain over immutable A0")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths["pairwise_figure"] = os.path.join(
        figure_dir, "stage_a_pairwise_gain_vs_a0.png"
    )
    fig.savefig(paths["pairwise_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    joined = comparison.merge(complexity, on="model_variant", suffixes=("", "_complexity"))
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for _, row in joined.iterrows():
        ax.scatter(
            row["parameter_count_max"],
            row["macro_corrected_candidate_test_nrmse"],
            s=75,
        )
        ax.annotate(
            str(row["model_variant"]).upper(),
            (
                row["parameter_count_max"],
                row["macro_corrected_candidate_test_nrmse"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Parameter count")
    ax.set_ylabel("Five-farm macro corrected-candidate NRMSE")
    ax.set_title("Accuracy-complexity trade-off (no hard parameter cap)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    paths["complexity_figure"] = os.path.join(
        figure_dir, "stage_a_accuracy_complexity.png"
    )
    fig.savefig(paths["complexity_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def write_report(comparison, selected, stage_b_variant, pairwise, output_dir):
    columns = [
        "model_variant",
        "variant_label",
        "macro_corrected_candidate_test_nrmse",
        "macro_corrected_candidate_test_nmae",
        "macro_corrected_candidate_test_nrmse_improvement_vs_a0",
        "macro_frozen_g0_fused_test_nrmse",
        "macro_frozen_g0_fused_test_nmae",
        "farms_candidate_nrmse_strictly_improved",
        "farms_candidate_nrmse_nondegraded",
        "farms_candidate_nmae_nondegraded",
        "parameter_count_max",
        "robust_selection_pass",
        "stage_b_unlock_pass",
        "numerically_lowest_candidate_nrmse",
        "selected",
    ]
    macro_pair = pairwise[pairwise["farm_id"].astype(str) == "macro"]
    stage_b_text = (
        f"Stage-B 条件已满足，推荐入口为 **{stage_b_variant.upper()}**。"
        if stage_b_variant is not None
        else "没有 A1--A5 通过全部 Stage-B 启动门槛，当前不得自动扩展 Stage-B。"
    )
    text = [
        "# 第三部分第一轮 Stage-A：测试集最终选型",
        "",
        f"corrected candidate 主口径最终选中 **{str(selected['model_variant']).upper()}**，"
        f"五场站宏平均 NRMSE=`{selected['macro_corrected_candidate_test_nrmse']:.9f}`，"
        f"NMAE=`{selected['macro_corrected_candidate_test_nmae']:.9f}`。",
        "",
        stage_b_text,
        "",
        "> 本轮遵照实验要求使用当前测试集选型，但该测试集已参与历史开发；"
        "协议固定标记为 `legacy_seen_test_selected`，不是最终盲测。",
        "> Stage-B 解锁只作为后续工程实验的启动条件；由于它同样由已见测试集"
        "计算，论文中不得表述为预注册确认性门槛已经通过。",
        "",
        "## A0--A5 candidate 主口径与 frozen-G0 诊断",
        "",
        comparison[columns].to_markdown(index=False),
        "",
        "## 相对 A0 的宏平均配对结果",
        "",
        macro_pair.to_markdown(index=False),
        "",
        "## 选型与 Stage-B 门槛",
        "",
        "- 数值最低 corrected-candidate NRMSE 单独保留；新变体还必须通过稳健"
        "选型门槛，否则正式结果回退 A0。通过者按 candidate NRMSE 排序，最优值"
        "的0.1%带内依次以 candidate NMAE、训练wrapper参数量、推理参数量和"
        "推理耗时打破平局。",
        "- 稳健选型要求 macro candidate NRMSE严格改善、macro candidate NMAE"
        "不退化、至少4/5场站candidate NRMSE不退化且至少3/5严格改善、至少4/5"
        "场站candidate NMAE不退化；dynamic/ramp-up/ramp-down/"
        "change_ge_20退化不超过0.5%，且不得连续3个及以上horizon系统退化超过0.5%。",
        "- frozen-G0 fused 只用于诊断旧门控能否转化candidate收益，不参与Stage-A"
        "正式选型。candidate改变后尚未重算soft oracle/Q90或重校准gate，选中项"
        "必须进入Stage-C闭环后才具备部署资格。",
        "- Stage-B 要求：5/5场站 candidate NRMSE严格改善、5/5 candidate NMAE"
        "不退化；macro candidate NRMSE/NMAE均至少改善0.5%；candidate在dynamic/"
        "ramp-up/ramp-down/change_ge_20 与16个horizon均不退化。",
        "- A0来自Stage-4B formal D0的hash校验只读引用；本脚本未训练、未复制、"
        "未重新forward A0。",
        "- A0按要求不续训，因此当前没有同训练预算的base fine-tune control；"
        "A1--A5相对A0的增益不能全部归因于新增目标。模块净效应应主要依靠"
        "A2/A3/A4/A5同父快照的受控差分；一区投稿前若需纯因果证据，应补A0R。",
        "",
    ]
    return _atomic_text(
        "\n".join(text),
        os.path.join(output_dir, "stage_a_test_final_selection.md"),
    )


def save_aggregate_tables(frames, output_dir):
    names = {
        "summary": "stage_a_test_summary.csv",
        "horizon": "stage_a_test_horizon.csv",
        "candidate": "stage_a_test_candidate.csv",
        "regime": "stage_a_test_regime.csv",
        "assignments": "stage_a_test_assignments.csv",
        "safety": "stage_a_test_safety.csv",
        "calibration": "stage_a_test_calibration.csv",
    }
    return {
        key: _atomic_csv(frames[key], os.path.join(output_dir, filename))
        for key, filename in names.items()
    }


def build_visual_inventory(summary, figures):
    rows = [
        {"scope": "aggregate", "model_variant": "all", "farm_id": "all", "kind": key, "path": path}
        for key, path in figures.items()
        if path
    ]
    for _, row in summary.iterrows():
        for key in ("single_window_figure_path", "weighted_curve_figure_path"):
            path = row.get(key)
            if isinstance(path, str) and path:
                rows.append(
                    {
                        "scope": "per_farm",
                        "model_variant": row["model_variant"],
                        "farm_id": str(row["farm_id"]),
                        "kind": key,
                        "path": path,
                    }
                )
    return pd.DataFrame(rows)


def publish_formal_marker(
    output_dir,
    table_paths,
    comparison_path,
    pairwise_path,
    complexity_path,
    selection_path,
    report_path,
    inventory_path,
    figure_paths,
    summary,
    selected,
    stage_b_variant,
    source_marker_path,
    training_marker_path,
    test_files,
):
    files = {
        "prediction_code": _file_record(__file__),
        "training_code": _file_record(stage_a_train.__file__),
        "source_stage4b_prediction_marker": _file_record(source_marker_path),
        "training_marker": _file_record(training_marker_path),
        "formal.comparison": _file_record(comparison_path),
        "formal.pairwise_vs_a0": _file_record(pairwise_path),
        "formal.complexity": _file_record(complexity_path),
        "formal.final_selection": _file_record(selection_path),
        "formal.report": _file_record(report_path),
        "formal.visual_inventory": _file_record(inventory_path),
    }
    for key, path in table_paths.items():
        files[f"formal.{key}"] = _file_record(path)
    for key, path in figure_paths.items():
        files[f"formal.{key}"] = _file_record(path)
    new_rows = summary[summary["model_variant"].isin(NEW_VARIANTS)]
    for _, row in new_rows.iterrows():
        prefix = f"{row['model_variant']}.{row['farm_id']}"
        for key in (
            "model_path",
            "artifact_path",
            "prediction_path",
            "horizon_metric_path",
            "candidate_metric_path",
            "regime_metric_path",
            "regime_assignment_path",
            "safety_diagnostics_path",
            "calibration_path",
            "gate_points_path",
            "candidate_archive_path",
            "single_window_path",
            "single_window_figure_path",
            "weighted_curve_path",
            "weighted_curve_figure_path",
        ):
            path = row.get(key)
            if isinstance(path, str) and path:
                files[f"{prefix}.{key}"] = _file_record(path)
    test_records = {
        farm_id: _file_record(path) for farm_id, path in test_files.items()
    }
    marker = {
        "status": "complete",
        "protocol_version": stage_a_train.PROTOCOL_VERSION,
        "architecture_version": stage_a_train.ARCHITECTURE_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": int(stage_a_train.RANDOM_SEED),
        "selection_split": "test",
        "test_used_for_selection": True,
        "test_reuse_status": TEST_REUSE_STATUS,
        "test_is_final_blind_evaluation": False,
        "selection_metric_scope": "corrected_candidate_forecast",
        "fused_role": "frozen_g0_diagnostic_only",
        "candidate_specific_gate_recalibrated": False,
        "stage_c_gate_closure_required_before_deployment": True,
        "selected_model_deployment_eligible": False,
        "continued_training_control_present": False,
        "variants": list(EXPECTED_VARIANTS),
        "expected_farm_ids": list(_expected_farms()),
        "a0_policy": (
            "direct_stage4b_formal_d0_reference_no_training_no_forward_no_copy"
        ),
        "a0_reference_forward_count": 0,
        "a0_retraining_forbidden": True,
        "a0_copy_forbidden": True,
        "new_prediction_count": int(len(new_rows)),
        "selected_variant": str(selected["model_variant"]),
        "selected_variant_robust_selection_pass": bool(
            selected["robust_selection_pass"]
        ),
        "numerically_lowest_candidate_nrmse_variant": str(
            summary.groupby("model_variant")["corrected_candidate_nrmse"]
            .mean()
            .idxmin()
        ),
        "numerically_lowest_frozen_g0_fused_nrmse_variant": str(
            summary.groupby("model_variant")["fused_test_nrmse"].mean().idxmin()
        ),
        "stage_b_unlocked": stage_b_variant is not None,
        "stage_b_recommended_variant": stage_b_variant,
        "stage_b_gate_computed_on_seen_test_not_confirmatory": True,
        "test_files": test_records,
        "files": files,
    }
    path = _atomic_json(marker, os.path.join(output_dir, FORMAL_MARKER_NAME))
    running = os.path.join(output_dir, RUNNING_MARKER_NAME)
    if os.path.isfile(running):
        os.remove(running)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default=",".join(ALL_VARIANTS),
        help="逗号分隔：a0,a1,a2,a3,a4,a5",
    )
    parser.add_argument(
        "--farms",
        default="",
        help="逗号分隔场站ID；空值为正式5场站",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="仅用于partial小样本检查；A0不可与截断样本混比",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args(argv)


def _discover_test_files(requested_farms, source_marker):
    discovered = {}
    for path in common_predict.discover_test_files():
        farm_id = str(common_predict.get_farm_id(path))
        if farm_id in requested_farms:
            if farm_id in discovered:
                raise ValueError(f"测试场站文件重复: {farm_id}")
            discovered[farm_id] = path
    if set(discovered) != set(requested_farms):
        raise FileNotFoundError(
            f"测试文件集合不完整: {set(discovered)} != {set(requested_farms)}"
        )
    source_records = source_marker.get("test_files", {})
    for farm_id, path in discovered.items():
        locked = source_records.get(farm_id)
        locked_path = _validate_record(f"A0 source test/{farm_id}", locked)
        if os.path.realpath(path) != os.path.realpath(locked_path):
            raise ValueError(f"{farm_id}当前测试CSV与A0 marker路径不一致")
    return discovered


def main(argv=None):
    args = parse_args(argv)
    configure_prediction_reproducibility()
    if tuple(ALL_VARIANTS) != EXPECTED_VARIANTS:
        raise ValueError(
            f"训练端A矩阵漂移: {ALL_VARIANTS} != {EXPECTED_VARIANTS}"
        )
    variants = _parse_list(args.variants, ALL_VARIANTS, "variants")
    farms = (
        _parse_list(args.farms, _expected_farms(), "farms")
        if args.farms
        else list(_expected_farms())
    )
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples必须为正")
    if args.max_samples is not None and A0_VARIANT in variants:
        raise ValueError(
            "A0只读引用完整Stage-4B结果，不能与--max-samples截断结果混比；"
            "小样本检查请显式使用--variants a1（或其它新增变体）"
        )
    full = bool(
        set(variants) == set(EXPECTED_VARIANTS)
        and set(farms) == set(_expected_farms())
        and args.max_samples is None
        and not args.skip_plots
    )
    output_root = (
        stage_a_train.RESULT_ROOT
        if full
        else os.path.join(
            stage_a_train.RESULT_ROOT,
            "partial_runs",
            args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
        )
    )
    output_dir = os.path.join(output_root, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)
    if full:
        _atomic_json(
            {
                "status": "running",
                "protocol_version": stage_a_train.PROTOCOL_VERSION,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "variants": variants,
                "farm_ids": farms,
            },
            os.path.join(output_dir, RUNNING_MARKER_NAME),
        )

    source_marker, source_frames, source_paths = validate_a0_source_bundle()
    requested_new = [variant for variant in variants if variant in NEW_VARIANTS]
    training_marker_path, training_marker = validate_training_bundle(requested_new)
    test_files = _discover_test_files(farms, source_marker)
    print(
        f"Stage-A预测 variants={variants}; farms={farms}; full={full}; "
        f"output={output_root}"
    )
    if A0_VARIANT in variants:
        print("A0：hash校验只读引用Stage-4B formal D0；不训练、不复制、不forward")

    collected = {key: [] for key in source_frames}
    if A0_VARIANT in variants:
        for key, frame in source_frames.items():
            collected[key].append(
                frame[frame["farm_id"].astype(str).isin(set(farms))].copy()
            )
    for farm_id in farms:
        test_file = test_files[farm_id]
        for variant in requested_new:
            print(f"\n===== Stage-A预测 {variant.upper()} farm={farm_id} =====")
            payload = predict_variant(
                variant, test_file, training_marker, max_samples=args.max_samples
            )
            result = save_payload(payload, output_root, skip_plots=args.skip_plots)
            for key in collected:
                collected[key].append(result[key])

    frames = {
        key: pd.concat(values, ignore_index=True) if values else pd.DataFrame()
        for key, values in collected.items()
    }
    table_paths = save_aggregate_tables(frames, output_dir)
    print(f"Stage-A聚合表已保存: {table_paths}")
    if not full:
        partial_manifest = {
            "status": "partial",
            "protocol_version": stage_a_train.PROTOCOL_VERSION,
            "variants": variants,
            "farm_ids": farms,
            "max_samples": args.max_samples,
            "skip_plots": bool(args.skip_plots),
            "formal_marker_published": False,
            "tables": {key: _file_record(path) for key, path in table_paths.items()},
        }
        path = _atomic_json(
            partial_manifest, os.path.join(output_dir, "stage_a_test_partial_manifest.json")
        )
        print(f"partial运行完成，不覆盖formal bundle: {path}")
        return

    validate_complete_matrix(frames)
    pairwise = build_pairwise_vs_a0(frames["summary"])
    comparison = build_comparison(
        frames["summary"],
        frames["candidate"],
        frames["regime"],
        pairwise,
    )
    selected, stage_b_variant, comparison = select_model(comparison)
    complexity = build_complexity(frames["summary"])
    comparison_path = _atomic_csv(
        comparison, os.path.join(output_dir, "stage_a_test_variant_comparison.csv")
    )
    pairwise_path = _atomic_csv(
        pairwise, os.path.join(output_dir, "stage_a_test_pairwise_vs_a0.csv")
    )
    complexity_path = _atomic_csv(
        complexity, os.path.join(output_dir, "stage_a_test_complexity.csv")
    )
    selection = comparison[comparison["selected"]].copy()
    selection_path = _atomic_csv(
        selection, os.path.join(output_dir, "stage_a_test_final_selection.csv")
    )
    figures = save_aggregate_figures(
        comparison,
        frames["summary"],
        frames["candidate"],
        frames["regime"],
        pairwise,
        complexity,
        output_dir,
    )
    report_path = write_report(
        comparison, selected, stage_b_variant, pairwise, output_dir
    )
    inventory = build_visual_inventory(frames["summary"], figures)
    inventory_path = _atomic_csv(
        inventory, os.path.join(output_dir, "stage_a_visual_inventory.csv")
    )
    reuse_manifest = pd.DataFrame(
        [
            {
                "model_variant": A0_VARIANT,
                "source_model_variant": SOURCE_VARIANT,
                "source_marker_path": os.path.abspath(SOURCE_MARKER),
                "source_marker_sha256": _sha256(SOURCE_MARKER),
                "source_table": key,
                "source_path": path,
                "source_sha256": _sha256(path),
                "training_count": 0,
                "forward_count": 0,
                "source_model_or_prediction_artifact_copy_count": 0,
                "metric_rows_materialized_into_stage_a_tables": True,
            }
            for key, path in source_paths.items()
        ]
    )
    reuse_path = _atomic_csv(
        reuse_manifest, os.path.join(output_dir, "stage_a_a0_source_reuse_manifest.csv")
    )
    table_paths["a0_source_reuse_manifest"] = reuse_path
    marker_path = publish_formal_marker(
        output_dir,
        table_paths,
        comparison_path,
        pairwise_path,
        complexity_path,
        selection_path,
        report_path,
        inventory_path,
        figures,
        frames["summary"],
        selected,
        stage_b_variant,
        SOURCE_MARKER,
        training_marker_path,
        test_files,
    )
    print(
        f"Stage-A正式测试bundle完成: {marker_path}\n"
        f"最终选中: {str(selected['model_variant']).upper()} | "
        "macro corrected-candidate NRMSE="
        f"{selected['macro_corrected_candidate_test_nrmse']:.9f}\n"
        "该candidate尚需Stage-C重新校准gate后才能部署。\n"
        f"Stage-B unlocked={stage_b_variant is not None}; "
        f"recommended={stage_b_variant}"
    )


if __name__ == "__main__":
    main()
