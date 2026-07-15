"""Stage-4B D0/D0R/D1/D2/D3 门控收益转化闭环：测试集预测、审计与选型。

正式矩阵只有五项：

* D0：只读、hash 校验地引用 Stage-4 T0，不训练、不 forward、不复制 artifact；
* D0R：冻结 T1 candidate + 新训练非因子化 direct gate；
* D1：冻结 F7 candidate + 非因子化 calibrated-safe gate；
* D2：冻结 T1 candidate + 非因子化 calibrated-safe gate；
* D3：冻结 T1 candidate + 因子化 calibrated-dynamic-safe gate。

fixed-G0-on-T1 仅是 D0R 的额外反事实回放诊断，不是第六个正式变体。
每个变体必须覆盖固定 5 场站；正式结果按当前测试集选择，因此标记为
``legacy_seen_test_selected``，不宣称为最终盲测。

默认执行完整 4x5 新模型预测并合并 D0 只读引用：

    python wind_time_freq_model_stage4b_predict.py

smoke/partial 会写入隔离目录，不发布正式 complete marker。
"""

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
import wind_time_freq_model_predict as stage4_predict
import wind_time_freq_model_stage4b_train as stage4b_train


OUTPUT_SUBDIR = "testdata_predict_output"
TEST_REUSE_STATUS = "legacy_seen_test_selected"
FORMAL_MARKER_NAME = "stage4b_gate_closure_test_bundle_complete.json"
ALL_VARIANTS = tuple(stage4b_train.VARIANT_SPECS)
NEW_VARIANTS = tuple(stage4b_train.TRAINABLE_VARIANTS)

SOURCE_ROOT = os.path.join("./wind_results", "time_freq_model")
SOURCE_OUTPUT = os.path.join(SOURCE_ROOT, OUTPUT_SUBDIR)
SOURCE_MARKER = os.path.join(SOURCE_OUTPUT, "time_freq_model_test_bundle_complete.json")
SOURCE_FILES = {
    "summary": "time_freq_model_test_summary.csv",
    "horizon": "time_freq_model_test_horizon.csv",
    "candidate": "time_freq_model_test_candidate.csv",
    "regime": "time_freq_model_test_regime.csv",
    "assignments": "time_freq_model_test_assignments.csv",
    "safety": "time_freq_model_test_safety.csv",
    "calibration": "time_freq_model_test_calibration.csv",
}

# 正式晋级条件（相对 D0）。
REQUIRED_MACRO_IMPROVEMENT = 0.002
NRMSE_TIE_TOL = 0.001
FARM_NONDEGRADE_ATOL = 1e-12
MIN_NONDEGRADED_FARMS = 4
MIN_STRICTLY_IMPROVED_FARMS = 3
RAMP_DEGRADATION_TOL = 0.005
CANDIDATE_OVERALL_TOL = 0.002
CANDIDATE_REGIME_TOL = 0.005
SAFETY_REGRET_TOL = 0.005
SAFETY_HARM_ABS_TOL = 0.002
PARAMETER_LIMIT = 30_000

PERSISTENCE_MAX_NORM_TOL = 1e-6
CANDIDATE_MAX_NORM_TOL = 1e-4
CANDIDATE_MEAN_NORM_TOL = 1e-6

# 复用 Stage-4 已经充分测试的原子写、逐场保存和评价函数。这些
# 函数在运行时查询模块全局变量，因此显式绑定 Stage-4B 训练模块；
# 不会改动 Stage-4 磁盘产物。
if not hasattr(stage4b_train, "get_time_freq_custom_objects"):
    stage4b_train.get_time_freq_custom_objects = (
        stage4b_train.get_stage4b_custom_objects
    )
if not hasattr(stage4b_train, "EXPECTED_ADAPTER_TRAINABLE_PARAMS"):
    stage4b_train.EXPECTED_ADAPTER_TRAINABLE_PARAMS = {
        variant: 0 for variant in ALL_VARIANTS
    }
stage4_predict.tf_train = stage4b_train
stage4_predict.ALL_VARIANTS = ALL_VARIANTS
stage4_predict.NEW_VARIANTS = NEW_VARIANTS
stage4_predict.TEST_REUSE_STATUS = TEST_REUSE_STATUS
stage4_predict.PERSISTENCE_CONTROL_MAX_NORM_TOL = PERSISTENCE_MAX_NORM_TOL

_sha256 = stage4_predict._sha256
_file_record = stage4_predict._file_record
_atomic_csv = stage4_predict._atomic_csv
_atomic_json = stage4_predict._atomic_json
_atomic_text = stage4_predict._atomic_text
_atomic_npz = stage4_predict._atomic_npz
_validate_record = stage4_predict._validate_record


def _relabel_reference(frame, key):
    """Filter immutable T0 rows and expose them as Stage-4B D0."""
    if "model_variant" not in frame:
        raise KeyError(f"Stage-4 source {key}缺少model_variant")
    frame = frame[frame["model_variant"].astype(str) == "t0"].copy()
    frame["source_model_family"] = "time_freq_model"
    frame["source_model_variant"] = "t0"
    frame["model_family"] = stage4b_train.MODEL_FAMILY
    frame["model_variant"] = "d0"
    if "model_name" in frame:
        frame["model_name"] = stage4b_train.variant_model_name("d0")
    if key == "summary":
        frame["variant_label"] = stage4b_train.VARIANT_SPECS["d0"]["label"]
        frame["result_source"] = (
            "hash_validated_stage4_t0_direct_reference_no_training_no_forward_no_copy"
        )
        frame["reference_only"] = True
        frame["selection_eligible"] = True
        frame["test_reuse_status"] = TEST_REUSE_STATUS
        frame["selection_split"] = "test"
        frame["test_used_for_selection"] = True
        frame["test_is_final_blind_evaluation"] = False
        frame["adapter_trainable_parameter_count"] = 0
    return frame


def validate_source_bundle():
    """Hash-validate Stage-4 bundle; return D0 frames and source T1 rows."""
    if not os.path.isfile(SOURCE_MARKER):
        raise FileNotFoundError(f"缺少Stage-4 complete marker: {SOURCE_MARKER}")
    with open(SOURCE_MARKER, "r", encoding="utf-8") as file:
        marker = json.load(file)
    if marker.get("status") != "complete":
        raise ValueError("Stage-4 source bundle不是complete")
    expected_farms = set(stage4b_train.expected_farm_ids())
    if set(map(str, marker.get("test_files", {}))) != expected_farms:
        raise ValueError("Stage-4 source marker未锁定完整5场站")
    for key, record in marker.get("files", {}).items():
        _validate_record(f"Stage-4 source files.{key}", record)
    for farm_id, record in marker.get("test_files", {}).items():
        _validate_record(f"Stage-4 source test.{farm_id}", record)

    frames, paths = {}, {}
    source_summary = None
    for key, filename in SOURCE_FILES.items():
        path = os.path.join(SOURCE_OUTPUT, filename)
        record = marker.get("files", {}).get(f"formal.{key}")
        if record is None:
            raise KeyError(f"Stage-4 marker缺少formal.{key}")
        validated = _validate_record(f"Stage-4 formal.{key}", record)
        if os.path.realpath(path) != os.path.realpath(validated):
            raise ValueError(f"Stage-4 formal.{key}路径不一致")
        full = pd.read_csv(path, dtype={"farm_id": str})
        if key == "summary":
            source_summary = full.copy()
        frames[key] = _relabel_reference(full, key)
        paths[key] = path
    stage4_predict._exact_five(
        frames["summary"], "D0 Stage-4 read-only reference", ("capacity_normalized_rmse",)
    )
    if source_summary is None:
        raise RuntimeError("Stage-4 source summary未加载")
    source_t1 = source_summary[source_summary["model_variant"].astype(str) == "t1"].copy()
    stage4_predict._exact_five(
        source_t1, "Stage-4 T1 candidate source", ("capacity_normalized_rmse",)
    )
    return marker, frames, source_t1, paths


def validate_training_bundle(required_variants):
    """Validate Stage-4B training marker without accepting partial runs."""
    if not required_variants:
        return None, None
    running_path = os.path.join(
        stage4b_train.RESULT_ROOT, stage4b_train.RUNNING_MARKER_NAME
    )
    if os.path.isfile(running_path):
        raise RuntimeError(
            f"Stage-4B训练仍在运行或上次未完整收尾，拒绝读取混合bundle: {running_path}"
        )
    path = os.path.join(stage4b_train.RESULT_ROOT, stage4b_train.TRAINING_MARKER_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少Stage-4B训练complete marker: {path}")
    with open(path, "r", encoding="utf-8") as file:
        marker = json.load(file)
    if marker.get("status") != "complete":
        raise ValueError("Stage-4B训练marker不是complete")
    if marker.get("protocol_version") != stage4b_train.PROTOCOL_VERSION:
        raise ValueError("Stage-4B训练marker协议不匹配")
    if marker.get("architecture_version") != stage4b_train.ARCHITECTURE_VERSION:
        raise ValueError("Stage-4B训练marker架构不匹配")
    expected = set(stage4b_train.expected_farm_ids())
    if set(map(str, marker.get("expected_farm_ids", ()))) != expected:
        raise ValueError("Stage-4B训练marker未覆盖固定5场站")
    for key, record in marker.get("files", {}).items():
        _validate_record(f"Stage-4B training files.{key}", record)
    for variant in required_variants:
        for farm_id in expected:
            for kind in ("model_path", "artifact_path"):
                if f"{variant}.{farm_id}.{kind}" not in marker.get("files", {}):
                    raise KeyError(f"训练marker缺少{variant}.{farm_id}.{kind}")
    return path, marker


def _load_model(variant, farm_id, marker):
    """Stage-4B-specific loader.

    D0R is intentionally a direct gate and has no soft-oracle/Q90 objective.  It
    must therefore carry an explicit N/A calibration record rather than a forged
    train-only Q90 vector.  D1--D3, in contrast, must carry a valid train-only
    per-horizon Q90 vector.
    """
    files = marker["files"]
    artifact_path = _validate_record(
        f"{variant}/{farm_id} artifact", files[f"{variant}.{farm_id}.artifact_path"]
    )
    artifact = joblib.load(artifact_path)
    stage4b_train.validate_dependency_code_records(
        artifact.get("dependency_code_records"),
        role=f"{variant}/{farm_id} prediction artifact",
    )
    q90_record = artifact.get("candidate_calibration", {})
    raw_q90 = q90_record.get("candidate_difference_q90", ())
    q90 = np.asarray(() if raw_q90 is None else raw_q90, dtype=float)
    checks = {
        "variant": artifact.get("variant_id") == variant,
        "farm": str(artifact.get("farm_id")) == str(farm_id),
        "family": artifact.get("model_family") == stage4b_train.MODEL_FAMILY,
        "architecture": artifact.get("architecture_version")
        == stage4b_train.ARCHITECTURE_VERSION,
        "protocol": artifact.get("protocol_version")
        == stage4b_train.PROTOCOL_VERSION,
        "schema": int(artifact.get("artifact_schema_version", -1))
        == stage4b_train.ARTIFACT_SCHEMA_VERSION,
        "seed": int(artifact.get("random_seed", -1)) == stage4b_train.RANDOM_SEED,
        "history": int(artifact.get("history_len", -1)) == stage4b_train.HISTORY_LEN,
        "forecast": int(artifact.get("forecast_len", -1))
        == stage4b_train.FORECAST_LEN,
        "params": int(artifact.get("total_params", -1))
        == stage4b_train.EXPECTED_TOTAL_PARAMS[variant],
        "candidate_weight_frozen": artifact.get(
            "candidate_snapshot_before_gate_sha256"
        )
        == artifact.get("candidate_snapshot_after_gate_sha256"),
        "candidate_output_frozen": artifact.get(
            "candidate_output_before_gate_sha256"
        )
        == artifact.get("candidate_output_after_gate_sha256"),
        "candidate_gate_drift_zero": float(
            artifact.get("candidate_gate_calibration_max_abs_drift", np.nan)
        )
        == 0.0,
        "candidate_frozen_all_phases": bool(
            artifact.get("gate_training", {}).get("candidate_frozen_all_phases")
        ),
    }
    scope = str(q90_record.get("scope", "")).lower()
    if variant == "d0r":
        checks["direct_gate_q90_explicitly_not_applicable"] = bool(
            q90.size == 0
            and q90_record.get("candidate_source") == "t1"
            and q90_record.get("candidate_snapshot_sha256")
            == artifact.get("candidate_snapshot_before_gate_sha256")
            and (
                "not_applicable" in scope
                or "not-applicable" in scope
                or scope in {"na", "n/a", "direct_gate_no_soft_oracle"}
            )
        )
    else:
        expected_source = "f7" if variant == "d1" else "t1"
        q90_float32 = q90.astype(np.float32, copy=False)
        checks["train_only_soft_oracle_q90"] = bool(
            q90.shape == (stage4b_train.FORECAST_LEN,)
            and np.isfinite(q90).all()
            and np.all(q90 >= 0.0)
            and "train" in scope
            and int(q90_record.get("sample_count", 0)) > 0
            and int(q90_record.get("element_count", -1))
            == int(q90_record.get("sample_count", 0))
            * stage4b_train.FORECAST_LEN
            and np.isclose(
                float(q90_record.get("quantile", np.nan)),
                stage4b_train.CALIBRATION_DIFFERENCE_QUANTILE,
                rtol=0.0,
                atol=1e-12,
            )
            and q90_record.get("candidate_source") == expected_source
            and q90_record.get("candidate_snapshot_sha256")
            == artifact.get("candidate_snapshot_before_gate_sha256")
            and q90_record.get("candidate_difference_q90_sha256")
            == stage4b_train._array_sha256([("q90", q90_float32)])
        )
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
        custom_objects=stage4b_train.get_stage4b_custom_objects(),
        compile=False,
    )
    if int(model.count_params()) != stage4b_train.EXPECTED_TOTAL_PARAMS[variant]:
        raise ValueError(f"{variant}/{farm_id}加载后参数量漂移")
    if int(model.count_params()) >= PARAMETER_LIMIT:
        raise ValueError(f"{variant}/{farm_id}超过30k参数上限")
    return artifact, artifact_path, model, model_path


def _normalize_output(value, shape, label):
    value = np.asarray(value, dtype=np.float64)
    if value.shape == (shape[0], 1):
        value = np.repeat(value, shape[1], axis=1)
    elif value.shape == (shape[1],):
        value = np.repeat(value[None, :], shape[0], axis=0)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{label} shape/finite异常: {value.shape} != {shape}")
    return value


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
    diagnostic = stage4b_train.diagnostic_model(model)
    started = time.perf_counter()
    outputs = diagnostic.predict(dataset, verbose=common_predict.PREDICT_VERBOSE)
    elapsed = float(time.perf_counter() - started)
    required = ("forecast", "persistence", "corrected", "gate", "q", "s")
    if not isinstance(outputs, dict) or any(key not in outputs for key in required):
        missing = set(required) - set(outputs if isinstance(outputs, dict) else ())
        raise TypeError(f"{variant}/{farm_id} diagnostic输出不完整: {missing}")
    shape = (n_samples, forecast_len)
    outputs = {
        key: _normalize_output(outputs[key], shape, f"{variant}/{farm_id}/{key}")
        for key in required
    }
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
            "result_source": "stage4b_single_formal_test_forward",
            "diagnostic_source": "same_forward_as_forecast",
            "inference_elapsed_seconds": elapsed,
            "inference_milliseconds_per_sample": 1000.0 * elapsed / n_samples,
        }
    )
    return payload


def save_payload(payload, output_root, skip_plots=False):
    result = stage4_predict.save_payload(payload, output_root, skip_plots)
    result["summary"]["result_source"] = "stage4b_single_formal_test_forward"
    result["summary"]["selection_eligible"] = True
    variant = str(payload["variant_id"])
    structural_adapter = 3152 if variant in {"d0r", "d2", "d3"} else 0
    # T1 adapter is structurally present but frozen throughout this round.  Keep
    # structural capacity separate from the zero adapter parameters trained here.
    result["summary"]["candidate_adapter_structural_parameter_count"] = (
        structural_adapter
    )
    result["summary"]["adapter_trainable_parameter_count"] = 0
    archive_path = result["paths"]["archive"]
    _atomic_npz(
        archive_path,
        schema_version=np.asarray("stage4b_gate_closure_candidate_archive_v1"),
        model_variant=np.asarray(variant),
        farm_id=np.asarray(payload["farm_id"]),
        candidate_source=np.asarray(payload["artifact"]["candidate_source"]),
        sample_id=payload["sample_id"],
        horizon_step=payload["horizon_step"],
        forecast_origin_time=payload["forecast_origin_time"],
        capacity=np.asarray(payload["capacity"]),
        y=payload["y_true"],
        P=payload["persistence"],
        C=payload["corrected"],
        F=payload["fused"],
        y_true=payload["y_true"],
        persistence=payload["persistence"],
        corrected=payload["corrected"],
        fused=payload["fused"],
        persistence_scaled=payload["persistence_scaled"],
        corrected_scaled=payload["corrected_scaled"],
        fused_scaled=payload["fused_scaled"],
        raw_gate=payload["raw_gate"],
        applied_gate=payload["applied_gate"],
        q=payload["q"],
        s=payload["s"],
    )
    result["summary"]["candidate_archive_sha256"] = _sha256(archive_path)
    return result


def _read_archive(row, label):
    path = row.get("candidate_archive_path")
    if not isinstance(path, str) or not os.path.isfile(path):
        raise FileNotFoundError(f"{label}缺少candidate archive: {path}")
    if _sha256(path) != row.get("candidate_archive_sha256"):
        raise ValueError(f"{label} candidate archive hash漂移")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def build_fixed_g0_replay(d0_summary, d0r_results, output_root, skip_plots=False):
    """Replay immutable D0/G0 gate on the newly reconstructed T1 candidate.

    The replay is diagnostic only.  It is saved under a dedicated subtree and is
    never appended to the five-variant formal selection frames.
    """
    source_rows = {
        str(row["farm_id"]): row for _, row in d0_summary.iterrows()
    }
    source = {
        farm_id: _read_archive(row, f"D0/{farm_id}")
        for farm_id, row in source_rows.items()
    }
    root = os.path.join(output_root, "d0r_fixed_g0_replay_diagnostic")
    os.makedirs(root, exist_ok=True)
    rows, paths, payloads = [], {}, []
    for result in d0r_results:
        target = result["payload"]
        farm_id = str(target["farm_id"])
        baseline = source[farm_id]
        source_row = source_rows[farm_id]
        for key in ("sample_id", "horizon_step", "forecast_origin_time"):
            if not np.array_equal(baseline[key], target[key]):
                raise ValueError(f"fixed-G0 replay/{farm_id} {key}未对齐")
        if not np.array_equal(baseline["y_true"], target["y_true"], equal_nan=True):
            raise ValueError(f"fixed-G0 replay/{farm_id}真值未对齐")
        baseline_capacity = float(np.asarray(baseline["capacity"]).reshape(-1)[0])
        if not np.isclose(
            baseline_capacity,
            float(target["capacity"]),
            rtol=1e-10,
            atol=1e-8,
        ):
            raise ValueError(f"fixed-G0 replay/{farm_id}容量不一致")
        gate_key = "applied_gate" if "applied_gate" in baseline else "raw_gate"
        gate = np.asarray(baseline[gate_key], dtype=np.float64)
        if gate.shape != target["corrected_scaled"].shape:
            raise ValueError(f"fixed-G0 replay/{farm_id} gate shape不匹配")
        fused_scaled = target["persistence_scaled"] + gate * (
            target["corrected_scaled"] - target["persistence_scaled"]
        )
        if "q" in baseline and np.asarray(baseline["q"]).shape == gate.shape:
            q = np.asarray(baseline["q"], dtype=np.float64)
        else:
            q = np.repeat(
                np.mean(gate, axis=1, keepdims=True), gate.shape[1], axis=1
            )
        s = (
            np.asarray(baseline["s"], dtype=np.float64)
            if "s" in baseline and np.asarray(baseline["s"]).shape == gate.shape
            else np.ones_like(gate)
        )
        outputs = {
            "forecast": fused_scaled,
            "persistence": target["persistence_scaled"],
            "corrected": target["corrected_scaled"],
            "gate": gate,
            "q": q,
            "s": s,
        }
        replay = gate_predict._build_payload(
            "d0r_fixed_g0_replay",
            farm_id,
            target["df"],
            target["artifact"],
            outputs,
            target["y_true"],
            target["capacity"],
            target["history_len"],
        )
        payloads.append(replay)
        model_name = "stage4b_d0r_fixed_g0_on_t1_replay"
        prediction = common_predict.build_prediction_frame(
            model_name,
            replay["df"],
            farm_id,
            replay["fused"],
            replay["y_true"],
            replay["history_len"],
            replay["forecast_len"],
        )
        pred_dir = os.path.join(root, "predictions")
        archive_dir = os.path.join(root, "candidate_archives")
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(archive_dir, exist_ok=True)
        pred_path = _atomic_csv(
            prediction, os.path.join(pred_dir, f"fixed_g0_replay_farm_{farm_id}.csv")
        )
        archive_path = _atomic_npz(
            os.path.join(archive_dir, f"fixed_g0_replay_archive_farm_{farm_id}.npz"),
            schema_version=np.asarray("stage4b_fixed_g0_replay_v1"),
            model_variant=np.asarray("d0r_fixed_g0_replay"),
            farm_id=np.asarray(farm_id),
            sample_id=replay["sample_id"],
            horizon_step=replay["horizon_step"],
            forecast_origin_time=replay["forecast_origin_time"],
            y_true=replay["y_true"],
            y=replay["y_true"],
            persistence=replay["persistence"],
            P=replay["persistence"],
            corrected=replay["corrected"],
            C=replay["corrected"],
            fused=replay["fused"],
            F=replay["fused"],
            persistence_scaled=replay["persistence_scaled"],
            corrected_scaled=replay["corrected_scaled"],
            fused_scaled=replay["fused_scaled"],
            fixed_g0_gate=gate,
            raw_gate=gate,
            applied_gate=gate,
            q=q,
            s=s,
            capacity=np.asarray(replay["capacity"]),
        )
        horizon = common_predict.metrics_by_horizon(
            model_name,
            farm_id,
            replay["y_true"],
            replay["fused"],
            replay["capacity"],
            replay["forecast_len"],
        )
        horizon["diagnostic_variant"] = "fixed_g0_on_t1_replay"
        horizon_path = _atomic_csv(
            horizon, os.path.join(root, f"fixed_g0_replay_horizon_farm_{farm_id}.csv")
        )
        regimes = gate_predict._regime_metrics(replay)
        regimes["model_family"] = stage4b_train.MODEL_FAMILY
        regimes["model_variant"] = "d0r_fixed_g0_replay"
        regimes["diagnostic_variant"] = "fixed_g0_on_t1_replay"
        regime_path = _atomic_csv(
            regimes, os.path.join(root, f"fixed_g0_replay_regime_farm_{farm_id}.csv")
        )
        safety = gate_predict.build_safety_scope_frame(replay)
        safety["model_family"] = stage4b_train.MODEL_FAMILY
        safety["model_variant"] = "d0r_fixed_g0_replay"
        safety["diagnostic_variant"] = "fixed_g0_on_t1_replay"
        safety_path = _atomic_csv(
            safety, os.path.join(root, f"fixed_g0_replay_safety_farm_{farm_id}.csv")
        )
        calibration = gate_predict.build_reliability_frame(replay)
        calibration["model_family"] = stage4b_train.MODEL_FAMILY
        calibration["model_variant"] = "d0r_fixed_g0_replay"
        calibration["diagnostic_variant"] = "fixed_g0_on_t1_replay"
        calibration_path = _atomic_csv(
            calibration,
            os.path.join(root, f"fixed_g0_replay_calibration_farm_{farm_id}.csv"),
        )
        overall = horizon[horizon["horizon_step"].astype(str) == "all"].iloc[0]
        utility = safety[
            (safety["scope_type"] == "overall")
            & (safety["scope_value"].astype(str) == "all")
        ].iloc[0]
        row = {**overall.to_dict()}
        for column in (
            "positive_regret_mean",
            "harm_rate_0_005",
            "oracle_brier",
            "ece_10bin",
        ):
            row[column] = utility[column]
        row.update(
            {
                "diagnostic_variant": "fixed_g0_on_t1_replay",
                "formal_selection_eligible": False,
                "prediction_path": pred_path,
                "candidate_archive_path": archive_path,
                "d0_gate_source_archive_path": source_row[
                    "candidate_archive_path"
                ],
                "d0_gate_source_archive_sha256": source_row[
                    "candidate_archive_sha256"
                ],
                "horizon_path": horizon_path,
                "regime_path": regime_path,
                "safety_path": safety_path,
                "calibration_path": calibration_path,
            }
        )
        rows.append(row)
        for key, value in {
            "prediction": pred_path,
            "archive": archive_path,
            "horizon": horizon_path,
            "regime": regime_path,
            "safety": safety_path,
            "calibration": calibration_path,
        }.items():
            paths[f"{farm_id}.{key}"] = value
        if not skip_plots:
            dirs = stage4_predict.prediction_dirs(
                "d0r_fixed_g0_replay_diagnostic", output_root
            )
            _, single = common_predict.save_single_window_plot(
                prediction, model_name, farm_id, dirs, replay["forecast_len"]
            )
            _, weighted, _ = common_predict.save_weighted_full_test_plot(
                prediction, model_name, farm_id, dirs, replay["capacity"]
            )
            paths[f"{farm_id}.single_figure"] = single
            paths[f"{farm_id}.weighted_figure"] = weighted
    summary = pd.DataFrame(rows)
    stage4_predict._exact_five(
        summary,
        "fixed-G0-on-T1 replay",
        ("capacity_normalized_rmse", "positive_regret_mean", "harm_rate_0_005"),
    )
    summary_path = _atomic_csv(
        summary, os.path.join(root, "fixed_g0_on_t1_replay_summary.csv")
    )
    paths["summary"] = summary_path
    return summary, payloads, paths


def _exact_five(frame, label, columns):
    return stage4_predict._exact_five(frame, label, columns)


def validate_complete_output_matrix(frames):
    """Validate five variants x five farms without Stage-4's hard-coded T0 key."""
    expected_variants = set(ALL_VARIANTS)
    expected_farms = set(stage4b_train.expected_farm_ids())
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
        "safety": (
            "model_variant",
            "farm_id",
            "scope_type",
            "scope_value",
        ),
        "calibration": ("model_variant", "farm_id", "gate_bin"),
    }

    def canonical(value):
        if pd.isna(value):
            return "<NA>"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        return str(value)

    if set(frames) != set(natural_keys):
        raise ValueError(f"正式表集不完整: {set(frames)}")
    for key, frame in frames.items():
        keys = natural_keys[key]
        missing = set(keys) - set(frame.columns)
        if missing:
            raise KeyError(f"正式{key}缺少自然键: {sorted(missing)}")
        if set(frame["model_variant"].astype(str)) != expected_variants:
            raise ValueError(f"正式{key}未覆盖D0/D0R/D1/D2/D3")
        if set(frame["farm_id"].astype(str)) != expected_farms:
            raise ValueError(f"正式{key}未覆盖固定5场站")
        normalized = frame.copy()
        for column in keys:
            normalized[column] = normalized[column].map(canonical)
        if normalized.duplicated(list(keys)).any():
            raise ValueError(f"正式{key}自然键重复")
        suffix = [column for column in keys if column not in ("model_variant", "farm_id")]
        for farm_id in sorted(expected_farms):
            baseline = normalized[
                (normalized["model_variant"] == "d0")
                & (normalized["farm_id"] == farm_id)
            ]
            if baseline.empty:
                raise ValueError(f"正式{key}/D0/{farm_id}缺失")
            for variant in ALL_VARIANTS:
                target = normalized[
                    (normalized["model_variant"] == variant)
                    & (normalized["farm_id"] == farm_id)
                ]
                if len(target) != len(baseline):
                    raise ValueError(f"正式{key}/{variant}/{farm_id}行数与D0不同")
                if suffix and set(target[suffix].itertuples(index=False, name=None)) != set(
                    baseline[suffix].itertuples(index=False, name=None)
                ):
                    raise ValueError(f"正式{key}/{variant}/{farm_id}自然键集与D0不同")
    summary = frames["summary"]
    if len(summary) != len(ALL_VARIANTS) * len(expected_farms):
        raise ValueError("正式summary不是5变体x5场站")


def _macro(frame, variant, column, **filters):
    part = frame[frame["model_variant"].astype(str) == variant]
    for key, value in filters.items():
        part = part[part[key].astype(str) == str(value)]
    part = _exact_five(part, f"{variant}/{filters}", (column,))
    return float(pd.to_numeric(part[column]).mean())


def build_comparison(summary, candidate, regime, replay_summary):
    rows = []
    for variant in ALL_VARIANTS:
        frame = _exact_five(
            summary[summary["model_variant"].astype(str) == variant],
            variant,
            (
                "capacity_normalized_rmse",
                "positive_regret_mean",
                "harm_rate_0_005",
                "oracle_brier",
                "ece_10bin",
                "parameter_count",
            ),
        )
        row = {
            "model_variant": variant,
            "variant_label": stage4b_train.VARIANT_SPECS[variant]["label"],
            "macro_test_nrmse": float(frame["capacity_normalized_rmse"].mean()),
            "macro_test_nmae": float(frame["capacity_normalized_mae"].mean()),
            "macro_positive_regret_mean": float(frame["positive_regret_mean"].mean()),
            "macro_harm_rate_0_005": float(frame["harm_rate_0_005"].mean()),
            "macro_oracle_brier": float(frame["oracle_brier"].mean()),
            "macro_ece_10bin": float(frame["ece_10bin"].mean()),
            "parameter_count_max": int(frame["parameter_count"].max()),
            "macro_inference_milliseconds_per_sample": float(
                pd.to_numeric(frame["inference_milliseconds_per_sample"]).mean()
            ),
            "corrected_overall_nrmse": _macro(
                candidate,
                variant,
                "capacity_normalized_rmse",
                candidate="corrected",
                horizon_step="all",
            ),
            "corrected_overall_nmae": _macro(
                candidate,
                variant,
                "capacity_normalized_mae",
                candidate="corrected",
                horizon_step="all",
            ),
        }
        for group in ("dynamic", "ramp_up", "ramp_down"):
            row[f"corrected_{group}_nrmse"] = _macro(
                regime,
                variant,
                "capacity_normalized_rmse",
                candidate="corrected",
                regime_group=group,
                horizon_step="all",
            )
            row[f"fused_{group}_nrmse"] = _macro(
                regime,
                variant,
                "capacity_normalized_rmse",
                candidate="fused",
                regime_group=group,
                horizon_step="all",
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    base = result[result["model_variant"] == "d0"].iloc[0]
    result["relative_macro_nrmse_vs_d0"] = (
        result["macro_test_nrmse"] / float(base["macro_test_nrmse"]) - 1.0
    )
    result["actual_macro_improvement_vs_d0"] = 1.0 - (
        result["macro_test_nrmse"] / float(base["macro_test_nrmse"])
    )
    result["fixed_g0_on_t1_replay_macro_nrmse"] = np.nan
    result["gate_retraining_effect_d0r_minus_replay"] = np.nan
    replay_nrmse = float(replay_summary["capacity_normalized_rmse"].mean())
    mask = result["model_variant"] == "d0r"
    result.loc[mask, "fixed_g0_on_t1_replay_macro_nrmse"] = replay_nrmse
    result.loc[mask, "gate_retraining_effect_d0r_minus_replay"] = (
        result.loc[mask, "macro_test_nrmse"] - replay_nrmse
    )

    farm_base = summary[summary["model_variant"] == "d0"].set_index("farm_id")[
        "capacity_normalized_rmse"
    ]
    flags = []
    for _, row in result.iterrows():
        variant = row["model_variant"]
        farm = summary[summary["model_variant"] == variant].set_index("farm_id")[
            "capacity_normalized_rmse"
        ].reindex(farm_base.index)
        nondegraded = int((farm <= farm_base + FARM_NONDEGRADE_ATOL).sum())
        improved = int((farm < farm_base - FARM_NONDEGRADE_ATOL).sum())
        macro = bool(
            row["macro_test_nrmse"]
            <= float(base["macro_test_nrmse"]) * (1.0 - REQUIRED_MACRO_IMPROVEMENT)
        )
        ramp = all(
            row[f"fused_{group}_nrmse"]
            <= float(base[f"fused_{group}_nrmse"])
            * (1.0 + RAMP_DEGRADATION_TOL)
            for group in ("ramp_up", "ramp_down")
        )
        candidate_overall = all(
            row[column] <= float(base[column]) * (1.0 + CANDIDATE_OVERALL_TOL)
            for column in ("corrected_overall_nrmse", "corrected_overall_nmae")
        )
        candidate_regime = all(
            row[f"corrected_{group}_nrmse"]
            <= float(base[f"corrected_{group}_nrmse"])
            * (1.0 + CANDIDATE_REGIME_TOL)
            for group in ("dynamic", "ramp_up", "ramp_down")
        )
        safety = bool(
            row["macro_positive_regret_mean"]
            <= float(base["macro_positive_regret_mean"])
            * (1.0 + SAFETY_REGRET_TOL)
            and row["macro_harm_rate_0_005"]
            <= float(base["macro_harm_rate_0_005"]) + SAFETY_HARM_ABS_TOL
        )
        parameter = bool(row["parameter_count_max"] < PARAMETER_LIMIT)
        guard = bool(
            macro
            and nondegraded >= MIN_NONDEGRADED_FARMS
            and improved >= MIN_STRICTLY_IMPROVED_FARMS
            and ramp
            and candidate_overall
            and candidate_regime
            and safety
            and parameter
        )
        if variant == "d0":
            guard = True
        flags.append(
            {
                "model_variant": variant,
                "macro_improves_at_least_0_2pct": macro,
                "farms_nondegraded_vs_d0": nondegraded,
                "at_least_4_farms_nondegraded": nondegraded
                >= MIN_NONDEGRADED_FARMS,
                "farms_strictly_improved_vs_d0": improved,
                "at_least_3_farms_strictly_improved": improved
                >= MIN_STRICTLY_IMPROVED_FARMS,
                "ramp_guard_pass": bool(ramp),
                "candidate_overall_guard_pass": bool(candidate_overall),
                "candidate_dynamic_ramp_guard_pass": bool(candidate_regime),
                "safety_guard_pass": safety,
                "parameter_under_30k": parameter,
                "selection_guard_pass": guard,
            }
        )
    return result.merge(pd.DataFrame(flags), on="model_variant", validate="one_to_one")


def select_model(comparison):
    """Choose among qualified models, using safety/complexity inside a 0.1% tie."""
    comparison = comparison.copy()
    lowest = comparison.sort_values("macro_test_nrmse", kind="stable").iloc[0]
    qualified = comparison[
        (comparison["model_variant"] != "d0")
        & comparison["selection_guard_pass"].astype(bool)
    ]
    if qualified.empty:
        selected = comparison[comparison["model_variant"] == "d0"].iloc[0]
        status = "fallback_d0_no_new_variant_passed_all_guards"
    else:
        best = float(qualified["macro_test_nrmse"].min())
        near = qualified[
            qualified["macro_test_nrmse"] <= best * (1.0 + NRMSE_TIE_TOL)
        ]
        selected = near.sort_values(
            [
                "macro_positive_regret_mean",
                "macro_oracle_brier",
                "parameter_count_max",
                "macro_inference_milliseconds_per_sample",
                "macro_test_nrmse",
            ],
            kind="stable",
        ).iloc[0]
        status = "qualified_nrmse_0_1pct_tie_then_safety_calibration_complexity"
    comparison["numerically_lowest"] = (
        comparison["model_variant"] == lowest["model_variant"]
    )
    comparison["selected"] = comparison["model_variant"] == selected["model_variant"]
    comparison["selection_status"] = status
    return comparison[comparison["selected"]].iloc[0], comparison


def validate_candidate_invariants(d0_summary, source_t1_summary, results):
    """Verify the two intended frozen candidate families against source archives."""
    d0 = {
        str(row["farm_id"]): _read_archive(row, f"D0/{row['farm_id']}")
        for _, row in d0_summary.iterrows()
    }
    t1 = {
        str(row["farm_id"]): _read_archive(row, f"source T1/{row['farm_id']}")
        for _, row in source_t1_summary.iterrows()
    }
    rows = []
    for result in results:
        payload = result["payload"]
        variant, farm_id = payload["variant_id"], str(payload["farm_id"])
        source_name = "stage4_t0_f7" if variant == "d1" else "stage4_t1"
        source = d0[farm_id] if variant == "d1" else t1[farm_id]
        for key in ("sample_id", "horizon_step", "forecast_origin_time"):
            if not np.array_equal(source[key], payload[key]):
                raise ValueError(f"{variant}/{farm_id} candidate {key}未对齐")
        if not np.array_equal(source["y_true"], payload["y_true"], equal_nan=True):
            raise ValueError(f"{variant}/{farm_id} candidate真值未对齐")
        capacity = float(payload["capacity"])
        persistence_drift = np.abs(
            np.asarray(payload["persistence"], float)
            - np.asarray(source["persistence"], float)
        ) / capacity
        corrected_drift = np.abs(
            np.asarray(payload["corrected"], float)
            - np.asarray(source["corrected"], float)
        ) / capacity
        row = {
            "model_variant": variant,
            "farm_id": farm_id,
            "expected_candidate_source": source_name,
            "persistence_capacity_normalized_max_abs_drift": float(
                np.max(persistence_drift)
            ),
            "persistence_capacity_normalized_mean_abs_drift": float(
                np.mean(persistence_drift)
            ),
            "corrected_capacity_normalized_max_abs_drift": float(
                np.max(corrected_drift)
            ),
            "corrected_capacity_normalized_mean_abs_drift": float(
                np.mean(corrected_drift)
            ),
            "persistence_scaled_exact": bool(
                np.array_equal(
                    source["persistence_scaled"],
                    payload["persistence_scaled"],
                    equal_nan=True,
                )
            ),
            "corrected_scaled_exact": bool(
                np.array_equal(
                    source["corrected_scaled"],
                    payload["corrected_scaled"],
                    equal_nan=True,
                )
            ),
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
        rows.append(row)
    frame = pd.DataFrame(rows)
    expected_rows = len(NEW_VARIANTS) * len(stage4b_train.expected_farm_ids())
    if (
        len(frame) != expected_rows
        or frame.duplicated(["model_variant", "farm_id"]).any()
        or set(frame["model_variant"].astype(str)) != set(NEW_VARIANTS)
    ):
        raise ValueError("candidate不变量不是4变体x5场站唯一矩阵")
    failures = frame[
        ~frame["persistence_control_pass"]
        | ~frame["corrected_candidate_control_pass"]
    ]
    if not failures.empty:
        raise ValueError(
            "D0/D1或T1/D0R/D2/D3 candidate不一致，禁止正式选型: "
            + str(failures.to_dict(orient="records"))
        )
    return frame, {
        "d0_d1_candidate_consistent": bool(
            frame[frame["model_variant"] == "d1"][
                "corrected_candidate_control_pass"
            ].all()
        ),
        "source_t1_d0r_d2_d3_candidate_consistent": bool(
            frame[frame["model_variant"].isin(("d0r", "d2", "d3"))][
                "corrected_candidate_control_pass"
            ].all()
        ),
        "persistence_consistent_all_new_variants": bool(
            frame["persistence_control_pass"].all()
        ),
        "candidate_max_norm_tolerance": CANDIDATE_MAX_NORM_TOL,
        "candidate_mean_norm_tolerance": CANDIDATE_MEAN_NORM_TOL,
    }


def _replay_macro_row(replay_summary, replay_payloads, comparison):
    regimes = pd.concat(
        [gate_predict._regime_metrics(payload) for payload in replay_payloads],
        ignore_index=True,
    )
    d0r = comparison[comparison["model_variant"] == "d0r"].iloc[0]
    row = {
        "model_variant": "fixed_g0_on_t1_replay",
        "macro_test_nrmse": float(replay_summary["capacity_normalized_rmse"].mean()),
        "macro_test_nmae": float(replay_summary["capacity_normalized_mae"].mean()),
        "corrected_overall_nrmse": float(d0r["corrected_overall_nrmse"]),
        "corrected_overall_nmae": float(d0r["corrected_overall_nmae"]),
        "macro_positive_regret_mean": float(
            replay_summary["positive_regret_mean"].mean()
        ),
        "macro_harm_rate_0_005": float(replay_summary["harm_rate_0_005"].mean()),
        "macro_oracle_brier": float(replay_summary["oracle_brier"].mean()),
        "macro_ece_10bin": float(replay_summary["ece_10bin"].mean()),
        "parameter_count_max": np.nan,
    }
    for group in ("dynamic", "ramp_up", "ramp_down"):
        part = regimes[
            (regimes["candidate"].astype(str) == "fused")
            & (regimes["regime_group"].astype(str) == group)
            & (regimes["horizon_step"].astype(str) == "all")
        ]
        _exact_five(part, f"replay/{group}", ("capacity_normalized_rmse",))
        row[f"fused_{group}_nrmse"] = float(
            part["capacity_normalized_rmse"].mean()
        )
        row[f"corrected_{group}_nrmse"] = float(d0r[f"corrected_{group}_nrmse"])
    return row


def build_controlled_contrasts(comparison, replay_summary, replay_payloads):
    """Predeclared controlled effects; signed deltas are left minus right."""
    lookup = {
        row["model_variant"]: row
        for _, row in comparison.set_index("model_variant", drop=False).iterrows()
    }
    lookup["fixed_g0_on_t1_replay"] = pd.Series(
        _replay_macro_row(replay_summary, replay_payloads, comparison)
    )
    specs = (
        (
            "candidate_effect_d2_minus_d1",
            "d2",
            "d1",
            "same_nonfactorized_calibrated_safe_gate_t1_candidate_vs_f7",
        ),
        (
            "auxiliary_objective_effect_d2_minus_d0r",
            "d2",
            "d0r",
            "same_t1_candidate_and_nonfactorized_topology_calibrated_safe_vs_direct",
        ),
        (
            "factorization_dynamic_joint_effect_d3_minus_d2",
            "d3",
            "d2",
            "same_t1_candidate_but_factorization_and_dynamic_auxiliary_change_together",
        ),
        (
            "gate_retraining_effect_d0r_minus_fixed_g0_replay",
            "d0r",
            "fixed_g0_on_t1_replay",
            "same_t1_candidate_new_direct_gate_vs_fixed_original_g0_gate",
        ),
    )
    metrics = (
        "macro_test_nrmse",
        "macro_test_nmae",
        "corrected_overall_nrmse",
        "fused_dynamic_nrmse",
        "fused_ramp_up_nrmse",
        "fused_ramp_down_nrmse",
        "macro_positive_regret_mean",
        "macro_harm_rate_0_005",
        "macro_oracle_brier",
        "macro_ece_10bin",
    )
    rows = []
    for effect, left, right, control in specs:
        for metric in metrics:
            left_value = float(lookup[left][metric])
            right_value = float(lookup[right][metric])
            rows.append(
                {
                    "effect": effect,
                    "left_variant": left,
                    "right_variant": right,
                    "controlled_condition": control,
                    "metric": metric,
                    "left_value": left_value,
                    "right_value": right_value,
                    "signed_delta_left_minus_right": left_value - right_value,
                    "relative_delta_left_vs_right": left_value / right_value - 1.0
                    if right_value != 0.0
                    else np.nan,
                    "negative_delta_means_improvement": True,
                }
            )
    return pd.DataFrame(rows)


def build_complexity(summary):
    frame = (
        summary.groupby("model_variant", as_index=False)
        .agg(
            parameter_count_max=("parameter_count", "max"),
            trainable_parameter_count_max=("trainable_parameter_count", "max"),
            inference_ms_per_sample_macro=(
                "inference_milliseconds_per_sample",
                "mean",
            ),
            recorded_training_seconds_macro=("gate_training_elapsed_seconds", "mean"),
        )
        .copy()
    )
    d0_params = int(
        frame.loc[frame["model_variant"] == "d0", "parameter_count_max"].iloc[0]
    )
    frame["parameter_delta_vs_d0"] = frame["parameter_count_max"] - d0_params
    frame["parameter_under_30k"] = frame["parameter_count_max"] < PARAMETER_LIMIT
    stage4b_updated = {
        "d0": 0,
        "d0r": 2553,
        "d1": 2553,
        "d2": 2553,
        "d3": 1993,
    }
    candidate_adapter = {
        "d0": 0,
        "d0r": 3152,
        "d1": 0,
        "d2": 3152,
        "d3": 3152,
    }
    frame["structural_parameter_count"] = frame["parameter_count_max"]
    frame["candidate_adapter_structural_parameter_count"] = frame[
        "model_variant"
    ].map(candidate_adapter).astype(int)
    frame["stage4b_trained_parameter_count"] = frame["model_variant"].map(
        stage4b_updated
    ).astype(int)
    frame["source_frozen_parameter_count"] = (
        frame["structural_parameter_count"]
        - frame["stage4b_trained_parameter_count"]
    )
    frame["stage4b_training_elapsed_seconds"] = frame[
        "recorded_training_seconds_macro"
    ]
    frame.loc[
        frame["model_variant"] == "d0", "stage4b_training_elapsed_seconds"
    ] = 0.0
    frame["source_historical_training_seconds"] = np.nan
    frame.loc[
        frame["model_variant"] == "d0", "source_historical_training_seconds"
    ] = frame.loc[
        frame["model_variant"] == "d0", "recorded_training_seconds_macro"
    ]
    frame["random_seed"] = stage4b_train.RANDOM_SEED
    frame["seed_count"] = 1
    frame["stability_scope"] = "single_seed_2026_no_multiseed_claim"
    return frame


def save_aggregate_figures(
    comparison,
    summary,
    horizon,
    calibration,
    safety,
    controlled_contrasts,
    replay_summary,
    output_dir,
):
    figure_dir = os.path.join(output_dir, "figures")
    cache_dir = os.path.join(output_dir, "matplotlib_cache")
    os.makedirs(figure_dir, exist_ok=True)
    plt = common_predict.setup_matplotlib({"matplotlib_cache": cache_dir})
    paths = {}

    ordered = comparison.sort_values("macro_test_nrmse", kind="stable")
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colors = ["#d62728" if bool(v) else "#4c78a8" for v in ordered["selected"]]
    ax.bar(ordered["model_variant"].str.upper(), ordered["macro_test_nrmse"], color=colors)
    ax.set_ylabel("Five-farm macro NRMSE")
    ax.set_title("Stage-4B guarded test-set ranking")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths["rank_figure"] = os.path.join(figure_dir, "stage4b_test_nrmse_rank.png")
    fig.savefig(paths["rank_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    matrix = summary.pivot(
        index="model_variant", columns="farm_id", values="capacity_normalized_rmse"
    ).reindex(index=ALL_VARIANTS, columns=stage4b_train.expected_farm_ids())
    if matrix.isna().any().any():
        raise ValueError("Stage-4B场站NRMSE热力图矩阵不完整")
    fig, ax = plt.subplots(figsize=(10.8, 4.7))
    image = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(matrix.index)), labels=[v.upper() for v in matrix.index])
    ax.set_xticks(
        np.arange(len(matrix.columns)), labels=[str(v)[-4:] for v in matrix.columns]
    )
    ax.set_xlabel("Farm ID (last 4 digits)")
    ax.set_title("Capacity-normalized RMSE by farm")
    fig.colorbar(image, ax=ax, label="NRMSE")
    fig.tight_layout()
    paths["farm_heatmap_figure"] = os.path.join(
        figure_dir, "stage4b_test_farm_heatmap.png"
    )
    fig.savefig(paths["farm_heatmap_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    numeric = horizon[horizon["horizon_step"].astype(str) != "all"].copy()
    numeric["h"] = pd.to_numeric(numeric["horizon_step"], errors="raise")
    macro = numeric.groupby(["model_variant", "h"], as_index=False)[
        "capacity_normalized_rmse"
    ].mean()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for variant in ALL_VARIANTS:
        part = macro[macro["model_variant"] == variant].sort_values("h")
        if len(part) != stage4b_train.FORECAST_LEN:
            raise ValueError(f"{variant}逐horizon指标不完整")
        ax.plot(part["h"], part["capacity_normalized_rmse"], marker="o", ms=3, label=variant.upper())
    ax.set_xlabel("Forecast horizon (15-min steps)")
    ax.set_ylabel("Five-farm macro NRMSE")
    ax.set_title("Horizon-wise test error")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    paths["horizon_figure"] = os.path.join(figure_dir, "stage4b_test_horizon_nrmse.png")
    fig.savefig(paths["horizon_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    d0r = summary[summary["model_variant"] == "d0r"].set_index("farm_id")
    replay = replay_summary.set_index("farm_id").reindex(d0r.index)
    x = np.arange(len(d0r))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - 0.18, replay["capacity_normalized_rmse"], 0.36, label="fixed G0 on T1")
    ax.bar(x + 0.18, d0r["capacity_normalized_rmse"], 0.36, label="retrained D0R")
    ax.set_xticks(x, labels=[str(v)[-4:] for v in d0r.index])
    ax.set_ylabel("NRMSE")
    ax.set_title("Gate retraining diagnostic")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths["replay_figure"] = os.path.join(
        figure_dir, "stage4b_d0r_fixed_g0_replay_comparison.png"
    )
    fig.savefig(paths["replay_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    effect = controlled_contrasts[
        controlled_contrasts["metric"] == "macro_test_nrmse"
    ].copy()
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    colors = ["#59a14f" if value < 0 else "#e15759" for value in effect["signed_delta_left_minus_right"]]
    ax.axhline(0.0, color="black", lw=1)
    ax.bar(effect["effect"], effect["signed_delta_left_minus_right"], color=colors)
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("Signed macro NRMSE delta (left - right)")
    ax.set_title("Predeclared controlled contrasts")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths["controlled_contrast_figure"] = os.path.join(
        figure_dir, "stage4b_controlled_contrast_nrmse.png"
    )
    fig.savefig(paths["controlled_contrast_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    valid_calibration = calibration.copy()
    for column in ("count", "mean_raw_gate", "corrected_better_rate"):
        valid_calibration[column] = pd.to_numeric(
            valid_calibration[column], errors="coerce"
        )
    valid_calibration = valid_calibration[
        (valid_calibration["count"] > 0)
        & np.isfinite(valid_calibration["mean_raw_gate"])
        & np.isfinite(valid_calibration["corrected_better_rate"])
    ]
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    for variant in ALL_VARIANTS:
        part = valid_calibration[
            valid_calibration["model_variant"] == variant
        ].copy()
        if part.empty:
            continue
        points = []
        for gate_bin, group in part.groupby("gate_bin", sort=True):
            weight = group["count"].to_numpy(float)
            points.append(
                (
                    gate_bin,
                    float(np.average(group["mean_raw_gate"], weights=weight)),
                    float(
                        np.average(
                            group["corrected_better_rate"], weights=weight
                        )
                    ),
                )
            )
        point = pd.DataFrame(points, columns=["gate_bin", "gate", "oracle"])
        ax.plot(point["gate"], point["oracle"], marker="o", label=variant.upper())
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set(xlabel="Mean gate probability", ylabel="Corrected-better frequency")
    ax.set_title("Five-farm test reliability")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    paths["reliability_figure"] = os.path.join(
        figure_dir, "stage4b_test_gate_reliability.png"
    )
    fig.savefig(paths["reliability_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    regime_order = ("stable", "dynamic", "ramp_up", "ramp_down")
    gate_regime = safety[
        (safety["scope_type"].astype(str) == "regime")
        & safety["scope_value"].astype(str).isin(regime_order)
    ].copy()
    gate_regime["gate_mean"] = pd.to_numeric(
        gate_regime["gate_mean"], errors="coerce"
    )
    gate_macro = (
        gate_regime.groupby(["model_variant", "scope_value"], as_index=False)[
            "gate_mean"
        ].mean()
    )
    matrix = gate_macro.pivot(
        index="model_variant", columns="scope_value", values="gate_mean"
    ).reindex(index=ALL_VARIANTS, columns=regime_order)
    if matrix.isna().any().any():
        raise ValueError("Stage-4B gate-by-regime可视化矩阵不完整")
    x = np.arange(len(regime_order))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for index, variant in enumerate(ALL_VARIANTS):
        ax.bar(
            x + (index - 2) * width,
            matrix.loc[variant].to_numpy(float),
            width,
            label=variant.upper(),
        )
    ax.set_xticks(x, labels=regime_order)
    ax.set_ylabel("Five-farm macro gate mean")
    ax.set_title("Gate response by wind-power regime")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    paths["gate_regime_figure"] = os.path.join(
        figure_dir, "stage4b_test_gate_by_regime.png"
    )
    fig.savefig(paths["gate_regime_figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def write_report(comparison, selected, controlled, invariants, output_dir):
    compact = [
        "model_variant",
        "macro_test_nrmse",
        "actual_macro_improvement_vs_d0",
        "farms_nondegraded_vs_d0",
        "farms_strictly_improved_vs_d0",
        "fused_ramp_up_nrmse",
        "fused_ramp_down_nrmse",
        "macro_positive_regret_mean",
        "macro_harm_rate_0_005",
        "parameter_count_max",
        "selection_guard_pass",
        "numerically_lowest",
        "selected",
    ]
    effect = controlled[controlled["metric"] == "macro_test_nrmse"]
    text = [
        "# Stage-4B门控收益转化闭环：测试集最终选型",
        "",
        f"最终选中 **{selected['model_variant'].upper()}**，5场站等权宏平均 NRMSE=`{selected['macro_test_nrmse']:.9f}`。",
        "",
        "本轮按用户指定在当前测试集选型；属于 `legacy_seen_test_selected`，不是最终盲测。",
        "",
        "## 五变体正式矩阵",
        "",
        comparison[compact].to_markdown(index=False),
        "",
        "## 受控对照（负的 signed delta 表示 left 更好）",
        "",
        effect.to_markdown(index=False),
        "",
        "## Candidate不变量",
        "",
        invariants.to_markdown(index=False),
        "",
        "## 守门与引用说明",
        "",
        "- D0只读引用Stage-4 T0；没有重训练、forward、模型复制或candidate archive复制。",
        "- 新变体必须相对D0的macro NRMSE实际改善至少0.2%，至少4/5场站不退化且至少3/5严格改善。",
        "- ramp-up/down不得超过D0+0.5%；candidate、安全指标不得越界，参数量必须<30k。",
        "- 先保留通过全部守门的新变体；NRMSE处于最优值0.1%带内时，再按regret、Brier、参数量、推理耗时和NRMSE排序。",
        "- 没有任何新变体合格时回退D0。",
        "- D3相对D2同时改变因子化拓扑和dynamic辅助项，只能解释为联合效应，不能单独归因为拓扑。",
        "- fixed-G0-on-T1 replay只用于诊断门控重训收益，不进入五变体排名。",
        "",
    ]
    return _atomic_text(
        "\n".join(text),
        os.path.join(output_dir, "stage4b_gate_closure_test_final_selection.md"),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default=os.getenv("WIND_STAGE4B_PREDICT_VARIANTS", ",".join(ALL_VARIANTS)),
        help="逗号分隔: d0,d0r,d1,d2,d3",
    )
    parser.add_argument(
        "--farms",
        default=os.getenv("WIND_STAGE4B_FARMS", ""),
        help="逗号分隔场站ID；空值为全部",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--run-id", default=None)
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
    os.environ.setdefault("PYTHONHASHSEED", str(stage4b_train.RANDOM_SEED))
    keras.utils.set_random_seed(stage4b_train.RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    variants = _parse_list(args.variants, ALL_VARIANTS, "variants")
    expected_farms = list(stage4b_train.expected_farm_ids())
    farms = (
        _parse_list(args.farms, expected_farms, "farms")
        if args.farms
        else expected_farms
    )
    if args.smoke:
        if args.variants == ",".join(ALL_VARIANTS):
            variants = ["d0r"]
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
        stage4b_train.RESULT_ROOT
        if full
        else os.path.join(
            stage4b_train.RESULT_ROOT,
            "partial_runs",
            args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    )
    output_dir = os.path.join(output_root, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)
    formal_marker = os.path.join(
        stage4b_train.RESULT_ROOT, OUTPUT_SUBDIR, FORMAL_MARKER_NAME
    )
    if full and os.path.exists(formal_marker):
        os.remove(formal_marker)

    source_marker, d0_frames, source_t1, source_paths = validate_source_bundle()
    training_marker_path, training_marker = validate_training_bundle(
        [variant for variant in variants if variant in NEW_VARIANTS]
    )
    source_test = {
        str(farm): record["path"]
        for farm, record in source_marker["test_files"].items()
    }
    results = []
    for farm_id in farms:
        test_file = source_test[farm_id]
        for variant in variants:
            if variant == "d0":
                continue
            print(f"\n===== Stage-4B预测 variant={variant} farm={farm_id} =====")
            payload = predict_variant(
                variant, test_file, training_marker, args.max_samples
            )
            results.append(save_payload(payload, output_root, args.skip_plots))

    if not full:
        requested_reference = d0_frames["summary"][
            d0_frames["summary"]["farm_id"].isin(farms)
        ] if "d0" in variants else pd.DataFrame()
        pieces = [requested_reference] + [item["summary"] for item in results]
        partial = pd.concat([item for item in pieces if not item.empty], ignore_index=True)
        path = _atomic_csv(
            partial, os.path.join(output_dir, "stage4b_gate_closure_partial_summary.csv")
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
        raise ValueError("Stage-4B正式预测必须是D0R/D1/D2/D3 x 5场站")
    frames = {
        key: pd.concat(
            [d0_frames[key]] + [item[key] for item in results],
            ignore_index=True,
            sort=False,
        )
        for key in SOURCE_FILES
    }
    validate_complete_output_matrix(frames)

    d0r_results = [item for item in results if item["payload"]["variant_id"] == "d0r"]
    replay_summary, replay_payloads, replay_paths = build_fixed_g0_replay(
        d0_frames["summary"], d0r_results, output_root, args.skip_plots
    )
    candidate_invariants, invariant_status = validate_candidate_invariants(
        d0_frames["summary"], source_t1, results
    )
    comparison = build_comparison(
        frames["summary"], frames["candidate"], frames["regime"], replay_summary
    )
    selected, comparison = select_model(comparison)
    controlled = build_controlled_contrasts(
        comparison, replay_summary, replay_payloads
    )
    complexity = build_complexity(frames["summary"])

    paths = {}
    for key, frame in frames.items():
        paths[key] = _atomic_csv(
            frame,
            os.path.join(output_dir, f"stage4b_gate_closure_test_{key}.csv"),
        )
    paths["comparison"] = _atomic_csv(
        comparison,
        os.path.join(output_dir, "stage4b_gate_closure_test_variant_comparison.csv"),
    )
    paths["final_selection"] = _atomic_csv(
        comparison[comparison["selected"]],
        os.path.join(output_dir, "stage4b_gate_closure_test_final_selection.csv"),
    )
    paths["candidate_invariants"] = _atomic_csv(
        candidate_invariants,
        os.path.join(output_dir, "stage4b_gate_closure_candidate_invariants.csv"),
    )
    paths["controlled_contrasts"] = _atomic_csv(
        controlled,
        os.path.join(output_dir, "stage4b_gate_closure_controlled_contrasts.csv"),
    )
    paths["complexity"] = _atomic_csv(
        complexity,
        os.path.join(output_dir, "stage4b_gate_closure_test_complexity.csv"),
    )
    paths.update(
        save_aggregate_figures(
            comparison,
            frames["summary"],
            frames["horizon"],
            frames["calibration"],
            frames["safety"],
            controlled,
            replay_summary,
            output_dir,
        )
    )

    source_rows = [
        {
            "source": "Stage-4 complete marker",
            "key": "marker",
            **_file_record(SOURCE_MARKER),
            "reuse_action": "hash_validated_read_only_dependency",
        }
    ]
    for key, path in source_paths.items():
        source_rows.append(
            {
                "source": "Stage-4 formal aggregate",
                "key": key,
                **_file_record(path),
                "reuse_action": "filter_t0_relabel_d0_no_inference_no_copy",
            }
        )
    for _, row in d0_frames["summary"].iterrows():
        for key in (
            "model_path",
            "artifact_path",
            "prediction_path",
            "candidate_archive_path",
            "single_window_figure_path",
            "weighted_curve_figure_path",
        ):
            value = row.get(key)
            if isinstance(value, str) and os.path.isfile(value):
                source_rows.append(
                    {
                        "source": f"Stage-4 T0 farm {row['farm_id']}",
                        "key": key,
                        **_file_record(value),
                        "reuse_action": "direct_path_reference_no_copy",
                    }
                )
    for _, row in source_t1.iterrows():
        value = row.get("candidate_archive_path")
        if isinstance(value, str) and os.path.isfile(value):
            source_rows.append(
                {
                    "source": f"Stage-4 T1 farm {row['farm_id']}",
                    "key": "candidate_archive_path",
                    **_file_record(value),
                    "reuse_action": "candidate_invariant_control_read_only",
                }
            )
    paths["source_manifest"] = _atomic_csv(
        pd.DataFrame(source_rows),
        os.path.join(output_dir, "stage4b_gate_closure_source_reuse_manifest.csv"),
    )
    paths["report"] = write_report(
        comparison, selected, controlled, candidate_invariants, output_dir
    )

    visual_candidates = []
    for key, path in paths.items():
        if isinstance(path, str) and path.lower().endswith(".png"):
            visual_candidates.append((f"aggregate.{key}", path))
    for key, path in replay_paths.items():
        if isinstance(path, str) and path.lower().endswith(".png"):
            visual_candidates.append((f"replay.{key}", path))
    for index, result in enumerate(results):
        for key, path in result["paths"].items():
            if isinstance(path, str) and path.lower().endswith(".png"):
                visual_candidates.append((f"result{index}.{key}", path))
    for _, row in d0_frames["summary"].iterrows():
        for key in ("single_window_figure_path", "weighted_curve_figure_path"):
            path = row.get(key)
            if isinstance(path, str) and path.lower().endswith(".png"):
                visual_candidates.append(
                    (f"source_d0.{row['farm_id']}.{key}", path)
                )
    visual_rows = []
    seen_visuals = set()
    for key, path in visual_candidates:
        real = os.path.realpath(path)
        if real in seen_visuals:
            continue
        seen_visuals.add(real)
        visual_rows.append({"key": key, **_file_record(path)})
    if not visual_rows:
        raise ValueError("正式Stage-4B bundle没有生成任何可视化图片")
    paths["visual_inventory"] = _atomic_csv(
        pd.DataFrame(visual_rows),
        os.path.join(output_dir, "stage4b_gate_closure_visual_inventory.csv"),
    )

    files = {
        "prediction_code": _file_record(__file__),
        "training_code": _file_record(stage4b_train.__file__),
        "dependency.stage4_prediction_helpers": _file_record(
            stage4_predict.__file__
        ),
        "dependency.controlled_gate_prediction_helpers": _file_record(
            gate_predict.__file__
        ),
        "dependency.common_prediction_helpers": _file_record(
            common_predict.__file__
        ),
        "stage4_source_marker": _file_record(SOURCE_MARKER),
        "training_marker": _file_record(training_marker_path),
    }
    files.update({f"formal.{key}": _file_record(path) for key, path in paths.items()})
    files.update(
        {f"replay.{key}": _file_record(path) for key, path in replay_paths.items()}
    )
    for index, result in enumerate(results):
        for key, path in result["paths"].items():
            if path:
                files[f"result{index}.{key}"] = _file_record(path)
    marker = {
        "status": "complete",
        "protocol_version": stage4b_train.PROTOCOL_VERSION,
        "architecture_version": stage4b_train.ARCHITECTURE_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": stage4b_train.RANDOM_SEED,
        "selection_split": "test",
        "test_used_for_selection": True,
        "test_reuse_status": TEST_REUSE_STATUS,
        "test_is_final_blind_evaluation": False,
        "variants": list(ALL_VARIANTS),
        "expected_farm_ids": expected_farms,
        "d0_policy": "direct_stage4_t0_reference_no_training_no_forward_no_copy",
        "fixed_g0_replay_policy": "diagnostic_only_not_sixth_formal_variant",
        "visualization_count": len(visual_rows),
        "candidate_invariants": invariant_status,
        "new_prediction_count": len(results),
        "selected_variant": str(selected["model_variant"]),
        "test_files": {
            farm: _file_record(source_test[farm]) for farm in expected_farms
        },
        "files": files,
    }
    marker_path = _atomic_json(marker, formal_marker)
    print(
        f"\nStage-4B测试集最终选择: {selected['model_variant']} / "
        f"macro NRMSE={selected['macro_test_nrmse']:.9f}"
    )
    print(f"正式报告: {paths['report']}")
    print(f"正式bundle marker: {marker_path}")


if __name__ == "__main__":
    main()
