"""第三阶段 G0--G4 测试集预测、门控诊断与最终选型入口。

执行协议：

* G0 的正式精度/逐horizon结果直接引用第二阶段 F7；仅对F7模型做一次
  diagnostic-only forward，重建逐点P/C/gate并与既有F7预测逐点核验。
* G1--G3 各执行一次模型前向，未来真实功率在前向结束后才读取。
* G4 从同一次 G3 scaled P/C/raw-pi 输出离线生成，不训练、不重复推理。
* G4 在用户明确要求下使用当前测试集，从预声明kappa网格选择一个跨场站、
  跨horizon统一阈值。报告明确标记legacy_seen/test-selected/not blind。
* hard top-1由相同G3输出生成，只是负对照，不进入G0--G4正式排名。

最终选择先执行Persistence安全/精度守门，再按5场站等权宏平均测试NRMSE；
若NRMSE差不超过相对0.1%，依次选择positive regret更低、Brier更低、参数更少者。
"""

import hashlib
import json
import os
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from tensorflow import keras

import wind_RegimeEncoder_PatchTST_feature_screen_predict as feature_predict
import wind_RegimeEncoder_PatchTST_predict as regime_predict
import wind_RegimeEncoder_PatchTST_train as regime_train
import wind_controlled_gate_cali_train as gate_train
import wind_dl_model_predict as common_predict

warnings.filterwarnings("ignore")


OUTPUT_SUBDIR = "testdata_predict_output"
TEST_REUSE_STATUS = "legacy_seen_test_selected"
SELECTION_METRIC = "capacity_normalized_rmse"
MACRO_SELECTION_METRIC = "macro_test_nrmse"
HARD_CONTROL_ID = "g4_hard"
NRMSE_GUARD_RELATIVE = 0.002
FARM_NRMSE_GUARD_RELATIVE = 0.01
MIN_FARMS_WITHIN_GUARD = 4
SAFETY_REDUCTION_TARGET = 0.20
NRMSE_TIE_RELATIVE = 0.001
SAFETY_NONDEGRADATION_TOLERANCE = 1e-12
G0_RECONSTRUCTION_MAX_CAPACITY_FRACTION = 5e-5
G0_RECONSTRUCTION_MEAN_CAPACITY_FRACTION = 1e-6

REGIME_GROUP_ORDER = tuple(regime_predict.REGIME_GROUP_ORDER)


def _assert_exact_farm_metrics(frame, label, required_columns):
    """Reject incomplete/duplicate/non-finite five-farm selection inputs."""
    if "farm_id" not in frame:
        raise KeyError(f"{label}缺少farm_id")
    expected = set(map(str, gate_train.expected_farm_ids()))
    actual = frame["farm_id"].astype(str)
    if (
        len(frame) != len(expected)
        or actual.duplicated().any()
        or set(actual) != expected
    ):
        raise ValueError(
            f"{label}必须恰好覆盖5个预期场站且每场一行: "
            f"rows={len(frame)}, farms={sorted(set(actual))}"
        )
    for column in required_columns:
        if column not in frame:
            raise KeyError(f"{label}缺少硬选型指标{column}")
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if values.shape != (len(expected),) or not np.isfinite(values).all():
            raise ValueError(f"{label}的{column}未覆盖5场站有限值: {values.tolist()}")


def configure_prediction_reproducibility():
    os.environ.setdefault("PYTHONHASHSEED", str(gate_train.RANDOM_SEED))
    keras.utils.set_random_seed(gate_train.RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _sha256(path, chunk_size=1024 * 1024):
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_to_csv(frame, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def _atomic_write_text(path, content):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            file.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def _atomic_write_json(value, path):
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


def _atomic_save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp.npz"
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def prediction_dirs(variant_id, create=True):
    if variant_id not in gate_train.VARIANT_SPECS:
        raise ValueError(f"未知G变体: {variant_id}")
    root = os.path.join(
        gate_train.variant_dirs(variant_id, create=create)["root"],
        OUTPUT_SUBDIR,
    )
    dirs = {
        "root": root,
        "predictions": os.path.join(root, "predictions"),
        "candidate_archives": os.path.join(root, "candidate_archives"),
        "candidate_metrics": os.path.join(root, "candidate_metrics"),
        "regime_assignments": os.path.join(root, "regime_assignments"),
        "regime_metrics": os.path.join(root, "regime_metrics"),
        "gate_diagnostics": os.path.join(root, "gate_diagnostics"),
        "safety_diagnostics": os.path.join(root, "safety_diagnostics"),
        "calibration": os.path.join(root, "calibration"),
        "single_windows": os.path.join(root, "single_window_comparisons"),
        "weighted_curves": os.path.join(root, "weighted_curves"),
        "figures": os.path.join(root, "figures"),
        "matplotlib_cache": os.path.join(root, "matplotlib_cache"),
    }
    if create:
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)
    return dirs


def comparison_dir():
    path = os.path.join(gate_train.RESULT_ROOT, OUTPUT_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def marker_path():
    return os.path.join(
        comparison_dir(), "controlled_gate_cali_test_bundle_complete.json"
    )


def _clear_prediction_marker():
    path = marker_path()
    if os.path.exists(path):
        os.remove(path)


def validate_training_bundle():
    path = os.path.join(gate_train.RESULT_ROOT, gate_train.TRAINING_MARKER_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少第三阶段训练complete marker: {path}")
    with open(path, "r", encoding="utf-8") as file:
        marker = json.load(file)
    if marker.get("status") != "complete":
        raise ValueError("第三阶段训练marker不是complete")
    if marker.get("protocol_version") != gate_train.PROTOCOL_VERSION:
        raise ValueError("第三阶段训练marker协议版本不匹配")
    if int(marker.get("new_model_count", -1)) != 15:
        raise ValueError("第三阶段训练marker新增模型数不是15")
    if set(map(str, marker.get("expected_farm_ids", ()))) != set(
        gate_train.expected_farm_ids()
    ):
        raise ValueError("第三阶段训练marker场站集合不匹配")
    if set(marker.get("variants", ())) != set(gate_train.VARIANT_SPECS):
        raise ValueError("第三阶段训练marker变体集合不匹配")
    expected_members = {
        f"{variant_id}.{farm_id}.{key}"
        for variant_id in gate_train.TRAINABLE_VARIANTS
        for farm_id in gate_train.expected_farm_ids()
        for key in (
            "model_path",
            "best_weights_path",
            "artifact_path",
            "history_path",
            "validation_diagnostics_path",
            "checkpoint_trace_path",
            "tail_path",
        )
    }
    if not expected_members.issubset(marker.get("files", {})):
        missing = sorted(expected_members - set(marker.get("files", {})))
        raise ValueError(f"第三阶段训练marker缺少正式成员: {missing}")
    for key, item in marker.get("files", {}).items():
        path_value = item.get("path")
        if not path_value or not os.path.exists(path_value):
            raise FileNotFoundError(f"训练marker成员缺失: {key}={path_value}")
        if _sha256(path_value) != item.get("sha256"):
            raise ValueError(f"训练marker成员hash不一致: {key}")
        if os.path.getsize(path_value) != int(item.get("size_bytes", -1)):
            raise ValueError(f"训练marker成员size不一致: {key}")
    return path, marker


def validate_stage2_source_bundle():
    path = feature_predict.bundle_completion_marker_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少第二阶段正式bundle marker: {path}")
    with open(path, "r", encoding="utf-8") as file:
        marker = json.load(file)
    if marker.get("status") != "complete":
        raise ValueError("第二阶段来源bundle不是complete")
    if gate_train.SOURCE_VARIANT not in set(marker.get("selection_variants", ())):
        raise ValueError("第二阶段来源bundle未包含F7")
    if set(map(str, marker.get("expected_test_farm_ids", ()))) != set(
        gate_train.expected_farm_ids()
    ):
        raise ValueError("第二阶段来源bundle场站集合不匹配")
    current_test_files = marker.get("current_test_files", {})
    if set(map(str, current_test_files)) != set(gate_train.expected_farm_ids()):
        raise ValueError("第二阶段来源bundle未锁定完整测试文件集合")
    for farm_id, item in current_test_files.items():
        test_path = item.get("path")
        if not test_path or not os.path.isfile(test_path):
            raise FileNotFoundError(f"第二阶段测试文件缺失: {farm_id}")
        if _sha256(test_path) != item.get("sha256"):
            raise ValueError(f"第二阶段测试文件hash漂移: {farm_id}")
        if os.path.getsize(test_path) != int(item.get("size_bytes", -1)):
            raise ValueError(f"第二阶段测试文件size漂移: {farm_id}")
    for key, item in marker.get("files", {}).items():
        source_path = item.get("path")
        if not source_path or not os.path.isfile(source_path):
            raise FileNotFoundError(f"第二阶段bundle成员缺失: {key}")
        if _sha256(source_path) != item.get("sha256"):
            raise ValueError(f"第二阶段bundle成员hash漂移: {key}")
        if os.path.getsize(source_path) != int(item.get("size_bytes", -1)):
            raise ValueError(f"第二阶段bundle成员size漂移: {key}")
    return path, marker


def discover_test_files():
    files = common_predict.discover_test_files()
    requested = os.getenv("WIND_CONTROLLED_GATE_FARMS")
    if requested:
        farms = {item.strip() for item in requested.split(",") if item.strip()}
        files = [
            path for path in files if str(common_predict.get_farm_id(path)) in farms
        ]
    return files


def get_requested_variants():
    raw = os.getenv("WIND_CONTROLLED_GATE_PREDICT_VARIANTS")
    if not raw:
        return list(gate_train.VARIANT_SPECS)
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = sorted(set(values) - set(gate_train.VARIANT_SPECS))
    if invalid:
        raise ValueError(f"未知预测G变体{invalid}")
    return list(dict.fromkeys(values))


def _artifact_path(variant_id, farm_id):
    model_name = gate_train.variant_model_name(variant_id)
    return os.path.join(
        gate_train.variant_dirs(variant_id, create=False)["preprocess"],
        f"{model_name}_farm_{farm_id}_preprocess.pkl",
    )


def load_artifact_and_model(variant_id, farm_id, training_marker):
    if variant_id not in gate_train.TRAINABLE_VARIANTS:
        raise ValueError(f"{variant_id}没有独立训练模型")
    artifact_path = _artifact_path(variant_id, farm_id)
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"缺少{variant_id}/{farm_id} artifact")
    artifact = joblib.load(artifact_path)
    spec = gate_train.VARIANT_SPECS[variant_id]
    loss_weights = artifact.get("loss_weights", {})
    calibration_config = artifact.get("calibration_candidate_difference_weight", {})
    q90 = np.asarray(
        calibration_config.get("candidate_difference_q90", ()), dtype=float
    )
    checks = {
        "artifact_schema": int(artifact.get("artifact_schema_version", -1))
        == gate_train.ARTIFACT_SCHEMA_VERSION,
        "variant": artifact.get("variant_id") == variant_id,
        "farm_id": str(artifact.get("farm_id")) == str(farm_id),
        "family": artifact.get("model_family") == gate_train.MODEL_FAMILY,
        "architecture": artifact.get("architecture_version")
        == gate_train.ARCHITECTURE_VERSION,
        "protocol": artifact.get("protocol_version") == gate_train.PROTOCOL_VERSION,
        "seed": int(artifact.get("random_seed", -1)) == gate_train.RANDOM_SEED,
        "features": list(artifact.get("selected_regime_feature_groups", ()))
        == ["P", "H", "D"],
        "feature_names": list(artifact.get("selected_regime_feature_names", ()))
        == list(gate_train.feature_train.selected_feature_names("f7")),
        "feature_count": int(artifact.get("selected_regime_feature_count", -1))
        == gate_train.SOURCE_FEATURE_COUNT,
        "history_len": int(artifact.get("history_len", -1)) == gate_train.HISTORY_LEN,
        "forecast_len": int(artifact.get("forecast_len", -1))
        == gate_train.FORECAST_LEN,
        "factorized": bool(artifact.get("factorized_gate"))
        == bool(spec["factorized_gate"]),
        "total_params": int(artifact.get("total_params", -1))
        == gate_train.EXPECTED_TOTAL_PARAMS[variant_id],
        "loss_calibration": np.isclose(
            float(loss_weights.get("calibration", np.nan)),
            spec["calibration_weight"],
        ),
        "loss_dynamic": np.isclose(
            float(loss_weights.get("dynamic", np.nan)), spec["dynamic_weight"]
        ),
        "loss_safety": np.isclose(
            float(loss_weights.get("safety", np.nan)), spec["safety_weight"]
        ),
        "calibration_q90": q90.shape == (gate_train.FORECAST_LEN,)
        and np.isfinite(q90).all()
        and np.all(q90 >= 0.0),
        "calibration_weight_floor": np.isclose(
            float(calibration_config.get("weight_floor", np.nan)),
            gate_train.CALIBRATION_WEIGHT_FLOOR,
        ),
        "calibration_enabled": bool(calibration_config.get("enabled"))
        == bool(spec["calibration_weight"] > 0.0),
        "calibration_train_scope": (
            spec["calibration_weight"] <= 0.0
            or (
                calibration_config.get("scope")
                == "per_farm_per_horizon_train_initial_frozen_f7"
                and int(calibration_config.get("estimation_sample_count", 0)) > 0
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{variant_id}/{farm_id} artifact不兼容: {failed}")
    model_path = artifact.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"缺少{variant_id}/{farm_id} model")
    if _sha256(model_path) != artifact.get("model_sha256"):
        raise ValueError(f"{variant_id}/{farm_id} model hash不一致")
    marker_files = training_marker["files"]
    for key, current_path in (
        ("artifact_path", artifact_path),
        ("model_path", model_path),
    ):
        member = marker_files[f"{variant_id}.{farm_id}.{key}"]
        if os.path.realpath(member["path"]) != os.path.realpath(current_path):
            raise ValueError(f"{variant_id}/{farm_id} {key}未被训练marker锁定")
        if _sha256(current_path) != member["sha256"]:
            raise ValueError(f"{variant_id}/{farm_id} {key}与训练marker hash不一致")
    model = keras.models.load_model(
        model_path,
        custom_objects=gate_train.get_controlled_gate_custom_objects(),
        compile=False,
    )
    if int(model.count_params()) != gate_train.EXPECTED_TOTAL_PARAMS[variant_id]:
        raise ValueError(f"{variant_id}/{farm_id}加载模型参数量漂移")
    return artifact, artifact_path, model, model_path


def _forward_controlled(model, dataset):
    diagnostic = gate_train._diagnostic_model(model)
    outputs = {
        key: [] for key in ("forecast", "persistence", "corrected", "gate", "q", "s")
    }
    for batch_x in dataset:
        result = diagnostic(batch_x, training=False)
        for key in outputs:
            outputs[key].append(np.asarray(result[key]))
    if not outputs["forecast"]:
        raise ValueError("测试dataset没有产生预测")
    return {key: np.concatenate(value) for key, value in outputs.items()}


def _forward_g0(model, dataset):
    diagnostic = gate_train._source_diagnostic_model(model)
    outputs = {key: [] for key in ("forecast", "persistence", "corrected", "gate")}
    for batch_x in dataset:
        result = diagnostic(batch_x, training=False)
        for key in outputs:
            outputs[key].append(np.asarray(result[key]))
    values = {key: np.concatenate(items) for key, items in outputs.items()}
    values["q"] = np.mean(values["gate"], axis=1, keepdims=True)
    values["q"] = np.repeat(values["q"], gate_train.FORECAST_LEN, axis=1)
    values["s"] = np.ones_like(values["gate"])
    return values


def _inverse_scaled(artifact, values, capacity):
    shape = np.asarray(values).shape
    physical = (
        artifact["scaler_y"]
        .inverse_transform(np.asarray(values).reshape(-1, 1))
        .reshape(shape)
    )
    return np.clip(physical, 0.0, float(capacity)).astype(np.float64)


def _build_payload(
    variant_id,
    farm_id,
    df,
    artifact,
    outputs,
    y_true,
    capacity,
    history_len,
    raw_gate=None,
    raw_fused_scaled=None,
    abstention_mask=None,
    selected_kappa=None,
):
    persistence_scaled = np.asarray(outputs["persistence"], dtype=np.float64)
    corrected_scaled = np.asarray(outputs["corrected"], dtype=np.float64)
    fused_scaled = np.asarray(outputs["forecast"], dtype=np.float64)
    applied_gate = np.asarray(outputs["gate"], dtype=np.float64)
    raw_gate = applied_gate if raw_gate is None else np.asarray(raw_gate, dtype=float)
    raw_fused_scaled = (
        fused_scaled
        if raw_fused_scaled is None
        else np.asarray(raw_fused_scaled, dtype=float)
    )
    q = np.asarray(outputs.get("q", np.nan), dtype=np.float64)
    s = np.asarray(outputs.get("s", np.nan), dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    expected_shape = fused_scaled.shape
    named_arrays = {
        "persistence_scaled": persistence_scaled,
        "corrected_scaled": corrected_scaled,
        "fused_scaled": fused_scaled,
        "raw_fused_scaled": raw_fused_scaled,
        "raw_gate": raw_gate,
        "applied_gate": applied_gate,
        "q": q,
        "s": s,
        "y_true": y_true,
    }
    if (
        len(expected_shape) != 2
        or expected_shape[1] != gate_train.FORECAST_LEN
        or expected_shape[1] != int(artifact.get("forecast_len", -1))
    ):
        raise ValueError(f"{variant_id}/{farm_id}预测shape异常: {expected_shape}")
    mismatched_shapes = {
        name: value.shape
        for name, value in named_arrays.items()
        if value.shape != expected_shape
    }
    if mismatched_shapes:
        raise ValueError(
            f"{variant_id}/{farm_id}二维输出shape不一致: {mismatched_shapes}"
        )
    for name, value in named_arrays.items():
        if name != "y_true" and not np.isfinite(value).all():
            raise FloatingPointError(f"{variant_id}/{farm_id} {name}包含非有限值")
    persistence = _inverse_scaled(artifact, persistence_scaled, capacity)
    corrected = _inverse_scaled(artifact, corrected_scaled, capacity)
    fused = _inverse_scaled(artifact, fused_scaled, capacity)
    raw_fused = _inverse_scaled(artifact, raw_fused_scaled, capacity)
    n_samples, forecast_len = fused.shape
    origins = df.index[history_len - 1 : history_len - 1 + n_samples]
    last_power = persistence[:, 0]
    regimes = regime_train.build_regime_targets_numpy(y_true, last_power, capacity)
    reconstruction = persistence_scaled + applied_gate * (
        corrected_scaled - persistence_scaled
    )
    reconstruction_error = float(np.max(np.abs(reconstruction - fused_scaled)))
    if reconstruction_error > 2e-6:
        raise ValueError(
            f"{variant_id}/{farm_id} scaled融合重构误差{reconstruction_error}"
        )
    if (
        not np.isfinite(applied_gate).all()
        or np.min(applied_gate) < -1e-7
        or np.max(applied_gate) > 1.0 + 1e-7
    ):
        raise ValueError(f"{variant_id}/{farm_id} applied gate不在[0,1]")
    if abstention_mask is None:
        abstention_mask = np.zeros_like(applied_gate, dtype=bool)
    abstention_mask = np.asarray(abstention_mask, dtype=bool)
    if abstention_mask.shape != expected_shape:
        raise ValueError(f"{variant_id}/{farm_id} abstention mask shape不一致")
    return {
        "variant_id": variant_id,
        "farm_id": str(farm_id),
        "artifact": artifact,
        "df": df,
        "history_len": int(history_len),
        "forecast_len": int(forecast_len),
        "capacity": float(capacity),
        "sample_id": np.arange(n_samples, dtype=np.int64),
        "horizon_step": np.arange(1, forecast_len + 1, dtype=np.int16),
        "forecast_origin_time": np.asarray(origins.astype(str), dtype=np.str_),
        "y_true": y_true,
        "persistence_scaled": persistence_scaled,
        "corrected_scaled": corrected_scaled,
        "fused_scaled": fused_scaled,
        "raw_fused_scaled": np.asarray(raw_fused_scaled, dtype=np.float64),
        "persistence": persistence,
        "corrected": corrected,
        "fused": fused,
        "raw_fused": raw_fused,
        "raw_gate": np.asarray(raw_gate, dtype=np.float64),
        "applied_gate": applied_gate,
        "q": q,
        "s": s,
        "abstention_mask": abstention_mask,
        "selected_kappa": selected_kappa,
        "regimes": regimes,
        "fusion_reconstruction_max_abs_error_scaled": reconstruction_error,
    }


def _source_f7_test_frames(stage2_marker):
    root = feature_predict.comparison_output_dir()
    paths = {
        "summary": os.path.join(
            root, "feature_screening_f0_f8_test_metrics_summary.csv"
        ),
        "horizon": os.path.join(
            root, "feature_screening_f0_f8_test_metrics_by_horizon_all.csv"
        ),
        "candidate": os.path.join(
            root, "feature_screening_f0_f8_test_candidate_all.csv"
        ),
    }
    for path in paths.values():
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少F7正式引用文件: {path}")
    marker_keys = {
        "summary": "selection.summary",
        "horizon": "selection.horizon",
        "candidate": "selection.candidate",
    }
    for key, marker_key in marker_keys.items():
        member = stage2_marker.get("files", {}).get(marker_key)
        if member is None:
            raise ValueError(f"第二阶段bundle未锁定{marker_key}")
        if os.path.realpath(member["path"]) != os.path.realpath(paths[key]):
            raise ValueError(f"第二阶段{key}路径与bundle不一致")
        if _sha256(paths[key]) != member["sha256"]:
            raise ValueError(f"第二阶段{key} hash与bundle不一致")
    frames = {
        key: pd.read_csv(path, dtype={"farm_id": str}) for key, path in paths.items()
    }
    for key, frame in frames.items():
        frames[key] = frame[
            frame["model_variant"].astype(str) == gate_train.SOURCE_VARIANT
        ].copy()
    if len(frames["summary"]) != 5:
        raise ValueError("F7正式test summary不是5场站")
    return frames, paths


def _load_source_prediction(row):
    path = feature_predict._first_present(
        row.get("source_prediction_path"), row.get("prediction_path")
    )
    path = feature_predict._resolve_existing_path(path)
    if path is None:
        raise FileNotFoundError("F7 summary缺少逐点预测CSV")
    frame = pd.read_csv(path)
    return frame.sort_values(["sample_id", "horizon_step"]).reset_index(drop=True), path


def _assert_same_datetimes(label, actual, expected):
    actual_values = pd.to_datetime(actual, errors="coerce").to_numpy(
        dtype="datetime64[ns]"
    )
    expected_values = pd.to_datetime(expected, errors="coerce").to_numpy(
        dtype="datetime64[ns]"
    )
    if (
        actual_values.shape != expected_values.shape
        or np.isnat(actual_values).any()
        or np.isnat(expected_values).any()
        or not np.array_equal(actual_values, expected_values)
    ):
        raise ValueError(f"G0诊断重建{label}与F7预测CSV不一致")


def _verify_g0_reconstruction(payload, source_row):
    source, path = _load_source_prediction(source_row)
    expected_sample = np.repeat(payload["sample_id"], payload["forecast_len"])
    expected_horizon = np.tile(payload["horizon_step"], len(payload["sample_id"]))
    if len(source) != len(expected_sample):
        raise ValueError("G0诊断重建与F7预测CSV行数不一致")
    if not np.array_equal(source["sample_id"].to_numpy(int), expected_sample):
        raise ValueError("G0诊断重建sample_id与F7不一致")
    if not np.array_equal(source["horizon_step"].to_numpy(int), expected_horizon):
        raise ValueError("G0诊断重建horizon与F7不一致")
    if "farm_id" in source and not np.all(
        source["farm_id"].astype(str).to_numpy() == payload["farm_id"]
    ):
        raise ValueError("G0诊断重建farm_id与F7不一致")
    expected_origins = np.repeat(
        payload["forecast_origin_time"], payload["forecast_len"]
    )
    if "forecast_origin_time" not in source:
        raise KeyError("F7预测CSV缺少forecast_origin_time")
    _assert_same_datetimes(
        "forecast_origin_time", source["forecast_origin_time"], expected_origins
    )
    history_len = payload["history_len"]
    n_samples = len(payload["sample_id"])
    expected_starts = np.repeat(
        np.asarray(
            payload["df"].index[history_len : history_len + n_samples].astype(str)
        ),
        payload["forecast_len"],
    )
    expected_targets = np.stack(
        [
            np.asarray(
                payload["df"]
                .index[history_len + horizon : history_len + horizon + n_samples]
                .astype(str)
            )
            for horizon in range(payload["forecast_len"])
        ],
        axis=1,
    ).reshape(-1)
    for column, expected in (
        ("forecast_start_time", expected_starts),
        ("target_time", expected_targets),
    ):
        if column not in source:
            raise KeyError(f"F7预测CSV缺少{column}")
        _assert_same_datetimes(column, source[column], expected)
    bitwise_compatible = feature_predict._csv_float_matches_archive(
        source["pred_power"], payload["fused"]
    )
    difference = np.abs(
        source["pred_power"].to_numpy(float) - payload["fused"].reshape(-1)
    )
    finite_difference = difference[np.isfinite(difference)]
    if not len(finite_difference):
        raise ValueError("G0诊断重建没有可比较的有限预测值")
    max_normalized_difference = float(finite_difference.max() / payload["capacity"])
    mean_normalized_difference = float(finite_difference.mean() / payload["capacity"])
    if not bitwise_compatible and (
        max_normalized_difference > G0_RECONSTRUCTION_MAX_CAPACITY_FRACTION
        or mean_normalized_difference > G0_RECONSTRUCTION_MEAN_CAPACITY_FRACTION
    ):
        raise ValueError(
            "G0诊断重建未在跨运行时容量容差内复现F7预测: "
            f"max_norm={max_normalized_difference}, "
            f"mean_norm={mean_normalized_difference}"
        )
    if not np.allclose(
        pd.to_numeric(source["actual_power"], errors="coerce"),
        payload["y_true"].reshape(-1),
        rtol=0.0,
        atol=1e-7,
        equal_nan=True,
    ):
        raise ValueError("G0诊断重建真值与F7预测CSV不一致")
    return {
        "path": path,
        "sha256": _sha256(path),
        "comparison_mode": (
            "strict_float32_compatible"
            if bitwise_compatible
            else "cross_runtime_capacity_tolerance"
        ),
        "max_abs_difference": float(finite_difference.max()),
        "mean_abs_difference": float(finite_difference.mean()),
        "max_capacity_normalized_difference": max_normalized_difference,
        "mean_capacity_normalized_difference": mean_normalized_difference,
    }


def predict_g0(test_file, source_frames):
    farm_id = str(common_predict.get_farm_id(test_file))
    source_model, artifact, artifact_path, model_path = gate_train.load_source_f7(
        farm_id
    )
    df, features, actual_power, capacity = common_predict.prepare_prediction_arrays(
        test_file, artifact
    )
    history_len = int(artifact["history_len"])
    forecast_len = int(artifact["forecast_len"])
    dataset, n_samples = common_predict.make_prediction_dataset(
        features, history_len, forecast_len
    )
    inference_start = time.perf_counter()
    outputs = _forward_g0(source_model, dataset)
    inference_elapsed = float(time.perf_counter() - inference_start)
    if len(outputs["forecast"]) != n_samples:
        raise ValueError("G0诊断重建样本数不一致")
    y_true = common_predict.build_truth_windows(
        actual_power, n_samples, history_len, forecast_len
    )
    payload = _build_payload(
        "g0",
        farm_id,
        df,
        artifact,
        outputs,
        y_true,
        capacity,
        history_len,
    )
    row = source_frames["summary"][source_frames["summary"]["farm_id"] == farm_id].iloc[
        0
    ]
    source_horizon = source_frames["horizon"][
        source_frames["horizon"]["farm_id"] == farm_id
    ].copy()
    if len(source_horizon) != forecast_len + 1:
        raise ValueError(f"G0/F7/{farm_id}正式horizon结果不是{forecast_len + 1}行")
    if _sha256(model_path) != row.get("loaded_model_sha256"):
        raise ValueError(f"G0/F7/{farm_id}模型hash与第二阶段正式summary不一致")
    if _sha256(artifact_path) != row.get("artifact_sha256"):
        raise ValueError(f"G0/F7/{farm_id} artifact hash与第二阶段正式summary不一致")
    for label, current_path, recorded_path in (
        ("model", model_path, row.get("loaded_model_path")),
        ("artifact", artifact_path, row.get("artifact_path")),
    ):
        if not isinstance(recorded_path, str) or os.path.realpath(
            current_path
        ) != os.path.realpath(recorded_path):
            raise ValueError(f"G0/F7/{farm_id} {label}路径与第二阶段summary不一致")
    source_verification = _verify_g0_reconstruction(payload, row)
    payload.update(
        {
            "model": source_model,
            "model_path": model_path,
            "model_sha256": _sha256(model_path),
            "artifact_path": artifact_path,
            "artifact_sha256": _sha256(artifact_path),
            "parameter_count": int(source_model.count_params()),
            "trainable_parameter_count": int(
                sum(
                    int(np.prod(weight.shape))
                    for weight in source_model.trainable_weights
                )
            ),
            "source_prediction_path": source_verification["path"],
            "source_prediction_sha256": source_verification["sha256"],
            "source_reconstruction_verification": source_verification,
            "source_formal_summary": row.to_dict(),
            "source_formal_horizon": source_horizon,
            "result_source": "direct_reference_existing_f7_result",
            "diagnostic_source": (
                "diagnostic_reconstruction_verified_against_legacy_f7_prediction"
            ),
            "reference_only": True,
            "inference_elapsed_seconds": inference_elapsed,
            "inference_milliseconds_per_sample": 1000.0 * inference_elapsed / n_samples,
        }
    )
    return payload


def predict_trainable_variant(variant_id, test_file, training_marker):
    farm_id = str(common_predict.get_farm_id(test_file))
    artifact, artifact_path, model, model_path = load_artifact_and_model(
        variant_id, farm_id, training_marker
    )
    df, features, actual_power, capacity = common_predict.prepare_prediction_arrays(
        test_file, artifact
    )
    history_len = int(artifact["history_len"])
    forecast_len = int(artifact["forecast_len"])
    dataset, n_samples = common_predict.make_prediction_dataset(
        features, history_len, forecast_len
    )
    inference_start = time.perf_counter()
    outputs = _forward_controlled(model, dataset)
    inference_elapsed = float(time.perf_counter() - inference_start)
    if len(outputs["forecast"]) != n_samples:
        raise ValueError(f"{variant_id}/{farm_id}预测样本数不一致")
    # 前向结束后才读取未来真实功率。
    y_true = common_predict.build_truth_windows(
        actual_power, n_samples, history_len, forecast_len
    )
    payload = _build_payload(
        variant_id,
        farm_id,
        df,
        artifact,
        outputs,
        y_true,
        capacity,
        history_len,
    )
    payload.update(
        {
            "model": model,
            "model_path": model_path,
            "model_sha256": _sha256(model_path),
            "artifact_path": artifact_path,
            "artifact_sha256": _sha256(artifact_path),
            "parameter_count": int(model.count_params()),
            "trainable_parameter_count": int(
                sum(int(np.prod(weight.shape)) for weight in model.trainable_weights)
            ),
            "result_source": "new_stage3_model_inference",
            "diagnostic_source": "same_forward_as_forecast",
            "reference_only": False,
            "inference_elapsed_seconds": inference_elapsed,
            "inference_milliseconds_per_sample": 1000.0 * inference_elapsed / n_samples,
        }
    )
    return payload


def _g4_payload_from_g3(g3, variant_id, gate):
    gate = np.asarray(gate, dtype=np.float64)
    fused_scaled = g3["persistence_scaled"] + gate * (
        g3["corrected_scaled"] - g3["persistence_scaled"]
    )
    outputs = {
        "persistence": g3["persistence_scaled"],
        "corrected": g3["corrected_scaled"],
        "forecast": fused_scaled,
        "gate": gate,
        "q": g3["q"],
        "s": g3["s"],
    }
    payload = _build_payload(
        variant_id,
        g3["farm_id"],
        g3["df"],
        g3["artifact"],
        outputs,
        g3["y_true"],
        g3["capacity"],
        g3["history_len"],
        raw_gate=g3["applied_gate"],
        raw_fused_scaled=g3["fused_scaled"],
        abstention_mask=gate <= 0.0,
    )
    payload.update(
        {
            "model": None,
            "model_path": g3["model_path"],
            "model_sha256": g3["model_sha256"],
            "artifact_path": g3["artifact_path"],
            "artifact_sha256": g3["artifact_sha256"],
            "parameter_count": g3["parameter_count"],
            "trainable_parameter_count": g3["trainable_parameter_count"],
            "posthoc_added_trainable_parameter_count": 0,
            "result_source": "posthoc_same_g3_forward_no_training",
            "diagnostic_source": "same_g3_forward",
            "reference_only": True,
            "inference_elapsed_seconds": g3["inference_elapsed_seconds"],
            "inference_milliseconds_per_sample": g3[
                "inference_milliseconds_per_sample"
            ],
        }
    )
    return payload


def _finite_flat(payload):
    arrays = (
        payload["y_true"],
        payload["persistence"],
        payload["corrected"],
        payload["fused"],
        payload["raw_gate"],
        payload["applied_gate"],
    )
    valid = np.ones(payload["y_true"].shape, dtype=bool)
    for value in arrays:
        valid &= np.isfinite(value)
    return valid


def _ece(probability, truth, bins=10, adaptive=False):
    probability = np.asarray(probability, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if not len(probability):
        return np.nan
    if adaptive:
        order = np.argsort(probability, kind="stable")
        groups = np.array_split(order, bins)
        return float(
            sum(
                len(group)
                / len(probability)
                * abs(probability[group].mean() - truth[group].mean())
                for group in groups
                if len(group)
            )
        )
    ids = np.minimum((np.clip(probability, 0, 1) * bins).astype(int), bins - 1)
    result = 0.0
    for bin_id in range(bins):
        mask = ids == bin_id
        if mask.any():
            result += mask.mean() * abs(probability[mask].mean() - truth[mask].mean())
    return float(result)


def _rank_correlation(first, second):
    first = pd.Series(np.asarray(first, dtype=float)).rank(method="average")
    second = pd.Series(np.asarray(second, dtype=float)).rank(method="average")
    if len(first) < 2 or first.std() <= 1e-12 or second.std() <= 1e-12:
        return np.nan
    return float(first.corr(second))


def _safe_quantile(values, quantile):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, quantile)) if len(values) else np.nan


def _utility_metrics(payload, point_mask=None):
    valid = _finite_flat(payload)
    if point_mask is not None:
        valid &= np.asarray(point_mask, dtype=bool)
    y = payload["y_true"][valid]
    persistence = payload["persistence"][valid]
    corrected = payload["corrected"][valid]
    fused = payload["fused"][valid]
    raw_fused = payload["raw_fused"][valid]
    raw_gate = payload["raw_gate"][valid]
    applied_gate = payload["applied_gate"][valid]
    capacity = payload["capacity"]
    if not len(y):
        return {"valid_count": 0}
    p_abs = np.abs(persistence - y) / capacity
    c_abs = np.abs(corrected - y) / capacity
    f_abs = np.abs(fused - y) / capacity
    raw_abs = np.abs(raw_fused - y) / capacity
    oracle = c_abs < p_abs
    prevalence = float(oracle.mean())
    brier = float(np.mean(np.square(raw_gate - oracle.astype(float))))
    baseline_brier = prevalence * (1.0 - prevalence)
    hard = raw_gate >= 0.5
    if np.unique(oracle).size == 2:
        auroc = float(roc_auc_score(oracle, raw_gate))
        auprc = float(average_precision_score(oracle, raw_gate))
        balanced = float(balanced_accuracy_score(oracle, hard))
    else:
        auroc = auprc = balanced = np.nan
    regret = f_abs - p_abs
    positive = np.maximum(0.0, regret)
    raw_regret = raw_abs - p_abs
    p95 = _safe_quantile(positive, 0.95)
    tail_count = max(1, int(np.ceil(0.05 * len(positive))))
    cvar = float(np.sort(positive)[-tail_count:].mean())
    oracle_abs = np.minimum(p_abs, c_abs)
    possible_gain = float(p_abs.mean() - oracle_abs.mean())
    captured_gain = float(p_abs.mean() - f_abs.mean())
    dynamic_magnitude = np.repeat(
        np.asarray(payload["regimes"]["change_magnitude"], dtype=float),
        payload["forecast_len"],
        axis=0,
    )
    flat_valid = valid.reshape(-1)
    dynamic_values = dynamic_magnitude[flat_valid]
    return {
        "valid_count": int(valid.sum()),
        "excluded_nonfinite_count": int(valid.size - valid.sum()),
        "gate_mean": float(raw_gate.mean()),
        "gate_std": float(raw_gate.std()),
        "gate_p10": _safe_quantile(raw_gate, 0.10),
        "gate_p50": _safe_quantile(raw_gate, 0.50),
        "gate_p90": _safe_quantile(raw_gate, 0.90),
        "gate_low_saturation_rate": float(np.mean(raw_gate < 0.05)),
        "gate_high_saturation_rate": float(np.mean(raw_gate > 0.95)),
        "applied_gate_mean": float(applied_gate.mean()),
        "corrected_better_prevalence": prevalence,
        "oracle_brier": brier,
        "brier_skill_score": (
            float(1.0 - brier / baseline_brier) if baseline_brier > 1e-12 else np.nan
        ),
        "ece_10bin": _ece(raw_gate, oracle, bins=10, adaptive=False),
        "adaptive_ece_10bin": _ece(raw_gate, oracle, bins=10, adaptive=True),
        "auroc": auroc,
        "auprc": auprc,
        "balanced_accuracy": balanced,
        "utility_gap": float(raw_gate[oracle].mean() - raw_gate[~oracle].mean())
        if oracle.any() and (~oracle).any()
        else np.nan,
        "positive_regret_mean": float(positive.mean()),
        "positive_regret_p95": p95,
        "positive_regret_cvar95": cvar,
        "harm_rate_gt_0": float(np.mean(regret > 0.0)),
        "harm_rate_0_005": float(np.mean(regret > 0.005)),
        "benefit_rate": float(np.mean(regret < 0.0)),
        "persistence_mae_normalized": float(p_abs.mean()),
        "fused_mae_normalized": float(f_abs.mean()),
        "oracle_mae_normalized": float(oracle_abs.mean()),
        "oracle_gap_closure": (
            captured_gain / possible_gain if possible_gain > 1e-12 else np.nan
        ),
        "coverage_rate": float(np.mean(applied_gate > 0.0)),
        "abstention_rate": float(np.mean(applied_gate <= 0.0)),
        "override_rate": float(np.mean(np.abs(applied_gate - raw_gate) > 1e-12)),
        "raw_positive_regret_mean": float(np.maximum(0.0, raw_regret).mean()),
        "avoided_normalized_mae": float(raw_abs.mean() - f_abs.mean()),
        "missed_oracle_gain": float(np.maximum(0.0, f_abs - oracle_abs).mean()),
        "gate_change_magnitude_spearman": _rank_correlation(raw_gate, dynamic_values),
    }


def _scope_masks(payload):
    shape = payload["y_true"].shape
    masks = {"overall": np.ones(shape, dtype=bool)}
    for horizon in range(shape[1]):
        mask = np.zeros(shape, dtype=bool)
        mask[:, horizon] = True
        masks[f"horizon_{horizon + 1}"] = mask
    sample_masks, _ = regime_predict._regime_masks(payload["regimes"])
    for name in REGIME_GROUP_ORDER:
        masks[f"regime_{name}"] = np.repeat(
            np.asarray(sample_masks[name], dtype=bool)[:, None],
            shape[1],
            axis=1,
        )
    return masks


def build_safety_scope_frame(payload):
    rows = []
    for scope, mask in _scope_masks(payload).items():
        if scope.startswith("horizon_"):
            scope_type = "horizon"
            scope_value = int(scope.split("_")[-1])
        elif scope.startswith("regime_"):
            scope_type = "regime"
            scope_value = scope[len("regime_") :]
        else:
            scope_type = "overall"
            scope_value = "all"
        rows.append(
            {
                "model_family": gate_train.MODEL_FAMILY,
                "model_variant": payload["variant_id"],
                "farm_id": payload["farm_id"],
                "scope_type": scope_type,
                "scope_value": scope_value,
                "raw_gate_for_calibration": True,
                "applied_gate_for_safety": True,
                **_utility_metrics(payload, mask),
            }
        )
    return pd.DataFrame(rows)


def build_reliability_frame(payload, bins=10):
    valid = _finite_flat(payload)
    probability = payload["raw_gate"][valid]
    oracle = np.abs(payload["corrected"][valid] - payload["y_true"][valid]) < np.abs(
        payload["persistence"][valid] - payload["y_true"][valid]
    )
    ids = np.minimum((np.clip(probability, 0, 1) * bins).astype(int), bins - 1)
    rows = []
    for bin_id in range(bins):
        mask = ids == bin_id
        rows.append(
            {
                "model_family": gate_train.MODEL_FAMILY,
                "model_variant": payload["variant_id"],
                "farm_id": payload["farm_id"],
                "gate_bin": bin_id,
                "gate_bin_left": bin_id / bins,
                "gate_bin_right": (bin_id + 1) / bins,
                "count": int(mask.sum()),
                "mean_raw_gate": float(probability[mask].mean())
                if mask.any()
                else np.nan,
                "corrected_better_rate": float(oracle[mask].mean())
                if mask.any()
                else np.nan,
                "finite_masked": True,
            }
        )
    return pd.DataFrame(rows)


def build_point_gate_frame(payload):
    n_samples = len(payload["sample_id"])
    horizon = payload["forecast_len"]
    sample_id = np.repeat(payload["sample_id"], horizon)
    horizon_step = np.tile(payload["horizon_step"], n_samples)
    origins = np.repeat(payload["forecast_origin_time"], horizon)
    y = payload["y_true"].reshape(-1)
    p = payload["persistence"].reshape(-1)
    c = payload["corrected"].reshape(-1)
    f = payload["fused"].reshape(-1)
    raw_gate = payload["raw_gate"].reshape(-1)
    applied_gate = payload["applied_gate"].reshape(-1)
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(c) & np.isfinite(f)
    return pd.DataFrame(
        {
            "model_variant": payload["variant_id"],
            "farm_id": payload["farm_id"],
            "sample_id": sample_id,
            "forecast_origin_time": origins,
            "horizon_step": horizon_step,
            "raw_gate": raw_gate,
            "applied_gate": applied_gate,
            "q": payload["q"].reshape(-1),
            "s": payload["s"].reshape(-1),
            "abstained": payload["abstention_mask"].reshape(-1),
            "corrected_better": np.where(valid, np.abs(c - y) < np.abs(p - y), False),
            "normalized_regret": np.where(
                valid,
                (np.abs(f - y) - np.abs(p - y)) / payload["capacity"],
                np.nan,
            ),
            "valid_oracle": valid,
        }
    )


def _candidate_metrics(payload):
    frames = []
    for name, values in (
        ("fused", payload["fused"]),
        ("persistence", payload["persistence"]),
        ("corrected", payload["corrected"]),
    ):
        model_name = (
            gate_train.variant_model_name(payload["variant_id"])
            if payload["variant_id"] in gate_train.VARIANT_SPECS
            else f"{gate_train.MODEL_FAMILY}_{payload['variant_id']}"
        )
        frame = common_predict.metrics_by_horizon(
            model_name,
            payload["farm_id"],
            payload["y_true"],
            values,
            payload["capacity"],
            payload["forecast_len"],
        )
        frame["model_family"] = gate_train.MODEL_FAMILY
        frame["model_variant"] = payload["variant_id"]
        frame["candidate"] = name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _regime_metrics(payload):
    rows = regime_predict.build_regime_metric_rows(
        payload["variant_id"],
        payload["farm_id"],
        payload["y_true"],
        {
            "fused": payload["fused"],
            "persistence": payload["persistence"],
            "corrected": payload["corrected"],
        },
        payload["regimes"],
        payload["capacity"],
    )
    frame = pd.DataFrame(rows)
    frame["model_family"] = gate_train.MODEL_FAMILY
    frame["model_variant"] = payload["variant_id"]
    return frame


def _assignment_frame(payload):
    frame = regime_predict._assignment_frame(
        payload["df"],
        payload["farm_id"],
        payload["regimes"],
        payload["persistence"][:, 0],
        len(payload["sample_id"]),
        payload["history_len"],
    )
    frame["model_family"] = gate_train.MODEL_FAMILY
    frame["model_variant"] = payload["variant_id"]
    return frame


def _archive_path(payload, dirs):
    model_name = (
        gate_train.variant_model_name(payload["variant_id"])
        if payload["variant_id"] in gate_train.VARIANT_SPECS
        else f"{gate_train.MODEL_FAMILY}_{payload['variant_id']}"
    )
    return os.path.join(
        dirs["candidate_archives"],
        f"{model_name}_candidate_archive_farm_{payload['farm_id']}.npz",
    )


def _horizon_key(value):
    text = str(value).strip().lower()
    return "all" if text == "all" else str(int(float(text)))


def _reuse_g0_formal_horizon_metrics(payload, recomputed):
    source = payload["source_formal_horizon"].copy()
    source["_horizon_key"] = source["horizon_step"].map(_horizon_key)
    if source["_horizon_key"].duplicated().any():
        raise ValueError(f"G0/F7/{payload['farm_id']}正式horizon键重复")
    source = source.set_index("_horizon_key")
    metric_columns = (
        "valid_count",
        "mae",
        "mse",
        "rmse",
        "mape",
        "smape",
        "r2",
        "capacity_normalized_mae",
        "capacity_normalized_rmse",
    )
    result = recomputed.copy()
    for row_index, row in result.iterrows():
        key = _horizon_key(row["horizon_step"])
        if key not in source.index:
            raise ValueError(f"G0/F7正式horizon缺少{key}")
        formal = source.loc[key]
        for column in metric_columns:
            left = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            right = pd.to_numeric(pd.Series([formal[column]]), errors="coerce").iloc[0]
            if not (
                (pd.isna(left) and pd.isna(right))
                or np.isclose(left, right, rtol=1e-5, atol=1e-6)
            ):
                raise ValueError(
                    f"G0/F7/{payload['farm_id']}正式{key}/{column}与重算不一致: "
                    f"{right} != {left}"
                )
            result.at[row_index, column] = formal[column]
    summary = payload["source_formal_summary"]
    overall = result[result["horizon_step"].astype(str) == "all"].iloc[0]
    for column in metric_columns:
        formal = summary.get(column)
        reused = overall[column]
        if not (
            (pd.isna(formal) and pd.isna(reused))
            or np.isclose(float(formal), float(reused), rtol=1e-10, atol=1e-10)
        ):
            raise ValueError(f"G0/F7正式summary与horizon的{column}不一致")
    result["formal_metric_source"] = "direct_reference_existing_f7_horizon"
    return result


def save_payload_outputs(payload, write_prediction=True, hard_control=False):
    dirs = prediction_dirs("g4" if hard_control else payload["variant_id"])
    if hard_control:
        for key in list(dirs):
            if key == "root":
                continue
            dirs[key] = os.path.join(dirs[key], "hard_control")
            os.makedirs(dirs[key], exist_ok=True)
    model_name = (
        f"{gate_train.MODEL_FAMILY}_{payload['variant_id']}"
        if hard_control
        else gate_train.variant_model_name(payload["variant_id"])
    )
    prediction_frame = common_predict.build_prediction_frame(
        model_name,
        payload["df"],
        payload["farm_id"],
        payload["fused"],
        payload["y_true"],
        payload["history_len"],
        payload["forecast_len"],
    )
    if write_prediction:
        prediction_path = os.path.join(
            dirs["predictions"],
            f"{model_name}_predictions_farm_{payload['farm_id']}.csv",
        )
        _atomic_to_csv(prediction_frame, prediction_path)
    else:
        prediction_path = payload["source_prediction_path"]
    horizon = common_predict.metrics_by_horizon(
        model_name,
        payload["farm_id"],
        payload["y_true"],
        payload["fused"],
        payload["capacity"],
        payload["forecast_len"],
    )
    if payload["variant_id"] == "g0" and not hard_control:
        horizon = _reuse_g0_formal_horizon_metrics(payload, horizon)
    horizon["model_family"] = gate_train.MODEL_FAMILY
    horizon["model_variant"] = payload["variant_id"]
    horizon_path = os.path.join(
        dirs["root"], f"{model_name}_metrics_by_horizon_farm_{payload['farm_id']}.csv"
    )
    _atomic_to_csv(horizon, horizon_path)
    candidates = _candidate_metrics(payload)
    candidate_path = os.path.join(
        dirs["candidate_metrics"],
        f"{model_name}_candidate_metrics_farm_{payload['farm_id']}.csv",
    )
    _atomic_to_csv(candidates, candidate_path)
    regimes = _regime_metrics(payload)
    regime_path = os.path.join(
        dirs["regime_metrics"],
        f"{model_name}_regime_metrics_farm_{payload['farm_id']}.csv",
    )
    _atomic_to_csv(regimes, regime_path)
    assignments = _assignment_frame(payload)
    assignment_path = os.path.join(
        dirs["regime_assignments"],
        f"{model_name}_regime_assignments_farm_{payload['farm_id']}.csv",
    )
    _atomic_to_csv(assignments, assignment_path)
    safety = build_safety_scope_frame(payload)
    safety_path = os.path.join(
        dirs["safety_diagnostics"],
        f"{model_name}_safety_by_scope_farm_{payload['farm_id']}.csv",
    )
    _atomic_to_csv(safety, safety_path)
    calibration = build_reliability_frame(payload)
    calibration_path = os.path.join(
        dirs["calibration"],
        f"{model_name}_reliability_farm_{payload['farm_id']}.csv",
    )
    _atomic_to_csv(calibration, calibration_path)
    point_gate = build_point_gate_frame(payload)
    gate_path = os.path.join(
        dirs["gate_diagnostics"],
        f"{model_name}_gate_points_farm_{payload['farm_id']}.csv",
    )
    _atomic_to_csv(point_gate, gate_path)
    archive_path = _archive_path(payload, dirs)
    _atomic_save_npz(
        archive_path,
        schema_version=np.asarray("controlled_gate_candidate_archive_v1"),
        model_variant=np.asarray(payload["variant_id"]),
        farm_id=np.asarray(payload["farm_id"]),
        sample_id=payload["sample_id"],
        horizon_step=payload["horizon_step"],
        forecast_origin_time=payload["forecast_origin_time"],
        y_true=payload["y_true"],
        persistence_scaled=payload["persistence_scaled"],
        corrected_scaled=payload["corrected_scaled"],
        fused_scaled=payload["fused_scaled"],
        raw_fused_scaled=payload["raw_fused_scaled"],
        persistence=payload["persistence"],
        corrected=payload["corrected"],
        fused=payload["fused"],
        raw_fused=payload["raw_fused"],
        raw_gate=payload["raw_gate"],
        applied_gate=payload["applied_gate"],
        q=payload["q"],
        s=payload["s"],
        abstention_mask=payload["abstention_mask"],
        selected_kappa=np.asarray(
            np.nan if payload["selected_kappa"] is None else payload["selected_kappa"]
        ),
        capacity=np.asarray(payload["capacity"]),
    )
    single_path = single_figure = weighted_path = weighted_figure = None
    weighted_metrics = {}
    if not hard_control:
        single_path, single_figure = common_predict.save_single_window_plot(
            prediction_frame,
            model_name,
            payload["farm_id"],
            dirs,
            payload["forecast_len"],
        )
        (
            weighted_path,
            weighted_figure,
            weighted_metrics,
        ) = common_predict.save_weighted_full_test_plot(
            prediction_frame,
            model_name,
            payload["farm_id"],
            dirs,
            payload["capacity"],
        )
    overall = horizon[horizon["horizon_step"].astype(str) == "all"].iloc[0]
    utility = safety[
        (safety["scope_type"] == "overall")
        & (safety["scope_value"].astype(str) == "all")
    ].iloc[0]
    summary = {
        **overall.to_dict(),
        **{
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
        },
        "model_family": gate_train.MODEL_FAMILY,
        "model_variant": payload["variant_id"],
        "variant_label": (
            gate_train.VARIANT_SPECS[payload["variant_id"]]["label"]
            if payload["variant_id"] in gate_train.VARIANT_SPECS
            else "G4 hard top-1 negative control"
        ),
        "farm_id": payload["farm_id"],
        "feature_groups": gate_train.SOURCE_FEATURE_GROUPS,
        "feature_count": gate_train.SOURCE_FEATURE_COUNT,
        "parameter_count": payload["parameter_count"],
        "trainable_parameter_count": payload["trainable_parameter_count"],
        "posthoc_added_trainable_parameter_count": payload.get(
            "posthoc_added_trainable_parameter_count"
        ),
        "inference_elapsed_seconds": payload["inference_elapsed_seconds"],
        "inference_milliseconds_per_sample": payload[
            "inference_milliseconds_per_sample"
        ],
        "reference_only": payload["reference_only"],
        "selection_eligible": not hard_control,
        "selected_kappa": payload["selected_kappa"],
        "result_source": payload["result_source"],
        "diagnostic_source": payload["diagnostic_source"],
        "test_reuse_status": TEST_REUSE_STATUS,
        "selection_split": "test",
        "test_used_for_selection": True,
        "test_is_final_blind_evaluation": False,
        "random_seed": gate_train.RANDOM_SEED,
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
        "source_reconstruction_comparison_mode": payload.get(
            "source_reconstruction_verification", {}
        ).get("comparison_mode"),
        "source_reconstruction_max_capacity_normalized_difference": payload.get(
            "source_reconstruction_verification", {}
        ).get("max_capacity_normalized_difference"),
        "source_reconstruction_mean_capacity_normalized_difference": payload.get(
            "source_reconstruction_verification", {}
        ).get("mean_capacity_normalized_difference"),
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
        "gate_points": point_gate,
        "paths": paths,
        "payload": payload,
    }


def _macro_for_payloads(payloads):
    rows = []
    for payload in payloads:
        metric = common_predict.calculate_metrics(
            payload["y_true"], payload["fused"], payload["capacity"]
        )
        utility = _utility_metrics(payload)
        rows.append(
            {
                "farm_id": payload["farm_id"],
                SELECTION_METRIC: metric[SELECTION_METRIC],
                **utility,
            }
        )
    frame = pd.DataFrame(rows)
    required = (
        "valid_count",
        SELECTION_METRIC,
        "positive_regret_mean",
        "harm_rate_0_005",
        "oracle_brier",
        "ece_10bin",
        "coverage_rate",
        "abstention_rate",
    )
    _assert_exact_farm_metrics(frame, "G4 kappa候选", required)
    if (pd.to_numeric(frame["valid_count"], errors="coerce") <= 0).any():
        raise ValueError("G4 kappa候选存在零有效样本场站")
    macro = {
        MACRO_SELECTION_METRIC: float(frame[SELECTION_METRIC].mean()),
        "macro_positive_regret_mean": float(frame["positive_regret_mean"].mean()),
        "macro_harm_rate_0_005": float(frame["harm_rate_0_005"].mean()),
        "macro_oracle_brier": float(frame["oracle_brier"].mean()),
        "macro_ece_10bin": float(frame["ece_10bin"].mean()),
        "macro_coverage_rate": float(frame["coverage_rate"].mean()),
        "macro_abstention_rate": float(frame["abstention_rate"].mean()),
        "farm_count": int(len(frame)),
    }
    return frame, macro


def _relative_reduction(candidate, baseline):
    candidate = float(candidate)
    baseline = float(baseline)
    if baseline <= SAFETY_NONDEGRADATION_TOLERANCE:
        return (
            0.0 if candidate <= baseline + SAFETY_NONDEGRADATION_TOLERANCE else -np.inf
        )
    return float(1.0 - candidate / baseline)


def _safety_reduction_guard(regret_reduction, harm_reduction):
    nondegrading = bool(
        regret_reduction >= -SAFETY_NONDEGRADATION_TOLERANCE
        and harm_reduction >= -SAFETY_NONDEGRADATION_TOLERANCE
    )
    target = bool(
        nondegrading
        and max(regret_reduction, harm_reduction) >= SAFETY_REDUCTION_TARGET
    )
    return nondegrading, target


def select_g4_kappa(g3_payloads, g0_payloads):
    _, g0_macro = _macro_for_payloads(g0_payloads)
    g0_farms, _ = _macro_for_payloads(g0_payloads)
    g0_nrmse = g0_farms.set_index("farm_id")[SELECTION_METRIC]
    g3_farms, g3_macro = _macro_for_payloads(g3_payloads)
    g3_nrmse = g3_farms.set_index("farm_id")[SELECTION_METRIC]
    rows = []
    candidate_payloads = {}
    for kappa in gate_train.G4_KAPPA_GRID:
        payloads = []
        for g3 in g3_payloads:
            applied_gate = np.where(
                g3["applied_gate"] >= kappa, g3["applied_gate"], 0.0
            )
            payload = _g4_payload_from_g3(g3, "g4", applied_gate)
            payload["selected_kappa"] = float(kappa)
            payloads.append(payload)
        farm_frame, macro = _macro_for_payloads(payloads)
        indexed = farm_frame.set_index("farm_id")[SELECTION_METRIC]
        relative_g0 = (indexed - g0_nrmse) / g0_nrmse
        relative_g3 = (indexed - g3_nrmse) / g3_nrmse
        nrmse_guard_g0 = macro[MACRO_SELECTION_METRIC] <= g0_macro[
            MACRO_SELECTION_METRIC
        ] * (1.0 + NRMSE_GUARD_RELATIVE)
        nrmse_guard_g3 = macro[MACRO_SELECTION_METRIC] <= g3_macro[
            MACRO_SELECTION_METRIC
        ] * (1.0 + NRMSE_GUARD_RELATIVE)
        farm_guard_count_g0 = int((relative_g0 <= FARM_NRMSE_GUARD_RELATIVE).sum())
        farm_guard_count_g3 = int((relative_g3 <= FARM_NRMSE_GUARD_RELATIVE).sum())
        regret_reduction_g0 = _relative_reduction(
            macro["macro_positive_regret_mean"],
            g0_macro["macro_positive_regret_mean"],
        )
        harm_reduction_g0 = _relative_reduction(
            macro["macro_harm_rate_0_005"],
            g0_macro["macro_harm_rate_0_005"],
        )
        regret_reduction_g3 = _relative_reduction(
            macro["macro_positive_regret_mean"],
            g3_macro["macro_positive_regret_mean"],
        )
        harm_reduction_g3 = _relative_reduction(
            macro["macro_harm_rate_0_005"],
            g3_macro["macro_harm_rate_0_005"],
        )
        safety_nondegrading_g0, safety_target_g0 = _safety_reduction_guard(
            regret_reduction_g0, harm_reduction_g0
        )
        safety_nondegrading_g3, safety_target_g3 = _safety_reduction_guard(
            regret_reduction_g3, harm_reduction_g3
        )
        row = {
            "kappa": float(kappa),
            **macro,
            "relative_nrmse_vs_g0": (
                macro[MACRO_SELECTION_METRIC] / g0_macro[MACRO_SELECTION_METRIC] - 1.0
            ),
            "relative_nrmse_vs_g3": (
                macro[MACRO_SELECTION_METRIC] / g3_macro[MACRO_SELECTION_METRIC] - 1.0
            ),
            "farms_within_1pct_g0": farm_guard_count_g0,
            "farms_within_1pct_g3": farm_guard_count_g3,
            "nrmse_guard_vs_g0_pass": bool(nrmse_guard_g0),
            "nrmse_guard_vs_g3_pass": bool(nrmse_guard_g3),
            "farm_guard_vs_g0_pass": farm_guard_count_g0 >= MIN_FARMS_WITHIN_GUARD,
            "farm_guard_vs_g3_pass": farm_guard_count_g3 >= MIN_FARMS_WITHIN_GUARD,
            "positive_regret_reduction_vs_g0": regret_reduction_g0,
            "harm_rate_reduction_vs_g0": harm_reduction_g0,
            "positive_regret_reduction_vs_g3": regret_reduction_g3,
            "harm_rate_reduction_vs_g3": harm_reduction_g3,
            "safety_nondegrading_vs_g0": safety_nondegrading_g0,
            "safety_nondegrading_vs_g3": safety_nondegrading_g3,
            "safety_reduction_target_vs_g0_pass": safety_target_g0,
            "safety_reduction_target_vs_g3_pass": safety_target_g3,
        }
        row["strict_eligible"] = bool(
            row["nrmse_guard_vs_g0_pass"]
            and row["nrmse_guard_vs_g3_pass"]
            and row["farm_guard_vs_g0_pass"]
            and row["farm_guard_vs_g3_pass"]
            and row["safety_reduction_target_vs_g0_pass"]
            and row["safety_reduction_target_vs_g3_pass"]
        )
        rows.append(row)
        candidate_payloads[float(kappa)] = payloads
    table = pd.DataFrame(rows)
    strict = table[table["strict_eligible"]].copy()
    if len(strict):
        selected = strict.sort_values("kappa", kind="stable").iloc[0]
        status = "minimum_kappa_passing_all_safety_and_accuracy_guards"
    else:
        fallback = table[
            table["nrmse_guard_vs_g0_pass"]
            & table["nrmse_guard_vs_g3_pass"]
            & table["farm_guard_vs_g0_pass"]
            & table["farm_guard_vs_g3_pass"]
            & table["safety_nondegrading_vs_g0"]
            & table["safety_nondegrading_vs_g3"]
        ].copy()
        if fallback.empty:
            fallback = table.copy()
            status = "fallback_no_kappa_passed_accuracy_guard"
            selected = fallback.sort_values(
                [
                    "macro_positive_regret_mean",
                    "macro_harm_rate_0_005",
                    MACRO_SELECTION_METRIC,
                    "kappa",
                ],
                kind="stable",
            ).iloc[0]
        else:
            status = "fallback_no_kappa_reached_20pct_safety_reduction"
            selected = fallback.sort_values("kappa", kind="stable").iloc[0]
    selected_kappa = float(selected["kappa"])
    table["selected"] = np.isclose(table["kappa"], selected_kappa)
    return selected_kappa, status, table, candidate_payloads[selected_kappa]


def hard_payloads_from_g3(g3_payloads, selected_kappa):
    payloads = []
    for g3 in g3_payloads:
        hard_gate = (g3["applied_gate"] >= selected_kappa).astype(np.float64)
        payload = _g4_payload_from_g3(g3, HARD_CONTROL_ID, hard_gate)
        payload["selected_kappa"] = float(selected_kappa)
        payloads.append(payload)
    return payloads


def _assert_payload_pair_alignment(reference, target):
    identity_fields = ("sample_id", "horizon_step", "forecast_origin_time")
    for field in identity_fields:
        if not np.array_equal(reference[field], target[field]):
            raise ValueError(
                f"{reference['variant_id']}->{target['variant_id']}/"
                f"{target['farm_id']}的{field}未对齐"
            )
    if not np.array_equal(reference["y_true"], target["y_true"], equal_nan=True):
        raise ValueError(
            f"{reference['variant_id']}->{target['variant_id']}/"
            f"{target['farm_id']}真值未对齐"
        )


def build_candidate_drift_report(payloads):
    comparisons = [("g0", variant_id, False) for variant_id in ("g1", "g2", "g3", "g4")]
    comparisons.append(("g3", "g4", True))
    rows = []
    expected_variants = set(gate_train.VARIANT_SPECS)
    if set(payloads) != expected_variants:
        raise ValueError(f"candidate drift需要完整G0--G4，实际{sorted(payloads)}")
    expected_farms = set(map(str, gate_train.expected_farm_ids()))
    for variant_id, items in payloads.items():
        farms = [str(item["farm_id"]) for item in items]
        if (
            len(items) != len(expected_farms)
            or len(set(farms)) != len(farms)
            or set(farms) != expected_farms
        ):
            raise ValueError(
                f"candidate drift/{variant_id}必须恰好覆盖5个预期场站: {farms}"
            )
    by_variant = {
        variant_id: {item["farm_id"]: item for item in items}
        for variant_id, items in payloads.items()
    }
    for baseline_id, target_id, strict_pair in comparisons:
        if set(by_variant[baseline_id]) != set(by_variant[target_id]):
            raise ValueError(f"{baseline_id}->{target_id}候选场站集合不一致")
        for farm_id, baseline in by_variant[baseline_id].items():
            target = by_variant[target_id][farm_id]
            _assert_payload_pair_alignment(baseline, target)
            pair_exact = bool(
                np.array_equal(
                    baseline["persistence_scaled"],
                    target["persistence_scaled"],
                    equal_nan=True,
                )
                and np.array_equal(
                    baseline["corrected_scaled"],
                    target["corrected_scaled"],
                    equal_nan=True,
                )
            )
            if strict_pair and not pair_exact:
                raise ValueError(f"{baseline_id}->{target_id}/{farm_id}候选发生漂移")
            scopes = [("all", np.ones_like(baseline["y_true"], dtype=bool))]
            for horizon in range(baseline["forecast_len"]):
                mask = np.zeros_like(baseline["y_true"], dtype=bool)
                mask[:, horizon] = True
                scopes.append((horizon + 1, mask))
            for horizon_step, scope in scopes:
                valid = scope.copy()
                for value in (
                    baseline["y_true"],
                    baseline["persistence"],
                    baseline["corrected"],
                    target["persistence"],
                    target["corrected"],
                ):
                    valid &= np.isfinite(value)
                p_scaled_difference = (
                    target["persistence_scaled"] - baseline["persistence_scaled"]
                )[valid]
                c_scaled_difference = (
                    target["corrected_scaled"] - baseline["corrected_scaled"]
                )[valid]
                p_difference = (target["persistence"] - baseline["persistence"])[valid]
                c_difference = (target["corrected"] - baseline["corrected"])[valid]
                baseline_oracle = np.abs(
                    baseline["corrected"][valid] - baseline["y_true"][valid]
                ) < np.abs(baseline["persistence"][valid] - baseline["y_true"][valid])
                target_oracle = np.abs(
                    target["corrected"][valid] - target["y_true"][valid]
                ) < np.abs(target["persistence"][valid] - target["y_true"][valid])
                target_gate = target["raw_gate"][valid]
                if valid.any():
                    p_scaled_max = float(np.max(np.abs(p_scaled_difference)))
                    c_scaled_max = float(np.max(np.abs(c_scaled_difference)))
                    p_nrmse_drift = float(
                        np.sqrt(np.mean(np.square(p_difference))) / baseline["capacity"]
                    )
                    c_nrmse_drift = float(
                        np.sqrt(np.mean(np.square(c_difference))) / baseline["capacity"]
                    )
                    baseline_prevalence = float(baseline_oracle.mean())
                    target_prevalence = float(target_oracle.mean())
                    oracle_agreement = float(np.mean(baseline_oracle == target_oracle))
                    fixed_baseline_oracle_brier = float(
                        np.mean(np.square(target_gate - baseline_oracle.astype(float)))
                    )
                    fixed_baseline_oracle_ece = _ece(
                        target_gate,
                        baseline_oracle,
                        bins=10,
                        adaptive=False,
                    )
                else:
                    p_scaled_max = c_scaled_max = np.nan
                    p_nrmse_drift = c_nrmse_drift = np.nan
                    baseline_prevalence = target_prevalence = np.nan
                    oracle_agreement = np.nan
                    fixed_baseline_oracle_brier = np.nan
                    fixed_baseline_oracle_ece = np.nan
                rows.append(
                    {
                        "baseline_variant": baseline_id,
                        "target_variant": target_id,
                        "farm_id": farm_id,
                        "horizon_step": horizon_step,
                        "valid_count": int(valid.sum()),
                        "strict_same_candidate_pair_expected": strict_pair,
                        "candidate_pair_exact_full_archive": pair_exact,
                        "persistence_scaled_max_abs_drift": p_scaled_max,
                        "corrected_scaled_max_abs_drift": c_scaled_max,
                        "persistence_normalized_rmse_drift": p_nrmse_drift,
                        "corrected_normalized_rmse_drift": c_nrmse_drift,
                        "baseline_oracle_prevalence": baseline_prevalence,
                        "target_oracle_prevalence": target_prevalence,
                        "oracle_label_agreement": oracle_agreement,
                        "fixed_baseline_oracle_brier": fixed_baseline_oracle_brier,
                        "fixed_baseline_oracle_ece_10bin": fixed_baseline_oracle_ece,
                        "calibration_comparison_scope": (
                            "strict_controlled_same_oracle"
                            if pair_exact
                            else "end_to_end_descriptive_different_oracle"
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    identity = [
        "baseline_variant",
        "target_variant",
        "farm_id",
        "horizon_step",
    ]
    if result.duplicated(identity).any():
        raise ValueError("candidate drift报告存在重复比较键")
    required = (
        "valid_count",
        "persistence_scaled_max_abs_drift",
        "corrected_scaled_max_abs_drift",
        "persistence_normalized_rmse_drift",
        "corrected_normalized_rmse_drift",
        "baseline_oracle_prevalence",
        "target_oracle_prevalence",
        "oracle_label_agreement",
        "fixed_baseline_oracle_brier",
        "fixed_baseline_oracle_ece_10bin",
    )
    for column in required:
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"candidate drift报告{column}包含非有限值")
    if (result["valid_count"] <= 0).any():
        raise ValueError("candidate drift报告存在零有效样本scope")
    return result


def attach_candidate_drift_summary(comparison, candidate_drift):
    overall = candidate_drift[
        (candidate_drift["baseline_variant"] == "g0")
        & (candidate_drift["horizon_step"].astype(str) == "all")
    ]
    rows = [
        {
            "model_variant": "g0",
            "candidate_pair_exact_vs_g0_all_farms": True,
            "corrected_candidate_max_abs_drift_vs_g0": 0.0,
            "corrected_candidate_nrmse_drift_vs_g0": 0.0,
            "oracle_label_agreement_vs_g0_macro": 1.0,
            "cross_variant_calibration_role": "reference",
            "macro_fixed_g0_oracle_brier": np.nan,
            "macro_fixed_g0_oracle_ece_10bin": np.nan,
        }
    ]
    for variant_id, frame in overall.groupby("target_variant"):
        _assert_exact_farm_metrics(
            frame,
            f"candidate drift G0->{variant_id}",
            (
                "valid_count",
                "corrected_scaled_max_abs_drift",
                "corrected_normalized_rmse_drift",
                "oracle_label_agreement",
                "fixed_baseline_oracle_brier",
                "fixed_baseline_oracle_ece_10bin",
            ),
        )
        exact = bool(frame["candidate_pair_exact_full_archive"].all())
        rows.append(
            {
                "model_variant": variant_id,
                "candidate_pair_exact_vs_g0_all_farms": exact,
                "corrected_candidate_max_abs_drift_vs_g0": float(
                    frame["corrected_scaled_max_abs_drift"].max()
                ),
                "corrected_candidate_nrmse_drift_vs_g0": float(
                    frame["corrected_normalized_rmse_drift"].mean()
                ),
                "oracle_label_agreement_vs_g0_macro": float(
                    frame["oracle_label_agreement"].mean()
                ),
                "cross_variant_calibration_role": (
                    "strict_controlled_same_oracle"
                    if exact
                    else "end_to_end_descriptive_different_oracle"
                ),
                "macro_fixed_g0_oracle_brier": float(
                    frame["fixed_baseline_oracle_brier"].mean()
                ),
                "macro_fixed_g0_oracle_ece_10bin": float(
                    frame["fixed_baseline_oracle_ece_10bin"].mean()
                ),
            }
        )
    if set(overall["target_variant"].astype(str)) != {"g1", "g2", "g3", "g4"}:
        raise ValueError("candidate drift缺少G0到G1--G4的完整overall比较")
    result = comparison.merge(
        pd.DataFrame(rows), on="model_variant", how="left", validate="one_to_one"
    )
    g0_mask = result["model_variant"] == "g0"
    result.loc[g0_mask, "macro_fixed_g0_oracle_brier"] = result.loc[
        g0_mask, "macro_oracle_brier"
    ]
    result.loc[g0_mask, "macro_fixed_g0_oracle_ece_10bin"] = result.loc[
        g0_mask, "macro_ece_10bin"
    ]
    fixed_columns = (
        "macro_fixed_g0_oracle_brier",
        "macro_fixed_g0_oracle_ece_10bin",
    )
    if (
        len(result) != len(gate_train.VARIANT_SPECS)
        or result["model_variant"].duplicated().any()
    ):
        raise ValueError("fixed-G0 oracle汇总不是G0--G4唯一矩阵")
    for column in fixed_columns:
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"fixed-G0 oracle汇总{column}包含非有限值")
    return result


def _concat_results(results, key):
    frames = [item[key] for item in results if key in item and item[key] is not None]
    return (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )


def build_variant_comparison(summary, safety):
    expected_variants = set(gate_train.VARIANT_SPECS)
    if set(summary["model_variant"].astype(str)) != expected_variants:
        raise ValueError("正式summary必须完整覆盖G0--G4")
    if summary.duplicated(["model_variant", "farm_id"]).any():
        raise ValueError("正式summary存在重复variant/farm")
    overall_safety = safety[
        (safety["scope_type"] == "overall")
        & (safety["scope_value"].astype(str) == "all")
    ]
    rows = []
    for variant_id, frame in summary.groupby("model_variant"):
        _assert_exact_farm_metrics(
            frame,
            f"{variant_id}正式summary",
            (
                SELECTION_METRIC,
                "parameter_count",
                "inference_milliseconds_per_sample",
            ),
        )
        safe = overall_safety[overall_safety["model_variant"] == variant_id]
        _assert_exact_farm_metrics(
            safe,
            f"{variant_id} overall安全指标",
            (
                "valid_count",
                "positive_regret_mean",
                "positive_regret_p95",
                "harm_rate_0_005",
                "oracle_brier",
                "ece_10bin",
                "gate_high_saturation_rate",
                "coverage_rate",
            ),
        )
        if (pd.to_numeric(safe["valid_count"], errors="coerce") <= 0).any():
            raise ValueError(f"{variant_id} overall安全指标存在零有效样本场站")
        variant_safety = safety[safety["model_variant"] == variant_id]
        stable_frame = variant_safety[
            (variant_safety["scope_type"] == "regime")
            & (variant_safety["scope_value"].astype(str) == "stable")
        ]
        _assert_exact_farm_metrics(
            stable_frame,
            f"{variant_id} stable门控指标",
            ("valid_count", "gate_mean"),
        )
        if (pd.to_numeric(stable_frame["valid_count"], errors="coerce") <= 0).any():
            raise ValueError(f"{variant_id} stable门控指标存在零有效样本场站")
        stable_rows = stable_frame.set_index(stable_frame["farm_id"].astype(str))
        dynamic_rows = variant_safety[
            (variant_safety["scope_type"] == "regime")
            & variant_safety["scope_value"].astype(str).isin(["ramp_up", "ramp_down"])
        ]
        dynamic_farms = set(dynamic_rows["farm_id"].astype(str))
        if dynamic_farms != set(map(str, gate_train.expected_farm_ids())):
            raise ValueError(f"{variant_id} dynamic门控指标未覆盖5个预期场站")
        if dynamic_rows.duplicated(["farm_id", "scope_value"]).any():
            raise ValueError(f"{variant_id} dynamic门控指标存在重复farm/scope")
        dynamic_by_farm = {}
        for farm_id, farm_rows in dynamic_rows.groupby("farm_id"):
            weights = farm_rows["valid_count"].to_numpy(dtype=float)
            gate_values = farm_rows["gate_mean"].to_numpy(dtype=float)
            usable = (weights > 0.0) & np.isfinite(gate_values)
            dynamic_by_farm[str(farm_id)] = (
                float(np.average(gate_values[usable], weights=weights[usable]))
                if usable.any()
                else np.nan
            )
        if (
            set(dynamic_by_farm) != set(map(str, gate_train.expected_farm_ids()))
            or not np.isfinite(list(dynamic_by_farm.values())).all()
        ):
            raise ValueError(f"{variant_id} dynamic门控宏平均存在缺失/非有限场站")
        gate_gaps = [
            dynamic_by_farm.get(str(farm_id), np.nan) - row["gate_mean"]
            for farm_id, row in stable_rows.iterrows()
        ]
        if (
            len(gate_gaps) != len(gate_train.expected_farm_ids())
            or not np.isfinite(gate_gaps).all()
        ):
            raise ValueError(f"{variant_id} dynamic-stable gate gap不是5场站有限值")
        dynamic_stable_gate_gap = float(np.mean(gate_gaps))
        rows.append(
            {
                "model_variant": variant_id,
                "variant_label": frame["variant_label"].iloc[0],
                "farm_count": int(frame["farm_id"].nunique()),
                "feature_groups": frame["feature_groups"].iloc[0],
                "parameter_count_max": int(frame["parameter_count"].max()),
                "macro_inference_milliseconds_per_sample": float(
                    frame["inference_milliseconds_per_sample"].mean()
                ),
                MACRO_SELECTION_METRIC: float(frame[SELECTION_METRIC].mean()),
                "cross_farm_nrmse_std": float(frame[SELECTION_METRIC].std(ddof=0)),
                "macro_positive_regret_mean": float(
                    safe["positive_regret_mean"].mean()
                ),
                "macro_positive_regret_p95": float(safe["positive_regret_p95"].mean()),
                "macro_harm_rate_0_005": float(safe["harm_rate_0_005"].mean()),
                "macro_oracle_brier": float(safe["oracle_brier"].mean()),
                "macro_ece_10bin": float(safe["ece_10bin"].mean()),
                "macro_high_saturation_rate": float(
                    safe["gate_high_saturation_rate"].mean()
                ),
                "macro_utility_gap": float(safe["utility_gap"].mean()),
                "macro_coverage_rate": float(safe["coverage_rate"].mean()),
                "dynamic_stable_gate_gap": dynamic_stable_gate_gap,
                "selected_kappa": frame["selected_kappa"].dropna().iloc[0]
                if frame["selected_kappa"].notna().any()
                else np.nan,
            }
        )
    comparison = pd.DataFrame(rows)
    g0 = comparison[comparison["model_variant"] == "g0"].iloc[0]
    g3 = comparison[comparison["model_variant"] == "g3"].iloc[0]
    g0_farms = summary[summary["model_variant"] == "g0"].set_index("farm_id")
    g3_farms = summary[summary["model_variant"] == "g3"].set_index("farm_id")
    guard_rows = []
    for _, row in comparison.iterrows():
        variant_id = row["model_variant"]
        farms = summary[summary["model_variant"] == variant_id].set_index("farm_id")
        relative = (farms[SELECTION_METRIC] - g0_farms[SELECTION_METRIC]) / g0_farms[
            SELECTION_METRIC
        ]
        regret_reduction = _relative_reduction(
            row["macro_positive_regret_mean"], g0["macro_positive_regret_mean"]
        )
        harm_reduction = _relative_reduction(
            row["macro_harm_rate_0_005"], g0["macro_harm_rate_0_005"]
        )
        safety_nondegrading, safety_target = _safety_reduction_guard(
            regret_reduction, harm_reduction
        )
        direct_regret_reduction = (
            _relative_reduction(
                row["macro_positive_regret_mean"],
                g3["macro_positive_regret_mean"],
            )
            if variant_id == "g4"
            else np.nan
        )
        direct_harm_reduction = (
            _relative_reduction(
                row["macro_harm_rate_0_005"], g3["macro_harm_rate_0_005"]
            )
            if variant_id == "g4"
            else np.nan
        )
        if variant_id == "g4":
            direct_nondegrading, direct_target = _safety_reduction_guard(
                direct_regret_reduction, direct_harm_reduction
            )
            direct_accuracy_guard = bool(
                row[MACRO_SELECTION_METRIC]
                <= g3[MACRO_SELECTION_METRIC] * (1.0 + NRMSE_GUARD_RELATIVE)
            )
            direct_farm_count = int(
                (
                    (farms[SELECTION_METRIC] - g3_farms[SELECTION_METRIC])
                    / g3_farms[SELECTION_METRIC]
                    <= FARM_NRMSE_GUARD_RELATIVE
                ).sum()
            )
        else:
            direct_nondegrading = direct_target = True
            direct_accuracy_guard = True
            direct_farm_count = len(farms)
        accuracy_guard = bool(
            row[MACRO_SELECTION_METRIC]
            <= g0[MACRO_SELECTION_METRIC] * (1.0 + NRMSE_GUARD_RELATIVE)
        )
        farm_count = int((relative <= FARM_NRMSE_GUARD_RELATIVE).sum())
        safety_pass = bool(variant_id == "g0" or safety_target)
        guard_rows.append(
            {
                "model_variant": variant_id,
                "relative_nrmse_vs_g0": (
                    row[MACRO_SELECTION_METRIC] / g0[MACRO_SELECTION_METRIC] - 1.0
                ),
                "farms_within_1pct_g0": farm_count,
                "positive_regret_reduction_vs_g0": regret_reduction,
                "harm_rate_reduction_vs_g0": harm_reduction,
                "safety_nondegrading_vs_g0": safety_nondegrading,
                "g4_positive_regret_reduction_vs_g3": direct_regret_reduction,
                "g4_harm_rate_reduction_vs_g3": direct_harm_reduction,
                "g4_safety_nondegrading_vs_g3": direct_nondegrading,
                "g4_direct_safety_target_pass": direct_target,
                "g4_accuracy_guard_vs_g3_pass": direct_accuracy_guard,
                "g4_farms_within_1pct_g3": direct_farm_count,
                "g4_farm_guard_vs_g3_pass": direct_farm_count >= MIN_FARMS_WITHIN_GUARD,
                "accuracy_guard_pass": accuracy_guard,
                "farm_guard_pass": farm_count >= MIN_FARMS_WITHIN_GUARD,
                "persistence_safety_pass": safety_pass,
                "selection_guard_pass": bool(
                    accuracy_guard
                    and farm_count >= MIN_FARMS_WITHIN_GUARD
                    and safety_pass
                    and direct_target
                    and direct_accuracy_guard
                    and direct_farm_count >= MIN_FARMS_WITHIN_GUARD
                ),
            }
        )
    comparison = comparison.merge(pd.DataFrame(guard_rows), on="model_variant")
    comparison = comparison.sort_values(MACRO_SELECTION_METRIC).reset_index(drop=True)
    comparison["nrmse_rank"] = np.arange(1, len(comparison) + 1)
    return comparison


def _regime_macro(regime, variant_id, regime_group, candidate="fused"):
    frame = regime[
        (regime["model_variant"] == variant_id)
        & (regime["regime_group"] == regime_group)
        & (regime["candidate"] == candidate)
        & (regime["horizon_step"].astype(str) == "all")
    ]
    _assert_exact_farm_metrics(
        frame,
        f"{variant_id}/{regime_group}/{candidate}工况指标",
        ("sample_count", "valid_count", SELECTION_METRIC),
    )
    if (pd.to_numeric(frame["sample_count"], errors="coerce") <= 0).any() or (
        pd.to_numeric(frame["valid_count"], errors="coerce") <= 0
    ).any():
        raise ValueError(f"{variant_id}/{regime_group}/{candidate}存在零有效样本场站")
    return float(pd.to_numeric(frame[SELECTION_METRIC]).to_numpy(dtype=float).mean())


def _scope_safety_macro(safety, variant_id, regime_group, metric):
    frame = safety[
        (safety["model_variant"] == variant_id)
        & (safety["scope_type"] == "regime")
        & (safety["scope_value"].astype(str) == regime_group)
    ]
    _assert_exact_farm_metrics(
        frame,
        f"{variant_id}/{regime_group}安全指标",
        ("valid_count", metric),
    )
    if (pd.to_numeric(frame["valid_count"], errors="coerce") <= 0).any():
        raise ValueError(f"{variant_id}/{regime_group}安全指标存在零有效样本场站")
    return float(pd.to_numeric(frame[metric]).to_numpy(dtype=float).mean())


def _persistence_gap_closure(g0_value, candidate_value, persistence_value):
    denominator = float(g0_value - persistence_value)
    if not np.isfinite(denominator) or denominator <= 0.0:
        return np.nan
    return float((g0_value - candidate_value) / denominator)


def add_predeclared_target_flags(comparison, regime, safety):
    comparison = comparison.copy()
    if (
        len(comparison) != len(gate_train.VARIANT_SPECS)
        or comparison["model_variant"].duplicated().any()
        or set(comparison["model_variant"].astype(str)) != set(gate_train.VARIANT_SPECS)
    ):
        raise ValueError("预声明资格输入不是G0--G4唯一矩阵")
    g0 = comparison.set_index("model_variant").loc["g0"]
    g0_stable = _regime_macro(regime, "g0", "stable")
    g0_low = _regime_macro(regime, "g0", "low_power")
    g0_up = _regime_macro(regime, "g0", "ramp_up")
    g0_down = _regime_macro(regime, "g0", "ramp_down")
    persistence_stable = _regime_macro(regime, "g0", "stable", candidate="persistence")
    persistence_low = _regime_macro(regime, "g0", "low_power", candidate="persistence")
    rows = []
    for _, item in comparison.iterrows():
        variant_id = item["model_variant"]
        stable = _regime_macro(regime, variant_id, "stable")
        low = _regime_macro(regime, variant_id, "low_power")
        up = _regime_macro(regime, variant_id, "ramp_up")
        down = _regime_macro(regime, variant_id, "ramp_down")
        stable_regret = _scope_safety_macro(
            safety, variant_id, "stable", "positive_regret_mean"
        )
        stable_harm = _scope_safety_macro(
            safety, variant_id, "stable", "harm_rate_0_005"
        )
        low_regret = _scope_safety_macro(
            safety, variant_id, "low_power", "positive_regret_mean"
        )
        low_harm = _scope_safety_macro(
            safety, variant_id, "low_power", "harm_rate_0_005"
        )
        g0_stable_regret = _scope_safety_macro(
            safety, "g0", "stable", "positive_regret_mean"
        )
        g0_stable_harm = _scope_safety_macro(safety, "g0", "stable", "harm_rate_0_005")
        g0_low_regret = _scope_safety_macro(
            safety, "g0", "low_power", "positive_regret_mean"
        )
        g0_low_harm = _scope_safety_macro(safety, "g0", "low_power", "harm_rate_0_005")
        rows.append(
            {
                "model_variant": variant_id,
                "stable_improvement_vs_g0": 1.0 - stable / g0_stable,
                "low_power_improvement_vs_g0": 1.0 - low / g0_low,
                "ramp_up_relative_change_vs_g0": up / g0_up - 1.0,
                "ramp_down_relative_change_vs_g0": down / g0_down - 1.0,
                "stable_persistence_gap_closure": _persistence_gap_closure(
                    g0_stable, stable, persistence_stable
                ),
                "low_power_persistence_gap_closure": _persistence_gap_closure(
                    g0_low, low, persistence_low
                ),
                "brier_reduction_vs_g0": _relative_reduction(
                    item["macro_fixed_g0_oracle_brier"],
                    g0["macro_fixed_g0_oracle_brier"],
                ),
                "ece_reduction_vs_g0": _relative_reduction(
                    item["macro_fixed_g0_oracle_ece_10bin"],
                    g0["macro_fixed_g0_oracle_ece_10bin"],
                ),
                "parameter_under_30k": item["parameter_count_max"]
                < gate_train.PARAMETER_LIMIT,
                "stable_positive_regret_reduction_vs_g0": _relative_reduction(
                    stable_regret, g0_stable_regret
                ),
                "stable_harm_rate_reduction_vs_g0": _relative_reduction(
                    stable_harm, g0_stable_harm
                ),
                "low_power_positive_regret_reduction_vs_g0": _relative_reduction(
                    low_regret, g0_low_regret
                ),
                "low_power_harm_rate_reduction_vs_g0": _relative_reduction(
                    low_harm, g0_low_harm
                ),
            }
        )
    flags = pd.DataFrame(rows)
    result = comparison.merge(flags, on="model_variant")
    result["stable_10pct_target"] = result["stable_improvement_vs_g0"] >= 0.10
    result["low_power_5pct_target"] = result["low_power_improvement_vs_g0"] >= 0.05
    result["ramp_guard_pass"] = (result["ramp_up_relative_change_vs_g0"] <= 0.005) & (
        result["ramp_down_relative_change_vs_g0"] <= 0.005
    )
    result["brier_10pct_target"] = result["brier_reduction_vs_g0"] >= 0.10
    result["ece_15pct_target"] = result["ece_reduction_vs_g0"] >= 0.15
    result["saturation_target"] = result["macro_high_saturation_rate"] < 0.50
    result["dynamic_stable_gap_0_15_target"] = result["dynamic_stable_gate_gap"] >= 0.15
    result["stable_gap_closure_25pct_target"] = (
        result["stable_persistence_gap_closure"] >= 0.25
    )
    result["low_power_gap_closure_20pct_target"] = (
        result["low_power_persistence_gap_closure"] >= 0.20
    )
    stable_safety_columns = [
        "stable_positive_regret_reduction_vs_g0",
        "stable_harm_rate_reduction_vs_g0",
    ]
    stable_safety_finite = np.isfinite(
        result[stable_safety_columns].to_numpy(dtype=float)
    ).all(axis=1)
    result["stable_safety_nondegrading"] = stable_safety_finite & (
        result[stable_safety_columns].min(axis=1, skipna=False)
        >= -SAFETY_NONDEGRADATION_TOLERANCE
    )
    result["stable_safety_20pct_target"] = stable_safety_finite & (
        result[stable_safety_columns].max(axis=1, skipna=False) >= 0.20
    )
    low_power_safety_columns = [
        "low_power_positive_regret_reduction_vs_g0",
        "low_power_harm_rate_reduction_vs_g0",
    ]
    low_power_safety_finite = np.isfinite(
        result[low_power_safety_columns].to_numpy(dtype=float)
    ).all(axis=1)
    result["low_power_safety_nondegrading"] = low_power_safety_finite & (
        result[low_power_safety_columns].min(axis=1, skipna=False)
        >= -SAFETY_NONDEGRADATION_TOLERANCE
    )
    result["low_power_safety_20pct_target"] = low_power_safety_finite & (
        result[low_power_safety_columns].max(axis=1, skipna=False) >= 0.20
    )
    target_columns = [
        "accuracy_guard_pass",
        "farm_guard_pass",
        "stable_10pct_target",
        "low_power_5pct_target",
        "stable_gap_closure_25pct_target",
        "low_power_gap_closure_20pct_target",
        "ramp_guard_pass",
        "brier_10pct_target",
        "ece_15pct_target",
        "saturation_target",
        "dynamic_stable_gap_0_15_target",
        "stable_safety_20pct_target",
        "low_power_safety_20pct_target",
        "parameter_under_30k",
    ]
    result["predeclared_qualification_pass"] = result[target_columns].all(axis=1)
    g4_mask = result["model_variant"] == "g4"
    result.loc[g4_mask, "predeclared_qualification_pass"] &= result.loc[
        g4_mask,
        [
            "g4_accuracy_guard_vs_g3_pass",
            "g4_farm_guard_vs_g3_pass",
            "persistence_safety_pass",
            "safety_nondegrading_vs_g0",
            "g4_safety_nondegrading_vs_g3",
            "g4_direct_safety_target_pass",
        ],
    ].all(axis=1)
    result["selection_guard_pass"] = result["predeclared_qualification_pass"]
    result.loc[result["model_variant"] == "g0", "selection_guard_pass"] = True
    result["secondary_targets_selection_role"] = "hard_predeclared_qualification"
    return result


def select_final_model(comparison):
    eligible_nonreference = comparison[
        comparison["selection_guard_pass"] & (comparison["model_variant"] != "g0")
    ].copy()
    if eligible_nonreference.empty:
        eligible = comparison[comparison["model_variant"] == "g0"].copy()
        selection_status = "fallback_g0_no_candidate_passed_guards"
    else:
        eligible = comparison[comparison["selection_guard_pass"]].copy()
        selection_status = "safety_then_accuracy_lexicographic"
    best_nrmse = float(eligible[MACRO_SELECTION_METRIC].min())
    near = eligible[
        eligible[MACRO_SELECTION_METRIC] <= best_nrmse * (1.0 + NRMSE_TIE_RELATIVE)
    ].copy()
    selected = near.sort_values(
        [
            "macro_positive_regret_mean",
            "macro_fixed_g0_oracle_brier",
            "parameter_count_max",
            "macro_inference_milliseconds_per_sample",
            MACRO_SELECTION_METRIC,
        ],
        kind="stable",
    ).iloc[0]
    comparison = comparison.copy()
    comparison["selected"] = comparison["model_variant"] == selected["model_variant"]
    comparison["selection_status"] = selection_status
    return selected, comparison


def _save_aggregate_figures(comparison, summary, horizon, calibration, output_dir):
    dirs = {"matplotlib_cache": os.path.join(output_dir, "matplotlib_cache")}
    plt = common_predict.setup_matplotlib(dirs)
    paths = {}
    ordered = comparison.sort_values(MACRO_SELECTION_METRIC)
    path = os.path.join(output_dir, "controlled_gate_cali_test_nrmse_rank.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#2a9d8f" if value else "#6c757d" for value in ordered["selected"]]
    ax.bar(ordered["model_variant"], ordered[MACRO_SELECTION_METRIC], color=colors)
    ax.set_ylabel("Five-farm macro NRMSE")
    ax.set_title("G0-G4 test selection")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["rank_figure"] = path

    path = os.path.join(output_dir, "controlled_gate_cali_test_farm_heatmap.png")
    matrix = summary.pivot(
        index="model_variant", columns="farm_id", values=SELECTION_METRIC
    ).reindex(ordered["model_variant"])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis_r")
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.set_xticks(
        np.arange(len(matrix.columns)),
        [str(item)[-4:] for item in matrix.columns],
        rotation=30,
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                f"{matrix.iloc[row, column]:.4f}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="NRMSE")
    ax.set_title("Per-farm test NRMSE")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["farm_heatmap"] = path

    path = os.path.join(output_dir, "controlled_gate_cali_test_horizon_nrmse.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    values = horizon[horizon["horizon_step"].astype(str) != "all"].copy()
    values["horizon_step"] = pd.to_numeric(values["horizon_step"])
    macro = values.groupby(["model_variant", "horizon_step"])[SELECTION_METRIC].mean()
    for variant_id in ordered["model_variant"]:
        line = macro.loc[variant_id]
        ax.plot(line.index * 15, line.values, marker="o", label=variant_id)
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("Macro NRMSE")
    ax.set_title("Horizon-wise test NRMSE")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["horizon_figure"] = path

    path = os.path.join(output_dir, "controlled_gate_cali_reliability.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    calibration_rows = []
    for (variant_id, gate_bin), group in calibration.groupby(
        ["model_variant", "gate_bin"], sort=False
    ):
        weights = group["count"].to_numpy(dtype=float)
        count = float(weights.sum())
        calibration_rows.append(
            {
                "model_variant": variant_id,
                "gate_bin": gate_bin,
                "mean_gate": (
                    float(
                        np.average(
                            group["mean_raw_gate"].fillna(0).to_numpy(float),
                            weights=weights,
                        )
                    )
                    if count
                    else np.nan
                ),
                "observed": (
                    float(
                        np.average(
                            group["corrected_better_rate"].fillna(0).to_numpy(float),
                            weights=weights,
                        )
                    )
                    if count
                    else np.nan
                ),
            }
        )
    cal = pd.DataFrame(calibration_rows)
    for variant_id, frame in cal.groupby("model_variant"):
        frame = frame.dropna(subset=["mean_gate", "observed"])
        ax.plot(frame["mean_gate"], frame["observed"], marker="o", label=variant_id)
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    ax.set_xlabel("Mean raw gate")
    ax.set_ylabel("Corrected-better frequency")
    ax.set_title("Finite-masked gate reliability")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["reliability_figure"] = path

    path = os.path.join(output_dir, "controlled_gate_cali_safety_accuracy.png")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        comparison[MACRO_SELECTION_METRIC],
        comparison["macro_positive_regret_mean"],
        s=70,
    )
    for _, row in comparison.iterrows():
        ax.annotate(
            row["model_variant"],
            (row[MACRO_SELECTION_METRIC], row["macro_positive_regret_mean"]),
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Macro test NRMSE (lower is better)")
    ax.set_ylabel("Mean positive regret (lower is safer)")
    ax.set_title("Accuracy-safety trade-off")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["safety_accuracy_figure"] = path
    return paths


def write_selection_report(
    comparison,
    selected,
    kappa_status,
    kappa_table,
    figure_paths,
    output_path,
):
    columns = [
        "model_variant",
        "feature_groups",
        "parameter_count_max",
        MACRO_SELECTION_METRIC,
        "macro_positive_regret_mean",
        "macro_harm_rate_0_005",
        "macro_oracle_brier",
        "macro_ece_10bin",
        "macro_fixed_g0_oracle_brier",
        "macro_fixed_g0_oracle_ece_10bin",
        "macro_high_saturation_rate",
        "predeclared_qualification_pass",
        "cross_variant_calibration_role",
        "selection_guard_pass",
        "selected",
    ]
    lines = [
        "# G0--G4两候选门控校准与Persistence保护测试集选型",
        "",
        f"最终选中 **{selected['model_variant']}**，5场站等权宏平均NRMSE="
        f"`{selected[MACRO_SELECTION_METRIC]:.9f}`。",
        "",
        "本轮按用户要求使用当前测试集选择统一G4 kappa并筛选最终模型；该测试段"
        "已参与此前结构选择，因此属于legacy-seen探索性结果，不是最终盲测。",
        "",
        "## 正式排名与安全守门",
        "",
        comparison[columns].to_markdown(index=False),
        "",
        "## G4统一kappa",
        "",
        f"kappa选择状态：`{kappa_status}`；选择值："
        f"`{kappa_table.loc[kappa_table['selected'], 'kappa'].iloc[0]:.2f}`。",
        "",
        kappa_table.to_markdown(index=False),
        "",
        "## 解释边界",
        "",
        "- G0直接引用F7正式精度；新增逐点候选/校准/安全字段来自一次只读诊断前向，"
        "且已核验F7既有预测CSV的样本/时间/真值键；预测数值优先执行float32位级"
        "兼容检查，跨TensorFlow/CUDA运行时则使用已归档的容量归一化严格容差。",
        "- G1--G3从同一F7快照独立初始化；soft oracle和dynamic标签只用于训练loss。",
        "- G1--G3最终phase会按既定方案联合微调residual，因此跨G0--G3的Brier/ECE"
        "属于端到端描述性比较；candidate drift与oracle标签一致率已单独归档。",
        "- Brier/ECE硬晋级使用同一G0候选对生成的固定oracle标签；各模型基于自身"
        "candidate oracle的校准指标仅作为部署效用诊断，避免candidate drift改变分类任务。",
        "- G4与hard负对照都来自同一次G3 scaled候选和raw gate；hard不参与排名。",
        "- G4阈值同时接受相对G0的全局守门和相对G3的直接保护守门；两个安全轴"
        "均不得恶化，并至少一个达到20%改善目标；最小合格κ先由这些全局守门确定，"
        "随后G4仍需通过与其它模型相同的全部分工况硬晋级门槛。",
        "- raw gate用于Brier/ECE/AUC；G4 applied gate用于融合精度与安全指标。",
        "- 所有选型macro均强制恰好覆盖5个预期场站；关键指标缺失、重复、零有效"
        "样本或非有限值会终止正式选型，不允许Pandas跳过NaN后以少于5场晋级。",
        "- stable/low-power/ramp、Persistence gap closure、Brier/ECE、饱和率、"
        "dynamic-stable gap与分工况安全目标均为硬晋级门槛；通过后再按"
        "NRMSE→后悔→Brier→复杂度筛选。",
        "- 当前仍为单seed=2026，且沿用legacy scaler/插值协议。",
        "",
        "## 图形",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in figure_paths.items())
    return _atomic_write_text(output_path, "\n".join(lines))


def _file_record(path):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"正式bundle文件不存在: {path}")
    return {
        "path": os.path.abspath(path),
        "sha256": _sha256(path),
        "size_bytes": os.path.getsize(path),
    }


def publish_prediction_marker(
    training_marker_path,
    stage2_marker_path,
    test_files,
    formal_paths,
    per_result_paths,
):
    if len(per_result_paths) != 30:
        raise ValueError(
            f"正式预测bundle应包含25个G结果+5个hard结果，实际{len(per_result_paths)}"
        )
    files = {
        "prediction_code": _file_record(__file__),
        "training_code": _file_record(gate_train.__file__),
        "training_marker": _file_record(training_marker_path),
        "stage2_source_marker": _file_record(stage2_marker_path),
    }
    for key, path in formal_paths.items():
        files[f"formal.{key}"] = _file_record(path)
    for result_index, paths in enumerate(per_result_paths):
        for key, path in paths.items():
            if path is None:
                continue
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"result{result_index}.{key}声明路径不存在: {path}"
                )
            files[f"result{result_index}.{key}"] = _file_record(path)
    marker = {
        "status": "complete",
        "protocol_version": gate_train.PROTOCOL_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": gate_train.RANDOM_SEED,
        "selection_split": "test",
        "test_used_for_selection": True,
        "test_is_final_blind_evaluation": False,
        "g4_kappa_selected_on_test": True,
        "variants": list(gate_train.VARIANT_SPECS),
        "hard_control_selection_eligible": False,
        "test_files": {
            str(common_predict.get_farm_id(path)): _file_record(path)
            for path in test_files
        },
        "files": files,
    }
    return _atomic_write_json(marker, marker_path())


def main():
    configure_prediction_reproducibility()
    training_marker_path, training_marker = validate_training_bundle()
    stage2_marker_path, stage2_marker = validate_stage2_source_bundle()
    test_files = discover_test_files()
    variants = get_requested_variants()
    farms = [str(common_predict.get_farm_id(path)) for path in test_files]
    full_matrix = set(variants) == set(gate_train.VARIANT_SPECS) and set(farms) == set(
        gate_train.expected_farm_ids()
    )
    if not test_files:
        raise FileNotFoundError("未找到第三阶段测试文件")
    if not full_matrix:
        raise ValueError(
            "G4统一test kappa和最终选型要求G0--G4完整5场站；"
            "请勿用partial覆盖正式第三阶段结果"
        )
    _clear_prediction_marker()
    source_frames, source_paths = _source_f7_test_frames(stage2_marker)
    payloads = {variant_id: [] for variant_id in ("g0", "g1", "g2", "g3")}
    for test_file in test_files:
        farm_id = str(common_predict.get_farm_id(test_file))
        print(f"\n===== Stage-3预测 farm={farm_id} =====")
        payloads["g0"].append(predict_g0(test_file, source_frames))
        for variant_id in gate_train.TRAINABLE_VARIANTS:
            payloads[variant_id].append(
                predict_trainable_variant(variant_id, test_file, training_marker)
            )

    selected_kappa, kappa_status, kappa_table, g4_payloads = select_g4_kappa(
        payloads["g3"], payloads["g0"]
    )
    for payload in g4_payloads:
        payload["selected_kappa"] = selected_kappa
    payloads["g4"] = g4_payloads
    hard_payloads = hard_payloads_from_g3(payloads["g3"], selected_kappa)
    candidate_drift = build_candidate_drift_report(payloads)

    policy_path = os.path.join(
        gate_train.variant_dirs("g4")["preprocess"],
        "controlled_gate_cali_g4_test_selected_policy.json",
    )
    policy = {
        "policy_schema_version": 1,
        "protocol_version": gate_train.PROTOCOL_VERSION,
        "source_variant": "g3",
        "source_models": {
            item["farm_id"]: {
                "model_path": item["model_path"],
                "model_sha256": item["model_sha256"],
                "artifact_path": item["artifact_path"],
                "artifact_sha256": item["artifact_sha256"],
            }
            for item in payloads["g3"]
        },
        "kappa_grid": list(gate_train.G4_KAPPA_GRID),
        "selected_kappa": selected_kappa,
        "selection_status": kappa_status,
        "selection_split": "test",
        "test_used_for_selection": True,
        "test_is_final_blind_evaluation": False,
        "same_kappa_all_farms_and_horizons": True,
        "kappa_selection_rule": (
            "minimum_kappa_passing_global_accuracy_and_safety_guards_vs_g0_and_g3"
        ),
        "hard_negative_control_rule": "one_if_raw_gate_ge_selected_kappa_else_zero",
        "fusion_space": "scaled_candidate_space_before_inverse_and_clip",
        "raw_gate_used_for_calibration": True,
        "applied_gate_used_for_safety_and_accuracy": True,
    }
    _atomic_write_json(policy, policy_path)

    results = []
    for variant_id in gate_train.VARIANT_SPECS:
        for payload in payloads[variant_id]:
            write_prediction = variant_id != "g0"
            results.append(
                save_payload_outputs(payload, write_prediction=write_prediction)
            )
    hard_results = [
        save_payload_outputs(payload, write_prediction=True, hard_control=True)
        for payload in hard_payloads
    ]

    summary = _concat_results(results, "summary")
    horizon = _concat_results(results, "horizon")
    candidate = _concat_results(results, "candidate")
    regime = _concat_results(results, "regime")
    assignments = _concat_results(results, "assignments")
    safety = _concat_results(results, "safety")
    calibration = _concat_results(results, "calibration")
    hard_summary = _concat_results(hard_results, "summary")
    if len(summary) != 25 or summary.duplicated(["model_variant", "farm_id"]).any():
        raise ValueError("G0--G4正式summary不是5×5唯一矩阵")
    comparison = build_variant_comparison(summary, safety)
    comparison = attach_candidate_drift_summary(comparison, candidate_drift)
    comparison = add_predeclared_target_flags(comparison, regime, safety)
    selected, comparison = select_final_model(comparison)
    output_dir = comparison_dir()
    paths = {
        "summary": os.path.join(
            output_dir, "controlled_gate_cali_test_metrics_summary.csv"
        ),
        "horizon": os.path.join(
            output_dir, "controlled_gate_cali_test_metrics_by_horizon.csv"
        ),
        "candidate": os.path.join(
            output_dir, "controlled_gate_cali_test_candidate_metrics.csv"
        ),
        "regime": os.path.join(
            output_dir, "controlled_gate_cali_test_regime_metrics.csv"
        ),
        "assignments": os.path.join(
            output_dir, "controlled_gate_cali_test_regime_assignments.csv"
        ),
        "safety": os.path.join(output_dir, "controlled_gate_cali_test_gate_safety.csv"),
        "calibration": os.path.join(
            output_dir, "controlled_gate_cali_test_reliability.csv"
        ),
        "comparison": os.path.join(
            output_dir, "controlled_gate_cali_test_variant_comparison.csv"
        ),
        "kappa": os.path.join(
            output_dir, "controlled_gate_cali_g4_kappa_test_selection.csv"
        ),
        "hard_control": os.path.join(
            output_dir, "controlled_gate_cali_hard_control_summary.csv"
        ),
        "candidate_drift": os.path.join(
            output_dir, "controlled_gate_cali_test_candidate_drift.csv"
        ),
        "final_selection": os.path.join(
            output_dir, "controlled_gate_cali_test_final_selection.csv"
        ),
        "policy": policy_path,
    }
    frames = {
        "summary": summary,
        "horizon": horizon,
        "candidate": candidate,
        "regime": regime,
        "assignments": assignments,
        "safety": safety,
        "calibration": calibration,
        "comparison": comparison,
        "kappa": kappa_table,
        "hard_control": hard_summary,
        "candidate_drift": candidate_drift,
        "final_selection": comparison[comparison["selected"]].copy(),
    }
    for key, frame in frames.items():
        _atomic_to_csv(frame, paths[key])
    figure_paths = _save_aggregate_figures(
        comparison, summary, horizon, calibration, output_dir
    )
    paths.update(figure_paths)
    report_path = os.path.join(
        output_dir, "controlled_gate_cali_test_final_selection.md"
    )
    write_selection_report(
        comparison,
        selected,
        kappa_status,
        kappa_table,
        figure_paths,
        report_path,
    )
    paths["selection_report"] = report_path
    source_manifest = pd.DataFrame(
        [
            {
                "source": "Stage2 completed source bundle",
                "key": "bundle_marker",
                "path": stage2_marker_path,
                "sha256": _sha256(stage2_marker_path),
                "reuse_action": "validated_complete_dependency_chain",
            },
            *[
                {
                    "source": "F7 formal test aggregate",
                    "key": key,
                    "path": path,
                    "sha256": _sha256(path),
                    "reuse_action": "read_only_reference_no_retraining",
                }
                for key, path in source_paths.items()
            ],
        ]
    )
    source_manifest_path = os.path.join(
        output_dir, "controlled_gate_cali_source_reuse_manifest.csv"
    )
    _atomic_to_csv(source_manifest, source_manifest_path)
    paths["source_manifest"] = source_manifest_path
    marker = publish_prediction_marker(
        training_marker_path,
        stage2_marker_path,
        test_files,
        paths,
        [item["paths"] for item in results + hard_results],
    )
    print(f"\nG4统一test-selected kappa={selected_kappa:.2f} ({kappa_status})")
    print(
        f"最终测试集选择: {selected['model_variant']} / "
        f"macro NRMSE={selected[MACRO_SELECTION_METRIC]:.9f}"
    )
    print(f"正式报告: {report_path}")
    print(f"预测bundle完成标志: {marker}")


if __name__ == "__main__":
    main()
