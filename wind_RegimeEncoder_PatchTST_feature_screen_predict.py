"""RegimeEncoder-PatchTST 显式工况特征筛选补充测试入口。

默认只对 F8/FP0/FP4 执行新推理。F0--F7 从已完成的
``f0_f7_test_selection_output`` 聚合文件只读复用，不重新训练或推理。
完整 F0--F8 按测试集5场站等权宏平均容量归一化 RMSE 排名；
FP0/FP4 是独立的 Frozen-Pair control，不进入最终模型选型。

该测试集已用于选型，因此输出明确标记 ``legacy_seen`` 且不属于
最终盲测。所有补充产物使用新目录和 ``feature_screening_f0_f8``
前缀，不覆盖旧 F0--F7、R2--R5 或 ``testdata_predict_output``。
"""

import glob
import hashlib
import json
import os
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

import wind_dl_model_predict as common_predict
import wind_RegimeEncoder_PatchTST_predict as regime_predict
import wind_RegimeEncoder_PatchTST_train as regime_train
from wind_RegimeEncoder_PatchTST_feature_screen_train import (
    ARCHITECTURE_VERSION,
    EXPECTED_PARAMETER_COUNTS,
    EXPECTED_TRAINABLE_PARAMETER_COUNTS,
    FULL_FEATURE_NAMES,
    LEGACY_SELECTION_VARIANTS,
    MODEL_FAMILY,
    NEW_TRAINING_VARIANTS,
    PROBE_VARIANTS,
    RANDOM_SEED,
    RESULT_ROOT,
    R4_SOURCE_VARIANT,
    SELECTION_VARIANTS,
    SUPPLEMENT_PROTOCOL_VERSION,
    TRAINING_COMPLETION_NAME,
    VARIANT_SPECS,
    build_feature_screen_model_from_artifact,
    get_feature_screen_custom_objects,
    selected_feature_names,
    validate_feature_training_protocol,
    validate_r4_reference_artifact,
    variant_dirs,
    variant_model_name,
)

warnings.filterwarnings("ignore")


LEGACY_OUTPUT_SUBDIR = "f0_f7_test_selection_output"
OUTPUT_SUBDIR = "f8_fp_supplement_test_output"
EXTENDED_OUTPUT_SUBDIR = "f0_f8_test_selection_output"
PROBE_OUTPUT_SUBDIR = "frozen_pair_probe_test_output"
ANALYSIS_OUTPUT_SUBDIR = "f0_f8_probe_analysis_output"
TEST_REUSE_STATUS = "legacy_seen_used_for_f0_f8_feature_selection"
SELECTION_METRIC = "capacity_normalized_rmse"
SELECTION_MACRO_METRIC = "macro_mean_capacity_normalized_rmse"
FP_SCALED_ATOL = 1e-7
FP_PHYSICAL_ATOL = 1e-5
DRIFT_RELATIVE_LIMIT = 0.002
PRACTICAL_NRMSE_RELATIVE_PCT = 0.05
CALIBRATION_ABSOLUTE_TOLERANCE = 0.001
SAFETY_REGRET_ABSOLUTE_TOLERANCE = 0.0001
SAFETY_HARM_RATE_ABSOLUTE_TOLERANCE = 0.005
MIN_PRACTICAL_FARM_CONSISTENCY = 4
EXPECTED_TEST_FARM_COUNT = 5

LEGACY_AGGREGATE_FILES = {
    "summary": "feature_screening_f0_f7_test_metrics_summary.csv",
    "horizon": "feature_screening_f0_f7_test_metrics_by_horizon_all.csv",
    "candidate": "feature_screening_f0_f7_test_candidate_all.csv",
    "regime": "feature_screening_f0_f7_test_metrics_by_regime_all.csv",
    "gate": "feature_screening_f0_f7_test_gate_all.csv",
    "calibration": "feature_screening_f0_f7_test_gate_calibration_all.csv",
    "assignments": "feature_screening_f0_f7_test_regime_assignments_all.csv",
}


def configure_prediction_reproducibility():
    os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_SEED))
    keras.utils.set_random_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _sha256(path, chunk_size=1024 * 1024):
    if not path:
        return None
    try:
        path = os.fspath(path)
    except TypeError:
        return None
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _is_missing_scalar(value):
    """Return True for empty/NaN scalar metadata without coercing arrays."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if np.ndim(missing) == 0 else False


def _first_present(*values):
    return next((value for value in values if not _is_missing_scalar(value)), None)


def _atomic_to_csv(frame, path):
    """Publish a CSV only after its complete temporary file has been written."""
    path = _assert_new_output_path(path)
    temporary = f"{path}.tmp"
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def _atomic_write_text(path, text):
    path = _assert_new_output_path(path)
    temporary = f"{path}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            file.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def _resolve_existing_path(path):
    if _is_missing_scalar(path):
        return None
    candidates = [os.fspath(path)]
    if not os.path.isabs(candidates[0]):
        candidates.append(os.path.join(os.path.dirname(__file__), candidates[0]))
    return next((value for value in candidates if os.path.exists(value)), None)


def _assert_new_output_path(path):
    """拒绝将补充实验文件写入旧F0--F7或原第二阶段目录。"""
    resolved = os.path.realpath(os.path.abspath(path))
    result_root = os.path.realpath(os.path.abspath(RESULT_ROOT))
    legacy_root = os.path.realpath(
        os.path.abspath(os.path.join(RESULT_ROOT, LEGACY_OUTPUT_SUBDIR))
    )
    original_stage2 = os.path.realpath(
        os.path.abspath(os.path.join(regime_train.RESULT_ROOT, "testdata_predict_output"))
    )
    try:
        inside_result_root = os.path.commonpath([resolved, result_root]) == result_root
    except ValueError:
        inside_result_root = False
    if not inside_result_root:
        raise ValueError(f"补充输出越出专用RESULT_ROOT: {path}")
    for protected in (legacy_root, original_stage2):
        try:
            overlaps = os.path.commonpath([resolved, protected]) == protected
        except ValueError:
            overlaps = False
        if overlaps:
            raise ValueError(f"补充输出不得覆盖保护目录: {path}")
    return path


def prediction_output_dirs(variant_id):
    if variant_id not in NEW_TRAINING_VARIANTS:
        raise ValueError(
            f"{variant_id} 不属于补充推理变体 {NEW_TRAINING_VARIANTS}；"
            "F0--F7必须只读复用旧聚合结果"
        )
    root = os.path.join(variant_dirs(variant_id, create=True)["root"], OUTPUT_SUBDIR)
    _assert_new_output_path(root)
    dirs = {
        "root": root,
        "predictions": os.path.join(root, "predictions"),
        "candidate_archives": os.path.join(root, "candidate_archives"),
        "figures": os.path.join(root, "figures"),
        "single_windows": os.path.join(root, "single_window_comparisons"),
        "weighted_curves": os.path.join(root, "weighted_curves"),
        "router_diagnostics": os.path.join(root, "router_diagnostics"),
        "regime_assignments": os.path.join(root, "regime_assignments"),
        "regime_metrics": os.path.join(root, "regime_metrics"),
        "gate_diagnostics": os.path.join(root, "gate_diagnostics"),
        "candidate_metrics": os.path.join(root, "candidate_metrics"),
        "matplotlib_cache": os.path.join(root, "matplotlib_cache"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def comparison_output_dir():
    path = os.path.join(RESULT_ROOT, EXTENDED_OUTPUT_SUBDIR)
    _assert_new_output_path(path)
    os.makedirs(path, exist_ok=True)
    return path


def probe_output_dir():
    path = os.path.join(RESULT_ROOT, PROBE_OUTPUT_SUBDIR)
    _assert_new_output_path(path)
    os.makedirs(path, exist_ok=True)
    return path


def analysis_output_dir():
    path = os.path.join(RESULT_ROOT, ANALYSIS_OUTPUT_SUBDIR)
    _assert_new_output_path(path)
    os.makedirs(path, exist_ok=True)
    return path


def bundle_completion_marker_path():
    path = os.path.join(
        RESULT_ROOT,
        ANALYSIS_OUTPUT_SUBDIR,
        "feature_screening_f0_f8_fp_bundle_complete.json",
    )
    _assert_new_output_path(path)
    return path


def _clear_bundle_completion_marker():
    path = bundle_completion_marker_path()
    if os.path.exists(path):
        os.remove(path)


def _publish_bundle_completion_marker(*path_groups):
    files = {}
    for group_name, group in path_groups:
        for name, path in group.items():
            resolved = _resolve_existing_path(path)
            if resolved is None or not os.path.isfile(resolved):
                raise FileNotFoundError(
                    f"正式bundle缺少{group_name}.{name}: {path}"
                )
            files[f"{group_name}.{name}"] = {
                "path": os.path.abspath(resolved),
                "sha256": _sha256(resolved),
                "size_bytes": os.path.getsize(resolved),
            }
    dependency_columns = (
        "prediction_path",
        "candidate_archive_path",
        "horizon_metric_path",
        "candidate_metric_path",
        "regime_assignment_path",
        "regime_metric_path",
        "gate_diagnostics_path",
        "gate_calibration_path",
    )
    for group_name, group in path_groups:
        summary_path = group.get("summary")
        if not summary_path:
            continue
        summary = pd.read_csv(summary_path)
        for _, row in summary.iterrows():
            variant_id = str(row.get("model_variant"))
            if group_name == "selection" and variant_id != "f8":
                # 旧F0--F7由已验hash的legacy source manifest负责；这里只为
                # 新推理证据建立bundle依赖，避免重复哈希数百MB旧预测。
                continue
            identity = f"{variant_id}.{row.get('farm_id')}"
            for column in dependency_columns:
                if column not in summary or _is_missing_scalar(row.get(column)):
                    continue
                resolved = _resolve_existing_path(row.get(column))
                if resolved is None or not os.path.isfile(resolved):
                    raise FileNotFoundError(
                        f"正式bundle缺少证据文件{identity}.{column}"
                    )
                files[f"{group_name}.evidence.{identity}.{column}"] = {
                    "path": os.path.abspath(resolved),
                    "sha256": _sha256(resolved),
                    "size_bytes": os.path.getsize(resolved),
                }
    current_test_files = {}
    for path in common_predict.discover_test_files():
        farm_id = str(common_predict.get_farm_id(path))
        if farm_id in set(expected_test_farm_ids()):
            current_test_files[farm_id] = {
                "path": os.path.abspath(path),
                "sha256": _sha256(path),
                "size_bytes": os.path.getsize(path),
            }
    if set(current_test_files) != set(expected_test_farm_ids()):
        raise ValueError("prediction bundle无法锁定全部5个当前test CSV")
    for group_name, group in path_groups:
        summary_path = group.get("summary")
        if not summary_path:
            continue
        summary = pd.read_csv(summary_path)
        new_rows = summary[
            summary["model_variant"].astype(str).isin(NEW_TRAINING_VARIANTS)
        ]
        for _, row in new_rows.iterrows():
            farm_id = str(row["farm_id"])
            if row.get("test_file_sha256") != current_test_files[farm_id]["sha256"]:
                raise ValueError(
                    f"{group_name}/{row['model_variant']}/{farm_id} test CSV hash漂移"
                )
    payload = {
        "status": "complete",
        "bundle_protocol_version": SUPPLEMENT_PROTOCOL_VERSION,
        "completed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "random_seed": RANDOM_SEED,
        "expected_test_farm_ids": list(expected_test_farm_ids()),
        "selection_variants": list(SELECTION_VARIANTS),
        "probe_variants": list(PROBE_VARIANTS),
        "multi_seed_experiment_run": False,
        "candidate_invariance_required": True,
        "current_test_files": current_test_files,
        "legacy_f0_f7_raw_test_file_hash_available": False,
        "input_version_limitation": (
            "legacy F0-F7 did not store raw test CSV hashes; cross-run identity is "
            "verified by sample/horizon/time/target alignment, while current F8/FP "
            "raw test hashes are locked here"
        ),
        "files": files,
    }
    marker = bundle_completion_marker_path()
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    _atomic_write_text(
        marker,
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    return marker


def legacy_output_dir():
    """只读返回已完成F0--F7聚合目录；本脚本永不在此创建文件。"""
    return os.path.join(RESULT_ROOT, LEGACY_OUTPUT_SUBDIR)


def expected_test_farm_ids():
    """Lock the formal matrix to the five farms in the completed F0--F7 run."""
    path = os.path.join(
        legacy_output_dir(),
        LEGACY_AGGREGATE_FILES["summary"],
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"无法锁定正式测试场站，缺少旧summary: {path}")
    frame = pd.read_csv(path, usecols=["model_variant", "farm_id"])
    frame["model_variant"] = frame["model_variant"].astype(str)
    frame["farm_id"] = frame["farm_id"].astype(str)
    expected = None
    for variant_id in LEGACY_SELECTION_VARIANTS:
        variant_farms = set(
            frame.loc[frame["model_variant"] == variant_id, "farm_id"]
        )
        if expected is None:
            expected = variant_farms
        elif variant_farms != expected:
            raise ValueError("旧F0--F7 summary的场站集合不一致")
    expected = expected or set()
    if len(expected) != EXPECTED_TEST_FARM_COUNT:
        raise ValueError(
            f"正式测试矩阵必须锁定{EXPECTED_TEST_FARM_COUNT}个场站，"
            f"旧summary实际为{sorted(expected)}"
        )
    if len(frame) != len(LEGACY_SELECTION_VARIANTS) * EXPECTED_TEST_FARM_COUNT:
        raise ValueError("旧F0--F7 summary不是8×5唯一完整矩阵")
    if frame.duplicated(["model_variant", "farm_id"]).any():
        raise ValueError("旧F0--F7 summary包含重复variant/farm键")
    return tuple(sorted(expected))


def validate_training_bundle_completion():
    """Require a hash-complete 15-model training bundle before formal inference."""
    marker_path = os.path.join(RESULT_ROOT, TRAINING_COMPLETION_NAME)
    if not os.path.isfile(marker_path):
        raise FileNotFoundError(
            "正式F8/FP预测前必须完成15个新增模型训练，缺少training complete标志: "
            f"{marker_path}"
        )
    with open(marker_path, "r", encoding="utf-8") as file:
        marker = json.load(file)
    if (
        marker.get("status") != "complete"
        or marker.get("supplement_protocol_version")
        != SUPPLEMENT_PROTOCOL_VERSION
        or int(marker.get("random_seed", -1)) != RANDOM_SEED
        or int(marker.get("new_model_count", -1))
        != len(NEW_TRAINING_VARIANTS) * EXPECTED_TEST_FARM_COUNT
        or tuple(marker.get("new_training_variants", ()))
        != tuple(NEW_TRAINING_VARIANTS)
        or tuple(marker.get("expected_farm_ids", ()))
        != tuple(expected_test_farm_ids())
    ):
        raise ValueError(f"training complete标志协议/数量不匹配: {marker_path}")
    files = marker.get("files")
    if not isinstance(files, dict):
        raise TypeError("training complete标志缺少files字典")
    for name, metadata in files.items():
        if not isinstance(metadata, dict):
            raise TypeError(f"training bundle文件元数据异常: {name}")
        path = _resolve_existing_path(metadata.get("path"))
        if (
            path is None
            or _sha256(path) != metadata.get("sha256")
            or os.path.getsize(path) != int(metadata.get("size_bytes", -1))
        ):
            raise ValueError(f"training bundle文件hash/大小失配: {name}")
    for variant_id in NEW_TRAINING_VARIANTS:
        for farm_id in expected_test_farm_ids():
            prefix = f"{variant_id}.{farm_id}"
            artifact = load_feature_artifact(variant_id, farm_id)
            expected_paths = {
                "artifact_path": artifact["artifact_path"],
                "model_path": artifact["model_path"],
                "best_weights_path": artifact["best_weights_path"],
            }
            for field, expected_path in expected_paths.items():
                key = f"{prefix}.{field}"
                if key not in files:
                    raise KeyError(f"training bundle缺少文件索引: {key}")
                marker_file = _resolve_existing_path(files[key].get("path"))
                actual_file = _resolve_existing_path(expected_path)
                if (
                    marker_file is None
                    or actual_file is None
                    or os.path.realpath(marker_file) != os.path.realpath(actual_file)
                ):
                    raise ValueError(f"training bundle路径身份不一致: {key}")
    return marker_path


def get_requested_prediction_variants():
    """默认只推理新增F8/FP0/FP4，旧F变体始终从聚合文件复用。"""
    raw = os.getenv("WIND_FEATURE_SCREEN_PREDICT_VARIANTS")
    if raw is None:
        raw = os.getenv("WIND_FEATURE_SCREEN_VARIANTS")
    if not raw:
        return list(NEW_TRAINING_VARIANTS)
    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if any(item in {"all", "*"} for item in requested):
        return list(NEW_TRAINING_VARIANTS)
    invalid = sorted(set(requested) - set(VARIANT_SPECS))
    if invalid:
        raise ValueError(f"未知变体{invalid}；可选{list(VARIANT_SPECS)}")
    old_requested = [item for item in requested if item in LEGACY_SELECTION_VARIANTS]
    if old_requested:
        print(f"旧变体{old_requested}将只读复用，不重新推理")
    return list(
        dict.fromkeys(item for item in requested if item in NEW_TRAINING_VARIANTS)
    )


def discover_requested_test_files():
    files = common_predict.discover_test_files()
    requested = os.getenv("WIND_FEATURE_SCREEN_FARMS")
    if not requested:
        return files
    farm_ids = {item.strip() for item in requested.split(",") if item.strip()}
    return [path for path in files if common_predict.get_farm_id(path) in farm_ids]


def _artifact_path(variant_id, farm_id):
    model_name = variant_model_name(variant_id)
    return os.path.join(
        variant_dirs(variant_id, create=False)["preprocess"],
        f"{model_name}_farm_{farm_id}_preprocess.pkl",
    )


def load_feature_artifact(variant_id, farm_id):
    if variant_id not in NEW_TRAINING_VARIANTS:
        raise ValueError(f"{variant_id} 不属于补充训练变体")
    path = _artifact_path(variant_id, farm_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少 {variant_id}/{farm_id} artifact: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise TypeError(f"artifact 必须为dict: {path}")
    if artifact.get("variant_id") != variant_id:
        raise ValueError(f"artifact变体不匹配: {path}")
    if artifact.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(f"artifact架构版本不匹配: {path}")
    if int(artifact.get("random_seed", -1)) != RANDOM_SEED:
        raise ValueError(f"artifact seed必须为{RANDOM_SEED}: {path}")
    expected_names = selected_feature_names(variant_id)
    if tuple(artifact.get("full_regime_feature_names", ())) != FULL_FEATURE_NAMES:
        raise ValueError(f"artifact完整43维特征定义已漂移: {path}")
    if tuple(artifact.get("selected_regime_feature_names", ())) != expected_names:
        raise ValueError(f"artifact特征子集与{variant_id}定义不一致: {path}")
    if int(artifact.get("selected_regime_feature_count", -1)) != len(expected_names):
        raise ValueError(f"artifact特征维数与{variant_id}定义不一致: {path}")
    spec = VARIANT_SPECS[variant_id]
    if tuple(artifact.get("selected_regime_feature_groups", ())) != tuple(
        spec["groups"]
    ):
        raise ValueError(f"artifact特征组与{variant_id}定义不一致: {path}")
    if bool(artifact.get("freeze_candidates", spec["freeze_candidates"])) != bool(
        spec["freeze_candidates"]
    ):
        raise ValueError(f"artifact候选冻结标记与{variant_id}定义不一致: {path}")
    if artifact.get("experiment_role", spec["experiment_role"]) != spec[
        "experiment_role"
    ]:
        raise ValueError(f"artifact实验角色与{variant_id}定义不一致: {path}")
    if artifact.get("supplement_protocol_version") != SUPPLEMENT_PROTOCOL_VERSION:
        raise ValueError(
            f"artifact补充协议版本不匹配: {path}: "
            f"{artifact.get('supplement_protocol_version')!r} != "
            f"{SUPPLEMENT_PROTOCOL_VERSION!r}"
        )
    validate_feature_training_protocol(
        artifact,
        path,
        candidate_loss_weight=(0.0 if variant_id in PROBE_VARIANTS else 0.50),
    )
    for path_key, hash_key in (
        ("model_path", "model_sha256"),
        ("best_weights_path", "best_weights_sha256"),
    ):
        resolved = _resolve_existing_path(artifact.get(path_key))
        stored_hash = artifact.get(hash_key)
        if (
            resolved is None
            or not isinstance(stored_hash, str)
            or _sha256(resolved) != stored_hash
        ):
            raise ValueError(
                f"artifact {path_key}/{hash_key}完整性校验失败: {path}"
            )
    required = (
        "input_cols",
        "target_index",
        "scaler_x",
        "scaler_y",
        "history_len",
        "forecast_len",
        "diagnostic_layers",
        "regime_label_config",
        "regime_feature_config",
    )
    missing = [key for key in required if key not in artifact]
    if missing:
        raise KeyError(f"artifact缺少字段{missing}: {path}")
    if list(artifact["regime_label_config"].get("change_band_edges", ())) != list(
        regime_train.CHANGE_BAND_EDGES
    ):
        raise ValueError(f"artifact工况阈值与当前代码不一致: {path}")
    if variant_id in PROBE_VARIANTS:
        probe_required = (
            "backbone_frozen",
            "candidate_supervision_loss_weight",
            "frozen_weights_sha256_before_training",
            "frozen_weights_sha256_after_training",
            "frozen_weights_exact_match_after_training",
            "candidate_output_sha256_before_training",
            "candidate_output_sha256_after_training",
            "candidate_output_exact_match_after_training",
            "source_model_sha256",
            "source_artifact_sha256",
        )
        probe_missing = [key for key in probe_required if key not in artifact]
        if probe_missing:
            raise KeyError(f"{variant_id} Frozen-Pair artifact缺少{probe_missing}: {path}")
        if not bool(artifact["backbone_frozen"]):
            raise ValueError(f"{variant_id} artifact未标记backbone_frozen: {path}")
        if float(artifact["candidate_supervision_loss_weight"]) != 0.0:
            raise ValueError(f"{variant_id} candidate loss必须为0: {path}")
        if not bool(artifact["frozen_weights_exact_match_after_training"]):
            raise ValueError(f"{variant_id}冻结权重训练前后不一致: {path}")
        if not bool(artifact["candidate_output_exact_match_after_training"]):
            raise ValueError(f"{variant_id}冻结候选训练前后不一致: {path}")
    artifact = dict(artifact)
    artifact["artifact_path"] = os.path.abspath(path)
    return artifact


def load_feature_model(variant_id, farm_id, artifact):
    model_path = _resolve_existing_path(artifact.get("model_path"))
    if model_path:
        model = keras.models.load_model(
            model_path,
            custom_objects=get_feature_screen_custom_objects(),
            compile=False,
        )
        loaded_path = os.path.abspath(model_path)
    else:
        weights_path = _resolve_existing_path(artifact.get("best_weights_path"))
        if not weights_path:
            raise FileNotFoundError(
                f"缺少 {variant_id}/{farm_id} 完整模型和最佳权重"
            )
        model = build_feature_screen_model_from_artifact(artifact)
        model.load_weights(weights_path)
        loaded_path = os.path.abspath(weights_path)
    count = int(model.count_params())
    if int(artifact.get("total_params", count)) != count:
        raise ValueError(
            f"artifact参数量{artifact.get('total_params')}与模型{count}不一致"
        )
    if count != EXPECTED_PARAMETER_COUNTS[variant_id]:
        raise ValueError(
            f"{variant_id}参数量{count:,}与冻结实验协议"
            f"{EXPECTED_PARAMETER_COUNTS[variant_id]:,}不一致"
        )
    trainable_count = int(
        sum(int(np.prod(variable.shape)) for variable in model.trainable_weights)
    )
    expected_trainable = EXPECTED_TRAINABLE_PARAMETER_COUNTS[variant_id]
    if trainable_count != expected_trainable:
        raise ValueError(
            f"{variant_id}可训练参数{trainable_count:,}与冻结协议"
            f"{expected_trainable:,}不一致"
        )
    artifact_trainable = int(artifact.get("trainable_params", trainable_count))
    if artifact_trainable != trainable_count:
        raise ValueError(
            f"{variant_id}/{farm_id} artifact可训练参数"
            f"{artifact_trainable:,}与模型{trainable_count:,}不一致"
        )
    return model, loaded_path


def _array_sha256(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _save_candidate_archive(
    dirs,
    model_name,
    farm_id,
    df,
    history_len,
    y_true,
    outputs,
    persistence,
    corrected,
    fused,
    gate,
    capacity,
):
    """保存补充变体的逐点候选，为Frozen-Pair硬校验提供证据。"""
    n_samples, forecast_len = y_true.shape
    origins = df.index[history_len - 1 : history_len - 1 + n_samples]
    path = os.path.join(
        dirs["candidate_archives"],
        f"{model_name}_candidate_archive_farm_{farm_id}.npz",
    )
    persistence_scaled = np.asarray(outputs["persistence_candidate"])
    corrected_scaled = np.asarray(outputs["corrected_candidate"])
    np.savez_compressed(
        path,
        schema_version=np.asarray("candidate_archive_v1"),
        farm_id=np.asarray(str(farm_id)),
        sample_id=np.arange(n_samples, dtype=np.int64),
        horizon_step=np.arange(1, forecast_len + 1, dtype=np.int16),
        forecast_origin_time=np.asarray(origins.astype(str), dtype=np.str_),
        y_true=np.asarray(y_true, dtype=np.float64),
        persistence_scaled=persistence_scaled,
        corrected_scaled=corrected_scaled,
        persistence=np.asarray(persistence, dtype=np.float64),
        corrected=np.asarray(corrected, dtype=np.float64),
        gate=np.asarray(gate, dtype=np.float64),
        fused=np.asarray(fused, dtype=np.float64),
        capacity=np.asarray(float(capacity), dtype=np.float64),
    )
    return {
        "candidate_archive_path": path,
        "candidate_archive_sha256": _sha256(path),
        "candidate_pair_scaled_sha256": _array_sha256(
            persistence_scaled, corrected_scaled
        ),
        "candidate_pair_physical_sha256": _array_sha256(
            np.asarray(persistence, dtype=np.float64),
            np.asarray(corrected, dtype=np.float64),
        ),
    }


def _relabel_frame(frame, variant_id):
    frame = frame.copy()
    frame["model_family"] = MODEL_FAMILY
    frame["model_variant"] = variant_id
    frame["variant_id"] = variant_id
    frame["model_name"] = variant_model_name(variant_id)
    return frame


def _gate_oracle_valid_mask(gate, y_true, persistence, corrected, fused=None):
    """Return the elementwise mask used by every gate/oracle diagnostic.

    ``valid_future`` is a sample-level flag and is true when *any* future point is
    finite.  It therefore cannot protect horizon-level calibration from isolated
    missing targets.  Keeping this mask in one place prevents NaN comparisons from
    being silently interpreted as ``corrected_better=False``.
    """
    arrays = (gate, y_true, persistence, corrected)
    if fused is not None:
        arrays = (*arrays, fused)
    valid = np.ones(np.asarray(gate).shape, dtype=bool)
    for values in arrays:
        value = np.asarray(values)
        if value.shape != valid.shape:
            raise ValueError(
                f"gate/oracle shape mismatch: {value.shape} != {valid.shape}"
            )
        valid &= np.isfinite(value)
    return valid


def _finite_gate_rows(
    variant_id,
    farm_id,
    gate,
    y_true,
    persistence,
    corrected,
    fused,
    regimes,
):
    """Keep target-free gate distributions legacy-comparable and mask oracle metrics."""
    gate = np.asarray(gate, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    persistence = np.asarray(persistence, dtype=float)
    corrected = np.asarray(corrected, dtype=float)
    fused = np.asarray(fused, dtype=float)
    oracle_valid = _gate_oracle_valid_mask(
        gate,
        y_true,
        persistence,
        corrected,
        fused,
    )
    gate_valid = np.isfinite(gate)
    candidate_valid = (
        gate_valid & np.isfinite(persistence) & np.isfinite(corrected)
    )
    regime_masks, _ = regime_predict._regime_masks(regimes)
    rows = []
    for regime_group in regime_predict.REGIME_GROUP_ORDER:
        regime_mask = np.asarray(regime_masks[regime_group], dtype=bool)
        for horizon in range(gate.shape[1]):
            # 与旧_gate_rows相同，gate分布只按sample-level regime mask取样，
            # 不因某一horizon真值缺失而改变；oracle字段另用逐点有限掩码。
            distribution_mask = regime_mask & gate_valid[:, horizon]
            candidate_mask = regime_mask & candidate_valid[:, horizon]
            oracle_mask = regime_mask & oracle_valid[:, horizon]
            values = gate[distribution_mask, horizon]
            if len(values):
                entropy = -(
                    values * np.log(np.clip(values, 1e-8, 1.0))
                    + (1.0 - values)
                    * np.log(np.clip(1.0 - values, 1e-8, 1.0))
                ) / np.log(2.0)
                fields = {
                    "gate_mean": float(values.mean()),
                    "gate_std": float(values.std()),
                    "gate_p10": float(np.quantile(values, 0.10)),
                    "gate_p50": float(np.quantile(values, 0.50)),
                    "gate_p90": float(np.quantile(values, 0.90)),
                    "gate_low_saturation_rate": float((values < 0.05).mean()),
                    "gate_high_saturation_rate": float((values > 0.95).mean()),
                    "gate_binary_entropy": float(entropy.mean()),
                }
            else:
                fields = {
                    key: np.nan
                    for key in (
                        "gate_mean",
                        "gate_std",
                        "gate_p10",
                        "gate_p50",
                        "gate_p90",
                        "gate_low_saturation_rate",
                        "gate_high_saturation_rate",
                        "gate_binary_entropy",
                    )
                }

            if candidate_mask.any():
                candidate_gap = (
                    corrected[candidate_mask, horizon]
                    - persistence[candidate_mask, horizon]
                )
                contribution = gate[candidate_mask, horizon] * candidate_gap
                fields.update(
                    {
                        "candidate_abs_gap_mean": float(
                            np.abs(candidate_gap).mean()
                        ),
                        "gate_contribution_abs_mean": float(
                            np.abs(contribution).mean()
                        ),
                    }
                )
            else:
                fields.update(
                    {
                        "candidate_abs_gap_mean": np.nan,
                        "gate_contribution_abs_mean": np.nan,
                    }
                )

            if oracle_mask.any():
                oracle_gate = gate[oracle_mask, horizon]
                p_error = np.square(
                    persistence[oracle_mask, horizon]
                    - y_true[oracle_mask, horizon]
                )
                c_error = np.square(
                    corrected[oracle_mask, horizon]
                    - y_true[oracle_mask, horizon]
                )
                f_error = np.square(
                    fused[oracle_mask, horizon] - y_true[oracle_mask, horizon]
                )
                corrected_better = c_error < p_error
                hard_choice = oracle_gate >= 0.5
                persistence_mse = float(p_error.mean())
                oracle_mse = float(np.minimum(p_error, c_error).mean())
                fused_mse = float(f_error.mean())
                possible_gain = persistence_mse - oracle_mse
                captured_gain = (
                    (persistence_mse - fused_mse) / possible_gain
                    if possible_gain > 1e-12
                    else np.nan
                )
                fields.update(
                    {
                        "corrected_better_rate": float(corrected_better.mean()),
                        "gate_hard_choice_accuracy": float(
                            (hard_choice == corrected_better).mean()
                        ),
                        "gate_oracle_brier": float(
                            np.square(
                                oracle_gate
                                - corrected_better.astype(float)
                            ).mean()
                        ),
                        "fused_mse": fused_mse,
                        "oracle_mse": oracle_mse,
                        "oracle_regret": fused_mse - oracle_mse,
                        "captured_oracle_gain": captured_gain,
                    }
                )
            else:
                fields.update(
                    {
                        key: np.nan
                        for key in (
                            "corrected_better_rate",
                            "gate_hard_choice_accuracy",
                            "gate_oracle_brier",
                            "fused_mse",
                            "oracle_mse",
                            "oracle_regret",
                            "captured_oracle_gain",
                        )
                    }
                )
            rows.append(
                {
                    "model_family": MODEL_FAMILY,
                    "model_variant": variant_id,
                    "farm_id": str(farm_id),
                    "regime_group": regime_group,
                    "horizon_step": horizon + 1,
                    "horizon_minutes": (horizon + 1) * 15,
                    "regime_sample_count": int(regime_mask.sum()),
                    "sample_count": int(distribution_mask.sum()),
                    "gate_distribution_sample_count": int(
                        distribution_mask.sum()
                    ),
                    "gate_distribution_excluded_nonfinite_count": int(
                        regime_mask.sum() - distribution_mask.sum()
                    ),
                    "candidate_valid_count": int(candidate_mask.sum()),
                    "oracle_valid_count": int(oracle_mask.sum()),
                    "excluded_nonfinite_count": int(
                        regime_mask.sum() - oracle_mask.sum()
                    ),
                    "gate_distribution_protocol": (
                        "legacy_compatible_sample_regime_mask_target_free"
                    ),
                    "elementwise_finite_masked": True,
                    **fields,
                }
            )
    return rows


def _finite_gate_calibration_rows(
    variant_id,
    farm_id,
    gate,
    y_true,
    persistence,
    corrected,
):
    """Build calibration bins after excluding every non-finite oracle point."""
    gate = np.asarray(gate, dtype=float)
    valid = _gate_oracle_valid_mask(gate, y_true, persistence, corrected)
    probability = gate[valid]
    oracle = (
        np.square(np.asarray(corrected)[valid] - np.asarray(y_true)[valid])
        < np.square(np.asarray(persistence)[valid] - np.asarray(y_true)[valid])
    )
    bin_ids = np.minimum((np.clip(probability, 0.0, 1.0) * 10).astype(int), 9)
    rows = []
    for bin_id in range(10):
        mask = bin_ids == bin_id
        rows.append(
            {
                "model_family": MODEL_FAMILY,
                "model_variant": variant_id,
                "farm_id": str(farm_id),
                "gate_bin": bin_id,
                "gate_bin_left": bin_id / 10.0,
                "gate_bin_right": (bin_id + 1) / 10.0,
                "count": int(mask.sum()),
                "mean_gate": (
                    float(probability[mask].mean()) if mask.any() else np.nan
                ),
                "corrected_better_rate": (
                    float(oracle[mask].mean()) if mask.any() else np.nan
                ),
                "total_valid_count": int(valid.sum()),
                "excluded_nonfinite_count": int(valid.size - valid.sum()),
                "elementwise_finite_masked": True,
            }
        )
    return rows


def predict_one_feature_variant_farm(variant_id, test_file):
    farm_id = common_predict.get_farm_id(test_file)
    model_name = variant_model_name(variant_id)
    dirs = prediction_output_dirs(variant_id)
    print(f"\n===== 预测 {model_name} / 风电场 {farm_id} =====")
    artifact = load_feature_artifact(variant_id, farm_id)
    model, loaded_model_path = load_feature_model(variant_id, farm_id, artifact)
    df, features, actual_power, capacity = common_predict.prepare_prediction_arrays(
        test_file,
        artifact,
    )
    history_len = int(artifact["history_len"])
    forecast_len = int(artifact["forecast_len"])
    n_samples = regime_predict._prediction_sample_count(
        features, history_len, forecast_len
    )
    pred_ds, dataset_samples = common_predict.make_prediction_dataset(
        features,
        history_len,
        forecast_len,
    )
    if dataset_samples != n_samples:
        raise ValueError("预测dataset样本数不一致")
    outputs = regime_predict._validate_diagnostics(
        regime_predict._diagnostic_forward(model, pred_ds, artifact),
        n_samples,
        forecast_len,
    )

    # 模型前向结束后才读取未来真实功率，用于测试指标和realized regime。
    y_true = common_predict.build_truth_windows(
        actual_power,
        n_samples,
        history_len,
        forecast_len,
    )
    # Use one representation for metrics, CSV and NPZ.  TensorFlow inference is
    # float32; promoting once here preserves those values exactly while avoiding
    # a shorter float32 CSV representation than the archive's float64 values.
    fused = np.asarray(
        regime_predict._inverse_candidate(
            artifact, outputs["forecast"], capacity
        ),
        dtype=np.float64,
    )
    persistence = np.asarray(
        regime_predict._inverse_candidate(
            artifact, outputs["persistence_candidate"], capacity
        ),
        dtype=np.float64,
    )
    corrected = np.asarray(
        regime_predict._inverse_candidate(
            artifact, outputs["corrected_candidate"], capacity
        ),
        dtype=np.float64,
    )
    gate = outputs["gate"]
    last_power = persistence[:, 0]
    regimes = regime_train.build_regime_targets_numpy(y_true, last_power, capacity)
    candidate_predictions = {
        "fused": fused,
        "persistence": persistence,
        "corrected": corrected,
    }
    archive_fields = _save_candidate_archive(
        dirs,
        model_name,
        farm_id,
        df,
        history_len,
        y_true,
        outputs,
        persistence,
        corrected,
        fused,
        gate,
        capacity,
    )

    pred_df = common_predict.build_prediction_frame(
        model_name,
        df,
        farm_id,
        fused,
        y_true,
        history_len,
        forecast_len,
    )
    prediction_path = os.path.join(
        dirs["predictions"], f"{model_name}_predictions_farm_{farm_id}.csv"
    )
    pred_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    metric_df = common_predict.metrics_by_horizon(
        model_name,
        farm_id,
        y_true,
        fused,
        capacity,
        forecast_len,
    )
    parameter_count = int(model.count_params())
    trainable_parameter_count = int(
        sum(int(np.prod(variable.shape)) for variable in model.trainable_weights)
    )
    metric_df["model_family"] = MODEL_FAMILY
    metric_df["model_variant"] = variant_id
    metric_df["variant_id"] = variant_id
    metric_df["feature_count"] = len(selected_feature_names(variant_id))
    metric_df["parameter_count"] = parameter_count
    horizon_metric_path = os.path.join(
        dirs["root"], f"{model_name}_metrics_by_horizon_farm_{farm_id}.csv"
    )
    metric_df.to_csv(horizon_metric_path, index=False, encoding="utf-8-sig")

    candidate_frame = _relabel_frame(
        regime_predict._candidate_metric_rows(
            variant_id,
            model_name,
            farm_id,
            y_true,
            candidate_predictions,
            capacity,
        ),
        variant_id,
    )
    candidate_path = os.path.join(
        dirs["candidate_metrics"],
        f"{model_name}_candidate_metrics_farm_{farm_id}.csv",
    )
    candidate_frame.to_csv(candidate_path, index=False, encoding="utf-8-sig")

    assignment_frame = regime_predict._assignment_frame(
        df,
        farm_id,
        regimes,
        last_power,
        n_samples,
        history_len,
    )
    assignment_frame = _relabel_frame(assignment_frame, variant_id)
    assignment_path = os.path.join(
        dirs["regime_assignments"],
        f"{model_name}_regime_assignments_farm_{farm_id}.csv",
    )
    assignment_frame.to_csv(assignment_path, index=False, encoding="utf-8-sig")

    regime_frame = _relabel_frame(
        pd.DataFrame(
            regime_predict.build_regime_metric_rows(
                variant_id,
                farm_id,
                y_true,
                candidate_predictions,
                regimes,
                capacity,
            )
        ),
        variant_id,
    )
    regime_path = os.path.join(
        dirs["regime_metrics"],
        f"{model_name}_regime_metrics_farm_{farm_id}.csv",
    )
    regime_frame.to_csv(regime_path, index=False, encoding="utf-8-sig")

    gate_frame = _relabel_frame(
        pd.DataFrame(
            _finite_gate_rows(
                variant_id,
                farm_id,
                gate,
                y_true,
                persistence,
                corrected,
                fused,
                regimes,
            )
        ),
        variant_id,
    )
    gate_path = os.path.join(
        dirs["gate_diagnostics"],
        f"{model_name}_gate_by_regime_horizon_farm_{farm_id}.csv",
    )
    gate_frame.to_csv(gate_path, index=False, encoding="utf-8-sig")
    oracle_valid = _gate_oracle_valid_mask(
        gate,
        y_true,
        persistence,
        corrected,
        fused,
    )
    corrected_better = np.zeros(gate.shape, dtype=bool)
    corrected_better[oracle_valid] = (
        np.square(corrected[oracle_valid] - y_true[oracle_valid])
        < np.square(persistence[oracle_valid] - y_true[oracle_valid])
    )
    calibration_frame = _relabel_frame(
        pd.DataFrame(
            _finite_gate_calibration_rows(
                variant_id,
                farm_id,
                gate,
                y_true,
                persistence,
                corrected,
            )
        ),
        variant_id,
    )
    calibration_path = os.path.join(
        dirs["gate_diagnostics"],
        f"{model_name}_gate_calibration_farm_{farm_id}.csv",
    )
    calibration_frame.to_csv(
        calibration_path, index=False, encoding="utf-8-sig"
    )
    heatmap_path, calibration_figure_path = regime_predict._save_gate_figures(
        gate_frame,
        calibration_frame,
        model_name,
        farm_id,
        dirs,
    )

    router_weights = np.stack([1.0 - gate, gate], axis=-1)
    _, router_fields = common_predict.save_router_diagnostics(
        router_weights,
        ["persistence", "corrected"],
        model_name,
        farm_id,
        dirs,
    )
    single_window_path, single_window_figure_path = (
        common_predict.save_single_window_plot(
            pred_df, model_name, farm_id, dirs, forecast_len
        )
    )
    (
        weighted_curve_path,
        weighted_curve_figure_path,
        weighted_metrics,
    ) = common_predict.save_weighted_full_test_plot(
        pred_df,
        model_name,
        farm_id,
        dirs,
        capacity,
    )

    all_metrics = metric_df[metric_df["horizon_step"] == "all"].iloc[0].to_dict()
    binary_entropy = -(
        gate * np.log(np.clip(gate, 1e-8, 1.0))
        + (1.0 - gate) * np.log(np.clip(1.0 - gate, 1e-8, 1.0))
    ) / np.log(2.0)
    spec = VARIANT_SPECS[variant_id]
    experiment_role = spec["experiment_role"]
    selection_eligible = bool(spec["selection_eligible"])
    all_metrics.update(
        {
            "model_family": MODEL_FAMILY,
            "model_variant": variant_id,
            "variant_id": variant_id,
            "variant_label": spec["label"],
            "feature_groups": "+".join(spec["groups"]),
            "feature_count": len(selected_feature_names(variant_id)),
            "feature_names": json.dumps(
                selected_feature_names(variant_id), ensure_ascii=False
            ),
            "architecture_version": ARCHITECTURE_VERSION,
            "random_seed": RANDOM_SEED,
            "result_source": "stage2_feature_screen_supplement_inference",
            "reference_only": False,
            "source_variant": "b2_persistence_residual",
            "experiment_role": experiment_role,
            "freeze_candidates": bool(spec["freeze_candidates"]),
            "selection_eligible": selection_eligible,
            "encoder_type": artifact.get("encoder_type"),
            "gate_type": artifact.get("gate_type"),
            "auxiliary_tasks": False,
            "parameter_count": parameter_count,
            "expected_parameter_count": EXPECTED_PARAMETER_COUNTS[variant_id],
            "trainable_parameter_count": trainable_parameter_count,
            "expected_trainable_parameter_count": (
                EXPECTED_TRAINABLE_PARAMETER_COUNTS[variant_id]
            ),
            "frozen_parameter_count": parameter_count - trainable_parameter_count,
            "loaded_model_path": loaded_model_path,
            "loaded_model_sha256": _sha256(loaded_model_path),
            "artifact_path": artifact["artifact_path"],
            "artifact_sha256": _sha256(artifact["artifact_path"]),
            "test_file_path": os.path.abspath(test_file),
            "test_file_sha256": _sha256(test_file),
            "prediction_path": prediction_path,
            "horizon_metric_path": horizon_metric_path,
            "candidate_metric_path": candidate_path,
            "regime_assignment_path": assignment_path,
            "regime_metric_path": regime_path,
            "gate_diagnostics_path": gate_path,
            "gate_calibration_path": calibration_path,
            "gate_heatmap_path": heatmap_path,
            "gate_calibration_figure_path": calibration_figure_path,
            "single_window_path": single_window_path,
            "single_window_figure_path": single_window_figure_path,
            "weighted_curve_path": weighted_curve_path,
            "weighted_curve_figure_path": weighted_curve_figure_path,
            "gate_mean": float(gate.mean()),
            "gate_std": float(gate.std()),
            "gate_sample_variation": float(np.std(gate, axis=0).mean()),
            "gate_binary_entropy": float(binary_entropy.mean()),
            "gate_saturation_low_rate": float((gate < 0.05).mean()),
            "gate_saturation_high_rate": float((gate > 0.95).mean()),
            "gate_oracle_choice_accuracy": float(
                ((gate[oracle_valid] >= 0.5) == corrected_better[oracle_valid]).mean()
                if oracle_valid.any()
                else np.nan
            ),
            "gate_oracle_brier": float(
                np.mean(
                    np.square(
                        gate[oracle_valid]
                        - corrected_better[oracle_valid].astype(float)
                    )
                )
                if oracle_valid.any()
                else np.nan
            ),
            "gate_oracle_valid_count": int(oracle_valid.sum()),
            "gate_oracle_excluded_nonfinite_count": int(
                oracle_valid.size - oracle_valid.sum()
            ),
            "gate_oracle_elementwise_finite_masked": True,
            "gate_change_magnitude_spearman": regime_predict._fixed_rank_correlation(
                gate.mean(axis=1), regimes["change_magnitude"]
            ),
            "fusion_reconstruction_max_abs_error": outputs[
                "fusion_reconstruction_max_abs_error"
            ],
            "evaluation_pipeline_version": regime_train.EVALUATION_PIPELINE_VERSION,
            "legacy_bidirectional_weather_imputation": True,
            "test_reuse_status": TEST_REUSE_STATUS,
            "source_test_reuse_status": "legacy_seen",
            "test_used_for_feature_selection": selection_eligible,
            "feature_screening_test_selection_eligible": selection_eligible,
            "test_selection_prohibited": not selection_eligible,
            "test_is_final_blind_evaluation": False,
            "selection_split": "test" if selection_eligible else "not_applicable",
            "selection_metric": (
                SELECTION_MACRO_METRIC if selection_eligible else "not_applicable"
            ),
            "training_code_path": os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "wind_RegimeEncoder_PatchTST_feature_screen_train.py",
                )
            ),
            "prediction_code_path": os.path.abspath(__file__),
            "prediction_code_sha256": _sha256(os.path.abspath(__file__)),
            **archive_fields,
            **router_fields,
            **{
                f"weighted_curve_{key}": value
                for key, value in weighted_metrics.items()
            },
        }
    )
    for metadata_key in (
        "backbone_frozen",
        "candidate_supervision_loss_weight",
        "source_model_sha256",
        "source_artifact_sha256",
        "frozen_weights_sha256_before_training",
        "frozen_weights_sha256_after_training",
        "frozen_weights_exact_match_after_training",
        "candidate_output_sha256_before_training",
        "candidate_output_sha256_after_training",
        "candidate_output_exact_match_after_training",
    ):
        if metadata_key in artifact:
            all_metrics[metadata_key] = artifact[metadata_key]
    print(
        f"{model_name} / {farm_id}: NRMSE="
        f"{all_metrics[SELECTION_METRIC]:.6f}, features="
        f"{len(selected_feature_names(variant_id))}, params={parameter_count:,}"
    )
    del model
    keras.backend.clear_session()
    return {
        "summary": pd.DataFrame([all_metrics]),
        "horizon": metric_df,
        "candidate": candidate_frame,
        "regime": regime_frame,
        "gate": gate_frame,
        "calibration": calibration_frame,
        "assignments": assignment_frame,
    }


def predict_feature_variant(variant_id, test_files):
    outputs = {
        "summary": [],
        "horizon": [],
        "candidate": [],
        "regime": [],
        "gate": [],
        "calibration": [],
        "assignments": [],
    }
    for test_file in test_files:
        result = predict_one_feature_variant_farm(variant_id, test_file)
        for key in outputs:
            outputs[key].append(result[key])
    combined = {
        key: pd.concat(frames, ignore_index=True, sort=False)
        for key, frames in outputs.items()
    }
    dirs = prediction_output_dirs(variant_id)
    model_name = variant_model_name(variant_id)
    requested_farm_ids = {
        str(common_predict.get_farm_id(path)) for path in test_files
    }
    full_farms = bool(
        not os.getenv("WIND_FEATURE_SCREEN_FARMS")
        and requested_farm_ids == set(expected_test_farm_ids())
    )
    suffix = _partial_suffix(combined["summary"], [variant_id], full_farms)
    _atomic_to_csv(
        combined["summary"],
        os.path.join(dirs["root"], f"{model_name}_test_metrics_summary{suffix}.csv"),
    )
    _atomic_to_csv(
        combined["horizon"],
        os.path.join(
            dirs["root"], f"{model_name}_test_metrics_by_horizon_all{suffix}.csv"
        ),
    )
    return combined


def _r4_output_root():
    return os.path.join(
        regime_train.variant_dirs(R4_SOURCE_VARIANT, create=False)["root"],
        "testdata_predict_output",
    )


def _r4_test_paths():
    root = _r4_output_root()
    source_name = regime_train.variant_model_name(R4_SOURCE_VARIANT)
    return {
        "root": root,
        "summary": os.path.join(root, f"{source_name}_test_metrics_summary.csv"),
        "horizon": os.path.join(
            root, f"{source_name}_test_metrics_by_horizon_all.csv"
        ),
        "predictions": os.path.join(root, "predictions"),
        "candidate": os.path.join(root, "candidate_metrics"),
        "regime": os.path.join(root, "regime_metrics"),
        "gate": os.path.join(root, "gate_diagnostics"),
        "assignments": os.path.join(root, "regime_assignments"),
    }


def _load_one_per_farm(directory, suffix_pattern, farm_ids):
    frames = []
    paths = []
    for farm_id in farm_ids:
        matches = sorted(glob.glob(os.path.join(directory, f"*{suffix_pattern}{farm_id}.csv")))
        if len(matches) != 1:
            raise ValueError(
                f"R4引用文件应唯一: dir={directory}, farm={farm_id}, matches={matches}"
            )
        frame = pd.read_csv(matches[0])
        frame["farm_id"] = frame["farm_id"].astype(str)
        frames.append(frame)
        paths.append(os.path.abspath(matches[0]))
    return pd.concat(frames, ignore_index=True, sort=False), paths


def load_f4_reference(test_files):
    """直接读取既有R4预测、指标和图路径，不重新推理或复制文件。"""
    paths = _r4_test_paths()
    for key in ("summary", "horizon"):
        if not os.path.exists(paths[key]):
            raise FileNotFoundError(f"F4引用所需R4 {key}不存在: {paths[key]}")
    farm_ids = [common_predict.get_farm_id(path) for path in test_files]
    farm_set = set(farm_ids)
    summary = pd.read_csv(paths["summary"])
    horizon = pd.read_csv(paths["horizon"])
    summary["farm_id"] = summary["farm_id"].astype(str)
    horizon["farm_id"] = horizon["farm_id"].astype(str)
    summary = summary[summary["farm_id"].isin(farm_set)].copy()
    horizon = horizon[horizon["farm_id"].isin(farm_set)].copy()
    if summary["farm_id"].nunique() != len(farm_set) or len(summary) != len(farm_set):
        raise ValueError("R4测试summary未按每场站一行覆盖全部请求场站")
    counts = horizon.groupby("farm_id").size()
    if set(counts.index) != farm_set or not (
        counts == regime_train.FORECAST_LEN + 1
    ).all():
        raise ValueError("R4测试horizon必须对每场站包含16步和all共17行")
    if not (
        pd.to_numeric(summary["parameter_count"], errors="coerce") == 21151
    ).all():
        raise ValueError("F4引用的R4参数量不是21,151")

    # 读取源artifact只做身份/43维定义核验；不重建模型、不执行推理。
    for _, row in summary.iterrows():
        artifact_path = _resolve_existing_path(row.get("artifact_path"))
        model_path = _resolve_existing_path(row.get("loaded_model_path"))
        if artifact_path is None or model_path is None:
            raise FileNotFoundError(
                f"F4/R4源artifact或模型不存在: farm={row['farm_id']}"
            )
        artifact = joblib.load(artifact_path)
        validate_r4_reference_artifact(artifact, artifact_path)

    source_model_names = summary["model_name"].copy()
    source_architecture = summary.get("architecture_version", pd.Series(index=summary.index))
    source_test_status = summary.get("test_reuse_status", pd.Series(index=summary.index))
    source_selection_flag = summary.get(
        "test_selection_prohibited", pd.Series(index=summary.index)
    )
    source_prediction_paths = summary["prediction_path"].copy()
    summary["source_model_name"] = source_model_names
    summary["source_model_family"] = regime_train.MODEL_FAMILY
    summary["source_model_variant"] = R4_SOURCE_VARIANT
    summary["source_architecture_version"] = source_architecture
    summary["source_test_reuse_status"] = source_test_status
    summary["source_test_selection_prohibited"] = source_selection_flag
    summary["source_summary_path"] = os.path.abspath(paths["summary"])
    summary["source_summary_sha256"] = _sha256(paths["summary"])
    summary["source_horizon_path"] = os.path.abspath(paths["horizon"])
    summary["source_horizon_sha256"] = _sha256(paths["horizon"])
    summary["source_prediction_path"] = source_prediction_paths
    summary["model_family"] = MODEL_FAMILY
    summary["model_name"] = variant_model_name("f4")
    summary["model_variant"] = "f4"
    summary["variant_id"] = "f4"
    summary["variant_label"] = VARIANT_SPECS["f4"]["label"]
    summary["feature_groups"] = "+".join(VARIANT_SPECS["f4"]["groups"])
    summary["feature_count"] = len(FULL_FEATURE_NAMES)
    summary["feature_names"] = json.dumps(FULL_FEATURE_NAMES, ensure_ascii=False)
    summary["expected_parameter_count"] = EXPECTED_PARAMETER_COUNTS["f4"]
    summary["result_source"] = "direct_reference_existing_r4_test_outputs"
    summary["reference_only"] = True
    summary["source_variant"] = R4_SOURCE_VARIANT
    summary["test_reuse_status"] = TEST_REUSE_STATUS
    summary["test_used_for_feature_selection"] = True
    summary["feature_screening_test_selection_eligible"] = True
    summary["test_selection_prohibited"] = False
    summary["test_is_final_blind_evaluation"] = False
    summary["selection_split"] = "test"
    summary["selection_metric"] = SELECTION_MACRO_METRIC
    summary["training_code_path"] = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "wind_RegimeEncoder_PatchTST_feature_screen_train.py",
        )
    )
    summary["prediction_code_path"] = os.path.abspath(__file__)
    summary["prediction_code_sha256"] = _sha256(os.path.abspath(__file__))

    horizon["source_model_name"] = horizon.get("model_name")
    horizon["source_model_variant"] = R4_SOURCE_VARIANT
    horizon["model_name"] = variant_model_name("f4")
    horizon["model_family"] = MODEL_FAMILY
    horizon["model_variant"] = "f4"
    horizon["variant_id"] = "f4"
    horizon["feature_count"] = len(FULL_FEATURE_NAMES)
    horizon["result_source"] = "direct_reference_existing_r4_test_outputs"

    candidate, candidate_paths = _load_one_per_farm(
        paths["candidate"], "candidate_metrics_farm_", farm_ids
    )
    regime, regime_paths = _load_one_per_farm(
        paths["regime"], "regime_metrics_farm_", farm_ids
    )
    gate, gate_paths = _load_one_per_farm(
        paths["gate"], "gate_by_regime_horizon_farm_", farm_ids
    )
    calibration, calibration_paths = _load_one_per_farm(
        paths["gate"], "gate_calibration_farm_", farm_ids
    )
    assignments, assignment_paths = _load_one_per_farm(
        paths["assignments"], "regime_assignments_farm_", farm_ids
    )
    optional = {}
    for key, frame, source_paths in (
        ("candidate", candidate, candidate_paths),
        ("regime", regime, regime_paths),
        ("gate", gate, gate_paths),
        ("calibration", calibration, calibration_paths),
        ("assignments", assignments, assignment_paths),
    ):
        frame = _relabel_frame(frame, "f4")
        frame["result_source"] = "direct_reference_existing_r4_test_outputs"
        frame["source_model_variant"] = R4_SOURCE_VARIANT
        frame["source_file_path"] = frame["farm_id"].map(
            dict(zip(farm_ids, source_paths))
        )
        optional[key] = frame
    return {
        "summary": summary,
        "horizon": horizon,
        **optional,
    }


def _read_csv_with_farm_id(path):
    frame = pd.read_csv(path, low_memory=False)
    if "farm_id" in frame:
        frame["farm_id"] = frame["farm_id"].astype(str)
    return frame


def load_legacy_f0_f7_aggregates(test_files):
    """只读复用已完成F0--F7聚合文件，不回退到模型推理。"""
    root = legacy_output_dir()
    farm_ids = [str(common_predict.get_farm_id(path)) for path in test_files]
    outputs = {}
    source_rows = []
    for key, filename in LEGACY_AGGREGATE_FILES.items():
        path = os.path.join(root, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"缺少只读F0--F7聚合文件{key}: {path}；"
                "协议禁止通过重新推理补齐"
            )
        frame = _read_csv_with_farm_id(path)
        if "model_variant" not in frame or "farm_id" not in frame:
            raise KeyError(f"旧聚合文件缺少model_variant/farm_id: {path}")
        frame = frame[
            frame["model_variant"].astype(str).isin(LEGACY_SELECTION_VARIANTS)
            & frame["farm_id"].isin(farm_ids)
        ].copy()
        frame["source_result_source"] = frame.get("result_source", np.nan)
        frame["result_source"] = "direct_reference_existing_f0_f7_aggregate"
        frame["source_aggregate_path"] = os.path.abspath(path)
        frame["source_aggregate_sha256"] = _sha256(path)
        outputs[key] = frame
        source_rows.append(
            {
                "aggregate_key": key,
                "source_path": os.path.abspath(path),
                "source_sha256": _sha256(path),
                "row_count_loaded": len(frame),
                "read_only_reuse": True,
            }
        )

    expected_summary = len(LEGACY_SELECTION_VARIANTS) * len(farm_ids)
    if len(outputs["summary"]) != expected_summary:
        raise ValueError(
            f"旧F0--F7 summary行数{len(outputs['summary'])} != {expected_summary}"
        )
    if set(outputs["summary"]["model_variant"].astype(str)) != set(
        LEGACY_SELECTION_VARIANTS
    ):
        raise ValueError("旧summary未完整覆盖F0--F7")
    if outputs["summary"].duplicated(["model_variant", "farm_id"]).any():
        raise ValueError("旧summary存在重复variant/farm键")
    expected_fixed_rows = {
        "horizon": expected_summary * (regime_train.FORECAST_LEN + 1),
        "candidate": expected_summary * 3 * (regime_train.FORECAST_LEN + 1),
        "gate": expected_summary * len(regime_predict.REGIME_GROUP_ORDER) * regime_train.FORECAST_LEN,
        "calibration": expected_summary * 10,
        "regime": expected_summary
        * len(regime_predict.REGIME_GROUP_ORDER)
        * 3
        * (regime_train.FORECAST_LEN + 1),
    }
    for key, expected in expected_fixed_rows.items():
        if len(outputs[key]) != expected:
            raise ValueError(f"旧{key}行数{len(outputs[key])} != {expected}")
    assignment_counts = outputs["assignments"].groupby("model_variant").size()
    if len(assignment_counts) != len(LEGACY_SELECTION_VARIANTS) or (
        assignment_counts != assignment_counts.iloc[0]
    ).any():
        raise ValueError("旧F0--F7 regime assignments样本数不一致")
    nrmse = pd.to_numeric(outputs["summary"][SELECTION_METRIC], errors="coerce")
    if not np.isfinite(nrmse).all():
        raise ValueError("旧F0--F7 summary包含非有限NRMSE")
    outputs["source_manifest"] = pd.DataFrame(source_rows)
    return outputs


def _load_prediction_truth(path):
    resolved = _resolve_existing_path(path)
    if resolved is None:
        raise FileNotFoundError(f"用于真值对齐的预测文件不存在: {path}")
    frame = pd.read_csv(resolved)
    required = {"sample_id", "horizon_step", "actual_power"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"预测文件缺少真值对齐列{sorted(missing)}: {resolved}")
    order = ["sample_id", "horizon_step"]
    return frame.sort_values(order).reset_index(drop=True)


def _csv_float_matches_archive(csv_values, archive_values, atol=1e-7):
    """Compare a numeric CSV column with its source archive safely.

    Older outputs wrote the float32 prediction directly to CSV, while the NPZ
    archive promoted the same values to float64.  Pandas writes a short decimal
    that round-trips losslessly to float32, but a comparison after ``read_csv``
    happens in float64 and can therefore exceed a small absolute tolerance.

    The compatibility path is deliberately strict: it is used only when the
    archive consists of exact float32 values promoted to float64, and the CSV
    round-trips to the identical float32 bits.  A genuine prediction mismatch is
    therefore not hidden behind a wider physical-unit tolerance.
    """
    csv_array = pd.to_numeric(csv_values, errors="coerce").to_numpy(
        dtype=np.float64
    )
    archive_array = np.asarray(archive_values, dtype=np.float64).reshape(-1)
    if csv_array.shape != archive_array.shape:
        return False
    if np.allclose(
        csv_array,
        archive_array,
        rtol=0.0,
        atol=atol,
        equal_nan=True,
    ):
        return True

    with np.errstate(over="ignore", invalid="ignore"):
        archive_float32 = archive_array.astype(np.float32)
        archive_roundtrip = archive_float32.astype(np.float64)
        csv_float32 = csv_array.astype(np.float32)
    archive_is_promoted_float32 = np.array_equal(
        archive_array,
        archive_roundtrip,
        equal_nan=True,
    )
    return archive_is_promoted_float32 and np.array_equal(
        csv_float32,
        archive_float32,
        equal_nan=True,
    )


def validate_truth_alignment(summary, expected_variants=SELECTION_VARIANTS):
    """保证所有F模型使用与F4完全相同的测试样本和真实值。"""
    expected_variants = tuple(expected_variants)
    for farm_id, farm_frame in summary.groupby("farm_id"):
        farm_frame = farm_frame[
            farm_frame["model_variant"].astype(str).isin(expected_variants)
        ]
        if set(farm_frame["model_variant"].astype(str)) != set(expected_variants):
            raise ValueError(
                f"场站{farm_id}没有覆盖{expected_variants}，无法校验真值"
            )
        reference_variant = "f4" if "f4" in expected_variants else expected_variants[0]
        f4_row = farm_frame[
            farm_frame["model_variant"].astype(str) == reference_variant
        ].iloc[0]
        reference_path = _first_present(
            f4_row.get("source_prediction_path"),
            f4_row.get("prediction_path"),
        )
        reference = _load_prediction_truth(reference_path)
        reference_keys = reference[["sample_id", "horizon_step"]].to_numpy()
        reference_truth = pd.to_numeric(
            reference["actual_power"], errors="coerce"
        ).to_numpy()
        reference_time = (
            reference["forecast_origin_time"].astype(str).to_numpy()
            if "forecast_origin_time" in reference
            else None
        )
        for _, row in farm_frame.iterrows():
            path = row.get("prediction_path")
            candidate = _load_prediction_truth(path)
            if len(candidate) != len(reference) or not np.array_equal(
                candidate[["sample_id", "horizon_step"]].to_numpy(), reference_keys
            ):
                raise ValueError(
                    f"{row['model_variant']}/{farm_id}与F4测试窗口键不一致"
                )
            candidate_truth = pd.to_numeric(
                candidate["actual_power"], errors="coerce"
            ).to_numpy()
            if not np.allclose(
                candidate_truth,
                reference_truth,
                rtol=0.0,
                atol=1e-7,
                equal_nan=True,
            ):
                raise ValueError(
                    f"{row['model_variant']}/{farm_id}与F4测试真实功率不一致"
                )
            if reference_time is not None:
                if "forecast_origin_time" not in candidate or not np.array_equal(
                    candidate["forecast_origin_time"].astype(str).to_numpy(),
                    reference_time,
                ):
                    raise ValueError(
                        f"{row['model_variant']}/{farm_id}与F4预测起报时刻不一致"
                    )


def build_test_comparison(summary):
    metric_names = (
        "mae",
        "mse",
        "rmse",
        "mape",
        "smape",
        "r2",
        "capacity_normalized_mae",
        "capacity_normalized_rmse",
        "weighted_curve_mae",
        "weighted_curve_rmse",
        "weighted_curve_capacity_normalized_mae",
        "weighted_curve_capacity_normalized_rmse",
        "gate_mean",
        "gate_oracle_brier",
    )
    f4 = summary[summary["model_variant"] == "f4"][[
        "farm_id",
        SELECTION_METRIC,
    ]].rename(columns={SELECTION_METRIC: "f4_nrmse"})
    rows = []
    # 兼容入口也必须严格限定为F0--F8选型集合；FP0/FP4只是
    # Frozen-Pair control，任何情况下都不能混入最终模型排名。
    for order, variant_id in enumerate(SELECTION_VARIANTS):
        frame = summary[summary["model_variant"] == variant_id].copy()
        parameters = pd.to_numeric(frame["parameter_count"], errors="coerce")
        paired = frame[["farm_id", SELECTION_METRIC]].merge(f4, on="farm_id")
        paired[SELECTION_METRIC] = pd.to_numeric(
            paired[SELECTION_METRIC], errors="coerce"
        )
        paired["f4_nrmse"] = pd.to_numeric(paired["f4_nrmse"], errors="coerce")
        row = {
            "variant_order": order,
            "model_variant": variant_id,
            "model_name": variant_model_name(variant_id),
            "feature_groups": "+".join(VARIANT_SPECS[variant_id]["groups"]),
            "feature_count": len(selected_feature_names(variant_id)),
            "farm_count": int(frame["farm_id"].astype(str).nunique()),
            "parameter_count_min": (
                int(parameters.min()) if parameters.notna().any() else np.nan
            ),
            "parameter_count_max": (
                int(parameters.max()) if parameters.notna().any() else np.nan
            ),
            "requires_training": VARIANT_SPECS[variant_id]["requires_training"],
            "result_source": (
                "stage2_feature_screen_model_inference"
                if VARIANT_SPECS[variant_id]["requires_training"]
                else "direct_reference_existing_r4_test_outputs"
            ),
            "selection_split": "test",
            "selection_metric": SELECTION_MACRO_METRIC,
            "test_reuse_status": TEST_REUSE_STATUS,
            "test_used_for_feature_selection": True,
            "test_is_final_blind_evaluation": False,
            "farms_better_than_f4": int(
                (paired[SELECTION_METRIC] < paired["f4_nrmse"]).sum()
            ),
            "farms_equal_to_f4": int(
                np.isclose(
                    paired[SELECTION_METRIC], paired["f4_nrmse"], atol=1e-12
                ).sum()
            ),
            "macro_nrmse_delta_vs_f4": float(
                (paired[SELECTION_METRIC] - paired["f4_nrmse"]).mean()
            ),
        }
        for metric in metric_names:
            values = (
                pd.to_numeric(frame[metric], errors="coerce")
                if metric in frame
                else pd.Series(dtype=float)
            )
            row[f"macro_mean_{metric}"] = float(values.mean())
            row[f"macro_std_{metric}"] = float(values.std(ddof=0))
        rows.append(row)
    comparison = pd.DataFrame(rows)
    if not np.isfinite(comparison[SELECTION_MACRO_METRIC]).all():
        raise ValueError("至少一个F变体的宏平均测试NRMSE不是有限值")
    order = comparison.sort_values(
        [
            SELECTION_MACRO_METRIC,
            "macro_std_capacity_normalized_rmse",
            "feature_count",
            "parameter_count_max",
            "variant_order",
        ],
        kind="mergesort",
    ).index
    ranks = pd.Series(np.arange(1, len(order) + 1), index=order)
    comparison["selection_rank"] = ranks
    comparison["selected_final_variant"] = comparison["selection_rank"] == 1
    comparison["selection_tie_break_order"] = (
        "macro_test_nrmse -> macro_test_nrmse_std -> feature_count -> "
        "parameter_count -> variant_order"
    )
    return comparison.sort_values("selection_rank").reset_index(drop=True)


def build_feature_contribution(summary):
    comparisons = (
        ("f0", "f1", "add_H", "在P上加入轮毂高度风速H"),
        ("f1", "f2", "add_M", "在P+H上加入多高度风速M"),
        ("f2", "f3", "add_D", "在P+H+M上加入风向D"),
        ("f3", "f4", "add_C", "在P+H+M+D上加入一致性C"),
        ("f5", "f3", "add_P", "在H+M+D上加入功率状态P"),
        ("f6", "f3", "add_H_reverse", "在P+M+D上加入轮毂风速H"),
        ("f7", "f3", "add_M_reverse", "在P+H+D上加入多高度风速M"),
        ("f7", "f8", "add_C_without_M", "在P+H+D上加入一致性C"),
    )
    rows = []
    values = summary[["model_variant", "farm_id", SELECTION_METRIC]].copy()
    values[SELECTION_METRIC] = pd.to_numeric(values[SELECTION_METRIC], errors="coerce")
    for source_id, target_id, change, description in comparisons:
        source = values[values["model_variant"] == source_id][
            ["farm_id", SELECTION_METRIC]
        ].rename(columns={SELECTION_METRIC: "source_nrmse"})
        target = values[values["model_variant"] == target_id][
            ["farm_id", SELECTION_METRIC]
        ].rename(columns={SELECTION_METRIC: "target_nrmse"})
        paired = source.merge(target, on="farm_id")
        delta = paired["target_nrmse"] - paired["source_nrmse"]
        relative = delta / paired["source_nrmse"] * 100.0
        rows.append(
            {
                "comparison": change,
                "description": description,
                "source_variant": source_id,
                "target_variant": target_id,
                "source_groups": "+".join(VARIANT_SPECS[source_id]["groups"]),
                "target_groups": "+".join(VARIANT_SPECS[target_id]["groups"]),
                "paired_farm_count": len(paired),
                "source_macro_test_nrmse": float(paired["source_nrmse"].mean()),
                "target_macro_test_nrmse": float(paired["target_nrmse"].mean()),
                "target_minus_source_nrmse": float(delta.mean()),
                "relative_change_pct": float(relative.mean()),
                "improves_macro_test_nrmse": bool(delta.mean() < 0),
                "farms_improved": int((delta < 0).sum()),
                "farms_degraded": int((delta > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_horizon_comparison(horizon):
    frame = horizon[horizon["horizon_step"].astype(str) != "all"].copy()
    frame["horizon_step"] = pd.to_numeric(frame["horizon_step"], errors="raise")
    frame[SELECTION_METRIC] = pd.to_numeric(
        frame[SELECTION_METRIC], errors="coerce"
    )
    grouped = (
        frame.groupby(["model_variant", "horizon_step"], as_index=False)
        .agg(
            macro_mean_capacity_normalized_rmse=(SELECTION_METRIC, "mean"),
            macro_std_capacity_normalized_rmse=(SELECTION_METRIC, "std"),
            farm_count=("farm_id", "nunique"),
        )
    )
    grouped["horizon_minutes"] = grouped["horizon_step"] * 15
    f4 = grouped[grouped["model_variant"] == "f4"][[
        "horizon_step",
        "macro_mean_capacity_normalized_rmse",
    ]].rename(
        columns={"macro_mean_capacity_normalized_rmse": "f4_macro_nrmse"}
    )
    grouped = grouped.merge(f4, on="horizon_step", how="left")
    grouped["delta_vs_f4"] = (
        grouped["macro_mean_capacity_normalized_rmse"] - grouped["f4_macro_nrmse"]
    )
    return grouped.sort_values(["model_variant", "horizon_step"])


def _save_selection_figures(comparison, summary, horizon_comparison, output_dir):
    dirs = {"matplotlib_cache": os.path.join(output_dir, "matplotlib_cache")}
    plt = common_predict.setup_matplotlib(dirs)
    paths = {}

    rank_path = os.path.join(output_dir, "feature_screening_f0_f8_test_nrmse_rank.png")
    rank = comparison.sort_values(SELECTION_MACRO_METRIC, ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [
        "tab:red" if bool(value) else "tab:blue"
        for value in rank["selected_final_variant"]
    ]
    ax.barh(rank["model_variant"], rank[SELECTION_MACRO_METRIC], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Macro mean test capacity-normalized RMSE")
    ax.set_title("F0--F8 test-set feature selection")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(rank_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["rank_figure"] = rank_path

    heatmap_path = os.path.join(output_dir, "feature_screening_f0_f8_test_farm_heatmap.png")
    pivot = summary.pivot(
        index="model_variant", columns="farm_id", values=SELECTION_METRIC
    ).reindex(SELECTION_VARIANTS)
    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_xticks(
        np.arange(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right"
    )
    ax.set_title("Per-farm test NRMSE (lower is better)")
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            ax.text(
                column,
                row,
                f"{pivot.iloc[row, column]:.4f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if pivot.iloc[row, column] > pivot.to_numpy().mean() else "black",
            )
    fig.colorbar(image, ax=ax, label="Capacity-normalized RMSE")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["farm_heatmap"] = heatmap_path

    horizon_path = os.path.join(output_dir, "feature_screening_f0_f8_test_horizon_nrmse.png")
    fig, ax = plt.subplots(figsize=(11, 6))
    for variant_id in SELECTION_VARIANTS:
        frame = horizon_comparison[
            horizon_comparison["model_variant"] == variant_id
        ]
        ax.plot(
            frame["horizon_minutes"],
            frame["macro_mean_capacity_normalized_rmse"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=variant_id,
        )
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("Macro mean capacity-normalized RMSE")
    ax.set_title("F0--F8 test NRMSE by horizon")
    ax.grid(alpha=0.3)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(horizon_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["horizon_figure"] = horizon_path

    pareto_path = os.path.join(output_dir, "feature_screening_f0_f8_test_pareto.png")
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in comparison.iterrows():
        selected = bool(row["selected_final_variant"])
        ax.scatter(
            row["parameter_count_max"],
            row[SELECTION_MACRO_METRIC],
            s=80 if selected else 45,
            color="tab:red" if selected else "tab:blue",
        )
        ax.annotate(
            f"{row['model_variant']} ({int(row['feature_count'])}F)",
            (row["parameter_count_max"], row[SELECTION_MACRO_METRIC]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Parameter count")
    ax.set_ylabel("Macro mean test capacity-normalized RMSE")
    ax.set_title("F0--F8 accuracy--complexity comparison")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(pareto_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["pareto_figure"] = pareto_path
    return paths


def _selection_markdown(comparison, contribution, output_path, figure_paths):
    selected = comparison[comparison["selected_final_variant"]].iloc[0]
    columns = [
        "selection_rank",
        "model_variant",
        "feature_groups",
        "feature_count",
        "parameter_count_max",
        SELECTION_MACRO_METRIC,
        "macro_std_capacity_normalized_rmse",
        "farms_better_than_f4",
    ]
    table = comparison[columns].copy()
    for column in (
        SELECTION_MACRO_METRIC,
        "macro_std_capacity_normalized_rmse",
    ):
        table[column] = table[column].map(lambda value: f"{value:.6f}")
    contribution_table = contribution[[
        "comparison",
        "source_variant",
        "target_variant",
        "target_minus_source_nrmse",
        "relative_change_pct",
        "farms_improved",
        "farms_degraded",
    ]].copy()
    contribution_table["target_minus_source_nrmse"] = contribution_table[
        "target_minus_source_nrmse"
    ].map(lambda value: f"{value:+.6f}")
    contribution_table["relative_change_pct"] = contribution_table[
        "relative_change_pct"
    ].map(lambda value: f"{value:+.3f}%")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("# F0--F8 显式工况特征筛选（测试集选型）\n\n")
        file.write(
            f"最终模型：**{selected['model_variant']}**，特征组 "
            f"`{selected['feature_groups']}`，测试集5场站等权宏平均NRMSE="
            f"`{selected[SELECTION_MACRO_METRIC]:.6f}`。\n\n"
        )
        file.write("## 选型口径\n\n")
        file.write(
            "- 主指标：每个场站全16步容量归一化RMSE，再对场站等权宏平均。\n"
            "- 主指标越低越优；完全同值依次按跨场站标准差、特征数、参数量和F编号破平。\n"
            "- F4直接引用既有R4模型、预测、诊断和图形，没有重新训练或推理。\n"
            "- 本测试段状态为`legacy_seen`且此次用于特征选型，因此不再是最终盲测。\n\n"
        )
        file.write("## 排名\n\n")
        file.write(table.to_markdown(index=False))
        file.write("\n\n## 特征组增量/反向消融\n\n")
        file.write(contribution_table.to_markdown(index=False))
        file.write("\n\n## 图形\n\n")
        for name, path in figure_paths.items():
            file.write(f"- {name}: `{os.path.abspath(path)}`\n")


def save_cross_variant_outputs(results, variants, full_matrix):
    """Deprecated unsafe writer retained only to fail loudly for old callers."""
    del results, variants, full_matrix
    raise RuntimeError(
        "save_cross_variant_outputs已停用；请使用先验收后发布的"
        "save_extended_selection_outputs"
    )


RESULT_KEYS = (
    "summary",
    "horizon",
    "candidate",
    "regime",
    "gate",
    "calibration",
    "assignments",
)


def _concat_results(results):
    combined = {}
    for key in RESULT_KEYS:
        frames = [result[key] for result in results if not result[key].empty]
        combined[key] = (
            pd.concat(frames, ignore_index=True, sort=False)
            if frames
            else pd.DataFrame()
        )
        if "farm_id" in combined[key]:
            combined[key]["farm_id"] = combined[key]["farm_id"].astype(str)
    return combined


def _validate_result_matrix_structure(combined, variants, expected_farm_ids):
    variants = tuple(variants)
    farms = {str(value) for value in expected_farm_ids}
    model_count = len(variants) * len(farms)
    regime_count = len(regime_predict.REGIME_GROUP_ORDER)
    specifications = {
        "summary": (
            model_count,
            ["model_variant", "farm_id"],
        ),
        "horizon": (
            model_count * (regime_train.FORECAST_LEN + 1),
            ["model_variant", "farm_id", "horizon_step"],
        ),
        "candidate": (
            model_count * 3 * (regime_train.FORECAST_LEN + 1),
            ["model_variant", "farm_id", "candidate", "horizon_step"],
        ),
        "regime": (
            model_count
            * regime_count
            * 3
            * (regime_train.FORECAST_LEN + 1),
            [
                "model_variant",
                "farm_id",
                "regime_group",
                "candidate",
                "horizon_step",
            ],
        ),
        "gate": (
            model_count * regime_count * regime_train.FORECAST_LEN,
            [
                "model_variant",
                "farm_id",
                "regime_group",
                "horizon_step",
            ],
        ),
        "calibration": (
            model_count * 10,
            ["model_variant", "farm_id", "gate_bin"],
        ),
    }
    for name, (expected_rows, keys) in specifications.items():
        frame = combined[name]
        missing = set(keys) - set(frame.columns)
        if missing:
            raise KeyError(f"正式{name}缺少结构键{sorted(missing)}")
        if (
            len(frame) != expected_rows
            or frame.duplicated(keys).any()
            or set(frame["model_variant"].astype(str)) != set(variants)
            or set(frame["farm_id"].astype(str)) != farms
        ):
            raise ValueError(
                f"正式{name}矩阵不完整/重复: rows={len(frame)}, "
                f"expected={expected_rows}"
            )
    assignments = combined["assignments"]
    assignment_keys = ["model_variant", "farm_id", "sample_id"]
    missing = set(assignment_keys) - set(assignments.columns)
    if missing:
        raise KeyError(f"正式assignments缺少结构键{sorted(missing)}")
    if (
        assignments.duplicated(assignment_keys).any()
        or set(assignments["model_variant"].astype(str)) != set(variants)
        or set(assignments["farm_id"].astype(str)) != farms
    ):
        raise ValueError("正式assignments模型/场站/样本键不完整或重复")
    counts = assignments.groupby(["farm_id", "model_variant"]).size().unstack()
    counts.index = counts.index.astype(str)
    if (
        set(counts.index) != farms
        or set(counts.columns.astype(str)) != set(variants)
        or counts.isna().any().any()
        or not (counts.nunique(axis=1) == 1).all()
        or not (counts > 0).all().all()
    ):
        raise ValueError("正式assignments各模型未按场站覆盖相同正样本数")


def _partial_suffix(summary, variants, full_matrix):
    if full_matrix:
        return ""
    farms = sorted(summary["farm_id"].astype(str).unique()) if not summary.empty else []
    raw = f"{'-'.join(variants)}__farms_{'-'.join(farms)}"
    tag = raw if len(raw) <= 140 else hashlib.sha1(raw.encode()).hexdigest()[:12]
    return f"_partial_{tag}"


def _extended_test_comparison(summary):
    metric_names = (
        "mae",
        "mse",
        "rmse",
        "mape",
        "smape",
        "r2",
        "capacity_normalized_mae",
        "capacity_normalized_rmse",
        "weighted_curve_mae",
        "weighted_curve_rmse",
        "weighted_curve_capacity_normalized_mae",
        "weighted_curve_capacity_normalized_rmse",
        "gate_mean",
        "gate_oracle_brier",
        "gate_saturation_high_rate",
        "gate_sample_variation",
    )
    f4 = summary[summary["model_variant"] == "f4"][[
        "farm_id",
        SELECTION_METRIC,
    ]].rename(columns={SELECTION_METRIC: "f4_nrmse"})
    rows = []
    for order, variant_id in enumerate(SELECTION_VARIANTS):
        frame = summary[summary["model_variant"] == variant_id].copy()
        if frame.empty:
            continue
        parameters = pd.to_numeric(frame["parameter_count"], errors="coerce")
        paired = frame[["farm_id", SELECTION_METRIC]].merge(f4, on="farm_id")
        values = pd.to_numeric(paired[SELECTION_METRIC], errors="coerce")
        reference = pd.to_numeric(paired["f4_nrmse"], errors="coerce")
        row = {
            "variant_order": order,
            "model_variant": variant_id,
            "model_name": variant_model_name(variant_id),
            "feature_groups": "+".join(VARIANT_SPECS[variant_id]["groups"]),
            "feature_count": len(selected_feature_names(variant_id)),
            "farm_count": int(frame["farm_id"].nunique()),
            "parameter_count_min": int(parameters.min()),
            "parameter_count_max": int(parameters.max()),
            "requires_new_inference": variant_id == "f8",
            "result_source": (
                "new_f8_inference"
                if variant_id == "f8"
                else "read_only_reuse_existing_f0_f7_aggregate"
            ),
            "selection_split": "test",
            "selection_metric": SELECTION_MACRO_METRIC,
            "test_reuse_status": TEST_REUSE_STATUS,
            "test_used_for_feature_selection": True,
            "test_is_final_blind_evaluation": False,
            "farms_better_than_f4": int((values < reference).sum()),
            "farms_close_to_f4": int(
                np.isclose(values, reference, rtol=1e-5, atol=1e-12).sum()
            ),
            "macro_nrmse_delta_vs_f4": float((values - reference).mean()),
        }
        for metric in metric_names:
            metric_values = pd.to_numeric(frame.get(metric), errors="coerce")
            if metric == "gate_oracle_brier":
                finite_protocol = frame.get(
                    "gate_oracle_elementwise_finite_masked",
                    pd.Series(False, index=frame.index),
                ).fillna(False).astype(bool)
                metric_values = metric_values.where(finite_protocol)
            row[f"macro_mean_{metric}"] = float(metric_values.mean())
            row[f"macro_std_{metric}"] = float(metric_values.std(ddof=0))
        oracle_protocol = frame.get(
            "gate_oracle_elementwise_finite_masked",
            pd.Series(False, index=frame.index),
        ).fillna(False).astype(bool)
        row["gate_oracle_metric_protocol"] = (
            "elementwise_finite_masked_v2"
            if oracle_protocol.all()
            else "legacy_unmasked_excluded_from_cross_variant_comparison"
        )
        rows.append(row)
    comparison = pd.DataFrame(rows)
    if set(comparison["model_variant"]) != set(SELECTION_VARIANTS):
        raise ValueError("F0--F8比较未完整覆盖选型变体")
    if not np.isfinite(comparison[SELECTION_MACRO_METRIC]).all():
        raise ValueError("F0--F8至少一个宏平均NRMSE非有限")
    ordered = comparison.sort_values(
        [
            SELECTION_MACRO_METRIC,
            "macro_std_capacity_normalized_rmse",
            "feature_count",
            "parameter_count_max",
            "variant_order",
        ],
        kind="mergesort",
    ).index
    comparison["selection_rank"] = pd.Series(
        np.arange(1, len(ordered) + 1), index=ordered
    )
    comparison["selected_final_variant"] = comparison["selection_rank"] == 1
    comparison["selection_tie_break_order"] = (
        "macro_test_nrmse -> macro_test_nrmse_std -> feature_count -> "
        "parameter_count -> variant_order"
    )
    return comparison.sort_values("selection_rank").reset_index(drop=True)


def _pair_effect_rows(summary):
    comparisons = (
        ("f0", "f1", "add_H", "P -> P+H"),
        ("f1", "f7", "add_D_without_M_C", "P+H -> P+H+D"),
        ("f7", "f8", "add_C_without_M", "P+H+D -> P+H+D+C"),
        ("f7", "f3", "add_M_without_C", "P+H+D -> P+H+M+D"),
        ("f3", "f4", "add_C_with_M", "P+H+M+D -> full"),
        ("f8", "f4", "add_M_with_C", "P+H+D+C -> full"),
        ("f5", "f3", "add_P", "H+M+D -> P+H+M+D"),
        ("f6", "f3", "add_H_reverse", "P+M+D -> P+H+M+D"),
    )
    values = summary[["model_variant", "farm_id", SELECTION_METRIC]].copy()
    values[SELECTION_METRIC] = pd.to_numeric(values[SELECTION_METRIC], errors="coerce")
    rows = []
    for source_id, target_id, change, description in comparisons:
        source = values[values["model_variant"] == source_id][
            ["farm_id", SELECTION_METRIC]
        ].rename(columns={SELECTION_METRIC: "source_nrmse"})
        target = values[values["model_variant"] == target_id][
            ["farm_id", SELECTION_METRIC]
        ].rename(columns={SELECTION_METRIC: "target_nrmse"})
        paired = source.merge(target, on="farm_id")
        delta = paired["target_nrmse"] - paired["source_nrmse"]
        source_macro = float(paired["source_nrmse"].mean())
        target_macro = float(paired["target_nrmse"].mean())
        rows.append(
            {
                "comparison": change,
                "description": description,
                "source_variant": source_id,
                "target_variant": target_id,
                "paired_farm_count": len(paired),
                "source_macro_test_nrmse": source_macro,
                "target_macro_test_nrmse": target_macro,
                "target_minus_source_nrmse": target_macro - source_macro,
                "macro_relative_change_pct": (
                    (target_macro / source_macro - 1.0) * 100.0
                ),
                "mean_paired_relative_change_pct": float(
                    (delta / paired["source_nrmse"] * 100.0).mean()
                ),
                "farms_improved": int((delta < 0).sum()),
                "farms_degraded": int((delta > 0).sum()),
            }
        )
    result = pd.DataFrame(rows)
    lookup = result.set_index("comparison")["target_minus_source_nrmse"]
    interaction = float(lookup["add_C_with_M"] - lookup["add_C_without_M"])
    result["M_x_C_interaction_nrmse"] = np.nan
    result.loc[result["comparison"] == "add_C_without_M", "M_x_C_interaction_nrmse"] = interaction
    return result


def _extended_horizon_comparison(horizon):
    frame = horizon[
        horizon["model_variant"].astype(str).isin(SELECTION_VARIANTS)
        & (horizon["horizon_step"].astype(str) != "all")
    ].copy()
    frame["horizon_step"] = pd.to_numeric(frame["horizon_step"], errors="raise")
    frame[SELECTION_METRIC] = pd.to_numeric(frame[SELECTION_METRIC], errors="coerce")
    grouped = (
        frame.groupby(["model_variant", "horizon_step"], as_index=False)
        .agg(
            macro_mean_capacity_normalized_rmse=(SELECTION_METRIC, "mean"),
            macro_std_capacity_normalized_rmse=(SELECTION_METRIC, lambda x: x.std(ddof=0)),
            farm_count=("farm_id", "nunique"),
        )
    )
    grouped["horizon_minutes"] = grouped["horizon_step"] * 15
    f4 = grouped[grouped["model_variant"] == "f4"][[
        "horizon_step",
        "macro_mean_capacity_normalized_rmse",
    ]].rename(columns={"macro_mean_capacity_normalized_rmse": "f4_macro_nrmse"})
    grouped = grouped.merge(f4, on="horizon_step", how="left")
    grouped["delta_vs_f4"] = (
        grouped["macro_mean_capacity_normalized_rmse"]
        - grouped["f4_macro_nrmse"]
    )
    return grouped.sort_values(["model_variant", "horizon_step"])


def _save_extended_figures(comparison, summary, horizon, output_dir):
    dirs = {"matplotlib_cache": os.path.join(output_dir, "matplotlib_cache")}
    plt = common_predict.setup_matplotlib(dirs)
    paths = {}
    rank = comparison.sort_values("selection_rank")

    path = os.path.join(output_dir, "feature_screening_f0_f8_test_nrmse_rank.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["tab:red" if value else "tab:blue" for value in rank["selected_final_variant"]]
    ax.barh(rank["model_variant"], rank[SELECTION_MACRO_METRIC], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Macro mean test capacity-normalized RMSE")
    ax.set_title("F0--F8 test-set feature selection")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["rank_figure"] = path

    path = os.path.join(output_dir, "feature_screening_f0_f8_test_farm_heatmap.png")
    pivot = summary.pivot(index="model_variant", columns="farm_id", values=SELECTION_METRIC).reindex(SELECTION_VARIANTS)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
    ax.set_title("F0--F8 per-farm test NRMSE")
    fig.colorbar(image, ax=ax, label="Capacity-normalized RMSE")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["farm_heatmap"] = path

    path = os.path.join(output_dir, "feature_screening_f0_f8_test_horizon_nrmse.png")
    fig, ax = plt.subplots(figsize=(11, 6))
    for variant_id in SELECTION_VARIANTS:
        frame = horizon[horizon["model_variant"] == variant_id]
        ax.plot(frame["horizon_minutes"], frame["macro_mean_capacity_normalized_rmse"], marker="o", markersize=3, linewidth=1.3, label=variant_id)
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("Macro mean capacity-normalized RMSE")
    ax.set_title("F0--F8 test NRMSE by horizon")
    ax.grid(alpha=0.3)
    ax.legend(ncol=5)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["horizon_figure"] = path

    path = os.path.join(output_dir, "feature_screening_f0_f8_test_pareto.png")
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in comparison.iterrows():
        selected = bool(row["selected_final_variant"])
        ax.scatter(row["parameter_count_max"], row[SELECTION_MACRO_METRIC], s=90 if selected else 50, color="tab:red" if selected else "tab:blue")
        ax.annotate(row["model_variant"], (row["parameter_count_max"], row[SELECTION_MACRO_METRIC]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Parameter count")
    ax.set_ylabel("Macro mean test capacity-normalized RMSE")
    ax.set_title("F0--F8 accuracy--complexity comparison")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["pareto_figure"] = path
    return paths


def _write_extended_selection_report(comparison, effects, path, figure_paths):
    selected = comparison[comparison["selected_final_variant"]].iloc[0]
    columns = [
        "selection_rank",
        "model_variant",
        "feature_groups",
        "feature_count",
        "parameter_count_max",
        SELECTION_MACRO_METRIC,
        "macro_std_capacity_normalized_rmse",
        "farms_better_than_f4",
    ]
    with open(path, "w", encoding="utf-8") as file:
        file.write("# F0--F8显式工况特征补充筛选\n\n")
        file.write(
            f"按测试集5场站等权宏平均NRMSE选中 **{selected['model_variant']}** "
            f"(`{selected[SELECTION_MACRO_METRIC]:.9f}`)。\n\n"
        )
        file.write(
            "- F0--F7从旧聚合文件只读复用，未重新推理。\n"
            "- F8为P+H+D+C，用于检验无M条件下C的端到端贡献。\n"
            "- FP0/FP4不参与本排名，在Frozen-Pair control报告中单独分析。\n"
            "- 本测试段是legacy_seen且已用于选型，不是最终盲测。\n"
            "- 整套补充实验仅在bundle complete JSON存在且所列hash通过时视为完整。\n\n"
        )
        file.write("## F0--F8排名\n\n")
        file.write(comparison[columns].to_markdown(index=False))
        file.write("\n\n## 特征组和M×C交互\n\n")
        file.write(effects.to_markdown(index=False))
        file.write("\n\n## 图形\n\n")
        for name, figure_path in figure_paths.items():
            file.write(f"- {name}: `{os.path.abspath(figure_path)}`\n")


def save_extended_selection_outputs(legacy, f8_result, full_matrix):
    output_dir = comparison_output_dir()
    results = [legacy] + ([f8_result] if f8_result is not None else [])
    combined = _concat_results(results)
    variants = list(SELECTION_VARIANTS if f8_result is not None else LEGACY_SELECTION_VARIANTS)
    suffix = _partial_suffix(combined["summary"], variants, full_matrix)
    paths = {}
    stems = {
        "summary": "feature_screening_f0_f8_test_metrics_summary",
        "horizon": "feature_screening_f0_f8_test_metrics_by_horizon_all",
        "candidate": "feature_screening_f0_f8_test_candidate_all",
        "regime": "feature_screening_f0_f8_test_metrics_by_regime_all",
        "gate": "feature_screening_f0_f8_test_gate_all",
        "calibration": "feature_screening_f0_f8_test_gate_calibration_all",
        "assignments": "feature_screening_f0_f8_test_regime_assignments_all",
    }
    comparison = effects = horizon = None
    if full_matrix:
        # 正式无suffix文件必须在内存中通过完整性和真值验收后才发布。
        if "valid_count" not in combined["summary"]:
            raise KeyError("F0--F8正式summary缺少valid_count")
        expected_farm_set = set(expected_test_farm_ids())
        expected_farms = len(expected_farm_set)
        expected_rows = len(SELECTION_VARIANTS) * expected_farms
        _validate_result_matrix_structure(
            combined,
            SELECTION_VARIANTS,
            expected_farm_set,
        )
        summary_keys = combined["summary"][["model_variant", "farm_id"]]
        summary_metric = pd.to_numeric(
            combined["summary"][SELECTION_METRIC], errors="coerce"
        )
        valid_count = pd.to_numeric(
            combined["summary"].get("valid_count"), errors="coerce"
        )
        if (
            len(combined["summary"]) != expected_rows
            or summary_keys.duplicated().any()
            or combined["summary"]["model_variant"].nunique()
            != len(SELECTION_VARIANTS)
            or combined["summary"]["farm_id"].nunique() != expected_farms
            or set(combined["summary"]["farm_id"].astype(str))
            != expected_farm_set
            or not np.isfinite(summary_metric).all()
            or not np.isfinite(valid_count).all()
            or not (valid_count > 0).all()
            or not (
                combined["summary"]
                .assign(_valid_count=valid_count)
                .groupby("farm_id")["_valid_count"]
                .nunique()
                == 1
            ).all()
        ):
            raise ValueError(
                f"F0--F8 summary必须是{expected_rows}个唯一、有限且"
                "真值有效数一致的variant/farm键"
            )
        validate_truth_alignment(combined["summary"], SELECTION_VARIANTS)
        raw_horizon = combined["horizon"]
        expected_raw_horizon_rows = expected_rows * (
            regime_train.FORECAST_LEN + 1
        )
        raw_horizon_metric = pd.to_numeric(
            raw_horizon[SELECTION_METRIC], errors="coerce"
        )
        if (
            len(raw_horizon) != expected_raw_horizon_rows
            or raw_horizon.duplicated(
                ["model_variant", "farm_id", "horizon_step"]
            ).any()
            or not np.isfinite(raw_horizon_metric).all()
        ):
            raise ValueError(
                "F0--F8逐场站horizon主指标不完整、重复或包含非有限值"
            )
        comparison = _extended_test_comparison(combined["summary"])
        effects = _pair_effect_rows(combined["summary"])
        horizon = _extended_horizon_comparison(combined["horizon"])
        expected_horizon_rows = len(SELECTION_VARIANTS) * regime_train.FORECAST_LEN
        if (
            len(horizon) != expected_horizon_rows
            or horizon[["model_variant", "horizon_step"]].duplicated().any()
            or not (horizon["farm_count"] == expected_farms).all()
        ):
            raise ValueError(
                "F0--F8 horizon比较未完整覆盖9个变体、16步和全部场站"
            )

    # partial也使用唯一suffix，正式运行则至此已经全部验收通过。
    for key, stem in stems.items():
        path = os.path.join(output_dir, f"{stem}{suffix}.csv")
        _atomic_to_csv(combined[key], path)
        paths[key] = path
    manifest_path = os.path.join(
        output_dir,
        f"feature_screening_f0_f8_legacy_source_manifest{suffix}.csv",
    )
    _atomic_to_csv(legacy["source_manifest"], manifest_path)
    paths["legacy_source_manifest"] = manifest_path

    if not full_matrix:
        note = os.path.join(
            output_dir,
            f"feature_screening_f0_f8_partial_note{suffix}.md",
        )
        _atomic_write_text(
            note,
            "# F0--F8部分运行\n\n"
            "未覆盖完整F0--F8和全部场站，不生成winner。\n",
        )
        paths["partial_note"] = note
        return combined, paths

    table_paths = {
        "comparison": os.path.join(output_dir, "feature_screening_f0_f8_test_variant_comparison.csv"),
        "feature_effects": os.path.join(output_dir, "feature_screening_f0_f8_test_feature_effects.csv"),
        "horizon_comparison": os.path.join(output_dir, "feature_screening_f0_f8_test_horizon_comparison.csv"),
        "final_selection": os.path.join(output_dir, "feature_screening_f0_f8_test_final_selection.csv"),
    }
    _atomic_to_csv(comparison, table_paths["comparison"])
    _atomic_to_csv(effects, table_paths["feature_effects"])
    _atomic_to_csv(horizon, table_paths["horizon_comparison"])
    _atomic_to_csv(
        comparison[comparison["selected_final_variant"]],
        table_paths["final_selection"],
    )
    figures = _save_extended_figures(comparison, combined["summary"], horizon, output_dir)
    report = os.path.join(output_dir, "feature_screening_f0_f8_test_final_selection.md")
    report_temporary = f"{report}.tmp"
    try:
        _write_extended_selection_report(
            comparison,
            effects,
            report_temporary,
            figures,
        )
        os.replace(report_temporary, report)
    finally:
        if os.path.exists(report_temporary):
            os.remove(report_temporary)
    paths.update(table_paths)
    paths.update(figures)
    paths["selection_report"] = report
    combined["comparison"] = comparison
    combined["feature_effects"] = effects
    combined["horizon_comparison"] = horizon
    return combined, paths


def _load_archive(path):
    resolved = _resolve_existing_path(path)
    if not resolved:
        raise FileNotFoundError(f"候选archive不存在: {path}")
    with np.load(resolved, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _validate_archive_identity_and_truth(
    archive,
    summary_row,
    farm_id,
    formal_reference,
):
    required = {
        "schema_version",
        "farm_id",
        "sample_id",
        "horizon_step",
        "forecast_origin_time",
        "y_true",
        "persistence_scaled",
        "corrected_scaled",
        "persistence",
        "corrected",
        "gate",
        "fused",
        "capacity",
    }
    missing = required - set(archive)
    if missing:
        raise KeyError(f"{farm_id}候选archive缺少字段{sorted(missing)}")
    schema = str(np.asarray(archive["schema_version"]).item())
    archive_farm = str(np.asarray(archive["farm_id"]).item())
    if schema != "candidate_archive_v1" or archive_farm != str(farm_id):
        raise ValueError(
            f"候选archive身份不匹配: schema={schema}, farm={archive_farm}"
        )
    shape = np.asarray(archive["gate"]).shape
    if len(shape) != 2 or shape[1] != regime_train.FORECAST_LEN:
        raise ValueError(f"{farm_id}候选archive gate形状异常: {shape}")
    for key in (
        "y_true",
        "persistence_scaled",
        "corrected_scaled",
        "persistence",
        "corrected",
        "fused",
    ):
        if np.asarray(archive[key]).shape != shape:
            raise ValueError(f"{farm_id}候选archive {key}形状不一致")
    sample_id = np.asarray(archive["sample_id"], dtype=np.int64)
    horizon_step = np.asarray(archive["horizon_step"], dtype=np.int64)
    origins = np.asarray(archive["forecast_origin_time"]).astype(str)
    if (
        not np.array_equal(sample_id, np.arange(shape[0], dtype=np.int64))
        or not np.array_equal(
            horizon_step,
            np.arange(1, regime_train.FORECAST_LEN + 1, dtype=np.int64),
        )
        or origins.shape != (shape[0],)
    ):
        raise ValueError(f"{farm_id}候选archive窗口键异常")
    capacity = float(np.asarray(archive["capacity"]).item())
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError(f"{farm_id}候选archive容量无效: {capacity}")
    artifact_path = _resolve_existing_path(summary_row.get("artifact_path"))
    if artifact_path is None:
        raise FileNotFoundError(f"{farm_id}候选archive缺少对应artifact")
    artifact = joblib.load(artifact_path)
    artifact_capacity = float(artifact.get("capacity", np.nan))
    if not np.isclose(capacity, artifact_capacity, rtol=0.0, atol=1e-9):
        raise ValueError(f"{farm_id}候选archive容量与artifact不一致")

    prediction = _load_prediction_truth(summary_row.get("prediction_path"))
    if "pred_power" not in prediction:
        raise KeyError(f"{farm_id}预测CSV缺少pred_power")
    expected_sample = np.repeat(sample_id, len(horizon_step))
    expected_horizon = np.tile(horizon_step, len(sample_id))
    expected_length = shape[0] * shape[1]
    if len(prediction) != expected_length:
        raise ValueError(
            f"{farm_id}候选archive与预测CSV行数不一致: "
            f"{expected_length} != {len(prediction)}"
        )
    if not np.array_equal(
        prediction["sample_id"].to_numpy(dtype=np.int64), expected_sample
    ) or not np.array_equal(
        prediction["horizon_step"].to_numpy(dtype=np.int64), expected_horizon
    ):
        raise ValueError(f"{farm_id}候选archive与预测CSV窗口键不一致")
    if not np.allclose(
        pd.to_numeric(prediction["actual_power"], errors="coerce"),
        np.asarray(archive["y_true"], dtype=float).reshape(-1),
        rtol=0.0,
        atol=1e-7,
        equal_nan=True,
    ):
        raise ValueError(f"{farm_id}候选archive与预测CSV真实值不一致")
    if not _csv_float_matches_archive(
        prediction["pred_power"], archive["fused"]
    ):
        raise ValueError(f"{farm_id}候选archive与预测CSV fused预测不一致")
    if "forecast_origin_time" in prediction and not np.array_equal(
        prediction["forecast_origin_time"].astype(str).to_numpy(),
        np.repeat(origins, len(horizon_step)),
    ):
        raise ValueError(f"{farm_id}候选archive起报时刻与预测CSV不一致")
    if (
        len(formal_reference) != len(prediction)
        or not np.array_equal(
            formal_reference[["sample_id", "horizon_step"]].to_numpy(),
            prediction[["sample_id", "horizon_step"]].to_numpy(),
        )
        or not np.allclose(
            pd.to_numeric(
                formal_reference["actual_power"], errors="coerce"
            ),
            np.asarray(archive["y_true"], dtype=float).reshape(-1),
            rtol=0.0,
            atol=1e-7,
            equal_nan=True,
        )
    ):
        raise ValueError(f"{farm_id}候选archive与正式F4测试基准不一致")
    recomputed = common_predict.calculate_metrics(
        archive["y_true"],
        archive["fused"],
        capacity,
    )
    if (
        int(recomputed["valid_count"])
        != int(summary_row.get("valid_count", -1))
        or not np.isclose(
            float(recomputed[SELECTION_METRIC]),
            float(summary_row.get(SELECTION_METRIC, np.nan)),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError(f"{farm_id} archive复算指标与FP summary不一致")
    return capacity


def validate_frozen_pair_archives(probe_summary):
    legacy_summary_path = os.path.join(
        legacy_output_dir(), LEGACY_AGGREGATE_FILES["summary"]
    )
    legacy_summary = _read_csv_with_farm_id(legacy_summary_path)
    f4_reference_rows = legacy_summary[
        legacy_summary["model_variant"].astype(str) == "f4"
    ].set_index("farm_id")
    rows = []
    for farm_id, frame in probe_summary.groupby("farm_id"):
        if set(frame["model_variant"]) != set(PROBE_VARIANTS):
            raise ValueError(f"Frozen-Pair场站{farm_id}未同时覆盖FP0/FP4")
        first_row = frame[frame["model_variant"] == "fp0"].iloc[0]
        second_row = frame[frame["model_variant"] == "fp4"].iloc[0]
        for metadata_key in (
            "source_model_sha256",
            "source_artifact_sha256",
            "frozen_weights_sha256_after_training",
        ):
            if (
                metadata_key not in frame
                or pd.isna(first_row.get(metadata_key))
                or first_row.get(metadata_key) != second_row.get(metadata_key)
            ):
                raise ValueError(
                    f"{farm_id} FP0/FP4 {metadata_key}不一致，"
                    "Frozen-Pair control无效"
                )
        if str(farm_id) not in f4_reference_rows.index:
            raise ValueError(f"Frozen-Pair场站{farm_id}缺少正式F4真值基准")
        f4_row = f4_reference_rows.loc[str(farm_id)]
        formal_reference = _load_prediction_truth(
            _first_present(
                f4_row.get("source_prediction_path"),
                f4_row.get("prediction_path"),
            )
        )
        first_path = _resolve_existing_path(first_row["candidate_archive_path"])
        second_path = _resolve_existing_path(second_row["candidate_archive_path"])
        first = _load_archive(first_path)
        second = _load_archive(second_path)
        first_archive_hash = _sha256(first_path)
        second_archive_hash = _sha256(second_path)
        if (
            first_archive_hash != first_row.get("candidate_archive_sha256")
            or second_archive_hash != second_row.get("candidate_archive_sha256")
        ):
            raise ValueError(f"{farm_id}候选archive文件SHA256与summary不一致")
        first_scaled_hash = _array_sha256(
            first["persistence_scaled"], first["corrected_scaled"]
        )
        second_scaled_hash = _array_sha256(
            second["persistence_scaled"], second["corrected_scaled"]
        )
        first_physical_hash = _array_sha256(
            first["persistence"], first["corrected"]
        )
        second_physical_hash = _array_sha256(
            second["persistence"], second["corrected"]
        )
        for row, scaled_hash, physical_hash in (
            (first_row, first_scaled_hash, first_physical_hash),
            (second_row, second_scaled_hash, second_physical_hash),
        ):
            if (
                scaled_hash != row.get("candidate_pair_scaled_sha256")
                or physical_hash != row.get("candidate_pair_physical_sha256")
            ):
                raise ValueError(
                    f"{farm_id}/{row['model_variant']}候选数组hash与summary不一致"
                )
        first_capacity = _validate_archive_identity_and_truth(
            first, first_row, farm_id, formal_reference
        )
        second_capacity = _validate_archive_identity_and_truth(
            second, second_row, farm_id, formal_reference
        )
        if not np.isclose(
            first_capacity, second_capacity, rtol=0.0, atol=1e-9
        ):
            raise ValueError(f"{farm_id} FP0/FP4容量不一致")
        for key in ("sample_id", "horizon_step", "forecast_origin_time"):
            if not np.array_equal(first[key], second[key]):
                raise ValueError(f"{farm_id} FP0/FP4 archive键{key}不一致")
        if not np.allclose(first["y_true"], second["y_true"], rtol=0, atol=1e-7, equal_nan=True):
            raise ValueError(f"{farm_id} FP0/FP4真值不一致")
        scaled_p = np.asarray(first["persistence_scaled"], dtype=float) - np.asarray(second["persistence_scaled"], dtype=float)
        scaled_c = np.asarray(first["corrected_scaled"], dtype=float) - np.asarray(second["corrected_scaled"], dtype=float)
        physical_p = np.asarray(first["persistence"], dtype=float) - np.asarray(second["persistence"], dtype=float)
        physical_c = np.asarray(first["corrected"], dtype=float) - np.asarray(second["corrected"], dtype=float)
        scaled_p_max = float(np.max(np.abs(scaled_p)))
        scaled_c_max = float(np.max(np.abs(scaled_c)))
        physical_p_max = float(np.max(np.abs(physical_p)))
        physical_c_max = float(np.max(np.abs(physical_c)))
        scaled_exact = bool(
            np.array_equal(first["persistence_scaled"], second["persistence_scaled"])
            and np.array_equal(first["corrected_scaled"], second["corrected_scaled"])
        )
        scaled_within = scaled_p_max <= FP_SCALED_ATOL and scaled_c_max <= FP_SCALED_ATOL
        physical_limit = max(
            FP_PHYSICAL_ATOL,
            float(first["capacity"]) * FP_SCALED_ATOL,
        )
        physical_within = physical_p_max <= physical_limit and physical_c_max <= physical_limit
        first_oracle_valid = (
            np.isfinite(first["y_true"])
            & np.isfinite(first["persistence"])
            & np.isfinite(first["corrected"])
        )
        second_oracle_valid = (
            np.isfinite(second["y_true"])
            & np.isfinite(second["persistence"])
            & np.isfinite(second["corrected"])
        )
        oracle_valid_mask_equal = bool(
            np.array_equal(first_oracle_valid, second_oracle_valid)
        )
        if not oracle_valid_mask_equal:
            raise ValueError(
                f"{farm_id} FP0/FP4 corrected-better有效点掩码不一致"
            )
        oracle_first = (
            np.square(
                first["corrected"][first_oracle_valid]
                - first["y_true"][first_oracle_valid]
            )
            < np.square(
                first["persistence"][first_oracle_valid]
                - first["y_true"][first_oracle_valid]
            )
        )
        oracle_second = (
            np.square(
                second["corrected"][second_oracle_valid]
                - second["y_true"][second_oracle_valid]
            )
            < np.square(
                second["persistence"][second_oracle_valid]
                - second["y_true"][second_oracle_valid]
            )
        )
        oracle_equal = bool(np.array_equal(oracle_first, oracle_second))
        scaled_hash_equal = first_scaled_hash == second_scaled_hash
        physical_hash_equal = first_physical_hash == second_physical_hash
        if not scaled_exact or not scaled_hash_equal:
            raise ValueError(
                f"{farm_id} Frozen-Pair标准化候选未通过位级/hash一致性: "
                f"exact={scaled_exact}, hash_equal={scaled_hash_equal}, "
                f"P={scaled_p_max}, C={scaled_c_max}"
            )
        if not physical_within:
            raise ValueError(
                f"{farm_id} Frozen-Pair物理量候选超出浮点容差: "
                f"P={physical_p_max}, C={physical_c_max}, limit={physical_limit}"
            )
        if not oracle_equal:
            raise ValueError(f"{farm_id} Frozen-Pair corrected-better oracle不一致")
        rows.append(
            {
                "farm_id": str(farm_id),
                "fp0_archive_path": first_row["candidate_archive_path"],
                "fp4_archive_path": second_row["candidate_archive_path"],
                "fp0_archive_sha256": first_row["candidate_archive_sha256"],
                "fp4_archive_sha256": second_row["candidate_archive_sha256"],
                "archive_file_sha256_recomputed": True,
                "candidate_scaled_hash_equal": scaled_hash_equal,
                "candidate_physical_hash_equal": physical_hash_equal,
                "candidate_physical_hash_required_for_pass": False,
                "candidate_pair_hashes_recomputed": True,
                "candidate_scaled_bitwise_exact": scaled_exact,
                "persistence_scaled_max_abs_diff": scaled_p_max,
                "corrected_scaled_max_abs_diff": scaled_c_max,
                "persistence_physical_max_abs_diff": physical_p_max,
                "corrected_physical_max_abs_diff": physical_c_max,
                "physical_tolerance": physical_limit,
                "oracle_label_exact": oracle_equal,
                "oracle_valid_mask_exact": oracle_valid_mask_equal,
                "oracle_valid_point_count": int(first_oracle_valid.sum()),
                "oracle_excluded_nonfinite_count": int(
                    first_oracle_valid.size - first_oracle_valid.sum()
                ),
                "oracle_elementwise_finite_masked": True,
                "source_model_sha256_equal": bool(
                    first_row["source_model_sha256"]
                    == second_row["source_model_sha256"]
                ),
                "source_artifact_sha256_equal": bool(
                    first_row["source_artifact_sha256"]
                    == second_row["source_artifact_sha256"]
                ),
                "frozen_weights_sha256_equal": bool(
                    first_row["frozen_weights_sha256_after_training"]
                    == second_row["frozen_weights_sha256_after_training"]
                ),
                "candidate_invariance_pass": bool(
                    scaled_exact
                    and scaled_hash_equal
                    and scaled_within
                    and physical_within
                    and oracle_valid_mask_equal
                    and oracle_equal
                ),
                "control_interpretation": "Frozen-Pair control: only C features and gate parameters may differ",
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != len(probe_summary["farm_id"].unique()) or not result["candidate_invariance_pass"].all():
        raise ValueError("Frozen-Pair候选一致性验收失败")
    return result


def _binary_ece(probability, truth, bins=10):
    probability = np.asarray(probability, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    valid = np.isfinite(probability) & np.isfinite(truth)
    probability = probability[valid]
    truth = truth[valid]
    if not len(probability):
        return np.nan
    ids = np.minimum((np.clip(probability, 0, 1) * bins).astype(int), bins - 1)
    error = 0.0
    for bin_id in range(bins):
        mask = ids == bin_id
        if mask.any():
            error += mask.mean() * abs(probability[mask].mean() - truth[mask].mean())
    return float(error)


def _probe_gate_utility_rows(probe_summary):
    rows = []
    for _, row in probe_summary.iterrows():
        archive = _load_archive(row["candidate_archive_path"])
        gate = np.asarray(archive["gate"], dtype=float)
        truth = np.asarray(archive["y_true"], dtype=float)
        persistence = np.asarray(archive["persistence"], dtype=float)
        corrected = np.asarray(archive["corrected"], dtype=float)
        fused = np.asarray(archive["fused"], dtype=float)
        capacity = float(archive["capacity"])
        valid = _gate_oracle_valid_mask(
            gate,
            truth,
            persistence,
            corrected,
            fused,
        )
        if not valid.any():
            raise ValueError(
                f"{row['model_variant']}/{row['farm_id']}没有可用的门控评价点"
            )
        flat_gate = gate[valid]
        flat_truth = truth[valid]
        flat_persistence = persistence[valid]
        flat_corrected = corrected[valid]
        flat_fused = fused[valid]
        flat_oracle = (
            np.square(flat_corrected - flat_truth)
            < np.square(flat_persistence - flat_truth)
        ).astype(int)
        try:
            auroc = float(roc_auc_score(flat_oracle, flat_gate))
            auprc = float(average_precision_score(flat_oracle, flat_gate))
            balanced = float(balanced_accuracy_score(flat_oracle, flat_gate >= 0.5))
        except ValueError:
            auroc = auprc = balanced = np.nan
        utility_gap = (
            float(flat_gate[flat_oracle == 1].mean() - flat_gate[flat_oracle == 0].mean())
            if np.unique(flat_oracle).size == 2
            else np.nan
        )
        regret = np.maximum(
            0.0,
            (
                np.abs(flat_fused - flat_truth)
                - np.abs(flat_persistence - flat_truth)
            )
            / capacity,
        )
        rows.append(
            {
                "model_variant": row["model_variant"],
                "farm_id": str(row["farm_id"]),
                "candidate_fixed": True,
                "total_point_count": int(valid.size),
                "valid_point_count": int(valid.sum()),
                "excluded_nonfinite_count": int(valid.size - valid.sum()),
                "elementwise_finite_masked": True,
                "gate_mean": float(flat_gate.mean()),
                "gate_std": float(flat_gate.std()),
                "gate_p10": float(np.quantile(flat_gate, 0.10)),
                "gate_p50": float(np.quantile(flat_gate, 0.50)),
                "gate_p90": float(np.quantile(flat_gate, 0.90)),
                "gate_high_saturation_rate": float((flat_gate > 0.95).mean()),
                "gate_low_saturation_rate": float((flat_gate < 0.05).mean()),
                "corrected_better_prevalence": float(flat_oracle.mean()),
                "oracle_brier": float(
                    np.mean(np.square(flat_gate - flat_oracle.astype(float)))
                ),
                "ece_10bin": _binary_ece(flat_gate, flat_oracle),
                "auroc": auroc,
                "auprc": auprc,
                "balanced_accuracy": balanced,
                "utility_gap": utility_gap,
                "positive_regret_mean": float(regret.mean()),
                "harm_rate_0_005": float((regret > 0.005).mean()),
                "positive_regret_p95": float(np.quantile(regret, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def save_probe_outputs(probe_results, full_pair):
    output_dir = probe_output_dir()
    combined = _concat_results(probe_results)
    variants = sorted(combined["summary"]["model_variant"].unique())
    suffix = _partial_suffix(combined["summary"], variants, full_pair)
    paths = {}
    invariance = utility = comparison = None
    if full_pair:
        if "valid_count" not in combined["summary"]:
            raise KeyError("Frozen-Pair正式summary缺少valid_count")
        expected_farm_set = set(expected_test_farm_ids())
        expected_farms = len(expected_farm_set)
        expected_rows = len(PROBE_VARIANTS) * expected_farms
        _validate_result_matrix_structure(
            combined,
            PROBE_VARIANTS,
            expected_farm_set,
        )
        summary_metric = pd.to_numeric(
            combined["summary"][SELECTION_METRIC], errors="coerce"
        )
        valid_count = pd.to_numeric(
            combined["summary"].get("valid_count"), errors="coerce"
        )
        if (
            len(combined["summary"]) != expected_rows
            or combined["summary"].duplicated(
                ["model_variant", "farm_id"]
            ).any()
            or set(combined["summary"]["model_variant"])
            != set(PROBE_VARIANTS)
            or combined["summary"]["farm_id"].nunique() != expected_farms
            or set(combined["summary"]["farm_id"].astype(str))
            != expected_farm_set
            or not np.isfinite(summary_metric).all()
            or not np.isfinite(valid_count).all()
            or not (valid_count > 0).all()
            or not (
                combined["summary"]
                .assign(_valid_count=valid_count)
                .groupby("farm_id")["_valid_count"]
                .nunique()
                == 1
            ).all()
        ):
            raise ValueError(
                f"Frozen-Pair summary必须是{expected_rows}个唯一、有限且"
                "真值有效数一致的variant/farm键"
            )
        raw_horizon = combined["horizon"]
        expected_raw_horizon_rows = expected_rows * (
            regime_train.FORECAST_LEN + 1
        )
        if (
            len(raw_horizon) != expected_raw_horizon_rows
            or raw_horizon.duplicated(
                ["model_variant", "farm_id", "horizon_step"]
            ).any()
            or not np.isfinite(
                pd.to_numeric(
                    raw_horizon[SELECTION_METRIC], errors="coerce"
                )
            ).all()
        ):
            raise ValueError(
                "Frozen-Pair逐场站horizon主指标不完整、重复或非有限"
            )
        # 候选一致性、真值与有限样本门控指标全部先验收，随后才发布正式文件。
        validate_truth_alignment(combined["summary"], PROBE_VARIANTS)
        invariance = validate_frozen_pair_archives(combined["summary"])
        utility = _probe_gate_utility_rows(combined["summary"])

    for key in RESULT_KEYS:
        path = os.path.join(
            output_dir,
            f"feature_screening_frozen_pair_{key}{suffix}.csv",
        )
        _atomic_to_csv(combined[key], path)
        paths[key] = path
    if not full_pair:
        note = os.path.join(
            output_dir,
            f"feature_screening_frozen_pair_partial_note{suffix}.md",
        )
        _atomic_write_text(
            note,
            "# Frozen-Pair部分运行\n\n"
            "FP0/FP4或全部场站未齐，不生成C归因结论。\n",
        )
        paths["partial_note"] = note
        return combined, paths

    comparison = (
        combined["summary"]
        .groupby("model_variant", as_index=False)
        .agg(
            farm_count=("farm_id", "nunique"),
            macro_test_nrmse=(SELECTION_METRIC, "mean"),
            macro_test_nrmse_std=(SELECTION_METRIC, lambda x: x.std(ddof=0)),
            macro_gate_mean=("gate_mean", "mean"),
            macro_gate_brier=("gate_oracle_brier", "mean"),
            total_params=("parameter_count", "max"),
            trainable_params=("trainable_parameter_count", "max"),
            frozen_params=("frozen_parameter_count", "max"),
        )
    )
    utility_macro = utility.groupby("model_variant", as_index=False).mean(numeric_only=True)
    comparison = comparison.merge(utility_macro, on="model_variant", how="left", suffixes=("", "_utility"))
    table_paths = {
        "candidate_invariance": os.path.join(output_dir, "feature_screening_frozen_pair_candidate_invariance.csv"),
        "gate_utility": os.path.join(output_dir, "feature_screening_frozen_pair_gate_utility_by_farm.csv"),
        "comparison": os.path.join(output_dir, "feature_screening_frozen_pair_comparison.csv"),
    }
    _atomic_to_csv(invariance, table_paths["candidate_invariance"])
    _atomic_to_csv(utility, table_paths["gate_utility"])
    _atomic_to_csv(comparison, table_paths["comparison"])
    report = os.path.join(output_dir, "feature_screening_frozen_pair_control_report.md")
    report_text = (
        "# FP0/FP4 Frozen-Pair control\n\n"
        "FP0与FP4使用同一个冻结B2 Persistence/corrected pair。"
        "候选一致性是归因的先决条件；FP不参与F0--F8最终选型。\n\n"
        "## 候选一致性\n\n"
        f"{invariance.to_markdown(index=False)}\n\n"
        "## 预测、门控和复杂度\n\n"
        f"{comparison.to_markdown(index=False)}\n\n"
        "只有在上表candidate_invariance_pass全部为True时，"
        "FP4−FP0才能解释为C对门控的直接条件贡献。\n"
    )
    _atomic_write_text(report, report_text)
    paths.update(table_paths)
    paths["probe_report"] = report
    combined["invariance"] = invariance
    combined["utility"] = utility
    combined["probe_comparison"] = comparison
    return combined, paths


def _history_epoch_count(path):
    resolved = _resolve_existing_path(path)
    if not resolved:
        return np.nan
    try:
        return int(len(pd.read_csv(resolved)))
    except Exception:
        return np.nan


def build_complexity_report(*summary_frames):
    summary = pd.concat(
        [
            frame
            for frame in summary_frames
            if frame is not None and not frame.empty
        ],
        ignore_index=True,
        sort=False,
    )
    if summary.empty:
        raise ValueError("复杂度报告没有模型summary")
    summary["farm_id"] = summary["farm_id"].astype(str)
    if summary.duplicated(["model_variant", "farm_id"]).any():
        raise ValueError("复杂度报告包含重复variant/farm键")
    expected_farms = len(expected_test_farm_ids())
    farm_counts = summary.groupby("model_variant")["farm_id"].nunique()
    variant_farm_sets = summary.groupby("model_variant")["farm_id"].agg(set)
    if (
        not (farm_counts == expected_farms).all()
        or not all(
            value == set(expected_test_farm_ids())
            for value in variant_farm_sets
        )
    ):
        raise ValueError(
            "复杂度正式报告要求每个模型覆盖全部测试场站: "
            f"{farm_counts.to_dict()}"
        )
    rows = []
    for _, row in summary.iterrows():
        artifact_path = _resolve_existing_path(row.get("artifact_path"))
        if artifact_path is None:
            raise FileNotFoundError(
                f"复杂度报告缺少artifact: {row['model_variant']}/{row['farm_id']}"
            )
        try:
            loaded = joblib.load(artifact_path)
        except Exception as error:
            raise RuntimeError(
                "复杂度报告无法读取artifact: "
                f"{row['model_variant']}/{row['farm_id']} / {artifact_path}"
            ) from error
        if not isinstance(loaded, dict):
            raise TypeError(f"复杂度artifact不是dict: {artifact_path}")
        artifact = loaded
        artifact_model_path = _resolve_existing_path(artifact.get("model_path"))
        loaded_inference_path = _resolve_existing_path(
            row.get("loaded_model_path")
        )
        model_path = artifact_model_path
        if model_path is None and (
            loaded_inference_path is not None
            and loaded_inference_path.lower().endswith(".keras")
        ):
            model_path = loaded_inference_path
        if model_path is None:
            raise FileNotFoundError(
                "复杂度报告缺少.keras模型归档: "
                f"{row['model_variant']}/{row['farm_id']}"
            )
        weights_path = _resolve_existing_path(artifact.get("best_weights_path"))
        total_value = pd.to_numeric(
            _first_present(row.get("parameter_count"), artifact.get("total_params")),
            errors="coerce",
        )
        trainable_value = pd.to_numeric(
            _first_present(
                row.get("trainable_parameter_count"),
                artifact.get("trainable_params"),
                total_value,
            ),
            errors="coerce",
        )
        if not np.isfinite(total_value) or not np.isfinite(trainable_value):
            raise ValueError(
                f"复杂度参数量无效: {row['model_variant']}/{row['farm_id']}"
            )
        total = int(total_value)
        trainable = int(trainable_value)
        role = _first_present(row.get("experiment_role"), "feature_candidate")
        rows.append(
            {
                "model_variant": row["model_variant"],
                "farm_id": str(row["farm_id"]),
                "experiment_role": role,
                "feature_count": row.get("feature_count", np.nan),
                "total_params": total,
                "trainable_params": trainable,
                "frozen_params": total - trainable,
                "frozen_fraction": (total - trainable) / total if total else np.nan,
                "parameter_storage_bytes_float32": total * np.dtype(np.float32).itemsize,
                "trainable_parameter_storage_bytes_float32": (
                    trainable * np.dtype(np.float32).itemsize
                ),
                "keras_archive_size_bytes": os.path.getsize(model_path),
                "keras_archive_size_scope": (
                    ".keras archive; may include compile/optimizer metadata; "
                    "not deployment memory"
                ),
                "inference_weights_file_size_bytes": (
                    os.path.getsize(weights_path) if weights_path else np.nan
                ),
                "training_elapsed_seconds": artifact.get("training_elapsed_seconds", np.nan),
                "actual_epoch_count": _history_epoch_count(artifact.get("history_path")),
                "test_macro_component_nrmse": row.get(SELECTION_METRIC, np.nan),
                "artifact_path": artifact_path,
                "model_path": model_path,
                "loaded_inference_path": loaded_inference_path,
                "artifact_load_status": "validated",
                "artifact_load_error": "",
                "measurement_scope": (
                    "artifact/filesystem/parameter count; latency, FLOPs, "
                    "throughput and peak memory not uniformly rebenchmarked"
                ),
                "runtime_benchmark_status": "not_measured_in_uniform_environment",
                "batch1_latency_p50_ms": np.nan,
                "batch1_latency_p95_ms": np.nan,
                "batch192_throughput_samples_per_second": np.nan,
                "macs": np.nan,
                "flops": np.nan,
                "peak_gpu_memory_bytes": np.nan,
                "python_version": platform.python_version(),
                "tensorflow_version": tf.__version__,
            }
        )
    by_farm = pd.DataFrame(rows).drop_duplicates(["model_variant", "farm_id"])
    macro = (
        by_farm.groupby("model_variant", as_index=False)
        .agg(
            farm_count=("farm_id", "nunique"),
            feature_count=("feature_count", "max"),
            total_params=("total_params", "max"),
            trainable_params=("trainable_params", "max"),
            frozen_params=("frozen_params", "max"),
            parameter_storage_bytes_float32=(
                "parameter_storage_bytes_float32",
                "max",
            ),
            trainable_parameter_storage_bytes_float32=(
                "trainable_parameter_storage_bytes_float32",
                "max",
            ),
            keras_archive_size_bytes_mean=("keras_archive_size_bytes", "mean"),
            inference_weights_file_size_bytes_mean=(
                "inference_weights_file_size_bytes",
                "mean",
            ),
            training_elapsed_seconds_mean=("training_elapsed_seconds", "mean"),
            actual_epoch_count_mean=("actual_epoch_count", "mean"),
            macro_test_nrmse=("test_macro_component_nrmse", "mean"),
            cross_farm_test_nrmse_std=(
                "test_macro_component_nrmse",
                lambda x: pd.to_numeric(x, errors="coerce").std(ddof=0),
            ),
            cross_farm_test_nrmse_min=("test_macro_component_nrmse", "min"),
            cross_farm_test_nrmse_max=("test_macro_component_nrmse", "max"),
        )
    )
    macro["cross_farm_test_nrmse_range"] = (
        macro["cross_farm_test_nrmse_max"]
        - macro["cross_farm_test_nrmse_min"]
    )
    macro["stability_scope"] = (
        "single-seed spatial stability across five farms; not multi-seed stability"
    )
    macro["runtime_benchmark_status"] = "not_measured_in_uniform_environment"
    return by_farm, macro


def _stage1_b2_test_frames():
    source_variant = "b2_persistence_residual"
    model_name = regime_train.stage1_variant_model_name(source_variant)
    root = os.path.join(
        regime_train.stage1_variant_dirs(source_variant, create=False)["root"],
        "testdata_predict_output",
    )
    summary_path = os.path.join(root, f"{model_name}_test_metrics_summary.csv")
    horizon_path = os.path.join(root, f"{model_name}_test_metrics_by_horizon_all.csv")
    if not os.path.exists(summary_path) or not os.path.exists(horizon_path):
        raise FileNotFoundError("candidate drift报告缺少Stage-1 B2测试指标")
    return _read_csv_with_farm_id(summary_path), _read_csv_with_farm_id(horizon_path)


def _validate_b2_truth_against_f4(b2_summary):
    legacy_summary = _read_csv_with_farm_id(
        os.path.join(
            legacy_output_dir(),
            LEGACY_AGGREGATE_FILES["summary"],
        )
    )
    f4 = legacy_summary[
        legacy_summary["model_variant"].astype(str) == "f4"
    ].set_index("farm_id")
    b2 = b2_summary.set_index("farm_id")
    expected = set(expected_test_farm_ids())
    if set(f4.index) != expected or set(b2.index) != expected:
        raise ValueError("B2/F4真值对齐未覆盖锁定的5个场站")
    for farm_id in sorted(expected):
        b2_truth = _load_prediction_truth(b2.loc[farm_id, "prediction_path"])
        f4_row = f4.loc[farm_id]
        f4_truth = _load_prediction_truth(
            _first_present(
                f4_row.get("source_prediction_path"),
                f4_row.get("prediction_path"),
            )
        )
        if (
            len(b2_truth) != len(f4_truth)
            or not np.array_equal(
                b2_truth[["sample_id", "horizon_step"]].to_numpy(),
                f4_truth[["sample_id", "horizon_step"]].to_numpy(),
            )
            or not np.allclose(
                pd.to_numeric(b2_truth["actual_power"], errors="coerce"),
                pd.to_numeric(f4_truth["actual_power"], errors="coerce"),
                rtol=0.0,
                atol=1e-7,
                equal_nan=True,
            )
        ):
            raise ValueError(f"Stage-1 B2与F4测试窗口/真值不一致: {farm_id}")
    return True


def build_candidate_drift_reports(selection_combined):
    expected_farms = len(expected_test_farm_ids())
    expected_overall = len(SELECTION_VARIANTS) * expected_farms
    candidate = selection_combined["candidate"].copy()
    overall = candidate[
        candidate["model_variant"].isin(SELECTION_VARIANTS)
        & (candidate["candidate"] == "corrected")
        & (candidate["horizon_step"].astype(str) == "all")
    ].copy()
    overall[SELECTION_METRIC] = pd.to_numeric(overall[SELECTION_METRIC], errors="coerce")
    if (
        len(overall) != expected_overall
        or overall.duplicated(["model_variant", "farm_id"]).any()
        or not np.isfinite(overall[SELECTION_METRIC]).all()
    ):
        raise ValueError(
            "corrected candidate总体指标必须完整覆盖9个F变体×全部场站"
        )
    b2_summary, b2_horizon = _stage1_b2_test_frames()
    b2_truth_alignment_pass = _validate_b2_truth_against_f4(b2_summary)
    reference = b2_summary[["farm_id", SELECTION_METRIC]].rename(columns={SELECTION_METRIC: "b2_corrected_nrmse"})
    reference["b2_corrected_nrmse"] = pd.to_numeric(
        reference["b2_corrected_nrmse"], errors="coerce"
    )
    if (
        len(reference) != expected_farms
        or reference.duplicated(["farm_id"]).any()
        or reference["farm_id"].nunique() != expected_farms
        or set(reference["farm_id"].astype(str))
        != set(expected_test_farm_ids())
        or not np.isfinite(reference["b2_corrected_nrmse"]).all()
    ):
        raise ValueError("Stage-1 B2 summary未按每场站唯一覆盖")
    fused = selection_combined["summary"][["model_variant", "farm_id", SELECTION_METRIC]].rename(columns={SELECTION_METRIC: "fused_nrmse"})
    fused["fused_nrmse"] = pd.to_numeric(
        fused["fused_nrmse"], errors="coerce"
    )
    if (
        len(fused) != expected_overall
        or fused.duplicated(["model_variant", "farm_id"]).any()
        or not np.isfinite(fused["fused_nrmse"]).all()
    ):
        raise ValueError("F0--F8 fused summary键不完整或重复")
    by_farm = (
        overall[["model_variant", "farm_id", SELECTION_METRIC]]
        .rename(columns={SELECTION_METRIC: "corrected_nrmse"})
        .merge(reference, on="farm_id", validate="many_to_one")
        .merge(fused, on=["model_variant", "farm_id"], validate="one_to_one")
    )
    if len(by_farm) != expected_overall:
        raise ValueError("candidate drift总体指标合并后覆盖不完整")
    by_farm["candidate_delta_vs_b2"] = by_farm["corrected_nrmse"] - by_farm["b2_corrected_nrmse"]
    by_farm["candidate_relative_delta_vs_b2"] = by_farm["corrected_nrmse"] / by_farm["b2_corrected_nrmse"] - 1.0
    by_farm["fused_minus_corrected_nrmse"] = by_farm["fused_nrmse"] - by_farm["corrected_nrmse"]
    by_farm["candidate_fixed"] = False
    by_farm["b2_f4_test_truth_alignment_pass"] = b2_truth_alignment_pass
    macro = (
        by_farm.groupby("model_variant", as_index=False)
        .agg(
            farm_count=("farm_id", "nunique"),
            corrected_macro_nrmse=("corrected_nrmse", "mean"),
            b2_macro_nrmse=("b2_corrected_nrmse", "mean"),
            candidate_delta_vs_b2=("candidate_delta_vs_b2", "mean"),
            candidate_relative_delta_vs_b2=("candidate_relative_delta_vs_b2", "mean"),
            fused_macro_nrmse=("fused_nrmse", "mean"),
            fused_minus_corrected_nrmse=("fused_minus_corrected_nrmse", "mean"),
        )
    )
    candidate_range = float(macro["corrected_macro_nrmse"].max() - macro["corrected_macro_nrmse"].min())
    relative_range = candidate_range / float(macro["corrected_macro_nrmse"].min())
    macro["f0_f8_corrected_range"] = candidate_range
    macro["f0_f8_corrected_relative_range"] = relative_range
    macro["aggregate_candidate_metric_drift_within_0_2pct"] = (
        relative_range <= DRIFT_RELATIVE_LIMIT
    )
    macro["assessment_scope"] = (
        "aggregate corrected-candidate NRMSE drift; not pointwise equality"
    )
    macro["pointwise_candidate_equality_tested"] = False
    macro["independent_feature_attribution_allowed"] = False

    corrected_h = candidate[
        candidate["model_variant"].isin(SELECTION_VARIANTS)
        & (candidate["candidate"] == "corrected")
        & (candidate["horizon_step"].astype(str) != "all")
    ].copy()
    corrected_h["horizon_step"] = pd.to_numeric(corrected_h["horizon_step"])
    corrected_h[SELECTION_METRIC] = pd.to_numeric(
        corrected_h[SELECTION_METRIC], errors="coerce"
    )
    expected_corrected_h = expected_overall * regime_train.FORECAST_LEN
    if (
        len(corrected_h) != expected_corrected_h
        or corrected_h.duplicated(
            ["model_variant", "farm_id", "horizon_step"]
        ).any()
        or not np.isfinite(corrected_h[SELECTION_METRIC]).all()
    ):
        raise ValueError(
            "corrected candidate逐时域指标必须完整覆盖9×场站×16步"
        )
    b2_h = b2_horizon[b2_horizon["horizon_step"].astype(str) != "all"].copy()
    b2_h["horizon_step"] = pd.to_numeric(b2_h["horizon_step"])
    b2_h = b2_h[["farm_id", "horizon_step", SELECTION_METRIC]].rename(columns={SELECTION_METRIC: "b2_corrected_nrmse"})
    b2_h["b2_corrected_nrmse"] = pd.to_numeric(
        b2_h["b2_corrected_nrmse"], errors="coerce"
    )
    expected_b2_h = expected_farms * regime_train.FORECAST_LEN
    if (
        len(b2_h) != expected_b2_h
        or b2_h.duplicated(["farm_id", "horizon_step"]).any()
        or not np.isfinite(b2_h["b2_corrected_nrmse"]).all()
    ):
        raise ValueError("Stage-1 B2逐时域指标未完整覆盖场站×16步")
    horizon = (
        corrected_h[[
            "model_variant",
            "farm_id",
            "horizon_step",
            SELECTION_METRIC,
        ]]
        .rename(columns={SELECTION_METRIC: "corrected_nrmse"})
        .merge(
            b2_h,
            on=["farm_id", "horizon_step"],
            validate="many_to_one",
        )
    )
    if len(horizon) != expected_corrected_h:
        raise ValueError("candidate drift逐时域指标合并后覆盖不完整")
    horizon["candidate_delta_vs_b2"] = horizon["corrected_nrmse"] - horizon["b2_corrected_nrmse"]
    horizon = horizon.groupby(
        ["model_variant", "horizon_step"], as_index=False
    ).agg(
        corrected_nrmse=("corrected_nrmse", "mean"),
        b2_corrected_nrmse=("b2_corrected_nrmse", "mean"),
        candidate_delta_vs_b2=("candidate_delta_vs_b2", "mean"),
        farm_count=("farm_id", "nunique"),
    )
    if (
        len(horizon) != len(SELECTION_VARIANTS) * regime_train.FORECAST_LEN
        or horizon.duplicated(["model_variant", "horizon_step"]).any()
        or not (horizon["farm_count"] == expected_farms).all()
        or not np.isfinite(
            horizon[
                [
                    "corrected_nrmse",
                    "b2_corrected_nrmse",
                    "candidate_delta_vs_b2",
                ]
            ].to_numpy(dtype=float)
        ).all()
    ):
        raise ValueError("candidate drift逐时域宏平均覆盖不完整")
    horizon["assessment_scope"] = (
        "aggregate horizon NRMSE drift; not pointwise candidate equality"
    )
    return by_farm, macro, horizon


def build_gate_interpretation_reports(selection_combined, probe_combined=None):
    summary_frames = [selection_combined["summary"]]
    gate_frames = [selection_combined["gate"]]
    if probe_combined is not None and not probe_combined["summary"].empty:
        summary_frames.append(probe_combined["summary"])
        gate_frames.append(probe_combined["gate"])
    summary = pd.concat(summary_frames, ignore_index=True, sort=False)
    gate = pd.concat(gate_frames, ignore_index=True, sort=False)
    summary_masked = summary.get(
        "gate_oracle_elementwise_finite_masked",
        pd.Series(False, index=summary.index),
    ).fillna(False).astype(bool)
    summary["gate_oracle_metric_protocol"] = np.where(
        summary_masked,
        "elementwise_finite_masked_v2",
        "legacy_unmasked_nonfinite_targets_excluded_from_oracle_comparison",
    )
    summary["gate_oracle_metric_comparable"] = summary_masked
    for column in ("gate_oracle_choice_accuracy", "gate_oracle_brier"):
        if column in summary:
            summary.loc[~summary_masked, column] = np.nan

    gate_masked = gate.get(
        "elementwise_finite_masked",
        pd.Series(False, index=gate.index),
    ).fillna(False).astype(bool)
    gate["gate_oracle_metric_protocol"] = np.where(
        gate_masked,
        "elementwise_finite_masked_v2",
        "legacy_unmasked_nonfinite_targets_excluded_from_oracle_comparison",
    )
    gate["gate_oracle_metric_comparable"] = gate_masked
    target_dependent_gate_columns = (
        "corrected_better_rate",
        "gate_hard_choice_accuracy",
        "gate_oracle_brier",
        "fused_mse",
        "oracle_mse",
        "oracle_regret",
        "captured_oracle_gain",
    )
    for column in target_dependent_gate_columns:
        if column in gate:
            gate.loc[~gate_masked, column] = np.nan
    overall_columns = [
        "model_variant",
        "farm_id",
        SELECTION_METRIC,
        "gate_mean",
        "gate_std",
        "gate_sample_variation",
        "gate_binary_entropy",
        "gate_saturation_low_rate",
        "gate_saturation_high_rate",
        "gate_oracle_choice_accuracy",
        "gate_oracle_brier",
        "gate_oracle_metric_protocol",
        "gate_oracle_metric_comparable",
    ]
    overall = summary[[column for column in overall_columns if column in summary]].copy()
    overall["candidate_fixed"] = overall["model_variant"].isin(PROBE_VARIANTS)
    overall["gate_distribution_metrics_comparable"] = True
    numeric_gate_columns = [
        column
        for column in (
            "gate_mean",
            "gate_std",
            "gate_p10",
            "gate_p50",
            "gate_p90",
            "gate_low_saturation_rate",
            "gate_high_saturation_rate",
            "gate_binary_entropy",
            "corrected_better_rate",
            "gate_hard_choice_accuracy",
            "gate_oracle_brier",
            "oracle_regret",
        )
        if column in gate
    ]
    by_regime = gate.groupby(["model_variant", "regime_group"], as_index=False)[numeric_gate_columns].mean()
    all_gate = gate[gate["regime_group"] == "all"].copy()
    by_horizon = all_gate.groupby(["model_variant", "horizon_step"], as_index=False)[numeric_gate_columns].mean()
    regime_gate = by_regime.pivot(index="model_variant", columns="regime_group", values="gate_mean")
    gaps = pd.DataFrame({"model_variant": regime_gate.index})
    gaps["dynamic_minus_stable_gate"] = regime_gate.get("dynamic", np.nan).to_numpy() - regime_gate.get("stable", np.nan).to_numpy()
    gaps["ramp_up_minus_stable_gate"] = regime_gate.get("ramp_up", np.nan).to_numpy() - regime_gate.get("stable", np.nan).to_numpy()
    gaps["ramp_down_minus_stable_gate"] = regime_gate.get("ramp_down", np.nan).to_numpy() - regime_gate.get("stable", np.nan).to_numpy()

    pairs = (
        ("f0", "f1", "H effect"),
        ("f1", "f7", "D effect without M/C"),
        ("f7", "f8", "C end-to-end effect without M"),
        ("f8", "f4", "M effect with C"),
        ("fp0", "fp4", "C direct gate effect under Frozen-Pair control"),
    )
    effect_rows = []
    for source_id, target_id, description in pairs:
        source = overall[overall["model_variant"] == source_id]
        target = overall[overall["model_variant"] == target_id]
        if source.empty or target.empty:
            continue
        paired = source.merge(target, on="farm_id", suffixes=("_source", "_target"))
        oracle_comparable = bool(
            len(paired)
            and paired.get(
                "gate_oracle_metric_comparable_source",
                pd.Series(False, index=paired.index),
            ).fillna(False).all()
            and paired.get(
                "gate_oracle_metric_comparable_target",
                pd.Series(False, index=paired.index),
            ).fillna(False).all()
        )
        row = {
            "comparison": f"{source_id}_to_{target_id}",
            "description": description,
            "source_variant": source_id,
            "target_variant": target_id,
            "candidate_fixed": source_id in PROBE_VARIANTS and target_id in PROBE_VARIANTS,
            "paired_farm_count": len(paired),
            "gate_distribution_metrics_comparable": True,
            "gate_oracle_metric_comparable": oracle_comparable,
            "gate_oracle_metric_note": (
                "both elementwise-finite-masked"
                if oracle_comparable
                else "legacy oracle fields excluded; use FP0/FP4 for C calibration attribution"
            ),
        }
        for metric in (
            SELECTION_METRIC,
            "gate_mean",
            "gate_sample_variation",
            "gate_saturation_high_rate",
            "gate_oracle_brier",
        ):
            if f"{metric}_source" not in paired:
                continue
            if metric == "gate_oracle_brier" and not oracle_comparable:
                row[f"delta_{metric}"] = np.nan
            else:
                row[f"delta_{metric}"] = float(
                    (
                        paired[f"{metric}_target"]
                        - paired[f"{metric}_source"]
                    ).mean()
                )
        effect_rows.append(row)
    return overall, by_regime, by_horizon, gaps, pd.DataFrame(effect_rows)


def _paired_model_delta(summary, source_id, target_id, metric):
    source = summary[summary["model_variant"] == source_id][["farm_id", metric]].rename(columns={metric: "source"})
    target = summary[summary["model_variant"] == target_id][["farm_id", metric]].rename(columns={metric: "target"})
    paired = source.merge(target, on="farm_id")
    delta = pd.to_numeric(paired["target"], errors="coerce") - pd.to_numeric(paired["source"], errors="coerce")
    source_macro = float(pd.to_numeric(paired["source"], errors="coerce").mean())
    target_macro = float(pd.to_numeric(paired["target"], errors="coerce").mean())
    return {
        "source_macro": source_macro,
        "target_macro": target_macro,
        "delta": target_macro - source_macro,
        "relative_change_pct": (target_macro / source_macro - 1.0) * 100.0,
        "farms_improved": int((delta < 0).sum()),
        "farms_degraded": int((delta > 0).sum()),
    }


def _paired_practical_assessment(
    frame,
    source_id,
    target_id,
    metric,
    *,
    lower_is_better,
    absolute_tolerance=None,
    relative_tolerance_pct=None,
):
    """Assess effect size and cross-farm consistency without a zero-only test."""
    expected_farm_ids = set(expected_test_farm_ids())
    expected_farm_count = len(expected_farm_ids)
    source = frame[frame["model_variant"] == source_id][
        ["farm_id", metric]
    ].rename(columns={metric: "source"})
    target = frame[frame["model_variant"] == target_id][
        ["farm_id", metric]
    ].rename(columns={metric: "target"})
    source["farm_id"] = source["farm_id"].astype(str)
    target["farm_id"] = target["farm_id"].astype(str)
    paired = source.merge(target, on="farm_id", validate="one_to_one")
    paired["source"] = pd.to_numeric(paired["source"], errors="coerce")
    paired["target"] = pd.to_numeric(paired["target"], errors="coerce")
    paired = paired[np.isfinite(paired["source"]) & np.isfinite(paired["target"])]
    complete_finite_coverage = bool(
        expected_farm_count
        and len(paired) == expected_farm_count
        and set(paired["farm_id"]) == expected_farm_ids
    )
    if paired.empty:
        return {
            "source_macro": np.nan,
            "target_macro": np.nan,
            "target_minus_source": np.nan,
            "relative_change_pct": np.nan,
            "practical_tolerance": np.nan,
            "paired_farm_count": 0,
            "expected_farm_count": expected_farm_count,
            "complete_finite_farm_coverage": False,
            "farms_practically_improved": 0,
            "farms_practically_degraded": 0,
            "practical_status": "not_evaluable",
        }
    if (absolute_tolerance is None) == (relative_tolerance_pct is None):
        raise ValueError("必须且只能指定absolute或relative实践容差")
    source_macro = float(paired["source"].mean())
    target_macro = float(paired["target"].mean())
    delta = paired["target"] - paired["source"]
    if relative_tolerance_pct is not None:
        macro_tolerance = abs(source_macro) * relative_tolerance_pct / 100.0
        per_farm_tolerance = (
            paired["source"].abs() * relative_tolerance_pct / 100.0
        )
        tolerance_label = (
            f"{relative_tolerance_pct:.6g}% relative to source"
        )
    else:
        macro_tolerance = float(absolute_tolerance)
        per_farm_tolerance = pd.Series(
            float(absolute_tolerance), index=paired.index
        )
        tolerance_label = f"{absolute_tolerance:.6g} absolute"
    signed_gain = -delta if lower_is_better else delta
    macro_signed_gain = (
        source_macro - target_macro if lower_is_better else target_macro - source_macro
    )
    improved = signed_gain > per_farm_tolerance
    degraded = signed_gain < -per_farm_tolerance
    if (
        macro_signed_gain > macro_tolerance
        and int(improved.sum()) >= MIN_PRACTICAL_FARM_CONSISTENCY
    ):
        status = "practical_improvement"
    elif (
        macro_signed_gain < -macro_tolerance
        and int(degraded.sum()) >= MIN_PRACTICAL_FARM_CONSISTENCY
    ):
        status = "practical_degradation"
    else:
        status = "no_clear_practical_effect"
    if not complete_finite_coverage:
        status = "not_evaluable"
    return {
        "source_macro": source_macro,
        "target_macro": target_macro,
        "target_minus_source": target_macro - source_macro,
        "relative_change_pct": (
            (target_macro / source_macro - 1.0) * 100.0
            if abs(source_macro) > 1e-15
            else np.nan
        ),
        "practical_tolerance": tolerance_label,
        "paired_farm_count": int(len(paired)),
        "expected_farm_count": expected_farm_count,
        "complete_finite_farm_coverage": complete_finite_coverage,
        "farms_practically_improved": int(improved.sum()),
        "farms_practically_degraded": int(degraded.sum()),
        "lower_is_better": bool(lower_is_better),
        "minimum_consistent_farms": MIN_PRACTICAL_FARM_CONSISTENCY,
        "practical_status": status,
    }


def save_analysis_reports(selection_combined, probe_combined=None):
    output_dir = analysis_output_dir()
    paths = {}
    complexity_by_farm, complexity_macro = build_complexity_report(
        selection_combined["summary"],
        probe_combined["summary"] if probe_combined is not None else None,
    )
    drift_by_farm, drift_macro, drift_horizon = build_candidate_drift_reports(selection_combined)
    gate_overall, gate_regime, gate_horizon, gate_gaps, gate_effects = build_gate_interpretation_reports(selection_combined, probe_combined)
    tables = {
        "complexity_by_farm": complexity_by_farm,
        "complexity_macro": complexity_macro,
        "candidate_drift_by_farm": drift_by_farm,
        "candidate_drift_macro": drift_macro,
        "candidate_drift_by_horizon": drift_horizon,
        "gate_overall": gate_overall,
        "gate_by_regime": gate_regime,
        "gate_by_horizon": gate_horizon,
        "gate_regime_gaps": gate_gaps,
        "gate_feature_effects": gate_effects,
    }
    if probe_combined is not None and "invariance" in probe_combined:
        tables["probe_candidate_invariance"] = probe_combined["invariance"]
        tables["probe_gate_utility"] = probe_combined["utility"]
    for name, frame in tables.items():
        path = os.path.join(output_dir, f"feature_screening_{name}.csv")
        _atomic_to_csv(frame, path)
        paths[name] = path

    f_accuracy = _paired_practical_assessment(
        selection_combined["summary"],
        "f7",
        "f8",
        SELECTION_METRIC,
        lower_is_better=True,
        relative_tolerance_pct=PRACTICAL_NRMSE_RELATIVE_PCT,
    )
    conclusion_rows = [
        {
            "axis": "fusion_accuracy",
            "comparison": "F7_to_F8_end_to_end_C_without_M",
            "source_variant": "f7",
            "target_variant": "f8",
            "metric": SELECTION_METRIC,
            "candidate_fixed": False,
            "evidence_role": "end_to_end_total_effect_candidate_drift_confounded",
            **f_accuracy,
        }
    ]
    probe_assessments = {}
    probe_complete = bool(
        probe_combined is not None
        and set(probe_combined["summary"]["model_variant"])
        == set(PROBE_VARIANTS)
        and "invariance" in probe_combined
        and probe_combined["invariance"]["candidate_invariance_pass"].all()
    )
    if probe_complete:
        assessment_specs = (
            (
                "fusion_accuracy",
                probe_combined["summary"],
                SELECTION_METRIC,
                True,
                None,
                PRACTICAL_NRMSE_RELATIVE_PCT,
                "direct_fusion_effect_with_bitwise_fixed_candidates",
            ),
            (
                "routing_calibration",
                probe_combined["utility"],
                "oracle_brier",
                True,
                CALIBRATION_ABSOLUTE_TOLERANCE,
                None,
                "finite_masked_oracle_calibration",
            ),
            (
                "routing_calibration",
                probe_combined["utility"],
                "ece_10bin",
                True,
                CALIBRATION_ABSOLUTE_TOLERANCE,
                None,
                "finite_masked_expected_calibration_error",
            ),
            (
                "routing_discrimination",
                probe_combined["utility"],
                "utility_gap",
                False,
                CALIBRATION_ABSOLUTE_TOLERANCE,
                None,
                "gate_separation_corrected_better_vs_worse",
            ),
            (
                "persistence_safety",
                probe_combined["utility"],
                "positive_regret_mean",
                True,
                SAFETY_REGRET_ABSOLUTE_TOLERANCE,
                None,
                "mean_positive_mae_regret_vs_persistence_normalized_by_capacity",
            ),
            (
                "persistence_safety",
                probe_combined["utility"],
                "harm_rate_0_005",
                True,
                SAFETY_HARM_RATE_ABSOLUTE_TOLERANCE,
                None,
                "rate_regret_exceeds_0.5pct_capacity",
            ),
        )
        for (
            axis,
            frame,
            metric,
            lower_is_better,
            absolute_tolerance,
            relative_tolerance_pct,
            evidence_role,
        ) in assessment_specs:
            assessment = _paired_practical_assessment(
                frame,
                "fp0",
                "fp4",
                metric,
                lower_is_better=lower_is_better,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance_pct=relative_tolerance_pct,
            )
            probe_assessments[metric] = assessment
            conclusion_rows.append(
                {
                    "axis": axis,
                    "comparison": f"FP0_to_FP4_{metric}",
                    "source_variant": "fp0",
                    "target_variant": "fp4",
                    "metric": metric,
                    "candidate_fixed": True,
                    "evidence_role": evidence_role,
                    **assessment,
                }
            )
    conclusion = pd.DataFrame(conclusion_rows)
    conclusion["single_seed"] = True
    conclusion["random_seed"] = RANDOM_SEED
    conclusion["evaluation_split"] = "legacy_seen_test"
    conclusion["final_blind_test"] = False
    conclusion_path = os.path.join(output_dir, "feature_screening_c_feature_conclusion.csv")
    _atomic_to_csv(conclusion, conclusion_path)
    paths["c_conclusion"] = conclusion_path

    report_path = os.path.join(output_dir, "feature_screening_c_feature_conclusion.md")
    status_cn = {
        "practical_improvement": "达到实践改善标准",
        "practical_degradation": "达到实践退化标准",
        "no_clear_practical_effect": "无明确实践效应",
        "not_evaluable": "不可评价",
    }
    report_lines = [
        "# C特征补充实验结论",
        "",
        "F8=P+H+D+C，与F7=P+H+D构成无M条件下的端到端C消融。"
        "NRMSE实践容差预设为相对0.05%，并要求全部5场站指标有限且至少"
        "4/5场站达到相应实践阈值。",
        (
            "FP分轴判定同样要求全部5场站有限且至少4/5场站达到实践阈值："
            "Brier/ECE/utility-gap的"
            f"绝对容差为{CALIBRATION_ABSOLUTE_TOLERANCE:g}，正向regret均值"
            f"容差为{SAFETY_REGRET_ABSOLUTE_TOLERANCE:g}，伤害率绝对容差为"
            f"{SAFETY_HARM_RATE_ABSOLUTE_TOLERANCE:g}。这些是单seed探索阶段的"
            "实践效应阈值，不是显著性检验。"
        ),
        "",
        "## 1. 融合精度",
        "",
        (
            f"F7→F8宏平均NRMSE变化 `{f_accuracy['target_minus_source']:+.9f}` "
            f"(`{f_accuracy['relative_change_pct']:+.4f}%`)；"
            f"{f_accuracy['farms_practically_improved']}/"
            f"{f_accuracy['expected_farm_count']}场站达到实践改善，"
            f"结论为“{status_cn[f_accuracy['practical_status']]}”。"
            "该比较包含候选网络联合微调，只表示端到端总效应。"
        ),
        "",
    ]
    if not probe_complete:
        report_lines.extend(
            [
                "FP0/FP4 Frozen-Pair control未完整通过，不能判断C的直接门控、"
                "校准和Persistence安全贡献。",
                "",
                "**当前判定：C在无M条件下是否无效仍待FP0/FP4完成后确定。**",
                "",
            ]
        )
    else:
        fp_accuracy = probe_assessments[SELECTION_METRIC]
        routing_metrics = ("oracle_brier", "ece_10bin", "utility_gap")
        safety_metrics = ("positive_regret_mean", "harm_rate_0_005")
        report_lines.extend(
            [
                (
                    "FP0=P+H+D门控，FP4=P+H+D+4个C特征门控；二者使用同一组"
                    "冻结B2 Persistence/corrected候选。名称中的0/4表示C特征数，"
                    "不是F0/F4模型复刻，且两者均不参与F0–F8排名。"
                ),
                "",
                (
                    f"FP0→FP4宏平均NRMSE变化 "
                    f"`{fp_accuracy['target_minus_source']:+.9f}` "
                    f"(`{fp_accuracy['relative_change_pct']:+.4f}%`)；"
                    f"{fp_accuracy['farms_practically_improved']}/"
                    f"{fp_accuracy['expected_farm_count']}场站达到实践改善，"
                    f"结论为“{status_cn[fp_accuracy['practical_status']]}”。"
                ),
                "",
                "## 2. 路由判别与校准",
                "",
            ]
        )
        routing_labels = {
            "oracle_brier": "Brier（越低越好）",
            "ece_10bin": "ECE-10（越低越好）",
            "utility_gap": "corrected优/劣样本门控间隔（越高越好）",
        }
        for metric in routing_metrics:
            item = probe_assessments[metric]
            report_lines.append(
                f"- {routing_labels[metric]}：FP4−FP0=`{item['target_minus_source']:+.6f}`，"
                f"{item['farms_practically_improved']}/"
                f"{item['expected_farm_count']}场站实践改善，"
                f"{status_cn[item['practical_status']]}。"
            )
        report_lines.extend(["", "## 3. Persistence安全性", ""])
        safety_labels = {
            "positive_regret_mean": "正向regret均值（越低越安全）",
            "harm_rate_0_005": "regret>0.5%容量的伤害率（越低越安全）",
        }
        for metric in safety_metrics:
            item = probe_assessments[metric]
            report_lines.append(
                f"- {safety_labels[metric]}：FP4−FP0=`{item['target_minus_source']:+.6f}`，"
                f"{item['farms_practically_improved']}/"
                f"{item['expected_farm_count']}场站实践改善，"
                f"{status_cn[item['practical_status']]}。"
            )

        direct_status = fp_accuracy["practical_status"]
        routing_statuses = [
            probe_assessments[item]["practical_status"]
            for item in routing_metrics
        ]
        safety_statuses = [
            probe_assessments[item]["practical_status"]
            for item in safety_metrics
        ]
        end_to_end_status = f_accuracy["practical_status"]
        if "practical_degradation" in safety_statuses:
            statement = (
                "C出现Persistence安全性退化，不能宣称安全保护价值；即使其他轴改善，"
                "也应保留这一风险结论。"
            )
        elif {direct_status, end_to_end_status} == {
            "practical_improvement",
            "practical_degradation",
        }:
            statement = (
                "固定候选贡献与F7→F8端到端总效应方向相反，属于混合证据。"
                "FP只说明条件门控贡献，不能覆盖F8联合训练后的相反结果；"
                "模型取舍仍以F0–F8正式排名为准。"
            )
        elif direct_status == "practical_degradation":
            statement = (
                "固定候选后C造成实践精度退化，不建议在无M结构中保留；"
                "其他轴即使改善，也不能掩盖这一融合精度代价。"
            )
        elif end_to_end_status == "practical_degradation":
            statement = (
                "F7→F8端到端总效应达到实践退化标准；即使FP或路由轴存在局部信号，"
                "也不能宣称C改善最终F8模型，模型取舍以正式排名为准。"
            )
        elif "practical_degradation" in routing_statuses:
            statement = (
                "路由判别/校准指标中存在实践退化，证据属于混合；"
                "必须与融合精度轴分别报告，不能概括为C单向有效。"
            )
        elif "not_evaluable" in [
            end_to_end_status,
            direct_status,
            *routing_statuses,
            *safety_statuses,
        ]:
            statement = (
                "至少一个预设评价轴未覆盖全部5个有限场站，证据不完整；"
                "只能逐轴报告，不能总体判定C在无M条件下有效或无效。"
            )
        elif (
            direct_status == "practical_improvement"
            and end_to_end_status == "practical_improvement"
        ):
            statement = (
                "固定候选贡献与F7→F8端到端总效应均达到实践改善标准，"
                "支持C在无M条件下有效。"
            )
        elif direct_status == "practical_improvement":
            statement = (
                "固定候选后C具有直接融合精度贡献，但F7→F8端到端总效应尚不明确；"
                "不能概括为最终F8模型已获益。"
            )
        elif end_to_end_status == "practical_improvement":
            statement = (
                "F7→F8端到端总效应改善，但Frozen-Pair未确认直接贡献；"
                "收益仍可能包含candidate drift/联合优化，不能作独立C归因。"
            )
        elif "practical_improvement" in routing_statuses:
            statement = (
                "C提供了可测的路由判别/校准信息，但尚未形成明确的融合精度收益；"
                "不能简单称为完全无效。"
            )
        elif "practical_improvement" in safety_statuses:
            statement = (
                "C未形成明确精度收益，但表现出Persistence安全改进，"
                "其价值应限定为安全轴。"
            )
        elif (
            direct_status == "no_clear_practical_effect"
            and end_to_end_status == "no_clear_practical_effect"
            and all(
                item == "no_clear_practical_effect"
                for item in routing_statuses + safety_statuses
            )
        ):
            statement = (
                "在当前单seed、legacy-seen测试协议和预设实践容差下，"
                "没有发现C在无M条件下的可辨识贡献，可将其视为本结构中的无效特征组。"
            )
        else:
            statement = "各评价轴证据混合，不能判定C在无M条件下有效或无效。"
        report_lines.extend(["", f"**分轴综合判定：** {statement}", ""])

    report_lines.extend(
        [
            "校准指标仅使用逐点有限真值掩码；旧F0–F7的legacy未掩码oracle字段"
            "不参与校准增量判断。上述属于已用于选型测试集上的seed=2026探索性结论，"
            "不是最终盲测；按本阶段要求不追加3-seed实验。",
            "",
        ]
    )
    _atomic_write_text(report_path, "\n".join(report_lines))
    paths["c_conclusion_report"] = report_path

    overview_path = os.path.join(output_dir, "feature_screening_complexity_drift_gate_report.md")
    drift_range = float(drift_macro["f0_f8_corrected_relative_range"].iloc[0])
    overview_text = (
        "# 复杂度、candidate drift与门控解释\n\n"
        f"F0--F8 corrected candidate宏平均NRMSE相对跨度为 "
        f"`{drift_range * 100:.4f}%`；0.2%聚合漂移阈值标记为 "
        f"`{'WITHIN' if drift_range <= DRIFT_RELATIVE_LIMIT else 'EXCEEDS'}`。"
        "这只是聚合性能漂移，不是候选逐点等价检验，也不授权独立特征归因。\n\n"
        "联合训练F模型的门控变化可能同时包含candidate drift。FP0/FP4通过"
        "标准化候选位级一致、物理候选容差一致及有限oracle一致性验收；只有该表"
        "全部通过时，FP4−FP0才用于C的直接门控归因。\n\n"
        "门控分布（均值、饱和率等）可跨旧模型描述；旧F0–F7的oracle校准字段因"
        "未逐点屏蔽缺失真值而从比较报告中置空，校准与安全结论只读取新FP有限掩码指标。\n\n"
        "复杂度表报告参数量、float32理论参数字节、.keras归档和独立权重文件大小；"
        ".keras可能包含编译/优化器元数据，不等同部署内存。跨场站NRMSE标准差/极差"
        "只表示seed=2026下的空间稳定性，不是多seed稳定性。延迟、FLOPs、吞吐和"
        "峰值显存未在统一硬件重新测量，因此明确留空而不补造数值。\n"
        "当前F8/FP所用5个原始test CSV的SHA256写入bundle marker；旧F0–F7运行"
        "当时未保存原始test CSV hash，旧侧只能通过sample/horizon/time/target逐点"
        "对齐追溯，这一输入版本限制已在marker中显式记录。\n"
    )
    _atomic_write_text(overview_path, overview_text)
    paths["analysis_overview"] = overview_path
    return paths


def main():
    configure_prediction_reproducibility()
    test_files = discover_requested_test_files()
    if not test_files:
        raise FileNotFoundError(
            f"未找到测试文件模式 {common_predict.TEST_FILE_PATTERN}"
        )
    new_variants = get_requested_prediction_variants()
    actual_farm_ids = {
        str(common_predict.get_farm_id(path)) for path in test_files
    }
    all_farms = bool(
        not os.getenv("WIND_FEATURE_SCREEN_FARMS")
        and actual_farm_ids == set(expected_test_farm_ids())
    )
    if new_variants:
        # partial也会更新被正式summary引用的逐场站预测/NPZ，因此任何新推理
        # 都先撤销旧完成标志；只有整套报告重建后才重新发布complete marker。
        _clear_bundle_completion_marker()
    if all_farms and new_variants:
        training_marker = validate_training_bundle_completion()
        print(f"训练bundle完整性验收通过: {training_marker}")
    print(
        f"发现{len(test_files)}个测试场站；旧F0--F7只读复用；"
        f"新推理变体: {new_variants}"
    )
    legacy = load_legacy_f0_f7_aggregates(test_files)
    inferred = {}
    for variant_id in new_variants:
        if variant_id not in NEW_TRAINING_VARIANTS:
            raise ValueError(f"禁止对非补充变体重新推理: {variant_id}")
        inferred[variant_id] = predict_feature_variant(variant_id, test_files)

    full_selection = all_farms and "f8" in inferred
    selection_combined, selection_paths = save_extended_selection_outputs(
        legacy,
        inferred.get("f8"),
        full_selection,
    )
    print("F0--F8独立选型输出:")
    for name, path in selection_paths.items():
        print(f"  {name}: {path}")

    probe_results = [inferred[item] for item in PROBE_VARIANTS if item in inferred]
    probe_combined = None
    probe_paths = {}
    full_pair = False
    if probe_results:
        full_pair = all_farms and set(PROBE_VARIANTS).issubset(inferred)
        probe_combined, probe_paths = save_probe_outputs(probe_results, full_pair)
        print("Frozen-Pair control独立输出:")
        for name, path in probe_paths.items():
            print(f"  {name}: {path}")

    analysis_paths = {}
    if full_selection:
        analysis_paths = save_analysis_reports(
            selection_combined,
            probe_combined if probe_combined is not None and "invariance" in probe_combined else None,
        )
        print("复杂度/candidate drift/门控/C结论报告:")
        for name, path in analysis_paths.items():
            print(f"  {name}: {path}")
        selected = pd.read_csv(selection_paths["final_selection"]).iloc[0]
        print(
            f"按测试集5场站宏平均NRMSE选定: "
            f"{selected['model_variant']} ({selected[SELECTION_MACRO_METRIC]:.6f})"
        )
        print("FP0/FP4为Frozen-Pair control，从未进入F0--F8排名")
        if full_pair and probe_combined is not None and "invariance" in probe_combined:
            marker = _publish_bundle_completion_marker(
                ("selection", selection_paths),
                ("frozen_pair", probe_paths),
                ("analysis", analysis_paths),
            )
            print(f"正式补充实验bundle完成标志: {marker}")
        else:
            print(
                "FP0/FP4未完整验收；不发布bundle complete标志，"
                "C直接归因仍不完整"
            )
    else:
        print("未完整覆盖F0--F8全场站；不生成最终winner")


if __name__ == "__main__":
    main()
