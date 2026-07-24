"""Part 3 / Round 3：JSFD001--JSFD014 十模型统一外部测试。

正式模式具有严格的 ``test-once`` 冻结门：在读取任何测试数组之前，必须确认
14 个场站、10 个模型的训练 task marker 和总 training complete marker 全部
存在且文件哈希有效。局部调试必须显式使用 ``--partial`` 或 ``--smoke``，输出
自动进入 ``partial_runs``，不会污染正式测试目录。

本文件只消费 Round 3 的无泄漏 preprocessing bundle 和从零训练产物；禁止读取
``processed_npz``。完整逐样本预测使用 gzip CSV，汇总指标使用普通 CSV。
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


RESULT_ROOT = Path(
    "./wind_results/part3_new_module_supplement/"
    "03_external14_leakage_free_strong_baseline_benchmark"
)
PREDICTION_DIRNAME = "testdata_predict_output"
PREDICTION_COMPLETE_NAME = "round3_external14_prediction_bundle_complete.json"
PREPROCESS_COMPLETE_NAME = "round3_preprocess_bundle_complete.json"
TRAINING_COMPLETE_CANDIDATES = (
    "round3_training_bundle_complete.json",
    "round3_external14_training_bundle_complete.json",
    "manifests/round3_training_bundle_complete.json",
    "manifests/round3_external14_training_bundle_complete.json",
    "manifests/training/round3_training_bundle_complete.json",
    "manifests/training/round3_external14_training_bundle_complete.json",
)
PROTOCOL_VERSION = "part3_round3_external14_test_once_v2"
EXPECTED_PREPROCESS_PROTOCOL_VERSION = (
    "part3_round3_external14_leakage_free_v2"
)
EXPECTED_TRAINING_PROTOCOL_VERSION = (
    "part3_round3_external14_unified_training_v2"
)
SEED = 2026
HISTORY_LEN = 96
FORECAST_LEN = 16
TIME_FREQ_MINUTES = 15
DEFAULT_BATCH_SIZE = 192
EXPECTED_INPUT_DIM = 45
EXPECTED_TARGET_INDEX = 44
EXPECTED_FEATURE_SCHEMA_HASH = (
    "a2f44e932044c2609a8c0e1cf6a446f37b4a0cfb71b8bf232a5bae6c568c680c"
)
EXPECTED_FARMS = tuple(f"JSFD{i:03d}" for i in range(1, 15))
MODEL_IDS = (
    "patchtst",
    "bilstm",
    "cnn_lstm",
    "cnn_resnet_gru",
    "wavenet",
    "transformer",
    "informer",
    "autoformer",
    "hr_moe_fets_patchtst",
    "windprism_f7_g0",
)
MODEL_DISPLAY_NAMES = {
    "patchtst": "PatchTST",
    "bilstm": "BiLSTM",
    "cnn_lstm": "CNN-LSTM",
    "cnn_resnet_gru": "CNN-ResNet-GRU",
    "wavenet": "WaveNet",
    "transformer": "Transformer",
    "informer": "Informer",
    "autoformer": "Autoformer",
    "hr_moe_fets_patchtst": "HR-MoE FeTS-PatchTST (B6/v5ab)",
    "windprism_f7_g0": "WindPRISM (F7/G0)",
}
PRIMARY_MODEL_ID = "windprism_f7_g0"
TIE_TOLERANCE = 1e-6


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _test_evaluation_provenance(formal):
    """Return precise test-use claims without overstating global blindness."""
    return {
        "test_reuse_status": (
            "frozen_current_round_external_evaluation_with_"
            "prior_dataset_exposure_disclosed"
            if formal
            else "nonformal_partial_or_smoke_diagnostic"
        ),
        # JSFD001--014 existed in an earlier teacher/processed-data workflow.
        # This round rebuilds raw Excel and freezes models before formal
        # prediction, but cannot truthfully claim that the dataset was never
        # seen anywhere in the wider project.
        "test_is_final_blind_evaluation": False,
        "test_used_for_selection": bool(formal),
        "selection_is_descriptive_not_confirmatory": True,
        "confirmatory_evaluation_scope": (
            "predeclared frozen-model comparisons and primary WindPRISM hypothesis"
            if formal
            else "none_nonformal_run"
        ),
        "winner_selection_scope": "descriptive test-set ranking",
        "test_execution_mode": (
            "one_shot_after_140_training_tasks_frozen"
            if formal
            else "partial_or_smoke_after_selected_tasks_frozen"
        ),
        "all_models_frozen_before_first_formal_test_prediction": bool(formal),
        "test_targets_used_for_training_or_validation_selection": False,
        "post_test_training_or_hyperparameter_changes_in_this_run": False,
        "test_results_used_for_further_tuning_within_this_run": False,
        "future_tuning_policy": (
            "any later test-driven change invalidates confirmatory use and "
            "must be disclosed as development reuse"
        ),
        "test_split_role": "within_station_chronological_holdout",
        "external_to_current_windprism_development": True,
        "dataset_globally_never_used": False,
        "dataset_prior_use_context": (
            "historical external-teacher/processed-data workflow; no Round-3 "
            "arrays, statistics, or weights reused"
        ),
        "legacy_processed_npz_used": False,
        "legacy_processed_npz_used_as_model_input": False,
        "legacy_weights_reused": False,
        "all_station_models_trained_from_scratch": bool(formal),
        "architecture_definitions_reused": True,
        "prediction_resume_allowed_only_with_identical_frozen_snapshot": True,
    }


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2)
    os.replace(temp, path)
    return path


def _atomic_csv(rows, path, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_json_safe(dict(row)) for row in rows]
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)
    return path


def _atomic_text(text, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        handle.write(str(text))
    os.replace(temp, path)
    return path


def _json_safe(value):
    try:
        import numpy as np
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, np.ndarray):
            return [_json_safe(item) for item in value.tolist()]
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_json(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON根对象必须为dict: {path}")
    return value


def _completion_declared(payload):
    status = str(payload.get("status", "")).lower()
    return bool(
        payload.get("complete")
        or payload.get("completed")
        or status in {"complete", "completed", "success", "succeeded"}
    )


def _forbid_legacy_path(path):
    resolved = Path(path).resolve()
    if "processed_npz" in {part.lower() for part in resolved.parts}:
        raise ValueError(f"Round 3禁止读取旧processed_npz产物: {resolved}")
    return resolved


def _resolve_existing(root, candidates):
    for item in candidates:
        path = Path(item)
        if not path.is_absolute():
            path = Path(root) / path
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"未找到候选文件: {candidates}")


def _normalize_requested(values, allowed, label):
    if not values:
        return list(allowed)
    result = []
    for raw in values:
        result.extend(part.strip() for part in str(raw).split(",") if part.strip())
    invalid = sorted(set(result) - set(allowed))
    if invalid:
        raise ValueError(f"未知{label}: {invalid}; 可选={list(allowed)}")
    return list(dict.fromkeys(result))


def _task_marker_path(root, model_id, farm_id):
    return Path(root) / "manifests" / "training" / f"{model_id}_{farm_id}.json"


def _prediction_marker_path(output_root, model_id, farm_id):
    return (
        Path(output_root)
        / "manifests"
        / "prediction"
        / f"{model_id}_{farm_id}.json"
    )


def _record_from_value(value):
    if isinstance(value, str):
        return {"path": value}
    if isinstance(value, dict) and value.get("path"):
        return value
    return None


def _extract_artifact_record(marker, role):
    aliases = {
        "model": ("model", "model_path", "keras_model", "saved_model"),
        "weights": ("weights", "weights_path", "best_weights", "best_weights_path"),
        "artifact": (
            "artifact",
            "artifact_path",
            "training_artifact",
            "preprocess_artifact",
        ),
        "history": ("history", "history_path", "history_csv"),
    }[role]
    containers = [marker]
    for key in ("artifacts", "files", "paths", "output_files"):
        if isinstance(marker.get(key), dict):
            containers.append(marker[key])
    for container in containers:
        for key in aliases:
            record = _record_from_value(container.get(key))
            if record:
                if not record.get("sha256"):
                    hash_aliases = {
                        "model": ("model_sha256",),
                        "weights": ("weights_sha256", "best_weights_sha256"),
                        "artifact": ("artifact_sha256",),
                        "history": ("history_sha256",),
                    }[role]
                    for hash_key in hash_aliases:
                        if marker.get(hash_key):
                            record["sha256"] = marker[hash_key]
                            break
                if (
                    role == "model"
                    and not record.get("size_bytes")
                    and marker.get("model_size_bytes") is not None
                ):
                    record["size_bytes"] = marker["model_size_bytes"]
                return record
    raise KeyError(f"训练marker缺少{role}文件记录")


def _training_source_hash(marker, role):
    aliases = {
        "bundle": (
            "preprocessing_bundle_sha256",
            "preprocess_bundle_sha256",
            "bundle_sha256",
        ),
        "array": (
            "array_sha256",
            "feature_array_sha256",
            "preprocessed_array_sha256",
        ),
    }[role]
    for key in aliases:
        if marker.get(key):
            return str(marker[key])
    for container_name in ("sources", "source_files", "preprocessing", "data_source"):
        container = marker.get(container_name)
        if not isinstance(container, dict):
            continue
        role_aliases = (
            ("preprocessing_bundle", "preprocess_bundle", "bundle")
            if role == "bundle"
            else ("feature_array", "array", "npz")
        )
        for key in role_aliases:
            value = container.get(key)
            if isinstance(value, dict) and value.get("sha256"):
                return str(value["sha256"])
        for key in aliases:
            if container.get(key):
                return str(container[key])
    return None


def _assert_station_matches_training(station, frozen_tasks, models, farm_id, formal):
    missing = []
    for model_id in models:
        marker = frozen_tasks[(model_id, farm_id)]["marker"]
        bundle_hash = _training_source_hash(marker, "bundle")
        array_hash = _training_source_hash(marker, "array")
        if bundle_hash is None or array_hash is None:
            missing.append(model_id)
            continue
        if bundle_hash != station["bundle_sha256"]:
            raise ValueError(
                f"{model_id}/{farm_id}训练所用preprocessing bundle与当前测试bundle不同"
            )
        if array_hash != station["array_sha256"]:
            raise ValueError(
                f"{model_id}/{farm_id}训练所用特征数组与当前测试数组不同"
            )
        marker_schema = marker.get("schema_hash") or marker.get("feature_schema_hash")
        if marker_schema and station["schema_hash"] and str(marker_schema) != station["schema_hash"]:
            raise ValueError(f"{model_id}/{farm_id}训练与测试feature schema hash不同")
    if missing and formal:
        raise ValueError(
            f"{farm_id}以下正式训练marker缺少bundle/array SHA-256，"
            f"无法证明训练测试使用同一预处理bundle: {missing}"
        )


def _validate_record(record, hash_cache=None):
    record = dict(record)
    path = Path(record["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_size = record.get("size_bytes")
    if expected_size is not None and int(expected_size) != path.stat().st_size:
        raise ValueError(f"文件大小漂移: {path}")
    expected_hash = record.get("sha256")
    if expected_hash:
        cache_key = (str(path), path.stat().st_size, path.stat().st_mtime_ns)
        actual = hash_cache.get(cache_key) if hash_cache is not None else None
        if actual is None:
            actual = _sha256(path)
            if hash_cache is not None:
                hash_cache[cache_key] = actual
        if actual != expected_hash:
            raise ValueError(f"文件SHA-256漂移: {path}")
    return path


def _find_training_complete(root):
    return _resolve_existing(root, TRAINING_COMPLETE_CANDIDATES)


def _validate_preprocess_complete(root):
    path = Path(root) / PREPROCESS_COMPLETE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"缺少Round 3预处理complete marker: {path}")
    payload = _read_json(path)
    if not _completion_declared(payload):
        raise ValueError(f"预处理marker未声明complete: {path}")
    if payload.get("protocol_version") != EXPECTED_PREPROCESS_PROTOCOL_VERSION:
        raise ValueError(f"预处理协议版本不是Round-3 v2: {path}")
    if set(map(str, payload.get("completed_farms", ()))) != set(
        EXPECTED_FARMS
    ):
        raise ValueError(f"预处理complete marker未精确覆盖14个场站: {path}")
    return path, payload


def _validate_all_training_frozen(root):
    """在测试数组被读取前冻结并验证全部 140 个训练任务。"""
    root = Path(root).resolve()
    complete_path = root / "round3_training_bundle_complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(
            f"正式测试只接受规范Round-3 v2训练complete marker: {complete_path}"
        )
    complete = _read_json(complete_path)
    if not _completion_declared(complete):
        raise ValueError(f"训练总marker未声明complete: {complete_path}")
    if complete.get("protocol_version") != EXPECTED_TRAINING_PROTOCOL_VERSION:
        raise ValueError(f"训练complete marker不是Round-3 v2: {complete_path}")
    expected_pairs = {
        (model_id, farm_id)
        for model_id in MODEL_IDS
        for farm_id in EXPECTED_FARMS
    }
    declared_pairs = {
        (str(item.get("model_id")), str(item.get("farm_id")))
        for item in complete.get("completed_tasks", ())
    }
    if (
        int(complete.get("expected_task_count", -1)) != len(expected_pairs)
        or int(complete.get("completed_task_count", -1)) != len(expected_pairs)
        or declared_pairs != expected_pairs
    ):
        raise ValueError("训练complete marker未精确声明10×14任务矩阵")
    hash_cache = {}
    for record in complete.get("summary_outputs", {}).values():
        _validate_record(record, hash_cache)
    batch_policy_path = _validate_record(
        {
            "path": complete.get("global_batch_policy_path"),
            "sha256": complete.get("global_batch_policy_sha256"),
        },
        hash_cache,
    )
    del batch_policy_path
    task_records = {
        (str(item.get("model_id")), str(item.get("farm_id"))): item
        for item in complete.get("task_marker_records", ())
    }
    if set(task_records) != expected_pairs:
        raise ValueError("训练complete marker缺少140个task marker哈希快照")
    tasks = {}
    for farm_id in EXPECTED_FARMS:
        for model_id in MODEL_IDS:
            path = _task_marker_path(root, model_id, farm_id)
            if not path.is_file():
                raise FileNotFoundError(f"正式测试缺少训练task marker: {path}")
            declared_record = task_records[(model_id, farm_id)]
            if Path(declared_record["path"]).resolve() != path.resolve():
                raise ValueError(f"训练task marker路径与complete快照不一致: {path}")
            _validate_record(declared_record, hash_cache)
            marker = _read_json(path)
            if not _completion_declared(marker):
                raise ValueError(f"训练task尚未complete: {path}")
            if marker.get("protocol_version") != EXPECTED_TRAINING_PROTOCOL_VERSION:
                raise ValueError(f"训练task协议版本漂移: {path}")
            if (
                marker.get("preprocess_protocol_version")
                != EXPECTED_PREPROCESS_PROTOCOL_VERSION
            ):
                raise ValueError(f"训练task预处理协议版本漂移: {path}")
            if str(marker.get("model_id", model_id)) != model_id:
                raise ValueError(f"训练task model_id漂移: {path}")
            if str(marker.get("farm_id", farm_id)) != farm_id:
                raise ValueError(f"训练task farm_id漂移: {path}")
            if (
                marker.get("training_initialization")
                != "from_scratch_seed_2026"
                or marker.get("pretrained_weights_loaded") is not False
            ):
                raise ValueError(f"训练task未证明从seed=2026随机初始化: {path}")
            if (
                marker.get("global_batch_policy_sha256")
                != complete.get("global_batch_policy_sha256")
            ):
                raise ValueError(f"训练task全局batch策略哈希漂移: {path}")
            model_record = _extract_artifact_record(marker, "model")
            _validate_record(model_record, hash_cache)
            for optional_role in ("weights", "artifact", "history"):
                try:
                    _validate_record(
                        _extract_artifact_record(marker, optional_role),
                        hash_cache,
                    )
                except KeyError:
                    # Round-3's task marker is itself the immutable training
                    # artifact; no redundant pickle is required.  Best
                    # weights and history remain mandatory.
                    if optional_role in {"weights", "history"}:
                        raise
            tasks[(model_id, farm_id)] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "marker": marker,
                "model_record": model_record,
            }
    return {
        "training_complete": _file_record(complete_path),
        "tasks": tasks,
    }


def _validate_selected_training(root, models, farms):
    """局部/冒烟模式仅验证被请求任务，不宣称正式测试完成。"""
    hash_cache = {}
    tasks = {}
    for farm_id in farms:
        for model_id in models:
            candidates = (
                _task_marker_path(root, model_id, farm_id),
                (
                    Path(root)
                    / "partial_runs"
                    / "smoke"
                    / "manifests"
                    / "training"
                    / f"{model_id}_{farm_id}.json"
                ),
            )
            path = next((item for item in candidates if item.is_file()), None)
            if path is None:
                raise FileNotFoundError(
                    f"局部预测未找到训练task marker: {list(map(str, candidates))}"
                )
            marker = _read_json(path)
            if not _completion_declared(marker):
                raise ValueError(f"局部预测所需训练task未完成: {path}")
            model_record = _extract_artifact_record(marker, "model")
            _validate_record(model_record, hash_cache)
            tasks[(model_id, farm_id)] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "marker": marker,
                "model_record": model_record,
            }
    return {"training_complete": None, "tasks": tasks}


def _first(payload, names, default=None):
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return default


def _scaler_stats(bundle, prefix, expected_dim, npz_values=None):
    import numpy as np
    npz_values = npz_values or {}
    mean = _first(
        npz_values,
        (f"{prefix}_mean", f"{prefix}_mean_"),
    )
    scale = _first(
        npz_values,
        (f"{prefix}_scale", f"{prefix}_scale_"),
    )
    nested = bundle.get("scaler_arrays", {})
    if mean is None:
        mean = _first(
            bundle,
            (f"{prefix}_mean", f"{prefix}_mean_"),
            _first(nested, (f"{prefix}_mean", f"{prefix}_mean_")),
        )
    if scale is None:
        scale = _first(
            bundle,
            (f"{prefix}_scale", f"{prefix}_scale_"),
            _first(nested, (f"{prefix}_scale", f"{prefix}_scale_")),
        )
    if mean is None or scale is None:
        obj = bundle.get(prefix)
        if obj is not None:
            mean = getattr(obj, "mean_", mean)
            scale = getattr(obj, "scale_", scale)
    if mean is None or scale is None:
        raise KeyError(
            f"preprocessing bundle缺少{prefix} mean/scale纯数组及兼容对象"
        )
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    if len(mean) != expected_dim or len(scale) != expected_dim:
        raise ValueError(
            f"{prefix}维数异常: mean={len(mean)}, scale={len(scale)}, "
            f"expected={expected_dim}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError(f"{prefix}统计量含非有限值")
    if np.any(scale <= 0):
        raise ValueError(f"{prefix} scale必须全部为正")
    return mean, scale


def _npz_value(npz, aliases, required=True):
    for key in aliases:
        if key in npz.files:
            return npz[key]
    if required:
        raise KeyError(f"NPZ缺少字段，允许别名={aliases}; 实际={npz.files}")
    return None


def _bundle_paths(root, farm_id):
    bundle = Path(root) / "preprocess" / farm_id / "preprocessing_bundle.joblib"
    array = Path(root) / "prepared_data" / "feature_arrays" / f"{farm_id}.npz"
    return _forbid_legacy_path(bundle), _forbid_legacy_path(array)


def _load_station_bundle(root, farm_id, smoke_limit=None):
    """读取一个场站；调用者必须先通过正式冻结门。"""
    import numpy as np

    bundle_path, default_array_path = _bundle_paths(root, farm_id)
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)
    # Never unpickle the sklearn-bearing joblib in the TensorFlow environment:
    # preprocessing and DL environments intentionally use different sklearn
    # minor versions.  Immutable numeric state comes from NPZ; JSON supplies
    # optional metadata, while the joblib file is identity-hashed only.
    regime_json = (
        Path(root) / "preprocess" / farm_id / "regime_feature_config.json"
    )
    manifest_path = (
        Path(root) / "manifests" / "preprocess" / f"{farm_id}.json"
    )
    bundle = _read_json(regime_json) if regime_json.is_file() else {}
    preprocess_manifest = (
        _read_json(manifest_path) if manifest_path.is_file() else {}
    )
    summary = preprocess_manifest.get("summary", {})
    if bundle and str(bundle.get("farm_id", farm_id)) != farm_id:
        raise ValueError(f"regime JSON场站身份漂移: {regime_json}")
    if summary and str(summary.get("farm_id", farm_id)) != farm_id:
        raise ValueError(f"preprocess manifest场站身份漂移: {manifest_path}")
    array_hint = _first(
        bundle,
        ("array_path", "feature_array_path", "npz_path"),
        str(default_array_path),
    )
    array_path = Path(array_hint)
    if not array_path.is_absolute():
        candidate = Path(root) / array_path
        array_path = candidate if candidate.is_file() else bundle_path.parent / array_path
    array_path = _forbid_legacy_path(array_path)
    if not array_path.is_file():
        raise FileNotFoundError(array_path)
    expected_array_hash = _first(bundle, ("array_sha256", "npz_sha256"))
    actual_array_hash = _sha256(array_path)
    if expected_array_hash and expected_array_hash != actual_array_hash:
        raise ValueError(f"{farm_id}特征数组SHA-256与bundle不一致")
    if summary.get("array_sha256") and summary["array_sha256"] != actual_array_hash:
        raise ValueError(f"{farm_id}特征数组SHA-256与preprocess manifest不一致")
    actual_bundle_hash = _sha256(bundle_path)
    if (
        summary.get("bundle_sha256")
        and summary["bundle_sha256"] != actual_bundle_hash
    ):
        raise ValueError(f"{farm_id}joblib身份哈希与preprocess manifest不一致")

    npz_scalers = {}
    npz_metadata = {}
    with np.load(array_path, allow_pickle=False) as npz:
        features = np.asarray(
            _npz_value(npz, ("features_scaled", "features", "X_scaled")),
            dtype=np.float32,
        )
        target_scaled = np.asarray(
            _npz_value(npz, ("target_scaled", "target", "y_scaled")),
            dtype=np.float32,
        ).reshape(-1)
        target_mw = np.asarray(
            _npz_value(npz, ("target_mw", "power_mw", "y_mw")),
            dtype=np.float64,
        ).reshape(-1)
        timestamps_ns = np.asarray(
            _npz_value(npz, ("timestamps_ns", "timestamp_ns", "time_ns")),
            dtype=np.int64,
        ).reshape(-1)
        power_valid = np.asarray(
            _npz_value(npz, ("power_valid", "target_valid")),
            dtype=bool,
        ).reshape(-1)
        test_origins = np.asarray(
            _npz_value(npz, ("test_origins", "origins_test")),
            dtype=np.int64,
        ).reshape(-1)
        for key in (
            "scaler_x_mean",
            "scaler_x_scale",
            "scaler_y_mean",
            "scaler_y_scale",
            "scaler_x_mean_",
            "scaler_x_scale_",
            "scaler_y_mean_",
            "scaler_y_scale_",
        ):
            if key in npz.files:
                npz_scalers[key] = np.asarray(npz[key])
        for key in (
            "input_cols",
            "feature_names",
            "target_index",
            "target_channel_index",
            "power_reference_mw",
            "capacity_mw",
            "train_only_power_reference_mw",
            "power_reference_kind",
            "schema_hash",
            "feature_schema_hash",
            "history_len",
            "forecast_len",
        ):
            if key in npz.files:
                value = np.asarray(npz[key])
                npz_metadata[key] = (
                    value.reshape(-1)[0].item()
                    if value.size == 1
                    else value.tolist()
                )

    history_len = int(
        _first(
            npz_metadata,
            ("history_len",),
            _first(bundle, ("history_len", "context_length"), HISTORY_LEN),
        )
    )
    forecast_len = int(
        _first(
            npz_metadata,
            ("forecast_len",),
            _first(bundle, ("forecast_len", "prediction_length"), FORECAST_LEN),
        )
    )
    if history_len != HISTORY_LEN or forecast_len != FORECAST_LEN:
        raise ValueError(
            f"{farm_id}窗口协议漂移: history={history_len}, forecast={forecast_len}"
        )
    raw_input_cols = _first(
        npz_metadata,
        ("input_cols", "feature_names"),
        _first(bundle, ("input_cols", "feature_names"), []),
    )
    if isinstance(raw_input_cols, str):
        try:
            raw_input_cols = json.loads(raw_input_cols)
        except json.JSONDecodeError:
            raw_input_cols = [item for item in raw_input_cols.split("|") if item]
    input_cols = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in list(raw_input_cols)
    ]
    target_index = int(
        _first(
            npz_metadata,
            ("target_index", "target_channel_index"),
            _first(
                bundle,
                ("target_index", "target_channel_index"),
                EXPECTED_TARGET_INDEX,
            ),
        )
    )
    if len(input_cols) != EXPECTED_INPUT_DIM or target_index != EXPECTED_TARGET_INDEX:
        raise ValueError(
            f"{farm_id}输入schema异常: columns={len(input_cols)}, "
            f"target_index={target_index}"
        )
    computed_schema_hash = hashlib.sha256(
        json.dumps(
            input_cols,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if computed_schema_hash != EXPECTED_FEATURE_SCHEMA_HASH:
        raise ValueError(f"{farm_id}输入列语义/顺序与冻结F7 schema不一致")
    declared_schema_hash = str(
        _first(
            npz_metadata,
            ("schema_hash", "feature_schema_hash"),
            _first(bundle, ("schema_hash", "feature_schema_hash"), ""),
        )
    )
    if declared_schema_hash and declared_schema_hash != computed_schema_hash:
        raise ValueError(f"{farm_id}预处理声明的schema hash与实际列不一致")

    length = len(features)
    if features.ndim != 2 or features.shape[1] != EXPECTED_INPUT_DIM:
        raise ValueError(f"{farm_id}features_scaled形状异常: {features.shape}")
    for name, value in (
        ("target_scaled", target_scaled),
        ("target_mw", target_mw),
        ("timestamps_ns", timestamps_ns),
        ("power_valid", power_valid),
    ):
        if len(value) != length:
            raise ValueError(f"{farm_id}/{name}长度{len(value)} != {length}")
    if len(test_origins) == 0:
        raise ValueError(f"{farm_id}测试origin为空")
    if np.any(np.diff(test_origins) <= 0):
        raise ValueError(f"{farm_id}测试origin必须严格递增且唯一")
    if test_origins.min() < HISTORY_LEN or test_origins.max() + FORECAST_LEN > length:
        raise ValueError(f"{farm_id}测试origin越界")
    offsets = np.arange(-HISTORY_LEN, FORECAST_LEN, dtype=np.int64)
    used = test_origins[:, None] + offsets[None, :]
    if not power_valid[used].all():
        raise ValueError(f"{farm_id}测试窗口包含预处理声明的无效功率点")
    if not np.isfinite(features[test_origins.min() - HISTORY_LEN :]).all():
        raise ValueError(f"{farm_id}测试所需features_scaled含非有限值")
    target_indices = test_origins[:, None] + np.arange(FORECAST_LEN)[None, :]
    if not np.isfinite(target_scaled[target_indices]).all():
        raise ValueError(f"{farm_id}测试target_scaled含非有限值")
    if not np.isfinite(target_mw[target_indices]).all():
        raise ValueError(f"{farm_id}测试target_mw含非有限值")
    if smoke_limit:
        test_origins = test_origins[: int(smoke_limit)]
        target_indices = target_indices[: int(smoke_limit)]

    x_mean, x_scale = _scaler_stats(
        bundle, "scaler_x", EXPECTED_INPUT_DIM, npz_scalers
    )
    y_mean, y_scale = _scaler_stats(
        bundle, "scaler_y", 1, npz_scalers
    )
    power_reference = float(
        _first(
            npz_metadata,
            ("power_reference_mw", "capacity_mw", "train_only_power_reference_mw"),
            _first(
                bundle,
                (
                    "power_reference_mw",
                    "capacity_mw",
                    "train_only_power_reference_mw",
                ),
                float("nan"),
            ),
        )
    )
    if not math.isfinite(power_reference) or power_reference <= 0:
        raise ValueError(f"{farm_id}缺少有效功率归一化参考值")
    return {
        "farm_id": farm_id,
        "bundle": bundle,
        "bundle_path": str(bundle_path),
        "bundle_sha256": actual_bundle_hash,
        "array_path": str(array_path),
        "array_sha256": actual_array_hash,
        "features": features,
        "target_scaled": target_scaled,
        "target_mw": target_mw,
        "timestamps_ns": timestamps_ns,
        "test_origins": test_origins,
        "target_indices": target_indices,
        "input_cols": input_cols,
        "target_index": target_index,
        "scaler_x_mean": x_mean,
        "scaler_x_scale": x_scale,
        "scaler_y_mean": float(y_mean[0]),
        "scaler_y_scale": float(y_scale[0]),
        "power_reference_mw": power_reference,
        "power_reference_kind": str(
            _first(
                npz_metadata,
                ("power_reference_kind",),
                _first(bundle, ("power_reference_kind",), "train_reference"),
            )
        ),
        "schema_hash": computed_schema_hash,
    }


def _configure_tensorflow():
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    try:
        keras.utils.set_random_seed(SEED)
    except AttributeError:
        pass
    return tf, keras


def _register_custom_layers(model_id):
    """动态导入原始结构模块，使Keras注册所有自定义层。"""
    custom_objects = {}
    if model_id == "patchtst":
        __import__("wind_dl_model_train")
    elif model_id in {
        "bilstm",
        "cnn_lstm",
        "cnn_resnet_gru",
        "wavenet",
        "transformer",
        "informer",
        "autoformer",
    }:
        __import__("wind_dl_other_models_train")
    elif model_id == "hr_moe_fets_patchtst":
        __import__("wind_dl_model_train")
        __import__("wind_FeTS_PatchTST_train")
    elif model_id == "windprism_f7_g0":
        module = __import__("wind_RegimeEncoder_PatchTST_feature_screen_train")
        __import__("wind_RegimeEncoder_PatchTST_train")
        custom_objects.update(module.get_feature_screen_custom_objects())
        round3_train = __import__(
            "wind_part3_round3_external14_all_models_train"
        )
        custom_objects.update(round3_train.get_round3_custom_objects())
    else:
        raise ValueError(model_id)
    return custom_objects


def _make_test_dataset(tf, station, batch_size):
    features = station["features"]
    origins = station["test_origins"]
    with tf.device("/CPU:0"):
        feature_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices(origins)

    def gather(origin):
        index = tf.range(origin - HISTORY_LEN, origin)
        x = tf.gather(feature_tensor, index)
        x.set_shape((HISTORY_LEN, EXPECTED_INPUT_DIM))
        return x

    options = tf.data.Options()
    options.experimental_deterministic = True
    return (
        ds.with_options(options)
        .map(gather, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        .batch(int(batch_size), drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )


def _metric_values(y_true, y_pred, reference):
    import numpy as np

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        return {name: float("nan") for name in ("mae", "rmse", "nmae", "nrmse", "r2", "smape", "mape")}
    truth = y_true[valid]
    pred = y_pred[valid]
    error = pred - truth
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    sst = float(np.sum(np.square(truth - np.mean(truth))))
    r2 = float(1.0 - np.sum(np.square(error)) / sst) if sst > 1e-12 else float("nan")
    smape = float(
        np.mean(2.0 * np.abs(error) / np.maximum(np.abs(truth) + np.abs(pred), 1e-6))
    )
    nonzero = np.abs(truth) >= max(1e-6, 0.01 * float(reference))
    mape = (
        float(np.mean(np.abs(error[nonzero]) / np.abs(truth[nonzero])))
        if nonzero.any()
        else float("nan")
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "nmae": mae / float(reference),
        "nrmse": rmse / float(reference),
        "r2": r2,
        "smape": smape,
        "mape": mape,
    }


def _huber_mean(y_true, y_pred, delta=1.0):
    import numpy as np
    error = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    loss = np.where(error <= delta, 0.5 * np.square(error), delta * (error - 0.5 * delta))
    return float(np.mean(loss))


def _extract_predictions(raw, model):
    if isinstance(raw, dict):
        forecast = raw.get("forecast_power")
        if forecast is None:
            forecast = next(iter(raw.values()))
        candidate = raw.get("candidate_forecast")
        return forecast, candidate
    if isinstance(raw, (list, tuple)):
        mapping = dict(zip(model.output_names, raw))
        return mapping.get("forecast_power", raw[0]), mapping.get("candidate_forecast")
    return raw, None


def _predict_with_diagnostics(tf, keras, model, model_id, dataset):
    """一次前向尽量提取主预测及结构专属诊断，避免重复读取测试数据。"""
    diagnostics = {}
    diagnostic_model = model
    output_names = ["forecast"]
    if model_id == "windprism_f7_g0":
        layer_names = (
            "forecast_power",
            "candidate_forecast",
            "persistence_forecast_candidate",
            "corrected_forecast_candidate",
            "correction_gate",
        )
        missing = [name for name in layer_names if not any(layer.name == name for layer in model.layers)]
        if missing:
            raise ValueError(f"WindPRISM模型缺少诊断层: {missing}")
        diagnostic_model = keras.Model(
            model.inputs,
            [model.get_layer(name).output for name in layer_names],
            name="Round3WindPRISMDiagnostics",
        )
        output_names = list(layer_names)
    elif model_id == "hr_moe_fets_patchtst":
        router_name = "horizon_regime_router"
        if any(layer.name == router_name for layer in model.layers):
            diagnostic_model = keras.Model(
                model.inputs,
                [model.output, model.get_layer(router_name).output],
                name="Round3HRMoEDiagnostics",
            )
            output_names = ["forecast_power", "router_weights"]

    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass
    started = time.monotonic()
    raw = diagnostic_model.predict(dataset, verbose=0)
    elapsed = float(time.monotonic() - started)
    if isinstance(raw, dict):
        arrays = raw
    elif isinstance(raw, (list, tuple)):
        arrays = dict(zip(output_names, raw))
    else:
        arrays = {output_names[0]: raw}
    if model_id == "windprism_f7_g0":
        forecast = arrays["forecast_power"]
        diagnostics = {
            "candidate_scaled": arrays["candidate_forecast"],
            "persistence_scaled": arrays["persistence_forecast_candidate"],
            "corrected_scaled": arrays["corrected_forecast_candidate"],
            "gate": arrays["correction_gate"],
        }
    else:
        forecast = arrays.get("forecast_power", arrays.get("forecast"))
        if forecast is None:
            forecast = next(iter(arrays.values()))
        if "router_weights" in arrays:
            diagnostics["router_weights"] = arrays["router_weights"]
    import numpy as np
    forecast = np.asarray(forecast, dtype=np.float64)
    for key, value in list(diagnostics.items()):
        diagnostics[key] = np.asarray(value, dtype=np.float64)
    peak = None
    try:
        peak = int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
    except Exception:
        pass
    return forecast, diagnostics, elapsed, peak


def _inverse_scaled(station, values):
    import numpy as np
    return (
        np.asarray(values, dtype=np.float64) * station["scaler_y_scale"]
        + station["scaler_y_mean"]
    )


def _timestamps_iso(values):
    import pandas as pd
    # Raw Excel timestamps are local station time without an authoritative
    # timezone offset.  Keep that semantics explicit instead of attaching a
    # misleading UTC ``Z`` suffix.
    return pd.to_datetime(values, unit="ns").strftime("%Y-%m-%dT%H:%M:%S")


def _save_sample_predictions(output_root, model_id, farm_id, station, truth, prediction):
    import pandas as pd

    origins = station["test_origins"]
    frame = pd.DataFrame(
        {
            "sample_index": range(len(origins)),
            "origin_index": origins,
            "decision_time": _timestamps_iso(
                station["timestamps_ns"][origins - 1]
            ),
            "target_start_time": _timestamps_iso(
                station["timestamps_ns"][origins]
            ),
        }
    )
    for horizon in range(FORECAST_LEN):
        frame[f"y_true_h{horizon + 1:02d}_mw"] = truth[:, horizon]
        frame[f"y_pred_h{horizon + 1:02d}_mw"] = prediction[:, horizon]
        frame[f"error_h{horizon + 1:02d}_mw"] = (
            prediction[:, horizon] - truth[:, horizon]
        )
    path = (
        Path(output_root)
        / "per_sample"
        / model_id
        / f"{model_id}_{farm_id}_test_predictions.csv.gz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.gz")
    frame.to_csv(temp, index=False, encoding="utf-8-sig", compression="gzip")
    os.replace(temp, path)
    return path


def _save_diagnostic_tables(
    output_root,
    model_id,
    farm_id,
    station,
    truth,
    forecast,
    diagnostics,
):
    import numpy as np
    rows = []
    extra_paths = {}
    reference = station["power_reference_mw"]
    if model_id == "windprism_f7_g0":
        candidate = np.clip(_inverse_scaled(station, diagnostics["candidate_scaled"]), 0.0, None)
        persistence = np.clip(
            _inverse_scaled(station, diagnostics["persistence_scaled"]), 0.0, None
        )
        corrected = np.clip(
            _inverse_scaled(station, diagnostics["corrected_scaled"]), 0.0, None
        )
        gate = diagnostics["gate"]
        for horizon in range(FORECAST_LEN):
            row = {
                "model_id": model_id,
                "farm_id": farm_id,
                "horizon": horizon + 1,
                "lead_minutes": (horizon + 1) * TIME_FREQ_MINUTES,
                "gate_mean": float(np.mean(gate[:, horizon])),
                "gate_std": float(np.std(gate[:, horizon])),
                "gate_q10": float(np.quantile(gate[:, horizon], 0.10)),
                "gate_q50": float(np.quantile(gate[:, horizon], 0.50)),
                "gate_q90": float(np.quantile(gate[:, horizon], 0.90)),
            }
            for role, values in (
                ("fused", forecast),
                ("candidate", candidate),
                ("persistence", persistence),
                ("corrected", corrected),
            ):
                metrics = _metric_values(
                    truth[:, horizon], values[:, horizon], reference
                )
                row.update({f"{role}_{key}": value for key, value in metrics.items()})
            rows.append(row)
        path = (
            Path(output_root)
            / "diagnostics"
            / model_id
            / f"{model_id}_{farm_id}_gate_candidate_by_horizon.csv"
        )
        _atomic_csv(rows, path)
        extra_paths["windprism_diagnostics"] = path
    elif model_id == "hr_moe_fets_patchtst" and "router_weights" in diagnostics:
        weights = diagnostics["router_weights"]
        if weights.ndim != 3 or weights.shape[1] != FORECAST_LEN:
            raise ValueError(f"HR-MoE router形状异常: {weights.shape}")
        if not np.isfinite(weights).all() or not np.allclose(
            weights.sum(axis=-1), 1.0, atol=1e-5
        ):
            raise ValueError("HR-MoE router权重无效")
        entropy = -np.sum(
            weights * np.log(np.clip(weights, 1e-8, 1.0)), axis=-1
        ) / np.log(weights.shape[-1])
        expert_names = ("long", "mid", "local", "persistence")
        for horizon in range(FORECAST_LEN):
            row = {
                "model_id": model_id,
                "farm_id": farm_id,
                "horizon": horizon + 1,
                "lead_minutes": (horizon + 1) * TIME_FREQ_MINUTES,
                "normalized_entropy_mean": float(np.mean(entropy[:, horizon])),
            }
            for idx in range(weights.shape[-1]):
                name = expert_names[idx] if idx < len(expert_names) else f"expert_{idx}"
                row[f"{name}_weight_mean"] = float(np.mean(weights[:, horizon, idx]))
                row[f"{name}_weight_std"] = float(np.std(weights[:, horizon, idx]))
            rows.append(row)
        path = (
            Path(output_root)
            / "diagnostics"
            / model_id
            / f"{model_id}_{farm_id}_router_by_horizon.csv"
        )
        _atomic_csv(rows, path)
        extra_paths["hr_router_diagnostics"] = path
    return rows, extra_paths


def _plot_task_visuals(
    output_root,
    model_id,
    farm_id,
    station,
    truth,
    prediction,
    horizon_rows,
    diagnostics,
):
    import numpy as np
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(output_root) / ".matplotlib_cache")
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(output_root) / "visualizations" / model_id / farm_id
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    h1_time = _timestamps_iso(
        station["timestamps_ns"][station["test_origins"]]
    )

    full_x = np.arange(len(h1_time))
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.plot(full_x, truth[:, 0], linewidth=0.7, label="Truth H1")
    ax.plot(
        full_x,
        prediction[:, 0],
        linewidth=0.65,
        alpha=0.85,
        label="Prediction H1",
    )
    ax.set_title(f"{MODEL_DISPLAY_NAMES[model_id]} / {farm_id} complete H1 test curve")
    ax.set_ylabel("Power (MW)")
    ax.grid(alpha=0.25)
    ax.legend()
    tick_count = min(8, len(h1_time))
    if tick_count:
        indices = np.linspace(0, len(h1_time) - 1, tick_count).astype(int)
        ax.set_xticks(indices)
        ax.set_xticklabels([h1_time[index][:10] for index in indices], rotation=30)
    path = directory / "complete_h1_prediction_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["complete_curve"] = path

    errors_h1 = prediction[:, 0] - truth[:, 0]
    center = int(np.argmax(np.abs(np.diff(truth[:, 0], prepend=truth[0, 0]))))
    radius = min(7 * 96 // 2, max(1, len(truth) // 2))
    start = max(0, center - radius)
    stop = min(len(truth), start + 7 * 96)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    local_x = np.arange(start, stop)
    ax.plot(local_x, truth[start:stop, 0], label="Truth H1")
    ax.plot(local_x, prediction[start:stop, 0], label="Prediction H1")
    ax.set_title(f"Representative dynamic window / {farm_id}")
    ax.set_xlabel("Test forecast origin")
    ax.set_ylabel("Power (MW)")
    ax.grid(alpha=0.25)
    ax.legend()
    path = directory / "representative_dynamic_window.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["local_window"] = path

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    flat_error = (prediction - truth).reshape(-1)
    axes[0].hist(flat_error, bins=80, alpha=0.85)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title("All-horizon error distribution")
    axes[0].set_xlabel("Prediction - truth (MW)")
    axes[0].grid(alpha=0.2)
    axes[1].hist(errors_h1, bins=60, alpha=0.85, color="tab:orange")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("H1 error distribution")
    axes[1].grid(alpha=0.2)
    path = directory / "error_distributions.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["error_distribution"] = path

    horizons = np.arange(1, FORECAST_LEN + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(horizons, [row["nrmse"] for row in horizon_rows], marker="o")
    axes[1].plot(horizons, [row["nmae"] for row in horizon_rows], marker="o")
    axes[0].set_ylabel("NRMSE")
    axes[1].set_ylabel("NMAE")
    axes[1].set_xlabel("Forecast horizon")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Horizon metrics / {MODEL_DISPLAY_NAMES[model_id]} / {farm_id}")
    path = directory / "horizon_metrics.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["horizon_metrics"] = path

    if model_id == "windprism_f7_g0" and "gate" in diagnostics:
        gate = diagnostics["gate"]
        fig, axes = plt.subplots(2, 1, figsize=(9, 7))
        axes[0].plot(horizons, np.mean(gate, axis=0), marker="o")
        axes[0].fill_between(
            horizons,
            np.quantile(gate, 0.1, axis=0),
            np.quantile(gate, 0.9, axis=0),
            alpha=0.2,
            label="Q10--Q90",
        )
        axes[0].set_ylabel("Corrected-candidate gate")
        axes[0].legend()
        axes[1].hist(gate.reshape(-1), bins=60)
        axes[1].set_xlabel("Gate value")
        for ax in axes:
            ax.grid(alpha=0.25)
        path = directory / "windprism_gate_diagnostics.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths["gate_visualization"] = path
    elif model_id == "hr_moe_fets_patchtst" and "router_weights" in diagnostics:
        weights = diagnostics["router_weights"]
        mean = weights.mean(axis=0)
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = ["long", "mid", "local", "persistence"][: mean.shape[1]]
        ax.stackplot(horizons, mean.T, labels=labels, alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("Mean router weight")
        ax.legend(loc="upper left", ncol=2)
        ax.grid(alpha=0.2)
        path = directory / "hr_moe_router_diagnostics.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths["router_visualization"] = path
    return paths


def _task_metrics(model_id, farm_id, station, truth, prediction, prediction_scaled):
    import numpy as np
    reference = station["power_reference_mw"]
    overall = _metric_values(truth, prediction, reference)
    if station["power_reference_kind"] == "train_power_q999":
        overall["trnmae"] = overall["nmae"]
        overall["trnrmse"] = overall["nrmse"]
    row = {
        "model_id": model_id,
        "model_display_name": MODEL_DISPLAY_NAMES[model_id],
        "farm_id": farm_id,
        "scope": "overall",
        "horizon": 0,
        "lead_minutes": 0,
        "n_samples": int(truth.shape[0]),
        "n_points": int(truth.size),
        "power_reference_mw": reference,
        "power_reference_kind": station["power_reference_kind"],
        "test_huber_scaled": _huber_mean(
            station["target_scaled"][station["target_indices"]],
            prediction_scaled,
        ),
        **overall,
    }
    horizon_rows = []
    for horizon in range(FORECAST_LEN):
        horizon_metrics = _metric_values(
            truth[:, horizon], prediction[:, horizon], reference
        )
        if station["power_reference_kind"] == "train_power_q999":
            horizon_metrics["trnmae"] = horizon_metrics["nmae"]
            horizon_metrics["trnrmse"] = horizon_metrics["nrmse"]
        horizon_rows.append(
            {
                "model_id": model_id,
                "model_display_name": MODEL_DISPLAY_NAMES[model_id],
                "farm_id": farm_id,
                "scope": "horizon",
                "horizon": horizon + 1,
                "lead_minutes": (horizon + 1) * TIME_FREQ_MINUTES,
                "n_samples": int(truth.shape[0]),
                "n_points": int(truth.shape[0]),
                "power_reference_mw": reference,
                "power_reference_kind": station["power_reference_kind"],
                **horizon_metrics,
            }
        )
    error = prediction - truth
    row.update(
        {
            "sum_abs_error": float(np.sum(np.abs(error))),
            "sum_squared_error": float(np.sum(np.square(error))),
            "sum_normalized_abs_error": float(
                np.sum(np.abs(error) / reference)
            ),
            "sum_normalized_squared_error": float(
                np.sum(np.square(error / reference))
            ),
            "truth_sum": float(np.sum(truth)),
            "truth_squared_sum": float(np.sum(np.square(truth))),
        }
    )
    return row, horizon_rows


def _final_paths(output_root, model_id, farm_id):
    root = Path(output_root)
    return {
        "sample": root / "per_sample" / model_id / f"{model_id}_{farm_id}_test_predictions.csv.gz",
        "farm_metrics": root / "per_farm" / model_id / f"{model_id}_{farm_id}_metrics.csv",
        "horizon_metrics": root / "per_horizon" / model_id / f"{model_id}_{farm_id}_horizon_metrics.csv",
    }


def _prediction_marker_valid(
    path,
    frozen_task_hash,
    array_hash,
    *,
    expected_model_id=None,
    expected_farm_id=None,
    expected_formal=None,
    expected_bundle_hash=None,
    expected_snapshot_hash=None,
):
    if not Path(path).is_file():
        return None
    try:
        marker = _read_json(path)
        if not _completion_declared(marker):
            return None
        if marker.get("protocol_version") != PROTOCOL_VERSION:
            return None
        if (
            expected_model_id is not None
            and marker.get("model_id") != expected_model_id
        ):
            return None
        if (
            expected_farm_id is not None
            and marker.get("farm_id") != expected_farm_id
        ):
            return None
        if (
            expected_formal is not None
            and marker.get("formal") is not bool(expected_formal)
        ):
            return None
        if marker.get("training_task_marker_sha256") != frozen_task_hash:
            return None
        if marker.get("test_array_sha256") != array_hash:
            return None
        if (
            expected_bundle_hash is not None
            and marker.get("preprocessing_bundle_sha256")
            != expected_bundle_hash
        ):
            return None
        if (
            expected_snapshot_hash is not None
            and marker.get("frozen_snapshot_sha256")
            != expected_snapshot_hash
        ):
            return None
        if not marker.get("power_reference_kind"):
            return None
        if "test_reuse_status" not in marker:
            return None
        horizon_rows = marker.get("horizon_metrics", ())
        if len(horizon_rows) != FORECAST_LEN:
            return None
        if {
            int(row.get("horizon", -1))
            for row in horizon_rows
        } != set(range(1, FORECAST_LEN + 1)):
            return None
        required_outputs = {
            "sample_predictions",
            "farm_metrics",
            "horizon_metrics",
        }
        if not required_outputs.issubset(marker.get("output_files", {})):
            return None
        for record in marker.get("output_files", {}).values():
            if isinstance(record, dict) and record.get("path"):
                _validate_record(record)
        return marker
    except Exception:
        return None


def _run_prediction_task(
    tf,
    keras,
    output_root,
    model_id,
    farm_id,
    station,
    frozen_task,
    batch_size,
    formal,
    snapshot_record,
):
    import numpy as np

    keras.backend.clear_session()
    gc.collect()
    custom_objects = _register_custom_layers(model_id)
    model_path = _validate_record(frozen_task["model_record"])
    model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
    )
    if len(model.inputs) != 1:
        raise ValueError(f"{model_id}/{farm_id}模型输入数量不是1")
    if tuple(model.input_shape[1:]) != (HISTORY_LEN, EXPECTED_INPUT_DIM):
        raise ValueError(
            f"{model_id}/{farm_id}输入形状漂移: {model.input_shape}"
        )
    dataset = _make_test_dataset(tf, station, batch_size)
    prediction_scaled, diagnostics, elapsed, peak_gpu = _predict_with_diagnostics(
        tf, keras, model, model_id, dataset
    )
    expected_shape = (len(station["test_origins"]), FORECAST_LEN)
    if prediction_scaled.shape != expected_shape:
        raise ValueError(
            f"{model_id}/{farm_id}预测形状{prediction_scaled.shape} != {expected_shape}"
        )
    if not np.isfinite(prediction_scaled).all():
        raise ValueError(f"{model_id}/{farm_id}预测含非有限值")
    truth = station["target_mw"][station["target_indices"]]
    prediction = np.clip(_inverse_scaled(station, prediction_scaled), 0.0, None)
    overall, horizon_rows = _task_metrics(
        model_id, farm_id, station, truth, prediction, prediction_scaled
    )
    overall.update(
        {
            "inference_seconds": elapsed,
            "samples_per_second": len(truth) / max(elapsed, 1e-9),
            "forecast_points_per_second": truth.size / max(elapsed, 1e-9),
            "peak_gpu_memory_bytes": peak_gpu,
            "total_params": int(model.count_params()),
            "model_size_bytes": int(model_path.stat().st_size),
            "effective_batch_size": int(batch_size),
        }
    )
    paths = _final_paths(output_root, model_id, farm_id)
    sample_path = _save_sample_predictions(
        output_root, model_id, farm_id, station, truth, prediction
    )
    _atomic_csv([overall], paths["farm_metrics"])
    _atomic_csv(horizon_rows, paths["horizon_metrics"])
    _, diagnostic_paths = _save_diagnostic_tables(
        output_root,
        model_id,
        farm_id,
        station,
        truth,
        prediction,
        diagnostics,
    )
    visual_paths = _plot_task_visuals(
        output_root,
        model_id,
        farm_id,
        station,
        truth,
        prediction,
        horizon_rows,
        diagnostics,
    )
    output_files = {
        "sample_predictions": _file_record(sample_path),
        "farm_metrics": _file_record(paths["farm_metrics"]),
        "horizon_metrics": _file_record(paths["horizon_metrics"]),
    }
    output_files.update(
        {name: _file_record(path) for name, path in diagnostic_paths.items()}
    )
    output_files.update(
        {f"plot_{name}": _file_record(path) for name, path in visual_paths.items()}
    )
    marker = {
        "status": "complete",
        "complete": True,
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": _utc_now(),
        "model_id": model_id,
        "model_display_name": MODEL_DISPLAY_NAMES[model_id],
        "farm_id": farm_id,
        "formal": bool(formal),
        "training_task_marker_path": frozen_task["path"],
        "training_task_marker_sha256": frozen_task["sha256"],
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "preprocessing_bundle_path": station["bundle_path"],
        "preprocessing_bundle_sha256": station["bundle_sha256"],
        "test_array_path": station["array_path"],
        "test_array_sha256": station["array_sha256"],
        "schema_hash": station["schema_hash"],
        "frozen_snapshot_path": snapshot_record["path"],
        "frozen_snapshot_sha256": snapshot_record["sha256"],
        "power_reference_kind": station["power_reference_kind"],
        "test_samples": len(station["test_origins"]),
        "batch_size": int(batch_size),
        **_test_evaluation_provenance(formal),
        "metrics": overall,
        "horizon_metrics": horizon_rows,
        "output_files": output_files,
    }
    marker_path = _prediction_marker_path(output_root, model_id, farm_id)
    _atomic_json(marker, marker_path)
    print(
        f"{model_id}/{farm_id}: NRMSE={overall['nrmse']:.6f}, "
        f"NMAE={overall['nmae']:.6f}, R2={overall['r2']:.6f}"
    )
    del model, dataset, diagnostics, prediction_scaled, prediction, truth
    keras.backend.clear_session()
    gc.collect()
    return marker


def _collect_prediction_markers(output_root, models, farms):
    markers = []
    for farm_id in farms:
        for model_id in models:
            path = _prediction_marker_path(output_root, model_id, farm_id)
            marker = _read_json(path)
            if not _completion_declared(marker):
                raise ValueError(f"预测task未完成: {path}")
            markers.append(marker)
    return markers


def _macro_micro_rows(per_farm_rows, models):
    import numpy as np
    rows = []
    metric_names = (
        "mae",
        "rmse",
        "nmae",
        "nrmse",
        "trnmae",
        "trnrmse",
        "r2",
        "smape",
        "mape",
    )
    for model_id in models:
        selected = [row for row in per_farm_rows if row["model_id"] == model_id]
        if not selected:
            continue
        macro = {
            "model_id": model_id,
            "model_display_name": MODEL_DISPLAY_NAMES[model_id],
            "aggregation": "macro_equal_farm",
            "farm_count": len(selected),
        }
        for metric in metric_names:
            values = np.asarray(
                [row.get(metric, np.nan) for row in selected], dtype=float
            )
            macro[metric] = float(np.nanmean(values))
            macro[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
            macro[f"{metric}_median"] = float(np.nanmedian(values))
        rows.append(macro)

        n = int(sum(int(row["n_points"]) for row in selected))
        sum_abs = sum(float(row["sum_abs_error"]) for row in selected)
        sum_sq = sum(float(row["sum_squared_error"]) for row in selected)
        sum_norm_abs = sum(
            float(row["sum_normalized_abs_error"]) for row in selected
        )
        sum_norm_sq = sum(
            float(row["sum_normalized_squared_error"]) for row in selected
        )
        truth_sum = sum(float(row["truth_sum"]) for row in selected)
        truth_sq_sum = sum(float(row["truth_squared_sum"]) for row in selected)
        sst = truth_sq_sum - truth_sum * truth_sum / max(n, 1)
        micro = {
            "model_id": model_id,
            "model_display_name": MODEL_DISPLAY_NAMES[model_id],
            "aggregation": "micro_pooled_points",
            "farm_count": len(selected),
            "n_points": n,
            "mae": sum_abs / n,
            "rmse": math.sqrt(sum_sq / n),
            "nmae": sum_norm_abs / n,
            "nrmse": math.sqrt(sum_norm_sq / n),
            "r2": 1.0 - sum_sq / sst if sst > 1e-12 else float("nan"),
        }
        if all("trnmae" in row and "trnrmse" in row for row in selected):
            micro["trnmae"] = micro["nmae"]
            micro["trnrmse"] = micro["nrmse"]
        rows.append(micro)
    return rows


def _rank_and_win_rows(per_farm_rows, models, farms):
    import numpy as np
    rank_rows = []
    rank_values = {model_id: [] for model_id in models}
    lookup = {
        (row["model_id"], row["farm_id"]): row for row in per_farm_rows
    }
    for farm_id in farms:
        ordered = sorted(
            models,
            key=lambda model_id: (
                float(lookup[(model_id, farm_id)]["nrmse"]),
                float(lookup[(model_id, farm_id)]["nmae"]),
            ),
        )
        previous = None
        previous_rank = None
        for position, model_id in enumerate(ordered, 1):
            value = float(lookup[(model_id, farm_id)]["nrmse"])
            if previous is not None and abs(value - previous) <= TIE_TOLERANCE:
                rank = previous_rank
            else:
                rank = position
            rank_values[model_id].append(float(rank))
            rank_rows.append(
                {
                    "farm_id": farm_id,
                    "model_id": model_id,
                    "model_display_name": MODEL_DISPLAY_NAMES[model_id],
                    "nrmse": value,
                    "nmae": float(lookup[(model_id, farm_id)]["nmae"]),
                    "rank": rank,
                }
            )
            previous = value
            previous_rank = rank
    average_rows = [
        {
            "model_id": model_id,
            "model_display_name": MODEL_DISPLAY_NAMES[model_id],
            "average_rank": float(np.mean(rank_values[model_id])),
            "median_rank": float(np.median(rank_values[model_id])),
            "best_rank_count": int(np.sum(np.asarray(rank_values[model_id]) == 1)),
        }
        for model_id in models
    ]
    win_rows = []
    for left_index, left in enumerate(models):
        for right in models[left_index + 1 :]:
            delta = np.asarray(
                [
                    float(lookup[(left, farm)]["nrmse"])
                    - float(lookup[(right, farm)]["nrmse"])
                    for farm in farms
                ]
            )
            win_rows.append(
                {
                    "metric": "nrmse",
                    "model_a": left,
                    "model_b": right,
                    "a_wins": int(np.sum(delta < -TIE_TOLERANCE)),
                    "ties": int(np.sum(np.abs(delta) <= TIE_TOLERANCE)),
                    "a_losses": int(np.sum(delta > TIE_TOLERANCE)),
                    "mean_a_minus_b": float(np.mean(delta)),
                    "tie_tolerance": TIE_TOLERANCE,
                }
            )
    return rank_rows, average_rows, win_rows


def _holm_adjust(p_values):
    import numpy as np
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def _significance_rows(per_farm_rows, models, farms):
    import numpy as np
    lookup = {
        (row["model_id"], row["farm_id"]): row for row in per_farm_rows
    }
    rng = np.random.default_rng(SEED)
    rows = []
    raw_p = []
    for baseline in models:
        if baseline == PRIMARY_MODEL_ID:
            continue
        delta = np.asarray(
            [
                float(lookup[(PRIMARY_MODEL_ID, farm)]["nrmse"])
                - float(lookup[(baseline, farm)]["nrmse"])
                for farm in farms
            ],
            dtype=float,
        )
        method = "wilcoxon_signed_rank"
        try:
            from scipy.stats import wilcoxon
            result = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
        except Exception:
            method = "exact_sign_test_fallback"
            nonzero = delta[np.abs(delta) > TIE_TOLERANCE]
            wins = int(np.sum(nonzero < 0))
            n = len(nonzero)
            tail = sum(math.comb(n, k) for k in range(0, min(wins, n - wins) + 1))
            p_value = min(1.0, 2.0 * tail / (2**n)) if n else 1.0
            statistic = float(wins)
        bootstrap = []
        for _ in range(10000):
            bootstrap.append(float(np.mean(rng.choice(delta, size=len(delta), replace=True))))
        row = {
            "primary_model": PRIMARY_MODEL_ID,
            "baseline_model": baseline,
            "metric": "nrmse",
            "delta_definition": "WindPRISM_minus_baseline",
            "mean_delta": float(np.mean(delta)),
            "median_delta": float(np.median(delta)),
            "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
            "test_method": method,
            "statistic": statistic,
            "p_value_raw": p_value,
            "farm_count": len(delta),
        }
        rows.append(row)
        raw_p.append(p_value)
    adjusted = _holm_adjust(raw_p)
    for row, value in zip(rows, adjusted):
        row["p_value_holm"] = float(value)
        row["significant_at_0_05"] = bool(value < 0.05)
    return rows


def _complexity_rows(markers, average_rank_rows, macro_micro):
    import numpy as np
    task_rows = []
    rank_lookup = {row["model_id"]: row["average_rank"] for row in average_rank_rows}
    macro_lookup = {
        row["model_id"]: row
        for row in macro_micro
        if row["aggregation"] == "macro_equal_farm"
    }
    for marker in markers:
        metric = marker["metrics"]
        task = marker
        task_rows.append(
            {
                "model_id": task["model_id"],
                "model_display_name": task["model_display_name"],
                "farm_id": task["farm_id"],
                "total_params": metric.get("total_params"),
                "model_size_bytes": metric.get("model_size_bytes"),
                "inference_seconds": metric.get("inference_seconds"),
                "samples_per_second": metric.get("samples_per_second"),
                "forecast_points_per_second": metric.get("forecast_points_per_second"),
                "peak_gpu_memory_bytes": metric.get("peak_gpu_memory_bytes"),
                "effective_batch_size": metric.get("effective_batch_size"),
            }
        )
    summary = []
    for model_id in sorted({row["model_id"] for row in task_rows}):
        selected = [row for row in task_rows if row["model_id"] == model_id]
        params = [float(row["total_params"]) for row in selected]
        summary.append(
            {
                "model_id": model_id,
                "model_display_name": MODEL_DISPLAY_NAMES[model_id],
                "farm_count": len(selected),
                "total_params": int(round(np.median(params))),
                "mean_model_size_bytes": float(
                    np.mean([row["model_size_bytes"] for row in selected])
                ),
                "mean_inference_seconds": float(
                    np.mean([row["inference_seconds"] for row in selected])
                ),
                "mean_samples_per_second": float(
                    np.mean([row["samples_per_second"] for row in selected])
                ),
                "macro_nrmse": macro_lookup[model_id]["nrmse"],
                "macro_nmae": macro_lookup[model_id]["nmae"],
                "average_rank": rank_lookup[model_id],
            }
        )
    for row in summary:
        row["pareto_nrmse_params"] = not any(
            other["total_params"] <= row["total_params"]
            and other["macro_nrmse"] <= row["macro_nrmse"]
            and (
                other["total_params"] < row["total_params"]
                or other["macro_nrmse"] < row["macro_nrmse"]
            )
            for other in summary
        )
    return task_rows, summary


def _plot_overview(output_root, per_farm, macro_micro, average_ranks, complexity, models, farms):
    import numpy as np
    os.environ.setdefault("MPLCONFIGDIR", str(Path(output_root) / ".matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(output_root) / "visualizations" / "overview"
    directory.mkdir(parents=True, exist_ok=True)
    lookup = {(row["model_id"], row["farm_id"]): row for row in per_farm}
    matrix = np.asarray(
        [[lookup[(model, farm)]["nrmse"] for farm in farms] for model in models],
        dtype=float,
    )
    paths = {}
    fig, ax = plt.subplots(figsize=(16, 7))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(farms)))
    ax.set_xticklabels(farms, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_DISPLAY_NAMES[item] for item in models])
    ax.set_title("Test NRMSE heatmap (model × farm)")
    fig.colorbar(image, ax=ax, label="NRMSE")
    path = directory / "nrmse_model_farm_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    paths["nrmse_heatmap"] = path

    macro = {
        row["model_id"]: row
        for row in macro_micro
        if row["aggregation"] == "macro_equal_farm"
    }
    order = sorted(models, key=lambda item: macro[item]["nrmse"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    x = np.arange(len(order))
    axes[0].bar(x, [macro[item]["nrmse"] for item in order])
    axes[1].bar(x, [macro[item]["nmae"] for item in order], color="tab:orange")
    axes[0].set_ylabel("Macro NRMSE")
    axes[1].set_ylabel("Macro NMAE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([MODEL_DISPLAY_NAMES[item] for item in order], rotation=35, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    path = directory / "macro_nrmse_nmae_bars.png"
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    paths["macro_bars"] = path

    fig, ax = plt.subplots(figsize=(10, 6))
    for row in complexity:
        ax.scatter(row["total_params"], row["macro_nrmse"], s=70)
        ax.annotate(
            row["model_id"],
            (row["total_params"], row["macro_nrmse"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Trainable model parameters (log scale)")
    ax.set_ylabel("Macro NRMSE")
    ax.set_title("Accuracy–complexity Pareto view")
    ax.grid(alpha=0.25)
    path = directory / "parameter_nrmse_pareto.png"
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    paths["pareto"] = path

    ordered_rank = sorted(average_ranks, key=lambda row: row["average_rank"])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(
        range(len(ordered_rank)),
        [row["average_rank"] for row in ordered_rank],
        color="tab:green",
    )
    ax.set_xticks(range(len(ordered_rank)))
    ax.set_xticklabels(
        [row["model_display_name"] for row in ordered_rank], rotation=35, ha="right"
    )
    ax.set_ylabel("Average farm rank (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    path = directory / "average_farm_rank.png"
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    paths["average_rank"] = path
    return paths


def _save_final_selection(
    output_root,
    per_farm,
    macro_micro,
    average_ranks,
    complexity_summary,
    formal,
):
    """Select the test winner using the frozen equal-farm protocol."""
    macro = {
        row["model_id"]: row
        for row in macro_micro
        if row["aggregation"] == "macro_equal_farm"
    }
    rank = {row["model_id"]: row for row in average_ranks}
    complexity = {row["model_id"]: row for row in complexity_summary}
    farm_rows = {}
    for row in per_farm:
        farm_rows.setdefault(row["model_id"], []).append(row)
    ranking = []
    for model_id, row in macro.items():
        model_farms = farm_rows[model_id]
        ranking.append(
            {
                "model_id": model_id,
                "model_display_name": MODEL_DISPLAY_NAMES[model_id],
                "macro_nrmse": float(row["nrmse"]),
                "macro_nmae": float(row["nmae"]),
                "macro_r2": float(row["r2"]),
                "worst_farm_nrmse": float(
                    max(item["nrmse"] for item in model_farms)
                ),
                "average_rank": float(rank[model_id]["average_rank"]),
                "farm_rank1_count": int(rank[model_id]["best_rank_count"]),
                "total_params": int(complexity[model_id]["total_params"]),
            }
        )
    ranking.sort(
        key=lambda item: (
            item["macro_nrmse"],
            item["macro_nmae"],
            item["average_rank"],
            item["total_params"],
        )
    )
    for position, row in enumerate(ranking, 1):
        row["final_rank"] = position
    selected = ranking[0]
    payload = {
        "status": "complete" if formal else "diagnostic_complete",
        "created_at_utc": _utc_now(),
        "selection_scope": "frozen test sets of requested farms",
        "primary_metric": "equal-farm macro NRMSE",
        "secondary_metric": "equal-farm macro NMAE",
        "tertiary_metric": "average per-farm NRMSE rank",
        "final_tiebreaker": "total parameter count",
        **_test_evaluation_provenance(formal),
        "selection_eligible": bool(formal),
        "publication_result_eligible": bool(formal),
        "ranking": ranking,
    }
    root = Path(output_root)
    if formal:
        payload.update(
            {
                "selected_model_id": selected["model_id"],
                "selected_model_display_name": selected[
                    "model_display_name"
                ],
                "selected_metrics": selected,
            }
        )
        json_path = root / "round3_external14_test_final_selection.json"
        markdown_path = root / "round3_external14_test_final_selection.md"
        heading = "# Part 3 Round 3 test-set final selection"
        lead = (
            f"Selected model: **{selected['model_display_name']} "
            f"(`{selected['model_id']}`)**."
        )
    else:
        payload.update(
            {
                "diagnostic_top_model_id": selected["model_id"],
                "diagnostic_top_metrics": selected,
                "selected_model_id": None,
            }
        )
        json_path = root / "round3_nonformal_diagnostic_ranking.json"
        markdown_path = root / "round3_nonformal_diagnostic_ranking.md"
        heading = "# Part 3 Round 3 non-formal diagnostic ranking"
        lead = (
            f"Diagnostic top row: **{selected['model_display_name']} "
            f"(`{selected['model_id']}`)**. This is not a final selection."
        )
    _atomic_json(payload, json_path)
    lines = [
        heading,
        "",
        lead,
        "",
        (
            "The frozen rule minimizes equal-farm Macro NRMSE, then Macro "
            "NMAE, average per-farm NRMSE rank, and finally parameter count."
        ),
        "",
        "| Rank | Model | Macro NRMSE | Macro NMAE | Macro R2 | "
        "Worst-farm NRMSE | Avg. rank | Farm wins | Parameters |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranking:
        lines.append(
            "| {final_rank} | {model_display_name} | {macro_nrmse:.6f} | "
            "{macro_nmae:.6f} | {macro_r2:.6f} | "
            "{worst_farm_nrmse:.6f} | {average_rank:.3f} | "
            "{farm_rank1_count} | {total_params:,} |".format(**row)
        )
    lines.extend(
        [
            "",
            (
                "This selection is descriptive for the frozen test protocol; "
                "the paired significance table must be consulted before "
                "claiming statistical superiority."
            ),
            (
                "It is not labelled globally blind: JSFD001--JSFD014 had "
                "historical exposure in a separate teacher-data workflow, "
                + (
                    "while formal Round-3 evaluation starts only after its "
                    "140 from-scratch tasks are frozen."
                    if formal
                    else "and this partial/smoke ranking is diagnostic only."
                )
            ),
            "",
        ]
    )
    _atomic_text("\n".join(lines), markdown_path)
    return {
        "selection_json": json_path,
        "selection_markdown": markdown_path,
    }


def _aggregate_and_save(output_root, markers, models, farms, formal):
    per_farm = [dict(marker["metrics"]) for marker in markers]
    per_horizon = [
        dict(row) for marker in markers for row in marker["horizon_metrics"]
    ]
    macro_micro = _macro_micro_rows(per_farm, models)
    rank_rows, average_ranks, win_rows = _rank_and_win_rows(
        per_farm, models, farms
    )
    significance = (
        _significance_rows(per_farm, models, farms)
        if PRIMARY_MODEL_ID in models and len(farms) >= 5
        else []
    )
    complexity_tasks, complexity_summary = _complexity_rows(
        markers, average_ranks, macro_micro
    )
    root = Path(output_root)
    paths = {
        "per_farm": _atomic_csv(
            per_farm, root / "round3_external14_test_metrics_per_farm.csv"
        ),
        "per_horizon": _atomic_csv(
            per_horizon, root / "round3_external14_test_metrics_by_horizon.csv"
        ),
        "macro_micro": _atomic_csv(
            macro_micro, root / "round3_external14_test_macro_micro.csv"
        ),
        "ranks": _atomic_csv(
            rank_rows, root / "round3_external14_per_farm_rank.csv"
        ),
        "average_ranks": _atomic_csv(
            average_ranks, root / "round3_external14_average_rank.csv"
        ),
        "win_tie_loss": _atomic_csv(
            win_rows, root / "round3_external14_win_tie_loss.csv"
        ),
        "complexity_tasks": _atomic_csv(
            complexity_tasks, root / "round3_external14_complexity_per_farm.csv"
        ),
        "complexity_summary": _atomic_csv(
            complexity_summary, root / "round3_external14_complexity.csv"
        ),
    }
    if significance:
        paths["significance"] = _atomic_csv(
            significance, root / "round3_external14_significance.csv"
        )
    plot_paths = _plot_overview(
        output_root,
        per_farm,
        macro_micro,
        average_ranks,
        complexity_summary,
        models,
        farms,
    )
    paths.update({f"plot_{key}": value for key, value in plot_paths.items()})
    paths.update(
        _save_final_selection(
            output_root,
            per_farm,
            macro_micro,
            average_ranks,
            complexity_summary,
            formal,
        )
    )
    return paths


def _write_inventory(output_root):
    root = Path(output_root).resolve()
    inventory_path = root / "round3_external14_output_inventory.csv"
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path == inventory_path or path.name == PREDICTION_COMPLETE_NAME:
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    _atomic_csv(rows, inventory_path)
    return inventory_path, rows


def _validate_inventory(output_root, inventory_record):
    """Revalidate every inventoried file and reject missing/extra outputs."""
    root = Path(output_root).resolve()
    inventory_path = _validate_record(inventory_record)
    with open(inventory_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    declared = set()
    for row in rows:
        relative = str(row["relative_path"])
        if relative in declared:
            raise ValueError(f"inventory包含重复路径: {relative}")
        declared.add(relative)
        _validate_record(
            {
                "path": root / relative,
                "size_bytes": int(row["size_bytes"]),
                "sha256": row["sha256"],
            }
        )
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.resolve() != inventory_path.resolve()
        and path.name != PREDICTION_COMPLETE_NAME
    }
    if actual != declared:
        raise ValueError(
            "prediction inventory与当前输出树不一致: "
            f"missing={sorted(declared - actual)[:5]}, "
            f"extra={sorted(actual - declared)[:5]}"
        )


def _freeze_snapshot(output_root, frozen, preprocess_record, formal):
    path = Path(output_root) / "manifests" / "frozen_training_snapshot.json"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "formal": bool(formal),
        "created_at_utc": _utc_now(),
        "preprocess_complete": preprocess_record,
        "training_complete": frozen.get("training_complete"),
        "tasks": {
            f"{model_id}/{farm_id}": {
                "path": task["path"],
                "sha256": task["sha256"],
                "model": task["model_record"],
            }
            for (model_id, farm_id), task in sorted(frozen["tasks"].items())
        },
    }
    if path.is_file():
        existing = _read_json(path)
        comparable_existing = dict(existing)
        comparable_payload = dict(payload)
        comparable_existing.pop("created_at_utc", None)
        comparable_payload.pop("created_at_utc", None)
        if comparable_existing != comparable_payload:
            raise ValueError(
                "已有预测输出的冻结训练快照与当前训练产物不同；"
                "拒绝在同一目录混合测试结果"
            )
        return path
    _atomic_json(payload, path)
    return path


def _revalidate_frozen(frozen):
    if frozen.get("training_complete"):
        _validate_record(frozen["training_complete"])
    for task in frozen["tasks"].values():
        path = Path(task["path"])
        if _sha256(path) != task["sha256"]:
            raise ValueError(f"预测期间训练task marker发生变化: {path}")
        _validate_record(task["model_record"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Part 3 Round 3 JSFD14 ten-model frozen test prediction"
    )
    parser.add_argument("--models", nargs="*", help="模型ID，空格或逗号分隔")
    parser.add_argument("--farms", nargs="*", help="JSFD场站ID，空格或逗号分隔")
    parser.add_argument("--partial", action="store_true", help="局部诊断；隔离输出")
    parser.add_argument("--smoke", action="store_true", help="首任务少量样本冒烟")
    parser.add_argument("--resume", action="store_true", help="复用通过哈希校验的预测task")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--smoke-samples", type=int, default=64)
    parser.add_argument("--result-root", default=str(RESULT_ROOT))
    parser.add_argument("--output-root", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.result_root).resolve()
    models = _normalize_requested(args.models, MODEL_IDS, "模型")
    farms = _normalize_requested(args.farms, EXPECTED_FARMS, "场站")
    if args.batch_size <= 0:
        raise ValueError("batch-size必须为正整数")
    if args.smoke:
        if not args.models:
            models = [MODEL_IDS[0]]
        if not args.farms:
            farms = [EXPECTED_FARMS[0]]
    subset = set(models) != set(MODEL_IDS) or set(farms) != set(EXPECTED_FARMS)
    formal = not args.partial and not args.smoke and not subset
    if subset and not (args.partial or args.smoke):
        raise ValueError("模型/场站子集预测必须显式使用--partial或--smoke")

    preprocess_path, _ = _validate_preprocess_complete(root)
    preprocess_record = _file_record(preprocess_path)
    # 关键因果门：正式模式在下方验证完140个冻结任务前，不读取任何NPZ测试值。
    frozen = (
        _validate_all_training_frozen(root)
        if formal
        else _validate_selected_training(root, models, farms)
    )
    if formal:
        output_root = root / PREDICTION_DIRNAME
    elif args.output_root:
        output_root = Path(args.output_root).resolve()
        formal_output_root = (root / PREDICTION_DIRNAME).resolve()
        if (
            output_root == formal_output_root
            or formal_output_root in output_root.parents
        ):
            raise ValueError(
                "partial/smoke输出不能位于正式testdata_predict_output目录树"
            )
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "smoke" if args.smoke else "partial"
        output_root = (
            root / "partial_runs" / "prediction" / f"{stamp}_{tag}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = _freeze_snapshot(
        output_root,
        frozen,
        preprocess_record,
        formal,
    )
    snapshot_record = _file_record(snapshot_path)
    complete_path = output_root / PREDICTION_COMPLETE_NAME
    if formal and complete_path.is_file():
        existing = _read_json(complete_path)
        if _completion_declared(existing):
            if existing.get("protocol_version") != PROTOCOL_VERSION:
                print(
                    "检测到旧协议正式预测marker；将在相同冻结训练条件下"
                    "重新生成v2预测归档"
                )
            else:
                if int(existing.get("task_count", -1)) != (
                    len(MODEL_IDS) * len(EXPECTED_FARMS)
                ):
                    raise ValueError("已有正式预测complete marker的任务数不完整")
                for record in existing.get("summary_files", {}).values():
                    _validate_record(record)
                _validate_inventory(output_root, existing["inventory"])
                for farm_id in EXPECTED_FARMS:
                    for model_id in MODEL_IDS:
                        marker_path = _prediction_marker_path(
                            output_root,
                            model_id,
                            farm_id,
                        )
                        marker = _read_json(marker_path)
                        task = frozen["tasks"][(model_id, farm_id)]
                        if _prediction_marker_valid(
                            marker_path,
                            task["sha256"],
                            _sha256(marker["test_array_path"]),
                            expected_model_id=model_id,
                            expected_farm_id=farm_id,
                            expected_formal=True,
                            expected_bundle_hash=_sha256(
                                marker["preprocessing_bundle_path"]
                            ),
                            expected_snapshot_hash=snapshot_record["sha256"],
                        ) is None:
                            raise ValueError(
                                f"已有正式预测task恢复校验失败: {marker_path}"
                            )
                _revalidate_frozen(frozen)
                if (
                    "test_reuse_status" in existing
                    and "test_is_final_blind_evaluation" in existing
                ):
                    print(
                        "正式测试bundle已经完成且冻结产物未变化: "
                        f"{complete_path}"
                    )
                    return 0
                print(
                    "检测到旧版正式预测marker缺少测试使用声明；"
                    "将在相同冻结快照下重新生成预测归档"
                )
    _atomic_json(
        {
            "protocol_version": PROTOCOL_VERSION,
            "created_at_utc": _utc_now(),
            "formal": formal,
            "models": models,
            "farms": farms,
            "history_len": HISTORY_LEN,
            "forecast_len": FORECAST_LEN,
            "time_frequency_minutes": TIME_FREQ_MINUTES,
            "batch_size": args.batch_size,
            "prediction_postprocessing": "inverse train-only scaler; lower clip at 0 MW; no upper clipping",
            "normalization": "per-farm declared capacity when verified, otherwise train-only power reference",
            "primary_selection_metric": "equal-farm macro NRMSE",
            "secondary_selection_metric": "equal-farm macro NMAE",
            "mape_truth_floor": "1% of per-farm power reference",
            "rank_tie_tolerance_nrmse": TIE_TOLERANCE,
            "significance": (
                "farm-level paired Wilcoxon; Holm correction; "
                "10,000-replicate paired farm bootstrap"
            ),
            "formal_test_gate": (
                "all 140 training tasks and the training complete marker "
                "validated before any test NPZ values are read"
            ),
            **_test_evaluation_provenance(formal),
        },
        output_root / "manifests" / "round3_prediction_protocol.json",
    )

    tf, keras = _configure_tensorflow()
    completed = []
    reused_prediction_task_count = 0
    new_prediction_task_count = 0
    failures = []
    for farm_id in farms:
        print(f"\n===== Round 3 test farm={farm_id} =====")
        station = _load_station_bundle(
            root,
            farm_id,
            smoke_limit=args.smoke_samples if args.smoke else None,
        )
        _assert_station_matches_training(
            station, frozen["tasks"], models, farm_id, formal
        )
        for model_id in models:
            task = frozen["tasks"][(model_id, farm_id)]
            marker_path = _prediction_marker_path(output_root, model_id, farm_id)
            cached = (
                _prediction_marker_valid(
                    marker_path,
                    task["sha256"],
                    station["array_sha256"],
                    expected_model_id=model_id,
                    expected_farm_id=farm_id,
                    expected_formal=formal,
                    expected_bundle_hash=station["bundle_sha256"],
                    expected_snapshot_hash=snapshot_record["sha256"],
                )
                if args.resume
                else None
            )
            if cached is not None:
                print(f"resume跳过 {model_id}/{farm_id}")
                completed.append(cached)
                reused_prediction_task_count += 1
                continue
            try:
                marker = _run_prediction_task(
                    tf,
                    keras,
                    output_root,
                    model_id,
                    farm_id,
                    station,
                    task,
                    args.batch_size,
                    formal,
                    snapshot_record,
                )
                completed.append(marker)
                new_prediction_task_count += 1
            except Exception as exc:
                failures.append(
                    {
                        "model_id": model_id,
                        "farm_id": farm_id,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                _atomic_json(
                    failures[-1],
                    output_root
                    / "manifests"
                    / "prediction_failures"
                    / f"{model_id}_{farm_id}.json",
                )
                print(f"预测失败 {model_id}/{farm_id}: {exc}", file=sys.stderr)
        del station
        gc.collect()

    if failures:
        _atomic_csv(
            failures, output_root / "round3_external14_prediction_failures.csv"
        )
        raise RuntimeError(
            f"{len(failures)}个预测任务失败；未生成complete marker"
        )
    markers = _collect_prediction_markers(output_root, models, farms)
    summary_paths = _aggregate_and_save(
        output_root,
        markers,
        models,
        farms,
        formal,
    )
    selection_payload = _read_json(summary_paths["selection_json"])
    _revalidate_frozen(frozen)
    inventory_path, inventory_rows = _write_inventory(output_root)
    completion = {
        "status": "complete",
        "complete": True,
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": _utc_now(),
        "formal": formal,
        "model_ids": models,
        "farm_ids": farms,
        "task_count": len(markers),
        "expected_formal_task_count": len(MODEL_IDS) * len(EXPECTED_FARMS),
        "new_prediction_task_count": new_prediction_task_count,
        "reused_prediction_task_count": reused_prediction_task_count,
        "all_140_models_frozen_before_this_test_run": bool(formal),
        "selected_tasks_frozen_before_this_test_run": True,
        "selection_eligible": bool(formal),
        "selected_model_id": (
            selection_payload.get("selected_model_id") if formal else None
        ),
        "final_selection": _file_record(summary_paths["selection_json"]),
        **_test_evaluation_provenance(formal),
        "preprocess_complete": preprocess_record,
        "training_complete": frozen.get("training_complete"),
        "frozen_snapshot": snapshot_record,
        "summary_files": {
            key: _file_record(path) for key, path in summary_paths.items()
        },
        "inventory": _file_record(inventory_path),
        "inventory_file_count": len(inventory_rows),
    }
    _atomic_json(completion, complete_path)
    print(f"\n预测完成: {complete_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
