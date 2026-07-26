"""Part 3 Round 3：JSFD001--JSFD014 强基线统一无泄漏训练。

父调度器本身不导入 TensorFlow。每个 ``model_id × farm_id`` 任务均在全新
子进程中构建模型、训练并退出，以可靠释放 GPU 上下文。数据只来自 Round 3
预处理生成的 NPZ/bundle；本文件不会调用旧工程的读 CSV、插值、缩放或切窗
函数。正式训练前，三个重模型会在最大训练场站各用独立进程完成一个完整
train+validation预检epoch；只有HR-MoE预检确认CUDA OOM时才锁定全局
batch=128，随后14个HR场站全部使用同一batch。

``itransformer``、``timesnet``、``timemixer`` 与 ``dlinear`` 是现代强
基线。当前统一追加协议会逐项校验、归档并复用原10模型的 10×14 个训练
产物，只训练四个新增模型的 4×14 个任务；不得重建原全局 batch policy，
也不得覆盖旧模型、权重或 history。为兼容已经按代际运行过的历史目录，
代码仍可从冻结的 13×14 pre-DLinear 状态仅补训 DLinear。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "part3_round3_external14_unified_training_v2"
PREPROCESS_PROTOCOL_VERSION = "part3_round3_external14_leakage_free_v2"
RESULT_ROOT = Path(
    "./wind_results/part3_new_module_supplement/"
    "03_external14_leakage_free_strong_baseline_benchmark"
)
EXPECTED_FARMS = tuple(f"JSFD{i:03d}" for i in range(1, 15))
LEGACY_MODEL_IDS = (
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
ITRANSFORMER_BASELINE_IDS = ("itransformer",)
PRE_TIMESNET_MODEL_IDS = LEGACY_MODEL_IDS + ITRANSFORMER_BASELINE_IDS
TIMESNET_BASELINE_IDS = ("timesnet",)
PRE_TIMEMIXER_MODEL_IDS = PRE_TIMESNET_MODEL_IDS + TIMESNET_BASELINE_IDS
TIMEMIXER_BASELINE_IDS = ("timemixer",)
PRE_DLINEAR_MODEL_IDS = PRE_TIMEMIXER_MODEL_IDS + TIMEMIXER_BASELINE_IDS
DLINEAR_BASELINE_IDS = ("dlinear",)
MODERN_TRAINABLE_MODEL_IDS = (
    ITRANSFORMER_BASELINE_IDS
    + TIMESNET_BASELINE_IDS
    + TIMEMIXER_BASELINE_IDS
    + DLINEAR_BASELINE_IDS
)
MODEL_IDS = LEGACY_MODEL_IDS + MODERN_TRAINABLE_MODEL_IDS
STAGED_EXTENSION_LINEAGE = "historical_staged_modern_extensions_v1"
UNIFIED_MODERN_EXTENSION_LINEAGE = (
    "base10_unified_four_modern_baselines_v1"
)
OTHER_MODELS = frozenset(LEGACY_MODEL_IDS[1:8])
HEAVY_FALLBACK_MODEL = "hr_moe_fets_patchtst"
PREFLIGHT_MODELS = (
    "hr_moe_fets_patchtst",
    "wavenet",
    "transformer",
)
CALIBRATION_MODELS = (
    "hr_moe_fets_patchtst",
    "autoformer",
    "patchtst",
)
HISTORY_LEN = 96
FORECAST_LEN = 16
INPUT_DIM = 45
TARGET_INDEX = 44
EXPECTED_FEATURE_SCHEMA_HASH = (
    "a2f44e932044c2609a8c0e1cf6a446f37b4a0cfb71b8bf232a5bae6c568c680c"
)
RANDOM_SEED = 2026
DEFAULT_BATCH_SIZE = 192
OOM_FALLBACK_BATCH_SIZE = 128
HISTORICAL_BATCH_SIZE = 256
LEARNING_RATE = 5e-4
CLIPNORM = 1.0
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6
ITRANSFORMER_D_MODEL = 512
ITRANSFORMER_NUM_HEADS = 8
ITRANSFORMER_ENCODER_LAYERS = 2
ITRANSFORMER_D_FF = 2048
ITRANSFORMER_DROPOUT = 0.1
ITRANSFORMER_NORM_EPSILON = 1e-5
TIMESNET_D_MODEL = 64
TIMESNET_D_FF = 64
TIMESNET_ENCODER_LAYERS = 2
TIMESNET_TOP_K = 5
TIMESNET_NUM_KERNELS = 6
TIMESNET_DROPOUT = 0.1
TIMESNET_NORM_EPSILON = 1e-5
TIMEMIXER_D_MODEL = 16
TIMEMIXER_D_FF = 32
TIMEMIXER_PDM_LAYERS = 2
TIMEMIXER_DOWNSAMPLING_LAYERS = 3
TIMEMIXER_DOWNSAMPLING_WINDOW = 2
TIMEMIXER_MOVING_AVERAGE = 25
TIMEMIXER_DROPOUT = 0.1
TIMEMIXER_NORM_EPSILON = 1e-5
TIMEMIXER_SCALE_LENGTHS = tuple(
    HISTORY_LEN // (TIMEMIXER_DOWNSAMPLING_WINDOW**level)
    for level in range(TIMEMIXER_DOWNSAMPLING_LAYERS + 1)
)
DLINEAR_MOVING_AVERAGE = 25
DLINEAR_INDIVIDUAL = False
EPOCHS = {
    **{name: 60 for name in OTHER_MODELS},
    "patchtst": 80,
    "hr_moe_fets_patchtst": 80,
    "windprism_f7_g0": 80,
    "itransformer": 60,
    "timesnet": 60,
    "timemixer": 60,
    "dlinear": 60,
}
EXPECTED_PARAMETER_COUNTS = {
    "patchtst": 210_960,
    "bilstm": 107_920,
    "cnn_lstm": 70_480,
    "cnn_resnet_gru": 118_544,
    "wavenet": 940_560,
    "transformer": 858_512,
    "informer": 484_240,
    "autoformer": 212_737,
    "hr_moe_fets_patchtst": 885_395,
    "windprism_f7_g0": 20_969,
    "itransformer": 6_363_664,
    "timesnet": 4_709_917,
    "timemixer": 61_017,
    "dlinear": 3_104,
}
_MISSING_SAFE_REGIME_LAYER_CLASS: Any | None = None
_ITRANSFORMER_LAYER_CLASSES: dict[str, Any] | None = None
_TIMESNET_LAYER_CLASSES: dict[str, Any] | None = None
_TIMEMIXER_LAYER_CLASSES: dict[str, Any] | None = None
_DLINEAR_LAYER_CLASSES: dict[str, Any] | None = None
OOM_EXIT_CODE = 86
EXPLICIT_CUDA_OOM_PATTERNS = (
    "cuda_error_out_of_memory",
    "cudaerrormemoryallocation",
    "cudnn_status_alloc_failed",
)
BATCH_POLICY_PATH = RESULT_ROOT / "manifests" / "round3_batch_policy.json"
PREFLIGHT_SUMMARY_PATH = (
    RESULT_ROOT / "data_audit" / "round3_gpu_preflight_summary.csv"
)
RESOURCE_PLAN_INITIAL_PATH = (
    RESULT_ROOT / "manifests" / "round3_resource_plan_initial.json"
)
RESOURCE_PLAN_CALIBRATED_PATH = (
    RESULT_ROOT / "manifests" / "round3_resource_plan_calibrated.json"
)
RUNTIME_PROGRESS_PATH = (
    RESULT_ROOT / "complexity" / "round3_runtime_progress.csv"
)
HISTORICAL_TASK_SECONDS = {
    "patchtst": 1035.0,
    "bilstm": 128.0,
    "cnn_lstm": 81.0,
    "cnn_resnet_gru": 136.0,
    "wavenet": 101.0,
    "transformer": 147.0,
    "informer": 288.0,
    "autoformer": 1755.0,
    "hr_moe_fets_patchtst": 3292.0,
    "windprism_f7_g0": 137.0,
    # 首次扩展前没有本项目实测值，仅用于ETA；训练完成后会被实测耗时替代。
    "itransformer": 900.0,
    "timesnet": 900.0,
    "timemixer": 900.0,
    "dlinear": 300.0,
}
PRE_DLINEAR_MODEL_MATRIX_REVISION = (
    "base10_plus_itransformer_plus_timesnet_plus_timemixer_extension_v3"
)
MODEL_MATRIX_REVISION = (
    "base10_plus_itransformer_plus_timesnet_plus_timemixer_plus_dlinear_"
    "extension_v4"
)
BASE10_TRAINING_CODE_SHA256 = (
    "d37c17db911dcf95023e9636c2a7881df2d1a2c62202a55a532cae4139bbe530"
)
ITRANSFORMER_EXTENSION_TRAINING_CODE_SHA256 = (
    "06072e0334fb9b785ceb5023d6808489505de27c048d67b19c90353442692e90"
)
TIMEMIXER_EXTENSION_TRAINING_CODE_SHA256 = (
    "b2351214ebc18ec3c35d514d391323985f692d865207d9c424b57c30e0274860"
)
BASE10_TRAINING_COMPLETE_ARCHIVE_PATH = (
    RESULT_ROOT
    / "manifests"
    / "extensions"
    / "itransformer"
    / "base10_training_bundle_complete.json"
)
BASE10_TRAINING_ARCHIVE_MANIFEST_PATH = (
    BASE10_TRAINING_COMPLETE_ARCHIVE_PATH.parent / "archive_manifest.json"
)
PRE_TIMESNET_TRAINING_ARCHIVE_ROOT = (
    RESULT_ROOT
    / "manifests"
    / "extensions"
    / "timesnet"
    / "pre_timesnet_11_training_state"
)
PRE_TIMESNET_TRAINING_ARCHIVE_MANIFEST_PATH = (
    PRE_TIMESNET_TRAINING_ARCHIVE_ROOT / "archive_manifest.json"
)
PRE_TIMEMIXER_TRAINING_ARCHIVE_ROOT = (
    RESULT_ROOT
    / "manifests"
    / "extensions"
    / "timemixer"
    / "pre_timemixer_12_training_state"
)
PRE_TIMEMIXER_TRAINING_ARCHIVE_MANIFEST_PATH = (
    PRE_TIMEMIXER_TRAINING_ARCHIVE_ROOT / "archive_manifest.json"
)
PRE_DLINEAR_TRAINING_ARCHIVE_ROOT = (
    RESULT_ROOT
    / "manifests"
    / "extensions"
    / "dlinear"
    / "pre_dlinear_13_training_state"
)
PRE_DLINEAR_TRAINING_ARCHIVE_MANIFEST_PATH = (
    PRE_DLINEAR_TRAINING_ARCHIVE_ROOT / "archive_manifest.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def parse_selection(raw: str, allowed: Iterable[str], label: str) -> list[str]:
    allowed = tuple(allowed)
    if raw.strip().lower() == "all":
        return list(allowed)
    values = []
    for token in re.split(r"[\s,]+", raw.strip()):
        if token and token not in values:
            values.append(token)
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"未知{label}: {unknown}; 可选={list(allowed)}")
    if not values:
        raise ValueError(f"{label}不能为空")
    return values


def artifact_paths(model_id: str, farm_id: str, smoke: bool = False) -> dict[str, Path]:
    root = RESULT_ROOT
    if smoke:
        root = root / "partial_runs" / "smoke"
    stem = f"{model_id}_{farm_id}"
    return {
        "array": RESULT_ROOT / "prepared_data" / "feature_arrays" / f"{farm_id}.npz",
        "bundle": RESULT_ROOT / "preprocess" / farm_id / "preprocessing_bundle.joblib",
        "preprocess_manifest": RESULT_ROOT / "manifests" / "preprocess" / f"{farm_id}.json",
        "model": root / "models" / model_id / f"{stem}.keras",
        "weights": root / "weights" / model_id / f"{stem}_best.weights.h5",
        "history": root / "history" / model_id / f"{stem}_history.csv",
        "history_plot": root / "history" / model_id / f"{stem}_history.png",
        "tensorboard": root / "tensorboard" / model_id / farm_id,
        "validation": root / "validation_metrics" / model_id / f"{stem}_validation.json",
        "overfit": root / "validation_metrics" / model_id / f"{stem}_overfit.json",
        "marker": (
            root / "manifests" / "training" / f"{stem}.json"
            if smoke
            else RESULT_ROOT / "manifests" / "training" / f"{stem}.json"
        ),
        "attempt_root": root / "attempts" / model_id / farm_id,
    }


def ensure_preprocess_complete() -> dict[str, Any]:
    marker = RESULT_ROOT / "round3_preprocess_bundle_complete.json"
    if not marker.is_file():
        raise FileNotFoundError(f"缺少完整预处理marker: {marker}")
    with open(marker, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "complete":
        raise ValueError(f"预处理marker不是complete: {marker}")
    if payload.get("protocol_version") != PREPROCESS_PROTOCOL_VERSION:
        raise ValueError(
            f"预处理协议版本不匹配: {payload.get('protocol_version')}"
        )
    completed = set(map(str, payload.get("completed_farms", ())))
    if completed != set(EXPECTED_FARMS):
        raise ValueError("预处理marker未覆盖JSFD001--JSFD014")
    return payload


def preprocess_summary(farm_id: str) -> dict[str, Any]:
    path = RESULT_ROOT / "manifests" / "preprocess" / f"{farm_id}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "complete":
        raise ValueError(f"预处理场站marker不是complete: {path}")
    if payload.get("protocol_version") != PREPROCESS_PROTOCOL_VERSION:
        raise ValueError(f"预处理场站协议版本漂移: {path}")
    summary = payload.get("summary", {})
    if str(summary.get("farm_id")) != farm_id:
        raise ValueError(f"预处理场站marker身份漂移: {path}")
    return summary


def largest_training_farm() -> str:
    rows = [
        (int(preprocess_summary(farm_id)["train_windows"]), farm_id)
        for farm_id in EXPECTED_FARMS
    ]
    return max(rows, key=lambda item: (item[0], item[1]))[1]


def legacy_base10_markers_valid_for_extension(policy_sha256: str) -> bool:
    """Prove that an old batch policy is still bound to all frozen base tasks."""
    for model_id in LEGACY_MODEL_IDS:
        for farm_id in EXPECTED_FARMS:
            path = artifact_paths(model_id, farm_id)["marker"]
            if not completed_marker_valid(path):
                return False
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    marker = json.load(handle)
            except Exception:
                return False
            if (
                marker.get("model_id") != model_id
                or marker.get("farm_id") != farm_id
                or marker.get("training_code_sha256")
                != BASE10_TRAINING_CODE_SHA256
                or marker.get("global_batch_policy_sha256")
                != policy_sha256
            ):
                return False
    return True


def copy_exact_file(source: Path, destination: Path) -> dict[str, Any]:
    """Copy one immutable artifact atomically and verify its content hash."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha = sha256_file(source)
    if destination.is_file():
        if sha256_file(destination) != source_sha:
            raise ValueError(f"既有训练归档与源SHA不一致: {destination}")
        return {
            "path": str(destination),
            "sha256": source_sha,
            "size_bytes": destination.stat().st_size,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(source, "rb") as input_handle, open(
        temporary, "wb"
    ) as output_handle:
        for block in iter(lambda: input_handle.read(1024 * 1024), b""):
            output_handle.write(block)
    os.replace(temporary, destination)
    if sha256_file(destination) != source_sha:
        raise ValueError(f"训练归档复制后SHA不一致: {destination}")
    return {
        "path": str(destination),
        "sha256": source_sha,
        "size_bytes": destination.stat().st_size,
    }


def archive_base10_training_complete() -> dict[str, Any]:
    """Archive the immutable 10×14 completion proof before modern extensions."""
    source = RESULT_ROOT / "round3_training_bundle_complete.json"
    archive = BASE10_TRAINING_COMPLETE_ARCHIVE_PATH
    manifest_path = BASE10_TRAINING_ARCHIVE_MANIFEST_PATH
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (
            manifest.get("status") != "complete"
            or int(manifest.get("expected_task_count", -1))
            != len(LEGACY_MODEL_IDS) * len(EXPECTED_FARMS)
            or tuple(manifest.get("expected_models", ())) != LEGACY_MODEL_IDS
        ):
            raise ValueError(f"base10训练归档身份漂移: {manifest_path}")
        for record in manifest.get("archived_records", {}).values():
            path = Path(record["path"])
            if (
                not path.is_file()
                or path.stat().st_size != int(record["size_bytes"])
                or sha256_file(path) != record["sha256"]
            ):
                raise ValueError(f"base10训练归档文件漂移: {path}")
        policy_sha = str(manifest.get("global_batch_policy_sha256", ""))
        if not policy_sha:
            archived_complete = Path(
                manifest["archived_records"]["training_complete"]["path"]
            )
            with open(archived_complete, "r", encoding="utf-8") as handle:
                archived_payload = json.load(handle)
            policy_sha = str(
                archived_payload.get("global_batch_policy_sha256", "")
            )
        if (
            not BATCH_POLICY_PATH.is_file()
            or sha256_file(BATCH_POLICY_PATH) != policy_sha
            or not legacy_base10_markers_valid_for_extension(policy_sha)
        ):
            raise ValueError(
                "base10训练归档与当前140个冻结marker/batch policy不一致"
            )
        frozen_records = {
            (str(item["model_id"]), str(item["farm_id"])): item
            for item in manifest.get("frozen_task_marker_records", ())
        }
        if frozen_records:
            expected_pairs = {
                (model_id, farm_id)
                for model_id in LEGACY_MODEL_IDS
                for farm_id in EXPECTED_FARMS
            }
            if set(frozen_records) != expected_pairs:
                raise ValueError("base10训练归档缺少140个冻结marker记录")
            for pair, record in frozen_records.items():
                path = Path(record["path"])
                if (
                    not path.is_file()
                    or sha256_file(path) != record["sha256"]
                    or not completed_marker_valid(path)
                ):
                    raise ValueError(f"base10冻结训练marker漂移: {pair}")
        return {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        }
    if not source.is_file():
        raise FileNotFoundError(f"缺少待归档base10训练complete marker: {source}")
    with open(source, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected_count = len(LEGACY_MODEL_IDS) * len(EXPECTED_FARMS)
    if (
        payload.get("status") != "complete"
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or int(payload.get("expected_task_count", -1)) != expected_count
        or int(payload.get("completed_task_count", -1)) != expected_count
        or tuple(payload.get("expected_models", ())) != LEGACY_MODEL_IDS
    ):
        raise ValueError(
            "现有training complete不是可追加iTransformer的原10模型冻结矩阵"
        )
    policy_sha = str(payload.get("global_batch_policy_sha256", ""))
    if not legacy_base10_markers_valid_for_extension(policy_sha):
        raise ValueError("原10模型的140个训练marker未通过追加扩展身份校验")
    frozen_records = []
    for model_id in LEGACY_MODEL_IDS:
        for farm_id in EXPECTED_FARMS:
            marker_path = artifact_paths(model_id, farm_id)["marker"]
            frozen_records.append(
                {
                    "model_id": model_id,
                    "farm_id": farm_id,
                    "path": str(marker_path.resolve()),
                    "sha256": sha256_file(marker_path),
                }
            )
    archived_records = {
        "training_complete": copy_exact_file(source, archive),
    }
    for key, record in payload.get("summary_outputs", {}).items():
        source_path = Path(record["path"])
        if (
            not source_path.is_file()
            or source_path.stat().st_size != int(record["size_bytes"])
            or sha256_file(source_path) != record["sha256"]
        ):
            raise ValueError(f"base10训练summary源记录漂移: {key}")
        archived_records[f"summary_{key}"] = copy_exact_file(
            source_path,
            archive.parent / "summary_files" / f"{key}{source_path.suffix}",
        )
    for key in (
        "resource_plan_initial",
        "resource_plan_calibrated",
        "runtime_progress",
    ):
        source_path = payload.get(f"{key}_path")
        expected_sha = payload.get(f"{key}_sha256")
        if source_path and expected_sha:
            source_path = Path(source_path)
            if sha256_file(source_path) != expected_sha:
                raise ValueError(f"base10 {key}源SHA漂移")
            archived_records[key] = copy_exact_file(
                source_path,
                archive.parent / "resource_state" / source_path.name,
            )
    manifest = {
        "status": "complete",
        "created_at": utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "model_matrix_revision_at_archive": payload.get(
            "model_matrix_revision"
        ),
        "expected_models": list(LEGACY_MODEL_IDS),
        "expected_farms": list(EXPECTED_FARMS),
        "expected_task_count": expected_count,
        "base10_training_complete_source_sha256": sha256_file(source),
        "global_batch_policy_sha256": policy_sha,
        "frozen_training_code_sha256_by_model": {
            model_id: BASE10_TRAINING_CODE_SHA256
            for model_id in LEGACY_MODEL_IDS
        },
        "frozen_task_marker_records": frozen_records,
        "archived_records": archived_records,
        "legacy_task_training_artifacts_modified": False,
    }
    atomic_json(manifest_path, manifest)
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "size_bytes": manifest_path.stat().st_size,
    }


def archive_pre_timesnet_training_complete() -> dict[str, Any]:
    """Atomically freeze the completed 11×14 matrix before TimesNet."""
    source = RESULT_ROOT / "round3_training_bundle_complete.json"
    archive_root = PRE_TIMESNET_TRAINING_ARCHIVE_ROOT
    manifest_path = PRE_TIMESNET_TRAINING_ARCHIVE_MANIFEST_PATH
    expected_pairs = {
        (model_id, farm_id)
        for model_id in PRE_TIMESNET_MODEL_IDS
        for farm_id in EXPECTED_FARMS
    }
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (
            manifest.get("status") != "complete"
            or tuple(manifest.get("expected_models", ()))
            != PRE_TIMESNET_MODEL_IDS
            or int(manifest.get("expected_task_count", -1))
            != len(expected_pairs)
        ):
            raise ValueError(f"pre-TimesNet训练归档身份漂移: {manifest_path}")
        for record in manifest.get("archived_records", {}).values():
            path = Path(record["path"])
            if (
                not path.is_file()
                or path.stat().st_size != int(record["size_bytes"])
                or sha256_file(path) != record["sha256"]
            ):
                raise ValueError(f"pre-TimesNet训练归档文件漂移: {path}")
        frozen_records = {
            (str(item["model_id"]), str(item["farm_id"])): item
            for item in manifest.get("frozen_task_marker_records", ())
        }
        if set(frozen_records) != expected_pairs:
            raise ValueError("pre-TimesNet归档缺少154个训练marker记录")
        for record in frozen_records.values():
            path = Path(record["path"])
            if (
                not path.is_file()
                or sha256_file(path) != record["sha256"]
                or not completed_marker_valid(path)
            ):
                raise ValueError(f"pre-TimesNet冻结训练marker漂移: {path}")
        return {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        }

    if archive_root.exists():
        raise ValueError(f"pre-TimesNet训练归档目录不完整: {archive_root}")
    if not source.is_file():
        raise FileNotFoundError(
            "缺少iTransformer扩展后的11模型training complete marker"
        )
    with open(source, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    declared_pairs = {
        (str(item.get("model_id")), str(item.get("farm_id")))
        for item in payload.get("completed_tasks", ())
    }
    if (
        payload.get("status") != "complete"
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or tuple(payload.get("expected_models", ()))
        != PRE_TIMESNET_MODEL_IDS
        or int(payload.get("expected_task_count", -1))
        != len(expected_pairs)
        or int(payload.get("completed_task_count", -1))
        != len(expected_pairs)
        or declared_pairs != expected_pairs
    ):
        raise ValueError(
            "TimesNet追加要求先完成11模型×14站矩阵；"
            "当前training complete不是154任务冻结态"
        )
    policy_sha = str(payload.get("global_batch_policy_sha256", ""))
    if (
        not BATCH_POLICY_PATH.is_file()
        or sha256_file(BATCH_POLICY_PATH) != policy_sha
    ):
        raise ValueError("pre-TimesNet训练complete绑定的batch policy已漂移")

    task_records = {
        (str(item.get("model_id")), str(item.get("farm_id"))): item
        for item in payload.get("task_marker_records", ())
    }
    if set(task_records) != expected_pairs:
        raise ValueError("pre-TimesNet训练complete缺少154个task marker哈希")
    frozen_records = []
    code_hashes_by_model: dict[str, set[str]] = {
        model_id: set() for model_id in PRE_TIMESNET_MODEL_IDS
    }
    for pair in sorted(expected_pairs):
        record = task_records[pair]
        path = Path(record["path"])
        if (
            not path.is_file()
            or sha256_file(path) != record.get("sha256")
            or not completed_marker_valid(path)
        ):
            raise ValueError(f"pre-TimesNet训练marker无效: {pair}")
        with open(path, "r", encoding="utf-8") as handle:
            marker = json.load(handle)
        if (
            (str(marker.get("model_id")), str(marker.get("farm_id")))
            != pair
            or marker.get("global_batch_policy_sha256") != policy_sha
        ):
            raise ValueError(f"pre-TimesNet训练marker身份/策略漂移: {pair}")
        code_hashes_by_model[pair[0]].add(
            str(marker.get("training_code_sha256", ""))
        )
        frozen_records.append(
            {
                "model_id": pair[0],
                "farm_id": pair[1],
                "path": str(path.resolve()),
                "sha256": record["sha256"],
            }
        )
    inconsistent = {
        model_id: sorted(values)
        for model_id, values in code_hashes_by_model.items()
        if len(values) != 1 or "" in values
    }
    if inconsistent:
        raise ValueError(
            f"pre-TimesNet模型跨场站训练代码SHA不一致: {inconsistent}"
        )
    for model_id in LEGACY_MODEL_IDS:
        if code_hashes_by_model[model_id] != {
            BASE10_TRAINING_CODE_SHA256
        }:
            raise ValueError(f"{model_id}不再绑定原base10训练代码SHA")

    for key, record in payload.get("summary_outputs", {}).items():
        path = Path(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"pre-TimesNet训练summary漂移: {key}")

    staging_parent = RESULT_ROOT / "partial_runs" / "archive_staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix="pre_timesnet_training_state_",
            dir=staging_parent,
        )
    ).resolve()

    def final_record(record: dict[str, Any]) -> dict[str, Any]:
        relative = Path(record["path"]).resolve().relative_to(staging_root)
        return {
            **record,
            "path": str((archive_root / relative).resolve()),
        }

    try:
        archived_records = {
            "training_complete": final_record(
                copy_exact_file(
                    source,
                    staging_root
                    / "pre_timesnet_11_training_bundle_complete.json",
                )
            )
        }
        for key, record in payload.get("summary_outputs", {}).items():
            source_path = Path(record["path"])
            archived_records[f"summary_{key}"] = final_record(
                copy_exact_file(
                    source_path,
                    staging_root
                    / "summary_files"
                    / f"{key}{source_path.suffix}",
                )
            )
        for key in (
            "resource_plan_initial",
            "resource_plan_calibrated",
            "runtime_progress",
        ):
            source_path = payload.get(f"{key}_path")
            expected_sha = payload.get(f"{key}_sha256")
            if source_path and expected_sha:
                source_path = Path(source_path)
                if sha256_file(source_path) != expected_sha:
                    raise ValueError(f"pre-TimesNet {key}源SHA漂移")
                archived_records[key] = final_record(
                    copy_exact_file(
                        source_path,
                        staging_root / "resource_state" / source_path.name,
                    )
                )
        manifest = {
            "status": "complete",
            "created_at": utc_now(),
            "protocol_version": PROTOCOL_VERSION,
            "model_matrix_revision": MODEL_MATRIX_REVISION,
            "expected_models": list(PRE_TIMESNET_MODEL_IDS),
            "expected_farms": list(EXPECTED_FARMS),
            "expected_task_count": len(expected_pairs),
            "pre_timesnet_training_complete_source_sha256": sha256_file(
                source
            ),
            "global_batch_policy_sha256": policy_sha,
            "frozen_training_code_sha256_by_model": {
                model_id: next(iter(values))
                for model_id, values in code_hashes_by_model.items()
            },
            "frozen_task_marker_records": frozen_records,
            "archived_records": archived_records,
            "pre_timesnet_model_artifacts_modified": False,
        }
        atomic_json(staging_root / "archive_manifest.json", manifest)
        archive_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_root, archive_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "size_bytes": manifest_path.stat().st_size,
    }


def archive_pre_timemixer_training_complete() -> dict[str, Any]:
    """Atomically freeze the completed 12×14 matrix before TimeMixer."""
    source = RESULT_ROOT / "round3_training_bundle_complete.json"
    archive_root = PRE_TIMEMIXER_TRAINING_ARCHIVE_ROOT
    manifest_path = PRE_TIMEMIXER_TRAINING_ARCHIVE_MANIFEST_PATH
    expected_pairs = {
        (model_id, farm_id)
        for model_id in PRE_TIMEMIXER_MODEL_IDS
        for farm_id in EXPECTED_FARMS
    }

    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (
            manifest.get("status") != "complete"
            or tuple(manifest.get("expected_models", ()))
            != PRE_TIMEMIXER_MODEL_IDS
            or int(manifest.get("expected_task_count", -1))
            != len(expected_pairs)
        ):
            raise ValueError(f"pre-TimeMixer训练归档身份漂移: {manifest_path}")
        frozen_records = {
            (str(item["model_id"]), str(item["farm_id"])): item
            for item in manifest.get("frozen_task_marker_records", ())
        }
        if set(frozen_records) != expected_pairs:
            raise ValueError("pre-TimeMixer归档缺少168个训练marker记录")
        for record in manifest.get("archived_records", {}).values():
            path = Path(record["path"])
            if (
                not path.is_file()
                or path.stat().st_size != int(record["size_bytes"])
                or sha256_file(path) != record["sha256"]
            ):
                raise ValueError(f"pre-TimeMixer训练归档文件漂移: {path}")
        for pair, record in frozen_records.items():
            path = Path(record["path"])
            if (
                not path.is_file()
                or sha256_file(path) != record["sha256"]
                or not completed_marker_valid(path)
            ):
                raise ValueError(f"pre-TimeMixer冻结训练marker漂移: {pair}")
        frozen_hashes = manifest.get(
            "frozen_training_code_sha256_by_model", {}
        )
        if set(frozen_hashes) != set(PRE_TIMEMIXER_MODEL_IDS):
            raise ValueError("pre-TimeMixer归档缺少12模型训练代码SHA")
        return {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        }

    if archive_root.exists():
        raise ValueError(f"pre-TimeMixer训练归档目录不完整: {archive_root}")
    if not source.is_file():
        raise FileNotFoundError(
            "缺少TimesNet扩展后的12模型training complete marker"
        )
    if not PRE_TIMESNET_TRAINING_ARCHIVE_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "缺少pre-TimesNet训练归档，无法建立连续代际证据链"
        )
    # 复核上一代归档及其154个live marker均未漂移。
    archive_pre_timesnet_training_complete()
    with open(source, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    declared_pairs = {
        (str(item.get("model_id")), str(item.get("farm_id")))
        for item in payload.get("completed_tasks", ())
    }
    if (
        payload.get("status") != "complete"
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or tuple(payload.get("expected_models", ()))
        != PRE_TIMEMIXER_MODEL_IDS
        or int(payload.get("expected_task_count", -1))
        != len(expected_pairs)
        or int(payload.get("completed_task_count", -1))
        != len(expected_pairs)
        or declared_pairs != expected_pairs
    ):
        raise ValueError(
            "TimeMixer追加要求先完成12模型×14站矩阵；"
            "当前training complete不是168任务冻结态"
        )
    previous_archive = payload.get(
        "pre_timesnet_training_complete_archive"
    )
    if (
        not isinstance(previous_archive, dict)
        or Path(previous_archive.get("path", "")).resolve()
        != PRE_TIMESNET_TRAINING_ARCHIVE_MANIFEST_PATH.resolve()
        or previous_archive.get("sha256")
        != sha256_file(PRE_TIMESNET_TRAINING_ARCHIVE_MANIFEST_PATH)
        or int(previous_archive.get("size_bytes", -1))
        != PRE_TIMESNET_TRAINING_ARCHIVE_MANIFEST_PATH.stat().st_size
    ):
        raise ValueError("12模型complete未正确绑定pre-TimesNet训练归档")
    policy_sha = str(payload.get("global_batch_policy_sha256", ""))
    if (
        not BATCH_POLICY_PATH.is_file()
        or sha256_file(BATCH_POLICY_PATH) != policy_sha
    ):
        raise ValueError("pre-TimeMixer训练complete绑定的batch policy已漂移")

    task_records = {
        (str(item.get("model_id")), str(item.get("farm_id"))): item
        for item in payload.get("task_marker_records", ())
    }
    if set(task_records) != expected_pairs:
        raise ValueError("pre-TimeMixer训练complete缺少168个task marker哈希")
    frozen_records: list[dict[str, Any]] = []
    code_hashes_by_model: dict[str, set[str]] = {
        model_id: set() for model_id in PRE_TIMEMIXER_MODEL_IDS
    }
    for pair in sorted(expected_pairs):
        record = task_records[pair]
        path = Path(record["path"])
        if (
            not path.is_file()
            or sha256_file(path) != record.get("sha256")
            or not completed_marker_valid(path)
        ):
            raise ValueError(f"pre-TimeMixer训练marker无效: {pair}")
        with open(path, "r", encoding="utf-8") as handle:
            marker = json.load(handle)
        if (
            (str(marker.get("model_id")), str(marker.get("farm_id")))
            != pair
            or marker.get("global_batch_policy_sha256") != policy_sha
        ):
            raise ValueError(f"pre-TimeMixer训练marker身份/策略漂移: {pair}")
        code_hashes_by_model[pair[0]].add(
            str(marker.get("training_code_sha256", ""))
        )
        frozen_records.append(
            {
                "model_id": pair[0],
                "farm_id": pair[1],
                "path": str(path.resolve()),
                "sha256": record["sha256"],
            }
        )
    inconsistent = {
        model_id: sorted(values)
        for model_id, values in code_hashes_by_model.items()
        if len(values) != 1 or "" in values
    }
    if inconsistent:
        raise ValueError(
            f"pre-TimeMixer模型跨场站训练代码SHA不一致: {inconsistent}"
        )
    for model_id in LEGACY_MODEL_IDS:
        if code_hashes_by_model[model_id] != {
            BASE10_TRAINING_CODE_SHA256
        }:
            raise ValueError(f"{model_id}不再绑定原base10训练代码SHA")

    for key, record in payload.get("summary_outputs", {}).items():
        path = Path(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"pre-TimeMixer训练summary漂移: {key}")

    staging_parent = RESULT_ROOT / "partial_runs" / "archive_staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix="pre_timemixer_training_state_",
            dir=staging_parent,
        )
    ).resolve()

    def final_record(record: dict[str, Any]) -> dict[str, Any]:
        relative = Path(record["path"]).resolve().relative_to(staging_root)
        return {**record, "path": str((archive_root / relative).resolve())}

    try:
        archived_records = {
            "training_complete": final_record(
                copy_exact_file(
                    source,
                    staging_root
                    / "pre_timemixer_12_training_bundle_complete.json",
                )
            ),
            "pre_timesnet_archive_manifest": final_record(
                copy_exact_file(
                    PRE_TIMESNET_TRAINING_ARCHIVE_MANIFEST_PATH,
                    staging_root
                    / "prior_generation"
                    / "pre_timesnet_archive_manifest.json",
                )
            ),
        }
        for key, record in payload.get("summary_outputs", {}).items():
            source_path = Path(record["path"])
            archived_records[f"summary_{key}"] = final_record(
                copy_exact_file(
                    source_path,
                    staging_root
                    / "summary_files"
                    / f"{key}{source_path.suffix}",
                )
            )
        for key in (
            "resource_plan_initial",
            "resource_plan_calibrated",
            "runtime_progress",
        ):
            source_path = payload.get(f"{key}_path")
            expected_sha = payload.get(f"{key}_sha256")
            if source_path and expected_sha:
                source_path = Path(source_path)
                if sha256_file(source_path) != expected_sha:
                    raise ValueError(f"pre-TimeMixer {key}源SHA漂移")
                archived_records[key] = final_record(
                    copy_exact_file(
                        source_path,
                        staging_root / "resource_state" / source_path.name,
                    )
                )
        manifest = {
            "status": "complete",
            "created_at": utc_now(),
            "protocol_version": PROTOCOL_VERSION,
            "model_matrix_revision_at_archive": payload.get(
                "model_matrix_revision"
            ),
            "expected_models": list(PRE_TIMEMIXER_MODEL_IDS),
            "expected_farms": list(EXPECTED_FARMS),
            "expected_task_count": len(expected_pairs),
            "pre_timemixer_training_complete_source_sha256": sha256_file(
                source
            ),
            "global_batch_policy_sha256": policy_sha,
            "frozen_training_code_sha256_by_model": {
                model_id: next(iter(values))
                for model_id, values in code_hashes_by_model.items()
            },
            "frozen_task_marker_records": frozen_records,
            "archived_records": archived_records,
            "pre_timemixer_model_artifacts_modified": False,
        }
        atomic_json(staging_root / "archive_manifest.json", manifest)
        archive_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_root, archive_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "size_bytes": manifest_path.stat().st_size,
    }


def archive_pre_dlinear_training_complete() -> dict[str, Any]:
    """Atomically freeze the completed 13×14 matrix before DLinear."""
    source = RESULT_ROOT / "round3_training_bundle_complete.json"
    archive_root = PRE_DLINEAR_TRAINING_ARCHIVE_ROOT
    manifest_path = PRE_DLINEAR_TRAINING_ARCHIVE_MANIFEST_PATH
    expected_pairs = {
        (model_id, farm_id)
        for model_id in PRE_DLINEAR_MODEL_IDS
        for farm_id in EXPECTED_FARMS
    }

    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (
            manifest.get("status") != "complete"
            or manifest.get("protocol_version") != PROTOCOL_VERSION
            or manifest.get("model_matrix_revision_at_archive")
            != PRE_DLINEAR_MODEL_MATRIX_REVISION
            or tuple(manifest.get("expected_models", ()))
            != PRE_DLINEAR_MODEL_IDS
            or int(manifest.get("expected_task_count", -1))
            != len(expected_pairs)
        ):
            raise ValueError(f"pre-DLinear训练归档身份漂移: {manifest_path}")
        frozen_records = {
            (str(item["model_id"]), str(item["farm_id"])): item
            for item in manifest.get("frozen_task_marker_records", ())
        }
        if set(frozen_records) != expected_pairs:
            raise ValueError("pre-DLinear归档缺少182个训练marker记录")
        for record in manifest.get("archived_records", {}).values():
            path = Path(record["path"])
            if (
                not path.is_file()
                or path.stat().st_size != int(record["size_bytes"])
                or sha256_file(path) != record["sha256"]
            ):
                raise ValueError(f"pre-DLinear训练归档文件漂移: {path}")
        for pair, record in frozen_records.items():
            path = Path(record["path"])
            if (
                not path.is_file()
                or sha256_file(path) != record["sha256"]
                or not completed_marker_valid(path)
            ):
                raise ValueError(f"pre-DLinear冻结训练marker漂移: {pair}")
        frozen_hashes = {
            str(model_id): str(code_sha)
            for model_id, code_sha in manifest.get(
                "frozen_training_code_sha256_by_model", {}
            ).items()
        }
        if set(frozen_hashes) != set(PRE_DLINEAR_MODEL_IDS):
            raise ValueError("pre-DLinear归档缺少13模型训练代码SHA")
        if (
            frozen_hashes.get("timemixer")
            != TIMEMIXER_EXTENSION_TRAINING_CODE_SHA256
        ):
            raise ValueError("pre-DLinear归档中的TimeMixer训练代码SHA漂移")
        return {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        }

    if archive_root.exists():
        raise ValueError(f"pre-DLinear训练归档目录不完整: {archive_root}")
    if not source.is_file():
        raise FileNotFoundError(
            "缺少TimeMixer扩展后的13模型training complete marker"
        )
    if not PRE_TIMEMIXER_TRAINING_ARCHIVE_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "缺少pre-TimeMixer训练归档，不能从当前182项live bundle"
            "反推上一代状态，无法建立连续代际证据链"
        )
    # 只能复核既存上一代归档；不能用13模型live complete重建12模型归档。
    archive_pre_timemixer_training_complete()
    with open(source, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    declared_pairs = {
        (str(item.get("model_id")), str(item.get("farm_id")))
        for item in payload.get("completed_tasks", ())
    }
    if (
        payload.get("status") != "complete"
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("model_matrix_revision")
        != PRE_DLINEAR_MODEL_MATRIX_REVISION
        or tuple(payload.get("expected_models", ()))
        != PRE_DLINEAR_MODEL_IDS
        or int(payload.get("expected_task_count", -1))
        != len(expected_pairs)
        or int(payload.get("completed_task_count", -1))
        != len(expected_pairs)
        or declared_pairs != expected_pairs
    ):
        raise ValueError(
            "DLinear追加要求先完成13模型×14站矩阵；"
            "当前training complete不是182任务冻结态"
        )
    previous_archive = payload.get(
        "pre_timemixer_training_complete_archive"
    )
    if (
        not isinstance(previous_archive, dict)
        or Path(previous_archive.get("path", "")).resolve()
        != PRE_TIMEMIXER_TRAINING_ARCHIVE_MANIFEST_PATH.resolve()
        or previous_archive.get("sha256")
        != sha256_file(PRE_TIMEMIXER_TRAINING_ARCHIVE_MANIFEST_PATH)
        or int(previous_archive.get("size_bytes", -1))
        != PRE_TIMEMIXER_TRAINING_ARCHIVE_MANIFEST_PATH.stat().st_size
    ):
        raise ValueError("13模型complete未正确绑定pre-TimeMixer训练归档")
    if (
        payload.get("timemixer_extension_training_code_sha256")
        != TIMEMIXER_EXTENSION_TRAINING_CODE_SHA256
    ):
        raise ValueError("13模型complete记录的TimeMixer训练代码SHA漂移")
    policy_sha = str(payload.get("global_batch_policy_sha256", ""))
    if (
        not BATCH_POLICY_PATH.is_file()
        or sha256_file(BATCH_POLICY_PATH) != policy_sha
    ):
        raise ValueError("pre-DLinear训练complete绑定的batch policy已漂移")

    task_records = {
        (str(item.get("model_id")), str(item.get("farm_id"))): item
        for item in payload.get("task_marker_records", ())
    }
    if set(task_records) != expected_pairs:
        raise ValueError("pre-DLinear训练complete缺少182个task marker哈希")
    with open(
        PRE_TIMEMIXER_TRAINING_ARCHIVE_MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as handle:
        prior_archive = json.load(handle)
    prior_hashes = {
        str(model_id): str(code_sha)
        for model_id, code_sha in prior_archive.get(
            "frozen_training_code_sha256_by_model", {}
        ).items()
    }
    if set(prior_hashes) != set(PRE_TIMEMIXER_MODEL_IDS):
        raise ValueError("pre-TimeMixer训练归档缺少12模型代码SHA")

    frozen_records: list[dict[str, Any]] = []
    code_hashes_by_model: dict[str, set[str]] = {
        model_id: set() for model_id in PRE_DLINEAR_MODEL_IDS
    }
    for pair in sorted(expected_pairs):
        record = task_records[pair]
        path = Path(record["path"])
        if (
            not path.is_file()
            or sha256_file(path) != record.get("sha256")
            or not completed_marker_valid(path)
        ):
            raise ValueError(f"pre-DLinear训练marker无效: {pair}")
        with open(path, "r", encoding="utf-8") as handle:
            marker = json.load(handle)
        if (
            (str(marker.get("model_id")), str(marker.get("farm_id")))
            != pair
            or marker.get("global_batch_policy_sha256") != policy_sha
        ):
            raise ValueError(f"pre-DLinear训练marker身份/策略漂移: {pair}")
        code_hashes_by_model[pair[0]].add(
            str(marker.get("training_code_sha256", ""))
        )
        frozen_records.append(
            {
                "model_id": pair[0],
                "farm_id": pair[1],
                "path": str(path.resolve()),
                "sha256": record["sha256"],
            }
        )
    inconsistent = {
        model_id: sorted(values)
        for model_id, values in code_hashes_by_model.items()
        if len(values) != 1 or "" in values
    }
    if inconsistent:
        raise ValueError(
            f"pre-DLinear模型跨场站训练代码SHA不一致: {inconsistent}"
        )
    for model_id in PRE_TIMEMIXER_MODEL_IDS:
        if code_hashes_by_model[model_id] != {
            prior_hashes[model_id]
        }:
            raise ValueError(f"{model_id}不再绑定pre-TimeMixer归档训练SHA")
    if code_hashes_by_model["timemixer"] != {
        TIMEMIXER_EXTENSION_TRAINING_CODE_SHA256
    }:
        raise ValueError("TimeMixer不再绑定其冻结训练代码SHA")

    for key, record in payload.get("summary_outputs", {}).items():
        path = Path(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"pre-DLinear训练summary漂移: {key}")

    staging_parent = RESULT_ROOT / "partial_runs" / "archive_staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix="pre_dlinear_training_state_",
            dir=staging_parent,
        )
    ).resolve()

    def final_record(record: dict[str, Any]) -> dict[str, Any]:
        relative = Path(record["path"]).resolve().relative_to(staging_root)
        return {**record, "path": str((archive_root / relative).resolve())}

    try:
        archived_records = {
            "training_complete": final_record(
                copy_exact_file(
                    source,
                    staging_root
                    / "pre_dlinear_13_training_bundle_complete.json",
                )
            ),
            "pre_timemixer_archive_manifest": final_record(
                copy_exact_file(
                    PRE_TIMEMIXER_TRAINING_ARCHIVE_MANIFEST_PATH,
                    staging_root
                    / "prior_generation"
                    / "pre_timemixer_archive_manifest.json",
                )
            ),
        }
        for key, record in payload.get("summary_outputs", {}).items():
            source_path = Path(record["path"])
            archived_records[f"summary_{key}"] = final_record(
                copy_exact_file(
                    source_path,
                    staging_root
                    / "summary_files"
                    / f"{key}{source_path.suffix}",
                )
            )
        for key in (
            "resource_plan_initial",
            "resource_plan_calibrated",
            "runtime_progress",
        ):
            source_path = payload.get(f"{key}_path")
            expected_sha = payload.get(f"{key}_sha256")
            if source_path and expected_sha:
                source_path = Path(source_path)
                if sha256_file(source_path) != expected_sha:
                    raise ValueError(f"pre-DLinear {key}源SHA漂移")
                archived_records[key] = final_record(
                    copy_exact_file(
                        source_path,
                        staging_root / "resource_state" / source_path.name,
                    )
                )
        manifest = {
            "status": "complete",
            "created_at": utc_now(),
            "protocol_version": PROTOCOL_VERSION,
            "model_matrix_revision_at_archive": payload.get(
                "model_matrix_revision"
            ),
            "expected_models": list(PRE_DLINEAR_MODEL_IDS),
            "expected_farms": list(EXPECTED_FARMS),
            "expected_task_count": len(expected_pairs),
            "pre_dlinear_training_complete_source_sha256": sha256_file(source),
            "global_batch_policy_sha256": policy_sha,
            "frozen_training_code_sha256_by_model": {
                model_id: next(iter(values))
                for model_id, values in code_hashes_by_model.items()
            },
            "frozen_task_marker_records": frozen_records,
            "archived_records": archived_records,
            "pre_dlinear_model_artifacts_modified": False,
        }
        atomic_json(staging_root / "archive_manifest.json", manifest)
        archive_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_root, archive_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "size_bytes": manifest_path.stat().st_size,
    }


def infer_extension_lineage() -> str:
    """Infer staged versus unified lineage without rewriting any evidence."""
    staged_archives = (
        PRE_TIMESNET_TRAINING_ARCHIVE_MANIFEST_PATH,
        PRE_TIMEMIXER_TRAINING_ARCHIVE_MANIFEST_PATH,
        PRE_DLINEAR_TRAINING_ARCHIVE_MANIFEST_PATH,
    )
    if any(path.is_file() for path in staged_archives):
        return STAGED_EXTENSION_LINEAGE

    complete_path = RESULT_ROOT / "round3_training_bundle_complete.json"
    if complete_path.is_file():
        try:
            with open(complete_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            declared_lineage = payload.get("extension_lineage")
            if declared_lineage in {
                STAGED_EXTENSION_LINEAGE,
                UNIFIED_MODERN_EXTENSION_LINEAGE,
            }:
                return str(declared_lineage)
            expected_models = tuple(payload.get("expected_models", ()))
            if expected_models in {
                PRE_TIMESNET_MODEL_IDS,
                PRE_TIMEMIXER_MODEL_IDS,
                PRE_DLINEAR_MODEL_IDS,
            }:
                return STAGED_EXTENSION_LINEAGE
        except Exception:
            pass

    current_code_sha = sha256_file(__file__)
    observed_hashes = set()
    for model_id in MODERN_TRAINABLE_MODEL_IDS:
        for farm_id in EXPECTED_FARMS:
            marker_path = artifact_paths(model_id, farm_id)["marker"]
            if not marker_path.is_file():
                continue
            try:
                with open(marker_path, "r", encoding="utf-8") as handle:
                    marker = json.load(handle)
            except Exception:
                continue
            if (
                marker.get("status") == "complete"
                and marker.get("model_id") == model_id
                and marker.get("farm_id") == farm_id
            ):
                declared_lineage = marker.get("extension_lineage")
                if declared_lineage == STAGED_EXTENSION_LINEAGE:
                    return STAGED_EXTENSION_LINEAGE
                if marker.get("training_code_sha256"):
                    observed_hashes.add(str(marker["training_code_sha256"]))
    if observed_hashes and observed_hashes != {current_code_sha}:
        return STAGED_EXTENSION_LINEAGE
    return UNIFIED_MODERN_EXTENSION_LINEAGE


def _archive_pointer_valid(
    record: Any,
    manifest_path: Path,
) -> bool:
    return bool(
        isinstance(record, dict)
        and manifest_path.is_file()
        and Path(record.get("path", "")).resolve() == manifest_path.resolve()
        and record.get("sha256") == sha256_file(manifest_path)
        and int(record.get("size_bytes", -1)) == manifest_path.stat().st_size
    )


def extended_training_bundle_valid() -> bool:
    """Return True for either immutable staged or unified 14×14 lineage."""
    path = RESULT_ROOT / "round3_training_bundle_complete.json"
    if not path.is_file():
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        expected = {
            (model_id, farm_id)
            for model_id in MODEL_IDS
            for farm_id in EXPECTED_FARMS
        }
        declared = {
            (str(item.get("model_id")), str(item.get("farm_id")))
            for item in payload.get("completed_tasks", ())
        }
        if (
            payload.get("status") != "complete"
            or payload.get("protocol_version") != PROTOCOL_VERSION
            or payload.get("model_matrix_revision")
            != MODEL_MATRIX_REVISION
            or tuple(payload.get("expected_models", ())) != MODEL_IDS
            or int(payload.get("expected_task_count", -1)) != len(expected)
            or int(payload.get("completed_task_count", -1)) != len(expected)
            or declared != expected
        ):
            return False
        lineage = payload.get("extension_lineage")
        if lineage not in {
            STAGED_EXTENSION_LINEAGE,
            UNIFIED_MODERN_EXTENSION_LINEAGE,
        }:
            # Compatibility for a historical staged complete written before
            # the explicit lineage field was introduced.
            if payload.get("pre_dlinear_training_complete_archive"):
                lineage = STAGED_EXTENSION_LINEAGE
            else:
                return False
        records = {
            (str(item.get("model_id")), str(item.get("farm_id"))): item
            for item in payload.get("task_marker_records", ())
        }
        if set(records) != expected:
            return False
        frozen_code_hashes: dict[str, str]
        modern_code_sha: str | None = None
        if lineage == STAGED_EXTENSION_LINEAGE:
            if not _archive_pointer_valid(
                payload.get("pre_dlinear_training_complete_archive"),
                PRE_DLINEAR_TRAINING_ARCHIVE_MANIFEST_PATH,
            ):
                return False
            with open(
                PRE_DLINEAR_TRAINING_ARCHIVE_MANIFEST_PATH,
                "r",
                encoding="utf-8",
            ) as handle:
                archive = json.load(handle)
            frozen_code_hashes = {
                str(model_id): str(code_sha)
                for model_id, code_sha in archive.get(
                    "frozen_training_code_sha256_by_model", {}
                ).items()
            }
            if (
                archive.get("status") != "complete"
                or archive.get("model_matrix_revision_at_archive")
                != PRE_DLINEAR_MODEL_MATRIX_REVISION
                or tuple(archive.get("expected_models", ()))
                != PRE_DLINEAR_MODEL_IDS
                or set(frozen_code_hashes) != set(PRE_DLINEAR_MODEL_IDS)
            ):
                return False
            dlinear_code_sha = str(
                payload.get("dlinear_extension_training_code_sha256", "")
            )
            if not dlinear_code_sha:
                return False
        else:
            if not _archive_pointer_valid(
                payload.get("base10_training_complete_archive"),
                BASE10_TRAINING_ARCHIVE_MANIFEST_PATH,
            ):
                return False
            with open(
                BASE10_TRAINING_ARCHIVE_MANIFEST_PATH,
                "r",
                encoding="utf-8",
            ) as handle:
                archive = json.load(handle)
            if (
                archive.get("status") != "complete"
                or tuple(archive.get("expected_models", ()))
                != LEGACY_MODEL_IDS
                or int(archive.get("expected_task_count", -1))
                != len(LEGACY_MODEL_IDS) * len(EXPECTED_FARMS)
            ):
                return False
            frozen_code_hashes = {
                model_id: BASE10_TRAINING_CODE_SHA256
                for model_id in LEGACY_MODEL_IDS
            }
            modern_code_sha = str(
                payload.get("modern_extension_training_code_sha256", "")
            )
            if not modern_code_sha:
                return False

        for pair, record in records.items():
            record_path = Path(record["path"])
            if (
                not record_path.is_file()
                or sha256_file(record_path) != record.get("sha256")
                or not completed_marker_valid(record_path)
            ):
                return False
            with open(record_path, "r", encoding="utf-8") as handle:
                marker = json.load(handle)
            if (
                str(marker.get("model_id")),
                str(marker.get("farm_id")),
            ) != pair:
                return False
            if lineage == UNIFIED_MODERN_EXTENSION_LINEAGE:
                expected_code_sha = (
                    modern_code_sha
                    if pair[0] in MODERN_TRAINABLE_MODEL_IDS
                    else frozen_code_hashes.get(pair[0])
                )
                if (
                    pair[0] in MODERN_TRAINABLE_MODEL_IDS
                    and marker.get("extension_lineage")
                    != UNIFIED_MODERN_EXTENSION_LINEAGE
                ):
                    return False
            else:
                expected_code_sha = (
                    dlinear_code_sha
                    if pair[0] in DLINEAR_BASELINE_IDS
                    else frozen_code_hashes.get(pair[0])
                )
                if (
                    pair[0] in DLINEAR_BASELINE_IDS
                    and marker.get("extension_lineage")
                    not in (None, STAGED_EXTENSION_LINEAGE)
                ):
                    return False
            if marker.get("training_code_sha256") != expected_code_sha:
                return False
        for record in payload.get("summary_outputs", {}).values():
            record_path = Path(record["path"])
            if (
                not record_path.is_file()
                or sha256_file(record_path) != record.get("sha256")
            ):
                return False
        policy_path = Path(payload["global_batch_policy_path"])
        if (
            not policy_path.is_file()
            or sha256_file(policy_path)
            != payload.get("global_batch_policy_sha256")
        ):
            return False
        return True
    except Exception:
        return False


def load_batch_policy(require_valid_sources: bool = True) -> dict[str, Any]:
    if not BATCH_POLICY_PATH.is_file():
        raise FileNotFoundError(f"缺少全局batch策略: {BATCH_POLICY_PATH}")
    with open(BATCH_POLICY_PATH, "r", encoding="utf-8") as handle:
        policy = json.load(handle)
    if policy.get("status") != "complete":
        raise ValueError("全局batch策略不是complete")
    if policy.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("全局batch策略协议版本漂移")
    if set(policy.get("preflight_models", ())) != set(PREFLIGHT_MODELS):
        raise ValueError("全局batch策略未覆盖三个预检模型")
    hr_batch = int(policy.get("hr_moe_effective_batch_size", -1))
    if hr_batch not in (DEFAULT_BATCH_SIZE, OOM_FALLBACK_BATCH_SIZE):
        raise ValueError(f"HR-MoE全局batch无效: {hr_batch}")
    effective = policy.get("model_effective_batch_sizes", {})
    if int(effective.get(HEAVY_FALLBACK_MODEL, -1)) != hr_batch:
        raise ValueError("HR-MoE预检结果与全局batch字段不一致")
    for model_id in ("wavenet", "transformer"):
        if int(effective.get(model_id, -1)) != DEFAULT_BATCH_SIZE:
            raise ValueError(f"{model_id}预检未锁定batch=192")
    if policy.get("gpu_preflight_verified") is not True:
        raise ValueError("全局batch策略没有真实GPU预检证明")
    gpu_names = policy.get("preflight_gpu_names_by_model", {})
    if any(not gpu_names.get(model_id) for model_id in PREFLIGHT_MODELS):
        raise ValueError("全局batch策略缺少预检GPU设备身份")
    summary_path = Path(str(policy.get("preflight_summary_path", "")))
    if (
        not summary_path.is_file()
        or sha256_file(summary_path) != policy.get("preflight_summary_sha256")
    ):
        raise ValueError("GPU预检summary缺失或SHA漂移")
    if require_valid_sources:
        farm_id = str(policy.get("preflight_farm_id"))
        if farm_id != largest_training_farm():
            raise ValueError("全局batch策略的最大训练场站已漂移")
        paths = artifact_paths("patchtst", farm_id)
        current_sources = {
            "array_sha256": sha256_file(paths["array"]),
            "preprocess_bundle_sha256": sha256_file(paths["bundle"]),
        }
        for key, value in current_sources.items():
            if policy.get(key) != value:
                raise ValueError(f"全局batch策略源身份漂移: {key}")
        current_code_sha = sha256_file(__file__)
        if policy.get("training_code_sha256") != current_code_sha:
            # Modern baselines are appended by generation.  Rebuilding the
            # original policy would change its SHA and invalidate the frozen
            # base markers, so compatibility is accepted only while every
            # base marker still binds the archived code SHA and policy SHA.
            if (
                policy.get("training_code_sha256")
                != BASE10_TRAINING_CODE_SHA256
                or not legacy_base10_markers_valid_for_extension(
                    sha256_file(BATCH_POLICY_PATH)
                )
            ):
                raise ValueError(
                    "全局batch策略training code身份漂移，且不满足base10追加兼容"
                )
    return policy


def formal_batch_size(model_id: str, policy: dict[str, Any]) -> int:
    if model_id == HEAVY_FALLBACK_MODEL:
        return int(policy["hr_moe_effective_batch_size"])
    return DEFAULT_BATCH_SIZE


def validate_task_policy(
    marker_path: Path,
    model_id: str,
    policy: dict[str, Any],
) -> None:
    with open(marker_path, "r", encoding="utf-8") as handle:
        marker = json.load(handle)
    expected_batch = formal_batch_size(model_id, policy)
    if int(marker.get("effective_batch_size", -1)) != expected_batch:
        raise ValueError(
            f"{model_id}/{marker.get('farm_id')} batch与全局策略不一致"
        )
    if marker.get("global_batch_policy_sha256") != sha256_file(BATCH_POLICY_PATH):
        raise ValueError(
            f"{model_id}/{marker.get('farm_id')} batch策略SHA已漂移"
        )


def completed_marker_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("status") != "complete":
            return False
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            return False
        if payload.get("preprocess_protocol_version") != PREPROCESS_PROTOCOL_VERSION:
            return False
        required_path_hashes = (
            ("model_path", "model_sha256"),
            ("weights_path", "weights_sha256"),
            ("history_path", "history_sha256"),
            ("history_plot_path", "history_plot_sha256"),
            ("validation_path", "validation_sha256"),
            ("overfit_path", "overfit_sha256"),
            ("array_path", "array_sha256"),
            ("preprocess_bundle_path", "preprocess_bundle_sha256"),
        )
        for path_key, hash_key in required_path_hashes:
            value = payload.get(path_key)
            expected = payload.get(hash_key)
            if not value or not expected or not Path(value).is_file():
                return False
            if sha256_file(value) != expected:
                return False
        if not payload.get("smoke"):
            policy_path = payload.get("global_batch_policy_path")
            policy_hash = payload.get("global_batch_policy_sha256")
            if (
                not policy_path
                or not policy_hash
                or not Path(policy_path).is_file()
                or sha256_file(policy_path) != policy_hash
            ):
                return False
        return True
    except Exception:
        return False


def is_oom_text(text: str) -> bool:
    lowered = text.lower()
    if any(pattern in lowered for pattern in EXPLICIT_CUDA_OOM_PATTERNS):
        return True
    memory_signal = any(
        pattern in lowered
        for pattern in (
            "resourceexhaustederror",
            "failed to allocate memory",
            "out of memory",
            "oom when allocating",
        )
    )
    gpu_signal = any(
        token in lowered
        for token in (
            "cuda",
            "gpu",
            "cudnn",
            "gpu_bfc",
            "/device:gpu",
        )
    )
    return memory_signal and gpu_signal


def is_confirmed_cuda_oom(
    exc: Exception,
    text: str,
    tf: Any,
    model: Any,
) -> bool:
    if is_oom_text(text):
        return True
    if tf is None or model is None:
        return False
    resource_exhausted = isinstance(exc, tf.errors.ResourceExhaustedError)
    memory_signal = any(
        token in text.lower()
        for token in ("oom", "out of memory", "failed to allocate")
    )
    model_on_gpu = False
    for variable in getattr(model, "weights", ()):
        device = str(getattr(variable, "device", ""))
        if not device and hasattr(variable, "handle"):
            device = str(getattr(variable.handle, "device", ""))
        if "GPU" in device.upper():
            model_on_gpu = True
            break
    return resource_exhausted and memory_signal and model_on_gpu


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all", help="all或逗号分隔model_id")
    parser.add_argument("--farms", default="all", help="all或逗号分隔JSFD编号")
    parser.add_argument("--resume", action="store_true", help="跳过身份校验通过的任务")
    parser.add_argument("--force", action="store_true", help="覆盖已有任务")
    parser.add_argument("--smoke", action="store_true", help="每任务仅跑1 epoch且隔离保存")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只完成全局GPU预检和batch策略，不启动正式训练",
    )
    parser.add_argument(
        "--force-preflight",
        action="store_true",
        help="忽略已有有效GPU预检策略并重新运行",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--preflight-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=MODEL_IDS, help=argparse.SUPPRESS)
    parser.add_argument("--farm", choices=EXPECTED_FARMS, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--attempt-dir", help=argparse.SUPPRESS)
    parser.add_argument(
        "--extension-lineage",
        choices=(
            STAGED_EXTENSION_LINEAGE,
            UNIFIED_MODERN_EXTENSION_LINEAGE,
            "smoke",
        ),
        help=argparse.SUPPRESS,
    )
    return parser


def _required_npz(path: Path) -> dict[str, np.ndarray]:
    required = {
        "features_scaled",
        "target_scaled",
        "train_origins",
        "val_origins",
        "input_cols",
        "scaler_y_mean",
        "scaler_y_scale",
        "scaler_x_mean",
        "scaler_x_scale",
        "power_reference_mw",
    }
    if not path.is_file():
        raise FileNotFoundError(path)
    if "processed_npz" in path.parts:
        raise ValueError(f"禁止使用旧processed_npz: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"{path}缺少数组: {missing}")
        # 刻意不访问test_origins、target_mw、timestamps_ns等测试侧数组。
        return {key: np.asarray(archive[key]) for key in required}


def _validate_origins(
    origins: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
    label: str,
) -> np.ndarray:
    origins = np.asarray(origins, dtype=np.int64).reshape(-1)
    if not len(origins):
        raise ValueError(f"{label} origins为空")
    if not np.array_equal(origins, np.unique(origins)):
        raise ValueError(f"{label} origins必须严格递增且唯一")
    if origins[0] < HISTORY_LEN or origins[-1] + FORECAST_LEN > len(target):
        raise ValueError(f"{label} origins越界")
    for start in range(0, len(origins), 4096):
        batch = origins[start : start + 4096]
        history_indices = batch[:, None] - HISTORY_LEN + np.arange(HISTORY_LEN)
        target_indices = batch[:, None] + np.arange(FORECAST_LEN)
        if not np.isfinite(features[history_indices]).all():
            raise ValueError(f"{label}历史窗口含非有限特征")
        if not np.isfinite(target[target_indices]).all():
            raise ValueError(f"{label}目标窗口含非有限值")
    return origins


def load_prepared(farm_id: str) -> dict[str, Any]:
    paths = artifact_paths("patchtst", farm_id)
    array_path, bundle_path = paths["array"], paths["bundle"]
    arrays = _required_npz(array_path)
    manifest_path = paths["preprocess_manifest"]
    if not bundle_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(bundle_path)
    bundle_sha = sha256_file(bundle_path)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    summary = manifest.get("summary", {})
    if manifest.get("protocol_version") != PREPROCESS_PROTOCOL_VERSION:
        raise ValueError(f"{farm_id}预处理协议漂移")
    input_cols = tuple(map(str, arrays["input_cols"].tolist()))
    if str(summary.get("farm_id")) != farm_id:
        raise ValueError(f"manifest场站身份不符: {summary.get('farm_id')} != {farm_id}")
    features = np.asarray(arrays["features_scaled"], dtype=np.float32)
    target = np.asarray(arrays["target_scaled"], dtype=np.float32).reshape(-1)
    if features.ndim != 2 or features.shape[1] != INPUT_DIM:
        raise ValueError(f"{farm_id}特征形状必须为(*,{INPUT_DIM}): {features.shape}")
    if len(features) != len(target):
        raise ValueError(f"{farm_id}完整时间轴长度不一致")
    if len(input_cols) != INPUT_DIM or input_cols[TARGET_INDEX] != "功率":
        raise ValueError(f"{farm_id}固定45通道/功率索引协议漂移")
    schema_hash = hashlib.sha256(
        json.dumps(
            list(input_cols),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if schema_hash != EXPECTED_FEATURE_SCHEMA_HASH:
        raise ValueError(f"{farm_id}输入列语义/顺序与原五场站F7协议不一致")
    if summary.get("schema_hash") != schema_hash:
        raise ValueError(f"{farm_id}manifest schema hash不一致")
    train_origins = _validate_origins(
        arrays["train_origins"], features, target, "train"
    )
    val_origins = _validate_origins(arrays["val_origins"], features, target, "validation")
    if train_origins[-1] + FORECAST_LEN > val_origins[0]:
        raise ValueError(f"{farm_id}训练和验证目标区间重叠")
    # 在任何模型构建前截断到最后一个验证标签；测试特征/标签不进入训练状态。
    allowed_stop = int(val_origins[-1] + FORECAST_LEN)
    features = np.ascontiguousarray(features[:allowed_stop])
    target = np.ascontiguousarray(target[:allowed_stop])
    array_sha = sha256_file(array_path)
    if summary.get("array_sha256") != array_sha:
        raise ValueError(f"{farm_id} manifest记录的array SHA不一致")
    if summary.get("bundle_sha256") != bundle_sha:
        raise ValueError(f"{farm_id} manifest记录的bundle SHA不一致")
    y_mean = float(np.asarray(arrays["scaler_y_mean"]).reshape(-1)[0])
    y_scale = float(np.asarray(arrays["scaler_y_scale"]).reshape(-1)[0])
    x_mean = np.asarray(arrays["scaler_x_mean"], dtype=np.float64).reshape(-1)
    x_scale = np.asarray(arrays["scaler_x_scale"], dtype=np.float64).reshape(-1)
    reference = float(np.asarray(arrays["power_reference_mw"]).reshape(-1)[0])
    if not all(np.isfinite([y_mean, y_scale, reference])) or min(y_scale, reference) <= 0:
        raise ValueError(f"{farm_id}目标缩放或训练参考值无效")
    target_mw = target.astype(np.float64) * y_scale + y_mean
    return {
        "farm_id": farm_id,
        "features": features,
        "target": target,
        "target_mw": target_mw,
        "train_origins": train_origins,
        "val_origins": val_origins,
        "input_cols": list(input_cols),
        "schema_hash": schema_hash,
        "target_index": TARGET_INDEX,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "power_reference_mw": reference,
        "power_reference_kind": "train_power_q999",
        "power_scale_ratio": float(x_scale[TARGET_INDEX] / y_scale),
        "power_scale_offset": float((x_mean[TARGET_INDEX] - y_mean) / y_scale),
        "regime_feature_config": build_regime_config(
            input_cols, x_mean, x_scale, reference
        ),
        "training_feasibility": str(summary.get("training_feasibility", "unknown")),
        "array_path": str(array_path.resolve()),
        "array_sha256": array_sha,
        "bundle_path": str(bundle_path.resolve()),
        "preprocess_bundle_sha256": bundle_sha,
    }


def build_regime_config(
    input_cols: tuple[str, ...],
    means: np.ndarray,
    scales: np.ndarray,
    reference: float,
) -> dict[str, Any]:
    speed_names = (
        "10米风速",
        "30米风速",
        "50米风速",
        "70米风速",
        "轮毂高度风速",
    )
    speed_indices = [input_cols.index(name) for name in speed_names]
    sin_index = input_cols.index("轮毂高度风向_sin")
    cos_index = input_cols.index("轮毂高度风向_cos")
    return {
        "target_channel_index": TARGET_INDEX,
        "power_mean": float(means[TARGET_INDEX]),
        "power_scale": float(scales[TARGET_INDEX]),
        "capacity": float(reference),
        "wind_speed_indices": speed_indices,
        "wind_speed_names": list(speed_names),
        "wind_speed_means": [float(means[index]) for index in speed_indices],
        "wind_speed_scales": [float(scales[index]) for index in speed_indices],
        "hub_wind_position": 4,
        "direction_sin_index": sin_index,
        "direction_cos_index": cos_index,
        "direction_sin_mean": float(means[sin_index]),
        "direction_sin_scale": float(scales[sin_index]),
        "direction_cos_mean": float(means[cos_index]),
        "direction_cos_scale": float(scales[cos_index]),
        "windows": [4, 8, 16, 32],
        "low_power_threshold": 0.02,
        "wind_speed_normalizer": 25.0,
        "selected_groups": ["P", "H", "D"],
        "missing_safe_direction_required": True,
    }


def get_missing_safe_regime_layer_class() -> Any:
    """Return the F7 layer that treats a physical ``(0, 0)`` direction as unknown.

    The upstream layer normalizes every direction pair before calculating turn
    statistics.  Without this override, Round-3's explicit unknown sentinel
    would be interpreted as a half-turn signal.  TensorFlow stays lazily
    imported so the parent scheduler never creates a GPU context.
    """
    global _MISSING_SAFE_REGIME_LAYER_CLASS
    if _MISSING_SAFE_REGIME_LAYER_CLASS is not None:
        return _MISSING_SAFE_REGIME_LAYER_CLASS

    import tensorflow as tf
    from tensorflow import keras
    import wind_RegimeEncoder_PatchTST_train as regime_source

    @keras.utils.register_keras_serializable(
        package="Round3WindPRISM",
        name="MissingSafeExplicitWindRegimeFeatures",
    )
    class MissingSafeExplicitWindRegimeFeatures(
        regime_source.ExplicitWindRegimeFeatures
    ):
        """Preserve all upstream features and replace only the four D features."""

        def call(self, inputs: Any) -> Any:
            features = super().call(inputs)
            if (
                self.direction_sin_index is None
                or self.direction_cos_index is None
            ):
                return features

            direction_sin = self._physical_channel(
                inputs,
                self.direction_sin_index,
                self.direction_sin_mean,
                self.direction_sin_scale,
            )
            direction_cos = self._physical_channel(
                inputs,
                self.direction_cos_index,
                self.direction_cos_mean,
                self.direction_cos_scale,
            )
            magnitude = tf.sqrt(
                tf.square(direction_sin) + tf.square(direction_cos)
            )
            valid = magnitude > tf.cast(0.5, inputs.dtype)
            safe_magnitude = tf.maximum(
                magnitude, tf.cast(1e-6, inputs.dtype)
            )
            direction_sin = tf.where(
                valid,
                direction_sin / safe_magnitude,
                tf.zeros_like(direction_sin),
            )
            direction_cos = tf.where(
                valid,
                direction_cos / safe_magnitude,
                tf.zeros_like(direction_cos),
            )

            def turn(lag: int) -> Any:
                pair_valid = valid[:, -1] & valid[:, -1 - lag]
                dot = (
                    direction_sin[:, -1] * direction_sin[:, -1 - lag]
                    + direction_cos[:, -1] * direction_cos[:, -1 - lag]
                )
                value = 0.5 * (
                    1.0 - tf.clip_by_value(dot, -1.0, 1.0)
                )
                return tf.where(pair_valid, value, tf.zeros_like(value))

            recent_sin = direction_sin[:, -16:]
            recent_cos = direction_cos[:, -16:]
            pair_valid = valid[:, -15:] & valid[:, -16:-1]
            consecutive_dot = (
                recent_sin[:, 1:] * recent_sin[:, :-1]
                + recent_cos[:, 1:] * recent_cos[:, :-1]
            )
            consecutive_turn = 0.5 * (
                1.0 - tf.clip_by_value(consecutive_dot, -1.0, 1.0)
            )
            valid_float = tf.cast(pair_valid, inputs.dtype)
            mean_turn = tf.reduce_sum(
                consecutive_turn * valid_float, axis=1
            ) / tf.maximum(
                tf.reduce_sum(valid_float, axis=1),
                tf.cast(1.0, inputs.dtype),
            )
            direction_features = tf.stack(
                [turn(1), turn(4), turn(16), mean_turn], axis=-1
            )

            # Upstream order: P(20), H(12), M(3), D(4), C(4).
            return tf.concat(
                [features[:, :35], direction_features, features[:, 39:]],
                axis=-1,
            )

    _MISSING_SAFE_REGIME_LAYER_CLASS = MissingSafeExplicitWindRegimeFeatures
    return _MISSING_SAFE_REGIME_LAYER_CLASS


def get_round3_custom_objects() -> dict[str, Any]:
    """Custom-object map required to reload Round-3 WindPRISM models."""
    layer_class = get_missing_safe_regime_layer_class()
    return {
        "MissingSafeExplicitWindRegimeFeatures": layer_class,
        (
            "Round3WindPRISM>"
            "MissingSafeExplicitWindRegimeFeatures"
        ): layer_class,
    }


def get_itransformer_layer_classes() -> dict[str, Any]:
    """Create serializable Keras layers without importing TensorFlow in parent."""
    global _ITRANSFORMER_LAYER_CLASSES
    if _ITRANSFORMER_LAYER_CLASSES is not None:
        return _ITRANSFORMER_LAYER_CLASSES

    import tensorflow as tf
    from tensorflow import keras

    @keras.utils.register_keras_serializable(package="Round3ITransformer")
    class ITransformerInstanceNormalization(keras.layers.Layer):
        """Official per-window/per-variate normalization and retained statistics."""

        def __init__(self, epsilon: float = 1e-5, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.epsilon = float(epsilon)

        def call(self, inputs: Any) -> tuple[Any, Any, Any]:
            # Official implementation detaches only the window mean. Population
            # variance (unbiased=False) is equivalent to tf.math.reduce_variance.
            mean = tf.stop_gradient(
                tf.reduce_mean(inputs, axis=1, keepdims=True)
            )
            centered = inputs - mean
            stdev = tf.sqrt(
                tf.math.reduce_variance(
                    centered, axis=1, keepdims=True
                )
                + tf.cast(self.epsilon, inputs.dtype)
            )
            normalized = centered / stdev
            return normalized, mean, stdev

        def get_config(self) -> dict[str, Any]:
            return {**super().get_config(), "epsilon": self.epsilon}

    @keras.utils.register_keras_serializable(package="Round3ITransformer")
    class ITransformerEncoderBlock(keras.layers.Layer):
        """Post-norm full-attention encoder operating on variate tokens."""

        def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            dropout: float = 0.1,
            activation: str = "gelu",
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            if int(d_model) % int(num_heads):
                raise ValueError("iTransformer d_model必须能被num_heads整除")
            self.d_model = int(d_model)
            self.num_heads = int(num_heads)
            self.d_ff = int(d_ff)
            self.dropout_rate = float(dropout)
            self.activation_name = str(activation)
            self.attention = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                value_dim=self.d_model // self.num_heads,
                dropout=self.dropout_rate,
                use_bias=True,
                output_shape=self.d_model,
                name="full_variate_attention",
            )
            self.attention_dropout = keras.layers.Dropout(
                self.dropout_rate, name="attention_residual_dropout"
            )
            self.attention_norm = keras.layers.LayerNormalization(
                epsilon=1e-5, name="attention_post_norm"
            )
            self.ffn_dense_1 = keras.layers.Dense(
                self.d_ff,
                activation=keras.activations.get(self.activation_name),
                name="ffn_expand",
            )
            self.ffn_dropout_1 = keras.layers.Dropout(
                self.dropout_rate, name="ffn_activation_dropout"
            )
            self.ffn_dense_2 = keras.layers.Dense(
                self.d_model, name="ffn_project"
            )
            self.ffn_dropout_2 = keras.layers.Dropout(
                self.dropout_rate, name="ffn_residual_dropout"
            )
            self.ffn_norm = keras.layers.LayerNormalization(
                epsilon=1e-5, name="ffn_post_norm"
            )

        def call(self, inputs: Any, training: bool | None = None) -> Any:
            attended = self.attention(
                query=inputs,
                value=inputs,
                key=inputs,
                use_causal_mask=False,
                training=training,
            )
            attended = self.attention_dropout(
                attended, training=training
            )
            attention_output = self.attention_norm(inputs + attended)
            ffn = self.ffn_dense_1(attention_output)
            ffn = self.ffn_dropout_1(ffn, training=training)
            ffn = self.ffn_dense_2(ffn)
            ffn = self.ffn_dropout_2(ffn, training=training)
            return self.ffn_norm(attention_output + ffn)

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "d_ff": self.d_ff,
                "dropout": self.dropout_rate,
                "activation": self.activation_name,
            }

    @keras.utils.register_keras_serializable(package="Round3ITransformer")
    class ITransformerTargetHead(keras.layers.Layer):
        """De-normalize all tokens, then emit target power in y-scaler units."""

        def __init__(
            self,
            target_index: int,
            power_scale_ratio: float,
            power_scale_offset: float,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.target_index = int(target_index)
            self.power_scale_ratio = float(power_scale_ratio)
            self.power_scale_offset = float(power_scale_offset)

        def call(self, inputs: list[Any] | tuple[Any, ...]) -> Any:
            forecasts, mean, stdev = inputs
            denormalized = forecasts * stdev + mean
            target_x_scaled = denormalized[:, :, self.target_index]
            return (
                target_x_scaled
                * tf.cast(self.power_scale_ratio, target_x_scaled.dtype)
                + tf.cast(self.power_scale_offset, target_x_scaled.dtype)
            )

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "target_index": self.target_index,
                "power_scale_ratio": self.power_scale_ratio,
                "power_scale_offset": self.power_scale_offset,
            }

    _ITRANSFORMER_LAYER_CLASSES = {
        "ITransformerInstanceNormalization": ITransformerInstanceNormalization,
        "ITransformerEncoderBlock": ITransformerEncoderBlock,
        "ITransformerTargetHead": ITransformerTargetHead,
    }
    return _ITRANSFORMER_LAYER_CLASSES


def get_itransformer_custom_objects() -> dict[str, Any]:
    """Custom-object map required to reload Round-3 iTransformer models."""
    classes = get_itransformer_layer_classes()
    result: dict[str, Any] = {}
    for name, layer_class in classes.items():
        result[name] = layer_class
        result[f"Round3ITransformer>{name}"] = layer_class
    return result


def build_itransformer_model(
    prepared: dict[str, Any],
    keras: Any,
) -> Any:
    """Faithful Keras iTransformer adapted from 96 steps to a 16-step target."""
    d_model = ITRANSFORMER_D_MODEL
    num_heads = ITRANSFORMER_NUM_HEADS
    encoder_layers = ITRANSFORMER_ENCODER_LAYERS
    d_ff = ITRANSFORMER_D_FF
    dropout = ITRANSFORMER_DROPOUT
    classes = get_itransformer_layer_classes()

    inputs = keras.layers.Input(
        shape=(HISTORY_LEN, INPUT_DIM), name="history_features"
    )
    normalized, window_mean, window_stdev = classes[
        "ITransformerInstanceNormalization"
    ](
        epsilon=ITRANSFORMER_NORM_EPSILON,
        name="instance_normalization",
    )(inputs)
    # Official DataEmbedding_inverted: B×L×N -> B×N×L, followed by one
    # shared Linear(L, d_model).  No positional embedding is used.
    tokens = keras.layers.Permute(
        (2, 1), name="invert_time_and_variate_axes"
    )(normalized)
    tokens = keras.layers.Dense(
        d_model, name="inverted_value_embedding"
    )(tokens)
    tokens = keras.layers.Dropout(
        dropout, name="embedding_dropout"
    )(tokens)
    for index in range(encoder_layers):
        tokens = classes["ITransformerEncoderBlock"](
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            activation="gelu",
            name=f"inverted_encoder_{index + 1}",
        )(tokens)
    tokens = keras.layers.LayerNormalization(
        epsilon=ITRANSFORMER_NORM_EPSILON,
        name="encoder_final_norm",
    )(tokens)
    all_variate_forecasts = keras.layers.Dense(
        FORECAST_LEN, name="shared_forecast_projector"
    )(tokens)
    all_variate_forecasts = keras.layers.Permute(
        (2, 1), name="restore_forecast_and_variate_axes"
    )(all_variate_forecasts)
    forecast_power = classes["ITransformerTargetHead"](
        target_index=TARGET_INDEX,
        power_scale_ratio=prepared["power_scale_ratio"],
        power_scale_offset=prepared["power_scale_offset"],
        name="forecast_power",
    )([all_variate_forecasts, window_mean, window_stdev])
    return keras.Model(
        inputs=inputs,
        outputs=forecast_power,
        name="Round3_iTransformer_96to16",
    )


def get_timesnet_layer_classes() -> dict[str, Any]:
    """Create serializable Keras TimesNet layers without parent-side TF import."""
    global _TIMESNET_LAYER_CLASSES
    if _TIMESNET_LAYER_CLASSES is not None:
        return _TIMESNET_LAYER_CLASSES

    import tensorflow as tf
    from tensorflow import keras

    @keras.utils.register_keras_serializable(package="Round3TimesNet")
    class TimesNetInstanceNormalization(keras.layers.Layer):
        """Official forecast normalization: detached mean, live population std."""

        def __init__(self, epsilon: float = 1e-5, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.epsilon = float(epsilon)

        def call(self, inputs: Any) -> tuple[Any, Any, Any]:
            mean = tf.stop_gradient(
                tf.reduce_mean(inputs, axis=1, keepdims=True)
            )
            centered = inputs - mean
            stdev = tf.sqrt(
                tf.math.reduce_variance(
                    centered, axis=1, keepdims=True
                )
                + tf.cast(self.epsilon, inputs.dtype)
            )
            return centered / stdev, mean, stdev

        def get_config(self) -> dict[str, Any]:
            return {**super().get_config(), "epsilon": self.epsilon}

    @keras.utils.register_keras_serializable(package="Round3TimesNet")
    class TimesNetCircularConv1D(keras.layers.Layer):
        """Keras equivalent of the official circular Conv1d token embedding."""

        def __init__(
            self,
            filters: int,
            kernel_size: int = 3,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            if int(kernel_size) % 2 != 1:
                raise ValueError("TimesNet circular kernel_size必须为奇数")
            self.filters = int(filters)
            self.kernel_size = int(kernel_size)
            self.conv = keras.layers.Conv1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                padding="valid",
                use_bias=False,
                kernel_initializer=keras.initializers.VarianceScaling(
                    scale=2.0,
                    mode="fan_in",
                    distribution="untruncated_normal",
                ),
                name="token_conv",
            )

        def call(self, inputs: Any) -> Any:
            radius = self.kernel_size // 2
            if radius == 0:
                padded = inputs
            else:
                padded = tf.concat(
                    [inputs[:, -radius:, :], inputs, inputs[:, :radius, :]],
                    axis=1,
                )
            return self.conv(padded)

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "filters": self.filters,
                "kernel_size": self.kernel_size,
            }

    @keras.utils.register_keras_serializable(package="Round3TimesNet")
    class TimesNetDataEmbedding(keras.layers.Layer):
        """Token convolution plus fixed sinusoidal position encoding."""

        def __init__(
            self,
            d_model: int,
            dropout: float = 0.1,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.d_model = int(d_model)
            self.dropout_rate = float(dropout)
            self.value_embedding = TimesNetCircularConv1D(
                filters=self.d_model,
                kernel_size=3,
                name="circular_value_embedding",
            )
            self.dropout = keras.layers.Dropout(
                self.dropout_rate, name="embedding_dropout"
            )

        def call(
            self,
            inputs: Any,
            training: bool | None = None,
        ) -> Any:
            values = self.value_embedding(inputs)
            dtype = values.dtype
            positions = tf.cast(
                tf.range(tf.shape(values)[1])[:, None], dtype
            )
            dimensions = tf.range(self.d_model)[None, :]
            paired_dimensions = 2 * tf.math.floordiv(dimensions, 2)
            rates = tf.pow(
                tf.cast(10_000.0, dtype),
                -tf.cast(paired_dimensions, dtype)
                / tf.cast(self.d_model, dtype),
            )
            angles = positions * rates
            encoding = tf.where(
                tf.equal(tf.math.floormod(dimensions, 2), 0),
                tf.sin(angles),
                tf.cos(angles),
            )
            return self.dropout(
                values + encoding[None, :, :],
                training=training,
            )

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "d_model": self.d_model,
                "dropout": self.dropout_rate,
            }

    @keras.utils.register_keras_serializable(package="Round3TimesNet")
    class TimesNetInceptionBlockV1(keras.layers.Layer):
        """Six averaged square-kernel Conv2D branches from official TimesNet."""

        def __init__(
            self,
            out_channels: int,
            num_kernels: int = 6,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.out_channels = int(out_channels)
            self.num_kernels = int(num_kernels)
            self.kernels = [
                keras.layers.Conv2D(
                    filters=self.out_channels,
                    kernel_size=2 * index + 1,
                    padding="same",
                    use_bias=True,
                    kernel_initializer=keras.initializers.VarianceScaling(
                        scale=2.0,
                        mode="fan_out",
                        distribution="untruncated_normal",
                    ),
                    bias_initializer="zeros",
                    name=f"kernel_{2 * index + 1}x{2 * index + 1}",
                )
                for index in range(self.num_kernels)
            ]

        def call(self, inputs: Any) -> Any:
            branches = [kernel(inputs) for kernel in self.kernels]
            return tf.add_n(branches) / tf.cast(
                self.num_kernels, inputs.dtype
            )

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "out_channels": self.out_channels,
                "num_kernels": self.num_kernels,
            }

    @keras.utils.register_keras_serializable(package="Round3TimesNet")
    class TimesNetBlock(keras.layers.Layer):
        """FFT-discovered temporal 2D variation block with adaptive fusion."""

        def __init__(
            self,
            d_model: int,
            d_ff: int,
            top_k: int = 5,
            num_kernels: int = 6,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.d_model = int(d_model)
            self.d_ff = int(d_ff)
            self.top_k = int(top_k)
            self.num_kernels = int(num_kernels)
            self.expand = TimesNetInceptionBlockV1(
                out_channels=self.d_ff,
                num_kernels=self.num_kernels,
                name="inception_expand",
            )
            self.activation = keras.layers.Activation(
                keras.activations.gelu, name="gelu"
            )
            self.project = TimesNetInceptionBlockV1(
                out_channels=self.d_model,
                num_kernels=self.num_kernels,
                name="inception_project",
            )

        def call(self, inputs: Any) -> Any:
            # tf.signal.rfft works on the last axis, so [B,T,C] -> [B,C,F].
            spectrum = tf.signal.rfft(tf.transpose(inputs, [0, 2, 1]))
            amplitude = tf.abs(spectrum)
            global_amplitude = tf.reduce_mean(amplitude, axis=[0, 1])
            # Official code sets DC to zero.  -inf preserves the intent while
            # preventing an all-flat batch from selecting index 0 and dividing
            # by zero when converting frequency into period.
            global_amplitude = tf.concat(
                [
                    tf.fill(
                        [1],
                        tf.cast(-float("inf"), global_amplitude.dtype),
                    ),
                    global_amplitude[1:],
                ],
                axis=0,
            )
            _, frequency_indices = tf.math.top_k(
                global_amplitude, k=self.top_k, sorted=True
            )
            frequency_indices = tf.stop_gradient(frequency_indices)
            total_length = tf.shape(inputs)[1]
            periods = tf.math.floordiv(total_length, frequency_indices)
            batch = tf.shape(inputs)[0]
            channels = tf.shape(inputs)[2]
            periodic_outputs = []
            for index in range(self.top_k):
                period = periods[index]
                padded_length = (
                    tf.math.floordiv(total_length + period - 1, period)
                    * period
                )
                padding_length = padded_length - total_length
                padded = tf.pad(
                    inputs,
                    tf.stack(
                        [
                            tf.constant([0, 0], dtype=tf.int32),
                            tf.stack(
                                [
                                    tf.constant(0, dtype=tf.int32),
                                    padding_length,
                                ]
                            ),
                            tf.constant([0, 0], dtype=tf.int32),
                        ]
                    ),
                )
                two_dimensional = tf.reshape(
                    padded,
                    tf.stack(
                        [
                            batch,
                            tf.math.floordiv(padded_length, period),
                            period,
                            channels,
                        ]
                    ),
                )
                two_dimensional.set_shape(
                    (None, None, None, self.d_model)
                )
                convolved = self.project(
                    self.activation(self.expand(two_dimensional))
                )
                restored = tf.reshape(
                    convolved,
                    tf.stack([batch, padded_length, channels]),
                )
                restored.set_shape((None, None, self.d_model))
                periodic_outputs.append(restored[:, :total_length, :])
            stacked = tf.stack(periodic_outputs, axis=-1)
            sample_amplitude = tf.reduce_mean(amplitude, axis=1)
            period_weights = tf.nn.softmax(
                tf.gather(
                    sample_amplitude,
                    frequency_indices,
                    axis=1,
                ),
                axis=1,
            )
            period_weights = period_weights[:, None, None, :]
            return (
                tf.reduce_sum(stacked * period_weights, axis=-1)
                + inputs
            )

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "d_model": self.d_model,
                "d_ff": self.d_ff,
                "top_k": self.top_k,
                "num_kernels": self.num_kernels,
            }

    @keras.utils.register_keras_serializable(package="Round3TimesNet")
    class TimesNetTargetHead(keras.layers.Layer):
        """Return the last 16 target steps after official de-normalization."""

        def __init__(
            self,
            forecast_len: int,
            target_index: int,
            power_scale_ratio: float,
            power_scale_offset: float,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.forecast_len = int(forecast_len)
            self.target_index = int(target_index)
            self.power_scale_ratio = float(power_scale_ratio)
            self.power_scale_offset = float(power_scale_offset)

        def call(self, inputs: list[Any] | tuple[Any, ...]) -> Any:
            forecasts, mean, stdev = inputs
            target_normalized = forecasts[
                :, -self.forecast_len :, self.target_index
            ]
            target_x_scaled = (
                target_normalized * stdev[:, :, self.target_index]
                + mean[:, :, self.target_index]
            )
            return (
                target_x_scaled
                * tf.cast(self.power_scale_ratio, target_x_scaled.dtype)
                + tf.cast(self.power_scale_offset, target_x_scaled.dtype)
            )

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "forecast_len": self.forecast_len,
                "target_index": self.target_index,
                "power_scale_ratio": self.power_scale_ratio,
                "power_scale_offset": self.power_scale_offset,
            }

    _TIMESNET_LAYER_CLASSES = {
        "TimesNetInstanceNormalization": TimesNetInstanceNormalization,
        "TimesNetCircularConv1D": TimesNetCircularConv1D,
        "TimesNetDataEmbedding": TimesNetDataEmbedding,
        "TimesNetInceptionBlockV1": TimesNetInceptionBlockV1,
        "TimesNetBlock": TimesNetBlock,
        "TimesNetTargetHead": TimesNetTargetHead,
    }
    return _TIMESNET_LAYER_CLASSES


def get_timesnet_custom_objects() -> dict[str, Any]:
    """Custom-object map required to reload Round-3 TimesNet models."""
    classes = get_timesnet_layer_classes()
    result: dict[str, Any] = {}
    for name, layer_class in classes.items():
        result[name] = layer_class
        result[f"Round3TimesNet>{name}"] = layer_class
    return result


def build_timesnet_model(
    prepared: dict[str, Any],
    keras: Any,
) -> Any:
    """Official TimesNet forecasting flow adapted to 96→16 wind power."""
    classes = get_timesnet_layer_classes()
    inputs = keras.layers.Input(
        shape=(HISTORY_LEN, INPUT_DIM), name="history_features"
    )
    normalized, window_mean, window_stdev = classes[
        "TimesNetInstanceNormalization"
    ](
        epsilon=TIMESNET_NORM_EPSILON,
        name="instance_normalization",
    )(inputs)
    embedded = classes["TimesNetDataEmbedding"](
        d_model=TIMESNET_D_MODEL,
        dropout=TIMESNET_DROPOUT,
        name="data_embedding",
    )(normalized)
    # Official forecast head first expands the latent temporal dimension from
    # seq_len to seq_len+pred_len; TimesBlocks therefore process 112 positions.
    aligned = keras.layers.Permute(
        (2, 1), name="latent_time_to_last_axis"
    )(embedded)
    aligned = keras.layers.Dense(
        HISTORY_LEN + FORECAST_LEN,
        name="temporal_alignment",
    )(aligned)
    encoded = keras.layers.Permute(
        (2, 1), name="restore_latent_time_axis"
    )(aligned)
    shared_norm = keras.layers.LayerNormalization(
        epsilon=TIMESNET_NORM_EPSILON,
        name="shared_timesblock_norm",
    )
    for index in range(TIMESNET_ENCODER_LAYERS):
        encoded = classes["TimesNetBlock"](
            d_model=TIMESNET_D_MODEL,
            d_ff=TIMESNET_D_FF,
            top_k=TIMESNET_TOP_K,
            num_kernels=TIMESNET_NUM_KERNELS,
            name=f"times_block_{index + 1}",
        )(encoded)
        encoded = shared_norm(encoded)
    all_variate_forecasts = keras.layers.Dense(
        INPUT_DIM, name="all_variate_projection"
    )(encoded)
    forecast_power = classes["TimesNetTargetHead"](
        forecast_len=FORECAST_LEN,
        target_index=TARGET_INDEX,
        power_scale_ratio=prepared["power_scale_ratio"],
        power_scale_offset=prepared["power_scale_offset"],
        name="forecast_power",
    )([all_variate_forecasts, window_mean, window_stdev])
    return keras.Model(
        inputs=inputs,
        outputs=forecast_power,
        name="Round3_TimesNet_96to16",
    )


def get_timemixer_layer_classes() -> dict[str, Any]:
    """Create serializable Keras TimeMixer layers without parent-side TF import."""
    global _TIMEMIXER_LAYER_CLASSES
    if _TIMEMIXER_LAYER_CLASSES is not None:
        return _TIMEMIXER_LAYER_CLASSES

    import tensorflow as tf
    from tensorflow import keras

    @keras.utils.register_keras_serializable(package="Round3TimeMixer")
    class TimeMixerReversibleNormalization(keras.layers.Layer):
        """Official per-scale Normalize/RevIN with detached statistics."""

        def __init__(
            self,
            num_features: int,
            epsilon: float = 1e-5,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.num_features = int(num_features)
            self.epsilon = float(epsilon)

        def build(self, input_shape: Any) -> None:
            if int(input_shape[-1]) != self.num_features:
                raise ValueError(
                    "TimeMixer Normalize输入变量数漂移: "
                    f"{input_shape[-1]} != {self.num_features}"
                )
            self.affine_weight = self.add_weight(
                name="affine_weight",
                shape=(self.num_features,),
                initializer="ones",
                trainable=True,
            )
            self.affine_bias = self.add_weight(
                name="affine_bias",
                shape=(self.num_features,),
                initializer="zeros",
                trainable=True,
            )
            super().build(input_shape)

        def call(self, inputs: Any) -> tuple[Any, Any, Any]:
            mean = tf.stop_gradient(
                tf.reduce_mean(inputs, axis=1, keepdims=True)
            )
            centered = inputs - mean
            stdev = tf.stop_gradient(
                tf.sqrt(
                    tf.math.reduce_variance(
                        centered, axis=1, keepdims=True
                    )
                    + tf.cast(self.epsilon, inputs.dtype)
                )
            )
            normalized = centered / stdev
            normalized = (
                normalized
                * tf.cast(self.affine_weight, normalized.dtype)
                + tf.cast(self.affine_bias, normalized.dtype)
            )
            return normalized, mean, stdev

        def denormalize(self, inputs: Any, mean: Any, stdev: Any) -> Any:
            restored = inputs - tf.cast(self.affine_bias, inputs.dtype)
            restored = restored / (
                tf.cast(self.affine_weight, inputs.dtype)
                + tf.cast(self.epsilon * self.epsilon, inputs.dtype)
            )
            return restored * stdev + mean

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "num_features": self.num_features,
                "epsilon": self.epsilon,
            }

    @keras.utils.register_keras_serializable(package="Round3TimeMixer")
    class TimeMixerCircularConv1D(keras.layers.Layer):
        """Keras equivalent of TimeMixer's circular Conv1d token embedding."""

        def __init__(
            self,
            filters: int,
            kernel_size: int = 3,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            if int(kernel_size) % 2 != 1:
                raise ValueError("TimeMixer circular kernel_size必须为奇数")
            self.filters = int(filters)
            self.kernel_size = int(kernel_size)
            self.conv = keras.layers.Conv1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                padding="valid",
                use_bias=False,
                kernel_initializer=keras.initializers.VarianceScaling(
                    scale=2.0,
                    mode="fan_in",
                    distribution="untruncated_normal",
                ),
                name="token_conv",
            )

        def call(self, inputs: Any) -> Any:
            radius = self.kernel_size // 2
            if radius == 0:
                padded = inputs
            else:
                padded = tf.concat(
                    [inputs[:, -radius:, :], inputs, inputs[:, :radius, :]],
                    axis=1,
                )
            return self.conv(padded)

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "filters": self.filters,
                "kernel_size": self.kernel_size,
            }

    @keras.utils.register_keras_serializable(package="Round3TimeMixer")
    class TimeMixerDataEmbeddingWithoutPosition(keras.layers.Layer):
        """Shared value embedding used by official DataEmbedding_wo_pos."""

        def __init__(
            self,
            d_model: int,
            dropout: float,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.d_model = int(d_model)
            self.dropout_rate = float(dropout)
            self.value_embedding = TimeMixerCircularConv1D(
                filters=self.d_model,
                kernel_size=3,
                name="circular_value_embedding",
            )
            self.dropout = keras.layers.Dropout(
                self.dropout_rate, name="embedding_dropout"
            )

        def call(
            self,
            inputs: Any,
            training: bool | None = None,
        ) -> Any:
            return self.dropout(
                self.value_embedding(inputs), training=training
            )

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "d_model": self.d_model,
                "dropout": self.dropout_rate,
            }

    @keras.utils.register_keras_serializable(package="Round3TimeMixer")
    class TimeMixerPastDecomposableMixing(keras.layers.Layer):
        """PDM: seasonal bottom-up and trend top-down multiscale mixing."""

        def __init__(
            self,
            scale_lengths: tuple[int, ...] | list[int],
            d_model: int,
            d_ff: int,
            moving_average: int,
            dropout: float,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.scale_lengths = tuple(int(value) for value in scale_lengths)
            self.d_model = int(d_model)
            self.d_ff = int(d_ff)
            self.moving_average = int(moving_average)
            self.dropout_rate = float(dropout)
            if self.moving_average % 2 != 1:
                raise ValueError("TimeMixer moving_average必须为奇数")
            if len(self.scale_lengths) < 2:
                raise ValueError("TimeMixer至少需要两个时间尺度")
            self.season_down_sampling_layers = []
            for index in range(len(self.scale_lengths) - 1):
                next_length = self.scale_lengths[index + 1]
                self.season_down_sampling_layers.append(
                    keras.Sequential(
                        [
                            keras.layers.Dense(next_length),
                            keras.layers.Activation(
                                keras.activations.gelu
                            ),
                            keras.layers.Dense(next_length),
                        ],
                        name=f"season_fine_to_coarse_{index + 1}",
                    )
                )
            self.trend_up_sampling_layers = []
            for index in reversed(range(len(self.scale_lengths) - 1)):
                fine_length = self.scale_lengths[index]
                self.trend_up_sampling_layers.append(
                    keras.Sequential(
                        [
                            keras.layers.Dense(fine_length),
                            keras.layers.Activation(
                                keras.activations.gelu
                            ),
                            keras.layers.Dense(fine_length),
                        ],
                        name=(
                            "trend_coarse_to_fine_"
                            f"{len(self.scale_lengths) - index - 1}"
                        ),
                    )
                )
            self.out_cross_layer = keras.Sequential(
                [
                    keras.layers.Dense(self.d_ff),
                    keras.layers.Activation(keras.activations.gelu),
                    keras.layers.Dense(self.d_model),
                ],
                name="channel_independent_out_cross",
            )
            # These modules are registered in the official PDM class although
            # its forward() does not invoke them.  Building the LayerNorm keeps
            # the upstream module/state structure and parameter accounting.
            self.official_registered_layer_norm = (
                keras.layers.LayerNormalization(
                    epsilon=1e-5,
                    name="official_registered_layer_norm",
                )
            )
            self.official_registered_dropout = keras.layers.Dropout(
                self.dropout_rate,
                name="official_registered_dropout",
            )

        def build(self, input_shape: Any) -> None:
            self.official_registered_layer_norm.build(
                (None, None, self.d_model)
            )
            super().build(input_shape)

        def _decompose(self, inputs: Any) -> tuple[Any, Any]:
            radius = (self.moving_average - 1) // 2
            padded = tf.concat(
                [
                    tf.repeat(inputs[:, :1, :], radius, axis=1),
                    inputs,
                    tf.repeat(inputs[:, -1:, :], radius, axis=1),
                ],
                axis=1,
            )
            trend = tf.nn.avg_pool1d(
                padded,
                ksize=self.moving_average,
                strides=1,
                padding="VALID",
            )
            return inputs - trend, trend

        def call(
            self,
            inputs: list[Any] | tuple[Any, ...],
            training: bool | None = None,
        ) -> list[Any]:
            if len(inputs) != len(self.scale_lengths):
                raise ValueError("TimeMixer PDM输入尺度数漂移")
            season_list = []
            trend_list = []
            for scale in inputs:
                season, trend = self._decompose(scale)
                season_list.append(tf.transpose(season, [0, 2, 1]))
                trend_list.append(tf.transpose(trend, [0, 2, 1]))

            out_high = season_list[0]
            out_low = season_list[1]
            mixed_seasons = [tf.transpose(out_high, [0, 2, 1])]
            for index, down_layer in enumerate(
                self.season_down_sampling_layers
            ):
                out_low = out_low + down_layer(
                    out_high, training=training
                )
                out_high = out_low
                if index + 2 <= len(season_list) - 1:
                    out_low = season_list[index + 2]
                mixed_seasons.append(tf.transpose(out_high, [0, 2, 1]))

            reversed_trends = list(reversed(trend_list))
            out_low = reversed_trends[0]
            out_high = reversed_trends[1]
            mixed_trends_reversed = [tf.transpose(out_low, [0, 2, 1])]
            for index, up_layer in enumerate(
                self.trend_up_sampling_layers
            ):
                out_high = out_high + up_layer(
                    out_low, training=training
                )
                out_low = out_high
                if index + 2 <= len(reversed_trends) - 1:
                    out_high = reversed_trends[index + 2]
                mixed_trends_reversed.append(
                    tf.transpose(out_low, [0, 2, 1])
                )
            mixed_trends = list(reversed(mixed_trends_reversed))

            outputs = []
            for original, season, trend, length in zip(
                inputs,
                mixed_seasons,
                mixed_trends,
                self.scale_lengths,
            ):
                mixed = season + trend
                # channel_independence=1 branch in the official PDM.
                mixed = original + self.out_cross_layer(
                    mixed, training=training
                )
                outputs.append(mixed[:, :length, :])
            return outputs

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "scale_lengths": list(self.scale_lengths),
                "d_model": self.d_model,
                "d_ff": self.d_ff,
                "moving_average": self.moving_average,
                "dropout": self.dropout_rate,
            }

    @keras.utils.register_keras_serializable(package="Round3TimeMixer")
    class TimeMixerForecastCore(keras.layers.Layer):
        """Original TimeMixer PDM+FMM flow adapted to the fixed wind schema."""

        def __init__(
            self,
            history_len: int,
            forecast_len: int,
            num_features: int,
            target_index: int,
            d_model: int,
            d_ff: int,
            pdm_layers: int,
            downsampling_layers: int,
            downsampling_window: int,
            moving_average: int,
            dropout: float,
            norm_epsilon: float,
            power_scale_ratio: float,
            power_scale_offset: float,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.history_len = int(history_len)
            self.forecast_len = int(forecast_len)
            self.num_features = int(num_features)
            self.target_index = int(target_index)
            self.d_model = int(d_model)
            self.d_ff = int(d_ff)
            self.pdm_layers = int(pdm_layers)
            self.downsampling_layers = int(downsampling_layers)
            self.downsampling_window = int(downsampling_window)
            self.moving_average = int(moving_average)
            self.dropout_rate = float(dropout)
            self.norm_epsilon = float(norm_epsilon)
            self.power_scale_ratio = float(power_scale_ratio)
            self.power_scale_offset = float(power_scale_offset)
            self.scale_lengths = tuple(
                self.history_len // (self.downsampling_window**level)
                for level in range(self.downsampling_layers + 1)
            )
            if self.scale_lengths != TIMEMIXER_SCALE_LENGTHS:
                raise ValueError(
                    f"TimeMixer时间尺度漂移: {self.scale_lengths}"
                )
            self.normalize_layers = [
                TimeMixerReversibleNormalization(
                    num_features=self.num_features,
                    epsilon=self.norm_epsilon,
                    name=f"scale_{index}_normalization",
                )
                for index in range(self.downsampling_layers + 1)
            ]
            self.embedding = TimeMixerDataEmbeddingWithoutPosition(
                d_model=self.d_model,
                dropout=self.dropout_rate,
                name="shared_data_embedding_without_position",
            )
            self.pdm_blocks = [
                TimeMixerPastDecomposableMixing(
                    scale_lengths=self.scale_lengths,
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    moving_average=self.moving_average,
                    dropout=self.dropout_rate,
                    name=f"past_decomposable_mixing_{index + 1}",
                )
                for index in range(self.pdm_layers)
            ]
            self.predict_layers = [
                keras.layers.Dense(
                    self.forecast_len,
                    name=f"scale_{index}_future_predictor",
                )
                for index in range(self.downsampling_layers + 1)
            ]
            self.projection_layer = keras.layers.Dense(
                1, name="shared_channel_independent_projection"
            )

        def call(
            self,
            inputs: Any,
            training: bool | None = None,
        ) -> Any:
            batch = tf.shape(inputs)[0]
            raw_scales = [inputs]
            current = inputs
            for _ in range(self.downsampling_layers):
                current = tf.nn.avg_pool1d(
                    current,
                    ksize=self.downsampling_window,
                    strides=self.downsampling_window,
                    padding="VALID",
                )
                raw_scales.append(current)

            embedded_scales = []
            scale_zero_mean = scale_zero_stdev = None
            for index, (raw_scale, normalizer, length) in enumerate(
                zip(
                    raw_scales,
                    self.normalize_layers,
                    self.scale_lengths,
                )
            ):
                normalized, mean, stdev = normalizer(raw_scale)
                if index == 0:
                    scale_zero_mean, scale_zero_stdev = mean, stdev
                channel_independent = tf.reshape(
                    tf.transpose(normalized, [0, 2, 1]),
                    [-1, length, 1],
                )
                channel_independent.set_shape((None, length, 1))
                embedded_scales.append(
                    self.embedding(
                        channel_independent, training=training
                    )
                )

            encoded_scales = embedded_scales
            for pdm_block in self.pdm_blocks:
                encoded_scales = pdm_block(
                    encoded_scales, training=training
                )

            scale_predictions = []
            for encoded, predictor in zip(
                encoded_scales, self.predict_layers
            ):
                aligned = predictor(
                    tf.transpose(encoded, [0, 2, 1])
                )
                aligned = tf.transpose(aligned, [0, 2, 1])
                projected = self.projection_layer(aligned)
                projected = tf.reshape(
                    projected,
                    [
                        batch,
                        self.num_features,
                        self.forecast_len,
                    ],
                )
                scale_predictions.append(
                    tf.transpose(projected, [0, 2, 1])
                )
            all_variate_forecasts = tf.add_n(scale_predictions)
            restored = self.normalize_layers[0].denormalize(
                all_variate_forecasts,
                scale_zero_mean,
                scale_zero_stdev,
            )
            target_x_scaled = restored[:, :, self.target_index]
            forecast_power = (
                target_x_scaled
                * tf.cast(self.power_scale_ratio, target_x_scaled.dtype)
                + tf.cast(self.power_scale_offset, target_x_scaled.dtype)
            )
            forecast_power.set_shape((None, self.forecast_len))
            return forecast_power

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "history_len": self.history_len,
                "forecast_len": self.forecast_len,
                "num_features": self.num_features,
                "target_index": self.target_index,
                "d_model": self.d_model,
                "d_ff": self.d_ff,
                "pdm_layers": self.pdm_layers,
                "downsampling_layers": self.downsampling_layers,
                "downsampling_window": self.downsampling_window,
                "moving_average": self.moving_average,
                "dropout": self.dropout_rate,
                "norm_epsilon": self.norm_epsilon,
                "power_scale_ratio": self.power_scale_ratio,
                "power_scale_offset": self.power_scale_offset,
            }

    _TIMEMIXER_LAYER_CLASSES = {
        "TimeMixerReversibleNormalization": (
            TimeMixerReversibleNormalization
        ),
        "TimeMixerCircularConv1D": TimeMixerCircularConv1D,
        "TimeMixerDataEmbeddingWithoutPosition": (
            TimeMixerDataEmbeddingWithoutPosition
        ),
        "TimeMixerPastDecomposableMixing": (
            TimeMixerPastDecomposableMixing
        ),
        "TimeMixerForecastCore": TimeMixerForecastCore,
    }
    return _TIMEMIXER_LAYER_CLASSES


def get_timemixer_custom_objects() -> dict[str, Any]:
    """Custom-object map required to reload Round-3 TimeMixer models."""
    classes = get_timemixer_layer_classes()
    result: dict[str, Any] = {}
    for name, layer_class in classes.items():
        result[name] = layer_class
        result[f"Round3TimeMixer>{name}"] = layer_class
    return result


def build_timemixer_model(
    prepared: dict[str, Any],
    keras: Any,
) -> Any:
    """Original TimeMixer forecasting flow adapted to 96→16 wind power."""
    classes = get_timemixer_layer_classes()
    inputs = keras.layers.Input(
        shape=(HISTORY_LEN, INPUT_DIM), name="history_features"
    )
    forecast_power = classes["TimeMixerForecastCore"](
        history_len=HISTORY_LEN,
        forecast_len=FORECAST_LEN,
        num_features=INPUT_DIM,
        target_index=TARGET_INDEX,
        d_model=TIMEMIXER_D_MODEL,
        d_ff=TIMEMIXER_D_FF,
        pdm_layers=TIMEMIXER_PDM_LAYERS,
        downsampling_layers=TIMEMIXER_DOWNSAMPLING_LAYERS,
        downsampling_window=TIMEMIXER_DOWNSAMPLING_WINDOW,
        moving_average=TIMEMIXER_MOVING_AVERAGE,
        dropout=TIMEMIXER_DROPOUT,
        norm_epsilon=TIMEMIXER_NORM_EPSILON,
        power_scale_ratio=prepared["power_scale_ratio"],
        power_scale_offset=prepared["power_scale_offset"],
        name="forecast_power",
    )(inputs)
    return keras.Model(
        inputs=inputs,
        outputs=forecast_power,
        name="Round3_TimeMixer_96to16",
    )


def get_dlinear_layer_classes() -> dict[str, Any]:
    """Build serializable Keras layers faithful to the official DLinear."""
    global _DLINEAR_LAYER_CLASSES
    if _DLINEAR_LAYER_CLASSES is not None:
        return _DLINEAR_LAYER_CLASSES

    import tensorflow as tf
    from tensorflow import keras

    @keras.utils.register_keras_serializable(package="Round3DLinear")
    class DLinearSeriesDecomposition(keras.layers.Layer):
        """Endpoint-replicated moving average plus seasonal remainder."""

        def __init__(self, kernel_size: int = 25, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.kernel_size = int(kernel_size)
            if self.kernel_size <= 0 or self.kernel_size % 2 != 1:
                raise ValueError("DLinear moving-average kernel must be positive odd")

        def call(self, inputs: Any) -> tuple[Any, Any]:
            pad = (self.kernel_size - 1) // 2
            front = tf.repeat(inputs[:, :1, :], repeats=pad, axis=1)
            end = tf.repeat(inputs[:, -1:, :], repeats=pad, axis=1)
            padded = tf.concat([front, inputs, end], axis=1)
            trend = tf.nn.avg_pool1d(
                padded,
                ksize=self.kernel_size,
                strides=1,
                padding="VALID",
                data_format="NWC",
            )
            seasonal = inputs - trend
            return seasonal, trend

        def get_config(self) -> dict[str, Any]:
            return {**super().get_config(), "kernel_size": self.kernel_size}

    @keras.utils.register_keras_serializable(package="Round3DLinear")
    class DLinearForecastCore(keras.layers.Layer):
        """Direct 96→16 seasonal/trend linear forecast on every variate."""

        def __init__(
            self,
            history_len: int,
            forecast_len: int,
            num_features: int,
            target_index: int,
            moving_average: int = 25,
            individual: bool = False,
            power_scale_ratio: float = 1.0,
            power_scale_offset: float = 0.0,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.history_len = int(history_len)
            self.forecast_len = int(forecast_len)
            self.num_features = int(num_features)
            self.target_index = int(target_index)
            self.moving_average = int(moving_average)
            self.individual = bool(individual)
            self.power_scale_ratio = float(power_scale_ratio)
            self.power_scale_offset = float(power_scale_offset)
            self.decomposition = DLinearSeriesDecomposition(
                kernel_size=self.moving_average,
                name="series_decomposition",
            )
            # PyTorch nn.Linear defaults to U(-1/sqrt(in), +1/sqrt(in)) for
            # both kernel and bias.  The official optional 1/L initializer is
            # commented out, so it must not be activated here.
            bound = float(1.0 / np.sqrt(self.history_len))

            def make_linear(name: str) -> Any:
                # Use fresh initializer instances so the two official heads do
                # not accidentally receive identical Keras draws.
                return keras.layers.Dense(
                    self.forecast_len,
                    kernel_initializer=keras.initializers.RandomUniform(
                        -bound, bound
                    ),
                    bias_initializer=keras.initializers.RandomUniform(
                        -bound, bound
                    ),
                    name=name,
                )

            if self.individual:
                self.seasonal_linears = [
                    make_linear(f"seasonal_linear_{index}")
                    for index in range(self.num_features)
                ]
                self.trend_linears = [
                    make_linear(f"trend_linear_{index}")
                    for index in range(self.num_features)
                ]
                self.seasonal_linear = None
                self.trend_linear = None
            else:
                self.seasonal_linears = []
                self.trend_linears = []
                self.seasonal_linear = make_linear("seasonal_linear")
                self.trend_linear = make_linear("trend_linear")

        def call(self, inputs: Any) -> Any:
            seasonal, trend = self.decomposition(inputs)
            seasonal = tf.transpose(seasonal, [0, 2, 1])
            trend = tf.transpose(trend, [0, 2, 1])
            if self.individual:
                seasonal_forecast = tf.stack(
                    [
                        layer(seasonal[:, index, :])
                        for index, layer in enumerate(self.seasonal_linears)
                    ],
                    axis=1,
                )
                trend_forecast = tf.stack(
                    [
                        layer(trend[:, index, :])
                        for index, layer in enumerate(self.trend_linears)
                    ],
                    axis=1,
                )
            else:
                assert self.seasonal_linear is not None
                assert self.trend_linear is not None
                seasonal_forecast = self.seasonal_linear(seasonal)
                trend_forecast = self.trend_linear(trend)
            all_variate_forecast = tf.transpose(
                seasonal_forecast + trend_forecast,
                [0, 2, 1],
            )
            target = all_variate_forecast[:, :, self.target_index]
            return (
                target * tf.cast(self.power_scale_ratio, target.dtype)
                + tf.cast(self.power_scale_offset, target.dtype)
            )

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "history_len": self.history_len,
                "forecast_len": self.forecast_len,
                "num_features": self.num_features,
                "target_index": self.target_index,
                "moving_average": self.moving_average,
                "individual": self.individual,
                "power_scale_ratio": self.power_scale_ratio,
                "power_scale_offset": self.power_scale_offset,
            }

    _DLINEAR_LAYER_CLASSES = {
        "DLinearSeriesDecomposition": DLinearSeriesDecomposition,
        "DLinearForecastCore": DLinearForecastCore,
    }
    return _DLINEAR_LAYER_CLASSES


def get_dlinear_custom_objects() -> dict[str, Any]:
    """Custom-object map required to reload Round-3 DLinear models."""
    classes = get_dlinear_layer_classes()
    result: dict[str, Any] = {}
    for name, layer_class in classes.items():
        result[name] = layer_class
        result[f"Round3DLinear>{name}"] = layer_class
    return result


def build_dlinear_model(
    prepared: dict[str, Any],
    keras: Any,
) -> Any:
    """Official shared-head DLinear adapted to a 96→16 wind-power MS task."""
    classes = get_dlinear_layer_classes()
    inputs = keras.layers.Input(
        shape=(HISTORY_LEN, INPUT_DIM), name="history_features"
    )
    forecast_power = classes["DLinearForecastCore"](
        history_len=HISTORY_LEN,
        forecast_len=FORECAST_LEN,
        num_features=INPUT_DIM,
        target_index=TARGET_INDEX,
        moving_average=DLINEAR_MOVING_AVERAGE,
        individual=DLINEAR_INDIVIDUAL,
        power_scale_ratio=prepared["power_scale_ratio"],
        power_scale_offset=prepared["power_scale_offset"],
        name="forecast_power",
    )(inputs)
    return keras.Model(
        inputs=inputs,
        outputs=forecast_power,
        name="Round3_DLinear_96to16",
    )


def configure_worker_environment() -> tuple[Any, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    os.environ.setdefault(
        "MPLCONFIGDIR", str((RESULT_ROOT / "matplotlib_cache").resolve())
    )
    import tensorflow as tf
    from tensorflow import keras

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    return tf, keras


def make_dataset(
    tf: Any,
    prepared: dict[str, Any],
    origins: np.ndarray,
    batch_size: int,
    training: bool,
    dual_targets: bool,
) -> Any:
    features = tf.convert_to_tensor(prepared["features"], dtype=tf.float32)
    target = tf.convert_to_tensor(prepared["target"], dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(
        np.asarray(origins, dtype=np.int64)
    )
    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(origins), 20_000),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    def gather(origin: Any) -> tuple[Any, Any]:
        x_indices = tf.range(origin - HISTORY_LEN, origin, dtype=tf.int64)
        y_indices = tf.range(origin, origin + FORECAST_LEN, dtype=tf.int64)
        x = tf.gather(features, x_indices)
        y = tf.gather(target, y_indices)
        x = tf.ensure_shape(x, (HISTORY_LEN, INPUT_DIM))
        y = tf.ensure_shape(y, (FORECAST_LEN,))
        if dual_targets:
            return x, {"forecast_power": y, "candidate_forecast": y}
        return x, y

    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    dataset = dataset.map(gather, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(int(batch_size), drop_remainder=False)
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model(
    model_id: str,
    prepared: dict[str, Any],
    keras: Any,
) -> Any:
    if model_id == "patchtst":
        import wind_dl_model_train as source

        keras.utils.set_random_seed(RANDOM_SEED)
        return source.build_patchtst_model(INPUT_DIM, TARGET_INDEX)
    if model_id in OTHER_MODELS:
        import wind_dl_other_models_train as source

        keras.utils.set_random_seed(RANDOM_SEED)
        builder = source.MODEL_BUILDERS[model_id]
        if model_id in {"informer", "autoformer"}:
            return builder((HISTORY_LEN, INPUT_DIM), input_cols=prepared["input_cols"])
        return builder((HISTORY_LEN, INPUT_DIM))
    if model_id == "hr_moe_fets_patchtst":
        import wind_FeTS_PatchTST_train as source

        keras.utils.set_random_seed(RANDOM_SEED)
        return source.build_fets_patchtst_model(
            input_dim=INPUT_DIM,
            target_channel_index=TARGET_INDEX,
            power_scale_ratio=prepared["power_scale_ratio"],
            power_scale_offset=prepared["power_scale_offset"],
        )
    if model_id == "itransformer":
        # Keras reimplementation of thuml/iTransformer.  The model remains a
        # pure encoder-only variate-token baseline; no WindPRISM residual,
        # Persistence candidate or regime gate is introduced.
        keras.utils.set_random_seed(RANDOM_SEED)
        return build_itransformer_model(prepared, keras)
    if model_id == "timesnet":
        # Keras reimplementation of thuml/Time-Series-Library TimesNet.  It
        # retains dynamic FFT period discovery and temporal 2D variation
        # modeling without adding WindPRISM-specific candidates or routing.
        keras.utils.set_random_seed(RANDOM_SEED)
        return build_timesnet_model(prepared, keras)
    if model_id == "timemixer":
        # Keras reimplementation of kwuking/TimeMixer.  It retains the
        # original channel-independent PDM and summed multiscale FMM path,
        # without adding WindPRISM candidates, residuals or regime routing.
        keras.utils.set_random_seed(RANDOM_SEED)
        return build_timemixer_model(prepared, keras)
    if model_id == "dlinear":
        # Keras reimplementation of honeywell21/DLinear.  It retains the
        # official shared seasonal/trend temporal heads and deliberately adds
        # no cross-variate projection or WindPRISM-specific component.
        keras.utils.set_random_seed(RANDOM_SEED)
        return build_dlinear_model(prepared, keras)
    if model_id == "windprism_f7_g0":
        import wind_RegimeEncoder_PatchTST_feature_screen_train as source
        import wind_RegimeEncoder_PatchTST_train as regime_source

        keras.utils.set_random_seed(RANDOM_SEED)
        if tuple(source.VARIANT_SPECS["f7"]["groups"]) != ("P", "H", "D"):
            raise ValueError("F7已不再对应P+H+D")
        missing_safe_layer = get_missing_safe_regime_layer_class()
        upstream_layer = regime_source.ExplicitWindRegimeFeatures
        regime_source.ExplicitWindRegimeFeatures = missing_safe_layer
        try:
            model = source.build_feature_screen_model(
                "f7",
                INPUT_DIM,
                TARGET_INDEX,
                prepared["power_scale_ratio"],
                prepared["power_scale_offset"],
                prepared["regime_feature_config"],
            )
        finally:
            regime_source.ExplicitWindRegimeFeatures = upstream_layer
        for layer in model.layers:
            layer.trainable = True
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=LEARNING_RATE, clipnorm=CLIPNORM
            ),
            loss={
                "forecast_power": keras.losses.Huber(delta=1.0),
                "candidate_forecast": keras.losses.Huber(delta=1.0),
            },
            loss_weights={"forecast_power": 1.0, "candidate_forecast": 0.5},
            metrics={
                "forecast_power": [
                    keras.metrics.MeanAbsoluteError(name="mae"),
                    keras.metrics.RootMeanSquaredError(name="rmse"),
                ],
                "candidate_forecast": [
                    keras.metrics.MeanAbsoluteError(name="mae")
                ],
            },
        )
        return model
    raise ValueError(model_id)


def extract_primary_prediction(
    prediction: Any,
    output_names: Iterable[str] | None = None,
) -> np.ndarray:
    if isinstance(prediction, dict):
        if "forecast_power" not in prediction:
            raise KeyError(f"多输出模型缺少forecast_power: {list(prediction)}")
        prediction = prediction["forecast_power"]
    elif isinstance(prediction, (tuple, list)):
        names = list(output_names or ())
        index = names.index("forecast_power") if "forecast_power" in names else 0
        prediction = prediction[index]
    values = np.asarray(prediction, dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or values.shape[1] != FORECAST_LEN:
        raise ValueError(f"预测输出形状异常: {values.shape}")
    if not np.isfinite(values).all():
        raise FloatingPointError("预测包含非有限值")
    return values


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, reference: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape or not np.isfinite(y_true).all():
        raise ValueError("验证真值/预测形状或有限性异常")
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denominator = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = (
        float(1.0 - np.sum(np.square(error)) / denominator)
        if denominator > 1e-12
        else float("nan")
    )
    smape = float(
        100.0
        * np.mean(
            2.0
            * np.abs(error)
            / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-6)
        )
    )
    return {
        "mae_mw": mae,
        "rmse_mw": rmse,
        "nmae": mae / reference,
        "nrmse": rmse / reference,
        "r2": r2,
        "smape_percent": smape,
    }


def evaluate_validation(
    model: Any,
    val_dataset: Any,
    prepared: dict[str, Any],
    model_id: str,
) -> dict[str, Any]:
    output = model.predict(val_dataset, verbose=0)
    output_names = list(getattr(model, "output_names", ()))
    forecast_scaled = extract_primary_prediction(output, output_names)
    origins = prepared["val_origins"]
    target_indices = origins[:, None] + np.arange(FORECAST_LEN)
    y_true_scaled = prepared["target"][target_indices].astype(np.float64)
    if forecast_scaled.shape != y_true_scaled.shape:
        raise ValueError("验证预测样本数与冻结origins不一致")
    residual = forecast_scaled - y_true_scaled
    absolute = np.abs(residual)
    huber = np.where(absolute <= 1.0, 0.5 * residual**2, absolute - 0.5)
    y_true = prepared["target_mw"][target_indices]
    y_pred = np.maximum(
        forecast_scaled * prepared["y_scale"] + prepared["y_mean"], 0.0
    )
    values: dict[str, Any] = {
        "farm_id": prepared["farm_id"],
        "model_id": model_id,
        "validation_windows": int(len(origins)),
        "val_huber_loss": float(np.mean(huber)),
        "val_forecast_huber_scaled": float(np.mean(huber)),
        **{f"val_{key}": value for key, value in metric_values(
            y_true, y_pred, prepared["power_reference_mw"]
        ).items()},
    }
    if prepared["power_reference_kind"] == "train_power_q999":
        values["val_trnmae"] = values["val_nmae"]
        values["val_trnrmse"] = values["val_nrmse"]
    candidate_output = None
    if isinstance(output, dict):
        candidate_output = output.get("candidate_forecast")
    elif isinstance(output, (tuple, list)) and "candidate_forecast" in output_names:
        candidate_output = output[output_names.index("candidate_forecast")]
    if candidate_output is not None:
        candidate_scaled = np.asarray(candidate_output, dtype=np.float64)
        candidate = np.maximum(
            candidate_scaled * prepared["y_scale"] + prepared["y_mean"], 0.0
        )
        values.update(
            {
                f"val_corrected_candidate_{key}": value
                for key, value in metric_values(
                    y_true, candidate, prepared["power_reference_mw"]
                ).items()
            }
        )
        if prepared["power_reference_kind"] == "train_power_q999":
            values["val_corrected_candidate_trnmae"] = values[
                "val_corrected_candidate_nmae"
            ]
            values["val_corrected_candidate_trnrmse"] = values[
                "val_corrected_candidate_nrmse"
            ]
    else:
        for key in ("mae_mw", "rmse_mw", "nmae", "nrmse", "r2", "smape_percent"):
            values[f"val_corrected_candidate_{key}"] = float("nan")
        values["val_corrected_candidate_trnmae"] = float("nan")
        values["val_corrected_candidate_trnrmse"] = float("nan")
    return values


def history_columns(frame: pd.DataFrame) -> dict[str, tuple[str | None, str | None]]:
    def pair(train_names: tuple[str, ...], val_names: tuple[str, ...]) -> tuple[str | None, str | None]:
        train = next((name for name in train_names if name in frame), None)
        val = next((name for name in val_names if name in frame), None)
        return train, val

    return {
        "loss": pair(
            ("loss", "forecast_power_loss"),
            ("val_loss", "val_forecast_power_loss"),
        ),
        "mae": pair(
            ("forecast_power_mae", "mae"),
            ("val_forecast_power_mae", "val_mae"),
        ),
        "rmse": pair(
            ("forecast_power_rmse", "rmse"),
            ("val_forecast_power_rmse", "val_rmse"),
        ),
    }


def save_history(history: Any, csv_path: Path, plot_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.DataFrame(history.history)
    if frame.empty:
        raise ValueError("训练history为空")
    numeric = frame.select_dtypes(include=[np.number])
    check = numeric.drop(columns=["lr", "learning_rate"], errors="ignore")
    if check.empty or not np.isfinite(check.to_numpy(dtype=float)).all():
        raise ValueError("训练history为空或包含非有限值")
    frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
    atomic_csv(csv_path, frame)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = history_columns(frame)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for axis, (metric, names) in zip(axes, pairs.items()):
        train_name, val_name = names
        if train_name:
            axis.plot(frame["epoch"], frame[train_name], label="train")
        if val_name:
            axis.plot(frame["epoch"], frame[val_name], label="validation")
        axis.set_title(metric.upper())
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        if train_name or val_name:
            axis.legend()
        else:
            axis.text(0.5, 0.5, "metric not logged", ha="center", va="center")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    loss_train, loss_val = pairs["loss"]
    rmse_train, rmse_val = pairs["rmse"]
    overfit: dict[str, Any] = {
        "epochs_ran": int(len(frame)),
        "validation_overfit_gap": float("nan"),
        "probable_overfit": False,
        "diagnostic_rule": (
            "last10 train loss strictly decreases while validation RMSE "
            "(or validation loss fallback) strictly increases"
        ),
    }
    if loss_train and loss_val:
        overfit["validation_overfit_gap"] = float(
            frame[loss_val].iloc[-1] - frame[loss_train].iloc[-1]
        )
    validation_trace = rmse_val or loss_val
    if loss_train and validation_trace and len(frame) >= 10:
        train_tail = frame[loss_train].to_numpy(dtype=float)[-10:]
        val_tail = frame[validation_trace].to_numpy(dtype=float)[-10:]
        overfit["probable_overfit"] = bool(
            np.all(np.diff(train_tail) < 0) and np.all(np.diff(val_tail) > 0)
        )
    return frame, overfit


def model_size_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    if path.is_dir():
        return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))
    return 0


def run_preflight_worker(args: argparse.Namespace) -> int:
    """在隔离子进程执行完整train+validation单epoch显存预检。"""
    if args.model not in PREFLIGHT_MODELS or not args.farm or not args.batch_size:
        raise ValueError("preflight worker参数不完整")
    model_id = str(args.model)
    farm_id = str(args.farm)
    batch_size = int(args.batch_size)
    attempt_dir = Path(args.attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    marker_path = attempt_dir / "preflight_attempt.json"
    started = time.monotonic()
    tf = keras = model = None
    try:
        tf, keras = configure_worker_environment()
        physical_gpus = tf.config.list_physical_devices("GPU")
        if not physical_gpus:
            raise RuntimeError(
                "GPU预检要求至少一个TensorFlow可见GPU；禁止用CPU结果发布显存策略"
            )
        prepared = load_prepared(farm_id)
        train_ds = make_dataset(
            tf,
            prepared,
            prepared["train_origins"],
            batch_size,
            training=True,
            dual_targets=False,
        )
        val_ds = make_dataset(
            tf,
            prepared,
            prepared["val_origins"],
            batch_size,
            training=False,
            dual_targets=False,
        )
        model = build_model(model_id, prepared, keras)
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=LEARNING_RATE, clipnorm=CLIPNORM
            ),
            loss=keras.losses.Huber(delta=1.0),
            metrics=[
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
        )
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.reset_memory_stats("GPU:0")
            except Exception:
                break
        fit_started = time.monotonic()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=1,
            verbose=2,
        )
        fit_seconds = time.monotonic() - fit_started
        history_values = pd.DataFrame(history.history).select_dtypes(
            include=[np.number]
        )
        if history_values.empty or not np.isfinite(
            history_values.to_numpy(dtype=float)
        ).all():
            raise FloatingPointError("GPU预检history为空或包含非有限值")
        peak_gpu_bytes = None
        try:
            peak_gpu_bytes = int(
                tf.config.experimental.get_memory_info("GPU:0")["peak"]
            )
        except Exception:
            pass
        payload = {
            "status": "complete",
            "created_at": utc_now(),
            "protocol_version": PROTOCOL_VERSION,
            "preflight_kind": "full_train_validation_single_epoch",
            "formal_artifact_published": False,
            "model_id": model_id,
            "farm_id": farm_id,
            "batch_size": batch_size,
            "random_seed": RANDOM_SEED,
            "training_initialization": "from_scratch_seed_2026",
            "pretrained_weights_loaded": False,
            "train_windows": int(len(prepared["train_origins"])),
            "validation_windows": int(len(prepared["val_origins"])),
            "train_steps": int(
                np.ceil(len(prepared["train_origins"]) / batch_size)
            ),
            "validation_steps": int(
                np.ceil(len(prepared["val_origins"]) / batch_size)
            ),
            "parameter_count": int(model.count_params()),
            "fit_seconds": fit_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_memory_bytes": peak_gpu_bytes,
            "physical_gpu_names": [
                str(getattr(gpu, "name", gpu)) for gpu in physical_gpus
            ],
            "gpu_preflight_verified": True,
            "array_sha256": prepared["array_sha256"],
            "preprocess_bundle_sha256": prepared["preprocess_bundle_sha256"],
            "training_code_sha256": sha256_file(__file__),
        }
        atomic_json(marker_path, payload)
        print(
            f"[preflight complete] {model_id}/{farm_id}, batch={batch_size}, "
            f"fit={fit_seconds:.1f}s"
        )
        return 0
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        oom = is_confirmed_cuda_oom(exc, text, tf, model)
        atomic_json(
            marker_path,
            {
                "status": "oom" if oom else "failed",
                "created_at": utc_now(),
                "protocol_version": PROTOCOL_VERSION,
                "model_id": model_id,
                "farm_id": farm_id,
                "batch_size": batch_size,
                "exception_type": type(exc).__name__,
                "message": str(exc)[:4000],
                "cuda_oom_confirmed": oom,
                "elapsed_seconds": time.monotonic() - started,
                "training_code_sha256": sha256_file(__file__),
            },
        )
        print(text, file=sys.stderr)
        return OOM_EXIT_CODE if oom else 1
    finally:
        if keras is not None:
            try:
                keras.backend.clear_session()
            except Exception:
                pass
        del model
        gc.collect()


def run_worker(args: argparse.Namespace) -> int:
    if not args.model or not args.farm or not args.batch_size:
        raise ValueError("worker必须指定--model/--farm/--batch-size")
    model_id, farm_id = args.model, args.farm
    batch_size = int(args.batch_size)
    smoke = bool(args.smoke)
    extension_lineage = str(args.extension_lineage or "")
    if smoke:
        extension_lineage = "smoke"
    elif model_id in MODERN_TRAINABLE_MODEL_IDS:
        if extension_lineage not in {
            STAGED_EXTENSION_LINEAGE,
            UNIFIED_MODERN_EXTENSION_LINEAGE,
        }:
            raise ValueError(
                f"{model_id}正式worker缺少有效extension lineage"
            )
    elif extension_lineage:
        raise ValueError(
            f"原10模型worker不应声明modern extension lineage: {model_id}"
        )
    batch_policy = None if smoke else load_batch_policy(require_valid_sources=True)
    if batch_policy is not None:
        expected_batch = formal_batch_size(model_id, batch_policy)
        if batch_size != expected_batch:
            raise ValueError(
                f"{model_id} worker batch={batch_size}与全局策略{expected_batch}不符"
            )
    paths = artifact_paths(model_id, farm_id, smoke=smoke)
    attempt_dir = Path(args.attempt_dir or paths["attempt_root"] / f"attempt_bs{batch_size}")
    attempt_dir.mkdir(parents=True, exist_ok=True)
    attempt_marker = attempt_dir / "attempt.json"
    # 正式complete marker只代表本次成功；重跑开始即撤销旧marker。
    paths["marker"].unlink(missing_ok=True)
    started_wall = utc_now()
    started = time.monotonic()
    tf = keras = model = None
    try:
        tf, keras = configure_worker_environment()
        if not smoke and not tf.config.list_physical_devices("GPU"):
            raise RuntimeError("正式训练要求TensorFlow可见GPU，当前仅检测到CPU")

        class FiniteLogGuard(keras.callbacks.Callback):
            def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
                for key, value in (logs or {}).items():
                    if key not in {"lr", "learning_rate"} and value is not None:
                        if not np.isfinite(float(value)):
                            raise FloatingPointError(
                                f"epoch={epoch + 1}日志{key}非有限: {value}"
                            )

        prepared = load_prepared(farm_id)
        dual_targets = model_id == "windprism_f7_g0"
        train_ds = make_dataset(
            tf,
            prepared,
            prepared["train_origins"],
            batch_size,
            training=True,
            dual_targets=dual_targets,
        )
        val_ds = make_dataset(
            tf,
            prepared,
            prepared["val_origins"],
            batch_size,
            training=False,
            dual_targets=dual_targets,
        )
        model = build_model(model_id, prepared, keras)
        if model_id != "windprism_f7_g0":
            # 覆盖可能由外部环境变量影响的旧模块compile状态；不改模型结构。
            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=LEARNING_RATE, clipnorm=CLIPNORM
                ),
                loss=keras.losses.Huber(delta=1.0),
                metrics=[
                    keras.metrics.MeanAbsoluteError(name="mae"),
                    keras.metrics.RootMeanSquaredError(name="rmse"),
                ],
            )
        if tuple(model.input_shape[1:]) != (HISTORY_LEN, INPUT_DIM):
            raise ValueError(f"模型输入形状漂移: {model.input_shape}")
        if model_id in {
            "itransformer",
            "timesnet",
            "timemixer",
            "dlinear",
        } and tuple(
            model.output_shape[1:]
        ) != (FORECAST_LEN,):
            raise ValueError(
                f"{model_id}输出必须为(None,{FORECAST_LEN}): "
                f"{model.output_shape}"
            )
        parameter_count = int(model.count_params())
        trainable_parameter_count = int(
            sum(int(np.prod(weight.shape)) for weight in model.trainable_weights)
        )
        expected_params = EXPECTED_PARAMETER_COUNTS[model_id]
        if parameter_count != expected_params:
            raise ValueError(
                f"{model_id}参数量漂移: {parameter_count:,} != {expected_params:,}"
            )

        for key in ("model", "weights", "history", "history_plot", "validation", "overfit"):
            paths[key].parent.mkdir(parents=True, exist_ok=True)
        paths["tensorboard"].mkdir(parents=True, exist_ok=True)
        monitor = (
            "val_forecast_power_loss"
            if model_id == "windprism_f7_g0"
            else "val_loss"
        )
        epochs = 1 if smoke else EPOCHS[model_id]
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(paths["weights"]),
                monitor=monitor,
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                mode="min",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                mode="min",
                factor=REDUCE_LR_FACTOR,
                patience=REDUCE_LR_PATIENCE,
                min_lr=MIN_LEARNING_RATE,
                verbose=1,
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(paths["tensorboard"] / datetime.now().strftime("%Y%m%d-%H%M%S")),
                histogram_freq=0,
                write_graph=True,
                profile_batch=0,
            ),
            keras.callbacks.TerminateOnNaN(),
            FiniteLogGuard(),
        ]
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.reset_memory_stats("GPU:0")
            except Exception:
                break
        fit_started = time.monotonic()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2,
        )
        fit_seconds = time.monotonic() - fit_started
        if not paths["weights"].is_file():
            raise FileNotFoundError(f"best checkpoint未生成: {paths['weights']}")
        model.load_weights(paths["weights"])
        history_frame, overfit = save_history(
            history, paths["history"], paths["history_plot"]
        )
        validation = evaluate_validation(model, val_ds, prepared, model_id)
        model.save(paths["model"])
        peak_gpu_bytes = None
        try:
            peak_gpu_bytes = int(
                tf.config.experimental.get_memory_info("GPU:0")["peak"]
            )
        except Exception:
            pass
        best_epoch = int(
            np.nanargmin(history_frame[monitor].to_numpy(dtype=float)) + 1
            if monitor in history_frame
            else len(history_frame)
        )
        overfit.update(
            {
                "model_id": model_id,
                "farm_id": farm_id,
                "training_feasibility": prepared["training_feasibility"],
            }
        )
        atomic_json(paths["validation"], validation)
        atomic_json(paths["overfit"], overfit)
        marker = {
            "status": "complete",
            "created_at": utc_now(),
            "started_at": started_wall,
            "protocol_version": PROTOCOL_VERSION,
            "preprocess_protocol_version": PREPROCESS_PROTOCOL_VERSION,
            "model_id": model_id,
            "model_name": model.name,
            "farm_id": farm_id,
            "extension_lineage": (
                extension_lineage
                if model_id in MODERN_TRAINABLE_MODEL_IDS
                else None
            ),
            "random_seed": RANDOM_SEED,
            "training_initialization": "from_scratch_seed_2026",
            "pretrained_weights_loaded": False,
            "history_len": HISTORY_LEN,
            "forecast_len": FORECAST_LEN,
            "input_dim": INPUT_DIM,
            "target_index": TARGET_INDEX,
            "schema_hash": prepared["schema_hash"],
            "requested_batch_size": DEFAULT_BATCH_SIZE,
            "effective_batch_size": batch_size,
            "batch_size": batch_size,
            # The formal task starts directly at the batch frozen by the
            # preflight policy; it is not itself an OOM retry.
            "fallback_triggered": False,
            "task_retry_after_oom": False,
            "global_policy_fallback_active": bool(
                batch_policy
                and batch_policy.get("hr_moe_global_fallback_triggered")
                and model_id == HEAVY_FALLBACK_MODEL
            ),
            "global_fallback_triggered": bool(
                batch_policy
                and batch_policy.get("hr_moe_global_fallback_triggered")
                and model_id == HEAVY_FALLBACK_MODEL
            ),
            "global_batch_policy_path": (
                str(BATCH_POLICY_PATH.resolve()) if batch_policy else None
            ),
            "global_batch_policy_sha256": (
                sha256_file(BATCH_POLICY_PATH) if batch_policy else None
            ),
            "global_batch_policy_reason": (
                batch_policy.get("hr_moe_batch_reason") if batch_policy else None
            ),
            "attempted_batch_sizes": [batch_size],
            "epochs_requested": epochs,
            "epochs_ran": int(len(history_frame)),
            "best_epoch": best_epoch,
            "early_stopping_monitor": monitor,
            "checkpoint_monitor": monitor,
            "learning_rate": LEARNING_RATE,
            "clipnorm": CLIPNORM,
            "optimizer": "Adam",
            "loss": "Huber(delta=1.0)",
            "candidate_supervision_loss_weight": (
                0.5 if model_id == "windprism_f7_g0" else None
            ),
            "missing_direction_semantics": (
                "physical (sin,cos)=(0,0) contributes zero D-feature evidence"
                if model_id == "windprism_f7_g0"
                else None
            ),
            "train_windows": int(len(prepared["train_origins"])),
            "validation_windows": int(len(prepared["val_origins"])),
            "training_feasibility": prepared["training_feasibility"],
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "fit_seconds": fit_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_memory_bytes": peak_gpu_bytes,
            "python_version": sys.version.split()[0],
            "tensorflow_version": tf.__version__,
            "keras_version": getattr(keras, "__version__", "tensorflow.keras"),
            "model_size_bytes": model_size_bytes(paths["model"]),
            "model_path": str(paths["model"].resolve()),
            "model_sha256": sha256_file(paths["model"]),
            "weights_path": str(paths["weights"].resolve()),
            "weights_sha256": sha256_file(paths["weights"]),
            "history_path": str(paths["history"].resolve()),
            "history_sha256": sha256_file(paths["history"]),
            "history_plot_path": str(paths["history_plot"].resolve()),
            "history_plot_sha256": sha256_file(paths["history_plot"]),
            "validation_path": str(paths["validation"].resolve()),
            "validation_sha256": sha256_file(paths["validation"]),
            "overfit_path": str(paths["overfit"].resolve()),
            "overfit_sha256": sha256_file(paths["overfit"]),
            "array_path": prepared["array_path"],
            "array_sha256": prepared["array_sha256"],
            "preprocess_bundle_path": prepared["bundle_path"],
            "preprocess_bundle_sha256": prepared["preprocess_bundle_sha256"],
            "power_reference_kind": prepared["power_reference_kind"],
            "power_reference_mw": prepared["power_reference_mw"],
            "validation_metrics": validation,
            "overfit_diagnostics": overfit,
            "expert_names": (
                ["long", "mid", "short", "persistence"]
                if model_id == "hr_moe_fets_patchtst"
                else ["persistence", "corrected"]
                if model_id == "windprism_f7_g0"
                else []
            ),
            "diagnostic_layers": (
                {
                    "forecast": "forecast_power",
                    "router": "horizon_regime_router",
                    "experts": [
                        "baseline_forecast_power",
                        "mid_forecast_candidate",
                        "local_forecast_candidate",
                        "persistence_forecast_candidate",
                    ],
                }
                if model_id == "hr_moe_fets_patchtst"
                else {
                    "forecast": "forecast_power",
                    "gate": "correction_gate",
                    "persistence_candidate": "persistence_forecast_candidate",
                    "corrected_candidate": "corrected_forecast_candidate",
                }
                if model_id == "windprism_f7_g0"
                else {"forecast": "forecast_power"}
            ),
            "smoke": smoke,
            "training_code_path": str(Path(__file__).resolve()),
            "training_code_sha256": sha256_file(__file__),
        }
        if model_id == "itransformer":
            marker.update(
                {
                    "model_matrix_revision": MODEL_MATRIX_REVISION,
                    "architecture_source": (
                        "https://github.com/thuml/iTransformer"
                    ),
                    "architecture_adaptation": {
                        "implementation": "Keras",
                        "task": "4h ultra-short-term wind power forecasting",
                        "history_steps": HISTORY_LEN,
                        "forecast_steps": FORECAST_LEN,
                        "time_frequency_minutes": 15,
                        "variate_tokens": INPUT_DIM,
                        "target_variate_index": TARGET_INDEX,
                        "d_model": ITRANSFORMER_D_MODEL,
                        "num_heads": ITRANSFORMER_NUM_HEADS,
                        "encoder_layers": ITRANSFORMER_ENCODER_LAYERS,
                        "d_ff": ITRANSFORMER_D_FF,
                        "dropout": ITRANSFORMER_DROPOUT,
                        "activation": "gelu",
                        "instance_normalization": True,
                        "attention_axis": "variates",
                        "causal_attention_mask": False,
                        "positional_embedding": False,
                        "decoder": False,
                        "round3_common_training_protocol": True,
                    },
                }
            )
        elif model_id == "timesnet":
            marker.update(
                {
                    "model_matrix_revision": MODEL_MATRIX_REVISION,
                    "architecture_source": (
                        "https://github.com/thuml/Time-Series-Library"
                    ),
                    "architecture_adaptation": {
                        "implementation": "Keras",
                        "upstream_model": "TimesNet",
                        "task": "4h ultra-short-term wind power forecasting",
                        "history_steps": HISTORY_LEN,
                        "forecast_steps": FORECAST_LEN,
                        "time_frequency_minutes": 15,
                        "input_variates": INPUT_DIM,
                        "target_variate_index": TARGET_INDEX,
                        "d_model": TIMESNET_D_MODEL,
                        "d_ff": TIMESNET_D_FF,
                        "encoder_layers": TIMESNET_ENCODER_LAYERS,
                        "top_k_periods": TIMESNET_TOP_K,
                        "inception_kernels": [
                            2 * index + 1
                            for index in range(TIMESNET_NUM_KERNELS)
                        ],
                        "dropout": TIMESNET_DROPOUT,
                        "activation": "gelu",
                        "instance_normalization": True,
                        "circular_token_conv_kernel": 3,
                        "sinusoidal_position_embedding": True,
                        "temporal_alignment": (
                            f"{HISTORY_LEN}->{HISTORY_LEN + FORECAST_LEN}"
                        ),
                        "dynamic_batch_fft_period_discovery": True,
                        "fft_dc_exclusion": "negative_infinity_safe_equivalent",
                        "period_weighting": "sample_amplitude_softmax",
                        "two_dimensional_variation_convolution": True,
                        "all_variate_projection_before_target_selection": True,
                        "x_mark": None,
                        "x_mark_reason": (
                            "the fixed 45-channel schema already contains "
                            "causal calendar features"
                        ),
                        "future_truth_entering_latent_112_positions": False,
                        "causal_statement": (
                            "the final 16 latent positions are generated only "
                            "from the 96-step historical embedding"
                        ),
                        "official_architecture_recipe": (
                            "ETTm1 15-minute seq_len=96 d_model=64 d_ff=64"
                        ),
                        "round3_common_training_protocol": True,
                        "windprism_specific_modules_added": False,
                    },
                }
            )
        elif model_id == "timemixer":
            marker.update(
                {
                    "model_matrix_revision": MODEL_MATRIX_REVISION,
                    "architecture_source": (
                        "https://github.com/kwuking/TimeMixer"
                    ),
                    "architecture_paper": (
                        "https://arxiv.org/abs/2405.14616"
                    ),
                    "architecture_adaptation": {
                        "implementation": "Keras",
                        "upstream_model": "TimeMixer",
                        "upstream_variant": "original_TimeMixer_not_TimeMixer++",
                        "task": "4h ultra-short-term wind power forecasting",
                        "history_steps": HISTORY_LEN,
                        "forecast_steps": FORECAST_LEN,
                        "time_frequency_minutes": 15,
                        "input_variates": INPUT_DIM,
                        "target_variate_index": TARGET_INDEX,
                        "channel_independence": 1,
                        "d_model": TIMEMIXER_D_MODEL,
                        "d_ff": TIMEMIXER_D_FF,
                        "pdm_layers": TIMEMIXER_PDM_LAYERS,
                        "downsampling_method": "average_pooling",
                        "downsampling_layers": (
                            TIMEMIXER_DOWNSAMPLING_LAYERS
                        ),
                        "downsampling_window": (
                            TIMEMIXER_DOWNSAMPLING_WINDOW
                        ),
                        "scale_lengths": list(TIMEMIXER_SCALE_LENGTHS),
                        "decomposition_method": "moving_average",
                        "moving_average_kernel": (
                            TIMEMIXER_MOVING_AVERAGE
                        ),
                        "seasonal_mixing_direction": (
                            "bottom_up_fine_to_coarse"
                        ),
                        "trend_mixing_direction": (
                            "top_down_coarse_to_fine"
                        ),
                        "past_decomposable_mixing": True,
                        "future_multipredictor_mixing": (
                            "independent_temporal_predictor_per_scale_then_sum"
                        ),
                        "shared_channel_independent_projection": True,
                        "per_scale_affine_normalization": True,
                        "scale_zero_denormalization": True,
                        "data_embedding": (
                            "shared_circular_conv_without_position"
                        ),
                        "dropout": TIMEMIXER_DROPOUT,
                        "activation": "gelu",
                        "x_mark": None,
                        "future_temporal_features": False,
                        "x_mark_reason": (
                            "the fixed 45-channel schema already contains "
                            "causal calendar features and the unified benchmark "
                            "does not supply a future-covariate tensor"
                        ),
                        "all_variate_forecast_before_target_selection": True,
                        "future_truth_entering_model": False,
                        "causal_statement": (
                            "all four 16-step forecasts are generated only "
                            "from downsampled views of the 96-step history"
                        ),
                        "official_registered_unused_pdm_layernorm_retained": (
                            True
                        ),
                        "official_architecture_recipe": (
                            "ETTm1 15-minute seq_len=96 e_layers=2 "
                            "down_sampling_layers=3 window=2 avg "
                            "d_model=16 d_ff=32 channel_independence=1"
                        ),
                        "round3_common_training_protocol": True,
                        "windprism_specific_modules_added": False,
                    },
                }
            )
        elif model_id == "dlinear":
            marker.update(
                {
                    "model_matrix_revision": MODEL_MATRIX_REVISION,
                    "architecture_source": (
                        "https://github.com/honeywell21/DLinear"
                    ),
                    "architecture_paper": (
                        "https://arxiv.org/abs/2205.13504"
                    ),
                    "architecture_adaptation": {
                        "implementation": "Keras",
                        "upstream_model": "DLinear",
                        "task": "4h ultra-short-term wind power forecasting",
                        "history_steps": HISTORY_LEN,
                        "forecast_steps": FORECAST_LEN,
                        "time_frequency_minutes": 15,
                        "input_variates": INPUT_DIM,
                        "target_variate_index": TARGET_INDEX,
                        "features_mode": (
                            "official_MS_equivalent_all_variates_forecast_"
                            "then_target_selection"
                        ),
                        "direct_multi_step_forecast": True,
                        "decomposition": "moving_average_plus_remainder",
                        "moving_average_kernel": DLINEAR_MOVING_AVERAGE,
                        "moving_average_stride": 1,
                        "moving_average_boundary": (
                            "repeat_first_and_last_12_steps_then_valid_pool"
                        ),
                        "individual": DLINEAR_INDIVIDUAL,
                        "shared_temporal_heads_across_variates": True,
                        "seasonal_linear_mapping": (
                            f"{HISTORY_LEN}->{FORECAST_LEN}"
                        ),
                        "trend_linear_mapping": (
                            f"{HISTORY_LEN}->{FORECAST_LEN}"
                        ),
                        "branch_fusion": "elementwise_sum",
                        "activation": None,
                        "dropout": 0.0,
                        "normalization": None,
                        "attention": False,
                        "embedding": False,
                        "cross_variate_mixing": False,
                        "non_target_variates_influence_target": False,
                        "future_truth_entering_model": False,
                        "linear_initializer": (
                            "pytorch_nn_linear_default_equivalent_"
                            "uniform_plus_minus_1_over_sqrt_96"
                        ),
                        "official_commented_mean_initializer_enabled": False,
                        "x_target_to_y_target_scaler_bridge": (
                            "deterministic_non_trainable_affine"
                        ),
                        "round3_common_training_protocol": True,
                        "training_protocol_adaptations": {
                            "seed": RANDOM_SEED,
                            "batch_size": batch_size,
                            "optimizer": "Adam",
                            "learning_rate": LEARNING_RATE,
                            "loss": "Huber(delta=1.0)",
                            "maximum_epochs": epochs,
                            "early_stopping_patience": (
                                EARLY_STOPPING_PATIENCE
                            ),
                        },
                        "windprism_specific_modules_added": False,
                    },
                }
            )
        atomic_json(paths["marker"], marker)
        atomic_json(
            attempt_marker,
            {
                "status": "complete",
                "model_id": model_id,
                "farm_id": farm_id,
                "batch_size": batch_size,
                "task_marker": str(paths["marker"].resolve()),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        print(
            f"[complete] {model_id}/{farm_id}: params={parameter_count:,}, "
            f"batch={batch_size}, val_nrmse={validation['val_nrmse']:.6f}"
        )
        return 0
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        oom = is_confirmed_cuda_oom(exc, text, tf, model)
        atomic_json(
            attempt_marker,
            {
                "status": "oom" if oom else "failed",
                "created_at": utc_now(),
                "model_id": model_id,
                "farm_id": farm_id,
                "batch_size": batch_size,
                "exception_type": type(exc).__name__,
                "message": str(exc)[:4000],
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        print(text, file=sys.stderr)
        return OOM_EXIT_CODE if oom else 1
    finally:
        if keras is not None:
            try:
                keras.backend.clear_session()
            except Exception:
                pass
        del model
        gc.collect()


def worker_command(
    model_id: str,
    farm_id: str,
    batch_size: int,
    attempt_dir: Path,
    smoke: bool,
    extension_lineage: str,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model",
        model_id,
        "--farm",
        farm_id,
        "--batch-size",
        str(batch_size),
        "--attempt-dir",
        str(attempt_dir.resolve()),
        "--extension-lineage",
        extension_lineage,
    ]
    if smoke:
        command.append("--smoke")
    return command


def preflight_worker_command(
    model_id: str,
    farm_id: str,
    batch_size: int,
    attempt_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--preflight-worker",
        "--model",
        model_id,
        "--farm",
        farm_id,
        "--batch-size",
        str(batch_size),
        "--attempt-dir",
        str(attempt_dir.resolve()),
    ]


def launch_preflight_worker(
    model_id: str,
    farm_id: str,
    batch_size: int,
) -> tuple[int, Path, Path, str]:
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    attempt_dir = (
        RESULT_ROOT
        / "partial_runs"
        / "gpu_preflight"
        / model_id
        / farm_id
        / f"batch_{batch_size}"
        / f"attempt_{attempt_id}"
    )
    attempt_dir.mkdir(parents=True, exist_ok=True)
    marker_path = attempt_dir / "preflight_attempt.json"
    marker_path.unlink(missing_ok=True)
    log_path = attempt_dir / "worker.log"
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES", "0"),
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "TF_DETERMINISTIC_OPS": "1",
            "PYTHONHASHSEED": str(RANDOM_SEED),
        }
    )
    command = preflight_worker_command(
        model_id, farm_id, batch_size, attempt_dir
    )
    print(
        f"\n[preflight launch] {model_id}/{farm_id}, batch={batch_size}; "
        f"log={log_path}"
    )
    tail: list[str] = []
    with open(log_path, "w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(Path.cwd()),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            print(line, end="")
            tail.append(line)
            if len(tail) > 300:
                tail.pop(0)
        code = int(process.wait())
    return code, marker_path, log_path, "".join(tail)


def read_preflight_attempt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"预检worker未生成marker: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_global_preflight(force: bool = False) -> dict[str, Any]:
    """执行/复用三模型全局预检并冻结HR-MoE的统一batch。"""
    if force and load_formal_markers_fast():
        raise RuntimeError(
            "已有正式训练task marker，禁止重建全局batch策略；"
            "若确需重建，必须先人工归档并清理全部正式任务"
        )
    if BATCH_POLICY_PATH.exists() and not force:
        try:
            policy = load_batch_policy(require_valid_sources=True)
        except Exception as exc:
            raise RuntimeError(
                "已有batch策略身份无效；请审计后使用--force-preflight重建"
            ) from exc
        print(
            "[preflight resume] "
            f"farm={policy['preflight_farm_id']}, "
            f"HR batch={policy['hr_moe_effective_batch_size']}"
        )
        return policy
    if force:
        BATCH_POLICY_PATH.unlink(missing_ok=True)

    farm_id = largest_training_farm()
    paths = artifact_paths("patchtst", farm_id)
    attempts: list[dict[str, Any]] = []
    successful: dict[str, dict[str, Any]] = {}
    hr_effective_batch = DEFAULT_BATCH_SIZE
    hr_fallback = False

    for model_id in PREFLIGHT_MODELS:
        code, marker_path, log_path, tail = launch_preflight_worker(
            model_id, farm_id, DEFAULT_BATCH_SIZE
        )
        attempt = read_preflight_attempt(marker_path)
        attempt["log_path"] = str(log_path.resolve())
        attempts.append(attempt)
        if code == 0 and attempt.get("status") == "complete":
            successful[model_id] = attempt
            continue
        oom = (
            code == OOM_EXIT_CODE
            and bool(attempt.get("cuda_oom_confirmed"))
        )
        if model_id != HEAVY_FALLBACK_MODEL:
            reason = "CUDA OOM" if oom else "非CUDA OOM失败"
            raise RuntimeError(
                f"{model_id} batch=192 GPU预检{reason}；协议不允许自动降batch。"
                f"log={log_path}"
            )
        if not oom:
            raise RuntimeError(
                "HR-MoE预检失败但未确认CUDA OOM，不得自动降batch；"
                f"log={log_path}"
            )
        hr_fallback = True
        hr_effective_batch = OOM_FALLBACK_BATCH_SIZE
        code, marker_path, log_path, tail = launch_preflight_worker(
            model_id, farm_id, OOM_FALLBACK_BATCH_SIZE
        )
        fallback_attempt = read_preflight_attempt(marker_path)
        fallback_attempt["log_path"] = str(log_path.resolve())
        attempts.append(fallback_attempt)
        if code != 0 or fallback_attempt.get("status") != "complete":
            raise RuntimeError(
                "HR-MoE batch=128 GPU预检仍失败，停止正式训练；"
                f"log={log_path}; tail={tail[-2000:]}"
            )
        successful[model_id] = fallback_attempt

    if set(successful) != set(PREFLIGHT_MODELS):
        raise AssertionError("GPU预检未完整覆盖三个模型")
    summary = pd.DataFrame(attempts)
    atomic_csv(PREFLIGHT_SUMMARY_PATH, summary)
    policy = {
        "status": "complete",
        "created_at": utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "preflight_kind": "full_train_validation_single_epoch",
        "preflight_farm_id": farm_id,
        "preflight_models": list(PREFLIGHT_MODELS),
        "gpu_preflight_verified": True,
        "preflight_gpu_names_by_model": {
            model_id: payload.get("physical_gpu_names", [])
            for model_id, payload in successful.items()
        },
        "default_batch_size": DEFAULT_BATCH_SIZE,
        "hr_moe_effective_batch_size": hr_effective_batch,
        "hr_moe_global_fallback_triggered": hr_fallback,
        "hr_moe_batch_reason": (
            "global_preflight_cuda_oom_192_then_128_passed"
            if hr_fallback
            else "global_preflight_192_passed"
        ),
        "model_effective_batch_sizes": {
            model_id: int(payload["batch_size"])
            for model_id, payload in successful.items()
        },
        "preflight_attempts": [
            {
                "model_id": item.get("model_id"),
                "batch_size": item.get("batch_size"),
                "status": item.get("status"),
                "cuda_oom_confirmed": item.get("cuda_oom_confirmed", False),
                "fit_seconds": item.get("fit_seconds"),
                "elapsed_seconds": item.get("elapsed_seconds"),
                "peak_gpu_memory_bytes": item.get("peak_gpu_memory_bytes"),
                "log_path": item.get("log_path"),
            }
            for item in attempts
        ],
        "preflight_summary_path": str(PREFLIGHT_SUMMARY_PATH.resolve()),
        "preflight_summary_sha256": sha256_file(PREFLIGHT_SUMMARY_PATH),
        "array_sha256": sha256_file(paths["array"]),
        "preprocess_bundle_sha256": sha256_file(paths["bundle"]),
        "training_code_sha256": sha256_file(__file__),
    }
    atomic_json(BATCH_POLICY_PATH, policy)
    print(
        f"[preflight complete] 最大训练场站={farm_id}, "
        f"HR-MoE正式batch={hr_effective_batch}"
    )
    return policy


def launch_worker(
    model_id: str,
    farm_id: str,
    batch_size: int,
    smoke: bool,
    extension_lineage: str,
) -> tuple[int, Path, str]:
    paths = artifact_paths(model_id, farm_id, smoke=smoke)
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    attempt_dir = (
        paths["attempt_root"]
        / f"attempt_bs{batch_size}_{attempt_id}"
    )
    attempt_dir.mkdir(parents=True, exist_ok=True)
    log_path = attempt_dir / "worker.log"
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES", "0"),
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "TF_DETERMINISTIC_OPS": "1",
            "PYTHONHASHSEED": str(RANDOM_SEED),
        }
    )
    command = worker_command(
        model_id,
        farm_id,
        batch_size,
        attempt_dir,
        smoke,
        extension_lineage,
    )
    print(
        f"\n[launch] {model_id}/{farm_id}, batch={batch_size}; "
        f"log={log_path}"
    )
    tail: list[str] = []
    with open(log_path, "w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(Path.cwd()),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            print(line, end="")
            tail.append(line)
            if len(tail) > 300:
                tail.pop(0)
        code = int(process.wait())
    return code, log_path, "".join(tail)


def load_all_formal_markers() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        for farm_id in EXPECTED_FARMS:
            path = artifact_paths(model_id, farm_id)["marker"]
            if not completed_marker_valid(path):
                continue
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("model_id") != model_id or payload.get("farm_id") != farm_id:
                raise ValueError(f"训练marker身份不一致: {path}")
            rows.append(payload)
    return rows


def load_formal_markers_fast() -> dict[tuple[str, str], dict[str, Any]]:
    markers: dict[tuple[str, str], dict[str, Any]] = {}
    for model_id in MODEL_IDS:
        for farm_id in EXPECTED_FARMS:
            path = artifact_paths(model_id, farm_id)["marker"]
            if not path.is_file():
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                continue
            if (
                payload.get("status") == "complete"
                and payload.get("model_id") == model_id
                and payload.get("farm_id") == farm_id
            ):
                markers[(model_id, farm_id)] = payload
    return markers


def ordered_formal_tasks(
    models: Iterable[str],
    farms: Iterable[str],
    largest_farm: str,
) -> list[tuple[str, str]]:
    model_set, farm_set = set(models), set(farms)
    priority = [
        (model_id, largest_farm)
        for model_id in CALIBRATION_MODELS
        if model_id in model_set and largest_farm in farm_set
    ]
    stable = [
        (model_id, farm_id)
        for model_id in MODEL_IDS
        if model_id in model_set
        for farm_id in EXPECTED_FARMS
        if farm_id in farm_set
    ]
    return priority + [task for task in stable if task not in set(priority)]


def write_initial_resource_plan(
    policy: dict[str, Any],
    largest_farm: str,
) -> dict[str, Any]:
    central_seconds = 14.0 * sum(
        seconds
        * HISTORICAL_BATCH_SIZE
        / formal_batch_size(model_id, policy)
        for model_id, seconds in HISTORICAL_TASK_SECONDS.items()
    )
    payload = {
        "status": "initial",
        "created_at": utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "estimation_basis": (
            "predeclared 80-120 GPU-hour planning envelope; per-model "
            "historical timing priors are diagnostic only and are replaced "
            "by formal calibration tasks"
        ),
        "expected_task_count": len(MODEL_IDS) * len(EXPECTED_FARMS),
        "largest_training_farm": largest_farm,
        "calibration_task_order": [
            {"model_id": model_id, "farm_id": largest_farm}
            for model_id in CALIBRATION_MODELS
        ],
        "hr_moe_effective_batch_size": policy["hr_moe_effective_batch_size"],
        "estimated_training_gpu_hours_center": 100.0,
        "estimated_training_gpu_hours_lower": 80.0,
        "estimated_training_gpu_hours_upper": 120.0,
        "historical_prior_scaled_task_sum_gpu_hours": (
            central_seconds / 3600.0
        ),
        "historical_task_seconds_at_batch256": HISTORICAL_TASK_SECONDS,
        "resource_reservation_gpu_hours": [80.0, 130.0],
        "batch_policy_path": str(BATCH_POLICY_PATH.resolve()),
        "batch_policy_sha256": sha256_file(BATCH_POLICY_PATH),
        "preflight_elapsed_gpu_hours": float(
            sum(
                float(item.get("elapsed_seconds") or 0.0)
                for item in policy.get("preflight_attempts", ())
            )
            / 3600.0
        ),
    }
    atomic_json(RESOURCE_PLAN_INITIAL_PATH, payload)
    return payload


def update_runtime_progress(
    policy: dict[str, Any],
    largest_farm: str,
) -> dict[str, Any]:
    policy_sha = sha256_file(BATCH_POLICY_PATH)
    markers: dict[tuple[str, str], dict[str, Any]] = {}
    for (model_id, farm_id), marker in load_formal_markers_fast().items():
        marker_path = artifact_paths(model_id, farm_id)["marker"]
        if not completed_marker_valid(marker_path):
            continue
        if marker.get("global_batch_policy_sha256") != policy_sha:
            continue
        if int(marker.get("effective_batch_size", -1)) != formal_batch_size(
            model_id,
            policy,
        ):
            continue
        markers[(model_id, farm_id)] = marker
    rates: dict[str, list[float]] = {model_id: [] for model_id in MODEL_IDS}
    epoch_counts: dict[str, list[int]] = {model_id: [] for model_id in MODEL_IDS}
    overheads: dict[str, list[float]] = {model_id: [] for model_id in MODEL_IDS}
    for (model_id, _), marker in markers.items():
        batch = int(marker["effective_batch_size"])
        epochs = max(1, int(marker["epochs_ran"]))
        steps = int(np.ceil(int(marker["train_windows"]) / batch)) + int(
            np.ceil(int(marker["validation_windows"]) / batch)
        )
        fit_seconds = float(marker["fit_seconds"])
        rates[model_id].append(fit_seconds / max(1, epochs * steps))
        epoch_counts[model_id].append(epochs)
        overheads[model_id].append(
            max(0.0, float(marker["elapsed_seconds"]) - fit_seconds)
        )

    rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        for farm_id in EXPECTED_FARMS:
            marker = markers.get((model_id, farm_id))
            summary = preprocess_summary(farm_id)
            batch = formal_batch_size(model_id, policy)
            train_windows = int(summary["train_windows"])
            val_windows = int(summary["validation_windows"])
            steps = int(np.ceil(train_windows / batch)) + int(
                np.ceil(val_windows / batch)
            )
            if marker:
                center = float(marker["elapsed_seconds"])
                lower = upper = center
                status = "complete"
            elif rates[model_id]:
                rate = float(np.median(rates[model_id]))
                expected_epochs = float(np.median(epoch_counts[model_id]))
                overhead = float(np.median(overheads[model_id]))
                center = overhead + rate * steps * expected_epochs
                lower = overhead + rate * steps * max(1.0, expected_epochs * 0.75)
                upper = overhead + rate * steps * EPOCHS[model_id]
                status = "pending_calibrated"
            else:
                center = (
                    float(HISTORICAL_TASK_SECONDS[model_id])
                    * HISTORICAL_BATCH_SIZE
                    / batch
                )
                lower, upper = 0.75 * center, 1.75 * center
                status = "pending_historical"
            rows.append(
                {
                    "model_id": model_id,
                    "farm_id": farm_id,
                    "status": status,
                    "train_windows": train_windows,
                    "validation_windows": val_windows,
                    "effective_batch_size": batch,
                    "epochs_ran": marker.get("epochs_ran") if marker else None,
                    "observed_fit_seconds": (
                        marker.get("fit_seconds") if marker else None
                    ),
                    "observed_elapsed_seconds": (
                        marker.get("elapsed_seconds") if marker else None
                    ),
                    "estimated_seconds_lower": lower,
                    "estimated_seconds_center": center,
                    "estimated_seconds_upper": upper,
                }
            )
    frame = pd.DataFrame(rows)
    atomic_csv(RUNTIME_PROGRESS_PATH, frame)
    pending = frame[~frame["status"].eq("complete")]
    completed_frame = frame[frame["status"].eq("complete")]
    observed_seconds = float(
        completed_frame["observed_elapsed_seconds"].fillna(0.0).sum()
    )
    summary_payload = {
        "created_at": utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "completed_task_count": int(frame["status"].eq("complete").sum()),
        "verified_task_count": len(markers),
        "remaining_task_count": int(len(pending)),
        "remaining_gpu_hours_lower": float(
            pending["estimated_seconds_lower"].sum() / 3600.0
        ),
        "remaining_gpu_hours_center": float(
            pending["estimated_seconds_center"].sum() / 3600.0
        ),
        "remaining_gpu_hours_upper": float(
            pending["estimated_seconds_upper"].sum() / 3600.0
        ),
        "observed_training_gpu_hours": observed_seconds / 3600.0,
        "estimated_total_gpu_hours_center": float(
            (
                observed_seconds
                + pending["estimated_seconds_center"].sum()
            )
            / 3600.0
        ),
        "runtime_progress_path": str(RUNTIME_PROGRESS_PATH.resolve()),
        "runtime_progress_sha256": sha256_file(RUNTIME_PROGRESS_PATH),
    }
    calibration_tasks = {
        (model_id, largest_farm) for model_id in CALIBRATION_MODELS
    }
    if calibration_tasks.issubset(markers):
        summary_payload.update(
            {
                "status": "calibrated",
                "largest_training_farm": largest_farm,
                "calibration_tasks": [
                    {
                        "model_id": model_id,
                        "farm_id": largest_farm,
                        "fit_seconds": markers[(model_id, largest_farm)][
                            "fit_seconds"
                        ],
                        "epochs_ran": markers[(model_id, largest_farm)][
                            "epochs_ran"
                        ],
                    }
                    for model_id in CALIBRATION_MODELS
                ],
                "batch_policy_sha256": sha256_file(BATCH_POLICY_PATH),
            }
        )
        atomic_json(RESOURCE_PLAN_CALIBRATED_PATH, summary_payload)
    return summary_payload


def write_summaries(
    markers: list[dict[str, Any]],
    report_root: Path,
) -> dict[str, str]:
    summary_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    overfit_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    for marker in markers:
        common = {
            "model_id": marker["model_id"],
            "farm_id": marker["farm_id"],
            "status": marker["status"],
        }
        validation = marker.get("validation_metrics", {})
        overfit = marker.get("overfit_diagnostics", {})
        summary_rows.append(
            {
                **common,
                "train_windows": marker.get("train_windows"),
                "validation_windows": marker.get("validation_windows"),
                "training_feasibility": marker.get("training_feasibility"),
                "requested_batch_size": marker.get("requested_batch_size"),
                "effective_batch_size": marker.get("effective_batch_size"),
                "fallback_triggered": marker.get("fallback_triggered"),
                "epochs_ran": marker.get("epochs_ran"),
                "best_epoch": marker.get("best_epoch"),
                "fit_seconds": marker.get("fit_seconds"),
                "parameter_count": marker.get("parameter_count"),
                "val_nmae": validation.get("val_nmae"),
                "val_nrmse": validation.get("val_nrmse"),
                "val_r2": validation.get("val_r2"),
                "model_path": marker.get("model_path"),
                "weights_path": marker.get("weights_path"),
            }
        )
        validation_rows.append({**common, **validation})
        complexity_rows.append(
            {
                **common,
                "parameter_count": marker.get("parameter_count"),
                "trainable_parameter_count": marker.get("trainable_parameter_count"),
                "model_size_bytes": marker.get("model_size_bytes"),
                "fit_seconds": marker.get("fit_seconds"),
                "elapsed_seconds": marker.get("elapsed_seconds"),
                "peak_gpu_memory_bytes": marker.get("peak_gpu_memory_bytes"),
                "epochs_ran": marker.get("epochs_ran"),
                "seconds_per_epoch": (
                    float(marker["fit_seconds"]) / max(1, int(marker["epochs_ran"]))
                    if marker.get("fit_seconds") is not None
                    else None
                ),
            }
        )
        overfit_rows.append({**common, **overfit})
        fallback_rows.append(
            {
                **common,
                "requested_batch_size": marker.get("requested_batch_size"),
                "effective_batch_size": marker.get("effective_batch_size"),
                "fallback_triggered": marker.get("fallback_triggered"),
                "global_policy_fallback_active": marker.get(
                    "global_policy_fallback_active"
                ),
                "task_retry_after_oom": marker.get("task_retry_after_oom"),
                "attempted_batch_sizes": json.dumps(
                    marker.get("attempted_batch_sizes", []), ensure_ascii=False
                ),
                "fallback_reason": marker.get(
                    "global_batch_policy_reason",
                    marker.get("fallback_reason"),
                ),
                "oom_exception_type": marker.get("oom_exception_type"),
                "oom_message": marker.get("oom_message"),
            }
        )
    outputs = {
        "summary": report_root / "round3_external14_training_summary.csv",
        "validation": report_root
        / "validation_metrics"
        / "round3_external14_validation_metrics.csv",
        "complexity": report_root
        / "complexity"
        / "round3_external14_training_complexity_runtime.csv",
        "overfit": report_root
        / "validation_metrics"
        / "round3_overfit_diagnostics.csv",
        "fallback": report_root
        / "complexity"
        / "round3_batch_fallback_summary.csv",
    }
    atomic_csv(outputs["summary"], pd.DataFrame(summary_rows))
    atomic_csv(outputs["validation"], pd.DataFrame(validation_rows))
    atomic_csv(outputs["complexity"], pd.DataFrame(complexity_rows))
    atomic_csv(outputs["overfit"], pd.DataFrame(overfit_rows))
    atomic_csv(outputs["fallback"], pd.DataFrame(fallback_rows))
    return {name: str(path.resolve()) for name, path in outputs.items()}


def finalize_bundle(
    extension_lineage: str | None = None,
) -> dict[str, Any]:
    markers = load_all_formal_markers()
    policy = load_batch_policy(require_valid_sources=True)
    policy_sha = sha256_file(BATCH_POLICY_PATH)
    current_training_code_sha = sha256_file(__file__)
    extension_lineage = extension_lineage or infer_extension_lineage()
    if extension_lineage not in {
        STAGED_EXTENSION_LINEAGE,
        UNIFIED_MODERN_EXTENSION_LINEAGE,
    }:
        raise ValueError(f"未知训练extension lineage: {extension_lineage}")

    frozen_pre_dlinear_code_hashes: dict[str, str] = {}
    if extension_lineage == STAGED_EXTENSION_LINEAGE:
        if PRE_DLINEAR_TRAINING_ARCHIVE_MANIFEST_PATH.is_file():
            with open(
                PRE_DLINEAR_TRAINING_ARCHIVE_MANIFEST_PATH,
                "r",
                encoding="utf-8",
            ) as handle:
                pre_dlinear_archive = json.load(handle)
            frozen_pre_dlinear_code_hashes = {
                str(model_id): str(code_sha)
                for model_id, code_sha in pre_dlinear_archive.get(
                    "frozen_training_code_sha256_by_model", {}
                ).items()
            }
            if (
                pre_dlinear_archive.get("status") != "complete"
                or pre_dlinear_archive.get(
                    "model_matrix_revision_at_archive"
                )
                != PRE_DLINEAR_MODEL_MATRIX_REVISION
                or tuple(pre_dlinear_archive.get("expected_models", ()))
                != PRE_DLINEAR_MODEL_IDS
                or set(frozen_pre_dlinear_code_hashes)
                != set(PRE_DLINEAR_MODEL_IDS)
            ):
                raise ValueError("pre-DLinear训练归档身份或代码SHA集合漂移")
        else:
            # Partial diagnostics before staged DLinear starts.
            frozen_pre_dlinear_code_hashes = {
                model_id: BASE10_TRAINING_CODE_SHA256
                for model_id in LEGACY_MODEL_IDS
            }
            frozen_pre_dlinear_code_hashes["itransformer"] = (
                ITRANSFORMER_EXTENSION_TRAINING_CODE_SHA256
            )
            if PRE_TIMEMIXER_TRAINING_ARCHIVE_MANIFEST_PATH.is_file():
                with open(
                    PRE_TIMEMIXER_TRAINING_ARCHIVE_MANIFEST_PATH,
                    "r",
                    encoding="utf-8",
                ) as handle:
                    pre_timemixer_archive = json.load(handle)
                frozen_pre_dlinear_code_hashes.update(
                    {
                        str(model_id): str(code_sha)
                        for model_id, code_sha in pre_timemixer_archive.get(
                            "frozen_training_code_sha256_by_model", {}
                        ).items()
                    }
                )
            frozen_pre_dlinear_code_hashes["timemixer"] = (
                TIMEMIXER_EXTENSION_TRAINING_CODE_SHA256
            )
    else:
        frozen_pre_dlinear_code_hashes = {
            str(model_id): str(code_sha)
            for model_id, code_sha in {
                model_id: BASE10_TRAINING_CODE_SHA256
                for model_id in LEGACY_MODEL_IDS
            }.items()
        }
    batches_by_model: dict[str, set[int]] = {
        model_id: set() for model_id in MODEL_IDS
    }
    staged_dlinear_code_sha = current_training_code_sha
    if extension_lineage == STAGED_EXTENSION_LINEAGE:
        dlinear_markers = [
            marker for marker in markers if marker["model_id"] == "dlinear"
        ]
        observed_dlinear_hashes = {
            str(marker.get("training_code_sha256", ""))
            for marker in dlinear_markers
        }
        if (
            len(dlinear_markers) == len(EXPECTED_FARMS)
            and len(observed_dlinear_hashes) == 1
            and "" not in observed_dlinear_hashes
        ):
            # Preserve a historically completed staged generation even after
            # later orchestration-only edits changed this source file SHA.
            staged_dlinear_code_sha = next(iter(observed_dlinear_hashes))
        elif observed_dlinear_hashes and observed_dlinear_hashes != {
            current_training_code_sha
        }:
            raise ValueError(
                "staged DLinear仅部分完成且训练代码SHA已变化，禁止混合代际"
            )
    for marker in markers:
        model_id = str(marker["model_id"])
        batches_by_model[model_id].add(int(marker["effective_batch_size"]))
        if marker.get("global_batch_policy_sha256") != policy_sha:
            raise ValueError(
                f"{model_id}/{marker['farm_id']}使用的全局batch策略SHA已漂移"
            )
        if extension_lineage == UNIFIED_MODERN_EXTENSION_LINEAGE:
            expected_code_sha = (
                current_training_code_sha
                if model_id in MODERN_TRAINABLE_MODEL_IDS
                else BASE10_TRAINING_CODE_SHA256
            )
            if (
                model_id in MODERN_TRAINABLE_MODEL_IDS
                and marker.get("extension_lineage")
                != UNIFIED_MODERN_EXTENSION_LINEAGE
            ):
                raise ValueError(
                    f"{model_id}/{marker['farm_id']}缺少unified lineage身份"
                )
        else:
            expected_code_sha = (
                staged_dlinear_code_sha
                if model_id in DLINEAR_BASELINE_IDS
                else frozen_pre_dlinear_code_hashes.get(model_id)
            )
            if (
                model_id in DLINEAR_BASELINE_IDS
                and marker.get("extension_lineage")
                not in (None, STAGED_EXTENSION_LINEAGE)
            ):
                raise ValueError(
                    f"{model_id}/{marker['farm_id']}staged lineage身份漂移"
                )
        if marker.get("training_code_sha256") != expected_code_sha:
            raise ValueError(
                f"{model_id}/{marker['farm_id']}训练代码SHA不符合追加矩阵身份"
            )
    mixed = {
        model_id: sorted(values)
        for model_id, values in batches_by_model.items()
        if len(values) > 1
    }
    if mixed:
        raise ValueError(f"同一模型跨场站混用batch: {mixed}")
    expected_hr_batch = int(policy["hr_moe_effective_batch_size"])
    observed_hr = batches_by_model[HEAVY_FALLBACK_MODEL]
    if observed_hr and observed_hr != {expected_hr_batch}:
        raise ValueError(
            f"HR-MoE正式batch{sorted(observed_hr)}与全局策略"
            f"{expected_hr_batch}不一致"
        )
    for model_id in MODEL_IDS:
        if model_id != HEAVY_FALLBACK_MODEL and batches_by_model[model_id]:
            if batches_by_model[model_id] != {DEFAULT_BATCH_SIZE}:
                raise ValueError(f"{model_id}正式batch不是统一192")
    identities = {(item["model_id"], item["farm_id"]) for item in markers}
    expected = {(model, farm) for model in MODEL_IDS for farm in EXPECTED_FARMS}
    required_archive_exists = (
        PRE_DLINEAR_TRAINING_ARCHIVE_MANIFEST_PATH.is_file()
        if extension_lineage == STAGED_EXTENSION_LINEAGE
        else BASE10_TRAINING_ARCHIVE_MANIFEST_PATH.is_file()
    )
    complete = (
        identities == expected
        and len(markers) == len(expected)
        and required_archive_exists
    )
    report_root = (
        RESULT_ROOT
        if complete
        else RESULT_ROOT / "partial_runs" / "training_summary"
    )
    outputs = write_summaries(markers, report_root)
    base10_archive_record = (
        {
            "path": str(BASE10_TRAINING_ARCHIVE_MANIFEST_PATH.resolve()),
            "sha256": sha256_file(BASE10_TRAINING_ARCHIVE_MANIFEST_PATH),
            "size_bytes": BASE10_TRAINING_ARCHIVE_MANIFEST_PATH.stat().st_size,
        }
        if BASE10_TRAINING_ARCHIVE_MANIFEST_PATH.is_file()
        else None
    )
    pre_timesnet_archive_record = (
        {
            "path": str(
                PRE_TIMESNET_TRAINING_ARCHIVE_MANIFEST_PATH.resolve()
            ),
            "sha256": sha256_file(
                PRE_TIMESNET_TRAINING_ARCHIVE_MANIFEST_PATH
            ),
            "size_bytes": (
                PRE_TIMESNET_TRAINING_ARCHIVE_MANIFEST_PATH.stat().st_size
            ),
        }
        if PRE_TIMESNET_TRAINING_ARCHIVE_MANIFEST_PATH.is_file()
        else None
    )
    pre_timemixer_archive_record = (
        {
            "path": str(
                PRE_TIMEMIXER_TRAINING_ARCHIVE_MANIFEST_PATH.resolve()
            ),
            "sha256": sha256_file(
                PRE_TIMEMIXER_TRAINING_ARCHIVE_MANIFEST_PATH
            ),
            "size_bytes": (
                PRE_TIMEMIXER_TRAINING_ARCHIVE_MANIFEST_PATH.stat().st_size
            ),
        }
        if PRE_TIMEMIXER_TRAINING_ARCHIVE_MANIFEST_PATH.is_file()
        else None
    )
    pre_dlinear_archive_record = (
        {
            "path": str(
                PRE_DLINEAR_TRAINING_ARCHIVE_MANIFEST_PATH.resolve()
            ),
            "sha256": sha256_file(
                PRE_DLINEAR_TRAINING_ARCHIVE_MANIFEST_PATH
            ),
            "size_bytes": (
                PRE_DLINEAR_TRAINING_ARCHIVE_MANIFEST_PATH.stat().st_size
            ),
        }
        if PRE_DLINEAR_TRAINING_ARCHIVE_MANIFEST_PATH.is_file()
        else None
    )
    payload = {
        "status": "complete" if complete else "partial",
        "created_at": utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "model_matrix_revision": MODEL_MATRIX_REVISION,
        "extension_lineage": extension_lineage,
        "expected_models": list(MODEL_IDS),
        "expected_farms": list(EXPECTED_FARMS),
        "expected_task_count": len(expected),
        "completed_task_count": len(markers),
        "completed_tasks": [
            {"model_id": item["model_id"], "farm_id": item["farm_id"]}
            for item in markers
        ],
        "summary_outputs": {
            name: {
                "path": path,
                "sha256": sha256_file(path),
                "size_bytes": Path(path).stat().st_size,
            }
            for name, path in outputs.items()
        },
        "task_marker_records": [
            {
                "model_id": item["model_id"],
                "farm_id": item["farm_id"],
                "path": str(
                    artifact_paths(
                        item["model_id"],
                        item["farm_id"],
                    )["marker"].resolve()
                ),
                "sha256": sha256_file(
                    artifact_paths(
                        item["model_id"],
                        item["farm_id"],
                    )["marker"]
                ),
            }
            for item in markers
        ],
        "batch_fallback_count": int(
            bool(policy["hr_moe_global_fallback_triggered"])
        ),
        "fallback_event_count": int(
            bool(policy["hr_moe_global_fallback_triggered"])
        ),
        "tasks_using_fallback_batch": int(
            sum(
                item["model_id"] == HEAVY_FALLBACK_MODEL
                and int(item["effective_batch_size"])
                == OOM_FALLBACK_BATCH_SIZE
                for item in markers
            )
        ),
        "global_batch_policy_path": str(BATCH_POLICY_PATH.resolve()),
        "global_batch_policy_sha256": policy_sha,
        "hr_moe_effective_batch_size": expected_hr_batch,
        "hr_moe_global_fallback_triggered": bool(
            policy["hr_moe_global_fallback_triggered"]
        ),
        "effective_batches_by_model": {
            model_id: sorted(values)
            for model_id, values in batches_by_model.items()
        },
        "resource_plan_initial_path": str(RESOURCE_PLAN_INITIAL_PATH.resolve()),
        "resource_plan_initial_sha256": (
            sha256_file(RESOURCE_PLAN_INITIAL_PATH)
            if RESOURCE_PLAN_INITIAL_PATH.is_file()
            else None
        ),
        "resource_plan_calibrated_path": (
            str(RESOURCE_PLAN_CALIBRATED_PATH.resolve())
            if RESOURCE_PLAN_CALIBRATED_PATH.is_file()
            else None
        ),
        "resource_plan_calibrated_sha256": (
            sha256_file(RESOURCE_PLAN_CALIBRATED_PATH)
            if RESOURCE_PLAN_CALIBRATED_PATH.is_file()
            else None
        ),
        "runtime_progress_path": (
            str(RUNTIME_PROGRESS_PATH.resolve())
            if RUNTIME_PROGRESS_PATH.is_file()
            else None
        ),
        "runtime_progress_sha256": (
            sha256_file(RUNTIME_PROGRESS_PATH)
            if RUNTIME_PROGRESS_PATH.is_file()
            else None
        ),
        "all_default_batch_192": bool(
            markers
            and all(int(item.get("effective_batch_size", -1)) == 192 for item in markers)
        ),
        "additive_baseline_extension": True,
        "base10_reused_task_count": int(
            sum(item["model_id"] in LEGACY_MODEL_IDS for item in markers)
        ),
        "itransformer_reused_task_count": int(
            sum(item["model_id"] == "itransformer" for item in markers)
            if extension_lineage == STAGED_EXTENSION_LINEAGE
            else 0
        ),
        "new_itransformer_task_count": int(
            sum(item["model_id"] == "itransformer" for item in markers)
            if extension_lineage == UNIFIED_MODERN_EXTENSION_LINEAGE
            else 0
        ),
        "pre_timesnet_reused_task_count": int(
            sum(
                item["model_id"] in PRE_TIMESNET_MODEL_IDS
                for item in markers
            )
            if extension_lineage == STAGED_EXTENSION_LINEAGE
            else sum(item["model_id"] in LEGACY_MODEL_IDS for item in markers)
        ),
        "new_timesnet_task_count": int(
            sum(item["model_id"] == "timesnet" for item in markers)
        ),
        "pre_timemixer_reused_task_count": int(
            sum(
                item["model_id"] in PRE_TIMEMIXER_MODEL_IDS
                for item in markers
            )
            if extension_lineage == STAGED_EXTENSION_LINEAGE
            else sum(item["model_id"] in LEGACY_MODEL_IDS for item in markers)
        ),
        "timesnet_reused_task_count": int(
            sum(item["model_id"] == "timesnet" for item in markers)
            if extension_lineage == STAGED_EXTENSION_LINEAGE
            else 0
        ),
        "new_timemixer_task_count": int(
            sum(item["model_id"] == "timemixer" for item in markers)
        ),
        "pre_dlinear_reused_task_count": int(
            sum(
                item["model_id"] in PRE_DLINEAR_MODEL_IDS
                for item in markers
            )
            if extension_lineage == STAGED_EXTENSION_LINEAGE
            else sum(item["model_id"] in LEGACY_MODEL_IDS for item in markers)
        ),
        "timemixer_reused_task_count": int(
            sum(item["model_id"] == "timemixer" for item in markers)
            if extension_lineage == STAGED_EXTENSION_LINEAGE
            else 0
        ),
        "new_dlinear_task_count": int(
            sum(item["model_id"] == "dlinear" for item in markers)
        ),
        "base10_training_complete_archive": base10_archive_record,
        "pre_timesnet_training_complete_archive": (
            pre_timesnet_archive_record
        ),
        "pre_timemixer_training_complete_archive": (
            pre_timemixer_archive_record
        ),
        "pre_dlinear_training_complete_archive": (
            pre_dlinear_archive_record
        ),
        "legacy_training_code_sha256": BASE10_TRAINING_CODE_SHA256,
        "modern_extension_training_code_sha256": (
            current_training_code_sha
            if extension_lineage == UNIFIED_MODERN_EXTENSION_LINEAGE
            else None
        ),
        "frozen_pre_timesnet_training_code_sha256_by_model": (
            {
                model_id: frozen_pre_dlinear_code_hashes.get(model_id)
                for model_id in PRE_TIMESNET_MODEL_IDS
            }
        ),
        "timesnet_extension_training_code_sha256": (
            frozen_pre_dlinear_code_hashes.get("timesnet")
        ),
        "frozen_pre_timemixer_training_code_sha256_by_model": (
            {
                model_id: frozen_pre_dlinear_code_hashes.get(model_id)
                for model_id in PRE_TIMEMIXER_MODEL_IDS
            }
        ),
        "timemixer_extension_training_code_sha256": (
            frozen_pre_dlinear_code_hashes.get("timemixer")
        ),
        "frozen_pre_dlinear_training_code_sha256_by_model": (
            frozen_pre_dlinear_code_hashes
        ),
        "dlinear_extension_training_code_sha256": (
            staged_dlinear_code_sha
            if extension_lineage == STAGED_EXTENSION_LINEAGE
            else current_training_code_sha
        ),
        "legacy_model_artifacts_modified_by_extension": False,
        "pre_timemixer_model_artifacts_modified_by_extension": False,
        "pre_dlinear_model_artifacts_modified_by_extension": False,
    }
    if complete:
        marker_path = RESULT_ROOT / "round3_training_bundle_complete.json"
        (RESULT_ROOT / "partial_runs" / "round3_training_partial.json").unlink(
            missing_ok=True
        )
    else:
        marker_path = RESULT_ROOT / "partial_runs" / "round3_training_partial.json"
    atomic_json(marker_path, payload)
    return payload


def run_parent(args: argparse.Namespace) -> int:
    ensure_preprocess_complete()
    models = parse_selection(args.models, MODEL_IDS, "模型")
    farms = parse_selection(args.farms, EXPECTED_FARMS, "场站")
    if args.smoke and args.preflight_only:
        raise ValueError("--smoke与--preflight-only不能同时使用")
    if args.smoke and args.farms.strip().lower() == "all":
        farms = [EXPECTED_FARMS[0]]
        print(f"[smoke] 未显式指定场站，仅使用{farms[0]}验证全部所选模型")
    if args.force and args.resume:
        raise ValueError("--force与--resume不能同时使用")
    extension_lineage = (
        "smoke" if args.smoke else infer_extension_lineage()
    )
    selected_modern = set(models).intersection(MODERN_TRAINABLE_MODEL_IDS)
    if not args.smoke and not selected_modern and not args.preflight_only:
        raise ValueError(
            "正式扩展必须至少选择一个现代基线；原10模型已冻结"
        )
    if (
        not args.smoke
        and extension_lineage == STAGED_EXTENSION_LINEAGE
        and "dlinear" not in selected_modern
        and not args.preflight_only
    ):
        raise ValueError(
            "historical staged lineage当前仅允许补齐DLinear；"
            "此前13模型已按代际冻结"
        )
    frozen_model_ids = (
        PRE_DLINEAR_MODEL_IDS
        if extension_lineage == STAGED_EXTENSION_LINEAGE
        else LEGACY_MODEL_IDS
    )
    if (
        not args.smoke
        and args.force
        and set(models).intersection(frozen_model_ids)
    ):
        raise ValueError(
            f"{extension_lineage}禁止--force覆盖冻结模型；"
            "仅显式选择需要重训的新增模型"
        )
    if not args.smoke and args.force_preflight:
        raise ValueError(
            "现代基线追加必须复用原10模型绑定的全局batch policy；"
            "禁止--force-preflight重建策略"
        )
    if not args.smoke:
        if extension_lineage == STAGED_EXTENSION_LINEAGE:
            archive_record = archive_pre_dlinear_training_complete()
            archive_label = "pre-DLinear 13-model"
        else:
            archive_record = archive_base10_training_complete()
            archive_label = "base10 unified-modern"
        print(
            f"[{extension_lineage}] 已冻结训练bundle ({archive_label}): "
            f"{archive_record['path']}"
        )
        selected_complete = all(
            completed_marker_valid(artifact_paths(model_id, farm_id)["marker"])
            for model_id in models
            for farm_id in farms
        )
        if args.resume and selected_complete and extended_training_bundle_valid():
            print(
                f"[{extension_lineage} resume] 14×14训练bundle及所选任务"
                "均通过哈希校验，无需重写任何训练产物"
            )
            return 0
    policy = None
    largest_farm = largest_training_farm()
    if not args.smoke:
        policy = ensure_global_preflight(force=bool(args.force_preflight))
        if args.preflight_only:
            write_initial_resource_plan(policy, largest_farm)
            update_runtime_progress(policy, largest_farm)
            print(
                f"[preflight-only complete] farm={largest_farm}, "
                f"HR batch={policy['hr_moe_effective_batch_size']}"
            )
            return 0
        # 保留原140项或182项complete作为可恢复证据，直到196项全部
        # 核验完成后才原子覆盖；精确副本已由对应lineage归档。
        if args.force:
            # 撤销可能存在的196项complete，避免强制重训失败时留下
            # 伪完成状态；冻结的上一代complete仍保留在归档中。
            (
                RESULT_ROOT / "round3_training_bundle_complete.json"
            ).unlink(missing_ok=True)
            for model_id, farm_id in ordered_formal_tasks(
                models, farms, largest_farm
            ):
                artifact_paths(model_id, farm_id)["marker"].unlink(
                    missing_ok=True
                )
        RESOURCE_PLAN_CALIBRATED_PATH.unlink(missing_ok=True)
        write_initial_resource_plan(policy, largest_farm)
        update_runtime_progress(policy, largest_farm)
    elif args.force_preflight:
        raise ValueError("--force-preflight不能与--smoke同时使用")

    completed = 0
    skipped = 0
    tasks = ordered_formal_tasks(models, farms, largest_farm)
    for model_id, farm_id in tasks:
        paths = artifact_paths(model_id, farm_id, smoke=args.smoke)
        if not args.smoke and model_id in frozen_model_ids:
            if not completed_marker_valid(paths["marker"]):
                raise ValueError(
                    f"冻结的{extension_lineage}任务缺失或漂移，禁止重算: "
                    f"{model_id}/{farm_id}"
                )
            with open(paths["marker"], "r", encoding="utf-8") as handle:
                frozen_marker = json.load(handle)
            if (
                str(frozen_marker.get("model_id")) != model_id
                or str(frozen_marker.get("farm_id")) != farm_id
            ):
                raise ValueError(
                    f"冻结任务marker身份漂移: {model_id}/{farm_id}"
                )
            if (
                extension_lineage == UNIFIED_MODERN_EXTENSION_LINEAGE
                and frozen_marker.get("training_code_sha256")
                != BASE10_TRAINING_CODE_SHA256
            ):
                raise ValueError(
                    f"base10冻结任务训练代码SHA漂移: {model_id}/{farm_id}"
                )
            if policy is not None:
                validate_task_policy(paths["marker"], model_id, policy)
            print(f"[frozen reuse] {model_id}/{farm_id}已完成，跳过")
            skipped += 1
            continue
        if completed_marker_valid(paths["marker"]):
            if args.resume:
                if policy is not None:
                    validate_task_policy(paths["marker"], model_id, policy)
                with open(paths["marker"], "r", encoding="utf-8") as handle:
                    completed_marker = json.load(handle)
                if (
                    str(completed_marker.get("model_id")) != model_id
                    or str(completed_marker.get("farm_id")) != farm_id
                ):
                    raise ValueError(
                        f"resume任务marker身份漂移: {model_id}/{farm_id}"
                    )
                if model_id in MODERN_TRAINABLE_MODEL_IDS:
                    if (
                        completed_marker.get("extension_lineage")
                        != extension_lineage
                    ):
                        raise ValueError(
                            f"resume任务lineage漂移: {model_id}/{farm_id}"
                        )
                    if (
                        extension_lineage
                        == UNIFIED_MODERN_EXTENSION_LINEAGE
                        and completed_marker.get("training_code_sha256")
                        != sha256_file(__file__)
                    ):
                        raise ValueError(
                            f"unified resume训练代码SHA漂移: "
                            f"{model_id}/{farm_id}"
                        )
                print(f"[resume] {model_id}/{farm_id}已完成，跳过")
                skipped += 1
                continue
            if not args.force:
                raise FileExistsError(
                    f"{model_id}/{farm_id}已有完整产物；使用--resume或--force"
                )
        batch_size = (
            DEFAULT_BATCH_SIZE
            if args.smoke
            else formal_batch_size(model_id, policy)
        )
        code, worker_log, tail = launch_worker(
            model_id,
            farm_id,
            batch_size,
            smoke=args.smoke,
            extension_lineage=extension_lineage,
        )
        if code != 0:
            oom = code == OOM_EXIT_CODE
            if oom:
                raise RuntimeError(
                    f"{model_id}/{farm_id}按全局policy batch={batch_size}"
                    "仍发生CUDA OOM；禁止逐场站降batch，以免协议混用。"
                    f"log={worker_log}"
                )
            raise RuntimeError(
                f"{model_id}/{farm_id}非CUDA OOM训练失败；"
                f"code={code}, log={worker_log}, tail={tail[-2000:]}"
            )
        if not completed_marker_valid(paths["marker"]):
            raise RuntimeError(f"worker成功退出但任务marker无效: {paths['marker']}")
        completed += 1
        if policy is not None:
            update_runtime_progress(policy, largest_farm)

    if args.smoke:
        payload = {
            "status": "complete",
            "created_at": utc_now(),
            "protocol_version": PROTOCOL_VERSION,
            "models": models,
            "farms": farms,
            "completed": completed,
            "skipped": skipped,
            "formal_artifacts_untouched": True,
        }
        atomic_json(
            RESULT_ROOT / "partial_runs" / "smoke" / "round3_smoke_complete.json",
            payload,
        )
        print(f"[smoke complete] {completed} tasks, {skipped} skipped")
        return 0

    runtime = update_runtime_progress(policy, largest_farm)
    bundle = finalize_bundle(extension_lineage)
    print(
        f"[training {bundle['status']}] 本轮完成={completed}, 跳过={skipped}, "
        f"全局进度={bundle['completed_task_count']}/{bundle['expected_task_count']}, "
        f"HR全局batch={bundle['hr_moe_effective_batch_size']}, "
        f"剩余ETA中心={runtime['remaining_gpu_hours_center']:.2f} GPU小时"
    )
    return 0


def main() -> None:
    args = build_parser().parse_args()
    if args.preflight_worker:
        raise SystemExit(run_preflight_worker(args))
    if args.worker:
        raise SystemExit(run_worker(args))
    try:
        code = run_parent(args)
    except Exception:
        if not args.smoke:
            try:
                finalize_bundle()
            except Exception as finalize_error:
                print(
                    f"[warning] 失败后的partial marker生成失败: {finalize_error}",
                    file=sys.stderr,
                )
        raise
    raise SystemExit(code)


if __name__ == "__main__":
    main()
