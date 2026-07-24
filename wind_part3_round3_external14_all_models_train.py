"""Part 3 Round 3：JSFD001--JSFD014 十模型统一无泄漏训练。

父调度器本身不导入 TensorFlow。每个 ``model_id × farm_id`` 任务均在全新
子进程中构建模型、训练并退出，以可靠释放 GPU 上下文。数据只来自 Round 3
预处理生成的 NPZ/bundle；本文件不会调用旧工程的读 CSV、插值、缩放或切窗
函数。正式训练前，三个重模型会在最大训练场站各用独立进程完成一个完整
train+validation预检epoch；只有HR-MoE预检确认CUDA OOM时才锁定全局
batch=128，随后14个HR场站全部使用同一batch。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import subprocess
import sys
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
OTHER_MODELS = frozenset(MODEL_IDS[1:8])
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
EPOCHS = {
    **{name: 60 for name in OTHER_MODELS},
    "patchtst": 80,
    "hr_moe_fets_patchtst": 80,
    "windprism_f7_g0": 80,
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
}
_MISSING_SAFE_REGIME_LAYER_CLASS: Any | None = None
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
}


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
        current = {
            "array_sha256": sha256_file(paths["array"]),
            "preprocess_bundle_sha256": sha256_file(paths["bundle"]),
            "training_code_sha256": sha256_file(__file__),
        }
        for key, value in current.items():
            if policy.get(key) != value:
                raise ValueError(f"全局batch策略源身份漂移: {key}")
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
    command = worker_command(model_id, farm_id, batch_size, attempt_dir, smoke)
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


def finalize_bundle() -> dict[str, Any]:
    markers = load_all_formal_markers()
    policy = load_batch_policy(require_valid_sources=True)
    policy_sha = sha256_file(BATCH_POLICY_PATH)
    batches_by_model: dict[str, set[int]] = {
        model_id: set() for model_id in MODEL_IDS
    }
    for marker in markers:
        model_id = str(marker["model_id"])
        batches_by_model[model_id].add(int(marker["effective_batch_size"]))
        if marker.get("global_batch_policy_sha256") != policy_sha:
            raise ValueError(
                f"{model_id}/{marker['farm_id']}使用的全局batch策略SHA已漂移"
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
    complete = identities == expected and len(markers) == len(expected)
    report_root = (
        RESULT_ROOT
        if complete
        else RESULT_ROOT / "partial_runs" / "training_summary"
    )
    outputs = write_summaries(markers, report_root)
    payload = {
        "status": "complete" if complete else "partial",
        "created_at": utc_now(),
        "protocol_version": PROTOCOL_VERSION,
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
        # 任一正式调度开始即撤销旧全局complete；只有重新核验140项后才发布。
        (RESULT_ROOT / "round3_training_bundle_complete.json").unlink(
            missing_ok=True
        )
        if args.force:
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
        if completed_marker_valid(paths["marker"]):
            if args.resume:
                if policy is not None:
                    validate_task_policy(paths["marker"], model_id, policy)
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
    bundle = finalize_bundle()
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
