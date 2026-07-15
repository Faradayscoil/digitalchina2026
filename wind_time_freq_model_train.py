"""第四阶段：最小 residual 与 T0--T3 因果时频矩阵训练入口。

本文件把实验口径固定为一个可审计的、低参数量受控消融矩阵：

    T0  已完成的 Stage-3 G0/F7；只读引用，不复制、不重新训练。
    M0  冻结 F7 corrected candidate，仅重建 train-only oracle/Q90 并训练统一门控。
    T1  冻结 F7 residual，在其上训练零初始化的轻量因果时间 adapter。
    T2  冻结 F7 residual，在其上训练仅读取历史窗口的轻量频率 adapter。
    T3  同时使用时间、频率表示及逐维乘性交互的轻量联合 adapter。

T1--T3 先只训练新增 adapter；随后冻结完整 candidate。M0/T1/T2/T3 都从
各自冻结 candidate 的训练窗口重新计算 soft oracle 与逐 horizon |C-P| Q90，
再使用完全相同的 factorized calibrated safe gate 训练协议。未来真值只作为
训练 target，不进入模型输入；频率分支只对 96 步历史功率做 rFFT。

默认正式运行生成 4×5 个新模型，并在 summary 中加入 5 行 T0 只读引用。
任何变体/场站/epoch override 或 ``--smoke-test`` 都写入独立 partial_runs，
不会覆盖正式 bundle，也不会发布 formal complete marker。
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

import wind_RegimeEncoder_PatchTST_feature_screen_train as feature_train
import wind_RegimeEncoder_PatchTST_train as regime_train
import wind_controlled_gate_cali_train as gate_train
import wind_dl_model_train as common_train
from wind_dl_model_train import (
    DATA_DIR,
    FORECAST_LEN,
    HISTORY_LEN,
    TARGET_COL,
    TIME_FREQ,
    make_window_dataset,
    set_global_seed,
)

warnings.filterwarnings("ignore")


MODEL_FAMILY = "time_freq_model"
ARCHITECTURE_VERSION = "time_freq_min_residual_t0_t3_v1"
ARTIFACT_SCHEMA_VERSION = 1
PROTOCOL_VERSION = "time_freq_residual_t0_t3_test_selected_v1"
RESULT_ROOT = os.path.join("./wind_results", MODEL_FAMILY)
SOURCE_VARIANT = "f7"
SOURCE_FEATURE_GROUPS = "P+H+D"
SOURCE_FEATURE_COUNT = 36
RANDOM_SEED = 2026

BATCH_SIZE = int(os.getenv("WIND_TIME_FREQ_BATCH_SIZE", "192"))
VALIDATION_SPLIT = float(os.getenv("WIND_TIME_FREQ_VALIDATION_SPLIT", "0.15"))
CANDIDATE_EPOCHS = int(os.getenv("WIND_TIME_FREQ_CANDIDATE_EPOCHS", "30"))
GATE_ONLY_EPOCHS = int(os.getenv("WIND_TIME_FREQ_GATE_ONLY_EPOCHS", "3"))
CONTEXT_EPOCHS = int(os.getenv("WIND_TIME_FREQ_CONTEXT_EPOCHS", "5"))
CALIBRATED_GATE_EPOCHS = int(os.getenv("WIND_TIME_FREQ_CALIBRATED_GATE_EPOCHS", "30"))
CANDIDATE_LEARNING_RATE = float(
    os.getenv("WIND_TIME_FREQ_CANDIDATE_LEARNING_RATE", "0.0001")
)
GATE_INITIAL_LR = float(os.getenv("WIND_TIME_FREQ_GATE_INITIAL_LR", "0.0001"))
GATE_CALIBRATED_LR = float(os.getenv("WIND_TIME_FREQ_GATE_CALIBRATED_LR", "0.00005"))
EARLY_STOPPING_PATIENCE = int(os.getenv("WIND_TIME_FREQ_PATIENCE", "6"))

PARAMETER_LIMIT = 30000
EXPECTED_TOTAL_PARAMS = {
    "m0": 20409,
    "t1": 23561,
    "t2": 21161,
    "t3": 24697,
}
EXPECTED_ADAPTER_TRAINABLE_PARAMS = {
    "m0": 0,
    "t1": 3152,
    "t2": 752,
    "t3": 4288,
}
EXPECTED_GATE_TRAINABLE_PARAMS = {
    "gate_only": 433,
    "context": 1993,
    "calibrated_gate": 1993,
}
CALIBRATION_WEIGHT = gate_train.CALIBRATION_WEIGHT
DYNAMIC_WEIGHT = gate_train.DYNAMIC_WEIGHT
SAFETY_WEIGHT = gate_train.SAFETY_WEIGHT
SOFT_ORACLE_TEMPERATURE = gate_train.SOFT_ORACLE_TEMPERATURE
CALIBRATION_DIFFERENCE_QUANTILE = gate_train.CALIBRATION_DIFFERENCE_QUANTILE
FACTORIZED_INITIAL_GATE = gate_train.FACTORIZED_INITIAL_GATE
CORRECTION_KERNEL_L2 = regime_train.CORRECTION_KERNEL_L2

VARIANT_SPECS = {
    "t0": {
        "label": "T0 Stage-3 G0/F7 direct reference",
        "requires_training": False,
        "adapter": "none",
        "gate": "existing_nonfactorized_g0",
        "description": "只读引用现有G0/F7模型与测试数据，不重复训练",
    },
    "m0": {
        "label": "M0 frozen minimal residual + calibrated factorized gate",
        "requires_training": True,
        "adapter": "none",
        "gate": "factorized_calibrated_safe",
        "description": "冻结F7 candidate的低成本门控/候选漂移控制",
    },
    "t1": {
        "label": "T1 causal temporal residual adapter",
        "requires_training": True,
        "adapter": "causal_temporal",
        "gate": "factorized_calibrated_safe",
        "description": "dilation=4因果卷积与recent/global历史时间表示",
    },
    "t2": {
        "label": "T2 history-only frequency residual adapter",
        "requires_training": True,
        "adapter": "history_rfft",
        "gate": "factorized_calibrated_safe",
        "description": "仅历史功率rFFT的低/中/高频能量统计",
    },
    "t3": {
        "label": "T3 causal time-frequency interaction adapter",
        "requires_training": True,
        "adapter": "causal_temporal_x_history_rfft",
        "gate": "factorized_calibrated_safe",
        "description": "时间、频率及逐维乘性交互的轻量联合增强",
    },
}
TRAINABLE_VARIANTS = ("m0", "t1", "t2", "t3")
REFERENCE_VARIANTS = ("t0",)
ADAPTER_VARIANTS = ("t1", "t2", "t3")

BASE_WEIGHTED_LAYER_NAMES = tuple(gate_train.COMMON_WEIGHTED_LAYER_NAMES)
CONTEXT_WEIGHTED_LAYER_NAMES = tuple(gate_train.CONTEXT_WEIGHTED_LAYER_NAMES)
ADAPTER_WEIGHTED_LAYER_NAMES = {
    "m0": (),
    "t1": (
        "tf_temporal_causal_conv",
        "tf_temporal_projection",
        "tf_temporal_hidden",
        "tf_residual_delta",
    ),
    "t2": ("tf_frequency_hidden", "tf_residual_delta"),
    "t3": (
        "tf_temporal_causal_conv",
        "tf_temporal_projection",
        "tf_frequency_projection",
        "tf_joint_hidden",
        "tf_residual_delta",
    ),
}

TRAINING_SUMMARY_NAME = "time_freq_model_training_metrics.csv"
MANIFEST_NAME = "time_freq_model_experiment_manifest.csv"
TRAINING_MARKER_NAME = "time_freq_model_training_bundle_complete.json"
PREDICTION_MARKER_RELATIVE_PATH = os.path.join(
    "testdata_predict_output", "time_freq_model_test_bundle_complete.json"
)


def configure_reproducibility():
    os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_SEED))
    set_global_seed(RANDOM_SEED)
    keras.utils.set_random_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


@keras.utils.register_keras_serializable(package="WindTimeFreqModel")
class HistoryBandSpectrum(layers.Layer):
    """Six deterministic spectral descriptors from historical target only."""

    def __init__(self, target_channel_index, history_len=HISTORY_LEN, **kwargs):
        super().__init__(**kwargs)
        self.target_channel_index = int(target_channel_index)
        self.history_len = int(history_len)
        if self.target_channel_index < 0 or self.history_len < 8:
            raise ValueError("HistoryBandSpectrum参数无效")

    def call(self, inputs):
        signal = inputs[:, -self.history_len :, self.target_channel_index]
        signal = signal - tf.reduce_mean(signal, axis=1, keepdims=True)
        window = tf.signal.hann_window(self.history_len, dtype=signal.dtype)
        spectrum = tf.signal.rfft(signal * window[tf.newaxis, :])
        power = tf.math.real(spectrum * tf.math.conj(spectrum))[:, 1:]
        bins = self.history_len // 2
        low_end = max(1, bins // 8)
        mid_end = max(low_end + 1, bins // 3)
        eps = tf.cast(keras.backend.epsilon(), power.dtype)
        low = tf.reduce_mean(power[:, :low_end], axis=1)
        mid = tf.reduce_mean(power[:, low_end:mid_end], axis=1)
        high = tf.reduce_mean(power[:, mid_end:], axis=1)
        total = tf.reduce_mean(power, axis=1)
        frequency = tf.cast(tf.range(1, bins + 1), power.dtype)
        centroid = tf.reduce_sum(power * frequency[tf.newaxis, :], axis=1) / (
            tf.reduce_sum(power, axis=1) + eps
        )
        centroid = centroid / tf.cast(bins, power.dtype)
        ratio = high / (low + eps)
        values = tf.stack(
            [
                tf.math.log1p(low),
                tf.math.log1p(mid),
                tf.math.log1p(high),
                tf.math.log1p(total),
                centroid,
                tf.math.log1p(ratio),
            ],
            axis=-1,
        )
        values = tf.where(tf.math.is_finite(values), values, tf.zeros_like(values))
        return tf.clip_by_value(values, -10.0, 10.0)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 6

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "target_channel_index": self.target_channel_index,
                "history_len": self.history_len,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="WindTimeFreqModel")
class LastSequenceToken(layers.Layer):
    def call(self, inputs):
        return inputs[:, -1, :]


def get_time_freq_custom_objects():
    objects = dict(gate_train.get_controlled_gate_custom_objects())
    for cls in (HistoryBandSpectrum, LastSequenceToken):
        objects[cls.__name__] = cls
        objects[f"WindTimeFreqModel>{cls.__name__}"] = cls
    return objects


def variant_model_name(variant_id):
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"未知T变体: {variant_id}")
    return f"{MODEL_FAMILY}_{variant_id}"


def variant_dirs(variant_id, create=True, result_root=None):
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"未知T变体: {variant_id}")
    root = os.path.join(RESULT_ROOT if result_root is None else result_root, variant_id)
    values = {
        "root": root,
        "models": os.path.join(root, "models"),
        "weights": os.path.join(root, "weights"),
        "preprocess": os.path.join(root, "preprocess"),
        "history": os.path.join(root, "history"),
        "tensorboard": os.path.join(root, "tensorboard"),
        "tails": os.path.join(root, "tails"),
        "validation_diagnostics": os.path.join(root, "validation_diagnostics"),
        "records": os.path.join(root, "records"),
    }
    if create:
        for path in values.values():
            os.makedirs(path, exist_ok=True)
    return values


def expected_farm_ids():
    values = tuple(str(item) for item in feature_train.expected_training_farm_ids())
    if len(values) != 5:
        raise ValueError(f"正式F7来源不是5场站: {values}")
    return values


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


def _array_sha256(named_arrays):
    digest = hashlib.sha256()
    for name, value in named_arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(name).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
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


def _atomic_joblib_dump(value, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    try:
        joblib.dump(value, temporary)
        if not isinstance(joblib.load(temporary), dict):
            raise TypeError("artifact重载后不是dict")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def _file_record(path):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"bundle成员不存在: {path}")
    return {
        "path": os.path.abspath(path),
        "sha256": _sha256(path),
        "size_bytes": int(os.path.getsize(path)),
    }


def _dependency_code_records():
    modules = {
        "feature_screen_train": feature_train,
        "regime_encoder_train": regime_train,
        "controlled_gate_train": gate_train,
        "common_train": common_train,
    }
    return {
        name: _file_record(os.path.realpath(module.__file__))
        for name, module in modules.items()
    }


def _weighted_snapshot(model, layer_names):
    values = []
    for name in layer_names:
        layer = model.get_layer(name)
        for index, weight in enumerate(layer.get_weights()):
            values.append((f"{name}:{index}", weight))
    return values


def _copy_base_weights(source_model, model):
    copied = []
    for name in BASE_WEIGHTED_LAYER_NAMES:
        source = source_model.get_layer(name).get_weights()
        target_layer = model.get_layer(name)
        target = target_layer.get_weights()
        if [item.shape for item in source] != [item.shape for item in target]:
            raise ValueError(f"F7->{model.name}层{name}权重形状不同")
        target_layer.set_weights(source)
        if any(
            not np.array_equal(left, right)
            for left, right in zip(source, target_layer.get_weights())
        ):
            raise ValueError(f"F7->{model.name}层{name}未精确复制")
        copied.append(name)
    return copied


def _temporal_vector(template):
    sequence = template.get_layer("residual_causal_conv_2").output
    enhanced = layers.Conv1D(
        16,
        kernel_size=3,
        dilation_rate=4,
        padding="causal",
        activation="gelu",
        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
        name="tf_temporal_causal_conv",
    )(sequence)
    recent = LastSequenceToken(name="tf_temporal_recent")(enhanced)
    pooled = layers.GlobalAveragePooling1D(name="tf_temporal_global_pool")(enhanced)
    merged = layers.Concatenate(name="tf_temporal_summary")([recent, pooled])
    return layers.Dense(
        16,
        activation="gelu",
        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
        name="tf_temporal_projection",
    )(merged)


def _frequency_features(inputs, target_channel_index):
    return HistoryBandSpectrum(
        target_channel_index=target_channel_index,
        history_len=HISTORY_LEN,
        name="tf_history_band_spectrum",
    )(inputs)


def _adapter_delta(variant_id, template, source_artifact):
    inputs = template.inputs[0]
    if variant_id == "t1":
        temporal = _temporal_vector(template)
        hidden = layers.Dense(
            32,
            activation="gelu",
            kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
            name="tf_temporal_hidden",
        )(temporal)
    elif variant_id == "t2":
        frequency = _frequency_features(inputs, int(source_artifact["target_index"]))
        hidden = layers.Dense(
            32,
            activation="gelu",
            kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
            name="tf_frequency_hidden",
        )(frequency)
    elif variant_id == "t3":
        temporal = _temporal_vector(template)
        frequency = _frequency_features(inputs, int(source_artifact["target_index"]))
        frequency = layers.Dense(
            16,
            activation="gelu",
            kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
            name="tf_frequency_projection",
        )(frequency)
        interaction = layers.Multiply(name="tf_time_frequency_interaction")(
            [temporal, frequency]
        )
        joint = layers.Concatenate(name="tf_joint_summary")(
            [temporal, frequency, interaction]
        )
        hidden = layers.Dense(
            32,
            activation="gelu",
            kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
            name="tf_joint_hidden",
        )(joint)
    else:
        raise ValueError(f"{variant_id}没有adapter")
    return layers.Dense(
        FORECAST_LEN,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
        name="tf_residual_delta",
    )(hidden)


def build_time_freq_model(variant_id, source_artifact, initial_gate_weight=None):
    """Build M0/T1/T2/T3 with a common calibrated factorized gate."""
    if variant_id not in TRAINABLE_VARIANTS:
        raise ValueError(f"{variant_id}不是可训练时频变体")
    configure_reproducibility()
    template = feature_train.build_feature_screen_model_from_artifact(source_artifact)
    persistence = template.get_layer("persistence_forecast_candidate").output
    base_corrected = template.get_layer("corrected_forecast_candidate").output
    if variant_id in ADAPTER_VARIANTS:
        delta = _adapter_delta(variant_id, template, source_artifact)
        corrected_raw = layers.Add(name="tf_adapter_corrected_add")(
            [base_corrected, delta]
        )
    else:
        corrected_raw = base_corrected
    corrected = layers.Activation("linear", name="time_freq_corrected_candidate")(
        corrected_raw
    )
    context = template.get_layer("regime_context").output
    gate, q_by_horizon, horizon_prior = gate_train._build_factorized_gate(context)
    forecast = regime_train.TwoCandidateGateFusion(name="forecast_power")(
        [persistence, corrected, gate]
    )
    candidate = layers.Activation("linear", name="candidate_forecast")(corrected)
    packet = layers.Concatenate(name="control_packet")(
        [gate, persistence, corrected, forecast, q_by_horizon, horizon_prior]
    )
    model = keras.Model(
        template.inputs,
        {
            "forecast_power": forecast,
            "candidate_forecast": candidate,
            "control_packet": packet,
        },
        name=f"WindTimeFreqModel_{variant_id.upper()}",
    )
    total_params = int(model.count_params())
    if total_params != EXPECTED_TOTAL_PARAMS[variant_id]:
        raise ValueError(
            f"{variant_id}参数量{total_params} != 预注册值"
            f"{EXPECTED_TOTAL_PARAMS[variant_id]}"
        )
    if total_params >= PARAMETER_LIMIT:
        raise ValueError(
            f"{variant_id}参数量{model.count_params()}不满足< {PARAMETER_LIMIT}"
        )
    return model


def diagnostic_model(model):
    packet = model.get_layer("control_packet").output
    h = FORECAST_LEN
    return keras.Model(
        model.inputs,
        {
            "forecast": model.get_layer("forecast_power").output,
            "persistence": model.get_layer("persistence_forecast_candidate").output,
            "corrected": model.get_layer("time_freq_corrected_candidate").output,
            "gate": packet[:, :h],
            "q": packet[:, 4 * h : 5 * h],
            "s": packet[:, 5 * h : 6 * h],
        },
    )


def _base_candidate_model(model):
    return keras.Model(
        model.inputs,
        model.get_layer("corrected_forecast_candidate").output,
    )


def _final_candidate_model(model):
    return keras.Model(
        model.inputs,
        model.get_layer("time_freq_corrected_candidate").output,
    )


def _plain_datasets(prepared):
    return make_window_dataset(
        prepared["features"],
        prepared["target"],
        HISTORY_LEN,
        FORECAST_LEN,
        BATCH_SIZE,
        VALIDATION_SPLIT,
    )


def _attach_gate_targets(dataset):
    def attach(batch_x, batch_y):
        return batch_x, {
            "forecast_power": batch_y,
            "candidate_forecast": batch_y,
            "control_packet": batch_y,
        }

    return dataset.map(
        attach, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    ).prefetch(tf.data.AUTOTUNE)


def _inverse_scaled(values, prepared):
    shape = np.asarray(values).shape
    result = (
        prepared["scaler_y"]
        .inverse_transform(np.asarray(values).reshape(-1, 1))
        .reshape(shape)
    )
    return np.clip(result, 0.0, float(prepared["capacity"]))


def _capacity_fraction(values, prepared):
    return _inverse_scaled(values, prepared) / float(prepared["capacity"])


def estimate_candidate_calibration_statistics(model, train_ds, prepared):
    """Recompute oracle and Q90 from this frozen candidate's train windows only."""
    diagnostic = diagnostic_model(model)
    oracle_sum = 0.0
    element_count = 0
    differences = []
    for batch_x, batch_y in train_ds:
        output = diagnostic(batch_x, training=False)
        truth = _capacity_fraction(batch_y.numpy(), prepared)
        persistence = _capacity_fraction(output["persistence"].numpy(), prepared)
        corrected = _capacity_fraction(output["corrected"].numpy(), prepared)
        e_p = np.abs(truth - persistence)
        e_c = np.abs(truth - corrected)
        advantage = (e_p - e_c) / (e_p + e_c + 1e-8)
        oracle = 1.0 / (1.0 + np.exp(-advantage / SOFT_ORACLE_TEMPERATURE))
        oracle_sum += float(oracle.sum())
        element_count += int(oracle.size)
        differences.append(np.abs(corrected - persistence))
    if not differences or element_count == 0:
        raise ValueError("训练窗口为空，无法重建soft oracle/Q90")
    difference = np.concatenate(differences, axis=0)
    raw_oracle_mean = float(oracle_sum / element_count)
    q90 = np.quantile(difference, CALIBRATION_DIFFERENCE_QUANTILE, axis=0).astype(
        np.float32
    )
    if q90.shape != (FORECAST_LEN,) or not np.isfinite(q90).all():
        raise FloatingPointError("candidate difference Q90无效")
    return {
        "soft_oracle_mean": raw_oracle_mean,
        "soft_oracle_mean_clipped_for_optional_initialization": float(
            np.clip(raw_oracle_mean, 0.05, 0.95)
        ),
        "candidate_difference_q90": q90,
        "sample_count": int(difference.shape[0]),
        "element_count": int(element_count),
        "scope": "per_farm_per_horizon_train_frozen_variant_candidate",
    }


def _validation_ece(probability, truth, bins=10):
    probability = np.asarray(probability, dtype=float)
    truth = np.asarray(truth, dtype=float)
    ids = np.minimum((np.clip(probability, 0, 1) * bins).astype(int), bins - 1)
    total = 0.0
    for bin_id in range(bins):
        mask = ids == bin_id
        if mask.any():
            total += float(mask.mean()) * abs(
                float(probability[mask].mean()) - float(truth[mask].mean())
            )
    return float(total)


def validation_diagnostics(model, val_ds, prepared, variant_id):
    diagnostic = diagnostic_model(model)
    output_values = {
        key: [] for key in ("forecast", "persistence", "corrected", "gate", "q", "s")
    }
    truths = []
    for batch_x, batch_y in val_ds:
        result = diagnostic(batch_x, training=False)
        truths.append(np.asarray(batch_y))
        for key in output_values:
            output_values[key].append(np.asarray(result[key]))
    if not truths:
        raise ValueError("验证集为空")
    truth = _inverse_scaled(np.concatenate(truths), prepared)
    values = {key: np.concatenate(value) for key, value in output_values.items()}
    persistence = _inverse_scaled(values["persistence"], prepared)
    corrected = _inverse_scaled(values["corrected"], prepared)
    forecast = _inverse_scaled(values["forecast"], prepared)
    gate = values["gate"]
    valid = (
        np.isfinite(truth)
        & np.isfinite(persistence)
        & np.isfinite(corrected)
        & np.isfinite(forecast)
        & np.isfinite(gate)
    )
    if not valid.any():
        raise FloatingPointError("验证输出没有有限元素")
    capacity = float(prepared["capacity"])
    error = forecast[valid] - truth[valid]
    candidate_error = corrected[valid] - truth[valid]
    persistence_abs = np.abs(persistence[valid] - truth[valid]) / capacity
    forecast_abs = np.abs(error) / capacity
    oracle = np.abs(candidate_error) < np.abs(persistence[valid] - truth[valid])
    positive_regret = np.maximum(0.0, forecast_abs - persistence_abs)
    row = {
        "variant_id": variant_id,
        "farm_id": str(prepared["farm_id"]),
        "valid_count": int(valid.sum()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "capacity_normalized_mae": float(np.mean(np.abs(error)) / capacity),
        "capacity_normalized_rmse": float(
            np.sqrt(np.mean(np.square(error))) / capacity
        ),
        "corrected_capacity_normalized_rmse": float(
            np.sqrt(np.mean(np.square(candidate_error))) / capacity
        ),
        "gate_mean": float(np.mean(gate)),
        "gate_std": float(np.std(gate)),
        "q_mean": float(np.mean(values["q"])),
        "s_mean": float(np.mean(values["s"])),
        "positive_regret_mean": float(np.mean(positive_regret)),
        "harm_rate_0_005": float(np.mean((forecast_abs - persistence_abs) > 0.005)),
        "oracle_brier": float(np.mean(np.square(gate[valid] - oracle.astype(float)))),
        "ece_10bin": _validation_ece(gate[valid], oracle),
        "diagnostic_scope": "validation_checkpoint_only_not_final_selection",
    }
    if not all(
        np.isfinite(value) for key, value in row.items() if isinstance(value, float)
    ):
        raise FloatingPointError("验证诊断含非有限数")
    return pd.DataFrame([row])


class CandidateValidationCheckpoint(keras.callbacks.Callback):
    def __init__(self, full_model, path, validation_dataset, prepared):
        super().__init__()
        self.full_model = full_model
        self.path = path
        self.validation_dataset = validation_dataset
        self.prepared = prepared
        self.best = np.inf
        self.records = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        values = []
        truths = []
        candidate = _final_candidate_model(self.full_model)
        for batch_x, batch_y in self.validation_dataset:
            values.append(np.asarray(candidate(batch_x, training=False)))
            truths.append(np.asarray(batch_y))
        prediction = _inverse_scaled(np.concatenate(values), self.prepared)
        truth = _inverse_scaled(np.concatenate(truths), self.prepared)
        nrmse = float(
            np.sqrt(np.mean(np.square(prediction - truth)))
            / float(self.prepared["capacity"])
        )
        if not np.isfinite(nrmse):
            raise FloatingPointError("candidate validation NRMSE非有限")
        logs["selection_val_candidate_nrmse"] = nrmse
        updated = nrmse < self.best - 1e-12
        self.records.append(
            {
                "phase_epoch": int(epoch),
                "candidate_validation_nrmse": nrmse,
                "checkpoint_updated": updated,
            }
        )
        if updated:
            self.best = nrmse
            self.full_model.save_weights(self.path)


class GateValidationCheckpoint(keras.callbacks.Callback):
    """Select across gate phases by NRMSE then regret/Brier in a 0.1% band."""

    def __init__(self, path, validation_dataset, prepared, variant_id):
        super().__init__()
        self.path = path
        self.validation_dataset = validation_dataset
        self.prepared = prepared
        self.variant_id = variant_id
        self.phase = None
        self.records = []
        self.snapshots = []
        self.best = np.inf
        self.best_regret = np.inf
        self.best_brier = np.inf
        self.best_phase = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = (
            validation_diagnostics(
                self.model, self.validation_dataset, self.prepared, self.variant_id
            )
            .iloc[0]
            .to_dict()
        )
        nrmse = float(row["capacity_normalized_rmse"])
        regret = float(row["positive_regret_mean"])
        brier = float(row["oracle_brier"])
        logs["selection_val_nrmse"] = nrmse
        logs["selection_val_positive_regret"] = regret
        logs["selection_val_brier"] = brier
        self.records.append(
            {
                "global_epoch": len(self.records),
                "phase": self.phase,
                "phase_epoch": int(epoch),
                **row,
            }
        )
        self.snapshots.append(
            [np.array(value, copy=True) for value in self.model.get_weights()]
        )

    def finalize(self):
        if not self.records or len(self.records) != len(self.snapshots):
            raise ValueError("门控checkpoint轨迹不完整")
        frame = pd.DataFrame(self.records)
        minimum = float(frame["capacity_normalized_rmse"].min())
        eligible = frame[frame["capacity_normalized_rmse"] <= minimum * 1.001].copy()
        index = int(
            eligible.sort_values(
                [
                    "positive_regret_mean",
                    "oracle_brier",
                    "capacity_normalized_rmse",
                    "global_epoch",
                ],
                kind="stable",
            ).index[0]
        )
        selected = frame.loc[index]
        self.model.set_weights(self.snapshots[index])
        self.model.save_weights(self.path)
        self.best = float(selected["capacity_normalized_rmse"])
        self.best_regret = float(selected["positive_regret_mean"])
        self.best_brier = float(selected["oracle_brier"])
        self.best_phase = str(selected["phase"])
        frame["selected_checkpoint"] = frame.index == index
        return frame


def _set_all_trainable(model, trainable=False):
    for layer in model.layers:
        layer.trainable = bool(trainable)


def _set_candidate_adapter_phase(model, variant_id):
    _set_all_trainable(model, False)
    for name in ADAPTER_WEIGHTED_LAYER_NAMES[variant_id]:
        model.get_layer(name).trainable = True
    model.get_layer("residual_dropout").rate = 0.0
    model.get_layer("regime_context_dropout").rate = 0.0


def _set_gate_phase(model, phase):
    if phase not in {"gate_only", "context", "calibrated_gate"}:
        raise ValueError(f"未知门控phase: {phase}")
    _set_all_trainable(model, False)
    for name in (
        "sample_dynamic_hidden",
        "sample_dynamic_dropout",
        "sample_dynamic_probability",
        "sample_dynamic_probability_by_horizon",
        "horizon_gate_prior",
        "controlled_gate",
    ):
        model.get_layer(name).trainable = True
    if phase in {"context", "calibrated_gate"}:
        for name in CONTEXT_WEIGHTED_LAYER_NAMES:
            model.get_layer(name).trainable = True
    model.get_layer("residual_dropout").rate = 0.0
    model.get_layer("regime_context_dropout").rate = (
        0.0 if phase == "gate_only" else float(feature_train.GATE_DROPOUT)
    )


def _trainable_parameter_count(model):
    return int(sum(int(np.prod(weight.shape)) for weight in model.trainable_weights))


def _compile_gate(model, prepared, q90, learning_rate):
    auxiliary = gate_train.ControlledGateAuxiliaryLoss(
        forecast_len=FORECAST_LEN,
        target_mean=float(prepared["scaler_y"].mean_[0]),
        target_scale=float(prepared["scaler_y"].scale_[0]),
        capacity=float(prepared["capacity"]),
        calibration_weight=CALIBRATION_WEIGHT,
        dynamic_weight=DYNAMIC_WEIGHT,
        safety_weight=SAFETY_WEIGHT,
        candidate_difference_q90=q90,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss={
            "forecast_power": keras.losses.Huber(delta=1.0),
            "candidate_forecast": keras.losses.Huber(delta=1.0),
            "control_packet": auxiliary,
        },
        loss_weights={
            "forecast_power": 1.0,
            "candidate_forecast": 0.0,
            "control_packet": 1.0,
        },
        metrics={
            "forecast_power": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ]
        },
    )


def _history_frame(histories):
    rows = []
    global_epoch = 0
    for phase, history in histories:
        count = len(next(iter(history.history.values()))) if history.history else 0
        for index in range(count):
            row = {"global_epoch": global_epoch, "phase": phase, "phase_epoch": index}
            row.update({key: values[index] for key, values in history.history.items()})
            rows.append(row)
            global_epoch += 1
    return pd.DataFrame(rows)


def _save_model_atomic(model, path):
    stem, extension = os.path.splitext(path)
    temporary = f"{stem}.tmp{extension}"
    try:
        model.save(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            if os.path.isdir(temporary):
                import shutil

                shutil.rmtree(temporary)
            else:
                os.remove(temporary)


def _train_paths(dirs, variant_id, farm_id):
    name = variant_model_name(variant_id)
    prefix = f"{name}_farm_{farm_id}"
    return {
        "model_path": os.path.join(dirs["models"], f"{prefix}.keras"),
        "weights_path": os.path.join(dirs["weights"], f"{prefix}_best.weights.h5"),
        "candidate_weights_path": os.path.join(
            dirs["weights"], f"{prefix}_candidate_best.weights.h5"
        ),
        "artifact_path": os.path.join(dirs["preprocess"], f"{prefix}_preprocess.pkl"),
        "candidate_history_path": os.path.join(
            dirs["history"], f"{prefix}_candidate_history.csv"
        ),
        "gate_history_path": os.path.join(
            dirs["history"], f"{prefix}_gate_history.csv"
        ),
        "validation_path": os.path.join(
            dirs["validation_diagnostics"], f"{prefix}_validation.csv"
        ),
        "checkpoint_trace_path": os.path.join(
            dirs["validation_diagnostics"], f"{prefix}_checkpoint_trace.csv"
        ),
        "tail_path": os.path.join(dirs["tails"], f"{prefix}_tail.csv"),
        "record_path": os.path.join(dirs["records"], f"{prefix}_record.json"),
    }


def _assert_source_compatible(prepared, artifact):
    gate_train._validate_prepared_against_source(prepared, artifact)
    if list(artifact.get("selected_regime_feature_groups", ())) != ["P", "H", "D"]:
        raise ValueError("来源F7不是P+H+D")


def _source_and_initial_model(variant_id, prepared):
    farm_id = str(prepared["farm_id"])
    source_model, artifact, artifact_path, source_model_path = (
        gate_train.load_source_f7(farm_id)
    )
    _assert_source_compatible(prepared, artifact)
    model = build_time_freq_model(variant_id, artifact)
    copied = _copy_base_weights(source_model, model)
    sample_x, _ = next(iter(_plain_datasets(prepared)[0]))
    sample_x = sample_x[:2]
    source_diag = gate_train._source_diagnostic_model(source_model)(
        sample_x, training=False
    )
    target_diag = diagnostic_model(model)(sample_x, training=False)
    for name in ("persistence", "corrected"):
        if not np.array_equal(
            np.asarray(source_diag[name]), np.asarray(target_diag[name])
        ):
            difference = float(
                np.max(
                    np.abs(
                        np.asarray(source_diag[name]) - np.asarray(target_diag[name])
                    )
                )
            )
            raise ValueError(f"{variant_id}零初始化未精确复现F7 {name}: {difference}")
    source_context = np.asarray(source_diag["context"])
    target_context = np.asarray(
        keras.Model(model.inputs, model.get_layer("regime_context").output)(
            sample_x, training=False
        )
    )
    if not np.array_equal(source_context, target_context):
        difference = float(np.max(np.abs(source_context - target_context)))
        raise ValueError(f"{variant_id}显式工况context未精确复现F7: {difference}")
    source_hash = _array_sha256(
        _weighted_snapshot(source_model, BASE_WEIGHTED_LAYER_NAMES)
    )
    initial_hash = _array_sha256(_weighted_snapshot(model, BASE_WEIGHTED_LAYER_NAMES))
    if source_hash != initial_hash:
        raise ValueError("F7公共权重快照复制hash不一致")
    return {
        "source_model": source_model,
        "source_artifact": artifact,
        "source_artifact_path": artifact_path,
        "source_model_path": source_model_path,
        "model": model,
        "sample_x": sample_x,
        "copied_layers": copied,
        "source_base_snapshot_sha256": source_hash,
        "initial_base_snapshot_sha256": initial_hash,
    }


def train_variant_for_farm(
    variant_id,
    prepared,
    result_root=None,
    candidate_epochs=CANDIDATE_EPOCHS,
    gate_only_epochs=GATE_ONLY_EPOCHS,
    context_epochs=CONTEXT_EPOCHS,
    calibrated_gate_epochs=CALIBRATED_GATE_EPOCHS,
):
    if variant_id not in TRAINABLE_VARIANTS:
        raise ValueError(f"禁止训练{variant_id}")
    keras.backend.clear_session()
    configure_reproducibility()
    farm_id = str(prepared["farm_id"])
    state = _source_and_initial_model(variant_id, prepared)
    model = state["model"]
    source_model = state["source_model"]
    plain_train, plain_val, train_samples, total_samples = _plain_datasets(prepared)
    dirs = variant_dirs(variant_id, result_root=result_root)
    paths = _train_paths(dirs, variant_id, farm_id)
    for name in ("weights_path", "candidate_weights_path"):
        if os.path.exists(paths[name]):
            os.remove(paths[name])

    base_snapshot_before = _array_sha256(
        _weighted_snapshot(model, regime_train.B2_WEIGHTED_LAYER_NAMES)
    )
    initial_candidate = np.asarray(
        diagnostic_model(model)(state["sample_x"], training=False)["corrected"]
    )
    candidate_history = pd.DataFrame()
    candidate_best_nrmse = np.nan
    candidate_start = time.monotonic()
    if variant_id in ADAPTER_VARIANTS:
        _set_candidate_adapter_phase(model, variant_id)
        adapter_trainable = _trainable_parameter_count(model)
        if adapter_trainable != EXPECTED_ADAPTER_TRAINABLE_PARAMS[variant_id]:
            raise ValueError(
                f"{variant_id} adapter可训练参数{adapter_trainable} != "
                f"{EXPECTED_ADAPTER_TRAINABLE_PARAMS[variant_id]}"
            )
        candidate_model = _final_candidate_model(model)
        candidate_model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=CANDIDATE_LEARNING_RATE, clipnorm=1.0
            ),
            loss=keras.losses.Huber(delta=1.0),
            metrics=[
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
        )
        checkpoint = CandidateValidationCheckpoint(
            model, paths["candidate_weights_path"], plain_val, prepared
        )
        finite_guard = feature_train.NonFiniteTrainingGuard()
        history = candidate_model.fit(
            plain_train,
            validation_data=plain_val,
            epochs=int(candidate_epochs),
            callbacks=[
                finite_guard,
                keras.callbacks.TerminateOnNaN(),
                checkpoint,
                keras.callbacks.EarlyStopping(
                    monitor="selection_val_candidate_nrmse",
                    mode="min",
                    patience=EARLY_STOPPING_PATIENCE,
                    restore_best_weights=False,
                    verbose=1,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="selection_val_candidate_nrmse",
                    mode="min",
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                    verbose=1,
                ),
                keras.callbacks.TensorBoard(
                    log_dir=os.path.join(
                        dirs["tensorboard"],
                        f"farm_{farm_id}",
                        datetime.now().strftime("%Y%m%d-%H%M%S"),
                        "candidate_adapter",
                    ),
                    histogram_freq=0,
                    profile_batch=0,
                ),
            ],
            verbose=1,
        )
        feature_train.ensure_finite_training_history(history, finite_guard)
        if not os.path.exists(paths["candidate_weights_path"]):
            raise FileNotFoundError("candidate checkpoint未生成")
        model.load_weights(paths["candidate_weights_path"])
        candidate_history = pd.DataFrame(history.history)
        candidate_history.insert(0, "epoch", np.arange(len(candidate_history)))
        candidate_history["selection_val_candidate_nrmse"] = [
            item["candidate_validation_nrmse"] for item in checkpoint.records
        ]
        candidate_best_nrmse = float(checkpoint.best)
    else:
        model.save_weights(paths["candidate_weights_path"])
        candidate_history = pd.DataFrame(
            [
                {
                    "epoch": -1,
                    "phase": "skipped_frozen_f7_candidate",
                    "selection_val_candidate_nrmse": validation_diagnostics(
                        model, plain_val, prepared, variant_id
                    ).iloc[0]["corrected_capacity_normalized_rmse"],
                }
            ]
        )
        candidate_best_nrmse = float(
            candidate_history.iloc[0]["selection_val_candidate_nrmse"]
        )
    candidate_elapsed = float(time.monotonic() - candidate_start)
    _atomic_to_csv(candidate_history, paths["candidate_history_path"])

    base_snapshot_after_candidate = _array_sha256(
        _weighted_snapshot(model, regime_train.B2_WEIGHTED_LAYER_NAMES)
    )
    if base_snapshot_after_candidate != base_snapshot_before:
        raise ValueError("adapter训练改变了冻结F7 residual")
    frozen_candidate = np.asarray(
        diagnostic_model(model)(state["sample_x"], training=False)["corrected"]
    )
    candidate_drift = float(np.max(np.abs(frozen_candidate - initial_candidate)))
    candidate_snapshot_before_gate = _array_sha256(
        _weighted_snapshot(
            model,
            tuple(regime_train.B2_WEIGHTED_LAYER_NAMES)
            + tuple(ADAPTER_WEIGHTED_LAYER_NAMES[variant_id]),
        )
    )
    calibration = estimate_candidate_calibration_statistics(
        model, plain_train, prepared
    )
    if calibration["sample_count"] != int(train_samples):
        raise ValueError("soft-oracle/Q90并非由完整训练窗口生成")

    train_ds = _attach_gate_targets(plain_train)
    val_ds = _attach_gate_targets(plain_val)
    gate_checkpoint = GateValidationCheckpoint(
        paths["weights_path"], plain_val, prepared, variant_id
    )
    histories = []
    gate_phase_trainable_params = {}
    phase_specs = (
        ("gate_only", int(gate_only_epochs), GATE_INITIAL_LR),
        ("context", int(context_epochs), GATE_INITIAL_LR),
        ("calibrated_gate", int(calibrated_gate_epochs), GATE_CALIBRATED_LR),
    )
    gate_start = time.monotonic()
    for phase, epochs, learning_rate in phase_specs:
        if epochs <= 0:
            continue
        _set_gate_phase(model, phase)
        trainable_count = _trainable_parameter_count(model)
        expected_trainable = EXPECTED_GATE_TRAINABLE_PARAMS[phase]
        if trainable_count != expected_trainable:
            raise ValueError(
                f"{variant_id}/{phase}可训练参数{trainable_count} != "
                f"预注册值{expected_trainable}"
            )
        gate_phase_trainable_params[phase] = trainable_count
        _compile_gate(
            model,
            prepared,
            calibration["candidate_difference_q90"],
            learning_rate,
        )
        gate_checkpoint.phase = phase
        finite_guard = feature_train.NonFiniteTrainingGuard()
        callbacks = [
            finite_guard,
            keras.callbacks.TerminateOnNaN(),
            gate_checkpoint,
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(
                    dirs["tensorboard"],
                    f"farm_{farm_id}",
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    phase,
                ),
                histogram_freq=0,
                profile_batch=0,
            ),
        ]
        if phase == "calibrated_gate":
            callbacks.extend(
                [
                    keras.callbacks.EarlyStopping(
                        monitor="selection_val_nrmse",
                        mode="min",
                        patience=EARLY_STOPPING_PATIENCE,
                        restore_best_weights=False,
                        verbose=1,
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor="selection_val_nrmse",
                        mode="min",
                        factor=0.5,
                        patience=3,
                        min_lr=1e-6,
                        verbose=1,
                    ),
                ]
            )
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        feature_train.ensure_finite_training_history(history, finite_guard)
        histories.append((phase, history))
    checkpoint_trace = gate_checkpoint.finalize()
    model.load_weights(paths["weights_path"])
    gate_elapsed = float(time.monotonic() - gate_start)
    gate_history = _history_frame(histories)
    if len(gate_history) != len(checkpoint_trace):
        raise ValueError("门控history与checkpoint trace长度不一致")
    _atomic_to_csv(gate_history, paths["gate_history_path"])
    _atomic_to_csv(checkpoint_trace, paths["checkpoint_trace_path"])

    candidate_snapshot_after_gate = _array_sha256(
        _weighted_snapshot(
            model,
            tuple(regime_train.B2_WEIGHTED_LAYER_NAMES)
            + tuple(ADAPTER_WEIGHTED_LAYER_NAMES[variant_id]),
        )
    )
    if candidate_snapshot_after_gate != candidate_snapshot_before_gate:
        raise ValueError("门控校准改变了冻结candidate")
    candidate_after_gate = np.asarray(
        diagnostic_model(model)(state["sample_x"], training=False)["corrected"]
    )
    candidate_gate_drift = float(
        np.max(np.abs(candidate_after_gate - frozen_candidate))
    )
    if candidate_gate_drift != 0.0:
        raise ValueError(f"门控校准后candidate输出不再逐点一致: {candidate_gate_drift}")
    validation = validation_diagnostics(model, plain_val, prepared, variant_id)
    _atomic_to_csv(validation, paths["validation_path"])
    prepared["train_df"].iloc[-HISTORY_LEN:].to_csv(paths["tail_path"], index=True)
    _save_model_atomic(model, paths["model_path"])
    restored = keras.models.load_model(
        paths["model_path"],
        custom_objects=get_time_freq_custom_objects(),
        compile=False,
    )
    expected = diagnostic_model(model)(state["sample_x"], training=False)
    actual = diagnostic_model(restored)(state["sample_x"], training=False)
    for name in expected:
        if not np.allclose(expected[name], actual[name], rtol=1e-7, atol=1e-7):
            raise ValueError(f"保存/重载{name}输出不一致")

    total_params = int(model.count_params())
    if total_params != EXPECTED_TOTAL_PARAMS[variant_id]:
        raise ValueError(
            f"{variant_id}最终参数量{total_params} != "
            f"预注册值{EXPECTED_TOTAL_PARAMS[variant_id]}"
        )
    if total_params >= PARAMETER_LIMIT:
        raise ValueError(f"{variant_id}参数量{total_params}超过上限")
    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_family": MODEL_FAMILY,
        "architecture_version": ARCHITECTURE_VERSION,
        "variant_id": variant_id,
        "variant_spec": dict(VARIANT_SPECS[variant_id]),
        "farm_id": farm_id,
        "random_seed": RANDOM_SEED,
        "history_len": HISTORY_LEN,
        "forecast_len": FORECAST_LEN,
        "time_freq": TIME_FREQ,
        "target_col": TARGET_COL,
        "train_file": os.path.abspath(prepared["train_file"]),
        "train_file_sha256": _sha256(prepared["train_file"]),
        "feature_cols": prepared["feature_cols"],
        "input_cols": prepared["input_cols"],
        "target_index": prepared["target_index"],
        "scaler_x": prepared["scaler_x"],
        "scaler_y": prepared["scaler_y"],
        "capacity": float(prepared["capacity"]),
        "power_scale_ratio": prepared["power_scale_ratio"],
        "power_scale_offset": prepared["power_scale_offset"],
        "regime_feature_config": prepared["regime_feature_config"],
        "selected_regime_feature_groups": ["P", "H", "D"],
        "selected_regime_feature_names": list(
            feature_train.selected_feature_names(SOURCE_VARIANT)
        ),
        "selected_regime_feature_count": SOURCE_FEATURE_COUNT,
        "adapter_definition": VARIANT_SPECS[variant_id]["adapter"],
        "candidate_training": {
            "enabled": variant_id in ADAPTER_VARIANTS,
            "epochs_requested": int(candidate_epochs),
            "epochs_actual": int(len(candidate_history))
            if variant_id in ADAPTER_VARIANTS
            else 0,
            "learning_rate": CANDIDATE_LEARNING_RATE,
            "best_validation_nrmse": candidate_best_nrmse,
            "base_f7_residual_frozen": True,
            "zero_initialized_delta_head": variant_id in ADAPTER_VARIANTS,
            "test_used": False,
        },
        "candidate_calibration": {
            "soft_oracle_temperature": SOFT_ORACLE_TEMPERATURE,
            "soft_oracle_train_mean": calibration["soft_oracle_mean"],
            "soft_oracle_train_mean_clipped_for_optional_initialization": calibration[
                "soft_oracle_mean_clipped_for_optional_initialization"
            ],
            "candidate_difference_q90": calibration[
                "candidate_difference_q90"
            ].tolist(),
            "quantile": CALIBRATION_DIFFERENCE_QUANTILE,
            "sample_count": calibration["sample_count"],
            "element_count": calibration["element_count"],
            "scope": calibration["scope"],
            "future_truth_role": "train_target_only",
        },
        "gate_training": {
            "topology": "pi_i_h=q_i*s_h",
            "candidate_frozen_all_phases": True,
            "calibration_weight": CALIBRATION_WEIGHT,
            "dynamic_weight": DYNAMIC_WEIGHT,
            "safety_weight": SAFETY_WEIGHT,
            "phases": [
                {
                    "phase": phase,
                    "epochs": epochs,
                    "learning_rate": lr,
                    "trainable_parameter_count": gate_phase_trainable_params.get(phase),
                }
                for phase, epochs, lr in phase_specs
            ],
        },
        "batch_size": BATCH_SIZE,
        "validation_split": VALIDATION_SPLIT,
        "selection_split": "test_in_prediction_script",
        "test_used_for_training": False,
        "test_is_final_blind_evaluation": False,
        "training_code_path": os.path.abspath(__file__),
        "training_code_sha256": _sha256(__file__),
        "dependency_code": _dependency_code_records(),
        "legacy_bidirectional_weather_imputation": True,
        "scaler_fit_scope": "full_train_file_including_validation",
        "validation_target_overlap_steps": FORECAST_LEN - 1,
        "copied_layer_names": state["copied_layers"],
        "source_f7_model_path": os.path.abspath(state["source_model_path"]),
        "source_f7_model_sha256": _sha256(state["source_model_path"]),
        "source_f7_artifact_path": os.path.abspath(state["source_artifact_path"]),
        "source_f7_artifact_sha256": _sha256(state["source_artifact_path"]),
        "source_base_snapshot_sha256": state["source_base_snapshot_sha256"],
        "initial_base_snapshot_sha256": state["initial_base_snapshot_sha256"],
        "post_candidate_base_snapshot_sha256": base_snapshot_after_candidate,
        "candidate_snapshot_before_gate_sha256": candidate_snapshot_before_gate,
        "candidate_snapshot_after_gate_sha256": candidate_snapshot_after_gate,
        "candidate_output_before_gate_sha256": _array_sha256(
            [("corrected", frozen_candidate)]
        ),
        "candidate_output_after_gate_sha256": _array_sha256(
            [("corrected", candidate_after_gate)]
        ),
        "candidate_gate_calibration_max_abs_drift": candidate_gate_drift,
        "candidate_probe_max_abs_drift_from_f7": candidate_drift,
        "candidate_probe_scope": "first_seeded_shuffled_train_batch_first_2_windows",
        # 保留旧键供首轮预测代码兼容；该值仅是probe，不是全测试集drift。
        "candidate_max_abs_drift_from_f7": candidate_drift,
        "model_path": os.path.abspath(paths["model_path"]),
        "model_sha256": _sha256(paths["model_path"]),
        "best_weights_path": os.path.abspath(paths["weights_path"]),
        "best_weights_sha256": _sha256(paths["weights_path"]),
        "candidate_weights_path": os.path.abspath(paths["candidate_weights_path"]),
        "candidate_weights_sha256": _sha256(paths["candidate_weights_path"]),
        "artifact_path": os.path.abspath(paths["artifact_path"]),
        "candidate_history_path": os.path.abspath(paths["candidate_history_path"]),
        "candidate_history_sha256": _sha256(paths["candidate_history_path"]),
        "gate_history_path": os.path.abspath(paths["gate_history_path"]),
        "gate_history_sha256": _sha256(paths["gate_history_path"]),
        "validation_diagnostics_path": os.path.abspath(paths["validation_path"]),
        "validation_diagnostics_sha256": _sha256(paths["validation_path"]),
        "checkpoint_trace_path": os.path.abspath(paths["checkpoint_trace_path"]),
        "checkpoint_trace_sha256": _sha256(paths["checkpoint_trace_path"]),
        "tail_path": os.path.abspath(paths["tail_path"]),
        "tail_sha256": _sha256(paths["tail_path"]),
        "total_params": total_params,
        "parameter_limit": PARAMETER_LIMIT,
        "candidate_training_elapsed_seconds": candidate_elapsed,
        "gate_training_elapsed_seconds": gate_elapsed,
        "train_samples": int(train_samples),
        "validation_samples": int(total_samples - train_samples),
        "best_validation_nrmse": gate_checkpoint.best,
        "best_validation_positive_regret": gate_checkpoint.best_regret,
        "best_validation_brier": gate_checkpoint.best_brier,
        "best_phase": gate_checkpoint.best_phase,
    }
    _atomic_joblib_dump(artifact, paths["artifact_path"])
    row = validation.iloc[0].to_dict()
    row.update(
        {
            "model_family": MODEL_FAMILY,
            "variant_id": variant_id,
            "variant_label": VARIANT_SPECS[variant_id]["label"],
            "farm_id": farm_id,
            "feature_groups": SOURCE_FEATURE_GROUPS,
            "feature_count": SOURCE_FEATURE_COUNT,
            "reference_only": False,
            "requires_training": True,
            "random_seed": RANDOM_SEED,
            "parameter_count": total_params,
            "parameter_limit": PARAMETER_LIMIT,
            "candidate_adapter": VARIANT_SPECS[variant_id]["adapter"],
            "candidate_probe_drift_max_abs": candidate_drift,
            "candidate_probe_scope": "first_seeded_shuffled_train_batch_first_2_windows",
            "candidate_drift_max_abs": candidate_drift,
            "candidate_gate_calibration_max_abs_drift": candidate_gate_drift,
            "training_code_path": os.path.abspath(__file__),
            "training_code_sha256": _sha256(__file__),
            "candidate_training_elapsed_seconds": candidate_elapsed,
            "gate_training_elapsed_seconds": gate_elapsed,
            "candidate_epoch_count": int(len(candidate_history))
            if variant_id in ADAPTER_VARIANTS
            else 0,
            "gate_epoch_count": int(len(gate_history)),
            "best_validation_nrmse": gate_checkpoint.best,
            "best_validation_positive_regret": gate_checkpoint.best_regret,
            "best_validation_brier": gate_checkpoint.best_brier,
            "best_phase": gate_checkpoint.best_phase,
            "source_variant": SOURCE_VARIANT,
            "source_model_path": os.path.abspath(state["source_model_path"]),
            "source_model_sha256": _sha256(state["source_model_path"]),
            "model_path": os.path.abspath(paths["model_path"]),
            "model_sha256": _sha256(paths["model_path"]),
            "best_weights_path": os.path.abspath(paths["weights_path"]),
            "best_weights_sha256": _sha256(paths["weights_path"]),
            "candidate_weights_path": os.path.abspath(paths["candidate_weights_path"]),
            "candidate_weights_sha256": _sha256(paths["candidate_weights_path"]),
            "artifact_path": os.path.abspath(paths["artifact_path"]),
            "artifact_sha256": _sha256(paths["artifact_path"]),
            "candidate_history_path": os.path.abspath(paths["candidate_history_path"]),
            "candidate_history_sha256": _sha256(paths["candidate_history_path"]),
            "gate_history_path": os.path.abspath(paths["gate_history_path"]),
            "gate_history_sha256": _sha256(paths["gate_history_path"]),
            "validation_diagnostics_path": os.path.abspath(paths["validation_path"]),
            "validation_diagnostics_sha256": _sha256(paths["validation_path"]),
            "checkpoint_trace_path": os.path.abspath(paths["checkpoint_trace_path"]),
            "checkpoint_trace_sha256": _sha256(paths["checkpoint_trace_path"]),
            "tail_path": os.path.abspath(paths["tail_path"]),
            "tail_sha256": _sha256(paths["tail_path"]),
            "record_path": os.path.abspath(paths["record_path"]),
            "result_source": "new_time_freq_training",
        }
    )
    _atomic_write_json(row, paths["record_path"])
    del restored, source_model, model
    keras.backend.clear_session()
    return row


def build_t0_reference_rows(farm_ids):
    source_path = os.path.join(gate_train.RESULT_ROOT, gate_train.TRAINING_SUMMARY_NAME)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"缺少Stage-3训练summary: {source_path}")
    frame = pd.read_csv(source_path, dtype={"farm_id": str})
    frame = frame[
        (frame["variant_id"].astype(str) == "g0")
        & frame["farm_id"].astype(str).isin([str(item) for item in farm_ids])
    ].copy()
    if len(frame) != len(farm_ids) or frame["farm_id"].nunique() != len(farm_ids):
        raise ValueError("T0/G0训练引用未覆盖请求场站唯一集合")
    rows = []
    for _, source in frame.iterrows():
        model_path = source["source_model_path"]
        artifact_path = source["source_artifact_path"]
        if _sha256(model_path) != source["source_model_sha256"]:
            raise ValueError(f"T0来源模型hash漂移: {source['farm_id']}")
        if _sha256(artifact_path) != source["source_artifact_sha256"]:
            raise ValueError(f"T0来源artifact hash漂移: {source['farm_id']}")
        rows.append(
            {
                "model_family": MODEL_FAMILY,
                "variant_id": "t0",
                "variant_label": VARIANT_SPECS["t0"]["label"],
                "farm_id": str(source["farm_id"]),
                "feature_groups": SOURCE_FEATURE_GROUPS,
                "feature_count": SOURCE_FEATURE_COUNT,
                "reference_only": True,
                "requires_training": False,
                "random_seed": RANDOM_SEED,
                "parameter_count": int(source["parameter_count"]),
                "source_variant": "controlled_gate_g0/f7",
                "source_model_path": os.path.abspath(model_path),
                "source_model_sha256": source["source_model_sha256"],
                "source_artifact_path": os.path.abspath(artifact_path),
                "source_artifact_sha256": source["source_artifact_sha256"],
                "source_summary_path": os.path.abspath(source_path),
                "source_summary_sha256": _sha256(source_path),
                "result_source": "direct_reference_existing_stage3_g0_no_training",
                "selection_split": "test_in_prediction_script",
            }
        )
    return rows


def _validate_completed_record(record_path, variant_id, farm_id):
    if not os.path.isfile(record_path):
        return None
    with open(record_path, "r", encoding="utf-8") as file:
        row = json.load(file)
    if row.get("variant_id") != variant_id or str(row.get("farm_id")) != str(farm_id):
        raise ValueError(f"resume record身份不一致: {record_path}")
    for path_key, hash_key in (
        ("model_path", "model_sha256"),
        ("best_weights_path", "best_weights_sha256"),
        ("candidate_weights_path", "candidate_weights_sha256"),
        ("artifact_path", "artifact_sha256"),
    ):
        if _sha256(row.get(path_key)) != row.get(hash_key):
            raise ValueError(f"resume文件hash不一致: {path_key}")
    artifact = joblib.load(row["artifact_path"])
    if (
        artifact.get("protocol_version") != PROTOCOL_VERSION
        or artifact.get("architecture_version") != ARCHITECTURE_VERSION
        or artifact.get("model_sha256") != row.get("model_sha256")
    ):
        raise ValueError("resume artifact协议或模型身份不一致")
    current_code_sha256 = _sha256(__file__)
    if (
        row.get("training_code_sha256") != current_code_sha256
        or artifact.get("training_code_sha256") != current_code_sha256
    ):
        raise ValueError("resume记录由不同训练代码生成；请使用--force重训")
    for path_key, hash_key in (
        ("train_file", "train_file_sha256"),
        ("source_f7_model_path", "source_f7_model_sha256"),
        ("source_f7_artifact_path", "source_f7_artifact_sha256"),
        ("candidate_history_path", "candidate_history_sha256"),
        ("gate_history_path", "gate_history_sha256"),
        ("validation_diagnostics_path", "validation_diagnostics_sha256"),
        ("checkpoint_trace_path", "checkpoint_trace_sha256"),
        ("tail_path", "tail_sha256"),
    ):
        if _sha256(artifact.get(path_key)) != artifact.get(hash_key):
            raise ValueError(f"resume artifact依赖已漂移: {path_key}")
    current_dependencies = _dependency_code_records()
    recorded_dependencies = artifact.get("dependency_code", {})
    if set(recorded_dependencies) != set(current_dependencies):
        raise ValueError("resume artifact依赖代码集合不完整")
    for name, current in current_dependencies.items():
        recorded = recorded_dependencies[name]
        if (
            os.path.realpath(recorded.get("path", ""))
            != os.path.realpath(current["path"])
            or recorded.get("sha256") != current["sha256"]
        ):
            raise ValueError(f"resume artifact依赖代码已漂移: {name}")
    if artifact.get("candidate_snapshot_before_gate_sha256") != artifact.get(
        "candidate_snapshot_after_gate_sha256"
    ):
        raise ValueError("resume artifact未证明门控阶段candidate冻结")
    if artifact.get("candidate_output_before_gate_sha256") != artifact.get(
        "candidate_output_after_gate_sha256"
    ):
        raise ValueError("resume artifact的candidate门控前后输出hash不一致")
    return row


def write_manifest(result_root=RESULT_ROOT, run_scope="formal"):
    rows = []
    for order, (variant_id, spec) in enumerate(VARIANT_SPECS.items()):
        rows.append(
            {
                "variant_order": order,
                "variant_id": variant_id,
                "label": spec["label"],
                "requires_training": spec["requires_training"],
                "adapter": spec["adapter"],
                "gate": spec["gate"],
                "description": spec["description"],
                "source_candidate": "f7_persistence_plus_light_residual",
                "base_candidate_frozen": variant_id != "t0",
                "candidate_retrained": variant_id in ADAPTER_VARIANTS,
                "soft_oracle_recomputed_after_candidate_change": variant_id != "t0",
                "q90_scope": "train_only_per_farm_per_horizon",
                "frequency_input_scope": "history_only"
                if variant_id in {"t2", "t3"}
                else "not_applicable",
                "factorized_calibrated_safe_gate": variant_id != "t0",
                "parameter_limit_exclusive": PARAMETER_LIMIT,
                "expected_total_params": EXPECTED_TOTAL_PARAMS.get(variant_id),
                "expected_adapter_trainable_params": (
                    EXPECTED_ADAPTER_TRAINABLE_PARAMS.get(variant_id)
                ),
                "random_seed": RANDOM_SEED,
                "batch_size": BATCH_SIZE,
                "selection_split": "test",
                "test_used_for_selection": True,
                "test_is_final_blind_evaluation": False,
                "protocol_version": PROTOCOL_VERSION,
                "run_scope": run_scope,
            }
        )
    return _atomic_to_csv(pd.DataFrame(rows), os.path.join(result_root, MANIFEST_NAME))


def publish_training_marker(summary_path, manifest_path, summary):
    new_rows = summary[summary["variant_id"].isin(TRAINABLE_VARIANTS)]
    if len(new_rows) != len(TRAINABLE_VARIANTS) * len(expected_farm_ids()):
        raise ValueError("正式新训练矩阵不是4×5")
    files = {
        "training_summary": _file_record(summary_path),
        "experiment_manifest": _file_record(manifest_path),
        "training_code": _file_record(__file__),
        "source_feature_training_marker": _file_record(
            os.path.join(
                feature_train.RESULT_ROOT, feature_train.TRAINING_COMPLETION_NAME
            )
        ),
        "source_stage3_training_marker": _file_record(
            os.path.join(gate_train.RESULT_ROOT, gate_train.TRAINING_MARKER_NAME)
        ),
        "source_stage3_prediction_marker": _file_record(
            os.path.join(
                gate_train.RESULT_ROOT, gate_train.PREDICTION_MARKER_RELATIVE_PATH
            )
        ),
    }
    for name, record in _dependency_code_records().items():
        files[f"dependency_code.{name}"] = record
    for _, row in new_rows.iterrows():
        prefix = f"{row['variant_id']}.{row['farm_id']}"
        for key in (
            "model_path",
            "best_weights_path",
            "candidate_weights_path",
            "artifact_path",
            "candidate_history_path",
            "gate_history_path",
            "validation_diagnostics_path",
            "checkpoint_trace_path",
            "tail_path",
            "record_path",
        ):
            files[f"{prefix}.{key}"] = _file_record(row[key])
    marker = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "expected_farm_ids": list(expected_farm_ids()),
        "variants": list(VARIANT_SPECS),
        "new_training_variants": list(TRAINABLE_VARIANTS),
        "new_model_count": int(len(new_rows)),
        "t0_reused_model_count": 5,
        "t0_retraining_forbidden": True,
        "parameter_limit_exclusive": PARAMETER_LIMIT,
        "files": files,
    }
    return _atomic_write_json(marker, os.path.join(RESULT_ROOT, TRAINING_MARKER_NAME))


def _validate_marker_file_records(marker_path, expected_protocol, critical_keys):
    if not os.path.isfile(marker_path):
        raise FileNotFoundError(f"缺少上游complete marker: {marker_path}")
    with open(marker_path, "r", encoding="utf-8") as file:
        marker = json.load(file)
    if marker.get("status") != "complete":
        raise ValueError(f"上游marker不是complete: {marker_path}")
    if (
        expected_protocol is not None
        and marker.get("protocol_version") != expected_protocol
    ):
        raise ValueError(
            f"上游marker协议漂移: {marker.get('protocol_version')} != "
            f"{expected_protocol}: {marker_path}"
        )
    records = marker.get("files", {})
    for key in critical_keys:
        record = records.get(key)
        if not isinstance(record, dict):
            raise KeyError(f"上游marker缺少关键文件记录{key}: {marker_path}")
        path = record.get("path")
        if _sha256(path) != record.get("sha256"):
            raise ValueError(f"上游关键文件hash漂移{key}: {path}")
    return marker


def validate_required_source_bundles():
    """Fail before training if Stage-2/3 source identities have drifted."""
    feature_marker_path = os.path.join(
        feature_train.RESULT_ROOT, feature_train.TRAINING_COMPLETION_NAME
    )
    feature_marker = _validate_marker_file_records(
        feature_marker_path,
        expected_protocol=None,
        critical_keys=("extended_training_summary", "legacy_f0_f7_training_summary"),
    )
    stage3_training_path = os.path.join(
        gate_train.RESULT_ROOT, gate_train.TRAINING_MARKER_NAME
    )
    stage3_training = _validate_marker_file_records(
        stage3_training_path,
        expected_protocol=gate_train.PROTOCOL_VERSION,
        critical_keys=(
            "training_summary",
            "experiment_manifest",
            "source_feature_training_marker",
        ),
    )
    source_feature_record = stage3_training["files"]["source_feature_training_marker"]
    if os.path.realpath(source_feature_record["path"]) != os.path.realpath(
        feature_marker_path
    ) or source_feature_record["sha256"] != _sha256(feature_marker_path):
        raise ValueError("Stage-3训练marker未锁定当前Stage-2 marker")
    stage3_prediction_path = os.path.join(
        gate_train.RESULT_ROOT, gate_train.PREDICTION_MARKER_RELATIVE_PATH
    )
    stage3_prediction = _validate_marker_file_records(
        stage3_prediction_path,
        expected_protocol=gate_train.PROTOCOL_VERSION,
        critical_keys=(
            "training_marker",
            "stage2_source_marker",
            "formal.summary",
            "formal.candidate",
            "formal.final_selection",
        ),
    )
    training_record = stage3_prediction["files"]["training_marker"]
    if os.path.realpath(training_record["path"]) != os.path.realpath(
        stage3_training_path
    ) or training_record["sha256"] != _sha256(stage3_training_path):
        raise ValueError("Stage-3预测marker未锁定当前Stage-3训练marker")
    return {
        "feature_marker_sha256": _sha256(feature_marker_path),
        "stage3_training_marker_sha256": _sha256(stage3_training_path),
        "stage3_prediction_marker_sha256": _sha256(stage3_prediction_path),
        "feature_status": feature_marker.get("status"),
        "stage3_training_status": stage3_training.get("status"),
        "stage3_prediction_status": stage3_prediction.get("status"),
    }


def _discover_train_files(requested_farms=None):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "wind_train_*.csv")))
    if requested_farms:
        farms = {str(item) for item in requested_farms}
        files = [
            path
            for path in files
            if re.search(r"wind_train_(\d+)\.csv$", os.path.basename(path)).group(1)
            in farms
        ]
    return files


def _parse_csv(value):
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default=os.getenv("WIND_TIME_FREQ_VARIANTS", ",".join(VARIANT_SPECS)),
        help="逗号分隔: t0,m0,t1,t2,t3",
    )
    parser.add_argument(
        "--farms",
        default=os.getenv("WIND_TIME_FREQ_FARMS", ""),
        help="逗号分隔场站ID；空值表示全部正式场站",
    )
    parser.add_argument("--candidate-epochs", type=int, default=CANDIDATE_EPOCHS)
    parser.add_argument("--gate-only-epochs", type=int, default=GATE_ONLY_EPOCHS)
    parser.add_argument("--context-epochs", type=int, default=CONTEXT_EPOCHS)
    parser.add_argument(
        "--calibrated-gate-epochs", type=int, default=CALIBRATED_GATE_EPOCHS
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="调试快捷覆盖candidate与calibrated-gate epoch；自动进入partial",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="忽略可验证resume记录并重训"
    )
    return parser.parse_args(argv)


def _formal_protocol(args, variants, farm_ids):
    return (
        not args.smoke_test
        and args.epochs is None
        and set(variants) == set(VARIANT_SPECS)
        and set(farm_ids) == set(expected_farm_ids())
        and args.candidate_epochs == CANDIDATE_EPOCHS == 30
        and args.gate_only_epochs == GATE_ONLY_EPOCHS == 3
        and args.context_epochs == CONTEXT_EPOCHS == 5
        and args.calibrated_gate_epochs == CALIBRATED_GATE_EPOCHS == 30
        and BATCH_SIZE == 192
        and np.isclose(VALIDATION_SPLIT, 0.15, rtol=0.0, atol=1e-12)
        and np.isclose(CANDIDATE_LEARNING_RATE, 1e-4, rtol=0.0, atol=1e-12)
        and np.isclose(GATE_INITIAL_LR, 1e-4, rtol=0.0, atol=1e-12)
        and np.isclose(GATE_CALIBRATED_LR, 5e-5, rtol=0.0, atol=1e-12)
        and EARLY_STOPPING_PATIENCE == 6
    )


def _validate_runtime_configuration(args, variants):
    if BATCH_SIZE <= 0:
        raise ValueError("batch_size必须为正整数")
    if not 0.0 < VALIDATION_SPLIT < 1.0:
        raise ValueError("validation_split必须位于(0,1)")
    learning_rates = {
        "candidate": CANDIDATE_LEARNING_RATE,
        "gate_initial": GATE_INITIAL_LR,
        "gate_calibrated": GATE_CALIBRATED_LR,
    }
    invalid_rates = {
        name: value for name, value in learning_rates.items() if value <= 0
    }
    if invalid_rates:
        raise ValueError(f"学习率必须为正: {invalid_rates}")
    epochs = {
        "candidate": int(args.candidate_epochs),
        "gate_only": int(args.gate_only_epochs),
        "context": int(args.context_epochs),
        "calibrated_gate": int(args.calibrated_gate_epochs),
    }
    if any(value < 0 for value in epochs.values()):
        raise ValueError(f"epoch不得为负: {epochs}")
    if set(variants).intersection(ADAPTER_VARIANTS) and epochs["candidate"] <= 0:
        raise ValueError("T1--T3必须至少训练1个candidate adapter epoch")
    if (
        set(variants).intersection(TRAINABLE_VARIANTS)
        and sum(epochs[name] for name in ("gate_only", "context", "calibrated_gate"))
        <= 0
    ):
        raise ValueError("可训练变体必须至少运行1个门控epoch")


def main(argv=None):
    args = parse_args(argv)
    configure_reproducibility()
    source_bundle_identity = validate_required_source_bundles()
    variants = list(dict.fromkeys(_parse_csv(args.variants)))
    invalid = sorted(set(variants) - set(VARIANT_SPECS))
    if invalid:
        raise ValueError(f"未知变体{invalid}; 可选{list(VARIANT_SPECS)}")
    farms = _parse_csv(args.farms) if args.farms else []
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs必须为正")
        args.candidate_epochs = args.epochs
        args.calibrated_gate_epochs = args.epochs
    if args.smoke_test:
        # 默认覆盖最复杂的时间+rFFT+交互分支，而不是只测无adapter的M0。
        variants = [
            "t3"
            if "t3" in variants
            else next((item for item in variants if item in TRAINABLE_VARIANTS), "t3")
        ]
        if not farms:
            farms = [expected_farm_ids()[0]]
        else:
            farms = farms[:1]
        args.candidate_epochs = 1
        args.gate_only_epochs = 1
        args.context_epochs = 1
        args.calibrated_gate_epochs = 1
    _validate_runtime_configuration(args, variants)
    train_files = _discover_train_files(farms)
    if not train_files:
        raise FileNotFoundError("没有匹配的训练文件")
    farm_ids = [regime_train.get_farm_id(path) for path in train_files]
    formal = _formal_protocol(args, variants, farm_ids)
    if formal:
        run_root = RESULT_ROOT
        run_scope = "formal"
        marker_path = os.path.join(RESULT_ROOT, TRAINING_MARKER_NAME)
        downstream_path = os.path.join(RESULT_ROOT, PREDICTION_MARKER_RELATIVE_PATH)
        for path in (marker_path, downstream_path):
            if os.path.exists(path):
                os.remove(path)
    else:
        tag = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_root = os.path.join(RESULT_ROOT, "partial_runs", tag)
        run_scope = "smoke_or_partial_or_protocol_override"
    manifest_path = write_manifest(run_root, run_scope)
    print(
        f"第四阶段场站={farm_ids}; 变体={variants}; 输出={run_root}; "
        f"formal={formal}; seed={RANDOM_SEED}; batch={BATCH_SIZE}"
    )
    print(f"上游bundle已验证: {source_bundle_identity}")
    rows = []
    if "t0" in variants:
        rows.extend(build_t0_reference_rows(farm_ids))
    trainable = [item for item in variants if item in TRAINABLE_VARIANTS]
    for train_file in train_files:
        prepared = regime_train._prepare_farm(train_file)
        for variant_id in trainable:
            dirs = variant_dirs(variant_id, result_root=run_root)
            record_path = _train_paths(dirs, variant_id, str(prepared["farm_id"]))[
                "record_path"
            ]
            completed = (
                None
                if args.force
                else _validate_completed_record(
                    record_path, variant_id, prepared["farm_id"]
                )
            )
            if completed is not None:
                print(f"跳过已验证完成模型: {variant_id}/{prepared['farm_id']}")
                rows.append(completed)
                continue
            print(
                f"\n===== {VARIANT_SPECS[variant_id]['label']} / "
                f"farm={prepared['farm_id']} ====="
            )
            rows.append(
                train_variant_for_farm(
                    variant_id,
                    prepared,
                    result_root=run_root,
                    candidate_epochs=args.candidate_epochs,
                    gate_only_epochs=args.gate_only_epochs,
                    context_epochs=args.context_epochs,
                    calibrated_gate_epochs=args.calibrated_gate_epochs,
                )
            )
    summary = pd.DataFrame(rows)
    if summary.empty or summary.duplicated(["variant_id", "farm_id"]).any():
        raise ValueError("训练/引用summary为空或存在重复键")
    summary_path = _atomic_to_csv(
        summary, os.path.join(run_root, TRAINING_SUMMARY_NAME)
    )
    print(f"训练汇总: {summary_path}")
    if formal:
        expected_rows = len(VARIANT_SPECS) * len(expected_farm_ids())
        if len(summary) != expected_rows:
            raise ValueError(f"正式summary应为{expected_rows}行，实际{len(summary)}")
        marker = publish_training_marker(summary_path, manifest_path, summary)
        print(f"正式训练bundle完成: {marker}")
        print("T0只读引用Stage-3 G0/F7，新增训练模型数=20")
    else:
        print("partial/smoke运行不覆盖正式summary，不发布complete marker")


if __name__ == "__main__":
    main()
