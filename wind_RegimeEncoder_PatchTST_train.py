"""RegimeEncoder-PatchTST 第二阶段显式工况编码实验训练入口。

第一阶段已经确认 B2（persistence + lightweight causal residual）是当前最小
有效主干。本阶段不再扩展大模型，而是把 B2 拆成两个完整预测候选：

    persistence candidate
    corrected candidate = persistence + causal residual

并用逐样本、逐 horizon 的门控做凸融合。实验矩阵固定为：

    R0  Stage-1 B0 persistence 冻结引用，不重复计算
    R1  Stage-1 B2 persistence residual 冻结引用，不重复训练
    R2  B2 两候选 + 逐 horizon 静态门控
    R3  B2 两候选 + 隐式 causal-Conv 上下文动态门控
    R4  B2 两候选 + 显式风电工况编码器动态门控
    R5  R4 + 训练期工况辅助任务
    R6  Stage-1 B6 全模型冻结引用，不重复训练

R2--R5 都从同一场站的 Stage-1 B2 最佳权重初始化，并给 corrected candidate
增加统一的直接监督，以缓解 ``gate * residual`` 的尺度不可辨识问题。R5 的
未来工况标签只作为 fit target；模型输入始终只有 96 步历史特征，推理不需要
任何未来标签。

默认固定 seed=2026、batch_size=192。可用 ``WIND_REGIME_VARIANTS`` 和
``WIND_REGIME_FARMS`` 做子集实验；子集只写 partial 汇总，不覆盖完整矩阵结果。
"""

import glob
import os
import re
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from wind_FeTS_PatchTST_min_train import (
    ARCHITECTURE_VERSION as STAGE1_ARCHITECTURE_VERSION,
    RESULT_ROOT as STAGE1_RESULT_ROOT,
    get_min_custom_objects,
    variant_dirs as stage1_variant_dirs,
    variant_model_name as stage1_variant_model_name,
)
from wind_FeTS_PatchTST_train import (
    CORRECTION_KERNEL_L2,
    NonFiniteTrainingGuard,
    PersistenceForecast,
    TakeLastToken,
    compute_power_scale_alignment,
    ensure_finite_training_history,
    validate_preprocessed_data,
)
from wind_dl_model_train import (
    DATA_DIR,
    FORECAST_LEN,
    HEAD_DROPOUT,
    HISTORY_LEN,
    TARGET_COL,
    TIME_FREQ,
    WIND_SPEED_COLS,
    build_scaled_arrays,
    load_and_preprocess,
    make_window_dataset,
    set_global_seed,
)

warnings.filterwarnings("ignore")


MODEL_FAMILY = "regime_encoder_patchtst"
ARCHITECTURE_VERSION = "regime_encoder_patchtst_stage2_v1"
ARTIFACT_SCHEMA_VERSION = 1
REGIME_LABEL_VERSION = "capacity_fixed_regime_v1"
EVALUATION_PIPELINE_VERSION = "stage1_legacy_v1"
RESULT_ROOT = os.path.join("./wind_results", MODEL_FAMILY)
TRAIN_FILE_PATTERN = "wind_train_*.csv"
RANDOM_SEED = 2026

# Stage-1 的实际完整实验使用 192；这里直接把 192 作为默认值，避免重新落回
# wind_dl_model_train.py 中曾经导致显存不足的 256。
BATCH_SIZE = int(os.getenv("WIND_REGIME_BATCH_SIZE", "192"))
EPOCHS = int(os.getenv("WIND_REGIME_EPOCHS", "60"))
VALIDATION_SPLIT = float(os.getenv("WIND_REGIME_VALIDATION_SPLIT", "0.15"))
LEARNING_RATE = float(os.getenv("WIND_REGIME_LEARNING_RATE", "0.0001"))

GATE_HIDDEN_DIM = int(os.getenv("WIND_REGIME_GATE_HIDDEN_DIM", "16"))
HORIZON_EMBEDDING_DIM = int(
    os.getenv("WIND_REGIME_HORIZON_EMBEDDING_DIM", "8")
)
REGIME_CONTEXT_DIM = int(os.getenv("WIND_REGIME_CONTEXT_DIM", "24"))
GATE_DROPOUT = float(os.getenv("WIND_REGIME_GATE_DROPOUT", "0.10"))
GATE_INITIAL_CORRECTED_WEIGHT = float(
    os.getenv("WIND_REGIME_GATE_INITIAL_CORRECTED_WEIGHT", "0.95")
)
CANDIDATE_LOSS_WEIGHT = float(
    os.getenv("WIND_REGIME_CANDIDATE_LOSS_WEIGHT", "0.50")
)
AUX_CLASS_LOSS_WEIGHT = float(
    os.getenv("WIND_REGIME_AUX_CLASS_LOSS_WEIGHT", "0.10")
)
AUX_LOW_POWER_LOSS_WEIGHT = float(
    os.getenv("WIND_REGIME_AUX_LOW_POWER_LOSS_WEIGHT", "0.05")
)
AUX_MAGNITUDE_LOSS_WEIGHT = float(
    os.getenv("WIND_REGIME_AUX_MAGNITUDE_LOSS_WEIGHT", "0.05")
)
IDEAL_PARAMETER_LIMIT = int(os.getenv("WIND_REGIME_IDEAL_PARAMS", "50000"))
HARD_PARAMETER_LIMIT = int(os.getenv("WIND_REGIME_MAX_PARAMS", "100000"))

STABLE_CHANGE_THRESHOLD = 0.02
LOW_POWER_THRESHOLD = 0.02
CHANGE_BAND_EDGES = (0.02, 0.05, 0.10, 0.20)
REGIME_WINDOWS = (4, 8, 16, 32)
WIND_SPEED_NORMALIZER = 25.0

REFERENCE_SOURCE_VARIANTS = {
    "r0_persistence_reference": "b0_persistence",
    "r1_b2_reference": "b2_persistence_residual",
    "r6_b6_reference": "b6_all_dynamic",
}

VARIANT_SPECS = {
    "r0_persistence_reference": {
        "label": "R0 Stage-1 B0 Persistence reference",
        "requires_training": False,
        "model_kind": "frozen_stage1_reference",
        "source_variant": "b0_persistence",
        "gate_type": "none",
        "encoder_type": "none",
        "auxiliary_tasks": False,
        "description": "直接引用第一阶段 B0；不生成新模型",
    },
    "r1_b2_reference": {
        "label": "R1 Stage-1 B2 backbone reference",
        "requires_training": False,
        "model_kind": "frozen_stage1_reference",
        "source_variant": "b2_persistence_residual",
        "gate_type": "none",
        "encoder_type": "b2_causal_residual",
        "auxiliary_tasks": False,
        "description": "直接引用第一阶段最小有效 B2；不重复训练",
    },
    "r2_horizon_gate": {
        "label": "R2 B2 + horizon-only two-candidate gate",
        "requires_training": True,
        "model_kind": "keras_network",
        "source_variant": "b2_persistence_residual",
        "gate_type": "static_horizon_sigmoid",
        "encoder_type": "horizon_only",
        "auxiliary_tasks": False,
        "description": "控制组：门控只随 horizon 变化，不读取样本工况",
    },
    "r3_implicit_conv_gate": {
        "label": "R3 B2 + implicit Conv regime gate",
        "requires_training": True,
        "model_kind": "keras_network",
        "source_variant": "b2_persistence_residual",
        "gate_type": "sample_horizon_sigmoid",
        "encoder_type": "implicit_causal_conv_context",
        "auxiliary_tasks": False,
        "description": "用 B2 隐式卷积表示产生逐样本逐 horizon 门控",
    },
    "r4_explicit_regime_gate": {
        "label": "R4 B2 + explicit wind regime encoder",
        "requires_training": True,
        "model_kind": "keras_network",
        "source_variant": "b2_persistence_residual",
        "gate_type": "sample_horizon_sigmoid",
        "encoder_type": "explicit_wind_regime_statistics",
        "auxiliary_tasks": False,
        "description": "历史功率/风速/风向/一致性显式统计驱动动态门控",
    },
    "r5_explicit_regime_aux": {
        "label": "R5 Explicit regime encoder + auxiliary tasks",
        "requires_training": True,
        "model_kind": "keras_network_multi_output",
        "source_variant": "b2_persistence_residual",
        "gate_type": "sample_horizon_sigmoid",
        "encoder_type": "explicit_wind_regime_statistics",
        "auxiliary_tasks": True,
        "description": "R4 加 stable/up/down、低功率和变化幅度辅助任务",
    },
    "r6_b6_reference": {
        "label": "R6 Stage-1 B6 full-model reference",
        "requires_training": False,
        "model_kind": "frozen_stage1_reference",
        "source_variant": "b6_all_dynamic",
        "gate_type": "source_four_expert_softmax",
        "encoder_type": "stage1_full_multiscale",
        "auxiliary_tasks": False,
        "description": "直接引用第一阶段 B6 结果，避免重复大模型实验",
    },
}

TRAINABLE_VARIANTS = tuple(
    variant_id
    for variant_id, spec in VARIANT_SPECS.items()
    if spec["requires_training"]
)

B2_WEIGHTED_LAYER_NAMES = (
    "residual_causal_conv_1",
    "residual_causal_conv_2",
    "residual_hidden",
    "persistence_residual",
)


def _validate_configuration():
    if BATCH_SIZE <= 0 or EPOCHS <= 0:
        raise ValueError("batch_size 和 epochs 必须为正整数")
    if not 0 < VALIDATION_SPLIT < 1:
        raise ValueError("validation_split 必须位于 (0, 1)")
    if LEARNING_RATE <= 0:
        raise ValueError("learning_rate 必须为正数")
    if not 0 < GATE_INITIAL_CORRECTED_WEIGHT < 1:
        raise ValueError("初始 corrected gate 权重必须位于 (0, 1)")
    for name, value in {
        "candidate_loss_weight": CANDIDATE_LOSS_WEIGHT,
        "aux_class_loss_weight": AUX_CLASS_LOSS_WEIGHT,
        "aux_low_power_loss_weight": AUX_LOW_POWER_LOSS_WEIGHT,
        "aux_magnitude_loss_weight": AUX_MAGNITUDE_LOSS_WEIGHT,
    }.items():
        if value < 0:
            raise ValueError(f"{name} 不能为负数")
    if HARD_PARAMETER_LIMIT < IDEAL_PARAMETER_LIMIT:
        raise ValueError("硬参数上限不能小于理想参数上限")


def configure_reproducibility():
    set_global_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def variant_model_name(variant_id):
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"未知第二阶段变体: {variant_id}")
    return f"{MODEL_FAMILY}_{variant_id}"


def variant_dirs(variant_id, create=True):
    root = os.path.join(RESULT_ROOT, variant_id)
    paths = {
        "root": root,
        "models": os.path.join(root, "models"),
        "weights": os.path.join(root, "weights"),
        "preprocess": os.path.join(root, "preprocess"),
        "history": os.path.join(root, "history"),
        "tensorboard": os.path.join(root, "tensorboard"),
        "tails": os.path.join(root, "tails"),
        "validation_diagnostics": os.path.join(
            root,
            "validation_diagnostics",
        ),
    }
    if create:
        os.makedirs(RESULT_ROOT, exist_ok=True)
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
    return paths


def get_requested_variants():
    raw = os.getenv("WIND_REGIME_VARIANTS")
    if not raw:
        return list(VARIANT_SPECS)
    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if any(item in {"all", "*"} for item in requested):
        return list(VARIANT_SPECS)
    invalid = sorted(set(requested) - set(VARIANT_SPECS))
    if invalid:
        raise ValueError(f"未知变体 {invalid}；可选: {list(VARIANT_SPECS)}")
    return list(dict.fromkeys(requested))


def get_farm_id(path):
    match = re.search(r"wind_train_(\d+)\.csv$", os.path.basename(path))
    if match:
        return match.group(1)
    return os.path.splitext(os.path.basename(path))[0]


def discover_train_files(data_dir=DATA_DIR):
    files = sorted(glob.glob(os.path.join(data_dir, TRAIN_FILE_PATTERN)))
    requested = os.getenv("WIND_REGIME_FARMS")
    if not requested:
        return files
    farm_ids = {item.strip() for item in requested.split(",") if item.strip()}
    return [path for path in files if get_farm_id(path) in farm_ids]


def explicit_regime_feature_names(windows=REGIME_WINDOWS):
    names = ["power_last", "power_mean_4", "power_mean_16", "power_mean_32"]
    for window in windows:
        names.extend(
            [
                f"power_slope_{window}",
                f"power_std_{window}",
                f"power_mean_abs_step_{window}",
            ]
        )
    names.extend(
        [
            "power_range_16",
            "power_range_32",
            "power_low_fraction_16",
            "power_low_fraction_32",
            "hub_wind_last",
            "hub_wind_mean_4",
            "hub_wind_mean_16",
        ]
    )
    for window in windows:
        names.extend([f"hub_wind_slope_{window}", f"hub_wind_std_{window}"])
    names.extend(
        [
            "hub_wind_mean_abs_step_16",
            "all_height_wind_last_mean",
            "all_height_wind_last_std",
            "hub_minus_height_mean",
            "direction_turn_lag_1",
            "direction_turn_lag_4",
            "direction_turn_lag_16",
            "direction_mean_turn_16",
            "power_wind_slope_product_8",
            "power_wind_slope_product_16",
            "power_minus_wind_cube_proxy",
            "power_wind_change_correlation_16",
        ]
    )
    return names


@keras.utils.register_keras_serializable(package="WindRegimeEncoderPatchTST")
class ExplicitWindRegimeFeatures(layers.Layer):
    """从历史窗口提取有物理含义且固定维数的风电工况统计量。"""

    def __init__(
        self,
        target_channel_index,
        power_mean,
        power_scale,
        capacity,
        wind_speed_indices,
        wind_speed_means,
        wind_speed_scales,
        hub_wind_position,
        direction_sin_index=None,
        direction_cos_index=None,
        direction_sin_mean=0.0,
        direction_sin_scale=1.0,
        direction_cos_mean=0.0,
        direction_cos_scale=1.0,
        windows=REGIME_WINDOWS,
        low_power_threshold=LOW_POWER_THRESHOLD,
        wind_speed_normalizer=WIND_SPEED_NORMALIZER,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if target_channel_index < 0:
            raise ValueError("target_channel_index 不能为负")
        if not np.isfinite(power_scale) or power_scale <= 0:
            raise ValueError("power_scale 必须为有限正数")
        if not np.isfinite(capacity) or capacity <= 0:
            raise ValueError("capacity 必须为有限正数")
        if not wind_speed_indices:
            raise ValueError("显式工况编码器至少需要一个历史风速通道")
        if not (
            len(wind_speed_indices)
            == len(wind_speed_means)
            == len(wind_speed_scales)
        ):
            raise ValueError("风速索引、均值和尺度长度不一致")
        if any(float(scale) <= 0 for scale in wind_speed_scales):
            raise ValueError("风速标准化尺度必须为正")
        if not 0 <= int(hub_wind_position) < len(wind_speed_indices):
            raise ValueError("hub_wind_position 越界")
        if sorted(int(value) for value in windows) != list(windows):
            raise ValueError("windows 必须是递增正整数")
        if min(windows) < 2 or max(windows) > HISTORY_LEN:
            raise ValueError("工况统计窗口超出历史长度")

        self.target_channel_index = int(target_channel_index)
        self.power_mean = float(power_mean)
        self.power_scale = float(power_scale)
        self.capacity = float(capacity)
        self.wind_speed_indices = tuple(int(value) for value in wind_speed_indices)
        self.wind_speed_means = tuple(float(value) for value in wind_speed_means)
        self.wind_speed_scales = tuple(float(value) for value in wind_speed_scales)
        self.hub_wind_position = int(hub_wind_position)
        self.direction_sin_index = (
            None if direction_sin_index is None else int(direction_sin_index)
        )
        self.direction_cos_index = (
            None if direction_cos_index is None else int(direction_cos_index)
        )
        self.direction_sin_mean = float(direction_sin_mean)
        self.direction_sin_scale = float(direction_sin_scale)
        self.direction_cos_mean = float(direction_cos_mean)
        self.direction_cos_scale = float(direction_cos_scale)
        self.windows = tuple(int(value) for value in windows)
        self.low_power_threshold = float(low_power_threshold)
        self.wind_speed_normalizer = float(wind_speed_normalizer)
        self.feature_names = tuple(explicit_regime_feature_names(self.windows))

    @staticmethod
    def _mean_abs_step(series):
        return tf.reduce_mean(tf.abs(series[:, 1:] - series[:, :-1]), axis=1)

    @staticmethod
    def _range(series):
        return tf.reduce_max(series, axis=1) - tf.reduce_min(series, axis=1)

    @staticmethod
    def _safe_correlation(first, second):
        first_centered = first - tf.reduce_mean(first, axis=1, keepdims=True)
        second_centered = second - tf.reduce_mean(second, axis=1, keepdims=True)
        numerator = tf.reduce_mean(first_centered * second_centered, axis=1)
        denominator = tf.sqrt(
            tf.reduce_mean(tf.square(first_centered), axis=1)
            * tf.reduce_mean(tf.square(second_centered), axis=1)
            + tf.cast(1e-8, first.dtype)
        )
        return numerator / denominator

    def _physical_channel(self, inputs, index, mean, scale):
        return (
            inputs[:, :, index] * tf.cast(scale, inputs.dtype)
            + tf.cast(mean, inputs.dtype)
        )

    def call(self, inputs):
        power = self._physical_channel(
            inputs,
            self.target_channel_index,
            self.power_mean,
            self.power_scale,
        )
        power = power / tf.cast(self.capacity, inputs.dtype)

        speed_series = []
        for index, mean, scale in zip(
            self.wind_speed_indices,
            self.wind_speed_means,
            self.wind_speed_scales,
        ):
            speed = self._physical_channel(inputs, index, mean, scale)
            speed_series.append(
                speed / tf.cast(self.wind_speed_normalizer, inputs.dtype)
            )
        all_speed = tf.stack(speed_series, axis=-1)
        hub_speed = all_speed[:, :, self.hub_wind_position]

        values = [
            power[:, -1],
            tf.reduce_mean(power[:, -4:], axis=1),
            tf.reduce_mean(power[:, -16:], axis=1),
            tf.reduce_mean(power[:, -32:], axis=1),
        ]
        for window in self.windows:
            segment = power[:, -window:]
            values.extend(
                [
                    segment[:, -1] - segment[:, 0],
                    tf.math.reduce_std(segment, axis=1),
                    self._mean_abs_step(segment),
                ]
            )
        values.extend(
            [
                self._range(power[:, -16:]),
                self._range(power[:, -32:]),
                tf.reduce_mean(
                    tf.cast(
                        power[:, -16:] <= self.low_power_threshold,
                        inputs.dtype,
                    ),
                    axis=1,
                ),
                tf.reduce_mean(
                    tf.cast(
                        power[:, -32:] <= self.low_power_threshold,
                        inputs.dtype,
                    ),
                    axis=1,
                ),
                hub_speed[:, -1],
                tf.reduce_mean(hub_speed[:, -4:], axis=1),
                tf.reduce_mean(hub_speed[:, -16:], axis=1),
            ]
        )
        for window in self.windows:
            segment = hub_speed[:, -window:]
            values.extend(
                [
                    segment[:, -1] - segment[:, 0],
                    tf.math.reduce_std(segment, axis=1),
                ]
            )
        height_last = all_speed[:, -1, :]
        height_mean = tf.reduce_mean(height_last, axis=1)
        values.extend(
            [
                self._mean_abs_step(hub_speed[:, -16:]),
                height_mean,
                tf.math.reduce_std(height_last, axis=1),
                hub_speed[:, -1] - height_mean,
            ]
        )

        if self.direction_sin_index is not None and self.direction_cos_index is not None:
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
            norm = tf.sqrt(
                tf.square(direction_sin) + tf.square(direction_cos) + 1e-8
            )
            direction_sin = direction_sin / norm
            direction_cos = direction_cos / norm

            def _turn(lag):
                dot = (
                    direction_sin[:, -1] * direction_sin[:, -1 - lag]
                    + direction_cos[:, -1] * direction_cos[:, -1 - lag]
                )
                return 0.5 * (1.0 - tf.clip_by_value(dot, -1.0, 1.0))

            recent_sin = direction_sin[:, -16:]
            recent_cos = direction_cos[:, -16:]
            consecutive_dot = (
                recent_sin[:, 1:] * recent_sin[:, :-1]
                + recent_cos[:, 1:] * recent_cos[:, :-1]
            )
            mean_turn = tf.reduce_mean(
                0.5 * (1.0 - tf.clip_by_value(consecutive_dot, -1.0, 1.0)),
                axis=1,
            )
            values.extend([_turn(1), _turn(4), _turn(16), mean_turn])
        else:
            zeros = tf.zeros_like(power[:, -1])
            values.extend([zeros, zeros, zeros, zeros])

        power_slope_8 = power[:, -1] - power[:, -8]
        power_slope_16 = power[:, -1] - power[:, -16]
        wind_slope_8 = hub_speed[:, -1] - hub_speed[:, -8]
        wind_slope_16 = hub_speed[:, -1] - hub_speed[:, -16]
        wind_cube_proxy = tf.pow(tf.clip_by_value(hub_speed[:, -1], 0.0, 1.5), 3)
        power_change = power[:, -15:] - power[:, -16:-1]
        wind_cube_recent = tf.pow(
            tf.clip_by_value(hub_speed[:, -16:], 0.0, 1.5),
            3,
        )
        wind_change = wind_cube_recent[:, 1:] - wind_cube_recent[:, :-1]
        values.extend(
            [
                power_slope_8 * wind_slope_8,
                power_slope_16 * wind_slope_16,
                power[:, -1] - wind_cube_proxy,
                self._safe_correlation(power_change, wind_change),
            ]
        )

        features = tf.stack(values, axis=-1)
        features = tf.clip_by_value(features, -5.0, 5.0)
        return tf.where(tf.math.is_finite(features), features, tf.zeros_like(features))

    def compute_output_shape(self, input_shape):
        return input_shape[0], len(self.feature_names)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "target_channel_index": self.target_channel_index,
                "power_mean": self.power_mean,
                "power_scale": self.power_scale,
                "capacity": self.capacity,
                "wind_speed_indices": list(self.wind_speed_indices),
                "wind_speed_means": list(self.wind_speed_means),
                "wind_speed_scales": list(self.wind_speed_scales),
                "hub_wind_position": self.hub_wind_position,
                "direction_sin_index": self.direction_sin_index,
                "direction_cos_index": self.direction_cos_index,
                "direction_sin_mean": self.direction_sin_mean,
                "direction_sin_scale": self.direction_sin_scale,
                "direction_cos_mean": self.direction_cos_mean,
                "direction_cos_scale": self.direction_cos_scale,
                "windows": list(self.windows),
                "low_power_threshold": self.low_power_threshold,
                "wind_speed_normalizer": self.wind_speed_normalizer,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="WindRegimeEncoderPatchTST")
class HorizonOnlyCorrectionGate(layers.Layer):
    """不读取样本状态、仅按预测步学习 corrected candidate 权重。"""

    def __init__(self, forecast_len, initial_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        if forecast_len <= 0:
            raise ValueError("forecast_len 必须为正")
        if not 0 < initial_weight < 1:
            raise ValueError("initial_weight 必须位于 (0, 1)")
        self.forecast_len = int(forecast_len)
        self.initial_weight = float(initial_weight)

    def build(self, input_shape):
        logit = np.log(self.initial_weight / (1.0 - self.initial_weight))
        self.horizon_logits = self.add_weight(
            name="horizon_logits",
            shape=(self.forecast_len,),
            initializer=keras.initializers.Constant(logit),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        gate = tf.nn.sigmoid(self.horizon_logits)
        return tf.broadcast_to(
            gate[tf.newaxis, :],
            [tf.shape(inputs)[0], self.forecast_len],
        )

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.forecast_len

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "forecast_len": self.forecast_len,
                "initial_weight": self.initial_weight,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="WindRegimeEncoderPatchTST")
class SampleHorizonCorrectionGate(layers.Layer):
    """由历史上下文和 horizon embedding 生成逐样本 sigmoid 门控。"""

    def __init__(
        self,
        forecast_len,
        hidden_dim=16,
        horizon_embedding_dim=8,
        dropout=0.1,
        initial_weight=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if forecast_len <= 0 or hidden_dim <= 0 or horizon_embedding_dim <= 0:
            raise ValueError("门控维度必须为正")
        if not 0 <= dropout < 1:
            raise ValueError("dropout 必须位于 [0, 1)")
        if not 0 < initial_weight < 1:
            raise ValueError("initial_weight 必须位于 (0, 1)")
        self.forecast_len = int(forecast_len)
        self.hidden_dim = int(hidden_dim)
        self.horizon_embedding_dim = int(horizon_embedding_dim)
        self.dropout_rate = float(dropout)
        self.initial_weight = float(initial_weight)

        self.context_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name="context_norm",
        )
        self.context_projection = layers.Dense(
            self.hidden_dim,
            activation="gelu",
            name="context_projection",
        )
        self.context_dropout = layers.Dropout(
            self.dropout_rate,
            name="context_dropout",
        )
        self.horizon_embedding = layers.Embedding(
            self.forecast_len,
            self.horizon_embedding_dim,
            name="horizon_embedding",
        )
        self.gate_hidden = layers.Dense(
            self.hidden_dim,
            activation="gelu",
            name="gate_hidden",
        )
        self.gate_dropout = layers.Dropout(
            self.dropout_rate,
            name="gate_dropout",
        )
        initial_logit = np.log(self.initial_weight / (1.0 - self.initial_weight))
        self.gate_logit = layers.Dense(
            1,
            kernel_initializer="zeros",
            bias_initializer=keras.initializers.Constant(initial_logit),
            name="gate_logit",
        )

    def call(self, inputs, training=None):
        context = self.context_norm(inputs)
        context = self.context_projection(context)
        context = self.context_dropout(context, training=training)
        context = tf.repeat(
            context[:, tf.newaxis, :],
            repeats=self.forecast_len,
            axis=1,
        )
        horizon_ids = tf.range(self.forecast_len)
        horizon = self.horizon_embedding(horizon_ids)
        horizon = tf.broadcast_to(
            horizon[tf.newaxis, :, :],
            [
                tf.shape(context)[0],
                self.forecast_len,
                self.horizon_embedding_dim,
            ],
        )
        hidden = self.gate_hidden(tf.concat([context, horizon], axis=-1))
        hidden = self.gate_dropout(hidden, training=training)
        return tf.nn.sigmoid(self.gate_logit(hidden)[..., 0])

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.forecast_len

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "forecast_len": self.forecast_len,
                "hidden_dim": self.hidden_dim,
                "horizon_embedding_dim": self.horizon_embedding_dim,
                "dropout": self.dropout_rate,
                "initial_weight": self.initial_weight,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="WindRegimeEncoderPatchTST")
class TwoCandidateGateFusion(layers.Layer):
    """计算 persistence + gate * (corrected - persistence)。"""

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
            raise ValueError("TwoCandidateGateFusion 需要三个输入")
        persistence, corrected, gate = inputs
        gate = tf.clip_by_value(gate, 0.0, 1.0)
        return persistence + gate * (corrected - persistence)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def _regime_feature_config(input_cols, scaler_x, capacity, target_index):
    if capacity is None or not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("显式工况编码实验需要有效装机容量")
    means = np.asarray(scaler_x.mean_, dtype=float)
    scales = np.asarray(scaler_x.scale_, dtype=float)
    speed_names = [name for name in WIND_SPEED_COLS if name in input_cols]
    if not speed_names:
        raise ValueError("input_cols 中没有原始高度风速通道")
    speed_indices = [input_cols.index(name) for name in speed_names]
    hub_name = "轮毂高度风速" if "轮毂高度风速" in speed_names else speed_names[-1]
    hub_position = speed_names.index(hub_name)

    sin_name = "轮毂高度风向_sin"
    cos_name = "轮毂高度风向_cos"
    sin_index = input_cols.index(sin_name) if sin_name in input_cols else None
    cos_index = input_cols.index(cos_name) if cos_name in input_cols else None
    return {
        "target_channel_index": int(target_index),
        "power_mean": float(means[target_index]),
        "power_scale": float(scales[target_index]),
        "capacity": float(capacity),
        "wind_speed_indices": speed_indices,
        "wind_speed_names": speed_names,
        "wind_speed_means": [float(means[index]) for index in speed_indices],
        "wind_speed_scales": [float(scales[index]) for index in speed_indices],
        "hub_wind_position": int(hub_position),
        "direction_sin_index": sin_index,
        "direction_cos_index": cos_index,
        "direction_sin_mean": float(means[sin_index]) if sin_index is not None else 0.0,
        "direction_sin_scale": (
            float(scales[sin_index]) if sin_index is not None else 1.0
        ),
        "direction_cos_mean": float(means[cos_index]) if cos_index is not None else 0.0,
        "direction_cos_scale": (
            float(scales[cos_index]) if cos_index is not None else 1.0
        ),
        "windows": list(REGIME_WINDOWS),
        "low_power_threshold": LOW_POWER_THRESHOLD,
        "wind_speed_normalizer": WIND_SPEED_NORMALIZER,
        "feature_names": explicit_regime_feature_names(),
    }


def _layer_feature_kwargs(config):
    allowed = {
        "target_channel_index",
        "power_mean",
        "power_scale",
        "capacity",
        "wind_speed_indices",
        "wind_speed_means",
        "wind_speed_scales",
        "hub_wind_position",
        "direction_sin_index",
        "direction_cos_index",
        "direction_sin_mean",
        "direction_sin_scale",
        "direction_cos_mean",
        "direction_cos_scale",
        "windows",
        "low_power_threshold",
        "wind_speed_normalizer",
    }
    return {key: value for key, value in config.items() if key in allowed}


def _build_b2_candidates(
    inputs,
    target_channel_index,
    power_scale_ratio,
    power_scale_offset,
):
    set_global_seed(RANDOM_SEED)
    persistence = PersistenceForecast(
        target_channel_index=target_channel_index,
        forecast_len=FORECAST_LEN,
        scale_ratio=power_scale_ratio,
        scale_offset=power_scale_offset,
        name="persistence_forecast_candidate",
    )(inputs)
    x = layers.Conv1D(
        32,
        kernel_size=5,
        padding="causal",
        activation="gelu",
        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
        name="residual_causal_conv_1",
    )(inputs)
    x = layers.Conv1D(
        32,
        kernel_size=3,
        padding="causal",
        dilation_rate=2,
        activation="gelu",
        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
        name="residual_causal_conv_2",
    )(x)
    recent = TakeLastToken(name="residual_recent_token")(x)
    pooled = layers.GlobalAveragePooling1D(name="residual_global_pool")(x)
    last_features = TakeLastToken(name="residual_last_history_features")(inputs)
    implicit_context = layers.Concatenate(name="residual_context")(
        [recent, pooled, last_features]
    )
    hidden = layers.Dense(
        64,
        activation="gelu",
        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
        name="residual_hidden",
    )(implicit_context)
    residual_head = layers.Dropout(
        HEAD_DROPOUT,
        name="residual_dropout",
    )(hidden)
    residual = layers.Dense(
        FORECAST_LEN,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
        name="persistence_residual",
    )(residual_head)
    corrected = layers.Add(name="corrected_forecast_candidate")(
        [persistence, residual]
    )
    return persistence, corrected, hidden


def _compile_regime_model(model, auxiliary_tasks):
    losses = {
        "forecast_power": keras.losses.Huber(delta=1.0),
        "candidate_forecast": keras.losses.Huber(delta=1.0),
    }
    loss_weights = {
        "forecast_power": 1.0,
        "candidate_forecast": CANDIDATE_LOSS_WEIGHT,
    }
    metrics = {
        "forecast_power": [
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
        "candidate_forecast": [
            keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    }
    if auxiliary_tasks:
        losses.update(
            {
                "regime_class": keras.losses.CategoricalCrossentropy(),
                "low_power_aux": keras.losses.BinaryCrossentropy(),
                "change_magnitude_aux": keras.losses.Huber(delta=0.05),
            }
        )
        loss_weights.update(
            {
                "regime_class": AUX_CLASS_LOSS_WEIGHT,
                "low_power_aux": AUX_LOW_POWER_LOSS_WEIGHT,
                "change_magnitude_aux": AUX_MAGNITUDE_LOSS_WEIGHT,
            }
        )
        metrics.update(
            {
                "regime_class": [
                    keras.metrics.CategoricalAccuracy(name="accuracy")
                ],
                "low_power_aux": [
                    keras.metrics.BinaryAccuracy(name="accuracy")
                ],
                "change_magnitude_aux": [
                    keras.metrics.MeanAbsoluteError(name="mae")
                ],
            }
        )
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0,
        ),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    return model


def build_regime_encoder_patchtst_model(
    variant_id,
    input_dim,
    target_channel_index,
    power_scale_ratio=1.0,
    power_scale_offset=0.0,
    regime_feature_config=None,
):
    """构建 R2--R5；模型唯一输入是历史特征张量。"""
    if variant_id not in TRAINABLE_VARIANTS:
        raise ValueError(f"{variant_id} 不是可训练的第二阶段变体")
    if target_channel_index is None or not 0 <= target_channel_index < input_dim:
        raise ValueError("历史功率目标通道索引无效")

    spec = VARIANT_SPECS[variant_id]
    inputs = keras.Input(
        shape=(HISTORY_LEN, input_dim),
        name="history_features",
    )
    persistence, corrected, implicit_context = _build_b2_candidates(
        inputs,
        target_channel_index,
        power_scale_ratio,
        power_scale_offset,
    )

    regime_context = None
    explicit_features = None
    if spec["encoder_type"] == "horizon_only":
        gate = HorizonOnlyCorrectionGate(
            forecast_len=FORECAST_LEN,
            initial_weight=GATE_INITIAL_CORRECTED_WEIGHT,
            name="correction_gate",
        )(inputs)
    elif spec["encoder_type"] == "implicit_causal_conv_context":
        gate = SampleHorizonCorrectionGate(
            forecast_len=FORECAST_LEN,
            hidden_dim=GATE_HIDDEN_DIM,
            horizon_embedding_dim=HORIZON_EMBEDDING_DIM,
            dropout=GATE_DROPOUT,
            initial_weight=GATE_INITIAL_CORRECTED_WEIGHT,
            name="correction_gate",
        )(implicit_context)
    elif spec["encoder_type"] == "explicit_wind_regime_statistics":
        if not regime_feature_config:
            raise ValueError("显式工况编码变体缺少 regime_feature_config")
        explicit_features = ExplicitWindRegimeFeatures(
            **_layer_feature_kwargs(regime_feature_config),
            name="explicit_regime_features",
        )(inputs)
        normalized = layers.LayerNormalization(
            epsilon=1e-6,
            name="explicit_regime_feature_norm",
        )(explicit_features)
        regime_context = layers.Dense(
            REGIME_CONTEXT_DIM,
            activation="gelu",
            kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
            name="regime_context_hidden",
        )(normalized)
        regime_context = layers.Dropout(
            GATE_DROPOUT,
            name="regime_context_dropout",
        )(regime_context)
        regime_context = layers.Dense(
            REGIME_CONTEXT_DIM,
            activation="gelu",
            name="regime_context",
        )(regime_context)
        gate = SampleHorizonCorrectionGate(
            forecast_len=FORECAST_LEN,
            hidden_dim=GATE_HIDDEN_DIM,
            horizon_embedding_dim=HORIZON_EMBEDDING_DIM,
            dropout=GATE_DROPOUT,
            initial_weight=GATE_INITIAL_CORRECTED_WEIGHT,
            name="correction_gate",
        )(regime_context)
    else:
        raise ValueError(f"不支持的 encoder_type: {spec['encoder_type']}")

    forecast = TwoCandidateGateFusion(name="forecast_power")(
        [persistence, corrected, gate]
    )
    candidate_forecast = layers.Activation(
        "linear",
        name="candidate_forecast",
    )(corrected)
    outputs = {
        "forecast_power": forecast,
        "candidate_forecast": candidate_forecast,
    }

    if spec["auxiliary_tasks"]:
        if regime_context is None:
            raise ValueError("辅助工况任务必须建立在显式 regime_context 上")
        auxiliary_hidden = layers.Dense(
            16,
            activation="gelu",
            name="regime_aux_hidden",
        )(regime_context)
        outputs.update(
            {
                "regime_class": layers.Dense(
                    3,
                    activation="softmax",
                    name="regime_class",
                )(auxiliary_hidden),
                "low_power_aux": layers.Dense(
                    1,
                    activation="sigmoid",
                    name="low_power_aux",
                )(auxiliary_hidden),
                "change_magnitude_aux": layers.Dense(
                    1,
                    activation="sigmoid",
                    name="change_magnitude_aux",
                )(auxiliary_hidden),
            }
        )

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"WindRegimeEncoderPatchTST_{variant_id}",
    )
    set_global_seed(RANDOM_SEED)
    return _compile_regime_model(model, spec["auxiliary_tasks"])


def build_regime_encoder_patchtst_model_from_artifact(artifact):
    if artifact.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(
            "artifact 架构版本不匹配: "
            f"{artifact.get('architecture_version')} != {ARCHITECTURE_VERSION}"
        )
    return build_regime_encoder_patchtst_model(
        variant_id=artifact["variant_id"],
        input_dim=len(artifact["input_cols"]),
        target_channel_index=int(artifact["target_index"]),
        power_scale_ratio=float(artifact["power_scale_ratio"]),
        power_scale_offset=float(artifact["power_scale_offset"]),
        regime_feature_config=artifact.get("regime_feature_config"),
    )


def get_regime_custom_objects():
    custom_objects = dict(get_min_custom_objects())
    custom_objects.update(
        {
            "PersistenceForecast": PersistenceForecast,
            "WindFeTSPatchTST>PersistenceForecast": PersistenceForecast,
            "TakeLastToken": TakeLastToken,
            "WindFeTSPatchTST>TakeLastToken": TakeLastToken,
        }
    )
    classes = (
        ExplicitWindRegimeFeatures,
        HorizonOnlyCorrectionGate,
        SampleHorizonCorrectionGate,
        TwoCandidateGateFusion,
    )
    for cls in classes:
        custom_objects[cls.__name__] = cls
        custom_objects[f"WindRegimeEncoderPatchTST>{cls.__name__}"] = cls
    return custom_objects


def _prepare_farm(train_file):
    farm_id = get_farm_id(train_file)
    train_df, feature_cols, capacity = load_and_preprocess(train_file, is_train=True)
    (
        features,
        target,
        input_cols,
        target_index,
        scaler_x,
        scaler_y,
    ) = build_scaled_arrays(train_df, feature_cols)
    validate_preprocessed_data(
        train_df,
        features,
        target,
        input_cols,
        target_index,
    )
    ratio, offset = compute_power_scale_alignment(
        scaler_x,
        scaler_y,
        target_index,
    )
    feature_config = _regime_feature_config(
        input_cols,
        scaler_x,
        capacity,
        target_index,
    )
    return {
        "farm_id": farm_id,
        "train_file": train_file,
        "train_df": train_df,
        "feature_cols": feature_cols,
        "capacity": float(capacity),
        "features": features,
        "target": target,
        "input_cols": input_cols,
        "target_index": target_index,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "power_scale_ratio": ratio,
        "power_scale_offset": offset,
        "regime_feature_config": feature_config,
    }


def _regime_targets_tensor(
    batch_x,
    batch_y,
    target_index,
    power_scale_ratio,
    power_scale_offset,
    target_mean,
    target_scale,
    capacity,
):
    last_y_scaled = (
        batch_x[:, -1, target_index] * power_scale_ratio + power_scale_offset
    )
    future_power = batch_y * target_scale + target_mean
    last_power = last_y_scaled * target_scale + target_mean
    change = (future_power - last_power[:, tf.newaxis]) / capacity
    magnitude = tf.reduce_max(tf.abs(change), axis=1)
    peak_index = tf.argmax(tf.abs(change), axis=1, output_type=tf.int32)
    signed_peak = tf.gather(change, peak_index, batch_dims=1)
    stable = magnitude <= STABLE_CHANGE_THRESHOLD
    up = tf.logical_and(tf.logical_not(stable), signed_peak >= 0.0)
    down = tf.logical_and(tf.logical_not(stable), signed_peak < 0.0)
    regime_class = tf.stack([stable, up, down], axis=1)
    low_power = (
        tf.reduce_mean(future_power / capacity, axis=1) <= LOW_POWER_THRESHOLD
    )
    return {
        "regime_class": tf.cast(regime_class, tf.float32),
        "low_power_aux": tf.cast(low_power[:, tf.newaxis], tf.float32),
        "change_magnitude_aux": tf.cast(
            tf.clip_by_value(magnitude[:, tf.newaxis], 0.0, 1.0),
            tf.float32,
        ),
    }


def _attach_training_targets(dataset, prepared, auxiliary_tasks):
    target_mean = tf.constant(float(prepared["scaler_y"].mean_[0]), tf.float32)
    target_scale = tf.constant(float(prepared["scaler_y"].scale_[0]), tf.float32)
    capacity = tf.constant(float(prepared["capacity"]), tf.float32)
    power_scale_ratio = tf.constant(
        float(prepared["power_scale_ratio"]),
        tf.float32,
    )
    power_scale_offset = tf.constant(
        float(prepared["power_scale_offset"]),
        tf.float32,
    )
    target_index = int(prepared["target_index"])

    def _map_targets(batch_x, batch_y):
        targets = {
            "forecast_power": batch_y,
            "candidate_forecast": batch_y,
        }
        if auxiliary_tasks:
            targets.update(
                _regime_targets_tensor(
                    batch_x,
                    batch_y,
                    target_index,
                    power_scale_ratio,
                    power_scale_offset,
                    target_mean,
                    target_scale,
                    capacity,
                )
            )
        return batch_x, targets

    return dataset.map(
        _map_targets,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    ).prefetch(tf.data.AUTOTUNE)


def _make_variant_datasets(prepared, auxiliary_tasks):
    train_ds, val_ds, train_samples, total_samples = make_window_dataset(
        prepared["features"],
        prepared["target"],
        HISTORY_LEN,
        FORECAST_LEN,
        BATCH_SIZE,
        VALIDATION_SPLIT,
    )
    return (
        _attach_training_targets(train_ds, prepared, auxiliary_tasks),
        _attach_training_targets(val_ds, prepared, auxiliary_tasks),
        train_samples,
        total_samples,
    )


def _resolve_existing_path(path):
    if not path:
        return None
    candidates = [os.fspath(path)]
    if not os.path.isabs(path):
        candidates.append(os.path.join(os.path.dirname(__file__), path))
    return next((value for value in candidates if os.path.exists(value)), None)


def _stage1_artifact_path(source_variant, farm_id):
    model_name = stage1_variant_model_name(source_variant)
    return os.path.join(
        stage1_variant_dirs(source_variant, create=False)["preprocess"],
        f"{model_name}_farm_{farm_id}_preprocess.pkl",
    )


def _load_stage1_artifact(source_variant, farm_id):
    path = _stage1_artifact_path(source_variant, farm_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少 Stage-1 引用 artifact: {path}")
    artifact = joblib.load(path)
    if artifact.get("variant_id") != source_variant:
        raise ValueError(f"Stage-1 artifact 变体不匹配: {path}")
    if artifact.get("architecture_version") != STAGE1_ARCHITECTURE_VERSION:
        raise ValueError(f"Stage-1 artifact 架构版本不匹配: {path}")
    if int(artifact.get("random_seed", -1)) != RANDOM_SEED:
        raise ValueError(f"Stage-1 artifact seed 不是 {RANDOM_SEED}: {path}")
    return artifact, os.path.abspath(path)


def _initialize_from_stage1_b2(model, prepared, sample_x):
    artifact, artifact_path = _load_stage1_artifact(
        "b2_persistence_residual",
        prepared["farm_id"],
    )
    source_model_path = _resolve_existing_path(artifact.get("model_path"))
    if source_model_path is None:
        default_path = os.path.join(
            stage1_variant_dirs("b2_persistence_residual", create=False)["models"],
            f"{stage1_variant_model_name('b2_persistence_residual')}_farm_"
            f"{prepared['farm_id']}.keras",
        )
        source_model_path = _resolve_existing_path(default_path)
    if source_model_path is None:
        raise FileNotFoundError(
            f"缺少场站 {prepared['farm_id']} 的 Stage-1 B2 完整模型"
        )
    source_model = keras.models.load_model(
        source_model_path,
        custom_objects=get_regime_custom_objects(),
        compile=False,
    )
    copied = []
    for layer_name in B2_WEIGHTED_LAYER_NAMES:
        source_layer = source_model.get_layer(layer_name)
        target_layer = model.get_layer(layer_name)
        source_weights = source_layer.get_weights()
        target_weights = target_layer.get_weights()
        if len(source_weights) != len(target_weights):
            raise ValueError(f"B2 层权重数量不匹配: {layer_name}")
        for source_value, target_value in zip(source_weights, target_weights):
            if source_value.shape != target_value.shape:
                raise ValueError(
                    f"B2 层 {layer_name} 权重形状不匹配: "
                    f"{source_value.shape} vs {target_value.shape}"
                )
        target_layer.set_weights(source_weights)
        copied.append(layer_name)
    source_prediction = np.asarray(source_model(sample_x, training=False), dtype=float)
    target_diagnostic = keras.Model(
        model.inputs,
        model.get_layer("corrected_forecast_candidate").output,
    )
    target_prediction = np.asarray(
        target_diagnostic(sample_x, training=False),
        dtype=float,
    )
    transfer_max_abs_error = float(
        np.max(np.abs(source_prediction - target_prediction))
    )
    if not np.allclose(
        source_prediction,
        target_prediction,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError(
            "Stage-1 B2 权重迁移后 corrected candidate 不等价，"
            f"最大误差={transfer_max_abs_error}"
        )
    del target_diagnostic
    del source_model
    return {
        "source_variant": "b2_persistence_residual",
        "source_artifact_path": artifact_path,
        "source_model_path": os.path.abspath(source_model_path),
        "copied_weight_layers": copied,
        "transfer_verified": True,
        "transfer_max_abs_error": transfer_max_abs_error,
    }


def _inverse_power(scaler_y, values):
    values = np.asarray(values).reshape(-1, 1)
    return scaler_y.inverse_transform(values).reshape(-1)


def build_regime_targets_numpy(y_true, last_power, capacity):
    """只用于训练/验证诊断；future truth 不会返回模型输入。"""
    y_true = np.asarray(y_true, dtype=float)
    last_power = np.asarray(last_power, dtype=float).reshape(-1)
    change = (y_true - last_power[:, None]) / float(capacity)
    finite = np.isfinite(change)
    valid_future = finite.any(axis=1)
    safe_abs_change = np.where(finite, np.abs(change), -np.inf)
    peak_index = np.argmax(safe_abs_change, axis=1)
    signed_peak = change[np.arange(len(change)), peak_index]
    magnitude = safe_abs_change[np.arange(len(change)), peak_index]
    magnitude = np.where(valid_future, magnitude, np.nan)
    stable = valid_future & (magnitude <= STABLE_CHANGE_THRESHOLD)
    regime_index = np.where(
        stable,
        0,
        np.where(valid_future & (signed_peak >= 0.0), 1, 2),
    )
    regime_name = np.asarray(["stable", "ramp_up", "ramp_down"])[regime_index]
    regime_name = np.where(valid_future, regime_name, "unknown")
    normalized_power = y_true / float(capacity)
    valid_counts = np.isfinite(normalized_power).sum(axis=1)
    power_sums = np.nansum(normalized_power, axis=1)
    mean_power = np.divide(
        power_sums,
        valid_counts,
        out=np.full(len(y_true), np.nan, dtype=float),
        where=valid_counts > 0,
    )
    low_power = valid_future & (mean_power <= LOW_POWER_THRESHOLD)
    return {
        "regime_index": regime_index.astype(int),
        "regime_name": regime_name,
        "low_power": low_power,
        "change_magnitude": magnitude,
        "valid_future": valid_future,
    }


def _physical_metrics(y_true, y_pred, capacity):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        return {"mae": np.nan, "rmse": np.nan, "nmae": np.nan, "nrmse": np.nan}
    mae = float(mean_absolute_error(y_true[valid], y_pred[valid]))
    rmse = float(np.sqrt(mean_squared_error(y_true[valid], y_pred[valid])))
    return {
        "mae": mae,
        "rmse": rmse,
        "nmae": mae / capacity,
        "nrmse": rmse / capacity,
    }


def _collect_validation_diagnostics(model, val_ds, prepared, variant_id):
    output_layers = [
        model.get_layer("forecast_power").output,
        model.get_layer("persistence_forecast_candidate").output,
        model.get_layer("corrected_forecast_candidate").output,
        model.get_layer("correction_gate").output,
    ]
    auxiliary = VARIANT_SPECS[variant_id]["auxiliary_tasks"]
    if auxiliary:
        output_layers.extend(
            [
                model.get_layer("regime_class").output,
                model.get_layer("low_power_aux").output,
                model.get_layer("change_magnitude_aux").output,
            ]
        )
    diagnostic_model = keras.Model(model.inputs, output_layers)
    predicted = diagnostic_model.predict(val_ds, verbose=0)
    if not isinstance(predicted, (list, tuple)):
        raise TypeError("诊断模型必须返回多个命名层输出")
    forecast_scaled, persistence_scaled, corrected_scaled, gate = [
        np.asarray(value) for value in predicted[:4]
    ]
    if gate.shape != forecast_scaled.shape:
        raise ValueError(f"gate 形状异常: {gate.shape} vs {forecast_scaled.shape}")
    if not np.isfinite(gate).all() or np.min(gate) < -1e-6 or np.max(gate) > 1 + 1e-6:
        raise FloatingPointError("验证 gate 包含非法值")
    reconstructed = persistence_scaled + gate * (
        corrected_scaled - persistence_scaled
    )
    reconstruction_error = float(np.max(np.abs(reconstructed - forecast_scaled)))
    if reconstruction_error > 1e-5:
        raise ValueError(f"两候选融合重构误差过大: {reconstruction_error}")

    x_batches = []
    y_batches = []
    for batch_x, batch_targets in val_ds:
        x_batches.append(batch_x.numpy())
        y_batches.append(batch_targets["forecast_power"].numpy())
    x_values = np.concatenate(x_batches, axis=0)
    y_true_scaled = np.concatenate(y_batches, axis=0)
    scaler_y = prepared["scaler_y"]
    forecast = _inverse_power(scaler_y, forecast_scaled).reshape(forecast_scaled.shape)
    persistence = _inverse_power(scaler_y, persistence_scaled).reshape(
        persistence_scaled.shape
    )
    corrected = _inverse_power(scaler_y, corrected_scaled).reshape(corrected_scaled.shape)
    y_true = _inverse_power(scaler_y, y_true_scaled).reshape(y_true_scaled.shape)
    capacity = prepared["capacity"]
    forecast = np.clip(forecast, 0, capacity)
    persistence = np.clip(persistence, 0, capacity)
    corrected = np.clip(corrected, 0, capacity)
    last_y_scaled = (
        x_values[:, -1, prepared["target_index"]]
        * prepared["power_scale_ratio"]
        + prepared["power_scale_offset"]
    )
    last_power = _inverse_power(scaler_y, last_y_scaled)
    regimes = build_regime_targets_numpy(y_true, last_power, capacity)

    rows = []
    masks = {
        "all": np.ones(len(y_true), dtype=bool),
        "stable": regimes["regime_name"] == "stable",
        "ramp_up": regimes["regime_name"] == "ramp_up",
        "ramp_down": regimes["regime_name"] == "ramp_down",
        "low_power": regimes["low_power"],
    }
    for group_name, mask in masks.items():
        for candidate_name, candidate_values in {
            "fused": forecast,
            "persistence": persistence,
            "corrected": corrected,
        }.items():
            metrics = _physical_metrics(y_true[mask], candidate_values[mask], capacity)
            rows.append(
                {
                    "model_family": MODEL_FAMILY,
                    "variant_id": variant_id,
                    "farm_id": prepared["farm_id"],
                    "regime_group": group_name,
                    "candidate": candidate_name,
                    "sample_count": int(mask.sum()),
                    **metrics,
                }
            )

    binary_entropy = -(
        gate * np.log(np.clip(gate, 1e-8, 1.0))
        + (1.0 - gate) * np.log(np.clip(1.0 - gate, 1e-8, 1.0))
    ) / np.log(2.0)
    corrected_better = np.square(corrected - y_true) < np.square(
        persistence - y_true
    )
    hard_choice = gate >= 0.5
    gate_fields = {
        "gate_mean": float(gate.mean()),
        "gate_std": float(gate.std()),
        "gate_sample_variation": float(np.std(gate, axis=0).mean()),
        "gate_binary_entropy": float(binary_entropy.mean()),
        "gate_saturation_low_rate": float((gate < 0.05).mean()),
        "gate_saturation_high_rate": float((gate > 0.95).mean()),
        "gate_oracle_choice_accuracy": float((hard_choice == corrected_better).mean()),
        "gate_oracle_brier": float(
            np.mean(np.square(gate - corrected_better.astype(float)))
        ),
        "fusion_reconstruction_max_abs_error": reconstruction_error,
    }
    gate_by_horizon = []
    for horizon in range(FORECAST_LEN):
        values = gate[:, horizon]
        gate_by_horizon.append(
            {
                "model_family": MODEL_FAMILY,
                "variant_id": variant_id,
                "farm_id": prepared["farm_id"],
                "horizon_step": horizon + 1,
                "gate_mean": float(values.mean()),
                "gate_std": float(values.std()),
                "gate_p10": float(np.quantile(values, 0.10)),
                "gate_p50": float(np.quantile(values, 0.50)),
                "gate_p90": float(np.quantile(values, 0.90)),
                "corrected_better_rate": float(corrected_better[:, horizon].mean()),
                "oracle_choice_accuracy": float(
                    (hard_choice[:, horizon] == corrected_better[:, horizon]).mean()
                ),
            }
        )

    auxiliary_fields = {}
    auxiliary_confusion = None
    if auxiliary:
        class_probability = np.asarray(predicted[4], dtype=float)
        low_probability = np.asarray(predicted[5], dtype=float).reshape(-1)
        magnitude_prediction = np.asarray(predicted[6], dtype=float).reshape(-1)
        true_class = regimes["regime_index"]
        predicted_class = np.argmax(class_probability, axis=1)
        auxiliary_confusion = confusion_matrix(
            true_class,
            predicted_class,
            labels=[0, 1, 2],
        )
        auxiliary_fields = {
            "aux_regime_accuracy": float(np.mean(predicted_class == true_class)),
            "aux_regime_macro_f1": float(
                f1_score(
                    true_class,
                    predicted_class,
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                )
            ),
            "aux_low_power_accuracy": float(
                np.mean((low_probability >= 0.5) == regimes["low_power"])
            ),
            "aux_low_power_brier": float(
                np.mean(
                    np.square(low_probability - regimes["low_power"].astype(float))
                )
            ),
            "aux_change_magnitude_mae": float(
                np.mean(
                    np.abs(
                        magnitude_prediction
                        - np.clip(regimes["change_magnitude"], 0.0, 1.0)
                    )
                )
            ),
        }

    return {
        "overall_metrics": {
            f"val_{key}": value
            for key, value in _physical_metrics(y_true, forecast, capacity).items()
        },
        "candidate_metrics": {
            f"val_candidate_{key}": value
            for key, value in _physical_metrics(y_true, corrected, capacity).items()
        },
        "persistence_metrics": {
            f"val_persistence_{key}": value
            for key, value in _physical_metrics(y_true, persistence, capacity).items()
        },
        "regime_rows": rows,
        "gate_rows": gate_by_horizon,
        "gate_fields": gate_fields,
        "auxiliary_fields": auxiliary_fields,
        "auxiliary_confusion": auxiliary_confusion,
    }


def _save_history(history, dirs, model_name, farm_id):
    frame = pd.DataFrame(history.history)
    frame.index = np.arange(1, len(frame) + 1)
    frame.index.name = "epoch"
    path = os.path.join(dirs["history"], f"{model_name}_history_farm_{farm_id}.csv")
    frame.to_csv(path, encoding="utf-8-sig")
    figure_path = os.path.join(
        dirs["history"],
        f"{model_name}_history_farm_{farm_id}.png",
    )
    try:
        os.environ["MPLCONFIGDIR"] = os.path.join(dirs["root"], "matplotlib_cache")
        os.environ["XDG_CACHE_HOME"] = os.environ["MPLCONFIGDIR"]
        os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        columns = [name for name in frame if name.startswith("val_")]
        fig, ax = plt.subplots(figsize=(11, 5))
        for name in ["loss", "forecast_power_loss", *columns[:2]]:
            if name in frame and name not in columns:
                ax.plot(frame.index, frame[name], label=name)
            val_name = f"val_{name}"
            if val_name in frame:
                ax.plot(frame.index, frame[val_name], label=val_name)
        ax.set_title(f"{model_name} - Farm {farm_id}")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"训练曲线保存失败: {exc}")
        figure_path = None
    return path, figure_path


def _train_paths(dirs, model_name, farm_id):
    return {
        "model_path": os.path.join(
            dirs["models"],
            f"{model_name}_farm_{farm_id}.keras",
        ),
        "best_weights_path": os.path.join(
            dirs["weights"],
            f"{model_name}_farm_{farm_id}_best.weights.h5",
        ),
        "artifact_path": os.path.join(
            dirs["preprocess"],
            f"{model_name}_farm_{farm_id}_preprocess.pkl",
        ),
        "tail_path": os.path.join(
            dirs["tails"],
            f"{model_name}_tail_farm_{farm_id}.csv",
        ),
    }


def _save_load_smoke_test(model, model_path, val_ds):
    if os.getenv("WIND_REGIME_SAVE_SMOKE_TEST", "1") == "0":
        return
    sample_x, _ = next(iter(val_ds))
    sample_x = sample_x[:2]
    diagnostic = keras.Model(model.inputs, model.get_layer("forecast_power").output)
    expected = np.asarray(diagnostic(sample_x, training=False), dtype=float)
    restored = keras.models.load_model(
        model_path,
        custom_objects=get_regime_custom_objects(),
        compile=False,
    )
    restored_diagnostic = keras.Model(
        restored.inputs,
        restored.get_layer("forecast_power").output,
    )
    actual = np.asarray(restored_diagnostic(sample_x, training=False), dtype=float)
    if not np.allclose(expected, actual, rtol=1e-6, atol=1e-6):
        raise ValueError("保存后重载模型的 forecast 输出不一致")
    del restored_diagnostic, restored


def train_variant_for_farm(variant_id, prepared):
    if variant_id not in TRAINABLE_VARIANTS:
        raise ValueError(f"引用变体 {variant_id} 不应进入训练函数")
    keras.backend.clear_session()
    configure_reproducibility()
    spec = VARIANT_SPECS[variant_id]
    model_name = variant_model_name(variant_id)
    dirs = variant_dirs(variant_id)
    paths = _train_paths(dirs, model_name, prepared["farm_id"])
    print(
        f"\n===== {spec['label']} / 风电场 {prepared['farm_id']} / "
        f"seed={RANDOM_SEED} ====="
    )

    train_ds, val_ds, train_samples, total_samples = _make_variant_datasets(
        prepared,
        spec["auxiliary_tasks"],
    )
    model = build_regime_encoder_patchtst_model(
        variant_id,
        len(prepared["input_cols"]),
        prepared["target_index"],
        prepared["power_scale_ratio"],
        prepared["power_scale_offset"],
        prepared["regime_feature_config"],
    )
    sample_x, _ = next(iter(train_ds))
    backbone_source = _initialize_from_stage1_b2(
        model,
        prepared,
        sample_x[:2],
    )
    total_params = int(model.count_params())
    trainable_params = int(
        sum(np.prod(variable.shape) for variable in model.trainable_weights)
    )
    if total_params > HARD_PARAMETER_LIMIT:
        raise ValueError(
            f"{variant_id} 参数量 {total_params:,} 超过硬上限 "
            f"{HARD_PARAMETER_LIMIT:,}"
        )
    if total_params > IDEAL_PARAMETER_LIMIT:
        print(
            f"警告: {variant_id} 参数量 {total_params:,} 超过理想上限 "
            f"{IDEAL_PARAMETER_LIMIT:,}"
        )

    monitor = "val_forecast_power_loss"
    tensorboard_log_dir = os.path.join(
        dirs["tensorboard"],
        f"farm_{prepared['farm_id']}",
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    guard = NonFiniteTrainingGuard()
    callbacks = [
        guard,
        keras.callbacks.TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=0,
            write_graph=True,
            update_freq="epoch",
            profile_batch=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            paths["best_weights_path"],
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]
    start_time = time.monotonic()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    ensure_finite_training_history(history, guard)
    if not os.path.exists(paths["best_weights_path"]):
        raise FileNotFoundError(f"未生成最佳权重: {paths['best_weights_path']}")
    history_path, history_plot_path = _save_history(
        history,
        dirs,
        model_name,
        prepared["farm_id"],
    )
    model.load_weights(paths["best_weights_path"])
    diagnostics = _collect_validation_diagnostics(
        model,
        val_ds,
        prepared,
        variant_id,
    )
    model.save(paths["model_path"])
    _save_load_smoke_test(model, paths["model_path"], val_ds)
    elapsed_seconds = float(time.monotonic() - start_time)

    regime_path = os.path.join(
        dirs["validation_diagnostics"],
        f"{model_name}_validation_regime_metrics_farm_{prepared['farm_id']}.csv",
    )
    pd.DataFrame(diagnostics["regime_rows"]).to_csv(
        regime_path,
        index=False,
        encoding="utf-8-sig",
    )
    gate_path = os.path.join(
        dirs["validation_diagnostics"],
        f"{model_name}_validation_gate_by_horizon_farm_{prepared['farm_id']}.csv",
    )
    pd.DataFrame(diagnostics["gate_rows"]).to_csv(
        gate_path,
        index=False,
        encoding="utf-8-sig",
    )
    confusion_path = None
    if diagnostics["auxiliary_confusion"] is not None:
        confusion_path = os.path.join(
            dirs["validation_diagnostics"],
            f"{model_name}_validation_aux_confusion_farm_"
            f"{prepared['farm_id']}.csv",
        )
        pd.DataFrame(
            diagnostics["auxiliary_confusion"],
            index=["true_stable", "true_ramp_up", "true_ramp_down"],
            columns=["pred_stable", "pred_ramp_up", "pred_ramp_down"],
        ).to_csv(confusion_path, encoding="utf-8-sig")

    prepared["train_df"].iloc[-HISTORY_LEN:].to_csv(
        paths["tail_path"],
        index=True,
    )
    model_size_bytes = os.path.getsize(paths["model_path"])
    model_output_names = list(model.output_names)
    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "model_name": model_name,
        "variant_id": variant_id,
        "variant_label": spec["label"],
        "variant_config": dict(spec),
        "architecture_version": ARCHITECTURE_VERSION,
        "farm_id": prepared["farm_id"],
        "train_file": prepared["train_file"],
        "feature_cols": prepared["feature_cols"],
        "input_cols": prepared["input_cols"],
        "target_col": TARGET_COL,
        "target_index": prepared["target_index"],
        "scaler_x": prepared["scaler_x"],
        "scaler_y": prepared["scaler_y"],
        "capacity": prepared["capacity"],
        "history_len": HISTORY_LEN,
        "forecast_len": FORECAST_LEN,
        "time_freq": TIME_FREQ,
        "random_seed": RANDOM_SEED,
        "deterministic_ops_requested": True,
        "training_mode": "stage1_b2_warm_start_finetune",
        "requires_keras_model": True,
        "model_kind": spec["model_kind"],
        "gate_type": spec["gate_type"],
        "encoder_type": spec["encoder_type"],
        "auxiliary_tasks": spec["auxiliary_tasks"],
        "model_output_names": model_output_names,
        "forecast_output_layer_name": "forecast_power",
        "candidate_output_layer_name": "corrected_forecast_candidate",
        "diagnostic_layers": {
            "forecast": "forecast_power",
            "gate": "correction_gate",
            "persistence_candidate": "persistence_forecast_candidate",
            "corrected_candidate": "corrected_forecast_candidate",
            "explicit_features": (
                "explicit_regime_features"
                if spec["encoder_type"] == "explicit_wind_regime_statistics"
                else None
            ),
            "regime_context": (
                "regime_context"
                if spec["encoder_type"] == "explicit_wind_regime_statistics"
                else None
            ),
            "regime_class": "regime_class" if spec["auxiliary_tasks"] else None,
            "low_power": "low_power_aux" if spec["auxiliary_tasks"] else None,
            "change_magnitude": (
                "change_magnitude_aux" if spec["auxiliary_tasks"] else None
            ),
        },
        "expert_names": ["persistence", "corrected"],
        "power_scale_ratio": prepared["power_scale_ratio"],
        "power_scale_offset": prepared["power_scale_offset"],
        "regime_feature_config": prepared["regime_feature_config"],
        "regime_label_config": {
            "version": REGIME_LABEL_VERSION,
            "threshold_source": "predeclared_capacity_fraction",
            "stable_change_threshold": STABLE_CHANGE_THRESHOLD,
            "low_power_threshold": LOW_POWER_THRESHOLD,
            "change_band_edges": list(CHANGE_BAND_EDGES),
            "class_names": ["stable", "ramp_up", "ramp_down"],
            "future_labels_are_training_targets_only": True,
        },
        "gate_hidden_dim": GATE_HIDDEN_DIM,
        "horizon_embedding_dim": HORIZON_EMBEDDING_DIM,
        "regime_context_dim": REGIME_CONTEXT_DIM,
        "gate_dropout": GATE_DROPOUT,
        "gate_initial_corrected_weight": GATE_INITIAL_CORRECTED_WEIGHT,
        "candidate_supervision_loss_weight": CANDIDATE_LOSS_WEIGHT,
        "auxiliary_loss_weights": {
            "regime_class": AUX_CLASS_LOSS_WEIGHT,
            "low_power": AUX_LOW_POWER_LOSS_WEIGHT,
            "change_magnitude": AUX_MAGNITUDE_LOSS_WEIGHT,
        },
        "correction_kernel_l2": CORRECTION_KERNEL_L2,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "validation_split": VALIDATION_SPLIT,
        "learning_rate": LEARNING_RATE,
        "early_stopping_monitor": monitor,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_bytes": model_size_bytes,
        "training_elapsed_seconds": elapsed_seconds,
        "train_samples": train_samples,
        "val_samples": total_samples - train_samples,
        "model_path": paths["model_path"],
        "best_weights_path": paths["best_weights_path"],
        "artifact_path": paths["artifact_path"],
        "history_path": history_path,
        "history_plot_path": history_plot_path,
        "tensorboard_log_dir": tensorboard_log_dir,
        "tail_path": paths["tail_path"],
        "validation_regime_metrics_path": regime_path,
        "validation_gate_diagnostics_path": gate_path,
        "validation_aux_confusion_path": confusion_path,
        "backbone_initialization": backbone_source,
        "evaluation_pipeline_version": EVALUATION_PIPELINE_VERSION,
        "legacy_bidirectional_weather_imputation": True,
        "scaler_fit_scope": "full_train_file_including_validation",
        "validation_target_overlap_steps": FORECAST_LEN - 1,
        "exploratory_legacy_comparison": True,
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(keras, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        **diagnostics["overall_metrics"],
        **diagnostics["candidate_metrics"],
        **diagnostics["persistence_metrics"],
        **diagnostics["gate_fields"],
        **diagnostics["auxiliary_fields"],
    }
    joblib.dump(artifact, paths["artifact_path"])

    result = {
        "model_family": MODEL_FAMILY,
        "model_name": model_name,
        "variant_id": variant_id,
        "variant_label": spec["label"],
        "farm_id": prepared["farm_id"],
        "requires_training": True,
        "result_source": "stage2_trained",
        "source_variant": "b2_persistence_residual",
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_bytes": model_size_bytes,
        "training_elapsed_seconds": elapsed_seconds,
        "train_samples": train_samples,
        "val_samples": total_samples - train_samples,
        **diagnostics["overall_metrics"],
        **diagnostics["candidate_metrics"],
        **diagnostics["persistence_metrics"],
        **diagnostics["gate_fields"],
        **diagnostics["auxiliary_fields"],
        "model_path": paths["model_path"],
        "best_weights_path": paths["best_weights_path"],
        "artifact_path": paths["artifact_path"],
        "history_path": history_path,
        "history_plot_path": history_plot_path,
        "validation_regime_metrics_path": regime_path,
        "validation_gate_diagnostics_path": gate_path,
        "source_model_path": backbone_source["source_model_path"],
        "source_artifact_path": backbone_source["source_artifact_path"],
    }
    print(
        f"{model_name} / {prepared['farm_id']}: "
        f"val NRMSE={result['val_nrmse']:.6f}, params={total_params:,}, "
        f"gate={result['gate_mean']:.4f}"
    )
    del model
    keras.backend.clear_session()
    return result


def _load_stage1_training_metrics():
    path = os.path.join(STAGE1_RESULT_ROOT, "stage1_training_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少第一阶段训练汇总: {path}")
    frame = pd.read_csv(path)
    frame["farm_id"] = frame["farm_id"].astype(str)
    return frame, os.path.abspath(path)


def _reference_result(variant_id, farm_id, source_frame, source_summary_path):
    spec = VARIANT_SPECS[variant_id]
    source_variant = spec["source_variant"]
    matches = source_frame[
        (source_frame["variant_id"] == source_variant)
        & (source_frame["farm_id"] == str(farm_id))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Stage-1 {source_variant}/{farm_id} 训练结果应唯一，实际 {len(matches)}"
        )
    source = matches.iloc[0].to_dict()
    artifact, artifact_path = _load_stage1_artifact(source_variant, farm_id)
    row = {
        "model_family": MODEL_FAMILY,
        "model_name": variant_model_name(variant_id),
        "variant_id": variant_id,
        "variant_label": spec["label"],
        "farm_id": str(farm_id),
        "requires_training": False,
        "result_source": "frozen_stage1_reference",
        "source_variant": source_variant,
        "source_model_name": source.get("model_name"),
        "source_training_summary_path": source_summary_path,
        "source_artifact_path": artifact_path,
        "source_model_path": artifact.get("model_path"),
        "random_seed": int(artifact["random_seed"]),
        "batch_size": artifact.get("batch_size"),
        "total_params": int(source.get("total_params", artifact.get("total_params", 0))),
        "trainable_params": int(
            source.get("trainable_params", artifact.get("trainable_params", 0))
        ),
        "model_size_bytes": source.get("model_size_bytes", 0),
        "training_elapsed_seconds": source.get("training_elapsed_seconds", 0),
        "train_samples": source.get("train_samples"),
        "val_samples": source.get("val_samples"),
        "val_mae": source.get("val_mae"),
        "val_rmse": source.get("val_rmse"),
        "val_nmae": source.get("val_capacity_normalized_mae"),
        "val_nrmse": source.get("val_capacity_normalized_rmse"),
        "artifact_path": artifact_path,
        "model_path": artifact.get("model_path"),
        "history_path": artifact.get("history_path"),
    }
    if source_variant == "b0_persistence":
        row["source_model_path"] = None
        row["model_path"] = None
    return row


def _write_experiment_manifest():
    rows = []
    for order, (variant_id, spec) in enumerate(VARIANT_SPECS.items()):
        rows.append(
            {
                "variant_order": order,
                "variant_id": variant_id,
                "model_name": variant_model_name(variant_id),
                "label": spec["label"],
                "requires_training": spec["requires_training"],
                "model_kind": spec["model_kind"],
                "source_variant": spec["source_variant"],
                "gate_type": spec["gate_type"],
                "encoder_type": spec["encoder_type"],
                "auxiliary_tasks": spec["auxiliary_tasks"],
                "candidate_supervision": spec["requires_training"],
                "candidate_loss_weight": (
                    CANDIDATE_LOSS_WEIGHT if spec["requires_training"] else 0.0
                ),
                "description": spec["description"],
                "random_seed": RANDOM_SEED,
                "default_batch_size": BATCH_SIZE,
                "selection_metric_source": "validation_only",
                "test_selection_prohibited": True,
            }
        )
    os.makedirs(RESULT_ROOT, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULT_ROOT, "stage2_experiment_manifest.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def build_validation_comparison(metrics_df, requested_variants):
    rows = []
    for order, variant_id in enumerate(requested_variants):
        frame = metrics_df[metrics_df["variant_id"] == variant_id]
        params = pd.to_numeric(frame.get("total_params"), errors="coerce")
        nrmse = pd.to_numeric(frame.get("val_nrmse"), errors="coerce")
        rows.append(
            {
                "variant_order": order,
                "variant_id": variant_id,
                "model_name": variant_model_name(variant_id),
                "farm_count": int(frame["farm_id"].astype(str).nunique()),
                "parameter_count_max": (
                    int(params.max()) if params.notna().any() else np.nan
                ),
                "macro_val_nrmse": float(nrmse.mean()),
                "std_val_nrmse": float(nrmse.std(ddof=0)),
                "requires_training": VARIANT_SPECS[variant_id]["requires_training"],
                "result_source": (
                    "stage2_trained"
                    if VARIANT_SPECS[variant_id]["requires_training"]
                    else "frozen_stage1_reference"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_validation_screening(metrics_df, expected_farm_count):
    """只做预声明的 validation 资格筛查；绝不读取测试结果。"""
    r1 = metrics_df[metrics_df["variant_id"] == "r1_b2_reference"][
        ["farm_id", "val_nrmse"]
    ].copy()
    r6 = metrics_df[metrics_df["variant_id"] == "r6_b6_reference"][
        ["farm_id", "val_nrmse"]
    ].copy()
    r1["farm_id"] = r1["farm_id"].astype(str)
    r6["farm_id"] = r6["farm_id"].astype(str)
    reference = r1.merge(r6, on="farm_id", suffixes=("_r1", "_r6"))
    reference["per_farm_best_reference"] = reference[
        ["val_nrmse_r1", "val_nrmse_r6"]
    ].min(axis=1)
    r1_macro = float(pd.to_numeric(r1["val_nrmse"], errors="coerce").mean())

    rows = []
    for variant_id in TRAINABLE_VARIANTS:
        candidate = metrics_df[metrics_df["variant_id"] == variant_id].copy()
        candidate["farm_id"] = candidate["farm_id"].astype(str)
        paired = reference.merge(
            candidate[["farm_id", "val_nrmse", "total_params"]],
            on="farm_id",
            how="inner",
        )
        macro = float(pd.to_numeric(candidate["val_nrmse"], errors="coerce").mean())
        farm_noninferior = int(
            (
                paired["val_nrmse"]
                <= paired["per_farm_best_reference"] * 1.01
            ).sum()
        )
        params = pd.to_numeric(candidate["total_params"], errors="coerce")
        complete = int(candidate["farm_id"].nunique()) == expected_farm_count
        macro_pass = bool(np.isfinite(macro) and macro <= r1_macro * 1.005)
        farm_pass = bool(len(paired) == expected_farm_count and farm_noninferior >= 4)
        parameter_pass = bool(params.notna().all() and params.max() <= HARD_PARAMETER_LIMIT)
        rows.append(
            {
                "variant_id": variant_id,
                "farm_count": int(candidate["farm_id"].nunique()),
                "macro_val_nrmse": macro,
                "r1_macro_val_nrmse": r1_macro,
                "relative_to_r1_pct": (macro / r1_macro - 1.0) * 100.0,
                "farms_within_1pct_of_per_farm_best_r1_r6": farm_noninferior,
                "complete_farms": complete,
                "macro_within_r1_0_5pct": macro_pass,
                "at_least_4_farms_within_reference_1pct": farm_pass,
                "under_hard_parameter_limit": parameter_pass,
                "passes_primary_validation_screen": bool(
                    complete and macro_pass and farm_pass and parameter_pass
                ),
                "selected_variant": False,
                "selection_note": (
                    "仅为validation资格筛查；需结合预声明工况指标后锁定，"
                    "禁止使用test自动选型"
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    _validate_configuration()
    configure_reproducibility()
    _write_experiment_manifest()
    variants = get_requested_variants()
    train_files = discover_train_files(DATA_DIR)
    if not train_files:
        raise FileNotFoundError(f"未在 {DATA_DIR} 找到训练文件")
    farm_ids = [get_farm_id(path) for path in train_files]
    print(f"固定随机种子: {RANDOM_SEED}；batch_size={BATCH_SIZE}")
    print(f"场站数: {len(train_files)}；第二阶段矩阵: {variants}")
    print(f"实际需要训练: {[v for v in variants if v in TRAINABLE_VARIANTS]}")

    source_frame, source_summary_path = _load_stage1_training_metrics()
    results = []
    for variant_id in variants:
        if not VARIANT_SPECS[variant_id]["requires_training"]:
            for farm_id in farm_ids:
                results.append(
                    _reference_result(
                        variant_id,
                        farm_id,
                        source_frame,
                        source_summary_path,
                    )
                )

    trainable_requested = [
        variant_id for variant_id in variants if variant_id in TRAINABLE_VARIANTS
    ]
    for train_file in train_files:
        if not trainable_requested:
            break
        prepared = _prepare_farm(train_file)
        for variant_id in trainable_requested:
            results.append(train_variant_for_farm(variant_id, prepared))
            pd.DataFrame(results).to_csv(
                os.path.join(RESULT_ROOT, "stage2_training_metrics_partial.csv"),
                index=False,
                encoding="utf-8-sig",
            )

    metrics_df = pd.DataFrame(results)
    expected_rows = len(VARIANT_SPECS) * len(
        sorted(glob.glob(os.path.join(DATA_DIR, TRAIN_FILE_PATTERN)))
    )
    is_complete_matrix = (
        set(variants) == set(VARIANT_SPECS)
        and not os.getenv("WIND_REGIME_FARMS")
        and len(metrics_df) == expected_rows
        and not metrics_df.duplicated(["variant_id", "farm_id"]).any()
    )
    metrics_filename = (
        "stage2_training_metrics.csv"
        if is_complete_matrix
        else "stage2_training_metrics_partial.csv"
    )
    metrics_path = os.path.join(RESULT_ROOT, metrics_filename)
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    comparison = build_validation_comparison(metrics_df, variants)
    comparison_filename = (
        "stage2_validation_comparison.csv"
        if is_complete_matrix
        else "stage2_validation_comparison_partial.csv"
    )
    comparison.to_csv(
        os.path.join(RESULT_ROOT, comparison_filename),
        index=False,
        encoding="utf-8-sig",
    )
    if is_complete_matrix:
        screening = build_validation_screening(metrics_df, len(farm_ids))
        screening.to_csv(
            os.path.join(RESULT_ROOT, "stage2_validation_screening.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    for variant_id, frame in metrics_df.groupby("variant_id"):
        if variant_id not in TRAINABLE_VARIANTS:
            continue
        per_variant_suffix = "" if is_complete_matrix else "_partial"
        frame.to_csv(
            os.path.join(
                variant_dirs(variant_id)["root"],
                f"{variant_model_name(variant_id)}_training_metrics"
                f"{per_variant_suffix}.csv",
            ),
            index=False,
            encoding="utf-8-sig",
        )
    print(f"第二阶段训练/引用矩阵已保存: {metrics_path}")
    if not is_complete_matrix:
        print("当前是子集或不完整运行；未覆盖完整矩阵汇总，也未生成自动筛查结论")


if __name__ == "__main__":
    main()
