"""FeTS-PatchTST 第一阶段最小有效结构搜索训练入口。

本文件保持公共数据处理、96→16 任务、PatchTST/FeTS 分支、Adam、Huber
损失和验证切分与现有风电深度学习脚本一致，只改变激活的专家和融合方式。
默认依次运行以下七个需要拟合的结构以及一个解析式 persistence 基线：

    B0  persistence（解析式，无需拟合）
    B1  long PatchTST
    B2  persistence + 轻量残差
    B3  long + persistence，逐 horizon 静态 softmax
    B4M long + mid + persistence，逐 horizon 静态 softmax
    B4S long + short FeTS + persistence，逐 horizon 静态 softmax
    B5  long + mid + short + persistence，逐 horizon 静态 softmax
    B6  long + mid + short + persistence，样本/horizon 动态 softmax

B6 与当前 v5-A/B 拓扑对应，并以固定 seed=2026 重训；已有 v5-A/B 结果作为
冻结历史参考，不在这里重复训练成一个与 B6 完全相同的伪消融。可通过环境变量
WIND_FETS_MIN_VARIANTS 和 WIND_FETS_MIN_FARMS 分批执行，但随机种子不可覆盖。
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from wind_FeTS_PatchTST_train import (
    ADAFE_KERNEL_SIZE,
    ADAFE_PADDING,
    CORRECTION_KERNEL_L2,
    FOURIER_DEGREE,
    HORIZON_EMBEDDING_DIM,
    LAYER_SCALE_INIT,
    LOCAL_N_LAYERS,
    LOCAL_PATCH_LEN,
    LOCAL_PATCH_STRIDE,
    LONG_CONTEXT_DIM,
    MID_N_LAYERS,
    MID_PATCH_LEN,
    MID_PATCH_STRIDE,
    POLYNOMIAL_DEGREE,
    ROUTER_DROPOUT,
    ROUTER_HIDDEN_DIM,
    TARGET_WEATHER_HEADS,
    ChannelIdentityEmbedding,
    ExpertConvexFusion,
    FeTSPatchExtract,
    HorizonRegimeRouter,
    LayerScaleFeTSFeatureBlock,
    NonFiniteTrainingGuard,
    PersistenceForecast,
    TakeLastToken,
    TargetWeatherCrossAttention,
    compute_power_scale_alignment,
    ensure_finite_training_history,
    validate_preprocessed_data,
)
from wind_dl_model_train import (
    BATCH_SIZE as BASELINE_BATCH_SIZE,
    DATA_DIR,
    D_FF,
    D_MODEL,
    DROPOUT,
    EPOCHS as BASELINE_EPOCHS,
    FORECAST_LEN,
    HEAD_DROPOUT,
    HISTORY_LEN,
    LEARNING_RATE as BASELINE_LEARNING_RATE,
    N_HEADS,
    N_LAYERS,
    PATCH_LEN,
    PATCH_STRIDE,
    TARGET_COL,
    TIME_FREQ,
    VALIDATION_SPLIT as BASELINE_VALIDATION_SPLIT,
    LearnablePositionEmbedding,
    MergeChannels,
    PatchExtract,
    RestoreChannels,
    TakeChannel,
    build_scaled_arrays,
    compute_patch_num,
    load_and_preprocess,
    make_window_dataset,
    set_global_seed,
    transformer_encoder,
)

warnings.filterwarnings("ignore")


MODEL_FAMILY = "fets_patchtst_min"
ARCHITECTURE_VERSION = "fets_patchtst_min_stage1_v1"
ARTIFACT_SCHEMA_VERSION = 1
RESULT_ROOT = os.path.join("./wind_results", MODEL_FAMILY)
TRAIN_FILE_PATTERN = "wind_train_*.csv"
RANDOM_SEED = 2026

BATCH_SIZE = int(os.getenv("WIND_FETS_MIN_BATCH_SIZE", str(BASELINE_BATCH_SIZE)))
EPOCHS = int(os.getenv("WIND_FETS_MIN_EPOCHS", str(BASELINE_EPOCHS)))
VALIDATION_SPLIT = float(
    os.getenv(
        "WIND_FETS_MIN_VALIDATION_SPLIT",
        str(BASELINE_VALIDATION_SPLIT),
    )
)
LEARNING_RATE = float(
    os.getenv("WIND_FETS_MIN_LEARNING_RATE", str(BASELINE_LEARNING_RATE))
)
FFN_RATIO = 2

# 有序字典同时定义融合时的专家顺序；该顺序会写入 artifact，并与 router
# 最后一维严格对应。B6 已覆盖当前完整 v5 拓扑，因此不重复训练同构 B7。
VARIANT_SPECS = {
    "b0_persistence": {
        "label": "B0 Persistence",
        "experts": ("persistence",),
        "fusion_type": "single_analytic",
        "router_type": "none",
        "description": "最后历史功率重复到未来16步的解析式基线",
    },
    "b1_long": {
        "label": "B1 Long PatchTST",
        "experts": ("long",),
        "fusion_type": "single_network",
        "router_type": "none",
        "description": "原生长尺度 PatchTST",
    },
    "b2_persistence_residual": {
        "label": "B2 Persistence + lightweight residual",
        "experts": ("persistence_residual",),
        "fusion_type": "single_network",
        "router_type": "none",
        "description": "持续性预测加轻量因果卷积残差",
    },
    "b3_long_persistence_static": {
        "label": "B3 Long + Persistence static horizon router",
        "experts": ("long", "persistence"),
        "fusion_type": "convex",
        "router_type": "static_horizon_softmax",
        "description": "长尺度与持续性逐horizon静态融合",
    },
    "b4m_long_mid_persistence_static": {
        "label": "B4M Long + Mid + Persistence static horizon router",
        "experts": ("long", "mid", "persistence"),
        "fusion_type": "convex",
        "router_type": "static_horizon_softmax",
        "description": "检验中尺度专家的独立增益",
    },
    "b4s_long_short_persistence_static": {
        "label": "B4S Long + Short + Persistence static horizon router",
        "experts": ("long", "short", "persistence"),
        "fusion_type": "convex",
        "router_type": "static_horizon_softmax",
        "description": "检验局部FeTS专家的独立增益",
    },
    "b5_all_static": {
        "label": "B5 All experts static horizon router",
        "experts": ("long", "mid", "short", "persistence"),
        "fusion_type": "convex",
        "router_type": "static_horizon_softmax",
        "description": "检验完整多尺度表示本身",
    },
    "b6_all_dynamic": {
        "label": "B6 All experts sample-horizon dynamic router",
        "experts": ("long", "mid", "short", "persistence"),
        "fusion_type": "convex",
        "router_type": "sample_horizon_dense_softmax",
        "description": "当前完整v5拓扑在seed=2026下的公平重训",
    },
}

EXPERT_INITIAL_BIAS = {
    "long": 2.0,
    "mid": 0.0,
    "short": 0.0,
    "persistence": -2.0,
}
EXPERT_OUTPUT_LAYER_NAMES = {
    "long": "baseline_forecast_power",
    "mid": "mid_forecast_candidate",
    "short": "local_forecast_candidate",
    "persistence": "persistence_forecast_candidate",
    "persistence_residual": "forecast_power",
}


@keras.utils.register_keras_serializable(package="WindFeTSPatchTSTMin")
class StaticHorizonRouter(layers.Layer):
    """仅随 horizon 变化、不读取样本内容的可学习 softmax router。"""

    def __init__(
        self,
        forecast_len,
        n_experts,
        initial_bias=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if int(forecast_len) <= 0 or int(n_experts) <= 1:
            raise ValueError("forecast_len 必须为正且 n_experts 必须大于1")
        if initial_bias is None:
            initial_bias = [0.0] * int(n_experts)
        if len(initial_bias) != int(n_experts):
            raise ValueError("initial_bias 长度必须等于 n_experts")
        values = np.asarray(initial_bias, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("initial_bias 必须全部为有限数")
        self.forecast_len = int(forecast_len)
        self.n_experts = int(n_experts)
        self.initial_bias = tuple(float(value) for value in values)

    def build(self, input_shape):
        initial_logits = np.tile(
            np.asarray(self.initial_bias, dtype=np.float32)[None, :],
            (self.forecast_len, 1),
        )
        self.horizon_logits = self.add_weight(
            name="horizon_logits",
            shape=(self.forecast_len, self.n_experts),
            initializer=keras.initializers.Constant(initial_logits),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        weights = tf.nn.softmax(self.horizon_logits, axis=-1)
        return tf.broadcast_to(
            weights[tf.newaxis, :, :],
            [tf.shape(inputs)[0], self.forecast_len, self.n_experts],
        )

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.forecast_len, self.n_experts

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "forecast_len": self.forecast_len,
                "n_experts": self.n_experts,
                "initial_bias": list(self.initial_bias),
            }
        )
        return config


def variant_model_name(variant_id):
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"未知最小结构变体: {variant_id}")
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
    }
    if create:
        os.makedirs(RESULT_ROOT, exist_ok=True)
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
    return paths


def get_requested_variants():
    raw = os.getenv("WIND_FETS_MIN_VARIANTS")
    if not raw:
        return list(VARIANT_SPECS)
    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if any(item in {"all", "*"} for item in requested):
        return list(VARIANT_SPECS)
    invalid = sorted(set(requested) - set(VARIANT_SPECS))
    if invalid:
        raise ValueError(f"未知变体 {invalid}；可选: {list(VARIANT_SPECS)}")
    return list(dict.fromkeys(requested))


def discover_train_files(data_dir=DATA_DIR):
    files = sorted(glob.glob(os.path.join(data_dir, TRAIN_FILE_PATTERN)))
    requested = os.getenv("WIND_FETS_MIN_FARMS")
    if not requested:
        return files
    farm_ids = {item.strip() for item in requested.split(",") if item.strip()}
    return [path for path in files if get_farm_id(path) in farm_ids]


def get_farm_id(path):
    match = re.search(r"wind_train_(\d+)\.csv$", os.path.basename(path))
    if match:
        return match.group(1)
    return os.path.splitext(os.path.basename(path))[0]


def configure_reproducibility():
    set_global_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _reset_branch_seed():
    """使同名分支在不同变体中从相同 seed=2026 开始初始化。"""
    set_global_seed(RANDOM_SEED)


def _build_long_expert(inputs, input_dim, target_channel_index):
    _reset_branch_seed()
    patch_num = compute_patch_num(HISTORY_LEN, PATCH_LEN, PATCH_STRIDE)
    x = PatchExtract(PATCH_LEN, PATCH_STRIDE, name="long_patch_extract")(inputs)
    x = layers.Dense(D_MODEL, name="long_patch_projection")(x)
    x = MergeChannels(name="long_merge_channels")(x)
    x = LearnablePositionEmbedding(
        patch_num,
        D_MODEL,
        name="long_position_embedding",
    )(x)
    x = layers.Dropout(DROPOUT, name="long_patch_dropout")(x)
    for idx in range(N_LAYERS):
        x = transformer_encoder(
            x,
            D_MODEL,
            N_HEADS,
            D_FF,
            DROPOUT,
            name=f"long_encoder_{idx + 1}",
        )

    x = RestoreChannels(
        input_dim,
        patch_num,
        D_MODEL,
        name="long_restore_channels",
    )(x)
    target_repr = TakeChannel(
        target_channel_index,
        name="long_target_power_channel",
    )(x)
    target_repr = layers.Flatten(name="long_target_flatten")(target_repr)
    global_context = layers.GlobalAveragePooling2D(
        name="long_channel_context_pool",
    )(x)
    head = layers.Concatenate(name="long_forecast_context")(
        [target_repr, global_context]
    )
    head = layers.Dropout(HEAD_DROPOUT, name="long_head_dropout")(head)
    head = layers.Dense(
        D_FF,
        activation="gelu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="long_forecast_ff",
    )(head)
    head = layers.Dropout(HEAD_DROPOUT, name="long_forecast_dropout")(head)
    forecast = layers.Dense(
        FORECAST_LEN,
        name="baseline_forecast_power",
    )(head)
    return forecast, target_repr, global_context


def _build_long_context(target_repr, global_context, forecast):
    _reset_branch_seed()
    context = layers.Concatenate(name="long_local_context_features")(
        [target_repr, global_context, forecast]
    )
    return layers.Dense(
        LONG_CONTEXT_DIM,
        activation="gelu",
        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
        name="long_to_local_context_projection",
    )(context)


def _build_mid_expert(inputs, input_dim, target_channel_index, long_forecast):
    _reset_branch_seed()
    patch_num = compute_patch_num(
        HISTORY_LEN,
        MID_PATCH_LEN,
        MID_PATCH_STRIDE,
    )
    x = PatchExtract(
        MID_PATCH_LEN,
        MID_PATCH_STRIDE,
        name="mid_patch_extract",
    )(inputs)
    x = layers.Dense(D_MODEL, name="mid_patch_projection")(x)
    x = MergeChannels(name="mid_merge_channels")(x)
    x = LearnablePositionEmbedding(
        patch_num,
        D_MODEL,
        name="mid_position_embedding",
    )(x)
    x = layers.Dropout(DROPOUT, name="mid_patch_dropout")(x)
    for idx in range(MID_N_LAYERS):
        x = transformer_encoder(
            x,
            D_MODEL,
            N_HEADS,
            D_FF,
            DROPOUT,
            name=f"mid_encoder_{idx + 1}",
        )
    x = RestoreChannels(
        input_dim,
        patch_num,
        D_MODEL,
        name="mid_restore_channels",
    )(x)
    target_repr = TakeChannel(
        target_channel_index,
        name="mid_target_power_channel",
    )(x)
    target_repr = layers.Flatten(name="mid_target_flatten")(target_repr)
    global_context = layers.GlobalAveragePooling2D(
        name="mid_channel_context_pool",
    )(x)
    context = layers.Concatenate(name="mid_forecast_context")(
        [target_repr, global_context]
    )
    head = layers.Dropout(HEAD_DROPOUT, name="mid_head_dropout")(context)
    head = layers.Dense(
        D_FF,
        activation="gelu",
        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
        name="mid_forecast_ff",
    )(head)
    head = layers.Dropout(HEAD_DROPOUT, name="mid_forecast_dropout")(head)
    residual = layers.Dense(
        FORECAST_LEN,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
        name="mid_forecast_residual",
    )(head)
    forecast = layers.Add(name="mid_forecast_candidate")([long_forecast, residual])
    return forecast, context


def _build_short_expert(
    inputs,
    input_dim,
    target_channel_index,
    long_forecast,
    long_context,
):
    _reset_branch_seed()
    patch_num = compute_patch_num(
        HISTORY_LEN,
        LOCAL_PATCH_LEN,
        LOCAL_PATCH_STRIDE,
    )
    x = FeTSPatchExtract(
        LOCAL_PATCH_LEN,
        LOCAL_PATCH_STRIDE,
        name="local_patch_extract",
    )(inputs)
    x = layers.Dense(D_MODEL, name="local_patch_embedding")(x)
    x = ChannelIdentityEmbedding(
        input_dim,
        D_MODEL,
        name="local_channel_embedding",
    )(x)
    x = LayerScaleFeTSFeatureBlock(
        d_model=D_MODEL,
        fourier_degree=FOURIER_DEGREE,
        poly_degree=POLYNOMIAL_DEGREE,
        ffn_ratio=FFN_RATIO,
        dropout=DROPOUT,
        layer_scale_init=LAYER_SCALE_INIT,
        kernel_size=ADAFE_KERNEL_SIZE,
        padding=ADAFE_PADDING,
        name="local_fets_feature_block",
    )(x)
    x = TargetWeatherCrossAttention(
        n_channels=input_dim,
        target_channel_index=target_channel_index,
        d_model=D_MODEL,
        n_heads=TARGET_WEATHER_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        name="local_power_to_weather_attention",
    )(x)
    x = LearnablePositionEmbedding(
        patch_num,
        D_MODEL,
        name="local_position_embedding",
    )(x)
    x = layers.Dropout(DROPOUT, name="local_patch_dropout")(x)
    for idx in range(LOCAL_N_LAYERS):
        x = transformer_encoder(
            x,
            D_MODEL,
            N_HEADS,
            D_FF,
            DROPOUT,
            name=f"local_encoder_{idx + 1}",
        )
    recent = TakeLastToken(name="local_recent_token")(x)
    global_context = layers.GlobalAveragePooling1D(
        name="local_global_pool",
    )(x)
    head = layers.Concatenate(name="local_forecast_context")(
        [recent, global_context, long_context]
    )
    head = layers.Dropout(HEAD_DROPOUT, name="local_head_dropout")(head)
    head = layers.Dense(D_FF, activation="gelu", name="local_forecast_ff")(head)
    head = layers.Dropout(HEAD_DROPOUT, name="local_forecast_dropout")(head)
    residual = layers.Dense(
        FORECAST_LEN,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
        name="local_forecast_residual",
    )(head)
    forecast = layers.Add(name="local_forecast_candidate")([long_forecast, residual])
    return forecast, recent, global_context


def _build_persistence_expert(
    inputs,
    target_channel_index,
    power_scale_ratio,
    power_scale_offset,
):
    return PersistenceForecast(
        target_channel_index=target_channel_index,
        forecast_len=FORECAST_LEN,
        scale_ratio=power_scale_ratio,
        scale_offset=power_scale_offset,
        name="persistence_forecast_candidate",
    )(inputs)


def _build_persistence_residual_model(
    inputs,
    target_channel_index,
    power_scale_ratio,
    power_scale_offset,
):
    _reset_branch_seed()
    persistence = _build_persistence_expert(
        inputs,
        target_channel_index,
        power_scale_ratio,
        power_scale_offset,
    )
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
    head = layers.Concatenate(name="residual_context")([recent, pooled, last_features])
    head = layers.Dense(
        64,
        activation="gelu",
        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
        name="residual_hidden",
    )(head)
    head = layers.Dropout(HEAD_DROPOUT, name="residual_dropout")(head)
    residual = layers.Dense(
        FORECAST_LEN,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
        name="persistence_residual",
    )(head)
    return layers.Add(name="forecast_power")([persistence, residual])


def _compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0,
        ),
        loss=keras.losses.Huber(delta=1.0),
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def build_fets_patchtst_min_model(
    variant_id,
    input_dim,
    target_channel_index,
    power_scale_ratio=1.0,
    power_scale_offset=0.0,
):
    """按 variant 条件构图；未启用的专家不会进入模型参数。"""
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"未知变体: {variant_id}")
    if variant_id == "b0_persistence":
        raise ValueError("B0 是解析式基线，不构建 Keras 模型")
    if target_channel_index is None or not 0 <= target_channel_index < input_dim:
        raise ValueError("历史功率目标通道索引无效")

    spec = VARIANT_SPECS[variant_id]
    expert_names = tuple(spec["experts"])
    inputs = keras.Input(
        shape=(HISTORY_LEN, input_dim),
        name="history_features",
    )

    if variant_id == "b2_persistence_residual":
        outputs = _build_persistence_residual_model(
            inputs,
            target_channel_index,
            power_scale_ratio,
            power_scale_offset,
        )
        model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="WindFeTSPatchTSTMinPersistenceResidual",
        )
        set_global_seed(RANDOM_SEED)
        return _compile_model(model)

    predictions = {}
    router_contexts = []
    long_forecast = None
    long_target_repr = None
    long_global_context = None
    long_context = None

    if "long" in expert_names or {"mid", "short"} & set(expert_names):
        (
            long_forecast,
            long_target_repr,
            long_global_context,
        ) = _build_long_expert(inputs, input_dim, target_channel_index)
        predictions["long"] = long_forecast

    if "mid" in expert_names:
        mid_forecast, mid_context = _build_mid_expert(
            inputs,
            input_dim,
            target_channel_index,
            long_forecast,
        )
        predictions["mid"] = mid_forecast
    else:
        mid_context = None

    if "short" in expert_names:
        long_context = _build_long_context(
            long_target_repr,
            long_global_context,
            long_forecast,
        )
        short_forecast, short_recent, short_global = _build_short_expert(
            inputs,
            input_dim,
            target_channel_index,
            long_forecast,
            long_context,
        )
        predictions["short"] = short_forecast
    else:
        short_recent = None
        short_global = None

    if "persistence" in expert_names:
        predictions["persistence"] = _build_persistence_expert(
            inputs,
            target_channel_index,
            power_scale_ratio,
            power_scale_offset,
        )

    if len(expert_names) == 1:
        outputs = layers.Activation("linear", name="forecast_power")(
            predictions[expert_names[0]]
        )
    else:
        initial_bias = [EXPERT_INITIAL_BIAS[name] for name in expert_names]
        _reset_branch_seed()
        if spec["router_type"] == "static_horizon_softmax":
            router_weights = StaticHorizonRouter(
                forecast_len=FORECAST_LEN,
                n_experts=len(expert_names),
                initial_bias=initial_bias,
                name="expert_router",
            )(inputs)
        elif spec["router_type"] == "sample_horizon_dense_softmax":
            if long_context is None:
                long_context = _build_long_context(
                    long_target_repr,
                    long_global_context,
                    long_forecast,
                )
            router_contexts.append(long_context)
            if mid_context is not None:
                router_contexts.append(
                    layers.Dense(
                        LONG_CONTEXT_DIM,
                        activation="gelu",
                        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
                        name="mid_router_context_projection",
                    )(mid_context)
                )
            if short_recent is not None:
                local_router_context = layers.Concatenate(
                    name="local_router_context_features",
                )([short_recent, short_global])
                router_contexts.append(
                    layers.Dense(
                        LONG_CONTEXT_DIM,
                        activation="gelu",
                        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
                        name="local_router_context_projection",
                    )(local_router_context)
                )
            router_contexts.append(
                TakeLastToken(name="router_last_history_features")(inputs)
            )
            router_context = layers.Concatenate(
                name="router_context_features",
            )(router_contexts)
            router_weights = HorizonRegimeRouter(
                forecast_len=FORECAST_LEN,
                n_experts=len(expert_names),
                hidden_dim=ROUTER_HIDDEN_DIM,
                horizon_embedding_dim=HORIZON_EMBEDDING_DIM,
                dropout=ROUTER_DROPOUT,
                initial_bias=initial_bias,
                name="expert_router",
            )(router_context)
        else:
            raise ValueError(f"不支持的 router: {spec['router_type']}")

        outputs = ExpertConvexFusion(
            n_experts=len(expert_names),
            name="forecast_power",
        )(
            [
                *[predictions[name] for name in expert_names],
                router_weights,
            ]
        )

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"WindFeTSPatchTSTMin_{variant_id}",
    )
    # 分支初始化后恢复统一训练随机流；数据 shuffle 也显式使用同一 seed。
    set_global_seed(RANDOM_SEED)
    return _compile_model(model)


def build_fets_patchtst_min_model_from_artifact(artifact):
    if artifact.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(
            "artifact 架构版本不匹配: "
            f"{artifact.get('architecture_version')} != {ARCHITECTURE_VERSION}"
        )
    variant_id = artifact["variant_id"]
    return build_fets_patchtst_min_model(
        variant_id=variant_id,
        input_dim=len(artifact["input_cols"]),
        target_channel_index=int(artifact["target_index"]),
        power_scale_ratio=float(artifact["power_scale_ratio"]),
        power_scale_offset=float(artifact["power_scale_offset"]),
    )


def get_min_custom_objects():
    return {
        "StaticHorizonRouter": StaticHorizonRouter,
        "WindFeTSPatchTSTMin>StaticHorizonRouter": StaticHorizonRouter,
    }


def _prepare_farm(train_file):
    farm_id = get_farm_id(train_file)
    train_df, feature_cols, capacity = load_and_preprocess(
        train_file,
        is_train=True,
    )
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
    return {
        "farm_id": farm_id,
        "train_file": train_file,
        "train_df": train_df,
        "feature_cols": feature_cols,
        "capacity": capacity,
        "features": features,
        "target": target,
        "input_cols": input_cols,
        "target_index": target_index,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "power_scale_ratio": ratio,
        "power_scale_offset": offset,
    }


def _inverse_power(scaler_y, values):
    values = np.asarray(values).reshape(-1, 1)
    return scaler_y.inverse_transform(values).reshape(-1)


def _physical_metrics(y_true_scaled, y_pred_scaled, scaler_y, capacity):
    y_true = _inverse_power(scaler_y, y_true_scaled)
    y_pred = _inverse_power(scaler_y, y_pred_scaled)
    y_pred = np.clip(y_pred, 0, capacity if capacity is not None else None)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "val_mae": mae,
        "val_rmse": rmse,
        "val_capacity_normalized_mae": (
            mae / capacity if capacity is not None and capacity > 0 else np.nan
        ),
        "val_capacity_normalized_rmse": (
            rmse / capacity if capacity is not None and capacity > 0 else np.nan
        ),
    }


def _evaluate_model(model, val_ds, scaler_y, capacity):
    y_true_scaled = np.concatenate(
        [batch_y.numpy() for _, batch_y in val_ds],
        axis=0,
    )
    y_pred_scaled = np.asarray(model.predict(val_ds, verbose=0), dtype=float)
    if not np.isfinite(y_pred_scaled).all():
        raise FloatingPointError("验证预测包含非有限值")
    return _physical_metrics(
        y_true_scaled,
        y_pred_scaled,
        scaler_y,
        capacity,
    )


def _evaluate_analytic_persistence(
    val_ds,
    target_index,
    forecast_len,
    scale_ratio,
    scale_offset,
    scaler_y,
    capacity,
):
    true_batches = []
    pred_batches = []
    for batch_x, batch_y in val_ds:
        last_x = batch_x[:, -1, target_index].numpy()
        last_y = last_x * scale_ratio + scale_offset
        pred = np.repeat(last_y[:, None], forecast_len, axis=1)
        true_batches.append(batch_y.numpy())
        pred_batches.append(pred)
    return _physical_metrics(
        np.concatenate(true_batches, axis=0),
        np.concatenate(pred_batches, axis=0),
        scaler_y,
        capacity,
    )


def _collect_router_statistics(model, val_ds, router_layer_name, expert_names):
    if not router_layer_name or len(expert_names) <= 1:
        return None
    router_model = keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer(router_layer_name).output,
    )
    weights = np.asarray(router_model.predict(val_ds, verbose=0), dtype=float)
    if weights.ndim != 3 or weights.shape[-1] != len(expert_names):
        raise ValueError(f"router 输出形状异常: {weights.shape}")
    if not np.isfinite(weights).all():
        raise FloatingPointError("router 权重包含非有限值")
    if not np.allclose(weights.sum(axis=-1), 1.0, atol=1e-5):
        raise ValueError("router 权重之和不为1")
    entropy = -np.sum(
        weights * np.log(np.clip(weights, 1e-8, 1.0)),
        axis=-1,
    ) / np.log(len(expert_names))
    return {
        "overall_mean": weights.mean(axis=(0, 1)),
        "overall_std": weights.std(axis=(0, 1)),
        "mean_by_horizon": weights.mean(axis=0),
        "std_by_horizon": weights.std(axis=0),
        "normalized_entropy_mean": float(entropy.mean()),
    }


def _save_history(history, dirs, model_name, farm_id):
    history_df = pd.DataFrame(history.history)
    history_df.index = np.arange(1, len(history_df) + 1)
    history_df.index.name = "epoch"
    history_path = os.path.join(
        dirs["history"],
        f"{model_name}_history_farm_{farm_id}.csv",
    )
    history_df.to_csv(history_path, encoding="utf-8-sig")
    plot_path = os.path.join(
        dirs["history"],
        f"{model_name}_history_farm_{farm_id}.png",
    )
    try:
        cache_dir = os.path.join(dirs["root"], "matplotlib_cache")
        os.environ["MPLCONFIGDIR"] = cache_dir
        os.environ["XDG_CACHE_HOME"] = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = [
            name
            for name in history_df.columns
            if not name.startswith("val_") and f"val_{name}" in history_df.columns
        ]
        fig, axes = plt.subplots(
            max(1, len(metrics)),
            1,
            figsize=(10, max(3, 2.8 * max(1, len(metrics)))),
            sharex=True,
        )
        axes = np.atleast_1d(axes)
        for ax, metric in zip(axes, metrics):
            ax.plot(history_df.index, history_df[metric], label=metric)
            ax.plot(
                history_df.index,
                history_df[f"val_{metric}"],
                label=f"val_{metric}",
            )
            ax.set_title(metric)
            ax.grid(alpha=0.3)
            ax.legend()
        axes[-1].set_xlabel("epoch")
        fig.suptitle(f"{model_name} - Farm {farm_id}", y=1.0)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"{model_name} 训练曲线保存失败: {exc}")
        plot_path = None
    return history_path, plot_path


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
    if os.getenv("WIND_FETS_MIN_SAVE_SMOKE_TEST", "1") == "0":
        return
    sample_x, _ = next(iter(val_ds))
    sample_x = sample_x[:2]
    expected = np.asarray(model(sample_x, training=False), dtype=float)
    restored = keras.models.load_model(
        model_path,
        custom_objects=get_min_custom_objects(),
        compile=False,
    )
    actual = np.asarray(restored(sample_x, training=False), dtype=float)
    if not np.allclose(expected, actual, rtol=1e-6, atol=1e-6):
        raise ValueError("保存后重新加载的模型输出不一致")
    del restored


def train_variant_for_farm(variant_id, prepared):
    keras.backend.clear_session()
    configure_reproducibility()
    spec = VARIANT_SPECS[variant_id]
    model_name = variant_model_name(variant_id)
    dirs = variant_dirs(variant_id)
    farm_id = prepared["farm_id"]
    paths = _train_paths(dirs, model_name, farm_id)
    print(f"\n===== {spec['label']} / 风电场 {farm_id} / seed=2026 =====")

    # 每个变体重新创建带 shuffle 的训练集，避免前一模型的 epoch/早停次数
    # 推进 tf.data 的内部迭代状态而破坏公平性。
    train_ds, val_ds, train_samples, total_samples = make_window_dataset(
        prepared["features"],
        prepared["target"],
        HISTORY_LEN,
        FORECAST_LEN,
        BATCH_SIZE,
        VALIDATION_SPLIT,
    )

    start_time = time.monotonic()
    history_path = None
    history_plot_path = None
    tensorboard_log_dir = None
    model_path = None
    best_weights_path = None
    router_statistics = None

    if variant_id == "b0_persistence":
        metrics = _evaluate_analytic_persistence(
            val_ds,
            prepared["target_index"],
            FORECAST_LEN,
            prepared["power_scale_ratio"],
            prepared["power_scale_offset"],
            prepared["scaler_y"],
            prepared["capacity"],
        )
        total_params = 0
        trainable_params = 0
        model_size_bytes = 0
        training_mode = "analytic_baseline"
        requires_keras_model = False
    else:
        model = build_fets_patchtst_min_model(
            variant_id,
            len(prepared["input_cols"]),
            prepared["target_index"],
            prepared["power_scale_ratio"],
            prepared["power_scale_offset"],
        )
        total_params = int(model.count_params())
        trainable_params = int(
            sum(np.prod(variable.shape) for variable in model.trainable_weights)
        )
        model_path = paths["model_path"]
        best_weights_path = paths["best_weights_path"]
        tensorboard_log_dir = os.path.join(
            dirs["tensorboard"],
            f"farm_{farm_id}",
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
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                best_weights_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
        ]
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1,
        )
        ensure_finite_training_history(history, guard)
        history_path, history_plot_path = _save_history(
            history,
            dirs,
            model_name,
            farm_id,
        )
        if not os.path.exists(best_weights_path):
            raise FileNotFoundError(f"未生成最佳权重: {best_weights_path}")
        model.load_weights(best_weights_path)
        metrics = _evaluate_model(
            model,
            val_ds,
            prepared["scaler_y"],
            prepared["capacity"],
        )
        router_layer_name = "expert_router" if spec["router_type"] != "none" else None
        router_statistics = _collect_router_statistics(
            model,
            val_ds,
            router_layer_name,
            spec["experts"],
        )
        model.save(model_path)
        _save_load_smoke_test(model, model_path, val_ds)
        model_size_bytes = os.path.getsize(model_path)
        training_mode = "keras_fit"
        requires_keras_model = True

    elapsed_seconds = float(time.monotonic() - start_time)
    prepared["train_df"].iloc[-HISTORY_LEN:].to_csv(
        paths["tail_path"],
        index=True,
    )

    router_layer_name = "expert_router" if spec["router_type"] != "none" else None
    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "model_name": model_name,
        "variant_id": variant_id,
        "variant_label": spec["label"],
        "variant_config": {
            **spec,
            "experts": list(spec["experts"]),
        },
        "architecture_version": ARCHITECTURE_VERSION,
        "farm_id": farm_id,
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
        "training_mode": training_mode,
        "requires_keras_model": requires_keras_model,
        "model_kind": (
            "analytic_persistence"
            if variant_id == "b0_persistence"
            else "keras_network"
        ),
        "expert_names": list(spec["experts"]),
        "expert_output_layer_names": {
            name: EXPERT_OUTPUT_LAYER_NAMES[name] for name in spec["experts"]
        },
        "fusion_type": spec["fusion_type"],
        "router_type": spec["router_type"],
        "router_layer_name": router_layer_name,
        "router_initial_bias": (
            [EXPERT_INITIAL_BIAS[name] for name in spec["experts"]]
            if router_layer_name
            else None
        ),
        "power_scale_ratio": prepared["power_scale_ratio"],
        "power_scale_offset": prepared["power_scale_offset"],
        "patch_len": PATCH_LEN,
        "patch_stride": PATCH_STRIDE,
        "mid_patch_len": MID_PATCH_LEN,
        "mid_patch_stride": MID_PATCH_STRIDE,
        "mid_n_layers": MID_N_LAYERS,
        "local_patch_len": LOCAL_PATCH_LEN,
        "local_patch_stride": LOCAL_PATCH_STRIDE,
        "local_n_layers": LOCAL_N_LAYERS,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "dropout": DROPOUT,
        "head_dropout": HEAD_DROPOUT,
        "fourier_degree": FOURIER_DEGREE,
        "poly_degree": POLYNOMIAL_DEGREE,
        "ffn_ratio": FFN_RATIO,
        "target_weather_heads": TARGET_WEATHER_HEADS,
        "layer_scale_init": LAYER_SCALE_INIT,
        "long_context_dim": LONG_CONTEXT_DIM,
        "router_hidden_dim": ROUTER_HIDDEN_DIM,
        "horizon_embedding_dim": HORIZON_EMBEDDING_DIM,
        "router_dropout": ROUTER_DROPOUT,
        "correction_kernel_l2": CORRECTION_KERNEL_L2,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "validation_split": VALIDATION_SPLIT,
        "learning_rate": LEARNING_RATE,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_bytes": model_size_bytes,
        "training_elapsed_seconds": elapsed_seconds,
        "train_samples": train_samples,
        "val_samples": total_samples - train_samples,
        "model_path": model_path,
        "best_weights_path": best_weights_path,
        "artifact_path": paths["artifact_path"],
        "history_path": history_path,
        "history_plot_path": history_plot_path,
        "tensorboard_log_dir": tensorboard_log_dir,
        "tail_path": paths["tail_path"],
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(keras, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        **metrics,
    }
    if router_statistics is not None:
        artifact.update(
            {
                "router_validation_overall_mean": (
                    router_statistics["overall_mean"].tolist()
                ),
                "router_validation_overall_std": (
                    router_statistics["overall_std"].tolist()
                ),
                "router_validation_mean_by_horizon": (
                    router_statistics["mean_by_horizon"].tolist()
                ),
                "router_validation_std_by_horizon": (
                    router_statistics["std_by_horizon"].tolist()
                ),
                "router_validation_normalized_entropy_mean": (
                    router_statistics["normalized_entropy_mean"]
                ),
            }
        )
    joblib.dump(artifact, paths["artifact_path"])

    result = {
        "model_family": MODEL_FAMILY,
        "model_name": model_name,
        "variant_id": variant_id,
        "variant_label": spec["label"],
        "farm_id": farm_id,
        "random_seed": RANDOM_SEED,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_bytes": model_size_bytes,
        "training_elapsed_seconds": elapsed_seconds,
        "train_samples": train_samples,
        "val_samples": total_samples - train_samples,
        **metrics,
        "model_path": model_path,
        "best_weights_path": best_weights_path,
        "artifact_path": paths["artifact_path"],
        "history_path": history_path,
        "history_plot_path": history_plot_path,
        "tail_path": paths["tail_path"],
    }
    if router_statistics is not None:
        result["router_normalized_entropy"] = router_statistics[
            "normalized_entropy_mean"
        ]
        result.update(
            {
                f"router_weight_{name}": float(weight)
                for name, weight in zip(
                    spec["experts"],
                    router_statistics["overall_mean"],
                )
            }
        )
    print(
        f"{model_name} / {farm_id}: val NRMSE="
        f"{metrics['val_capacity_normalized_rmse']:.6f}, "
        f"params={total_params:,}"
    )
    if variant_id != "b0_persistence":
        del model
    keras.backend.clear_session()
    return result


def _write_variant_manifest():
    rows = []
    for variant_id, spec in VARIANT_SPECS.items():
        rows.append(
            {
                "variant_id": variant_id,
                "model_name": variant_model_name(variant_id),
                "label": spec["label"],
                "experts": "+".join(spec["experts"]),
                "fusion_type": spec["fusion_type"],
                "router_type": spec["router_type"],
                "description": spec["description"],
                "random_seed": RANDOM_SEED,
            }
        )
    os.makedirs(RESULT_ROOT, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULT_ROOT, "stage1_variant_manifest.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def main():
    _write_variant_manifest()
    variants = get_requested_variants()
    train_files = discover_train_files(DATA_DIR)
    if not train_files:
        raise FileNotFoundError(f"未在 {DATA_DIR} 找到训练文件")
    print(f"固定主随机种子: {RANDOM_SEED}")
    print(f"场站数: {len(train_files)}；变体: {variants}")

    all_results = []
    for train_file in train_files:
        prepared = _prepare_farm(train_file)
        for variant_id in variants:
            all_results.append(train_variant_for_farm(variant_id, prepared))
            # 长批量实验中断时保留已经完成的组合汇总；单场站 artifact 和模型
            # 在各自训练结束时已经生成，不依赖整个矩阵全部完成。
            pd.DataFrame(all_results).to_csv(
                os.path.join(RESULT_ROOT, "stage1_training_metrics_partial.csv"),
                index=False,
                encoding="utf-8-sig",
            )

    metrics_df = pd.DataFrame(all_results)
    metrics_path = os.path.join(
        RESULT_ROOT,
        "stage1_training_metrics.csv",
    )
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    for variant_id, frame in metrics_df.groupby("variant_id"):
        frame.to_csv(
            os.path.join(
                variant_dirs(variant_id)["root"],
                f"{variant_model_name(variant_id)}_training_metrics.csv",
            ),
            index=False,
            encoding="utf-8-sig",
        )
    print(f"第一阶段训练完成: {metrics_path}")


if __name__ == "__main__":
    main()
