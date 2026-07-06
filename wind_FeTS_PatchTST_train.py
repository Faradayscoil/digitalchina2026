"""FeTS-PatchTST 在风电超短期功率预测任务上的训练入口。

模型完整保留原生 PatchTST 长尺度分支，并增加中尺度 PatchTST 与局部 FeTS
两个候选专家：

    原生长分支: patch(16, 8) -> PatchTST -> baseline forecast
    中尺度分支: patch(8, 4) -> PatchTST -> baseline + mid residual
    局部 FeTS: patch(4, 2) -> channel embedding
               -> AdaFE/DSFFN LayerScale residual
               -> power-query/weather-key-value attention
               -> local temporal encoder
               -> long-context fusion -> baseline + local residual
    持续性专家: 最后历史功率在目标标准化空间重复 16 步
    最终输出: 输入状态与 horizon 条件化的 softmax router 对四个专家凸融合

模型复用原生 PatchTST 基线的数据预处理、历史窗口、时序编码器、预测 head、
验证划分、优化器和损失。此文件不包含 k-fold、多随机种子、RevIN、
自蒸馏、外部 teacher、频谱路由、尺度相似度、top-k 稀疏化或双向尺度交互。
"""

import glob
import os
import re
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from wind_dl_model_train import (
    BATCH_SIZE as BASELINE_BATCH_SIZE,
    DATA_DIR,
    D_FF as BASELINE_D_FF,
    D_MODEL as BASELINE_D_MODEL,
    DROPOUT as BASELINE_DROPOUT,
    EPOCHS as BASELINE_EPOCHS,
    FORECAST_LEN,
    HEAD_DROPOUT as BASELINE_HEAD_DROPOUT,
    HISTORY_LEN,
    LEARNING_RATE as BASELINE_LEARNING_RATE,
    N_HEADS as BASELINE_N_HEADS,
    N_LAYERS as BASELINE_N_LAYERS,
    PATCH_LEN as BASELINE_PATCH_LEN,
    PATCH_STRIDE as BASELINE_PATCH_STRIDE,
    TARGET_COL,
    TIME_FREQ,
    VALIDATION_SPLIT as BASELINE_VALIDATION_SPLIT,
    LearnablePositionEmbedding,
    MergeChannels,
    PatchExtract,
    RestoreChannels,
    TakeChannel,
    build_scaled_arrays,
    load_and_preprocess,
    make_window_dataset,
    transformer_encoder,
)

warnings.filterwarnings('ignore')


MODEL_NAME = 'fets_patchtst'
ARCHITECTURE_VERSION = 'fets_patchtst_horizon_regime_moe_v5ab'
OFFICIAL_REPOSITORY = 'https://github.com/lllucky111/FeTS'
OFFICIAL_REVISION = 'd908e434b70f3cf69065004e295db13cdb9790b2'
TRAIN_FILE_PATTERN = 'wind_train_*.csv'

MODEL_DIR = os.path.join('./wind_results', MODEL_NAME)
SAVED_MODEL_DIR = os.path.join(MODEL_DIR, 'models')
WEIGHTS_DIR = os.path.join(MODEL_DIR, 'weights')
PREPROCESS_DIR = os.path.join(MODEL_DIR, 'preprocess')
HISTORY_DIR = os.path.join(MODEL_DIR, 'history')
TENSORBOARD_LOG_DIR = os.path.join(MODEL_DIR, 'tensorboard')
TAIL_DIR = os.path.join(MODEL_DIR, 'tails')

BATCH_SIZE = int(os.getenv('WIND_FETS_BATCH_SIZE', str(BASELINE_BATCH_SIZE)))
EPOCHS = int(os.getenv('WIND_FETS_EPOCHS', str(BASELINE_EPOCHS)))
VALIDATION_SPLIT = float(
    os.getenv('WIND_FETS_VALIDATION_SPLIT', str(BASELINE_VALIDATION_SPLIT))
)
LEARNING_RATE = float(
    os.getenv('WIND_FETS_LEARNING_RATE', str(BASELINE_LEARNING_RATE))
)

# 与 817fe4... 原生 PatchTST 保持一致，避免同时改变 patch 和表示宽度。
PATCH_LEN = BASELINE_PATCH_LEN
PATCH_STRIDE = BASELINE_PATCH_STRIDE
D_MODEL = BASELINE_D_MODEL
N_HEADS = BASELINE_N_HEADS
N_LAYERS = BASELINE_N_LAYERS
D_FF = BASELINE_D_FF
DROPOUT = BASELINE_DROPOUT
HEAD_DROPOUT = BASELINE_HEAD_DROPOUT

# 局部分支使用 1 小时 patch、30 分钟步长，保留 patch 内 15 分钟细节。
LOCAL_PATCH_LEN = 4
LOCAL_PATCH_STRIDE = 2
LOCAL_N_LAYERS = 2

# 中尺度专家填补局部 4/2 与长尺度 16/8 之间的表示空白。
MID_PATCH_LEN = 8
MID_PATCH_STRIDE = 4
MID_N_LAYERS = 2

TARGET_WEATHER_HEADS = N_HEADS
LAYER_SCALE_INIT = 1e-3
LONG_CONTEXT_DIM = D_MODEL

# v5-A 使用样本状态与 horizon 共同决定四个专家的凸融合权重。router 输出层
# zero-init，使初始权重严格由 bias 控制并以原生长尺度专家为主。
EXPERT_NAMES = ('long', 'mid', 'short', 'persistence')
ROUTER_HIDDEN_DIM = 64
HORIZON_EMBEDDING_DIM = 16
ROUTER_DROPOUT = 0.1
ROUTER_INITIAL_BIAS = (2.0, 0.0, 0.0, -2.0)

CORRECTION_KERNEL_L2 = 1e-4

# FeTS 官方 Fourier/polynomial mask 和 DSFFN 配置。
FOURIER_DEGREE = 2
POLYNOMIAL_DEGREE = 2
ADAFE_KERNEL_SIZE = 5
ADAFE_PADDING = 2
FFN_RATIO = 2
# 仅限制直通估计器的反向代理幅度；官方二值 mask 的前向结果不变。
SOFT_MASK_LOGIT_CLIP = 20.0


def discover_train_files(data_dir=DATA_DIR):
    """查找按场站拆分的训练文件。"""
    return sorted(glob.glob(os.path.join(data_dir, TRAIN_FILE_PATTERN)))


def get_farm_id(path):
    basename = os.path.basename(path)
    match = re.search(r'wind_train_(\d+)\.csv$', basename)
    if match:
        return match.group(1)
    return os.path.splitext(basename)[0]


def compute_patch_num(context_window, patch_len, stride):
    """复现官方 ReplicationPad1d + unfold 后的 patch 数量。"""
    effective_len = context_window + stride if patch_len != stride else context_window
    return (effective_len - patch_len) // stride + 1


@keras.utils.register_keras_serializable(package='WindFeTSPatchTST')
class FeTSPatchExtract(layers.Layer):
    """官方 FeTS 的通道独立 patch 切分和尾部复制填充。"""

    def __init__(self, patch_len, stride, **kwargs):
        super().__init__(**kwargs)
        if patch_len <= 0 or stride <= 0:
            raise ValueError('patch_len 和 stride 必须为正整数')
        self.patch_len = int(patch_len)
        self.stride = int(stride)

    def call(self, inputs):
        # [batch, history, channel] -> [batch, channel, history]
        x = tf.transpose(inputs, [0, 2, 1])
        if self.patch_len != self.stride:
            x = tf.concat(
                [x, tf.repeat(x[:, :, -1:], repeats=self.stride, axis=-1)],
                axis=-1,
            )
        return tf.signal.frame(
            x,
            frame_length=self.patch_len,
            frame_step=self.stride,
            axis=-1,
        )

    def compute_output_shape(self, input_shape):
        patch_num = None
        if input_shape[1] is not None:
            patch_num = compute_patch_num(
                input_shape[1],
                self.patch_len,
                self.stride,
            )
        return input_shape[0], input_shape[2], patch_num, self.patch_len

    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_len': self.patch_len,
            'stride': self.stride,
        })
        return config


@keras.utils.register_keras_serializable(package='WindFeTSPatchTST')
class FourierPolynomialMask(layers.Layer):
    """Fourier 与 polynomial 基函数生成的 FeTS 可学习 mask。"""

    def __init__(
        self,
        input_dim,
        output_dim,
        fourier_degree,
        poly_degree,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError('input_dim 和 output_dim 必须为正整数')
        if fourier_degree <= 0 or poly_degree < 0:
            raise ValueError('fourier_degree 必须为正，poly_degree 不能为负')

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.fourier_degree = int(fourier_degree)
        self.poly_degree = int(poly_degree)
        self.interaction = layers.Dense(self.output_dim, name='interaction')

    def build(self, input_shape):
        self.cos_coeffs = self.add_weight(
            name='cos_coeffs',
            shape=(self.input_dim, self.output_dim, self.fourier_degree + 1),
            initializer=keras.initializers.RandomNormal(
                stddev=1.0 / (self.input_dim * (self.fourier_degree + 1))
            ),
            trainable=True,
        )
        self.sin_coeffs = self.add_weight(
            name='sin_coeffs',
            shape=(self.input_dim, self.output_dim, self.fourier_degree),
            initializer=keras.initializers.RandomNormal(
                stddev=1.0 / (self.input_dim * self.fourier_degree)
            ),
            trainable=True,
        )
        self.poly_coeffs = self.add_weight(
            name='poly_coeffs',
            shape=(self.input_dim, self.output_dim, self.poly_degree + 1),
            # 与官方表达式 1 / input_dim * (poly_degree + 1) 保持一致。
            initializer=keras.initializers.RandomNormal(
                stddev=(self.poly_degree + 1) / self.input_dim
            ),
            trainable=True,
        )
        self.mask_bias = self.add_weight(
            name='mask_bias',
            shape=(1, self.output_dim),
            initializer='zeros',
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        dtype = inputs.dtype
        pi = tf.cast(np.pi, dtype)
        k_cos = tf.cast(tf.range(self.fourier_degree + 1), dtype)
        k_sin = tf.cast(tf.range(1, self.fourier_degree + 1), dtype)

        expanded = inputs[..., None]
        x_cos = tf.cos(expanded * k_cos * pi)
        x_sin = tf.sin(expanded * k_sin * pi)
        # 不使用 tf.pow(x, 0)：TensorFlow 在 x == 0 时对该表达式的反向
        # 梯度是 NaN。逐次乘法与 [x^0, x^1, ..., x^p] 前向等价，且零点
        # 梯度有限。
        poly_terms = [tf.ones_like(inputs)]
        current_power = tf.ones_like(inputs)
        for _ in range(self.poly_degree):
            current_power = current_power * inputs
            poly_terms.append(current_power)
        x_poly = tf.stack(poly_terms, axis=-1)

        y_cos = tf.einsum('bid,iod->bo', x_cos, self.cos_coeffs)
        y_sin = tf.einsum('bid,iod->bo', x_sin, self.sin_coeffs)
        y_poly = tf.einsum('bid,iod->bo', x_poly, self.poly_coeffs)
        mask = y_cos + y_sin + y_poly + self.mask_bias
        return self.interaction(mask)

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'fourier_degree': self.fourier_degree,
            'poly_degree': self.poly_degree,
        })
        return config


@keras.utils.register_keras_serializable(package='WindFeTSPatchTST')
class AdaptiveFeatureExtraction(layers.Layer):
    """AdaFE：使用动态二值 mask 在表示维度上进行局部滑窗聚合。"""

    def __init__(
        self,
        d_model,
        fourier_degree,
        poly_degree,
        kernel_size=5,
        padding=2,
        soft_mask_logit_clip=SOFT_MASK_LOGIT_CLIP,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if kernel_size <= 0 or padding < 0:
            raise ValueError('kernel_size 必须为正，padding 不能为负')
        if 2 * padding != kernel_size - 1:
            raise ValueError('AdaFE 需要 2 * padding == kernel_size - 1 以保持维度')
        if soft_mask_logit_clip <= 0:
            raise ValueError('soft_mask_logit_clip 必须为正数')

        self.d_model = int(d_model)
        self.fourier_degree = int(fourier_degree)
        self.poly_degree = int(poly_degree)
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        self.soft_mask_logit_clip = float(soft_mask_logit_clip)
        self.mask = FourierPolynomialMask(
            self.d_model,
            self.d_model,
            self.fourier_degree,
            self.poly_degree,
            name='fourier_polynomial_mask',
        )

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.kernel_size,),
            initializer=keras.initializers.RandomNormal(),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        mask = self.mask(inputs)
        # inputs 的每一行对应一个独立 patch；只在该 patch 的表示维度内计算
        # 阈值，避免同一样本的预测随 batch 组成或 batch size 改变。
        threshold = tf.reduce_mean(mask, axis=-1, keepdims=True)
        hard_active = tf.cast(mask > threshold, inputs.dtype)
        # 前向仍是官方二值 mask；直通估计器让 Fourier/polynomial 参数能够
        # 获得梯度，避免框架迁移后这些“可学习”参数实际被冻结。
        centered_mask = tf.clip_by_value(
            mask - threshold,
            tf.cast(-self.soft_mask_logit_clip, inputs.dtype),
            tf.cast(self.soft_mask_logit_clip, inputs.dtype),
        )
        soft_active = tf.sigmoid(centered_mask)
        active = soft_active + tf.stop_gradient(hard_active - soft_active)

        paddings = [[0, 0], [self.padding, self.padding]]
        x_pad = tf.pad(inputs, paddings)
        active_pad = tf.pad(active, paddings)
        x_windows = tf.signal.frame(
            x_pad,
            frame_length=self.kernel_size,
            frame_step=1,
            axis=1,
        )
        active_windows = tf.signal.frame(
            active_pad,
            frame_length=self.kernel_size,
            frame_step=1,
            axis=1,
        )
        return tf.einsum('blk,k->bl', x_windows * active_windows, self.kernel)

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'fourier_degree': self.fourier_degree,
            'poly_degree': self.poly_degree,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'soft_mask_logit_clip': self.soft_mask_logit_clip,
        })
        return config


@keras.utils.register_keras_serializable(package='WindFeTSPatchTST')
class DualScaleFeedForward(layers.Layer):
    """DSFFN：融合逐 patch 局部表示与 patch 维全局均值。"""

    def __init__(self, d_model, ffn_ratio, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.ffn_ratio = int(ffn_ratio)
        self.dropout_rate = float(dropout)
        d_ff = self.d_model * self.ffn_ratio

        self.pointwise_in = layers.Conv1D(
            d_ff,
            kernel_size=1,
            name='pointwise_in',
        )
        self.activation = layers.Activation('gelu', name='gelu')
        self.combine = layers.Conv1D(
            d_ff,
            kernel_size=1,
            name='combine',
        )
        self.pointwise_out = layers.Conv1D(
            self.d_model,
            kernel_size=1,
            name='pointwise_out',
        )
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=None):
        # inputs: [batch, channel, d_model, patch_num]
        shape = tf.shape(inputs)
        batch_size, n_channels, patch_num = shape[0], shape[1], shape[3]
        x = tf.transpose(inputs, [0, 1, 3, 2])
        x = tf.reshape(x, [batch_size * n_channels, patch_num, self.d_model])

        local_features = self.activation(self.pointwise_in(x))
        global_features = tf.reduce_mean(x, axis=1, keepdims=True)
        global_features = tf.repeat(global_features, repeats=patch_num, axis=1)
        combined = tf.concat([local_features, global_features], axis=-1)

        x = self.combine(combined)
        x = self.pointwise_out(x)
        x = self.dropout(x, training=training)
        x = tf.reshape(
            x,
            [batch_size, n_channels, patch_num, self.d_model],
        )
        return tf.transpose(x, [0, 1, 3, 2])

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'ffn_ratio': self.ffn_ratio,
            'dropout': self.dropout_rate,
        })
        return config


@keras.utils.register_keras_serializable(package='WindFeTSPatchTST')
class FeTSFeatureBlock(layers.Layer):
    """第一轮模型的 FeTS 处理块，仅用于旧模型反序列化兼容。"""

    def __init__(
        self,
        d_model,
        fourier_degree,
        poly_degree,
        ffn_ratio,
        dropout,
        kernel_size=5,
        padding=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.fourier_degree = int(fourier_degree)
        self.poly_degree = int(poly_degree)
        self.ffn_ratio = int(ffn_ratio)
        self.dropout_rate = float(dropout)
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)

        self.adafe = AdaptiveFeatureExtraction(
            self.d_model,
            self.fourier_degree,
            self.poly_degree,
            kernel_size=self.kernel_size,
            padding=self.padding,
            name='adafe',
        )
        self.layer_norm = layers.LayerNormalization(
            epsilon=1e-5,
            name='feature_layer_norm',
        )
        self.dsffn = DualScaleFeedForward(
            self.d_model,
            self.ffn_ratio,
            dropout=self.dropout_rate,
            name='dsffn',
        )

    def call(self, inputs, training=None):
        # patch projection 输入为 [batch, channel, patch_num, d_model]。
        shape = tf.shape(inputs)
        batch_size, n_channels, patch_num = shape[0], shape[1], shape[2]
        residual = tf.transpose(inputs, [0, 1, 3, 2])

        x = tf.reshape(inputs, [-1, self.d_model])
        x = self.adafe(x)
        x = self.layer_norm(x)
        x = tf.reshape(
            x,
            [batch_size, n_channels, patch_num, self.d_model],
        )
        x = tf.transpose(x, [0, 1, 3, 2])
        x = self.dsffn(x, training=training)
        return x + residual

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'fourier_degree': self.fourier_degree,
            'poly_degree': self.poly_degree,
            'ffn_ratio': self.ffn_ratio,
            'dropout': self.dropout_rate,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
        })
        return config


@keras.utils.register_keras_serializable(package='WindFeTSPatchTST')
class ChannelIdentityEmbedding(layers.Layer):
    """为每个输入变量添加可学习身份，避免跨变量注意力丢失特征语义。"""

    def __init__(self, n_channels, d_model, init_stddev=0.02, **kwargs):
        super().__init__(**kwargs)
        if n_channels <= 0 or d_model <= 0:
            raise ValueError('n_channels 和 d_model 必须为正整数')
        self.n_channels = int(n_channels)
        self.d_model = int(d_model)
        self.init_stddev = float(init_stddev)

    def build(self, input_shape):
        if input_shape[1] is not None and int(input_shape[1]) != self.n_channels:
            raise ValueError(
                f'输入通道数 {input_shape[1]} 与配置 {self.n_channels} 不一致'
            )
        if input_shape[-1] is not None and int(input_shape[-1]) != self.d_model:
            raise ValueError(
                f'输入表示维度 {input_shape[-1]} 与配置 {self.d_model} 不一致'
            )
        self.channel_embedding = self.add_weight(
            name='channel_embedding',
            shape=(1, self.n_channels, 1, self.d_model),
            initializer=keras.initializers.RandomNormal(
                stddev=self.init_stddev,
            ),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + tf.cast(self.channel_embedding, inputs.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_channels': self.n_channels,
            'd_model': self.d_model,
            'init_stddev': self.init_stddev,
        })
        return config


@keras.utils.register_keras_serializable(package='WindFeTSPatchTST')
class LayerScaleFeTSFeatureBlock(layers.Layer):
    """AdaFE/DSFFN 适配器，以小幅 LayerScale 残差保护原始 patch 表示。"""

    def __init__(
        self,
        d_model,
        fourier_degree,
        poly_degree,
        ffn_ratio,
        dropout,
        layer_scale_init=1e-3,
        kernel_size=5,
        padding=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if layer_scale_init < 0:
            raise ValueError('layer_scale_init 不能为负数')
        self.d_model = int(d_model)
        self.fourier_degree = int(fourier_degree)
        self.poly_degree = int(poly_degree)
        self.ffn_ratio = int(ffn_ratio)
        self.dropout_rate = float(dropout)
        self.layer_scale_init = float(layer_scale_init)
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)

        self.adafe = AdaptiveFeatureExtraction(
            self.d_model,
            self.fourier_degree,
            self.poly_degree,
            kernel_size=self.kernel_size,
            padding=self.padding,
            name='adafe',
        )
        self.layer_norm = layers.LayerNormalization(
            epsilon=1e-5,
            name='feature_layer_norm',
        )
        self.dsffn = DualScaleFeedForward(
            self.d_model,
            self.ffn_ratio,
            dropout=self.dropout_rate,
            name='dsffn',
        )

    def build(self, input_shape):
        if input_shape[-1] is not None and int(input_shape[-1]) != self.d_model:
            raise ValueError(
                f'输入表示维度 {input_shape[-1]} 与配置 {self.d_model} 不一致'
            )
        self.layer_scale = self.add_weight(
            name='layer_scale',
            shape=(self.d_model,),
            initializer=keras.initializers.Constant(self.layer_scale_init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        # inputs/residual: [batch, channel, patch_num, d_model]
        shape = tf.shape(inputs)
        batch_size, n_channels, patch_num = shape[0], shape[1], shape[2]

        x = tf.reshape(inputs, [-1, self.d_model])
        x = self.adafe(x)
        x = self.layer_norm(x)
        x = tf.reshape(
            x,
            [batch_size, n_channels, patch_num, self.d_model],
        )
        x = tf.transpose(x, [0, 1, 3, 2])
        x = self.dsffn(x, training=training)
        x = tf.transpose(x, [0, 1, 3, 2])

        scale = tf.reshape(
            tf.cast(self.layer_scale, inputs.dtype),
            [1, 1, 1, self.d_model],
        )
        return inputs + scale * x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'fourier_degree': self.fourier_degree,
            'poly_degree': self.poly_degree,
            'ffn_ratio': self.ffn_ratio,
            'dropout': self.dropout_rate,
            'layer_scale_init': self.layer_scale_init,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
        })
        return config


@keras.utils.register_keras_serializable(package='WindFeTSPatchTST')
class FeTSChannelPatchTranspose(layers.Layer):
    """[batch, channel, d_model, patch] -> [batch, channel, patch, d_model]。"""

    def call(self, inputs):
        return tf.transpose(inputs, [0, 1, 3, 2])

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[3], input_shape[2]


@keras.utils.register_keras_serializable(package='WindFeTSPatchTST')
class PatchCrossChannelAttention(layers.Layer):
    """第一轮全通道自注意力，仅用于旧模型反序列化兼容。"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        if d_model % n_heads != 0:
            raise ValueError('d_model 必须能被 cross-channel n_heads 整除')
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_ff = int(d_ff)
        self.dropout_rate = float(dropout)

        self.attention = layers.MultiHeadAttention(
            num_heads=self.n_heads,
            key_dim=self.d_model // self.n_heads,
            dropout=self.dropout_rate,
            name='channel_mha',
        )
        self.attention_dropout = layers.Dropout(self.dropout_rate)
        self.attention_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name='channel_attention_norm',
        )
        self.ff_in = layers.Dense(
            self.d_ff,
            activation='gelu',
            name='channel_ff_in',
        )
        self.ff_dropout = layers.Dropout(self.dropout_rate)
        self.ff_out = layers.Dense(self.d_model, name='channel_ff_out')
        self.output_dropout = layers.Dropout(self.dropout_rate)
        self.output_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name='channel_output_norm',
        )

    def call(self, inputs, training=None):
        # [B, C, N, D] -> [B*N, C, D]，每个 patch 独立沿 C 维注意力。
        shape = tf.shape(inputs)
        batch_size, n_channels, patch_num = shape[0], shape[1], shape[2]
        x = tf.transpose(inputs, [0, 2, 1, 3])
        x = tf.reshape(
            x,
            [batch_size * patch_num, n_channels, self.d_model],
        )

        attention = self.attention(x, x, training=training)
        x = self.attention_norm(
            x + self.attention_dropout(attention, training=training)
        )
        ff = self.ff_in(x)
        ff = self.ff_dropout(ff, training=training)
        ff = self.ff_out(ff)
        x = self.output_norm(
            x + self.output_dropout(ff, training=training)
        )

        x = tf.reshape(
            x,
            [batch_size, patch_num, n_channels, self.d_model],
        )
        return tf.transpose(x, [0, 2, 1, 3])

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'dropout': self.dropout_rate,
        })
        return config


@keras.utils.register_keras_serializable(package='WindFeTSPatchTST')
class TargetWeatherCrossAttention(layers.Layer):
    """每个 patch 内以历史功率为 Query、其它变量为 Key/Value。"""

    def __init__(
        self,
        n_channels,
        target_channel_index,
        d_model,
        n_heads,
        d_ff,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if n_channels < 2:
            raise ValueError('目标-气象交叉注意力至少需要两个输入通道')
        if not 0 <= target_channel_index < n_channels:
            raise ValueError('target_channel_index 超出输入通道范围')
        if d_model % n_heads != 0:
            raise ValueError('d_model 必须能被 target-weather n_heads 整除')
        self.n_channels = int(n_channels)
        self.target_channel_index = int(target_channel_index)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_ff = int(d_ff)
        self.dropout_rate = float(dropout)

        self.attention = layers.MultiHeadAttention(
            num_heads=self.n_heads,
            key_dim=self.d_model // self.n_heads,
            dropout=self.dropout_rate,
            name='power_to_weather_mha',
        )
        self.attention_dropout = layers.Dropout(self.dropout_rate)
        self.attention_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name='power_weather_attention_norm',
        )
        self.ff_in = layers.Dense(
            self.d_ff,
            activation='gelu',
            name='power_weather_ff_in',
        )
        self.ff_dropout = layers.Dropout(self.dropout_rate)
        self.ff_out = layers.Dense(
            self.d_model,
            name='power_weather_ff_out',
        )
        self.output_dropout = layers.Dropout(self.dropout_rate)
        self.output_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name='power_weather_output_norm',
        )

    def call(self, inputs, training=None):
        # inputs: [B, C, N, D]。每个 patch 的功率 token 只读取非功率变量。
        shape = tf.shape(inputs)
        batch_size, patch_num = shape[0], shape[2]

        power = inputs[
            :,
            self.target_channel_index:self.target_channel_index + 1,
            :,
            :,
        ]
        if self.target_channel_index == 0:
            weather = inputs[:, 1:, :, :]
        elif self.target_channel_index == self.n_channels - 1:
            weather = inputs[:, :-1, :, :]
        else:
            weather = tf.concat(
                [
                    inputs[:, :self.target_channel_index, :, :],
                    inputs[:, self.target_channel_index + 1:, :, :],
                ],
                axis=1,
            )

        power = tf.transpose(power, [0, 2, 1, 3])
        weather = tf.transpose(weather, [0, 2, 1, 3])
        power = tf.reshape(
            power,
            [batch_size * patch_num, 1, self.d_model],
        )
        weather = tf.reshape(
            weather,
            [batch_size * patch_num, self.n_channels - 1, self.d_model],
        )

        attention = self.attention(
            query=power,
            value=weather,
            key=weather,
            training=training,
        )
        x = self.attention_norm(
            power + self.attention_dropout(attention, training=training)
        )
        ff = self.ff_in(x)
        ff = self.ff_dropout(ff, training=training)
        ff = self.ff_out(ff)
        x = self.output_norm(
            x + self.output_dropout(ff, training=training)
        )

        x = tf.reshape(x, [batch_size, patch_num, self.d_model])
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2], input_shape[3]

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_channels': self.n_channels,
            'target_channel_index': self.target_channel_index,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'dropout': self.dropout_rate,
        })
        return config


@keras.utils.register_keras_serializable(package='WindFeTSPatchTST')
class TakeLastToken(layers.Layer):
    """选择局部 patch 序列的最后一个 token。"""

    def call(self, inputs):
        return inputs[:, -1, :]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


@keras.utils.register_keras_serializable(package='WindFeTSPatchTST')
class SelectChannel(layers.Layer):
    """从 [batch, channel, ...] 中选择目标功率通道。"""

    def __init__(self, channel_index, **kwargs):
        super().__init__(**kwargs)
        self.channel_index = int(channel_index)

    def call(self, inputs):
        return inputs[:, self.channel_index, ...]

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + tuple(input_shape[2:])

    def get_config(self):
        config = super().get_config()
        config.update({'channel_index': self.channel_index})
        return config


@keras.utils.register_keras_serializable(package='WindFeTSPatchTST')
class HorizonScaledResidualAdd(layers.Layer):
    """用按 horizon 学习的受限 scale 将局部修正添加到长尺度基线。"""

    def __init__(
        self,
        forecast_len,
        initial_scale=0.1,
        max_scale=1.0,
        scale_l2=1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if forecast_len <= 0:
            raise ValueError('forecast_len 必须为正整数')
        if max_scale <= 0:
            raise ValueError('max_scale 必须为正数')
        if not 0 <= initial_scale <= max_scale:
            raise ValueError('initial_scale 必须位于 [0, max_scale] 内')
        if scale_l2 < 0:
            raise ValueError('scale_l2 不能为负数')
        self.forecast_len = int(forecast_len)
        self.initial_scale = float(initial_scale)
        self.max_scale = float(max_scale)
        self.scale_l2 = float(scale_l2)

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError('HorizonScaledResidualAdd 需要 [baseline, correction]')
        for shape in input_shape:
            if shape[-1] is not None and int(shape[-1]) != self.forecast_len:
                raise ValueError(
                    f'输入预测长度 {shape[-1]} 与 forecast_len='
                    f'{self.forecast_len} 不一致'
                )
        self.correction_scale = self.add_weight(
            name='correction_scale',
            shape=(self.forecast_len,),
            initializer=keras.initializers.Constant(self.initial_scale),
            regularizer=regularizers.l2(self.scale_l2),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        baseline, correction = inputs
        scale = tf.clip_by_value(
            tf.cast(self.correction_scale, correction.dtype),
            clip_value_min=0.0,
            clip_value_max=self.max_scale,
        )
        return baseline + correction * scale

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            'forecast_len': self.forecast_len,
            'initial_scale': self.initial_scale,
            'max_scale': self.max_scale,
            'scale_l2': self.scale_l2,
        })
        return config


@keras.utils.register_keras_serializable(package='WindFeTSPatchTST')
class PersistenceForecast(layers.Layer):
    """将最后历史功率转换到目标标准化空间并重复到所有 horizon。"""

    def __init__(
        self,
        target_channel_index,
        forecast_len,
        scale_ratio=1.0,
        scale_offset=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if target_channel_index < 0:
            raise ValueError('target_channel_index 不能为负数')
        if forecast_len <= 0:
            raise ValueError('forecast_len 必须为正整数')
        if not np.isfinite(scale_ratio) or scale_ratio <= 0:
            raise ValueError('scale_ratio 必须为有限正数')
        if not np.isfinite(scale_offset):
            raise ValueError('scale_offset 必须为有限数')
        self.target_channel_index = int(target_channel_index)
        self.forecast_len = int(forecast_len)
        self.scale_ratio = float(scale_ratio)
        self.scale_offset = float(scale_offset)

    def call(self, inputs):
        last_power_x_scaled = inputs[:, -1, self.target_channel_index]
        last_power_y_scaled = (
            last_power_x_scaled
            * tf.cast(self.scale_ratio, inputs.dtype)
            + tf.cast(self.scale_offset, inputs.dtype)
        )
        return tf.repeat(
            last_power_y_scaled[:, tf.newaxis],
            repeats=self.forecast_len,
            axis=1,
        )

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.forecast_len

    def get_config(self):
        config = super().get_config()
        config.update({
            'target_channel_index': self.target_channel_index,
            'forecast_len': self.forecast_len,
            'scale_ratio': self.scale_ratio,
            'scale_offset': self.scale_offset,
        })
        return config


@keras.utils.register_keras_serializable(package='WindFeTSPatchTST')
class HorizonRegimeRouter(layers.Layer):
    """根据当前窗口上下文和预测步长生成逐样本专家权重。"""

    def __init__(
        self,
        forecast_len,
        n_experts,
        hidden_dim=64,
        horizon_embedding_dim=16,
        dropout=0.1,
        initial_bias=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if forecast_len <= 0 or n_experts <= 1:
            raise ValueError('forecast_len 必须为正且 n_experts 必须大于 1')
        if hidden_dim <= 0 or horizon_embedding_dim <= 0:
            raise ValueError('router 表示维度必须为正整数')
        if not 0 <= dropout < 1:
            raise ValueError('router dropout 必须位于 [0, 1)')
        if initial_bias is None:
            initial_bias = [0.0] * int(n_experts)
        if len(initial_bias) != int(n_experts):
            raise ValueError('initial_bias 长度必须等于 n_experts')
        if not np.isfinite(np.asarray(initial_bias, dtype=float)).all():
            raise ValueError('initial_bias 必须全部为有限数')

        self.forecast_len = int(forecast_len)
        self.n_experts = int(n_experts)
        self.hidden_dim = int(hidden_dim)
        self.horizon_embedding_dim = int(horizon_embedding_dim)
        self.dropout_rate = float(dropout)
        self.initial_bias = tuple(float(value) for value in initial_bias)

        self.context_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name='context_norm',
        )
        self.context_projection = layers.Dense(
            self.hidden_dim,
            activation='gelu',
            name='context_projection',
        )
        self.context_dropout = layers.Dropout(
            self.dropout_rate,
            name='context_dropout',
        )
        self.horizon_embedding = layers.Embedding(
            input_dim=self.forecast_len,
            output_dim=self.horizon_embedding_dim,
            name='horizon_embedding',
        )
        self.router_hidden = layers.Dense(
            self.hidden_dim,
            activation='gelu',
            name='router_hidden',
        )
        self.router_dropout = layers.Dropout(
            self.dropout_rate,
            name='router_dropout',
        )
        self.router_logits = layers.Dense(
            self.n_experts,
            kernel_initializer='zeros',
            bias_initializer=keras.initializers.Constant(self.initial_bias),
            name='router_logits',
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
        router_input = tf.concat([context, horizon], axis=-1)
        router_hidden = self.router_hidden(router_input)
        router_hidden = self.router_dropout(router_hidden, training=training)
        logits = self.router_logits(router_hidden)
        return tf.nn.softmax(logits, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.forecast_len, self.n_experts

    def get_config(self):
        config = super().get_config()
        config.update({
            'forecast_len': self.forecast_len,
            'n_experts': self.n_experts,
            'hidden_dim': self.hidden_dim,
            'horizon_embedding_dim': self.horizon_embedding_dim,
            'dropout': self.dropout_rate,
            'initial_bias': list(self.initial_bias),
        })
        return config


@keras.utils.register_keras_serializable(package='WindFeTSPatchTST')
class ExpertConvexFusion(layers.Layer):
    """使用逐样本、逐 horizon 的 softmax 权重融合完整预测专家。"""

    def __init__(self, n_experts, **kwargs):
        super().__init__(**kwargs)
        if n_experts <= 1:
            raise ValueError('n_experts 必须大于 1')
        self.n_experts = int(n_experts)

    def build(self, input_shape):
        if (
            not isinstance(input_shape, (list, tuple))
            or len(input_shape) != self.n_experts + 1
        ):
            raise ValueError(
                'ExpertConvexFusion 需要 n_experts 个预测和一个 router 权重'
            )
        reference_shape = input_shape[0]
        for expert_shape in input_shape[1:self.n_experts]:
            if tuple(expert_shape[1:]) != tuple(reference_shape[1:]):
                raise ValueError('所有专家预测形状必须一致')
        router_shape = input_shape[-1]
        if (
            router_shape[-1] is not None
            and int(router_shape[-1]) != self.n_experts
        ):
            raise ValueError('router 最后一维必须等于 n_experts')
        super().build(input_shape)

    def call(self, inputs):
        expert_predictions = inputs[:self.n_experts]
        router_weights = inputs[-1]
        expert_stack = tf.stack(expert_predictions, axis=-1)
        return tf.reduce_sum(expert_stack * router_weights, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({'n_experts': self.n_experts})
        return config


def build_fets_patchtst_model(
    input_dim,
    target_channel_index,
    history_len=HISTORY_LEN,
    forecast_len=FORECAST_LEN,
    patch_len=PATCH_LEN,
    patch_stride=PATCH_STRIDE,
    d_model=D_MODEL,
    dropout=DROPOUT,
    head_dropout=HEAD_DROPOUT,
    fourier_degree=FOURIER_DEGREE,
    poly_degree=POLYNOMIAL_DEGREE,
    ffn_ratio=FFN_RATIO,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    mid_patch_len=MID_PATCH_LEN,
    mid_patch_stride=MID_PATCH_STRIDE,
    mid_n_layers=MID_N_LAYERS,
    local_patch_len=LOCAL_PATCH_LEN,
    local_patch_stride=LOCAL_PATCH_STRIDE,
    local_n_layers=LOCAL_N_LAYERS,
    target_weather_heads=TARGET_WEATHER_HEADS,
    layer_scale_init=LAYER_SCALE_INIT,
    long_context_dim=LONG_CONTEXT_DIM,
    router_hidden_dim=ROUTER_HIDDEN_DIM,
    horizon_embedding_dim=HORIZON_EMBEDDING_DIM,
    router_dropout=ROUTER_DROPOUT,
    router_initial_bias=ROUTER_INITIAL_BIAS,
    power_scale_ratio=1.0,
    power_scale_offset=0.0,
    correction_kernel_l2=CORRECTION_KERNEL_L2,
):
    """构建长/中/短/持续性专家动态凸融合的 FeTS-PatchTST。"""
    if target_channel_index is None:
        raise ValueError('FeTS-PatchTST 需要将历史功率作为输入通道')
    if not 0 <= target_channel_index < input_dim:
        raise ValueError('target_channel_index 超出输入通道范围')
    if input_dim < 2:
        raise ValueError('目标-气象交叉注意力至少需要两个输入通道')
    if d_model % n_heads != 0:
        raise ValueError('d_model 必须能被 PatchTST n_heads 整除')
    if d_model % target_weather_heads != 0:
        raise ValueError('d_model 必须能被 target_weather_heads 整除')
    if mid_n_layers <= 0 or local_n_layers <= 0:
        raise ValueError('mid_n_layers 和 local_n_layers 必须为正整数')
    if long_context_dim <= 0:
        raise ValueError('long_context_dim 必须为正整数')
    if router_hidden_dim <= 0 or horizon_embedding_dim <= 0:
        raise ValueError('router 表示维度必须为正整数')
    if len(router_initial_bias) != len(EXPERT_NAMES):
        raise ValueError('router_initial_bias 长度必须等于专家数量')
    if correction_kernel_l2 < 0:
        raise ValueError('correction_kernel_l2 不能为负数')

    long_patch_num = compute_patch_num(
        history_len,
        patch_len,
        patch_stride,
    )
    mid_patch_num = compute_patch_num(
        history_len,
        mid_patch_len,
        mid_patch_stride,
    )
    local_patch_num = compute_patch_num(
        history_len,
        local_patch_len,
        local_patch_stride,
    )
    inputs = keras.Input(
        shape=(history_len, input_dim),
        name='history_features',
    )

    # 长尺度分支逐层保留 817fe4... 的原生 PatchTST 结构，并作为动态
    # 专家融合的安全基线。新增分支不会改写这条前向路径。
    long_x = PatchExtract(
        patch_len,
        patch_stride,
        name='long_patch_extract',
    )(inputs)
    long_x = layers.Dense(
        d_model,
        name='long_patch_projection',
    )(long_x)
    long_x = MergeChannels(name='long_merge_channels')(long_x)
    long_x = LearnablePositionEmbedding(
        long_patch_num,
        d_model=d_model,
        name='long_position_embedding',
    )(long_x)
    long_x = layers.Dropout(
        dropout,
        name='long_patch_dropout',
    )(long_x)
    for idx in range(n_layers):
        long_x = transformer_encoder(
            long_x,
            d_model,
            n_heads,
            d_ff,
            dropout,
            name=f'long_encoder_{idx + 1}',
        )

    long_x = RestoreChannels(
        input_dim,
        long_patch_num,
        d_model,
        name='long_restore_channels',
    )(long_x)
    long_target_repr = TakeChannel(
        target_channel_index,
        name='long_target_power_channel',
    )(long_x)
    long_target_repr = layers.Flatten(
        name='long_target_flatten',
    )(long_target_repr)
    long_global_context = layers.GlobalAveragePooling2D(
        name='long_channel_context_pool',
    )(long_x)

    baseline_head = layers.Concatenate(
        name='long_forecast_context',
    )([long_target_repr, long_global_context])
    baseline_head = layers.Dropout(
        head_dropout,
        name='long_head_dropout',
    )(baseline_head)
    baseline_head = layers.Dense(
        d_ff,
        activation='gelu',
        kernel_regularizer=regularizers.l2(1e-4),
        name='long_forecast_ff',
    )(baseline_head)
    baseline_head = layers.Dropout(
        head_dropout,
        name='long_forecast_dropout',
    )(baseline_head)
    baseline_forecast = layers.Dense(
        forecast_len,
        name='baseline_forecast_power',
    )(baseline_head)
    long_local_context = layers.Concatenate(
        name='long_local_context_features',
    )([
        long_target_repr,
        long_global_context,
        baseline_forecast,
    ])
    long_local_context = layers.Dense(
        long_context_dim,
        activation='gelu',
        kernel_regularizer=regularizers.l2(correction_kernel_l2),
        name='long_to_local_context_projection',
    )(long_local_context)

    # 中尺度专家使用 8/4 PatchTST，覆盖局部 4/2 与长尺度 16/8 之间的
    # 约 1--2 小时状态。其预测以原生长尺度输出为锚点并学习零初始化残差。
    mid_x = PatchExtract(
        mid_patch_len,
        mid_patch_stride,
        name='mid_patch_extract',
    )(inputs)
    mid_x = layers.Dense(
        d_model,
        name='mid_patch_projection',
    )(mid_x)
    mid_x = MergeChannels(name='mid_merge_channels')(mid_x)
    mid_x = LearnablePositionEmbedding(
        mid_patch_num,
        d_model=d_model,
        name='mid_position_embedding',
    )(mid_x)
    mid_x = layers.Dropout(
        dropout,
        name='mid_patch_dropout',
    )(mid_x)
    for idx in range(mid_n_layers):
        mid_x = transformer_encoder(
            mid_x,
            d_model,
            n_heads,
            d_ff,
            dropout,
            name=f'mid_encoder_{idx + 1}',
        )

    mid_x = RestoreChannels(
        input_dim,
        mid_patch_num,
        d_model,
        name='mid_restore_channels',
    )(mid_x)
    mid_target_repr = TakeChannel(
        target_channel_index,
        name='mid_target_power_channel',
    )(mid_x)
    mid_target_repr = layers.Flatten(
        name='mid_target_flatten',
    )(mid_target_repr)
    mid_global_context = layers.GlobalAveragePooling2D(
        name='mid_channel_context_pool',
    )(mid_x)
    mid_context_features = layers.Concatenate(
        name='mid_forecast_context',
    )([mid_target_repr, mid_global_context])
    mid_router_context = layers.Dense(
        long_context_dim,
        activation='gelu',
        kernel_regularizer=regularizers.l2(correction_kernel_l2),
        name='mid_router_context_projection',
    )(mid_context_features)

    mid_head = layers.Dropout(
        head_dropout,
        name='mid_head_dropout',
    )(mid_context_features)
    mid_head = layers.Dense(
        d_ff,
        activation='gelu',
        kernel_regularizer=regularizers.l2(correction_kernel_l2),
        name='mid_forecast_ff',
    )(mid_head)
    mid_head = layers.Dropout(
        head_dropout,
        name='mid_forecast_dropout',
    )(mid_head)
    mid_residual = layers.Dense(
        forecast_len,
        kernel_initializer='zeros',
        bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(correction_kernel_l2),
        name='mid_forecast_residual',
    )(mid_head)
    mid_forecast = layers.Add(
        name='mid_forecast_candidate',
    )([baseline_forecast, mid_residual])

    # 局部分支保留 4 个原始 15 分钟点/patch，以 2 点步长建模快速爬坡。
    local_x = FeTSPatchExtract(
        local_patch_len,
        local_patch_stride,
        name='local_patch_extract',
    )(inputs)
    local_x = layers.Dense(
        d_model,
        name='local_patch_embedding',
    )(local_x)
    local_x = ChannelIdentityEmbedding(
        input_dim,
        d_model,
        name='local_channel_embedding',
    )(local_x)
    local_x = LayerScaleFeTSFeatureBlock(
        d_model=d_model,
        fourier_degree=fourier_degree,
        poly_degree=poly_degree,
        ffn_ratio=ffn_ratio,
        dropout=dropout,
        layer_scale_init=layer_scale_init,
        kernel_size=ADAFE_KERNEL_SIZE,
        padding=ADAFE_PADDING,
        name='local_fets_feature_block',
    )(local_x)
    local_x = TargetWeatherCrossAttention(
        n_channels=input_dim,
        target_channel_index=target_channel_index,
        d_model=d_model,
        n_heads=target_weather_heads,
        d_ff=d_ff,
        dropout=dropout,
        name='local_power_to_weather_attention',
    )(local_x)
    local_x = LearnablePositionEmbedding(
        local_patch_num,
        d_model,
        name='local_position_embedding',
    )(local_x)
    local_x = layers.Dropout(
        dropout,
        name='local_patch_dropout',
    )(local_x)
    for idx in range(local_n_layers):
        local_x = transformer_encoder(
            local_x,
            d_model,
            n_heads,
            d_ff,
            dropout,
            name=f'local_encoder_{idx + 1}',
        )

    local_recent = TakeLastToken(
        name='local_recent_token',
    )(local_x)
    local_global = layers.GlobalAveragePooling1D(
        name='local_global_pool',
    )(local_x)
    local_router_context = layers.Concatenate(
        name='local_router_context_features',
    )([local_recent, local_global])
    local_router_context = layers.Dense(
        long_context_dim,
        activation='gelu',
        kernel_regularizer=regularizers.l2(correction_kernel_l2),
        name='local_router_context_projection',
    )(local_router_context)
    local_head = layers.Concatenate(
        name='local_forecast_context',
    )([
        local_recent,
        local_global,
        long_local_context,
    ])
    local_head = layers.Dropout(
        head_dropout,
        name='local_head_dropout',
    )(local_head)
    local_head = layers.Dense(
        d_ff,
        activation='gelu',
        name='local_forecast_ff',
    )(local_head)
    local_head = layers.Dropout(
        head_dropout,
        name='local_forecast_dropout',
    )(local_head)
    local_residual = layers.Dense(
        forecast_len,
        kernel_initializer='zeros',
        bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(correction_kernel_l2),
        name='local_forecast_residual',
    )(local_head)
    local_forecast = layers.Add(
        name='local_forecast_candidate',
    )([baseline_forecast, local_residual])

    persistence_forecast = PersistenceForecast(
        target_channel_index=target_channel_index,
        forecast_len=forecast_len,
        scale_ratio=power_scale_ratio,
        scale_offset=power_scale_offset,
        name='persistence_forecast_candidate',
    )(inputs)

    # v5-A 暂不引入显式频谱/尺度相似度特征。router 只读取三个分支已经
    # 学到的历史上下文以及最后时刻全部已观测输入，不接触未来真实功率。
    last_history_features = TakeLastToken(
        name='router_last_history_features',
    )(inputs)
    router_context = layers.Concatenate(
        name='router_context_features',
    )([
        long_local_context,
        mid_router_context,
        local_router_context,
        last_history_features,
    ])
    router_weights = HorizonRegimeRouter(
        forecast_len=forecast_len,
        n_experts=len(EXPERT_NAMES),
        hidden_dim=router_hidden_dim,
        horizon_embedding_dim=horizon_embedding_dim,
        dropout=router_dropout,
        initial_bias=router_initial_bias,
        name='horizon_regime_router',
    )(router_context)

    outputs = ExpertConvexFusion(
        n_experts=len(EXPERT_NAMES),
        name='forecast_power',
    )([
        baseline_forecast,
        mid_forecast,
        local_forecast,
        persistence_forecast,
        router_weights,
    ])

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name='WindFeTSPatchTST',
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0,
        ),
        # 与原生风电 PatchTST 保持相同基础损失，以隔离结构差异。
        loss=keras.losses.Huber(delta=1.0),
        metrics=[
            keras.metrics.MeanAbsoluteError(name='mae'),
            keras.metrics.RootMeanSquaredError(name='rmse'),
        ],
    )
    return model


def validate_preprocessed_data(
    train_df,
    features,
    target,
    input_cols,
    target_index,
):
    """在进入模型前校验公共预处理结果，避免把数据错误误判为模型发散。"""
    if TARGET_COL not in train_df.columns:
        raise ValueError(f"预处理结果缺少目标列 '{TARGET_COL}'")
    if target_index is None or input_cols[target_index] != TARGET_COL:
        raise ValueError('历史功率通道索引与公共预处理结果不一致')
    if features.ndim != 2 or features.shape[1] != len(input_cols):
        raise ValueError(
            f'特征形状与列数不一致: features={features.shape}, '
            f'input_cols={len(input_cols)}'
        )
    if target.ndim != 1 or len(target) != len(features):
        raise ValueError(
            f'目标形状与特征长度不一致: target={target.shape}, '
            f'features={features.shape}'
        )
    if not np.isfinite(features).all():
        bad_count = int((~np.isfinite(features)).sum())
        raise ValueError(f'公共预处理后的输入包含 {bad_count} 个非有限值')
    if not np.isfinite(target).all():
        bad_count = int((~np.isfinite(target)).sum())
        raise ValueError(f'公共预处理后的目标包含 {bad_count} 个非有限值')


def compute_power_scale_alignment(scaler_x, scaler_y, target_index):
    """计算输入目标通道标准分数到输出目标标准分数的仿射变换。"""
    x_scale = float(scaler_x.scale_[target_index])
    x_mean = float(scaler_x.mean_[target_index])
    y_scale = float(scaler_y.scale_[0])
    y_mean = float(scaler_y.mean_[0])
    if min(x_scale, y_scale) <= 0:
        raise ValueError('功率 scaler 的 scale 必须为正数')

    scale_ratio = x_scale / y_scale
    scale_offset = (x_mean - y_mean) / y_scale
    if not np.isfinite([scale_ratio, scale_offset]).all():
        raise ValueError('功率输入/输出标准化对齐参数包含非有限值')
    return float(scale_ratio), float(scale_offset)


def collect_router_statistics(model, dataset):
    """汇总验证集动态路由权重，不改变训练目标。"""
    router_model = keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer('horizon_regime_router').output,
        name='WindFeTSPatchTSTRouterDiagnostics',
    )
    weights = np.asarray(router_model.predict(dataset, verbose=0), dtype=float)
    expected_shape = (
        None,
        model.output_shape[-1],
        len(EXPERT_NAMES),
    )
    if (
        weights.ndim != 3
        or weights.shape[1] != expected_shape[1]
        or weights.shape[2] != expected_shape[2]
    ):
        raise ValueError(
            f'router 输出形状异常: {weights.shape}, '
            f'期望 [样本, {expected_shape[1]}, {expected_shape[2]}]'
        )
    if not np.isfinite(weights).all():
        raise FloatingPointError('router 权重包含非有限值')
    if not np.allclose(weights.sum(axis=-1), 1.0, atol=1e-5):
        raise ValueError('router 权重之和不为 1')

    entropy = -np.sum(
        weights * np.log(np.clip(weights, 1e-8, 1.0)),
        axis=-1,
    ) / np.log(len(EXPERT_NAMES))
    return {
        'overall_mean': weights.mean(axis=(0, 1)),
        'overall_std': weights.std(axis=(0, 1)),
        'mean_by_horizon': weights.mean(axis=0),
        'std_by_horizon': weights.std(axis=0),
        'normalized_entropy_mean': float(entropy.mean()),
    }


class NonFiniteTrainingGuard(keras.callbacks.Callback):
    """检测到非有限训练指标后停止，防止保存损坏模型。"""

    def __init__(self):
        super().__init__()
        self.failure = None

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        bad_metrics = {}
        for name, value in logs.items():
            if value is None:
                continue
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                bad_metrics[name] = numeric_value
        if bad_metrics:
            self.failure = {
                'batch': int(batch),
                'metrics': bad_metrics,
            }
            self.model.stop_training = True
            print(
                '\n检测到非有限训练指标，已停止当前场站训练: '
                f'batch={batch}, metrics={bad_metrics}'
            )


def ensure_finite_training_history(history, guard):
    if guard.failure is not None:
        raise FloatingPointError(
            'FeTS-PatchTST 训练发生数值发散，未保存模型。'
            f"首个异常 batch={guard.failure['batch']}, "
            f"metrics={guard.failure['metrics']}"
        )

    for metric_name, values in history.history.items():
        numeric_values = np.asarray(values, dtype=float)
        if not np.isfinite(numeric_values).all():
            raise FloatingPointError(
                f'FeTS-PatchTST 的 {metric_name} 包含非有限值，未保存模型'
            )


def inverse_power(scaler_y, values):
    values = np.asarray(values).reshape(-1, 1)
    return scaler_y.inverse_transform(values).reshape(-1)


def evaluate_model(model, val_ds, scaler_y, capacity=None):
    y_true_scaled = np.concatenate(
        [y_batch.numpy() for _, y_batch in val_ds],
        axis=0,
    )
    y_pred_scaled = model.predict(val_ds, verbose=0)

    y_true = inverse_power(scaler_y, y_true_scaled)
    y_pred = inverse_power(scaler_y, y_pred_scaled)
    if not np.isfinite(y_pred).all():
        bad_count = int((~np.isfinite(y_pred)).sum())
        raise FloatingPointError(
            f'验证集预测包含 {bad_count} 个非有限值，未保存模型'
        )
    y_pred = np.clip(y_pred, 0, capacity if capacity is not None else None)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    normalized_mae = np.nan
    normalized_rmse = np.nan
    if capacity is not None and capacity > 0:
        normalized_mae = mae / capacity
        normalized_rmse = rmse / capacity
    return {
        'val_mae': mae,
        'val_rmse': rmse,
        'val_capacity_normalized_mae': normalized_mae,
        'val_capacity_normalized_rmse': normalized_rmse,
    }


def save_history_artifacts(history, farm_id):
    history_df = pd.DataFrame(history.history)
    history_df.index = np.arange(1, len(history_df) + 1)
    history_df.index.name = 'epoch'

    history_path = os.path.join(
        HISTORY_DIR,
        f'{MODEL_NAME}_history_farm_{farm_id}.csv',
    )
    history_df.to_csv(history_path, encoding='utf-8-sig')

    plot_path = os.path.join(
        HISTORY_DIR,
        f'{MODEL_NAME}_history_farm_{farm_id}.png',
    )
    try:
        cache_dir = os.path.join(MODEL_DIR, 'matplotlib_cache')
        os.environ['MPLCONFIGDIR'] = cache_dir
        os.environ['XDG_CACHE_HOME'] = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        metric_names = [
            key for key in history_df.columns
            if not key.startswith('val_') and f'val_{key}' in history_df.columns
        ]
        n_axes = max(1, len(metric_names))
        fig, axes = plt.subplots(
            n_axes,
            1,
            figsize=(10, max(3, 2.8 * n_axes)),
            sharex=True,
        )
        if n_axes == 1:
            axes = [axes]

        for ax, metric in zip(axes, metric_names):
            ax.plot(history_df.index, history_df[metric], label=f'train_{metric}')
            ax.plot(
                history_df.index,
                history_df[f'val_{metric}'],
                label=f'val_{metric}',
            )
            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.grid(alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel('epoch')
        fig.suptitle(
            f'Wind FeTS-PatchTST Training History - Farm {farm_id}',
            y=1.0,
        )
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as exc:
        plot_path = None
        print(f'训练曲线图片保存失败: {exc}')

    return history_path, plot_path


def ensure_output_dirs():
    for path in (
        MODEL_DIR,
        SAVED_MODEL_DIR,
        WEIGHTS_DIR,
        PREPROCESS_DIR,
        HISTORY_DIR,
        TENSORBOARD_LOG_DIR,
        TAIL_DIR,
    ):
        os.makedirs(path, exist_ok=True)


def train_one_farm(train_file):
    farm_id = get_farm_id(train_file)
    print(f'\n===== 训练 FeTS-PatchTST / 风电场 {farm_id} =====')

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
    power_scale_ratio, power_scale_offset = compute_power_scale_alignment(
        scaler_x,
        scaler_y,
        target_index,
    )
    # 与 wind_dl_model_train.py 使用完全相同的滑窗及时间顺序验证切分。
    train_ds, val_ds, train_samples, total_samples = make_window_dataset(
        features,
        target,
        HISTORY_LEN,
        FORECAST_LEN,
        BATCH_SIZE,
        VALIDATION_SPLIT,
    )

    long_patch_num = compute_patch_num(
        HISTORY_LEN,
        PATCH_LEN,
        PATCH_STRIDE,
    )
    mid_patch_num = compute_patch_num(
        HISTORY_LEN,
        MID_PATCH_LEN,
        MID_PATCH_STRIDE,
    )
    local_patch_num = compute_patch_num(
        HISTORY_LEN,
        LOCAL_PATCH_LEN,
        LOCAL_PATCH_STRIDE,
    )
    print(f'原始/重采样后数据形状: {train_df.shape}')
    print(
        f'输入通道数: {len(input_cols)}，样本数: {total_samples}，'
        f'训练/验证: {train_samples}/{total_samples - train_samples}'
    )
    print(
        f'长尺度Patch: patch_len={PATCH_LEN}, stride={PATCH_STRIDE}, '
        f'patch_num={long_patch_num}'
    )
    print(
        f'中尺度Patch: patch_len={MID_PATCH_LEN}, '
        f'stride={MID_PATCH_STRIDE}, patch_num={mid_patch_num}'
    )
    print(
        f'局部FeTS Patch: patch_len={LOCAL_PATCH_LEN}, '
        f'stride={LOCAL_PATCH_STRIDE}, patch_num={local_patch_num}'
    )
    print(
        f'动态专家: {EXPERT_NAMES}；router hidden={ROUTER_HIDDEN_DIM}, '
        f'horizon embedding={HORIZON_EMBEDDING_DIM}, '
        f'initial bias={ROUTER_INITIAL_BIAS}'
    )
    print(
        f'功率标准化对齐: ratio={power_scale_ratio:.8f}, '
        f'offset={power_scale_offset:.8f}；残差头 L2={CORRECTION_KERNEL_L2}'
    )

    model = build_fets_patchtst_model(
        len(input_cols),
        target_index,
        power_scale_ratio=power_scale_ratio,
        power_scale_offset=power_scale_offset,
    )
    model.summary()

    best_weights_path = os.path.join(
        WEIGHTS_DIR,
        f'{MODEL_NAME}_farm_{farm_id}_best.weights.h5',
    )
    model_path = os.path.join(
        SAVED_MODEL_DIR,
        f'{MODEL_NAME}_farm_{farm_id}.keras',
    )
    tensorboard_log_dir = os.path.join(
        TENSORBOARD_LOG_DIR,
        f'farm_{farm_id}',
        datetime.now().strftime('%Y%m%d-%H%M%S'),
    )
    non_finite_guard = NonFiniteTrainingGuard()
    callbacks = [
        non_finite_guard,
        keras.callbacks.TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=0,
            write_graph=True,
            update_freq='epoch',
            profile_batch=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            best_weights_path,
            monitor='val_loss',
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
    history_path, history_plot_path = save_history_artifacts(history, farm_id)
    ensure_finite_training_history(history, non_finite_guard)

    if not os.path.exists(best_weights_path):
        raise FileNotFoundError(
            f'训练完成但未生成最佳权重，未保存模型: {best_weights_path}'
        )
    model.load_weights(best_weights_path)

    metrics = evaluate_model(model, val_ds, scaler_y, capacity)
    router_statistics = collect_router_statistics(model, val_ds)
    model.save(model_path)
    print(
        f"验证集反归一化 MAE: {metrics['val_mae']:.4f}, "
        f"RMSE: {metrics['val_rmse']:.4f}"
    )
    print(
        '验证集平均专家权重: '
        + ', '.join(
            f'{name}={weight:.6f}'
            for name, weight in zip(
                EXPERT_NAMES,
                router_statistics['overall_mean'],
            )
        )
    )
    print(
        '验证集归一化路由熵: '
        f"{router_statistics['normalized_entropy_mean']:.6f}"
    )

    artifact = {
        'model_name': MODEL_NAME,
        'architecture_version': ARCHITECTURE_VERSION,
        'farm_id': farm_id,
        'feature_cols': feature_cols,
        'input_cols': input_cols,
        'target_col': TARGET_COL,
        'target_index': target_index,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'capacity': capacity,
        'history_len': HISTORY_LEN,
        'forecast_len': FORECAST_LEN,
        'time_freq': TIME_FREQ,
        'patch_len': PATCH_LEN,
        'patch_stride': PATCH_STRIDE,
        'mid_patch_len': MID_PATCH_LEN,
        'mid_patch_stride': MID_PATCH_STRIDE,
        'mid_n_layers': MID_N_LAYERS,
        'local_patch_len': LOCAL_PATCH_LEN,
        'local_patch_stride': LOCAL_PATCH_STRIDE,
        'local_n_layers': LOCAL_N_LAYERS,
        'd_model': D_MODEL,
        'dropout': DROPOUT,
        'head_dropout': HEAD_DROPOUT,
        'fourier_degree': FOURIER_DEGREE,
        'poly_degree': POLYNOMIAL_DEGREE,
        'ffn_ratio': FFN_RATIO,
        'n_heads': N_HEADS,
        'n_layers': N_LAYERS,
        'd_ff': D_FF,
        'target_weather_heads': TARGET_WEATHER_HEADS,
        'layer_scale_init': LAYER_SCALE_INIT,
        'long_context_dim': LONG_CONTEXT_DIM,
        'long_local_context_fusion': (
            'target_representation+global_context+baseline_forecast'
        ),
        'expert_names': list(EXPERT_NAMES),
        'expert_fusion': 'sample_horizon_conditioned_dense_softmax',
        'expert_candidate_type': 'baseline_anchored_residual',
        'router_hidden_dim': ROUTER_HIDDEN_DIM,
        'horizon_embedding_dim': HORIZON_EMBEDDING_DIM,
        'router_dropout': ROUTER_DROPOUT,
        'router_initial_bias': list(ROUTER_INITIAL_BIAS),
        'router_top_k': None,
        'router_validation_overall_mean': (
            router_statistics['overall_mean'].tolist()
        ),
        'router_validation_overall_std': (
            router_statistics['overall_std'].tolist()
        ),
        'router_validation_mean_by_horizon': (
            router_statistics['mean_by_horizon'].tolist()
        ),
        'router_validation_std_by_horizon': (
            router_statistics['std_by_horizon'].tolist()
        ),
        'router_validation_normalized_entropy_mean': (
            router_statistics['normalized_entropy_mean']
        ),
        'power_scale_ratio': power_scale_ratio,
        'power_scale_offset': power_scale_offset,
        'correction_kernel_l2': CORRECTION_KERNEL_L2,
        'adafe_threshold': 'per_patch',
        'channel_identity_embedding': True,
        'cross_channel_attention': False,
        'cross_channel_fusion': 'power_query_weather_key_value',
        'long_branch': 'original_patchtst',
        'mid_branch': 'patchtst_8_4',
        'mid_residual_initializer': 'zeros',
        'local_residual_initializer': 'zeros',
        'persistence_expert': True,
        'official_repository': OFFICIAL_REPOSITORY,
        'official_revision': OFFICIAL_REVISION,
        'model_path': model_path,
        'best_weights_path': best_weights_path,
        'tensorboard_log_dir': tensorboard_log_dir,
        'history_path': history_path,
        'history_plot_path': history_plot_path,
        **metrics,
    }
    artifact_path = os.path.join(
        PREPROCESS_DIR,
        f'{MODEL_NAME}_farm_{farm_id}_preprocess.pkl',
    )
    joblib.dump(artifact, artifact_path)

    tail_path = os.path.join(
        TAIL_DIR,
        f'{MODEL_NAME}_tail_farm_{farm_id}.csv',
    )
    train_df.iloc[-HISTORY_LEN:].to_csv(tail_path, index=True)

    print(f'最终模型: {model_path}')
    print(f'最佳 checkpoint: {best_weights_path}')
    print(f'预处理参数: {artifact_path}')
    print(f'TensorBoard 日志: {tensorboard_log_dir}')

    result = {
        **metrics,
        'farm_id': farm_id,
        'model_path': model_path,
        'best_weights_path': best_weights_path,
        'artifact_path': artifact_path,
        'tail_path': tail_path,
        'tensorboard_log_dir': tensorboard_log_dir,
        'history_path': history_path,
        'history_plot_path': history_plot_path,
        'router_normalized_entropy': (
            router_statistics['normalized_entropy_mean']
        ),
        **{
            f'router_weight_{name}': float(weight)
            for name, weight in zip(
                EXPERT_NAMES,
                router_statistics['overall_mean'],
            )
        },
    }
    del model
    keras.backend.clear_session()
    return result


def main():
    ensure_output_dirs()
    train_files = discover_train_files(DATA_DIR)
    if not train_files:
        raise FileNotFoundError(f'未在 {DATA_DIR} 找到风电训练文件')

    print(f'发现 {len(train_files)} 个风电训练文件')
    results = [train_one_farm(train_file) for train_file in train_files]
    metrics_df = pd.DataFrame(results)
    metrics_path = os.path.join(
        MODEL_DIR,
        f'{MODEL_NAME}_training_metrics.csv',
    )
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f'\n训练完成，指标已保存至 {metrics_path}')
    print(f'启动 TensorBoard: tensorboard --logdir {TENSORBOARD_LOG_DIR}')


if __name__ == '__main__':
    main()
