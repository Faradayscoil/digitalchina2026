"""FeTS-PatchTST 在风电超短期功率预测任务上的训练入口。

模型在原生 PatchTST 上接入按 lllucky111/FeTS 官方张量逻辑重写的模块：

    patch embedding
    -> batch-independent AdaFE -> DSFFN
    -> per-patch cross-channel attention
    -> PatchTST positional embedding + Transformer encoders
    -> target/global-context MLP head

模型复用原生 PatchTST 基线的数据预处理、历史窗口、时序编码器、预测 head、
验证划分、优化器和损失。此文件不包含 k-fold、多随机种子、RevIN、
自蒸馏、外部 teacher 或输出专家等额外策略。
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
    RestoreChannels,
    TakeChannel,
    build_scaled_arrays,
    load_and_preprocess,
    make_window_dataset,
    transformer_encoder,
)

warnings.filterwarnings('ignore')


MODEL_NAME = 'fets_patchtst'
ARCHITECTURE_VERSION = 'fets_patchtst_hybrid_v2'
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
CROSS_CHANNEL_HEADS = N_HEADS

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
    """官方 AdaFE + LayerNorm + DSFFN + residual 完整处理块。"""

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
class FeTSChannelPatchTranspose(layers.Layer):
    """[batch, channel, d_model, patch] -> [batch, channel, patch, d_model]。"""

    def call(self, inputs):
        return tf.transpose(inputs, [0, 1, 3, 2])

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[3], input_shape[2]


@keras.utils.register_keras_serializable(package='WindFeTSPatchTST')
class PatchCrossChannelAttention(layers.Layer):
    """在每个 temporal patch 内对所有变量执行跨通道自注意力。"""

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
    cross_channel_heads=CROSS_CHANNEL_HEADS,
):
    """构建 FeTS patch 增强 + 跨变量融合 + PatchTST 时序主干。"""
    if target_channel_index is None:
        raise ValueError('FeTS-PatchTST 需要将历史功率作为输入通道')
    if not 0 <= target_channel_index < input_dim:
        raise ValueError('target_channel_index 超出输入通道范围')
    if d_model % n_heads != 0:
        raise ValueError('d_model 必须能被 PatchTST n_heads 整除')
    if d_model % cross_channel_heads != 0:
        raise ValueError('d_model 必须能被 cross_channel_heads 整除')

    patch_num = compute_patch_num(history_len, patch_len, patch_stride)
    inputs = keras.Input(
        shape=(history_len, input_dim),
        name='history_features',
    )
    x = FeTSPatchExtract(
        patch_len,
        patch_stride,
        name='patch_extract',
    )(inputs)
    x = layers.Dense(d_model, name='patch_embedding')(x)
    x = FeTSFeatureBlock(
        d_model=d_model,
        fourier_degree=fourier_degree,
        poly_degree=poly_degree,
        ffn_ratio=ffn_ratio,
        dropout=dropout,
        kernel_size=ADAFE_KERNEL_SIZE,
        padding=ADAFE_PADDING,
        name='fets_feature_block',
    )(x)
    x = FeTSChannelPatchTranspose(
        name='restore_patch_order',
    )(x)
    x = PatchCrossChannelAttention(
        d_model=d_model,
        n_heads=cross_channel_heads,
        d_ff=d_ff,
        dropout=dropout,
        name='cross_channel_attention',
    )(x)

    # 恢复 817fe4... 原生 PatchTST 的 patch 时序建模主干。
    x = MergeChannels(name='merge_channels')(x)
    x = LearnablePositionEmbedding(
        patch_num,
        d_model,
        name='position_embedding',
    )(x)
    x = layers.Dropout(dropout, name='patch_dropout')(x)
    for idx in range(n_layers):
        x = transformer_encoder(
            x,
            d_model,
            n_heads,
            d_ff,
            dropout,
            name=f'encoder_{idx + 1}',
        )

    x = RestoreChannels(
        input_dim,
        patch_num,
        d_model,
        name='restore_channels',
    )(x)
    target_repr = TakeChannel(
        target_channel_index,
        name='target_power_channel',
    )(x)
    target_repr = layers.Flatten(name='target_flatten')(target_repr)
    global_context = layers.GlobalAveragePooling2D(
        name='channel_context_pool',
    )(x)

    head = layers.Concatenate(
        name='forecast_context',
    )([target_repr, global_context])
    head = layers.Dropout(
        head_dropout,
        name='head_dropout',
    )(head)
    head = layers.Dense(
        d_ff,
        activation='gelu',
        kernel_regularizer=regularizers.l2(1e-4),
        name='forecast_ff',
    )(head)
    head = layers.Dropout(
        head_dropout,
        name='forecast_dropout',
    )(head)
    outputs = layers.Dense(
        forecast_len,
        name='forecast_power',
    )(head)

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
    # 与 wind_dl_model_train.py 使用完全相同的滑窗及时间顺序验证切分。
    train_ds, val_ds, train_samples, total_samples = make_window_dataset(
        features,
        target,
        HISTORY_LEN,
        FORECAST_LEN,
        BATCH_SIZE,
        VALIDATION_SPLIT,
    )

    patch_num = compute_patch_num(HISTORY_LEN, PATCH_LEN, PATCH_STRIDE)
    print(f'原始/重采样后数据形状: {train_df.shape}')
    print(
        f'输入通道数: {len(input_cols)}，样本数: {total_samples}，'
        f'训练/验证: {train_samples}/{total_samples - train_samples}'
    )
    print(
        f'Patch设置: patch_len={PATCH_LEN}, stride={PATCH_STRIDE}, '
        f'patch_num={patch_num}'
    )

    model = build_fets_patchtst_model(
        len(input_cols),
        target_index,
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
    model.save(model_path)
    print(
        f"验证集反归一化 MAE: {metrics['val_mae']:.4f}, "
        f"RMSE: {metrics['val_rmse']:.4f}"
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
        'd_model': D_MODEL,
        'dropout': DROPOUT,
        'head_dropout': HEAD_DROPOUT,
        'fourier_degree': FOURIER_DEGREE,
        'poly_degree': POLYNOMIAL_DEGREE,
        'ffn_ratio': FFN_RATIO,
        'n_heads': N_HEADS,
        'n_layers': N_LAYERS,
        'd_ff': D_FF,
        'cross_channel_heads': CROSS_CHANNEL_HEADS,
        'adafe_threshold': 'per_patch',
        'cross_channel_attention': True,
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
