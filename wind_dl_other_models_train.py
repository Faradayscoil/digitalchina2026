import glob
import os
import random
import re
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from wind_dl_model_train import (
    BATCH_SIZE as PATCHTST_BATCH_SIZE,
    DATA_DIR,
    FORECAST_LEN,
    HISTORY_LEN,
    TARGET_COL,
    TIME_FREQ,
    build_scaled_arrays,
    load_and_preprocess,
    make_window_dataset,
)

warnings.filterwarnings('ignore')


TRAIN_FILE_PATTERN = 'wind_train_*.csv'
BASE_RESULT_DIR = r'./wind_results'
DEFAULT_MODEL_NAMES = [
    'bilstm',
    'cnn_lstm',
    'cnn_resnet_gru',
    'wavenet',
    'transformer',
    'informer',
    'autoformer',
]

seed = 2026
BATCH_SIZE = int(os.getenv('WIND_DL_BATCH_SIZE', PATCHTST_BATCH_SIZE))
EPOCHS = int(os.getenv('WIND_DL_EPOCHS', 60))
VALIDATION_SPLIT = float(os.getenv('WIND_DL_VALIDATION_SPLIT', 0.15))
LEARNING_RATE = float(os.getenv('WIND_DL_LEARNING_RATE', 5e-4))

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
DROPOUT = 0.15
HEAD_DROPOUT = 0.2
L2_REG = 1e-4
INFORMER_FACTOR = 5
AUTOFORMER_FACTOR = 3
AUTOFORMER_MOVING_AVG = 25
AUTOFORMER_LABEL_LEN = min(48, HISTORY_LEN)
TIME_FEATURE_COLS = [
    'minute_sin',
    'minute_cos',
    'dow_sin',
    'dow_cos',
    'doy_sin',
    'doy_cos',
    'month_sin',
    'month_cos',
]


def set_global_seed(value):
    random.seed(value)
    np.random.seed(value)
    tf.random.set_seed(value)
    try:
        keras.utils.set_random_seed(value)
    except AttributeError:
        pass


def discover_train_files(data_dir=DATA_DIR):
    """只读取 wind_split 根目录下人工重命名好的 wind_train_<farm_id>.csv。"""
    return sorted(glob.glob(os.path.join(data_dir, TRAIN_FILE_PATTERN)))


def get_farm_id(path):
    basename = os.path.basename(path)
    match = re.search(r'wind_train_(\d+)\.csv$', basename)
    if match:
        return match.group(1)
    return os.path.splitext(basename)[0]


def get_requested_model_names():
    names = os.getenv('WIND_DL_MODEL_NAMES')
    if not names:
        return DEFAULT_MODEL_NAMES
    requested = [name.strip().lower() for name in names.split(',') if name.strip()]
    invalid = sorted(set(requested) - set(DEFAULT_MODEL_NAMES))
    if invalid:
        raise ValueError(f'未知模型名称: {invalid}; 可选: {DEFAULT_MODEL_NAMES}')
    return requested


def model_dirs(model_name):
    root = os.path.join(BASE_RESULT_DIR, model_name)
    dirs = {
        'root': root,
        'models': os.path.join(root, 'models'),
        'weights': os.path.join(root, 'weights'),
        'preprocess': os.path.join(root, 'preprocess'),
        'history': os.path.join(root, 'history'),
        'tensorboard': os.path.join(root, 'tensorboard'),
        'tails': os.path.join(root, 'tails'),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def inverse_power(scaler_y, values):
    values = np.asarray(values).reshape(-1, 1)
    return scaler_y.inverse_transform(values).reshape(-1)


def evaluate_model(model, val_ds, scaler_y, capacity=None):
    y_true_scaled = []
    for _, y_batch in val_ds:
        y_true_scaled.append(y_batch.numpy())
    y_true_scaled = np.concatenate(y_true_scaled, axis=0)
    y_pred_scaled = model.predict(val_ds, verbose=0)

    y_true = inverse_power(scaler_y, y_true_scaled)
    y_pred = inverse_power(scaler_y, y_pred_scaled)
    if capacity is not None:
        y_pred = np.clip(y_pred, 0, capacity)
    else:
        y_pred = np.clip(y_pred, 0, None)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = np.nan
    if len(y_true) > 1 and np.nanstd(y_true) > 1e-6:
        r2 = r2_score(y_true, y_pred)

    norm_mae = np.nan
    norm_rmse = np.nan
    if capacity is not None and capacity > 0:
        norm_mae = float(mae / capacity)
        norm_rmse = float(rmse / capacity)

    return {
        'val_inverse_mae': float(mae),
        'val_inverse_mse': float(mse),
        'val_inverse_rmse': rmse,
        'val_inverse_r2': r2,
        'val_capacity_normalized_mae': norm_mae,
        'val_capacity_normalized_rmse': norm_rmse,
    }


def save_history_artifacts(history, model_name, farm_id, dirs):
    history_df = pd.DataFrame(history.history)
    history_df.index = np.arange(1, len(history_df) + 1)
    history_df.index.name = 'epoch'

    history_path = os.path.join(dirs['history'], f'{model_name}_history_farm_{farm_id}.csv')
    history_df.to_csv(history_path, encoding='utf-8-sig')

    plot_path = os.path.join(dirs['history'], f'{model_name}_history_farm_{farm_id}.png')
    try:
        cache_dir = os.path.join(dirs['root'], 'matplotlib_cache')
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
        single_names = [
            key for key in history_df.columns
            if not key.startswith('val_') and f'val_{key}' not in history_df.columns
        ]
        n_axes = max(1, len(metric_names) + len(single_names))
        fig, axes = plt.subplots(n_axes, 1, figsize=(10, max(3, 2.8 * n_axes)), sharex=True)
        if n_axes == 1:
            axes = [axes]

        axis_idx = 0
        for metric in metric_names:
            ax = axes[axis_idx]
            ax.plot(history_df.index, history_df[metric], label=f'train_{metric}')
            ax.plot(history_df.index, history_df[f'val_{metric}'], label=f'val_{metric}')
            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.grid(alpha=0.3)
            ax.legend()
            axis_idx += 1

        for metric in single_names:
            ax = axes[axis_idx]
            ax.plot(history_df.index, history_df[metric], label=metric)
            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.grid(alpha=0.3)
            ax.legend()
            axis_idx += 1

        axes[-1].set_xlabel('epoch')
        fig.suptitle(f'{model_name} Training History - Farm {farm_id}', y=1.0)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as exc:
        plot_path = None
        print(f'{model_name} 场站 {farm_id} 训练曲线图片保存失败: {exc}')

    return history_path, plot_path


def compile_forecast_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss=keras.losses.Huber(delta=1.0),
        metrics=[
            keras.metrics.MeanAbsoluteError(name='mae'),
            keras.metrics.RootMeanSquaredError(name='rmse'),
        ],
    )
    return model


def dense_forecast_head(x, forecast_len=FORECAST_LEN, name='forecast'):
    x = layers.Flatten(name=f'{name}_flatten')(x)
    x = layers.Dropout(HEAD_DROPOUT, name=f'{name}_dropout')(x)
    x = layers.Dense(
        D_FF,
        activation='gelu',
        kernel_regularizer=regularizers.l2(L2_REG),
        name=f'{name}_dense',
    )(x)
    x = layers.Dropout(HEAD_DROPOUT, name=f'{name}_dense_dropout')(x)
    return layers.Dense(forecast_len, name=f'{name}_power')(x)


@keras.utils.register_keras_serializable(package='WindInformer')
class FixedPositionEmbedding(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        position = tf.cast(tf.range(seq_len)[:, None], tf.float32)
        dims = tf.cast(tf.range(self.d_model)[None, :], tf.float32)
        angle_rates = tf.pow(
            10000.0,
            -2.0 * tf.floor(dims / 2.0) / tf.cast(self.d_model, tf.float32),
        )
        angles = position * angle_rates
        even_dims = tf.equal(tf.math.mod(tf.range(self.d_model), 2), 0)
        position_encoding = tf.where(even_dims[None, :], tf.sin(angles), tf.cos(angles))
        return position_encoding[None, :, :]

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model})
        return config


@keras.utils.register_keras_serializable(package='WindInformer')
class CircularTokenEmbedding(layers.Layer):
    def __init__(self, d_model, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.token_conv = layers.Conv1D(
            d_model,
            kernel_size,
            padding='valid',
            kernel_initializer=keras.initializers.HeNormal(),
            name='token_conv',
        )

    def call(self, inputs):
        if self.pad == 0:
            return self.token_conv(inputs)
        padded = tf.concat([inputs[:, -self.pad:, :], inputs, inputs[:, :self.pad, :]], axis=1)
        return self.token_conv(padded)

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 'kernel_size': self.kernel_size})
        return config


@keras.utils.register_keras_serializable(package='WindInformer')
class InformerDataEmbedding(layers.Layer):
    def __init__(self, d_model, time_feature_indices=None, dropout=DROPOUT, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.time_feature_indices = list(time_feature_indices or [])
        self.dropout_rate = dropout
        self.value_embedding = CircularTokenEmbedding(d_model, kernel_size=3, name='value_embedding')
        self.position_embedding = FixedPositionEmbedding(d_model, name='position_embedding')
        self.temporal_embedding = layers.Dense(d_model, name='temporal_embedding')
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        value = self.value_embedding(inputs)
        position = self.position_embedding(inputs)
        if self.time_feature_indices:
            time_features = tf.gather(inputs, self.time_feature_indices, axis=-1)
            temporal = self.temporal_embedding(time_features)
        else:
            temporal = 0.0
        return self.dropout(value + position + temporal, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'time_feature_indices': self.time_feature_indices,
            'dropout': self.dropout_rate,
        })
        return config


@keras.utils.register_keras_serializable(package='WindInformer')
class ProbSparseSelfAttention(layers.Layer):
    def __init__(self, d_model, n_heads, factor=INFORMER_FACTOR,
                 dropout=DROPOUT, **kwargs):
        super().__init__(**kwargs)
        if d_model % n_heads != 0:
            raise ValueError('d_model 必须能被 n_heads 整除')
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.dropout_rate = dropout
        self.head_dim = d_model // n_heads
        self.query_projection = layers.Dense(d_model, name='query_projection')
        self.key_projection = layers.Dense(d_model, name='key_projection')
        self.value_projection = layers.Dense(d_model, name='value_projection')
        self.out_projection = layers.Dense(d_model, name='out_projection')
        self.attn_dropout = layers.Dropout(dropout)

    def _reshape_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, [batch_size, seq_len, self.n_heads, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    def _sample_size(self, length):
        length_float = tf.cast(length, tf.float32)
        sample_size = tf.cast(tf.math.ceil(tf.math.log(length_float + 1.0)) * self.factor, tf.int32)
        sample_size = tf.maximum(sample_size, 1)
        return tf.minimum(sample_size, tf.cast(length, tf.int32))

    def call(self, inputs, training=None):
        queries = self._reshape_heads(self.query_projection(inputs))
        keys = self._reshape_heads(self.key_projection(inputs))
        values = self._reshape_heads(self.value_projection(inputs))

        batch_size = tf.shape(inputs)[0]
        seq_q = tf.shape(queries)[2]
        seq_k = tf.shape(keys)[2]
        sample_k = self._sample_size(seq_k)
        n_top = self._sample_size(seq_q)

        sampled_key_indices = tf.random.uniform(
            shape=[seq_q, sample_k],
            minval=0,
            maxval=seq_k,
            dtype=tf.int32,
        )
        sampled_keys = tf.gather(keys, sampled_key_indices, axis=2)
        sampled_scores = tf.reduce_sum(
            queries[:, :, :, None, :] * sampled_keys,
            axis=-1,
        )

        sparsity = tf.reduce_max(sampled_scores, axis=-1) - tf.reduce_mean(sampled_scores, axis=-1)
        top_query_indices = tf.math.top_k(sparsity, k=n_top, sorted=False).indices
        top_queries = tf.gather(queries, top_query_indices, axis=2, batch_dims=2)

        scale = tf.math.rsqrt(tf.cast(self.head_dim, tf.float32))
        scores_top = tf.einsum('bhud,bhld->bhul', top_queries, keys) * scale
        attention_top = tf.nn.softmax(scores_top, axis=-1)
        attention_top = self.attn_dropout(attention_top, training=training)
        context_top = tf.einsum('bhul,bhld->bhud', attention_top, values)

        value_mean = tf.reduce_mean(values, axis=2, keepdims=True)
        context = tf.tile(value_mean, [1, 1, seq_q, 1])

        batch_indices = tf.broadcast_to(
            tf.reshape(tf.range(batch_size), [batch_size, 1, 1]),
            [batch_size, self.n_heads, n_top],
        )
        head_indices = tf.broadcast_to(
            tf.reshape(tf.range(self.n_heads), [1, self.n_heads, 1]),
            [batch_size, self.n_heads, n_top],
        )
        scatter_indices = tf.stack([batch_indices, head_indices, top_query_indices], axis=-1)
        context = tf.tensor_scatter_nd_update(
            context,
            tf.reshape(scatter_indices, [-1, 3]),
            tf.reshape(context_top, [-1, self.head_dim]),
        )

        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, [batch_size, seq_q, self.d_model])
        return self.out_projection(context)

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'factor': self.factor,
            'dropout': self.dropout_rate,
        })
        return config


def informer_encoder_block(x, d_model=D_MODEL, n_heads=N_HEADS,
                           d_ff=D_FF, dropout=DROPOUT, name='informer_encoder'):
    attn = ProbSparseSelfAttention(
        d_model=d_model,
        n_heads=n_heads,
        factor=INFORMER_FACTOR,
        dropout=dropout,
        name=f'{name}_probsparse_attention',
    )(x)
    x = layers.Add(name=f'{name}_attn_add')([x, layers.Dropout(dropout)(attn)])
    x = layers.LayerNormalization(epsilon=1e-6, name=f'{name}_attn_norm')(x)

    ff = layers.Conv1D(d_ff, 1, activation='gelu', name=f'{name}_ff1')(x)
    ff = layers.Dropout(dropout, name=f'{name}_ff_dropout')(ff)
    ff = layers.Conv1D(d_model, 1, name=f'{name}_ff2')(ff)
    x = layers.Add(name=f'{name}_ff_add')([x, layers.Dropout(dropout)(ff)])
    return layers.LayerNormalization(epsilon=1e-6, name=f'{name}_ff_norm')(x)


def informer_distil_layer(x, name='informer_distil'):
    x = CircularTokenEmbedding(D_MODEL, kernel_size=3, name=f'{name}_circular_conv')(x)
    x = layers.BatchNormalization(name=f'{name}_batch_norm')(x)
    x = layers.ELU(name=f'{name}_elu')(x)
    return layers.MaxPooling1D(
        pool_size=3,
        strides=2,
        padding='same',
        name=f'{name}_max_pool',
    )(x)


@keras.utils.register_keras_serializable(package='WindAutoformer')
class MovingAverage(layers.Layer):
    def __init__(self, kernel_size=AUTOFORMER_MOVING_AVG, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2
        self.avg_pool = layers.AveragePooling1D(
            pool_size=kernel_size,
            strides=1,
            padding='valid',
        )

    def call(self, inputs):
        if self.pad == 0:
            return inputs
        front = tf.repeat(inputs[:, :1, :], repeats=self.pad, axis=1)
        end = tf.repeat(inputs[:, -1:, :], repeats=self.pad, axis=1)
        padded = tf.concat([front, inputs, end], axis=1)
        return self.avg_pool(padded)

    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


@keras.utils.register_keras_serializable(package='WindAutoformer')
class SeriesDecomposition(layers.Layer):
    def __init__(self, kernel_size=AUTOFORMER_MOVING_AVG, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.moving_average = MovingAverage(kernel_size, name='moving_average')

    def call(self, inputs):
        trend = self.moving_average(inputs)
        seasonal = inputs - trend
        return seasonal, trend

    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


@keras.utils.register_keras_serializable(package='WindAutoformer')
class AutoformerLayerNorm(layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.layer_norm = layers.LayerNormalization(epsilon=epsilon)

    def call(self, inputs):
        normalized = self.layer_norm(inputs)
        bias = tf.reduce_mean(normalized, axis=1, keepdims=True)
        return normalized - bias

    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config


@keras.utils.register_keras_serializable(package='WindAutoformer')
class AutoformerDataEmbedding(layers.Layer):
    def __init__(self, d_model, time_feature_indices=None, dropout=DROPOUT, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.time_feature_indices = list(time_feature_indices or [])
        self.dropout_rate = dropout
        self.value_embedding = CircularTokenEmbedding(d_model, kernel_size=3, name='value_embedding')
        self.temporal_embedding = layers.Dense(d_model, use_bias=False, name='temporal_embedding')
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        value = self.value_embedding(inputs)
        if self.time_feature_indices:
            time_features = tf.gather(inputs, self.time_feature_indices, axis=-1)
            temporal = self.temporal_embedding(time_features)
        else:
            temporal = 0.0
        return self.dropout(value + temporal, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'time_feature_indices': self.time_feature_indices,
            'dropout': self.dropout_rate,
        })
        return config


@keras.utils.register_keras_serializable(package='WindAutoformer')
class AutoformerDecoderInitializer(layers.Layer):
    def __init__(self, label_len=AUTOFORMER_LABEL_LEN, pred_len=FORECAST_LEN, **kwargs):
        super().__init__(**kwargs)
        self.label_len = label_len
        self.pred_len = pred_len

    def call(self, inputs):
        seasonal_init, trend_init, encoded_values = inputs
        future_mean = tf.reduce_mean(encoded_values, axis=1, keepdims=True)
        future_mean = tf.repeat(future_mean, repeats=self.pred_len, axis=1)
        future_zeros = tf.zeros_like(future_mean)
        trend = tf.concat([trend_init[:, -self.label_len:, :], future_mean], axis=1)
        seasonal = tf.concat([seasonal_init[:, -self.label_len:, :], future_zeros], axis=1)
        return seasonal, trend

    def get_config(self):
        config = super().get_config()
        config.update({
            'label_len': self.label_len,
            'pred_len': self.pred_len,
        })
        return config


@keras.utils.register_keras_serializable(package='WindAutoformer')
class SeriesWiseAutoCorrelation(layers.Layer):
    def __init__(self, d_model, n_heads, factor=AUTOFORMER_FACTOR,
                 dropout=DROPOUT, **kwargs):
        super().__init__(**kwargs)
        if d_model % n_heads != 0:
            raise ValueError('d_model 必须能被 n_heads 整除')
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.dropout_rate = dropout
        self.head_dim = d_model // n_heads
        self.query_projection = layers.Dense(d_model, name='query_projection')
        self.key_projection = layers.Dense(d_model, name='key_projection')
        self.value_projection = layers.Dense(d_model, name='value_projection')
        self.out_projection = layers.Dense(d_model, name='out_projection')
        self.dropout = layers.Dropout(dropout)

    def _reshape_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, [batch_size, seq_len, self.n_heads, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    def _align_length(self, x, target_len):
        current_len = tf.shape(x)[2]

        def pad_to_target():
            pad_len = target_len - current_len
            paddings = tf.stack([
                tf.constant([0, 0], dtype=tf.int32),
                tf.constant([0, 0], dtype=tf.int32),
                tf.stack([tf.constant(0, dtype=tf.int32), pad_len]),
                tf.constant([0, 0], dtype=tf.int32),
            ])
            return tf.pad(x, paddings)

        def slice_to_target():
            return x[:, :, :target_len, :]

        return tf.cond(current_len < target_len, pad_to_target, slice_to_target)

    def _top_k(self, length):
        length_float = tf.cast(length, tf.float32)
        top_k = tf.cast(tf.math.ceil(tf.math.log(length_float + 1.0)) * self.factor, tf.int32)
        top_k = tf.maximum(top_k, 1)
        return tf.minimum(top_k, tf.cast(length, tf.int32))

    def call(self, inputs, training=None):
        queries, keys, values = inputs
        queries = self._reshape_heads(self.query_projection(queries))
        keys = self._reshape_heads(self.key_projection(keys))
        values = self._reshape_heads(self.value_projection(values))

        seq_len = tf.shape(queries)[2]
        keys = self._align_length(keys, seq_len)
        values = self._align_length(values, seq_len)

        queries_fft = tf.signal.rfft(
            tf.transpose(queries, [0, 1, 3, 2]),
            fft_length=tf.reshape(seq_len, [1]),
        )
        keys_fft = tf.signal.rfft(
            tf.transpose(keys, [0, 1, 3, 2]),
            fft_length=tf.reshape(seq_len, [1]),
        )
        corr = tf.signal.irfft(
            queries_fft * tf.math.conj(keys_fft),
            fft_length=tf.reshape(seq_len, [1]),
        )

        values_time = tf.transpose(values, [0, 1, 3, 2])
        mean_corr = tf.reduce_mean(corr, axis=[1, 2])
        top_k = self._top_k(seq_len)
        weights, delays = tf.math.top_k(mean_corr, k=top_k, sorted=False)
        weights = tf.nn.softmax(weights, axis=-1)
        weights = self.dropout(weights, training=training)

        values_twice = tf.concat([values_time, values_time], axis=-1)
        batch_size = tf.shape(values_time)[0]
        base_index = tf.range(seq_len, dtype=tf.int32)
        base_index = tf.reshape(base_index, [1, 1, 1, seq_len, 1])
        delays = tf.reshape(delays, [batch_size, 1, 1, 1, top_k])
        gather_index = base_index + delays
        gather_index = tf.broadcast_to(
            gather_index,
            [batch_size, self.n_heads, self.head_dim, seq_len, top_k],
        )
        delayed_values = tf.gather(values_twice, gather_index, axis=-1, batch_dims=3)
        weights = tf.reshape(weights, [batch_size, 1, 1, 1, top_k])
        aggregated = tf.reduce_sum(delayed_values * weights, axis=-1)

        aggregated = tf.transpose(aggregated, [0, 3, 1, 2])
        aggregated = tf.reshape(aggregated, [batch_size, seq_len, self.d_model])
        return self.out_projection(aggregated)

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'factor': self.factor,
            'dropout': self.dropout_rate,
        })
        return config


@keras.utils.register_keras_serializable(package='WindAutoformer')
class AutoformerEncoderLayer(layers.Layer):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
                 moving_avg=AUTOFORMER_MOVING_AVG, dropout=DROPOUT, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.moving_avg = moving_avg
        self.dropout_rate = dropout
        self.auto_correlation = SeriesWiseAutoCorrelation(
            d_model,
            n_heads,
            factor=AUTOFORMER_FACTOR,
            dropout=dropout,
            name='auto_correlation',
        )
        self.dropout = layers.Dropout(dropout)
        self.decomp1 = SeriesDecomposition(moving_avg, name='decomp1')
        self.decomp2 = SeriesDecomposition(moving_avg, name='decomp2')
        self.conv1 = layers.Conv1D(d_ff, 1, activation='gelu', name='ff_conv1')
        self.conv2 = layers.Conv1D(d_model, 1, name='ff_conv2')

    def call(self, inputs, training=None):
        attn = self.auto_correlation([inputs, inputs, inputs], training=training)
        x = inputs + self.dropout(attn, training=training)
        x, _ = self.decomp1(x)
        y = self.conv1(x)
        y = self.dropout(y, training=training)
        y = self.conv2(y)
        y = self.dropout(y, training=training)
        seasonal, _ = self.decomp2(x + y)
        return seasonal

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'moving_avg': self.moving_avg,
            'dropout': self.dropout_rate,
        })
        return config


@keras.utils.register_keras_serializable(package='WindAutoformer')
class AutoformerDecoderLayer(layers.Layer):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
                 moving_avg=AUTOFORMER_MOVING_AVG, dropout=DROPOUT, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.moving_avg = moving_avg
        self.dropout_rate = dropout
        self.self_correlation = SeriesWiseAutoCorrelation(
            d_model,
            n_heads,
            factor=AUTOFORMER_FACTOR,
            dropout=dropout,
            name='self_correlation',
        )
        self.cross_correlation = SeriesWiseAutoCorrelation(
            d_model,
            n_heads,
            factor=AUTOFORMER_FACTOR,
            dropout=dropout,
            name='cross_correlation',
        )
        self.dropout = layers.Dropout(dropout)
        self.decomp1 = SeriesDecomposition(moving_avg, name='decomp1')
        self.decomp2 = SeriesDecomposition(moving_avg, name='decomp2')
        self.decomp3 = SeriesDecomposition(moving_avg, name='decomp3')
        self.conv1 = layers.Conv1D(d_ff, 1, activation='gelu', name='ff_conv1')
        self.conv2 = layers.Conv1D(d_model, 1, name='ff_conv2')
        self.trend_projection = layers.Conv1D(
            d_model,
            3,
            padding='same',
            name='trend_projection',
        )

    def call(self, inputs, training=None):
        x, cross = inputs
        self_attn = self.self_correlation([x, x, x], training=training)
        x = x + self.dropout(self_attn, training=training)
        x, trend1 = self.decomp1(x)

        cross_attn = self.cross_correlation([x, cross, cross], training=training)
        x = x + self.dropout(cross_attn, training=training)
        x, trend2 = self.decomp2(x)

        y = self.conv1(x)
        y = self.dropout(y, training=training)
        y = self.conv2(y)
        y = self.dropout(y, training=training)
        x, trend3 = self.decomp3(x + y)

        residual_trend = self.trend_projection(trend1 + trend2 + trend3)
        return x, residual_trend

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'moving_avg': self.moving_avg,
            'dropout': self.dropout_rate,
        })
        return config


def transformer_encoder_block(x, d_model=D_MODEL, n_heads=N_HEADS,
                              d_ff=D_FF, dropout=DROPOUT, name='encoder'):
    attn = layers.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=max(1, d_model // n_heads),
        dropout=dropout,
        name=f'{name}_mha',
    )(x, x)
    x = layers.Add(name=f'{name}_attn_add')([x, layers.Dropout(dropout)(attn)])
    x = layers.LayerNormalization(epsilon=1e-6, name=f'{name}_attn_norm')(x)

    ff = layers.Dense(d_ff, activation='gelu', name=f'{name}_ff1')(x)
    ff = layers.Dropout(dropout, name=f'{name}_ff_dropout')(ff)
    ff = layers.Dense(d_model, name=f'{name}_ff2')(ff)
    x = layers.Add(name=f'{name}_ff_add')([x, layers.Dropout(dropout)(ff)])
    return layers.LayerNormalization(epsilon=1e-6, name=f'{name}_ff_norm')(x)


def residual_conv_block(x, filters=D_MODEL, kernel_size=3, dilation_rate=1,
                        dropout=DROPOUT, name='residual_conv'):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same', name=f'{name}_shortcut')(shortcut)

    y = layers.Conv1D(
        filters,
        kernel_size,
        padding='same',
        dilation_rate=dilation_rate,
        activation='gelu',
        kernel_regularizer=regularizers.l2(L2_REG),
        name=f'{name}_conv1',
    )(x)
    y = layers.Dropout(dropout, name=f'{name}_dropout1')(y)
    y = layers.Conv1D(
        filters,
        kernel_size,
        padding='same',
        dilation_rate=dilation_rate,
        kernel_regularizer=regularizers.l2(L2_REG),
        name=f'{name}_conv2',
    )(y)
    y = layers.Add(name=f'{name}_add')([shortcut, y])
    y = layers.Activation('gelu', name=f'{name}_gelu')(y)
    return layers.LayerNormalization(epsilon=1e-6, name=f'{name}_norm')(y)


def build_bilstm_model(input_shape):
    inputs = keras.Input(shape=input_shape, name='history_features')
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=DROPOUT),
        name='bilstm_1',
    )(inputs)
    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False, dropout=DROPOUT),
        name='bilstm_2',
    )(x)
    x = layers.Dense(D_FF, activation='gelu', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.Dropout(HEAD_DROPOUT)(x)
    outputs = layers.Dense(FORECAST_LEN, name='forecast_power')(x)
    return compile_forecast_model(keras.Model(inputs, outputs, name='WindBiLSTM'))


def build_cnn_lstm_model(input_shape):
    inputs = keras.Input(shape=input_shape, name='history_features')
    x = layers.Conv1D(64, 5, padding='same', activation='gelu', name='cnn_lstm_conv1')(inputs)
    x = layers.BatchNormalization(name='cnn_lstm_bn1')(x)
    x = layers.Conv1D(64, 3, padding='same', activation='gelu', name='cnn_lstm_conv2')(x)
    x = layers.Dropout(DROPOUT, name='cnn_lstm_conv_dropout')(x)
    x = layers.LSTM(64, return_sequences=False, dropout=DROPOUT, name='cnn_lstm_lstm')(x)
    x = layers.Dense(D_FF, activation='gelu', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.Dropout(HEAD_DROPOUT)(x)
    outputs = layers.Dense(FORECAST_LEN, name='forecast_power')(x)
    return compile_forecast_model(keras.Model(inputs, outputs, name='WindCNNLSTM'))


def build_cnn_resnet_gru_model(input_shape):
    inputs = keras.Input(shape=input_shape, name='history_features')
    x = layers.Conv1D(D_MODEL, 3, padding='same', activation='gelu', name='stem_conv')(inputs)
    for idx, dilation in enumerate([1, 2, 4]):
        x = residual_conv_block(x, D_MODEL, 3, dilation, name=f'resnet_block_{idx + 1}')
    x = layers.GRU(64, return_sequences=False, dropout=DROPOUT, name='resnet_gru')(x)
    x = layers.Dense(D_FF, activation='gelu', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.Dropout(HEAD_DROPOUT)(x)
    outputs = layers.Dense(FORECAST_LEN, name='forecast_power')(x)
    return compile_forecast_model(keras.Model(inputs, outputs, name='WindCNNResNetGRU'))


def build_wavenet_model(input_shape):
    inputs = keras.Input(shape=input_shape, name='history_features')
    x = layers.Conv1D(D_MODEL, 1, padding='same', name='wavenet_input_projection')(inputs)
    skips = []
    for idx, dilation in enumerate([1, 2, 4, 8, 16, 32]):
        tanh_out = layers.Conv1D(
            D_MODEL,
            2,
            padding='causal',
            dilation_rate=dilation,
            activation='tanh',
            name=f'wavenet_tanh_{idx + 1}',
        )(x)
        sigm_out = layers.Conv1D(
            D_MODEL,
            2,
            padding='causal',
            dilation_rate=dilation,
            activation='sigmoid',
            name=f'wavenet_sigmoid_{idx + 1}',
        )(x)
        gated = layers.Multiply(name=f'wavenet_gate_{idx + 1}')([tanh_out, sigm_out])
        skip = layers.Conv1D(D_MODEL, 1, padding='same', name=f'wavenet_skip_{idx + 1}')(gated)
        residual = layers.Conv1D(D_MODEL, 1, padding='same', name=f'wavenet_residual_{idx + 1}')(gated)
        x = layers.Add(name=f'wavenet_add_{idx + 1}')([x, residual])
        skips.append(skip)

    x = layers.Add(name='wavenet_skip_sum')(skips)
    x = layers.Activation('gelu', name='wavenet_skip_gelu')(x)
    x = layers.Conv1D(D_MODEL, 1, activation='gelu', name='wavenet_post_conv')(x)
    outputs = dense_forecast_head(x, name='wavenet_forecast')
    return compile_forecast_model(keras.Model(inputs, outputs, name='WindWaveNet'))


def build_transformer_model(input_shape):
    inputs = keras.Input(shape=input_shape, name='history_features')
    x = layers.Dense(D_MODEL, name='transformer_input_projection')(inputs)
    for idx in range(N_LAYERS):
        x = transformer_encoder_block(x, name=f'transformer_encoder_{idx + 1}')
    outputs = dense_forecast_head(x, name='transformer_forecast')
    return compile_forecast_model(keras.Model(inputs, outputs, name='WindTransformer'))


def get_time_feature_indices(input_cols):
    return [idx for idx, col in enumerate(input_cols) if col in TIME_FEATURE_COLS]


def build_informer_model(input_shape, input_cols=None):
    """
    Informer-style Keras implementation:
    TokenEmbedding + Positional/TemporalEmbedding + ProbSparse Attention encoder + distilling.
    """
    time_feature_indices = get_time_feature_indices(input_cols or [])
    inputs = keras.Input(shape=input_shape, name='history_features')
    x = InformerDataEmbedding(
        D_MODEL,
        time_feature_indices=time_feature_indices,
        dropout=DROPOUT,
        name='informer_data_embedding',
    )(inputs)
    for idx in range(N_LAYERS):
        x = informer_encoder_block(x, name=f'informer_encoder_{idx + 1}')
        if idx < N_LAYERS - 1:
            x = informer_distil_layer(x, name=f'informer_distil_{idx + 1}')
    outputs = dense_forecast_head(x, name='informer_forecast')
    return compile_forecast_model(keras.Model(inputs, outputs, name='WindInformer'))


def build_autoformer_model(input_shape, input_cols=None):
    """
    Autoformer-style Keras implementation:
    DataEmbedding_wo_pos + deep decomposition + series-wise autocorrelation decoder.
    """
    time_feature_indices = get_time_feature_indices(input_cols or [])
    inputs = keras.Input(shape=input_shape, name='history_features')
    x = AutoformerDataEmbedding(
        D_MODEL,
        time_feature_indices=time_feature_indices,
        dropout=DROPOUT,
        name='autoformer_data_embedding',
    )(inputs)
    seasonal_init, trend_init = SeriesDecomposition(
        AUTOFORMER_MOVING_AVG,
        name='autoformer_initial_decomp',
    )(x)

    enc_out = x
    for idx in range(N_LAYERS):
        enc_out = AutoformerEncoderLayer(
            D_MODEL,
            N_HEADS,
            D_FF,
            moving_avg=AUTOFORMER_MOVING_AVG,
            dropout=DROPOUT,
            name=f'autoformer_encoder_{idx + 1}',
        )(enc_out)
    enc_out = AutoformerLayerNorm(name='autoformer_encoder_norm')(enc_out)

    seasonal_dec, trend_dec = AutoformerDecoderInitializer(
        AUTOFORMER_LABEL_LEN,
        FORECAST_LEN,
        name='autoformer_decoder_init',
    )([seasonal_init, trend_init, x])
    dec_out = AutoformerDataEmbedding(
        D_MODEL,
        time_feature_indices=[],
        dropout=DROPOUT,
        name='autoformer_decoder_embedding',
    )(seasonal_dec)

    for idx in range(N_LAYERS):
        dec_out, residual_trend = AutoformerDecoderLayer(
            D_MODEL,
            N_HEADS,
            D_FF,
            moving_avg=AUTOFORMER_MOVING_AVG,
            dropout=DROPOUT,
            name=f'autoformer_decoder_{idx + 1}',
        )([dec_out, enc_out])
        trend_dec = layers.Add(name=f'autoformer_decoder_trend_add_{idx + 1}')(
            [trend_dec, residual_trend])

    seasonal_part = AutoformerLayerNorm(name='autoformer_decoder_norm')(dec_out)
    dec_out = layers.Add(name='autoformer_seasonal_trend_add')(
        [seasonal_part, trend_dec])
    forecast_seq = layers.Cropping1D(
        cropping=(AUTOFORMER_LABEL_LEN, 0),
        name='autoformer_forecast_slice',
    )(dec_out)
    forecast_seq = layers.Dense(1, name='autoformer_power_projection')(forecast_seq)
    outputs = layers.Flatten(name='autoformer_forecast_power')(forecast_seq)
    return compile_forecast_model(keras.Model(inputs, outputs, name='WindAutoformer'))


MODEL_BUILDERS = {
    'bilstm': build_bilstm_model,
    'cnn_lstm': build_cnn_lstm_model,
    'cnn_resnet_gru': build_cnn_resnet_gru_model,
    'wavenet': build_wavenet_model,
    'transformer': build_transformer_model,
    'informer': build_informer_model,
    'autoformer': build_autoformer_model,
}


def train_one_model_for_farm(model_name, train_file):
    farm_id = get_farm_id(train_file)
    dirs = model_dirs(model_name)
    print(f'\n===== 训练模型 {model_name} / 风电场 {farm_id} =====')

    train_df, feature_cols, capacity = load_and_preprocess(train_file, is_train=True)
    features, target, input_cols, target_index, scaler_x, scaler_y = build_scaled_arrays(
        train_df, feature_cols)
    train_ds, val_ds, train_samples, total_samples = make_window_dataset(
        features,
        target,
        HISTORY_LEN,
        FORECAST_LEN,
        BATCH_SIZE,
        VALIDATION_SPLIT,
    )

    print(f'数据形状: {train_df.shape}')
    print(f'输入通道数: {len(input_cols)}，样本数: {total_samples}，训练/验证: {train_samples}/{total_samples - train_samples}')

    builder = MODEL_BUILDERS[model_name]
    if model_name in {'informer', 'autoformer'}:
        model = builder((HISTORY_LEN, len(input_cols)), input_cols=input_cols)
    else:
        model = builder((HISTORY_LEN, len(input_cols)))
    model.summary()

    model_path = os.path.join(dirs['models'], f'{model_name}_farm_{farm_id}.keras')
    best_weights_path = os.path.join(dirs['weights'], f'{model_name}_farm_{farm_id}_best.weights.h5')
    tensorboard_log_dir = os.path.join(
        dirs['tensorboard'],
        f'farm_{farm_id}',
        datetime.now().strftime('%Y%m%d-%H%M%S'),
    )

    callbacks = [
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

    if os.path.exists(best_weights_path):
        model.load_weights(best_weights_path)
    model.save(model_path)

    history_path, history_plot_path = save_history_artifacts(history, model_name, farm_id, dirs)
    eval_metrics = evaluate_model(model, val_ds, scaler_y, capacity)
    print(
        f"验证集反归一化 MAE: {eval_metrics['val_inverse_mae']:.4f}, "
        f"RMSE: {eval_metrics['val_inverse_rmse']:.4f}"
    )

    artifact = {
        'model_name': model_name,
        'farm_id': farm_id,
        'feature_cols': feature_cols,
        'input_cols': input_cols,
        'target_col': TARGET_COL,
        'target_index': target_index,
        'time_feature_indices': get_time_feature_indices(input_cols),
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'capacity': capacity,
        'history_len': HISTORY_LEN,
        'forecast_len': FORECAST_LEN,
        'time_freq': TIME_FREQ,
        'autoformer_label_len': AUTOFORMER_LABEL_LEN if model_name == 'autoformer' else None,
        'autoformer_moving_avg': AUTOFORMER_MOVING_AVG if model_name == 'autoformer' else None,
        'autoformer_factor': AUTOFORMER_FACTOR if model_name == 'autoformer' else None,
        'model_path': model_path,
        'best_weights_path': best_weights_path,
        'tensorboard_log_dir': tensorboard_log_dir,
        'history_path': history_path,
        'history_plot_path': history_plot_path,
        **eval_metrics,
    }
    artifact_path = os.path.join(dirs['preprocess'], f'{model_name}_farm_{farm_id}_preprocess.pkl')
    joblib.dump(artifact, artifact_path)

    tail_path = os.path.join(dirs['tails'], f'{model_name}_tail_farm_{farm_id}.csv')
    train_df.iloc[-HISTORY_LEN:].to_csv(tail_path, index=True)

    result = {
        'model_name': model_name,
        'farm_id': farm_id,
        'train_file': train_file,
        'model_path': model_path,
        'best_weights_path': best_weights_path,
        'artifact_path': artifact_path,
        'tail_path': tail_path,
        'tensorboard_log_dir': tensorboard_log_dir,
        'history_path': history_path,
        'history_plot_path': history_plot_path,
        'train_samples': train_samples,
        'val_samples': total_samples - train_samples,
        **eval_metrics,
    }
    return result


def train_model_family(model_name, train_files):
    dirs = model_dirs(model_name)
    rows = []
    for train_file in train_files:
        rows.append(train_one_model_for_farm(model_name, train_file))

    metrics = pd.DataFrame(rows)
    metrics_path = os.path.join(dirs['root'], f'{model_name}_training_metrics.csv')
    metrics.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f'\n{model_name} 训练完成，指标已保存至 {metrics_path}')
    print(f'TensorBoard: tensorboard --logdir {dirs["tensorboard"]}')
    return metrics


if __name__ == '__main__':
    set_global_seed(seed)

    train_files = discover_train_files(DATA_DIR)
    if not train_files:
        raise FileNotFoundError(f'未在 {DATA_DIR} 找到 {TRAIN_FILE_PATTERN}')

    requested_model_names = get_requested_model_names()
    print(f'发现 {len(train_files)} 个风电训练文件')
    print(f'将训练模型: {requested_model_names}')
    print(f'训练轮数: {EPOCHS}, batch_size: {BATCH_SIZE}, validation_split: {VALIDATION_SPLIT}')

    all_metrics = []
    for model_name in requested_model_names:
        all_metrics.append(train_model_family(model_name, train_files))

    summary = pd.concat(all_metrics, ignore_index=True)
    summary_path = os.path.join(BASE_RESULT_DIR, 'wind_dl_other_models_training_metrics.csv')
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f'\n全部深度学习对比模型训练完成，汇总指标已保存至 {summary_path}')
