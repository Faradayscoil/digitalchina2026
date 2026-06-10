import glob
import os
import random
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers

warnings.filterwarnings('ignore')


DATA_DIR = r'./wind_split'
MODEL_DIR = r'./wind_results/patchtst'
TRAIN_FILE_PATTERN = 'wind_train_*.csv'

seed = 2026
TIME_FREQ = '15min'
HISTORY_LEN = 96          # 24小时历史窗口
FORECAST_LEN = 16         # 未来4小时预测
TARGET_COL = '功率'

BATCH_SIZE = 256
EPOCHS = 80
VALIDATION_SPLIT = 0.15
LEARNING_RATE = 5e-4

PATCH_LEN = 16            # 每个patch覆盖4小时
PATCH_STRIDE = 8          # patch之间重叠2小时
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 128
DROPOUT = 0.15
HEAD_DROPOUT = 0.2
USE_POWER_HISTORY = True

WIND_SPEED_COLS = ['10米风速', '30米风速', '50米风速', '70米风速', '轮毂高度风速']
WIND_DIR_COLS = ['10米风向', '30米风向', '50米风向', '70米风向', '轮毂高度风向']
WEATHER_LIMITS = {
    '10m气温': (-50, 60),
    '10m气压': (850, 1100),
    '10m湿度': (0, 100),
}


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


def repair_weather_column_swaps(df):
    """修复个别行中气压/湿度疑似互换的异常值。"""
    if '10m气压' in df.columns and '10m湿度' in df.columns:
        swap_mask = (df['10m气压'] < 300) & (df['10m湿度'] > 300)
        if swap_mask.any():
            pressure = df.loc[swap_mask, '10m气压'].copy()
            df.loc[swap_mask, '10m气压'] = df.loc[swap_mask, '10m湿度']
            df.loc[swap_mask, '10m湿度'] = pressure
    return df


def add_time_features(df):
    minute_of_day = df.index.hour * 60 + df.index.minute
    day_of_year = df.index.dayofyear
    day_of_week = df.index.dayofweek
    month = df.index.month

    df['minute_sin'] = np.sin(2 * np.pi * minute_of_day / 1440)
    df['minute_cos'] = np.cos(2 * np.pi * minute_of_day / 1440)
    df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    df['doy_sin'] = np.sin(2 * np.pi * day_of_year / 366)
    df['doy_cos'] = np.cos(2 * np.pi * day_of_year / 366)
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    return df


def add_wind_physics_features(df):
    speed_cols = [col for col in WIND_SPEED_COLS if col in df.columns]
    for col in speed_cols:
        df[f'{col}_sq'] = df[col] ** 2
        df[f'{col}_cube'] = df[col] ** 3

    hub_col = '轮毂高度风速'
    if hub_col in df.columns:
        for col in speed_cols:
            if col == hub_col:
                continue
            df[f'{hub_col}_minus_{col}'] = df[hub_col] - df[col]
            df[f'{hub_col}_ratio_{col}'] = df[hub_col] / df[col].clip(lower=0.5)
    return df


def load_and_preprocess(data_path, is_train=True):
    """
    风电短期预测预处理：
    1. 恢复15分钟等间隔序列；
    2. 清洗天气/功率异常；
    3. 风向转sin/cos；
    4. 添加时间周期、风速平方/立方、垂直风切变特征。
    """
    df = pd.read_csv(data_path, parse_dates=['时间'])
    df = df.sort_values('时间').drop_duplicates('时间')
    df.set_index('时间', inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    capacity = None
    if '装机' in df.columns:
        capacity_values = df['装机'].replace(0, np.nan).dropna()
        if not capacity_values.empty:
            capacity = float(capacity_values.median())

    df = repair_weather_column_swaps(df)
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=TIME_FREQ)
    df = df.reindex(full_index)
    df.index.name = '时间'

    if '装机' in df.columns:
        if capacity is not None:
            df['装机'] = df['装机'].fillna(capacity)
        df.drop(columns=['装机'], inplace=True)

    for col in WIND_SPEED_COLS:
        if col in df.columns:
            df[col] = df[col].where(df[col].between(0, 60))

    for col, (lower, upper) in WEATHER_LIMITS.items():
        if col in df.columns:
            df[col] = df[col].where(df[col].between(lower, upper))

    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].clip(lower=0)
        if capacity is not None:
            df[TARGET_COL] = df[TARGET_COL].clip(upper=capacity)
    elif is_train:
        raise ValueError(f"训练数据中缺少目标列 '{TARGET_COL}'")

    for col in WIND_DIR_COLS:
        if col in df.columns:
            radians = np.deg2rad(df[col] % 360)
            df[f'{col}_sin'] = np.sin(radians)
            df[f'{col}_cos'] = np.cos(radians)
            df.drop(columns=[col], inplace=True)

    df.interpolate(method='time', limit_direction='both', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)

    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].clip(lower=0)
        if capacity is not None:
            df[TARGET_COL] = df[TARGET_COL].clip(upper=capacity)

    df = add_time_features(df)
    df = add_wind_physics_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.interpolate(method='time', limit_direction='both', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)

    feature_cols = [col for col in df.columns if col != TARGET_COL]
    df = df.astype(np.float32)
    return df, feature_cols, capacity


def build_scaled_arrays(train_df, feature_cols):
    input_cols = feature_cols.copy()
    if USE_POWER_HISTORY and TARGET_COL in train_df.columns:
        input_cols.append(TARGET_COL)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    features = scaler_x.fit_transform(train_df[input_cols].values).astype(np.float32)
    target = scaler_y.fit_transform(train_df[[TARGET_COL]].values).ravel().astype(np.float32)
    target_index = input_cols.index(TARGET_COL) if TARGET_COL in input_cols else None
    return features, target, input_cols, target_index, scaler_x, scaler_y


def make_window_dataset(features, target, history_len, forecast_len, batch_size,
                        validation_split, shuffle_train=True):
    n_samples = len(features) - history_len - forecast_len + 1
    if n_samples <= 0:
        raise ValueError("数据量不足，无法构造完整历史窗口和预测窗口")

    target_windows = np.lib.stride_tricks.sliding_window_view(target, forecast_len)
    target_windows = target_windows[history_len:history_len + n_samples].astype(np.float32)

    split_idx = int(n_samples * (1 - validation_split))
    split_idx = max(1, min(split_idx, n_samples - 1))

    def _dataset(start, sample_count, shuffle):
        data_slice = features[start:start + sample_count + history_len - 1]
        target_slice = target_windows[start:start + sample_count]
        ds = keras.utils.timeseries_dataset_from_array(
            data=data_slice,
            targets=target_slice,
            sequence_length=history_len,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=batch_size,
            seed=seed if shuffle else None,
        )
        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds = _dataset(0, split_idx, shuffle_train)
    val_ds = _dataset(split_idx, n_samples - split_idx, False)
    return train_ds, val_ds, split_idx, n_samples


def compute_patch_num(context_window, patch_len, stride, padding_patch='end'):
    effective_len = context_window + stride if padding_patch == 'end' else context_window
    return int((effective_len - patch_len) / stride + 1)


@keras.utils.register_keras_serializable(package='WindPatchTST')
class PatchExtract(layers.Layer):
    def __init__(self, patch_len, stride, padding_patch='end', **kwargs):
        super().__init__(**kwargs)
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

    def call(self, inputs):
        x = tf.transpose(inputs, [0, 2, 1])
        if self.padding_patch == 'end':
            last = x[:, :, -1:]
            x = tf.concat([x, tf.repeat(last, repeats=self.stride, axis=-1)], axis=-1)
        return tf.signal.frame(x, frame_length=self.patch_len, frame_step=self.stride, axis=-1)

    def compute_output_shape(self, input_shape):
        seq_len = input_shape[1]
        patch_num = None
        if seq_len is not None:
            patch_num = compute_patch_num(seq_len, self.patch_len, self.stride, self.padding_patch)
        return input_shape[0], input_shape[2], patch_num, self.patch_len

    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_len': self.patch_len,
            'stride': self.stride,
            'padding_patch': self.padding_patch,
        })
        return config


@keras.utils.register_keras_serializable(package='WindPatchTST')
class MergeChannels(layers.Layer):
    def call(self, inputs):
        shape = tf.shape(inputs)
        return tf.reshape(inputs, [shape[0] * shape[1], shape[2], shape[3]])

    def compute_output_shape(self, input_shape):
        return None, input_shape[2], input_shape[3]


@keras.utils.register_keras_serializable(package='WindPatchTST')
class RestoreChannels(layers.Layer):
    def __init__(self, n_channels, patch_num, d_model, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.patch_num = patch_num
        self.d_model = d_model

    def call(self, inputs):
        batch = tf.shape(inputs)[0] // self.n_channels
        return tf.reshape(inputs, [batch, self.n_channels, self.patch_num, self.d_model])

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.n_channels, self.patch_num, self.d_model

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_channels': self.n_channels,
            'patch_num': self.patch_num,
            'd_model': self.d_model,
        })
        return config


@keras.utils.register_keras_serializable(package='WindPatchTST')
class LearnablePositionEmbedding(layers.Layer):
    def __init__(self, patch_num, d_model, **kwargs):
        super().__init__(**kwargs)
        self.patch_num = patch_num
        self.d_model = d_model

    def build(self, input_shape):
        self.position = self.add_weight(
            name='position',
            shape=(1, self.patch_num, self.d_model),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, inputs):
        return inputs + self.position

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'patch_num': self.patch_num, 'd_model': self.d_model})
        return config


@keras.utils.register_keras_serializable(package='WindPatchTST')
class TakeChannel(layers.Layer):
    def __init__(self, channel_index, **kwargs):
        super().__init__(**kwargs)
        self.channel_index = channel_index

    def call(self, inputs):
        return inputs[:, self.channel_index, :, :]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2], input_shape[3]

    def get_config(self):
        config = super().get_config()
        config.update({'channel_index': self.channel_index})
        return config


def transformer_encoder(x, d_model, n_heads, d_ff, dropout, name):
    attn = layers.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=d_model // n_heads,
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


def build_patchtst_model(input_dim, target_channel_index):
    if target_channel_index is None:
        raise ValueError("PatchTST短期风电模型需要将历史功率作为输入通道")

    patch_num = compute_patch_num(HISTORY_LEN, PATCH_LEN, PATCH_STRIDE)
    inputs = keras.Input(shape=(HISTORY_LEN, input_dim), name='history_features')

    x = PatchExtract(PATCH_LEN, PATCH_STRIDE, name='patch_extract')(inputs)
    x = layers.Dense(D_MODEL, name='patch_projection')(x)
    x = MergeChannels(name='merge_channels')(x)
    x = LearnablePositionEmbedding(patch_num, D_MODEL, name='position_embedding')(x)
    x = layers.Dropout(DROPOUT, name='patch_dropout')(x)

    for idx in range(N_LAYERS):
        x = transformer_encoder(x, D_MODEL, N_HEADS, D_FF, DROPOUT, name=f'encoder_{idx + 1}')

    x = RestoreChannels(input_dim, patch_num, D_MODEL, name='restore_channels')(x)
    target_repr = TakeChannel(target_channel_index, name='target_power_channel')(x)
    target_repr = layers.Flatten(name='target_flatten')(target_repr)
    global_context = layers.GlobalAveragePooling2D(name='channel_context_pool')(x)

    head = layers.Concatenate(name='forecast_context')([target_repr, global_context])
    head = layers.Dropout(HEAD_DROPOUT, name='head_dropout')(head)
    head = layers.Dense(
        D_FF,
        activation='gelu',
        kernel_regularizer=regularizers.l2(1e-4),
        name='forecast_ff',
    )(head)
    head = layers.Dropout(HEAD_DROPOUT, name='forecast_dropout')(head)
    outputs = layers.Dense(FORECAST_LEN, name='forecast_power')(head)

    model = keras.Model(inputs=inputs, outputs=outputs, name='WindPatchTST')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss=keras.losses.Huber(delta=1.0),
        metrics=[
            keras.metrics.MeanAbsoluteError(name='mae'),
            keras.metrics.RootMeanSquaredError(name='rmse'),
        ],
    )
    return model


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
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse


def train_one_farm(train_file):
    farm_id = get_farm_id(train_file)
    print(f'\n===== 训练风电场 {farm_id} =====')

    train_df, feature_cols, capacity = load_and_preprocess(train_file, is_train=True)
    features, target, input_cols, target_index, scaler_x, scaler_y = build_scaled_arrays(
        train_df, feature_cols)
    train_ds, val_ds, train_samples, total_samples = make_window_dataset(
        features, target, HISTORY_LEN, FORECAST_LEN, BATCH_SIZE, VALIDATION_SPLIT)

    print(f'原始/重采样后数据形状: {train_df.shape}')
    print(f'输入通道数: {len(input_cols)}，样本数: {total_samples}，训练/验证: {train_samples}/{total_samples - train_samples}')
    print(f'Patch设置: patch_len={PATCH_LEN}, stride={PATCH_STRIDE}, patch_num={compute_patch_num(HISTORY_LEN, PATCH_LEN, PATCH_STRIDE)}')

    model = build_patchtst_model(len(input_cols), target_index)
    model.summary()

    model_path = os.path.join(MODEL_DIR, f'patchtst_farm_{farm_id}.keras')
    callbacks = [
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
            model_path,
            monitor='val_loss',
            save_best_only=True,
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

    mae, rmse = evaluate_model(model, val_ds, scaler_y, capacity)
    print(f'验证集反归一化 MAE: {mae:.4f}, RMSE: {rmse:.4f}')

    artifact = {
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
        'model_path': model_path,
        'val_mae': mae,
        'val_rmse': rmse,
    }
    artifact_path = os.path.join(MODEL_DIR, f'patchtst_farm_{farm_id}_preprocess.pkl')
    joblib.dump(artifact, artifact_path)

    tail_path = os.path.join(MODEL_DIR, f'patchtst_tail_farm_{farm_id}.csv')
    train_df.iloc[-HISTORY_LEN:].to_csv(tail_path, index=True)

    return {
        'farm_id': farm_id,
        'model_path': model_path,
        'artifact_path': artifact_path,
        'tail_path': tail_path,
        'history': history.history,
        'val_mae': mae,
        'val_rmse': rmse,
    }


if __name__ == '__main__':
    set_global_seed(seed)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_files = discover_train_files(DATA_DIR)
    if not train_files:
        raise FileNotFoundError(f'未在 {DATA_DIR} 找到风电训练文件')

    print(f'发现 {len(train_files)} 个风电训练文件')
    results = []
    for file_path in train_files:
        results.append(train_one_farm(file_path))

    metrics = pd.DataFrame([
        {
            'farm_id': item['farm_id'],
            'val_mae': item['val_mae'],
            'val_rmse': item['val_rmse'],
            'model_path': item['model_path'],
            'artifact_path': item['artifact_path'],
        }
        for item in results
    ])
    metrics_path = os.path.join(MODEL_DIR, 'patchtst_training_metrics.csv')
    metrics.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f'\n训练完成，指标已保存至 {metrics_path}')
