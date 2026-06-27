import glob
import json
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
from tensorflow.keras import mixed_precision

from wind_dl_model_train import (
    DATA_DIR,
    D_FF,
    D_MODEL,
    DROPOUT,
    FORECAST_LEN,
    HEAD_DROPOUT,
    HISTORY_LEN,
    N_HEADS,
    N_LAYERS,
    PATCH_LEN,
    PATCH_STRIDE,
    TARGET_COL,
    TIME_FREQ,
    TRAIN_FILE_PATTERN,
    USE_POWER_HISTORY,
    LearnablePositionEmbedding,
    MergeChannels,
    PatchExtract,
    RestoreChannels,
    TakeChannel,
    build_scaled_arrays,
    compute_patch_num,
    load_and_preprocess,
    transformer_encoder,
)

warnings.filterwarnings('ignore')


TUNED_MODEL_NAME = 'tuned_patchtst'
MODEL_DIR = os.path.join('./wind_results', TUNED_MODEL_NAME)
SAVED_MODEL_DIR = os.path.join('./models', TUNED_MODEL_NAME)
WEIGHTS_DIR = os.path.join(MODEL_DIR, 'weights')
TEACHER_DIR = os.path.join(MODEL_DIR, 'teacher')
PREPROCESS_DIR = os.path.join(MODEL_DIR, 'preprocess')
HISTORY_DIR = os.path.join(MODEL_DIR, 'history')
TENSORBOARD_LOG_DIR = os.path.join(MODEL_DIR, 'tensorboard')
TAIL_DIR = os.path.join(MODEL_DIR, 'tails')
DISTILL_DIR = os.path.join(MODEL_DIR, 'distillation')

seed = 2026
BATCH_SIZE = int(os.getenv('WIND_TUNED_BATCH_SIZE', '192'))
COLD_START_EPOCHS = int(os.getenv('WIND_TUNED_COLD_START_EPOCHS', '25'))
DISTILL_EPOCHS = int(os.getenv('WIND_TUNED_DISTILL_EPOCHS', '45'))
VALIDATION_SPLIT = float(os.getenv('WIND_TUNED_VALIDATION_SPLIT', '0.15'))
BASE_LEARNING_RATE = float(os.getenv('WIND_TUNED_BASE_LR', '5e-4'))
DISTILL_LEARNING_RATE = float(os.getenv('WIND_TUNED_DISTILL_LR', '2.5e-4'))
WEIGHT_DECAY = float(os.getenv('WIND_TUNED_WEIGHT_DECAY', '1e-4'))

DISTILL_ALPHA = float(os.getenv('WIND_TUNED_DISTILL_ALPHA', '0.35'))
TEACHER_KEEP_RATIO = float(os.getenv('WIND_TUNED_TEACHER_KEEP_RATIO', '0.70'))
HORIZON_DECAY = float(os.getenv('WIND_TUNED_HORIZON_DECAY', '0.93'))
PHYSICAL_PENALTY_WEIGHT = float(os.getenv('WIND_TUNED_PHYSICAL_WEIGHT', '0.02'))
SMOOTHNESS_WEIGHT = float(os.getenv('WIND_TUNED_SMOOTHNESS_WEIGHT', '0.005'))
HUBER_DELTA = float(os.getenv('WIND_TUNED_HUBER_DELTA', '1.0'))

INPUT_NOISE_STD = float(os.getenv('WIND_TUNED_INPUT_NOISE_STD', '0.01'))
CHANNEL_DROPOUT = float(os.getenv('WIND_TUNED_CHANNEL_DROPOUT', '0.05'))
USE_MIXED_PRECISION = os.getenv('WIND_TUNED_MIXED_PRECISION', '1') == '1'


def configure_runtime():
    if USE_MIXED_PRECISION:
        try:
            mixed_precision.set_global_policy('mixed_float16')
            print('Mixed precision enabled: mixed_float16')
        except Exception as exc:
            print(f'Mixed precision setup failed, continue with default policy: {exc}')


def set_global_seed(value):
    random.seed(value)
    np.random.seed(value)
    tf.random.set_seed(value)
    try:
        keras.utils.set_random_seed(value)
    except AttributeError:
        pass


def ensure_dirs():
    for path in [
        MODEL_DIR,
        SAVED_MODEL_DIR,
        WEIGHTS_DIR,
        TEACHER_DIR,
        PREPROCESS_DIR,
        HISTORY_DIR,
        TENSORBOARD_LOG_DIR,
        TAIL_DIR,
        DISTILL_DIR,
    ]:
        os.makedirs(path, exist_ok=True)


def discover_train_files(data_dir=DATA_DIR):
    return sorted(glob.glob(os.path.join(data_dir, TRAIN_FILE_PATTERN)))


def get_farm_id(path):
    basename = os.path.basename(path)
    match = re.search(r'wind_train_(\d+)\.csv$', basename)
    if match:
        return match.group(1)
    return os.path.splitext(basename)[0]


def make_window_targets(features, target, history_len=HISTORY_LEN, forecast_len=FORECAST_LEN,
                        validation_split=VALIDATION_SPLIT):
    n_samples = len(features) - history_len - forecast_len + 1
    if n_samples <= 0:
        raise ValueError('数据量不足，无法构造完整历史窗口和预测窗口')

    target_windows = np.lib.stride_tricks.sliding_window_view(target, forecast_len)
    y = target_windows[history_len:history_len + n_samples].astype(np.float32)

    split_idx = int(n_samples * (1 - validation_split))
    split_idx = max(1, min(split_idx, n_samples - 1))
    return y[:split_idx], y[split_idx:], split_idx, n_samples


def combine_targets(y_true, teacher_pred=None, confidence=None):
    y_true = np.asarray(y_true, dtype=np.float32)
    if teacher_pred is None:
        teacher_pred = np.zeros_like(y_true, dtype=np.float32)
    else:
        teacher_pred = np.asarray(teacher_pred, dtype=np.float32)

    if confidence is None:
        confidence = np.zeros((len(y_true), 1), dtype=np.float32)
    else:
        confidence = np.asarray(confidence, dtype=np.float32).reshape(-1, 1)

    return np.concatenate([y_true, teacher_pred, confidence], axis=1).astype(np.float32)


def make_supervised_dataset(features, targets, start, sample_count,
                            history_len=HISTORY_LEN, batch_size=BATCH_SIZE,
                            shuffle=False):
    data_slice = features[start:start + sample_count + history_len - 1]
    ds = keras.utils.timeseries_dataset_from_array(
        data=data_slice,
        targets=targets,
        sequence_length=history_len,
        sequence_stride=1,
        shuffle=shuffle,
        batch_size=batch_size,
        seed=seed if shuffle else None,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def make_feature_dataset(features, start, sample_count,
                         history_len=HISTORY_LEN, batch_size=BATCH_SIZE):
    data_slice = features[start:start + sample_count + history_len - 1]
    ds = keras.utils.timeseries_dataset_from_array(
        data=data_slice,
        targets=None,
        sequence_length=history_len,
        sequence_stride=1,
        shuffle=False,
        batch_size=batch_size,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def horizon_weights(forecast_len=FORECAST_LEN, decay=HORIZON_DECAY):
    weights = decay ** np.arange(forecast_len, dtype=np.float32)
    weights = weights / np.mean(weights)
    return weights.astype(np.float32)


@keras.utils.register_keras_serializable(package='WindTunedPatchTST')
class RepeatLastTarget(layers.Layer):
    def __init__(self, target_channel_index, forecast_len=FORECAST_LEN, **kwargs):
        super().__init__(**kwargs)
        self.target_channel_index = target_channel_index
        self.forecast_len = forecast_len

    def call(self, inputs):
        last_value = inputs[:, -1, self.target_channel_index:self.target_channel_index + 1]
        return tf.repeat(last_value, repeats=self.forecast_len, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.forecast_len

    def get_config(self):
        config = super().get_config()
        config.update({
            'target_channel_index': self.target_channel_index,
            'forecast_len': self.forecast_len,
        })
        return config


@keras.utils.register_keras_serializable(package='WindTunedPatchTST')
class TunedPatchTSTLoss(keras.losses.Loss):
    def __init__(self, forecast_len=FORECAST_LEN, horizon_weight_values=None,
                 distill_alpha=DISTILL_ALPHA, zero_scaled=0.0, capacity_scaled=None,
                 physical_penalty_weight=PHYSICAL_PENALTY_WEIGHT,
                 smoothness_weight=SMOOTHNESS_WEIGHT, delta=HUBER_DELTA,
                 name='tuned_patchtst_loss'):
        super().__init__(name=name)
        self.forecast_len = forecast_len
        self.horizon_weight_values = (
            list(horizon_weight_values)
            if horizon_weight_values is not None
            else list(horizon_weights(forecast_len))
        )
        self.distill_alpha = distill_alpha
        self.zero_scaled = float(zero_scaled)
        self.capacity_scaled = None if capacity_scaled is None else float(capacity_scaled)
        self.physical_penalty_weight = physical_penalty_weight
        self.smoothness_weight = smoothness_weight
        self.delta = delta

    def _huber(self, error):
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        return 0.5 * tf.square(quadratic) + self.delta * linear

    def call(self, y_true, y_pred):
        actual = y_true[:, :self.forecast_len]
        teacher = y_true[:, self.forecast_len:2 * self.forecast_len]
        confidence = y_true[:, 2 * self.forecast_len:2 * self.forecast_len + 1]
        weights = tf.constant(self.horizon_weight_values, dtype=y_pred.dtype)
        weights = tf.reshape(weights, [1, self.forecast_len])

        supervised = tf.reduce_mean(weights * self._huber(actual - y_pred), axis=1)
        distill = tf.reduce_mean(weights * self._huber(teacher - y_pred), axis=1)
        loss = supervised + self.distill_alpha * tf.squeeze(confidence, axis=-1) * distill
        loss = tf.reduce_mean(loss)

        lower_penalty = tf.reduce_mean(tf.square(tf.nn.relu(self.zero_scaled - y_pred)))
        physical_penalty = lower_penalty
        if self.capacity_scaled is not None:
            upper_penalty = tf.reduce_mean(tf.square(tf.nn.relu(y_pred - self.capacity_scaled)))
            physical_penalty = physical_penalty + upper_penalty

        if self.forecast_len > 1:
            smoothness_penalty = tf.reduce_mean(tf.square(y_pred[:, 1:] - y_pred[:, :-1]))
        else:
            smoothness_penalty = 0.0

        return (
            loss
            + self.physical_penalty_weight * physical_penalty
            + self.smoothness_weight * smoothness_penalty
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'forecast_len': self.forecast_len,
            'horizon_weight_values': self.horizon_weight_values,
            'distill_alpha': self.distill_alpha,
            'zero_scaled': self.zero_scaled,
            'capacity_scaled': self.capacity_scaled,
            'physical_penalty_weight': self.physical_penalty_weight,
            'smoothness_weight': self.smoothness_weight,
            'delta': self.delta,
        })
        return config


@keras.utils.register_keras_serializable(package='WindTunedPatchTST')
def actual_mae(y_true, y_pred):
    actual = y_true[:, :FORECAST_LEN]
    return tf.reduce_mean(tf.abs(actual - y_pred))


@keras.utils.register_keras_serializable(package='WindTunedPatchTST')
def actual_rmse(y_true, y_pred):
    actual = y_true[:, :FORECAST_LEN]
    return tf.sqrt(tf.reduce_mean(tf.square(actual - y_pred)))


def scaled_bounds(scaler_y, capacity=None):
    zero_scaled = float(scaler_y.transform([[0.0]])[0, 0])
    capacity_scaled = None
    if capacity is not None and capacity > 0:
        capacity_scaled = float(scaler_y.transform([[capacity]])[0, 0])
    return zero_scaled, capacity_scaled


def make_optimizer(initial_lr, steps_per_epoch, epochs):
    decay_steps = max(1, int(steps_per_epoch * max(1, epochs)))
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        alpha=0.1,
    )
    try:
        return keras.optimizers.AdamW(
            learning_rate=schedule,
            weight_decay=WEIGHT_DECAY,
            clipnorm=1.0,
        )
    except AttributeError:
        return keras.optimizers.Adam(learning_rate=schedule, clipnorm=1.0)


def compile_tuned_model(model, scaler_y, capacity, initial_lr, steps_per_epoch,
                        epochs, distill_alpha):
    zero_scaled, capacity_scaled = scaled_bounds(scaler_y, capacity)
    loss = TunedPatchTSTLoss(
        forecast_len=FORECAST_LEN,
        horizon_weight_values=horizon_weights(),
        distill_alpha=distill_alpha,
        zero_scaled=zero_scaled,
        capacity_scaled=capacity_scaled,
    )
    model.compile(
        optimizer=make_optimizer(initial_lr, steps_per_epoch, epochs),
        loss=loss,
        metrics=[actual_mae, actual_rmse],
    )
    return model


def build_tuned_patchtst_model(input_dim, target_channel_index):
    if target_channel_index is None:
        raise ValueError('Tuned PatchTST 需要将历史功率作为输入通道')

    patch_num = compute_patch_num(HISTORY_LEN, PATCH_LEN, PATCH_STRIDE)
    inputs = keras.Input(shape=(HISTORY_LEN, input_dim), name='history_features')

    x_input = layers.GaussianNoise(INPUT_NOISE_STD, name='input_noise')(inputs)
    x_input = layers.SpatialDropout1D(CHANNEL_DROPOUT, name='channel_dropout')(x_input)

    x = PatchExtract(PATCH_LEN, PATCH_STRIDE, name='patch_extract')(x_input)
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
    residual = layers.Dense(
        FORECAST_LEN,
        kernel_initializer='zeros',
        bias_initializer='zeros',
        dtype='float32',
        name='forecast_residual',
    )(head)
    baseline = RepeatLastTarget(
        target_channel_index,
        FORECAST_LEN,
        name='persistence_baseline',
    )(inputs)
    outputs = layers.Add(name='forecast_power')([baseline, residual])
    outputs = layers.Activation('linear', dtype='float32', name='forecast_power_float32')(outputs)
    return keras.Model(inputs=inputs, outputs=outputs, name='WindTunedPatchTST')


def inverse_power(scaler_y, values):
    values = np.asarray(values).reshape(-1, 1)
    return scaler_y.inverse_transform(values).reshape(-1)


def evaluate_model(model, val_feature_ds, y_val, scaler_y, capacity=None):
    y_pred_scaled = model.predict(val_feature_ds, verbose=0)
    y_true = inverse_power(scaler_y, y_val)
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
    return {
        'val_inverse_mae': float(mae),
        'val_inverse_mse': float(mse),
        'val_inverse_rmse': rmse,
        'val_inverse_r2': r2,
    }


def teacher_confidence(y_true, teacher_pred, keep_ratio=TEACHER_KEEP_RATIO):
    sample_mae = np.mean(np.abs(y_true - teacher_pred), axis=1)
    threshold = float(np.quantile(sample_mae, keep_ratio))
    confidence = (sample_mae <= threshold).astype(np.float32)
    return confidence, sample_mae, threshold


def save_distillation_stats(farm_id, train_mae, train_conf, train_threshold,
                            val_mae, val_conf, val_threshold):
    rows = []
    for split, mae_values, conf, threshold in [
        ('train', train_mae, train_conf, train_threshold),
        ('validation', val_mae, val_conf, val_threshold),
    ]:
        rows.append({
            'farm_id': farm_id,
            'split': split,
            'samples': int(len(mae_values)),
            'teacher_keep_ratio': TEACHER_KEEP_RATIO,
            'accepted_samples': int(np.sum(conf > 0)),
            'accepted_ratio': float(np.mean(conf > 0)),
            'teacher_mae_mean_scaled': float(np.mean(mae_values)),
            'teacher_mae_median_scaled': float(np.median(mae_values)),
            'teacher_mae_p70_scaled': float(np.quantile(mae_values, 0.70)),
            'teacher_mae_p90_scaled': float(np.quantile(mae_values, 0.90)),
            'acceptance_threshold_scaled': threshold,
        })
    stats_df = pd.DataFrame(rows)
    stats_path = os.path.join(DISTILL_DIR, f'tuned_patchtst_distillation_stats_farm_{farm_id}.csv')
    stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
    return stats_path


def save_history_artifacts(histories, farm_id):
    frames = []
    for stage, history in histories:
        history_df = pd.DataFrame(history.history)
        history_df.insert(0, 'stage', stage)
        history_df.insert(1, 'epoch', np.arange(1, len(history_df) + 1))
        frames.append(history_df)

    combined = pd.concat(frames, ignore_index=True)
    history_path = os.path.join(HISTORY_DIR, f'tuned_patchtst_history_farm_{farm_id}.csv')
    combined.to_csv(history_path, index=False, encoding='utf-8-sig')

    plot_path = os.path.join(HISTORY_DIR, f'tuned_patchtst_history_farm_{farm_id}.png')
    try:
        cache_dir = os.path.join(MODEL_DIR, 'matplotlib_cache')
        os.environ['MPLCONFIGDIR'] = cache_dir
        os.environ['XDG_CACHE_HOME'] = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        metric_names = [
            key for key in combined.columns
            if key not in {'stage', 'epoch'} and not key.startswith('val_')
            and f'val_{key}' in combined.columns
        ]
        if not metric_names:
            metric_names = ['loss']
        fig, axes = plt.subplots(
            len(metric_names),
            1,
            figsize=(10, max(3, 2.8 * len(metric_names))),
            sharex=False,
        )
        if len(metric_names) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metric_names):
            for stage, stage_df in combined.groupby('stage'):
                ax.plot(stage_df['epoch'], stage_df[metric], label=f'{stage}_{metric}')
                val_metric = f'val_{metric}'
                if val_metric in stage_df:
                    ax.plot(stage_df['epoch'], stage_df[val_metric], label=f'{stage}_{val_metric}')
            ax.set_title(metric)
            ax.set_xlabel('epoch within stage')
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle(f'Tuned PatchTST Training History - Farm {farm_id}', y=1.0)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as exc:
        plot_path = None
        print(f'训练曲线图片保存失败: {exc}')

    return history_path, plot_path


def save_config():
    config = {
        'model_name': TUNED_MODEL_NAME,
        'data_dir': DATA_DIR,
        'history_len': HISTORY_LEN,
        'forecast_len': FORECAST_LEN,
        'patch_len': PATCH_LEN,
        'patch_stride': PATCH_STRIDE,
        'd_model': D_MODEL,
        'n_heads': N_HEADS,
        'n_layers': N_LAYERS,
        'd_ff': D_FF,
        'batch_size': BATCH_SIZE,
        'cold_start_epochs': COLD_START_EPOCHS,
        'distill_epochs': DISTILL_EPOCHS,
        'validation_split': VALIDATION_SPLIT,
        'base_learning_rate': BASE_LEARNING_RATE,
        'distill_learning_rate': DISTILL_LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'distill_alpha': DISTILL_ALPHA,
        'teacher_keep_ratio': TEACHER_KEEP_RATIO,
        'horizon_decay': HORIZON_DECAY,
        'physical_penalty_weight': PHYSICAL_PENALTY_WEIGHT,
        'smoothness_weight': SMOOTHNESS_WEIGHT,
        'input_noise_std': INPUT_NOISE_STD,
        'channel_dropout': CHANNEL_DROPOUT,
        'use_mixed_precision': USE_MIXED_PRECISION,
        'use_power_history': USE_POWER_HISTORY,
    }
    config_path = os.path.join(MODEL_DIR, 'tuned_patchtst_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return config_path


def train_one_farm(train_file):
    farm_id = get_farm_id(train_file)
    print(f'\n===== 训练 tuned PatchTST / 风电场 {farm_id} =====')

    train_df, feature_cols, capacity = load_and_preprocess(train_file, is_train=True)
    features, target, input_cols, target_index, scaler_x, scaler_y = build_scaled_arrays(
        train_df, feature_cols)
    y_train, y_val, train_samples, total_samples = make_window_targets(
        features,
        target,
        HISTORY_LEN,
        FORECAST_LEN,
        VALIDATION_SPLIT,
    )

    print(f'数据形状: {train_df.shape}')
    print(f'输入通道数: {len(input_cols)}，样本数: {total_samples}，训练/验证: {train_samples}/{total_samples - train_samples}')
    print(f'Patch设置: patch_len={PATCH_LEN}, stride={PATCH_STRIDE}, patch_num={compute_patch_num(HISTORY_LEN, PATCH_LEN, PATCH_STRIDE)}')
    print('训练流程: cold-start teacher -> filtered self-distillation student')

    y_train_stage1 = combine_targets(y_train)
    y_val_stage1 = combine_targets(y_val)
    val_samples = total_samples - train_samples
    steps_per_epoch = int(np.ceil(train_samples / BATCH_SIZE))
    train_ds_stage1 = make_supervised_dataset(
        features,
        y_train_stage1,
        start=0,
        sample_count=train_samples,
        shuffle=True,
    )
    val_ds_stage1 = make_supervised_dataset(
        features,
        y_val_stage1,
        start=train_samples,
        sample_count=val_samples,
        shuffle=False,
    )

    model = build_tuned_patchtst_model(len(input_cols), target_index)
    compile_tuned_model(
        model,
        scaler_y,
        capacity,
        BASE_LEARNING_RATE,
        steps_per_epoch,
        COLD_START_EPOCHS,
        distill_alpha=0.0,
    )
    model.summary()

    teacher_weights_path = os.path.join(
        WEIGHTS_DIR,
        f'tuned_patchtst_teacher_farm_{farm_id}_best.weights.h5',
    )
    teacher_model_path = os.path.join(
        TEACHER_DIR,
        f'tuned_patchtst_teacher_farm_{farm_id}.keras',
    )
    final_weights_path = os.path.join(
        WEIGHTS_DIR,
        f'tuned_patchtst_farm_{farm_id}_best.weights.h5',
    )
    final_model_path = os.path.join(
        SAVED_MODEL_DIR,
        f'tuned_patchtst_farm_{farm_id}.keras',
    )

    teacher_log_dir = os.path.join(
        TENSORBOARD_LOG_DIR,
        f'farm_{farm_id}',
        'cold_start',
        datetime.now().strftime('%Y%m%d-%H%M%S'),
    )
    distill_log_dir = os.path.join(
        TENSORBOARD_LOG_DIR,
        f'farm_{farm_id}',
        'distill',
        datetime.now().strftime('%Y%m%d-%H%M%S'),
    )

    teacher_callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=teacher_log_dir,
            histogram_freq=0,
            write_graph=True,
            update_freq='epoch',
            profile_batch=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            teacher_weights_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    teacher_history = model.fit(
        train_ds_stage1,
        validation_data=val_ds_stage1,
        epochs=COLD_START_EPOCHS,
        callbacks=teacher_callbacks,
        verbose=1,
    )

    if os.path.exists(teacher_weights_path):
        model.load_weights(teacher_weights_path)
    model.save(teacher_model_path)

    train_feature_ds = make_feature_dataset(features, 0, train_samples)
    val_feature_ds = make_feature_dataset(features, train_samples, val_samples)
    teacher_train_pred = model.predict(train_feature_ds, verbose=0)
    teacher_val_pred = model.predict(val_feature_ds, verbose=0)
    train_conf, train_teacher_mae, train_threshold = teacher_confidence(
        y_train,
        teacher_train_pred,
        TEACHER_KEEP_RATIO,
    )
    val_conf, val_teacher_mae, val_threshold = teacher_confidence(
        y_val,
        teacher_val_pred,
        TEACHER_KEEP_RATIO,
    )
    distill_stats_path = save_distillation_stats(
        farm_id,
        train_teacher_mae,
        train_conf,
        train_threshold,
        val_teacher_mae,
        val_conf,
        val_threshold,
    )
    print(
        f'teacher筛选: train accepted={np.mean(train_conf > 0):.3f}, '
        f'val accepted={np.mean(val_conf > 0):.3f}'
    )

    y_train_stage2 = combine_targets(y_train, teacher_train_pred, train_conf)
    y_val_stage2 = combine_targets(y_val, teacher_val_pred, val_conf)
    train_ds_stage2 = make_supervised_dataset(
        features,
        y_train_stage2,
        start=0,
        sample_count=train_samples,
        shuffle=True,
    )
    val_ds_stage2 = make_supervised_dataset(
        features,
        y_val_stage2,
        start=train_samples,
        sample_count=val_samples,
        shuffle=False,
    )

    compile_tuned_model(
        model,
        scaler_y,
        capacity,
        DISTILL_LEARNING_RATE,
        steps_per_epoch,
        DISTILL_EPOCHS,
        distill_alpha=DISTILL_ALPHA,
    )
    distill_callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=distill_log_dir,
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
        keras.callbacks.ModelCheckpoint(
            final_weights_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    distill_history = model.fit(
        train_ds_stage2,
        validation_data=val_ds_stage2,
        epochs=DISTILL_EPOCHS,
        callbacks=distill_callbacks,
        verbose=1,
    )

    if os.path.exists(final_weights_path):
        model.load_weights(final_weights_path)
    model.save(final_model_path)

    histories = [
        ('cold_start', teacher_history),
        ('distill', distill_history),
    ]
    history_path, history_plot_path = save_history_artifacts(histories, farm_id)
    eval_metrics = evaluate_model(model, val_feature_ds, y_val, scaler_y, capacity)
    print(
        f"验证集反归一化 MAE: {eval_metrics['val_inverse_mae']:.4f}, "
        f"RMSE: {eval_metrics['val_inverse_rmse']:.4f}"
    )

    artifact = {
        'model_name': TUNED_MODEL_NAME,
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
        'teacher_model_path': teacher_model_path,
        'teacher_weights_path': teacher_weights_path,
        'model_path': final_model_path,
        'best_weights_path': final_weights_path,
        'teacher_tensorboard_log_dir': teacher_log_dir,
        'distill_tensorboard_log_dir': distill_log_dir,
        'history_path': history_path,
        'history_plot_path': history_plot_path,
        'distillation_stats_path': distill_stats_path,
        'distill_alpha': DISTILL_ALPHA,
        'teacher_keep_ratio': TEACHER_KEEP_RATIO,
        'horizon_decay': HORIZON_DECAY,
        **eval_metrics,
    }
    artifact_path = os.path.join(
        PREPROCESS_DIR,
        f'tuned_patchtst_farm_{farm_id}_preprocess.pkl',
    )
    joblib.dump(artifact, artifact_path)

    tail_path = os.path.join(TAIL_DIR, f'tuned_patchtst_tail_farm_{farm_id}.csv')
    train_df.iloc[-HISTORY_LEN:].to_csv(tail_path, index=True)

    return {
        'farm_id': farm_id,
        'model_path': final_model_path,
        'best_weights_path': final_weights_path,
        'teacher_model_path': teacher_model_path,
        'teacher_weights_path': teacher_weights_path,
        'artifact_path': artifact_path,
        'tail_path': tail_path,
        'history_path': history_path,
        'history_plot_path': history_plot_path,
        'distillation_stats_path': distill_stats_path,
        'teacher_tensorboard_log_dir': teacher_log_dir,
        'distill_tensorboard_log_dir': distill_log_dir,
        'train_samples': train_samples,
        'val_samples': total_samples - train_samples,
        **eval_metrics,
    }


def main():
    configure_runtime()
    set_global_seed(seed)
    ensure_dirs()
    config_path = save_config()

    train_files = discover_train_files(DATA_DIR)
    if not train_files:
        raise FileNotFoundError(f'未在 {DATA_DIR} 找到 {TRAIN_FILE_PATTERN}')

    print(f'发现 {len(train_files)} 个风电训练文件')
    print(f'tuned PatchTST配置已保存: {config_path}')
    print(f'输出目录: {MODEL_DIR}')

    rows = []
    for train_file in train_files:
        rows.append(train_one_farm(train_file))

    metrics = pd.DataFrame(rows)
    metrics_path = os.path.join(MODEL_DIR, 'tuned_patchtst_training_metrics.csv')
    metrics.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f'\n训练完成，指标已保存至 {metrics_path}')
    print(f'TensorBoard: tensorboard --logdir {TENSORBOARD_LOG_DIR}')


if __name__ == '__main__':
    main()
