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
    WIND_SPEED_COLS,
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
ABLATION_DIR = os.path.join(MODEL_DIR, 'ablation')
ABLATION_ALL_METRICS_PATH = os.path.join(
    ABLATION_DIR,
    'tuned_patchtst_ablation_metrics_all_farms.csv',
)
ABLATION_MODULE_SUMMARY_PATH = os.path.join(
    ABLATION_DIR,
    'tuned_patchtst_ablation_module_summary.csv',
)
ROUND2_METRICS_PATH = os.path.join(
    ABLATION_DIR,
    'tuned_patchtst_ablation_round2_metrics_all_farms.csv',
)
ROUND2_MODULE_SUMMARY_PATH = os.path.join(
    ABLATION_DIR,
    'tuned_patchtst_ablation_round2_module_summary.csv',
)

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
RUN_ABLATION = os.getenv('WIND_TUNED_RUN_ABLATION', '1') == '1'
# 第二轮默认复用第一轮结果，仅训练新增分支；每个variant仍可用独立环境变量覆盖。
RUN_PREVIOUS_ABLATIONS = os.getenv(
    'WIND_TUNED_RUN_PREVIOUS_ABLATIONS',
    '0',
) == '1'
RUN_ROUND2_ABLATIONS = os.getenv(
    'WIND_TUNED_RUN_ROUND2_ABLATIONS',
    '1',
) == '1'
REUSE_PREVIOUS_ABLATION_RESULTS = os.getenv(
    'WIND_TUNED_REUSE_PREVIOUS_ABLATION_RESULTS',
    '1',
) == '1'
EXP_WEIGHT_HALFLIFE_STEPS = float(
    os.getenv('WIND_TUNED_EXP_WEIGHT_HALFLIFE_STEPS', '4.0')
)

REVIN_EPSILON = float(os.getenv('WIND_TUNED_REVIN_EPSILON', '1e-5'))
CNN_ADAPTER_FILTERS = int(os.getenv('WIND_TUNED_ADAPTER_FILTERS', '32'))
NEW_RAMP_LOSS_WEIGHT = float(os.getenv('WIND_TUNED_NEW_RAMP_WEIGHT', '0.03'))
NEW_RELATIVE_LOSS_WEIGHT = float(os.getenv('WIND_TUNED_NEW_RELATIVE_WEIGHT', '0.03'))
NEW_PHYSICAL_PENALTY_WEIGHT = float(os.getenv('WIND_TUNED_NEW_PHYSICAL_WEIGHT', '0.01'))
RELATIVE_POWER_FLOOR = float(os.getenv('WIND_TUNED_RELATIVE_FLOOR', '0.05'))
SWA_START_FRACTION = float(os.getenv('WIND_TUNED_SWA_START_FRACTION', '0.50'))
MAX_ALLOWED_NRMSE_DEGRADATION = float(
    os.getenv('WIND_TUNED_MAX_NRMSE_DEGRADATION', '0.02')
)

ABLATION_VARIANTS = [
    {
        'name': 'baseline',
        'round': 1,
        'parent_variant': None,
        'execution_env': 'WIND_TUNED_RUN_BASELINE',
        'added_module': 'original_tuned_patchtst',
        'use_revin': False,
        'use_cnn_adapter': False,
        'use_balanced_loss': False,
        'use_swa': False,
    },
    {
        'name': 'revin',
        'round': 1,
        'parent_variant': 'baseline',
        'execution_env': 'WIND_TUNED_RUN_REVIN',
        'added_module': 'power_revin',
        'use_revin': True,
        'use_cnn_adapter': False,
        'use_balanced_loss': False,
        'use_swa': False,
    },
    {
        'name': 'revin_cnn_adapter',
        'round': 1,
        'parent_variant': 'revin',
        'execution_env': 'WIND_TUNED_RUN_REVIN_CNN_ADAPTER',
        'added_module': 'zero_init_cnn_adapter',
        'use_revin': True,
        'use_cnn_adapter': True,
        'use_balanced_loss': False,
        'use_swa': False,
    },
    {
        'name': 'revin_cnn_adapter_balanced_loss',
        'round': 1,
        'parent_variant': 'revin_cnn_adapter',
        'execution_env': 'WIND_TUNED_RUN_REVIN_CNN_ADAPTER_BALANCED_LOSS',
        'added_module': 'balanced_loss',
        'use_revin': True,
        'use_cnn_adapter': True,
        'use_balanced_loss': True,
        'use_swa': False,
    },
    {
        'name': 'revin_cnn_adapter_balanced_loss_swa',
        'round': 1,
        'parent_variant': 'revin_cnn_adapter_balanced_loss',
        'execution_env': 'WIND_TUNED_RUN_REVIN_CNN_ADAPTER_BALANCED_LOSS_SWA',
        'added_module': 'swa',
        'use_revin': True,
        'use_cnn_adapter': True,
        'use_balanced_loss': True,
        'use_swa': True,
    },
    {
        'name': 'revin_balanced_loss',
        'round': 2,
        'parent_variant': 'revin',
        'execution_env': 'WIND_TUNED_RUN_REVIN_BALANCED_LOSS',
        'added_module': 'balanced_loss_without_cnn_adapter',
        'use_revin': True,
        'use_cnn_adapter': False,
        'use_balanced_loss': True,
        'use_swa': False,
    },
]


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
        ABLATION_DIR,
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


def parse_env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError(f'环境变量 {name} 应为0/1或true/false，当前为: {value}')


def get_ablation_execution_plan():
    requested = os.getenv('WIND_TUNED_ABLATION_VARIANTS', '').strip()
    requested_names = {
        name.strip()
        for name in requested.split(',')
        if name.strip()
    }
    valid_names = {variant['name'] for variant in ABLATION_VARIANTS}
    missing = sorted(requested_names - valid_names)
    if missing:
        raise ValueError(
            f'未知消融variant: {missing}; 可选: {sorted(valid_names)}'
        )

    plan = []
    for variant in ABLATION_VARIANTS:
        if requested:
            execute = variant['name'] in requested_names
        else:
            round_default = (
                RUN_PREVIOUS_ABLATIONS
                if variant['round'] == 1
                else RUN_ROUND2_ABLATIONS
            )
            execute = parse_env_bool(variant['execution_env'], round_default)
        plan.append({**variant, 'execute': execute})
    return plan


def get_ablation_variants():
    return [
        variant
        for variant in get_ablation_execution_plan()
        if variant['execute']
    ]


def get_adapter_channel_indices(input_cols):
    adapter_cols = [TARGET_COL] + [
        col for col in WIND_SPEED_COLS
        if col in input_cols
    ]
    return [input_cols.index(col) for col in adapter_cols if col in input_cols]


def make_variant_dirs(variant_name):
    root = os.path.join(ABLATION_DIR, variant_name)
    dirs = {
        'root': root,
        'models': os.path.join(root, 'models'),
        'weights': os.path.join(root, 'weights'),
        'teachers': os.path.join(root, 'teacher'),
        'preprocess': os.path.join(root, 'preprocess'),
        'history': os.path.join(root, 'history'),
        'tensorboard': os.path.join(root, 'tensorboard'),
        'distillation': os.path.join(root, 'distillation'),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def load_previous_ablation_results():
    if not REUSE_PREVIOUS_ABLATION_RESULTS:
        return []

    metric_paths = []
    if os.path.exists(ABLATION_ALL_METRICS_PATH):
        metric_paths.append(ABLATION_ALL_METRICS_PATH)
    metric_paths.extend(sorted(glob.glob(os.path.join(
        ABLATION_DIR,
        'tuned_patchtst_ablation_metrics_farm_*.csv',
    ))))
    if not metric_paths:
        print(f'未找到历史消融结果，将只使用本轮训练结果: {ABLATION_DIR}')
        return []

    variant_map = {
        variant['name']: variant
        for variant in ABLATION_VARIANTS
    }
    records_by_key = {}
    for metric_path in metric_paths:
        previous = pd.read_csv(metric_path, dtype={'farm_id': str})
        for record in previous.to_dict('records'):
            variant = variant_map.get(record.get('variant'))
            if variant is None:
                continue
            record.update({
                'farm_id': str(record['farm_id']),
                'round': variant['round'],
                'parent_variant': variant['parent_variant'],
                'result_source': 'reused_previous_result',
            })
            key = (record['farm_id'], record['variant'])
            records_by_key[key] = record
    records = list(records_by_key.values())
    print(
        f'已从 {len(metric_paths)} 个CSV加载历史消融结果 '
        f'{len(records)} 条'
    )
    return records


def result_artifacts_exist(result):
    required_paths = [
        result.get('model_path'),
        result.get('artifact_path'),
    ]
    return all(
        isinstance(path, str) and path and os.path.exists(path)
        for path in required_paths
    )


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
class PowerRevIN(layers.Layer):
    def __init__(self, target_channel_index, epsilon=REVIN_EPSILON, **kwargs):
        super().__init__(**kwargs)
        self.target_channel_index = target_channel_index
        self.epsilon = epsilon

    def call(self, inputs):
        target = inputs[
            :,
            :,
            self.target_channel_index:self.target_channel_index + 1,
        ]
        center = tf.reduce_mean(target, axis=1, keepdims=True)
        variance = tf.reduce_mean(tf.square(target - center), axis=1, keepdims=True)
        scale = tf.sqrt(variance + self.epsilon)
        normalized_target = (target - center) / scale
        normalized_inputs = tf.concat(
            [
                inputs[:, :, :self.target_channel_index],
                normalized_target,
                inputs[:, :, self.target_channel_index + 1:],
            ],
            axis=-1,
        )
        return normalized_inputs, center, scale

    def compute_output_shape(self, input_shape):
        stats_shape = (input_shape[0], 1, 1)
        return input_shape, stats_shape, stats_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'target_channel_index': self.target_channel_index,
            'epsilon': self.epsilon,
        })
        return config


@keras.utils.register_keras_serializable(package='WindTunedPatchTST')
class PowerRevINDenormalize(layers.Layer):
    def call(self, inputs):
        forecast, center, scale = inputs
        center = tf.squeeze(center, axis=1)
        scale = tf.squeeze(scale, axis=1)
        return forecast * scale + center

    def compute_output_shape(self, input_shape):
        return input_shape[0]


@keras.utils.register_keras_serializable(package='WindTunedPatchTST')
class SelectInputChannels(layers.Layer):
    def __init__(self, channel_indices, **kwargs):
        super().__init__(**kwargs)
        self.channel_indices = [int(index) for index in channel_indices]

    def call(self, inputs):
        return tf.gather(inputs, self.channel_indices, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], len(self.channel_indices)

    def get_config(self):
        config = super().get_config()
        config.update({'channel_indices': self.channel_indices})
        return config


@keras.utils.register_keras_serializable(package='WindTunedPatchTST')
class ZeroInitResidualAdapter(layers.Layer):
    def build(self, input_shape):
        self.gate = self.add_weight(
            name='adapter_gate',
            shape=(),
            initializer='zeros',
            regularizer=regularizers.l2(1e-4),
            trainable=True,
        )

    def call(self, inputs):
        base, adapter = inputs
        gate = tf.tanh(tf.cast(self.gate, base.dtype))
        return base + gate * adapter

    def compute_output_shape(self, input_shape):
        return input_shape[0]


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
class BalancedTunedPatchTSTLoss(keras.losses.Loss):
    def __init__(self, forecast_len=FORECAST_LEN, distill_alpha=DISTILL_ALPHA,
                 zero_scaled=0.0, capacity_scaled=None,
                 ramp_loss_weight=NEW_RAMP_LOSS_WEIGHT,
                 relative_loss_weight=NEW_RELATIVE_LOSS_WEIGHT,
                 physical_penalty_weight=NEW_PHYSICAL_PENALTY_WEIGHT,
                 relative_power_floor=RELATIVE_POWER_FLOOR,
                 delta=HUBER_DELTA, name='balanced_tuned_patchtst_loss'):
        super().__init__(name=name)
        self.forecast_len = forecast_len
        self.distill_alpha = float(distill_alpha)
        self.zero_scaled = float(zero_scaled)
        self.capacity_scaled = None if capacity_scaled is None else float(capacity_scaled)
        self.ramp_loss_weight = float(ramp_loss_weight)
        self.relative_loss_weight = float(relative_loss_weight)
        self.physical_penalty_weight = float(physical_penalty_weight)
        self.relative_power_floor = float(relative_power_floor)
        self.delta = float(delta)

    def _huber(self, error):
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        return 0.5 * tf.square(quadratic) + self.delta * linear

    def call(self, y_true, y_pred):
        actual = y_true[:, :self.forecast_len]
        teacher = y_true[:, self.forecast_len:2 * self.forecast_len]
        confidence = y_true[:, 2 * self.forecast_len:2 * self.forecast_len + 1]

        supervised = tf.reduce_mean(self._huber(actual - y_pred), axis=1)
        distill = tf.reduce_mean(self._huber(teacher - y_pred), axis=1)
        loss = supervised + self.distill_alpha * tf.squeeze(confidence, axis=-1) * distill
        loss = tf.reduce_mean(loss)

        if self.forecast_len > 1:
            true_ramp = actual[:, 1:] - actual[:, :-1]
            pred_ramp = y_pred[:, 1:] - y_pred[:, :-1]
            ramp_loss = tf.reduce_mean(self._huber(true_ramp - pred_ramp))
        else:
            ramp_loss = 0.0

        if self.capacity_scaled is not None:
            capacity_span = max(self.capacity_scaled - self.zero_scaled, 1e-6)
            actual_pu = (actual - self.zero_scaled) / capacity_span
            pred_pu = (y_pred - self.zero_scaled) / capacity_span
            denominator = tf.maximum(
                tf.abs(actual_pu) + tf.abs(pred_pu),
                self.relative_power_floor,
            )
            relative_loss = tf.reduce_mean(
                2.0 * tf.abs(actual_pu - pred_pu) / denominator
            )
        else:
            relative_loss = 0.0

        lower_penalty = tf.reduce_mean(tf.square(tf.nn.relu(self.zero_scaled - y_pred)))
        physical_penalty = lower_penalty
        if self.capacity_scaled is not None:
            upper_penalty = tf.reduce_mean(tf.square(tf.nn.relu(y_pred - self.capacity_scaled)))
            physical_penalty = physical_penalty + upper_penalty

        return (
            loss
            + self.ramp_loss_weight * ramp_loss
            + self.relative_loss_weight * relative_loss
            + self.physical_penalty_weight * physical_penalty
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'forecast_len': self.forecast_len,
            'distill_alpha': self.distill_alpha,
            'zero_scaled': self.zero_scaled,
            'capacity_scaled': self.capacity_scaled,
            'ramp_loss_weight': self.ramp_loss_weight,
            'relative_loss_weight': self.relative_loss_weight,
            'physical_penalty_weight': self.physical_penalty_weight,
            'relative_power_floor': self.relative_power_floor,
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


class StochasticWeightAveraging(keras.callbacks.Callback):
    def __init__(self, start_epoch):
        super().__init__()
        self.start_epoch = max(1, int(start_epoch))
        self.average_weights = None
        self.snapshot_count = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 < self.start_epoch:
            return

        current_weights = self.model.get_weights()
        self.snapshot_count += 1
        if self.average_weights is None:
            self.average_weights = [
                np.array(weight, copy=True)
                for weight in current_weights
            ]
            return

        factor = 1.0 / self.snapshot_count
        for index, current in enumerate(current_weights):
            self.average_weights[index] += (
                current - self.average_weights[index]
            ) * factor

    def on_train_end(self, logs=None):
        if self.average_weights is not None:
            self.model.set_weights(self.average_weights)


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
                        epochs, distill_alpha, use_balanced_loss=False):
    zero_scaled, capacity_scaled = scaled_bounds(scaler_y, capacity)
    if use_balanced_loss:
        loss = BalancedTunedPatchTSTLoss(
            forecast_len=FORECAST_LEN,
            distill_alpha=distill_alpha,
            zero_scaled=zero_scaled,
            capacity_scaled=capacity_scaled,
        )
    else:
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


def build_tuned_patchtst_model(input_dim, target_channel_index, use_revin=False,
                               use_cnn_adapter=False, adapter_channel_indices=None):
    if target_channel_index is None:
        raise ValueError('Tuned PatchTST 需要将历史功率作为输入通道')
    if use_cnn_adapter and not adapter_channel_indices:
        raise ValueError('启用CNN Adapter时必须提供功率/风速通道索引')

    patch_num = compute_patch_num(HISTORY_LEN, PATCH_LEN, PATCH_STRIDE)
    inputs = keras.Input(shape=(HISTORY_LEN, input_dim), name='history_features')

    model_input = inputs
    revin_center = None
    revin_scale = None
    if use_revin:
        model_input, revin_center, revin_scale = PowerRevIN(
            target_channel_index,
            REVIN_EPSILON,
            dtype='float32',
            name='power_revin',
        )(inputs)

    x_input = layers.GaussianNoise(INPUT_NOISE_STD, name='input_noise')(model_input)
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

    if use_cnn_adapter:
        adapter = SelectInputChannels(
            adapter_channel_indices,
            name='adapter_input_channels',
        )(x_input)
        adapter = layers.SeparableConv1D(
            CNN_ADAPTER_FILTERS,
            kernel_size=3,
            padding='same',
            activation='gelu',
            name='cnn_adapter_conv3',
        )(adapter)
        adapter = layers.SeparableConv1D(
            CNN_ADAPTER_FILTERS,
            kernel_size=5,
            padding='same',
            activation='gelu',
            name='cnn_adapter_conv5',
        )(adapter)
        adapter = layers.GlobalAveragePooling1D(name='cnn_adapter_pool')(adapter)
        adapter = layers.Dense(
            D_FF,
            activation='gelu',
            name='cnn_adapter_projection',
        )(adapter)
        adapter = layers.Dropout(HEAD_DROPOUT, name='cnn_adapter_dropout')(adapter)
        head = ZeroInitResidualAdapter(name='zero_init_cnn_adapter')([head, adapter])

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
    )(model_input)
    forecast_name = 'forecast_power_normalized' if use_revin else 'forecast_power'
    outputs = layers.Add(name=forecast_name)([baseline, residual])
    if use_revin:
        outputs = PowerRevINDenormalize(
            dtype='float32',
            name='forecast_power',
        )([outputs, revin_center, revin_scale])
    outputs = layers.Activation('linear', dtype='float32', name='forecast_power_float32')(outputs)
    return keras.Model(inputs=inputs, outputs=outputs, name='WindTunedPatchTST')


def inverse_power(scaler_y, values):
    values = np.asarray(values).reshape(-1, 1)
    return scaler_y.inverse_transform(values).reshape(-1)


def calculate_composite_metrics(y_true, y_pred, capacity=None):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) == 0:
        raise ValueError('验证集中没有可用于评价的功率点')

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = np.nan
    if len(y_true) > 1 and np.nanstd(y_true) > 1e-6:
        r2 = r2_score(y_true, y_pred)

    normalizer = capacity
    if normalizer is None or normalizer <= 0:
        normalizer = max(float(np.nanmax(y_true) - np.nanmin(y_true)), 1.0)
    normalized_mae = float(mae / normalizer)
    normalized_rmse = float(rmse / normalizer)
    denominator = np.maximum(
        np.abs(y_true) + np.abs(y_pred),
        RELATIVE_POWER_FLOOR * normalizer,
    )
    smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / denominator) * 100.0)
    r2_penalty = 1.0 if not np.isfinite(r2) else max(0.0, 1.0 - float(r2))
    score = (
        0.45 * normalized_rmse
        + 0.30 * normalized_mae
        + 0.15 * (smape / 100.0)
        + 0.10 * r2_penalty
    )
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': rmse,
        'r2': r2,
        'capacity_normalized_mae': normalized_mae,
        'capacity_normalized_rmse': normalized_rmse,
        'stable_smape': smape,
        'composite_score': float(score),
    }


def aggregate_exponential_weighted_predictions(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError('指数加权聚合要求y_true/y_pred均为相同的二维窗口数组')

    n_samples, forecast_len = y_true.shape
    timeline_len = n_samples + forecast_len - 1
    pred_sum = np.zeros(timeline_len, dtype=np.float64)
    weight_sum = np.zeros(timeline_len, dtype=np.float64)
    actual_sum = np.zeros(timeline_len, dtype=np.float64)
    actual_count = np.zeros(timeline_len, dtype=np.float64)
    horizon_weights_values = 0.5 ** (
        np.arange(forecast_len, dtype=np.float64) / EXP_WEIGHT_HALFLIFE_STEPS
    )

    for horizon_index, weight in enumerate(horizon_weights_values):
        target_slice = slice(horizon_index, horizon_index + n_samples)
        valid_pred = np.isfinite(y_pred[:, horizon_index])
        valid_actual = np.isfinite(y_true[:, horizon_index])
        pred_sum[target_slice] += np.where(
            valid_pred,
            y_pred[:, horizon_index] * weight,
            0.0,
        )
        weight_sum[target_slice] += valid_pred.astype(np.float64) * weight
        actual_sum[target_slice] += np.where(
            valid_actual,
            y_true[:, horizon_index],
            0.0,
        )
        actual_count[target_slice] += valid_actual.astype(np.float64)

    valid = (weight_sum > 0) & (actual_count > 0)
    actual = actual_sum[valid] / actual_count[valid]
    prediction = pred_sum[valid] / weight_sum[valid]
    return actual, prediction


def evaluate_model(model, val_feature_ds, y_val, scaler_y, capacity=None):
    y_pred_scaled = model.predict(val_feature_ds, verbose=0)
    y_true = inverse_power(scaler_y, y_val).reshape(y_val.shape)
    y_pred = inverse_power(scaler_y, y_pred_scaled).reshape(y_pred_scaled.shape)
    if capacity is not None:
        y_pred = np.clip(y_pred, 0, capacity)
    else:
        y_pred = np.clip(y_pred, 0, None)

    window_metrics = calculate_composite_metrics(y_true, y_pred, capacity)
    weighted_true, weighted_pred = aggregate_exponential_weighted_predictions(
        y_true,
        y_pred,
    )
    weighted_metrics = calculate_composite_metrics(
        weighted_true,
        weighted_pred,
        capacity,
    )
    composite_score = (
        0.70 * window_metrics['composite_score']
        + 0.30 * weighted_metrics['composite_score']
    )
    return {
        'val_inverse_mae': window_metrics['mae'],
        'val_inverse_mse': window_metrics['mse'],
        'val_inverse_rmse': window_metrics['rmse'],
        'val_inverse_r2': window_metrics['r2'],
        'val_capacity_normalized_mae': window_metrics['capacity_normalized_mae'],
        'val_capacity_normalized_rmse': window_metrics['capacity_normalized_rmse'],
        'val_stable_smape': window_metrics['stable_smape'],
        'val_window_composite_score': window_metrics['composite_score'],
        'val_weighted_curve_mae': weighted_metrics['mae'],
        'val_weighted_curve_rmse': weighted_metrics['rmse'],
        'val_weighted_curve_r2': weighted_metrics['r2'],
        'val_weighted_curve_capacity_normalized_mae': (
            weighted_metrics['capacity_normalized_mae']
        ),
        'val_weighted_curve_capacity_normalized_rmse': (
            weighted_metrics['capacity_normalized_rmse']
        ),
        'val_weighted_curve_stable_smape': weighted_metrics['stable_smape'],
        'val_weighted_curve_composite_score': weighted_metrics['composite_score'],
        'val_composite_score': float(composite_score),
    }


def teacher_confidence(y_true, teacher_pred, keep_ratio=TEACHER_KEEP_RATIO):
    sample_mae = np.mean(np.abs(y_true - teacher_pred), axis=1)
    threshold = float(np.quantile(sample_mae, keep_ratio))
    confidence = (sample_mae <= threshold).astype(np.float32)
    return confidence, sample_mae, threshold


def save_distillation_stats(farm_id, train_mae, train_conf, train_threshold,
                            val_mae, val_conf, val_threshold,
                            output_dir=DISTILL_DIR, filename_prefix='tuned_patchtst'):
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
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(
        output_dir,
        f'{filename_prefix}_distillation_stats_farm_{farm_id}.csv',
    )
    stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
    return stats_path


def save_history_artifacts(histories, farm_id, output_dir=HISTORY_DIR,
                           filename_prefix='tuned_patchtst'):
    frames = []
    for stage, history in histories:
        history_df = pd.DataFrame(history.history)
        history_df.insert(0, 'stage', stage)
        history_df.insert(1, 'epoch', np.arange(1, len(history_df) + 1))
        frames.append(history_df)

    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(
        output_dir,
        f'{filename_prefix}_history_farm_{farm_id}.csv',
    )
    combined.to_csv(history_path, index=False, encoding='utf-8-sig')

    plot_path = os.path.join(
        output_dir,
        f'{filename_prefix}_history_farm_{farm_id}.png',
    )
    try:
        cache_dir = os.path.join(output_dir, 'matplotlib_cache')
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
        'run_ablation': RUN_ABLATION,
        'run_previous_ablations': RUN_PREVIOUS_ABLATIONS,
        'run_round2_ablations': RUN_ROUND2_ABLATIONS,
        'reuse_previous_ablation_results': REUSE_PREVIOUS_ABLATION_RESULTS,
        'ablation_variants': ABLATION_VARIANTS,
        'ablation_execution_plan': get_ablation_execution_plan(),
        'revin_epsilon': REVIN_EPSILON,
        'cnn_adapter_filters': CNN_ADAPTER_FILTERS,
        'new_ramp_loss_weight': NEW_RAMP_LOSS_WEIGHT,
        'new_relative_loss_weight': NEW_RELATIVE_LOSS_WEIGHT,
        'new_physical_penalty_weight': NEW_PHYSICAL_PENALTY_WEIGHT,
        'relative_power_floor': RELATIVE_POWER_FLOOR,
        'exp_weight_halflife_steps': EXP_WEIGHT_HALFLIFE_STEPS,
        'swa_start_fraction': SWA_START_FRACTION,
        'max_allowed_nrmse_degradation': MAX_ALLOWED_NRMSE_DEGRADATION,
    }
    config_path = os.path.join(MODEL_DIR, 'tuned_patchtst_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return config_path


def train_ablation_variant(farm_id, variant, features, y_train, y_val,
                           train_samples, total_samples, input_cols,
                           target_index, adapter_channel_indices,
                           scaler_x, scaler_y, feature_cols, capacity):
    variant_name = variant['name']
    dirs = make_variant_dirs(variant_name)
    val_samples = total_samples - train_samples
    steps_per_epoch = int(np.ceil(train_samples / BATCH_SIZE))

    keras.backend.clear_session()
    set_global_seed(seed)
    print(
        f'\n--- 消融variant={variant_name}: '
        f"RevIN={variant['use_revin']}, "
        f"CNN Adapter={variant['use_cnn_adapter']}, "
        f"Balanced loss={variant['use_balanced_loss']}, "
        f"SWA={variant['use_swa']} ---"
    )

    y_train_stage1 = combine_targets(y_train)
    y_val_stage1 = combine_targets(y_val)
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
    train_feature_ds = make_feature_dataset(features, 0, train_samples)
    val_feature_ds = make_feature_dataset(features, train_samples, val_samples)

    model = build_tuned_patchtst_model(
        len(input_cols),
        target_index,
        use_revin=variant['use_revin'],
        use_cnn_adapter=variant['use_cnn_adapter'],
        adapter_channel_indices=adapter_channel_indices,
    )
    compile_tuned_model(
        model,
        scaler_y,
        capacity,
        BASE_LEARNING_RATE,
        steps_per_epoch,
        COLD_START_EPOCHS,
        distill_alpha=0.0,
        use_balanced_loss=variant['use_balanced_loss'],
    )
    print(f'variant={variant_name}, parameters={model.count_params():,}')

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    teacher_weights_path = os.path.join(
        dirs['weights'],
        f'tuned_patchtst_{variant_name}_teacher_farm_{farm_id}_best.weights.h5',
    )
    teacher_model_path = os.path.join(
        dirs['teachers'],
        f'tuned_patchtst_{variant_name}_teacher_farm_{farm_id}.keras',
    )
    raw_weights_path = os.path.join(
        dirs['weights'],
        f'tuned_patchtst_{variant_name}_farm_{farm_id}_raw_best.weights.h5',
    )
    averaged_weights_path = os.path.join(
        dirs['weights'],
        f'tuned_patchtst_{variant_name}_farm_{farm_id}_swa.weights.h5',
    )
    final_weights_path = os.path.join(
        dirs['weights'],
        f'tuned_patchtst_{variant_name}_farm_{farm_id}_best.weights.h5',
    )
    final_model_path = os.path.join(
        dirs['models'],
        f'tuned_patchtst_{variant_name}_farm_{farm_id}.keras',
    )
    teacher_log_dir = os.path.join(
        dirs['tensorboard'],
        f'farm_{farm_id}',
        'cold_start',
        timestamp,
    )
    distill_log_dir = os.path.join(
        dirs['tensorboard'],
        f'farm_{farm_id}',
        'distill',
        timestamp,
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
        output_dir=dirs['distillation'],
        filename_prefix=f'tuned_patchtst_{variant_name}',
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
        use_balanced_loss=variant['use_balanced_loss'],
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
            raw_weights_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]
    swa_callback = None
    if variant['use_swa']:
        swa_start_epoch = max(
            1,
            min(5, int(np.ceil(DISTILL_EPOCHS * SWA_START_FRACTION))),
        )
        swa_callback = StochasticWeightAveraging(swa_start_epoch)
        distill_callbacks.append(swa_callback)

    distill_history = model.fit(
        train_ds_stage2,
        validation_data=val_ds_stage2,
        epochs=DISTILL_EPOCHS,
        callbacks=distill_callbacks,
        verbose=1,
    )

    selected_weight_source = 'raw_best'
    raw_metrics = None
    averaged_metrics = None
    if variant['use_swa'] and swa_callback is not None and swa_callback.snapshot_count > 0:
        model.save_weights(averaged_weights_path)
        averaged_metrics = evaluate_model(
            model,
            val_feature_ds,
            y_val,
            scaler_y,
            capacity,
        )
        if os.path.exists(raw_weights_path):
            model.load_weights(raw_weights_path)
        raw_metrics = evaluate_model(
            model,
            val_feature_ds,
            y_val,
            scaler_y,
            capacity,
        )
        if averaged_metrics['val_composite_score'] < raw_metrics['val_composite_score']:
            model.load_weights(averaged_weights_path)
            selected_weight_source = 'swa'
    elif os.path.exists(raw_weights_path):
        model.load_weights(raw_weights_path)

    model.save_weights(final_weights_path)
    model.save(final_model_path)
    eval_metrics = evaluate_model(
        model,
        val_feature_ds,
        y_val,
        scaler_y,
        capacity,
    )
    histories = [
        ('cold_start', teacher_history),
        ('distill', distill_history),
    ]
    history_path, history_plot_path = save_history_artifacts(
        histories,
        farm_id,
        output_dir=dirs['history'],
        filename_prefix=f'tuned_patchtst_{variant_name}',
    )

    artifact = {
        'model_name': TUNED_MODEL_NAME,
        'farm_id': farm_id,
        'ablation_variant': variant_name,
        'ablation_round': variant['round'],
        'parent_variant': variant['parent_variant'],
        'added_module': variant['added_module'],
        'use_revin': variant['use_revin'],
        'use_cnn_adapter': variant['use_cnn_adapter'],
        'use_balanced_loss': variant['use_balanced_loss'],
        'use_swa': variant['use_swa'],
        'selected_weight_source': selected_weight_source,
        'adapter_channel_indices': adapter_channel_indices,
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
        'raw_best_weights_path': raw_weights_path,
        'swa_weights_path': (
            averaged_weights_path
            if os.path.exists(averaged_weights_path)
            else None
        ),
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
        dirs['preprocess'],
        f'tuned_patchtst_{variant_name}_farm_{farm_id}_preprocess.pkl',
    )
    joblib.dump(artifact, artifact_path)

    result = {
        'farm_id': farm_id,
        'variant': variant_name,
        'round': variant['round'],
        'parent_variant': variant['parent_variant'],
        'result_source': 'trained_current_run',
        'added_module': variant['added_module'],
        'use_revin': variant['use_revin'],
        'use_cnn_adapter': variant['use_cnn_adapter'],
        'use_balanced_loss': variant['use_balanced_loss'],
        'use_swa': variant['use_swa'],
        'selected_weight_source': selected_weight_source,
        'model_path': final_model_path,
        'best_weights_path': final_weights_path,
        'artifact_path': artifact_path,
        'history_path': history_path,
        'distillation_stats_path': distill_stats_path,
        'train_samples': train_samples,
        'val_samples': val_samples,
        'raw_val_composite_score': (
            raw_metrics['val_composite_score']
            if raw_metrics is not None
            else np.nan
        ),
        'swa_val_composite_score': (
            averaged_metrics['val_composite_score']
            if averaged_metrics is not None
            else np.nan
        ),
        **eval_metrics,
    }
    print(
        f"variant={variant_name}: score={eval_metrics['val_composite_score']:.6f}, "
        f"NRMSE={eval_metrics['val_capacity_normalized_rmse']:.6f}, "
        f"SMAPE={eval_metrics['val_stable_smape']:.3f}%"
    )
    return result


def promote_selected_variant(train_df, selected_result):
    farm_id = str(selected_result['farm_id'])
    selected_artifact = joblib.load(selected_result['artifact_path'])
    model = keras.models.load_model(
        selected_result['model_path'],
        compile=False,
    )

    canonical_model_path = os.path.join(
        SAVED_MODEL_DIR,
        f'tuned_patchtst_farm_{farm_id}.keras',
    )
    canonical_weights_path = os.path.join(
        WEIGHTS_DIR,
        f'tuned_patchtst_farm_{farm_id}_best.weights.h5',
    )
    canonical_artifact_path = os.path.join(
        PREPROCESS_DIR,
        f'tuned_patchtst_farm_{farm_id}_preprocess.pkl',
    )
    model.save(canonical_model_path)
    model.save_weights(canonical_weights_path)

    selected_artifact.update({
        'selected_ablation_variant': selected_result['variant'],
        'selected_ablation_round': selected_result.get('round'),
        'selected_parent_variant': selected_result.get('parent_variant'),
        'selected_weight_source': selected_result.get('selected_weight_source'),
        'selected_by': 'validation_composite_score',
        'source_variant_model_path': selected_result['model_path'],
        'source_variant_weights_path': selected_result['best_weights_path'],
        'model_path': canonical_model_path,
        'best_weights_path': canonical_weights_path,
    })
    joblib.dump(selected_artifact, canonical_artifact_path)

    tail_path = os.path.join(TAIL_DIR, f'tuned_patchtst_tail_farm_{farm_id}.csv')
    train_df.iloc[-HISTORY_LEN:].to_csv(tail_path, index=True)

    selection_path = os.path.join(
        ABLATION_DIR,
        f'tuned_patchtst_selected_variant_farm_{farm_id}.json',
    )
    selection_record = {
        'farm_id': farm_id,
        'selected_variant': selected_result['variant'],
        'selected_ablation_round': selected_result.get('round'),
        'selected_parent_variant': selected_result.get('parent_variant'),
        'selected_weight_source': selected_result['selected_weight_source'],
        'selection_metric': 'val_composite_score',
        'val_composite_score': float(selected_result['val_composite_score']),
        'val_capacity_normalized_rmse': float(
            selected_result['val_capacity_normalized_rmse']
        ),
        'canonical_model_path': canonical_model_path,
        'canonical_weights_path': canonical_weights_path,
        'canonical_artifact_path': canonical_artifact_path,
    }
    with open(selection_path, 'w', encoding='utf-8') as file:
        json.dump(selection_record, file, ensure_ascii=False, indent=2)

    return {
        **selected_result,
        'model_path': canonical_model_path,
        'best_weights_path': canonical_weights_path,
        'artifact_path': canonical_artifact_path,
        'tail_path': tail_path,
        'selection_path': selection_path,
    }


def select_variant_for_farm(results):
    if not results:
        raise ValueError('没有可供选择的消融实验结果')

    promotable = [
        result for result in results
        if result_artifacts_exist(result)
    ]
    if not promotable:
        raise FileNotFoundError('消融结果存在，但没有可加载的模型与artifact用于晋升')
    skipped = len(results) - len(promotable)
    if skipped:
        print(f'警告: {skipped} 个历史variant缺少模型/artifact，不参与最终晋升')

    baseline = next(
        (result for result in promotable if result['variant'] == 'baseline'),
        None,
    )
    eligible = list(promotable)
    if baseline is not None:
        baseline_nrmse = baseline['val_capacity_normalized_rmse']
        upper_limit = baseline_nrmse * (1.0 + MAX_ALLOWED_NRMSE_DEGRADATION)
        eligible = [
            result for result in promotable
            if result['variant'] == 'baseline'
            or result['val_capacity_normalized_rmse'] <= upper_limit
        ]
    return min(eligible, key=lambda result: result['val_composite_score'])


def train_one_farm_ablation(train_file, previous_results=None):
    farm_id = get_farm_id(train_file)
    print(f'\n===== tuned PatchTST 单变量消融 / 风电场 {farm_id} =====')
    previous_results = previous_results or []

    train_df, feature_cols, capacity = load_and_preprocess(train_file, is_train=True)
    features, target, input_cols, target_index, scaler_x, scaler_y = build_scaled_arrays(
        train_df,
        feature_cols,
    )
    y_train, y_val, train_samples, total_samples = make_window_targets(
        features,
        target,
        HISTORY_LEN,
        FORECAST_LEN,
        VALIDATION_SPLIT,
    )
    adapter_channel_indices = get_adapter_channel_indices(input_cols)
    print(
        f'数据形状: {train_df.shape}，输入通道: {len(input_cols)}，'
        f'训练/验证样本: {train_samples}/{total_samples - train_samples}'
    )
    print(
        'CNN Adapter通道: '
        f'{[input_cols[index] for index in adapter_channel_indices]}'
    )

    result_by_variant = {
        result['variant']: dict(result)
        for result in previous_results
        if str(result.get('farm_id')) == str(farm_id)
    }
    trained_results = []
    for variant in get_ablation_execution_plan():
        variant_name = variant['name']
        if variant['execute']:
            print(f'执行variant训练: {variant_name}')
            result = train_ablation_variant(
                farm_id,
                variant,
                features,
                y_train,
                y_val,
                train_samples,
                total_samples,
                input_cols,
                target_index,
                adapter_channel_indices,
                scaler_x,
                scaler_y,
                feature_cols,
                capacity,
            )
            result_by_variant[variant_name] = result
            trained_results.append(result)
        elif variant_name in result_by_variant:
            print(f'复用历史variant结果: {variant_name}')
        else:
            print(f'跳过variant且无历史结果: {variant_name}')

    variant_order = [variant['name'] for variant in ABLATION_VARIANTS]
    results = [
        result_by_variant[name]
        for name in variant_order
        if name in result_by_variant
    ]
    if not results:
        raise ValueError(
            f'场站 {farm_id} 没有任何可用消融结果；'
            '请至少启用一个variant或开启历史结果复用'
        )

    farm_results = pd.DataFrame(results)
    farm_metrics_path = os.path.join(
        ABLATION_DIR,
        f'tuned_patchtst_ablation_metrics_farm_{farm_id}.csv',
    )
    farm_results.to_csv(farm_metrics_path, index=False, encoding='utf-8-sig')

    selected = select_variant_for_farm(results)
    selected = promote_selected_variant(train_df, selected)
    selected['ablation_metrics_path'] = farm_metrics_path
    print(
        f"场站 {farm_id} 选择variant={selected['variant']}，"
        f"score={selected['val_composite_score']:.6f}"
    )
    return selected, results, trained_results


def summarize_ablation(all_results, selected_results):
    detail = pd.DataFrame(all_results)
    selected_names = pd.Series(
        [result['variant'] for result in selected_results]
    ).value_counts()
    variant_order = [variant['name'] for variant in ABLATION_VARIANTS]
    rows = []

    baseline = detail[detail['variant'] == 'baseline'][
        ['farm_id', 'val_composite_score', 'val_capacity_normalized_rmse']
    ].rename(columns={
        'val_composite_score': 'baseline_score',
        'val_capacity_normalized_rmse': 'baseline_nrmse',
    })
    variant_map = {
        variant['name']: variant
        for variant in ABLATION_VARIANTS
    }
    for variant in variant_order:
        variant_df = detail[detail['variant'] == variant]
        if variant_df.empty:
            continue
        variant_config = variant_map[variant]
        parent_variant = variant_config['parent_variant']
        mean_score = float(variant_df['val_composite_score'].mean())
        compared = variant_df.merge(baseline, on='farm_id', how='left')
        score_delta = compared['val_composite_score'] - compared['baseline_score']
        nrmse_ratio = (
            compared['val_capacity_normalized_rmse']
            / compared['baseline_nrmse']
            - 1.0
        )
        parent_score_delta = pd.Series(dtype=float)
        if parent_variant is not None:
            parent_df = detail[detail['variant'] == parent_variant][
                ['farm_id', 'val_composite_score']
            ].rename(columns={'val_composite_score': 'parent_score'})
            parent_compared = variant_df.merge(parent_df, on='farm_id', how='inner')
            parent_score_delta = (
                parent_compared['val_composite_score']
                - parent_compared['parent_score']
            )
        rows.append({
            'variant': variant,
            'round': variant_config['round'],
            'parent_variant': parent_variant,
            'added_module': variant_df['added_module'].iloc[0],
            'farms': int(len(variant_df)),
            'mean_composite_score': mean_score,
            'incremental_score_delta': (
                np.nan
                if parent_score_delta.empty
                else float(parent_score_delta.mean())
            ),
            'improved_farms_vs_parent': (
                0
                if parent_score_delta.empty
                else int((parent_score_delta < 0).sum())
            ),
            'mean_score_delta_vs_baseline': float(score_delta.mean()),
            'improved_farms_vs_baseline': int((score_delta < 0).sum()),
            'mean_nrmse_change_vs_baseline': float(nrmse_ratio.mean()),
            'max_nrmse_degradation_vs_baseline': float(nrmse_ratio.max()),
            'passes_nrmse_guardrail': bool(
                nrmse_ratio.max() <= MAX_ALLOWED_NRMSE_DEGRADATION
            ),
            'selected_farms': int(selected_names.get(variant, 0)),
        })

    summary = pd.DataFrame(rows)
    eligible = summary[summary['passes_nrmse_guardrail']]
    if eligible.empty:
        global_best_variant = 'baseline'
    else:
        global_best_variant = eligible.loc[
            eligible['mean_composite_score'].idxmin(),
            'variant',
        ]
    summary['global_best_variant'] = global_best_variant
    return detail, summary, global_best_variant


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
    all_ablation_results = []
    current_run_results = []
    if RUN_ABLATION:
        execution_plan = get_ablation_execution_plan()
        print('消融执行计划:')
        for variant in execution_plan:
            action = '训练' if variant['execute'] else '复用/跳过'
            print(
                f"  round{variant['round']} {variant['name']}: {action} "
                f"(开关 {variant['execution_env']})"
            )
        previous_results = load_previous_ablation_results()
        for train_file in train_files:
            selected, variant_results, trained_results = train_one_farm_ablation(
                train_file,
                previous_results=previous_results,
            )
            rows.append(selected)
            all_ablation_results.extend(variant_results)
            current_run_results.extend(trained_results)

        detail, summary, global_best_variant = summarize_ablation(
            all_ablation_results,
            rows,
        )
        detail.to_csv(
            ABLATION_ALL_METRICS_PATH,
            index=False,
            encoding='utf-8-sig',
        )
        summary.to_csv(
            ABLATION_MODULE_SUMMARY_PATH,
            index=False,
            encoding='utf-8-sig',
        )

        round2_detail = pd.DataFrame(current_run_results)
        round2_summary = summary[summary['round'] == 2].copy()
        round2_detail.to_csv(
            ROUND2_METRICS_PATH,
            index=False,
            encoding='utf-8-sig',
        )
        round2_summary.to_csv(
            ROUND2_MODULE_SUMMARY_PATH,
            index=False,
            encoding='utf-8-sig',
        )
        print(f'合并消融明细: {ABLATION_ALL_METRICS_PATH}')
        print(f'合并模块贡献汇总: {ABLATION_MODULE_SUMMARY_PATH}')
        print(f'第二轮训练明细: {ROUND2_METRICS_PATH}')
        print(f'第二轮模块汇总: {ROUND2_MODULE_SUMMARY_PATH}')
        print(f'跨场站最优variant: {global_best_variant}')
    else:
        print('消融关闭，按原 tuned PatchTST 流程训练')
        for train_file in train_files:
            rows.append(train_one_farm(train_file))

    metrics = pd.DataFrame(rows)
    metrics_path = os.path.join(MODEL_DIR, 'tuned_patchtst_training_metrics.csv')
    metrics.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f'\n训练完成，指标已保存至 {metrics_path}')
    print(f'TensorBoard: tensorboard --logdir {TENSORBOARD_LOG_DIR}')


if __name__ == '__main__':
    main()
