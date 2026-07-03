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
    preprocess_wind_dataframe,
    transformer_encoder,
)
from wind_supplementary_preprocess import WEATHER_COLS as SUPPLEMENTARY_RAW_COLUMNS

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
ROUND3_METRICS_PATH = os.path.join(
    ABLATION_DIR,
    'tuned_patchtst_ablation_round3_metrics_all_farms.csv',
)
ROUND3_MODULE_SUMMARY_PATH = os.path.join(
    ABLATION_DIR,
    'tuned_patchtst_ablation_round3_module_summary.csv',
)
ROUND4_METRICS_PATH = os.path.join(
    ABLATION_DIR,
    'tuned_patchtst_ablation_round4_metrics_all_farms.csv',
)
ROUND4_MODULE_SUMMARY_PATH = os.path.join(
    ABLATION_DIR,
    'tuned_patchtst_ablation_round4_module_summary.csv',
)
ROUND5_METRICS_PATH = os.path.join(
    ABLATION_DIR,
    'tuned_patchtst_ablation_round5_metrics_all_farms.csv',
)
ROUND5_MODULE_SUMMARY_PATH = os.path.join(
    ABLATION_DIR,
    'tuned_patchtst_ablation_round5_module_summary.csv',
)
ROUND_OUTPUT_PATHS = {
    2: (ROUND2_METRICS_PATH, ROUND2_MODULE_SUMMARY_PATH),
    3: (ROUND3_METRICS_PATH, ROUND3_MODULE_SUMMARY_PATH),
    4: (ROUND4_METRICS_PATH, ROUND4_MODULE_SUMMARY_PATH),
    5: (ROUND5_METRICS_PATH, ROUND5_MODULE_SUMMARY_PATH),
}

seed = 2026
ENABLE_TUNED_TRAINING = os.getenv(
    'WIND_TUNED_ENABLE_TRAINING',
    '0',
) == '1'
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
# 历史消融默认全部关闭，避免直接运行脚本时重训并覆盖canonical模型。
RUN_PREVIOUS_ABLATIONS = os.getenv(
    'WIND_TUNED_RUN_PREVIOUS_ABLATIONS',
    '0',
) == '1'
RUN_ROUND2_ABLATIONS = os.getenv(
    'WIND_TUNED_RUN_ROUND2_ABLATIONS',
    '0',
) == '1'
RUN_ROUND3_ABLATIONS = os.getenv(
    'WIND_TUNED_RUN_ROUND3_ABLATIONS',
    '0',
) == '1'
RUN_ROUND4_ABLATIONS = os.getenv(
    'WIND_TUNED_RUN_ROUND4_ABLATIONS',
    '0',
) == '1'
RUN_ROUND5_ABLATIONS = os.getenv(
    'WIND_TUNED_RUN_ROUND5_ABLATIONS',
    '0',
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
RAMP_EXPERT_CONTEXT_LEN = int(os.getenv(
    'WIND_TUNED_RAMP_CONTEXT_LEN',
    '32',
))
RAMP_EXPERT_FILTERS = int(os.getenv(
    'WIND_TUNED_RAMP_FILTERS',
    '48',
))
RAMP_EXPERT_DILATIONS = tuple(
    int(value.strip())
    for value in os.getenv(
        'WIND_TUNED_RAMP_DILATIONS',
        '1,2,4,8',
    ).split(',')
    if value.strip()
)
NEW_RAMP_LOSS_WEIGHT = float(os.getenv('WIND_TUNED_NEW_RAMP_WEIGHT', '0.03'))
NEW_RELATIVE_LOSS_WEIGHT = float(os.getenv('WIND_TUNED_NEW_RELATIVE_WEIGHT', '0.03'))
NEW_PHYSICAL_PENALTY_WEIGHT = float(os.getenv('WIND_TUNED_NEW_PHYSICAL_WEIGHT', '0.01'))
RELATIVE_POWER_FLOOR = float(os.getenv('WIND_TUNED_RELATIVE_FLOOR', '0.05'))
RMSE_MSE_LOSS_WEIGHT = float(os.getenv('WIND_TUNED_RMSE_MSE_WEIGHT', '0.10'))
RMSE_HORIZON_END_WEIGHT = float(os.getenv('WIND_TUNED_RMSE_HORIZON_END_WEIGHT', '1.25'))
MULTI_SEEDS = tuple(
    int(value.strip())
    for value in os.getenv('WIND_TUNED_MULTI_SEEDS', '2026').split(',')
    if value.strip()
)
SWA_START_FRACTION = float(os.getenv('WIND_TUNED_SWA_START_FRACTION', '0.50'))
MAX_ALLOWED_NRMSE_DEGRADATION = float(
    os.getenv('WIND_TUNED_MAX_NRMSE_DEGRADATION', '0.02')
)

SUPPLEMENTARY_CACHE_DIR = os.getenv(
    'WIND_TUNED_SUPPLEMENTARY_CACHE_DIR',
    './wind_split/supplementary_other_wind_data/processed_npz',
)
SUPPLEMENTARY_PRETRAIN_EPOCHS = int(os.getenv(
    'WIND_TUNED_SUPPLEMENTARY_PRETRAIN_EPOCHS',
    '3',
))
SUPPLEMENTARY_PRETRAIN_LR = float(os.getenv(
    'WIND_TUNED_SUPPLEMENTARY_PRETRAIN_LR',
    '2e-4',
))
SUPPLEMENTARY_MAX_WINDOWS_PER_STATION = int(os.getenv(
    'WIND_TUNED_SUPPLEMENTARY_MAX_WINDOWS_PER_STATION',
    '8192',
))
SUPPLEMENTARY_MIN_WINDOWS_PER_STATION = int(os.getenv(
    'WIND_TUNED_SUPPLEMENTARY_MIN_WINDOWS_PER_STATION',
    '1024',
))
SUPPLEMENTARY_SCALED_FEATURE_CLIP = float(os.getenv(
    'WIND_TUNED_SUPPLEMENTARY_SCALED_FEATURE_CLIP',
    '8.0',
))
SUPPLEMENTARY_STATIONS = tuple(
    value.strip()
    for value in os.getenv('WIND_TUNED_SUPPLEMENTARY_STATIONS', '').split(',')
    if value.strip()
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
    {
        'name': 'revin_balanced_loss_multiseed',
        'round': 3,
        'parent_variant': 'revin_balanced_loss',
        'execution_env': 'WIND_TUNED_RUN_REVIN_BALANCED_LOSS_MULTISEED',
        'added_module': 'multi_seed_and_validation_ensemble',
        'use_revin': True,
        'use_cnn_adapter': False,
        'use_balanced_loss': True,
        'use_rmse_balanced_loss': False,
        'use_swa': False,
        'use_distillation': True,
        'multi_seed': True,
    },
    {
        'name': 'revin_rmse_balanced_loss',
        'round': 4,
        'parent_variant': 'revin_balanced_loss',
        'execution_env': 'WIND_TUNED_RUN_REVIN_RMSE_BALANCED_LOSS',
        'added_module': 'rmse_balanced_loss',
        'use_revin': True,
        'use_cnn_adapter': False,
        'use_balanced_loss': False,
        'use_rmse_balanced_loss': True,
        'use_swa': False,
        'use_distillation': True,
        'multi_seed': False,
    },
    {
        'name': 'revin_rmse_balanced_loss_no_distill',
        'round': 5,
        'parent_variant': 'revin_rmse_balanced_loss',
        'execution_env': 'WIND_TUNED_RUN_REVIN_RMSE_BALANCED_LOSS_NO_DISTILL',
        'added_module': 'disable_self_distillation',
        'use_revin': True,
        'use_cnn_adapter': False,
        'use_balanced_loss': False,
        'use_rmse_balanced_loss': True,
        'use_swa': False,
        'use_distillation': False,
        'multi_seed': False,
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
            round_defaults = {
                1: RUN_PREVIOUS_ABLATIONS,
                2: RUN_ROUND2_ABLATIONS,
                3: RUN_ROUND3_ABLATIONS,
                4: RUN_ROUND4_ABLATIONS,
                5: RUN_ROUND5_ABLATIONS,
            }
            round_default = round_defaults.get(variant['round'], False)
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


def resolve_training_seed(value, default=seed):
    if value is None or pd.isna(value):
        return int(default)
    return int(value)


def load_saved_seed_member_result(farm_id, variant, member_seed):
    if not REUSE_PREVIOUS_ABLATION_RESULTS:
        return None

    storage_name = f"{variant['name']}_seed_{member_seed}"
    dirs = make_variant_dirs(storage_name)
    artifact_path = os.path.join(
        dirs['preprocess'],
        f'tuned_patchtst_{storage_name}_farm_{farm_id}_preprocess.pkl',
    )
    if not os.path.exists(artifact_path):
        return None

    artifact = joblib.load(artifact_path)
    artifact_seed = resolve_training_seed(
        artifact.get('training_seed'),
        member_seed,
    )
    model_path = artifact.get('model_path')
    if artifact_seed != member_seed or not model_path or not os.path.exists(model_path):
        return None
    required_metrics = {
        'val_composite_score',
        'val_capacity_normalized_rmse',
    }
    if not required_metrics.issubset(artifact):
        return None

    return {
        'farm_id': str(farm_id),
        'variant': variant['name'],
        'storage_variant': storage_name,
        'round': variant['round'],
        'parent_variant': variant['parent_variant'],
        'result_source': 'reused_seed_member',
        'added_module': variant['added_module'],
        'use_revin': variant['use_revin'],
        'use_cnn_adapter': variant['use_cnn_adapter'],
        'use_ramp_expert': variant.get('use_ramp_expert', False),
        'ramp_fusion_mode': variant.get('ramp_fusion_mode', 'none'),
        'use_balanced_loss': variant['use_balanced_loss'],
        'use_rmse_balanced_loss': variant.get(
            'use_rmse_balanced_loss',
            False,
        ),
        'use_swa': variant['use_swa'],
        'use_distillation': variant.get('use_distillation', True),
        'use_supplementary_teacher_pretraining': variant.get(
            'use_supplementary_teacher_pretraining',
            False,
        ),
        'training_seed': member_seed,
        'multi_seed': True,
        'selected_weight_source': artifact.get(
            'selected_weight_source',
            'raw_best',
        ),
        'model_path': model_path,
        'best_weights_path': artifact.get('best_weights_path'),
        'artifact_path': artifact_path,
        'history_path': artifact.get('history_path'),
        'distillation_stats_path': artifact.get('distillation_stats_path'),
        **{
            key: value
            for key, value in artifact.items()
            if key.startswith('val_') or key.startswith('teacher_val_')
        },
    }


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


class SupplementaryWindowSequence(keras.utils.Sequence):
    """Balanced on-demand windows without materializing all 96xfeature tensors."""

    def __init__(self, station_arrays, batch_size=BATCH_SIZE,
                 shuffle=True, random_seed=seed):
        self.station_arrays = station_arrays
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(random_seed)
        pairs = []
        for station_index, station in enumerate(station_arrays):
            starts = np.asarray(station['window_starts'], dtype=np.int64)
            if len(starts):
                pairs.append(np.column_stack([
                    np.full(len(starts), station_index, dtype=np.int64),
                    starts,
                ]))
        if not pairs:
            raise ValueError('补充数据中没有可用于teacher预训练的连续窗口')
        self.pairs = np.concatenate(pairs, axis=0)
        self.order = np.arange(len(self.pairs), dtype=np.int64)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.order) / self.batch_size))

    def __getitem__(self, batch_index):
        start = batch_index * self.batch_size
        end = min(start + self.batch_size, len(self.order))
        selected_pairs = self.pairs[self.order[start:end]]
        x_batch = []
        y_batch = []
        for station_index, window_start in selected_pairs:
            station = self.station_arrays[int(station_index)]
            window_start = int(window_start)
            x_batch.append(
                station['features'][
                    window_start:window_start + HISTORY_LEN
                ]
            )
            target_start = window_start + HISTORY_LEN
            y_batch.append(
                station['target'][
                    target_start:target_start + FORECAST_LEN
                ]
            )
        x_batch = np.asarray(x_batch, dtype=np.float32)
        y_batch = np.asarray(y_batch, dtype=np.float32)
        return x_batch, combine_targets(y_batch)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def _valid_supplementary_window_starts(quality_mask):
    window_len = HISTORY_LEN + FORECAST_LEN
    quality_mask = np.asarray(quality_mask, dtype=np.int16)
    if len(quality_mask) < window_len:
        return np.empty(0, dtype=np.int64)
    valid_counts = np.convolve(
        quality_mask,
        np.ones(window_len, dtype=np.int16),
        mode='valid',
    )
    return np.flatnonzero(valid_counts == window_len).astype(np.int64)


def build_supplementary_transfer_bundle(
        input_cols, scaler_x, scaler_y, target_capacity,
        cache_dir=SUPPLEMENTARY_CACHE_DIR,
        max_windows_per_station=SUPPLEMENTARY_MAX_WINDOWS_PER_STATION,
        min_windows_per_station=SUPPLEMENTARY_MIN_WINDOWS_PER_STATION):
    """Map unrelated farms to the target farm's per-unit power/scaler space."""
    if target_capacity is None or target_capacity <= 0:
        raise ValueError('补充数据迁移需要有效的目标场站装机容量')

    cache_paths = sorted(glob.glob(os.path.join(cache_dir, 'JSFD*_15min.npz')))
    if SUPPLEMENTARY_STATIONS:
        allowed = set(SUPPLEMENTARY_STATIONS)
        cache_paths = [
            path for path in cache_paths
            if os.path.basename(path).split('_15min.npz')[0] in allowed
        ]
    if not cache_paths:
        raise FileNotFoundError(
            f'未在 {cache_dir} 找到补充数据缓存。请先执行 '
            'python wind_supplementary_preprocess.py'
        )

    station_arrays = []
    station_rows = []
    for cache_path in cache_paths:
        with np.load(cache_path, allow_pickle=False) as cached:
            metadata = json.loads(str(cached['metadata_json'].item()))
            raw_values = cached['raw_values'].astype(np.float32)
            timestamps_ns = cached['timestamps_ns'].astype(np.int64)
            source_power = cached['power_mw'].astype(np.float32)
            quality_mask = cached['quality_mask'].astype(bool)
            source_capacity = float(cached['capacity_mw'])

        station_id = metadata.get(
            'station_id',
            os.path.basename(cache_path).split('_15min.npz')[0],
        )
        if raw_values.shape[1] != len(SUPPLEMENTARY_RAW_COLUMNS):
            print(
                f'跳过 {station_id}: raw_values列数={raw_values.shape[1]}，'
                f'期望={len(SUPPLEMENTARY_RAW_COLUMNS)}'
            )
            continue

        external_power_pu = np.clip(
            source_power / max(source_capacity, 1e-6),
            0.0,
            1.0,
        )
        external_frame = pd.DataFrame(
            raw_values,
            columns=SUPPLEMENTARY_RAW_COLUMNS,
            index=pd.to_datetime(timestamps_ns),
        )
        external_frame.index.name = '时间'
        external_frame['装机'] = float(target_capacity)
        external_frame[TARGET_COL] = (
            external_power_pu * float(target_capacity)
        ).astype(np.float32)
        processed, _, _ = preprocess_wind_dataframe(
            external_frame,
            is_train=True,
            capacity=float(target_capacity),
        )
        if len(processed) != len(quality_mask):
            print(
                f'跳过 {station_id}: 预处理后长度 {len(processed)} '
                f'与quality mask {len(quality_mask)} 不一致'
            )
            continue

        missing_cols = [
            column for column in input_cols
            if column not in processed.columns
        ]
        if missing_cols:
            print(f'{station_id}: 补齐缺失迁移特征 {missing_cols}')
            for column in missing_cols:
                processed[column] = 0.0

        scaled_features = scaler_x.transform(
            processed[input_cols].to_numpy()
        ).astype(np.float32)
        if SUPPLEMENTARY_SCALED_FEATURE_CLIP > 0:
            scaled_features = np.clip(
                scaled_features,
                -SUPPLEMENTARY_SCALED_FEATURE_CLIP,
                SUPPLEMENTARY_SCALED_FEATURE_CLIP,
            )
        scaled_target = scaler_y.transform(
            processed[[TARGET_COL]].to_numpy()
        ).ravel().astype(np.float32)

        starts = _valid_supplementary_window_starts(quality_mask)
        available_windows = len(starts)
        if available_windows < min_windows_per_station:
            print(
                f'跳过 {station_id}: 连续有效窗口 {available_windows} '
                f'< {min_windows_per_station}'
            )
            continue
        if max_windows_per_station > 0 and available_windows > max_windows_per_station:
            station_rng = np.random.default_rng(
                seed + sum(ord(value) for value in station_id)
            )
            starts = np.sort(station_rng.choice(
                starts,
                size=max_windows_per_station,
                replace=False,
            ))

        station_arrays.append({
            'station_id': station_id,
            'features': scaled_features,
            'target': scaled_target,
            'window_starts': starts,
        })
        station_rows.append({
            'station_id': station_id,
            'cache_path': cache_path,
            'source_capacity_mw': source_capacity,
            'rows': len(processed),
            'available_valid_windows': available_windows,
            'selected_windows': len(starts),
        })

    if not station_arrays:
        raise ValueError('没有补充场站通过teacher预训练质量门槛')
    return {
        'station_arrays': station_arrays,
        'station_metrics': station_rows,
        'station_count': len(station_arrays),
        'selected_window_count': int(sum(
            len(station['window_starts'])
            for station in station_arrays
        )),
        'cache_dir': cache_dir,
    }


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
class TakeRecentTimesteps(layers.Layer):
    def __init__(self, context_len, **kwargs):
        super().__init__(**kwargs)
        self.context_len = int(context_len)

    def call(self, inputs):
        return inputs[:, -self.context_len:, :]

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.context_len, input_shape[2]

    def get_config(self):
        config = super().get_config()
        config.update({'context_len': self.context_len})
        return config


@keras.utils.register_keras_serializable(package='WindTunedPatchTST')
class CumulativeRampForecast(layers.Layer):
    """Convert horizon increments into a trajectory around persistence."""

    def call(self, inputs):
        persistence, increments = inputs
        return persistence + tf.cumsum(increments, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


@keras.utils.register_keras_serializable(package='WindTunedPatchTST')
class HorizonExpertFusion(layers.Layer):
    """Sample- and horizon-specific convex fusion of forecast experts."""

    def __init__(self, forecast_len, num_experts, initial_logits,
                 **kwargs):
        super().__init__(**kwargs)
        self.forecast_len = int(forecast_len)
        self.num_experts = int(num_experts)
        self.initial_logits = [
            float(value) for value in initial_logits
        ]
        if len(self.initial_logits) != self.num_experts:
            raise ValueError('initial_logits长度必须等于num_experts')
        bias_values = np.tile(
            np.asarray(self.initial_logits, dtype=np.float32),
            self.forecast_len,
        )
        self.gate_projection = layers.Dense(
            self.forecast_len * self.num_experts,
            kernel_initializer='zeros',
            bias_initializer=keras.initializers.Constant(bias_values),
            name='gate_projection',
        )

    def call(self, inputs):
        context = inputs[0]
        experts = inputs[1:]
        if len(experts) != self.num_experts:
            raise ValueError('输入expert数量与num_experts不一致')
        logits = self.gate_projection(context)
        logits = tf.reshape(
            logits,
            [-1, self.forecast_len, self.num_experts],
        )
        weights = tf.nn.softmax(logits, axis=-1)
        stacked = tf.stack(experts, axis=-1)
        weights = tf.cast(weights, stacked.dtype)
        return tf.reduce_sum(weights * stacked, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def get_config(self):
        config = super().get_config()
        config.update({
            'forecast_len': self.forecast_len,
            'num_experts': self.num_experts,
            'initial_logits': self.initial_logits,
        })
        return config


@keras.utils.register_keras_serializable(package='WindTunedPatchTST')
class TunedPatchTSTLoss(keras.losses.Loss):
    def __init__(self, forecast_len=FORECAST_LEN, horizon_weight_values=None,
                 distill_alpha=DISTILL_ALPHA, zero_scaled=0.0, capacity_scaled=None,
                 physical_penalty_weight=PHYSICAL_PENALTY_WEIGHT,
                 smoothness_weight=SMOOTHNESS_WEIGHT, delta=HUBER_DELTA,
                 name='tuned_patchtst_loss',
                 reduction=keras.losses.Reduction.AUTO):
        super().__init__(name=name, reduction=reduction)
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
                 delta=HUBER_DELTA, name='balanced_tuned_patchtst_loss',
                 reduction=keras.losses.Reduction.AUTO):
        super().__init__(name=name, reduction=reduction)
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
class RMSEBalancedTunedPatchTSTLoss(BalancedTunedPatchTSTLoss):
    def __init__(self, forecast_len=FORECAST_LEN, distill_alpha=DISTILL_ALPHA,
                 zero_scaled=0.0, capacity_scaled=None,
                 ramp_loss_weight=NEW_RAMP_LOSS_WEIGHT,
                 relative_loss_weight=NEW_RELATIVE_LOSS_WEIGHT,
                 physical_penalty_weight=NEW_PHYSICAL_PENALTY_WEIGHT,
                 relative_power_floor=RELATIVE_POWER_FLOOR,
                 mse_loss_weight=RMSE_MSE_LOSS_WEIGHT,
                 horizon_end_weight=RMSE_HORIZON_END_WEIGHT,
                 delta=HUBER_DELTA, name='rmse_balanced_tuned_patchtst_loss',
                 reduction=keras.losses.Reduction.AUTO):
        super().__init__(
            forecast_len=forecast_len,
            distill_alpha=distill_alpha,
            zero_scaled=zero_scaled,
            capacity_scaled=capacity_scaled,
            ramp_loss_weight=ramp_loss_weight,
            relative_loss_weight=relative_loss_weight,
            physical_penalty_weight=physical_penalty_weight,
            relative_power_floor=relative_power_floor,
            delta=delta,
            name=name,
            reduction=reduction,
        )
        self.mse_loss_weight = float(mse_loss_weight)
        self.horizon_end_weight = float(horizon_end_weight)

    def call(self, y_true, y_pred):
        base_loss = super().call(y_true, y_pred)
        actual = y_true[:, :self.forecast_len]
        horizon_weight_values = tf.linspace(
            tf.cast(1.0, y_pred.dtype),
            tf.cast(self.horizon_end_weight, y_pred.dtype),
            self.forecast_len,
        )
        horizon_weight_values /= tf.reduce_mean(horizon_weight_values)
        weighted_mse = tf.reduce_mean(
            horizon_weight_values[tf.newaxis, :] * tf.square(actual - y_pred)
        )
        return base_loss + self.mse_loss_weight * weighted_mse

    def get_config(self):
        config = super().get_config()
        config.update({
            'mse_loss_weight': self.mse_loss_weight,
            'horizon_end_weight': self.horizon_end_weight,
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
                        epochs, distill_alpha, use_balanced_loss=False,
                        use_rmse_balanced_loss=False):
    zero_scaled, capacity_scaled = scaled_bounds(scaler_y, capacity)
    if use_rmse_balanced_loss:
        loss = RMSEBalancedTunedPatchTSTLoss(
            forecast_len=FORECAST_LEN,
            distill_alpha=distill_alpha,
            zero_scaled=zero_scaled,
            capacity_scaled=capacity_scaled,
        )
    elif use_balanced_loss:
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


def pretrain_teacher_with_supplementary(
        model, supplementary_bundle, scaler_y, capacity, variant,
        training_seed, tensorboard_log_dir):
    sequence = SupplementaryWindowSequence(
        supplementary_bundle['station_arrays'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        random_seed=training_seed,
    )
    compile_tuned_model(
        model,
        scaler_y,
        capacity,
        SUPPLEMENTARY_PRETRAIN_LR,
        len(sequence),
        SUPPLEMENTARY_PRETRAIN_EPOCHS,
        distill_alpha=0.0,
        use_balanced_loss=variant['use_balanced_loss'],
        use_rmse_balanced_loss=variant.get(
            'use_rmse_balanced_loss',
            False,
        ),
    )
    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=0,
            write_graph=True,
            update_freq='epoch',
            profile_batch=0,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]
    print(
        '补充数据teacher预训练: '
        f"stations={supplementary_bundle['station_count']}, "
        f"windows={supplementary_bundle['selected_window_count']}, "
        f'epochs={SUPPLEMENTARY_PRETRAIN_EPOCHS}'
    )
    history = model.fit(
        sequence,
        epochs=SUPPLEMENTARY_PRETRAIN_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def build_cnn_ramp_expert(x_input, baseline, channel_indices):
    if not channel_indices:
        raise ValueError('CNN ramp expert需要功率/风速通道索引')
    if RAMP_EXPERT_CONTEXT_LEN > HISTORY_LEN:
        raise ValueError('CNN ramp context不能超过历史窗口长度')

    ramp = SelectInputChannels(
        channel_indices,
        name='ramp_expert_input_channels',
    )(x_input)
    ramp = TakeRecentTimesteps(
        RAMP_EXPERT_CONTEXT_LEN,
        name='ramp_expert_recent_context',
    )(ramp)
    ramp = layers.Conv1D(
        RAMP_EXPERT_FILTERS,
        kernel_size=1,
        padding='same',
        name='ramp_expert_input_projection',
    )(ramp)
    for block_index, dilation in enumerate(RAMP_EXPERT_DILATIONS, start=1):
        shortcut = ramp
        block = layers.SeparableConv1D(
            RAMP_EXPERT_FILTERS,
            kernel_size=3,
            padding='causal',
            dilation_rate=dilation,
            activation='gelu',
            name=f'ramp_expert_block_{block_index}_conv1',
        )(ramp)
        block = layers.Dropout(
            DROPOUT,
            name=f'ramp_expert_block_{block_index}_dropout',
        )(block)
        block = layers.SeparableConv1D(
            RAMP_EXPERT_FILTERS,
            kernel_size=3,
            padding='causal',
            dilation_rate=dilation,
            name=f'ramp_expert_block_{block_index}_conv2',
        )(block)
        ramp = layers.Add(
            name=f'ramp_expert_block_{block_index}_add',
        )([shortcut, block])
        ramp = layers.LayerNormalization(
            epsilon=1e-6,
            name=f'ramp_expert_block_{block_index}_norm',
        )(ramp)
        ramp = layers.Activation(
            'gelu',
            name=f'ramp_expert_block_{block_index}_activation',
        )(ramp)

    # Flatten preserves the location of recent ramp events; unlike the old
    # CNN Adapter, no global time averaging is used.
    ramp_context = layers.Flatten(name='ramp_expert_temporal_flatten')(ramp)
    ramp_context = layers.Dense(
        D_FF,
        activation='gelu',
        kernel_regularizer=regularizers.l2(1e-4),
        name='ramp_expert_context',
    )(ramp_context)
    ramp_context = layers.Dropout(
        HEAD_DROPOUT,
        name='ramp_expert_context_dropout',
    )(ramp_context)
    increments = layers.Dense(
        FORECAST_LEN,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer='zeros',
        dtype='float32',
        name='ramp_expert_horizon_increments',
    )(ramp_context)
    ramp_forecast = CumulativeRampForecast(
        name='ramp_expert_forecast',
    )([baseline, increments])
    ramp_residual = layers.Subtract(
        name='ramp_expert_cumulative_residual',
    )([ramp_forecast, baseline])
    return ramp_forecast, ramp_residual, ramp_context


def build_tuned_patchtst_model(
        input_dim, target_channel_index, use_revin=False,
        use_cnn_adapter=False, adapter_channel_indices=None,
        use_ramp_expert=False, ramp_fusion_mode='none'):
    if target_channel_index is None:
        raise ValueError('Tuned PatchTST 需要将历史功率作为输入通道')
    if use_cnn_adapter and not adapter_channel_indices:
        raise ValueError('启用CNN Adapter时必须提供功率/风速通道索引')
    valid_fusion_modes = {
        'none',
        'residual',
        'two_expert_gating',
        'three_expert_gating',
    }
    if ramp_fusion_mode not in valid_fusion_modes:
        raise ValueError(
            f'未知ramp_fusion_mode={ramp_fusion_mode}; '
            f'可选={sorted(valid_fusion_modes)}'
        )
    if use_ramp_expert != (ramp_fusion_mode != 'none'):
        raise ValueError(
            'use_ramp_expert与ramp_fusion_mode必须同步启用或关闭'
        )
    if use_ramp_expert and not adapter_channel_indices:
        raise ValueError('启用CNN ramp expert时必须提供功率/风速通道索引')

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
    if not use_ramp_expert:
        outputs = layers.Add(name=forecast_name)([baseline, residual])
    else:
        long_forecast = layers.Add(
            name='long_patchtst_forecast',
        )([baseline, residual])
        ramp_forecast, ramp_residual, ramp_context = build_cnn_ramp_expert(
            x_input,
            baseline,
            adapter_channel_indices,
        )
        fusion_context = layers.Concatenate(
            name='ramp_fusion_context',
        )([head, ramp_context])
        if ramp_fusion_mode == 'residual':
            outputs = ZeroInitResidualAdapter(
                name='zero_init_ramp_trajectory_adapter',
            )([long_forecast, ramp_residual])
        elif ramp_fusion_mode == 'two_expert_gating':
            outputs = HorizonExpertFusion(
                FORECAST_LEN,
                2,
                initial_logits=[4.0, 0.0],
                name='long_ramp_horizon_fusion',
            )([fusion_context, long_forecast, ramp_forecast])
        else:
            outputs = HorizonExpertFusion(
                FORECAST_LEN,
                3,
                initial_logits=[4.0, 0.0, 0.0],
                name='long_ramp_persistence_horizon_fusion',
            )([
                fusion_context,
                long_forecast,
                ramp_forecast,
                baseline,
            ])
        outputs = layers.Activation(
            'linear',
            name=forecast_name,
        )(outputs)
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


def evaluate_scaled_predictions(y_pred_scaled, y_val, scaler_y, capacity=None):
    y_pred_scaled = np.asarray(y_pred_scaled)
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


def evaluate_model(model, val_feature_ds, y_val, scaler_y, capacity=None):
    y_pred_scaled = model.predict(val_feature_ds, verbose=0)
    return evaluate_scaled_predictions(
        y_pred_scaled,
        y_val,
        scaler_y,
        capacity,
    )


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
        'enable_tuned_training': ENABLE_TUNED_TRAINING,
        'run_ablation': RUN_ABLATION,
        'run_previous_ablations': RUN_PREVIOUS_ABLATIONS,
        'run_round2_ablations': RUN_ROUND2_ABLATIONS,
        'run_round3_ablations': RUN_ROUND3_ABLATIONS,
        'run_round4_ablations': RUN_ROUND4_ABLATIONS,
        'run_round5_ablations': RUN_ROUND5_ABLATIONS,
        'reuse_previous_ablation_results': REUSE_PREVIOUS_ABLATION_RESULTS,
        'ablation_variants': ABLATION_VARIANTS,
        'ablation_execution_plan': get_ablation_execution_plan(),
        'revin_epsilon': REVIN_EPSILON,
        'cnn_adapter_filters': CNN_ADAPTER_FILTERS,
        'ramp_expert_context_len': RAMP_EXPERT_CONTEXT_LEN,
        'ramp_expert_filters': RAMP_EXPERT_FILTERS,
        'ramp_expert_dilations': list(RAMP_EXPERT_DILATIONS),
        'new_ramp_loss_weight': NEW_RAMP_LOSS_WEIGHT,
        'new_relative_loss_weight': NEW_RELATIVE_LOSS_WEIGHT,
        'new_physical_penalty_weight': NEW_PHYSICAL_PENALTY_WEIGHT,
        'relative_power_floor': RELATIVE_POWER_FLOOR,
        'rmse_mse_loss_weight': RMSE_MSE_LOSS_WEIGHT,
        'rmse_horizon_end_weight': RMSE_HORIZON_END_WEIGHT,
        'multi_seeds': list(MULTI_SEEDS),
        'exp_weight_halflife_steps': EXP_WEIGHT_HALFLIFE_STEPS,
        'swa_start_fraction': SWA_START_FRACTION,
        'max_allowed_nrmse_degradation': MAX_ALLOWED_NRMSE_DEGRADATION,
        'supplementary_cache_dir': SUPPLEMENTARY_CACHE_DIR,
        'supplementary_pretrain_epochs': SUPPLEMENTARY_PRETRAIN_EPOCHS,
        'supplementary_pretrain_learning_rate': SUPPLEMENTARY_PRETRAIN_LR,
        'supplementary_max_windows_per_station': (
            SUPPLEMENTARY_MAX_WINDOWS_PER_STATION
        ),
        'supplementary_min_windows_per_station': (
            SUPPLEMENTARY_MIN_WINDOWS_PER_STATION
        ),
        'supplementary_scaled_feature_clip': (
            SUPPLEMENTARY_SCALED_FEATURE_CLIP
        ),
        'supplementary_stations': list(SUPPLEMENTARY_STATIONS),
    }
    config_path = os.path.join(MODEL_DIR, 'tuned_patchtst_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return config_path


def train_ablation_variant(farm_id, variant, features, y_train, y_val,
                           train_samples, total_samples, input_cols,
                           target_index, adapter_channel_indices,
                           scaler_x, scaler_y, feature_cols, capacity,
                           training_seed=None, storage_name=None,
                           supplementary_bundle=None):
    variant_name = variant['name']
    training_seed = seed if training_seed is None else int(training_seed)
    storage_name = storage_name or variant_name
    dirs = make_variant_dirs(storage_name)
    val_samples = total_samples - train_samples
    steps_per_epoch = int(np.ceil(train_samples / BATCH_SIZE))

    keras.backend.clear_session()
    set_global_seed(training_seed)
    print(
        f'\n--- 消融variant={variant_name}, seed={training_seed}: '
        f"RevIN={variant['use_revin']}, "
        f"CNN Adapter={variant['use_cnn_adapter']}, "
        f"CNN ramp expert={variant.get('use_ramp_expert', False)}, "
        f"Ramp fusion={variant.get('ramp_fusion_mode', 'none')}, "
        f"Balanced loss={variant['use_balanced_loss']}, "
        f"RMSE loss={variant.get('use_rmse_balanced_loss', False)}, "
        f"External teacher={variant.get('use_supplementary_teacher_pretraining', False)}, "
        f"Distill={variant.get('use_distillation', True)}, "
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
        use_ramp_expert=variant.get('use_ramp_expert', False),
        ramp_fusion_mode=variant.get('ramp_fusion_mode', 'none'),
    )
    print(f'variant={variant_name}, parameters={model.count_params():,}')

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    supplementary_pretrain_weights_path = os.path.join(
        dirs['weights'],
        f'tuned_patchtst_{storage_name}_external_pretrain_'
        f'farm_{farm_id}.weights.h5',
    )
    teacher_weights_path = os.path.join(
        dirs['weights'],
        f'tuned_patchtst_{storage_name}_teacher_farm_{farm_id}_best.weights.h5',
    )
    teacher_model_path = os.path.join(
        dirs['teachers'],
        f'tuned_patchtst_{storage_name}_teacher_farm_{farm_id}.keras',
    )
    raw_weights_path = os.path.join(
        dirs['weights'],
        f'tuned_patchtst_{storage_name}_farm_{farm_id}_raw_best.weights.h5',
    )
    averaged_weights_path = os.path.join(
        dirs['weights'],
        f'tuned_patchtst_{storage_name}_farm_{farm_id}_swa.weights.h5',
    )
    final_weights_path = os.path.join(
        dirs['weights'],
        f'tuned_patchtst_{storage_name}_farm_{farm_id}_best.weights.h5',
    )
    final_model_path = os.path.join(
        dirs['models'],
        f'tuned_patchtst_{storage_name}_farm_{farm_id}.keras',
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
        (
            'distill'
            if variant.get('use_distillation', True)
            else 'supervised_continue'
        ),
        timestamp,
    )
    supplementary_log_dir = os.path.join(
        dirs['tensorboard'],
        f'farm_{farm_id}',
        'supplementary_teacher_pretrain',
        timestamp,
    )

    use_supplementary_teacher = variant.get(
        'use_supplementary_teacher_pretraining',
        False,
    )
    supplementary_history = None
    if use_supplementary_teacher:
        if supplementary_bundle is None:
            supplementary_bundle = build_supplementary_transfer_bundle(
                input_cols,
                scaler_x,
                scaler_y,
                capacity,
            )
        supplementary_history = pretrain_teacher_with_supplementary(
            model,
            supplementary_bundle,
            scaler_y,
            capacity,
            variant,
            training_seed,
            supplementary_log_dir,
        )
        model.save_weights(supplementary_pretrain_weights_path)

    # Supplementary pretraining is only an initialization.  The teacher is
    # always fine-tuned and selected on the target farm's chronological split.
    compile_tuned_model(
        model,
        scaler_y,
        capacity,
        BASE_LEARNING_RATE,
        steps_per_epoch,
        COLD_START_EPOCHS,
        distill_alpha=0.0,
        use_balanced_loss=variant['use_balanced_loss'],
        use_rmse_balanced_loss=variant.get('use_rmse_balanced_loss', False),
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

    use_distillation = variant.get('use_distillation', True)
    if use_distillation:
        teacher_train_pred = model.predict(train_feature_ds, verbose=0)
        teacher_val_pred = model.predict(val_feature_ds, verbose=0)
        teacher_eval_metrics = evaluate_scaled_predictions(
            teacher_val_pred,
            y_val,
            scaler_y,
            capacity,
        )
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
            filename_prefix=f'tuned_patchtst_{storage_name}',
        )
        y_train_stage2 = combine_targets(y_train, teacher_train_pred, train_conf)
        y_val_stage2 = combine_targets(y_val, teacher_val_pred, val_conf)
        stage2_distill_alpha = DISTILL_ALPHA
    else:
        distill_stats_path = None
        teacher_val_pred = model.predict(val_feature_ds, verbose=0)
        teacher_eval_metrics = evaluate_scaled_predictions(
            teacher_val_pred,
            y_val,
            scaler_y,
            capacity,
        )
        y_train_stage2 = combine_targets(y_train)
        y_val_stage2 = combine_targets(y_val)
        stage2_distill_alpha = 0.0
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
        distill_alpha=stage2_distill_alpha,
        use_balanced_loss=variant['use_balanced_loss'],
        use_rmse_balanced_loss=variant.get('use_rmse_balanced_loss', False),
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
    histories = []
    if supplementary_history is not None:
        histories.append(('supplementary_pretrain', supplementary_history))
    histories.extend([
        ('cold_start', teacher_history),
        (
            'distill' if use_distillation else 'supervised_continue',
            distill_history,
        ),
    ])
    history_path, history_plot_path = save_history_artifacts(
        histories,
        farm_id,
        output_dir=dirs['history'],
        filename_prefix=f'tuned_patchtst_{storage_name}',
    )

    artifact = {
        'model_name': TUNED_MODEL_NAME,
        'farm_id': farm_id,
        'ablation_variant': variant_name,
        'storage_variant': storage_name,
        'ablation_round': variant['round'],
        'parent_variant': variant['parent_variant'],
        'added_module': variant['added_module'],
        'use_revin': variant['use_revin'],
        'use_cnn_adapter': variant['use_cnn_adapter'],
        'use_ramp_expert': variant.get('use_ramp_expert', False),
        'ramp_fusion_mode': variant.get('ramp_fusion_mode', 'none'),
        'ramp_expert_context_len': RAMP_EXPERT_CONTEXT_LEN,
        'ramp_expert_filters': RAMP_EXPERT_FILTERS,
        'ramp_expert_dilations': list(RAMP_EXPERT_DILATIONS),
        'use_balanced_loss': variant['use_balanced_loss'],
        'use_rmse_balanced_loss': variant.get('use_rmse_balanced_loss', False),
        'use_swa': variant['use_swa'],
        'use_distillation': use_distillation,
        'use_supplementary_teacher_pretraining': use_supplementary_teacher,
        'training_seed': training_seed,
        'multi_seed': variant.get('multi_seed', False),
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
        'supplementary_pretrain_weights_path': (
            supplementary_pretrain_weights_path
            if use_supplementary_teacher
            else None
        ),
        'supplementary_cache_dir': (
            supplementary_bundle['cache_dir']
            if use_supplementary_teacher
            else None
        ),
        'supplementary_station_count': (
            supplementary_bundle['station_count']
            if use_supplementary_teacher
            else 0
        ),
        'supplementary_selected_window_count': (
            supplementary_bundle['selected_window_count']
            if use_supplementary_teacher
            else 0
        ),
        'supplementary_station_metrics': (
            supplementary_bundle['station_metrics']
            if use_supplementary_teacher
            else []
        ),
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
        'distill_alpha': stage2_distill_alpha,
        'teacher_keep_ratio': TEACHER_KEEP_RATIO,
        'horizon_decay': HORIZON_DECAY,
        **{
            f'teacher_{key}': value
            for key, value in teacher_eval_metrics.items()
        },
        **eval_metrics,
    }
    artifact_path = os.path.join(
        dirs['preprocess'],
        f'tuned_patchtst_{storage_name}_farm_{farm_id}_preprocess.pkl',
    )
    joblib.dump(artifact, artifact_path)

    result = {
        'farm_id': farm_id,
        'variant': variant_name,
        'storage_variant': storage_name,
        'round': variant['round'],
        'parent_variant': variant['parent_variant'],
        'result_source': 'trained_current_run',
        'added_module': variant['added_module'],
        'use_revin': variant['use_revin'],
        'use_cnn_adapter': variant['use_cnn_adapter'],
        'use_ramp_expert': variant.get('use_ramp_expert', False),
        'ramp_fusion_mode': variant.get('ramp_fusion_mode', 'none'),
        'ramp_expert_context_len': RAMP_EXPERT_CONTEXT_LEN,
        'ramp_expert_filters': RAMP_EXPERT_FILTERS,
        'ramp_expert_dilations': list(RAMP_EXPERT_DILATIONS),
        'use_balanced_loss': variant['use_balanced_loss'],
        'use_rmse_balanced_loss': variant.get('use_rmse_balanced_loss', False),
        'use_swa': variant['use_swa'],
        'use_distillation': use_distillation,
        'use_supplementary_teacher_pretraining': use_supplementary_teacher,
        'training_seed': training_seed,
        'multi_seed': variant.get('multi_seed', False),
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
        **{
            f'teacher_{key}': value
            for key, value in teacher_eval_metrics.items()
        },
        **eval_metrics,
    }
    print(
        f"variant={variant_name}: score={eval_metrics['val_composite_score']:.6f}, "
        f"NRMSE={eval_metrics['val_capacity_normalized_rmse']:.6f}, "
        f"SMAPE={eval_metrics['val_stable_smape']:.3f}%"
    )
    return result


def train_multiseed_ablation_variant(
        farm_id, variant, parent_result, features, y_train, y_val,
        train_samples, total_samples, input_cols, target_index,
        adapter_channel_indices, scaler_x, scaler_y, feature_cols, capacity):
    if not MULTI_SEEDS:
        raise ValueError('WIND_TUNED_MULTI_SEEDS至少需要包含一个随机种子')

    member_results = []
    parent_seed = (
        resolve_training_seed(parent_result.get('training_seed'), seed)
        if parent_result
        else None
    )
    supplementary_bundle = None
    if variant.get('use_supplementary_teacher_pretraining', False):
        supplementary_bundle = build_supplementary_transfer_bundle(
            input_cols,
            scaler_x,
            scaler_y,
            capacity,
        )
    for member_seed in MULTI_SEEDS:
        if (
            variant.get('reuse_parent_seed_member', True)
            and parent_result is not None
            and member_seed == parent_seed
            and result_artifacts_exist(parent_result)
        ):
            member_result = dict(parent_result)
            member_result['training_seed'] = member_seed
            member_result['member_source'] = 'reused_parent_variant'
            print(
                f"多seed复用父分支: variant={parent_result['variant']}, "
                f'seed={member_seed}'
            )
        else:
            member_result = load_saved_seed_member_result(
                farm_id,
                variant,
                member_seed,
            )
            if member_result is not None:
                member_result['member_source'] = 'reused_saved_seed_member'
                print(
                    f"多seed复用已保存成员: variant={variant['name']}, "
                    f'seed={member_seed}'
                )
            else:
                storage_name = f"{variant['name']}_seed_{member_seed}"
                member_result = train_ablation_variant(
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
                    training_seed=member_seed,
                    storage_name=storage_name,
                    supplementary_bundle=supplementary_bundle,
                )
                member_result['member_source'] = 'trained_seed_member'
        member_results.append(member_result)

    val_samples = total_samples - train_samples
    val_feature_ds = make_feature_dataset(
        features,
        train_samples,
        val_samples,
    )
    member_predictions = []
    for member_result in member_results:
        keras.backend.clear_session()
        member_model = keras.models.load_model(
            member_result['model_path'],
            compile=False,
        )
        member_predictions.append(
            member_model.predict(val_feature_ds, verbose=0)
        )
        del member_model

    ensemble_prediction = np.mean(np.stack(member_predictions, axis=0), axis=0)
    ensemble_metrics = evaluate_scaled_predictions(
        ensemble_prediction,
        y_val,
        scaler_y,
        capacity,
    )
    best_member = min(
        member_results,
        key=lambda result: (
            result['val_capacity_normalized_rmse'],
            result['val_composite_score'],
        ),
    )
    is_multi_seed = len(member_results) > 1
    use_seed_ensemble = (
        is_multi_seed
        and ensemble_metrics['val_capacity_normalized_rmse']
        <= best_member['val_capacity_normalized_rmse']
        and ensemble_metrics['val_composite_score']
        <= best_member['val_composite_score']
    )
    selected_metrics = (
        ensemble_metrics
        if use_seed_ensemble
        else {
            key: value
            for key, value in best_member.items()
            if key.startswith('val_')
        }
    )
    selected_weight_source = (
        'seed_ensemble'
        if use_seed_ensemble
        else f"best_seed_{best_member['training_seed']}"
    )
    seed_nrmse_values = np.asarray([
        result['val_capacity_normalized_rmse']
        for result in member_results
    ], dtype=float)
    seed_score_values = np.asarray([
        result['val_composite_score']
        for result in member_results
    ], dtype=float)
    seed_statistics = {
        'seed_nrmse_mean': float(np.mean(seed_nrmse_values)),
        'seed_nrmse_std': float(np.std(seed_nrmse_values)),
        'seed_composite_score_mean': float(np.mean(seed_score_values)),
        'seed_composite_score_std': float(np.std(seed_score_values)),
    }

    dirs = make_variant_dirs(variant['name'])
    member_metrics_path = os.path.join(
        dirs['root'],
        f"tuned_patchtst_{variant['name']}_seed_metrics_farm_{farm_id}.csv",
    )
    member_metric_rows = []
    for member_result in member_results:
        member_metric_rows.append({
            'farm_id': farm_id,
            'variant': variant['name'],
            'training_seed': member_result['training_seed'],
            'member_source': member_result.get('member_source'),
            'model_path': member_result['model_path'],
            'artifact_path': member_result['artifact_path'],
            **{
                key: value
                for key, value in member_result.items()
                if key.startswith('val_') or key.startswith('teacher_val_')
            },
        })
    if is_multi_seed:
        member_metric_rows.append({
            'farm_id': farm_id,
            'variant': variant['name'],
            'training_seed': 'ensemble',
            'member_source': 'mean_prediction',
            'model_path': '',
            'artifact_path': '',
            **ensemble_metrics,
        })
    pd.DataFrame(member_metric_rows).to_csv(
        member_metrics_path,
        index=False,
        encoding='utf-8-sig',
    )

    selected_artifact = joblib.load(best_member['artifact_path'])
    ensemble_model_paths = [
        result['model_path']
        for result in member_results
    ]
    selected_artifact.update({
        'ablation_variant': variant['name'],
        'storage_variant': variant['name'],
        'ablation_round': variant['round'],
        'parent_variant': variant['parent_variant'],
        'added_module': variant['added_module'],
        'use_revin': variant['use_revin'],
        'use_cnn_adapter': variant['use_cnn_adapter'],
        'use_balanced_loss': variant['use_balanced_loss'],
        'use_rmse_balanced_loss': variant.get(
            'use_rmse_balanced_loss',
            False,
        ),
        'use_swa': variant['use_swa'],
        'use_distillation': variant.get('use_distillation', True),
        'use_supplementary_teacher_pretraining': variant.get(
            'use_supplementary_teacher_pretraining',
            False,
        ),
        'multi_seed': is_multi_seed,
        'multi_seed_values': list(MULTI_SEEDS),
        'use_seed_ensemble': use_seed_ensemble,
        'ensemble_model_paths': ensemble_model_paths,
        'ensemble_member_count': len(ensemble_model_paths),
        'selected_weight_source': selected_weight_source,
        'seed_member_metrics_path': member_metrics_path,
        'model_path': best_member['model_path'],
        'best_weights_path': best_member['best_weights_path'],
        **seed_statistics,
        **selected_metrics,
    })
    artifact_path = os.path.join(
        dirs['preprocess'],
        f"tuned_patchtst_{variant['name']}_farm_{farm_id}_preprocess.pkl",
    )
    joblib.dump(selected_artifact, artifact_path)

    result = {
        'farm_id': farm_id,
        'variant': variant['name'],
        'storage_variant': variant['name'],
        'round': variant['round'],
        'parent_variant': variant['parent_variant'],
        'result_source': 'trained_current_run',
        'added_module': variant['added_module'],
        'use_revin': variant['use_revin'],
        'use_cnn_adapter': variant['use_cnn_adapter'],
        'use_balanced_loss': variant['use_balanced_loss'],
        'use_rmse_balanced_loss': variant.get('use_rmse_balanced_loss', False),
        'use_swa': variant['use_swa'],
        'use_distillation': variant.get('use_distillation', True),
        'use_supplementary_teacher_pretraining': variant.get(
            'use_supplementary_teacher_pretraining',
            False,
        ),
        'training_seed': best_member['training_seed'],
        'multi_seed': is_multi_seed,
        'multi_seed_values': ','.join(str(value) for value in MULTI_SEEDS),
        'use_seed_ensemble': use_seed_ensemble,
        'ensemble_member_count': len(ensemble_model_paths),
        'selected_weight_source': selected_weight_source,
        'model_path': best_member['model_path'],
        'best_weights_path': best_member['best_weights_path'],
        'artifact_path': artifact_path,
        'history_path': best_member.get('history_path'),
        'distillation_stats_path': best_member.get('distillation_stats_path'),
        'seed_member_metrics_path': member_metrics_path,
        'train_samples': train_samples,
        'val_samples': val_samples,
        'raw_val_composite_score': np.nan,
        'swa_val_composite_score': np.nan,
        **{
            key: value
            for key, value in best_member.items()
            if key.startswith('teacher_val_')
        },
        **seed_statistics,
        **selected_metrics,
    }
    print(
        f"多seed选择: {selected_weight_source}, "
        f"score={selected_metrics['val_composite_score']:.6f}, "
        f"NRMSE={selected_metrics['val_capacity_normalized_rmse']:.6f}"
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
        'selected_training_seed': selected_result.get('training_seed'),
        'selected_use_seed_ensemble': selected_result.get(
            'use_seed_ensemble',
            False,
        ),
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
        'selected_training_seed': selected_result.get('training_seed'),
        'selected_use_seed_ensemble': selected_result.get(
            'use_seed_ensemble',
            False,
        ),
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
            if variant.get('multi_seed', False):
                result = train_multiseed_ablation_variant(
                    farm_id,
                    variant,
                    result_by_variant.get(variant.get('parent_variant')),
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
            else:
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
        parent_nrmse_delta = pd.Series(dtype=float)
        if parent_variant is not None:
            parent_df = detail[detail['variant'] == parent_variant][
                [
                    'farm_id',
                    'val_composite_score',
                    'val_capacity_normalized_rmse',
                ]
            ].rename(columns={
                'val_composite_score': 'parent_score',
                'val_capacity_normalized_rmse': 'parent_nrmse',
            })
            parent_compared = variant_df.merge(parent_df, on='farm_id', how='inner')
            parent_score_delta = (
                parent_compared['val_composite_score']
                - parent_compared['parent_score']
            )
            parent_nrmse_delta = (
                parent_compared['val_capacity_normalized_rmse']
                / parent_compared['parent_nrmse']
                - 1.0
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
            'mean_nrmse_change_vs_parent': (
                np.nan
                if parent_nrmse_delta.empty
                else float(parent_nrmse_delta.mean())
            ),
            'nrmse_improved_farms_vs_parent': (
                0
                if parent_nrmse_delta.empty
                else int((parent_nrmse_delta < 0).sum())
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
    if not ENABLE_TUNED_TRAINING:
        print(
            'tuned PatchTST历史训练入口默认关闭，未执行训练。'
            '如需显式运行前五轮代码，请设置 '
            'WIND_TUNED_ENABLE_TRAINING=1，并单独开启所需variant。'
        )
        return
    if (
        RUN_ABLATION
        and not any(
            variant['execute']
            for variant in get_ablation_execution_plan()
        )
    ):
        print(
            'WIND_TUNED_ENABLE_TRAINING已开启，但没有启用任何历史消融'
            'variant；为避免仅复用结果却覆盖canonical artifact，本次不执行。'
        )
        return
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

        print(f'合并消融明细: {ABLATION_ALL_METRICS_PATH}')
        print(f'合并模块贡献汇总: {ABLATION_MODULE_SUMMARY_PATH}')
        for round_number, (metrics_path, round_summary_path) in ROUND_OUTPUT_PATHS.items():
            round_results = [
                result for result in current_run_results
                if int(result['round']) == round_number
            ]
            if not round_results:
                continue
            pd.DataFrame(round_results).to_csv(
                metrics_path,
                index=False,
                encoding='utf-8-sig',
            )
            summary[summary['round'] == round_number].to_csv(
                round_summary_path,
                index=False,
                encoding='utf-8-sig',
            )
            print(f'第{round_number}轮训练明细: {metrics_path}')
            print(f'第{round_number}轮模块汇总: {round_summary_path}')
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
