import glob
import json
import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras

from wind_dl_model_train import (
    BATCH_SIZE as PATCHTST_BATCH_SIZE,
    DATA_DIR,
    FORECAST_LEN,
    HISTORY_LEN,
    MODEL_DIR as PATCHTST_MODEL_DIR,
    SAVED_MODEL_DIR,
    TARGET_COL,
    LearnablePositionEmbedding,
    MergeChannels,
    PatchExtract,
    RestoreChannels,
    TakeChannel,
    build_patchtst_model,
    load_and_preprocess,
)
from wind_dl_other_models_train import (
    BASE_RESULT_DIR,
    BATCH_SIZE as OTHER_BATCH_SIZE,
    DEFAULT_MODEL_NAMES as OTHER_MODEL_NAMES,
    MODEL_BUILDERS,
    AutoformerDataEmbedding,
    AutoformerDecoderInitializer,
    AutoformerDecoderLayer,
    AutoformerEncoderLayer,
    AutoformerLayerNorm,
    CircularTokenEmbedding,
    FixedPositionEmbedding,
    InformerDataEmbedding,
    MovingAverage,
    ProbSparseSelfAttention,
    SeriesDecomposition,
    SeriesWiseAutoCorrelation,
    seed,
    set_global_seed,
)
from wind_dl_tuned_patchtst_train import (
    TUNED_MODEL_NAME,
    SAVED_MODEL_DIR as TUNED_SAVED_MODEL_DIR,
    WEIGHTS_DIR as TUNED_WEIGHTS_DIR,
    BalancedTunedPatchTSTLoss,
    CumulativeRampForecast,
    HorizonExpertFusion,
    PowerRevIN,
    PowerRevINDenormalize,
    RMSEBalancedTunedPatchTSTLoss,
    RepeatLastTarget,
    SelectInputChannels,
    TakeRecentTimesteps,
    TunedPatchTSTLoss,
    ZeroInitResidualAdapter,
    actual_mae,
    actual_rmse,
    build_tuned_patchtst_model,
)

warnings.filterwarnings('ignore')


TEST_FILE_PATTERN = 'wind_test_*.csv'
TIME_COL = '时间'
PATCHTST_MODEL_NAME = 'patchtst'
EXTERNAL_TEACHER_MODEL_NAME = 'tuned_patchtst_external_teacher'
RAMP_TRAJECTORY_MODEL_NAME = 'tuned_patchtst_ramp_trajectory'
RAMP_GATED_MODEL_NAME = 'tuned_patchtst_ramp_gated'
RAMP_PERSISTENCE_GATED_MODEL_NAME = (
    'tuned_patchtst_ramp_persistence_gated'
)
TUNED_DERIVED_MODEL_NAMES = {
    TUNED_MODEL_NAME,
    EXTERNAL_TEACHER_MODEL_NAME,
    RAMP_TRAJECTORY_MODEL_NAME,
    RAMP_GATED_MODEL_NAME,
    RAMP_PERSISTENCE_GATED_MODEL_NAME,
}
ALL_MODEL_NAMES = [
    PATCHTST_MODEL_NAME,
    TUNED_MODEL_NAME,
    EXTERNAL_TEACHER_MODEL_NAME,
    RAMP_TRAJECTORY_MODEL_NAME,
    RAMP_GATED_MODEL_NAME,
    RAMP_PERSISTENCE_GATED_MODEL_NAME,
] + OTHER_MODEL_NAMES
OUTPUT_SUBDIR = 'testdata_predict_output'
PRED_BATCH_SIZE = max(256, PATCHTST_BATCH_SIZE, OTHER_BATCH_SIZE)
EXP_WEIGHT_HALFLIFE_STEPS = 4.0
PREDICT_VERBOSE = int(os.getenv('WIND_DL_PREDICT_VERBOSE', '1'))
ALLOW_HISTORICAL_ROUND6_EQUAL_WEIGHT = os.getenv(
    'WIND_DL_ALLOW_HISTORICAL_ROUND6_EQUAL_WEIGHT',
    '0',
) == '1'


def discover_test_files(data_dir=DATA_DIR):
    return sorted(glob.glob(os.path.join(data_dir, TEST_FILE_PATTERN)))


def get_farm_id(path):
    basename = os.path.basename(path)
    match = re.search(r'wind_test_(\d+)\.csv$', basename)
    if match:
        return match.group(1)
    return os.path.splitext(basename)[0]


def get_requested_model_names():
    names = os.getenv('WIND_DL_MODEL_NAMES')
    if not names:
        return ALL_MODEL_NAMES

    requested = [name.strip().lower() for name in names.split(',') if name.strip()]
    if any(name in {'all', '*'} for name in requested):
        return ALL_MODEL_NAMES

    invalid = sorted(set(requested) - set(ALL_MODEL_NAMES))
    if invalid:
        raise ValueError(f'未知模型名称: {invalid}; 可选: {ALL_MODEL_NAMES}')
    return requested


def model_output_dirs(model_name):
    output_dir = os.path.join(BASE_RESULT_DIR, model_name, OUTPUT_SUBDIR)
    dirs = {
        'root': output_dir,
        'predictions': os.path.join(output_dir, 'predictions'),
        'figures': os.path.join(output_dir, 'figures'),
        'single_windows': os.path.join(output_dir, 'single_window_comparisons'),
        'weighted_curves': os.path.join(output_dir, 'weighted_curves'),
        'matplotlib_cache': os.path.join(output_dir, 'matplotlib_cache'),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def get_custom_objects():
    return {
        'PatchExtract': PatchExtract,
        'WindPatchTST>PatchExtract': PatchExtract,
        'MergeChannels': MergeChannels,
        'WindPatchTST>MergeChannels': MergeChannels,
        'RestoreChannels': RestoreChannels,
        'WindPatchTST>RestoreChannels': RestoreChannels,
        'LearnablePositionEmbedding': LearnablePositionEmbedding,
        'WindPatchTST>LearnablePositionEmbedding': LearnablePositionEmbedding,
        'TakeChannel': TakeChannel,
        'WindPatchTST>TakeChannel': TakeChannel,
        'RepeatLastTarget': RepeatLastTarget,
        'WindTunedPatchTST>RepeatLastTarget': RepeatLastTarget,
        'TunedPatchTSTLoss': TunedPatchTSTLoss,
        'WindTunedPatchTST>TunedPatchTSTLoss': TunedPatchTSTLoss,
        'BalancedTunedPatchTSTLoss': BalancedTunedPatchTSTLoss,
        'WindTunedPatchTST>BalancedTunedPatchTSTLoss': BalancedTunedPatchTSTLoss,
        'RMSEBalancedTunedPatchTSTLoss': RMSEBalancedTunedPatchTSTLoss,
        'WindTunedPatchTST>RMSEBalancedTunedPatchTSTLoss': (
            RMSEBalancedTunedPatchTSTLoss
        ),
        'PowerRevIN': PowerRevIN,
        'WindTunedPatchTST>PowerRevIN': PowerRevIN,
        'PowerRevINDenormalize': PowerRevINDenormalize,
        'WindTunedPatchTST>PowerRevINDenormalize': PowerRevINDenormalize,
        'SelectInputChannels': SelectInputChannels,
        'WindTunedPatchTST>SelectInputChannels': SelectInputChannels,
        'ZeroInitResidualAdapter': ZeroInitResidualAdapter,
        'WindTunedPatchTST>ZeroInitResidualAdapter': ZeroInitResidualAdapter,
        'TakeRecentTimesteps': TakeRecentTimesteps,
        'WindTunedPatchTST>TakeRecentTimesteps': TakeRecentTimesteps,
        'CumulativeRampForecast': CumulativeRampForecast,
        'WindTunedPatchTST>CumulativeRampForecast': (
            CumulativeRampForecast
        ),
        'HorizonExpertFusion': HorizonExpertFusion,
        'WindTunedPatchTST>HorizonExpertFusion': HorizonExpertFusion,
        'actual_mae': actual_mae,
        'WindTunedPatchTST>actual_mae': actual_mae,
        'actual_rmse': actual_rmse,
        'WindTunedPatchTST>actual_rmse': actual_rmse,
        'FixedPositionEmbedding': FixedPositionEmbedding,
        'WindInformer>FixedPositionEmbedding': FixedPositionEmbedding,
        'CircularTokenEmbedding': CircularTokenEmbedding,
        'WindInformer>CircularTokenEmbedding': CircularTokenEmbedding,
        'InformerDataEmbedding': InformerDataEmbedding,
        'WindInformer>InformerDataEmbedding': InformerDataEmbedding,
        'ProbSparseSelfAttention': ProbSparseSelfAttention,
        'WindInformer>ProbSparseSelfAttention': ProbSparseSelfAttention,
        'MovingAverage': MovingAverage,
        'WindAutoformer>MovingAverage': MovingAverage,
        'SeriesDecomposition': SeriesDecomposition,
        'WindAutoformer>SeriesDecomposition': SeriesDecomposition,
        'AutoformerLayerNorm': AutoformerLayerNorm,
        'WindAutoformer>AutoformerLayerNorm': AutoformerLayerNorm,
        'AutoformerDataEmbedding': AutoformerDataEmbedding,
        'WindAutoformer>AutoformerDataEmbedding': AutoformerDataEmbedding,
        'AutoformerDecoderInitializer': AutoformerDecoderInitializer,
        'WindAutoformer>AutoformerDecoderInitializer': AutoformerDecoderInitializer,
        'SeriesWiseAutoCorrelation': SeriesWiseAutoCorrelation,
        'WindAutoformer>SeriesWiseAutoCorrelation': SeriesWiseAutoCorrelation,
        'AutoformerEncoderLayer': AutoformerEncoderLayer,
        'WindAutoformer>AutoformerEncoderLayer': AutoformerEncoderLayer,
        'AutoformerDecoderLayer': AutoformerDecoderLayer,
        'WindAutoformer>AutoformerDecoderLayer': AutoformerDecoderLayer,
    }


def load_artifact(model_name, farm_id):
    if model_name == PATCHTST_MODEL_NAME:
        artifact_path = os.path.join(
            PATCHTST_MODEL_DIR,
            f'patchtst_farm_{farm_id}_preprocess.pkl',
        )
    else:
        artifact_path = os.path.join(
            BASE_RESULT_DIR,
            model_name,
            'preprocess',
            f'{model_name}_farm_{farm_id}_preprocess.pkl',
        )

    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f'未找到 {model_name} 场站 {farm_id} 的预处理文件: {artifact_path}')
    artifact = joblib.load(artifact_path)
    artifact['artifact_path'] = artifact_path
    return artifact


def get_tuned_ablation_trace_fields(model_name, artifact):
    if model_name not in TUNED_DERIVED_MODEL_NAMES:
        return {}
    multi_seed_values = artifact.get('multi_seed_values')
    if isinstance(multi_seed_values, (list, tuple)):
        multi_seed_values = ','.join(str(value) for value in multi_seed_values)
    return {
        'selected_ablation_variant': artifact.get(
            'selected_ablation_variant',
            artifact.get('ablation_variant', 'legacy_tuned_patchtst'),
        ),
        'selected_ablation_round': artifact.get(
            'selected_ablation_round',
            artifact.get('ablation_round'),
        ),
        'selected_parent_variant': artifact.get(
            'selected_parent_variant',
            artifact.get('parent_variant'),
        ),
        'selected_weight_source': artifact.get('selected_weight_source'),
        'use_revin': artifact.get('use_revin', False),
        'use_cnn_adapter': artifact.get('use_cnn_adapter', False),
        'use_ramp_expert': artifact.get('use_ramp_expert', False),
        'ramp_fusion_mode': artifact.get('ramp_fusion_mode', 'none'),
        'ramp_expert_context_len': artifact.get(
            'ramp_expert_context_len'
        ),
        'ramp_expert_filters': artifact.get('ramp_expert_filters'),
        'ramp_expert_dilations': artifact.get(
            'ramp_expert_dilations'
        ),
        'structural_ablation_step': artifact.get(
            'structural_ablation_step'
        ),
        'structural_parent_variant': artifact.get(
            'structural_parent_variant'
        ),
        'use_balanced_loss': artifact.get('use_balanced_loss', False),
        'use_rmse_balanced_loss': artifact.get(
            'use_rmse_balanced_loss',
            False,
        ),
        'use_swa': artifact.get('use_swa', False),
        'use_distillation': artifact.get('use_distillation', True),
        'use_supplementary_teacher_pretraining': artifact.get(
            'use_supplementary_teacher_pretraining',
            False,
        ),
        'multi_seed': artifact.get('multi_seed', False),
        'multi_seed_values': multi_seed_values,
        'use_seed_ensemble': artifact.get('use_seed_ensemble', False),
        'ensemble_member_count': artifact.get('ensemble_member_count', 1),
        'seed_nrmse_mean': artifact.get('seed_nrmse_mean'),
        'seed_nrmse_std': artifact.get('seed_nrmse_std'),
        'seed_composite_score_mean': artifact.get(
            'seed_composite_score_mean'
        ),
        'seed_composite_score_std': artifact.get(
            'seed_composite_score_std'
        ),
        'selection_val_composite_score': artifact.get('val_composite_score'),
        'selection_val_capacity_normalized_rmse': artifact.get(
            'val_capacity_normalized_rmse'
        ),
        'source_variant_model_path': artifact.get('source_variant_model_path'),
        'external_teacher_candidate_selected': artifact.get(
            'external_teacher_candidate_selected'
        ),
        'external_teacher_fallback': artifact.get(
            'external_teacher_fallback'
        ),
        'external_teacher_candidate_val_nrmse': artifact.get(
            'external_teacher_candidate_val_nrmse'
        ),
        'external_teacher_parent_val_nrmse': artifact.get(
            'external_teacher_parent_val_nrmse'
        ),
        'teacher_val_capacity_normalized_rmse': artifact.get(
            'teacher_val_capacity_normalized_rmse'
        ),
        'teacher_val_composite_score': artifact.get(
            'teacher_val_composite_score'
        ),
        'supplementary_station_count': artifact.get(
            'supplementary_station_count',
            0,
        ),
        'supplementary_selected_window_count': artifact.get(
            'supplementary_selected_window_count',
            0,
        ),
    }


def build_model_from_weights(model_name, artifact):
    if model_name == PATCHTST_MODEL_NAME:
        return build_patchtst_model(len(artifact['input_cols']), artifact['target_index'])
    if model_name in TUNED_DERIVED_MODEL_NAMES:
        return build_tuned_patchtst_model(
            len(artifact['input_cols']),
            artifact['target_index'],
            use_revin=artifact.get('use_revin', False),
            use_cnn_adapter=artifact.get('use_cnn_adapter', False),
            adapter_channel_indices=artifact.get('adapter_channel_indices'),
            use_ramp_expert=artifact.get('use_ramp_expert', False),
            ramp_fusion_mode=artifact.get('ramp_fusion_mode', 'none'),
        )

    input_shape = (artifact.get('history_len', HISTORY_LEN), len(artifact['input_cols']))
    builder = MODEL_BUILDERS[model_name]
    if model_name in {'informer', 'autoformer'}:
        return builder(input_shape, input_cols=artifact['input_cols'])
    return builder(input_shape)


def load_trained_model(model_name, farm_id, artifact):
    if (
        model_name in TUNED_DERIVED_MODEL_NAMES
        and artifact.get('use_seed_ensemble', False)
    ):
        ensemble_paths = artifact.get('ensemble_model_paths') or []
        missing_paths = [
            path for path in ensemble_paths
            if not os.path.exists(path)
        ]
        if not ensemble_paths:
            raise FileNotFoundError(
                f'tuned_patchtst 场站 {farm_id} 标记为seed集成，'
                '但artifact中没有ensemble_model_paths'
            )
        if missing_paths:
            raise FileNotFoundError(
                f'tuned_patchtst 场站 {farm_id} 缺少seed成员模型: {missing_paths}'
            )
        models = [
            keras.models.load_model(
                path,
                custom_objects=get_custom_objects(),
                compile=False,
            )
            for path in ensemble_paths
        ]
        return models, json.dumps(ensemble_paths, ensure_ascii=False)

    if model_name == PATCHTST_MODEL_NAME:
        model_path = artifact.get('model_path') or os.path.join(
            SAVED_MODEL_DIR,
            f'patchtst_farm_{farm_id}.keras',
        )
        best_weights_path = artifact.get('best_weights_path') or os.path.join(
            PATCHTST_MODEL_DIR,
            f'patchtst_farm_{farm_id}_best.weights.h5',
        )
    elif model_name in TUNED_DERIVED_MODEL_NAMES:
        if model_name == TUNED_MODEL_NAME:
            default_model_dir = TUNED_SAVED_MODEL_DIR
            default_weights_dir = TUNED_WEIGHTS_DIR
        else:
            default_model_dir = os.path.join('./models', model_name)
            default_weights_dir = os.path.join(
                BASE_RESULT_DIR,
                model_name,
                'weights',
            )
        model_path = artifact.get('model_path') or os.path.join(
            default_model_dir,
            f'{model_name}_farm_{farm_id}.keras',
        )
        best_weights_path = artifact.get('best_weights_path') or os.path.join(
            default_weights_dir,
            f'{model_name}_farm_{farm_id}_best.weights.h5',
        )
    else:
        model_path = artifact.get('model_path') or os.path.join(
            BASE_RESULT_DIR,
            model_name,
            'models',
            f'{model_name}_farm_{farm_id}.keras',
        )
        best_weights_path = artifact.get('best_weights_path') or os.path.join(
            BASE_RESULT_DIR,
            model_name,
            'weights',
            f'{model_name}_farm_{farm_id}_best.weights.h5',
        )

    if os.path.exists(model_path):
        model = keras.models.load_model(
            model_path,
            custom_objects=get_custom_objects(),
            compile=False,
        )
        return model, model_path

    if not os.path.exists(best_weights_path):
        raise FileNotFoundError(
            f'未找到 {model_name} 场站 {farm_id} 的完整模型或最佳权重: '
            f'{model_path}, {best_weights_path}')

    model = build_model_from_weights(model_name, artifact)
    model.load_weights(best_weights_path)
    return model, best_weights_path


def load_actual_power_series(data_path, index, capacity=None):
    raw_df = pd.read_csv(data_path, parse_dates=[TIME_COL])
    if TIME_COL not in raw_df.columns or TARGET_COL not in raw_df.columns:
        return pd.Series(np.nan, index=index, dtype=float)

    raw_df = raw_df[[TIME_COL, TARGET_COL]].sort_values(TIME_COL).drop_duplicates(TIME_COL)
    raw_df.set_index(TIME_COL, inplace=True)
    raw_df[TARGET_COL] = pd.to_numeric(raw_df[TARGET_COL], errors='coerce')

    actual = raw_df[TARGET_COL].reindex(index).astype(float)
    actual = actual.clip(lower=0)
    if capacity is not None:
        actual = actual.clip(upper=capacity)
    return actual


def prepare_prediction_arrays(test_file, artifact):
    df, _, file_capacity = load_and_preprocess(test_file, is_train=False)
    capacity = artifact.get('capacity') or file_capacity
    actual_power = load_actual_power_series(test_file, df.index, capacity)

    # Only already-observed historical power is used by model input windows.
    # Future real power stays outside inputs and is used only for metrics/plots.
    historical_power = actual_power.ffill().fillna(0)
    df[TARGET_COL] = historical_power.astype(np.float32)

    input_cols = artifact['input_cols']
    missing_cols = [col for col in input_cols if col not in df.columns]
    if missing_cols:
        for col in missing_cols:
            df[col] = 0.0
        print(f'警告：测试集缺少 {missing_cols}，已用0补齐')

    features = artifact['scaler_x'].transform(df[input_cols].values).astype(np.float32)
    return df, features, actual_power.values.astype(float), capacity


def make_prediction_dataset(features, history_len=HISTORY_LEN, forecast_len=FORECAST_LEN):
    n_samples = len(features) - history_len - forecast_len + 1
    if n_samples <= 0:
        raise ValueError('测试集长度不足，无法构造完整历史窗口和预测窗口')

    data_slice = features[:n_samples + history_len - 1]
    ds = keras.utils.timeseries_dataset_from_array(
        data=data_slice,
        targets=None,
        sequence_length=history_len,
        sequence_stride=1,
        shuffle=False,
        batch_size=PRED_BATCH_SIZE,
    )
    return ds.prefetch(tf.data.AUTOTUNE), n_samples


def inverse_power(scaler_y, values):
    values = np.asarray(values).reshape(-1, 1)
    return scaler_y.inverse_transform(values).reshape(-1)


def build_truth_windows(actual_power, n_samples,
                        history_len=HISTORY_LEN, forecast_len=FORECAST_LEN):
    rows = []
    for sample_idx in range(n_samples):
        start = sample_idx + history_len
        rows.append(actual_power[start:start + forecast_len])
    return np.asarray(rows, dtype=float)


def build_prediction_frame(model_name, df, farm_id, y_pred, y_true,
                           history_len=HISTORY_LEN, forecast_len=FORECAST_LEN):
    n_samples = y_pred.shape[0]
    origin_times = df.index[history_len - 1:history_len - 1 + n_samples]
    forecast_start_times = df.index[history_len:history_len + n_samples]
    rows = []

    for horizon_idx in range(forecast_len):
        target_times = df.index[
            history_len + horizon_idx:history_len + horizon_idx + n_samples]
        rows.append(pd.DataFrame({
            'model_name': model_name,
            'farm_id': farm_id,
            'sample_id': np.arange(n_samples),
            'forecast_origin_time': origin_times,
            'forecast_start_time': forecast_start_times,
            'target_time': target_times,
            'horizon_step': horizon_idx + 1,
            'horizon_minutes': (horizon_idx + 1) * 15,
            'pred_power': y_pred[:, horizon_idx],
            'actual_power': y_true[:, horizon_idx],
        }))

    pred_df = pd.concat(rows, ignore_index=True)
    pred_df['error'] = pred_df['pred_power'] - pred_df['actual_power']
    pred_df['abs_error'] = pred_df['error'].abs()
    pred_df['squared_error'] = pred_df['error'] ** 2
    pred_df['valid_actual'] = np.isfinite(pred_df['actual_power'])
    return pred_df


def calculate_metrics(y_true, y_pred, capacity=None):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        return {
            'valid_count': 0,
            'mae': np.nan,
            'mse': np.nan,
            'rmse': np.nan,
            'mape': np.nan,
            'smape': np.nan,
            'r2': np.nan,
            'capacity_normalized_mae': np.nan,
            'capacity_normalized_rmse': np.nan,
        }

    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]
    mae = mean_absolute_error(yt, yp)
    mse = mean_squared_error(yt, yp)
    rmse = float(np.sqrt(mse))

    nonzero_mask = np.abs(yt) > 1e-6
    mape = np.nan
    if nonzero_mask.any():
        mape = float(np.mean(
            np.abs((yt[nonzero_mask] - yp[nonzero_mask]) / yt[nonzero_mask])) * 100)

    denominator = np.abs(yt) + np.abs(yp)
    smape_mask = denominator > 1e-6
    smape = np.nan
    if smape_mask.any():
        smape = float(np.mean(
            2 * np.abs(yp[smape_mask] - yt[smape_mask]) / denominator[smape_mask]) * 100)

    r2 = np.nan
    if valid_count > 1 and np.nanstd(yt) > 1e-6:
        r2 = r2_score(yt, yp)

    norm_mae = np.nan
    norm_rmse = np.nan
    if capacity is not None and capacity > 0:
        norm_mae = float(mae / capacity)
        norm_rmse = float(rmse / capacity)

    return {
        'valid_count': valid_count,
        'mae': float(mae),
        'mse': float(mse),
        'rmse': rmse,
        'mape': mape,
        'smape': smape,
        'r2': r2,
        'capacity_normalized_mae': norm_mae,
        'capacity_normalized_rmse': norm_rmse,
    }


def metrics_by_horizon(model_name, farm_id, y_true, y_pred,
                       capacity=None, forecast_len=FORECAST_LEN):
    rows = []
    all_metrics = calculate_metrics(y_true, y_pred, capacity)
    all_metrics.update({
        'model_name': model_name,
        'farm_id': farm_id,
        'horizon_step': 'all',
        'horizon_minutes': 'all',
    })
    rows.append(all_metrics)

    for horizon_idx in range(forecast_len):
        metrics = calculate_metrics(y_true[:, horizon_idx], y_pred[:, horizon_idx], capacity)
        metrics.update({
            'model_name': model_name,
            'farm_id': farm_id,
            'horizon_step': horizon_idx + 1,
            'horizon_minutes': (horizon_idx + 1) * 15,
        })
        rows.append(metrics)

    return pd.DataFrame(rows)


def setup_matplotlib(dirs):
    cache_dir = dirs['matplotlib_cache']
    os.environ['MPLCONFIGDIR'] = cache_dir
    os.environ['XDG_CACHE_HOME'] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def save_single_window_plot(pred_df, model_name, farm_id, dirs, forecast_len=FORECAST_LEN):
    valid_df = pred_df[pred_df['valid_actual']].copy()
    if valid_df.empty:
        return None, None

    valid_counts = pred_df.groupby('sample_id')['valid_actual'].sum()
    complete_windows = valid_counts[valid_counts == forecast_len].index.to_numpy()
    if len(complete_windows) > 0:
        sample_id = complete_windows[len(complete_windows) // 2]
    else:
        partial_windows = valid_counts[valid_counts > 0].index.to_numpy()
        sample_id = partial_windows[len(partial_windows) // 2]

    window_df = pred_df[pred_df['sample_id'] == sample_id].sort_values('horizon_step').copy()
    window_path = os.path.join(
        dirs['single_windows'],
        f'{model_name}_single_4h_window_farm_{farm_id}.csv',
    )
    window_df.to_csv(window_path, index=False, encoding='utf-8-sig')

    figure_path = os.path.join(
        dirs['figures'],
        f'{model_name}_single_4h_window_farm_{farm_id}.png',
    )
    try:
        plt = setup_matplotlib(dirs)
        origin_time = window_df['forecast_origin_time'].iloc[0]
        start_time = window_df['forecast_start_time'].iloc[0]

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(window_df['target_time'], window_df['actual_power'],
                marker='o', label='Actual power', linewidth=1.6)
        ax.plot(window_df['target_time'], window_df['pred_power'],
                marker='s', label=f'{model_name} 4h window prediction', linewidth=1.4)
        ax.set_title(
            f'{model_name} Farm {farm_id} Single 4h Forecast Window\n'
            f'origin={origin_time}, forecast_start={start_time}'
        )
        ax.set_xlabel('Target time')
        ax.set_ylabel('Power')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(figure_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as exc:
        print(f'{model_name} 场站 {farm_id} 单窗口可视化保存失败: {exc}')
        figure_path = None

    return window_path, figure_path


def build_exponential_weighted_timeline(pred_df):
    valid_df = pred_df[pred_df['valid_actual']].copy()
    if valid_df.empty:
        return pd.DataFrame()

    valid_df['exp_weight'] = 0.5 ** (
        (valid_df['horizon_step'].astype(float) - 1.0) / EXP_WEIGHT_HALFLIFE_STEPS
    )
    valid_df['weighted_pred_power'] = valid_df['pred_power'] * valid_df['exp_weight']

    timeline = valid_df.groupby('target_time', as_index=False).agg(
        actual_power=('actual_power', 'mean'),
        weighted_pred_sum=('weighted_pred_power', 'sum'),
        total_weight=('exp_weight', 'sum'),
        prediction_count=('pred_power', 'size'),
        min_horizon_step=('horizon_step', 'min'),
        max_horizon_step=('horizon_step', 'max'),
    )
    timeline['pred_power'] = timeline['weighted_pred_sum'] / timeline['total_weight']
    timeline = timeline.sort_values('target_time')
    return timeline[[
        'target_time',
        'actual_power',
        'pred_power',
        'prediction_count',
        'total_weight',
        'min_horizon_step',
        'max_horizon_step',
    ]]


def save_weighted_full_test_plot(pred_df, model_name, farm_id, dirs, capacity=None):
    timeline = build_exponential_weighted_timeline(pred_df)
    if timeline.empty:
        return None, None, {}

    weighted_curve_path = os.path.join(
        dirs['weighted_curves'],
        f'{model_name}_weighted_curve_farm_{farm_id}.csv',
    )
    timeline.to_csv(weighted_curve_path, index=False, encoding='utf-8-sig')

    figure_path = os.path.join(
        dirs['figures'],
        f'{model_name}_weighted_full_test_farm_{farm_id}.png',
    )
    try:
        plt = setup_matplotlib(dirs)

        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(timeline['target_time'], timeline['actual_power'],
                label='Actual power', linewidth=1.6)
        ax.plot(timeline['target_time'], timeline['pred_power'],
                label=f'{model_name} exponential weighted prediction', linewidth=1.3)
        ax.set_title(
            f'{model_name} Farm {farm_id} Full Test Prediction vs Actual\n'
            f'exponential weight half-life={EXP_WEIGHT_HALFLIFE_STEPS:g} horizon steps'
        )
        ax.set_xlabel('Time')
        ax.set_ylabel('Power')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(figure_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as exc:
        print(f'{model_name} 场站 {farm_id} 全测试集指数加权可视化保存失败: {exc}')
        figure_path = None

    weighted_metrics = calculate_metrics(
        timeline['actual_power'].values,
        timeline['pred_power'].values,
        capacity,
    )
    return weighted_curve_path, figure_path, weighted_metrics


def predict_one_farm(model_name, test_file):
    farm_id = get_farm_id(test_file)
    dirs = model_output_dirs(model_name)
    print(f'\n===== 预测 {model_name} / 风电场 {farm_id} =====')

    artifact = load_artifact(model_name, farm_id)
    ablation_trace_fields = get_tuned_ablation_trace_fields(model_name, artifact)
    if ablation_trace_fields:
        print(
            '入选消融variant: '
            f"{ablation_trace_fields['selected_ablation_variant']}，"
            f"round={ablation_trace_fields['selected_ablation_round']}，"
            f"weight={ablation_trace_fields['selected_weight_source']}"
        )
    model, loaded_model_path = load_trained_model(model_name, farm_id, artifact)
    df, features, actual_power, capacity = prepare_prediction_arrays(test_file, artifact)
    history_len = artifact.get('history_len', HISTORY_LEN)
    forecast_len = artifact.get('forecast_len', FORECAST_LEN)

    pred_ds, n_samples = make_prediction_dataset(features, history_len, forecast_len)
    if isinstance(model, list):
        print(f'使用 {len(model)} 个seed成员模型进行均值集成')
        member_predictions = [
            member_model.predict(pred_ds, verbose=PREDICT_VERBOSE)
            for member_model in model
        ]
        y_pred_scaled = np.mean(
            np.stack(member_predictions, axis=0),
            axis=0,
        )
    else:
        y_pred_scaled = model.predict(pred_ds, verbose=PREDICT_VERBOSE)
    y_pred = inverse_power(artifact['scaler_y'], y_pred_scaled).reshape(-1, forecast_len)
    if y_pred.shape[0] != n_samples:
        raise ValueError(
            f'{model_name} 场站 {farm_id} 预测样本数不一致: {y_pred.shape[0]} vs {n_samples}')

    if capacity is not None:
        y_pred = np.clip(y_pred, 0, capacity)
    else:
        y_pred = np.clip(y_pred, 0, None)

    y_true = build_truth_windows(actual_power, n_samples, history_len, forecast_len)
    pred_df = build_prediction_frame(
        model_name,
        df,
        farm_id,
        y_pred,
        y_true,
        history_len,
        forecast_len,
    )

    pred_path = os.path.join(
        dirs['predictions'],
        f'{model_name}_predictions_farm_{farm_id}.csv',
    )
    pred_df.to_csv(pred_path, index=False, encoding='utf-8-sig')

    metric_df = metrics_by_horizon(model_name, farm_id, y_true, y_pred, capacity, forecast_len)
    for field, value in ablation_trace_fields.items():
        metric_df[field] = value
    horizon_metric_path = os.path.join(
        dirs['root'],
        f'{model_name}_metrics_by_horizon_farm_{farm_id}.csv',
    )
    metric_df.to_csv(horizon_metric_path, index=False, encoding='utf-8-sig')

    single_window_path, single_window_figure_path = save_single_window_plot(
        pred_df,
        model_name,
        farm_id,
        dirs,
        forecast_len,
    )
    weighted_curve_path, weighted_curve_figure_path, weighted_metrics = save_weighted_full_test_plot(
        pred_df,
        model_name,
        farm_id,
        dirs,
        capacity,
    )

    all_metrics = metric_df[metric_df['horizon_step'] == 'all'].iloc[0].to_dict()
    weighted_metric_fields = {
        f'weighted_curve_{key}': value
        for key, value in weighted_metrics.items()
    }
    all_metrics.update({
        'loaded_model_path': loaded_model_path,
        'artifact_path': artifact['artifact_path'],
        'prediction_path': pred_path,
        'horizon_metric_path': horizon_metric_path,
        'single_window_path': single_window_path,
        'single_window_figure_path': single_window_figure_path,
        'weighted_curve_path': weighted_curve_path,
        'weighted_curve_figure_path': weighted_curve_figure_path,
        **ablation_trace_fields,
        **weighted_metric_fields,
    })
    print(f"{model_name} 场站 {farm_id}: MAE={all_metrics['mae']:.4f}, "
          f"RMSE={all_metrics['rmse']:.4f}")

    if isinstance(model, list):
        for member_model in model:
            del member_model
    del model
    keras.backend.clear_session()
    return all_metrics, metric_df


def historical_round6_requires_saved_predictions(model_name, test_files):
    if (
        model_name != TUNED_MODEL_NAME
        or ALLOW_HISTORICAL_ROUND6_EQUAL_WEIGHT
    ):
        return False

    incompatible_farms = []
    for test_file in test_files:
        farm_id = get_farm_id(test_file)
        try:
            artifact = load_artifact(model_name, farm_id)
        except FileNotFoundError:
            continue
        selected_round = artifact.get(
            'selected_ablation_round',
            artifact.get('ablation_round'),
        )
        try:
            selected_round = int(selected_round)
        except (TypeError, ValueError):
            selected_round = None
        raw_weights = artifact.get('ensemble_weights')
        weights = np.asarray(
            [] if raw_weights is None else raw_weights,
            dtype=float,
        )
        if (
            selected_round == 6
            and len(weights) > 1
            and not np.allclose(weights, np.mean(weights))
        ):
            incompatible_farms.append(farm_id)

    if incompatible_farms:
        print(
            '跳过 tuned_patchtst 整个模型族：当前canonical artifact包含'
            f'第六轮非均匀权重（场站 {incompatible_farms}），而活动预测代码'
            '只支持等权。请直接使用已保存的第六轮CSV；如确需覆盖式等权重跑，'
            '显式设置 WIND_DL_ALLOW_HISTORICAL_ROUND6_EQUAL_WEIGHT=1。'
        )
        return True
    return False


def predict_model_family(model_name, test_files):
    if historical_round6_requires_saved_predictions(model_name, test_files):
        return pd.DataFrame(), pd.DataFrame()

    dirs = model_output_dirs(model_name)
    summary_rows = []
    horizon_metric_frames = []

    for test_file in test_files:
        try:
            metrics, horizon_metrics = predict_one_farm(model_name, test_file)
        except FileNotFoundError as exc:
            print(f'跳过 {model_name} {os.path.basename(test_file)}: {exc}')
            continue
        summary_rows.append(metrics)
        horizon_metric_frames.append(horizon_metrics)

    if not summary_rows:
        print(f'{model_name} 没有生成预测结果')
        return pd.DataFrame(), pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(dirs['root'], f'{model_name}_test_metrics_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

    all_horizon_metrics = pd.concat(horizon_metric_frames, ignore_index=True)
    all_horizon_path = os.path.join(dirs['root'], f'{model_name}_test_metrics_by_horizon_all.csv')
    all_horizon_metrics.to_csv(all_horizon_path, index=False, encoding='utf-8-sig')

    print(f'{model_name} 汇总指标已保存: {summary_path}')
    print(f'{model_name} 分horizon指标已保存: {all_horizon_path}')
    return summary_df, all_horizon_metrics


def main():
    set_global_seed(seed)

    test_files = discover_test_files(DATA_DIR)
    if not test_files:
        raise FileNotFoundError(f'未在 {DATA_DIR} 找到 {TEST_FILE_PATTERN}')

    requested_model_names = get_requested_model_names()
    print(f'发现 {len(test_files)} 个风电测试文件')
    print(f'将预测模型: {requested_model_names}')

    all_summary = []
    all_horizon = []
    for model_name in requested_model_names:
        summary_df, horizon_df = predict_model_family(model_name, test_files)
        if not summary_df.empty:
            all_summary.append(summary_df)
        if not horizon_df.empty:
            all_horizon.append(horizon_df)

    if all_summary:
        global_summary = pd.concat(all_summary, ignore_index=True)
        summary_filename = (
            'wind_dl_all_models_test_metrics_summary.csv'
            if requested_model_names == ALL_MODEL_NAMES
            else 'wind_dl_selected_models_test_metrics_summary.csv'
        )
        global_summary_path = os.path.join(
            BASE_RESULT_DIR,
            summary_filename,
        )
        global_summary.to_csv(global_summary_path, index=False, encoding='utf-8-sig')
        print(f'模型汇总指标已保存: {global_summary_path}')

    if all_horizon:
        global_horizon = pd.concat(all_horizon, ignore_index=True)
        horizon_filename = (
            'wind_dl_all_models_test_metrics_by_horizon_all.csv'
            if requested_model_names == ALL_MODEL_NAMES
            else 'wind_dl_selected_models_test_metrics_by_horizon_all.csv'
        )
        global_horizon_path = os.path.join(
            BASE_RESULT_DIR,
            horizon_filename,
        )
        global_horizon.to_csv(global_horizon_path, index=False, encoding='utf-8-sig')
        print(f'模型分horizon指标已保存: {global_horizon_path}')

    print('全部深度学习模型测试集预测完成')


if __name__ == '__main__':
    main()
