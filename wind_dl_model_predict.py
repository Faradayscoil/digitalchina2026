import glob
import hashlib
import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras

from wind_FeTS_PatchTST_train import (
    ARCHITECTURE_VERSION as FETS_PATCHTST_ARCHITECTURE_VERSION,
    BATCH_SIZE as FETS_PATCHTST_BATCH_SIZE,
    PREPROCESS_DIR as FETS_PATCHTST_PREPROCESS_DIR,
    SAVED_MODEL_DIR as FETS_PATCHTST_SAVED_MODEL_DIR,
    WEIGHTS_DIR as FETS_PATCHTST_WEIGHTS_DIR,
    EXPERT_NAMES as FETS_PATCHTST_EXPERT_NAMES,
    AdaptiveFeatureExtraction,
    ChannelIdentityEmbedding,
    DualScaleFeedForward,
    ExpertConvexFusion,
    FeTSChannelPatchTranspose,
    FeTSFeatureBlock,
    FeTSPatchExtract,
    FourierPolynomialMask,
    HorizonRegimeRouter,
    HorizonScaledResidualAdd,
    LayerScaleFeTSFeatureBlock,
    PatchCrossChannelAttention,
    PersistenceForecast,
    SelectChannel,
    TakeLastToken,
    TargetWeatherCrossAttention,
    build_fets_patchtst_model,
)
from wind_dl_model_train import (
    BATCH_SIZE as PATCHTST_BATCH_SIZE,
    D_FF as PATCHTST_D_FF,
    D_MODEL as PATCHTST_D_MODEL,
    DATA_DIR,
    DROPOUT as PATCHTST_DROPOUT,
    FORECAST_LEN,
    HEAD_DROPOUT as PATCHTST_HEAD_DROPOUT,
    HISTORY_LEN,
    MODEL_DIR as PATCHTST_RESULT_DIR,
    N_HEADS as PATCHTST_N_HEADS,
    N_LAYERS as PATCHTST_N_LAYERS,
    PATCH_LEN as PATCHTST_PATCH_LEN,
    PATCH_STRIDE as PATCHTST_PATCH_STRIDE,
    PREPROCESS_DIR as PATCHTST_PREPROCESS_DIR,
    SAVED_MODEL_DIR as PATCHTST_SAVED_MODEL_DIR,
    TARGET_COL,
    WEIGHTS_DIR as PATCHTST_WEIGHTS_DIR,
    LearnablePositionEmbedding,
    MergeChannels,
    PatchExtract,
    RestoreChannels,
    TakeChannel,
    build_patchtst_model,
    compute_patch_num as compute_patchtst_patch_num,
    load_and_preprocess,
    transformer_encoder as patchtst_transformer_encoder,
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
from wind_RegimeEncoder_PatchTST_feature_screen_train import (
    build_feature_screen_model,
    get_feature_screen_custom_objects,
)

warnings.filterwarnings('ignore')


TEST_FILE_PATTERN = 'wind_test_*.csv'
TIME_COL = '时间'
PATCHTST_MODEL_NAME = 'patchtst'
FETS_PATCHTST_MODEL_NAME = 'fets_patchtst'
PART3_ROUND2_STRONG_BASELINE_MODEL_NAME = (
    'part3_round2_f7_g0_strong_baseline'
)
PART3_ROUND2_STRONG_BASELINE_VARIANT_ID = 'sb_f7_g0_bs256'
PART3_ROUND2_STRONG_BASELINE_EXPERT_NAMES = (
    'persistence',
    'corrected',
)
PART3_ROUND2_STRONG_BASELINE_RESULT_ROOT = os.path.join(
    './wind_results',
    'part3_new_module_supplement',
    '02_strong_baseline_f7_g0_fair_training',
)
PATCHTST_LEGACY_ROUND2_DIR = os.path.join(PATCHTST_RESULT_DIR, '第2轮训练结果')
ALL_MODEL_NAMES = [
    PATCHTST_MODEL_NAME,
    FETS_PATCHTST_MODEL_NAME,
] + OTHER_MODEL_NAMES + [PART3_ROUND2_STRONG_BASELINE_MODEL_NAME]
LEGACY_STRONG_COMPARISON_MODEL_NAMES = tuple(
    [PATCHTST_MODEL_NAME, FETS_PATCHTST_MODEL_NAME] + OTHER_MODEL_NAMES
)
STRONG_COMPARISON_EXPECTED_FARM_COUNT = 5
OUTPUT_SUBDIR = 'testdata_predict_output'
PRED_BATCH_SIZE = max(
    256,
    PATCHTST_BATCH_SIZE,
    FETS_PATCHTST_BATCH_SIZE,
    OTHER_BATCH_SIZE,
)
EXP_WEIGHT_HALFLIFE_STEPS = 4.0
PREDICT_VERBOSE = int(os.getenv('WIND_DL_PREDICT_VERBOSE', '1'))


@keras.utils.register_keras_serializable(package='WindPatchTST')
class RepeatLastTarget(keras.layers.Layer):
    def __init__(self, target_channel_index, forecast_len=FORECAST_LEN, **kwargs):
        super().__init__(**kwargs)
        self.target_channel_index = int(target_channel_index)
        self.forecast_len = int(forecast_len)

    def call(self, inputs):
        last_value = inputs[
            :,
            -1,
            self.target_channel_index:self.target_channel_index + 1,
        ]
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


@keras.utils.register_keras_serializable(package='WindPatchTST')
class HorizonGatedForecast(keras.layers.Layer):
    def __init__(
        self,
        forecast_len=FORECAST_LEN,
        init_near=2.0,
        init_far=-2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.forecast_len = int(forecast_len)
        self.init_near = float(init_near)
        self.init_far = float(init_far)

    def build(self, input_shape):
        init_values = np.linspace(
            self.init_near,
            self.init_far,
            self.forecast_len,
            dtype=np.float32,
        )
        self.gate_logits = self.add_weight(
            name='horizon_gate_logits',
            shape=(self.forecast_len,),
            initializer=keras.initializers.Constant(init_values),
            trainable=True,
        )

    def call(self, inputs):
        baseline, direct_forecast, residual = inputs
        gate = tf.sigmoid(
            tf.cast(self.gate_logits, baseline.dtype),
        )[tf.newaxis, :]
        return gate * baseline + (1.0 - gate) * direct_forecast + residual

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.forecast_len

    def get_config(self):
        config = super().get_config()
        config.update({
            'forecast_len': self.forecast_len,
            'init_near': self.init_near,
            'init_far': self.init_far,
        })
        return config


def build_legacy_patchtst_round2_model(input_dim, target_channel_index, artifact):
    """重建 wind_results/patchtst/第2轮训练结果 对应的原生 PatchTST。"""
    if target_channel_index is None:
        raise ValueError('PatchTST 短期风电模型需要将历史功率作为输入通道')

    history_len = int(artifact.get('history_len', HISTORY_LEN))
    forecast_len = int(artifact.get('forecast_len', FORECAST_LEN))
    patch_len = int(artifact.get('patch_len', PATCHTST_PATCH_LEN))
    patch_stride = int(artifact.get('patch_stride', PATCHTST_PATCH_STRIDE))
    cnn_stem_dropout = float(artifact.get('cnn_stem_dropout', 0.05))

    patch_num = compute_patchtst_patch_num(
        history_len,
        patch_len,
        patch_stride,
    )
    inputs = keras.Input(
        shape=(history_len, input_dim),
        name='history_features',
    )

    cnn_stem = keras.layers.Conv1D(
        input_dim,
        kernel_size=3,
        padding='same',
        activation='gelu',
        kernel_regularizer=keras.regularizers.l2(1e-5),
        name='local_cnn_stem',
    )(inputs)
    cnn_stem = keras.layers.Dropout(
        cnn_stem_dropout,
        name='local_cnn_stem_dropout',
    )(cnn_stem)
    x_input = keras.layers.Add(name='history_plus_local_stem')(
        [inputs, cnn_stem],
    )

    local_context = keras.layers.Conv1D(
        PATCHTST_D_MODEL,
        kernel_size=3,
        padding='same',
        activation='gelu',
        kernel_regularizer=keras.regularizers.l2(1e-5),
        name='local_context_conv3',
    )(x_input)
    local_context = keras.layers.Conv1D(
        PATCHTST_D_MODEL,
        kernel_size=5,
        padding='same',
        activation='gelu',
        kernel_regularizer=keras.regularizers.l2(1e-5),
        name='local_context_conv5',
    )(local_context)
    local_context = keras.layers.GlobalAveragePooling1D(
        name='local_context_pool',
    )(local_context)

    x = PatchExtract(
        patch_len,
        patch_stride,
        name='patch_extract',
    )(x_input)
    x = keras.layers.Dense(
        PATCHTST_D_MODEL,
        name='patch_projection',
    )(x)
    x = MergeChannels(name='merge_channels')(x)
    x = LearnablePositionEmbedding(
        patch_num,
        PATCHTST_D_MODEL,
        name='position_embedding',
    )(x)
    x = keras.layers.Dropout(
        PATCHTST_DROPOUT,
        name='patch_dropout',
    )(x)

    for idx in range(PATCHTST_N_LAYERS):
        x = patchtst_transformer_encoder(
            x,
            PATCHTST_D_MODEL,
            PATCHTST_N_HEADS,
            PATCHTST_D_FF,
            PATCHTST_DROPOUT,
            name=f'encoder_{idx + 1}',
        )

    x = RestoreChannels(
        input_dim,
        patch_num,
        PATCHTST_D_MODEL,
        name='restore_channels',
    )(x)
    target_repr = TakeChannel(
        target_channel_index,
        name='target_power_channel',
    )(x)
    target_repr = keras.layers.Flatten(name='target_flatten')(target_repr)
    global_context = keras.layers.GlobalAveragePooling2D(
        name='channel_context_pool',
    )(x)

    head = keras.layers.Concatenate(name='forecast_context')(
        [target_repr, global_context, local_context],
    )
    head = keras.layers.Dropout(
        PATCHTST_HEAD_DROPOUT,
        name='head_dropout',
    )(head)
    head = keras.layers.Dense(
        PATCHTST_D_FF,
        activation='gelu',
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name='forecast_ff',
    )(head)
    head = keras.layers.Dropout(
        PATCHTST_HEAD_DROPOUT,
        name='forecast_dropout',
    )(head)
    direct_forecast = keras.layers.Dense(
        forecast_len,
        name='direct_forecast',
    )(head)
    residual = keras.layers.Dense(
        forecast_len,
        kernel_initializer='zeros',
        bias_initializer='zeros',
        name='forecast_residual',
    )(head)
    baseline = RepeatLastTarget(
        target_channel_index,
        forecast_len,
        name='persistence_baseline',
    )(inputs)
    outputs = HorizonGatedForecast(
        forecast_len,
        name='forecast_power',
    )([baseline, direct_forecast, residual])

    return keras.Model(inputs=inputs, outputs=outputs, name='WindPatchTST')


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
    if model_name == PART3_ROUND2_STRONG_BASELINE_MODEL_NAME:
        output_dir = os.path.join(
            PART3_ROUND2_STRONG_BASELINE_RESULT_ROOT,
            OUTPUT_SUBDIR,
        )
    else:
        output_dir = os.path.join(BASE_RESULT_DIR, model_name, OUTPUT_SUBDIR)
    dirs = {
        'root': output_dir,
        'predictions': os.path.join(output_dir, 'predictions'),
        'figures': os.path.join(output_dir, 'figures'),
        'single_windows': os.path.join(output_dir, 'single_window_comparisons'),
        'weighted_curves': os.path.join(output_dir, 'weighted_curves'),
        'router_diagnostics': os.path.join(output_dir, 'router_diagnostics'),
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
        'WindPatchTST>RepeatLastTarget': RepeatLastTarget,
        'HorizonGatedForecast': HorizonGatedForecast,
        'WindPatchTST>HorizonGatedForecast': HorizonGatedForecast,
        'FeTSPatchExtract': FeTSPatchExtract,
        'WindFeTSPatchTST>FeTSPatchExtract': FeTSPatchExtract,
        'FourierPolynomialMask': FourierPolynomialMask,
        'WindFeTSPatchTST>FourierPolynomialMask': FourierPolynomialMask,
        'AdaptiveFeatureExtraction': AdaptiveFeatureExtraction,
        'WindFeTSPatchTST>AdaptiveFeatureExtraction': AdaptiveFeatureExtraction,
        'DualScaleFeedForward': DualScaleFeedForward,
        'WindFeTSPatchTST>DualScaleFeedForward': DualScaleFeedForward,
        'FeTSFeatureBlock': FeTSFeatureBlock,
        'WindFeTSPatchTST>FeTSFeatureBlock': FeTSFeatureBlock,
        'ChannelIdentityEmbedding': ChannelIdentityEmbedding,
        'WindFeTSPatchTST>ChannelIdentityEmbedding': ChannelIdentityEmbedding,
        'LayerScaleFeTSFeatureBlock': LayerScaleFeTSFeatureBlock,
        'WindFeTSPatchTST>LayerScaleFeTSFeatureBlock': LayerScaleFeTSFeatureBlock,
        'HorizonScaledResidualAdd': HorizonScaledResidualAdd,
        'WindFeTSPatchTST>HorizonScaledResidualAdd': HorizonScaledResidualAdd,
        'PersistenceForecast': PersistenceForecast,
        'WindFeTSPatchTST>PersistenceForecast': PersistenceForecast,
        'HorizonRegimeRouter': HorizonRegimeRouter,
        'WindFeTSPatchTST>HorizonRegimeRouter': HorizonRegimeRouter,
        'ExpertConvexFusion': ExpertConvexFusion,
        'WindFeTSPatchTST>ExpertConvexFusion': ExpertConvexFusion,
        'FeTSChannelPatchTranspose': FeTSChannelPatchTranspose,
        'WindFeTSPatchTST>FeTSChannelPatchTranspose': FeTSChannelPatchTranspose,
        'PatchCrossChannelAttention': PatchCrossChannelAttention,
        'WindFeTSPatchTST>PatchCrossChannelAttention': PatchCrossChannelAttention,
        'TargetWeatherCrossAttention': TargetWeatherCrossAttention,
        'WindFeTSPatchTST>TargetWeatherCrossAttention': TargetWeatherCrossAttention,
        'TakeLastToken': TakeLastToken,
        'WindFeTSPatchTST>TakeLastToken': TakeLastToken,
        'SelectChannel': SelectChannel,
        'WindFeTSPatchTST>SelectChannel': SelectChannel,
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


def get_part3_round2_strong_baseline_custom_objects():
    """Return the F7/G0 objects without changing legacy model deserialization."""
    return get_feature_screen_custom_objects()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_part3_round2_strong_baseline_artifact(artifact, artifact_path):
    """Reject protocol drift before adding the retrained F7/G0 to baselines."""
    required = (
        'model_name',
        'variant_id',
        'farm_id',
        'training_mode',
        'initialization_mode',
        'warm_start',
        'loaded_pretrained_weights',
        'batch_size',
        'input_cols',
        'target_index',
        'scaler_x',
        'scaler_y',
        'history_len',
        'forecast_len',
        'power_scale_ratio',
        'power_scale_offset',
        'regime_feature_config',
        'model_path',
        'best_weights_path',
        'diagnostic_layers',
        'expert_names',
        'primary_prediction_output',
        'epochs',
        'validation_split',
        'learning_rate',
        'optimizer',
        'clipnorm',
        'candidate_supervision_loss_weight',
        'early_stopping_monitor',
        'checkpoint_monitor',
        'early_stopping_patience',
        'reduce_lr_patience',
        'reduce_lr_factor',
        'minimum_learning_rate',
        'model_sha256',
        'best_weights_sha256',
    )
    missing = [key for key in required if key not in artifact]
    if missing:
        raise KeyError(
            f'第三部分第二轮 artifact缺少字段 {missing}: {artifact_path}'
        )
    if artifact['model_name'] != PART3_ROUND2_STRONG_BASELINE_MODEL_NAME:
        raise ValueError(
            f'第三部分第二轮 model_name不匹配: '
            f"{artifact['model_name']}"
        )
    if artifact['variant_id'] != PART3_ROUND2_STRONG_BASELINE_VARIANT_ID:
        raise ValueError(
            f'第三部分第二轮 variant_id不匹配: '
            f"{artifact['variant_id']}"
        )
    if artifact['training_mode'] != 'single_stage_from_scratch_fair_protocol':
        raise ValueError(
            '强基线必须由单阶段from-scratch协议产生，'
            f"实际为 {artifact['training_mode']}"
        )
    if (
        artifact['initialization_mode'] != 'from_scratch'
        or bool(artifact['warm_start'])
        or bool(artifact['loaded_pretrained_weights'])
    ):
        raise ValueError(
            '强基线不允许旧B2/F7权重初始化: '
            f"initialization={artifact['initialization_mode']}, "
            f"warm_start={artifact['warm_start']}, "
            f"loaded_pretrained_weights={artifact['loaded_pretrained_weights']}"
        )
    if int(artifact['batch_size']) != 256:
        raise ValueError(
            '强基线协议要求 batch_size=256，'
            f"实际为 {artifact['batch_size']}"
        )
    if int(artifact['history_len']) != HISTORY_LEN:
        raise ValueError(
            f'强基线历史长度必须为{HISTORY_LEN}: '
            f"{artifact['history_len']}"
        )
    if int(artifact['forecast_len']) != FORECAST_LEN:
        raise ValueError(
            f'强基线预测长度必须为{FORECAST_LEN}: '
            f"{artifact['forecast_len']}"
        )
    exact_protocol = {
        'epochs': 80,
        'early_stopping_patience': 10,
        'reduce_lr_patience': 4,
    }
    for key, expected in exact_protocol.items():
        if int(artifact[key]) != expected:
            raise ValueError(
                f'强基线协议 {key}漂移: '
                f"{artifact[key]} != {expected}"
            )
    float_protocol = {
        'validation_split': 0.15,
        'learning_rate': 5e-4,
        'clipnorm': 1.0,
        'candidate_supervision_loss_weight': 0.5,
        'reduce_lr_factor': 0.5,
        'minimum_learning_rate': 1e-6,
    }
    for key, expected in float_protocol.items():
        if not np.isclose(
            float(artifact[key]),
            expected,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f'强基线协议 {key}漂移: '
                f"{artifact[key]} != {expected}"
            )
    if artifact['optimizer'] != 'Adam':
        raise ValueError(f"强基线optimizer不是Adam: {artifact['optimizer']}")
    for monitor_key in ('early_stopping_monitor', 'checkpoint_monitor'):
        if artifact[monitor_key] != 'val_forecast_power_loss':
            raise ValueError(
                f'强基线 {monitor_key}不是主预测损失: '
                f"{artifact[monitor_key]}"
            )
    if artifact['primary_prediction_output'] != 'forecast_power':
        raise ValueError(
            '强基线主预测输出必须为forecast_power'
        )
    if tuple(artifact['expert_names']) != (
        PART3_ROUND2_STRONG_BASELINE_EXPERT_NAMES
    ):
        raise ValueError(
            '强基线两候选名称必须依次为persistence/corrected: '
            f"{artifact['expert_names']}"
        )
    expected_root = os.path.abspath(PART3_ROUND2_STRONG_BASELINE_RESULT_ROOT)
    for path_key in ('model_path', 'best_weights_path'):
        recorded_path = os.path.abspath(os.fspath(artifact[path_key]))
        if os.path.commonpath((expected_root, recorded_path)) != expected_root:
            raise ValueError(
                f'强基线 {path_key}越出第三部分第二轮目录: '
                f'{recorded_path}'
            )
    diagnostics = artifact['diagnostic_layers']
    expected_diagnostics = {
        'forecast': 'forecast_power',
        'gate': 'correction_gate',
        'persistence_candidate': 'persistence_forecast_candidate',
        'corrected_candidate': 'corrected_forecast_candidate',
    }
    missing_diagnostics = sorted(set(expected_diagnostics) - set(diagnostics))
    if missing_diagnostics:
        raise KeyError(
            '第三部分第二轮 diagnostic_layers缺少 '
            f'{missing_diagnostics}: {artifact_path}'
        )
    mismatched_diagnostics = {
        key: diagnostics.get(key)
        for key, expected in expected_diagnostics.items()
        if diagnostics.get(key) != expected
    }
    if mismatched_diagnostics:
        raise ValueError(
            '强基线F7/G0诊断层语义漂移: '
            f'{mismatched_diagnostics}'
        )


def load_artifact(model_name, farm_id):
    if model_name == PATCHTST_MODEL_NAME:
        artifact_candidates = [
            (
                os.path.join(
                    PATCHTST_PREPROCESS_DIR,
                    f'patchtst_farm_{farm_id}_preprocess.pkl',
                ),
                'standard',
            ),
            (
                os.path.join(
                    PATCHTST_LEGACY_ROUND2_DIR,
                    f'patchtst_farm_{farm_id}_preprocess.pkl',
                ),
                'legacy_round2',
            ),
            (
                os.path.join(
                    PATCHTST_RESULT_DIR,
                    f'patchtst_farm_{farm_id}_preprocess.pkl',
                ),
                'legacy_root',
            ),
        ]
        artifact_path = None
        patchtst_layout = None
        for candidate_path, candidate_layout in artifact_candidates:
            if os.path.exists(candidate_path):
                artifact_path = candidate_path
                patchtst_layout = candidate_layout
                break
    elif model_name == FETS_PATCHTST_MODEL_NAME:
        artifact_path = os.path.join(
            FETS_PATCHTST_PREPROCESS_DIR,
            f'{FETS_PATCHTST_MODEL_NAME}_farm_{farm_id}_preprocess.pkl',
        )
    elif model_name == PART3_ROUND2_STRONG_BASELINE_MODEL_NAME:
        artifact_path = os.path.join(
            PART3_ROUND2_STRONG_BASELINE_RESULT_ROOT,
            'preprocess',
            f'{model_name}_farm_{farm_id}_preprocess.pkl',
        )
        patchtst_layout = None
    else:
        artifact_path = os.path.join(
            BASE_RESULT_DIR,
            model_name,
            'preprocess',
            f'{model_name}_farm_{farm_id}_preprocess.pkl',
        )
        patchtst_layout = None

    if not artifact_path or not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f'未找到 {model_name} 场站 {farm_id} 的预处理文件: {artifact_path}')
    artifact = joblib.load(artifact_path)
    if model_name == PART3_ROUND2_STRONG_BASELINE_MODEL_NAME:
        validate_part3_round2_strong_baseline_artifact(
            artifact,
            artifact_path,
        )
        if str(artifact['farm_id']) != str(farm_id):
            raise ValueError(
                '强基线artifact场站身份与测试文件不一致: '
                f"{artifact['farm_id']} != {farm_id}; {artifact_path}"
            )
    if model_name == PATCHTST_MODEL_NAME:
        artifact['_patchtst_artifact_layout'] = patchtst_layout
        if patchtst_layout == 'standard':
            artifact['model_path'] = artifact.get('model_path') or os.path.join(
                PATCHTST_SAVED_MODEL_DIR,
                f'patchtst_farm_{farm_id}.keras',
            )
            artifact['best_weights_path'] = artifact.get('best_weights_path') or os.path.join(
                PATCHTST_WEIGHTS_DIR,
                f'patchtst_farm_{farm_id}_best.weights.h5',
            )
        elif patchtst_layout == 'legacy_round2':
            artifact['model_path'] = os.path.join(
                PATCHTST_LEGACY_ROUND2_DIR,
                f'patchtst_farm_{farm_id}.keras',
            )
            artifact['best_weights_path'] = os.path.join(
                PATCHTST_LEGACY_ROUND2_DIR,
                f'patchtst_farm_{farm_id}_best.weights.h5',
            )
        else:
            artifact['model_path'] = artifact.get('model_path') or os.path.join(
                PATCHTST_SAVED_MODEL_DIR,
                f'patchtst_farm_{farm_id}.keras',
            )
            artifact['best_weights_path'] = artifact.get('best_weights_path') or os.path.join(
                PATCHTST_WEIGHTS_DIR,
                f'patchtst_farm_{farm_id}_best.weights.h5',
            )
    if (
        model_name == FETS_PATCHTST_MODEL_NAME
        and artifact.get('architecture_version')
        != FETS_PATCHTST_ARCHITECTURE_VERSION
    ):
        raise FileNotFoundError(
            f'场站 {farm_id} 的 FeTS-PatchTST artifact 结构版本为 '
            f"{artifact.get('architecture_version', 'unknown')}，"
            f'当前预测代码要求 {FETS_PATCHTST_ARCHITECTURE_VERSION}；'
            '请使用当前训练脚本重新训练后再预测'
        )
    artifact['artifact_path'] = artifact_path
    return artifact


def build_model_from_weights(model_name, artifact):
    if model_name == PATCHTST_MODEL_NAME:
        if (
            artifact.get('_patchtst_artifact_layout') == 'legacy_round2'
            or 'cnn_stem_dropout' in artifact
            or 'horizon_decay' in artifact
        ):
            return build_legacy_patchtst_round2_model(
                len(artifact['input_cols']),
                artifact['target_index'],
                artifact,
            )
        return build_patchtst_model(len(artifact['input_cols']), artifact['target_index'])
    if model_name == FETS_PATCHTST_MODEL_NAME:
        return build_fets_patchtst_model(
            len(artifact['input_cols']),
            artifact['target_index'],
            history_len=artifact.get('history_len', HISTORY_LEN),
            forecast_len=artifact.get('forecast_len', FORECAST_LEN),
            patch_len=artifact.get('patch_len', 16),
            patch_stride=artifact.get('patch_stride', 8),
            d_model=artifact.get('d_model', 64),
            dropout=artifact.get('dropout', 0.15),
            head_dropout=artifact.get('head_dropout', 0.2),
            fourier_degree=artifact.get('fourier_degree', 2),
            poly_degree=artifact.get('poly_degree', 2),
            ffn_ratio=artifact.get('ffn_ratio', 2),
            n_heads=artifact.get('n_heads', 4),
            n_layers=artifact.get('n_layers', 3),
            d_ff=artifact.get('d_ff', 128),
            mid_patch_len=artifact.get('mid_patch_len', 8),
            mid_patch_stride=artifact.get('mid_patch_stride', 4),
            mid_n_layers=artifact.get('mid_n_layers', 2),
            local_patch_len=artifact.get('local_patch_len', 4),
            local_patch_stride=artifact.get('local_patch_stride', 2),
            local_n_layers=artifact.get('local_n_layers', 2),
            target_weather_heads=artifact.get('target_weather_heads', 4),
            layer_scale_init=artifact.get('layer_scale_init', 1e-3),
            long_context_dim=artifact.get('long_context_dim', 64),
            router_hidden_dim=artifact.get('router_hidden_dim', 64),
            horizon_embedding_dim=artifact.get('horizon_embedding_dim', 16),
            router_dropout=artifact.get('router_dropout', 0.1),
            router_initial_bias=artifact.get(
                'router_initial_bias',
                [2.0, 0.0, 0.0, -2.0],
            ),
            power_scale_ratio=artifact.get('power_scale_ratio', 1.0),
            power_scale_offset=artifact.get('power_scale_offset', 0.0),
            correction_kernel_l2=artifact.get('correction_kernel_l2', 1e-4),
        )
    if model_name == PART3_ROUND2_STRONG_BASELINE_MODEL_NAME:
        return build_feature_screen_model(
            variant_id='f7',
            input_dim=len(artifact['input_cols']),
            target_channel_index=int(artifact['target_index']),
            power_scale_ratio=float(artifact['power_scale_ratio']),
            power_scale_offset=float(artifact['power_scale_offset']),
            regime_feature_config=artifact['regime_feature_config'],
        )

    input_shape = (artifact.get('history_len', HISTORY_LEN), len(artifact['input_cols']))
    builder = MODEL_BUILDERS[model_name]
    if model_name in {'informer', 'autoformer'}:
        return builder(input_shape, input_cols=artifact['input_cols'])
    return builder(input_shape)


def load_trained_model(model_name, farm_id, artifact):
    if model_name == PATCHTST_MODEL_NAME:
        model_path = artifact.get('model_path') or os.path.join(
            PATCHTST_SAVED_MODEL_DIR,
            f'patchtst_farm_{farm_id}.keras',
        )
        best_weights_path = artifact.get('best_weights_path') or os.path.join(
            PATCHTST_WEIGHTS_DIR,
            f'patchtst_farm_{farm_id}_best.weights.h5',
        )
    elif model_name == FETS_PATCHTST_MODEL_NAME:
        model_path = artifact.get('model_path') or os.path.join(
            FETS_PATCHTST_SAVED_MODEL_DIR,
            f'{FETS_PATCHTST_MODEL_NAME}_farm_{farm_id}.keras',
        )
        best_weights_path = artifact.get('best_weights_path') or os.path.join(
            FETS_PATCHTST_WEIGHTS_DIR,
            f'{FETS_PATCHTST_MODEL_NAME}_farm_{farm_id}_best.weights.h5',
        )
    elif model_name == PART3_ROUND2_STRONG_BASELINE_MODEL_NAME:
        model_path = artifact.get('model_path') or os.path.join(
            PART3_ROUND2_STRONG_BASELINE_RESULT_ROOT,
            'models',
            f'{model_name}_farm_{farm_id}.keras',
        )
        best_weights_path = artifact.get('best_weights_path') or os.path.join(
            PART3_ROUND2_STRONG_BASELINE_RESULT_ROOT,
            'weights',
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
        if (
            model_name == PART3_ROUND2_STRONG_BASELINE_MODEL_NAME
            and _file_sha256(model_path) != artifact['model_sha256']
        ):
            raise ValueError(
                f'强基线F7/G0模型hash与artifact不一致: {model_path}'
            )
        custom_objects = (
            get_part3_round2_strong_baseline_custom_objects()
            if model_name == PART3_ROUND2_STRONG_BASELINE_MODEL_NAME
            else get_custom_objects()
        )
        model = keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
        )
        if model_name == PART3_ROUND2_STRONG_BASELINE_MODEL_NAME:
            validate_part3_round2_strong_baseline_model(model, artifact)
        return model, model_path

    if not os.path.exists(best_weights_path):
        raise FileNotFoundError(
            f'未找到 {model_name} 场站 {farm_id} 的完整模型或最佳权重: '
            f'{model_path}, {best_weights_path}')

    if (
        model_name == PART3_ROUND2_STRONG_BASELINE_MODEL_NAME
        and _file_sha256(best_weights_path) != artifact['best_weights_sha256']
    ):
        raise ValueError(
            f'强基线F7/G0权重hash与artifact不一致: '
            f'{best_weights_path}'
        )

    model = build_model_from_weights(model_name, artifact)
    model.load_weights(best_weights_path)
    if model_name == PART3_ROUND2_STRONG_BASELINE_MODEL_NAME:
        validate_part3_round2_strong_baseline_model(model, artifact)
    return model, best_weights_path


def validate_part3_round2_strong_baseline_model(model, artifact):
    """Validate the full fused F7/G0 inference graph selected for comparison."""
    if len(model.inputs) != 1:
        raise ValueError('强基线F7/G0推理图必须只有一个历史输入')
    expected_shape = (
        int(artifact['history_len']),
        len(artifact['input_cols']),
    )
    actual_shape = tuple(int(value) for value in model.input_shape[1:])
    if actual_shape != expected_shape:
        raise ValueError(
            f'强基线F7/G0输入形状不一致: '
            f'{actual_shape} != {expected_shape}'
        )
    for diagnostic_name, layer_name in artifact['diagnostic_layers'].items():
        if diagnostic_name in {
            'forecast',
            'gate',
            'persistence_candidate',
            'corrected_candidate',
        }:
            try:
                model.get_layer(layer_name)
            except ValueError as exc:
                raise ValueError(
                    f'强基线F7/G0缺少诊断层 '
                    f'{diagnostic_name}={layer_name}'
                ) from exc
    actual_params = int(model.count_params())
    recorded_params = int(artifact.get('total_params', actual_params))
    if recorded_params != actual_params:
        raise ValueError(
            f'强基线F7/G0参数量与artifact不一致: '
            f'{actual_params} != {recorded_params}'
        )


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


def predict_with_router_diagnostics(model_name, model, pred_ds):
    """FeTS-PatchTST 单次前向同时返回预测和动态路由权重。"""
    if model_name != FETS_PATCHTST_MODEL_NAME:
        return model.predict(pred_ds, verbose=PREDICT_VERBOSE), None

    router_layer = model.get_layer('horizon_regime_router')
    diagnostic_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.output, router_layer.output],
        name='WindFeTSPatchTSTPredictDiagnostics',
    )
    predictions, router_weights = diagnostic_model.predict(
        pred_ds,
        verbose=PREDICT_VERBOSE,
    )
    return predictions, np.asarray(router_weights, dtype=float)


def predict_part3_round2_strong_baseline_with_diagnostics(
    model,
    pred_ds,
    artifact,
):
    """Predict the fused F7/G0 output and retain its two-candidate evidence."""
    names = artifact['diagnostic_layers']
    diagnostic_model = keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(names['forecast']).output,
            model.get_layer(names['persistence_candidate']).output,
            model.get_layer(names['corrected_candidate']).output,
            model.get_layer(names['gate']).output,
        ],
        name='Part3Round2F7G0StrongBaselineDiagnostics',
    )
    forecast, persistence, corrected, gate = diagnostic_model.predict(
        pred_ds,
        verbose=PREDICT_VERBOSE,
    )
    arrays = {
        'forecast': np.asarray(forecast),
        'persistence': np.asarray(persistence),
        'corrected': np.asarray(corrected),
        'gate': np.asarray(gate),
    }
    expected_forecast_len = int(artifact['forecast_len'])
    expected_shape = (arrays['forecast'].shape[0], expected_forecast_len)
    for key, values in arrays.items():
        if values.shape != expected_shape:
            raise ValueError(
                f'强基线F7/G0 {key}输出形状不一致: '
                f'{values.shape} != {expected_shape}'
            )
        if not np.isfinite(values).all():
            raise FloatingPointError(f'强基线F7/G0 {key}包含非有限值')
    gate = arrays['gate'].astype(float)
    if np.any((gate < -1e-6) | (gate > 1.0 + 1e-6)):
        raise ValueError('强基线F7/G0 correction gate超出[0, 1]')
    reconstructed = (
        arrays['persistence']
        + gate * (arrays['corrected'] - arrays['persistence'])
    )
    if not np.allclose(
        reconstructed,
        arrays['forecast'],
        rtol=1e-5,
        atol=1e-6,
    ):
        max_difference = float(
            np.max(np.abs(reconstructed - arrays['forecast']))
        )
        raise ValueError(
            '强基线F7/G0主输出不等于两候选门控重建: '
            f'max_abs={max_difference}'
        )
    router_weights = np.stack([1.0 - gate, gate], axis=-1)
    return arrays['forecast'], router_weights, arrays


def save_router_diagnostics(
    router_weights,
    expert_names,
    model_name,
    farm_id,
    dirs,
):
    """保存逐 horizon 的路由均值、离散程度和分位数。"""
    if router_weights is None:
        return None, {}
    if router_weights.ndim != 3:
        raise ValueError(f'router 权重必须为三维，实际为 {router_weights.shape}')
    if len(expert_names) != router_weights.shape[-1]:
        raise ValueError('artifact 专家名称数量与 router 输出不一致')
    if not np.isfinite(router_weights).all():
        raise FloatingPointError('测试集 router 权重包含非有限值')
    if not np.allclose(router_weights.sum(axis=-1), 1.0, atol=1e-5):
        raise ValueError('测试集 router 权重之和不为 1')

    entropy = -np.sum(
        router_weights
        * np.log(np.clip(router_weights, 1e-8, 1.0)),
        axis=-1,
    ) / np.log(len(expert_names))
    rows = []
    for horizon_idx in range(router_weights.shape[1]):
        horizon_weights = router_weights[:, horizon_idx, :]
        row = {
            'model_name': model_name,
            'farm_id': farm_id,
            'horizon_step': horizon_idx + 1,
            'horizon_minutes': (horizon_idx + 1) * 15,
            'normalized_entropy_mean': float(entropy[:, horizon_idx].mean()),
        }
        for expert_idx, expert_name in enumerate(expert_names):
            values = horizon_weights[:, expert_idx]
            row.update({
                f'{expert_name}_weight_mean': float(values.mean()),
                f'{expert_name}_weight_std': float(values.std()),
                f'{expert_name}_weight_p10': float(np.quantile(values, 0.10)),
                f'{expert_name}_weight_p90': float(np.quantile(values, 0.90)),
            })
        rows.append(row)

    diagnostics_path = os.path.join(
        dirs['router_diagnostics'],
        f'{model_name}_router_weights_farm_{farm_id}.csv',
    )
    pd.DataFrame(rows).to_csv(
        diagnostics_path,
        index=False,
        encoding='utf-8-sig',
    )
    overall_fields = {
        'router_diagnostics_path': diagnostics_path,
        'router_normalized_entropy': float(entropy.mean()),
    }
    overall_mean = router_weights.mean(axis=(0, 1))
    overall_fields.update({
        f'router_weight_{expert_name}': float(weight)
        for expert_name, weight in zip(expert_names, overall_mean)
    })
    return diagnostics_path, overall_fields


def predict_one_farm(model_name, test_file):
    farm_id = get_farm_id(test_file)
    dirs = model_output_dirs(model_name)
    print(f'\n===== 预测 {model_name} / 风电场 {farm_id} =====')

    artifact = load_artifact(model_name, farm_id)
    model, loaded_model_path = load_trained_model(model_name, farm_id, artifact)
    df, features, actual_power, capacity = prepare_prediction_arrays(test_file, artifact)
    history_len = artifact.get('history_len', HISTORY_LEN)
    forecast_len = artifact.get('forecast_len', FORECAST_LEN)

    pred_ds, n_samples = make_prediction_dataset(features, history_len, forecast_len)
    strong_baseline_diagnostics = None
    if model_name == PART3_ROUND2_STRONG_BASELINE_MODEL_NAME:
        (
            y_pred_scaled,
            router_weights,
            strong_baseline_diagnostics,
        ) = predict_part3_round2_strong_baseline_with_diagnostics(
            model,
            pred_ds,
            artifact,
        )
    else:
        y_pred_scaled, router_weights = predict_with_router_diagnostics(
            model_name,
            model,
            pred_ds,
        )
    y_pred = inverse_power(artifact['scaler_y'], y_pred_scaled).reshape(-1, forecast_len)
    if y_pred.shape[0] != n_samples:
        raise ValueError(
            f'{model_name} 场站 {farm_id} 预测样本数不一致: {y_pred.shape[0]} vs {n_samples}')
    if router_weights is not None and router_weights.shape[:2] != (
        n_samples,
        forecast_len,
    ):
        raise ValueError(
            f'{model_name} 场站 {farm_id} router 形状不一致: '
            f'{router_weights.shape} vs ({n_samples}, {forecast_len}, 专家数)'
        )

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
    strong_baseline_metric_fields = {}
    if strong_baseline_diagnostics is not None:
        persistence = inverse_power(
            artifact['scaler_y'],
            strong_baseline_diagnostics['persistence'],
        ).reshape(-1, forecast_len)
        corrected = inverse_power(
            artifact['scaler_y'],
            strong_baseline_diagnostics['corrected'],
        ).reshape(-1, forecast_len)
        if capacity is not None:
            persistence = np.clip(persistence, 0, capacity)
            corrected = np.clip(corrected, 0, capacity)
        else:
            persistence = np.clip(persistence, 0, None)
            corrected = np.clip(corrected, 0, None)
        gate = np.asarray(strong_baseline_diagnostics['gate'], dtype=float)
        pred_df['persistence_power'] = persistence.T.reshape(-1)
        pred_df['corrected_candidate_power'] = corrected.T.reshape(-1)
        pred_df['corrected_gate_weight'] = gate.T.reshape(-1)
        persistence_metrics = calculate_metrics(y_true, persistence, capacity)
        corrected_metrics = calculate_metrics(y_true, corrected, capacity)
        strong_baseline_metric_fields = {
            'model_family': 'part3_new_module_supplement',
            'model_variant': PART3_ROUND2_STRONG_BASELINE_VARIANT_ID,
            'training_protocol': 'strong_baseline_fair_training_bs256',
            'training_mode': artifact['training_mode'],
            'batch_size': int(artifact['batch_size']),
            'parameter_count': int(model.count_params()),
            'gate_corrected_weight_mean': float(gate.mean()),
            'persistence_mae': persistence_metrics['mae'],
            'persistence_rmse': persistence_metrics['rmse'],
            'persistence_capacity_normalized_mae': persistence_metrics[
                'capacity_normalized_mae'
            ],
            'persistence_capacity_normalized_rmse': persistence_metrics[
                'capacity_normalized_rmse'
            ],
            'corrected_candidate_mae': corrected_metrics['mae'],
            'corrected_candidate_rmse': corrected_metrics['rmse'],
            'corrected_candidate_capacity_normalized_mae': corrected_metrics[
                'capacity_normalized_mae'
            ],
            'corrected_candidate_capacity_normalized_rmse': corrected_metrics[
                'capacity_normalized_rmse'
            ],
        }

    pred_path = os.path.join(
        dirs['predictions'],
        f'{model_name}_predictions_farm_{farm_id}.csv',
    )
    pred_df.to_csv(pred_path, index=False, encoding='utf-8-sig')

    metric_df = metrics_by_horizon(model_name, farm_id, y_true, y_pred, capacity, forecast_len)
    horizon_metric_path = os.path.join(
        dirs['root'],
        f'{model_name}_metrics_by_horizon_farm_{farm_id}.csv',
    )
    metric_df.to_csv(horizon_metric_path, index=False, encoding='utf-8-sig')

    if model_name == PART3_ROUND2_STRONG_BASELINE_MODEL_NAME:
        expert_names = list(PART3_ROUND2_STRONG_BASELINE_EXPERT_NAMES)
    else:
        expert_names = artifact.get(
            'expert_names',
            list(FETS_PATCHTST_EXPERT_NAMES),
        )
    _, router_metric_fields = save_router_diagnostics(
        router_weights,
        expert_names,
        model_name,
        farm_id,
        dirs,
    )
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
        **router_metric_fields,
        **weighted_metric_fields,
        **strong_baseline_metric_fields,
    })
    print(f"{model_name} 场站 {farm_id}: MAE={all_metrics['mae']:.4f}, "
          f"RMSE={all_metrics['rmse']:.4f}")

    del model
    keras.backend.clear_session()
    return all_metrics, metric_df


def predict_model_family(model_name, test_files):
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


def _validate_legacy_strong_comparison_frame(frame, frame_name):
    """Require the existing nine-model result matrix before read-only reuse."""
    required = {'model_name', 'farm_id', 'horizon_step'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f'{frame_name}缺少列: {missing}')
    checked = frame.copy()
    checked['model_name'] = checked['model_name'].astype(str)
    checked['farm_id'] = checked['farm_id'].astype(str)
    actual_models = set(checked['model_name'])
    expected_models = set(LEGACY_STRONG_COMPARISON_MODEL_NAMES)
    if actual_models != expected_models:
        raise ValueError(
            f'{frame_name}旧基线模型集不完整: '
            f'missing={sorted(expected_models - actual_models)}, '
            f'extra={sorted(actual_models - expected_models)}'
        )
    if checked.duplicated(['model_name', 'farm_id', 'horizon_step']).any():
        raise ValueError(f'{frame_name}存在重复model/farm/horizon键')
    farm_counts = checked.groupby('model_name')['farm_id'].nunique()
    if not (farm_counts == STRONG_COMPARISON_EXPECTED_FARM_COUNT).all():
        raise ValueError(
            f'{frame_name}不是9模型×5场站完整矩阵: '
            f'{farm_counts.to_dict()}'
        )
    farm_sets = checked.groupby('model_name')['farm_id'].agg(
        lambda values: frozenset(values)
    )
    if farm_sets.nunique() != 1:
        raise ValueError(
            f'{frame_name}旧9模型的场站集不一致: '
            f'{farm_sets.to_dict()}'
        )
    model_farm_pairs = checked[['model_name', 'farm_id']].drop_duplicates()
    if len(model_farm_pairs) != (
        len(LEGACY_STRONG_COMPARISON_MODEL_NAMES)
        * STRONG_COMPARISON_EXPECTED_FARM_COUNT
    ):
        raise ValueError(f'{frame_name}旧基线model/farm键不完整')
    if frame_name == 'global summary':
        if len(checked) != len(model_farm_pairs):
            raise ValueError('global summary每个model/farm必须只有一行')
        if not (checked['horizon_step'].astype(str) == 'all').all():
            raise ValueError('global summary只能包含horizon_step=all')
    else:
        row_counts = checked.groupby(['model_name', 'farm_id']).size()
        if not (row_counts == FORECAST_LEN + 1).all():
            raise ValueError(
                f'global horizon每个model/farm应有{FORECAST_LEN + 1}行: '
                f'{row_counts[row_counts != FORECAST_LEN + 1].to_dict()}'
            )
        expected_horizons = {'all', *map(str, range(1, FORECAST_LEN + 1))}
        for key, group in checked.groupby(['model_name', 'farm_id']):
            actual_horizons = set(group['horizon_step'].astype(str))
            if actual_horizons != expected_horizons:
                raise ValueError(
                    f'global horizon {key}的horizon集不完整: '
                    f'{sorted(actual_horizons)}'
                )
    return checked


def load_existing_legacy_strong_comparison_results():
    """Read, never recompute, the complete 9x5 legacy baseline matrix."""
    summary_path = os.path.join(
        BASE_RESULT_DIR,
        'wind_dl_all_models_test_metrics_summary.csv',
    )
    horizon_path = os.path.join(
        BASE_RESULT_DIR,
        'wind_dl_all_models_test_metrics_by_horizon_all.csv',
    )
    missing_paths = [
        path for path in (summary_path, horizon_path) if not os.path.exists(path)
    ]
    if missing_paths:
        raise FileNotFoundError(
            '仅运行第三部分第二轮模型时需要旧基线全局结果: '
            f'{missing_paths}'
        )
    summary_all = pd.read_csv(summary_path, dtype={'farm_id': str})
    horizon_all = pd.read_csv(horizon_path, dtype={'farm_id': str})
    # A repeated strong-baseline run may see the 50-row file written by its
    # previous run.  Reuse only the immutable nine legacy families.
    summary = summary_all[
        summary_all['model_name'].astype(str).isin(
            LEGACY_STRONG_COMPARISON_MODEL_NAMES
        )
    ].copy()
    horizon = horizon_all[
        horizon_all['model_name'].astype(str).isin(
            LEGACY_STRONG_COMPARISON_MODEL_NAMES
        )
    ].copy()
    return (
        _validate_legacy_strong_comparison_frame(summary, 'global summary'),
        _validate_legacy_strong_comparison_frame(horizon, 'global horizon'),
    )


def _validate_new_strong_baseline_matrix(summary, horizon):
    expected_model = PART3_ROUND2_STRONG_BASELINE_MODEL_NAME
    selected_summary = summary[
        summary['model_name'].astype(str) == expected_model
    ].copy()
    selected_horizon = horizon[
        horizon['model_name'].astype(str) == expected_model
    ].copy()
    if len(selected_summary) != STRONG_COMPARISON_EXPECTED_FARM_COUNT:
        raise ValueError(
            '强基线新模型测试汇总不是5场站: '
            f'{len(selected_summary)}'
        )
    if selected_summary['farm_id'].astype(str).nunique() != (
        STRONG_COMPARISON_EXPECTED_FARM_COUNT
    ):
        raise ValueError('强基线新模型测试汇总场站键不唯一')
    row_counts = selected_horizon.groupby(
        selected_horizon['farm_id'].astype(str)
    ).size()
    if len(row_counts) != STRONG_COMPARISON_EXPECTED_FARM_COUNT or not (
        row_counts == FORECAST_LEN + 1
    ).all():
        raise ValueError(
            '强基线新模型分horizon结果不是5场站×17行: '
            f'{row_counts.to_dict()}'
        )


def validate_part3_round2_all_model_comparison(global_summary, global_horizon):
    """Require a complete 10-model matrix before replacing the global CSVs."""
    _validate_new_strong_baseline_matrix(global_summary, global_horizon)
    legacy_summary = global_summary[
        global_summary['model_name'].astype(str).isin(
            LEGACY_STRONG_COMPARISON_MODEL_NAMES
        )
    ]
    legacy_horizon = global_horizon[
        global_horizon['model_name'].astype(str).isin(
            LEGACY_STRONG_COMPARISON_MODEL_NAMES
        )
    ]
    _validate_legacy_strong_comparison_frame(
        legacy_summary,
        'global summary',
    )
    _validate_legacy_strong_comparison_frame(
        legacy_horizon,
        'global horizon',
    )
    legacy_farms = set(legacy_summary['farm_id'].astype(str))
    strong_farms = set(
        global_summary.loc[
            global_summary['model_name'].astype(str)
            == PART3_ROUND2_STRONG_BASELINE_MODEL_NAME,
            'farm_id',
        ].astype(str)
    )
    if strong_farms != legacy_farms:
        raise ValueError(
            '强基线新模型与旧9模型的场站集不一致: '
            f'new={sorted(strong_farms)}, legacy={sorted(legacy_farms)}'
        )
    expected_models = {
        *LEGACY_STRONG_COMPARISON_MODEL_NAMES,
        PART3_ROUND2_STRONG_BASELINE_MODEL_NAME,
    }
    actual_models = set(global_summary['model_name'].astype(str))
    if actual_models != expected_models:
        raise ValueError(
            '强基线全模型对比集不是10个预定模型: '
            f'missing={sorted(expected_models - actual_models)}, '
            f'extra={sorted(actual_models - expected_models)}'
        )


def save_part3_round2_all_model_comparison(global_summary, global_horizon):
    """Archive the 10-model comparison beside the new round-2 experiment."""
    validate_part3_round2_all_model_comparison(
        global_summary,
        global_horizon,
    )
    dirs = model_output_dirs(PART3_ROUND2_STRONG_BASELINE_MODEL_NAME)
    summary_path = os.path.join(
        dirs['root'],
        'part3_round2_all_models_test_metrics_summary.csv',
    )
    horizon_path = os.path.join(
        dirs['root'],
        'part3_round2_all_models_test_metrics_by_horizon.csv',
    )
    global_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    global_horizon.to_csv(horizon_path, index=False, encoding='utf-8-sig')

    macro = global_summary.copy()
    numeric_columns = [
        'mae',
        'rmse',
        'r2',
        'capacity_normalized_mae',
        'capacity_normalized_rmse',
    ]
    macro = macro.groupby('model_name', as_index=False).agg(
        farm_count=('farm_id', 'nunique'),
        **{
            f'macro_{column}': (column, 'mean')
            for column in numeric_columns
        },
    )
    macro = macro.rename(
        columns={
            'macro_capacity_normalized_mae': 'macro_nmae',
            'macro_capacity_normalized_rmse': 'macro_nrmse',
        }
    )
    macro['macro_nrmse_rank'] = macro['macro_nrmse'].rank(
        method='min',
        ascending=True,
    ).astype(int)
    macro['macro_nmae_rank'] = macro['macro_nmae'].rank(
        method='min',
        ascending=True,
    ).astype(int)
    macro = macro.sort_values(
        ['macro_nrmse_rank', 'macro_nmae_rank', 'model_name']
    )
    macro_path = os.path.join(
        dirs['root'],
        'part3_round2_all_models_test_macro_comparison.csv',
    )
    macro.to_csv(macro_path, index=False, encoding='utf-8-sig')
    print(f'第三部分第二轮10模型汇总已保存: {summary_path}')
    print(f'第三部分第二轮10模型Macro排名已保存: {macro_path}')


def main():
    set_global_seed(seed)

    test_files = discover_test_files(DATA_DIR)
    if not test_files:
        raise FileNotFoundError(f'未在 {DATA_DIR} 找到 {TEST_FILE_PATTERN}')

    requested_model_names = get_requested_model_names()
    strong_baseline_requested = (
        PART3_ROUND2_STRONG_BASELINE_MODEL_NAME in requested_model_names
    )
    strong_baseline_only = requested_model_names == [
        PART3_ROUND2_STRONG_BASELINE_MODEL_NAME
    ]
    complete_default_all_requested = (
        set(requested_model_names) == set(ALL_MODEL_NAMES)
    )
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

    if (
        strong_baseline_requested
        and not complete_default_all_requested
        and all_summary
        and all_horizon
    ):
        legacy_summary, legacy_horizon = (
            load_existing_legacy_strong_comparison_results()
        )
        all_summary.insert(0, legacy_summary)
        all_horizon.insert(0, legacy_horizon)
    if strong_baseline_only and (not all_summary or not all_horizon):
        raise FileNotFoundError(
            '未生成第三部分第二轮强基线任一场站预测；'
            '请先完成新训练脚本的5场站训练'
        )

    if all_summary and all_horizon:
        preview_summary = pd.concat(all_summary, ignore_index=True)
        preview_horizon = pd.concat(all_horizon, ignore_index=True)
        if PART3_ROUND2_STRONG_BASELINE_MODEL_NAME in set(
            preview_summary['model_name'].astype(str)
        ):
            preview_summary['farm_id'] = preview_summary['farm_id'].astype(str)
            preview_horizon['farm_id'] = preview_horizon['farm_id'].astype(str)
            preview_summary = preview_summary.drop_duplicates(
                ['model_name', 'farm_id', 'horizon_step'],
                keep='last',
            )
            preview_horizon = preview_horizon.drop_duplicates(
                ['model_name', 'farm_id', 'horizon_step'],
                keep='last',
            )
            validate_part3_round2_all_model_comparison(
                preview_summary,
                preview_horizon,
            )

    global_summary = None
    if all_summary:
        global_summary = pd.concat(all_summary, ignore_index=True)
        if PART3_ROUND2_STRONG_BASELINE_MODEL_NAME in set(
            global_summary['model_name'].astype(str)
        ):
            global_summary['farm_id'] = global_summary['farm_id'].astype(str)
            global_summary = global_summary.drop_duplicates(
                ['model_name', 'farm_id', 'horizon_step'],
                keep='last',
            )
        global_summary_path = os.path.join(
            BASE_RESULT_DIR,
            'wind_dl_all_models_test_metrics_summary.csv',
        )
        global_summary.to_csv(global_summary_path, index=False, encoding='utf-8-sig')
        print(f'全部模型汇总指标已保存: {global_summary_path}')

    global_horizon = None
    if all_horizon:
        global_horizon = pd.concat(all_horizon, ignore_index=True)
        if PART3_ROUND2_STRONG_BASELINE_MODEL_NAME in set(
            global_horizon['model_name'].astype(str)
        ):
            global_horizon['farm_id'] = global_horizon['farm_id'].astype(str)
            global_horizon = global_horizon.drop_duplicates(
                ['model_name', 'farm_id', 'horizon_step'],
                keep='last',
            )
        global_horizon_path = os.path.join(
            BASE_RESULT_DIR,
            'wind_dl_all_models_test_metrics_by_horizon_all.csv',
        )
        global_horizon.to_csv(global_horizon_path, index=False, encoding='utf-8-sig')
        print(f'全部模型分horizon指标已保存: {global_horizon_path}')

    if (
        global_summary is not None
        and global_horizon is not None
        and PART3_ROUND2_STRONG_BASELINE_MODEL_NAME
        in set(global_summary['model_name'].astype(str))
    ):
        save_part3_round2_all_model_comparison(
            global_summary,
            global_horizon,
        )

    print('全部深度学习模型测试集预测完成')


if __name__ == '__main__':
    main()
