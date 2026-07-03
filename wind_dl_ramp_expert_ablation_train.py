"""CNN ramp expert structural ablation for tuned PatchTST.

The completed ``revin_balanced_loss`` parent is metrics-only and is never
retrained here.  Each new structure uses seed 2026, keeps the same target-farm
training protocol, and is promoted to an independent prediction namespace.
Completed new candidates are reused before any training switch is considered.
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from wind_dl_model_train import (
    DATA_DIR,
    FORECAST_LEN,
    HISTORY_LEN,
    TRAIN_FILE_PATTERN,
    build_scaled_arrays,
    load_and_preprocess,
)
from wind_dl_tuned_patchtst_train import (
    ABLATION_ALL_METRICS_PATH,
    RAMP_EXPERT_CONTEXT_LEN,
    RAMP_EXPERT_DILATIONS,
    RAMP_EXPERT_FILTERS,
    TUNED_MODEL_NAME,
    configure_runtime,
    discover_train_files,
    get_adapter_channel_indices,
    get_farm_id,
    make_variant_dirs,
    make_window_targets,
    set_global_seed,
    train_ablation_variant,
)


PARENT_VARIANT = 'revin_balanced_loss'
STRUCTURE_VALIDATION_SEED = int(os.getenv(
    'WIND_RAMP_EXPERT_SEED',
    '2026',
))
ENABLE_RAMP_EXPERT_TRAINING = os.getenv(
    'WIND_RAMP_EXPERT_ENABLE_TRAINING',
    '0',
) == '1'
REUSE_COMPLETED_CANDIDATES = os.getenv(
    'WIND_RAMP_EXPERT_REUSE_COMPLETED',
    '1',
) == '1'

RAMP_TRAJECTORY_MODEL_NAME = 'tuned_patchtst_ramp_trajectory'
RAMP_GATED_MODEL_NAME = 'tuned_patchtst_ramp_gated'
RAMP_PERSISTENCE_GATED_MODEL_NAME = (
    'tuned_patchtst_ramp_persistence_gated'
)

STRUCTURAL_VARIANTS = [
    {
        'name': 'revin_balanced_loss_ramp_trajectory',
        'model_name': RAMP_TRAJECTORY_MODEL_NAME,
        'step': 'B',
        'round': 8,
        'parent_variant': PARENT_VARIANT,
        'added_module': 'causal_dilated_cnn_ramp_trajectory',
        'train_env': 'WIND_RAMP_EXPERT_TRAIN_TRAJECTORY',
        'use_revin': True,
        'use_cnn_adapter': False,
        'use_ramp_expert': True,
        'ramp_fusion_mode': 'residual',
        'use_balanced_loss': True,
        'use_rmse_balanced_loss': False,
        'use_swa': False,
        'use_distillation': True,
        'multi_seed': False,
        'use_supplementary_teacher_pretraining': False,
    },
    {
        'name': 'revin_balanced_loss_ramp_gated',
        'model_name': RAMP_GATED_MODEL_NAME,
        'step': 'C',
        'round': 8,
        'parent_variant': 'revin_balanced_loss_ramp_trajectory',
        'added_module': 'sample_horizon_long_ramp_gating',
        'train_env': 'WIND_RAMP_EXPERT_TRAIN_GATED',
        'use_revin': True,
        'use_cnn_adapter': False,
        'use_ramp_expert': True,
        'ramp_fusion_mode': 'two_expert_gating',
        'use_balanced_loss': True,
        'use_rmse_balanced_loss': False,
        'use_swa': False,
        'use_distillation': True,
        'multi_seed': False,
        'use_supplementary_teacher_pretraining': False,
    },
    {
        'name': 'revin_balanced_loss_ramp_persistence_gated',
        'model_name': RAMP_PERSISTENCE_GATED_MODEL_NAME,
        'step': 'D',
        'round': 8,
        'parent_variant': 'revin_balanced_loss_ramp_gated',
        'added_module': 'persistence_expert_horizon_gating',
        'train_env': 'WIND_RAMP_EXPERT_TRAIN_PERSISTENCE_GATED',
        'use_revin': True,
        'use_cnn_adapter': False,
        'use_ramp_expert': True,
        'ramp_fusion_mode': 'three_expert_gating',
        'use_balanced_loss': True,
        'use_rmse_balanced_loss': False,
        'use_swa': False,
        'use_distillation': True,
        'multi_seed': False,
        'use_supplementary_teacher_pretraining': False,
    },
]

RESULT_DIR = os.path.join('./wind_results', 'ramp_expert_ablation')
DETAIL_METRICS_PATH = os.path.join(
    RESULT_DIR,
    'ramp_expert_ablation_metrics_all_farms.csv',
)
SUMMARY_METRICS_PATH = os.path.join(
    RESULT_DIR,
    'ramp_expert_ablation_summary.csv',
)
CONFIG_PATH = os.path.join(
    RESULT_DIR,
    'ramp_expert_ablation_config.json',
)


def parse_env_bool(name, default=True):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError(f'环境变量 {name} 应为0/1或true/false，当前为: {value}')


def ensure_dirs():
    os.makedirs(RESULT_DIR, exist_ok=True)
    for variant in STRUCTURAL_VARIANTS:
        model_name = variant['model_name']
        for path in [
            os.path.join('./models', model_name),
            os.path.join('./wind_results', model_name, 'weights'),
            os.path.join('./wind_results', model_name, 'preprocess'),
            os.path.join('./wind_results', model_name, 'tails'),
            os.path.join('./wind_results', model_name, 'selection'),
        ]:
            os.makedirs(path, exist_ok=True)


def load_parent_results():
    """Read the completed front-five parent; there is no parent train switch."""
    if not os.path.exists(ABLATION_ALL_METRICS_PATH):
        raise FileNotFoundError(
            f'缺少前五轮消融结果: {ABLATION_ALL_METRICS_PATH}'
        )
    detail = pd.read_csv(
        ABLATION_ALL_METRICS_PATH,
        dtype={'farm_id': str},
    )
    if 'round' in detail:
        detail = detail[pd.to_numeric(
            detail['round'],
            errors='coerce',
        ) <= 5]
    detail = detail[detail['variant'] == PARENT_VARIANT].copy()
    if detail.empty:
        raise ValueError(f'未找到前五轮父模型variant={PARENT_VARIANT}')

    results = {}
    for record in detail.to_dict('records'):
        record.update({
            'farm_id': str(record['farm_id']),
            'step': 'A',
            'result_source': 'reused_front_five_metrics',
            'use_ramp_expert': False,
            'ramp_fusion_mode': 'none',
        })
        results[record['farm_id']] = record
    return results


def candidate_artifact_path(farm_id, variant):
    dirs = make_variant_dirs(variant['name'])
    return os.path.join(
        dirs['preprocess'],
        f"tuned_patchtst_{variant['name']}_farm_{farm_id}_preprocess.pkl",
    )


def load_completed_candidate(farm_id, variant):
    if not REUSE_COMPLETED_CANDIDATES:
        return None
    artifact_path = candidate_artifact_path(farm_id, variant)
    if not os.path.exists(artifact_path):
        return None

    artifact = joblib.load(artifact_path)
    try:
        artifact_seed = int(artifact.get('training_seed', -1))
    except (TypeError, ValueError):
        artifact_seed = -1
    model_path = artifact.get('model_path')
    weights_path = artifact.get('best_weights_path')
    required_metrics = {
        'val_composite_score',
        'val_capacity_normalized_rmse',
    }
    if (
        artifact.get('ablation_variant') != variant['name']
        or artifact_seed != STRUCTURE_VALIDATION_SEED
        or not required_metrics.issubset(artifact)
        or not model_path
        or not os.path.exists(model_path)
        or not weights_path
        or not os.path.exists(weights_path)
    ):
        return None

    return {
        'farm_id': str(farm_id),
        'variant': variant['name'],
        'storage_variant': variant['name'],
        'step': variant['step'],
        'round': variant['round'],
        'parent_variant': variant['parent_variant'],
        'result_source': 'reused_completed_candidate',
        'added_module': variant['added_module'],
        'use_revin': variant['use_revin'],
        'use_cnn_adapter': variant['use_cnn_adapter'],
        'use_ramp_expert': True,
        'ramp_fusion_mode': variant['ramp_fusion_mode'],
        'ramp_expert_context_len': artifact.get(
            'ramp_expert_context_len',
            RAMP_EXPERT_CONTEXT_LEN,
        ),
        'ramp_expert_filters': artifact.get(
            'ramp_expert_filters',
            RAMP_EXPERT_FILTERS,
        ),
        'ramp_expert_dilations': artifact.get(
            'ramp_expert_dilations',
            list(RAMP_EXPERT_DILATIONS),
        ),
        'use_balanced_loss': variant['use_balanced_loss'],
        'use_rmse_balanced_loss': False,
        'use_swa': False,
        'use_distillation': True,
        'use_supplementary_teacher_pretraining': False,
        'training_seed': STRUCTURE_VALIDATION_SEED,
        'multi_seed': False,
        'selected_weight_source': artifact.get(
            'selected_weight_source',
            'raw_best',
        ),
        'model_path': model_path,
        'best_weights_path': weights_path,
        'artifact_path': artifact_path,
        'history_path': artifact.get('history_path'),
        'distillation_stats_path': artifact.get(
            'distillation_stats_path'
        ),
        **{
            key: value
            for key, value in artifact.items()
            if key.startswith('val_') or key.startswith('teacher_val_')
        },
    }


def promote_candidate(train_df, candidate, variant, parent_result):
    """Copy one structural candidate to its own prediction namespace."""
    farm_id = str(candidate['farm_id'])
    model_name = variant['model_name']
    model_path = os.path.join(
        './models',
        model_name,
        f'{model_name}_farm_{farm_id}.keras',
    )
    weights_path = os.path.join(
        './wind_results',
        model_name,
        'weights',
        f'{model_name}_farm_{farm_id}_best.weights.h5',
    )
    artifact_path = os.path.join(
        './wind_results',
        model_name,
        'preprocess',
        f'{model_name}_farm_{farm_id}_preprocess.pkl',
    )
    tail_path = os.path.join(
        './wind_results',
        model_name,
        'tails',
        f'{model_name}_tail_farm_{farm_id}.csv',
    )
    selection_path = os.path.join(
        './wind_results',
        model_name,
        'selection',
        f'{model_name}_selection_farm_{farm_id}.json',
    )
    promoted_paths = {
        'farm_id': farm_id,
        'model_name': model_name,
        'variant': candidate['variant'],
        'step': variant['step'],
        'model_path': model_path,
        'best_weights_path': weights_path,
        'artifact_path': artifact_path,
        'tail_path': tail_path,
        'selection_path': selection_path,
    }
    required_paths = [
        model_path,
        weights_path,
        artifact_path,
        tail_path,
        selection_path,
    ]
    if all(os.path.exists(path) for path in required_paths):
        canonical_artifact = joblib.load(artifact_path)
        if (
            canonical_artifact.get('selected_ablation_variant')
            == candidate['variant']
            and canonical_artifact.get('training_seed')
            == STRUCTURE_VALIDATION_SEED
        ):
            print(f'复用已晋升预测模型: {model_name} farm={farm_id}')
            return promoted_paths

    artifact = joblib.load(candidate['artifact_path'])
    model = keras.models.load_model(
        candidate['model_path'],
        compile=False,
    )
    model.save(model_path)
    model.save_weights(weights_path)

    artifact.update({
        'model_name': model_name,
        'source_model_name': TUNED_MODEL_NAME,
        'experiment_name': 'cnn_ramp_expert_structural_ablation_v1',
        'experiment_created_at': datetime.now().isoformat(
            timespec='seconds'
        ),
        'structural_ablation_step': variant['step'],
        'structural_parent_metrics_only': True,
        'structural_parent_variant': variant['parent_variant'],
        'structural_parent_val_nrmse': parent_result.get(
            'val_capacity_normalized_rmse'
        ),
        'structural_parent_val_composite': parent_result.get(
            'val_composite_score'
        ),
        'selected_ablation_variant': candidate['variant'],
        'selected_ablation_round': candidate['round'],
        'selected_parent_variant': candidate['parent_variant'],
        'selected_weight_source': candidate.get('selected_weight_source'),
        'source_variant_model_path': candidate['model_path'],
        'source_variant_weights_path': candidate['best_weights_path'],
        'model_path': model_path,
        'best_weights_path': weights_path,
    })
    joblib.dump(artifact, artifact_path)

    train_df.iloc[-HISTORY_LEN:].to_csv(tail_path, index=True)

    selection = {
        'farm_id': farm_id,
        'model_name': model_name,
        'variant': candidate['variant'],
        'step': variant['step'],
        'parent_variant': variant['parent_variant'],
        'selection_rule': (
            'independent_structural_candidate_no_champion_fallback'
        ),
        'training_seed': STRUCTURE_VALIDATION_SEED,
        'val_capacity_normalized_rmse': float(
            candidate['val_capacity_normalized_rmse']
        ),
        'val_composite_score': float(
            candidate['val_composite_score']
        ),
        'canonical_model_path': model_path,
        'canonical_weights_path': weights_path,
        'canonical_artifact_path': artifact_path,
    }
    with open(selection_path, 'w', encoding='utf-8') as file:
        json.dump(selection, file, ensure_ascii=False, indent=2)

    del model
    keras.backend.clear_session()
    return promoted_paths


def train_one_farm(train_file, parent_result):
    farm_id = get_farm_id(train_file)
    if parent_result is None:
        raise ValueError(f'场站 {farm_id} 缺少前五轮父模型结果')
    print(f'\n===== CNN ramp expert结构消融 / 风电场 {farm_id} =====')

    train_df, feature_cols, capacity = load_and_preprocess(
        train_file,
        is_train=True,
    )
    features, target, input_cols, target_index, scaler_x, scaler_y = (
        build_scaled_arrays(train_df, feature_cols)
    )
    y_train, y_val, train_samples, total_samples = make_window_targets(
        features,
        target,
        HISTORY_LEN,
        FORECAST_LEN,
    )
    adapter_channel_indices = get_adapter_channel_indices(input_cols)

    results = [dict(parent_result)]
    result_by_variant = {PARENT_VARIANT: results[0]}
    promoted = []
    for variant in STRUCTURAL_VARIANTS:
        candidate = load_completed_candidate(farm_id, variant)
        if candidate is not None:
            print(
                f"复用完成模型: step={variant['step']} "
                f"variant={variant['name']}"
            )
        elif not parse_env_bool(variant['train_env'], True):
            print(
                f"训练开关关闭且无完整产物，跳过: {variant['train_env']}=0"
            )
            continue
        else:
            print(
                f"训练新结构: step={variant['step']} "
                f"variant={variant['name']}, "
                f"seed={STRUCTURE_VALIDATION_SEED}"
            )
            candidate = train_ablation_variant(
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
                training_seed=STRUCTURE_VALIDATION_SEED,
                storage_name=variant['name'],
            )
            candidate['step'] = variant['step']

        parent = result_by_variant.get(variant['parent_variant'])
        if parent is None:
            print(
                f"警告: {variant['name']} 缺少父结果，保留候选但无法计算"
                '增量对照'
            )
            parent = parent_result
        results.append(candidate)
        result_by_variant[variant['name']] = candidate
        promoted.append(promote_candidate(
            train_df,
            candidate,
            variant,
            parent,
        ))
    return results, promoted


def summarize(results):
    detail = pd.DataFrame(results)
    rows = []
    variants = [{
        'name': PARENT_VARIANT,
        'step': 'A',
        'parent_variant': None,
        'added_module': 'completed_front_five_parent',
    }] + STRUCTURAL_VARIANTS
    for variant in variants:
        current = detail[detail['variant'] == variant['name']]
        if current.empty:
            continue
        row = {
            'step': variant['step'],
            'variant': variant['name'],
            'parent_variant': variant['parent_variant'],
            'added_module': variant['added_module'],
            'farms': int(len(current)),
            'mean_val_composite_score': float(
                current['val_composite_score'].mean()
            ),
            'mean_val_capacity_normalized_rmse': float(
                current['val_capacity_normalized_rmse'].mean()
            ),
            'mean_composite_delta_vs_parent': np.nan,
            'mean_nrmse_delta_vs_parent': np.nan,
            'composite_improved_farms_vs_parent': 0,
            'nrmse_improved_farms_vs_parent': 0,
        }
        parent_name = variant['parent_variant']
        if parent_name is not None:
            parent = detail[detail['variant'] == parent_name][[
                'farm_id',
                'val_composite_score',
                'val_capacity_normalized_rmse',
            ]].rename(columns={
                'val_composite_score': 'parent_composite',
                'val_capacity_normalized_rmse': 'parent_nrmse',
            })
            compared = current.merge(parent, on='farm_id', how='inner')
            composite_delta = (
                compared['val_composite_score']
                - compared['parent_composite']
            )
            nrmse_delta = (
                compared['val_capacity_normalized_rmse']
                - compared['parent_nrmse']
            )
            if not compared.empty:
                row.update({
                    'mean_composite_delta_vs_parent': float(
                        composite_delta.mean()
                    ),
                    'mean_nrmse_delta_vs_parent': float(
                        nrmse_delta.mean()
                    ),
                    'composite_improved_farms_vs_parent': int(
                        (composite_delta < 0).sum()
                    ),
                    'nrmse_improved_farms_vs_parent': int(
                        (nrmse_delta < 0).sum()
                    ),
                })
        rows.append(row)
    return detail, pd.DataFrame(rows)


def save_config():
    config = {
        'experiment': 'cnn_ramp_expert_structural_ablation_v1',
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'training_seed': STRUCTURE_VALIDATION_SEED,
        'parent_variant': PARENT_VARIANT,
        'parent_training_enabled': False,
        'reuse_completed_candidates': REUSE_COMPLETED_CANDIDATES,
        'ramp_expert_context_len': RAMP_EXPERT_CONTEXT_LEN,
        'ramp_expert_filters': RAMP_EXPERT_FILTERS,
        'ramp_expert_dilations': list(RAMP_EXPERT_DILATIONS),
        'variants': [
            {
                **variant,
                'train_if_missing': parse_env_bool(
                    variant['train_env'],
                    True,
                ),
            }
            for variant in STRUCTURAL_VARIANTS
        ],
    }
    with open(CONFIG_PATH, 'w', encoding='utf-8') as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


def main():
    configure_runtime()
    set_global_seed(STRUCTURE_VALIDATION_SEED)
    if not ENABLE_RAMP_EXPERT_TRAINING:
        print(
            'CNN ramp expert结构消融默认关闭，未执行训练。'
            '前五轮revin_balanced_loss只读复用，不存在重训开关。'
            '如需训练缺失的新结构，请设置 '
            'WIND_RAMP_EXPERT_ENABLE_TRAINING=1。'
        )
        return

    ensure_dirs()
    save_config()
    parent_results = load_parent_results()
    train_files = discover_train_files(DATA_DIR)
    if not train_files:
        raise FileNotFoundError(
            f'未在 {DATA_DIR} 找到 {TRAIN_FILE_PATTERN}'
        )

    print(f'固定结构消融seed: {STRUCTURE_VALIDATION_SEED}')
    print('父模型: 前五轮revin_balanced_loss指标只读复用')
    print(f'完成候选优先复用: {REUSE_COMPLETED_CANDIDATES}')
    all_results = []
    all_promoted = []
    for train_file in train_files:
        farm_id = get_farm_id(train_file)
        results, promoted = train_one_farm(
            train_file,
            parent_results.get(str(farm_id)),
        )
        all_results.extend(results)
        all_promoted.extend(promoted)

    detail, summary = summarize(all_results)
    detail.to_csv(
        DETAIL_METRICS_PATH,
        index=False,
        encoding='utf-8-sig',
    )
    summary.to_csv(
        SUMMARY_METRICS_PATH,
        index=False,
        encoding='utf-8-sig',
    )
    pd.DataFrame(all_promoted).to_csv(
        os.path.join(RESULT_DIR, 'ramp_expert_promoted_models.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    print(f'结构消融明细: {DETAIL_METRICS_PATH}')
    print(f'结构消融汇总: {SUMMARY_METRICS_PATH}')


if __name__ == '__main__':
    main()
