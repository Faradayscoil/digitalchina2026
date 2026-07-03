"""Train tuned PatchTST with unrelated-farm teacher pretraining.

The experiment is deliberately isolated from ``tuned_patchtst`` canonical
artifacts.  The current structure-validation profile trains one seed and
promotes that candidate to ``tuned_patchtst_external_teacher`` for prediction;
the round-3 parent metrics are retained only as a reference.
"""

import json
import os
from datetime import datetime

import joblib
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
    ABLATION_DIR,
    TUNED_MODEL_NAME,
    configure_runtime,
    discover_train_files,
    get_adapter_channel_indices,
    get_farm_id,
    make_window_targets,
    set_global_seed,
    train_ablation_variant,
)


EXTERNAL_TEACHER_MODEL_NAME = 'tuned_patchtst_external_teacher'
RESULT_DIR = os.path.join('./wind_results', EXTERNAL_TEACHER_MODEL_NAME)
MODEL_DIR = os.path.join('./models', EXTERNAL_TEACHER_MODEL_NAME)
WEIGHTS_DIR = os.path.join(RESULT_DIR, 'weights')
PREPROCESS_DIR = os.path.join(RESULT_DIR, 'preprocess')
TAIL_DIR = os.path.join(RESULT_DIR, 'tails')
SELECTION_DIR = os.path.join(RESULT_DIR, 'selection')
TRAINING_METRICS_PATH = os.path.join(
    RESULT_DIR,
    f'{EXTERNAL_TEACHER_MODEL_NAME}_training_metrics.csv',
)
CANDIDATE_METRICS_PATH = os.path.join(
    RESULT_DIR,
    f'{EXTERNAL_TEACHER_MODEL_NAME}_candidate_metrics.csv',
)

ENABLE_EXTERNAL_TEACHER_TRAINING = os.getenv(
    'WIND_EXTERNAL_TEACHER_ENABLE_TRAINING',
    '0',
) == '1'
STRUCTURE_VALIDATION_SEED = int(os.getenv(
    'WIND_EXTERNAL_TEACHER_SEED',
    '2026',
))
PARENT_VARIANT = 'revin_balanced_loss_multiseed'
EXTERNAL_VARIANT = {
    'name': (
        'revin_balanced_loss_external_teacher_seed_'
        f'{STRUCTURE_VALIDATION_SEED}'
    ),
    'round': 7,
    'parent_variant': PARENT_VARIANT,
    'added_module': 'supplementary_supervised_teacher_pretraining',
    'use_revin': True,
    'use_cnn_adapter': False,
    'use_balanced_loss': True,
    'use_rmse_balanced_loss': False,
    'use_swa': False,
    'use_distillation': True,
    'multi_seed': False,
    'use_supplementary_teacher_pretraining': True,
}


def ensure_dirs():
    for path in [
        RESULT_DIR,
        MODEL_DIR,
        WEIGHTS_DIR,
        PREPROCESS_DIR,
        TAIL_DIR,
        SELECTION_DIR,
    ]:
        os.makedirs(path, exist_ok=True)


def load_parent_results():
    if not os.path.exists(ABLATION_ALL_METRICS_PATH):
        raise FileNotFoundError(
            f'缺少前五轮消融结果: {ABLATION_ALL_METRICS_PATH}'
        )
    detail = pd.read_csv(
        ABLATION_ALL_METRICS_PATH,
        dtype={'farm_id': str},
    )
    if 'round' in detail:
        detail = detail[pd.to_numeric(detail['round'], errors='coerce') <= 5]
    detail = detail[detail['variant'] == PARENT_VARIANT]
    if detail.empty:
        raise ValueError(f'未找到父模型variant={PARENT_VARIANT}')

    results = {}
    for record in detail.to_dict('records'):
        record['farm_id'] = str(record['farm_id'])
        record['result_source'] = 'front_five_parent_result'
        results[record['farm_id']] = record
    return results


def promote_to_external_namespace(train_df, selected_result, candidate_result,
                                  parent_result):
    farm_id = str(selected_result['farm_id'])
    selected_artifact = joblib.load(selected_result['artifact_path'])
    model = keras.models.load_model(
        selected_result['model_path'],
        compile=False,
    )

    canonical_model_path = os.path.join(
        MODEL_DIR,
        f'{EXTERNAL_TEACHER_MODEL_NAME}_farm_{farm_id}.keras',
    )
    canonical_weights_path = os.path.join(
        WEIGHTS_DIR,
        f'{EXTERNAL_TEACHER_MODEL_NAME}_farm_{farm_id}_best.weights.h5',
    )
    canonical_artifact_path = os.path.join(
        PREPROCESS_DIR,
        f'{EXTERNAL_TEACHER_MODEL_NAME}_farm_{farm_id}_preprocess.pkl',
    )
    model.save(canonical_model_path)
    model.save_weights(canonical_weights_path)

    candidate_selected = True
    selected_artifact.update({
        'model_name': EXTERNAL_TEACHER_MODEL_NAME,
        'source_model_name': TUNED_MODEL_NAME,
        'experiment_name': 'supplementary_external_teacher_v1',
        'experiment_created_at': datetime.now().isoformat(timespec='seconds'),
        'external_teacher_candidate_variant': candidate_result['variant'],
        'external_teacher_parent_variant': parent_result['variant'],
        'external_teacher_candidate_selected': candidate_selected,
        'external_teacher_fallback': False,
        'external_teacher_selection_rule': (
            'structure_validation_single_seed_no_ensemble_no_fallback'
        ),
        'external_teacher_candidate_val_nrmse': candidate_result[
            'val_capacity_normalized_rmse'
        ],
        'external_teacher_candidate_val_composite': candidate_result[
            'val_composite_score'
        ],
        'external_teacher_parent_val_nrmse': parent_result[
            'val_capacity_normalized_rmse'
        ],
        'external_teacher_parent_val_composite': parent_result[
            'val_composite_score'
        ],
        'selected_ablation_variant': selected_result['variant'],
        'selected_ablation_round': selected_result.get('round'),
        'selected_parent_variant': selected_result.get('parent_variant'),
        'selected_weight_source': selected_result.get(
            'selected_weight_source'
        ),
        'source_variant_model_path': selected_result['model_path'],
        'source_variant_weights_path': selected_result.get(
            'best_weights_path'
        ),
        'model_path': canonical_model_path,
        'best_weights_path': canonical_weights_path,
    })
    joblib.dump(selected_artifact, canonical_artifact_path)

    tail_path = os.path.join(
        TAIL_DIR,
        f'{EXTERNAL_TEACHER_MODEL_NAME}_tail_farm_{farm_id}.csv',
    )
    train_df.iloc[-HISTORY_LEN:].to_csv(tail_path, index=True)

    selection_path = os.path.join(
        SELECTION_DIR,
        f'{EXTERNAL_TEACHER_MODEL_NAME}_selection_farm_{farm_id}.json',
    )
    selection_record = {
        'farm_id': farm_id,
        'candidate_variant': candidate_result['variant'],
        'parent_variant': parent_result['variant'],
        'candidate_selected': candidate_selected,
        'selected_variant': selected_result['variant'],
        'selection_rule': selected_artifact[
            'external_teacher_selection_rule'
        ],
        'candidate_val_capacity_normalized_rmse': float(
            candidate_result['val_capacity_normalized_rmse']
        ),
        'candidate_val_composite_score': float(
            candidate_result['val_composite_score']
        ),
        'parent_val_capacity_normalized_rmse': float(
            parent_result['val_capacity_normalized_rmse']
        ),
        'parent_val_composite_score': float(
            parent_result['val_composite_score']
        ),
        'canonical_model_path': canonical_model_path,
        'canonical_weights_path': canonical_weights_path,
        'canonical_artifact_path': canonical_artifact_path,
    }
    with open(selection_path, 'w', encoding='utf-8') as file:
        json.dump(selection_record, file, ensure_ascii=False, indent=2)

    del model
    keras.backend.clear_session()
    return {
        'model_name': EXTERNAL_TEACHER_MODEL_NAME,
        'farm_id': farm_id,
        'selected_variant': selected_result['variant'],
        'candidate_selected': candidate_selected,
        'external_teacher_fallback': False,
        'selected_weight_source': selected_result.get(
            'selected_weight_source'
        ),
        'model_path': canonical_model_path,
        'best_weights_path': canonical_weights_path,
        'artifact_path': canonical_artifact_path,
        'tail_path': tail_path,
        'selection_path': selection_path,
        'candidate_val_capacity_normalized_rmse': candidate_result[
            'val_capacity_normalized_rmse'
        ],
        'candidate_val_composite_score': candidate_result[
            'val_composite_score'
        ],
        'parent_val_capacity_normalized_rmse': parent_result[
            'val_capacity_normalized_rmse'
        ],
        'parent_val_composite_score': parent_result[
            'val_composite_score'
        ],
        'selected_val_capacity_normalized_rmse': selected_result[
            'val_capacity_normalized_rmse'
        ],
        'selected_val_composite_score': selected_result[
            'val_composite_score'
        ],
    }


def train_one_farm(train_file, parent_result):
    farm_id = get_farm_id(train_file)
    print(
        f'\n===== external-teacher tuned PatchTST / 风电场 {farm_id} ====='
    )
    if parent_result is None:
        raise ValueError(f'场站 {farm_id} 缺少前五轮父模型结果')

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

    candidate = train_ablation_variant(
        farm_id,
        EXTERNAL_VARIANT,
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
        storage_name=EXTERNAL_VARIANT['name'],
    )
    selected = candidate
    print(
        f'场站 {farm_id}: 单seed结构验证模式，直接保留external teacher；'
        '父模型指标只作参考，不执行回退'
    )
    promoted = promote_to_external_namespace(
        train_df,
        selected,
        candidate,
        parent_result,
    )
    return promoted, candidate


def main():
    configure_runtime()
    set_global_seed(STRUCTURE_VALIDATION_SEED)
    if not ENABLE_EXTERNAL_TEACHER_TRAINING:
        print(
            'external-teacher训练默认关闭，未执行训练。请先运行 '
            'python wind_supplementary_preprocess.py，确认报告后设置 '
            'WIND_EXTERNAL_TEACHER_ENABLE_TRAINING=1。'
        )
        return

    ensure_dirs()
    parent_results = load_parent_results()
    train_files = discover_train_files(DATA_DIR)
    if not train_files:
        raise FileNotFoundError(
            f'未在 {DATA_DIR} 找到 {TRAIN_FILE_PATTERN}'
        )
    print(f'补充teacher结构验证seed: {STRUCTURE_VALIDATION_SEED}')
    print(f'候选模型产物目录: {ABLATION_DIR}/{EXTERNAL_VARIANT["name"]}')
    print(f'独立预测模型目录: {RESULT_DIR}')

    selected_rows = []
    candidate_rows = []
    for train_file in train_files:
        farm_id = get_farm_id(train_file)
        selected, candidate = train_one_farm(
            train_file,
            parent_results.get(str(farm_id)),
        )
        selected_rows.append(selected)
        candidate_rows.append(candidate)

    pd.DataFrame(candidate_rows).to_csv(
        CANDIDATE_METRICS_PATH,
        index=False,
        encoding='utf-8-sig',
    )
    pd.DataFrame(selected_rows).to_csv(
        TRAINING_METRICS_PATH,
        index=False,
        encoding='utf-8-sig',
    )
    print(f'external-teacher候选指标: {CANDIDATE_METRICS_PATH}')
    print(f'external-teacher选择结果: {TRAINING_METRICS_PATH}')


if __name__ == '__main__':
    main()
