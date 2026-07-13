"""RegimeEncoder-PatchTST 第二阶段显式工况特征筛选 F0--F7。

本脚本只改变 R4 显式工况编码器送入门控 MLP 的特征组，B2 两候选主干、
门控结构、损失函数、训练轮数和随机种子均保持一致。43 维特征按物理含义分为：

    P: 功率状态（20）
    H: 轮毂高度风速（12）
    M: 多高度风速（3）
    D: 风向变化（4）
    C: 功率--风速一致性（4）

固定矩阵：

    F0=P, F1=P+H, F2=P+H+M, F3=P+H+M+D,
    F4=P+H+M+D+C（直接引用既有 R4，不训练、不复制模型），
    F5=H+M+D, F6=P+M+D, F7=P+H+D。

所有新增产物写入
``wind_results/regime_encoder_patchtst/stage2_feature_screening_f0_f7``，
不会写入原 R2--R5 目录或原 ``testdata_predict_output``。

可选环境变量：

    WIND_FEATURE_SCREEN_VARIANTS=f0,f1,...,f7
    WIND_FEATURE_SCREEN_FARMS=<farm_id,...>
    WIND_FEATURE_SCREEN_BATCH_SIZE=192
    WIND_FEATURE_SCREEN_EPOCHS=60
    WIND_FEATURE_SCREEN_SAVE_SMOKE_TEST=1
"""

import glob
import hashlib
import json
import os
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

import wind_RegimeEncoder_PatchTST_train as regime_train
from wind_FeTS_PatchTST_train import (
    CORRECTION_KERNEL_L2,
    NonFiniteTrainingGuard,
    ensure_finite_training_history,
)
from wind_dl_model_train import (
    DATA_DIR,
    FORECAST_LEN,
    HISTORY_LEN,
    TARGET_COL,
    TIME_FREQ,
    make_window_dataset,
    set_global_seed,
)

warnings.filterwarnings("ignore")


MODEL_FAMILY = "regime_encoder_patchtst_feature_screen"
ARCHITECTURE_VERSION = "regime_encoder_patchtst_stage2_feature_screen_v1"
ARTIFACT_SCHEMA_VERSION = 1
EXPERIMENT_DIRNAME = "stage2_feature_screening_f0_f7"
RESULT_ROOT = os.path.join(regime_train.RESULT_ROOT, EXPERIMENT_DIRNAME)
R4_SOURCE_VARIANT = "r4_explicit_regime_gate"
RANDOM_SEED = 2026

BATCH_SIZE = int(os.getenv("WIND_FEATURE_SCREEN_BATCH_SIZE", "192"))
EPOCHS = int(os.getenv("WIND_FEATURE_SCREEN_EPOCHS", "60"))
VALIDATION_SPLIT = float(
    os.getenv("WIND_FEATURE_SCREEN_VALIDATION_SPLIT", "0.15")
)
LEARNING_RATE = float(os.getenv("WIND_FEATURE_SCREEN_LEARNING_RATE", "0.0001"))
CANDIDATE_LOSS_WEIGHT = float(
    os.getenv(
        "WIND_FEATURE_SCREEN_CANDIDATE_LOSS_WEIGHT",
        "0.50",
    )
)
IDEAL_PARAMETER_LIMIT = int(os.getenv("WIND_FEATURE_SCREEN_IDEAL_PARAMS", "30000"))
HARD_PARAMETER_LIMIT = int(os.getenv("WIND_FEATURE_SCREEN_MAX_PARAMS", "100000"))

# 与既有R4冻结一致；不继承可能残留的WIND_REGIME_*环境变量。
GATE_HIDDEN_DIM = 16
HORIZON_EMBEDDING_DIM = 8
REGIME_CONTEXT_DIM = 24
GATE_DROPOUT = 0.10
GATE_INITIAL_CORRECTED_WEIGHT = 0.95

EXPECTED_PARAMETER_COUNTS = {
    "f0": 20553,
    "f1": 20865,
    "f2": 20943,
    "f3": 21047,
    "f4": 21151,
    "f5": 20527,
    "f6": 20735,
    "f7": 20969,
}

FULL_FEATURE_NAMES = tuple(regime_train.explicit_regime_feature_names())

FEATURE_GROUPS = {
    "P": (
        "power_last",
        "power_mean_4",
        "power_mean_16",
        "power_mean_32",
        "power_slope_4",
        "power_std_4",
        "power_mean_abs_step_4",
        "power_slope_8",
        "power_std_8",
        "power_mean_abs_step_8",
        "power_slope_16",
        "power_std_16",
        "power_mean_abs_step_16",
        "power_slope_32",
        "power_std_32",
        "power_mean_abs_step_32",
        "power_range_16",
        "power_range_32",
        "power_low_fraction_16",
        "power_low_fraction_32",
    ),
    "H": (
        "hub_wind_last",
        "hub_wind_mean_4",
        "hub_wind_mean_16",
        "hub_wind_slope_4",
        "hub_wind_std_4",
        "hub_wind_slope_8",
        "hub_wind_std_8",
        "hub_wind_slope_16",
        "hub_wind_std_16",
        "hub_wind_slope_32",
        "hub_wind_std_32",
        "hub_wind_mean_abs_step_16",
    ),
    "M": (
        "all_height_wind_last_mean",
        "all_height_wind_last_std",
        "hub_minus_height_mean",
    ),
    "D": (
        "direction_turn_lag_1",
        "direction_turn_lag_4",
        "direction_turn_lag_16",
        "direction_mean_turn_16",
    ),
    "C": (
        "power_wind_slope_product_8",
        "power_wind_slope_product_16",
        "power_minus_wind_cube_proxy",
        "power_wind_change_correlation_16",
    ),
}

VARIANT_SPECS = {
    "f0": {
        "directory_name": "f0_power",
        "label": "F0 power-state features",
        "groups": ("P",),
        "requires_training": True,
        "description": "仅保留历史功率状态特征",
    },
    "f1": {
        "directory_name": "f1_power_hub",
        "label": "F1 power + hub-wind features",
        "groups": ("P", "H"),
        "requires_training": True,
        "description": "F0 加入轮毂高度风速特征",
    },
    "f2": {
        "directory_name": "f2_power_hub_multiheight",
        "label": "F2 power + hub + multi-height features",
        "groups": ("P", "H", "M"),
        "requires_training": True,
        "description": "F1 加入多高度风速特征",
    },
    "f3": {
        "directory_name": "f3_power_hub_multiheight_direction",
        "label": "F3 power + hub + multi-height + direction features",
        "groups": ("P", "H", "M", "D"),
        "requires_training": True,
        "description": "F2 加入风向变化特征",
    },
    "f4": {
        "directory_name": "f4_full_r4_reference",
        "label": "F4 full 43-feature R4 reference",
        "groups": ("P", "H", "M", "D", "C"),
        "requires_training": False,
        "description": "完整43维特征；直接引用既有R4，不重复训练",
    },
    "f5": {
        "directory_name": "f5_no_power",
        "label": "F5 hub + multi-height + direction (no power)",
        "groups": ("H", "M", "D"),
        "requires_training": True,
        "description": "无功率状态特征的反向消融",
    },
    "f6": {
        "directory_name": "f6_no_hub",
        "label": "F6 power + multi-height + direction (no hub)",
        "groups": ("P", "M", "D"),
        "requires_training": True,
        "description": "无轮毂高度风速特征的反向消融",
    },
    "f7": {
        "directory_name": "f7_no_multiheight",
        "label": "F7 power + hub + direction (no multi-height)",
        "groups": ("P", "H", "D"),
        "requires_training": True,
        "description": "无多高度风速特征的反向消融",
    },
}

TRAINABLE_VARIANTS = tuple(
    variant_id
    for variant_id, spec in VARIANT_SPECS.items()
    if spec["requires_training"]
)


def selected_feature_names(variant_id):
    """按原43维顺序返回变体真正送入编码器的特征。"""
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"未知特征筛选变体: {variant_id}")
    enabled = {
        name
        for group in VARIANT_SPECS[variant_id]["groups"]
        for name in FEATURE_GROUPS[group]
    }
    return tuple(name for name in FULL_FEATURE_NAMES if name in enabled)


def _validate_feature_matrix():
    flattened = [name for group in FEATURE_GROUPS.values() for name in group]
    if len(flattened) != 43 or len(set(flattened)) != 43:
        raise ValueError("P/H/M/D/C 特征组必须无重复地覆盖43维特征")
    if set(flattened) != set(FULL_FEATURE_NAMES):
        missing = sorted(set(FULL_FEATURE_NAMES) - set(flattened))
        extra = sorted(set(flattened) - set(FULL_FEATURE_NAMES))
        raise ValueError(f"特征组与R4定义不一致；missing={missing}, extra={extra}")
    expected_counts = {
        "f0": 20,
        "f1": 32,
        "f2": 35,
        "f3": 39,
        "f4": 43,
        "f5": 19,
        "f6": 27,
        "f7": 36,
    }
    actual_counts = {
        variant_id: len(selected_feature_names(variant_id))
        for variant_id in VARIANT_SPECS
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"F0--F7 特征维数异常: {actual_counts} != {expected_counts}"
        )


@keras.utils.register_keras_serializable(package="WindRegimeFeatureScreen")
class RegimeFeatureSubset(layers.Layer):
    """按特征名执行真实维度裁剪，而不是用零值伪消融。"""

    def __init__(
        self,
        selected_names,
        full_names=FULL_FEATURE_NAMES,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.full_names = tuple(str(name) for name in full_names)
        self.selected_names = tuple(str(name) for name in selected_names)
        if not self.full_names or not self.selected_names:
            raise ValueError("full_names 和 selected_names 不能为空")
        if len(set(self.full_names)) != len(self.full_names):
            raise ValueError("full_names 包含重复特征")
        if len(set(self.selected_names)) != len(self.selected_names):
            raise ValueError("selected_names 包含重复特征")
        missing = [name for name in self.selected_names if name not in self.full_names]
        if missing:
            raise ValueError(f"selected_names 不属于完整特征: {missing}")
        self.indices = tuple(self.full_names.index(name) for name in self.selected_names)

    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=-1)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (len(self.indices),)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "selected_names": list(self.selected_names),
                "full_names": list(self.full_names),
            }
        )
        return config


def configure_reproducibility():
    set_global_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _validate_configuration():
    _validate_feature_matrix()
    if BATCH_SIZE <= 0 or EPOCHS <= 0:
        raise ValueError("batch_size 和 epochs 必须为正整数")
    if not 0 < VALIDATION_SPLIT < 1:
        raise ValueError("validation_split 必须位于 (0, 1)")
    if LEARNING_RATE <= 0 or CANDIDATE_LOSS_WEIGHT < 0:
        raise ValueError("学习率必须为正，候选监督权重不能为负")
    if HARD_PARAMETER_LIMIT < IDEAL_PARAMETER_LIMIT:
        raise ValueError("硬参数上限不能小于理想参数上限")


def variant_model_name(variant_id):
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"未知特征筛选变体: {variant_id}")
    return f"{MODEL_FAMILY}_{variant_id}"


def variant_dirs(variant_id, create=True):
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"未知特征筛选变体: {variant_id}")
    root = os.path.join(RESULT_ROOT, VARIANT_SPECS[variant_id]["directory_name"])
    paths = {
        "root": root,
        "models": os.path.join(root, "models"),
        "weights": os.path.join(root, "weights"),
        "preprocess": os.path.join(root, "preprocess"),
        "history": os.path.join(root, "history"),
        "tensorboard": os.path.join(root, "tensorboard"),
        "tails": os.path.join(root, "tails"),
        "validation_diagnostics": os.path.join(root, "validation_diagnostics"),
    }
    if create:
        os.makedirs(RESULT_ROOT, exist_ok=True)
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
    return paths


def get_requested_variants():
    raw = os.getenv("WIND_FEATURE_SCREEN_VARIANTS")
    if not raw:
        return list(VARIANT_SPECS)
    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if any(item in {"all", "*"} for item in requested):
        return list(VARIANT_SPECS)
    invalid = sorted(set(requested) - set(VARIANT_SPECS))
    if invalid:
        raise ValueError(f"未知变体 {invalid}；可选: {list(VARIANT_SPECS)}")
    return list(dict.fromkeys(requested))


def discover_train_files(data_dir=DATA_DIR):
    # 不继承原第二阶段入口的 WIND_REGIME_FARMS，避免旧环境变量意外缩小F实验。
    files = sorted(
        glob.glob(os.path.join(data_dir, regime_train.TRAIN_FILE_PATTERN))
    )
    requested = os.getenv("WIND_FEATURE_SCREEN_FARMS")
    if not requested:
        return files
    farm_ids = {item.strip() for item in requested.split(",") if item.strip()}
    return [path for path in files if regime_train.get_farm_id(path) in farm_ids]


def _compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0,
        ),
        loss={
            "forecast_power": keras.losses.Huber(delta=1.0),
            "candidate_forecast": keras.losses.Huber(delta=1.0),
        },
        loss_weights={
            "forecast_power": 1.0,
            "candidate_forecast": CANDIDATE_LOSS_WEIGHT,
        },
        metrics={
            "forecast_power": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
            "candidate_forecast": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
        },
    )
    return model


def build_feature_screen_model(
    variant_id,
    input_dim,
    target_channel_index,
    power_scale_ratio=1.0,
    power_scale_offset=0.0,
    regime_feature_config=None,
):
    """构建F0/F1/F2/F3/F5/F6/F7；F4只能直接引用R4。"""
    if variant_id not in TRAINABLE_VARIANTS:
        raise ValueError(f"{variant_id} 不是需要训练的特征筛选变体")
    if target_channel_index is None or not 0 <= target_channel_index < input_dim:
        raise ValueError("历史功率目标通道索引无效")
    if not regime_feature_config:
        raise ValueError("缺少 regime_feature_config")

    configure_reproducibility()
    inputs = keras.Input(shape=(HISTORY_LEN, input_dim), name="history_features")
    persistence, corrected, _ = regime_train._build_b2_candidates(
        inputs,
        target_channel_index,
        power_scale_ratio,
        power_scale_offset,
    )
    full_features = regime_train.ExplicitWindRegimeFeatures(
        **regime_train._layer_feature_kwargs(regime_feature_config),
        name="explicit_regime_features_full43",
    )(inputs)
    selected_names = selected_feature_names(variant_id)
    explicit_features = RegimeFeatureSubset(
        selected_names=selected_names,
        full_names=FULL_FEATURE_NAMES,
        name="explicit_regime_features",
    )(full_features)
    normalized = layers.LayerNormalization(
        epsilon=1e-6,
        name="explicit_regime_feature_norm",
    )(explicit_features)
    context = layers.Dense(
        REGIME_CONTEXT_DIM,
        activation="gelu",
        kernel_regularizer=regularizers.l2(CORRECTION_KERNEL_L2),
        name="regime_context_hidden",
    )(normalized)
    context = layers.Dropout(
        GATE_DROPOUT,
        name="regime_context_dropout",
    )(context)
    context = layers.Dense(
        REGIME_CONTEXT_DIM,
        activation="gelu",
        name="regime_context",
    )(context)
    gate = regime_train.SampleHorizonCorrectionGate(
        forecast_len=FORECAST_LEN,
        hidden_dim=GATE_HIDDEN_DIM,
        horizon_embedding_dim=HORIZON_EMBEDDING_DIM,
        dropout=GATE_DROPOUT,
        initial_weight=GATE_INITIAL_CORRECTED_WEIGHT,
        name="correction_gate",
    )(context)
    forecast = regime_train.TwoCandidateGateFusion(name="forecast_power")(
        [persistence, corrected, gate]
    )
    candidate_forecast = layers.Activation(
        "linear",
        name="candidate_forecast",
    )(corrected)
    model = keras.Model(
        inputs=inputs,
        outputs={
            "forecast_power": forecast,
            "candidate_forecast": candidate_forecast,
        },
        name=f"WindRegimeEncoderFeatureScreen_{variant_id.upper()}",
    )
    return _compile_model(model)


def build_feature_screen_model_from_artifact(artifact):
    if artifact.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(
            "artifact 架构版本不匹配: "
            f"{artifact.get('architecture_version')} != {ARCHITECTURE_VERSION}"
        )
    variant_id = artifact.get("variant_id")
    if tuple(artifact.get("selected_regime_feature_names", ())) != (
        selected_feature_names(variant_id)
    ):
        raise ValueError(f"artifact 的 {variant_id} 特征子集与当前定义不一致")
    return build_feature_screen_model(
        variant_id=variant_id,
        input_dim=len(artifact["input_cols"]),
        target_channel_index=int(artifact["target_index"]),
        power_scale_ratio=float(artifact["power_scale_ratio"]),
        power_scale_offset=float(artifact["power_scale_offset"]),
        regime_feature_config=artifact["regime_feature_config"],
    )


def get_feature_screen_custom_objects():
    custom_objects = dict(regime_train.get_regime_custom_objects())
    custom_objects.update(
        {
            "RegimeFeatureSubset": RegimeFeatureSubset,
            "WindRegimeFeatureScreen>RegimeFeatureSubset": RegimeFeatureSubset,
        }
    )
    return custom_objects


def _attach_targets(dataset):
    def map_targets(batch_x, batch_y):
        return batch_x, {
            "forecast_power": batch_y,
            "candidate_forecast": batch_y,
        }

    return dataset.map(
        map_targets,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    ).prefetch(tf.data.AUTOTUNE)


def _make_datasets(prepared):
    train_ds, val_ds, train_samples, total_samples = make_window_dataset(
        prepared["features"],
        prepared["target"],
        HISTORY_LEN,
        FORECAST_LEN,
        BATCH_SIZE,
        VALIDATION_SPLIT,
    )
    return (
        _attach_targets(train_ds),
        _attach_targets(val_ds),
        train_samples,
        total_samples,
    )


def _assert_stage1_b2_compatible(prepared):
    source, source_path = regime_train._load_stage1_artifact(
        "b2_persistence_residual",
        prepared["farm_id"],
    )
    checks = {
        "input_cols": list(source.get("input_cols", ()))
        == list(prepared["input_cols"]),
        "target_index": int(source.get("target_index", -1))
        == int(prepared["target_index"]),
        "history_len": int(source.get("history_len", -1)) == HISTORY_LEN,
        "forecast_len": int(source.get("forecast_len", -1)) == FORECAST_LEN,
        "capacity": np.isclose(
            float(source.get("capacity", np.nan)),
            float(prepared["capacity"]),
            rtol=1e-10,
            atol=1e-8,
        ),
        "scaler_x_mean": np.allclose(
            source["scaler_x"].mean_,
            prepared["scaler_x"].mean_,
            rtol=1e-8,
            atol=1e-8,
        ),
        "scaler_x_scale": np.allclose(
            source["scaler_x"].scale_,
            prepared["scaler_x"].scale_,
            rtol=1e-8,
            atol=1e-8,
        ),
        "scaler_y_mean": np.allclose(
            source["scaler_y"].mean_,
            prepared["scaler_y"].mean_,
            rtol=1e-8,
            atol=1e-8,
        ),
        "scaler_y_scale": np.allclose(
            source["scaler_y"].scale_,
            prepared["scaler_y"].scale_,
            rtol=1e-8,
            atol=1e-8,
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"场站 {prepared['farm_id']} 与Stage-1 B2预处理不一致: {failed}"
        )
    return os.path.abspath(source_path)


def _collect_validation_diagnostics(model, val_ds, prepared, variant_id):
    diagnostics = regime_train._collect_validation_diagnostics(
        model,
        val_ds,
        prepared,
        R4_SOURCE_VARIANT,
    )
    for key in ("regime_rows", "gate_rows"):
        for row in diagnostics[key]:
            row["model_family"] = MODEL_FAMILY
            row["variant_id"] = variant_id
    return diagnostics


def _train_paths(dirs, model_name, farm_id):
    return {
        "model_path": os.path.join(dirs["models"], f"{model_name}_farm_{farm_id}.keras"),
        "best_weights_path": os.path.join(
            dirs["weights"], f"{model_name}_farm_{farm_id}_best.weights.h5"
        ),
        "artifact_path": os.path.join(
            dirs["preprocess"], f"{model_name}_farm_{farm_id}_preprocess.pkl"
        ),
        "tail_path": os.path.join(
            dirs["tails"], f"{model_name}_tail_farm_{farm_id}.csv"
        ),
    }


def _save_load_smoke_test(model, model_path, val_ds):
    if os.getenv("WIND_FEATURE_SCREEN_SAVE_SMOKE_TEST", "1") == "0":
        return
    sample_x, _ = next(iter(val_ds))
    sample_x = sample_x[:2]
    diagnostic = keras.Model(model.inputs, model.get_layer("forecast_power").output)
    expected = np.asarray(diagnostic(sample_x, training=False), dtype=float)
    restored = keras.models.load_model(
        model_path,
        custom_objects=get_feature_screen_custom_objects(),
        compile=False,
    )
    restored_diagnostic = keras.Model(
        restored.inputs,
        restored.get_layer("forecast_power").output,
    )
    actual = np.asarray(restored_diagnostic(sample_x, training=False), dtype=float)
    if not np.allclose(expected, actual, rtol=1e-6, atol=1e-6):
        raise ValueError("保存后重载模型的 forecast 输出不一致")
    del restored_diagnostic, restored, diagnostic


def _file_sha256(path, chunk_size=1024 * 1024):
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def train_variant_for_farm(variant_id, prepared):
    if variant_id not in TRAINABLE_VARIANTS:
        raise ValueError(f"引用变体 {variant_id} 不应进入训练函数")
    keras.backend.clear_session()
    configure_reproducibility()
    spec = VARIANT_SPECS[variant_id]
    model_name = variant_model_name(variant_id)
    dirs = variant_dirs(variant_id)
    paths = _train_paths(dirs, model_name, prepared["farm_id"])
    print(
        f"\n===== {spec['label']} / 风电场 {prepared['farm_id']} / "
        f"seed={RANDOM_SEED} / features={len(selected_feature_names(variant_id))} ====="
    )

    source_preprocess_path = _assert_stage1_b2_compatible(prepared)
    train_ds, val_ds, train_samples, total_samples = _make_datasets(prepared)
    model = build_feature_screen_model(
        variant_id,
        len(prepared["input_cols"]),
        prepared["target_index"],
        prepared["power_scale_ratio"],
        prepared["power_scale_offset"],
        prepared["regime_feature_config"],
    )
    sample_x, _ = next(iter(train_ds))
    backbone_source = regime_train._initialize_from_stage1_b2(
        model,
        prepared,
        sample_x[:2],
    )
    total_params = int(model.count_params())
    trainable_params = int(
        sum(int(np.prod(variable.shape)) for variable in model.trainable_weights)
    )
    if total_params > HARD_PARAMETER_LIMIT:
        raise ValueError(
            f"{variant_id} 参数量 {total_params:,} 超过硬上限 {HARD_PARAMETER_LIMIT:,}"
        )
    if total_params > IDEAL_PARAMETER_LIMIT:
        print(
            f"警告: {variant_id} 参数量 {total_params:,} 超过理想上限 "
            f"{IDEAL_PARAMETER_LIMIT:,}"
        )
    expected_params = EXPECTED_PARAMETER_COUNTS[variant_id]
    if total_params != expected_params:
        raise ValueError(
            f"{variant_id}参数量{total_params:,}与冻结实验协议"
            f"{expected_params:,}不一致；请检查输入通道或结构是否漂移"
        )

    monitor = "val_forecast_power_loss"
    tensorboard_log_dir = os.path.join(
        dirs["tensorboard"],
        f"farm_{prepared['farm_id']}",
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    guard = NonFiniteTrainingGuard()
    callbacks = [
        guard,
        keras.callbacks.TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=0,
            write_graph=True,
            update_freq="epoch",
            profile_batch=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            paths["best_weights_path"],
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]
    start_time = time.monotonic()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    ensure_finite_training_history(history, guard)
    if not os.path.exists(paths["best_weights_path"]):
        raise FileNotFoundError(f"未生成最佳权重: {paths['best_weights_path']}")
    history_path, history_plot_path = regime_train._save_history(
        history,
        dirs,
        model_name,
        prepared["farm_id"],
    )
    model.load_weights(paths["best_weights_path"])
    diagnostics = _collect_validation_diagnostics(
        model,
        val_ds,
        prepared,
        variant_id,
    )
    model.save(paths["model_path"])
    _save_load_smoke_test(model, paths["model_path"], val_ds)
    elapsed_seconds = float(time.monotonic() - start_time)

    regime_path = os.path.join(
        dirs["validation_diagnostics"],
        f"{model_name}_validation_regime_metrics_farm_{prepared['farm_id']}.csv",
    )
    pd.DataFrame(diagnostics["regime_rows"]).to_csv(
        regime_path, index=False, encoding="utf-8-sig"
    )
    gate_path = os.path.join(
        dirs["validation_diagnostics"],
        f"{model_name}_validation_gate_by_horizon_farm_{prepared['farm_id']}.csv",
    )
    pd.DataFrame(diagnostics["gate_rows"]).to_csv(
        gate_path, index=False, encoding="utf-8-sig"
    )
    prepared["train_df"].iloc[-HISTORY_LEN:].to_csv(paths["tail_path"], index=True)

    names = selected_feature_names(variant_id)
    indices = [FULL_FEATURE_NAMES.index(name) for name in names]
    model_size_bytes = os.path.getsize(paths["model_path"])
    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "model_name": model_name,
        "variant_id": variant_id,
        "variant_label": spec["label"],
        "variant_config": dict(spec),
        "architecture_version": ARCHITECTURE_VERSION,
        "farm_id": prepared["farm_id"],
        "train_file": prepared["train_file"],
        "feature_cols": prepared["feature_cols"],
        "input_cols": prepared["input_cols"],
        "target_col": TARGET_COL,
        "target_index": prepared["target_index"],
        "scaler_x": prepared["scaler_x"],
        "scaler_y": prepared["scaler_y"],
        "capacity": prepared["capacity"],
        "history_len": HISTORY_LEN,
        "forecast_len": FORECAST_LEN,
        "time_freq": TIME_FREQ,
        "random_seed": RANDOM_SEED,
        "deterministic_ops_requested": True,
        "training_mode": "stage1_b2_warm_start_feature_subset_finetune",
        "requires_keras_model": True,
        "model_kind": "keras_network",
        "gate_type": "sample_horizon_sigmoid",
        "encoder_type": "explicit_wind_regime_feature_subset",
        "auxiliary_tasks": False,
        "model_output_names": list(model.output_names),
        "forecast_output_layer_name": "forecast_power",
        "candidate_output_layer_name": "corrected_forecast_candidate",
        "diagnostic_layers": {
            "forecast": "forecast_power",
            "gate": "correction_gate",
            "persistence_candidate": "persistence_forecast_candidate",
            "corrected_candidate": "corrected_forecast_candidate",
            "explicit_features_full": "explicit_regime_features_full43",
            "explicit_features": "explicit_regime_features",
            "regime_context": "regime_context",
            "regime_class": None,
            "low_power": None,
            "change_magnitude": None,
        },
        "expert_names": ["persistence", "corrected"],
        "power_scale_ratio": prepared["power_scale_ratio"],
        "power_scale_offset": prepared["power_scale_offset"],
        "regime_feature_config": prepared["regime_feature_config"],
        "full_regime_feature_names": list(FULL_FEATURE_NAMES),
        "selected_regime_feature_names": list(names),
        "selected_regime_feature_indices": indices,
        "selected_regime_feature_groups": list(spec["groups"]),
        "selected_regime_feature_count": len(names),
        "regime_label_config": {
            "version": regime_train.REGIME_LABEL_VERSION,
            "threshold_source": "predeclared_capacity_fraction",
            "stable_change_threshold": regime_train.STABLE_CHANGE_THRESHOLD,
            "low_power_threshold": regime_train.LOW_POWER_THRESHOLD,
            "change_band_edges": list(regime_train.CHANGE_BAND_EDGES),
            "class_names": ["stable", "ramp_up", "ramp_down"],
            "future_labels_are_training_targets_only": True,
        },
        "gate_hidden_dim": GATE_HIDDEN_DIM,
        "horizon_embedding_dim": HORIZON_EMBEDDING_DIM,
        "regime_context_dim": REGIME_CONTEXT_DIM,
        "gate_dropout": GATE_DROPOUT,
        "gate_initial_corrected_weight": GATE_INITIAL_CORRECTED_WEIGHT,
        "candidate_supervision_loss_weight": CANDIDATE_LOSS_WEIGHT,
        "correction_kernel_l2": CORRECTION_KERNEL_L2,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "validation_split": VALIDATION_SPLIT,
        "learning_rate": LEARNING_RATE,
        "early_stopping_monitor": monitor,
        "total_params": total_params,
        "expected_total_params": expected_params,
        "trainable_params": trainable_params,
        "model_size_bytes": model_size_bytes,
        "training_elapsed_seconds": elapsed_seconds,
        "train_samples": train_samples,
        "val_samples": total_samples - train_samples,
        "model_path": paths["model_path"],
        "best_weights_path": paths["best_weights_path"],
        "artifact_path": paths["artifact_path"],
        "history_path": history_path,
        "history_plot_path": history_plot_path,
        "tensorboard_log_dir": tensorboard_log_dir,
        "tail_path": paths["tail_path"],
        "validation_regime_metrics_path": regime_path,
        "validation_gate_diagnostics_path": gate_path,
        "backbone_initialization": backbone_source,
        "source_preprocess_compatibility_path": source_preprocess_path,
        "evaluation_pipeline_version": regime_train.EVALUATION_PIPELINE_VERSION,
        "legacy_bidirectional_weather_imputation": True,
        "scaler_fit_scope": "full_train_file_including_validation",
        "validation_target_overlap_steps": FORECAST_LEN - 1,
        "exploratory_legacy_comparison": True,
        "selection_metric_source": "test_macro_capacity_normalized_rmse",
        "test_used_for_feature_selection": True,
        "test_is_final_blind_evaluation": False,
        "training_code_path": os.path.abspath(__file__),
        "training_code_sha256": _file_sha256(os.path.abspath(__file__)),
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(keras, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        **diagnostics["overall_metrics"],
        **diagnostics["candidate_metrics"],
        **diagnostics["persistence_metrics"],
        **diagnostics["gate_fields"],
    }
    joblib.dump(artifact, paths["artifact_path"])

    result = {
        "model_family": MODEL_FAMILY,
        "model_name": model_name,
        "variant_id": variant_id,
        "variant_label": spec["label"],
        "feature_groups": "+".join(spec["groups"]),
        "feature_count": len(names),
        "feature_names": json.dumps(names, ensure_ascii=False),
        "farm_id": prepared["farm_id"],
        "requires_training": True,
        "result_source": "stage2_feature_screen_trained",
        "source_variant": "b2_persistence_residual",
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "total_params": total_params,
        "expected_total_params": expected_params,
        "trainable_params": trainable_params,
        "model_size_bytes": model_size_bytes,
        "training_elapsed_seconds": elapsed_seconds,
        "train_samples": train_samples,
        "val_samples": total_samples - train_samples,
        **diagnostics["overall_metrics"],
        **diagnostics["candidate_metrics"],
        **diagnostics["persistence_metrics"],
        **diagnostics["gate_fields"],
        "model_path": paths["model_path"],
        "model_sha256": _file_sha256(paths["model_path"]),
        "best_weights_path": paths["best_weights_path"],
        "artifact_path": paths["artifact_path"],
        "history_path": history_path,
        "history_plot_path": history_plot_path,
        "validation_regime_metrics_path": regime_path,
        "validation_gate_diagnostics_path": gate_path,
        "source_model_path": backbone_source["source_model_path"],
        "source_artifact_path": backbone_source["source_artifact_path"],
    }
    print(
        f"{model_name} / {prepared['farm_id']}: "
        f"val NRMSE={result['val_nrmse']:.6f}, params={total_params:,}, "
        f"features={len(names)}, gate={result['gate_mean']:.4f}"
    )
    del model
    keras.backend.clear_session()
    return result


def _r4_training_summary_path():
    source_name = regime_train.variant_model_name(R4_SOURCE_VARIANT)
    return os.path.join(
        regime_train.variant_dirs(R4_SOURCE_VARIANT, create=False)["root"],
        f"{source_name}_training_metrics.csv",
    )


def validate_r4_reference_artifact(artifact, artifact_path="<memory>"):
    """验证F4确为当前冻结协议下的完整R4，而非同名但漂移的模型。"""
    source_feature_names = tuple(
        artifact.get("regime_feature_config", {}).get("feature_names", ())
    )
    if source_feature_names != FULL_FEATURE_NAMES:
        raise ValueError(
            f"F4/R4源artifact不是当前43维完整工况定义: {artifact_path}"
        )
    if artifact.get("variant_id") != R4_SOURCE_VARIANT:
        raise ValueError(f"F4/R4源artifact变体不匹配: {artifact_path}")
    if artifact.get("architecture_version") != regime_train.ARCHITECTURE_VERSION:
        raise ValueError(f"F4/R4源artifact架构版本不匹配: {artifact_path}")
    if int(artifact.get("random_seed", -1)) != RANDOM_SEED:
        raise ValueError(f"F4/R4源artifact seed不是{RANDOM_SEED}: {artifact_path}")
    expected = {
        "total_params": EXPECTED_PARAMETER_COUNTS["f4"],
        "gate_hidden_dim": GATE_HIDDEN_DIM,
        "horizon_embedding_dim": HORIZON_EMBEDDING_DIM,
        "regime_context_dim": REGIME_CONTEXT_DIM,
        "gate_dropout": GATE_DROPOUT,
        "gate_initial_corrected_weight": GATE_INITIAL_CORRECTED_WEIGHT,
        "candidate_supervision_loss_weight": CANDIDATE_LOSS_WEIGHT,
    }
    mismatched = []
    for key, expected_value in expected.items():
        actual = artifact.get(key)
        if isinstance(expected_value, float):
            matches = actual is not None and np.isclose(
                float(actual), expected_value, rtol=0.0, atol=1e-12
            )
        else:
            matches = actual is not None and int(actual) == expected_value
        if not matches:
            mismatched.append(f"{key}={actual!r} (expected {expected_value!r})")
    if mismatched:
        raise ValueError(
            f"F4/R4源artifact超参数已漂移: {artifact_path}: "
            + "; ".join(mismatched)
        )


def load_f4_training_reference(farm_ids):
    """只读取R4训练产物；不创建F4模型、权重或副本。"""
    path = _r4_training_summary_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"F4引用所需R4训练汇总不存在: {path}")
    source = pd.read_csv(path)
    source["farm_id"] = source["farm_id"].astype(str)
    source = source[source["farm_id"].isin(set(map(str, farm_ids)))].copy()
    if source["farm_id"].nunique() != len(set(map(str, farm_ids))):
        raise ValueError("R4训练汇总未覆盖所有请求场站")
    if (pd.to_numeric(source["total_params"], errors="coerce") != 21151).any():
        raise ValueError("F4引用的R4参数量不是预期的21,151")

    resolved_artifacts = []
    resolved_models = []
    for _, row in source.iterrows():
        artifact_path = regime_train._resolve_existing_path(row.get("artifact_path"))
        model_path = regime_train._resolve_existing_path(row.get("model_path"))
        if artifact_path is None or model_path is None:
            raise FileNotFoundError(
                f"F4/R4源artifact或模型不存在: farm={row['farm_id']}"
            )
        artifact = joblib.load(artifact_path)
        validate_r4_reference_artifact(artifact, artifact_path)
        resolved_artifacts.append(os.path.abspath(artifact_path))
        resolved_models.append(os.path.abspath(model_path))

    source_names = source["model_name"].copy()
    source_artifacts = pd.Series(resolved_artifacts, index=source.index)
    source_models = pd.Series(resolved_models, index=source.index)
    source["source_model_name"] = source_names
    source["source_model_variant"] = R4_SOURCE_VARIANT
    source["source_training_summary_path"] = os.path.abspath(path)
    source["source_artifact_path"] = source_artifacts
    source["source_model_path"] = source_models
    source["source_model_sha256"] = source_models.map(_file_sha256)
    source["model_family"] = MODEL_FAMILY
    source["model_name"] = variant_model_name("f4")
    source["variant_id"] = "f4"
    source["variant_label"] = VARIANT_SPECS["f4"]["label"]
    source["feature_groups"] = "+".join(VARIANT_SPECS["f4"]["groups"])
    source["feature_count"] = len(FULL_FEATURE_NAMES)
    source["feature_names"] = json.dumps(FULL_FEATURE_NAMES, ensure_ascii=False)
    source["expected_total_params"] = EXPECTED_PARAMETER_COUNTS["f4"]
    source["requires_training"] = False
    source["result_source"] = "direct_reference_existing_r4"
    source["source_variant"] = R4_SOURCE_VARIANT
    source["reference_only"] = True
    source["training_elapsed_seconds"] = 0.0
    # artifact_path/model_path仍保留源路径，明确表示引用而非新副本。
    return source


def _write_manifest():
    os.makedirs(RESULT_ROOT, exist_ok=True)
    rows = []
    for order, (variant_id, spec) in enumerate(VARIANT_SPECS.items()):
        names = selected_feature_names(variant_id)
        rows.append(
            {
                "variant_order": order,
                "variant_id": variant_id,
                "model_name": variant_model_name(variant_id),
                "directory_name": spec["directory_name"],
                "label": spec["label"],
                "feature_groups": "+".join(spec["groups"]),
                "feature_group_count": len(spec["groups"]),
                "feature_count": len(names),
                "feature_names": json.dumps(names, ensure_ascii=False),
                "has_power_group": "P" in spec["groups"],
                "has_hub_wind_group": "H" in spec["groups"],
                "has_multiheight_group": "M" in spec["groups"],
                "has_direction_group": "D" in spec["groups"],
                "has_consistency_group": "C" in spec["groups"],
                "requires_training": spec["requires_training"],
                "result_source": (
                    "stage2_feature_screen_trained"
                    if spec["requires_training"]
                    else "direct_reference_existing_r4"
                ),
                "source_variant": (
                    "b2_persistence_residual"
                    if spec["requires_training"]
                    else R4_SOURCE_VARIANT
                ),
                "description": spec["description"],
                "random_seed": RANDOM_SEED,
                "default_batch_size": BATCH_SIZE,
                "expected_parameter_count": EXPECTED_PARAMETER_COUNTS[variant_id],
                "selection_split": "test",
                "selection_metric": "macro_mean_capacity_normalized_rmse",
                "test_used_for_feature_selection": True,
                "test_is_final_blind_evaluation": False,
                "source_test_reuse_status": "legacy_seen",
                "training_code_path": os.path.abspath(__file__),
                "prediction_code_path": os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        "wind_RegimeEncoder_PatchTST_feature_screen_predict.py",
                    )
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULT_ROOT, "feature_screening_experiment_manifest.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def _validation_comparison(metrics_df, variants):
    rows = []
    for order, variant_id in enumerate(variants):
        frame = metrics_df[metrics_df["variant_id"] == variant_id]
        nrmse = pd.to_numeric(frame["val_nrmse"], errors="coerce")
        params = pd.to_numeric(frame["total_params"], errors="coerce")
        rows.append(
            {
                "variant_order": order,
                "variant_id": variant_id,
                "feature_groups": "+".join(VARIANT_SPECS[variant_id]["groups"]),
                "feature_count": len(selected_feature_names(variant_id)),
                "farm_count": int(frame["farm_id"].astype(str).nunique()),
                "parameter_count_max": (
                    int(params.max()) if params.notna().any() else np.nan
                ),
                "macro_val_nrmse_descriptive": float(nrmse.mean()),
                "std_val_nrmse_descriptive": float(nrmse.std(ddof=0)),
                "used_for_final_feature_selection": False,
                "final_selection_source": "test",
            }
        )
    return pd.DataFrame(rows)


def _partial_tag(variants, farm_ids):
    variant_tag = "-".join(variants)
    farm_tag = "-".join(str(value) for value in farm_ids)
    raw = f"{variant_tag}__farms_{farm_tag}"
    return raw if len(raw) <= 150 else hashlib.sha1(raw.encode()).hexdigest()[:12]


def main():
    _validate_configuration()
    configure_reproducibility()
    _write_manifest()
    variants = get_requested_variants()
    train_files = discover_train_files(DATA_DIR)
    if not train_files:
        raise FileNotFoundError(f"未在 {DATA_DIR} 找到请求的训练文件")
    farm_ids = [regime_train.get_farm_id(path) for path in train_files]
    trainable = [variant_id for variant_id in variants if variant_id in TRAINABLE_VARIANTS]
    print(f"固定随机种子: {RANDOM_SEED}；batch_size={BATCH_SIZE}")
    print(f"场站数: {len(train_files)}；F矩阵: {variants}")
    print(f"实际新增训练: {trainable}")
    if "f4" in variants:
        print("F4直接引用既有R4模型/权重/结果，不重复训练或复制")

    results = []
    if "f4" in variants:
        results.extend(load_f4_training_reference(farm_ids).to_dict("records"))

    progress_path = os.path.join(
        RESULT_ROOT,
        f"feature_screening_training_progress_{_partial_tag(variants, farm_ids)}.csv",
    )
    for train_file in train_files:
        if not trainable:
            break
        prepared = regime_train._prepare_farm(train_file)
        for variant_id in trainable:
            results.append(train_variant_for_farm(variant_id, prepared))
            pd.DataFrame(results).to_csv(
                progress_path,
                index=False,
                encoding="utf-8-sig",
            )

    metrics_df = pd.DataFrame(results)
    all_train_files = sorted(glob.glob(os.path.join(DATA_DIR, regime_train.TRAIN_FILE_PATTERN)))
    expected_rows = len(VARIANT_SPECS) * len(all_train_files)
    is_complete = (
        set(variants) == set(VARIANT_SPECS)
        and not os.getenv("WIND_FEATURE_SCREEN_FARMS")
        and len(metrics_df) == expected_rows
        and not metrics_df.duplicated(["variant_id", "farm_id"]).any()
    )
    suffix = "" if is_complete else f"_partial_{_partial_tag(variants, farm_ids)}"
    metrics_path = os.path.join(
        RESULT_ROOT,
        f"feature_screening_training_metrics{suffix}.csv",
    )
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    comparison = _validation_comparison(metrics_df, variants)
    comparison.to_csv(
        os.path.join(
            RESULT_ROOT,
            f"feature_screening_validation_descriptive{suffix}.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"F0--F7训练/引用汇总已保存: {metrics_path}")
    if not is_complete:
        print("当前是子集运行；文件名带partial标签，不会覆盖完整F矩阵汇总")
    print("最终模型不在此处按验证集选择；请运行专用预测脚本按测试集宏平均NRMSE筛选")


if __name__ == "__main__":
    main()
