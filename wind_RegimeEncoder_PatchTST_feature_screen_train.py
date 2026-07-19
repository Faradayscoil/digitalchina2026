"""RegimeEncoder-PatchTST 第二阶段显式工况特征筛选 F0--F8。

本脚本只改变 R4 显式工况编码器送入门控 MLP 的特征组，B2 两候选主干、
门控结构、损失函数、训练轮数和随机种子均保持一致。43 维特征按物理含义分为：

    P: 功率状态（20）
    H: 轮毂高度风速（12）
    M: 多高度风速（3）
    D: 风向变化（4）
    C: 功率--风速一致性（4）

原始矩阵与本次补充：

    F0=P, F1=P+H, F2=P+H+M, F3=P+H+M+D,
    F4=P+H+M+D+C（直接引用既有 R4，不训练、不复制模型），
    F5=H+M+D, F6=P+M+D, F7=P+H+D,
    F8=P+H+D+C（无M条件下检验C）。

为排除联合微调造成的 corrected candidate drift，另设两个不参与最终模型排名的
冻结候选探针：FP0=P+H+D、FP4=P+H+D+C。两者都复制并冻结同一场站的
Stage-1 B2 persistence/corrected candidate，只训练显式工况编码器和门控。
因此 FP0--FP4 的差异可归因于4个C特征对路由的作用。

默认运行只新增 F8/FP0/FP4 共15个场站模型；F0--F7训练结果从既有汇总只读
复用，绝不再次训练。不会执行多seed实验。

所有新增产物写入
``wind_results/regime_encoder_patchtst/stage2_feature_screening_f0_f7``，
不会写入原 R2--R5 目录或原 ``testdata_predict_output``。

变体/场站可筛选；训练超参数被协议锁定（偏离下列值会拒绝运行）：

    WIND_FEATURE_SCREEN_VARIANTS=f0,...,f8,fp0,fp4
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
from datetime import datetime, timezone

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
SUPPLEMENT_PROTOCOL_VERSION = "feature_screen_f8_fp_frozen_b2_v1"
EXPERIMENT_DIRNAME = "stage2_feature_screening_f0_f7"
RESULT_ROOT = os.path.join(regime_train.RESULT_ROOT, EXPERIMENT_DIRNAME)
R4_SOURCE_VARIANT = "r4_explicit_regime_gate"
RANDOM_SEED = 2026
EXPECTED_FARM_COUNT = 5

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
LEGACY_JOINT_PROTOCOL = {
    "batch_size": 192,
    "epochs": 60,
    "validation_split": 0.15,
    "learning_rate": 1e-4,
    "candidate_supervision_loss_weight": 0.50,
    "random_seed": RANDOM_SEED,
}
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
    "f8": 21073,
    "fp0": 20969,
    "fp4": 21073,
}

EXPECTED_TRAINABLE_PARAMETER_COUNTS = {
    **EXPECTED_PARAMETER_COUNTS,
    "fp0": 2553,
    "fp4": 2657,
}

LEGACY_SELECTION_VARIANTS = tuple(f"f{index}" for index in range(8))
SELECTION_VARIANTS = tuple(f"f{index}" for index in range(9))
PROBE_VARIANTS = ("fp0", "fp4")
NEW_TRAINING_VARIANTS = ("f8", *PROBE_VARIANTS)
REUSED_TRAINING_VARIANTS = LEGACY_SELECTION_VARIANTS
FROZEN_B2_PARAMETER_COUNT = 18416
LEGACY_TRAINING_SUMMARY = os.path.join(
    RESULT_ROOT,
    "feature_screening_training_metrics.csv",
)
EXTENDED_TRAINING_SUMMARY_NAME = (
    "feature_screening_f0_f8_probe_training_metrics.csv"
)
EXTENDED_MANIFEST_NAME = "feature_screening_f0_f8_probe_experiment_manifest.csv"
EXTENDED_VALIDATION_NAME = (
    "feature_screening_f0_f8_probe_validation_descriptive.csv"
)
EXTENDED_PROGRESS_PREFIX = "feature_screening_f0_f8_probe_training_progress"
TRAINING_COMPLETION_NAME = (
    "feature_screening_f0_f8_probe_training_bundle_complete.json"
)
PREDICTION_COMPLETION_RELATIVE_PATH = os.path.join(
    "f0_f8_probe_analysis_output",
    "feature_screening_f0_f8_fp_bundle_complete.json",
)

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
    "f8": {
        "directory_name": "f8_no_multiheight_with_consistency",
        "label": "F8 power + hub + direction + consistency (no multi-height)",
        "groups": ("P", "H", "D", "C"),
        "requires_training": True,
        "freeze_candidates": False,
        "selection_eligible": True,
        "experiment_role": "feature_candidate",
        "description": "在F7上加入C，检验无M条件下一致性特征的独立贡献",
    },
    "fp0": {
        "directory_name": "fp0_frozen_candidate_phd",
        "label": "FP0 frozen B2 candidate + P+H+D gate",
        "groups": ("P", "H", "D"),
        "requires_training": True,
        "freeze_candidates": True,
        "selection_eligible": False,
        "experiment_role": "frozen_candidate_probe",
        "description": "冻结同一B2候选；0个C特征的门控探针",
    },
    "fp4": {
        "directory_name": "fp4_frozen_candidate_phdc",
        "label": "FP4 frozen B2 candidate + P+H+D+C gate",
        "groups": ("P", "H", "D", "C"),
        "requires_training": True,
        "freeze_candidates": True,
        "selection_eligible": False,
        "experiment_role": "frozen_candidate_probe",
        "description": "冻结同一B2候选；加入4个C特征的门控探针",
    },
}

for _variant_id, _spec in VARIANT_SPECS.items():
    _spec.setdefault("freeze_candidates", False)
    _spec.setdefault("selection_eligible", _variant_id in SELECTION_VARIANTS)
    _spec.setdefault("experiment_role", "feature_candidate")
    _spec.setdefault(
        "reuse_existing",
        _variant_id in REUSED_TRAINING_VARIANTS,
    )

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
        "f8": 40,
        "fp0": 36,
        "fp4": 40,
    }
    actual_counts = {
        variant_id: len(selected_feature_names(variant_id))
        for variant_id in VARIANT_SPECS
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"F0--F8/FP 特征维数异常: {actual_counts} != {expected_counts}"
        )
    if set(EXPECTED_PARAMETER_COUNTS) != set(VARIANT_SPECS):
        raise ValueError("总参数量冻结表没有完整覆盖全部特征变体")
    if set(EXPECTED_TRAINABLE_PARAMETER_COUNTS) != set(VARIANT_SPECS):
        raise ValueError("可训练参数量冻结表没有完整覆盖全部特征变体")
    for variant_id in PROBE_VARIANTS:
        frozen_count = (
            EXPECTED_PARAMETER_COUNTS[variant_id]
            - EXPECTED_TRAINABLE_PARAMETER_COUNTS[variant_id]
        )
        if frozen_count != FROZEN_B2_PARAMETER_COUNT:
            raise ValueError(
                f"{variant_id}冻结参数量{frozen_count:,}不是Stage-1 B2的"
                f"{FROZEN_B2_PARAMETER_COUNT:,}"
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
    if os.getenv("WIND_FEATURE_SCREEN_SAVE_SMOKE_TEST", "1") != "1":
        raise ValueError("正式F8/FP训练必须启用保存后重载smoke test")
    if BATCH_SIZE <= 0 or EPOCHS <= 0:
        raise ValueError("batch_size 和 epochs 必须为正整数")
    if not 0 < VALIDATION_SPLIT < 1:
        raise ValueError("validation_split 必须位于 (0, 1)")
    if LEARNING_RATE <= 0 or CANDIDATE_LOSS_WEIGHT < 0:
        raise ValueError("学习率必须为正，候选监督权重不能为负")
    if HARD_PARAMETER_LIMIT < IDEAL_PARAMETER_LIMIT:
        raise ValueError("硬参数上限不能小于理想参数上限")
    actual_protocol = {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "validation_split": VALIDATION_SPLIT,
        "learning_rate": LEARNING_RATE,
        "candidate_supervision_loss_weight": CANDIDATE_LOSS_WEIGHT,
        "random_seed": RANDOM_SEED,
    }
    mismatched = []
    for key, expected in LEGACY_JOINT_PROTOCOL.items():
        actual = actual_protocol[key]
        if isinstance(expected, float):
            matches = np.isclose(float(actual), expected, rtol=0.0, atol=1e-12)
        else:
            matches = int(actual) == int(expected)
        if not matches:
            mismatched.append(f"{key}={actual!r} (expected {expected!r})")
    if mismatched:
        raise ValueError(
            "F8/FP必须与既有F0--F7使用同一训练协议；"
            "请清除残留的WIND_FEATURE_SCREEN_*训练覆盖: "
            + "; ".join(mismatched)
        )


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


def expected_training_farm_ids():
    if not os.path.isfile(LEGACY_TRAINING_SUMMARY):
        raise FileNotFoundError(
            f"无法锁定正式训练场站，缺少旧summary: {LEGACY_TRAINING_SUMMARY}"
        )
    frame = pd.read_csv(
        LEGACY_TRAINING_SUMMARY,
        usecols=["variant_id", "farm_id"],
    )
    frame["variant_id"] = frame["variant_id"].astype(str)
    frame["farm_id"] = frame["farm_id"].astype(str)
    expected = None
    for variant_id in LEGACY_SELECTION_VARIANTS:
        farms = set(frame.loc[frame["variant_id"] == variant_id, "farm_id"])
        if expected is None:
            expected = farms
        elif farms != expected:
            raise ValueError("旧F0--F7训练summary场站集合不一致")
    expected = expected or set()
    if (
        len(expected) != EXPECTED_FARM_COUNT
        or len(frame) != len(LEGACY_SELECTION_VARIANTS) * EXPECTED_FARM_COUNT
        or frame.duplicated(["variant_id", "farm_id"]).any()
    ):
        raise ValueError("旧F0--F7训练summary不是8×5唯一完整矩阵")
    return tuple(sorted(expected))


def candidate_loss_weight_for_variant(variant_id):
    """冻结候选探针不对不可训练的corrected candidate重复施加常数损失。"""
    return 0.0 if VARIANT_SPECS[variant_id]["freeze_candidates"] else CANDIDATE_LOSS_WEIGHT


def _configure_candidate_trainability(model, variant_id):
    """在compile前锁定FP候选，并关闭不会受``trainable``影响的Dropout。"""
    freeze_candidates = bool(VARIANT_SPECS[variant_id]["freeze_candidates"])
    for layer_name in regime_train.B2_WEIGHTED_LAYER_NAMES:
        model.get_layer(layer_name).trainable = not freeze_candidates
    residual_dropout = model.get_layer("residual_dropout")
    if freeze_candidates:
        # Dropout没有权重；只设置trainable=False并不会阻止fit时随机丢弃。
        # rate=0使训练期与Stage-1 B2的确定性推理候选完全一致。
        residual_dropout.rate = 0.0
        residual_dropout.trainable = False
    return freeze_candidates


def _compile_model(model, candidate_loss_weight=CANDIDATE_LOSS_WEIGHT):
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
            "candidate_forecast": float(candidate_loss_weight),
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
    """构建可训练F/FP变体；F4只能直接引用既有R4。"""
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
    _configure_candidate_trainability(model, variant_id)
    return _compile_model(
        model,
        candidate_loss_weight=candidate_loss_weight_for_variant(variant_id),
    )


def build_feature_screen_model_from_artifact(artifact):
    if artifact.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(
            "artifact 架构版本不匹配: "
            f"{artifact.get('architecture_version')} != {ARCHITECTURE_VERSION}"
        )
    variant_id = artifact.get("variant_id")
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"artifact包含未知特征变体: {variant_id}")
    if tuple(artifact.get("selected_regime_feature_names", ())) != (
        selected_feature_names(variant_id)
    ):
        raise ValueError(f"artifact 的 {variant_id} 特征子集与当前定义不一致")
    expected_freeze = bool(VARIANT_SPECS[variant_id]["freeze_candidates"])
    artifact_freeze = bool(
        artifact.get(
            "freeze_candidates",
            artifact.get("backbone_frozen", False),
        )
    )
    if artifact_freeze != expected_freeze:
        raise ValueError(
            f"artifact 的 {variant_id} 冻结协议不一致: "
            f"{artifact_freeze} != {expected_freeze}"
        )
    expected_candidate_weight = candidate_loss_weight_for_variant(variant_id)
    artifact_candidate_weight = float(
        artifact.get(
            "candidate_supervision_loss_weight",
            CANDIDATE_LOSS_WEIGHT,
        )
    )
    if not np.isclose(
        artifact_candidate_weight,
        expected_candidate_weight,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"artifact 的 {variant_id} candidate loss权重不一致: "
            f"{artifact_candidate_weight} != {expected_candidate_weight}"
        )
    if expected_freeze and not bool(
        artifact.get("residual_dropout_disabled_for_frozen_candidate", False)
    ):
        raise ValueError(f"artifact 的 {variant_id} 未声明关闭residual dropout")
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


def _save_load_smoke_test(model, model_path, val_ds, variant_id):
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
    restored_trainable_params = int(
        sum(int(np.prod(variable.shape)) for variable in restored.trainable_weights)
    )
    expected_trainable_params = EXPECTED_TRAINABLE_PARAMETER_COUNTS[variant_id]
    if restored_trainable_params != expected_trainable_params:
        raise ValueError(
            f"保存后重载{variant_id}可训练参数量{restored_trainable_params:,} != "
            f"{expected_trainable_params:,}"
        )
    _validate_frozen_candidate_configuration(restored, variant_id)
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


def _atomic_to_csv(frame, path, **kwargs):
    temporary = f"{path}.tmp"
    try:
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
            **kwargs,
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def _atomic_joblib_dump(value, path):
    temporary = f"{path}.tmp"
    try:
        joblib.dump(value, temporary)
        restored = joblib.load(temporary)
        if isinstance(value, dict) and not isinstance(restored, dict):
            raise TypeError(f"joblib临时artifact重载类型异常: {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def _atomic_write_json(value, path):
    temporary = f"{path}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def _training_completion_path():
    return os.path.join(RESULT_ROOT, TRAINING_COMPLETION_NAME)


def _clear_training_completion_marker():
    path = _training_completion_path()
    if os.path.exists(path):
        os.remove(path)


def _clear_downstream_prediction_completion_marker():
    path = os.path.join(RESULT_ROOT, PREDICTION_COMPLETION_RELATIVE_PATH)
    if os.path.exists(path):
        os.remove(path)


def _publish_training_completion_marker(
    metrics_df,
    metrics_path,
    validation_path,
):
    expected_new_rows = len(NEW_TRAINING_VARIANTS) * len(
        expected_training_farm_ids()
    )
    new_rows = metrics_df[
        metrics_df["variant_id"].isin(NEW_TRAINING_VARIANTS)
    ].copy()
    if (
        len(new_rows) != expected_new_rows
        or new_rows.duplicated(["variant_id", "farm_id"]).any()
    ):
        raise ValueError("训练bundle完成标志前，F8/FP0/FP4产物矩阵不完整")
    files = {
        "extended_manifest": os.path.join(RESULT_ROOT, EXTENDED_MANIFEST_NAME),
        "extended_training_summary": metrics_path,
        "extended_validation_descriptive": validation_path,
        "legacy_f0_f7_training_summary": LEGACY_TRAINING_SUMMARY,
    }
    for _, row in new_rows.iterrows():
        prefix = f"{row['variant_id']}.{row['farm_id']}"
        for field in ("model_path", "artifact_path", "best_weights_path"):
            path = row.get(field)
            if not isinstance(path, str) or not os.path.isfile(path):
                raise FileNotFoundError(f"训练bundle缺少{prefix}.{field}: {path}")
            files[f"{prefix}.{field}"] = path
        if _file_sha256(row["model_path"]) != row.get("model_sha256"):
            raise ValueError(f"训练bundle模型hash不一致: {prefix}")
        if _file_sha256(row["best_weights_path"]) != row.get(
            "best_weights_sha256"
        ):
            raise ValueError(f"训练bundle权重hash不一致: {prefix}")
        artifact = joblib.load(row["artifact_path"])
        if (
            artifact.get("variant_id") != row["variant_id"]
            or str(artifact.get("farm_id")) != str(row["farm_id"])
            or artifact.get("model_sha256") != row.get("model_sha256")
            or artifact.get("best_weights_sha256")
            != row.get("best_weights_sha256")
        ):
            raise ValueError(f"训练bundle artifact身份/hash不一致: {prefix}")
    hashed_files = {}
    for name, path in files.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"训练bundle缺少{name}: {path}")
        hashed_files[name] = {
            "path": os.path.abspath(path),
            "sha256": _file_sha256(path),
            "size_bytes": os.path.getsize(path),
        }
    payload = {
        "status": "complete",
        "supplement_protocol_version": SUPPLEMENT_PROTOCOL_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "new_training_variants": list(NEW_TRAINING_VARIANTS),
        "expected_farm_ids": list(expected_training_farm_ids()),
        "new_model_count": int(len(new_rows)),
        "reused_f0_f7_model_count": int(
            metrics_df["variant_id"].isin(REUSED_TRAINING_VARIANTS).sum()
        ),
        "multi_seed_experiment_run": False,
        "files": hashed_files,
    }
    return _atomic_write_json(payload, _training_completion_path())


def _numpy_values_sha256(named_values):
    """对具名numpy数组做包含名称、形状和dtype的稳定哈希。"""
    digest = hashlib.sha256()
    for name, values in named_values:
        value = np.ascontiguousarray(np.asarray(values))
        digest.update(str(name).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _snapshot_b2_weighted_layers(model):
    snapshot = {}
    for layer_name in regime_train.B2_WEIGHTED_LAYER_NAMES:
        snapshot[layer_name] = tuple(
            np.array(value, copy=True)
            for value in model.get_layer(layer_name).get_weights()
        )
    return snapshot


def _snapshot_sha256(snapshot):
    named_values = []
    for layer_name in regime_train.B2_WEIGHTED_LAYER_NAMES:
        for weight_index, value in enumerate(snapshot[layer_name]):
            named_values.append((f"{layer_name}:{weight_index}", value))
    return _numpy_values_sha256(named_values)


def _assert_exact_snapshot_match(before, after, variant_id):
    changed = []
    for layer_name in regime_train.B2_WEIGHTED_LAYER_NAMES:
        before_values = before.get(layer_name, ())
        after_values = after.get(layer_name, ())
        if len(before_values) != len(after_values):
            changed.append(f"{layer_name}:weight_count")
            continue
        for weight_index, (before_value, after_value) in enumerate(
            zip(before_values, after_values)
        ):
            if not np.array_equal(before_value, after_value):
                changed.append(f"{layer_name}:{weight_index}")
    if changed:
        raise ValueError(
            f"{variant_id}冻结B2权重在训练前后发生变化: {changed}"
        )


def _corrected_candidate_values(model, sample_x):
    diagnostic = keras.Model(
        model.inputs,
        model.get_layer("corrected_forecast_candidate").output,
    )
    values = np.asarray(diagnostic(sample_x, training=False))
    del diagnostic
    return values


def _validate_frozen_candidate_configuration(model, variant_id):
    if variant_id not in PROBE_VARIANTS:
        return
    incorrectly_trainable = [
        layer_name
        for layer_name in regime_train.B2_WEIGHTED_LAYER_NAMES
        if model.get_layer(layer_name).trainable
    ]
    if incorrectly_trainable:
        raise ValueError(
            f"{variant_id}仍有可训练B2层: {incorrectly_trainable}"
        )
    residual_dropout = model.get_layer("residual_dropout")
    if not np.isclose(float(residual_dropout.rate), 0.0, rtol=0.0, atol=0.0):
        raise ValueError(f"{variant_id}未关闭residual_dropout")
    frozen_params = sum(
        int(np.prod(weight.shape))
        for layer_name in regime_train.B2_WEIGHTED_LAYER_NAMES
        for weight in model.get_layer(layer_name).weights
    )
    if frozen_params != FROZEN_B2_PARAMETER_COUNT:
        raise ValueError(
            f"{variant_id}冻结B2参数量{frozen_params:,} != "
            f"{FROZEN_B2_PARAMETER_COUNT:,}"
        )


def train_variant_for_farm(variant_id, prepared):
    if variant_id not in NEW_TRAINING_VARIANTS:
        raise ValueError(
            f"补充协议只允许新增训练{NEW_TRAINING_VARIANTS}；"
            f"{variant_id}必须从既有F0--F7 summary只读复用"
        )
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
    freeze_candidates = bool(spec["freeze_candidates"])
    candidate_loss_weight = candidate_loss_weight_for_variant(variant_id)
    _validate_frozen_candidate_configuration(model, variant_id)
    frozen_snapshot_before = None
    frozen_weights_sha256_before = None
    candidate_values_before = None
    candidate_output_sha256_before = None
    if freeze_candidates:
        frozen_snapshot_before = _snapshot_b2_weighted_layers(model)
        frozen_weights_sha256_before = _snapshot_sha256(frozen_snapshot_before)
        candidate_values_before = _corrected_candidate_values(model, sample_x[:2])
        candidate_output_sha256_before = _numpy_values_sha256(
            [("corrected_candidate", candidate_values_before)]
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
    expected_trainable_params = EXPECTED_TRAINABLE_PARAMETER_COUNTS[variant_id]
    if trainable_params != expected_trainable_params:
        raise ValueError(
            f"{variant_id}可训练参数量{trainable_params:,}与冻结实验协议"
            f"{expected_trainable_params:,}不一致"
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
    frozen_weights_sha256_after = None
    frozen_weights_exact_match = None
    candidate_output_sha256_after = None
    candidate_output_exact_match = None
    post_training_candidate_max_abs_error = None
    if freeze_candidates:
        frozen_snapshot_after = _snapshot_b2_weighted_layers(model)
        frozen_weights_sha256_after = _snapshot_sha256(frozen_snapshot_after)
        _assert_exact_snapshot_match(
            frozen_snapshot_before,
            frozen_snapshot_after,
            variant_id,
        )
        frozen_weights_exact_match = bool(
            frozen_weights_sha256_before == frozen_weights_sha256_after
        )
        if not frozen_weights_exact_match:
            raise ValueError(f"{variant_id}冻结B2权重哈希在训练前后不一致")
        candidate_values_after = _corrected_candidate_values(model, sample_x[:2])
        candidate_output_sha256_after = _numpy_values_sha256(
            [("corrected_candidate", candidate_values_after)]
        )
        candidate_output_exact_match = bool(
            np.array_equal(candidate_values_before, candidate_values_after)
        )
        post_training_candidate_max_abs_error = float(
            np.max(np.abs(candidate_values_before - candidate_values_after))
        )
        if not candidate_output_exact_match:
            raise ValueError(
                f"{variant_id}冻结corrected candidate训练前后不完全一致，"
                f"最大误差={post_training_candidate_max_abs_error}"
            )
    diagnostics = _collect_validation_diagnostics(
        model,
        val_ds,
        prepared,
        variant_id,
    )
    model_stem, model_extension = os.path.splitext(paths["model_path"])
    temporary_model_path = f"{model_stem}.tmp{model_extension}"
    try:
        model.save(temporary_model_path)
        _save_load_smoke_test(
            model,
            temporary_model_path,
            val_ds,
            variant_id,
        )
        os.replace(temporary_model_path, paths["model_path"])
    finally:
        if os.path.isfile(temporary_model_path):
            os.remove(temporary_model_path)
    elapsed_seconds = float(time.monotonic() - start_time)

    regime_path = os.path.join(
        dirs["validation_diagnostics"],
        f"{model_name}_validation_regime_metrics_farm_{prepared['farm_id']}.csv",
    )
    _atomic_to_csv(
        pd.DataFrame(diagnostics["regime_rows"]),
        regime_path,
    )
    gate_path = os.path.join(
        dirs["validation_diagnostics"],
        f"{model_name}_validation_gate_by_horizon_farm_{prepared['farm_id']}.csv",
    )
    _atomic_to_csv(
        pd.DataFrame(diagnostics["gate_rows"]),
        gate_path,
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
        "supplement_protocol_version": SUPPLEMENT_PROTOCOL_VERSION,
        "experiment_role": spec["experiment_role"],
        "selection_eligible": bool(spec["selection_eligible"]),
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
        "training_mode": (
            "stage1_b2_frozen_candidate_gate_only_probe"
            if freeze_candidates
            else "stage1_b2_warm_start_feature_subset_finetune"
        ),
        "freeze_candidates": freeze_candidates,
        "backbone_frozen": freeze_candidates,
        "frozen_candidate_layer_names": (
            list(regime_train.B2_WEIGHTED_LAYER_NAMES)
            if freeze_candidates
            else []
        ),
        "frozen_candidate_parameter_count": (
            FROZEN_B2_PARAMETER_COUNT if freeze_candidates else 0
        ),
        "residual_dropout_disabled_for_frozen_candidate": freeze_candidates,
        "residual_dropout_rate_during_training": float(
            model.get_layer("residual_dropout").rate
        ),
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
        "candidate_supervision_loss_weight": candidate_loss_weight,
        "correction_kernel_l2": CORRECTION_KERNEL_L2,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "validation_split": VALIDATION_SPLIT,
        "learning_rate": LEARNING_RATE,
        "early_stopping_monitor": monitor,
        "total_params": total_params,
        "expected_total_params": expected_params,
        "expected_trainable_params": expected_trainable_params,
        "trainable_params": trainable_params,
        "model_size_bytes": model_size_bytes,
        "training_elapsed_seconds": elapsed_seconds,
        "train_samples": train_samples,
        "val_samples": total_samples - train_samples,
        "model_path": paths["model_path"],
        "model_sha256": _file_sha256(paths["model_path"]),
        "best_weights_path": paths["best_weights_path"],
        "best_weights_sha256": _file_sha256(paths["best_weights_path"]),
        "artifact_path": paths["artifact_path"],
        "history_path": history_path,
        "history_plot_path": history_plot_path,
        "tensorboard_log_dir": tensorboard_log_dir,
        "tail_path": paths["tail_path"],
        "validation_regime_metrics_path": regime_path,
        "validation_gate_diagnostics_path": gate_path,
        "backbone_initialization": backbone_source,
        "source_model_path": backbone_source["source_model_path"],
        "source_model_sha256": _file_sha256(
            backbone_source["source_model_path"]
        ),
        "source_artifact_path": backbone_source["source_artifact_path"],
        "source_artifact_sha256": _file_sha256(
            backbone_source["source_artifact_path"]
        ),
        "source_preprocess_compatibility_path": source_preprocess_path,
        "source_preprocess_compatibility_sha256": _file_sha256(
            source_preprocess_path
        ),
        "frozen_weights_sha256_before_training": frozen_weights_sha256_before,
        "frozen_weights_sha256_after_training": frozen_weights_sha256_after,
        "frozen_weights_exact_match_after_training": frozen_weights_exact_match,
        "candidate_output_sha256_before_training": candidate_output_sha256_before,
        "candidate_output_sha256_after_training": candidate_output_sha256_after,
        "candidate_output_exact_match_after_training": candidate_output_exact_match,
        "post_training_candidate_max_abs_error": (
            post_training_candidate_max_abs_error
        ),
        "evaluation_pipeline_version": regime_train.EVALUATION_PIPELINE_VERSION,
        "legacy_bidirectional_weather_imputation": True,
        "scaler_fit_scope": "full_train_file_including_validation",
        "validation_target_overlap_steps": FORECAST_LEN - 1,
        "exploratory_legacy_comparison": True,
        "selection_metric_source": "test_macro_capacity_normalized_rmse",
        "test_used_for_feature_selection": bool(spec["selection_eligible"]),
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
    _atomic_joblib_dump(artifact, paths["artifact_path"])

    result = {
        "model_family": MODEL_FAMILY,
        "model_name": model_name,
        "variant_id": variant_id,
        "variant_label": spec["label"],
        "experiment_role": spec["experiment_role"],
        "selection_eligible": bool(spec["selection_eligible"]),
        "supplement_protocol_version": SUPPLEMENT_PROTOCOL_VERSION,
        "feature_groups": "+".join(spec["groups"]),
        "feature_count": len(names),
        "feature_names": json.dumps(names, ensure_ascii=False),
        "farm_id": prepared["farm_id"],
        "requires_training": True,
        "result_source": "stage2_feature_screen_trained",
        "current_run_action": "train_new_supplement_variant",
        "reused_existing_training_result": False,
        "source_variant": "b2_persistence_residual",
        "freeze_candidates": freeze_candidates,
        "backbone_frozen": freeze_candidates,
        "candidate_supervision_loss_weight": candidate_loss_weight,
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "total_params": total_params,
        "expected_total_params": expected_params,
        "expected_trainable_params": expected_trainable_params,
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
        "best_weights_sha256": _file_sha256(paths["best_weights_path"]),
        "artifact_path": paths["artifact_path"],
        "history_path": history_path,
        "history_plot_path": history_plot_path,
        "validation_regime_metrics_path": regime_path,
        "validation_gate_diagnostics_path": gate_path,
        "source_model_path": backbone_source["source_model_path"],
        "source_model_sha256": _file_sha256(
            backbone_source["source_model_path"]
        ),
        "source_artifact_path": backbone_source["source_artifact_path"],
        "source_artifact_sha256": _file_sha256(
            backbone_source["source_artifact_path"]
        ),
        "frozen_candidate_parameter_count": (
            FROZEN_B2_PARAMETER_COUNT if freeze_candidates else 0
        ),
        "frozen_weights_sha256_before_training": frozen_weights_sha256_before,
        "frozen_weights_sha256_after_training": frozen_weights_sha256_after,
        "frozen_weights_exact_match_after_training": frozen_weights_exact_match,
        "candidate_output_sha256_before_training": candidate_output_sha256_before,
        "candidate_output_sha256_after_training": candidate_output_sha256_after,
        "candidate_output_exact_match_after_training": candidate_output_exact_match,
        "post_training_candidate_max_abs_error": (
            post_training_candidate_max_abs_error
        ),
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


def validate_feature_training_protocol(
    artifact,
    artifact_path="<memory>",
    candidate_loss_weight=CANDIDATE_LOSS_WEIGHT,
):
    """Validate the fixed protocol shared by legacy F models and supplements."""
    expected = dict(LEGACY_JOINT_PROTOCOL)
    expected["candidate_supervision_loss_weight"] = float(candidate_loss_weight)
    mismatched = []
    for key, expected_value in expected.items():
        actual = artifact.get(key)
        if actual is None:
            matches = False
        elif isinstance(expected_value, float):
            matches = np.isclose(
                float(actual), expected_value, rtol=0.0, atol=1e-12
            )
        else:
            matches = int(actual) == int(expected_value)
        if not matches:
            mismatched.append(
                f"{key}={actual!r} (expected {expected_value!r})"
            )
    if mismatched:
        raise ValueError(
            f"feature-screen训练协议不一致: {artifact_path}: "
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


def load_reused_f0_f7_training_results(variant_ids, farm_ids):
    """从既有40行F0--F7汇总只读复用结果，不创建或改写旧模型产物。"""
    requested_variants = tuple(dict.fromkeys(variant_ids))
    invalid = sorted(set(requested_variants) - set(REUSED_TRAINING_VARIANTS))
    if invalid:
        raise ValueError(f"只读复用函数收到非F0--F7变体: {invalid}")
    if not requested_variants:
        return pd.DataFrame()
    if not os.path.exists(LEGACY_TRAINING_SUMMARY):
        raise FileNotFoundError(
            f"缺少F0--F7既有40行训练汇总: {LEGACY_TRAINING_SUMMARY}"
        )
    source_all = pd.read_csv(LEGACY_TRAINING_SUMMARY)
    required_columns = {
        "variant_id",
        "farm_id",
        "total_params",
        "feature_count",
        "feature_names",
        "artifact_path",
        "model_path",
    }
    missing_columns = sorted(required_columns - set(source_all.columns))
    if missing_columns:
        raise KeyError(f"F0--F7旧汇总缺少字段: {missing_columns}")
    source_all["variant_id"] = source_all["variant_id"].astype(str)
    source_all["farm_id"] = source_all["farm_id"].astype(str)
    if len(source_all) != 40:
        raise ValueError(f"F0--F7旧汇总应为40行，实际{len(source_all)}行")
    if set(source_all["variant_id"]) != set(REUSED_TRAINING_VARIANTS):
        raise ValueError("F0--F7旧汇总的变体集合不完整")
    if source_all.duplicated(["variant_id", "farm_id"]).any():
        raise ValueError("F0--F7旧汇总存在重复variant/farm键")
    counts = source_all.groupby("variant_id")["farm_id"].nunique()
    if not (counts == 5).all():
        raise ValueError(f"F0--F7旧汇总并非每个变体5场站: {counts.to_dict()}")

    requested_farms = tuple(dict.fromkeys(str(value) for value in farm_ids))
    source = source_all[
        source_all["variant_id"].isin(requested_variants)
        & source_all["farm_id"].isin(requested_farms)
    ].copy()
    expected_rows = len(requested_variants) * len(requested_farms)
    if len(source) != expected_rows:
        raise ValueError(
            f"F0--F7旧汇总未覆盖请求矩阵: {len(source)} != {expected_rows}"
        )

    for row_index, row in source.iterrows():
        variant_id = row["variant_id"]
        expected_names = selected_feature_names(variant_id)
        try:
            stored_names = tuple(json.loads(row["feature_names"]))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"旧汇总{variant_id}/{row['farm_id']} feature_names非法"
            ) from exc
        if stored_names != expected_names:
            raise ValueError(
                f"旧汇总{variant_id}/{row['farm_id']}特征子集已漂移"
            )
        if int(row["feature_count"]) != len(expected_names):
            raise ValueError(
                f"旧汇总{variant_id}/{row['farm_id']}特征数不一致"
            )
        if int(row["total_params"]) != EXPECTED_PARAMETER_COUNTS[variant_id]:
            raise ValueError(
                f"旧汇总{variant_id}/{row['farm_id']}参数量不一致"
            )
        artifact_path = regime_train._resolve_existing_path(row["artifact_path"])
        model_path = regime_train._resolve_existing_path(row["model_path"])
        if artifact_path is None or model_path is None:
            raise FileNotFoundError(
                f"旧汇总{variant_id}/{row['farm_id']} artifact或model不存在"
            )
        artifact = joblib.load(artifact_path)
        if variant_id == "f4":
            validate_r4_reference_artifact(artifact, artifact_path)
        else:
            if artifact.get("architecture_version") != ARCHITECTURE_VERSION:
                raise ValueError(f"旧artifact架构版本不匹配: {artifact_path}")
            if artifact.get("variant_id") != variant_id:
                raise ValueError(f"旧artifact变体不匹配: {artifact_path}")
            if tuple(
                artifact.get("selected_regime_feature_names", ())
            ) != expected_names:
                raise ValueError(f"旧artifact特征子集不匹配: {artifact_path}")
            if int(artifact.get("total_params", -1)) != EXPECTED_PARAMETER_COUNTS[
                variant_id
            ]:
                raise ValueError(f"旧artifact参数量不匹配: {artifact_path}")
        validate_feature_training_protocol(artifact, artifact_path)
        stored_model_hash = row.get("model_sha256")
        actual_model_hash = _file_sha256(model_path)
        if (
            isinstance(stored_model_hash, str)
            and stored_model_hash
            and stored_model_hash != actual_model_hash
        ):
            raise ValueError(
                f"旧模型SHA256不匹配: {variant_id}/{row['farm_id']}"
            )
        source.at[row_index, "resolved_reused_artifact_path"] = os.path.abspath(
            artifact_path
        )
        source.at[row_index, "resolved_reused_model_path"] = os.path.abspath(
            model_path
        )
        source.at[row_index, "resolved_reused_model_sha256"] = actual_model_hash

    source["current_run_action"] = "reuse_existing_f0_f7_result"
    source["reused_existing_training_result"] = True
    source["reused_training_summary_path"] = os.path.abspath(
        LEGACY_TRAINING_SUMMARY
    )
    source["reused_training_summary_sha256"] = _file_sha256(
        LEGACY_TRAINING_SUMMARY
    )
    source["supplement_protocol_version"] = SUPPLEMENT_PROTOCOL_VERSION
    source["experiment_role"] = source["variant_id"].map(
        {key: value["experiment_role"] for key, value in VARIANT_SPECS.items()}
    )
    source["selection_eligible"] = source["variant_id"].map(
        {key: bool(value["selection_eligible"]) for key, value in VARIANT_SPECS.items()}
    )
    source["freeze_candidates"] = False
    source["backbone_frozen"] = False
    source["expected_trainable_params"] = source["variant_id"].map(
        EXPECTED_TRAINABLE_PARAMETER_COUNTS
    )
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
                "current_run_action": (
                    "reuse_existing_f0_f7_result"
                    if variant_id in REUSED_TRAINING_VARIANTS
                    else "train_new_supplement_variant"
                ),
                "reuse_existing": bool(spec["reuse_existing"]),
                "experiment_role": spec["experiment_role"],
                "selection_eligible": bool(spec["selection_eligible"]),
                "freeze_candidates": bool(spec["freeze_candidates"]),
                "candidate_supervision_loss_weight": (
                    candidate_loss_weight_for_variant(variant_id)
                    if spec["requires_training"]
                    else CANDIDATE_LOSS_WEIGHT
                ),
                "result_source": (
                    "reuse_existing_feature_screen_summary"
                    if variant_id in REUSED_TRAINING_VARIANTS
                    else "stage2_feature_screen_supplement_trained"
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
                "expected_trainable_parameter_count": (
                    EXPECTED_TRAINABLE_PARAMETER_COUNTS[variant_id]
                ),
                "supplement_protocol_version": SUPPLEMENT_PROTOCOL_VERSION,
                "selection_split": "test",
                "selection_metric": "macro_mean_capacity_normalized_rmse",
                "test_used_for_feature_selection": bool(
                    spec["selection_eligible"]
                ),
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
    _atomic_to_csv(
        pd.DataFrame(rows),
        os.path.join(RESULT_ROOT, EXTENDED_MANIFEST_NAME),
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
                "experiment_role": VARIANT_SPECS[variant_id]["experiment_role"],
                "selection_eligible": bool(
                    VARIANT_SPECS[variant_id]["selection_eligible"]
                ),
                "freeze_candidates": bool(
                    VARIANT_SPECS[variant_id]["freeze_candidates"]
                ),
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
    variants = get_requested_variants()
    if any(item in NEW_TRAINING_VARIANTS for item in variants):
        # 新模型可能覆盖正式bundle成员；训练开始前撤销旧完成标志，只有全部
        # 15个新增模型及汇总再次验收后才重新发布。
        _clear_training_completion_marker()
        _clear_downstream_prediction_completion_marker()
    _write_manifest()
    train_files = discover_train_files(DATA_DIR)
    if not train_files:
        raise FileNotFoundError(f"未在 {DATA_DIR} 找到请求的训练文件")
    farm_ids = [regime_train.get_farm_id(path) for path in train_files]
    reused = [
        variant_id for variant_id in variants if variant_id in REUSED_TRAINING_VARIANTS
    ]
    trainable = [
        variant_id for variant_id in variants if variant_id in NEW_TRAINING_VARIANTS
    ]
    print(f"固定随机种子: {RANDOM_SEED}；batch_size={BATCH_SIZE}")
    print(f"场站数: {len(train_files)}；F/FP矩阵: {variants}")
    print(f"只读复用既有训练结果: {reused}")
    print(f"实际新增训练: {trainable}")
    if reused:
        print(
            "F0--F7直接读取既有40行summary；不重新训练、复制或修改旧模型产物"
        )

    results = []
    if reused:
        results.extend(
            load_reused_f0_f7_training_results(reused, farm_ids).to_dict("records")
        )

    progress_path = os.path.join(
        RESULT_ROOT,
        f"{EXTENDED_PROGRESS_PREFIX}_{_partial_tag(variants, farm_ids)}.csv",
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
    all_train_farm_ids = {
        str(regime_train.get_farm_id(path)) for path in all_train_files
    }
    expected_farm_set = set(expected_training_farm_ids())
    expected_rows = len(VARIANT_SPECS) * len(expected_farm_set)
    is_complete = (
        set(variants) == set(VARIANT_SPECS)
        and not os.getenv("WIND_FEATURE_SCREEN_FARMS")
        and all_train_farm_ids == expected_farm_set
        and len(metrics_df) == expected_rows
        and not metrics_df.duplicated(["variant_id", "farm_id"]).any()
    )
    partial_tag = _partial_tag(variants, farm_ids)
    if is_complete:
        metrics_filename = EXTENDED_TRAINING_SUMMARY_NAME
        validation_filename = EXTENDED_VALIDATION_NAME
    else:
        metrics_stem, _ = os.path.splitext(EXTENDED_TRAINING_SUMMARY_NAME)
        validation_stem, _ = os.path.splitext(EXTENDED_VALIDATION_NAME)
        metrics_filename = f"{metrics_stem}_partial_{partial_tag}.csv"
        validation_filename = f"{validation_stem}_partial_{partial_tag}.csv"
    metrics_path = os.path.join(RESULT_ROOT, metrics_filename)
    _atomic_to_csv(metrics_df, metrics_path)
    comparison = _validation_comparison(metrics_df, variants)
    validation_path = os.path.join(RESULT_ROOT, validation_filename)
    _atomic_to_csv(
        comparison,
        validation_path,
    )
    print(f"F0--F8/FP训练与只读引用汇总已保存: {metrics_path}")
    if not is_complete:
        print("当前是子集运行；文件名带partial标签，不会覆盖完整F矩阵汇总")
    else:
        completion_path = _publish_training_completion_marker(
            metrics_df,
            metrics_path,
            validation_path,
        )
        print(f"F8/FP训练bundle完成标志: {completion_path}")
    print("最终模型不在此处按验证集选择；请运行专用预测脚本按测试集宏平均NRMSE筛选")


if __name__ == "__main__":
    main()
