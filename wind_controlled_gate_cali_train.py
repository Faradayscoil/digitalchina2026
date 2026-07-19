"""第三阶段两候选受控门控校准与 Persistence 保护实验训练入口。

第二阶段已按测试集锁定 F7=P+H+D。本阶段固定 seed=2026，并使用同一
Persistence / lightweight corrected residual 两候选及36维显式工况上下文。
实验矩阵严格定义为：

    G0  F7 非因子化 sample×horizon sigmoid gate；直接引用，不重复训练。
    G1  因子化 gate pi(i,h)=q(i)*s(h) + dynamic supervision。
    G2  非因子化 gate + soft-oracle calibration + Persistence safety loss。
    G3  因子化 gate + calibration + dynamic + safety（完整软门控）。
    G4  G3 + 统一阈值 Persistence abstention；预测期离线生成，不训练。

G1--G3 均从同一场站 F7 快照独立初始化，不串行继承其他 G 变体。训练目标为

    L_base = L_fused + 0.5 * L_corrected

并按变体加入：

    G1: +0.05 L_dynamic
    G2: +0.10 L_cal +0.05 L_safe
    G3: +0.10 L_cal +0.05 L_dynamic +0.05 L_safe

未来真实功率只用于训练 target。soft oracle、dynamic target 和 safety regret
在当前 batch 内由 y/P/C/F 计算并 stop_gradient，不会进入模型历史输入。
校准Brier按初始冻结F7训练候选的逐horizon |C-P| Q90加权，权重下限0.25；
Q90仅由训练窗口预估并固化到artifact，不使用validation/test。
G4 及 hard top-1 负对照由预测脚本从同一次 G3 输出生成。

默认训练 G1--G3×5 场站；可用 ``WIND_CONTROLLED_GATE_VARIANTS`` 和
``WIND_CONTROLLED_GATE_FARMS`` 运行 partial，但 partial 不发布 complete marker。
"""

import glob
import hashlib
import json
import os
import re
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import wind_RegimeEncoder_PatchTST_feature_screen_train as feature_train
import wind_RegimeEncoder_PatchTST_train as regime_train
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


MODEL_FAMILY = "controlled_gate_cali"
ARCHITECTURE_VERSION = "controlled_gate_cali_stage3_v1"
ARTIFACT_SCHEMA_VERSION = 2
PROTOCOL_VERSION = "controlled_gate_g0_g4_test_selected_v2"
RESULT_ROOT = os.path.join("./wind_results", MODEL_FAMILY)
SOURCE_VARIANT = "f7"
SOURCE_FEATURE_GROUPS = "P+H+D"
SOURCE_FEATURE_COUNT = 36
RANDOM_SEED = 2026

BATCH_SIZE = int(os.getenv("WIND_CONTROLLED_GATE_BATCH_SIZE", "192"))
VALIDATION_SPLIT = float(os.getenv("WIND_CONTROLLED_GATE_VALIDATION_SPLIT", "0.15"))
PHASE_GATE_EPOCHS = int(os.getenv("WIND_CONTROLLED_GATE_GATE_ONLY_EPOCHS", "3"))
PHASE_CONTEXT_EPOCHS = int(os.getenv("WIND_CONTROLLED_GATE_CONTEXT_EPOCHS", "5"))
PHASE_JOINT_EPOCHS = int(os.getenv("WIND_CONTROLLED_GATE_JOINT_EPOCHS", "30"))
PHASE_INITIAL_LR = float(os.getenv("WIND_CONTROLLED_GATE_INITIAL_LR", "0.0001"))
PHASE_JOINT_LR = float(os.getenv("WIND_CONTROLLED_GATE_JOINT_LR", "0.00005"))
EARLY_STOPPING_PATIENCE = int(os.getenv("WIND_CONTROLLED_GATE_PATIENCE", "6"))

CALIBRATION_WEIGHT = 0.10
DYNAMIC_WEIGHT = 0.05
SAFETY_WEIGHT = 0.05
CORRECTED_WEIGHT = 0.50
SOFT_ORACLE_TEMPERATURE = 0.10
CALIBRATION_DIFFERENCE_QUANTILE = 0.90
CALIBRATION_WEIGHT_FLOOR = 0.25
DYNAMIC_START_FRACTION = 0.02
DYNAMIC_WIDTH_FRACTION = 0.08
FACTORIZED_INITIAL_GATE = 0.45
G4_KAPPA_GRID = (0.45, 0.50, 0.55, 0.60, 0.65)
PARAMETER_LIMIT = 30000
EXPECTED_TOTAL_PARAMS = {
    "g1": 20409,
    "g2": 20969,
    "g3": 20409,
}
EXPECTED_TRAINABLE_PARAMS_BY_PHASE = {
    "g1": {"gate_only": 433, "context": 1993, "joint": 20409},
    "g2": {"gate_only": 993, "context": 2553, "joint": 20969},
    "g3": {"gate_only": 433, "context": 1993, "joint": 20409},
}

VARIANT_SPECS = {
    "g0": {
        "label": "G0 F7 non-factorized gate reference",
        "requires_training": False,
        "factorized_gate": False,
        "calibration_weight": 0.0,
        "dynamic_weight": 0.0,
        "safety_weight": 0.0,
        "source": "feature_screen_f7",
        "description": "直接引用第二阶段F7；不重复训练",
    },
    "g1": {
        "label": "G1 factorized regime-horizon gate",
        "requires_training": True,
        "factorized_gate": True,
        "calibration_weight": 0.0,
        "dynamic_weight": DYNAMIC_WEIGHT,
        "safety_weight": 0.0,
        "source": SOURCE_VARIANT,
        "description": "pi=q_i*s_h，仅加入dynamic supervision",
    },
    "g2": {
        "label": "G2 calibrated non-factorized safe gate",
        "requires_training": True,
        "factorized_gate": False,
        "calibration_weight": CALIBRATION_WEIGHT,
        "dynamic_weight": 0.0,
        "safety_weight": SAFETY_WEIGHT,
        "source": SOURCE_VARIANT,
        "description": "G0拓扑 + soft-oracle weighted Brier + safety regret",
    },
    "g3": {
        "label": "G3 full factorized calibrated safe gate",
        "requires_training": True,
        "factorized_gate": True,
        "calibration_weight": CALIBRATION_WEIGHT,
        "dynamic_weight": DYNAMIC_WEIGHT,
        "safety_weight": SAFETY_WEIGHT,
        "source": SOURCE_VARIANT,
        "description": "factorized + calibration + dynamic + safety",
    },
    "g4": {
        "label": "G4 G3 + Persistence abstention",
        "requires_training": False,
        "factorized_gate": True,
        "calibration_weight": CALIBRATION_WEIGHT,
        "dynamic_weight": DYNAMIC_WEIGHT,
        "safety_weight": SAFETY_WEIGHT,
        "source": "g3_posthoc",
        "description": "G3同次输出按统一kappa执行Persistence abstention",
    },
}
TRAINABLE_VARIANTS = ("g1", "g2", "g3")
REFERENCE_VARIANTS = ("g0", "g4")
_CALIBRATION_STATS_CACHE = {}

COMMON_WEIGHTED_LAYER_NAMES = (
    "residual_causal_conv_1",
    "residual_causal_conv_2",
    "residual_hidden",
    "persistence_residual",
    "explicit_regime_feature_norm",
    "regime_context_hidden",
    "regime_context",
)
RESIDUAL_WEIGHTED_LAYER_NAMES = tuple(regime_train.B2_WEIGHTED_LAYER_NAMES)
CONTEXT_WEIGHTED_LAYER_NAMES = (
    "explicit_regime_feature_norm",
    "regime_context_hidden",
    "regime_context",
)

TRAINING_SUMMARY_NAME = "controlled_gate_cali_training_metrics.csv"
MANIFEST_NAME = "controlled_gate_cali_experiment_manifest.csv"
TRAINING_MARKER_NAME = "controlled_gate_cali_training_bundle_complete.json"
PREDICTION_MARKER_RELATIVE_PATH = os.path.join(
    "testdata_predict_output",
    "controlled_gate_cali_test_bundle_complete.json",
)


def configure_reproducibility():
    os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_SEED))
    set_global_seed(RANDOM_SEED)
    keras.utils.set_random_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _validate_protocol_configuration():
    expected = {
        "batch_size": 192,
        "validation_split": 0.15,
        "gate_epochs": 3,
        "context_epochs": 5,
        "joint_epochs": 30,
        "initial_lr": 1e-4,
        "joint_lr": 5e-5,
        "patience": 6,
    }
    actual = {
        "batch_size": BATCH_SIZE,
        "validation_split": VALIDATION_SPLIT,
        "gate_epochs": PHASE_GATE_EPOCHS,
        "context_epochs": PHASE_CONTEXT_EPOCHS,
        "joint_epochs": PHASE_JOINT_EPOCHS,
        "initial_lr": PHASE_INITIAL_LR,
        "joint_lr": PHASE_JOINT_LR,
        "patience": EARLY_STOPPING_PATIENCE,
    }
    mismatched = {
        key: (actual[key], value)
        for key, value in expected.items()
        if not np.isclose(actual[key], value, rtol=0.0, atol=1e-12)
    }
    allow_override = (
        os.getenv("WIND_CONTROLLED_GATE_ALLOW_PROTOCOL_OVERRIDE", "0") == "1"
    )
    if mismatched and not allow_override:
        raise ValueError(
            "正式G0--G4协议不允许超参数漂移；如仅做调试请显式设置"
            f"WIND_CONTROLLED_GATE_ALLOW_PROTOCOL_OVERRIDE=1: {mismatched}"
        )
    if mismatched:
        print(
            "警告：当前为protocol override调试运行，不会发布正式complete marker: "
            f"{mismatched}"
        )
    if tuple(feature_train.selected_feature_names(SOURCE_VARIANT)) != tuple(
        name
        for group in ("P", "H", "D")
        for name in feature_train.FEATURE_GROUPS[group]
    ):
        raise ValueError("当前F7已不再对应预注册的P+H+D特征顺序")
    if (
        len(feature_train.selected_feature_names(SOURCE_VARIANT))
        != SOURCE_FEATURE_COUNT
    ):
        raise ValueError("F7显式特征数不再是36")
    return not mismatched


def variant_model_name(variant_id):
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"未知G变体: {variant_id}")
    return f"{MODEL_FAMILY}_{variant_id}"


def variant_dirs(variant_id, create=True, result_root=None):
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"未知G变体: {variant_id}")
    root = os.path.join(RESULT_ROOT if result_root is None else result_root, variant_id)
    dirs = {
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
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)
    return dirs


def get_requested_variants():
    raw = os.getenv("WIND_CONTROLLED_GATE_VARIANTS")
    if not raw:
        return list(VARIANT_SPECS)
    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = sorted(set(requested) - set(VARIANT_SPECS))
    if invalid:
        raise ValueError(f"未知G变体{invalid}; 可选{list(VARIANT_SPECS)}")
    return list(dict.fromkeys(requested))


def get_farm_id(path):
    match = re.search(r"wind_train_(\d+)\.csv$", os.path.basename(path))
    if not match:
        raise ValueError(f"无法从训练文件解析场站ID: {path}")
    return match.group(1)


def discover_train_files(data_dir=DATA_DIR):
    files = sorted(glob.glob(os.path.join(data_dir, "wind_train_*.csv")))
    requested = os.getenv("WIND_CONTROLLED_GATE_FARMS")
    if requested:
        farm_ids = {item.strip() for item in requested.split(",") if item.strip()}
        files = [path for path in files if get_farm_id(path) in farm_ids]
    return files


def expected_farm_ids():
    source = feature_train.expected_training_farm_ids()
    if len(source) != 5:
        raise ValueError(f"F7正式训练场站数不是5: {source}")
    return tuple(str(item) for item in source)


def _sha256(path, chunk_size=1024 * 1024):
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _resolve_existing_path(path):
    if not isinstance(path, (str, os.PathLike)) or not os.fspath(path).strip():
        return None
    value = os.fspath(path)
    candidates = [value]
    if not os.path.isabs(value):
        candidates.append(os.path.join(os.path.dirname(__file__), value))
    return next((item for item in candidates if os.path.exists(item)), None)


def _array_sha256(named_arrays):
    digest = hashlib.sha256()
    for name, value in named_arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(name).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_to_csv(frame, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def _atomic_write_json(value, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def _atomic_joblib_dump(value, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    try:
        joblib.dump(value, temporary)
        restored = joblib.load(temporary)
        if not isinstance(restored, dict):
            raise TypeError(f"artifact重载后不是dict: {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


@keras.utils.register_keras_serializable(package="ControlledGateCali")
class FactorizedHorizonPrior(layers.Layer):
    """Learn s_h and broadcast it to the current batch."""

    def __init__(self, forecast_len, initial_value, **kwargs):
        super().__init__(**kwargs)
        if forecast_len <= 0 or not 0 < initial_value < 1:
            raise ValueError("FactorizedHorizonPrior参数无效")
        self.forecast_len = int(forecast_len)
        self.initial_value = float(initial_value)

    def build(self, input_shape):
        logit = np.log(self.initial_value / (1.0 - self.initial_value))
        self.horizon_logits = self.add_weight(
            name="horizon_logits",
            shape=(self.forecast_len,),
            initializer=keras.initializers.Constant(logit),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        prior = tf.nn.sigmoid(self.horizon_logits)
        return tf.broadcast_to(
            prior[tf.newaxis, :],
            [tf.shape(inputs)[0], self.forecast_len],
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "forecast_len": self.forecast_len,
                "initial_value": self.initial_value,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="ControlledGateCali")
class BroadcastScalarToHorizon(layers.Layer):
    def __init__(self, forecast_len, **kwargs):
        super().__init__(**kwargs)
        self.forecast_len = int(forecast_len)

    def call(self, inputs):
        value = tf.reshape(inputs, [tf.shape(inputs)[0], 1])
        return tf.repeat(value, repeats=self.forecast_len, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"forecast_len": self.forecast_len})
        return config


@keras.utils.register_keras_serializable(package="ControlledGateCali")
class HorizonGateMean(layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-1, keepdims=True)


@keras.utils.register_keras_serializable(package="ControlledGateCali")
class OnesHorizonPrior(layers.Layer):
    def call(self, inputs):
        return tf.ones_like(inputs)


@keras.utils.register_keras_serializable(package="ControlledGateCali")
class ControlledGateAuxiliaryLoss(keras.losses.Loss):
    """Current-batch soft-oracle calibration, dynamic and safety objective."""

    def __init__(
        self,
        forecast_len,
        target_mean,
        target_scale,
        capacity,
        calibration_weight,
        dynamic_weight,
        safety_weight,
        candidate_difference_q90,
        oracle_temperature=SOFT_ORACLE_TEMPERATURE,
        calibration_weight_floor=CALIBRATION_WEIGHT_FLOOR,
        dynamic_start=DYNAMIC_START_FRACTION,
        dynamic_width=DYNAMIC_WIDTH_FRACTION,
        name="controlled_gate_auxiliary_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if target_scale <= 0 or capacity <= 0 or oracle_temperature <= 0:
            raise ValueError("ControlledGateAuxiliaryLoss尺度参数无效")
        self.forecast_len = int(forecast_len)
        self.target_mean = float(target_mean)
        self.target_scale = float(target_scale)
        self.capacity = float(capacity)
        self.calibration_weight = float(calibration_weight)
        self.dynamic_weight = float(dynamic_weight)
        self.safety_weight = float(safety_weight)
        candidate_difference_q90 = np.asarray(
            candidate_difference_q90, dtype=np.float32
        ).reshape(-1)
        if (
            candidate_difference_q90.size != self.forecast_len
            or not np.isfinite(candidate_difference_q90).all()
            or np.any(candidate_difference_q90 < 0.0)
        ):
            raise ValueError("candidate_difference_q90必须是非负有限horizon向量")
        if not 0.0 < calibration_weight_floor <= 1.0:
            raise ValueError("calibration_weight_floor必须位于(0,1]")
        self.candidate_difference_q90 = candidate_difference_q90.tolist()
        self.oracle_temperature = float(oracle_temperature)
        self.calibration_weight_floor = float(calibration_weight_floor)
        self.dynamic_start = float(dynamic_start)
        self.dynamic_width = float(dynamic_width)

    def _capacity_fraction(self, scaled):
        physical = scaled * self.target_scale + self.target_mean
        return tf.clip_by_value(physical / self.capacity, 0.0, 1.0)

    def call(self, y_true, packet):
        h = self.forecast_len
        gate = packet[:, 0:h]
        persistence = packet[:, h : 2 * h]
        corrected = packet[:, 2 * h : 3 * h]
        fused = packet[:, 3 * h : 4 * h]
        q_by_horizon = packet[:, 4 * h : 5 * h]

        truth_fraction = self._capacity_fraction(y_true)
        persistence_fraction = self._capacity_fraction(persistence)
        corrected_fraction = self._capacity_fraction(corrected)
        fused_fraction = self._capacity_fraction(fused)
        e_p = tf.abs(truth_fraction - persistence_fraction)
        e_c = tf.abs(truth_fraction - corrected_fraction)
        e_f = tf.abs(truth_fraction - fused_fraction)

        advantage = (e_p - e_c) / (e_p + e_c + keras.backend.epsilon())
        oracle = tf.stop_gradient(tf.nn.sigmoid(advantage / self.oracle_temperature))
        candidate_difference = tf.abs(corrected_fraction - persistence_fraction)
        q90 = tf.convert_to_tensor(
            self.candidate_difference_q90,
            dtype=packet.dtype,
        )[tf.newaxis, :]
        normalized_difference = tf.clip_by_value(
            candidate_difference / (q90 + keras.backend.epsilon()),
            0.0,
            1.0,
        )
        difference_weight = tf.stop_gradient(
            self.calibration_weight_floor
            + (1.0 - self.calibration_weight_floor) * normalized_difference
        )
        weighted_brier = tf.reduce_sum(difference_weight * tf.square(gate - oracle)) / (
            tf.reduce_sum(difference_weight) + keras.backend.epsilon()
        )

        max_persistence_error = tf.reduce_max(e_p, axis=-1)
        dynamic_target = tf.stop_gradient(
            tf.clip_by_value(
                (max_persistence_error - self.dynamic_start) / self.dynamic_width,
                0.0,
                1.0,
            )
        )
        q = q_by_horizon[:, 0]
        dynamic_loss = tf.reduce_mean(tf.square(q - dynamic_target))

        positive_regret = tf.nn.relu(e_f - e_p)
        safety_loss = tf.reduce_mean((1.0 - oracle) * positive_regret)
        return (
            self.calibration_weight * weighted_brier
            + self.dynamic_weight * dynamic_loss
            + self.safety_weight * safety_loss
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "forecast_len": self.forecast_len,
                "target_mean": self.target_mean,
                "target_scale": self.target_scale,
                "capacity": self.capacity,
                "calibration_weight": self.calibration_weight,
                "dynamic_weight": self.dynamic_weight,
                "safety_weight": self.safety_weight,
                "candidate_difference_q90": self.candidate_difference_q90,
                "oracle_temperature": self.oracle_temperature,
                "calibration_weight_floor": self.calibration_weight_floor,
                "dynamic_start": self.dynamic_start,
                "dynamic_width": self.dynamic_width,
            }
        )
        return config


class ValidationSelectionCheckpoint(keras.callbacks.Callback):
    """Checkpoint by validation NRMSE, regret and Brier across all phases."""

    def __init__(self, path, validation_dataset, prepared, variant_id):
        super().__init__()
        self.path = path
        self.validation_dataset = validation_dataset
        self.prepared = prepared
        self.variant_id = variant_id
        self.best = np.inf
        self.min_nrmse_seen = np.inf
        self.best_positive_regret = np.inf
        self.best_brier = np.inf
        self.best_phase = None
        self.phase = None
        self.global_epoch = 0
        self.records = []
        self.weight_snapshots = []
        self.diagnostic_model = None

    def set_model(self, model):
        super().set_model(model)
        if self.diagnostic_model is None:
            self.diagnostic_model = _diagnostic_model(model)

    def _is_better(self, nrmse, positive_regret, brier):
        self.min_nrmse_seen = min(self.min_nrmse_seen, nrmse)
        if not np.isfinite(self.best):
            return True
        tie_band_upper = self.min_nrmse_seen * (1.0 + 0.001)
        if self.best > tie_band_upper:
            return True
        if nrmse <= tie_band_upper:
            if positive_regret < self.best_positive_regret - 1e-12:
                return True
            if (
                np.isclose(
                    positive_regret,
                    self.best_positive_regret,
                    rtol=0.0,
                    atol=1e-12,
                )
                and brier < self.best_brier - 1e-12
            ):
                return True
        return False

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        diagnostics, _, _ = _validation_diagnostics(
            self.model,
            self.validation_dataset,
            self.prepared,
            self.variant_id,
            diagnostic_model=self.diagnostic_model,
        )
        row = diagnostics.iloc[0].to_dict()
        nrmse = float(row["capacity_normalized_rmse"])
        positive_regret = float(row["positive_regret_mean"])
        brier = float(row["oracle_brier"])
        for name, value in {
            "selection_val_nrmse": nrmse,
            "selection_val_positive_regret": positive_regret,
            "selection_val_brier": brier,
        }.items():
            if not np.isfinite(value):
                raise FloatingPointError(f"checkpoint验证指标{name}非有限: {value}")
            logs[name] = value
        checkpoint_updated = self._is_better(nrmse, positive_regret, brier)
        self.records.append(
            {
                "global_epoch": self.global_epoch,
                "phase": self.phase,
                "phase_epoch": int(epoch),
                **row,
                "fit_val_loss": logs.get("val_loss"),
                "online_checkpoint_updated": checkpoint_updated,
            }
        )
        self.weight_snapshots.append(
            [np.array(value, copy=True) for value in self.model.get_weights()]
        )
        self.global_epoch += 1
        if checkpoint_updated:
            self.best = nrmse
            self.best_positive_regret = positive_regret
            self.best_brier = brier
            self.best_phase = self.phase
            self.model.save_weights(self.path)

    def finalize_from_all_epochs(self):
        if not self.records or len(self.records) != len(self.weight_snapshots):
            raise ValueError("checkpoint缺少完整逐epoch指标/权重快照")
        frame = pd.DataFrame(self.records)
        minimum_nrmse = float(frame["capacity_normalized_rmse"].min())
        eligible = frame[
            frame["capacity_normalized_rmse"] <= minimum_nrmse * (1.0 + 0.001)
        ].copy()
        selected_index = int(
            eligible.sort_values(
                [
                    "positive_regret_mean",
                    "oracle_brier",
                    "capacity_normalized_rmse",
                    "global_epoch",
                ],
                kind="stable",
            ).index[0]
        )
        selected = frame.loc[selected_index]
        self.model.set_weights(self.weight_snapshots[selected_index])
        self.model.save_weights(self.path)
        self.min_nrmse_seen = minimum_nrmse
        self.best = float(selected["capacity_normalized_rmse"])
        self.best_positive_regret = float(selected["positive_regret_mean"])
        self.best_brier = float(selected["oracle_brier"])
        self.best_phase = str(selected["phase"])
        return selected_index


def get_controlled_gate_custom_objects():
    objects = dict(feature_train.get_feature_screen_custom_objects())
    classes = (
        FactorizedHorizonPrior,
        BroadcastScalarToHorizon,
        HorizonGateMean,
        OnesHorizonPrior,
        ControlledGateAuxiliaryLoss,
    )
    for cls in classes:
        objects[cls.__name__] = cls
        objects[f"ControlledGateCali>{cls.__name__}"] = cls
    return objects


def _source_f7_artifact_path(farm_id):
    model_name = feature_train.variant_model_name(SOURCE_VARIANT)
    return os.path.join(
        feature_train.variant_dirs(SOURCE_VARIANT, create=False)["preprocess"],
        f"{model_name}_farm_{farm_id}_preprocess.pkl",
    )


def load_source_f7(farm_id):
    artifact_path = _source_f7_artifact_path(farm_id)
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"缺少F7 artifact: {artifact_path}")
    artifact = joblib.load(artifact_path)
    expected_names = list(feature_train.selected_feature_names(SOURCE_VARIANT))
    checks = {
        "variant": artifact.get("variant_id") == SOURCE_VARIANT,
        "architecture": artifact.get("architecture_version")
        == feature_train.ARCHITECTURE_VERSION,
        "seed": int(artifact.get("random_seed", -1)) == RANDOM_SEED,
        "feature_groups": list(artifact.get("selected_regime_feature_groups", ()))
        == ["P", "H", "D"],
        "feature_names": list(artifact.get("selected_regime_feature_names", ()))
        == expected_names,
        "feature_count": int(artifact.get("selected_regime_feature_count", -1))
        == SOURCE_FEATURE_COUNT,
        "params": int(artifact.get("total_params", -1))
        == feature_train.EXPECTED_PARAMETER_COUNTS[SOURCE_VARIANT],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"F7/{farm_id}来源协议不兼容: {failed}")
    model_path = artifact.get("model_path")
    model_path = _resolve_existing_path(model_path)
    if model_path is None:
        raise FileNotFoundError(f"缺少F7完整模型: {model_path}")
    source_summary, _ = _source_f7_training_summary()
    source_row = source_summary[source_summary["farm_id"].astype(str) == str(farm_id)]
    if len(source_row) != 1:
        raise ValueError(f"F7/{farm_id}训练来源summary不是唯一一行")
    source_row = source_row.iloc[0]
    recorded_model_path = _resolve_existing_path(source_row.get("model_path"))
    recorded_artifact_path = _resolve_existing_path(source_row.get("artifact_path"))
    if (
        recorded_model_path is None
        or os.path.realpath(recorded_model_path) != os.path.realpath(model_path)
        or recorded_artifact_path is None
        or os.path.realpath(recorded_artifact_path) != os.path.realpath(artifact_path)
    ):
        raise ValueError(f"F7/{farm_id} artifact与正式训练summary身份不一致")
    current_model_sha256 = _sha256(model_path)
    if current_model_sha256 != source_row.get("model_sha256"):
        raise ValueError(f"F7/{farm_id}模型hash与正式训练summary不一致")
    model = keras.models.load_model(
        model_path,
        custom_objects=feature_train.get_feature_screen_custom_objects(),
        compile=False,
    )
    if (
        int(model.count_params())
        != feature_train.EXPECTED_PARAMETER_COUNTS[SOURCE_VARIANT]
    ):
        raise ValueError("F7加载模型参数量异常")
    return model, artifact, artifact_path, model_path


def _validate_prepared_against_source(prepared, artifact):
    checks = {
        "input_cols": list(prepared["input_cols"])
        == list(artifact.get("input_cols", ())),
        "target_index": int(prepared["target_index"])
        == int(artifact.get("target_index", -1)),
        "capacity": np.isclose(
            float(prepared["capacity"]),
            float(artifact.get("capacity", np.nan)),
            rtol=1e-10,
            atol=1e-8,
        ),
        "scaler_x_mean": np.allclose(
            prepared["scaler_x"].mean_,
            artifact["scaler_x"].mean_,
            rtol=1e-8,
            atol=1e-8,
        ),
        "scaler_x_scale": np.allclose(
            prepared["scaler_x"].scale_,
            artifact["scaler_x"].scale_,
            rtol=1e-8,
            atol=1e-8,
        ),
        "scaler_y_mean": np.allclose(
            prepared["scaler_y"].mean_,
            artifact["scaler_y"].mean_,
            rtol=1e-8,
            atol=1e-8,
        ),
        "scaler_y_scale": np.allclose(
            prepared["scaler_y"].scale_,
            artifact["scaler_y"].scale_,
            rtol=1e-8,
            atol=1e-8,
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"F7/{prepared['farm_id']}预处理不一致: {failed}")


def _plain_datasets(prepared):
    return make_window_dataset(
        prepared["features"],
        prepared["target"],
        HISTORY_LEN,
        FORECAST_LEN,
        BATCH_SIZE,
        VALIDATION_SPLIT,
    )


def _attach_training_targets(dataset):
    def attach(batch_x, batch_y):
        return batch_x, {
            "forecast_power": batch_y,
            "candidate_forecast": batch_y,
            "control_packet": batch_y,
        }

    return dataset.map(
        attach,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    ).prefetch(tf.data.AUTOTUNE)


def _source_diagnostic_model(source_model):
    return keras.Model(
        source_model.inputs,
        {
            "persistence": source_model.get_layer(
                "persistence_forecast_candidate"
            ).output,
            "corrected": source_model.get_layer("corrected_forecast_candidate").output,
            "gate": source_model.get_layer("correction_gate").output,
            "forecast": source_model.get_layer("forecast_power").output,
            "context": source_model.get_layer("regime_context").output,
        },
    )


def _scaled_to_capacity_fraction(values, prepared):
    values = np.asarray(values, dtype=float)
    physical = values * float(prepared["scaler_y"].scale_[0]) + float(
        prepared["scaler_y"].mean_[0]
    )
    return np.clip(physical / float(prepared["capacity"]), 0.0, 1.0)


def estimate_initial_calibration_statistics(source_model, train_ds, prepared):
    diagnostic = _source_diagnostic_model(source_model)
    total = 0.0
    count = 0
    candidate_differences = []
    for batch_x, batch_y in train_ds:
        output = diagnostic(batch_x, training=False)
        truth = _scaled_to_capacity_fraction(batch_y.numpy(), prepared)
        persistence = _scaled_to_capacity_fraction(
            output["persistence"].numpy(), prepared
        )
        corrected = _scaled_to_capacity_fraction(output["corrected"].numpy(), prepared)
        e_p = np.abs(truth - persistence)
        e_c = np.abs(truth - corrected)
        advantage = (e_p - e_c) / (e_p + e_c + 1e-8)
        oracle = 1.0 / (1.0 + np.exp(-advantage / SOFT_ORACLE_TEMPERATURE))
        total += float(oracle.sum())
        count += int(oracle.size)
        candidate_differences.append(np.abs(corrected - persistence))
    if count == 0:
        raise ValueError("训练集无法估计soft-oracle与候选差异分位数")
    differences = np.concatenate(candidate_differences, axis=0)
    q90 = np.quantile(
        differences,
        CALIBRATION_DIFFERENCE_QUANTILE,
        axis=0,
    )
    if q90.shape != (FORECAST_LEN,) or not np.isfinite(q90).all():
        raise ValueError("训练集候选差异Q90不是有限horizon向量")
    return {
        "soft_oracle_mean": float(np.clip(total / count, 0.05, 0.95)),
        "candidate_difference_q90": q90.astype(np.float32),
        "sample_count": int(differences.shape[0]),
        "element_count": int(differences.size),
    }


def _build_factorized_gate(context):
    component_init = float(np.sqrt(FACTORIZED_INITIAL_GATE))
    initial_logit = float(np.log(component_init / (1.0 - component_init)))
    hidden = layers.Dense(
        feature_train.GATE_HIDDEN_DIM,
        activation="gelu",
        name="sample_dynamic_hidden",
    )(context)
    hidden = layers.Dropout(
        feature_train.GATE_DROPOUT,
        name="sample_dynamic_dropout",
    )(hidden)
    q = layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(initial_logit),
        name="sample_dynamic_probability",
    )(hidden)
    q_by_horizon = BroadcastScalarToHorizon(
        FORECAST_LEN,
        name="sample_dynamic_probability_by_horizon",
    )(q)
    horizon_prior = FactorizedHorizonPrior(
        FORECAST_LEN,
        component_init,
        name="horizon_gate_prior",
    )(q)
    gate = layers.Multiply(name="controlled_gate")([q_by_horizon, horizon_prior])
    return gate, q_by_horizon, horizon_prior


def build_controlled_gate_model(variant_id, source_artifact, initial_gate_weight):
    if variant_id not in TRAINABLE_VARIANTS:
        raise ValueError(f"{variant_id}不是可训练G变体")
    configure_reproducibility()
    template = feature_train.build_feature_screen_model_from_artifact(source_artifact)
    persistence = template.get_layer("persistence_forecast_candidate").output
    corrected = template.get_layer("corrected_forecast_candidate").output
    context = template.get_layer("regime_context").output
    spec = VARIANT_SPECS[variant_id]
    if spec["factorized_gate"]:
        gate, q_by_horizon, horizon_prior = _build_factorized_gate(context)
    else:
        gate = regime_train.SampleHorizonCorrectionGate(
            forecast_len=FORECAST_LEN,
            hidden_dim=feature_train.GATE_HIDDEN_DIM,
            horizon_embedding_dim=feature_train.HORIZON_EMBEDDING_DIM,
            dropout=feature_train.GATE_DROPOUT,
            initial_weight=float(initial_gate_weight),
            name="controlled_gate",
        )(context)
        q = HorizonGateMean(name="sample_dynamic_probability")(gate)
        q_by_horizon = BroadcastScalarToHorizon(
            FORECAST_LEN,
            name="sample_dynamic_probability_by_horizon",
        )(q)
        horizon_prior = OnesHorizonPrior(name="horizon_gate_prior")(gate)
    forecast = regime_train.TwoCandidateGateFusion(name="forecast_power")(
        [persistence, corrected, gate]
    )
    candidate = layers.Activation("linear", name="candidate_forecast")(corrected)
    packet = layers.Concatenate(name="control_packet")(
        [gate, persistence, corrected, forecast, q_by_horizon, horizon_prior]
    )
    return keras.Model(
        template.inputs,
        {
            "forecast_power": forecast,
            "candidate_forecast": candidate,
            "control_packet": packet,
        },
        name=f"ControlledGateCali_{variant_id.upper()}",
    )


def _copy_common_weights(source_model, target_model):
    copied = []
    for name in COMMON_WEIGHTED_LAYER_NAMES:
        source_layer = source_model.get_layer(name)
        target_layer = target_model.get_layer(name)
        source_weights = source_layer.get_weights()
        target_weights = target_layer.get_weights()
        if [item.shape for item in source_weights] != [
            item.shape for item in target_weights
        ]:
            raise ValueError(f"F7->{target_model.name}层{name}权重形状不一致")
        target_layer.set_weights(source_weights)
        if any(
            not np.array_equal(left, right)
            for left, right in zip(source_weights, target_layer.get_weights())
        ):
            raise ValueError(f"层{name}复制后不完全一致")
        copied.append(name)
    return copied


def _weighted_layer_snapshot(model, names):
    values = []
    for name in names:
        layer = model.get_layer(name)
        for index, value in enumerate(layer.get_weights()):
            values.append((f"{name}:{index}", value))
    return values


def _diagnostic_model(model):
    packet = model.get_layer("control_packet").output
    h = FORECAST_LEN
    return keras.Model(
        model.inputs,
        {
            "forecast": model.get_layer("forecast_power").output,
            "persistence": model.get_layer("persistence_forecast_candidate").output,
            "corrected": model.get_layer("corrected_forecast_candidate").output,
            "gate": packet[:, 0:h],
            "q": packet[:, 4 * h : 5 * h],
            "s": packet[:, 5 * h : 6 * h],
        },
    )


def _assert_source_initialization(source_model, target_model, sample_x):
    source = _source_diagnostic_model(source_model)(sample_x, training=False)
    target = _diagnostic_model(target_model)(sample_x, training=False)
    for key in ("persistence", "corrected"):
        left = np.asarray(source[key])
        right = np.asarray(target[key])
        if not np.array_equal(left, right):
            difference = float(np.max(np.abs(left - right)))
            raise ValueError(f"F7->{target_model.name} {key}初值漂移: {difference}")
    source_context = np.asarray(source["context"])
    target_context = np.asarray(
        keras.Model(
            target_model.inputs, target_model.get_layer("regime_context").output
        )(sample_x, training=False)
    )
    if not np.array_equal(source_context, target_context):
        raise ValueError("F7显式工况context复制后不完全一致")


def _set_training_phase(model, phase):
    if phase not in {"gate_only", "context", "joint"}:
        raise ValueError(f"未知训练phase: {phase}")
    for layer in model.layers:
        layer.trainable = False
    gate_names = {
        "sample_dynamic_hidden",
        "sample_dynamic_dropout",
        "sample_dynamic_probability",
        "sample_dynamic_probability_by_horizon",
        "horizon_gate_prior",
        "controlled_gate",
    }
    for name in gate_names:
        try:
            model.get_layer(name).trainable = True
        except ValueError:
            pass
    if phase in {"context", "joint"}:
        for name in CONTEXT_WEIGHTED_LAYER_NAMES:
            model.get_layer(name).trainable = True
    if phase == "joint":
        for name in RESIDUAL_WEIGHTED_LAYER_NAMES:
            model.get_layer(name).trainable = True

    residual_dropout = model.get_layer("residual_dropout")
    residual_dropout.rate = (
        float(regime_train.HEAD_DROPOUT) if phase == "joint" else 0.0
    )
    context_dropout = model.get_layer("regime_context_dropout")
    context_dropout.rate = (
        float(feature_train.GATE_DROPOUT) if phase != "gate_only" else 0.0
    )


def _trainable_parameter_count(model):
    return int(sum(int(np.prod(weight.shape)) for weight in model.trainable_weights))


def _compile_for_phase(
    model,
    variant_id,
    prepared,
    learning_rate,
    candidate_difference_q90,
):
    spec = VARIANT_SPECS[variant_id]
    auxiliary_loss = ControlledGateAuxiliaryLoss(
        forecast_len=FORECAST_LEN,
        target_mean=float(prepared["scaler_y"].mean_[0]),
        target_scale=float(prepared["scaler_y"].scale_[0]),
        capacity=float(prepared["capacity"]),
        calibration_weight=spec["calibration_weight"],
        dynamic_weight=spec["dynamic_weight"],
        safety_weight=spec["safety_weight"],
        candidate_difference_q90=candidate_difference_q90,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
        ),
        loss={
            "forecast_power": keras.losses.Huber(delta=1.0),
            "candidate_forecast": keras.losses.Huber(delta=1.0),
            "control_packet": auxiliary_loss,
        },
        loss_weights={
            "forecast_power": 1.0,
            "candidate_forecast": CORRECTED_WEIGHT,
            "control_packet": 1.0,
        },
        metrics={
            "forecast_power": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ]
        },
    )


def _train_paths(dirs, variant_id, farm_id):
    name = variant_model_name(variant_id)
    return {
        "model_path": os.path.join(dirs["models"], f"{name}_farm_{farm_id}.keras"),
        "weights_path": os.path.join(
            dirs["weights"], f"{name}_farm_{farm_id}_best.weights.h5"
        ),
        "artifact_path": os.path.join(
            dirs["preprocess"], f"{name}_farm_{farm_id}_preprocess.pkl"
        ),
        "history_path": os.path.join(
            dirs["history"], f"{name}_history_farm_{farm_id}.csv"
        ),
        "tail_path": os.path.join(dirs["tails"], f"{name}_tail_farm_{farm_id}.csv"),
        "validation_path": os.path.join(
            dirs["validation_diagnostics"],
            f"{name}_validation_diagnostics_farm_{farm_id}.csv",
        ),
        "checkpoint_trace_path": os.path.join(
            dirs["validation_diagnostics"],
            f"{name}_checkpoint_trace_farm_{farm_id}.csv",
        ),
    }


def _history_frame(histories):
    rows = []
    global_epoch = 0
    for phase, history in histories:
        keys = list(history.history)
        epoch_count = len(next(iter(history.history.values()))) if keys else 0
        for local_epoch in range(epoch_count):
            row = {
                "global_epoch": global_epoch,
                "phase": phase,
                "phase_epoch": local_epoch,
            }
            for key, values in history.history.items():
                row[key] = values[local_epoch]
            rows.append(row)
            global_epoch += 1
    return pd.DataFrame(rows)


def _inverse_scaled(values, prepared):
    shape = np.asarray(values).shape
    physical = (
        prepared["scaler_y"]
        .inverse_transform(np.asarray(values).reshape(-1, 1))
        .reshape(shape)
    )
    return np.clip(physical, 0.0, float(prepared["capacity"]))


def _validation_ece(probability, truth, bins=10):
    probability = np.asarray(probability, dtype=float)
    truth = np.asarray(truth, dtype=float)
    ids = np.minimum((np.clip(probability, 0.0, 1.0) * bins).astype(int), bins - 1)
    value = 0.0
    for bin_id in range(bins):
        mask = ids == bin_id
        if mask.any():
            value += mask.mean() * abs(probability[mask].mean() - truth[mask].mean())
    return float(value)


def _validation_diagnostics(
    model,
    val_ds,
    prepared,
    variant_id,
    diagnostic_model=None,
):
    diagnostic = (
        _diagnostic_model(model) if diagnostic_model is None else diagnostic_model
    )
    outputs = {
        key: [] for key in ("forecast", "persistence", "corrected", "gate", "q", "s")
    }
    truths = []
    for batch_x, batch_y in val_ds:
        result = diagnostic(batch_x, training=False)
        truths.append(np.asarray(batch_y))
        for key in outputs:
            outputs[key].append(np.asarray(result[key]))
    if not truths:
        raise ValueError("验证集为空")
    y_scaled = np.concatenate(truths)
    values = {key: np.concatenate(items) for key, items in outputs.items()}
    y = _inverse_scaled(y_scaled, prepared)
    persistence = _inverse_scaled(values["persistence"], prepared)
    corrected = _inverse_scaled(values["corrected"], prepared)
    forecast = _inverse_scaled(values["forecast"], prepared)
    gate = values["gate"]
    capacity = float(prepared["capacity"])
    valid = (
        np.isfinite(y)
        & np.isfinite(forecast)
        & np.isfinite(persistence)
        & np.isfinite(corrected)
        & np.isfinite(gate)
    )
    error = forecast[valid] - y[valid]
    corrected_error = corrected[valid] - y[valid]
    p_error = np.abs(persistence[valid] - y[valid]) / capacity
    f_error = np.abs(forecast[valid] - y[valid]) / capacity
    positive_regret = np.maximum(0.0, f_error - p_error)
    oracle = np.abs(corrected[valid] - y[valid]) < np.abs(persistence[valid] - y[valid])
    oracle_brier = float(np.mean(np.square(gate[valid] - oracle.astype(float))))
    row = {
        "variant_id": variant_id,
        "farm_id": str(prepared["farm_id"]),
        "valid_count": int(valid.sum()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "capacity_normalized_mae": float(np.mean(np.abs(error)) / capacity),
        "capacity_normalized_rmse": float(
            np.sqrt(np.mean(np.square(error))) / capacity
        ),
        "corrected_capacity_normalized_rmse": float(
            np.sqrt(np.mean(np.square(corrected_error))) / capacity
        ),
        "gate_mean": float(np.mean(gate)),
        "gate_std": float(np.std(gate)),
        "gate_low_saturation_rate": float(np.mean(gate < 0.05)),
        "gate_high_saturation_rate": float(np.mean(gate > 0.95)),
        "q_mean": float(np.mean(values["q"][:, 0])),
        "s_mean": float(np.mean(values["s"])),
        "positive_regret_mean": float(np.mean(positive_regret)),
        "harm_rate_0_005": float(np.mean((f_error - p_error) > 0.005)),
        "oracle_brier": oracle_brier,
        "ece_10bin": _validation_ece(gate[valid], oracle),
        "diagnostic_scope": "validation_checkpoint_selection_and_descriptive",
    }
    return pd.DataFrame([row]), values, y_scaled


def _save_model_atomic(model, path):
    stem, extension = os.path.splitext(path)
    temporary = f"{stem}.tmp{extension}"
    try:
        model.save(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            if os.path.isdir(temporary):
                import shutil

                shutil.rmtree(temporary)
            else:
                os.remove(temporary)


def train_variant_for_farm(variant_id, prepared, result_root=None):
    if variant_id not in TRAINABLE_VARIANTS:
        raise ValueError(f"禁止训练引用/后处理变体{variant_id}")
    keras.backend.clear_session()
    configure_reproducibility()
    farm_id = str(prepared["farm_id"])
    source_model, source_artifact, source_artifact_path, source_model_path = (
        load_source_f7(farm_id)
    )
    _validate_prepared_against_source(prepared, source_artifact)
    plain_train, plain_val, train_samples, total_samples = _plain_datasets(prepared)
    calibration_statistics = None
    if VARIANT_SPECS[variant_id]["calibration_weight"] > 0.0:
        calibration_cache_key = (
            farm_id,
            _sha256(source_model_path),
            _sha256(prepared["train_file"]),
            float(VALIDATION_SPLIT),
        )
        if calibration_cache_key not in _CALIBRATION_STATS_CACHE:
            _CALIBRATION_STATS_CACHE[calibration_cache_key] = (
                estimate_initial_calibration_statistics(
                    source_model,
                    plain_train,
                    prepared,
                )
            )
        calibration_statistics = _CALIBRATION_STATS_CACHE[calibration_cache_key]
        if calibration_statistics["sample_count"] != int(train_samples):
            raise ValueError(
                f"{variant_id}/{farm_id}校准Q90样本数"
                f"{calibration_statistics['sample_count']} != train_samples {train_samples}"
            )
    soft_oracle_mean = (
        calibration_statistics["soft_oracle_mean"]
        if variant_id == "g2" and calibration_statistics is not None
        else None
    )
    candidate_difference_q90 = (
        np.asarray(
            calibration_statistics["candidate_difference_q90"],
            dtype=np.float32,
        )
        if calibration_statistics is not None
        else np.zeros(FORECAST_LEN, dtype=np.float32)
    )
    initial_gate = (
        soft_oracle_mean if soft_oracle_mean is not None else FACTORIZED_INITIAL_GATE
    )
    model = build_controlled_gate_model(
        variant_id,
        source_artifact,
        initial_gate,
    )
    copied_layers = _copy_common_weights(source_model, model)
    sample_x, _ = next(iter(plain_train))
    sample_x = sample_x[:2]
    _assert_source_initialization(source_model, model, sample_x)
    source_snapshot_hash = _array_sha256(
        _weighted_layer_snapshot(source_model, COMMON_WEIGHTED_LAYER_NAMES)
    )
    initial_snapshot_hash = _array_sha256(
        _weighted_layer_snapshot(model, COMMON_WEIGHTED_LAYER_NAMES)
    )
    if source_snapshot_hash != initial_snapshot_hash:
        raise ValueError("F7公共权重快照复制hash不一致")
    initial_gate_hash = _array_sha256(
        _weighted_layer_snapshot(
            model,
            tuple(
                name
                for name in (
                    "sample_dynamic_hidden",
                    "sample_dynamic_probability",
                    "horizon_gate_prior",
                    "controlled_gate",
                )
                if any(layer.name == name for layer in model.layers)
            ),
        )
    )
    initial_candidate = np.asarray(
        _diagnostic_model(model)(sample_x, training=False)["corrected"]
    )
    initial_candidate_hash = _array_sha256([("corrected", initial_candidate)])

    train_ds = _attach_training_targets(plain_train)
    val_ds = _attach_training_targets(plain_val)
    dirs = variant_dirs(variant_id, result_root=result_root)
    paths = _train_paths(dirs, variant_id, farm_id)
    if os.path.exists(paths["weights_path"]):
        os.remove(paths["weights_path"])
    checkpoint = ValidationSelectionCheckpoint(
        paths["weights_path"],
        plain_val,
        prepared,
        variant_id,
    )
    tensorboard_root = os.path.join(
        dirs["tensorboard"],
        f"farm_{farm_id}",
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    histories = []
    phase_specs = (
        ("gate_only", PHASE_GATE_EPOCHS, PHASE_INITIAL_LR),
        ("context", PHASE_CONTEXT_EPOCHS, PHASE_INITIAL_LR),
        ("joint", PHASE_JOINT_EPOCHS, PHASE_JOINT_LR),
    )
    phase_trainable_params = {}
    start = time.monotonic()
    for phase, epochs, learning_rate in phase_specs:
        _set_training_phase(model, phase)
        trainable_count = _trainable_parameter_count(model)
        expected_trainable = EXPECTED_TRAINABLE_PARAMS_BY_PHASE[variant_id][phase]
        if trainable_count != expected_trainable:
            raise ValueError(
                f"{variant_id}/{phase}可训练参数{trainable_count} != "
                f"协议值{expected_trainable}"
            )
        phase_trainable_params[phase] = trainable_count
        _compile_for_phase(
            model,
            variant_id,
            prepared,
            learning_rate,
            candidate_difference_q90,
        )
        checkpoint.phase = phase
        finite_guard = feature_train.NonFiniteTrainingGuard()
        callbacks = [
            finite_guard,
            keras.callbacks.TerminateOnNaN(),
            checkpoint,
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(tensorboard_root, phase),
                histogram_freq=0,
                write_graph=phase == "gate_only",
                profile_batch=0,
            ),
        ]
        if phase == "joint":
            callbacks.extend(
                [
                    keras.callbacks.EarlyStopping(
                        monitor="selection_val_nrmse",
                        mode="min",
                        patience=EARLY_STOPPING_PATIENCE,
                        restore_best_weights=False,
                        verbose=1,
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor="selection_val_nrmse",
                        mode="min",
                        factor=0.5,
                        patience=3,
                        min_lr=1e-6,
                        verbose=1,
                    ),
                ]
            )
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        feature_train.ensure_finite_training_history(history, finite_guard)
        histories.append((phase, history))
    selected_checkpoint_index = checkpoint.finalize_from_all_epochs()
    if (
        checkpoint.best_phase is None
        or not np.isfinite(checkpoint.best)
        or not os.path.exists(paths["weights_path"])
    ):
        raise FileNotFoundError("跨phase最佳权重未生成")
    model.load_weights(paths["weights_path"])
    elapsed = float(time.monotonic() - start)
    final_candidate = np.asarray(
        _diagnostic_model(model)(sample_x, training=False)["corrected"]
    )
    final_candidate_hash = _array_sha256([("corrected", final_candidate)])
    candidate_max_abs_drift = float(np.max(np.abs(final_candidate - initial_candidate)))
    final_snapshot_hash = _array_sha256(
        _weighted_layer_snapshot(model, COMMON_WEIGHTED_LAYER_NAMES)
    )

    history_frame = _history_frame(histories)
    _atomic_to_csv(history_frame, paths["history_path"])
    checkpoint_trace = pd.DataFrame(checkpoint.records)
    if len(checkpoint_trace) != len(history_frame) or checkpoint_trace.empty:
        raise ValueError("checkpoint逐epoch验证轨迹与训练history长度不一致")
    checkpoint_trace["selected_checkpoint"] = False
    if selected_checkpoint_index not in checkpoint_trace.index:
        raise ValueError("全epoch复合checkpoint索引不在验证轨迹中")
    checkpoint_trace.loc[selected_checkpoint_index, "selected_checkpoint"] = True
    _atomic_to_csv(checkpoint_trace, paths["checkpoint_trace_path"])
    validation_frame, _, _ = _validation_diagnostics(
        model, plain_val, prepared, variant_id
    )
    _atomic_to_csv(validation_frame, paths["validation_path"])
    prepared["train_df"].iloc[-HISTORY_LEN:].to_csv(paths["tail_path"], index=True)
    _save_model_atomic(model, paths["model_path"])
    restored = keras.models.load_model(
        paths["model_path"],
        custom_objects=get_controlled_gate_custom_objects(),
        compile=False,
    )
    expected_outputs = _diagnostic_model(model)(sample_x, training=False)
    actual_outputs = _diagnostic_model(restored)(sample_x, training=False)
    for output_name in ("forecast", "persistence", "corrected", "gate", "q", "s"):
        expected = np.asarray(expected_outputs[output_name])
        actual = np.asarray(actual_outputs[output_name])
        if not np.allclose(expected, actual, rtol=1e-7, atol=1e-7):
            raise ValueError(f"受控门控模型保存/重载{output_name}不一致")

    total_params = int(model.count_params())
    trainable_params = _trainable_parameter_count(model)
    expected_total_params = EXPECTED_TOTAL_PARAMS[variant_id]
    if total_params != expected_total_params:
        raise ValueError(
            f"{variant_id}参数量{total_params} != 协议值{expected_total_params}"
        )
    if total_params >= PARAMETER_LIMIT:
        raise ValueError(f"{variant_id}参数量{total_params}超过30k预声明上限")
    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_family": MODEL_FAMILY,
        "architecture_version": ARCHITECTURE_VERSION,
        "variant_id": variant_id,
        "variant_spec": dict(VARIANT_SPECS[variant_id]),
        "farm_id": farm_id,
        "train_file": os.path.abspath(prepared["train_file"]),
        "train_file_sha256": _sha256(prepared["train_file"]),
        "random_seed": RANDOM_SEED,
        "history_len": HISTORY_LEN,
        "forecast_len": FORECAST_LEN,
        "time_freq": TIME_FREQ,
        "target_col": TARGET_COL,
        "feature_cols": prepared["feature_cols"],
        "input_cols": prepared["input_cols"],
        "target_index": prepared["target_index"],
        "scaler_x": prepared["scaler_x"],
        "scaler_y": prepared["scaler_y"],
        "capacity": float(prepared["capacity"]),
        "power_scale_ratio": prepared["power_scale_ratio"],
        "power_scale_offset": prepared["power_scale_offset"],
        "regime_feature_config": prepared["regime_feature_config"],
        "selected_regime_feature_groups": ["P", "H", "D"],
        "selected_regime_feature_names": list(
            feature_train.selected_feature_names(SOURCE_VARIANT)
        ),
        "selected_regime_feature_count": SOURCE_FEATURE_COUNT,
        "factorized_gate": bool(VARIANT_SPECS[variant_id]["factorized_gate"]),
        "soft_oracle_temperature": SOFT_ORACLE_TEMPERATURE,
        "soft_oracle_train_mean": soft_oracle_mean,
        "calibration_candidate_difference_weight": {
            "enabled": bool(VARIANT_SPECS[variant_id]["calibration_weight"] > 0.0),
            "formula": "0.25+0.75*clip(abs(C-P)/(Q90_train_h+eps),0,1)",
            "candidate_difference_q90": candidate_difference_q90.tolist(),
            "quantile": CALIBRATION_DIFFERENCE_QUANTILE,
            "scope": "per_farm_per_horizon_train_initial_frozen_f7",
            "domain": "physical_power_divided_by_capacity",
            "weight_floor": CALIBRATION_WEIGHT_FLOOR,
            "epsilon": float(keras.backend.epsilon()),
            "estimation_sample_count": (
                calibration_statistics["sample_count"]
                if calibration_statistics is not None
                else 0
            ),
            "estimation_element_count": (
                calibration_statistics["element_count"]
                if calibration_statistics is not None
                else 0
            ),
            "candidate_source_model_sha256": _sha256(source_model_path),
        },
        "initial_gate_weight": initial_gate,
        "loss_weights": {
            "fused": 1.0,
            "corrected": CORRECTED_WEIGHT,
            "calibration": VARIANT_SPECS[variant_id]["calibration_weight"],
            "dynamic": VARIANT_SPECS[variant_id]["dynamic_weight"],
            "safety": VARIANT_SPECS[variant_id]["safety_weight"],
        },
        "dynamic_target": {
            "start_capacity_fraction": DYNAMIC_START_FRACTION,
            "width_capacity_fraction": DYNAMIC_WIDTH_FRACTION,
        },
        "training_phases": [
            {
                "phase": phase,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "trainable_parameter_count": phase_trainable_params[phase],
            }
            for phase, epochs, learning_rate in phase_specs
        ],
        "batch_size": BATCH_SIZE,
        "validation_split": VALIDATION_SPLIT,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "checkpoint_selection": {
            "primary": "validation_capacity_normalized_rmse",
            "near_tie_relative": 0.001,
            "tie_breakers": ["validation_positive_regret", "validation_brier"],
            "not_selected_by_val_loss_only": True,
            "global_min_nrmse_tie_band_anchor": True,
            "selection_mode": "strict_reselection_from_all_epoch_weight_snapshots",
        },
        "candidate_recomputed_each_batch": True,
        "future_targets_are_fit_targets_only": True,
        "test_used_for_training": False,
        "selection_split": "test_in_prediction_script",
        "test_is_final_blind_evaluation": False,
        "legacy_bidirectional_weather_imputation": True,
        "scaler_fit_scope": "full_train_file_including_validation",
        "validation_target_overlap_steps": FORECAST_LEN - 1,
        "copied_layer_names": copied_layers,
        "source_f7_artifact_path": os.path.abspath(source_artifact_path),
        "source_f7_artifact_sha256": _sha256(source_artifact_path),
        "source_f7_model_path": os.path.abspath(source_model_path),
        "source_f7_model_sha256": _sha256(source_model_path),
        "source_common_snapshot_sha256": source_snapshot_hash,
        "initial_common_snapshot_sha256": initial_snapshot_hash,
        "final_common_snapshot_sha256": final_snapshot_hash,
        "initial_gate_head_sha256": initial_gate_hash,
        "initial_corrected_candidate_sha256": initial_candidate_hash,
        "final_corrected_candidate_sha256": final_candidate_hash,
        "corrected_candidate_max_abs_drift": candidate_max_abs_drift,
        "model_path": os.path.abspath(paths["model_path"]),
        "model_sha256": _sha256(paths["model_path"]),
        "best_weights_path": os.path.abspath(paths["weights_path"]),
        "best_weights_sha256": _sha256(paths["weights_path"]),
        "artifact_path": os.path.abspath(paths["artifact_path"]),
        "history_path": os.path.abspath(paths["history_path"]),
        "validation_diagnostics_path": os.path.abspath(paths["validation_path"]),
        "checkpoint_trace_path": os.path.abspath(paths["checkpoint_trace_path"]),
        "tail_path": os.path.abspath(paths["tail_path"]),
        "tensorboard_log_dir": os.path.abspath(tensorboard_root),
        "total_params": total_params,
        "trainable_params_final_phase": trainable_params,
        "training_elapsed_seconds": elapsed,
        "train_samples": int(train_samples),
        "validation_samples": int(total_samples - train_samples),
        "best_validation_nrmse": checkpoint.best,
        "minimum_validation_nrmse_seen": checkpoint.min_nrmse_seen,
        "best_validation_positive_regret": checkpoint.best_positive_regret,
        "best_validation_brier": checkpoint.best_brier,
        "best_phase": checkpoint.best_phase,
    }
    _atomic_joblib_dump(artifact, paths["artifact_path"])
    row = dict(validation_frame.iloc[0])
    row.update(
        {
            "model_family": MODEL_FAMILY,
            "variant_id": variant_id,
            "variant_label": VARIANT_SPECS[variant_id]["label"],
            "feature_groups": SOURCE_FEATURE_GROUPS,
            "feature_count": SOURCE_FEATURE_COUNT,
            "reference_only": False,
            "requires_training": True,
            "random_seed": RANDOM_SEED,
            "parameter_count": total_params,
            "trainable_parameter_count": trainable_params,
            "training_elapsed_seconds": elapsed,
            "actual_epoch_count": len(history_frame),
            "best_validation_nrmse": checkpoint.best,
            "minimum_validation_nrmse_seen": checkpoint.min_nrmse_seen,
            "best_validation_positive_regret": checkpoint.best_positive_regret,
            "best_validation_brier": checkpoint.best_brier,
            "best_phase": checkpoint.best_phase,
            "source_variant": SOURCE_VARIANT,
            "source_model_path": os.path.abspath(source_model_path),
            "source_model_sha256": _sha256(source_model_path),
            "model_path": os.path.abspath(paths["model_path"]),
            "model_sha256": _sha256(paths["model_path"]),
            "best_weights_path": os.path.abspath(paths["weights_path"]),
            "best_weights_sha256": _sha256(paths["weights_path"]),
            "artifact_path": os.path.abspath(paths["artifact_path"]),
            "artifact_sha256": _sha256(paths["artifact_path"]),
            "history_path": os.path.abspath(paths["history_path"]),
            "validation_diagnostics_path": os.path.abspath(paths["validation_path"]),
            "checkpoint_trace_path": os.path.abspath(paths["checkpoint_trace_path"]),
            "tail_path": os.path.abspath(paths["tail_path"]),
            "candidate_drift_max_abs": candidate_max_abs_drift,
            "result_source": "new_stage3_training",
        }
    )
    del restored, source_model, model
    keras.backend.clear_session()
    return row


def _source_f7_training_summary():
    path = os.path.join(
        feature_train.RESULT_ROOT,
        "feature_screening_training_metrics.csv",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少F0--F7训练汇总: {path}")
    frame = pd.read_csv(path, dtype={"farm_id": str})
    frame = frame[frame["variant_id"].astype(str) == SOURCE_VARIANT].copy()
    if len(frame) != 5 or frame["farm_id"].nunique() != 5:
        raise ValueError("F7训练引用不是5场站唯一矩阵")
    return frame, path


def build_g0_reference_rows(farm_ids):
    source, path = _source_f7_training_summary()
    source = source[source["farm_id"].isin(set(map(str, farm_ids)))].copy()
    expected = set(map(str, farm_ids))
    if (
        set(source["farm_id"]) != expected
        or len(source) != len(expected)
        or source["farm_id"].duplicated().any()
    ):
        raise ValueError("G0/F7训练引用未覆盖请求场站唯一集合")
    rows = []
    for _, item in source.iterrows():
        model_path = _resolve_existing_path(item.get("model_path"))
        artifact_path = _resolve_existing_path(item.get("artifact_path"))
        if model_path is None or artifact_path is None:
            raise FileNotFoundError(f"G0/F7来源文件缺失: farm={item['farm_id']}")
        model_sha256 = _sha256(model_path)
        recorded_model_sha256 = item.get("model_sha256")
        if (
            isinstance(recorded_model_sha256, str)
            and recorded_model_sha256
            and model_sha256 != recorded_model_sha256
        ):
            raise ValueError(f"G0/F7源模型hash漂移: farm={item['farm_id']}")
        rows.append(
            {
                "model_family": MODEL_FAMILY,
                "variant_id": "g0",
                "variant_label": VARIANT_SPECS["g0"]["label"],
                "farm_id": str(item["farm_id"]),
                "feature_groups": SOURCE_FEATURE_GROUPS,
                "feature_count": SOURCE_FEATURE_COUNT,
                "reference_only": True,
                "requires_training": False,
                "random_seed": RANDOM_SEED,
                "parameter_count": int(item["total_params"]),
                "trainable_parameter_count": int(item["trainable_params"]),
                "source_variant": SOURCE_VARIANT,
                "source_model_path": os.path.abspath(model_path),
                "source_model_sha256": model_sha256,
                "source_artifact_path": os.path.abspath(artifact_path),
                "source_artifact_sha256": _sha256(artifact_path),
                "source_summary_path": os.path.abspath(path),
                "source_summary_sha256": _sha256(path),
                "result_source": "direct_reference_existing_f7_training",
                "validation_scope": "source_descriptive_not_stage3_selection",
            }
        )
    return rows


def write_manifest(result_root=RESULT_ROOT, run_scope="formal"):
    os.makedirs(result_root, exist_ok=True)
    rows = []
    for order, (variant_id, spec) in enumerate(VARIANT_SPECS.items()):
        rows.append(
            {
                "variant_order": order,
                "variant_id": variant_id,
                "label": spec["label"],
                "requires_training": spec["requires_training"],
                "factorized_gate": spec["factorized_gate"],
                "calibration_weight": spec["calibration_weight"],
                "dynamic_weight": spec["dynamic_weight"],
                "safety_weight": spec["safety_weight"],
                "source": spec["source"],
                "description": spec["description"],
                "feature_groups": SOURCE_FEATURE_GROUPS,
                "feature_count": SOURCE_FEATURE_COUNT,
                "expected_total_params": EXPECTED_TOTAL_PARAMS.get(variant_id),
                "random_seed": RANDOM_SEED,
                "batch_size": BATCH_SIZE,
                "selection_split": "test",
                "test_used_for_selection": True,
                "test_is_final_blind_evaluation": False,
                "g4_kappa_grid": json.dumps(G4_KAPPA_GRID),
                "calibration_difference_quantile": CALIBRATION_DIFFERENCE_QUANTILE,
                "calibration_weight_floor": CALIBRATION_WEIGHT_FLOOR,
                "checkpoint_primary": "validation_capacity_normalized_rmse",
                "checkpoint_tie_breakers": "positive_regret,brier",
                "protocol_version": PROTOCOL_VERSION,
                "run_scope": run_scope,
            }
        )
    path = os.path.join(result_root, MANIFEST_NAME)
    _atomic_to_csv(pd.DataFrame(rows), path)
    return path


def _marker_path():
    return os.path.join(RESULT_ROOT, TRAINING_MARKER_NAME)


def _clear_completion_markers():
    for path in (
        _marker_path(),
        os.path.join(RESULT_ROOT, PREDICTION_MARKER_RELATIVE_PATH),
    ):
        if os.path.exists(path):
            os.remove(path)


def _file_record(path):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"complete marker成员不存在: {path}")
    return {
        "path": os.path.abspath(path),
        "sha256": _sha256(path),
        "size_bytes": os.path.getsize(path),
    }


def publish_training_marker(summary_path, manifest_path, summary):
    files = {
        "training_summary": _file_record(summary_path),
        "experiment_manifest": _file_record(manifest_path),
        "training_code": _file_record(__file__),
        "source_feature_training_marker": _file_record(
            os.path.join(
                feature_train.RESULT_ROOT,
                feature_train.TRAINING_COMPLETION_NAME,
            )
        ),
    }
    new_rows = summary[summary["variant_id"].isin(TRAINABLE_VARIANTS)]
    for _, row in new_rows.iterrows():
        prefix = f"{row['variant_id']}.{row['farm_id']}"
        for key in (
            "model_path",
            "best_weights_path",
            "artifact_path",
            "history_path",
            "validation_diagnostics_path",
            "checkpoint_trace_path",
            "tail_path",
        ):
            files[f"{prefix}.{key}"] = _file_record(row[key])
    marker = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "expected_farm_ids": list(expected_farm_ids()),
        "variants": list(VARIANT_SPECS),
        "new_training_variants": list(TRAINABLE_VARIANTS),
        "new_model_count": int(len(new_rows)),
        "g0_reused_model_count": 5,
        "g4_requires_training": False,
        "g4_kappa_selection_split": "test",
        "test_is_final_blind_evaluation": False,
        "files": files,
    }
    return _atomic_write_json(marker, _marker_path())


def _partial_suffix(variants, farms):
    variant_tag = "-".join(variants)
    farm_tag = "-".join(map(str, farms))
    return f"_partial_{variant_tag}__farms_{farm_tag}"


def main():
    configure_reproducibility()
    formal_protocol = _validate_protocol_configuration()
    variants = get_requested_variants()
    train_files = discover_train_files()
    if not train_files:
        raise FileNotFoundError("未找到第三阶段训练文件")
    farm_ids = [get_farm_id(path) for path in train_files]
    trainable_requested = [item for item in variants if item in TRAINABLE_VARIANTS]
    full_matrix = (
        formal_protocol
        and set(variants) == set(VARIANT_SPECS)
        and set(farm_ids) == set(expected_farm_ids())
    )
    if full_matrix and trainable_requested:
        _clear_completion_markers()
    if full_matrix:
        run_result_root = RESULT_ROOT
        run_scope = "formal"
    else:
        partial_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + _partial_suffix(
            variants, farm_ids
        )
        run_result_root = os.path.join(RESULT_ROOT, "partial_runs", partial_name)
        run_scope = "partial_or_protocol_override"
    manifest_path = write_manifest(run_result_root, run_scope=run_scope)
    print(
        f"第三阶段场站={farm_ids}; 变体={variants}; "
        f"实际新增训练={trainable_requested}; 输出={run_result_root}"
    )
    rows = []
    if "g0" in variants:
        rows.extend(build_g0_reference_rows(farm_ids))
    for train_file in train_files:
        prepared = regime_train._prepare_farm(train_file)
        for variant_id in trainable_requested:
            print(
                f"\n===== {VARIANT_SPECS[variant_id]['label']} / "
                f"farm={prepared['farm_id']} / seed={RANDOM_SEED} ====="
            )
            rows.append(
                train_variant_for_farm(
                    variant_id,
                    prepared,
                    result_root=run_result_root,
                )
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("本次没有训练或引用结果可保存")
    if summary.duplicated(["variant_id", "farm_id"]).any():
        raise ValueError("训练汇总存在重复variant/farm")
    expected_new = len(trainable_requested) * len(farm_ids)
    actual_new = int(summary["variant_id"].isin(trainable_requested).sum())
    if actual_new != expected_new:
        raise ValueError(f"新增训练行数{actual_new} != {expected_new}")
    summary_path = os.path.join(run_result_root, TRAINING_SUMMARY_NAME)
    _atomic_to_csv(summary, summary_path)
    print(f"训练汇总: {summary_path}")
    if full_matrix:
        expected_rows = 5 * (1 + len(TRAINABLE_VARIANTS))
        if len(summary) != expected_rows:
            raise ValueError(
                f"正式训练summary应为{expected_rows}行，实际{len(summary)}"
            )
        marker = publish_training_marker(summary_path, manifest_path, summary)
        print(f"训练bundle完成标志: {marker}")
        print("G4为G3测试输出后处理，不创建重复模型/权重")
    else:
        print("partial运行不发布complete marker，也不覆盖正式summary")


if __name__ == "__main__":
    main()
