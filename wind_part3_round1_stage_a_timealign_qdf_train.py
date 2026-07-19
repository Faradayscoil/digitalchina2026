"""第三部分新模块开发 / Round-1 Stage-A 训练入口。

本轮以现行 X0（亦即 D0/T0/G0/F7）为唯一父快照，验证两类训练期增强：

* TimeAlign 风电化改造：只在训练期使用未来真值残差 teacher，将过去表示与
  ``Y - Persistence`` 的局部 patch 形态和全局 patch 关系对齐，并由只看历史
  P+H+D 的 stable/dynamic/ramp 软工况调节逐样本对齐强度；
* regime-conditioned QDF：用只由历史 P+H+D context 决定的软工况，构造
  半正定的 16-horizon 二次误差目标。

实验矩阵：

    A0  X0/F7 只读引用，不训练、不复制模型；
    A1  X0 + regime-QDF；
    A2  X0 + local residual alignment；
    A3  X0 + global residual alignment；
    A4  X0 + local + global residual alignment；
    A5  A4 + regime-QDF。

A1--A5 每个场站均从同一个 F7 文件独立加载。父模型中只有 B2 residual 的四个
有权层允许微调；Persistence、P+H+D context 和 G0 gate 严格冻结。future
teacher、student projector 与 QDF 只属于训练 wrapper，正式 ``.keras`` 模型只有
``history_features`` 一个输入，推理时完全不存在未来输入。

重要协议边界：本文件只发现 ``wind_train_*.csv``，不会导入预测模块，也不会
读取任何测试 CSV/测试结果。测试集上的一次性比较由配套 predict 文件负责。

参考思想（clean-room Keras adaptation）：
https://github.com/TROUBADOUR000/TimeAlign
upstream main commit: ab2dff5bde250f82e29d8755f87a494921857d71
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
from datetime import datetime, timezone

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import wind_controlled_gate_cali_train as gate_train
import wind_RegimeEncoder_PatchTST_feature_screen_train as feature_train
import wind_RegimeEncoder_PatchTST_train as regime_train
from wind_FeTS_PatchTST_train import NonFiniteTrainingGuard, ensure_finite_training_history
from wind_dl_model_train import (
    DATA_DIR,
    FORECAST_LEN,
    HISTORY_LEN,
    make_window_dataset,
    set_global_seed,
)

MODEL_FAMILY = "part3_stagea_timealign_qdf"
ARCHITECTURE_VERSION = "part3_round1_stagea_timealign_qdf_v1"
PROTOCOL_VERSION = "part3_round1_stagea_legacy_test_selected_v1"
ARTIFACT_SCHEMA_VERSION = 1
RESULT_ROOT = os.path.join(
    ".",
    "wind_results",
    "part3_new_module_supplement",
    "01_stage_a_timealign_residual_alignment_qdf",
)
TRAINING_MARKER_NAME = "stage_a_training_bundle_complete.json"
RUNNING_MARKER_NAME = "stage_a_training_bundle_running.json"
TRAINING_SUMMARY_NAME = "stage_a_training_metrics.csv"
EXPERIMENT_MANIFEST_NAME = "stage_a_experiment_manifest.csv"
COMPLEXITY_REPORT_NAME = "stage_a_training_complexity.csv"

SOURCE_VARIANT = "f7"
SOURCE_ALIAS = "x0/d0/t0/g0/f7"
RANDOM_SEED = 2026
EXPECTED_FARM_COUNT = 5
HISTORY_PATCH_COUNT = 4
FUTURE_PATCH_LENGTH = FORECAST_LEN // HISTORY_PATCH_COUNT
ALIGNMENT_DIM = 32
TEACHER_FF_DIM = 64
TEACHER_BLOCKS = 2

DEFAULT_BATCH_SIZE = 192
DEFAULT_EPOCHS = 60
DEFAULT_VALIDATION_SPLIT = 0.15
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_PATIENCE = 8
DEFAULT_TEACHER_WARMUP_EPOCHS = 3
FUSED_DIAGNOSTIC_WEIGHT = 0.0
CANDIDATE_PRIMARY_WEIGHT = 1.0
TEACHER_RECONSTRUCTION_WEIGHT = 0.20
LOCAL_ALIGNMENT_WEIGHT = 0.10
GLOBAL_ALIGNMENT_WEIGHT = 0.05
QDF_WEIGHT = 0.25
RAMP_WEIGHT = 0.10
QDF_IDENTITY_WEIGHT = 1e-3
ALIGNMENT_LOCAL_MARGIN = 0.0
ALIGNMENT_GLOBAL_MARGIN = 0.0
LOCAL_REGIME_MULTIPLIERS = (0.50, 1.00, 1.50)
GLOBAL_REGIME_MULTIPLIERS = (0.75, 1.20, 1.50)
EPSILON = 1e-6
IDEAL_INFERENCE_PARAMETER_REFERENCE = 30_000

UPSTREAM_REPOSITORY = "https://github.com/TROUBADOUR000/TimeAlign"
UPSTREAM_COMMIT = "ab2dff5bde250f82e29d8755f87a494921857d71"

if FORECAST_LEN % HISTORY_PATCH_COUNT:
    raise RuntimeError("FORECAST_LEN 必须能被4个未来1小时patch整除")


VARIANT_SPECS = {
    "a0": {
        "label": "A0 X0/D0/T0/G0/F7 read-only reference",
        "directory_name": "a0_x0_reference",
        "requires_training": False,
        "local_alignment": False,
        "global_alignment": False,
        "qdf": False,
        "selection_eligible": True,
    },
    "a1": {
        "label": "A1 X0 + regime-conditioned QDF",
        "directory_name": "a1_regime_qdf",
        "requires_training": True,
        "local_alignment": False,
        "global_alignment": False,
        "qdf": True,
        "selection_eligible": True,
    },
    "a2": {
        "label": "A2 X0 + local residual alignment",
        "directory_name": "a2_local_residual_alignment",
        "requires_training": True,
        "local_alignment": True,
        "global_alignment": False,
        "qdf": False,
        "selection_eligible": True,
    },
    "a3": {
        "label": "A3 X0 + global residual alignment",
        "directory_name": "a3_global_residual_alignment",
        "requires_training": True,
        "local_alignment": False,
        "global_alignment": True,
        "qdf": False,
        "selection_eligible": True,
    },
    "a4": {
        "label": "A4 X0 + local/global residual alignment",
        "directory_name": "a4_local_global_residual_alignment",
        "requires_training": True,
        "local_alignment": True,
        "global_alignment": True,
        "qdf": False,
        "selection_eligible": True,
    },
    "a5": {
        "label": "A5 A4 + regime-conditioned QDF",
        "directory_name": "a5_local_global_alignment_regime_qdf",
        "requires_training": True,
        "local_alignment": True,
        "global_alignment": True,
        "qdf": True,
        "selection_eligible": True,
    },
}

ALL_VARIANTS = tuple(VARIANT_SPECS)
TRAINABLE_VARIANTS = tuple(
    variant for variant, spec in VARIANT_SPECS.items() if spec["requires_training"]
)
RESIDUAL_WEIGHTED_LAYER_NAMES = tuple(regime_train.B2_WEIGHTED_LAYER_NAMES)
_TRAIN_ONLY_STATISTICS_CACHE = {}


def configure_reproducibility():
    set_global_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path, chunk_size=1024 * 1024):
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(named_arrays):
    digest = hashlib.sha256()
    for name, value in named_arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(name).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _weights_snapshot(model, excluded_layer_names=()):
    excluded = set(excluded_layer_names)
    arrays = []
    for layer in model.layers:
        if layer.name in excluded:
            continue
        for index, weight in enumerate(layer.weights):
            arrays.append((f"{layer.name}/{index}", np.asarray(weight.numpy()).copy()))
    return arrays


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


def expected_farm_ids():
    farm_ids = tuple(str(value) for value in feature_train.expected_training_farm_ids())
    if len(farm_ids) != EXPECTED_FARM_COUNT or len(set(farm_ids)) != EXPECTED_FARM_COUNT:
        raise ValueError(f"F7来源不是5场站唯一集合: {farm_ids}")
    return tuple(sorted(farm_ids))


def variant_model_name(variant_id):
    variant_id = str(variant_id).lower()
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"未知Stage-A变体: {variant_id}")
    return f"{MODEL_FAMILY}_{variant_id}"


def variant_dirs(variant_id, result_root=RESULT_ROOT, create=True):
    variant_id = str(variant_id).lower()
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"未知Stage-A变体: {variant_id}")
    root = os.path.join(result_root, VARIANT_SPECS[variant_id]["directory_name"])
    paths = {
        "root": root,
        "models": os.path.join(root, "models"),
        "weights": os.path.join(root, "weights"),
        "preprocess": os.path.join(root, "preprocess"),
        "history": os.path.join(root, "history"),
        "tensorboard": os.path.join(root, "tensorboard"),
        "validation_diagnostics": os.path.join(root, "validation_diagnostics"),
        "training_visualizations": os.path.join(root, "training_visualizations"),
        "references": os.path.join(root, "references"),
    }
    if create:
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
    return paths


def _training_paths(variant_id, farm_id, result_root):
    dirs = variant_dirs(variant_id, result_root=result_root)
    prefix = f"{variant_model_name(variant_id)}_farm_{farm_id}"
    return {
        "model": os.path.join(dirs["models"], f"{prefix}.keras"),
        "weights": os.path.join(dirs["weights"], f"{prefix}_trainer_best.weights.h5"),
        "artifact": os.path.join(dirs["preprocess"], f"{prefix}_preprocess.pkl"),
        "history": os.path.join(dirs["history"], f"{prefix}_history.csv"),
        "history_figure": os.path.join(dirs["history"], f"{prefix}_history.png"),
        "validation": os.path.join(
            dirs["validation_diagnostics"], f"{prefix}_validation.json"
        ),
        "tensorboard": os.path.join(dirs["tensorboard"], f"farm_{farm_id}"),
        "reference": os.path.join(dirs["references"], f"{prefix}_reference.json"),
    }


def diagnostic_model(model):
    """Expose the common inference packet expected by the Stage-A predictor."""
    return keras.Model(
        model.inputs,
        {
            "forecast": model.get_layer("forecast_power").output,
            "candidate": model.get_layer("candidate_forecast").output,
            "persistence": model.get_layer("persistence_forecast_candidate").output,
            "corrected": model.get_layer("corrected_forecast_candidate").output,
            "gate": model.get_layer("correction_gate").output,
            "residual": model.get_layer("persistence_residual").output,
            "residual_hidden": model.get_layer("residual_hidden").output,
            "regime_context": model.get_layer("regime_context").output,
        },
        name=f"{model.name}_stagea_diagnostic",
    )


@keras.utils.register_keras_serializable(package="WindPart3StageA")
class FutureResidualTeacher(layers.Layer):
    """Training-only future residual encoder/autoencoder.

    The teacher consumes capacity-normalized ``Y-P`` and is never connected to the
    exported inference model. Alignment callers must stop-gradient its tokens;
    reconstruction remains differentiable so the teacher learns a meaningful target.
    """

    def __init__(
        self,
        patch_count=HISTORY_PATCH_COUNT,
        patch_length=FUTURE_PATCH_LENGTH,
        d_model=ALIGNMENT_DIM,
        d_ff=TEACHER_FF_DIM,
        blocks=TEACHER_BLOCKS,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_count = int(patch_count)
        self.patch_length = int(patch_length)
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.blocks = int(blocks)
        self.dropout_rate = float(dropout)
        self.patch_projection = layers.Dense(self.d_model, name="patch_projection")
        self.ffn_layers = []
        self.dropout_layers = []
        self.norm_layers = []
        for index in range(self.blocks):
            self.ffn_layers.append(
                keras.Sequential(
                    [
                        layers.Dense(self.d_ff, activation="gelu"),
                        layers.Dense(self.d_model),
                    ],
                    name=f"teacher_ffn_{index}",
                )
            )
            self.dropout_layers.append(
                layers.Dropout(self.dropout_rate, name=f"teacher_dropout_{index}")
            )
            self.norm_layers.append(
                layers.LayerNormalization(epsilon=1e-6, name=f"teacher_norm_{index}")
            )
        self.decoder = layers.Dense(self.patch_length, name="teacher_decoder")
        self.position = None

    def build(self, input_shape):
        self.position = self.add_weight(
            name="teacher_patch_position",
            shape=(self.patch_count, self.d_model),
            initializer=keras.initializers.RandomNormal(stddev=0.02, seed=2137),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, residual, training=None):
        patches = tf.reshape(
            residual,
            [-1, self.patch_count, self.patch_length],
        )
        tokens = self.patch_projection(patches) + self.position[tf.newaxis, :, :]
        for ffn, dropout, norm in zip(
            self.ffn_layers, self.dropout_layers, self.norm_layers
        ):
            update = dropout(ffn(tokens), training=training)
            tokens = norm(tokens + update)
        reconstruction = tf.reshape(
            self.decoder(tokens), [-1, self.patch_count * self.patch_length]
        )
        return tokens, reconstruction

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_count": self.patch_count,
                "patch_length": self.patch_length,
                "d_model": self.d_model,
                "d_ff": self.d_ff,
                "blocks": self.blocks,
                "dropout": self.dropout_rate,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="WindPart3StageA")
class ResidualStudentProjector(layers.Layer):
    """Past student tokens with a mandatory path through predicted residual.

    ``hidden`` supplies the history representation while the projection of
    ``persistence_residual`` ensures alignment gradients cannot be absorbed solely by
    a new mapper without reaching the residual forecast head.
    """

    def __init__(
        self,
        patch_count=HISTORY_PATCH_COUNT,
        patch_length=FUTURE_PATCH_LENGTH,
        d_model=ALIGNMENT_DIM,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_count = int(patch_count)
        self.patch_length = int(patch_length)
        self.d_model = int(d_model)
        self.hidden_projection = layers.Dense(
            self.patch_count * self.d_model,
            name="student_hidden_projection",
        )
        self.residual_projection = layers.Dense(
            self.d_model,
            use_bias=False,
            name="student_residual_projection",
        )
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="student_norm")

    def call(self, inputs):
        hidden, normalized_residual = inputs
        hidden_tokens = tf.reshape(
            self.hidden_projection(hidden), [-1, self.patch_count, self.d_model]
        )
        residual_patches = tf.reshape(
            normalized_residual, [-1, self.patch_count, self.patch_length]
        )
        residual_tokens = self.residual_projection(residual_patches)
        return self.norm(hidden_tokens + residual_tokens)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_count": self.patch_count,
                "patch_length": self.patch_length,
                "d_model": self.d_model,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="WindPart3StageA")
class RegimeQuadraticObjective(layers.Layer):
    """Three-regime PSD 16-horizon quadratic objective.

    ``W_k=L_k L_k^T+eps I`` is trace-normalized to H. Responsibilities are fixed
    history-only soft assignments supplied by the trainer; they receive no gradient.
    """

    def __init__(self, regimes=3, horizon=FORECAST_LEN, epsilon=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.regimes = int(regimes)
        self.horizon = int(horizon)
        self.epsilon = float(epsilon)
        self.raw_lower = None

    def build(self, input_shape):
        diagonal_value = np.log(np.expm1(1.0 - self.epsilon))
        initial = np.zeros((self.regimes, self.horizon, self.horizon), np.float32)
        for regime in range(self.regimes):
            np.fill_diagonal(initial[regime], diagonal_value)
        self.raw_lower = self.add_weight(
            name="qdf_raw_cholesky",
            shape=initial.shape,
            initializer=keras.initializers.Constant(initial),
            trainable=True,
        )
        super().build(input_shape)

    def matrices(self):
        lower = tf.linalg.band_part(self.raw_lower, -1, 0)
        diagonal = tf.linalg.diag_part(lower)
        lower = tf.linalg.set_diag(lower, tf.nn.softplus(diagonal) + self.epsilon)
        identity = tf.eye(self.horizon, batch_shape=[self.regimes], dtype=lower.dtype)
        matrices = tf.matmul(lower, lower, transpose_b=True) + self.epsilon * identity
        trace = tf.linalg.trace(matrices)
        matrices = matrices * (
            tf.cast(self.horizon, matrices.dtype) / trace
        )[:, tf.newaxis, tf.newaxis]
        return matrices

    def call(self, inputs):
        error, responsibilities = inputs
        responsibilities = tf.stop_gradient(responsibilities)
        matrices = self.matrices()
        sample_matrix = tf.einsum("bk,kij->bij", responsibilities, matrices)
        quadratic = tf.einsum("bi,bij,bj->b", error, sample_matrix, error)
        quadratic = tf.reduce_mean(quadratic / tf.cast(self.horizon, error.dtype))
        identity = tf.eye(self.horizon, batch_shape=[self.regimes], dtype=error.dtype)
        identity_penalty = tf.reduce_mean(tf.square(matrices - identity))
        return quadratic, identity_penalty

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "regimes": self.regimes,
                "horizon": self.horizon,
                "epsilon": self.epsilon,
            }
        )
        return config


def _huber_mean(error, delta):
    absolute = tf.abs(error)
    delta = tf.cast(delta, error.dtype)
    return tf.reduce_mean(
        tf.where(
            absolute <= delta,
            0.5 * tf.square(error),
            delta * (absolute - 0.5 * delta),
        )
    )


def _weighted_batch_mean(values, sample_weight=None):
    values = tf.convert_to_tensor(values)
    if sample_weight is None:
        return tf.reduce_mean(values)
    sample_weight = tf.stop_gradient(tf.cast(sample_weight, values.dtype))
    return tf.reduce_sum(values * sample_weight) / (
        tf.reduce_sum(sample_weight) + tf.cast(EPSILON, values.dtype)
    )


def _local_alignment(
    student,
    teacher,
    sample_weight=None,
    margin=ALIGNMENT_LOCAL_MARGIN,
):
    student = tf.math.l2_normalize(student, axis=-1)
    teacher = tf.math.l2_normalize(teacher, axis=-1)
    # Direction-sensitive: ramp-up and ramp-down may not be treated as equivalent.
    cosine = tf.reduce_sum(student * teacher, axis=-1)
    per_sample = tf.reduce_mean(tf.nn.gelu(1.0 - cosine - margin), axis=-1)
    return _weighted_batch_mean(per_sample, sample_weight)


def _global_alignment(
    student,
    teacher,
    sample_weight=None,
    margin=ALIGNMENT_GLOBAL_MARGIN,
):
    student = tf.math.l2_normalize(student, axis=-1)
    teacher = tf.math.l2_normalize(teacher, axis=-1)
    student_gram = tf.matmul(student, student, transpose_b=True)
    teacher_gram = tf.matmul(teacher, teacher, transpose_b=True)
    per_sample = tf.reduce_mean(
        tf.nn.relu(tf.abs(student_gram - teacher_gram) - margin), axis=(1, 2)
    )
    return _weighted_batch_mean(per_sample, sample_weight)


def _balanced_alignment(local_loss, global_loss):
    average = tf.stop_gradient(0.5 * (local_loss + global_loss))
    local_weight = tf.clip_by_value(
        average / (tf.stop_gradient(local_loss) + EPSILON), 0.1, 10.0
    )
    global_weight = tf.clip_by_value(
        average / (tf.stop_gradient(global_loss) + EPSILON), 0.1, 10.0
    )
    return local_weight * local_loss + global_weight * global_loss


@keras.utils.register_keras_serializable(package="WindPart3StageA")
class StageATrainingModel(keras.Model):
    """Training wrapper; exported inference model never contains future teacher."""

    def __init__(
        self,
        inference_model,
        variant_id,
        target_scale,
        capacity,
        residual_q90,
        regime_centroids,
        regime_context_scale,
        regime_temperature=1.0,
        teacher_enabled=True,
        qdf_enabled=False,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name or f"StageATrainer_{variant_id.upper()}", **kwargs)
        self.inference_model = inference_model
        self.variant_id = str(variant_id)
        self.variant_spec = dict(VARIANT_SPECS[self.variant_id])
        self.target_scale_value = float(target_scale)
        self.capacity_value = float(capacity)
        self.residual_q90_value = np.asarray(residual_q90, np.float32)
        self.regime_centroids_value = np.asarray(regime_centroids, np.float32)
        self.regime_context_scale_value = np.asarray(regime_context_scale, np.float32)
        self.regime_temperature = float(regime_temperature)
        self.teacher_enabled = bool(teacher_enabled)
        self.qdf_enabled = bool(qdf_enabled)
        context_dim = int(inference_model.get_layer("regime_context").output.shape[-1])
        if not np.isfinite(self.target_scale_value) or self.target_scale_value <= 0:
            raise ValueError("target_scale必须是有限正数")
        if not np.isfinite(self.capacity_value) or self.capacity_value <= 0:
            raise ValueError("capacity必须是有限正数")
        if (
            self.residual_q90_value.shape != (FORECAST_LEN,)
            or not np.isfinite(self.residual_q90_value).all()
            or (self.residual_q90_value < EPSILON).any()
        ):
            raise ValueError("train-only residual_q90必须是有限正值的16维向量")
        if (
            self.regime_centroids_value.shape != (3, context_dim)
            or not np.isfinite(self.regime_centroids_value).all()
        ):
            raise ValueError("regime_centroids必须是有限的[3, context_dim]矩阵")
        if (
            self.regime_context_scale_value.shape != (context_dim,)
            or not np.isfinite(self.regime_context_scale_value).all()
            or (self.regime_context_scale_value < EPSILON).any()
        ):
            raise ValueError("regime_context_scale必须是有限正值context向量")
        self.forward_model = diagnostic_model(inference_model)
        self.future_teacher = (
            FutureResidualTeacher(name="future_residual_teacher")
            if self.teacher_enabled
            else None
        )
        self.student_projector = (
            ResidualStudentProjector(name="residual_student_projector")
            if self.teacher_enabled
            else None
        )
        self.qdf_objective = (
            RegimeQuadraticObjective(name="regime_qdf_objective")
            if self.qdf_enabled
            else None
        )
        self.teacher_only = False

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.forecast_loss_tracker = keras.metrics.Mean(name="forecast_loss")
        self.candidate_loss_tracker = keras.metrics.Mean(name="candidate_loss")
        self.reconstruction_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.local_tracker = keras.metrics.Mean(name="local_alignment_loss")
        self.global_tracker = keras.metrics.Mean(name="global_alignment_loss")
        self.qdf_tracker = keras.metrics.Mean(name="qdf_loss")
        self.ramp_tracker = keras.metrics.Mean(name="ramp_loss")
        self.forecast_mae = keras.metrics.MeanAbsoluteError(name="forecast_mae")
        self.forecast_rmse = keras.metrics.RootMeanSquaredError(name="forecast_rmse")
        self.candidate_mae = keras.metrics.MeanAbsoluteError(name="candidate_mae")
        self.candidate_rmse = keras.metrics.RootMeanSquaredError(name="candidate_rmse")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.forecast_loss_tracker,
            self.candidate_loss_tracker,
            self.reconstruction_tracker,
            self.local_tracker,
            self.global_tracker,
            self.qdf_tracker,
            self.ramp_tracker,
            self.forecast_mae,
            self.forecast_rmse,
            self.candidate_mae,
            self.candidate_rmse,
        ]

    def call(self, inputs, training=None):
        packet = self.forward_model(inputs, training=training)
        return {
            "forecast_power": packet["forecast"],
            "candidate_forecast": packet["candidate"],
        }

    def _responsibilities(self, context):
        centroids = tf.convert_to_tensor(self.regime_centroids_value, context.dtype)
        scale = tf.convert_to_tensor(self.regime_context_scale_value, context.dtype)
        standardized = (context[:, tf.newaxis, :] - centroids[tf.newaxis, :, :]) / scale[
            tf.newaxis, tf.newaxis, :
        ]
        logits = -tf.reduce_mean(tf.square(standardized), axis=-1)
        return tf.nn.softmax(logits / self.regime_temperature, axis=-1)

    def _loss_components(self, x, y, training):
        packet = self.forward_model(x, training=training)
        forecast = packet["forecast"]
        candidate = packet["candidate"]
        persistence = packet["persistence"]
        # Difference in scaled target units -> physical power -> per-unit capacity.
        unit_factor = tf.cast(self.target_scale_value / self.capacity_value, tf.float32)
        forecast_error = (forecast - y) * unit_factor
        candidate_error = (candidate - y) * unit_factor
        true_residual = (y - persistence) * unit_factor
        predicted_residual = packet["residual"] * unit_factor
        responsibilities = None
        if self.teacher_enabled or self.qdf_enabled:
            responsibilities = tf.stop_gradient(
                self._responsibilities(packet["regime_context"])
            )

        forecast_loss = _huber_mean(forecast_error, delta=0.05)
        candidate_loss = _huber_mean(candidate_error, delta=0.05)
        ramp_error = candidate_error[:, 1:] - candidate_error[:, :-1]
        ramp_loss = _huber_mean(ramp_error, delta=0.02)

        reconstruction_loss = tf.constant(0.0, tf.float32)
        local_loss = tf.constant(0.0, tf.float32)
        global_loss = tf.constant(0.0, tf.float32)
        alignment_term = tf.constant(0.0, tf.float32)
        if self.teacher_enabled:
            q90 = tf.convert_to_tensor(self.residual_q90_value, tf.float32)
            normalized_true = true_residual / q90[tf.newaxis, :]
            normalized_predicted = predicted_residual / q90[tf.newaxis, :]
            teacher_tokens, reconstruction = self.future_teacher(
                normalized_true, training=training
            )
            student_tokens = self.student_projector(
                [packet["residual_hidden"], normalized_predicted]
            )
            reconstruction_loss = _huber_mean(
                reconstruction - normalized_true, delta=1.0
            )
            stopped_teacher = tf.stop_gradient(teacher_tokens)
            local_regime_weight = tf.reduce_sum(
                responsibilities
                * tf.constant(LOCAL_REGIME_MULTIPLIERS, dtype=tf.float32)[
                    tf.newaxis, :
                ],
                axis=-1,
            )
            global_regime_weight = tf.reduce_sum(
                responsibilities
                * tf.constant(GLOBAL_REGIME_MULTIPLIERS, dtype=tf.float32)[
                    tf.newaxis, :
                ],
                axis=-1,
            )
            if self.variant_spec["local_alignment"]:
                local_loss = _local_alignment(
                    student_tokens,
                    stopped_teacher,
                    sample_weight=local_regime_weight,
                )
            if self.variant_spec["global_alignment"]:
                global_loss = _global_alignment(
                    student_tokens,
                    stopped_teacher,
                    sample_weight=global_regime_weight,
                )
            if self.variant_spec["local_alignment"] and self.variant_spec["global_alignment"]:
                alignment_term = _balanced_alignment(local_loss, global_loss)
            elif self.variant_spec["local_alignment"]:
                alignment_term = local_loss
            elif self.variant_spec["global_alignment"]:
                alignment_term = global_loss

        qdf_loss = tf.constant(0.0, tf.float32)
        qdf_identity = tf.constant(0.0, tf.float32)
        if self.qdf_enabled:
            candidate_qdf, qdf_identity = self.qdf_objective(
                [candidate_error, responsibilities]
            )
            qdf_loss = candidate_qdf

        if self.teacher_only:
            total = TEACHER_RECONSTRUCTION_WEIGHT * reconstruction_loss
        else:
            # Stage A isolates corrected-candidate quality. The frozen legacy G0 fused
            # forecast is logged only; gate re-calibration belongs to later Stage C.
            total = (
                FUSED_DIAGNOSTIC_WEIGHT * forecast_loss
                + CANDIDATE_PRIMARY_WEIGHT * candidate_loss
            )
            total += TEACHER_RECONSTRUCTION_WEIGHT * reconstruction_loss
            if self.variant_spec["local_alignment"] and self.variant_spec["global_alignment"]:
                total += 0.5 * (LOCAL_ALIGNMENT_WEIGHT + GLOBAL_ALIGNMENT_WEIGHT) * alignment_term
            elif self.variant_spec["local_alignment"]:
                total += LOCAL_ALIGNMENT_WEIGHT * alignment_term
            elif self.variant_spec["global_alignment"]:
                total += GLOBAL_ALIGNMENT_WEIGHT * alignment_term
            if self.qdf_enabled:
                total += (
                    QDF_WEIGHT * qdf_loss
                    + RAMP_WEIGHT * ramp_loss
                    + QDF_IDENTITY_WEIGHT * qdf_identity
                )
        if not self.teacher_only and self.inference_model.losses:
            total += tf.add_n(self.inference_model.losses)
        return {
            "total": total,
            "forecast_loss": forecast_loss,
            "candidate_loss": candidate_loss,
            "reconstruction_loss": reconstruction_loss,
            "local_loss": local_loss,
            "global_loss": global_loss,
            "qdf_loss": qdf_loss,
            "ramp_loss": ramp_loss,
            "forecast": forecast,
            "candidate": candidate,
        }

    def _update_metrics(self, components, y):
        self.loss_tracker.update_state(components["total"])
        self.forecast_loss_tracker.update_state(components["forecast_loss"])
        self.candidate_loss_tracker.update_state(components["candidate_loss"])
        self.reconstruction_tracker.update_state(components["reconstruction_loss"])
        self.local_tracker.update_state(components["local_loss"])
        self.global_tracker.update_state(components["global_loss"])
        self.qdf_tracker.update_state(components["qdf_loss"])
        self.ramp_tracker.update_state(components["ramp_loss"])
        self.forecast_mae.update_state(y, components["forecast"])
        self.forecast_rmse.update_state(y, components["forecast"])
        self.candidate_mae.update_state(y, components["candidate"])
        self.candidate_rmse.update_state(y, components["candidate"])

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            components = self._loss_components(x, y, training=True)
        variables = self.trainable_variables
        gradients = tape.gradient(components["total"], variables)
        pairs = [(gradient, variable) for gradient, variable in zip(gradients, variables) if gradient is not None]
        if not pairs:
            raise RuntimeError("Stage-A训练没有任何有效梯度")
        self.optimizer.apply_gradients(pairs)
        self._update_metrics(components, y)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        x, y = data
        components = self._loss_components(x, y, training=False)
        self._update_metrics(components, y)
        return {metric.name: metric.result() for metric in self.metrics}

    def get_config(self):
        # The wrapper is checkpoint-only. The formal inference model is saved separately.
        config = super().get_config()
        config.update(
            {
                "variant_id": self.variant_id,
                "target_scale": self.target_scale_value,
                "capacity": self.capacity_value,
                "residual_q90": self.residual_q90_value.tolist(),
                "regime_centroids": self.regime_centroids_value.tolist(),
                "regime_context_scale": self.regime_context_scale_value.tolist(),
                "regime_temperature": self.regime_temperature,
                "teacher_enabled": self.teacher_enabled,
                "qdf_enabled": self.qdf_enabled,
            }
        )
        return config


def get_stagea_custom_objects():
    objects = {}
    for module in (feature_train, gate_train):
        getter = getattr(module, "get_feature_screen_custom_objects", None)
        if getter is not None:
            objects.update(getter())
        getter = getattr(module, "get_controlled_gate_custom_objects", None)
        if getter is not None:
            objects.update(getter())
    for cls in (
        FutureResidualTeacher,
        ResidualStudentProjector,
        RegimeQuadraticObjective,
        StageATrainingModel,
    ):
        objects[cls.__name__] = cls
        objects[f"WindPart3StageA>{cls.__name__}"] = cls
    return objects


def _discover_train_files(requested_farms=None):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "wind_train_*.csv")))
    requested = set(map(str, requested_farms or ()))
    selected = []
    for path in files:
        basename = os.path.basename(path)
        if "test" in basename.lower() or not basename.startswith("wind_train_"):
            raise ValueError(f"Stage-A训练发现疑似测试文件，已拒绝: {path}")
        farm_id = regime_train.get_farm_id(path)
        if not requested or farm_id in requested:
            selected.append(path)
    if requested and {regime_train.get_farm_id(path) for path in selected} != requested:
        found = {regime_train.get_farm_id(path) for path in selected}
        raise FileNotFoundError(f"请求场站训练文件不完整: missing={sorted(requested-found)}")
    if not selected:
        raise FileNotFoundError(f"{DATA_DIR} 下没有Stage-A训练文件")
    return selected


def _plain_datasets(prepared, batch_size, validation_split, shuffle_train=True):
    return make_window_dataset(
        prepared["features"],
        prepared["target"],
        HISTORY_LEN,
        FORECAST_LEN,
        batch_size,
        validation_split,
        shuffle_train=shuffle_train,
    )


def _sequential_history_dataset(features, sample_count, batch_size):
    data_slice = features[: sample_count + HISTORY_LEN - 1]
    dataset = keras.utils.timeseries_dataset_from_array(
        data=data_slice,
        targets=None,
        sequence_length=HISTORY_LEN,
        sequence_stride=1,
        shuffle=False,
        batch_size=batch_size,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def _train_window_arrays(prepared, train_samples):
    target = np.asarray(prepared["target"], np.float32)
    target_windows = np.lib.stride_tricks.sliding_window_view(target, FORECAST_LEN)
    target_windows = target_windows[
        HISTORY_LEN : HISTORY_LEN + train_samples
    ].astype(np.float32)
    target_history_scaled = (
        np.asarray(prepared["features"][:, prepared["target_index"]], np.float32)
        * float(prepared["power_scale_ratio"])
        + float(prepared["power_scale_offset"])
    )
    history_windows = np.lib.stride_tricks.sliding_window_view(
        target_history_scaled, HISTORY_LEN
    )[:train_samples].astype(np.float32)
    if len(target_windows) != train_samples or len(history_windows) != train_samples:
        raise ValueError("train-only窗口统计与dataset样本数不一致")
    return history_windows, target_windows


def _compute_train_only_statistics(
    prepared,
    inference_model,
    train_samples,
    batch_size,
):
    """Compute every teacher/QDF statistic from the chronological train split only."""
    history_windows, future_windows = _train_window_arrays(prepared, train_samples)
    target_scale = float(prepared["scaler_y"].scale_[0])
    capacity = float(prepared["capacity"])
    unit_factor = target_scale / capacity
    persistence = history_windows[:, -1:]
    true_residual = (future_windows - persistence) * unit_factor
    residual_q90 = np.quantile(np.abs(true_residual), 0.90, axis=0).astype(np.float32)
    residual_q90 = np.maximum(residual_q90, np.float32(1e-4))

    physical_history = history_windows * unit_factor
    recent16 = physical_history[:, -16:]
    recent8 = physical_history[:, -8:]
    activity = np.mean(np.abs(np.diff(recent16, axis=1)), axis=1)
    slope = np.abs(recent8[:, -1] - recent8[:, 0])
    thresholds = {
        "activity_q40": float(np.quantile(activity, 0.40)),
        "activity_q75": float(np.quantile(activity, 0.75)),
        "slope_q40": float(np.quantile(slope, 0.40)),
        "slope_q75": float(np.quantile(slope, 0.75)),
    }
    stable = (activity <= thresholds["activity_q40"]) & (
        slope <= thresholds["slope_q40"]
    )
    ramp = (~stable) & (
        (activity >= thresholds["activity_q75"])
        | (slope >= thresholds["slope_q75"])
    )
    labels = np.ones(train_samples, dtype=np.int32)  # dynamic
    labels[stable] = 0
    labels[ramp] = 2

    context_extractor = keras.Model(
        inference_model.inputs,
        inference_model.get_layer("regime_context").output,
        name=f"{inference_model.name}_train_context_extractor",
    )
    context_parts = []
    for batch_x in _sequential_history_dataset(
        prepared["features"], train_samples, batch_size
    ):
        context_parts.append(
            np.asarray(context_extractor(batch_x, training=False), np.float32)
        )
    context = np.concatenate(context_parts, axis=0)
    if len(context) != train_samples or not np.isfinite(context).all():
        raise ValueError("train-only P+H+D context统计不完整或包含非有限值")
    global_median = np.median(context, axis=0)
    context_scale = 1.4826 * np.median(
        np.abs(context - global_median[None, :]), axis=0
    )
    context_scale = np.maximum(context_scale.astype(np.float32), np.float32(1e-3))
    centroids = []
    counts = {}
    for class_id, name in enumerate(("stable", "dynamic", "ramp")):
        class_context = context[labels == class_id]
        counts[name] = int(len(class_context))
        if len(class_context) < 8:
            raise ValueError(
                f"train-only {name}工况仅{len(class_context)}个窗口，无法构造QDF centroid"
            )
        centroids.append(np.median(class_context, axis=0))
    centroids = np.asarray(centroids, np.float32)
    if not np.isfinite(centroids).all():
        raise ValueError("train-only regime centroids包含非有限值")
    return {
        "residual_q90": residual_q90,
        "regime_centroids": centroids,
        "regime_context_scale": context_scale,
        "regime_thresholds": thresholds,
        "regime_counts": counts,
        "train_stat_sample_count": int(train_samples),
        "residual_q90_sha256": _array_sha256([("residual_q90", residual_q90)]),
        "regime_stats_sha256": _array_sha256(
            [("centroids", centroids), ("scale", context_scale)]
        ),
    }


def _configure_core_trainability(model):
    """Freeze the complete X0 graph except the four weighted B2 residual layers."""
    for layer in model.layers:
        layer.trainable = False
    for name in RESIDUAL_WEIGHTED_LAYER_NAMES:
        model.get_layer(name).trainable = True
    # Dropout has no weights. Residual dropout remains active as source regularization;
    # all frozen context/gate dropout must be disabled during trainer(training=True).
    residual_dropout = model.get_layer("residual_dropout")
    residual_dropout.trainable = True
    for name in ("regime_context_dropout",):
        layer = model.get_layer(name)
        layer.rate = 0.0
        layer.trainable = False
    gate = model.get_layer("correction_gate")
    gate.trainable = False
    for attribute in ("context_dropout", "gate_dropout"):
        dropout = getattr(gate, attribute, None)
        if dropout is not None:
            dropout.rate = 0.0
            dropout.trainable = False

    actual = {
        layer.name
        for layer in model.layers
        if layer.trainable_weights
    }
    expected = set(RESIDUAL_WEIGHTED_LAYER_NAMES)
    if actual != expected:
        raise ValueError(f"X0可训练核心层异常: actual={sorted(actual)}, expected={sorted(expected)}")
    return tuple(sorted(actual))


def _packet_arrays(model, batch_x):
    packet = diagnostic_model(model)(batch_x, training=False)
    return {key: np.asarray(value) for key, value in packet.items()}


def _identity_hashes(packet):
    keys = ("persistence", "corrected", "gate", "forecast")
    return {key: _array_sha256([(key, packet[key])]) for key in keys}


def _assert_packet_exact(left, right, keys, label):
    drift = {}
    for key in keys:
        a = np.asarray(left[key])
        b = np.asarray(right[key])
        value = float(np.max(np.abs(a - b))) if a.size else 0.0
        drift[key] = value
        if not np.array_equal(a, b):
            raise ValueError(f"{label}: {key}没有精确复现父X0，max_abs={value}")
    return drift


def _prepare_source(train_file):
    if "test" in os.path.basename(train_file).lower():
        raise ValueError(f"训练入口禁止测试文件: {train_file}")
    prepared = regime_train._prepare_farm(train_file)
    farm_id = str(prepared["farm_id"])
    model, artifact, artifact_path, model_path = gate_train.load_source_f7(farm_id)
    gate_train._validate_prepared_against_source(prepared, artifact)
    if int(model.count_params()) != feature_train.EXPECTED_PARAMETER_COUNTS[SOURCE_VARIANT]:
        raise ValueError(f"F7/{farm_id}参数量身份异常")
    return prepared, model, artifact, artifact_path, model_path


def _history_frame(histories):
    frames = []
    offset = 0
    for phase, history in histories:
        values = dict(history.history)
        length = max((len(value) for value in values.values()), default=0)
        frame = pd.DataFrame({key: list(value) for key, value in values.items()})
        frame.insert(0, "phase_epoch", np.arange(1, length + 1))
        frame.insert(0, "epoch", np.arange(offset + 1, offset + length + 1))
        frame.insert(0, "phase", phase)
        frames.append(frame)
        offset += length
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _plot_history(frame, path, title):
    if frame.empty:
        raise ValueError("训练history为空，不能生成三联图")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = (
        ("loss", "val_loss", "Capacity-normalized objective"),
        ("candidate_mae", "val_candidate_mae", "Candidate MAE (scaled target)"),
        ("candidate_rmse", "val_candidate_rmse", "Candidate RMSE (scaled target)"),
    )
    for axis, (train_key, val_key, label) in zip(axes, panels):
        if train_key in frame:
            axis.plot(frame["epoch"], frame[train_key], label="train")
        if val_key in frame:
            axis.plot(frame["epoch"], frame[val_key], label="validation")
        for boundary in frame.loc[frame["phase"].ne(frame["phase"].shift()), "epoch"].iloc[1:]:
            axis.axvline(boundary - 0.5, color="grey", linestyle="--", alpha=0.4)
        axis.set_title(label)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(title)
    figure.tight_layout()
    temporary = f"{path}.tmp.png"
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, path)
    return path


def _qdf_diagnostics(trainer):
    if trainer.qdf_objective is None:
        return {
            "enabled": False,
            "degeneration_guard_pass": True,
            "matrices": None,
        }
    matrices = np.asarray(trainer.qdf_objective.matrices().numpy(), np.float64)
    rows = []
    guard = True
    for index, name in enumerate(("stable", "dynamic", "ramp")):
        matrix = matrices[index]
        eigenvalues = np.linalg.eigvalsh(matrix)
        condition = float(np.max(eigenvalues) / max(np.min(eigenvalues), 1e-12))
        offdiag = matrix - np.diag(np.diag(matrix))
        identity_distance = float(np.linalg.norm(matrix - np.eye(FORECAST_LEN), ord="fro"))
        passed = bool(
            np.isfinite(matrix).all()
            and float(np.min(eigenvalues)) > 0.0
            and condition <= 1e4
            and np.isclose(np.trace(matrix), FORECAST_LEN, rtol=1e-4, atol=1e-4)
        )
        guard = guard and passed
        rows.append(
            {
                "regime": name,
                "min_eigenvalue": float(np.min(eigenvalues)),
                "max_eigenvalue": float(np.max(eigenvalues)),
                "condition_number": condition,
                "offdiagonal_frobenius": float(np.linalg.norm(offdiag, ord="fro")),
                "identity_frobenius": identity_distance,
                "trace": float(np.trace(matrix)),
                "guard_pass": passed,
            }
        )
    if not guard:
        raise ValueError(f"QDF矩阵退化守门失败: {rows}")
    return {
        "enabled": True,
        "degeneration_guard_pass": guard,
        "matrices": matrices.tolist(),
        "per_regime": rows,
    }


def _count_parameters(weights):
    return int(sum(int(np.prod(weight.shape)) for weight in weights))


def _validation_physical_metrics(model, dataset, prepared):
    truth_parts, forecast_parts, candidate_parts = [], [], []
    diag = diagnostic_model(model)
    for batch_x, batch_y in dataset:
        packet = diag(batch_x, training=False)
        truth_parts.append(np.asarray(batch_y))
        forecast_parts.append(np.asarray(packet["forecast"]))
        candidate_parts.append(np.asarray(packet["candidate"]))
    truth_scaled = np.concatenate(truth_parts)
    forecast_scaled = np.concatenate(forecast_parts)
    candidate_scaled = np.concatenate(candidate_parts)
    scale = float(prepared["scaler_y"].scale_[0])
    capacity = float(prepared["capacity"])

    def metrics(prediction):
        error = (prediction - truth_scaled) * scale
        return {
            "nrmse": float(np.sqrt(np.mean(np.square(error))) / capacity),
            "nmae": float(np.mean(np.abs(error)) / capacity),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "mae": float(np.mean(np.abs(error))),
        }

    return {
        "sample_count": int(len(truth_scaled)),
        "forecast": metrics(forecast_scaled),
        "candidate": metrics(candidate_scaled),
    }


def _build_trainer(
    model,
    variant_id,
    prepared,
    statistics,
):
    spec = VARIANT_SPECS[variant_id]
    teacher_enabled = bool(spec["local_alignment"] or spec["global_alignment"])
    qdf_enabled = bool(spec["qdf"])
    trainer = StageATrainingModel(
        inference_model=model,
        variant_id=variant_id,
        target_scale=float(prepared["scaler_y"].scale_[0]),
        capacity=float(prepared["capacity"]),
        residual_q90=statistics["residual_q90"],
        regime_centroids=statistics["regime_centroids"],
        regime_context_scale=statistics["regime_context_scale"],
        teacher_enabled=teacher_enabled,
        qdf_enabled=qdf_enabled,
    )
    if teacher_enabled != (variant_id in {"a2", "a3", "a4", "a5"}):
        raise ValueError(f"{variant_id} teacher启用矩阵异常")
    if qdf_enabled != (variant_id in {"a1", "a5"}):
        raise ValueError(f"{variant_id} QDF启用矩阵异常")
    return trainer


def _compile_trainer(trainer, learning_rate):
    optimizer = keras.optimizers.Adam(
        learning_rate=float(learning_rate),
        clipnorm=1.0,
    )
    trainer.compile(optimizer=optimizer)
    build = getattr(optimizer, "build", None)
    if build is not None:
        build(trainer.trainable_variables)
    return trainer


def _build_auxiliary_variables(trainer, sample_x, sample_y):
    # Calling the complete loss once creates teacher/student/QDF variables before
    # ModelCheckpoint/save_weights. It performs no update.
    _ = trainer(sample_x, training=False)
    components = trainer._loss_components(sample_x, sample_y, training=False)
    if not trainer.built:
        raise RuntimeError("Stage-A training wrapper在dummy forward后仍未built")
    values = [float(value.numpy()) for key, value in components.items() if key not in {"forecast", "candidate"}]
    if not np.isfinite(values).all():
        raise ValueError("Stage-A初始loss包含非有限值")


def _base_trainable_parameter_count(model):
    return _count_parameters(
        [
            weight
            for name in RESIDUAL_WEIGHTED_LAYER_NAMES
            for weight in model.get_layer(name).trainable_weights
        ]
    )


def _write_reference(variant_id, prepared, source_artifact_path, source_model_path, result_root):
    farm_id = str(prepared["farm_id"])
    paths = _training_paths(variant_id, farm_id, result_root)
    reference = {
        "model_family": MODEL_FAMILY,
        "architecture_version": ARCHITECTURE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "variant_id": "a0",
        "variant_label": VARIANT_SPECS["a0"]["label"],
        "farm_id": farm_id,
        "reference_only": True,
        "requires_training": False,
        "source_alias": SOURCE_ALIAS,
        "source_variant": SOURCE_VARIANT,
        "source_model_path": os.path.abspath(source_model_path),
        "source_model_sha256": _sha256(source_model_path),
        "source_artifact_path": os.path.abspath(source_artifact_path),
        "source_artifact_sha256": _sha256(source_artifact_path),
        "inference_parameter_count": int(feature_train.EXPECTED_PARAMETER_COUNTS["f7"]),
        "training_wrapper_parameter_count": 0,
        "training_only_parameter_count": 0,
        "random_seed": RANDOM_SEED,
        "created_at": _utc_now(),
        "no_model_copied": True,
    }
    _atomic_write_json(reference, paths["reference"])
    reference["reference_path"] = os.path.abspath(paths["reference"])
    reference["reference_sha256"] = _sha256(paths["reference"])
    return reference


def _train_variant_for_farm(
    variant_id,
    train_file,
    result_root,
    batch_size,
    epochs,
    validation_split,
    learning_rate,
    patience,
    teacher_warmup_epochs,
):
    if variant_id not in TRAINABLE_VARIANTS:
        raise ValueError(f"{variant_id}不是可训练Stage-A变体")
    keras.backend.clear_session()
    configure_reproducibility()
    prepared, model, source_artifact, source_artifact_path, source_model_path = (
        _prepare_source(train_file)
    )
    farm_id = str(prepared["farm_id"])
    paths = _training_paths(variant_id, farm_id, result_root)
    print(f"\n===== Stage-A {variant_id.upper()} / farm={farm_id} / seed={RANDOM_SEED} =====")

    train_ds, val_ds, train_samples, total_samples = _plain_datasets(
        prepared,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle_train=True,
    )
    sample_x, sample_y = next(iter(train_ds))
    probe_x = sample_x[: min(4, int(sample_x.shape[0]))]
    source_packet = _packet_arrays(model, probe_x)
    source_initial_hashes = _identity_hashes(source_packet)
    source_model_sha256 = _sha256(source_model_path)

    trainable_core_layers = _configure_core_trainability(model)
    configured_packet = _packet_arrays(model, probe_x)
    initial_identity_drift = _assert_packet_exact(
        source_packet,
        configured_packet,
        ("persistence", "corrected", "gate", "forecast"),
        f"{variant_id}/{farm_id}初始身份",
    )
    frozen_snapshot_before = _weights_snapshot(
        model, excluded_layer_names=RESIDUAL_WEIGHTED_LAYER_NAMES
    )
    frozen_sha_before = _array_sha256(frozen_snapshot_before)
    all_core_sha_before_warmup = _array_sha256(_weights_snapshot(model))

    statistics_cache_key = (
        farm_id,
        source_model_sha256,
        int(train_samples),
        float(validation_split),
    )
    statistics_reused = statistics_cache_key in _TRAIN_ONLY_STATISTICS_CACHE
    if statistics_reused:
        statistics = _TRAIN_ONLY_STATISTICS_CACHE[statistics_cache_key]
    else:
        statistics = _compute_train_only_statistics(
            prepared,
            model,
            train_samples=train_samples,
            batch_size=batch_size,
        )
        _TRAIN_ONLY_STATISTICS_CACHE[statistics_cache_key] = statistics
    trainer = _build_trainer(model, variant_id, prepared, statistics)
    _build_auxiliary_variables(trainer, sample_x, sample_y)

    inference_parameter_count = int(model.count_params())
    inference_trainable_parameter_count = _base_trainable_parameter_count(model)
    training_wrapper_parameter_count = int(
        _count_parameters(trainer.weights)
    )
    training_wrapper_trainable_parameter_count = int(
        _count_parameters(trainer.trainable_weights)
    )
    training_only_parameter_count = int(
        training_wrapper_parameter_count - inference_parameter_count
    )
    training_only_trainable_parameter_count = int(
        training_wrapper_trainable_parameter_count - inference_trainable_parameter_count
    )
    if training_only_parameter_count < 0 or training_only_trainable_parameter_count < 0:
        raise ValueError("训练wrapper参数量小于inference参数量")

    histories = []
    start = time.monotonic()
    if trainer.teacher_enabled and teacher_warmup_epochs > 0:
        trainer.teacher_only = True
        _compile_trainer(trainer, learning_rate)
        warmup_guard = NonFiniteTrainingGuard()
        warmup = trainer.fit(
            train_ds,
            validation_data=val_ds,
            epochs=int(teacher_warmup_epochs),
            callbacks=[warmup_guard],
            verbose=1,
        )
        ensure_finite_training_history(warmup, warmup_guard)
        histories.append(("teacher_warmup", warmup))
        all_core_sha_after_warmup = _array_sha256(_weights_snapshot(model))
        if all_core_sha_after_warmup != all_core_sha_before_warmup:
            raise ValueError(
                f"{variant_id}/{farm_id} teacher-only warmup意外修改了X0核心权重"
            )
    else:
        all_core_sha_after_warmup = all_core_sha_before_warmup

    trainer.teacher_only = False
    _compile_trainer(trainer, learning_rate)
    guard = NonFiniteTrainingGuard()
    tensorboard_dir = os.path.join(
        paths["tensorboard"], datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    callbacks = [
        guard,
        keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=0,
            write_graph=True,
            update_freq="epoch",
            profile_batch=0,
        ),
        keras.callbacks.ModelCheckpoint(
            paths["weights"],
            monitor="val_candidate_rmse",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_candidate_rmse",
            mode="min",
            patience=int(patience),
            restore_best_weights=False,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_candidate_rmse",
            mode="min",
            factor=0.5,
            patience=max(2, int(patience) // 3),
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    objective_history = trainer.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(epochs),
        callbacks=callbacks,
        verbose=1,
    )
    ensure_finite_training_history(objective_history, guard)
    histories.append(("stage_a_objective", objective_history))
    if not os.path.isfile(paths["weights"]):
        raise FileNotFoundError(f"未生成验证集最佳checkpoint: {paths['weights']}")
    trainer.load_weights(paths["weights"])
    elapsed = float(time.monotonic() - start)

    history_frame = _history_frame(histories)
    if history_frame.empty or not np.isfinite(
        history_frame.select_dtypes(include=[np.number]).to_numpy()
    ).all():
        raise ValueError(f"{variant_id}/{farm_id} history为空或包含非有限值")
    _atomic_to_csv(history_frame, paths["history"])
    _plot_history(
        history_frame,
        paths["history_figure"],
        f"Stage-A {variant_id.upper()} / farm {farm_id}",
    )

    frozen_snapshot_after = _weights_snapshot(
        model, excluded_layer_names=RESIDUAL_WEIGHTED_LAYER_NAMES
    )
    frozen_sha_after = _array_sha256(frozen_snapshot_after)
    if frozen_sha_after != frozen_sha_before:
        raise ValueError(f"{variant_id}/{farm_id}非residual父权重发生漂移")
    final_packet = _packet_arrays(model, probe_x)
    frozen_output_drift = _assert_packet_exact(
        source_packet,
        final_packet,
        ("persistence", "gate"),
        f"{variant_id}/{farm_id}冻结输出",
    )

    validation_metrics = _validation_physical_metrics(model, val_ds, prepared)
    _atomic_write_json(validation_metrics, paths["validation"])
    qdf_diagnostics = _qdf_diagnostics(trainer)

    # Checkpoint-only wrapper rebuild/load smoke. Never serialize the wrapper itself.
    rebuilt_model, _, _, rebuilt_source_model_path = gate_train.load_source_f7(farm_id)
    if _sha256(rebuilt_source_model_path) != source_model_sha256:
        raise ValueError("wrapper重建时F7父模型hash发生变化")
    _configure_core_trainability(rebuilt_model)
    rebuilt = _build_trainer(rebuilt_model, variant_id, prepared, statistics)
    _build_auxiliary_variables(rebuilt, sample_x, sample_y)
    rebuilt.load_weights(paths["weights"])
    rebuilt_packet = _packet_arrays(rebuilt_model, probe_x)
    reload_drift = {}
    for key in ("persistence", "corrected", "gate", "forecast"):
        value = float(np.max(np.abs(rebuilt_packet[key] - final_packet[key])))
        reload_drift[key] = value
        if not np.array_equal(rebuilt_packet[key], final_packet[key]):
            raise ValueError(
                f"{variant_id}/{farm_id} trainer checkpoint重建{key}不一致: {value}"
            )

    if len(model.inputs) != 1 or model.inputs[0].name.split(":")[0] != "history_features":
        raise ValueError("正式inference模型不是唯一history_features输入")
    forbidden = ("future_teacher", "teacher_decoder", "student_projector", "qdf_objective")
    offending = [
        layer.name for layer in model.layers if any(token in layer.name for token in forbidden)
    ]
    if offending:
        raise ValueError(f"训练期teacher/QDF意外进入inference图: {offending}")
    temporary_model_path = f"{paths['model']}.tmp.keras"
    if os.path.exists(temporary_model_path):
        os.remove(temporary_model_path)
    model.save(temporary_model_path)
    temporary_restored = keras.models.load_model(
        temporary_model_path, custom_objects=get_stagea_custom_objects(), compile=False
    )
    temporary_packet = _packet_arrays(temporary_restored, probe_x)
    for key in ("persistence", "corrected", "gate", "forecast"):
        if not np.array_equal(temporary_packet[key], final_packet[key]):
            raise ValueError(f"临时inference模型保存重载{key}不一致")
    os.replace(temporary_model_path, paths["model"])
    restored = keras.models.load_model(
        paths["model"], custom_objects=get_stagea_custom_objects(), compile=False
    )
    if len(restored.inputs) != 1:
        raise ValueError("保存后重载的Stage-A模型不是单历史输入")
    restored_packet = _packet_arrays(restored, probe_x)
    exported_drift = {}
    for key in ("persistence", "corrected", "gate", "forecast"):
        value = float(np.max(np.abs(restored_packet[key] - final_packet[key])))
        exported_drift[key] = value
        if not np.array_equal(restored_packet[key], final_packet[key]):
            raise ValueError(f"正式inference模型保存重载{key}不一致: {value}")

    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "architecture_version": ARCHITECTURE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "variant_id": variant_id,
        "variant_label": VARIANT_SPECS[variant_id]["label"],
        "variant_config": VARIANT_SPECS[variant_id],
        "farm_id": farm_id,
        "random_seed": RANDOM_SEED,
        "history_len": HISTORY_LEN,
        "forecast_len": FORECAST_LEN,
        "input_cols": list(prepared["input_cols"]),
        "feature_cols": list(prepared["feature_cols"]),
        "target_index": int(prepared["target_index"]),
        "capacity": float(prepared["capacity"]),
        "scaler_x": prepared["scaler_x"],
        "scaler_y": prepared["scaler_y"],
        "power_scale_ratio": float(prepared["power_scale_ratio"]),
        "power_scale_offset": float(prepared["power_scale_offset"]),
        "regime_feature_config": prepared["regime_feature_config"],
        "train_file": os.path.abspath(train_file),
        "train_file_sha256": _sha256(train_file),
        "train_samples": int(train_samples),
        "validation_samples": int(total_samples - train_samples),
        "validation_split": float(validation_split),
        "batch_size": int(batch_size),
        "objective_epochs": int(epochs),
        "teacher_warmup_epochs": int(teacher_warmup_epochs if trainer.teacher_enabled else 0),
        "learning_rate": float(learning_rate),
        "early_stopping_monitor": "val_candidate_rmse",
        "stage_a_selection_target": "corrected_candidate; frozen_g0_fused_is_diagnostic_only",
        "test_data_read_during_training": False,
        "a0_equal_budget_finetune_control_present": False,
        "comparison_limitation": (
            "A0 is a read-only parent; A1-A5 include additional candidate fine-tuning. "
            "Absolute gains versus A0 may contain a continuation-training component."
        ),
        "source_alias": SOURCE_ALIAS,
        "source_variant": SOURCE_VARIANT,
        "source_model_path": os.path.abspath(source_model_path),
        "source_model_sha256": source_model_sha256,
        "source_artifact_path": os.path.abspath(source_artifact_path),
        "source_artifact_sha256": _sha256(source_artifact_path),
        "source_initial_output_sha256": source_initial_hashes,
        "initial_x0_identity_max_abs_drift": initial_identity_drift,
        "trainable_core_layer_names": list(trainable_core_layers),
        "frozen_nonresidual_weights_before_sha256": frozen_sha_before,
        "frozen_nonresidual_weights_after_sha256": frozen_sha_after,
        "frozen_nonresidual_weights_exact_match": True,
        "teacher_warmup_core_before_sha256": all_core_sha_before_warmup,
        "teacher_warmup_core_after_sha256": all_core_sha_after_warmup,
        "teacher_warmup_core_exact_match": True,
        "persistence_gate_final_max_abs_drift": frozen_output_drift,
        "trainer_checkpoint_reload_max_abs_drift": reload_drift,
        "exported_model_reload_max_abs_drift": exported_drift,
        "inference_input_names": [tensor.name.split(":")[0] for tensor in model.inputs],
        "future_teacher_in_inference_graph": False,
        "future_target_training_only": True,
        "teacher_removed_at_inference": True,
        "training_wrapper_serialized": False,
        "training_wrapper_storage": "weights_checkpoint_only_rebuilt_from_artifact",
        "teacher_future_truth_scope": "train_and_validation_labels_only; never model input",
        "teacher_target_stop_gradient_for_alignment": True,
        "teacher_reconstruction_updates_teacher": bool(trainer.teacher_enabled),
        "teacher_enabled": bool(trainer.teacher_enabled),
        "teacher_patch_count": HISTORY_PATCH_COUNT,
        "teacher_patch_length": FUTURE_PATCH_LENGTH,
        "teacher_alignment_dimension": ALIGNMENT_DIM,
        "student_design": "single_final_layer_residual_patch_projection_plus_hidden_token",
        "alignment_design": (
            "direction_sensitive_local_cosine_and_patch_gram_global_with_"
            "history_only_phd_soft_regime_weights"
        ),
        "alignment_regime_conditioning": {
            "history_only": True,
            "responsibility_source": "fixed_train_only_phd_context_centroids",
            "temperature": float(trainer.regime_temperature),
            "regime_order": ["stable", "dynamic", "ramp"],
            "local_multipliers": list(LOCAL_REGIME_MULTIPLIERS),
            "global_multipliers": list(GLOBAL_REGIME_MULTIPLIERS),
            "weights_stop_gradient": True,
            "batch_normalized_mean_one": True,
        },
        "a4_balance_design": (
            "history_only_phd_sample_weighting_then_detached_loss_magnitude_balance"
        ),
        "qdf_enabled": bool(trainer.qdf_enabled),
        "qdf_diagnostics": qdf_diagnostics,
        "train_only_statistics": {
            "residual_q90": statistics["residual_q90"].tolist(),
            "regime_centroids": statistics["regime_centroids"].tolist(),
            "regime_context_scale": statistics["regime_context_scale"].tolist(),
            "regime_thresholds": statistics["regime_thresholds"],
            "regime_counts": statistics["regime_counts"],
            "sample_count": statistics["train_stat_sample_count"],
            "residual_q90_sha256": statistics["residual_q90_sha256"],
            "regime_stats_sha256": statistics["regime_stats_sha256"],
            "reused_in_same_process": statistics_reused,
        },
        "loss_weights": {
            "frozen_g0_fused_diagnostic": FUSED_DIAGNOSTIC_WEIGHT,
            "candidate": CANDIDATE_PRIMARY_WEIGHT,
            "teacher_reconstruction": TEACHER_RECONSTRUCTION_WEIGHT,
            "local_alignment": LOCAL_ALIGNMENT_WEIGHT,
            "global_alignment": GLOBAL_ALIGNMENT_WEIGHT,
            "qdf": QDF_WEIGHT if trainer.qdf_enabled else 0.0,
            "ramp": RAMP_WEIGHT if trainer.qdf_enabled else 0.0,
            "qdf_identity": QDF_IDENTITY_WEIGHT if trainer.qdf_enabled else 0.0,
        },
        "inference_parameter_count": inference_parameter_count,
        "total_params": inference_parameter_count,
        "inference_trainable_parameter_count_during_stagea": inference_trainable_parameter_count,
        "trainable_parameter_count": inference_trainable_parameter_count,
        "training_wrapper_parameter_count": training_wrapper_parameter_count,
        "training_wrapper_trainable_parameter_count": training_wrapper_trainable_parameter_count,
        "training_only_parameter_count": training_only_parameter_count,
        "training_only_trainable_parameter_count": training_only_trainable_parameter_count,
        "ideal_30k_reference_exceeded": bool(
            inference_parameter_count > IDEAL_INFERENCE_PARAMETER_REFERENCE
        ),
        "parameter_limit_enforced": False,
        "validation_metrics": validation_metrics,
        "training_elapsed_seconds": elapsed,
        "model_path": os.path.abspath(paths["model"]),
        "model_sha256": _sha256(paths["model"]),
        "best_weights_path": os.path.abspath(paths["weights"]),
        "best_weights_sha256": _sha256(paths["weights"]),
        "history_path": os.path.abspath(paths["history"]),
        "history_sha256": _sha256(paths["history"]),
        "history_figure_path": os.path.abspath(paths["history_figure"]),
        "history_figure_sha256": _sha256(paths["history_figure"]),
        "validation_diagnostics_path": os.path.abspath(paths["validation"]),
        "validation_diagnostics_sha256": _sha256(paths["validation"]),
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "implementation_note": "clean-room Keras wind residual adaptation",
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(keras, "__version__", None),
        "numpy_version": np.__version__,
        "created_at": _utc_now(),
        "training_code_path": os.path.abspath(__file__),
        "training_code_sha256": _sha256(__file__),
    }
    artifact["artifact_path"] = os.path.abspath(paths["artifact"])
    _atomic_joblib_dump(artifact, paths["artifact"])

    row = {
        "model_family": MODEL_FAMILY,
        "variant_id": variant_id,
        "variant_label": VARIANT_SPECS[variant_id]["label"],
        "farm_id": farm_id,
        "reference_only": False,
        "requires_training": True,
        "random_seed": RANDOM_SEED,
        "train_samples": int(train_samples),
        "validation_samples": int(total_samples - train_samples),
        "best_objective_epoch": int(
            np.nanargmin(np.asarray(objective_history.history["val_candidate_rmse"])) + 1
        ),
        "val_forecast_nrmse": validation_metrics["forecast"]["nrmse"],
        "val_forecast_nmae": validation_metrics["forecast"]["nmae"],
        "val_candidate_nrmse": validation_metrics["candidate"]["nrmse"],
        "val_candidate_nmae": validation_metrics["candidate"]["nmae"],
        "teacher_enabled": bool(trainer.teacher_enabled),
        "qdf_enabled": bool(trainer.qdf_enabled),
        "inference_parameter_count": inference_parameter_count,
        "training_wrapper_parameter_count": training_wrapper_parameter_count,
        "training_only_parameter_count": training_only_parameter_count,
        "inference_trainable_parameter_count_during_stagea": inference_trainable_parameter_count,
        "training_wrapper_trainable_parameter_count": training_wrapper_trainable_parameter_count,
        "training_only_trainable_parameter_count": training_only_trainable_parameter_count,
        "parameter_limit_enforced": False,
        "source_model_path": os.path.abspath(source_model_path),
        "source_model_sha256": source_model_sha256,
        "model_path": os.path.abspath(paths["model"]),
        "model_sha256": _sha256(paths["model"]),
        "artifact_path": os.path.abspath(paths["artifact"]),
        "artifact_sha256": _sha256(paths["artifact"]),
        "best_weights_path": os.path.abspath(paths["weights"]),
        "best_weights_sha256": _sha256(paths["weights"]),
        "history_path": os.path.abspath(paths["history"]),
        "history_figure_path": os.path.abspath(paths["history_figure"]),
        "training_elapsed_seconds": elapsed,
        "frozen_identity_pass": True,
        "inference_reload_pass": True,
        "qdf_degeneration_guard_pass": bool(qdf_diagnostics["degeneration_guard_pass"]),
        "result_source": "new_stage_a_training",
    }
    return row


def _reference_summary_row(reference):
    return {
        "model_family": MODEL_FAMILY,
        "variant_id": "a0",
        "variant_label": VARIANT_SPECS["a0"]["label"],
        "farm_id": str(reference["farm_id"]),
        "reference_only": True,
        "requires_training": False,
        "random_seed": RANDOM_SEED,
        "train_samples": np.nan,
        "validation_samples": np.nan,
        "best_objective_epoch": np.nan,
        "val_forecast_nrmse": np.nan,
        "val_forecast_nmae": np.nan,
        "val_candidate_nrmse": np.nan,
        "val_candidate_nmae": np.nan,
        "teacher_enabled": False,
        "qdf_enabled": False,
        "inference_parameter_count": reference["inference_parameter_count"],
        "training_wrapper_parameter_count": 0,
        "training_only_parameter_count": 0,
        "inference_trainable_parameter_count_during_stagea": 0,
        "training_wrapper_trainable_parameter_count": 0,
        "training_only_trainable_parameter_count": 0,
        "parameter_limit_enforced": False,
        "source_model_path": reference["source_model_path"],
        "source_model_sha256": reference["source_model_sha256"],
        "model_path": reference["source_model_path"],
        "model_sha256": reference["source_model_sha256"],
        "artifact_path": reference["source_artifact_path"],
        "artifact_sha256": reference["source_artifact_sha256"],
        "best_weights_path": None,
        "best_weights_sha256": None,
        "history_path": None,
        "history_figure_path": None,
        "training_elapsed_seconds": 0.0,
        "frozen_identity_pass": True,
        "inference_reload_pass": True,
        "qdf_degeneration_guard_pass": True,
        "reference_path": reference["reference_path"],
        "reference_sha256": reference["reference_sha256"],
        "result_source": "read_only_existing_x0_f7_reference",
    }


def _complexity_frame(summary):
    columns = [
        "variant_id",
        "farm_id",
        "reference_only",
        "inference_parameter_count",
        "inference_trainable_parameter_count_during_stagea",
        "training_wrapper_parameter_count",
        "training_wrapper_trainable_parameter_count",
        "training_only_parameter_count",
        "training_only_trainable_parameter_count",
        "training_elapsed_seconds",
        "parameter_limit_enforced",
    ]
    return summary.loc[:, columns].copy()


def _write_manifest(result_root, variants, farms, args, scope):
    rows = []
    for variant_id in variants:
        spec = VARIANT_SPECS[variant_id]
        rows.append(
            {
                "model_family": MODEL_FAMILY,
                "architecture_version": ARCHITECTURE_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "variant_id": variant_id,
                "variant_label": spec["label"],
                "directory_name": spec["directory_name"],
                "requires_training": spec["requires_training"],
                "reference_only": not spec["requires_training"],
                "local_alignment": spec["local_alignment"],
                "global_alignment": spec["global_alignment"],
                "alignment_history_only_phd_conditioned": bool(
                    spec["local_alignment"] or spec["global_alignment"]
                ),
                "alignment_regime_order": "stable,dynamic,ramp",
                "local_regime_multipliers": ",".join(
                    map(str, LOCAL_REGIME_MULTIPLIERS)
                ),
                "global_regime_multipliers": ",".join(
                    map(str, GLOBAL_REGIME_MULTIPLIERS)
                ),
                "qdf": spec["qdf"],
                "source_alias": SOURCE_ALIAS,
                "source_variant": SOURCE_VARIANT,
                "random_seed": RANDOM_SEED,
                "batch_size": int(args.batch_size),
                "objective_epochs": int(args.epochs),
                "teacher_warmup_epochs": int(args.warmup_epochs),
                "validation_split": float(args.validation_split),
                "learning_rate": float(args.learning_rate),
                "patience": int(args.patience),
                "farm_count": len(farms),
                "farms": ",".join(farms),
                "run_scope": scope,
                "result_root": os.path.abspath(result_root),
                "test_data_read_during_training": False,
                "a0_equal_budget_finetune_control_present": False,
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_commit": UPSTREAM_COMMIT,
            }
        )
    path = os.path.join(result_root, EXPERIMENT_MANIFEST_NAME)
    _atomic_to_csv(pd.DataFrame(rows), path)
    return path


def _validate_training_bundle(summary, variants, farms, formal):
    expected_pairs = {(variant, farm) for variant in variants for farm in farms}
    actual_pairs = {
        (str(row.variant_id), str(row.farm_id))
        for row in summary.itertuples(index=False)
    }
    if actual_pairs != expected_pairs or len(summary) != len(expected_pairs):
        raise ValueError(
            f"Stage-A训练矩阵不完整: missing={sorted(expected_pairs-actual_pairs)}, "
            f"extra={sorted(actual_pairs-expected_pairs)}"
        )
    if summary.duplicated(["variant_id", "farm_id"]).any():
        raise ValueError("Stage-A summary存在重复variant/farm")
    trained = summary[~summary["reference_only"].astype(bool)]
    required_true = (
        "frozen_identity_pass",
        "inference_reload_pass",
        "qdf_degeneration_guard_pass",
    )
    for column in required_true:
        if not trained[column].astype(bool).all():
            raise ValueError(f"Stage-A完整性列未全部通过: {column}")
    for row in trained.itertuples(index=False):
        for path_key, hash_key in (
            ("model_path", "model_sha256"),
            ("artifact_path", "artifact_sha256"),
            ("best_weights_path", "best_weights_sha256"),
        ):
            path = getattr(row, path_key)
            expected_hash = getattr(row, hash_key)
            if not os.path.isfile(path) or _sha256(path) != expected_hash:
                raise ValueError(
                    f"Stage-A产物缺失或hash漂移: {row.variant_id}/{row.farm_id}/{path_key}"
                )
        if not os.path.isfile(row.history_path) or not os.path.isfile(row.history_figure_path):
            raise FileNotFoundError(
                f"Stage-A history产物缺失: {row.variant_id}/{row.farm_id}"
            )
    source_hashes = summary.groupby("farm_id")["source_model_sha256"].nunique()
    if (source_hashes != 1).any():
        raise ValueError("A0--A5同一场站没有使用同一个X0/F7父快照")
    if formal:
        if tuple(sorted(variants)) != tuple(sorted(ALL_VARIANTS)):
            raise ValueError("formal bundle没有覆盖A0--A5")
        if tuple(sorted(farms)) != expected_farm_ids():
            raise ValueError("formal bundle没有覆盖正式5场站")
        if len(summary) != len(ALL_VARIANTS) * EXPECTED_FARM_COUNT:
            raise ValueError("formal bundle不是6×5矩阵")


def _file_record(path):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"marker待锁定文件不存在: {path}")
    return {
        "path": os.path.abspath(path),
        "sha256": _sha256(path),
        "size_bytes": int(os.path.getsize(path)),
    }


def _bundle_files(summary, summary_path, manifest_path, complexity_path):
    files = {
        "training_summary": _file_record(summary_path),
        "experiment_manifest": _file_record(manifest_path),
        "complexity": _file_record(complexity_path),
        "training_code": _file_record(__file__),
    }
    for row in summary.itertuples(index=False):
        prefix = f"{row.variant_id}.{row.farm_id}"
        files[f"{prefix}.model_path"] = _file_record(row.model_path)
        files[f"{prefix}.artifact_path"] = _file_record(row.artifact_path)
        if not bool(row.reference_only):
            files[f"{prefix}.weights_path"] = _file_record(row.best_weights_path)
            files[f"{prefix}.history_path"] = _file_record(row.history_path)
            files[f"{prefix}.history_figure_path"] = _file_record(
                row.history_figure_path
            )
            artifact = joblib.load(row.artifact_path)
            validation_path = artifact.get("validation_diagnostics_path")
            files[f"{prefix}.validation_path"] = _file_record(validation_path)
        else:
            files[f"{prefix}.reference_path"] = _file_record(row.reference_path)
    return files


def _parse_csv_argument(value, valid, label):
    if value is None:
        return None
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    if any(item in {"all", "*"} for item in selected):
        return list(valid)
    invalid = sorted(set(selected) - set(valid))
    if invalid:
        raise ValueError(f"未知{label}: {invalid}; 可选={list(valid)}")
    return list(dict.fromkeys(selected))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Part-3 Round-1 Stage-A TimeAlign/QDF training"
    )
    parser.add_argument(
        "--variants",
        help="逗号分隔a0--a5；默认formal全矩阵",
    )
    parser.add_argument("--farms", help="逗号分隔场站ID；默认正式5场站")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_TEACHER_WARMUP_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--validation-split", type=float, default=DEFAULT_VALIDATION_SPLIT)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="单场站、A0+A5、1 objective epoch、1 teacher warmup的partial smoke",
    )
    parser.add_argument(
        "--result-root",
        help="仅用于显式partial存档；formal目录固定，不能覆盖",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="清除同目录旧running marker后重新执行（已有complete仍拒绝）",
    )
    return parser.parse_args(argv)


def _validate_cli(args):
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs/batch-size/patience必须为正")
    if args.warmup_epochs < 0:
        raise ValueError("warmup-epochs不能为负")
    if not 0.0 < args.validation_split < 1.0:
        raise ValueError("validation-split必须位于(0,1)")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate必须为正")


def main(argv=None):
    args = _parse_args(argv)
    _validate_cli(args)
    configure_reproducibility()

    formal_farms = expected_farm_ids()
    variants = _parse_csv_argument(args.variants, ALL_VARIANTS, "变体")
    farms = _parse_csv_argument(args.farms, formal_farms, "场站")
    variants = variants or list(ALL_VARIANTS)
    farms = farms or list(formal_farms)
    if args.smoke:
        if args.variants is None:
            variants = ["a0", "a5"]
        if args.farms is None:
            farms = [formal_farms[0]]
        args.epochs = 1
        args.warmup_epochs = 1

    formal = bool(
        not args.smoke
        and args.result_root is None
        and tuple(sorted(variants)) == tuple(sorted(ALL_VARIANTS))
        and tuple(sorted(farms)) == tuple(sorted(formal_farms))
        and args.epochs == DEFAULT_EPOCHS
        and args.warmup_epochs == DEFAULT_TEACHER_WARMUP_EPOCHS
        and args.batch_size == DEFAULT_BATCH_SIZE
        and np.isclose(args.validation_split, DEFAULT_VALIDATION_SPLIT)
        and np.isclose(args.learning_rate, DEFAULT_LEARNING_RATE)
        and args.patience == DEFAULT_PATIENCE
    )
    scope = "formal" if formal else "partial"
    if formal:
        result_root = RESULT_ROOT
    elif args.result_root:
        result_root = os.path.abspath(args.result_root)
        if os.path.realpath(result_root) == os.path.realpath(RESULT_ROOT):
            raise ValueError("非formal参数不能直接写入正式RESULT_ROOT")
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_root = os.path.join(RESULT_ROOT, "partial_runs", f"{stamp}_{'smoke' if args.smoke else 'partial'}")
    os.makedirs(result_root, exist_ok=True)

    complete_path = os.path.join(result_root, TRAINING_MARKER_NAME)
    running_path = os.path.join(result_root, RUNNING_MARKER_NAME)
    if os.path.exists(complete_path):
        raise FileExistsError(f"Stage-A训练bundle已经complete，拒绝覆盖: {complete_path}")
    if os.path.exists(running_path):
        if not args.resume:
            raise FileExistsError(
                f"存在旧running marker；确认无进程后用--resume: {running_path}"
            )
        os.remove(running_path)

    train_files = _discover_train_files(farms)
    file_by_farm = {regime_train.get_farm_id(path): path for path in train_files}
    if set(file_by_farm) != set(farms):
        raise ValueError("请求场站与训练文件集合不一致")
    manifest_path = _write_manifest(result_root, variants, farms, args, scope)
    running = {
        "status": "running",
        "run_scope": scope,
        "model_family": MODEL_FAMILY,
        "architecture_version": ARCHITECTURE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "variants": variants,
        "farms": farms,
        "result_root": os.path.abspath(result_root),
        "training_reads_test_data": False,
        "started_at": _utc_now(),
    }
    _atomic_write_json(running, running_path)

    rows = []
    try:
        for farm_id in farms:
            train_file = file_by_farm[farm_id]
            # A0 reference is materialized only as a path/hash record.
            if "a0" in variants:
                prepared, _, _, source_artifact_path, source_model_path = _prepare_source(
                    train_file
                )
                reference = _write_reference(
                    "a0",
                    prepared,
                    source_artifact_path,
                    source_model_path,
                    result_root,
                )
                rows.append(_reference_summary_row(reference))
                _atomic_to_csv(
                    pd.DataFrame(rows),
                    os.path.join(result_root, "stage_a_training_progress.csv"),
                )
            for variant_id in variants:
                if variant_id == "a0":
                    continue
                row = _train_variant_for_farm(
                    variant_id=variant_id,
                    train_file=train_file,
                    result_root=result_root,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    validation_split=args.validation_split,
                    learning_rate=args.learning_rate,
                    patience=args.patience,
                    teacher_warmup_epochs=args.warmup_epochs,
                )
                rows.append(row)
                _atomic_to_csv(
                    pd.DataFrame(rows),
                    os.path.join(result_root, "stage_a_training_progress.csv"),
                )
        summary = pd.DataFrame(rows)
        _validate_training_bundle(summary, variants, farms, formal=formal)
        summary_path = _atomic_to_csv(
            summary, os.path.join(result_root, TRAINING_SUMMARY_NAME)
        )
        complexity_path = _atomic_to_csv(
            _complexity_frame(summary),
            os.path.join(result_root, COMPLEXITY_REPORT_NAME),
        )
        files = _bundle_files(
            summary,
            summary_path=summary_path,
            manifest_path=manifest_path,
            complexity_path=complexity_path,
        )
        marker = {
            "status": "complete",
            "run_scope": scope,
            "formal_complete": formal,
            "model_family": MODEL_FAMILY,
            "architecture_version": ARCHITECTURE_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "random_seed": RANDOM_SEED,
            "variants": variants,
            "farms": farms,
            "expected_farm_ids": list(formal_farms if formal else farms),
            "selection_variants": variants,
            "new_training_variants": [
                variant for variant in variants if variant in TRAINABLE_VARIANTS
            ],
            "matrix_rows": int(len(summary)),
            "expected_matrix_rows": int(len(variants) * len(farms)),
            "a0_read_only_reference": True,
            "a0_equal_budget_finetune_control_present": False,
            "a0_comparison_continuation_training_caveat_recorded": True,
            "same_f7_snapshot_per_farm_verified": True,
            "training_reads_test_data": False,
            "teacher_absent_from_inference_models": True,
            "parameter_limit_enforced": False,
            "summary_path": os.path.abspath(summary_path),
            "summary_sha256": _sha256(summary_path),
            "manifest_path": os.path.abspath(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "complexity_path": os.path.abspath(complexity_path),
            "complexity_sha256": _sha256(complexity_path),
            "files": files,
            "result_root": os.path.abspath(result_root),
            "completed_at": _utc_now(),
        }
        _atomic_write_json(marker, complete_path)
        if os.path.exists(running_path):
            os.remove(running_path)
        print(f"\nStage-A训练完成: {complete_path}")
        return marker
    except Exception as error:
        running.update(
            {
                "status": "failed",
                "failed_at": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_rows": len(rows),
            }
        )
        _atomic_write_json(running, running_path)
        raise


if __name__ == "__main__":
    main()
