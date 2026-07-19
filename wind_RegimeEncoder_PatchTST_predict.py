"""RegimeEncoder-PatchTST 第二阶段统一预测与诊断入口。

R2--R5 使用本阶段模型执行一次前向，并同时提取最终预测、persistence
candidate、corrected candidate 和 correction gate。R0、R1、R6 按实验设计
直接读取第一阶段 B0、B2、B6 已保存的预测结果，不重复推理或复制源文件。

本脚本复用工程既有的整体/逐 horizon 指标、单窗口图和指数加权全时段图，
同时新增固定容量阈值的工况分层、门控饱和/熵、候选 oracle、门控校准以及
R5 辅助任务诊断。所有 realized regime 标签都在模型前向完成后由真实未来功率
生成，仅用于事后评价。

注意：当前测试段已经参与第一阶段分析，且为与直接引用结果保持同一口径，
R2--R5 继续使用 Stage-1 legacy 预处理。因此输出会明确标记
``test_reuse_status=legacy_seen``，测试结果只作描述，不执行自动选型。
"""

import glob
import hashlib
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from tensorflow import keras

import wind_dl_model_predict as common_predict
from wind_FeTS_PatchTST_min_train import (
    variant_dirs as stage1_variant_dirs,
    variant_model_name as stage1_variant_model_name,
)
from wind_RegimeEncoder_PatchTST_train import (
    ARCHITECTURE_VERSION,
    CHANGE_BAND_EDGES,
    EVALUATION_PIPELINE_VERSION,
    FORECAST_LEN,
    LOW_POWER_THRESHOLD,
    MODEL_FAMILY,
    RANDOM_SEED,
    REFERENCE_SOURCE_VARIANTS,
    REGIME_LABEL_VERSION,
    RESULT_ROOT,
    STABLE_CHANGE_THRESHOLD,
    TRAINABLE_VARIANTS,
    VARIANT_SPECS,
    build_regime_encoder_patchtst_model_from_artifact,
    build_regime_targets_numpy,
    get_regime_custom_objects,
    get_requested_variants,
    variant_dirs,
    variant_model_name,
)

warnings.filterwarnings("ignore")


OUTPUT_SUBDIR = "testdata_predict_output"
TEST_REUSE_STATUS = "legacy_seen"
REGIME_GROUP_ORDER = (
    "all",
    "stable",
    "dynamic",
    "ramp_up",
    "ramp_down",
    "low_power",
    "change_00_02",
    "change_02_05",
    "change_05_10",
    "change_10_20",
    "change_ge_20",
)


def configure_prediction_reproducibility():
    os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_SEED))
    keras.utils.set_random_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _sha256(path, chunk_size=1024 * 1024):
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


def _resolve_existing_path(path):
    if not path:
        return None
    candidates = [os.fspath(path)]
    if not os.path.isabs(path):
        candidates.append(os.path.join(os.path.dirname(__file__), path))
    return next((value for value in candidates if os.path.exists(value)), None)


def prediction_output_dirs(variant_id):
    root = os.path.join(variant_dirs(variant_id, create=True)["root"], OUTPUT_SUBDIR)
    dirs = {
        "root": root,
        "predictions": os.path.join(root, "predictions"),
        "figures": os.path.join(root, "figures"),
        "single_windows": os.path.join(root, "single_window_comparisons"),
        "weighted_curves": os.path.join(root, "weighted_curves"),
        "router_diagnostics": os.path.join(root, "router_diagnostics"),
        "regime_assignments": os.path.join(root, "regime_assignments"),
        "regime_metrics": os.path.join(root, "regime_metrics"),
        "gate_diagnostics": os.path.join(root, "gate_diagnostics"),
        "candidate_metrics": os.path.join(root, "candidate_metrics"),
        "auxiliary_diagnostics": os.path.join(root, "auxiliary_diagnostics"),
        "matplotlib_cache": os.path.join(root, "matplotlib_cache"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def comparison_output_dir():
    path = os.path.join(RESULT_ROOT, OUTPUT_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def discover_requested_test_files():
    files = common_predict.discover_test_files()
    requested = os.getenv("WIND_REGIME_FARMS")
    if not requested:
        return files
    farm_ids = {item.strip() for item in requested.split(",") if item.strip()}
    return [path for path in files if common_predict.get_farm_id(path) in farm_ids]


def _artifact_path(variant_id, farm_id):
    model_name = variant_model_name(variant_id)
    return os.path.join(
        variant_dirs(variant_id, create=False)["preprocess"],
        f"{model_name}_farm_{farm_id}_preprocess.pkl",
    )


def load_variant_artifact(variant_id, farm_id):
    if variant_id not in TRAINABLE_VARIANTS:
        raise ValueError(f"引用变体 {variant_id} 没有第二阶段模型 artifact")
    path = _artifact_path(variant_id, farm_id)
    if not os.path.exists(path):
        wildcard = sorted(
            glob.glob(
                os.path.join(
                    variant_dirs(variant_id, create=False)["preprocess"],
                    f"*farm_{farm_id}_preprocess.pkl",
                )
            )
        )
        path = wildcard[0] if len(wildcard) == 1 else path
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少 {variant_id}/{farm_id} artifact: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise TypeError(f"artifact 必须是 dict: {path}")
    if artifact.get("variant_id") != variant_id:
        raise ValueError(f"artifact 变体不匹配: {path}")
    if artifact.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(f"artifact 架构版本不匹配: {path}")
    if int(artifact.get("random_seed", -1)) != RANDOM_SEED:
        raise ValueError(f"artifact seed 必须为 {RANDOM_SEED}: {path}")
    required = (
        "input_cols",
        "target_index",
        "scaler_x",
        "scaler_y",
        "history_len",
        "forecast_len",
        "diagnostic_layers",
        "regime_label_config",
    )
    missing = [name for name in required if name not in artifact]
    if missing:
        raise KeyError(f"artifact 缺少字段 {missing}: {path}")
    if list(artifact["regime_label_config"].get("change_band_edges", [])) != list(
        CHANGE_BAND_EDGES
    ):
        raise ValueError(f"artifact 工况阈值与当前预测脚本不一致: {path}")
    artifact = dict(artifact)
    artifact["artifact_path"] = os.path.abspath(path)
    return artifact


def get_prediction_custom_objects():
    custom_objects = dict(common_predict.get_custom_objects())
    custom_objects.update(get_regime_custom_objects())
    return custom_objects


def load_variant_model(variant_id, farm_id, artifact):
    model_path = _resolve_existing_path(artifact.get("model_path"))
    if model_path:
        model = keras.models.load_model(
            model_path,
            custom_objects=get_prediction_custom_objects(),
            compile=False,
        )
        loaded_path = os.path.abspath(model_path)
    else:
        weights_path = _resolve_existing_path(artifact.get("best_weights_path"))
        if not weights_path:
            raise FileNotFoundError(
                f"缺少 {variant_id}/{farm_id} 完整模型和最佳权重"
            )
        model = build_regime_encoder_patchtst_model_from_artifact(artifact)
        model.load_weights(weights_path)
        loaded_path = os.path.abspath(weights_path)
    count = int(model.count_params())
    if int(artifact.get("total_params", count)) != count:
        raise ValueError(
            f"artifact 参数量 {artifact.get('total_params')} 与模型 {count} 不一致"
        )
    return model, loaded_path


def _prediction_sample_count(features, history_len, forecast_len):
    count = len(features) - history_len - forecast_len + 1
    if count <= 0:
        raise ValueError("测试集长度不足，无法构造完整历史/预测窗口")
    return int(count)


def _diagnostic_forward(model, pred_ds, artifact):
    names = artifact["diagnostic_layers"]
    required_keys = (
        "forecast",
        "gate",
        "persistence_candidate",
        "corrected_candidate",
    )
    missing = [key for key in required_keys if not names.get(key)]
    if missing:
        raise KeyError(f"artifact diagnostic_layers 缺少 {missing}")
    ordered_keys = list(required_keys)
    for key in ("regime_class", "low_power", "change_magnitude"):
        if names.get(key):
            ordered_keys.append(key)
    outputs = []
    for key in ordered_keys:
        try:
            outputs.append(model.get_layer(names[key]).output)
        except ValueError as exc:
            raise ValueError(f"模型缺少诊断层 {key}={names[key]}") from exc
    diagnostic_model = keras.Model(
        model.inputs,
        outputs,
        name="WindRegimeEncoderPatchTSTPredictDiagnostics",
    )
    values = diagnostic_model.predict(
        pred_ds,
        verbose=common_predict.PREDICT_VERBOSE,
    )
    if not isinstance(values, (list, tuple)):
        values = [values]
    return {key: np.asarray(value) for key, value in zip(ordered_keys, values)}


def _validate_diagnostics(outputs, n_samples, forecast_len):
    expected_shape = (n_samples, forecast_len)
    for key in ("forecast", "gate", "persistence_candidate", "corrected_candidate"):
        value = np.asarray(outputs[key])
        if value.shape == (n_samples, forecast_len, 1):
            value = value[..., 0]
            outputs[key] = value
        if value.shape != expected_shape:
            raise ValueError(f"{key} 输出形状 {value.shape} != {expected_shape}")
        if not np.isfinite(value).all():
            raise FloatingPointError(f"{key} 输出包含非有限值")
    gate = outputs["gate"]
    if np.min(gate) < -1e-6 or np.max(gate) > 1.0 + 1e-6:
        raise ValueError("correction_gate 不在 [0, 1]")
    reconstructed = outputs["persistence_candidate"] + gate * (
        outputs["corrected_candidate"] - outputs["persistence_candidate"]
    )
    error = float(np.max(np.abs(reconstructed - outputs["forecast"])))
    if error > 1e-5:
        raise ValueError(f"两候选融合重构失败，最大误差={error}")
    outputs["fusion_reconstruction_max_abs_error"] = error
    if "regime_class" in outputs:
        probability = np.asarray(outputs["regime_class"], dtype=float)
        if probability.shape != (n_samples, 3):
            raise ValueError(f"regime_class 形状异常: {probability.shape}")
        if not np.isfinite(probability).all() or not np.allclose(
            probability.sum(axis=1),
            1.0,
            atol=1e-5,
        ):
            raise ValueError("regime_class 必须是有限的三类概率")
        for key in ("low_power", "change_magnitude"):
            value = np.asarray(outputs[key], dtype=float)
            if value.shape not in {(n_samples,), (n_samples, 1)}:
                raise ValueError(f"{key} 形状异常: {value.shape}")
            if (
                not np.isfinite(value).all()
                or np.min(value) < -1e-6
                or np.max(value) > 1.0 + 1e-6
            ):
                raise ValueError(f"{key} 必须是 [0,1] 内有限概率/幅度")
    return outputs


def _inverse_candidate(artifact, values, capacity):
    result = common_predict.inverse_power(artifact["scaler_y"], values).reshape(
        values.shape
    )
    return np.clip(result, 0, capacity if capacity is not None else None)


def _regime_masks(regimes):
    valid = np.asarray(regimes.get("valid_future", np.ones(len(regimes["regime_name"]))))
    valid = valid.astype(bool)
    magnitude = np.asarray(regimes["change_magnitude"], dtype=float)
    bands = np.digitize(magnitude, CHANGE_BAND_EDGES, right=True)
    masks = {
        "all": valid,
        "stable": valid & (regimes["regime_name"] == "stable"),
        "dynamic": valid & (regimes["regime_name"] != "stable"),
        "ramp_up": valid & (regimes["regime_name"] == "ramp_up"),
        "ramp_down": valid & (regimes["regime_name"] == "ramp_down"),
        "low_power": valid & np.asarray(regimes["low_power"], dtype=bool),
    }
    names = (
        "change_00_02",
        "change_02_05",
        "change_05_10",
        "change_10_20",
        "change_ge_20",
    )
    masks.update({name: valid & (bands == index) for index, name in enumerate(names)})
    return masks, bands


def build_regime_metric_rows(
    variant_id,
    farm_id,
    y_true,
    candidate_predictions,
    regimes,
    capacity,
):
    rows = []
    masks, _ = _regime_masks(regimes)
    for regime_group in REGIME_GROUP_ORDER:
        mask = masks[regime_group]
        for candidate_name, y_pred in candidate_predictions.items():
            all_metrics = common_predict.calculate_metrics(
                y_true[mask],
                y_pred[mask],
                capacity,
            )
            rows.append(
                {
                    "model_family": MODEL_FAMILY,
                    "model_variant": variant_id,
                    "farm_id": str(farm_id),
                    "regime_group": regime_group,
                    "candidate": candidate_name,
                    "horizon_step": "all",
                    "horizon_minutes": "all",
                    "sample_count": int(mask.sum()),
                    "label_version": REGIME_LABEL_VERSION,
                    "threshold_source": "predeclared_capacity_fraction",
                    "evaluation_only": True,
                    **all_metrics,
                }
            )
            for horizon in range(y_true.shape[1]):
                metrics = common_predict.calculate_metrics(
                    y_true[mask, horizon],
                    y_pred[mask, horizon],
                    capacity,
                )
                rows.append(
                    {
                        "model_family": MODEL_FAMILY,
                        "model_variant": variant_id,
                        "farm_id": str(farm_id),
                        "regime_group": regime_group,
                        "candidate": candidate_name,
                        "horizon_step": horizon + 1,
                        "horizon_minutes": (horizon + 1) * 15,
                        "sample_count": int(mask.sum()),
                        "label_version": REGIME_LABEL_VERSION,
                        "threshold_source": "predeclared_capacity_fraction",
                        "evaluation_only": True,
                        **metrics,
                    }
                )
    return rows


def _candidate_metric_rows(
    variant_id,
    model_name,
    farm_id,
    y_true,
    candidate_predictions,
    capacity,
):
    frames = []
    for candidate_name, y_pred in candidate_predictions.items():
        frame = common_predict.metrics_by_horizon(
            model_name,
            farm_id,
            y_true,
            y_pred,
            capacity,
            y_true.shape[1],
        )
        frame["model_family"] = MODEL_FAMILY
        frame["model_variant"] = variant_id
        frame["candidate"] = candidate_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _safe_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def _gate_rows(
    variant_id,
    farm_id,
    gate,
    y_true,
    persistence,
    corrected,
    fused,
    regimes,
):
    rows = []
    masks, _ = _regime_masks(regimes)
    corrected_better = np.square(corrected - y_true) < np.square(
        persistence - y_true
    )
    hard_choice = gate >= 0.5
    candidate_gap = corrected - persistence
    contribution = gate * candidate_gap
    for regime_group in REGIME_GROUP_ORDER:
        mask = masks[regime_group]
        for horizon in range(gate.shape[1]):
            values = gate[mask, horizon]
            if len(values):
                binary_entropy = -(
                    values * np.log(np.clip(values, 1e-8, 1.0))
                    + (1.0 - values)
                    * np.log(np.clip(1.0 - values, 1e-8, 1.0))
                ) / np.log(2.0)
                p_error = np.square(
                    persistence[mask, horizon] - y_true[mask, horizon]
                )
                c_error = np.square(corrected[mask, horizon] - y_true[mask, horizon])
                f_error = np.square(fused[mask, horizon] - y_true[mask, horizon])
                oracle_error = np.minimum(p_error, c_error)
                persistence_mse = _safe_mean(p_error)
                oracle_mse = _safe_mean(oracle_error)
                fused_mse = _safe_mean(f_error)
                possible_gain = persistence_mse - oracle_mse
                captured_gain = (
                    (persistence_mse - fused_mse) / possible_gain
                    if np.isfinite(possible_gain) and possible_gain > 1e-12
                    else np.nan
                )
                row = {
                    "gate_mean": float(values.mean()),
                    "gate_std": float(values.std()),
                    "gate_p10": float(np.quantile(values, 0.10)),
                    "gate_p50": float(np.quantile(values, 0.50)),
                    "gate_p90": float(np.quantile(values, 0.90)),
                    "gate_low_saturation_rate": float((values < 0.05).mean()),
                    "gate_high_saturation_rate": float((values > 0.95).mean()),
                    "gate_binary_entropy": float(binary_entropy.mean()),
                    "corrected_better_rate": float(
                        corrected_better[mask, horizon].mean()
                    ),
                    "gate_hard_choice_accuracy": float(
                        (
                            hard_choice[mask, horizon]
                            == corrected_better[mask, horizon]
                        ).mean()
                    ),
                    "gate_oracle_brier": float(
                        np.mean(
                            np.square(
                                values
                                - corrected_better[mask, horizon].astype(float)
                            )
                        )
                    ),
                    "candidate_abs_gap_mean": float(
                        np.mean(np.abs(candidate_gap[mask, horizon]))
                    ),
                    "gate_contribution_abs_mean": float(
                        np.mean(np.abs(contribution[mask, horizon]))
                    ),
                    "fused_mse": fused_mse,
                    "oracle_mse": oracle_mse,
                    "oracle_regret": fused_mse - oracle_mse,
                    "captured_oracle_gain": captured_gain,
                }
            else:
                row = {
                    key: np.nan
                    for key in (
                        "gate_mean",
                        "gate_std",
                        "gate_p10",
                        "gate_p50",
                        "gate_p90",
                        "gate_low_saturation_rate",
                        "gate_high_saturation_rate",
                        "gate_binary_entropy",
                        "corrected_better_rate",
                        "gate_hard_choice_accuracy",
                        "gate_oracle_brier",
                        "candidate_abs_gap_mean",
                        "gate_contribution_abs_mean",
                        "fused_mse",
                        "oracle_mse",
                        "oracle_regret",
                        "captured_oracle_gain",
                    )
                }
            rows.append(
                {
                    "model_family": MODEL_FAMILY,
                    "model_variant": variant_id,
                    "farm_id": str(farm_id),
                    "regime_group": regime_group,
                    "horizon_step": horizon + 1,
                    "horizon_minutes": (horizon + 1) * 15,
                    "sample_count": int(mask.sum()),
                    **row,
                }
            )
    return rows


def _gate_calibration_rows(variant_id, farm_id, gate, corrected_better):
    flat_gate = np.asarray(gate, dtype=float).reshape(-1)
    flat_truth = np.asarray(corrected_better, dtype=float).reshape(-1)
    bin_ids = np.minimum((flat_gate * 10).astype(int), 9)
    rows = []
    for bin_id in range(10):
        mask = bin_ids == bin_id
        rows.append(
            {
                "model_family": MODEL_FAMILY,
                "model_variant": variant_id,
                "farm_id": str(farm_id),
                "gate_bin": bin_id,
                "gate_bin_left": bin_id / 10.0,
                "gate_bin_right": (bin_id + 1) / 10.0,
                "count": int(mask.sum()),
                "mean_gate": _safe_mean(flat_gate[mask]),
                "corrected_better_rate": _safe_mean(flat_truth[mask]),
            }
        )
    return rows


def _fixed_rank_correlation(first, second):
    first = pd.Series(np.asarray(first, dtype=float)).rank(method="average").to_numpy()
    second = pd.Series(np.asarray(second, dtype=float)).rank(method="average").to_numpy()
    if len(first) < 2 or np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


def _expected_calibration_error(probabilities, true_class, bins=10):
    confidence = np.max(probabilities, axis=1)
    prediction = np.argmax(probabilities, axis=1)
    correct = (prediction == true_class).astype(float)
    bin_ids = np.minimum((confidence * bins).astype(int), bins - 1)
    error = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(error)


def _auxiliary_metrics(outputs, regimes, variant_id, farm_id):
    if "regime_class" not in outputs:
        return None, None
    probability = np.asarray(outputs["regime_class"], dtype=float)
    low_probability = np.asarray(outputs["low_power"], dtype=float).reshape(-1)
    magnitude_prediction = np.asarray(outputs["change_magnitude"], dtype=float).reshape(-1)
    true_class = np.asarray(regimes["regime_index"], dtype=int)
    valid = np.asarray(regimes.get("valid_future", np.ones(len(true_class))), dtype=bool)
    probability = probability[valid]
    low_probability = low_probability[valid]
    magnitude_prediction = magnitude_prediction[valid]
    true_class = true_class[valid]
    true_low = np.asarray(regimes["low_power"], dtype=bool)[valid]
    true_magnitude = np.clip(
        np.asarray(regimes["change_magnitude"], dtype=float)[valid],
        0.0,
        1.0,
    )
    if len(true_class) == 0:
        return None, None
    predicted_class = np.argmax(probability, axis=1)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        true_class,
        predicted_class,
        labels=[0, 1, 2],
        zero_division=0,
    )
    one_hot = np.eye(3)[true_class]
    fields = {
        "model_family": MODEL_FAMILY,
        "model_variant": variant_id,
        "farm_id": str(farm_id),
        "valid_samples": int(valid.sum()),
        "regime_accuracy": float(np.mean(predicted_class == true_class)),
        "regime_macro_f1": float(
            f1_score(
                true_class,
                predicted_class,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "regime_log_loss": float(log_loss(true_class, probability, labels=[0, 1, 2])),
        "regime_multiclass_brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
        "regime_ece": _expected_calibration_error(probability, true_class),
        "low_power_accuracy": float(np.mean((low_probability >= 0.5) == true_low)),
        "low_power_brier": float(
            np.mean(np.square(low_probability - true_low.astype(float)))
        ),
        "change_magnitude_mae": float(
            np.mean(np.abs(magnitude_prediction - true_magnitude))
        ),
    }
    for index, name in enumerate(("stable", "ramp_up", "ramp_down")):
        fields.update(
            {
                f"{name}_precision": float(precision[index]),
                f"{name}_recall": float(recall[index]),
                f"{name}_f1": float(per_class_f1[index]),
                f"{name}_support": int(support[index]),
            }
        )
    matrix = confusion_matrix(true_class, predicted_class, labels=[0, 1, 2])
    return fields, matrix


def _save_gate_figures(gate_frame, calibration_frame, model_name, farm_id, dirs):
    heatmap_path = os.path.join(
        dirs["figures"],
        f"{model_name}_gate_by_regime_farm_{farm_id}.png",
    )
    calibration_path = os.path.join(
        dirs["figures"],
        f"{model_name}_gate_calibration_farm_{farm_id}.png",
    )
    try:
        plt = common_predict.setup_matplotlib(dirs)
        selected = gate_frame[
            gate_frame["regime_group"].isin(
                ["stable", "ramp_up", "ramp_down", "low_power"]
            )
        ]
        pivot = selected.pivot(
            index="regime_group",
            columns="horizon_step",
            values="gate_mean",
        )
        fig, ax = plt.subplots(figsize=(11, 4))
        image = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
        ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns)
        ax.set_xlabel("Horizon step")
        ax.set_title(f"{model_name} Farm {farm_id}: corrected gate by regime")
        fig.colorbar(image, ax=ax, label="Corrected candidate weight")
        fig.tight_layout()
        fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        valid_calibration = calibration_frame[calibration_frame["count"] > 0]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
        ax.plot(
            valid_calibration["mean_gate"],
            valid_calibration["corrected_better_rate"],
            marker="o",
            label="observed",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean corrected gate")
        ax.set_ylabel("Corrected candidate better rate")
        ax.set_title(f"{model_name} Farm {farm_id}: gate calibration")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(calibration_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"门控诊断图保存失败: {exc}")
        heatmap_path = None
        calibration_path = None
    return heatmap_path, calibration_path


def _save_aux_confusion(matrix, model_name, farm_id, dirs):
    if matrix is None:
        return None, None
    csv_path = os.path.join(
        dirs["auxiliary_diagnostics"],
        f"{model_name}_aux_confusion_farm_{farm_id}.csv",
    )
    frame = pd.DataFrame(
        matrix,
        index=["true_stable", "true_ramp_up", "true_ramp_down"],
        columns=["pred_stable", "pred_ramp_up", "pred_ramp_down"],
    )
    frame.to_csv(csv_path, encoding="utf-8-sig")
    figure_path = os.path.join(
        dirs["figures"],
        f"{model_name}_aux_confusion_farm_{farm_id}.png",
    )
    try:
        plt = common_predict.setup_matplotlib(dirs)
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(3), labels=["stable", "ramp_up", "ramp_down"])
        ax.set_yticks(range(3), labels=["stable", "ramp_up", "ramp_down"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{model_name} Farm {farm_id}: auxiliary regime confusion")
        for row in range(3):
            for column in range(3):
                ax.text(column, row, int(matrix[row, column]), ha="center", va="center")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"辅助任务混淆矩阵图保存失败: {exc}")
        figure_path = None
    return csv_path, figure_path


def _assignment_frame(df, farm_id, regimes, last_power, n_samples, history_len):
    _, bands = _regime_masks(regimes)
    band_names = np.asarray(
        [
            "change_00_02",
            "change_02_05",
            "change_05_10",
            "change_10_20",
            "change_ge_20",
        ]
    )
    origins = df.index[history_len - 1 : history_len - 1 + n_samples]
    valid_future = np.asarray(
        regimes.get("valid_future", np.ones(n_samples)),
        dtype=bool,
    )
    change_band = band_names[np.clip(bands, 0, 4)]
    change_band = np.where(valid_future, change_band, "unknown")
    return pd.DataFrame(
        {
            "farm_id": str(farm_id),
            "sample_id": np.arange(n_samples),
            "forecast_origin_time": origins,
            "last_history_power": last_power,
            "realized_regime": regimes["regime_name"],
            "low_power": regimes["low_power"],
            "future_max_change_capacity_fraction": regimes["change_magnitude"],
            "change_band": change_band,
            "valid_future": valid_future,
            "label_version": REGIME_LABEL_VERSION,
            "stable_threshold": STABLE_CHANGE_THRESHOLD,
            "low_power_threshold": LOW_POWER_THRESHOLD,
            "threshold_source": "predeclared_capacity_fraction",
            "evaluation_only": True,
        }
    )


def predict_one_trained_variant_farm(variant_id, test_file):
    farm_id = common_predict.get_farm_id(test_file)
    model_name = variant_model_name(variant_id)
    dirs = prediction_output_dirs(variant_id)
    print(f"\n===== 预测 {model_name} / 风电场 {farm_id} =====")
    artifact = load_variant_artifact(variant_id, farm_id)
    model, loaded_model_path = load_variant_model(variant_id, farm_id, artifact)
    df, features, actual_power, capacity = common_predict.prepare_prediction_arrays(
        test_file,
        artifact,
    )
    history_len = int(artifact["history_len"])
    forecast_len = int(artifact["forecast_len"])
    n_samples = _prediction_sample_count(features, history_len, forecast_len)
    pred_ds, dataset_samples = common_predict.make_prediction_dataset(
        features,
        history_len,
        forecast_len,
    )
    if dataset_samples != n_samples:
        raise ValueError("预测 dataset 样本数不一致")
    outputs = _validate_diagnostics(
        _diagnostic_forward(model, pred_ds, artifact),
        n_samples,
        forecast_len,
    )

    # 模型前向在这里已经完成；下面才读取未来真实功率并生成 realized regime。
    y_true = common_predict.build_truth_windows(
        actual_power,
        n_samples,
        history_len,
        forecast_len,
    )
    fused = _inverse_candidate(artifact, outputs["forecast"], capacity)
    persistence = _inverse_candidate(
        artifact,
        outputs["persistence_candidate"],
        capacity,
    )
    corrected = _inverse_candidate(
        artifact,
        outputs["corrected_candidate"],
        capacity,
    )
    gate = outputs["gate"]
    last_power = persistence[:, 0]
    regimes = build_regime_targets_numpy(y_true, last_power, capacity)
    candidate_predictions = {
        "fused": fused,
        "persistence": persistence,
        "corrected": corrected,
    }

    pred_df = common_predict.build_prediction_frame(
        model_name,
        df,
        farm_id,
        fused,
        y_true,
        history_len,
        forecast_len,
    )
    prediction_path = os.path.join(
        dirs["predictions"],
        f"{model_name}_predictions_farm_{farm_id}.csv",
    )
    pred_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    metric_df = common_predict.metrics_by_horizon(
        model_name,
        farm_id,
        y_true,
        fused,
        capacity,
        forecast_len,
    )
    parameter_count = int(model.count_params())
    trainable_parameter_count = int(
        sum(int(np.prod(variable.shape)) for variable in model.trainable_weights)
    )
    metric_df["model_family"] = MODEL_FAMILY
    metric_df["model_variant"] = variant_id
    metric_df["parameter_count"] = parameter_count
    horizon_metric_path = os.path.join(
        dirs["root"],
        f"{model_name}_metrics_by_horizon_farm_{farm_id}.csv",
    )
    metric_df.to_csv(horizon_metric_path, index=False, encoding="utf-8-sig")

    candidate_metric_df = _candidate_metric_rows(
        variant_id,
        model_name,
        farm_id,
        y_true,
        candidate_predictions,
        capacity,
    )
    candidate_metric_path = os.path.join(
        dirs["candidate_metrics"],
        f"{model_name}_candidate_metrics_farm_{farm_id}.csv",
    )
    candidate_metric_df.to_csv(
        candidate_metric_path,
        index=False,
        encoding="utf-8-sig",
    )

    assignment_df = _assignment_frame(
        df,
        farm_id,
        regimes,
        last_power,
        n_samples,
        history_len,
    )
    assignment_path = os.path.join(
        dirs["regime_assignments"],
        f"{model_name}_regime_assignments_farm_{farm_id}.csv",
    )
    assignment_df.to_csv(assignment_path, index=False, encoding="utf-8-sig")
    regime_rows = build_regime_metric_rows(
        variant_id,
        farm_id,
        y_true,
        candidate_predictions,
        regimes,
        capacity,
    )
    regime_frame = pd.DataFrame(regime_rows)
    regime_metric_path = os.path.join(
        dirs["regime_metrics"],
        f"{model_name}_regime_metrics_farm_{farm_id}.csv",
    )
    regime_frame.to_csv(regime_metric_path, index=False, encoding="utf-8-sig")

    gate_rows = _gate_rows(
        variant_id,
        farm_id,
        gate,
        y_true,
        persistence,
        corrected,
        fused,
        regimes,
    )
    gate_frame = pd.DataFrame(gate_rows)
    gate_path = os.path.join(
        dirs["gate_diagnostics"],
        f"{model_name}_gate_by_regime_horizon_farm_{farm_id}.csv",
    )
    gate_frame.to_csv(gate_path, index=False, encoding="utf-8-sig")
    corrected_better = np.square(corrected - y_true) < np.square(
        persistence - y_true
    )
    calibration_frame = pd.DataFrame(
        _gate_calibration_rows(variant_id, farm_id, gate, corrected_better)
    )
    calibration_path = os.path.join(
        dirs["gate_diagnostics"],
        f"{model_name}_gate_calibration_farm_{farm_id}.csv",
    )
    calibration_frame.to_csv(calibration_path, index=False, encoding="utf-8-sig")
    heatmap_path, calibration_figure_path = _save_gate_figures(
        gate_frame,
        calibration_frame,
        model_name,
        farm_id,
        dirs,
    )

    router_weights = np.stack([1.0 - gate, gate], axis=-1)
    _, router_fields = common_predict.save_router_diagnostics(
        router_weights,
        ["persistence", "corrected"],
        model_name,
        farm_id,
        dirs,
    )
    single_window_path, single_window_figure_path = (
        common_predict.save_single_window_plot(
            pred_df,
            model_name,
            farm_id,
            dirs,
            forecast_len,
        )
    )
    (
        weighted_curve_path,
        weighted_curve_figure_path,
        weighted_metrics,
    ) = common_predict.save_weighted_full_test_plot(
        pred_df,
        model_name,
        farm_id,
        dirs,
        capacity,
    )

    auxiliary_fields, auxiliary_matrix = _auxiliary_metrics(
        outputs,
        regimes,
        variant_id,
        farm_id,
    )
    auxiliary_summary_path = None
    if auxiliary_fields is not None:
        auxiliary_summary_path = os.path.join(
            dirs["auxiliary_diagnostics"],
            f"{model_name}_aux_metrics_farm_{farm_id}.csv",
        )
        pd.DataFrame([auxiliary_fields]).to_csv(
            auxiliary_summary_path,
            index=False,
            encoding="utf-8-sig",
        )
    auxiliary_confusion_path, auxiliary_confusion_figure_path = _save_aux_confusion(
        auxiliary_matrix,
        model_name,
        farm_id,
        dirs,
    )

    all_metrics = metric_df[metric_df["horizon_step"] == "all"].iloc[0].to_dict()
    binary_entropy = -(
        gate * np.log(np.clip(gate, 1e-8, 1.0))
        + (1.0 - gate) * np.log(np.clip(1.0 - gate, 1e-8, 1.0))
    ) / np.log(2.0)
    all_metrics.update(
        {
            "model_family": MODEL_FAMILY,
            "model_variant": variant_id,
            "variant_id": variant_id,
            "architecture_version": ARCHITECTURE_VERSION,
            "random_seed": RANDOM_SEED,
            "result_source": "stage2_model_inference",
            "reference_only": False,
            "source_variant": "b2_persistence_residual",
            "encoder_type": artifact.get("encoder_type"),
            "gate_type": artifact.get("gate_type"),
            "auxiliary_tasks": artifact.get("auxiliary_tasks"),
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "loaded_model_path": loaded_model_path,
            "artifact_path": artifact["artifact_path"],
            "artifact_sha256": _sha256(artifact["artifact_path"]),
            "prediction_path": prediction_path,
            "horizon_metric_path": horizon_metric_path,
            "candidate_metric_path": candidate_metric_path,
            "regime_assignment_path": assignment_path,
            "regime_metric_path": regime_metric_path,
            "gate_diagnostics_path": gate_path,
            "gate_calibration_path": calibration_path,
            "gate_heatmap_path": heatmap_path,
            "gate_calibration_figure_path": calibration_figure_path,
            "auxiliary_summary_path": auxiliary_summary_path,
            "auxiliary_confusion_path": auxiliary_confusion_path,
            "auxiliary_confusion_figure_path": auxiliary_confusion_figure_path,
            "single_window_path": single_window_path,
            "single_window_figure_path": single_window_figure_path,
            "weighted_curve_path": weighted_curve_path,
            "weighted_curve_figure_path": weighted_curve_figure_path,
            "gate_mean": float(gate.mean()),
            "gate_std": float(gate.std()),
            "gate_sample_variation": float(np.std(gate, axis=0).mean()),
            "gate_binary_entropy": float(binary_entropy.mean()),
            "gate_saturation_low_rate": float((gate < 0.05).mean()),
            "gate_saturation_high_rate": float((gate > 0.95).mean()),
            "gate_oracle_choice_accuracy": float(
                ((gate >= 0.5) == corrected_better).mean()
            ),
            "gate_oracle_brier": float(
                np.mean(np.square(gate - corrected_better.astype(float)))
            ),
            "gate_change_magnitude_spearman": _fixed_rank_correlation(
                gate.mean(axis=1),
                regimes["change_magnitude"],
            ),
            "fusion_reconstruction_max_abs_error": outputs[
                "fusion_reconstruction_max_abs_error"
            ],
            "evaluation_pipeline_version": EVALUATION_PIPELINE_VERSION,
            "legacy_bidirectional_weather_imputation": True,
            "test_reuse_status": TEST_REUSE_STATUS,
            "test_selection_prohibited": True,
            **router_fields,
            **{
                f"weighted_curve_{key}": value
                for key, value in weighted_metrics.items()
            },
            **(auxiliary_fields or {}),
        }
    )
    print(
        f"{model_name} / {farm_id}: NRMSE="
        f"{all_metrics['capacity_normalized_rmse']:.6f}, "
        f"gate={all_metrics['gate_mean']:.4f}, params={parameter_count:,}"
    )
    del model
    keras.backend.clear_session()
    return {
        "summary": all_metrics,
        "horizon": metric_df,
        "candidate": candidate_metric_df,
        "regime": regime_frame,
        "gate": gate_frame,
        "calibration": calibration_frame,
        "auxiliary": (
            pd.DataFrame([auxiliary_fields])
            if auxiliary_fields is not None
            else pd.DataFrame()
        ),
    }


def predict_trained_variant(variant_id, test_files):
    if variant_id not in TRAINABLE_VARIANTS:
        raise ValueError(f"{variant_id} 不是本阶段可推理模型")
    outputs = {
        "summary": [],
        "horizon": [],
        "candidate": [],
        "regime": [],
        "gate": [],
        "calibration": [],
        "auxiliary": [],
    }
    for test_file in test_files:
        result = predict_one_trained_variant_farm(variant_id, test_file)
        outputs["summary"].append(result["summary"])
        for key in outputs:
            if key == "summary":
                continue
            if not result[key].empty:
                outputs[key].append(result[key])

    combined = {
        "summary": pd.DataFrame(outputs["summary"]),
    }
    for key in outputs:
        if key == "summary":
            continue
        combined[key] = (
            pd.concat(outputs[key], ignore_index=True)
            if outputs[key]
            else pd.DataFrame()
        )
    dirs = prediction_output_dirs(variant_id)
    model_name = variant_model_name(variant_id)
    full_farm_run = (
        not os.getenv("WIND_REGIME_FARMS")
        and len(test_files) == len(common_predict.discover_test_files())
    )
    suffix = "" if full_farm_run else "_partial"
    combined["summary"].to_csv(
        os.path.join(
            dirs["root"],
            f"{model_name}_test_metrics_summary{suffix}.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )
    combined["horizon"].to_csv(
        os.path.join(
            dirs["root"],
            f"{model_name}_test_metrics_by_horizon_all{suffix}.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )
    return combined


def _stage1_output_paths(source_variant):
    model_name = stage1_variant_model_name(source_variant)
    root = os.path.join(
        stage1_variant_dirs(source_variant, create=False)["root"],
        OUTPUT_SUBDIR,
    )
    return {
        "root": root,
        "summary": os.path.join(root, f"{model_name}_test_metrics_summary.csv"),
        "horizon": os.path.join(
            root,
            f"{model_name}_test_metrics_by_horizon_all.csv",
        ),
        "predictions": os.path.join(root, "predictions"),
    }


def _stage1_artifact(source_variant, farm_id):
    model_name = stage1_variant_model_name(source_variant)
    path = os.path.join(
        stage1_variant_dirs(source_variant, create=False)["preprocess"],
        f"{model_name}_farm_{farm_id}_preprocess.pkl",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少 Stage-1 reference artifact: {path}")
    return joblib.load(path), os.path.abspath(path)


def _source_prediction_path(source_variant, farm_id):
    model_name = stage1_variant_model_name(source_variant)
    path = os.path.join(
        _stage1_output_paths(source_variant)["predictions"],
        f"{model_name}_predictions_farm_{farm_id}.csv",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少 Stage-1 reference prediction: {path}")
    return os.path.abspath(path)


def _prediction_arrays_from_frame(path, forecast_len=FORECAST_LEN):
    frame = pd.read_csv(path)
    required = {"sample_id", "horizon_step", "pred_power", "actual_power"}
    if not required.issubset(frame.columns):
        raise KeyError(f"源预测缺少列 {sorted(required - set(frame.columns))}: {path}")
    prediction = frame.pivot(
        index="sample_id",
        columns="horizon_step",
        values="pred_power",
    ).sort_index(axis=1)
    truth = frame.pivot(
        index="sample_id",
        columns="horizon_step",
        values="actual_power",
    ).sort_index(axis=1)
    if prediction.shape[1] != forecast_len or truth.shape != prediction.shape:
        raise ValueError(f"Stage-1 源预测形状异常: {path}, {prediction.shape}")
    return truth.to_numpy(dtype=float), prediction.to_numpy(dtype=float), frame


def _reference_regime_outputs(variant_id, source_variant, farm_id):
    source_prediction_path = _source_prediction_path(source_variant, farm_id)
    y_true, y_pred, source_frame = _prediction_arrays_from_frame(source_prediction_path)
    b0_path = _source_prediction_path("b0_persistence", farm_id)
    b0_true, b0_prediction, _ = _prediction_arrays_from_frame(b0_path)
    if y_true.shape != b0_true.shape or not np.allclose(
        y_true,
        b0_true,
        equal_nan=True,
        atol=1e-7,
    ):
        raise ValueError(f"Stage-1 reference truth 不一致: {source_variant}/{farm_id}")
    artifact, artifact_path = _stage1_artifact(source_variant, farm_id)
    capacity = float(artifact["capacity"])
    last_power = b0_prediction[:, 0]
    regimes = build_regime_targets_numpy(y_true, last_power, capacity)
    regime_frame = pd.DataFrame(
        build_regime_metric_rows(
            variant_id,
            farm_id,
            y_true,
            {"fused": y_pred},
            regimes,
            capacity,
        )
    )
    origin_rows = (
        source_frame[source_frame["horizon_step"] == 1]
        .sort_values("sample_id")
        .drop_duplicates("sample_id")
    )
    assignment = pd.DataFrame(
        {
            "farm_id": str(farm_id),
            "sample_id": np.arange(len(y_true)),
            "forecast_origin_time": origin_rows["forecast_origin_time"].to_numpy(),
            "last_history_power": last_power,
            "realized_regime": regimes["regime_name"],
            "low_power": regimes["low_power"],
            "future_max_change_capacity_fraction": regimes["change_magnitude"],
            "valid_future": regimes.get("valid_future", True),
            "label_version": REGIME_LABEL_VERSION,
            "evaluation_only": True,
            "source_prediction_path": source_prediction_path,
        }
    )
    return regime_frame, assignment, artifact, artifact_path, source_prediction_path


def load_reference_variant(variant_id, test_files):
    if variant_id not in REFERENCE_SOURCE_VARIANTS:
        raise ValueError(f"{variant_id} 不是冻结引用变体")
    source_variant = REFERENCE_SOURCE_VARIANTS[variant_id]
    paths = _stage1_output_paths(source_variant)
    for key in ("summary", "horizon"):
        if not os.path.exists(paths[key]):
            raise FileNotFoundError(f"缺少 Stage-1 引用 {key}: {paths[key]}")
    summary = pd.read_csv(paths["summary"])
    horizon = pd.read_csv(paths["horizon"])
    summary["farm_id"] = summary["farm_id"].astype(str)
    horizon["farm_id"] = horizon["farm_id"].astype(str)
    requested_farms = [common_predict.get_farm_id(path) for path in test_files]
    summary = summary[summary["farm_id"].isin(requested_farms)].copy()
    horizon = horizon[horizon["farm_id"].isin(requested_farms)].copy()
    if summary["farm_id"].nunique() != len(requested_farms):
        raise ValueError(f"{source_variant} 引用 summary 未覆盖全部请求场站")
    horizon_counts = horizon.groupby("farm_id").size()
    if set(horizon_counts.index) != set(requested_farms) or not (
        horizon_counts == FORECAST_LEN + 1
    ).all():
        raise ValueError(f"{source_variant} 引用 horizon 不是每场站17行")

    regime_frames = []
    assignment_frames = []
    source_metadata = {}
    for farm_id in requested_farms:
        (
            regime_frame,
            assignment,
            artifact,
            artifact_path,
            prediction_path,
        ) = _reference_regime_outputs(variant_id, source_variant, farm_id)
        regime_frames.append(regime_frame)
        assignment_frames.append(assignment)
        source_model_path = _resolve_existing_path(artifact.get("model_path"))
        source_metadata[farm_id] = {
            "source_artifact_path": artifact_path,
            "source_artifact_sha256": _sha256(artifact_path),
            "source_model_path": (
                os.path.abspath(source_model_path) if source_model_path else None
            ),
            "source_model_sha256": _sha256(source_model_path),
            "source_prediction_path": prediction_path,
        }

    original_model_names = summary["model_name"].copy()
    summary["source_model_name"] = original_model_names
    summary["source_model_family"] = "fets_patchtst_min"
    summary["source_variant"] = source_variant
    summary["source_summary_path"] = os.path.abspath(paths["summary"])
    summary["source_summary_sha256"] = _sha256(paths["summary"])
    summary["source_horizon_path"] = os.path.abspath(paths["horizon"])
    summary["source_horizon_sha256"] = _sha256(paths["horizon"])
    summary["model_name"] = variant_model_name(variant_id)
    summary["model_family"] = MODEL_FAMILY
    summary["model_variant"] = variant_id
    summary["variant_id"] = variant_id
    summary["result_source"] = "frozen_stage1_prediction_reference"
    summary["reference_only"] = True
    summary["test_reuse_status"] = TEST_REUSE_STATUS
    summary["test_selection_prohibited"] = True
    summary["evaluation_pipeline_version"] = EVALUATION_PIPELINE_VERSION
    summary["legacy_bidirectional_weather_imputation"] = True
    for column in (
        "source_artifact_path",
        "source_artifact_sha256",
        "source_model_path",
        "source_model_sha256",
        "source_prediction_path",
    ):
        summary[column] = summary["farm_id"].map(
            {farm: values[column] for farm, values in source_metadata.items()}
        )

    horizon["source_model_name"] = horizon["model_name"]
    horizon["source_model_family"] = "fets_patchtst_min"
    horizon["source_variant"] = source_variant
    horizon["model_name"] = variant_model_name(variant_id)
    horizon["model_family"] = MODEL_FAMILY
    horizon["model_variant"] = variant_id
    horizon["variant_id"] = variant_id
    horizon["result_source"] = "frozen_stage1_prediction_reference"
    return {
        "summary": summary,
        "horizon": horizon,
        "candidate": pd.DataFrame(),
        "regime": pd.concat(regime_frames, ignore_index=True),
        "gate": pd.DataFrame(),
        "calibration": pd.DataFrame(),
        "auxiliary": pd.DataFrame(),
        "assignments": pd.concat(assignment_frames, ignore_index=True),
    }


def build_descriptive_comparison(summary_df, requested_variants):
    metrics = (
        "mae",
        "mse",
        "rmse",
        "mape",
        "smape",
        "r2",
        "capacity_normalized_mae",
        "capacity_normalized_rmse",
        "weighted_curve_mae",
        "weighted_curve_rmse",
        "weighted_curve_capacity_normalized_mae",
        "weighted_curve_capacity_normalized_rmse",
    )
    rows = []
    for order, variant_id in enumerate(requested_variants):
        frame = summary_df[summary_df["model_variant"] == variant_id]
        parameters = (
            pd.to_numeric(frame["parameter_count"], errors="coerce")
            if "parameter_count" in frame
            else pd.Series(dtype=float)
        )
        row = {
            "variant_order": order,
            "model_variant": variant_id,
            "model_name": variant_model_name(variant_id),
            "farm_count": int(frame["farm_id"].astype(str).nunique()),
            "parameter_count_min": (
                int(parameters.min()) if parameters.notna().any() else np.nan
            ),
            "parameter_count_max": (
                int(parameters.max()) if parameters.notna().any() else np.nan
            ),
            "comparison_role": (
                "stage2_candidate"
                if variant_id in TRAINABLE_VARIANTS
                else "frozen_reference"
            ),
            "metric_source": "legacy_seen_test_descriptive_only",
            "eligible_for_test_based_selection": False,
        }
        for metric in metrics:
            values = (
                pd.to_numeric(frame[metric], errors="coerce")
                if metric in frame
                else pd.Series(dtype=float)
            )
            row[f"macro_mean_{metric}"] = float(values.mean())
            row[f"macro_std_{metric}"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def _save_descriptive_pareto(comparison, output_dir):
    path = os.path.join(output_dir, f"{MODEL_FAMILY}_test_descriptive_pareto.png")
    frame = comparison.dropna(
        subset=["parameter_count_max", "macro_mean_capacity_normalized_rmse"]
    )
    if frame.empty:
        return None
    try:
        dirs = {
            "matplotlib_cache": os.path.join(output_dir, "matplotlib_cache")
        }
        plt = common_predict.setup_matplotlib(dirs)
        fig, ax = plt.subplots(figsize=(10, 6))
        for _, row in frame.iterrows():
            candidate = row["comparison_role"] == "stage2_candidate"
            ax.scatter(
                row["parameter_count_max"],
                row["macro_mean_capacity_normalized_rmse"],
                marker="o" if candidate else "s",
                color="tab:blue" if candidate else "tab:gray",
            )
            ax.annotate(
                row["model_variant"],
                (
                    row["parameter_count_max"],
                    row["macro_mean_capacity_normalized_rmse"],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xscale("symlog", linthresh=1000)
        ax.set_xlabel("Parameter count (symlog)")
        ax.set_ylabel("Macro mean capacity-normalized RMSE")
        ax.set_title("Stage-2 legacy-seen test comparison (descriptive only)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"描述性 Pareto 图保存失败: {exc}")
        return None
    return path


def save_cross_variant_outputs(results, requested_variants, full_matrix):
    output_dir = comparison_output_dir()
    suffix = "" if full_matrix else "_partial"
    summary = pd.concat(
        [result["summary"] for result in results if not result["summary"].empty],
        ignore_index=True,
        sort=False,
    )
    horizon = pd.concat(
        [result["horizon"] for result in results if not result["horizon"].empty],
        ignore_index=True,
        sort=False,
    )
    regime = pd.concat(
        [result["regime"] for result in results if not result["regime"].empty],
        ignore_index=True,
        sort=False,
    )
    candidate_frames = [
        result["candidate"] for result in results if not result["candidate"].empty
    ]
    gate_frames = [result["gate"] for result in results if not result["gate"].empty]
    calibration_frames = [
        result["calibration"]
        for result in results
        if not result["calibration"].empty
    ]
    auxiliary_frames = [
        result["auxiliary"] for result in results if not result["auxiliary"].empty
    ]
    assignment_frames = [
        result["assignments"]
        for result in results
        if "assignments" in result and not result["assignments"].empty
    ]

    paths = {
        "summary": os.path.join(
            output_dir,
            f"{MODEL_FAMILY}_test_metrics_summary{suffix}.csv",
        ),
        "horizon": os.path.join(
            output_dir,
            f"{MODEL_FAMILY}_test_metrics_by_horizon_all{suffix}.csv",
        ),
        "regime": os.path.join(
            output_dir,
            f"{MODEL_FAMILY}_test_metrics_by_regime_all{suffix}.csv",
        ),
        "comparison": os.path.join(
            output_dir,
            f"{MODEL_FAMILY}_test_comparison_descriptive{suffix}.csv",
        ),
    }
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    horizon.to_csv(paths["horizon"], index=False, encoding="utf-8-sig")
    regime.to_csv(paths["regime"], index=False, encoding="utf-8-sig")
    comparison = build_descriptive_comparison(summary, requested_variants)
    comparison.to_csv(paths["comparison"], index=False, encoding="utf-8-sig")

    optional = {
        "candidate": candidate_frames,
        "gate": gate_frames,
        "gate_calibration": calibration_frames,
        "auxiliary": auxiliary_frames,
        "reference_regime_assignments": assignment_frames,
    }
    for name, frames in optional.items():
        if not frames:
            continue
        path = os.path.join(
            output_dir,
            f"{MODEL_FAMILY}_test_{name}_all{suffix}.csv",
        )
        pd.concat(frames, ignore_index=True, sort=False).to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )
        paths[name] = path
    paths["pareto_figure"] = _save_descriptive_pareto(comparison, output_dir)

    note_path = os.path.join(
        output_dir,
        f"{MODEL_FAMILY}_test_evaluation_note{suffix}.md",
    )
    with open(note_path, "w", encoding="utf-8") as file:
        file.write(
            "# RegimeEncoder-PatchTST 第二阶段测试说明\n\n"
            "- 本测试段已用于第一阶段结构分析，状态为 `legacy_seen`。\n"
            "- R0、R1、R6 直接引用第一阶段 B0、B2、B6 已保存结果。\n"
            "- 为保持引用结果可比，R2--R5 沿用 Stage-1 legacy 预处理口径。\n"
            "- 工况标签只用于模型前向完成后的分层评价。\n"
            "- 本脚本不根据测试指标选择模型；最终锁定只能来自完整验证结果。\n"
        )
    paths["evaluation_note"] = note_path
    return paths


def main():
    configure_prediction_reproducibility()
    test_files = discover_requested_test_files()
    if not test_files:
        raise FileNotFoundError(
            f"未找到测试文件模式 {common_predict.TEST_FILE_PATTERN}"
        )
    variants = get_requested_variants()
    if not variants:
        raise ValueError("没有请求任何第二阶段变体")
    print(f"发现 {len(test_files)} 个测试场站；变体: {variants}")
    print("R0/R1/R6 将直接引用 Stage-1 预测，不重复模型推理")

    results = []
    for variant_id in variants:
        if variant_id in REFERENCE_SOURCE_VARIANTS:
            results.append(load_reference_variant(variant_id, test_files))
        elif variant_id in TRAINABLE_VARIANTS:
            results.append(predict_trained_variant(variant_id, test_files))
        else:
            raise ValueError(f"未知变体: {variant_id}")

    all_test_files = common_predict.discover_test_files()
    expected_rows = len(VARIANT_SPECS) * len(all_test_files)
    actual_rows = sum(len(result["summary"]) for result in results)
    combined_keys = []
    for result in results:
        if result["summary"].empty:
            continue
        combined_keys.extend(
            zip(
                result["summary"]["model_variant"].astype(str),
                result["summary"]["farm_id"].astype(str),
            )
        )
    full_matrix = (
        set(variants) == set(VARIANT_SPECS)
        and not os.getenv("WIND_REGIME_FARMS")
        and actual_rows == expected_rows
        and len(set(combined_keys)) == expected_rows
    )
    paths = save_cross_variant_outputs(results, variants, full_matrix)
    print("第二阶段预测汇总已保存:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    if not full_matrix:
        print("当前为子集/不完整预测；只写 partial 汇总，不覆盖完整矩阵文件")
    print("测试集仅作描述性评价；脚本未执行基于 test 的模型选择")


if __name__ == "__main__":
    main()
