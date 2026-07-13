"""RegimeEncoder-PatchTST 显式工况特征筛选 F0--F7 测试入口。

F0/F1/F2/F3/F5/F6/F7 从专用特征筛选目录加载模型并生成与既有
RegimeEncoder-PatchTST 一致的整体、逐 horizon、工况、候选、门控和可视化
产物。F4 不重新训练、不重新推理、不复制大文件，直接读取既有 R4 测试结果
并保留所有源模型、预测和图形路径。

按用户指定，本脚本在完整 ``8 variants × all test farms`` 矩阵上，以各场站
等权宏平均 ``capacity_normalized_rmse`` 选择最终最优 F 模型。由于该测试段
此前已经被分析且本次又用于选型，输出会明确标注 ``legacy_seen``、
``test_used_for_feature_selection=True`` 和 ``test_is_final_blind=False``。

所有新文件写入特征筛选专用的 ``f0_f7_test_selection_output``，不会写入原
R2--R5 目录或原 ``testdata_predict_output``。
"""

import glob
import hashlib
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import wind_dl_model_predict as common_predict
import wind_RegimeEncoder_PatchTST_predict as regime_predict
import wind_RegimeEncoder_PatchTST_train as regime_train
from wind_RegimeEncoder_PatchTST_feature_screen_train import (
    ARCHITECTURE_VERSION,
    EXPECTED_PARAMETER_COUNTS,
    FULL_FEATURE_NAMES,
    MODEL_FAMILY,
    RANDOM_SEED,
    RESULT_ROOT,
    R4_SOURCE_VARIANT,
    TRAINABLE_VARIANTS,
    VARIANT_SPECS,
    build_feature_screen_model_from_artifact,
    get_feature_screen_custom_objects,
    get_requested_variants,
    selected_feature_names,
    validate_r4_reference_artifact,
    variant_dirs,
    variant_model_name,
)

warnings.filterwarnings("ignore")


OUTPUT_SUBDIR = "f0_f7_test_selection_output"
TEST_REUSE_STATUS = "legacy_seen_used_for_f0_f7_feature_selection"
SELECTION_METRIC = "capacity_normalized_rmse"
SELECTION_MACRO_METRIC = "macro_mean_capacity_normalized_rmse"


def configure_prediction_reproducibility():
    os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_SEED))
    keras.utils.set_random_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _sha256(path, chunk_size=1024 * 1024):
    if not path:
        return None
    try:
        path = os.fspath(path)
    except TypeError:
        return None
    if not os.path.exists(path):
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
    if not path or (isinstance(path, float) and np.isnan(path)):
        return None
    candidates = [os.fspath(path)]
    if not os.path.isabs(candidates[0]):
        candidates.append(os.path.join(os.path.dirname(__file__), candidates[0]))
    return next((value for value in candidates if os.path.exists(value)), None)


def prediction_output_dirs(variant_id):
    if variant_id not in TRAINABLE_VARIANTS:
        raise ValueError(f"{variant_id} 没有新增推理目录；F4必须直接引用R4")
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
    requested = os.getenv("WIND_FEATURE_SCREEN_FARMS")
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


def load_feature_artifact(variant_id, farm_id):
    if variant_id not in TRAINABLE_VARIANTS:
        raise ValueError(f"{variant_id} 没有新增特征筛选artifact")
    path = _artifact_path(variant_id, farm_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少 {variant_id}/{farm_id} artifact: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise TypeError(f"artifact 必须为dict: {path}")
    if artifact.get("variant_id") != variant_id:
        raise ValueError(f"artifact变体不匹配: {path}")
    if artifact.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(f"artifact架构版本不匹配: {path}")
    if int(artifact.get("random_seed", -1)) != RANDOM_SEED:
        raise ValueError(f"artifact seed必须为{RANDOM_SEED}: {path}")
    expected_names = selected_feature_names(variant_id)
    if tuple(artifact.get("full_regime_feature_names", ())) != FULL_FEATURE_NAMES:
        raise ValueError(f"artifact完整43维特征定义已漂移: {path}")
    if tuple(artifact.get("selected_regime_feature_names", ())) != expected_names:
        raise ValueError(f"artifact特征子集与{variant_id}定义不一致: {path}")
    if int(artifact.get("selected_regime_feature_count", -1)) != len(expected_names):
        raise ValueError(f"artifact特征维数与{variant_id}定义不一致: {path}")
    required = (
        "input_cols",
        "target_index",
        "scaler_x",
        "scaler_y",
        "history_len",
        "forecast_len",
        "diagnostic_layers",
        "regime_label_config",
        "regime_feature_config",
    )
    missing = [key for key in required if key not in artifact]
    if missing:
        raise KeyError(f"artifact缺少字段{missing}: {path}")
    if list(artifact["regime_label_config"].get("change_band_edges", ())) != list(
        regime_train.CHANGE_BAND_EDGES
    ):
        raise ValueError(f"artifact工况阈值与当前代码不一致: {path}")
    artifact = dict(artifact)
    artifact["artifact_path"] = os.path.abspath(path)
    return artifact


def load_feature_model(variant_id, farm_id, artifact):
    model_path = _resolve_existing_path(artifact.get("model_path"))
    if model_path:
        model = keras.models.load_model(
            model_path,
            custom_objects=get_feature_screen_custom_objects(),
            compile=False,
        )
        loaded_path = os.path.abspath(model_path)
    else:
        weights_path = _resolve_existing_path(artifact.get("best_weights_path"))
        if not weights_path:
            raise FileNotFoundError(
                f"缺少 {variant_id}/{farm_id} 完整模型和最佳权重"
            )
        model = build_feature_screen_model_from_artifact(artifact)
        model.load_weights(weights_path)
        loaded_path = os.path.abspath(weights_path)
    count = int(model.count_params())
    if int(artifact.get("total_params", count)) != count:
        raise ValueError(
            f"artifact参数量{artifact.get('total_params')}与模型{count}不一致"
        )
    if count != EXPECTED_PARAMETER_COUNTS[variant_id]:
        raise ValueError(
            f"{variant_id}参数量{count:,}与冻结实验协议"
            f"{EXPECTED_PARAMETER_COUNTS[variant_id]:,}不一致"
        )
    return model, loaded_path


def _relabel_frame(frame, variant_id):
    frame = frame.copy()
    frame["model_family"] = MODEL_FAMILY
    frame["model_variant"] = variant_id
    frame["variant_id"] = variant_id
    frame["model_name"] = variant_model_name(variant_id)
    return frame


def predict_one_feature_variant_farm(variant_id, test_file):
    farm_id = common_predict.get_farm_id(test_file)
    model_name = variant_model_name(variant_id)
    dirs = prediction_output_dirs(variant_id)
    print(f"\n===== 预测 {model_name} / 风电场 {farm_id} =====")
    artifact = load_feature_artifact(variant_id, farm_id)
    model, loaded_model_path = load_feature_model(variant_id, farm_id, artifact)
    df, features, actual_power, capacity = common_predict.prepare_prediction_arrays(
        test_file,
        artifact,
    )
    history_len = int(artifact["history_len"])
    forecast_len = int(artifact["forecast_len"])
    n_samples = regime_predict._prediction_sample_count(
        features, history_len, forecast_len
    )
    pred_ds, dataset_samples = common_predict.make_prediction_dataset(
        features,
        history_len,
        forecast_len,
    )
    if dataset_samples != n_samples:
        raise ValueError("预测dataset样本数不一致")
    outputs = regime_predict._validate_diagnostics(
        regime_predict._diagnostic_forward(model, pred_ds, artifact),
        n_samples,
        forecast_len,
    )

    # 模型前向结束后才读取未来真实功率，用于测试指标和realized regime。
    y_true = common_predict.build_truth_windows(
        actual_power,
        n_samples,
        history_len,
        forecast_len,
    )
    fused = regime_predict._inverse_candidate(artifact, outputs["forecast"], capacity)
    persistence = regime_predict._inverse_candidate(
        artifact, outputs["persistence_candidate"], capacity
    )
    corrected = regime_predict._inverse_candidate(
        artifact, outputs["corrected_candidate"], capacity
    )
    gate = outputs["gate"]
    last_power = persistence[:, 0]
    regimes = regime_train.build_regime_targets_numpy(y_true, last_power, capacity)
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
        dirs["predictions"], f"{model_name}_predictions_farm_{farm_id}.csv"
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
    metric_df["variant_id"] = variant_id
    metric_df["feature_count"] = len(selected_feature_names(variant_id))
    metric_df["parameter_count"] = parameter_count
    horizon_metric_path = os.path.join(
        dirs["root"], f"{model_name}_metrics_by_horizon_farm_{farm_id}.csv"
    )
    metric_df.to_csv(horizon_metric_path, index=False, encoding="utf-8-sig")

    candidate_frame = _relabel_frame(
        regime_predict._candidate_metric_rows(
            variant_id,
            model_name,
            farm_id,
            y_true,
            candidate_predictions,
            capacity,
        ),
        variant_id,
    )
    candidate_path = os.path.join(
        dirs["candidate_metrics"],
        f"{model_name}_candidate_metrics_farm_{farm_id}.csv",
    )
    candidate_frame.to_csv(candidate_path, index=False, encoding="utf-8-sig")

    assignment_frame = regime_predict._assignment_frame(
        df,
        farm_id,
        regimes,
        last_power,
        n_samples,
        history_len,
    )
    assignment_frame = _relabel_frame(assignment_frame, variant_id)
    assignment_path = os.path.join(
        dirs["regime_assignments"],
        f"{model_name}_regime_assignments_farm_{farm_id}.csv",
    )
    assignment_frame.to_csv(assignment_path, index=False, encoding="utf-8-sig")

    regime_frame = _relabel_frame(
        pd.DataFrame(
            regime_predict.build_regime_metric_rows(
                variant_id,
                farm_id,
                y_true,
                candidate_predictions,
                regimes,
                capacity,
            )
        ),
        variant_id,
    )
    regime_path = os.path.join(
        dirs["regime_metrics"],
        f"{model_name}_regime_metrics_farm_{farm_id}.csv",
    )
    regime_frame.to_csv(regime_path, index=False, encoding="utf-8-sig")

    gate_frame = _relabel_frame(
        pd.DataFrame(
            regime_predict._gate_rows(
                variant_id,
                farm_id,
                gate,
                y_true,
                persistence,
                corrected,
                fused,
                regimes,
            )
        ),
        variant_id,
    )
    gate_path = os.path.join(
        dirs["gate_diagnostics"],
        f"{model_name}_gate_by_regime_horizon_farm_{farm_id}.csv",
    )
    gate_frame.to_csv(gate_path, index=False, encoding="utf-8-sig")
    corrected_better = np.square(corrected - y_true) < np.square(
        persistence - y_true
    )
    calibration_frame = _relabel_frame(
        pd.DataFrame(
            regime_predict._gate_calibration_rows(
                variant_id, farm_id, gate, corrected_better
            )
        ),
        variant_id,
    )
    calibration_path = os.path.join(
        dirs["gate_diagnostics"],
        f"{model_name}_gate_calibration_farm_{farm_id}.csv",
    )
    calibration_frame.to_csv(
        calibration_path, index=False, encoding="utf-8-sig"
    )
    heatmap_path, calibration_figure_path = regime_predict._save_gate_figures(
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
            pred_df, model_name, farm_id, dirs, forecast_len
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

    all_metrics = metric_df[metric_df["horizon_step"] == "all"].iloc[0].to_dict()
    binary_entropy = -(
        gate * np.log(np.clip(gate, 1e-8, 1.0))
        + (1.0 - gate) * np.log(np.clip(1.0 - gate, 1e-8, 1.0))
    ) / np.log(2.0)
    spec = VARIANT_SPECS[variant_id]
    all_metrics.update(
        {
            "model_family": MODEL_FAMILY,
            "model_variant": variant_id,
            "variant_id": variant_id,
            "variant_label": spec["label"],
            "feature_groups": "+".join(spec["groups"]),
            "feature_count": len(selected_feature_names(variant_id)),
            "feature_names": json.dumps(
                selected_feature_names(variant_id), ensure_ascii=False
            ),
            "architecture_version": ARCHITECTURE_VERSION,
            "random_seed": RANDOM_SEED,
            "result_source": "stage2_feature_screen_model_inference",
            "reference_only": False,
            "source_variant": "b2_persistence_residual",
            "encoder_type": artifact.get("encoder_type"),
            "gate_type": artifact.get("gate_type"),
            "auxiliary_tasks": False,
            "parameter_count": parameter_count,
            "expected_parameter_count": EXPECTED_PARAMETER_COUNTS[variant_id],
            "trainable_parameter_count": trainable_parameter_count,
            "loaded_model_path": loaded_model_path,
            "loaded_model_sha256": _sha256(loaded_model_path),
            "artifact_path": artifact["artifact_path"],
            "artifact_sha256": _sha256(artifact["artifact_path"]),
            "prediction_path": prediction_path,
            "horizon_metric_path": horizon_metric_path,
            "candidate_metric_path": candidate_path,
            "regime_assignment_path": assignment_path,
            "regime_metric_path": regime_path,
            "gate_diagnostics_path": gate_path,
            "gate_calibration_path": calibration_path,
            "gate_heatmap_path": heatmap_path,
            "gate_calibration_figure_path": calibration_figure_path,
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
            "gate_change_magnitude_spearman": regime_predict._fixed_rank_correlation(
                gate.mean(axis=1), regimes["change_magnitude"]
            ),
            "fusion_reconstruction_max_abs_error": outputs[
                "fusion_reconstruction_max_abs_error"
            ],
            "evaluation_pipeline_version": regime_train.EVALUATION_PIPELINE_VERSION,
            "legacy_bidirectional_weather_imputation": True,
            "test_reuse_status": TEST_REUSE_STATUS,
            "source_test_reuse_status": "legacy_seen",
            "test_used_for_feature_selection": True,
            "feature_screening_test_selection_eligible": True,
            "test_selection_prohibited": False,
            "test_is_final_blind_evaluation": False,
            "selection_split": "test",
            "selection_metric": SELECTION_MACRO_METRIC,
            "training_code_path": os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "wind_RegimeEncoder_PatchTST_feature_screen_train.py",
                )
            ),
            "prediction_code_path": os.path.abspath(__file__),
            "prediction_code_sha256": _sha256(os.path.abspath(__file__)),
            **router_fields,
            **{
                f"weighted_curve_{key}": value
                for key, value in weighted_metrics.items()
            },
        }
    )
    print(
        f"{model_name} / {farm_id}: NRMSE="
        f"{all_metrics[SELECTION_METRIC]:.6f}, features="
        f"{len(selected_feature_names(variant_id))}, params={parameter_count:,}"
    )
    del model
    keras.backend.clear_session()
    return {
        "summary": pd.DataFrame([all_metrics]),
        "horizon": metric_df,
        "candidate": candidate_frame,
        "regime": regime_frame,
        "gate": gate_frame,
        "calibration": calibration_frame,
        "assignments": assignment_frame,
    }


def predict_feature_variant(variant_id, test_files):
    outputs = {
        "summary": [],
        "horizon": [],
        "candidate": [],
        "regime": [],
        "gate": [],
        "calibration": [],
        "assignments": [],
    }
    for test_file in test_files:
        result = predict_one_feature_variant_farm(variant_id, test_file)
        for key in outputs:
            outputs[key].append(result[key])
    combined = {
        key: pd.concat(frames, ignore_index=True, sort=False)
        for key, frames in outputs.items()
    }
    dirs = prediction_output_dirs(variant_id)
    model_name = variant_model_name(variant_id)
    full_farms = (
        not os.getenv("WIND_FEATURE_SCREEN_FARMS")
        and len(test_files) == len(common_predict.discover_test_files())
    )
    suffix = "" if full_farms else "_partial"
    combined["summary"].to_csv(
        os.path.join(dirs["root"], f"{model_name}_test_metrics_summary{suffix}.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    combined["horizon"].to_csv(
        os.path.join(
            dirs["root"], f"{model_name}_test_metrics_by_horizon_all{suffix}.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )
    return combined


def _r4_output_root():
    return os.path.join(
        regime_train.variant_dirs(R4_SOURCE_VARIANT, create=False)["root"],
        "testdata_predict_output",
    )


def _r4_test_paths():
    root = _r4_output_root()
    source_name = regime_train.variant_model_name(R4_SOURCE_VARIANT)
    return {
        "root": root,
        "summary": os.path.join(root, f"{source_name}_test_metrics_summary.csv"),
        "horizon": os.path.join(
            root, f"{source_name}_test_metrics_by_horizon_all.csv"
        ),
        "predictions": os.path.join(root, "predictions"),
        "candidate": os.path.join(root, "candidate_metrics"),
        "regime": os.path.join(root, "regime_metrics"),
        "gate": os.path.join(root, "gate_diagnostics"),
        "assignments": os.path.join(root, "regime_assignments"),
    }


def _load_one_per_farm(directory, suffix_pattern, farm_ids):
    frames = []
    paths = []
    for farm_id in farm_ids:
        matches = sorted(glob.glob(os.path.join(directory, f"*{suffix_pattern}{farm_id}.csv")))
        if len(matches) != 1:
            raise ValueError(
                f"R4引用文件应唯一: dir={directory}, farm={farm_id}, matches={matches}"
            )
        frame = pd.read_csv(matches[0])
        frame["farm_id"] = frame["farm_id"].astype(str)
        frames.append(frame)
        paths.append(os.path.abspath(matches[0]))
    return pd.concat(frames, ignore_index=True, sort=False), paths


def load_f4_reference(test_files):
    """直接读取既有R4预测、指标和图路径，不重新推理或复制文件。"""
    paths = _r4_test_paths()
    for key in ("summary", "horizon"):
        if not os.path.exists(paths[key]):
            raise FileNotFoundError(f"F4引用所需R4 {key}不存在: {paths[key]}")
    farm_ids = [common_predict.get_farm_id(path) for path in test_files]
    farm_set = set(farm_ids)
    summary = pd.read_csv(paths["summary"])
    horizon = pd.read_csv(paths["horizon"])
    summary["farm_id"] = summary["farm_id"].astype(str)
    horizon["farm_id"] = horizon["farm_id"].astype(str)
    summary = summary[summary["farm_id"].isin(farm_set)].copy()
    horizon = horizon[horizon["farm_id"].isin(farm_set)].copy()
    if summary["farm_id"].nunique() != len(farm_set) or len(summary) != len(farm_set):
        raise ValueError("R4测试summary未按每场站一行覆盖全部请求场站")
    counts = horizon.groupby("farm_id").size()
    if set(counts.index) != farm_set or not (
        counts == regime_train.FORECAST_LEN + 1
    ).all():
        raise ValueError("R4测试horizon必须对每场站包含16步和all共17行")
    if not (
        pd.to_numeric(summary["parameter_count"], errors="coerce") == 21151
    ).all():
        raise ValueError("F4引用的R4参数量不是21,151")

    # 读取源artifact只做身份/43维定义核验；不重建模型、不执行推理。
    for _, row in summary.iterrows():
        artifact_path = _resolve_existing_path(row.get("artifact_path"))
        model_path = _resolve_existing_path(row.get("loaded_model_path"))
        if artifact_path is None or model_path is None:
            raise FileNotFoundError(
                f"F4/R4源artifact或模型不存在: farm={row['farm_id']}"
            )
        artifact = joblib.load(artifact_path)
        validate_r4_reference_artifact(artifact, artifact_path)

    source_model_names = summary["model_name"].copy()
    source_architecture = summary.get("architecture_version", pd.Series(index=summary.index))
    source_test_status = summary.get("test_reuse_status", pd.Series(index=summary.index))
    source_selection_flag = summary.get(
        "test_selection_prohibited", pd.Series(index=summary.index)
    )
    source_prediction_paths = summary["prediction_path"].copy()
    summary["source_model_name"] = source_model_names
    summary["source_model_family"] = regime_train.MODEL_FAMILY
    summary["source_model_variant"] = R4_SOURCE_VARIANT
    summary["source_architecture_version"] = source_architecture
    summary["source_test_reuse_status"] = source_test_status
    summary["source_test_selection_prohibited"] = source_selection_flag
    summary["source_summary_path"] = os.path.abspath(paths["summary"])
    summary["source_summary_sha256"] = _sha256(paths["summary"])
    summary["source_horizon_path"] = os.path.abspath(paths["horizon"])
    summary["source_horizon_sha256"] = _sha256(paths["horizon"])
    summary["source_prediction_path"] = source_prediction_paths
    summary["model_family"] = MODEL_FAMILY
    summary["model_name"] = variant_model_name("f4")
    summary["model_variant"] = "f4"
    summary["variant_id"] = "f4"
    summary["variant_label"] = VARIANT_SPECS["f4"]["label"]
    summary["feature_groups"] = "+".join(VARIANT_SPECS["f4"]["groups"])
    summary["feature_count"] = len(FULL_FEATURE_NAMES)
    summary["feature_names"] = json.dumps(FULL_FEATURE_NAMES, ensure_ascii=False)
    summary["expected_parameter_count"] = EXPECTED_PARAMETER_COUNTS["f4"]
    summary["result_source"] = "direct_reference_existing_r4_test_outputs"
    summary["reference_only"] = True
    summary["source_variant"] = R4_SOURCE_VARIANT
    summary["test_reuse_status"] = TEST_REUSE_STATUS
    summary["test_used_for_feature_selection"] = True
    summary["feature_screening_test_selection_eligible"] = True
    summary["test_selection_prohibited"] = False
    summary["test_is_final_blind_evaluation"] = False
    summary["selection_split"] = "test"
    summary["selection_metric"] = SELECTION_MACRO_METRIC
    summary["training_code_path"] = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "wind_RegimeEncoder_PatchTST_feature_screen_train.py",
        )
    )
    summary["prediction_code_path"] = os.path.abspath(__file__)
    summary["prediction_code_sha256"] = _sha256(os.path.abspath(__file__))

    horizon["source_model_name"] = horizon.get("model_name")
    horizon["source_model_variant"] = R4_SOURCE_VARIANT
    horizon["model_name"] = variant_model_name("f4")
    horizon["model_family"] = MODEL_FAMILY
    horizon["model_variant"] = "f4"
    horizon["variant_id"] = "f4"
    horizon["feature_count"] = len(FULL_FEATURE_NAMES)
    horizon["result_source"] = "direct_reference_existing_r4_test_outputs"

    candidate, candidate_paths = _load_one_per_farm(
        paths["candidate"], "candidate_metrics_farm_", farm_ids
    )
    regime, regime_paths = _load_one_per_farm(
        paths["regime"], "regime_metrics_farm_", farm_ids
    )
    gate, gate_paths = _load_one_per_farm(
        paths["gate"], "gate_by_regime_horizon_farm_", farm_ids
    )
    calibration, calibration_paths = _load_one_per_farm(
        paths["gate"], "gate_calibration_farm_", farm_ids
    )
    assignments, assignment_paths = _load_one_per_farm(
        paths["assignments"], "regime_assignments_farm_", farm_ids
    )
    optional = {}
    for key, frame, source_paths in (
        ("candidate", candidate, candidate_paths),
        ("regime", regime, regime_paths),
        ("gate", gate, gate_paths),
        ("calibration", calibration, calibration_paths),
        ("assignments", assignments, assignment_paths),
    ):
        frame = _relabel_frame(frame, "f4")
        frame["result_source"] = "direct_reference_existing_r4_test_outputs"
        frame["source_model_variant"] = R4_SOURCE_VARIANT
        frame["source_file_path"] = frame["farm_id"].map(
            dict(zip(farm_ids, source_paths))
        )
        optional[key] = frame
    return {
        "summary": summary,
        "horizon": horizon,
        **optional,
    }


def _load_prediction_truth(path):
    resolved = _resolve_existing_path(path)
    if resolved is None:
        raise FileNotFoundError(f"用于真值对齐的预测文件不存在: {path}")
    frame = pd.read_csv(resolved)
    required = {"sample_id", "horizon_step", "actual_power"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"预测文件缺少真值对齐列{sorted(missing)}: {resolved}")
    order = ["sample_id", "horizon_step"]
    return frame.sort_values(order).reset_index(drop=True)


def validate_truth_alignment(summary):
    """保证所有F模型使用与F4完全相同的测试样本和真实值。"""
    for farm_id, farm_frame in summary.groupby("farm_id"):
        if set(farm_frame["model_variant"]) != set(VARIANT_SPECS):
            raise ValueError(f"场站{farm_id}没有覆盖完整F0--F7，无法校验真值")
        f4_row = farm_frame[farm_frame["model_variant"] == "f4"].iloc[0]
        reference_path = f4_row.get("source_prediction_path") or f4_row.get(
            "prediction_path"
        )
        reference = _load_prediction_truth(reference_path)
        reference_keys = reference[["sample_id", "horizon_step"]].to_numpy()
        reference_truth = pd.to_numeric(
            reference["actual_power"], errors="coerce"
        ).to_numpy()
        reference_time = (
            reference["forecast_origin_time"].astype(str).to_numpy()
            if "forecast_origin_time" in reference
            else None
        )
        for _, row in farm_frame.iterrows():
            path = row.get("prediction_path")
            candidate = _load_prediction_truth(path)
            if len(candidate) != len(reference) or not np.array_equal(
                candidate[["sample_id", "horizon_step"]].to_numpy(), reference_keys
            ):
                raise ValueError(
                    f"{row['model_variant']}/{farm_id}与F4测试窗口键不一致"
                )
            candidate_truth = pd.to_numeric(
                candidate["actual_power"], errors="coerce"
            ).to_numpy()
            if not np.allclose(
                candidate_truth,
                reference_truth,
                rtol=0.0,
                atol=1e-7,
                equal_nan=True,
            ):
                raise ValueError(
                    f"{row['model_variant']}/{farm_id}与F4测试真实功率不一致"
                )
            if reference_time is not None:
                if "forecast_origin_time" not in candidate or not np.array_equal(
                    candidate["forecast_origin_time"].astype(str).to_numpy(),
                    reference_time,
                ):
                    raise ValueError(
                        f"{row['model_variant']}/{farm_id}与F4预测起报时刻不一致"
                    )


def build_test_comparison(summary):
    metric_names = (
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
        "gate_mean",
        "gate_oracle_brier",
    )
    f4 = summary[summary["model_variant"] == "f4"][[
        "farm_id",
        SELECTION_METRIC,
    ]].rename(columns={SELECTION_METRIC: "f4_nrmse"})
    rows = []
    for order, variant_id in enumerate(VARIANT_SPECS):
        frame = summary[summary["model_variant"] == variant_id].copy()
        parameters = pd.to_numeric(frame["parameter_count"], errors="coerce")
        paired = frame[["farm_id", SELECTION_METRIC]].merge(f4, on="farm_id")
        paired[SELECTION_METRIC] = pd.to_numeric(
            paired[SELECTION_METRIC], errors="coerce"
        )
        paired["f4_nrmse"] = pd.to_numeric(paired["f4_nrmse"], errors="coerce")
        row = {
            "variant_order": order,
            "model_variant": variant_id,
            "model_name": variant_model_name(variant_id),
            "feature_groups": "+".join(VARIANT_SPECS[variant_id]["groups"]),
            "feature_count": len(selected_feature_names(variant_id)),
            "farm_count": int(frame["farm_id"].astype(str).nunique()),
            "parameter_count_min": (
                int(parameters.min()) if parameters.notna().any() else np.nan
            ),
            "parameter_count_max": (
                int(parameters.max()) if parameters.notna().any() else np.nan
            ),
            "requires_training": VARIANT_SPECS[variant_id]["requires_training"],
            "result_source": (
                "stage2_feature_screen_model_inference"
                if VARIANT_SPECS[variant_id]["requires_training"]
                else "direct_reference_existing_r4_test_outputs"
            ),
            "selection_split": "test",
            "selection_metric": SELECTION_MACRO_METRIC,
            "test_reuse_status": TEST_REUSE_STATUS,
            "test_used_for_feature_selection": True,
            "test_is_final_blind_evaluation": False,
            "farms_better_than_f4": int(
                (paired[SELECTION_METRIC] < paired["f4_nrmse"]).sum()
            ),
            "farms_equal_to_f4": int(
                np.isclose(
                    paired[SELECTION_METRIC], paired["f4_nrmse"], atol=1e-12
                ).sum()
            ),
            "macro_nrmse_delta_vs_f4": float(
                (paired[SELECTION_METRIC] - paired["f4_nrmse"]).mean()
            ),
        }
        for metric in metric_names:
            values = (
                pd.to_numeric(frame[metric], errors="coerce")
                if metric in frame
                else pd.Series(dtype=float)
            )
            row[f"macro_mean_{metric}"] = float(values.mean())
            row[f"macro_std_{metric}"] = float(values.std(ddof=0))
        rows.append(row)
    comparison = pd.DataFrame(rows)
    if not np.isfinite(comparison[SELECTION_MACRO_METRIC]).all():
        raise ValueError("至少一个F变体的宏平均测试NRMSE不是有限值")
    order = comparison.sort_values(
        [
            SELECTION_MACRO_METRIC,
            "macro_std_capacity_normalized_rmse",
            "feature_count",
            "parameter_count_max",
            "variant_order",
        ],
        kind="mergesort",
    ).index
    ranks = pd.Series(np.arange(1, len(order) + 1), index=order)
    comparison["selection_rank"] = ranks
    comparison["selected_final_variant"] = comparison["selection_rank"] == 1
    comparison["selection_tie_break_order"] = (
        "macro_test_nrmse -> macro_test_nrmse_std -> feature_count -> "
        "parameter_count -> variant_order"
    )
    return comparison.sort_values("selection_rank").reset_index(drop=True)


def build_feature_contribution(summary):
    comparisons = (
        ("f0", "f1", "add_H", "在P上加入轮毂高度风速H"),
        ("f1", "f2", "add_M", "在P+H上加入多高度风速M"),
        ("f2", "f3", "add_D", "在P+H+M上加入风向D"),
        ("f3", "f4", "add_C", "在P+H+M+D上加入一致性C"),
        ("f5", "f3", "add_P", "在H+M+D上加入功率状态P"),
        ("f6", "f3", "add_H_reverse", "在P+M+D上加入轮毂风速H"),
        ("f7", "f3", "add_M_reverse", "在P+H+D上加入多高度风速M"),
    )
    rows = []
    values = summary[["model_variant", "farm_id", SELECTION_METRIC]].copy()
    values[SELECTION_METRIC] = pd.to_numeric(values[SELECTION_METRIC], errors="coerce")
    for source_id, target_id, change, description in comparisons:
        source = values[values["model_variant"] == source_id][
            ["farm_id", SELECTION_METRIC]
        ].rename(columns={SELECTION_METRIC: "source_nrmse"})
        target = values[values["model_variant"] == target_id][
            ["farm_id", SELECTION_METRIC]
        ].rename(columns={SELECTION_METRIC: "target_nrmse"})
        paired = source.merge(target, on="farm_id")
        delta = paired["target_nrmse"] - paired["source_nrmse"]
        relative = delta / paired["source_nrmse"] * 100.0
        rows.append(
            {
                "comparison": change,
                "description": description,
                "source_variant": source_id,
                "target_variant": target_id,
                "source_groups": "+".join(VARIANT_SPECS[source_id]["groups"]),
                "target_groups": "+".join(VARIANT_SPECS[target_id]["groups"]),
                "paired_farm_count": len(paired),
                "source_macro_test_nrmse": float(paired["source_nrmse"].mean()),
                "target_macro_test_nrmse": float(paired["target_nrmse"].mean()),
                "target_minus_source_nrmse": float(delta.mean()),
                "relative_change_pct": float(relative.mean()),
                "improves_macro_test_nrmse": bool(delta.mean() < 0),
                "farms_improved": int((delta < 0).sum()),
                "farms_degraded": int((delta > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_horizon_comparison(horizon):
    frame = horizon[horizon["horizon_step"].astype(str) != "all"].copy()
    frame["horizon_step"] = pd.to_numeric(frame["horizon_step"], errors="raise")
    frame[SELECTION_METRIC] = pd.to_numeric(
        frame[SELECTION_METRIC], errors="coerce"
    )
    grouped = (
        frame.groupby(["model_variant", "horizon_step"], as_index=False)
        .agg(
            macro_mean_capacity_normalized_rmse=(SELECTION_METRIC, "mean"),
            macro_std_capacity_normalized_rmse=(SELECTION_METRIC, "std"),
            farm_count=("farm_id", "nunique"),
        )
    )
    grouped["horizon_minutes"] = grouped["horizon_step"] * 15
    f4 = grouped[grouped["model_variant"] == "f4"][[
        "horizon_step",
        "macro_mean_capacity_normalized_rmse",
    ]].rename(
        columns={"macro_mean_capacity_normalized_rmse": "f4_macro_nrmse"}
    )
    grouped = grouped.merge(f4, on="horizon_step", how="left")
    grouped["delta_vs_f4"] = (
        grouped["macro_mean_capacity_normalized_rmse"] - grouped["f4_macro_nrmse"]
    )
    return grouped.sort_values(["model_variant", "horizon_step"])


def _save_selection_figures(comparison, summary, horizon_comparison, output_dir):
    dirs = {"matplotlib_cache": os.path.join(output_dir, "matplotlib_cache")}
    plt = common_predict.setup_matplotlib(dirs)
    paths = {}

    rank_path = os.path.join(output_dir, "feature_screening_f0_f7_test_nrmse_rank.png")
    rank = comparison.sort_values(SELECTION_MACRO_METRIC, ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [
        "tab:red" if bool(value) else "tab:blue"
        for value in rank["selected_final_variant"]
    ]
    ax.barh(rank["model_variant"], rank[SELECTION_MACRO_METRIC], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Macro mean test capacity-normalized RMSE")
    ax.set_title("F0--F7 test-set feature selection")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(rank_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["rank_figure"] = rank_path

    heatmap_path = os.path.join(output_dir, "feature_screening_f0_f7_test_farm_heatmap.png")
    pivot = summary.pivot(
        index="model_variant", columns="farm_id", values=SELECTION_METRIC
    ).reindex(VARIANT_SPECS)
    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_xticks(
        np.arange(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right"
    )
    ax.set_title("Per-farm test NRMSE (lower is better)")
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            ax.text(
                column,
                row,
                f"{pivot.iloc[row, column]:.4f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if pivot.iloc[row, column] > pivot.to_numpy().mean() else "black",
            )
    fig.colorbar(image, ax=ax, label="Capacity-normalized RMSE")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["farm_heatmap"] = heatmap_path

    horizon_path = os.path.join(output_dir, "feature_screening_f0_f7_test_horizon_nrmse.png")
    fig, ax = plt.subplots(figsize=(11, 6))
    for variant_id in VARIANT_SPECS:
        frame = horizon_comparison[
            horizon_comparison["model_variant"] == variant_id
        ]
        ax.plot(
            frame["horizon_minutes"],
            frame["macro_mean_capacity_normalized_rmse"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=variant_id,
        )
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("Macro mean capacity-normalized RMSE")
    ax.set_title("F0--F7 test NRMSE by horizon")
    ax.grid(alpha=0.3)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(horizon_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["horizon_figure"] = horizon_path

    pareto_path = os.path.join(output_dir, "feature_screening_f0_f7_test_pareto.png")
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in comparison.iterrows():
        selected = bool(row["selected_final_variant"])
        ax.scatter(
            row["parameter_count_max"],
            row[SELECTION_MACRO_METRIC],
            s=80 if selected else 45,
            color="tab:red" if selected else "tab:blue",
        )
        ax.annotate(
            f"{row['model_variant']} ({int(row['feature_count'])}F)",
            (row["parameter_count_max"], row[SELECTION_MACRO_METRIC]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Parameter count")
    ax.set_ylabel("Macro mean test capacity-normalized RMSE")
    ax.set_title("F0--F7 accuracy--complexity comparison")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(pareto_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["pareto_figure"] = pareto_path
    return paths


def _selection_markdown(comparison, contribution, output_path, figure_paths):
    selected = comparison[comparison["selected_final_variant"]].iloc[0]
    columns = [
        "selection_rank",
        "model_variant",
        "feature_groups",
        "feature_count",
        "parameter_count_max",
        SELECTION_MACRO_METRIC,
        "macro_std_capacity_normalized_rmse",
        "farms_better_than_f4",
    ]
    table = comparison[columns].copy()
    for column in (
        SELECTION_MACRO_METRIC,
        "macro_std_capacity_normalized_rmse",
    ):
        table[column] = table[column].map(lambda value: f"{value:.6f}")
    contribution_table = contribution[[
        "comparison",
        "source_variant",
        "target_variant",
        "target_minus_source_nrmse",
        "relative_change_pct",
        "farms_improved",
        "farms_degraded",
    ]].copy()
    contribution_table["target_minus_source_nrmse"] = contribution_table[
        "target_minus_source_nrmse"
    ].map(lambda value: f"{value:+.6f}")
    contribution_table["relative_change_pct"] = contribution_table[
        "relative_change_pct"
    ].map(lambda value: f"{value:+.3f}%")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("# F0--F7 显式工况特征筛选（测试集选型）\n\n")
        file.write(
            f"最终模型：**{selected['model_variant']}**，特征组 "
            f"`{selected['feature_groups']}`，测试集5场站等权宏平均NRMSE="
            f"`{selected[SELECTION_MACRO_METRIC]:.6f}`。\n\n"
        )
        file.write("## 选型口径\n\n")
        file.write(
            "- 主指标：每个场站全16步容量归一化RMSE，再对场站等权宏平均。\n"
            "- 主指标越低越优；完全同值依次按跨场站标准差、特征数、参数量和F编号破平。\n"
            "- F4直接引用既有R4模型、预测、诊断和图形，没有重新训练或推理。\n"
            "- 本测试段状态为`legacy_seen`且此次用于特征选型，因此不再是最终盲测。\n\n"
        )
        file.write("## 排名\n\n")
        file.write(table.to_markdown(index=False))
        file.write("\n\n## 特征组增量/反向消融\n\n")
        file.write(contribution_table.to_markdown(index=False))
        file.write("\n\n## 图形\n\n")
        for name, path in figure_paths.items():
            file.write(f"- {name}: `{os.path.abspath(path)}`\n")


def save_cross_variant_outputs(results, variants, full_matrix):
    output_dir = comparison_output_dir()
    keys = (
        "summary",
        "horizon",
        "candidate",
        "regime",
        "gate",
        "calibration",
        "assignments",
    )
    combined = {}
    for key in keys:
        frames = [result[key] for result in results if not result[key].empty]
        combined[key] = (
            pd.concat(frames, ignore_index=True, sort=False)
            if frames
            else pd.DataFrame()
        )
    if full_matrix:
        suffix = ""
    else:
        farms = sorted(combined["summary"]["farm_id"].astype(str).unique())
        raw_tag = f"{'-'.join(variants)}__farms_{'-'.join(farms)}"
        tag = (
            raw_tag
            if len(raw_tag) <= 150
            else hashlib.sha1(raw_tag.encode()).hexdigest()[:12]
        )
        suffix = f"_partial_{tag}"
    paths = {}
    filenames = {
        "summary": "feature_screening_f0_f7_test_metrics_summary",
        "horizon": "feature_screening_f0_f7_test_metrics_by_horizon_all",
        "candidate": "feature_screening_f0_f7_test_candidate_all",
        "regime": "feature_screening_f0_f7_test_metrics_by_regime_all",
        "gate": "feature_screening_f0_f7_test_gate_all",
        "calibration": "feature_screening_f0_f7_test_gate_calibration_all",
        "assignments": "feature_screening_f0_f7_test_regime_assignments_all",
    }
    for key, stem in filenames.items():
        if combined[key].empty:
            continue
        path = os.path.join(output_dir, f"{stem}{suffix}.csv")
        combined[key].to_csv(path, index=False, encoding="utf-8-sig")
        paths[key] = path

    if full_matrix:
        validate_truth_alignment(combined["summary"])
        comparison = build_test_comparison(combined["summary"])
        contribution = build_feature_contribution(combined["summary"])
        horizon_comparison = build_horizon_comparison(combined["horizon"])
        comparison_path = os.path.join(
            output_dir, "feature_screening_f0_f7_test_variant_comparison.csv"
        )
        contribution_path = os.path.join(
            output_dir, "feature_screening_f0_f7_test_feature_contribution.csv"
        )
        horizon_path = os.path.join(
            output_dir, "feature_screening_f0_f7_test_horizon_comparison.csv"
        )
        selection_path = os.path.join(
            output_dir, "feature_screening_f0_f7_test_final_selection.csv"
        )
        comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
        contribution.to_csv(contribution_path, index=False, encoding="utf-8-sig")
        horizon_comparison.to_csv(horizon_path, index=False, encoding="utf-8-sig")
        comparison[comparison["selected_final_variant"]].to_csv(
            selection_path, index=False, encoding="utf-8-sig"
        )
        figure_paths = _save_selection_figures(
            comparison, combined["summary"], horizon_comparison, output_dir
        )
        report_path = os.path.join(
            output_dir, "feature_screening_f0_f7_test_final_selection.md"
        )
        _selection_markdown(comparison, contribution, report_path, figure_paths)
        paths.update(
            {
                "comparison": comparison_path,
                "feature_contribution": contribution_path,
                "horizon_comparison": horizon_path,
                "final_selection": selection_path,
                "selection_report": report_path,
                **figure_paths,
            }
        )
    else:
        note_path = os.path.join(
            output_dir, f"feature_screening_f0_f7_test_partial_note{suffix}.md"
        )
        with open(note_path, "w", encoding="utf-8") as file:
            file.write(
                "# F0--F7 部分测试运行\n\n"
                "当前没有覆盖完整F0--F7和全部测试场站，因此只保存partial指标，"
                "未生成最终最优模型结论。\n"
            )
        paths["partial_note"] = note_path
    return paths


def main():
    configure_prediction_reproducibility()
    test_files = discover_requested_test_files()
    if not test_files:
        raise FileNotFoundError(
            f"未找到测试文件模式 {common_predict.TEST_FILE_PATTERN}"
        )
    variants = get_requested_variants()
    print(f"发现{len(test_files)}个测试场站；F矩阵: {variants}")
    if "f4" in variants:
        print("F4直接引用既有R4测试结果/预测/可视化，不重新推理或复制")

    results = []
    for variant_id in variants:
        if variant_id == "f4":
            results.append(load_f4_reference(test_files))
        elif variant_id in TRAINABLE_VARIANTS:
            results.append(predict_feature_variant(variant_id, test_files))
        else:
            raise ValueError(f"未知F变体: {variant_id}")

    all_test_files = common_predict.discover_test_files()
    expected_rows = len(VARIANT_SPECS) * len(all_test_files)
    actual_rows = sum(len(result["summary"]) for result in results)
    combined_keys = []
    for result in results:
        combined_keys.extend(
            zip(
                result["summary"]["model_variant"].astype(str),
                result["summary"]["farm_id"].astype(str),
            )
        )
    full_matrix = (
        set(variants) == set(VARIANT_SPECS)
        and not os.getenv("WIND_FEATURE_SCREEN_FARMS")
        and actual_rows == expected_rows
        and len(set(combined_keys)) == expected_rows
    )
    paths = save_cross_variant_outputs(results, variants, full_matrix)
    print("F0--F7测试输出已保存:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    if full_matrix:
        selected = pd.read_csv(paths["final_selection"]).iloc[0]
        print(
            f"按测试集5场站宏平均NRMSE选定: {selected['model_variant']} "
            f"({selected[SELECTION_MACRO_METRIC]:.6f})"
        )
        print("注意：该测试集已用于选型，不能再称为最终盲测")
    else:
        print("当前为子集/不完整预测；只写partial文件，不生成最终选型结论")


if __name__ == "__main__":
    main()
