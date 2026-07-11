"""FeTS-PatchTST 第一阶段最小有效结构统一测试集预测。

本文件只负责读取 ``wind_FeTS_PatchTST_min_train.py`` 生成的各变体
artifact/模型，并复用 ``wind_dl_model_predict.py`` 的数据对齐、评价指标和
可视化实现。它不会读写现有 ``wind_dl_all_models_*.csv``，避免局部消融实验
覆盖正式全模型汇总。
"""

import glob
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import wind_dl_model_predict as common_predict
from wind_FeTS_PatchTST_min_train import (
    ARCHITECTURE_VERSION,
    MODEL_FAMILY,
    RANDOM_SEED,
    RESULT_ROOT,
    VARIANT_SPECS,
    StaticHorizonRouter,
    build_fets_patchtst_min_model_from_artifact,
    get_requested_variants,
    variant_dirs,
    variant_model_name,
)


warnings.filterwarnings("ignore")


OUTPUT_SUBDIR = "testdata_predict_output"
REFERENCE_VARIANT_ID = "b6_all_dynamic"
MACRO_NRMSE_TOLERANCE = 0.005
FARM_NRMSE_TOLERANCE = 0.01
MIN_NON_INFERIOR_FARMS = 4


def configure_prediction_reproducibility():
    """固定预测期随机状态；训练期的 seed 仍以 artifact 校验为准。"""
    os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_SEED))
    keras.utils.set_random_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _first_path(mapping, *keys):
    for key in keys:
        value = mapping.get(key)
        if value:
            return os.fspath(value)
    return None


def _variant_root(variant_id, paths=None):
    paths = paths or variant_dirs(variant_id, create=True)
    root = _first_path(
        paths,
        "root",
        "variant_root",
        "result_root",
        "model_dir",
    )
    if root:
        return root
    return os.path.join(RESULT_ROOT, variant_id)


def prediction_output_dirs(variant_id):
    """创建与既有预测脚本相同的输出目录结构。"""
    train_paths = variant_dirs(variant_id, create=True)
    output_root = os.path.join(
        _variant_root(variant_id, train_paths),
        OUTPUT_SUBDIR,
    )
    dirs = {
        "root": output_root,
        "predictions": os.path.join(output_root, "predictions"),
        "figures": os.path.join(output_root, "figures"),
        "single_windows": os.path.join(
            output_root,
            "single_window_comparisons",
        ),
        "weighted_curves": os.path.join(output_root, "weighted_curves"),
        "router_diagnostics": os.path.join(
            output_root,
            "router_diagnostics",
        ),
        "matplotlib_cache": os.path.join(output_root, "matplotlib_cache"),
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
    requested = os.getenv("WIND_FETS_MIN_FARMS")
    if not requested:
        return files
    farm_ids = {item.strip() for item in requested.split(",") if item.strip()}
    return [path for path in files if common_predict.get_farm_id(path) in farm_ids]


def _artifact_candidates(variant_id, farm_id):
    paths = variant_dirs(variant_id, create=False)
    preprocess_dir = _first_path(
        paths,
        "preprocess",
        "preprocess_dir",
        "artifacts",
        "artifact_dir",
    )
    if not preprocess_dir:
        preprocess_dir = os.path.join(
            _variant_root(variant_id, paths),
            "preprocess",
        )

    model_name = variant_model_name(variant_id)
    expected_names = [
        f"{model_name}_farm_{farm_id}_preprocess.pkl",
        f"{MODEL_FAMILY}_{variant_id}_farm_{farm_id}_preprocess.pkl",
    ]
    candidates = [
        os.path.join(preprocess_dir, name) for name in dict.fromkeys(expected_names)
    ]
    wildcard_matches = sorted(
        glob.glob(os.path.join(preprocess_dir, f"*farm_{farm_id}_preprocess.pkl"))
    )
    for match in wildcard_matches:
        if match not in candidates:
            candidates.append(match)
    return candidates


def load_variant_artifact(variant_id, farm_id):
    candidates = _artifact_candidates(variant_id, farm_id)
    artifact_path = next(
        (path for path in candidates if os.path.exists(path)),
        None,
    )
    if artifact_path is None:
        raise FileNotFoundError(
            f"未找到变体 {variant_id} 场站 {farm_id} 的预处理 artifact；"
            f"已检查: {candidates}"
        )

    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise TypeError(f"artifact 必须是 dict: {artifact_path}")

    artifact_variant = artifact.get("model_variant", artifact.get("variant_id"))
    if artifact_variant != variant_id:
        raise ValueError(
            f"artifact 变体不匹配: 请求={variant_id}, "
            f"artifact={artifact_variant}, path={artifact_path}"
        )
    if artifact.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(
            f"artifact 结构版本不匹配: 请求={ARCHITECTURE_VERSION}, "
            f"artifact={artifact.get('architecture_version', 'unknown')}, "
            f"path={artifact_path}"
        )
    artifact_seed = artifact.get("random_seed")
    if artifact_seed is None or int(artifact_seed) != int(RANDOM_SEED):
        raise ValueError(
            f"artifact 随机种子必须为 {RANDOM_SEED}，实际为 "
            f"{artifact_seed}: {artifact_path}"
        )

    required_keys = (
        "input_cols",
        "target_index",
        "scaler_x",
        "scaler_y",
        "history_len",
        "forecast_len",
    )
    missing_keys = [key for key in required_keys if key not in artifact]
    if missing_keys:
        raise KeyError(f"artifact 缺少预测必需字段 {missing_keys}: {artifact_path}")

    artifact = dict(artifact)
    artifact["artifact_path"] = os.path.abspath(artifact_path)
    artifact.setdefault("model_family", MODEL_FAMILY)
    artifact.setdefault("model_name", variant_model_name(variant_id))
    artifact.setdefault("model_variant", variant_id)
    artifact.setdefault("requires_keras_model", True)
    artifact.setdefault("model_kind", "keras")
    artifact.setdefault("router_type", "none")
    artifact.setdefault("router_layer_name", None)
    artifact.setdefault("expert_names", [])
    return artifact


def get_min_custom_objects():
    """合并既有自定义层，并显式登记 min 静态 horizon router。"""
    custom_objects = dict(common_predict.get_custom_objects())
    custom_objects["StaticHorizonRouter"] = StaticHorizonRouter
    custom_objects["WindFeTSPatchTSTMin>StaticHorizonRouter"] = StaticHorizonRouter
    try:
        registered_name = keras.saving.get_registered_name(
            StaticHorizonRouter,
        )
    except AttributeError:
        registered_name = None
    if registered_name:
        custom_objects[registered_name] = StaticHorizonRouter
    return custom_objects


def _existing_path(path, variant_id):
    if not path:
        return None
    path = os.fspath(path)
    candidates = [path]
    if not os.path.isabs(path):
        project_root = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(project_root, path))
        candidates.append(
            os.path.join(
                _variant_root(variant_id),
                os.path.basename(path),
            )
        )
    return next(
        (candidate for candidate in candidates if os.path.exists(candidate)),
        None,
    )


def _default_model_paths(variant_id, farm_id):
    paths = variant_dirs(variant_id, create=False)
    model_dir = _first_path(
        paths,
        "models",
        "saved_models",
        "saved_model_dir",
        "models_dir",
    ) or os.path.join(_variant_root(variant_id, paths), "models")
    weights_dir = _first_path(
        paths,
        "weights",
        "weights_dir",
    ) or os.path.join(_variant_root(variant_id, paths), "weights")
    model_name = variant_model_name(variant_id)
    return (
        os.path.join(model_dir, f"{model_name}_farm_{farm_id}.keras"),
        os.path.join(
            weights_dir,
            f"{model_name}_farm_{farm_id}_best.weights.h5",
        ),
    )


def load_variant_model(variant_id, farm_id, artifact):
    """优先加载完整模型，缺失时按 artifact 精确重建并加载权重。"""
    if not bool(artifact.get("requires_keras_model", True)):
        if artifact.get("model_kind") != "analytic_persistence":
            raise ValueError(
                f"{variant_id} 声明无需 Keras 模型，但 model_kind 不是 "
                f"analytic_persistence: {artifact.get('model_kind')}"
            )
        return None, "analytic_persistence"

    default_model_path, default_weights_path = _default_model_paths(
        variant_id,
        farm_id,
    )
    requested_model_path = artifact.get("model_path", default_model_path)
    requested_weights_path = artifact.get(
        "best_weights_path",
        default_weights_path,
    )
    model_path = _existing_path(requested_model_path, variant_id)
    weights_path = _existing_path(requested_weights_path, variant_id)

    if model_path:
        model = keras.models.load_model(
            model_path,
            custom_objects=get_min_custom_objects(),
            compile=False,
        )
        return model, os.path.abspath(model_path)

    if not weights_path:
        raise FileNotFoundError(
            f"未找到变体 {variant_id} 场站 {farm_id} 的完整模型或权重: "
            f"{requested_model_path}, {requested_weights_path}"
        )

    model = build_fets_patchtst_min_model_from_artifact(artifact)
    model.load_weights(weights_path)
    return model, os.path.abspath(weights_path)


def _prediction_sample_count(features, history_len, forecast_len):
    n_samples = len(features) - history_len - forecast_len + 1
    if n_samples <= 0:
        raise ValueError("测试集长度不足，无法构造完整历史窗口和预测窗口")
    return int(n_samples)


def analytic_persistence_prediction(features, artifact, n_samples):
    """在 scaler_y 空间生成解析式 persistence 预测。"""
    history_len = int(artifact["history_len"])
    forecast_len = int(artifact["forecast_len"])
    target_index = int(artifact["target_index"])
    if target_index < 0 or target_index >= features.shape[1]:
        raise IndexError(f"历史功率通道索引越界: {target_index} vs {features.shape[1]}")

    last_power_x_scaled = features[
        history_len - 1 : history_len - 1 + n_samples,
        target_index,
    ]
    if len(last_power_x_scaled) != n_samples:
        raise ValueError("解析式 persistence 样本数与预期不一致")

    scale_ratio = float(artifact.get("power_scale_ratio", 1.0))
    scale_offset = float(artifact.get("power_scale_offset", 0.0))
    y_scaled = last_power_x_scaled * scale_ratio + scale_offset
    predictions = np.repeat(
        y_scaled[:, np.newaxis],
        repeats=forecast_len,
        axis=1,
    ).astype(np.float32)
    if not np.isfinite(predictions).all():
        raise FloatingPointError("解析式 persistence 预测包含非有限值")
    return predictions


def predict_keras_with_optional_router(model, pred_ds, artifact):
    """单次前向返回预测，以及 artifact 指定的可选 router 权重。"""
    router_type = str(artifact.get("router_type", "none")).lower()
    router_layer_name = artifact.get("router_layer_name")
    if router_type in {"none", "", "null"}:
        if router_layer_name:
            raise ValueError("router_type=none 时 router_layer_name 必须为空")
        predictions = model.predict(
            pred_ds,
            verbose=common_predict.PREDICT_VERBOSE,
        )
        return np.asarray(predictions), None
    if not router_layer_name:
        raise ValueError(
            f"router_type={router_type}，但 artifact 未提供 router_layer_name"
        )

    try:
        router_layer = model.get_layer(router_layer_name)
    except ValueError as exc:
        raise ValueError(
            f"模型中不存在 artifact 指定的 router 层 {router_layer_name!r}"
        ) from exc

    diagnostic_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.output, router_layer.output],
        name="WindFeTSPatchTSTMinPredictDiagnostics",
    )
    predictions, router_weights = diagnostic_model.predict(
        pred_ds,
        verbose=common_predict.PREDICT_VERBOSE,
    )
    return (
        np.asarray(predictions),
        np.asarray(router_weights, dtype=float),
    )


def _parameter_counts(model, artifact):
    if model is None:
        return 0, 0
    parameter_count = int(model.count_params())
    trainable_parameter_count = int(
        sum(int(np.prod(tuple(variable.shape))) for variable in model.trainable_weights)
    )
    artifact_count = artifact.get(
        "parameter_count",
        artifact.get(
            "total_parameter_count",
            artifact.get("total_params"),
        ),
    )
    if artifact_count is not None and int(artifact_count) != parameter_count:
        raise ValueError(
            f"artifact 参数量 {artifact_count} 与加载模型 {parameter_count} 不一致"
        )
    return parameter_count, trainable_parameter_count


def _validate_router_output(router_weights, artifact, n_samples, forecast_len):
    if router_weights is None:
        return []
    if router_weights.ndim != 3 or router_weights.shape[:2] != (
        n_samples,
        forecast_len,
    ):
        raise ValueError(
            f"router 输出形状异常: {router_weights.shape}，期望 "
            f"({n_samples}, {forecast_len}, 专家数)"
        )
    expert_names = list(artifact.get("expert_names") or [])
    if len(expert_names) != router_weights.shape[-1]:
        raise ValueError(
            f"artifact expert_names 数量 {len(expert_names)} 与 router "
            f"输出 {router_weights.shape[-1]} 不一致"
        )
    if len(expert_names) <= 1:
        raise ValueError("单专家变体不应配置 router 或计算归一化路由熵")
    return expert_names


def predict_one_variant_farm(variant_id, test_file):
    farm_id = common_predict.get_farm_id(test_file)
    model_name = variant_model_name(variant_id)
    dirs = prediction_output_dirs(variant_id)
    print(f"\n===== 预测 {model_name} / 风电场 {farm_id} =====")

    artifact = load_variant_artifact(variant_id, farm_id)
    df, features, actual_power, capacity = common_predict.prepare_prediction_arrays(
        test_file,
        artifact,
    )
    history_len = int(artifact["history_len"])
    forecast_len = int(artifact["forecast_len"])
    n_samples = _prediction_sample_count(
        features,
        history_len,
        forecast_len,
    )

    model, loaded_model_path = load_variant_model(
        variant_id,
        farm_id,
        artifact,
    )
    if model is None:
        y_pred_scaled = analytic_persistence_prediction(
            features,
            artifact,
            n_samples,
        )
        router_weights = None
    else:
        pred_ds, dataset_n_samples = common_predict.make_prediction_dataset(
            features,
            history_len,
            forecast_len,
        )
        if dataset_n_samples != n_samples:
            raise ValueError(
                f"预测数据集样本数不一致: {dataset_n_samples} vs {n_samples}"
            )
        y_pred_scaled, router_weights = predict_keras_with_optional_router(
            model,
            pred_ds,
            artifact,
        )

    y_pred = common_predict.inverse_power(
        artifact["scaler_y"],
        y_pred_scaled,
    ).reshape(-1, forecast_len)
    if y_pred.shape != (n_samples, forecast_len):
        raise ValueError(
            f"{model_name} 场站 {farm_id} 预测形状不一致: "
            f"{y_pred.shape} vs ({n_samples}, {forecast_len})"
        )
    expert_names = _validate_router_output(
        router_weights,
        artifact,
        n_samples,
        forecast_len,
    )

    if capacity is not None:
        y_pred = np.clip(y_pred, 0, capacity)
    else:
        y_pred = np.clip(y_pred, 0, None)

    y_true = common_predict.build_truth_windows(
        actual_power,
        n_samples,
        history_len,
        forecast_len,
    )
    pred_df = common_predict.build_prediction_frame(
        model_name,
        df,
        farm_id,
        y_pred,
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
        y_pred,
        capacity,
        forecast_len,
    )
    parameter_count, trainable_parameter_count = _parameter_counts(
        model,
        artifact,
    )
    metric_df["model_family"] = MODEL_FAMILY
    metric_df["model_variant"] = variant_id
    metric_df["parameter_count"] = parameter_count
    horizon_metric_path = os.path.join(
        dirs["root"],
        f"{model_name}_metrics_by_horizon_farm_{farm_id}.csv",
    )
    metric_df.to_csv(
        horizon_metric_path,
        index=False,
        encoding="utf-8-sig",
    )

    _, router_metric_fields = common_predict.save_router_diagnostics(
        router_weights,
        expert_names,
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

    all_metrics = metric_df[metric_df["horizon_step"] == "all"].iloc[0].to_dict()
    all_metrics.update(
        {
            "model_family": MODEL_FAMILY,
            "model_variant": variant_id,
            "architecture_version": ARCHITECTURE_VERSION,
            "random_seed": RANDOM_SEED,
            "model_kind": artifact.get("model_kind"),
            "router_type": artifact.get("router_type"),
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "loaded_model_path": loaded_model_path,
            "artifact_path": artifact["artifact_path"],
            "prediction_path": prediction_path,
            "horizon_metric_path": horizon_metric_path,
            "single_window_path": single_window_path,
            "single_window_figure_path": single_window_figure_path,
            "weighted_curve_path": weighted_curve_path,
            "weighted_curve_figure_path": weighted_curve_figure_path,
            **router_metric_fields,
            **{
                f"weighted_curve_{key}": value
                for key, value in weighted_metrics.items()
            },
        }
    )
    print(
        f"{model_name} 场站 {farm_id}: MAE={all_metrics['mae']:.4f}, "
        f"RMSE={all_metrics['rmse']:.4f}, 参数量={parameter_count}"
    )

    if model is not None:
        del model
    keras.backend.clear_session()
    return all_metrics, metric_df


def predict_variant(variant_id, test_files):
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"未知变体 {variant_id!r}；可选: {list(VARIANT_SPECS)}")
    dirs = prediction_output_dirs(variant_id)
    model_name = variant_model_name(variant_id)
    summary_rows = []
    horizon_frames = []

    for test_file in test_files:
        try:
            metrics, horizon_metrics = predict_one_variant_farm(
                variant_id,
                test_file,
            )
        except FileNotFoundError as exc:
            print(f"跳过 {model_name} {os.path.basename(test_file)}: {exc}")
            continue
        summary_rows.append(metrics)
        horizon_frames.append(horizon_metrics)

    if not summary_rows:
        print(f"{model_name} 没有生成预测结果")
        return pd.DataFrame(), pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(
        dirs["root"],
        f"{model_name}_test_metrics_summary.csv",
    )
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    horizon_df = pd.concat(horizon_frames, ignore_index=True)
    horizon_path = os.path.join(
        dirs["root"],
        f"{model_name}_test_metrics_by_horizon_all.csv",
    )
    horizon_df.to_csv(horizon_path, index=False, encoding="utf-8-sig")
    print(f"{model_name} 汇总指标已保存: {summary_path}")
    print(f"{model_name} 分 horizon 指标已保存: {horizon_path}")
    return summary_df, horizon_df


def build_variant_comparison(summary_df, requested_variants):
    metric_columns = (
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
        variant_df = summary_df[summary_df["model_variant"] == variant_id].copy()
        row = {
            "variant_order": order,
            "model_variant": variant_id,
            "model_name": variant_model_name(variant_id),
            "farm_count": int(variant_df["farm_id"].nunique()),
        }
        parameter_values = pd.to_numeric(
            variant_df.get("parameter_count"),
            errors="coerce",
        ).dropna()
        row["parameter_count_min"] = (
            int(parameter_values.min()) if not parameter_values.empty else np.nan
        )
        row["parameter_count_max"] = (
            int(parameter_values.max()) if not parameter_values.empty else np.nan
        )
        for column in metric_columns:
            if column not in variant_df:
                row[f"macro_mean_{column}"] = np.nan
                row[f"macro_std_{column}"] = np.nan
                continue
            values = pd.to_numeric(variant_df[column], errors="coerce")
            row[f"macro_mean_{column}"] = float(values.mean())
            row[f"macro_std_{column}"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def build_minimal_effective_selection(
    summary_df,
    comparison_df,
    requested_variants,
):
    """按 B6 相对容差与跨场站一致性选择参数量最小的有效结构。"""
    reference_df = summary_df[summary_df["model_variant"] == REFERENCE_VARIANT_ID][
        ["farm_id", "capacity_normalized_rmse"]
    ].copy()
    reference_df["farm_id"] = reference_df["farm_id"].astype(str)
    reference_df["reference_nrmse"] = pd.to_numeric(
        reference_df["capacity_normalized_rmse"],
        errors="coerce",
    )
    reference_df = reference_df[np.isfinite(reference_df["reference_nrmse"])][
        ["farm_id", "reference_nrmse"]
    ]

    rows = []
    for order, variant_id in enumerate(requested_variants):
        variant_df = summary_df[summary_df["model_variant"] == variant_id][
            ["farm_id", "capacity_normalized_rmse"]
        ].copy()
        variant_df["farm_id"] = variant_df["farm_id"].astype(str)
        variant_df["variant_nrmse"] = pd.to_numeric(
            variant_df["capacity_normalized_rmse"],
            errors="coerce",
        )
        variant_df = variant_df[np.isfinite(variant_df["variant_nrmse"])][
            ["farm_id", "variant_nrmse"]
        ]
        paired = reference_df.merge(variant_df, on="farm_id", how="inner")

        common_farm_count = len(paired)
        reference_macro_nrmse = (
            float(paired["reference_nrmse"].mean()) if common_farm_count else np.nan
        )
        variant_macro_nrmse = (
            float(paired["variant_nrmse"].mean()) if common_farm_count else np.nan
        )
        if reference_macro_nrmse > 0 and np.isfinite(variant_macro_nrmse):
            macro_relative_gap = variant_macro_nrmse / reference_macro_nrmse - 1.0
        else:
            macro_relative_gap = np.nan
        farms_within_tolerance = (
            int(
                (
                    paired["variant_nrmse"]
                    <= paired["reference_nrmse"] * (1.0 + FARM_NRMSE_TOLERANCE)
                ).sum()
            )
            if common_farm_count
            else 0
        )

        macro_pass = bool(
            np.isfinite(macro_relative_gap)
            and macro_relative_gap <= MACRO_NRMSE_TOLERANCE + 1e-12
        )
        farm_pass = bool(
            common_farm_count >= MIN_NON_INFERIOR_FARMS
            and farms_within_tolerance >= MIN_NON_INFERIOR_FARMS
        )
        comparison_row = comparison_df[comparison_df["model_variant"] == variant_id]
        if comparison_row.empty:
            parameter_count = np.nan
        else:
            parameter_count = comparison_row.iloc[0]["parameter_count_max"]
        eligible = bool(macro_pass and farm_pass and np.isfinite(parameter_count))
        rows.append(
            {
                "variant_order": order,
                "model_variant": variant_id,
                "model_name": variant_model_name(variant_id),
                "parameter_count": parameter_count,
                "reference_variant": REFERENCE_VARIANT_ID,
                "common_farm_count": common_farm_count,
                "reference_macro_nrmse": reference_macro_nrmse,
                "variant_macro_nrmse": variant_macro_nrmse,
                "macro_relative_gap_pct": (
                    macro_relative_gap * 100.0
                    if np.isfinite(macro_relative_gap)
                    else np.nan
                ),
                "macro_within_0_5pct": macro_pass,
                "farms_within_1pct": farms_within_tolerance,
                "at_least_4_farms_within_1pct": farm_pass,
                "eligible_minimal_effective": eligible,
                "selected_minimal_effective": False,
            }
        )

    selection_df = pd.DataFrame(rows)
    eligible_df = selection_df[selection_df["eligible_minimal_effective"]].copy()
    if not eligible_df.empty:
        eligible_df = eligible_df.sort_values(
            [
                "parameter_count",
                "variant_macro_nrmse",
                "variant_order",
            ],
            kind="stable",
        )
        selected_index = eligible_df.index[0]
        selection_df.loc[
            selected_index,
            "selected_minimal_effective",
        ] = True
    return selection_df


def _format_markdown_value(value):
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return "是" if value else "否"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value).replace("|", "\\|")


def save_selection_markdown(selection_df, path):
    selected = selection_df[selection_df["selected_minimal_effective"]]
    selected_name = selected.iloc[0]["model_variant"] if not selected.empty else "无"
    columns = [
        "model_variant",
        "parameter_count",
        "common_farm_count",
        "variant_macro_nrmse",
        "macro_relative_gap_pct",
        "farms_within_1pct",
        "eligible_minimal_effective",
        "selected_minimal_effective",
    ]
    labels = [
        "变体",
        "参数量",
        "共同场站数",
        "宏平均 NRMSE",
        "相对 B6 差值(%)",
        "不劣于1%的场站数",
        "满足条件",
        "最终选择",
    ]
    lines = [
        "# FeTS-PatchTST 最小有效结构选择",
        "",
        f"- 参考变体：`{REFERENCE_VARIANT_ID}`",
        "- 宏平均 NRMSE 容差：相对参考不超过 0.5%",
        "- 跨场站条件：至少 4 个场站的 NRMSE 不劣于参考 1%",
        "- 选择规则：满足上述条件后，优先选择最大场站参数量最小的变体",
        f"- 最终选择：`{selected_name}`",
        "",
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] * len(labels)) + " |",
    ]
    for _, row in selection_df.iterrows():
        lines.append(
            "| "
            + " | ".join(_format_markdown_value(row[column]) for column in columns)
            + " |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def save_parameter_nrmse_pareto(comparison_df, selection_df, output_dir):
    """绘制参数量—五站宏平均 NRMSE，用于识别最小有效结构。"""
    figure_path = os.path.join(
        output_dir,
        f"{MODEL_FAMILY}_parameter_nrmse_pareto.png",
    )
    plot_df = comparison_df[
        [
            "model_variant",
            "parameter_count_max",
            "macro_mean_capacity_normalized_rmse",
        ]
    ].copy()
    plot_df["parameter_count_max"] = pd.to_numeric(
        plot_df["parameter_count_max"],
        errors="coerce",
    )
    plot_df["macro_mean_capacity_normalized_rmse"] = pd.to_numeric(
        plot_df["macro_mean_capacity_normalized_rmse"],
        errors="coerce",
    )
    plot_df = plot_df.dropna()
    if plot_df.empty:
        return None
    selected_ids = set(
        selection_df.loc[
            selection_df["selected_minimal_effective"],
            "model_variant",
        ]
    )
    try:
        cache_dir = os.path.join(output_dir, "matplotlib_cache")
        os.environ["MPLCONFIGDIR"] = cache_dir
        os.environ["XDG_CACHE_HOME"] = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        for _, row in plot_df.iterrows():
            selected = row["model_variant"] in selected_ids
            ax.scatter(
                row["parameter_count_max"],
                row["macro_mean_capacity_normalized_rmse"],
                s=130 if selected else 65,
                marker="*" if selected else "o",
                color="tab:red" if selected else "tab:blue",
                zorder=3,
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
        ax.set_title("FeTS-PatchTST Stage-1 Complexity-Accuracy Comparison")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"参数量—NRMSE Pareto 图保存失败: {exc}")
        return None
    return figure_path


def save_cross_variant_outputs(
    all_summary,
    all_horizon,
    requested_variants,
):
    output_dir = comparison_output_dir()
    summary_df = pd.concat(all_summary, ignore_index=True)
    horizon_df = pd.concat(all_horizon, ignore_index=True)
    summary_path = os.path.join(
        output_dir,
        f"{MODEL_FAMILY}_test_metrics_summary.csv",
    )
    horizon_path = os.path.join(
        output_dir,
        f"{MODEL_FAMILY}_test_metrics_by_horizon_all.csv",
    )
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    horizon_df.to_csv(horizon_path, index=False, encoding="utf-8-sig")

    comparison_df = build_variant_comparison(
        summary_df,
        requested_variants,
    )
    comparison_path = os.path.join(
        output_dir,
        f"{MODEL_FAMILY}_variant_comparison.csv",
    )
    comparison_df.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    selection_df = build_minimal_effective_selection(
        summary_df,
        comparison_df,
        requested_variants,
    )
    selection_path = os.path.join(
        output_dir,
        f"{MODEL_FAMILY}_minimal_effective_selection.csv",
    )
    selection_markdown_path = os.path.join(
        output_dir,
        f"{MODEL_FAMILY}_minimal_effective_selection.md",
    )
    selection_df.to_csv(
        selection_path,
        index=False,
        encoding="utf-8-sig",
    )
    save_selection_markdown(selection_df, selection_markdown_path)
    pareto_figure_path = save_parameter_nrmse_pareto(
        comparison_df,
        selection_df,
        output_dir,
    )
    return {
        "summary_path": summary_path,
        "horizon_path": horizon_path,
        "comparison_path": comparison_path,
        "selection_path": selection_path,
        "selection_markdown_path": selection_markdown_path,
        "pareto_figure_path": pareto_figure_path,
    }


def main():
    configure_prediction_reproducibility()
    test_files = discover_requested_test_files()
    if not test_files:
        raise FileNotFoundError(
            f"未找到测试文件模式 {common_predict.TEST_FILE_PATTERN}"
        )

    requested_variants = list(get_requested_variants())
    invalid_variants = [
        variant_id
        for variant_id in requested_variants
        if variant_id not in VARIANT_SPECS
    ]
    if invalid_variants:
        raise ValueError(f"未知变体 {invalid_variants}；可选: {list(VARIANT_SPECS)}")
    if not requested_variants:
        raise ValueError("没有请求任何 FeTS-PatchTST min 变体")

    print(f"发现 {len(test_files)} 个风电测试文件")
    print(f"将预测变体: {requested_variants}")
    all_summary = []
    all_horizon = []
    completed_variants = []
    for variant_id in requested_variants:
        summary_df, horizon_df = predict_variant(variant_id, test_files)
        if summary_df.empty or horizon_df.empty:
            continue
        all_summary.append(summary_df)
        all_horizon.append(horizon_df)
        completed_variants.append(variant_id)

    if not all_summary:
        print("没有变体生成预测结果，未创建跨变体汇总")
        return

    output_paths = save_cross_variant_outputs(
        all_summary,
        all_horizon,
        completed_variants,
    )
    print("跨变体汇总已保存:")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")
    if REFERENCE_VARIANT_ID not in completed_variants:
        print(
            f"警告：本次结果不含参考变体 {REFERENCE_VARIANT_ID}，"
            "选择报告不会选出最小有效结构"
        )
    print("FeTS-PatchTST 第一阶段最小结构测试集预测完成")


if __name__ == "__main__":
    main()
