"""Backfill and audit visualizations for the completed time-frequency experiment.

This is deliberately a post-processing program.  The completed training and
prediction bundle markers contain SHA256 hashes of the original programs, so
adding plotting code to either program after a formal run would invalidate the
archive.  This module therefore only reads frozen CSV/JSON artifacts and writes
new PNG files plus its own independent manifest.

Generated training plots
------------------------
* T1--T3 corrected-candidate history: loss, MAE and RMSE (train/validation).
* M0--T3 gate history: loss, forecast MAE and forecast RMSE
  (train/validation), with the three gate-training phases marked.

Generated prediction diagnostics
--------------------------------
* M0--T3 per-farm gate-by-regime/horizon heatmaps and reliability curves.
* Aggregate gate reliability and accuracy-safety plots omitted by the formal
  prediction program.
* Aggregate corrected-candidate horizon and dynamic/ramp regime views needed
  to explain why candidate gains did or did not survive the final fusion.

T0 is a direct Stage-3 G0 reference and was not retrained or re-predicted in
this experiment.  Its original prediction figures are validated and referenced
in the manifest rather than copied.  Likewise M0's frozen F7 candidate has no
training trajectory; a fake one-point curve is never produced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parent
RESULT_ROOT = WORKSPACE / "wind_results" / "time_freq_model"
TRAINING_MARKER = RESULT_ROOT / "time_freq_model_training_bundle_complete.json"
PREDICTION_ROOT = RESULT_ROOT / "testdata_predict_output"
PREDICTION_MARKER = PREDICTION_ROOT / "time_freq_model_test_bundle_complete.json"
MANIFEST_PATH = RESULT_ROOT / "time_freq_model_visualization_backfill_manifest.json"
INVENTORY_PATH = RESULT_ROOT / "time_freq_model_visualization_backfill_inventory.csv"
MATPLOTLIB_CACHE = RESULT_ROOT / "matplotlib_cache_visualization"

PROTOCOL_VERSION = "time_freq_visualization_backfill_v1"
ALL_VARIANTS = ("t0", "m0", "t1", "t2", "t3")
TRAINED_VARIANTS = ("m0", "t1", "t2", "t3")
CANDIDATE_TRAINED_VARIANTS = ("t1", "t2", "t3")
REGIME_ORDER = ("stable", "ramp_up", "ramp_down", "low_power")
GATE_PHASE_ORDER = ("gate_only", "context", "calibrated_gate")
FORECAST_STEPS = tuple(range(1, 17))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(WORKSPACE).as_posix()
    except ValueError:
        return str(resolved)


def load_json_strict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON文件不存在: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON顶层必须为对象: {path}")
    return payload


def descriptor_path(descriptor: dict[str, Any]) -> Path:
    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"归档descriptor缺少有效path: {descriptor}")
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return WORKSPACE / path


def verify_descriptor(
    descriptor: dict[str, Any], label: str, *, require_png: bool = False
) -> Path:
    path = descriptor_path(descriptor)
    if not path.is_file():
        raise FileNotFoundError(f"归档成员不存在 [{label}]: {path}")
    expected_size = descriptor.get("size_bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise ValueError(
            f"归档成员大小不一致 [{label}]: "
            f"expected={expected_size}, actual={path.stat().st_size}, path={path}"
        )
    expected_hash = descriptor.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"归档descriptor缺少有效SHA256 [{label}]")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"归档成员SHA256不一致 [{label}]: "
            f"expected={expected_hash}, actual={actual_hash}, path={path}"
        )
    if require_png:
        validate_png(path)
    return path


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label}缺少列: {missing}")


def require_finite(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{label}包含空值或非有限数值")


def setup_matplotlib():
    MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CACHE)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    return plt


def validate_png(path: Path) -> tuple[int, int]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"PNG不存在或为空: {path}")
    from PIL import Image

    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        if image.format != "PNG" or width < 400 or height < 250:
            raise ValueError(
                f"PNG格式或尺寸异常: format={image.format}, "
                f"size={width}x{height}, path={path}"
            )
    return int(width), int(height)


def atomic_save_figure(plt, figure, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.stem}.",
            suffix=".tmp.png",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        figure.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        width, height = validate_png(temporary)
        os.replace(temporary, output_path)
        temporary = None
    finally:
        plt.close(figure)
        if temporary is not None and temporary.exists():
            temporary.unlink()
    width, height = validate_png(output_path)
    return {
        "path": relative_path(output_path),
        "sha256": sha256_file(output_path),
        "size_bytes": output_path.stat().st_size,
        "width": width,
        "height": height,
    }


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.stem}.",
            suffix=".tmp.json",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_inventory(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "category",
        "plot_kind",
        "variant",
        "farm_id",
        "status",
        "output_path",
        "output_sha256",
        "size_bytes",
        "width",
        "height",
        "source_paths",
        "source_sha256",
        "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            prefix=f".{path.stem}.",
            suffix=".tmp.csv",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                sources = row.get("sources", [])
                output = row.get("output", {})
                writer.writerow(
                    {
                        "category": row.get("category", ""),
                        "plot_kind": row.get("plot_kind", ""),
                        "variant": row.get("variant", ""),
                        "farm_id": row.get("farm_id", ""),
                        "status": row.get("status", ""),
                        "output_path": output.get("path", row.get("reference_path", "")),
                        "output_sha256": output.get(
                            "sha256", row.get("reference_sha256", "")
                        ),
                        "size_bytes": output.get(
                            "size_bytes", row.get("reference_size_bytes", "")
                        ),
                        "width": output.get("width", row.get("reference_width", "")),
                        "height": output.get("height", row.get("reference_height", "")),
                        "source_paths": "|".join(
                            str(source.get("path", "")) for source in sources
                        ),
                        "source_sha256": "|".join(
                            str(source.get("sha256", "")) for source in sources
                        ),
                        "note": row.get("note", ""),
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def source_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def plot_candidate_history(plt, frame: pd.DataFrame, variant: str, farm_id: str):
    x = frame["epoch"].to_numpy(dtype=int) + 1
    panels = (
        ("loss", "val_loss", "Loss"),
        ("mae", "val_mae", "MAE"),
        ("rmse", "val_rmse", "RMSE"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))
    for axis, (train_column, val_column, title) in zip(axes, panels):
        axis.plot(x, frame[train_column], marker="o", label="Train", color="#4c78a8")
        axis.plot(
            x,
            frame[val_column],
            marker="o",
            label="Validation",
            color="#f58518",
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.legend()
    figure.suptitle(
        f"{variant.upper()} corrected-candidate training — Farm {farm_id}", y=1.02
    )
    figure.tight_layout()
    return figure


def plot_gate_history(plt, frame: pd.DataFrame, variant: str, farm_id: str):
    x = frame["global_epoch"].to_numpy(dtype=int) + 1
    panels = (
        ("loss", "val_loss", "Composite loss"),
        ("forecast_power_mae", "val_forecast_power_mae", "Forecast MAE"),
        ("forecast_power_rmse", "val_forecast_power_rmse", "Forecast RMSE"),
    )
    colors = {
        "gate_only": "#e8f1fa",
        "context": "#fff2df",
        "calibrated_gate": "#e8f5e9",
    }
    phase_ranges: list[tuple[str, float, float]] = []
    for phase in GATE_PHASE_ORDER:
        positions = x[frame["phase"].to_numpy(str) == phase]
        phase_ranges.append((phase, float(positions.min()) - 0.5, float(positions.max()) + 0.5))

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))
    for axis, (train_column, val_column, title) in zip(axes, panels):
        for phase, start, stop in phase_ranges:
            axis.axvspan(start, stop, color=colors[phase], alpha=0.55, zorder=0)
        axis.plot(x, frame[train_column], label="Train", color="#4c78a8", zorder=2)
        axis.plot(x, frame[val_column], label="Validation", color="#f58518", zorder=2)
        for _, start, _ in phase_ranges[1:]:
            axis.axvline(start, color="#666666", linestyle="--", linewidth=0.8)
        axis.set_xlabel("Global epoch")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.legend()
    for phase, start, stop in phase_ranges:
        axes[0].text(
            (start + stop) / 2,
            0.97,
            phase,
            ha="center",
            va="top",
            fontsize=8,
            color="#444444",
            transform=axes[0].get_xaxis_transform(),
        )
    figure.suptitle(f"{variant.upper()} calibrated-gate training — Farm {farm_id}", y=1.02)
    figure.tight_layout()
    return figure


def plot_gate_by_regime(plt, matrix: pd.DataFrame, variant: str, farm_id: str):
    figure, axis = plt.subplots(figsize=(11.8, 4.3))
    image = axis.imshow(
        matrix.to_numpy(float), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis"
    )
    axis.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    axis.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns)
    axis.set_xlabel("Forecast horizon (15-min steps)")
    axis.set_ylabel("Realized regime")
    axis.set_title(f"{variant.upper()} applied corrected-candidate gate — Farm {farm_id}")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix.iloc[row, column])
            color = "white" if value < 0.35 or value > 0.70 else "black"
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)
    figure.colorbar(image, ax=axis, label="Mean applied gate")
    figure.tight_layout()
    return figure


def plot_farm_reliability(
    plt, calibration: pd.DataFrame, variant: str, farm_id: str
):
    valid = calibration[
        (calibration["count"] > 0)
        & calibration["mean_raw_gate"].notna()
        & calibration["corrected_better_rate"].notna()
    ].copy()
    figure, axis = plt.subplots(figsize=(6.2, 5.8))
    axis.plot([0, 1], [0, 1], "--", color="#555555", linewidth=1.0, label="Ideal")
    axis.plot(
        valid["mean_raw_gate"],
        valid["corrected_better_rate"],
        marker="o",
        linewidth=1.8,
        color="#4c78a8",
        label="Observed",
    )
    for _, row in valid.iterrows():
        axis.annotate(
            f"n={int(row['count'])}",
            (row["mean_raw_gate"], row["corrected_better_rate"]),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean raw gate")
    axis.set_ylabel("Corrected-candidate better frequency")
    axis.set_title(f"{variant.upper()} finite-masked gate reliability — Farm {farm_id}")
    axis.legend()
    figure.tight_layout()
    return figure


def aggregate_calibration(calibration: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, gate_bin), group in calibration.groupby(
        ["model_variant", "gate_bin"], sort=False
    ):
        valid = group[
            (group["count"] > 0)
            & group["mean_raw_gate"].notna()
            & group["corrected_better_rate"].notna()
        ]
        weights = valid["count"].to_numpy(float)
        count = float(weights.sum())
        rows.append(
            {
                "model_variant": str(variant),
                "gate_bin": int(gate_bin),
                "count": int(count),
                "mean_gate": (
                    float(np.average(valid["mean_raw_gate"].to_numpy(float), weights=weights))
                    if count
                    else np.nan
                ),
                "observed": (
                    float(
                        np.average(
                            valid["corrected_better_rate"].to_numpy(float), weights=weights
                        )
                    )
                    if count
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_aggregate_reliability(plt, calibration: pd.DataFrame):
    aggregated = aggregate_calibration(calibration)
    figure, axis = plt.subplots(figsize=(7.0, 6.2))
    axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1.0, label="Ideal")
    for variant in ALL_VARIANTS:
        frame = aggregated[
            (aggregated["model_variant"] == variant)
            & aggregated["mean_gate"].notna()
            & aggregated["observed"].notna()
        ].sort_values("gate_bin")
        axis.plot(frame["mean_gate"], frame["observed"], marker="o", label=variant.upper())
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Count-weighted mean raw gate")
    axis.set_ylabel("Corrected-candidate better frequency")
    axis.set_title("Five-farm finite-masked gate reliability (test set)")
    axis.legend(ncol=2)
    figure.tight_layout()
    return figure


def plot_safety_accuracy(plt, comparison: pd.DataFrame):
    figure, axis = plt.subplots(figsize=(7.2, 5.5))
    colors = ["#d62728" if bool(value) else "#4c78a8" for value in comparison["selected"]]
    axis.scatter(
        comparison["macro_test_nrmse"],
        comparison["macro_positive_regret_mean"],
        s=85,
        c=colors,
    )
    annotation_offsets = {
        "t0": (6, 6),
        "m0": (-30, 12),
        "t1": (8, 10),
        "t2": (-30, -18),
        "t3": (8, -18),
    }
    for _, row in comparison.iterrows():
        variant = str(row["model_variant"])
        axis.annotate(
            variant.upper(),
            (row["macro_test_nrmse"], row["macro_positive_regret_mean"]),
            xytext=annotation_offsets[variant],
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.7},
        )
    axis.set_xlabel("Five-farm macro test NRMSE (lower is better)")
    axis.set_ylabel("Mean positive regret (lower is safer)")
    axis.set_title("Accuracy-safety trade-off (red = selected)")
    figure.tight_layout()
    return figure


def plot_corrected_candidate_horizon(plt, candidate: pd.DataFrame):
    frame = candidate[
        (candidate["candidate"] == "corrected")
        & (candidate["horizon_step"].astype(str) != "all")
    ].copy()
    frame["horizon_step_numeric"] = pd.to_numeric(frame["horizon_step"], errors="raise")
    macro = (
        frame.groupby(["model_variant", "horizon_step_numeric"], as_index=False)[
            "capacity_normalized_rmse"
        ]
        .mean()
        .sort_values(["model_variant", "horizon_step_numeric"])
    )
    figure, axis = plt.subplots(figsize=(8.6, 5.0))
    for variant in ALL_VARIANTS:
        part = macro[macro["model_variant"] == variant]
        if tuple(part["horizon_step_numeric"].astype(int)) != FORECAST_STEPS:
            raise ValueError(f"{variant} corrected candidate逐horizon矩阵不完整")
        axis.plot(
            part["horizon_step_numeric"],
            part["capacity_normalized_rmse"],
            marker="o",
            markersize=3,
            label=variant.upper(),
        )
    axis.set_xticks(FORECAST_STEPS)
    axis.set_xlabel("Forecast horizon (15-min steps)")
    axis.set_ylabel("Five-farm macro NRMSE")
    axis.set_title("Corrected-candidate horizon error (test set)")
    axis.legend(ncol=3)
    figure.tight_layout()
    return figure


def plot_regime_nrmse(plt, regime: pd.DataFrame):
    regimes = ("dynamic", "ramp_up", "ramp_down")
    frame = regime[
        regime["regime_group"].isin(regimes)
        & regime["candidate"].isin(("corrected", "fused"))
        & (regime["horizon_step"].astype(str) == "all")
    ].copy()
    macro = frame.groupby(
        ["regime_group", "candidate", "model_variant"], as_index=False
    )["capacity_normalized_rmse"].mean()
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), sharey=False)
    x = np.arange(len(ALL_VARIANTS), dtype=float)
    width = 0.36
    for axis, regime_name in zip(axes, regimes):
        selected = macro[macro["regime_group"] == regime_name]
        corrected = selected[selected["candidate"] == "corrected"].set_index(
            "model_variant"
        )["capacity_normalized_rmse"].reindex(ALL_VARIANTS)
        fused = selected[selected["candidate"] == "fused"].set_index("model_variant")[
            "capacity_normalized_rmse"
        ].reindex(ALL_VARIANTS)
        if corrected.isna().any() or fused.isna().any():
            raise ValueError(f"{regime_name}工况corrected/fused矩阵不完整")
        axis.bar(x - width / 2, corrected, width, label="Corrected", color="#4c78a8")
        axis.bar(x + width / 2, fused, width, label="Fused", color="#f58518")
        axis.set_xticks(x, [variant.upper() for variant in ALL_VARIANTS])
        axis.set_xlabel("Variant")
        axis.set_ylabel("Five-farm macro NRMSE")
        axis.set_title(regime_name)
        axis.legend()
    figure.suptitle("Corrected candidate vs final fusion by realized regime (test set)", y=1.02)
    figure.tight_layout()
    return figure


def source_signature(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [source_record(path) for path in paths]


def previous_jobs() -> dict[str, dict[str, Any]]:
    if not MANIFEST_PATH.is_file():
        return {}
    try:
        manifest = load_json_strict(MANIFEST_PATH)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    jobs = manifest.get("jobs", [])
    if not isinstance(jobs, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, dict):
            continue
        output = job.get("output")
        if isinstance(output, dict) and isinstance(output.get("path"), str):
            result[output["path"]] = job
    return result


def render_job(
    *,
    plt,
    jobs: list[dict[str, Any]],
    prior: dict[str, dict[str, Any]],
    script_hash: str,
    category: str,
    plot_kind: str,
    variant: str,
    farm_id: str,
    sources: list[Path],
    output_path: Path,
    renderer: Callable[[], Any],
    force: bool,
) -> None:
    signature = source_signature(sources)
    output_key = relative_path(output_path)
    old = prior.get(output_key)
    if not force and old is not None and output_path.is_file():
        old_output = old.get("output", {})
        if (
            old.get("script_sha256") == script_hash
            and old.get("sources") == signature
            and old_output.get("sha256") == sha256_file(output_path)
        ):
            width, height = validate_png(output_path)
            output = {
                "path": output_key,
                "sha256": old_output["sha256"],
                "size_bytes": output_path.stat().st_size,
                "width": width,
                "height": height,
            }
            jobs.append(
                {
                    "category": category,
                    "plot_kind": plot_kind,
                    "variant": variant,
                    "farm_id": farm_id,
                    "status": "current",
                    "script_sha256": script_hash,
                    "sources": signature,
                    "output": output,
                }
            )
            return
    figure = renderer()
    output = atomic_save_figure(plt, figure, output_path)
    jobs.append(
        {
            "category": category,
            "plot_kind": plot_kind,
            "variant": variant,
            "farm_id": farm_id,
            "status": "generated",
            "script_sha256": script_hash,
            "sources": signature,
            "output": output,
        }
    )


def build_gate_regime_matrix(gate_path: Path, assignment_path: Path, label: str) -> pd.DataFrame:
    gate = pd.read_csv(
        gate_path,
        usecols=["farm_id", "sample_id", "horizon_step", "applied_gate"],
        dtype={"farm_id": "string", "sample_id": "int64"},
    )
    assignment = pd.read_csv(
        assignment_path,
        usecols=["farm_id", "sample_id", "realized_regime", "low_power"],
        dtype={"farm_id": "string", "sample_id": "int64"},
    )
    if assignment.duplicated(["farm_id", "sample_id"]).any():
        raise ValueError(f"{label} regime assignment自然键重复")
    require_finite(gate, ["sample_id", "horizon_step", "applied_gate"], label)
    if not gate["horizon_step"].isin(FORECAST_STEPS).all():
        raise ValueError(f"{label}包含非法horizon_step")
    if not gate["applied_gate"].between(0, 1).all():
        raise ValueError(f"{label} applied_gate超出[0,1]")
    merged = gate.merge(
        assignment,
        on=["farm_id", "sample_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError(f"{label} gate与regime assignment未完整匹配")
    low_power = merged["low_power"]
    if low_power.dtype != bool:
        low_power = low_power.astype(str).str.lower().map({"true": True, "false": False})
    if low_power.isna().any():
        raise ValueError(f"{label} low_power无法解析为布尔值")
    rows: list[pd.DataFrame] = []
    for regime_name in REGIME_ORDER:
        if regime_name == "low_power":
            part = merged[low_power]
        else:
            part = merged[merged["realized_regime"] == regime_name]
        grouped = part.groupby("horizon_step", as_index=False)["applied_gate"].mean()
        grouped["regime_group"] = regime_name
        rows.append(grouped)
    matrix = (
        pd.concat(rows, ignore_index=True)
        .pivot(index="regime_group", columns="horizon_step", values="applied_gate")
        .reindex(index=REGIME_ORDER, columns=FORECAST_STEPS)
    )
    if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy(float)).all():
        raise ValueError(f"{label} gate-by-regime 4x16矩阵不完整")
    return matrix


def validate_candidate_history(frame: pd.DataFrame, label: str) -> None:
    columns = ("epoch", "loss", "mae", "rmse", "val_loss", "val_mae", "val_rmse")
    require_columns(frame, columns, label)
    require_finite(frame, columns, label)
    epoch = frame["epoch"].to_numpy(dtype=int)
    if len(epoch) < 2 or not np.array_equal(epoch, np.arange(len(epoch))):
        raise ValueError(f"{label} epoch必须从0连续递增且至少两轮")


def validate_gate_history(frame: pd.DataFrame, label: str) -> None:
    columns = (
        "global_epoch",
        "phase",
        "loss",
        "forecast_power_mae",
        "forecast_power_rmse",
        "val_loss",
        "val_forecast_power_mae",
        "val_forecast_power_rmse",
    )
    require_columns(frame, columns, label)
    require_finite(frame, [column for column in columns if column != "phase"], label)
    epoch = frame["global_epoch"].to_numpy(dtype=int)
    if len(epoch) < 3 or not np.array_equal(epoch, np.arange(len(epoch))):
        raise ValueError(f"{label} global_epoch必须从0连续递增")
    phases = tuple(frame["phase"].astype(str).drop_duplicates())
    if phases != GATE_PHASE_ORDER:
        raise ValueError(f"{label} phase顺序异常: {phases}")


def validate_calibration(frame: pd.DataFrame, label: str) -> None:
    columns = ("gate_bin", "count", "mean_raw_gate", "corrected_better_rate")
    require_columns(frame, columns, label)
    if len(frame) != 10 or tuple(frame["gate_bin"].astype(int)) != tuple(range(10)):
        raise ValueError(f"{label}必须恰含10个有序gate bins")
    if (frame["count"] < 0).any() or int(frame["count"].sum()) <= 0:
        raise ValueError(f"{label} count非法")
    valid = frame[frame["count"] > 0]
    require_finite(valid, ["mean_raw_gate", "corrected_better_rate"], label)
    if not valid["mean_raw_gate"].between(0, 1).all() or not valid[
        "corrected_better_rate"
    ].between(0, 1).all():
        raise ValueError(f"{label}可靠性概率超出[0,1]")


def preflight() -> dict[str, Any]:
    training = load_json_strict(TRAINING_MARKER)
    prediction = load_json_strict(PREDICTION_MARKER)
    if training.get("status") != "complete":
        raise ValueError("时频训练bundle尚未完成")
    if prediction.get("status") != "complete":
        raise ValueError("时频预测bundle尚未完成")
    if training.get("protocol_version") != prediction.get("protocol_version"):
        raise ValueError("训练/预测protocol_version不一致")
    farms = tuple(str(value) for value in training.get("expected_farm_ids", []))
    if len(farms) != 5 or len(set(farms)) != 5:
        raise ValueError(f"expected_farm_ids应为5个唯一场站，实际={farms}")
    if tuple(str(value) for value in prediction.get("expected_farm_ids", [])) != farms:
        raise ValueError("训练/预测expected_farm_ids不一致")
    if tuple(training.get("variants", [])) != ALL_VARIANTS:
        raise ValueError(f"训练variant矩阵异常: {training.get('variants')}")
    if tuple(prediction.get("variants", [])) != ALL_VARIANTS:
        raise ValueError(f"预测variant矩阵异常: {prediction.get('variants')}")
    if prediction.get("selected_variant") != "t0":
        raise ValueError(f"正式selection不是预期T0: {prediction.get('selected_variant')}")

    training_files = training.get("files", {})
    prediction_files = prediction.get("files", {})
    if not isinstance(training_files, dict) or not isinstance(prediction_files, dict):
        raise ValueError("bundle files字段非法")

    # The prediction marker must still point to this exact frozen training marker.
    verify_descriptor(prediction_files["training_marker"], "prediction.training_marker")

    candidate_frames: dict[tuple[str, str], tuple[pd.DataFrame, Path]] = {}
    gate_frames: dict[tuple[str, str], tuple[pd.DataFrame, Path]] = {}
    not_applicable: list[dict[str, Any]] = []
    for farm_id in farms:
        for variant in TRAINED_VARIANTS:
            candidate_key = f"{variant}.{farm_id}.candidate_history_path"
            gate_key = f"{variant}.{farm_id}.gate_history_path"
            candidate_path = verify_descriptor(training_files[candidate_key], candidate_key)
            gate_path = verify_descriptor(training_files[gate_key], gate_key)
            candidate = pd.read_csv(candidate_path)
            gate = pd.read_csv(gate_path)
            validate_gate_history(gate, gate_key)
            gate_frames[(variant, farm_id)] = (gate, gate_path)
            if variant == "m0":
                require_columns(
                    candidate,
                    ["epoch", "phase", "selection_val_candidate_nrmse"],
                    candidate_key,
                )
                if (
                    len(candidate) != 1
                    or int(candidate.iloc[0]["epoch"]) != -1
                    or str(candidate.iloc[0]["phase"]) != "skipped_frozen_f7_candidate"
                ):
                    raise ValueError(f"{candidate_key}不符合冻结F7 candidate语义")
                not_applicable.append(
                    {
                        "category": "training",
                        "plot_kind": "candidate_history_3panel",
                        "variant": variant,
                        "farm_id": farm_id,
                        "status": "not_applicable",
                        "sources": source_signature([candidate_path]),
                        "note": "M0 candidate is the frozen F7 residual; no candidate training trajectory exists.",
                    }
                )
            else:
                validate_candidate_history(candidate, candidate_key)
                candidate_frames[(variant, farm_id)] = (candidate, candidate_path)
        for plot_kind in ("candidate_history_3panel", "gate_history_3panel"):
            not_applicable.append(
                {
                    "category": "training",
                    "plot_kind": plot_kind,
                    "variant": "t0",
                    "farm_id": farm_id,
                    "status": "not_applicable",
                    "sources": [],
                    "note": "T0 is a direct Stage-3 G0 reference and was not trained in this bundle.",
                }
            )

    formal_paths: dict[str, Path] = {}
    for key in ("calibration", "comparison", "candidate", "regime"):
        marker_key = f"formal.{key}"
        formal_paths[key] = verify_descriptor(prediction_files[marker_key], marker_key)
    calibration_all = pd.read_csv(formal_paths["calibration"])
    comparison = pd.read_csv(formal_paths["comparison"])
    candidate_all = pd.read_csv(formal_paths["candidate"])
    regime_all = pd.read_csv(formal_paths["regime"])
    require_columns(
        comparison,
        [
            "model_variant",
            "macro_test_nrmse",
            "macro_positive_regret_mean",
            "selected",
        ],
        "formal.comparison",
    )
    if tuple(comparison["model_variant"].astype(str)) != ALL_VARIANTS:
        raise ValueError("formal.comparison variant顺序/矩阵异常")
    if int(comparison["selected"].astype(bool).sum()) != 1 or not bool(
        comparison.loc[comparison["model_variant"] == "t0", "selected"].iloc[0]
    ):
        raise ValueError("formal.comparison selected标记异常")
    require_columns(
        candidate_all,
        ["model_variant", "farm_id", "candidate", "horizon_step", "capacity_normalized_rmse"],
        "formal.candidate",
    )
    require_columns(
        regime_all,
        [
            "model_variant",
            "farm_id",
            "regime_group",
            "candidate",
            "horizon_step",
            "capacity_normalized_rmse",
        ],
        "formal.regime",
    )

    existing: list[dict[str, Any]] = []
    for key in (
        "formal.rank_figure",
        "formal.farm_heatmap_figure",
        "formal.horizon_figure",
        "formal.complementarity_figure",
    ):
        path = verify_descriptor(prediction_files[key], key, require_png=True)
        width, height = validate_png(path)
        existing.append(
            {
                "category": "prediction",
                "plot_kind": key.removeprefix("formal."),
                "variant": "all",
                "farm_id": "all",
                "status": "validated_existing",
                "reference_path": relative_path(path),
                "reference_sha256": sha256_file(path),
                "reference_size_bytes": path.stat().st_size,
                "reference_width": width,
                "reference_height": height,
                "note": "Created by the frozen formal prediction program; not overwritten.",
            }
        )

    result_prefixes = sorted(
        {
            key.split(".", 1)[0]
            for key in prediction_files
            if re.fullmatch(r"result\d+\..+", key)
        },
        key=lambda value: int(value.removeprefix("result")),
    )
    if len(result_prefixes) != 20:
        raise ValueError(f"正式新预测应恰含20组result，实际={len(result_prefixes)}")
    diagnostics: dict[tuple[str, str], dict[str, Any]] = {}
    tuple_pattern = re.compile(r"time_freq_model_(m0|t1|t2|t3)_.+_farm_(\d+)")
    for prefix in result_prefixes:
        paths: dict[str, Path] = {}
        for artifact in ("gate", "assignment", "calibration"):
            key = f"{prefix}.{artifact}"
            paths[artifact] = verify_descriptor(prediction_files[key], key)
        match = tuple_pattern.search(paths["gate"].stem)
        if match is None:
            raise ValueError(f"无法从gate路径解析variant/farm: {paths['gate']}")
        variant, farm_id = match.group(1), match.group(2)
        if variant not in TRAINED_VARIANTS or farm_id not in farms:
            raise ValueError(f"result自然键越界: variant={variant}, farm={farm_id}")
        natural_key = (variant, farm_id)
        if natural_key in diagnostics:
            raise ValueError(f"result自然键重复: {natural_key}")
        calibration = pd.read_csv(paths["calibration"])
        validate_calibration(calibration, f"{prefix}.calibration")
        if (
            set(calibration["model_variant"].astype(str)) != {variant}
            or set(calibration["farm_id"].astype(str)) != {farm_id}
        ):
            raise ValueError(f"{prefix}.calibration身份字段不一致")
        matrix = build_gate_regime_matrix(paths["gate"], paths["assignment"], prefix)
        diagnostics[natural_key] = {
            "paths": paths,
            "calibration": calibration,
            "gate_regime_matrix": matrix,
        }
        for artifact in ("single_figure", "weighted_figure"):
            key = f"{prefix}.{artifact}"
            figure_path = verify_descriptor(prediction_files[key], key, require_png=True)
            width, height = validate_png(figure_path)
            existing.append(
                {
                    "category": "prediction",
                    "plot_kind": artifact,
                    "variant": variant,
                    "farm_id": farm_id,
                    "status": "validated_existing",
                    "reference_path": relative_path(figure_path),
                    "reference_sha256": sha256_file(figure_path),
                    "reference_size_bytes": figure_path.stat().st_size,
                    "reference_width": width,
                    "reference_height": height,
                    "note": "Created by the frozen formal prediction program; not overwritten.",
                }
            )
    expected_keys = {(variant, farm) for variant in TRAINED_VARIANTS for farm in farms}
    if set(diagnostics) != expected_keys:
        raise ValueError(f"预测诊断矩阵不完整: missing={sorted(expected_keys - set(diagnostics))}")

    stage3_marker_path = verify_descriptor(prediction_files["stage3_marker"], "stage3_marker")
    stage3 = load_json_strict(stage3_marker_path)
    stage3_files = stage3.get("files", {})
    t0_references: list[dict[str, Any]] = []
    for index, farm_id in enumerate(farms):
        for artifact in ("single_figure", "weighted_figure"):
            key = f"result{index}.{artifact}"
            path = verify_descriptor(stage3_files[key], f"stage3.{key}", require_png=True)
            if f"farm_{farm_id}" not in path.stem or "_g0_" not in path.stem:
                raise ValueError(f"Stage-3 T0源图身份不一致: {path}")
            width, height = validate_png(path)
            t0_references.append(
                {
                    "category": "prediction",
                    "plot_kind": artifact,
                    "variant": "t0",
                    "farm_id": farm_id,
                    "status": "source_reference",
                    "reference_path": relative_path(path),
                    "reference_sha256": sha256_file(path),
                    "reference_size_bytes": path.stat().st_size,
                    "reference_width": width,
                    "reference_height": height,
                    "note": "T0 directly references Stage-3 G0; the PNG is validated in place, not copied.",
                }
            )

    return {
        "training_marker": training,
        "prediction_marker": prediction,
        "farms": farms,
        "candidate_frames": candidate_frames,
        "gate_frames": gate_frames,
        "diagnostics": diagnostics,
        "formal_paths": formal_paths,
        "calibration_all": calibration_all,
        "comparison": comparison,
        "candidate_all": candidate_all,
        "regime_all": regime_all,
        "not_applicable": not_applicable,
        "existing": existing,
        "t0_references": t0_references,
        "stage3_marker_path": stage3_marker_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and backfill time-frequency training/prediction visualizations."
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="validate every source and print the expected matrix without writing images",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate backfill-owned PNG files even when the prior manifest is current",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("[1/4] 校验正式训练/预测bundle及全部补图输入……", flush=True)
    data = preflight()
    expected_generated = {
        "candidate_training": len(CANDIDATE_TRAINED_VARIANTS) * len(data["farms"]),
        "gate_training": len(TRAINED_VARIANTS) * len(data["farms"]),
        "per_farm_prediction_diagnostics": 2 * len(TRAINED_VARIANTS) * len(data["farms"]),
        "aggregate_prediction_diagnostics": 4,
    }
    expected_total = sum(expected_generated.values())
    print(
        "预检通过：需补训练图35张、逐场站预测诊断图40张、汇总预测图4张；"
        "现有预测图44张及T0源图10张已校验。",
        flush=True,
    )
    if args.audit_only:
        print(json.dumps(expected_generated, ensure_ascii=False, indent=2))
        return

    print("[2/4] 生成/复核训练三子图……", flush=True)
    plt = setup_matplotlib()
    script_hash = sha256_file(Path(__file__))
    prior = previous_jobs()
    jobs: list[dict[str, Any]] = []
    jobs.extend(data["not_applicable"])
    jobs.extend(data["existing"])
    jobs.extend(data["t0_references"])
    for (variant, farm_id), (frame, source_path) in data["candidate_frames"].items():
        output_path = source_path.with_suffix(".png")
        render_job(
            plt=plt,
            jobs=jobs,
            prior=prior,
            script_hash=script_hash,
            category="training",
            plot_kind="candidate_history_3panel",
            variant=variant,
            farm_id=farm_id,
            sources=[source_path],
            output_path=output_path,
            renderer=lambda frame=frame, variant=variant, farm_id=farm_id: plot_candidate_history(
                plt, frame, variant, farm_id
            ),
            force=args.force,
        )
    for (variant, farm_id), (frame, source_path) in data["gate_frames"].items():
        output_path = source_path.with_suffix(".png")
        render_job(
            plt=plt,
            jobs=jobs,
            prior=prior,
            script_hash=script_hash,
            category="training",
            plot_kind="gate_history_3panel",
            variant=variant,
            farm_id=farm_id,
            sources=[source_path],
            output_path=output_path,
            renderer=lambda frame=frame, variant=variant, farm_id=farm_id: plot_gate_history(
                plt, frame, variant, farm_id
            ),
            force=args.force,
        )

    print("[3/4] 补齐门控预测诊断与论文解释图……", flush=True)
    for (variant, farm_id), diagnostic in data["diagnostics"].items():
        figure_dir = RESULT_ROOT / variant / "testdata_predict_output" / "figures"
        paths = diagnostic["paths"]
        render_job(
            plt=plt,
            jobs=jobs,
            prior=prior,
            script_hash=script_hash,
            category="prediction",
            plot_kind="gate_by_regime_horizon",
            variant=variant,
            farm_id=farm_id,
            sources=[paths["gate"], paths["assignment"]],
            output_path=figure_dir
            / f"time_freq_model_{variant}_gate_by_regime_farm_{farm_id}.png",
            renderer=lambda diagnostic=diagnostic, variant=variant, farm_id=farm_id: plot_gate_by_regime(
                plt, diagnostic["gate_regime_matrix"], variant, farm_id
            ),
            force=args.force,
        )
        render_job(
            plt=plt,
            jobs=jobs,
            prior=prior,
            script_hash=script_hash,
            category="prediction",
            plot_kind="gate_calibration",
            variant=variant,
            farm_id=farm_id,
            sources=[paths["calibration"]],
            output_path=figure_dir
            / f"time_freq_model_{variant}_gate_calibration_farm_{farm_id}.png",
            renderer=lambda diagnostic=diagnostic, variant=variant, farm_id=farm_id: plot_farm_reliability(
                plt, diagnostic["calibration"], variant, farm_id
            ),
            force=args.force,
        )

    aggregate_specs = (
        (
            "aggregate_gate_reliability",
            [data["formal_paths"]["calibration"]],
            PREDICTION_ROOT / "figures" / "time_freq_model_test_reliability.png",
            lambda: plot_aggregate_reliability(plt, data["calibration_all"]),
        ),
        (
            "aggregate_accuracy_safety",
            [data["formal_paths"]["comparison"]],
            PREDICTION_ROOT / "figures" / "time_freq_model_test_safety_accuracy.png",
            lambda: plot_safety_accuracy(plt, data["comparison"]),
        ),
        (
            "corrected_candidate_horizon_nrmse",
            [data["formal_paths"]["candidate"]],
            PREDICTION_ROOT
            / "figures"
            / "time_freq_model_test_corrected_candidate_horizon_nrmse.png",
            lambda: plot_corrected_candidate_horizon(plt, data["candidate_all"]),
        ),
        (
            "corrected_vs_fused_regime_nrmse",
            [data["formal_paths"]["regime"]],
            PREDICTION_ROOT / "figures" / "time_freq_model_test_regime_nrmse.png",
            lambda: plot_regime_nrmse(plt, data["regime_all"]),
        ),
    )
    for plot_kind, sources, output_path, renderer in aggregate_specs:
        render_job(
            plt=plt,
            jobs=jobs,
            prior=prior,
            script_hash=script_hash,
            category="prediction",
            plot_kind=plot_kind,
            variant="all",
            farm_id="all",
            sources=sources,
            output_path=output_path,
            renderer=renderer,
            force=args.force,
        )

    owned = [job for job in jobs if job.get("status") in {"generated", "current"}]
    if len(owned) != expected_total:
        raise RuntimeError(f"补图矩阵数量异常: expected={expected_total}, actual={len(owned)}")
    if len(data["existing"]) != 44 or len(data["t0_references"]) != 10:
        raise RuntimeError("现有预测图/T0源图审计数量异常")

    print("[4/4] 写入独立可视化manifest与CSV清单……", flush=True)
    counts: dict[str, int] = {}
    for job in jobs:
        status = str(job.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    manifest = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": source_record(Path(__file__)),
        "source_training_marker": source_record(TRAINING_MARKER),
        "source_prediction_marker": source_record(PREDICTION_MARKER),
        "source_stage3_marker": source_record(data["stage3_marker_path"]),
        "selection_split": data["prediction_marker"].get("selection_split"),
        "test_reuse_status": data["prediction_marker"].get("test_reuse_status"),
        "selected_variant": data["prediction_marker"].get("selected_variant"),
        "expected_farm_ids": list(data["farms"]),
        "expected_backfill_matrix": expected_generated,
        "counts_by_status": counts,
        "policies": {
            "formal_train_predict_code_unchanged": True,
            "t0_figures_referenced_not_copied": True,
            "m0_frozen_candidate_curve_not_fabricated": True,
            "backfill_outputs_owned_only_by_this_script": True,
        },
        "inventory_csv": relative_path(INVENTORY_PATH),
        "jobs": jobs,
    }
    atomic_write_inventory(jobs, INVENTORY_PATH)
    atomic_write_json(manifest, MANIFEST_PATH)
    generated = counts.get("generated", 0)
    current = counts.get("current", 0)
    print(
        f"完成：本次生成{generated}张、沿用已校验补图{current}张；"
        f"补图总矩阵{expected_total}张。\n"
        f"manifest: {relative_path(MANIFEST_PATH)}\n"
        f"inventory: {relative_path(INVENTORY_PATH)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
