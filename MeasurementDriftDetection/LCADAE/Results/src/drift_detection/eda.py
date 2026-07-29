from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .datasets import SKAB_SENSOR_COLUMNS, load_gas_dataset, load_skab_sequences, skab_summary
from .io_utils import atomic_write_dataframe
from .preprocessing import GasContextResidualizer, RobustSeriesScaler, feature_columns
from .windows import make_windows


def _figure_dir(config: dict[str, Any]) -> Path:
    path = Path(config["paths"]["results"]) / "data" / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _data_result_dir(config: dict[str, Any]) -> Path:
    path = Path(config["paths"]["results"]) / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _sample_indices(n_rows: int, max_rows: int, seed: int) -> np.ndarray:
    if n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_rows, replace=False))


def _safe_corr(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        return np.eye(values.shape[1] if values.ndim == 2 else 1, dtype=float)
    corr = np.corrcoef(values, rowvar=False)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def _corr_distance(reference: np.ndarray, current: np.ndarray) -> float:
    ref_corr = _safe_corr(reference)
    cur_corr = _safe_corr(current)
    denominator = max(1.0, np.sqrt(ref_corr.size))
    return float(np.linalg.norm(cur_corr - ref_corr, ord="fro") / denominator)


def _pca_coordinates(
    values: np.ndarray,
    metadata: pd.DataFrame,
    max_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    indices = _sample_indices(len(values), max_rows=max_rows, seed=seed)
    sampled = np.asarray(values, dtype=np.float32)[indices]
    n_components = 2 if sampled.shape[1] >= 2 else 1
    pca = PCA(n_components=n_components, random_state=seed)
    coordinates = pca.fit_transform(sampled)
    output = metadata.iloc[indices].reset_index(drop=True).copy()
    output["pc1"] = coordinates[:, 0]
    output["pc2"] = coordinates[:, 1] if coordinates.shape[1] > 1 else 0.0
    return output, pca.explained_variance_ratio_


def _write_markdown(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def analyze_gas_dataset(config: dict[str, Any], seed: int = 42) -> dict[str, Path]:
    settings = config["data"]["gas"]
    eda = config.get("eda", {})
    max_rows = int(eda.get("max_rows_for_pca", 6000))
    dpi = int(eda.get("figure_dpi", 150))
    result_dir = _data_result_dir(config)
    figure_dir = _figure_dir(config)

    gas = load_gas_dataset(config["paths"]["raw_gas"])
    features = feature_columns(gas)
    train_mask = gas["batch"].isin(settings["train_batches"])
    train = gas.loc[train_mask].copy()

    quality = pd.DataFrame(
        [
            {
                "dataset": "gas_sensor",
                "rows": len(gas),
                "channels": len(features),
                "batches": int(gas["batch"].nunique()),
                "gas_classes": int(gas["gas_code"].nunique()),
                "missing_feature_values": int(gas[features].isna().sum().sum()),
                "duplicate_rows": int(gas.duplicated(subset=["batch", "row_in_batch"]).sum()),
                "minimum_concentration": float(gas["concentration"].min()),
                "median_concentration": float(gas["concentration"].median()),
                "maximum_concentration": float(gas["concentration"].max()),
            }
        ]
    )
    quality_path = result_dir / "gas_data_quality_summary.csv"
    atomic_write_dataframe(quality, quality_path)

    batch_summary = (
        gas.groupby("batch", as_index=False)
        .agg(
            rows=("gas_code", "size"),
            gas_classes=("gas_code", "nunique"),
            concentration_min=("concentration", "min"),
            concentration_median=("concentration", "median"),
            concentration_max=("concentration", "max"),
        )
        .sort_values("batch")
    )
    batch_summary_path = result_dir / "gas_batch_summary.csv"
    atomic_write_dataframe(batch_summary, batch_summary_path)

    composition = pd.crosstab(gas["batch"], gas["gas_name"])
    composition_path = result_dir / "gas_batch_gas_composition.csv"
    atomic_write_dataframe(composition.reset_index(), composition_path)

    concentration_summary = (
        gas.groupby(["gas_code", "gas_name"], as_index=False)
        .agg(
            rows=("concentration", "size"),
            concentration_min=("concentration", "min"),
            concentration_median=("concentration", "median"),
            concentration_mean=("concentration", "mean"),
            concentration_max=("concentration", "max"),
        )
        .sort_values("gas_code")
    )
    concentration_summary_path = result_dir / "gas_concentration_summary.csv"
    atomic_write_dataframe(concentration_summary, concentration_summary_path)

    residualizer = GasContextResidualizer.fit(
        train,
        features,
        ridge_alpha=float(settings.get("ridge_alpha", 1.0)),
    )
    residuals = residualizer.transform(gas)
    train_residuals = residuals[train_mask.to_numpy()]
    train_corr = _safe_corr(train_residuals)

    drift_rows: list[dict[str, float | int]] = []
    sensor_rows: list[dict[str, float | int]] = []
    feature_to_sensor = np.asarray([(index // 8) + 1 for index in range(len(features))], dtype=int)
    for batch in sorted(gas["batch"].unique()):
        mask = gas["batch"].to_numpy() == batch
        current = residuals[mask]
        abs_current = np.abs(current)
        drift_rows.append(
            {
                "batch": int(batch),
                "rows": int(mask.sum()),
                "mean_absolute_residual": float(abs_current.mean()),
                "median_absolute_residual": float(np.median(abs_current)),
                "root_mean_squared_residual": float(np.sqrt(np.mean(current**2))),
                "p95_absolute_residual": float(np.quantile(abs_current, 0.95)),
                "correlation_distance_from_train": float(
                    np.linalg.norm(_safe_corr(current) - train_corr, ord="fro") / max(1.0, np.sqrt(train_corr.size))
                ),
            }
        )
        for sensor_index in sorted(np.unique(feature_to_sensor)):
            columns = np.flatnonzero(feature_to_sensor == sensor_index)
            sensor_rows.append(
                {
                    "batch": int(batch),
                    "sensor_index": int(sensor_index),
                    "mean_absolute_residual": float(np.mean(np.abs(current[:, columns]))),
                    "root_mean_squared_residual": float(np.sqrt(np.mean(current[:, columns] ** 2))),
                }
            )

    drift = pd.DataFrame(drift_rows)
    drift_path = result_dir / "gas_natural_drift_by_batch.csv"
    atomic_write_dataframe(drift, drift_path)
    sensor_drift = pd.DataFrame(sensor_rows)
    sensor_drift_path = result_dir / "gas_sensor_group_drift_by_batch.csv"
    atomic_write_dataframe(sensor_drift, sensor_drift_path)

    pca_metadata = gas[["batch", "gas_code", "gas_name", "concentration"]].copy()
    pca_frame, explained = _pca_coordinates(residuals, pca_metadata, max_rows=max_rows, seed=seed)
    pca_frame["pc1_explained_variance"] = float(explained[0]) if len(explained) else float("nan")
    pca_frame["pc2_explained_variance"] = float(explained[1]) if len(explained) > 1 else 0.0
    pca_path = result_dir / "gas_residual_pca_coordinates.csv"
    atomic_write_dataframe(pca_frame, pca_path)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(batch_summary["batch"].astype(str), batch_summary["rows"])
    ax.set_title("Gas Sensor samples by temporal batch")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Number of samples")
    gas_count_figure = _save_figure(fig, figure_dir / "gas_samples_by_batch.png", dpi)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    matrix = composition.to_numpy(dtype=float)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_title("Gas-class composition across temporal batches")
    ax.set_xlabel("Gas class")
    ax.set_ylabel("Batch")
    ax.set_xticks(np.arange(len(composition.columns)))
    ax.set_xticklabels(composition.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(composition.index)))
    ax.set_yticklabels(composition.index)
    fig.colorbar(image, ax=ax, label="Sample count")
    composition_figure = _save_figure(fig, figure_dir / "gas_composition_heatmap.png", dpi)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(drift["batch"], drift["mean_absolute_residual"], marker="o", label="Mean absolute residual")
    ax.plot(drift["batch"], drift["p95_absolute_residual"], marker="s", label="95th percentile")
    ax.set_title("Context-adjusted natural drift across Gas Sensor batches")
    ax.set_xlabel("Temporal batch")
    ax.set_ylabel("Standardized residual magnitude")
    ax.legend()
    ax.grid(alpha=0.25)
    drift_figure = _save_figure(fig, figure_dir / "gas_natural_drift_index.png", dpi)

    pivot = sensor_drift.pivot(index="sensor_index", columns="batch", values="mean_absolute_residual")
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_title("Sensor-group residual magnitude by batch")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Physical sensor group")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(image, ax=ax, label="Mean absolute residual")
    sensor_figure = _save_figure(fig, figure_dir / "gas_sensor_group_drift_heatmap.png", dpi)

    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    scatter = ax.scatter(pca_frame["pc1"], pca_frame["pc2"], c=pca_frame["batch"], s=10, alpha=0.55)
    ax.set_title("PCA of context-adjusted Gas Sensor residuals")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    pc2_variance = explained[1] * 100 if len(explained) > 1 else 0.0
    ax.set_ylabel(f"PC2 ({pc2_variance:.1f}% variance)")
    fig.colorbar(scatter, ax=ax, label="Batch")
    pca_figure = _save_figure(fig, figure_dir / "gas_residual_pca.png", dpi)

    return {
        "quality": quality_path,
        "batch_summary": batch_summary_path,
        "composition": composition_path,
        "concentration_summary": concentration_summary_path,
        "natural_drift": drift_path,
        "sensor_group_drift": sensor_drift_path,
        "pca_coordinates": pca_path,
        "figure_samples": gas_count_figure,
        "figure_composition": composition_figure,
        "figure_natural_drift": drift_figure,
        "figure_sensor_drift": sensor_figure,
        "figure_pca": pca_figure,
    }


def analyze_skab_dataset(config: dict[str, Any], seed: int = 42) -> dict[str, Path]:
    eda = config.get("eda", {})
    max_rows = int(eda.get("max_rows_for_pca", 6000))
    representative_points = int(eda.get("representative_points", 1200))
    top_channels = int(eda.get("top_channels_to_plot", 4))
    dpi = int(eda.get("figure_dpi", 150))
    result_dir = _data_result_dir(config)
    figure_dir = _figure_dir(config)

    normal_sequences, anomaly_sequences = load_skab_sequences(config["paths"]["raw_skab"])
    file_summary = skab_summary(config["paths"]["raw_skab"])
    file_summary_path = result_dir / "skab_file_summary_eda.csv"
    atomic_write_dataframe(file_summary, file_summary_path)

    all_frames = normal_sequences + anomaly_sequences
    total_rows = sum(len(frame) for frame in all_frames)
    quality = pd.DataFrame(
        [
            {
                "dataset": "skab",
                "rows": total_rows,
                "channels": len(SKAB_SENSOR_COLUMNS),
                "normal_files": len(normal_sequences),
                "anomaly_files": len(anomaly_sequences),
                "missing_sensor_values_after_loading": int(
                    sum(frame[SKAB_SENSOR_COLUMNS].isna().sum().sum() for frame in all_frames)
                ),
                "duplicate_timestamps": int(
                    sum(frame["datetime"].duplicated().sum() for frame in all_frames if "datetime" in frame.columns)
                ),
                "anomaly_points": int(sum(frame["anomaly"].fillna(0).sum() for frame in anomaly_sequences)),
                "changepoints": int(sum(frame["changepoint"].fillna(0).sum() for frame in anomaly_sequences)),
            }
        ]
    )
    quality_path = result_dir / "skab_data_quality_summary.csv"
    atomic_write_dataframe(quality, quality_path)

    normal_array = np.concatenate(
        [frame[SKAB_SENSOR_COLUMNS].to_numpy(dtype=np.float32) for frame in normal_sequences], axis=0
    )
    scaler = RobustSeriesScaler.fit(normal_array)
    normal_scaled = scaler.transform(normal_array)
    reference_corr = _safe_corr(normal_scaled)

    shift_rows: list[dict[str, Any]] = []
    for split, sequences in [("normal", normal_sequences), ("anomaly", anomaly_sequences)]:
        for sequence_id, frame in enumerate(sequences):
            values = scaler.transform(frame[SKAB_SENSOR_COLUMNS].to_numpy(dtype=np.float32))
            source = Path(str(frame["source_file"].iloc[0])).name
            shift_rows.append(
                {
                    "split": split,
                    "sequence_id": sequence_id,
                    "source_file": source,
                    "rows": len(frame),
                    "anomaly_ratio": float(frame["anomaly"].fillna(0).mean()),
                    "changepoint_ratio": float(frame["changepoint"].fillna(0).mean()),
                    "median_absolute_level": float(np.median(np.abs(values))),
                    "root_mean_squared_level": float(np.sqrt(np.mean(values**2))),
                    "correlation_distance_from_normal": _corr_distance(normal_scaled, values),
                }
            )
    shift = pd.DataFrame(shift_rows)
    shift_path = result_dir / "skab_sequence_distribution_shift.csv"
    atomic_write_dataframe(shift, shift_path)

    sensor_stats_rows: list[dict[str, Any]] = []
    for state, sequences in [("normal", normal_sequences), ("anomaly_files", anomaly_sequences)]:
        values = np.concatenate(
            [frame[SKAB_SENSOR_COLUMNS].to_numpy(dtype=np.float32) for frame in sequences], axis=0
        )
        for channel_index, channel in enumerate(SKAB_SENSOR_COLUMNS):
            column = values[:, channel_index]
            sensor_stats_rows.append(
                {
                    "state": state,
                    "channel": channel,
                    "mean": float(np.mean(column)),
                    "standard_deviation": float(np.std(column)),
                    "median": float(np.median(column)),
                    "minimum": float(np.min(column)),
                    "maximum": float(np.max(column)),
                }
            )
    sensor_stats = pd.DataFrame(sensor_stats_rows)
    sensor_stats_path = result_dir / "skab_sensor_statistics.csv"
    atomic_write_dataframe(sensor_stats, sensor_stats_path)

    pca_values: list[np.ndarray] = []
    pca_meta: list[pd.DataFrame] = []
    if normal_sequences:
        normal_frame = pd.concat(normal_sequences, ignore_index=True)
        normal_values = scaler.transform(normal_frame[SKAB_SENSOR_COLUMNS].to_numpy(dtype=np.float32))
        pca_values.append(normal_values)
        pca_meta.append(
            pd.DataFrame(
                {
                    "state": np.where(normal_frame["anomaly"].fillna(0).to_numpy() > 0, "anomaly", "normal"),
                    "source_type": "normal_file",
                }
            )
        )
    if anomaly_sequences:
        anomaly_frame = pd.concat(anomaly_sequences, ignore_index=True)
        anomaly_values = scaler.transform(anomaly_frame[SKAB_SENSOR_COLUMNS].to_numpy(dtype=np.float32))
        pca_values.append(anomaly_values)
        pca_meta.append(
            pd.DataFrame(
                {
                    "state": np.where(anomaly_frame["anomaly"].fillna(0).to_numpy() > 0, "anomaly", "normal"),
                    "source_type": "anomaly_file",
                }
            )
        )
    combined_values = np.concatenate(pca_values, axis=0)
    combined_meta = pd.concat(pca_meta, ignore_index=True)
    pca_frame, explained = _pca_coordinates(combined_values, combined_meta, max_rows=max_rows, seed=seed)
    pca_frame["pc1_explained_variance"] = float(explained[0]) if len(explained) else float("nan")
    pca_frame["pc2_explained_variance"] = float(explained[1]) if len(explained) > 1 else 0.0
    pca_path = result_dir / "skab_pca_coordinates.csv"
    atomic_write_dataframe(pca_frame, pca_path)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ordered = file_summary.sort_values(["split", "anomaly_points"], ascending=[True, False]).reset_index(drop=True)
    labels = [Path(value).stem for value in ordered["source_file"]]
    ax.bar(np.arange(len(ordered)), ordered["anomaly_points"])
    ax.set_title("SKAB anomaly-point coverage by source file")
    ax.set_xlabel("Source sequence")
    ax.set_ylabel("Anomaly points")
    step = max(1, len(labels) // 12)
    ticks = np.arange(0, len(labels), step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[index] for index in ticks], rotation=45, ha="right")
    coverage_figure = _save_figure(fig, figure_dir / "skab_anomaly_coverage.png", dpi)

    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    states = sorted(pca_frame["state"].unique())
    for state in states:
        subset = pca_frame[pca_frame["state"] == state]
        ax.scatter(subset["pc1"], subset["pc2"], s=10, alpha=0.45, label=state)
    ax.set_title("PCA of robust-scaled SKAB sensor observations")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    pc2_variance = explained[1] * 100 if len(explained) > 1 else 0.0
    ax.set_ylabel(f"PC2 ({pc2_variance:.1f}% variance)")
    ax.legend()
    pca_figure = _save_figure(fig, figure_dir / "skab_pca.png", dpi)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    image = ax.imshow(reference_corr, vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_title("Correlation structure of normal SKAB operation")
    ax.set_xticks(np.arange(len(SKAB_SENSOR_COLUMNS)))
    ax.set_xticklabels(SKAB_SENSOR_COLUMNS, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(SKAB_SENSOR_COLUMNS)))
    ax.set_yticklabels(SKAB_SENSOR_COLUMNS)
    fig.colorbar(image, ax=ax, label="Correlation")
    correlation_figure = _save_figure(fig, figure_dir / "skab_normal_correlation.png", dpi)

    representative = None
    for frame in anomaly_sequences:
        if frame["anomaly"].fillna(0).sum() > 0:
            representative = frame
            break
    if representative is None:
        representative = anomaly_sequences[0]
    representative = representative.iloc[:representative_points].reset_index(drop=True)
    scaled_rep = scaler.transform(representative[SKAB_SENSOR_COLUMNS].to_numpy(dtype=np.float32))
    variance_order = np.argsort(np.var(scaled_rep, axis=0))[::-1][: max(1, top_channels)]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x_axis = np.arange(len(representative))
    for offset, channel_index in enumerate(variance_order):
        ax.plot(x_axis, scaled_rep[:, channel_index] + offset * 5.0, label=SKAB_SENSOR_COLUMNS[channel_index])
    anomaly_mask = representative["anomaly"].fillna(0).to_numpy(dtype=int) > 0
    inside = False
    start = 0
    for index, value in enumerate(anomaly_mask):
        if value and not inside:
            start = index
            inside = True
        if inside and (not value or index == len(anomaly_mask) - 1):
            end = index if not value else index + 1
            ax.axvspan(start, end, alpha=0.12)
            inside = False
    ax.set_title("Representative SKAB sequence with native anomaly intervals")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Robust-scaled value with vertical offsets")
    ax.legend(loc="upper right", ncol=2)
    representative_figure = _save_figure(fig, figure_dir / "skab_representative_sequence.png", dpi)

    plot_shift = shift.sort_values("correlation_distance_from_normal", ascending=False).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_shift["source_file"], plot_shift["correlation_distance_from_normal"])
    ax.set_title("Largest SKAB correlation shifts relative to normal operation")
    ax.set_xlabel("Normalized correlation distance")
    ax.set_ylabel("Sequence")
    shift_figure = _save_figure(fig, figure_dir / "skab_correlation_shift_ranking.png", dpi)

    return {
        "quality": quality_path,
        "file_summary": file_summary_path,
        "sequence_shift": shift_path,
        "sensor_statistics": sensor_stats_path,
        "pca_coordinates": pca_path,
        "figure_coverage": coverage_figure,
        "figure_pca": pca_figure,
        "figure_correlation": correlation_figure,
        "figure_representative": representative_figure,
        "figure_shift": shift_figure,
    }


def analyze_controlled_benchmarks(config: dict[str, Any]) -> dict[str, Path]:
    eda = config.get("eda", {})
    representative_points = int(eda.get("representative_points", 1200))
    top_channels = int(eda.get("top_channels_to_plot", 4))
    dpi = int(eda.get("figure_dpi", 150))
    result_dir = _data_result_dir(config)
    figure_dir = _figure_dir(config)
    window_size = int(config["data"]["window_size"])
    stride = int(config["data"]["stride"])

    summary_rows: list[dict[str, Any]] = []
    fault_rows: list[dict[str, Any]] = []
    figure_paths: dict[str, Path] = {}
    for dataset in ["gas", "skab"]:
        benchmark_path = Path(config["paths"]["benchmarks"]) / f"{dataset}_controlled.npz"
        with np.load(benchmark_path, allow_pickle=False) as bundle:
            train_series = bundle["train_series"]
            validation_series = bundle["validation_series"]
            test_series = bundle["test_series"]
            clean_series = bundle["clean_test_series"]
            labels = bundle["test_labels"]
            fault_types = bundle["test_fault_types"].astype(str)
            channel_names = bundle["channel_names"].astype(str)

        train_windows = make_windows(train_series, window_size, stride)["windows"]
        validation_windows = make_windows(validation_series, window_size, stride)["windows"]
        test_windows = make_windows(test_series, window_size, stride, labels=labels)["windows"]
        summary_rows.append(
            {
                "dataset": dataset,
                "channels": int(test_series.shape[1]),
                "train_points": int(len(train_series)),
                "validation_points": int(len(validation_series)),
                "test_points": int(len(test_series)),
                "train_windows": int(len(train_windows)),
                "validation_windows": int(len(validation_windows)),
                "test_windows": int(len(test_windows)),
                "anomaly_points": int(labels.sum()),
                "anomaly_ratio": float(labels.mean()),
                "fault_families": int(len(set(fault_types) - {"normal"})),
                "window_size": window_size,
                "stride": stride,
            }
        )
        for fault_type in sorted(set(fault_types)):
            mask = fault_types == fault_type
            fault_rows.append(
                {
                    "dataset": dataset,
                    "fault_type": fault_type,
                    "points": int(mask.sum()),
                    "point_ratio": float(mask.mean()),
                    "mean_absolute_injection_delta": float(
                        np.mean(np.abs(test_series[mask] - clean_series[mask])) if mask.any() else 0.0
                    ),
                }
            )

        delta = np.mean(np.abs(test_series - clean_series), axis=0)
        channel_indices = np.argsort(delta)[::-1][: max(1, top_channels)]
        n_points = min(len(test_series), representative_points)
        fig, axes = plt.subplots(len(channel_indices), 1, figsize=(11, 2.5 * len(channel_indices)), sharex=True)
        if len(channel_indices) == 1:
            axes = [axes]
        x_axis = np.arange(n_points)
        anomaly_mask = labels[:n_points] > 0
        for ax, channel_index in zip(axes, channel_indices):
            ax.plot(x_axis, clean_series[:n_points, channel_index], label="Clean reference", linewidth=1.0)
            ax.plot(x_axis, test_series[:n_points, channel_index], label="Injected series", linewidth=1.0)
            inside = False
            start = 0
            for index, value in enumerate(anomaly_mask):
                if value and not inside:
                    start = index
                    inside = True
                if inside and (not value or index == len(anomaly_mask) - 1):
                    end = index if not value else index + 1
                    ax.axvspan(start, end, alpha=0.12)
                    inside = False
            ax.set_ylabel(channel_names[channel_index])
            ax.grid(alpha=0.2)
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("Time index")
        fig.suptitle(f"{dataset.upper()} controlled systematic-drift examples", y=1.01)
        figure_paths[f"figure_{dataset}_controlled"] = _save_figure(
            fig, figure_dir / f"{dataset}_controlled_drift_examples.png", dpi
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = result_dir / "cross_domain_benchmark_summary.csv"
    atomic_write_dataframe(summary, summary_path)
    fault_coverage = pd.DataFrame(fault_rows)
    fault_coverage_path = result_dir / "cross_domain_fault_coverage.csv"
    atomic_write_dataframe(fault_coverage, fault_coverage_path)

    manifests = []
    for dataset, filename in [
        ("gas", "gas_fault_injection_manifest.csv"),
        ("skab", "skab_fault_injection_manifest.csv"),
    ]:
        path = result_dir / filename
        if path.exists():
            frame = pd.read_csv(path)
            frame["dataset_short"] = dataset
            manifests.append(frame)
    if manifests:
        combined_manifest = pd.concat(manifests, ignore_index=True)
        severity_summary = (
            combined_manifest.groupby(["dataset_short", "fault_type"], as_index=False)
            .agg(
                events=("event_id", "count"),
                mean_duration=("duration", "mean"),
                mean_severity=("severity", "mean"),
                minimum_severity=("severity", "min"),
                maximum_severity=("severity", "max"),
                mean_affected_channels=("affected_channel_count", "mean"),
            )
        )
    else:
        severity_summary = pd.DataFrame(
            columns=[
                "dataset_short",
                "fault_type",
                "events",
                "mean_duration",
                "mean_severity",
                "minimum_severity",
                "maximum_severity",
                "mean_affected_channels",
            ]
        )
    severity_path = result_dir / "cross_domain_fault_scenario_summary.csv"
    atomic_write_dataframe(severity_summary, severity_path)

    return {
        "benchmark_summary": summary_path,
        "fault_coverage": fault_coverage_path,
        "fault_scenarios": severity_path,
        **figure_paths,
    }


def write_title_alignment_checklist(config: dict[str, Any]) -> Path:
    result_dir = _data_result_dir(config)
    configured_seeds = list(config["project"].get("seed_list", []))
    profile = str(config.get("experiment", {}).get("profile", "manuscript"))
    grid = config["data"]["fault_injection"]["scenario_grid"].get(profile, {})
    scenario_count = (
        len(config["data"]["fault_injection"]["enabled_faults"])
        * len(grid.get("severities", []))
        * len(grid.get("duration_windows", []))
        * int(grid.get("replicates", 0))
    )
    rows = [
        {
            "requirement": "Temporal or sequence-aware data splitting",
            "status": "implemented",
            "current_evidence": "Gas batches are split chronologically; SKAB normal data are split sequentially.",
            "recommended_code_action": "Retain the chronological split in all manuscript experiments.",
        },
        {
            "requirement": "Context adjustment for legitimate operating-condition changes",
            "status": "implemented_for_gas",
            "current_evidence": "Gas identity and concentration are modeled before drift scoring.",
            "recommended_code_action": "Report native SKAB process anomalies separately from controlled measurement drift.",
        },
        {
            "requirement": "Shared controlled fault protocol across domains",
            "status": "implemented",
            "current_evidence": f"The {profile} profile defines {scenario_count} shared scenario IDs with fixed severity, duration, replicate, start, and channel-fraction factors.",
            "recommended_code_action": "Use scenario-macro metrics as the primary controlled-drift results.",
        },
        {
            "requirement": "Realistic mixed-fault prevalence",
            "status": "implemented",
            "current_evidence": "The mixed benchmark targets approximately 10% point-level anomaly prevalence.",
            "recommended_code_action": "Report the achieved prevalence for each domain.",
        },
        {
            "requirement": "Natural and native anomaly validation",
            "status": "implemented_pending_execution",
            "current_evidence": "Natural Gas batch scoring and native SKAB sequence evaluation are implemented in experiments_v2.py.",
            "recommended_code_action": "Complete all seeds and report these evidence layers separately.",
        },
        {
            "requirement": "Cross-domain representation transfer",
            "status": "implemented_pending_execution",
            "current_evidence": "Shared temporal blocks can be transferred in both directions with target-data fractions from 1% to 100%.",
            "recommended_code_action": "Compare paired scratch and transfer runs for every seed and target fraction.",
        },
        {
            "requirement": "Robustness across random seeds",
            "status": "configured",
            "current_evidence": f"Configured independent seeds: {configured_seeds}.",
            "recommended_code_action": "Do not aggregate repeated executions of the same seed as independent runs.",
        },
        {
            "requirement": "Sensitivity to drift severity and duration",
            "status": "implemented",
            "current_evidence": "A fixed severity-duration grid is generated for both domains.",
            "recommended_code_action": "Report performance by fault type, severity, and duration.",
        },
        {
            "requirement": "Sensor localization evaluation",
            "status": "implemented",
            "current_evidence": "Controlled benchmarks include exact time-by-channel masks and LCAD-AE exports per-channel scores.",
            "recommended_code_action": "Report channel AUPRC, top-1 accuracy, and top-k channel recall.",
        },
        {
            "requirement": "Lightweight-model evidence and ablation",
            "status": "implemented_pending_execution",
            "current_evidence": "Parameter count, model size, latency, and four component variants are configured.",
            "recommended_code_action": "Complete the full five-seed ablation before finalizing the Discussion.",
        },
        {
            "requirement": "Statistical comparison",
            "status": "implemented_pending_execution",
            "current_evidence": "Bootstrap confidence intervals and paired Wilcoxon tests with Holm correction are implemented.",
            "recommended_code_action": "Run Notebook 04 only after all required model groups complete.",
        },
    ]
    path = result_dir / "research_title_alignment_checklist.csv"
    atomic_write_dataframe(pd.DataFrame(rows), path)
    return path


def write_dataset_analysis_report(config: dict[str, Any]) -> Path:
    result_dir = _data_result_dir(config)
    gas_quality = pd.read_csv(result_dir / "gas_data_quality_summary.csv").iloc[0]
    gas_drift = pd.read_csv(result_dir / "gas_natural_drift_by_batch.csv")
    skab_quality = pd.read_csv(result_dir / "skab_data_quality_summary.csv").iloc[0]
    skab_shift = pd.read_csv(result_dir / "skab_sequence_distribution_shift.csv")
    benchmark = pd.read_csv(result_dir / "cross_domain_benchmark_summary.csv")

    earliest = gas_drift.sort_values("batch").iloc[0]
    latest = gas_drift.sort_values("batch").iloc[-1]
    strongest = gas_drift.sort_values("mean_absolute_residual", ascending=False).iloc[0]
    top_skab = skab_shift.sort_values("correlation_distance_from_normal", ascending=False).head(3)

    lines = [
        "# Dataset Analysis Summary",
        "",
        "## Gas Sensor domain",
        "",
        f"- The dataset contains {int(gas_quality['rows'])} observations, {int(gas_quality['channels'])} engineered channels, and {int(gas_quality['batches'])} temporal batches.",
        f"- After gas identity and concentration adjustment, the mean absolute residual changes from {earliest['mean_absolute_residual']:.4f} in Batch {int(earliest['batch'])} to {latest['mean_absolute_residual']:.4f} in Batch {int(latest['batch'])}.",
        f"- Batch {int(strongest['batch'])} has the largest mean absolute residual ({strongest['mean_absolute_residual']:.4f}) and should be treated as the strongest natural-shift period, not automatically as a labeled fault period.",
        "- The batch-composition table must be examined together with the residual drift index because gas-class imbalance can otherwise be mistaken for instrumental drift.",
        "",
        "## SKAB domain",
        "",
        f"- The parsed SKAB collection contains {int(skab_quality['rows'])} observations across {int(skab_quality['normal_files'])} normal files and {int(skab_quality['anomaly_files'])} files containing operating disturbances or faults.",
        f"- Native labels contain {int(skab_quality['anomaly_points'])} anomaly points and {int(skab_quality['changepoints'])} changepoints.",
        "- The sequences with the largest correlation departure from the normal reference are:",
    ]
    for _, row in top_skab.iterrows():
        lines.append(
            f"  - `{row['source_file']}`: correlation distance {row['correlation_distance_from_normal']:.4f}, anomaly ratio {row['anomaly_ratio']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Cross-domain benchmark",
            "",
        ]
    )
    for _, row in benchmark.iterrows():
        lines.append(
            f"- {row['dataset']}: {int(row['channels'])} channels, {int(row['train_windows'])} training windows, {int(row['test_windows'])} controlled-test windows, and an anomaly-point ratio of {row['anomaly_ratio']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Natural later-batch shift in the Gas Sensor dataset and native SKAB anomalies are observational evidence. Controlled injected faults provide exact onset, type, duration, severity, and affected-channel ground truth. These two evidence sources must be reported separately in the manuscript.",
        ]
    )
    return _write_markdown(result_dir / "dataset_analysis_summary.md", lines)


def run_dataset_eda(config: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    """Run reproducible dataset visualization and diagnostics for both domains."""
    outputs: dict[str, Any] = {
        "gas": analyze_gas_dataset(config, seed=seed),
        "skab": analyze_skab_dataset(config, seed=seed),
        "controlled": analyze_controlled_benchmarks(config),
    }
    outputs["title_alignment_checklist"] = write_title_alignment_checklist(config)
    outputs["analysis_report"] = write_dataset_analysis_report(config)
    return outputs
