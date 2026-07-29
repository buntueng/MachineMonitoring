from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .fault_injection import apply_fault_scenario
from .metrics import classification_metrics, localization_metrics
from .windows import make_windows

ScoreWindows = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray | None]]
ScoreSeries = Callable[[np.ndarray, np.ndarray], np.ndarray]


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _macro_summary(frame: pd.DataFrame) -> dict[str, float]:
    metrics = [
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "event_recall",
        "mean_detection_delay_windows",
        "false_positive_rate",
        "false_alarm_segments",
        "channel_auprc",
        "channel_auroc",
        "top1_localization_accuracy",
        "topk_channel_recall",
    ]
    return {
        metric: float(pd.to_numeric(frame[metric], errors="coerce").mean())
        for metric in metrics
        if metric in frame.columns
    }


def evaluate_mixed_benchmark(
    bundle: dict[str, np.ndarray],
    score_windows: ScoreWindows,
    threshold: float,
    window_size: int,
    stride: int,
    score_series: ScoreSeries | None = None,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray | None]:
    windows = make_windows(
        bundle["test_series"],
        window_size,
        stride,
        labels=bundle["test_labels"],
        fault_types=bundle["test_fault_types"],
        event_ids=bundle["test_event_ids"],
        channel_labels=bundle.get("test_channel_labels"),
    )
    if score_series is not None:
        scores = score_series(bundle["test_series"], windows["end_indices"])
        channel_scores = None
    else:
        scores, channel_scores = score_windows(windows["windows"])
    metrics = classification_metrics(windows["labels"], scores, threshold, event_ids=windows["event_ids"])
    if channel_scores is not None and "channel_labels" in windows:
        metrics.update(localization_metrics(windows["channel_labels"], channel_scores))
    score_frame = pd.DataFrame(
        {
            "window_index": np.arange(len(scores)),
            "start_index": windows["start_indices"],
            "end_index": windows["end_indices"],
            "label": windows["labels"],
            "fault_type": windows["fault_types"],
            "event_id": windows["event_ids"],
            "score": scores,
            "threshold": threshold,
            "prediction": (scores >= threshold).astype(int),
        }
    )
    return metrics, score_frame, channel_scores


def evaluate_scenario_grid(
    clean_series: np.ndarray,
    manifest: pd.DataFrame,
    score_windows: ScoreWindows,
    thresholds: dict[str, float],
    window_size: int,
    stride: int,
    score_series: ScoreSeries | None = None,
    representative_scores_per_fault: bool = True,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate independently generated scenarios with one fixed trained model."""
    primary_threshold = float(thresholds["primary"])
    scenario_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    representative_frames: list[pd.DataFrame] = []
    cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    represented_faults: set[str] = set()

    for _, scenario in manifest.iterrows():
        injected, labels, fault_types, event_ids, channel_labels, event = apply_fault_scenario(
            clean_series, scenario, window_size
        )
        windows = make_windows(
            injected,
            window_size,
            stride,
            labels=labels,
            fault_types=fault_types,
            event_ids=event_ids,
            channel_labels=channel_labels,
        )
        if score_series is not None:
            scores = score_series(injected, windows["end_indices"])
            channel_scores = None
        else:
            scores, channel_scores = score_windows(windows["windows"])
        metrics = classification_metrics(windows["labels"], scores, primary_threshold, windows["event_ids"])
        if channel_scores is not None:
            metrics.update(localization_metrics(windows["channel_labels"], channel_scores))
        row = {**scenario.to_dict(), **metrics}
        row.update(
            {
                "event_start": event.start,
                "event_end": event.end,
                "duration_points": event.end - event.start,
                "affected_channel_count": len(event.channels),
                "affected_channels": ",".join(map(str, event.channels)),
                "anomalous_window_count": int(windows["labels"].sum()),
                "total_window_count": int(len(windows["labels"])),
            }
        )
        scenario_rows.append(row)
        cache.append((windows["labels"].copy(), windows["event_ids"].copy(), np.asarray(scores).copy()))
        fault = str(scenario["fault_type"])
        if representative_scores_per_fault and fault not in represented_faults:
            representative_frames.append(
                pd.DataFrame(
                    {
                        "scenario_id": str(scenario["scenario_id"]),
                        "fault_type": fault,
                        "window_index": np.arange(len(scores)),
                        "start_index": windows["start_indices"],
                        "end_index": windows["end_indices"],
                        "label": windows["labels"],
                        "score": scores,
                        "threshold": primary_threshold,
                        "prediction": (scores >= primary_threshold).astype(int),
                    }
                )
            )
            represented_faults.add(fault)

    scenario_frame = pd.DataFrame(scenario_rows)
    summary = _macro_summary(scenario_frame)
    summary["scenario_count"] = float(len(scenario_frame))
    summary["fault_type_count"] = float(scenario_frame["fault_type"].nunique())
    summary["severity_level_count"] = float(scenario_frame["severity"].nunique())
    summary["duration_level_count"] = float(scenario_frame["duration_windows"].nunique())

    for threshold_name, threshold in thresholds.items():
        if threshold_name == "primary":
            continue
        per_scenario: list[dict[str, float]] = []
        for labels_cached, event_ids_cached, scores_cached in cache:
            metric = classification_metrics(labels_cached, scores_cached, float(threshold), event_ids_cached)
            per_scenario.append(metric)
        temporary = pd.DataFrame(per_scenario)
        sensitivity_rows.append(
            {
                "threshold_name": threshold_name,
                "threshold": float(threshold),
                **_macro_summary(temporary),
                "scenario_count": len(temporary),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    representatives = pd.concat(representative_frames, ignore_index=True) if representative_frames else pd.DataFrame()
    return summary, scenario_frame, sensitivity, representatives


def evaluate_gas_natural(
    natural_path: str | Path,
    score_windows: ScoreWindows,
    threshold: float,
    validation_scores: np.ndarray,
    window_size: int,
    stride: int,
    score_series: ScoreSeries | None = None,
) -> pd.DataFrame:
    bundle = load_npz(natural_path)
    series = bundle["test_series"]
    if len(series) < window_size:
        return pd.DataFrame()
    windows = make_windows(series, window_size, stride)
    if score_series is not None:
        scores = score_series(series, windows["end_indices"])
    else:
        scores, _ = score_windows(windows["windows"])
    batches = bundle["batches"][windows["end_indices"]]
    reference_median = float(np.median(validation_scores))
    reference_iqr = float(np.quantile(validation_scores, 0.75) - np.quantile(validation_scores, 0.25))
    reference_iqr = max(reference_iqr, np.finfo(float).eps)
    frame = pd.DataFrame({"batch": batches, "score": scores, "exceeds_threshold": scores >= threshold})
    output = (
        frame.groupby("batch", as_index=False)
        .agg(
            window_count=("score", "size"),
            median_score=("score", "median"),
            mean_score=("score", "mean"),
            p95_score=("score", lambda values: float(np.quantile(values, 0.95))),
            threshold_exceedance_rate=("exceeds_threshold", "mean"),
        )
    )
    output["robust_shift_from_validation"] = (output["median_score"] - reference_median) / reference_iqr
    output["threshold"] = threshold
    return output


def evaluate_skab_native(
    native_manifest_path: str | Path,
    score_windows: ScoreWindows,
    threshold: float,
    window_size: int,
    stride: int,
    score_series: ScoreSeries | None = None,
) -> pd.DataFrame:
    manifest = pd.read_csv(native_manifest_path)
    rows: list[dict[str, Any]] = []
    for _, record in manifest.iterrows():
        path = Path(str(record["benchmark_file"]))
        if not path.exists():
            continue
        bundle = load_npz(path)
        if len(bundle["test_series"]) < window_size:
            continue
        windows = make_windows(
            bundle["test_series"],
            window_size,
            stride,
            labels=bundle["test_labels"],
        )
        if score_series is not None:
            scores = score_series(bundle["test_series"], windows["end_indices"])
        else:
            scores, _ = score_windows(windows["windows"])
        metrics = classification_metrics(windows["labels"], scores, threshold)
        rows.append(
            {
                "sequence_id": int(record["sequence_id"]),
                "source_file": str(record["source_file"]),
                "point_count": int(record["rows"]),
                "anomaly_points": int(record["anomaly_points"]),
                "changepoints": int(record["changepoints"]),
                **metrics,
            }
        )
    return pd.DataFrame(rows)
