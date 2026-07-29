from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_from_validation(scores: np.ndarray, quantile: float = 0.99) -> float:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No finite validation scores were available")
    return float(np.quantile(values, quantile))


def mad_threshold(scores: np.ndarray, multiplier: float = 6.0) -> float:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No finite validation scores were available")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * max(mad, np.finfo(float).eps)
    return median + float(multiplier) * robust_sigma


def threshold_candidates(scores: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    settings = config["training"]["threshold"]
    candidates: dict[str, float] = {}
    for quantile in settings.get("quantile_grid", [settings.get("primary_quantile", 0.99)]):
        candidates[f"quantile_{float(quantile):g}"] = threshold_from_validation(scores, float(quantile))
    for multiplier in settings.get("mad_multiplier_grid", []):
        candidates[f"mad_{float(multiplier):g}"] = mad_threshold(scores, float(multiplier))
    primary_method = str(settings.get("primary_method", "quantile"))
    if primary_method == "quantile":
        primary_name = f"quantile_{float(settings.get('primary_quantile', 0.99)):g}"
    elif primary_method == "mad":
        primary_name = f"mad_{float(settings.get('primary_mad_multiplier', 6.0)):g}"
        if primary_name not in candidates:
            candidates[primary_name] = mad_threshold(scores, float(settings.get("primary_mad_multiplier", 6.0)))
    else:
        raise KeyError(f"Unknown primary threshold method: {primary_method}")
    candidates = {"primary": candidates[primary_name], **candidates}
    return candidates


def _safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")


def _safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")


def event_metrics(labels: np.ndarray, predictions: np.ndarray, event_ids: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(labels).astype(int)
    pred = np.asarray(predictions).astype(int)
    if event_ids is None:
        event_ids = np.zeros_like(y, dtype=int)
        current = 0
        inside = False
        for index, value in enumerate(y):
            if value == 1 and not inside:
                current += 1
                inside = True
            elif value == 0:
                inside = False
            if value == 1:
                event_ids[index] = current
    event_ids = np.asarray(event_ids).astype(int)
    unique_events = [event for event in np.unique(event_ids) if event > 0]
    detected = 0
    delays: list[int] = []
    for event in unique_events:
        indices = np.flatnonzero(event_ids == event)
        if len(indices) == 0:
            continue
        predicted_indices = indices[pred[indices] == 1]
        if len(predicted_indices):
            detected += 1
            delays.append(int(predicted_indices[0] - indices[0]))
    false_alarm_segments = 0
    in_false_alarm = False
    for label, prediction in zip(y, pred):
        is_false_alarm = label == 0 and prediction == 1
        if is_false_alarm and not in_false_alarm:
            false_alarm_segments += 1
            in_false_alarm = True
        elif not is_false_alarm:
            in_false_alarm = False
    return {
        "event_count": float(len(unique_events)),
        "detected_event_count": float(detected),
        "event_recall": float(detected / len(unique_events)) if unique_events else float("nan"),
        "mean_detection_delay_windows": float(np.mean(delays)) if delays else float("nan"),
        "median_detection_delay_windows": float(np.median(delays)) if delays else float("nan"),
        "false_alarm_segments": float(false_alarm_segments),
    }


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    event_ids: np.ndarray | None = None,
) -> dict[str, float]:
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": _safe_roc_auc(y, s),
        "auprc": _safe_auprc(y, s),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_positive": float(tp),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "false_negative_rate": float(fn / max(1, fn + tp)),
    }
    metrics.update(event_metrics(y, pred, event_ids=event_ids))
    return metrics


def per_fault_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float, fault_types: np.ndarray) -> list[dict[str, Any]]:
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=float)
    faults = np.asarray(fault_types).astype(str)
    rows: list[dict[str, Any]] = []
    for fault in sorted(set(faults) - {"normal"}):
        mask = (faults == fault) | (faults == "normal")
        if mask.sum() == 0:
            continue
        row = {"fault_type": fault}
        row.update(classification_metrics(y[mask], s[mask], threshold))
        rows.append(row)
    return rows


def localization_metrics(channel_labels: np.ndarray, channel_scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(channel_labels, dtype=np.int8)
    scores = np.asarray(channel_scores, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError(f"channel label shape {labels.shape} does not match score shape {scores.shape}")
    anomalous_windows = labels.sum(axis=1) > 0
    flat_labels = labels[anomalous_windows].reshape(-1)
    flat_scores = scores[anomalous_windows].reshape(-1)
    if len(flat_labels) == 0:
        return {
            "channel_auprc": float("nan"),
            "channel_auroc": float("nan"),
            "top1_localization_accuracy": float("nan"),
            "topk_channel_recall": float("nan"),
            "localized_window_count": 0.0,
        }
    top1_hits: list[float] = []
    topk_recalls: list[float] = []
    for true_row, score_row in zip(labels[anomalous_windows], scores[anomalous_windows]):
        true_indices = np.flatnonzero(true_row > 0)
        if len(true_indices) == 0:
            continue
        top1 = int(np.argmax(score_row))
        top1_hits.append(float(top1 in set(true_indices.tolist())))
        predicted = np.argsort(score_row)[-len(true_indices):]
        overlap = len(set(predicted.tolist()) & set(true_indices.tolist()))
        topk_recalls.append(float(overlap / len(true_indices)))
    return {
        "channel_auprc": _safe_auprc(flat_labels, flat_scores),
        "channel_auroc": _safe_roc_auc(flat_labels, flat_scores),
        "top1_localization_accuracy": float(np.mean(top1_hits)) if top1_hits else float("nan"),
        "topk_channel_recall": float(np.mean(topk_recalls)) if topk_recalls else float("nan"),
        "localized_window_count": float(len(top1_hits)),
    }
