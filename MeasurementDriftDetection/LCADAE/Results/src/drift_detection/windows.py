from __future__ import annotations

import numpy as np


def make_windows(
    series: np.ndarray,
    window_size: int,
    stride: int,
    labels: np.ndarray | None = None,
    fault_types: np.ndarray | None = None,
    event_ids: np.ndarray | None = None,
    channel_labels: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Create end-aligned sliding windows and optional window-level labels."""
    x = np.asarray(series, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("series must have shape [time, channels]")
    if len(x) < window_size:
        raise ValueError(f"Series length {len(x)} is smaller than window_size {window_size}")
    starts = np.arange(0, len(x) - window_size + 1, stride, dtype=np.int64)
    windows = np.stack([x[start : start + window_size] for start in starts]).astype(np.float32)
    ends = starts + window_size - 1
    output: dict[str, np.ndarray] = {"windows": windows, "start_indices": starts, "end_indices": ends}
    if labels is not None:
        y = np.asarray(labels)
        output["labels"] = np.asarray([int(y[start : start + window_size].max()) for start in starts], dtype=np.int8)
    if fault_types is not None:
        values = np.asarray(fault_types)
        window_faults: list[str] = []
        for start in starts:
            segment = values[start : start + window_size]
            non_normal = [str(value) for value in segment if str(value) != "normal"]
            window_faults.append(non_normal[0] if non_normal else "normal")
        output["fault_types"] = np.asarray(window_faults, dtype="<U24")
    if event_ids is not None:
        values = np.asarray(event_ids)
        window_events = []
        for start in starts:
            segment = values[start : start + window_size]
            non_zero = segment[segment > 0]
            window_events.append(int(non_zero[0]) if len(non_zero) else 0)
        output["event_ids"] = np.asarray(window_events, dtype=np.int32)
    if channel_labels is not None:
        values = np.asarray(channel_labels, dtype=np.int8)
        if values.shape != x.shape:
            raise ValueError("channel_labels must have shape [time, channels]")
        output["channel_labels"] = np.stack(
            [values[start : start + window_size].max(axis=0) for start in starts]
        ).astype(np.int8)
    return output


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    return np.asarray(windows, dtype=np.float32).reshape(len(windows), -1)
