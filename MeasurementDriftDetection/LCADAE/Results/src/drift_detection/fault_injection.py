from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

FAULT_TYPES = ["additive", "proportional", "gradual", "noise", "stuck", "common_mode"]


@dataclass
class FaultEvent:
    event_id: int
    fault_type: str
    start: int
    end: int
    severity: float
    channels: list[int]
    scenario_id: str = ""


def _non_overlapping_start(
    rng: np.random.Generator,
    n_points: int,
    duration: int,
    occupied: np.ndarray,
    margin: int,
    attempts: int = 1000,
) -> int | None:
    upper = n_points - duration
    if upper <= 0:
        return None
    for _ in range(attempts):
        start = int(rng.integers(0, upper + 1))
        left = max(0, start - margin)
        right = min(n_points, start + duration + margin)
        if not occupied[left:right].any():
            return start
    return None


def _apply_fault(
    x: np.ndarray,
    start: int,
    end: int,
    channels: list[int],
    fault_type: str,
    severity: float,
    rng: np.random.Generator,
) -> list[int]:
    duration = end - start
    selected = list(channels)
    segment = x[start:end, selected].copy()
    channel_count = len(selected)
    if fault_type == "additive":
        signs = rng.choice([-1.0, 1.0], size=(1, channel_count)).astype(np.float32)
        x[start:end, selected] = segment + signs * severity
    elif fault_type == "proportional":
        signs = rng.choice([-1.0, 1.0], size=(1, channel_count)).astype(np.float32)
        scale = 1.0 + signs * min(0.75, 0.15 * severity)
        x[start:end, selected] = segment * scale
    elif fault_type == "gradual":
        ramp = np.linspace(0.0, severity, duration, dtype=np.float32)[:, None]
        signs = rng.choice([-1.0, 1.0], size=(1, channel_count)).astype(np.float32)
        x[start:end, selected] = segment + ramp * signs
    elif fault_type == "noise":
        noise = rng.normal(0.0, severity, size=segment.shape).astype(np.float32)
        x[start:end, selected] = segment + noise
    elif fault_type == "stuck":
        anchor = x[max(0, start - 1), selected]
        x[start:end, selected] = np.repeat(anchor[None, :], duration, axis=0)
    elif fault_type == "common_mode":
        sign = float(rng.choice([-1.0, 1.0]))
        x[start:end, selected] = segment + sign * severity
    else:
        raise KeyError(f"Unsupported fault type: {fault_type}")
    return selected


def inject_faults(
    series: np.ndarray,
    seed: int,
    enabled_faults: Iterable[str] = FAULT_TYPES,
    event_count: int = 6,
    min_duration: int = 16,
    max_duration: int = 32,
    min_severity: float = 0.75,
    max_severity: float = 2.5,
    channel_fraction_min: float = 0.08,
    channel_fraction_max: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[FaultEvent]]:
    """Inject a sparse mixed-fault benchmark into a standardized normal series.

    Returns the injected series, point labels, fault labels, event IDs,
    point-by-channel ground-truth mask, and event metadata.
    """
    x = np.asarray(series, dtype=np.float32).copy()
    if x.ndim != 2:
        raise ValueError("series must have shape [time, channels]")
    n_points, n_channels = x.shape
    if n_points < max(32, min_duration * 2):
        raise ValueError("The held-out sequence is too short for controlled fault injection")
    rng = np.random.default_rng(seed)
    labels = np.zeros(n_points, dtype=np.int8)
    event_ids = np.zeros(n_points, dtype=np.int32)
    fault_types = np.full(n_points, "normal", dtype="<U24")
    channel_labels = np.zeros((n_points, n_channels), dtype=np.int8)
    occupied = np.zeros(n_points, dtype=bool)
    events: list[FaultEvent] = []
    fault_list = [fault for fault in enabled_faults if fault in FAULT_TYPES]
    if not fault_list:
        raise ValueError("No supported fault types were enabled")
    for event_index in range(int(event_count)):
        duration = int(rng.integers(min_duration, max_duration + 1))
        duration = min(duration, max(4, n_points // max(2, event_count)))
        start = _non_overlapping_start(rng, n_points, duration, occupied, margin=max(2, min_duration // 4))
        if start is None:
            break
        end = start + duration
        fraction = float(rng.uniform(channel_fraction_min, channel_fraction_max))
        channel_count = int(np.clip(round(n_channels * fraction), 1, n_channels))
        fault_type = fault_list[event_index % len(fault_list)]
        if fault_type == "common_mode":
            channel_count = max(channel_count, int(np.ceil(n_channels * 0.5)))
        channels = sorted(rng.choice(n_channels, size=min(channel_count, n_channels), replace=False).tolist())
        severity = float(rng.uniform(min_severity, max_severity))
        _apply_fault(x, start, end, channels, fault_type, severity, rng)
        labels[start:end] = 1
        event_ids[start:end] = event_index + 1
        fault_types[start:end] = fault_type
        channel_labels[start:end, channels] = 1
        occupied[start:end] = True
        events.append(FaultEvent(event_index + 1, fault_type, start, end, severity, channels))
    return x, labels, fault_types, event_ids, channel_labels, events


def build_scenario_protocol(config: dict[str, Any], profile: str | None = None, seed: int = 2026) -> pd.DataFrame:
    """Build a domain-independent severity-duration protocol.

    The same scenario IDs, severities, durations, starts, and channel fractions
    are reused in both domains. Domain-specific point indices and channel IDs are
    resolved only when a scenario is applied.
    """
    profile_name = profile or str(config.get("experiment", {}).get("profile", "manuscript"))
    grid_root = config["data"]["fault_injection"]["scenario_grid"]
    if profile_name not in grid_root:
        raise KeyError(f"Unknown scenario profile: {profile_name}")
    grid = grid_root[profile_name]
    severities = [float(value) for value in grid["severities"]]
    durations = [int(value) for value in grid["duration_windows"]]
    replicates = int(grid["replicates"])
    fractions = [float(value) for value in grid_root["channel_fractions"]]
    start_min = float(grid_root.get("normalized_start_min", 0.05))
    start_max = float(grid_root.get("normalized_start_max", 0.85))
    fault_types = list(config["data"]["fault_injection"]["enabled_faults"])
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for fault_type in fault_types:
        for severity in severities:
            for duration_windows in durations:
                for replicate in range(replicates):
                    fraction = fractions[replicate % len(fractions)]
                    if fault_type == "common_mode":
                        fraction = max(0.50, fraction)
                    scenario_id = f"{fault_type}_s{severity:g}_d{duration_windows}_r{replicate}"
                    rows.append(
                        {
                            "scenario_id": scenario_id,
                            "fault_type": fault_type,
                            "severity": severity,
                            "duration_windows": duration_windows,
                            "replicate": replicate,
                            "channel_fraction": fraction,
                            "normalized_start": float(rng.uniform(start_min, start_max)),
                            "scenario_seed": int(rng.integers(1, 2**31 - 1)),
                            "profile": profile_name,
                        }
                    )
    return pd.DataFrame(rows)


def apply_fault_scenario(
    clean_series: np.ndarray,
    scenario: dict[str, Any] | pd.Series,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, FaultEvent]:
    """Apply one independent controlled scenario to a clean held-out series."""
    row = dict(scenario)
    x = np.asarray(clean_series, dtype=np.float32).copy()
    if x.ndim != 2:
        raise ValueError("clean_series must have shape [time, channels]")
    n_points, n_channels = x.shape
    duration = max(2, int(row["duration_windows"]) * int(window_size))
    duration = min(duration, max(2, n_points - 1))
    max_start = max(0, n_points - duration)
    start = int(round(float(row["normalized_start"]) * max_start))
    start = int(np.clip(start, 0, max_start))
    end = start + duration
    fraction = float(row["channel_fraction"])
    channel_count = int(np.clip(round(n_channels * fraction), 1, n_channels))
    if str(row["fault_type"]) == "common_mode":
        channel_count = max(channel_count, int(np.ceil(n_channels * 0.5)))
    rng = np.random.default_rng(int(row["scenario_seed"]))
    channels = sorted(rng.choice(n_channels, size=min(channel_count, n_channels), replace=False).tolist())
    _apply_fault(x, start, end, channels, str(row["fault_type"]), float(row["severity"]), rng)
    labels = np.zeros(n_points, dtype=np.int8)
    labels[start:end] = 1
    fault_types = np.full(n_points, "normal", dtype="<U24")
    fault_types[start:end] = str(row["fault_type"])
    event_ids = np.zeros(n_points, dtype=np.int32)
    event_ids[start:end] = 1
    channel_labels = np.zeros((n_points, n_channels), dtype=np.int8)
    channel_labels[start:end, channels] = 1
    event = FaultEvent(
        event_id=1,
        fault_type=str(row["fault_type"]),
        start=start,
        end=end,
        severity=float(row["severity"]),
        channels=channels,
        scenario_id=str(row["scenario_id"]),
    )
    return x, labels, fault_types, event_ids, channel_labels, event
