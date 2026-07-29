from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .datasets import SKAB_SENSOR_COLUMNS, load_gas_dataset, load_skab_sequences, skab_summary
from .fault_injection import FaultEvent, build_scenario_protocol, inject_faults
from .io_utils import atomic_write_dataframe, safe_npz, save_json
from .preprocessing import GasContextResidualizer, RobustSeriesScaler, feature_columns


def _events_to_frame(events: list[FaultEvent], dataset: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "scenario_id": event.scenario_id,
                "event_id": event.event_id,
                "fault_type": event.fault_type,
                "start": event.start,
                "end": event.end,
                "duration": event.end - event.start,
                "severity": event.severity,
                "affected_channel_count": len(event.channels),
                "affected_channels": ",".join(map(str, event.channels)),
            }
            for event in events
        ]
    )


def _inject_main_benchmark(series: np.ndarray, seed: int, config: dict[str, Any]):
    fault = config["data"]["fault_injection"]
    event_count = int(fault.get("main_event_count", len(fault["enabled_faults"])))
    target_fraction = float(fault.get("main_target_anomaly_fraction", 0.10))
    duration = max(4, int(round(len(series) * target_fraction / max(1, event_count))))
    min_duration = max(4, int(round(duration * 0.80)))
    max_duration = max(min_duration, int(round(duration * 1.20)))
    return inject_faults(
        series,
        seed=seed,
        enabled_faults=fault["enabled_faults"],
        event_count=event_count,
        min_duration=min_duration,
        max_duration=max_duration,
        min_severity=float(fault.get("main_min_severity", 0.75)),
        max_severity=float(fault.get("main_max_severity", 2.5)),
        channel_fraction_min=float(fault["channel_fraction_min"]),
        channel_fraction_max=float(fault["channel_fraction_max"]),
    )


def _save_scenario_manifest(
    config: dict[str, Any],
    dataset: str,
    clean_series: np.ndarray,
    channel_names: list[str] | np.ndarray,
    seed: int,
) -> Path:
    profile = str(config.get("experiment", {}).get("profile", "manuscript"))
    protocol = build_scenario_protocol(config, profile=profile, seed=seed + 1984)
    protocol.insert(0, "dataset", dataset)
    protocol["clean_test_points"] = int(len(clean_series))
    protocol["channel_count"] = int(clean_series.shape[1])
    protocol["window_size"] = int(config["data"]["window_size"])
    protocol["stride"] = int(config["data"]["stride"])
    protocol["channel_names"] = ",".join(map(str, channel_names))
    path = Path(config["paths"]["benchmarks"]) / f"{dataset}_scenario_manifest.csv"
    atomic_write_dataframe(protocol, path)
    return path


def prepare_gas_benchmark(config: dict[str, Any], seed: int = 42) -> dict[str, Path]:
    paths = config["paths"]
    settings = config["data"]["gas"]
    gas = load_gas_dataset(paths["raw_gas"])
    features = feature_columns(gas)
    train = gas[gas["batch"].isin(settings["train_batches"])].copy()
    validation_batch = gas[gas["batch"] == int(settings["validation_batch"])].copy()
    fraction = float(settings.get("validation_fraction_within_batch", 0.30))
    split_index = max(1, min(len(validation_batch) - 1, int(len(validation_batch) * fraction)))
    validation = validation_batch.iloc[:split_index].copy()
    controlled_frames = [validation_batch.iloc[split_index:].copy()]
    for batch in settings.get("controlled_test_batches", []):
        if int(batch) != int(settings["validation_batch"]):
            controlled_frames.append(gas[gas["batch"] == int(batch)].copy())
    controlled_base = pd.concat(controlled_frames, ignore_index=True)
    natural = gas[gas["batch"].isin(settings["natural_test_batches"])].copy()

    residualizer = GasContextResidualizer.fit(train, features, ridge_alpha=float(settings.get("ridge_alpha", 1.0)))
    train_series = residualizer.transform(train)
    validation_series = residualizer.transform(validation)
    controlled_series = residualizer.transform(controlled_base)
    natural_series = (
        residualizer.transform(natural)
        if len(natural)
        else np.empty((0, len(features)), dtype=np.float32)
    )

    injected, labels, fault_types, event_ids, channel_labels, events = _inject_main_benchmark(
        controlled_series, seed, config
    )
    benchmark_path = Path(paths["benchmarks"]) / "gas_controlled.npz"
    safe_npz(
        benchmark_path,
        train_series=train_series,
        validation_series=validation_series,
        test_series=injected,
        clean_test_series=controlled_series,
        test_labels=labels,
        test_fault_types=fault_types,
        test_event_ids=event_ids,
        test_channel_labels=channel_labels,
        channel_names=np.asarray(features, dtype="<U32"),
        domain=np.asarray(["gas_sensor"], dtype="<U32"),
        benchmark_version=np.asarray([config["project"]["experiment_version"]], dtype="<U32"),
    )
    natural_path = Path(paths["benchmarks"]) / "gas_natural_later_batches.npz"
    safe_npz(
        natural_path,
        test_series=natural_series,
        batches=natural["batch"].to_numpy(dtype=np.int16),
        gas_codes=natural["gas_code"].to_numpy(dtype=np.int8),
        concentrations=natural["concentration"].to_numpy(dtype=np.float32),
        channel_names=np.asarray(features, dtype="<U32"),
    )
    model_path = Path(paths["processed"]) / "gas_context_residualizer.joblib"
    joblib.dump(residualizer, model_path)
    event_path = Path(paths["results"]) / "data" / "gas_fault_injection_manifest.csv"
    atomic_write_dataframe(_events_to_frame(events, "gas_sensor"), event_path)
    scenario_path = _save_scenario_manifest(config, "gas_sensor", controlled_series, features, seed)
    summary = pd.DataFrame(
        [
            {"dataset": "gas_sensor", "split": "train", "rows": len(train), "channels": len(features)},
            {"dataset": "gas_sensor", "split": "validation", "rows": len(validation), "channels": len(features)},
            {"dataset": "gas_sensor", "split": "controlled_test", "rows": len(controlled_base), "channels": len(features)},
            {"dataset": "gas_sensor", "split": "natural_later_batches", "rows": len(natural), "channels": len(features)},
        ]
    )
    summary["main_anomaly_ratio"] = float(labels.mean())
    summary_path = Path(paths["results"]) / "data" / "gas_preparation_summary.csv"
    atomic_write_dataframe(summary, summary_path)
    return {
        "benchmark": benchmark_path,
        "natural": natural_path,
        "scenario_manifest": scenario_path,
        "residualizer": model_path,
        "event_manifest": event_path,
        "summary": summary_path,
    }


def _normal_training_series(normal_sequences: list[pd.DataFrame], anomaly_sequences: list[pd.DataFrame]) -> np.ndarray:
    if normal_sequences:
        return np.concatenate(
            [frame[SKAB_SENSOR_COLUMNS].to_numpy(dtype=np.float32) for frame in normal_sequences], axis=0
        )
    prefixes = []
    for frame in anomaly_sequences:
        anomaly_positions = np.flatnonzero(
            (frame["anomaly"].to_numpy() > 0) | (frame["changepoint"].to_numpy() > 0)
        )
        stop = int(anomaly_positions[0]) if len(anomaly_positions) else len(frame)
        if stop > 20:
            prefixes.append(frame.iloc[:stop][SKAB_SENSOR_COLUMNS].to_numpy(dtype=np.float32))
    if not prefixes:
        raise ValueError("Unable to derive normal SKAB training data")
    return np.concatenate(prefixes, axis=0)


def prepare_skab_benchmark(config: dict[str, Any], seed: int = 42) -> dict[str, Path]:
    paths = config["paths"]
    settings = config["data"]["skab"]
    normal_sequences, anomaly_sequences = load_skab_sequences(paths["raw_skab"])
    normal = _normal_training_series(normal_sequences, anomaly_sequences)
    n = len(normal)
    train_end = max(1, int(n * float(settings["train_fraction"])))
    validation_end = max(
        train_end + 1,
        int(n * (float(settings["train_fraction"]) + float(settings["validation_fraction"]))),
    )
    validation_end = min(validation_end, n - 1)
    raw_train = normal[:train_end]
    raw_validation = normal[train_end:validation_end]
    raw_controlled = normal[validation_end:]
    scaler = RobustSeriesScaler.fit(raw_train)
    train_series = scaler.transform(raw_train)
    validation_series = scaler.transform(raw_validation)
    controlled_series = scaler.transform(raw_controlled)
    injected, labels, fault_types, event_ids, channel_labels, events = _inject_main_benchmark(
        controlled_series, seed, config
    )

    benchmark_path = Path(paths["benchmarks"]) / "skab_controlled.npz"
    safe_npz(
        benchmark_path,
        train_series=train_series,
        validation_series=validation_series,
        test_series=injected,
        clean_test_series=controlled_series,
        test_labels=labels,
        test_fault_types=fault_types,
        test_event_ids=event_ids,
        test_channel_labels=channel_labels,
        channel_names=np.asarray(SKAB_SENSOR_COLUMNS, dtype="<U32"),
        domain=np.asarray(["skab"], dtype="<U32"),
        benchmark_version=np.asarray([config["project"]["experiment_version"]], dtype="<U32"),
    )
    scaler_path = Path(paths["processed"]) / "skab_robust_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    event_path = Path(paths["results"]) / "data" / "skab_fault_injection_manifest.csv"
    atomic_write_dataframe(_events_to_frame(events, "skab"), event_path)
    scenario_path = _save_scenario_manifest(config, "skab", controlled_series, SKAB_SENSOR_COLUMNS, seed)

    native_dir = Path(paths["processed"]) / "benchmarks" / "skab_native"
    native_dir.mkdir(parents=True, exist_ok=True)
    native_records = []
    for index, frame in enumerate(anomaly_sequences):
        values = scaler.transform(frame[SKAB_SENSOR_COLUMNS].to_numpy(dtype=np.float32))
        labels_native = frame["anomaly"].fillna(0).to_numpy(dtype=np.int8)
        changepoints = frame["changepoint"].fillna(0).to_numpy(dtype=np.int8)
        source = Path(str(frame["source_file"].iloc[0]))
        output = native_dir / f"{index:03d}_{source.stem}.npz"
        safe_npz(
            output,
            test_series=values,
            test_labels=labels_native,
            test_changepoints=changepoints,
            channel_names=np.asarray(SKAB_SENSOR_COLUMNS, dtype="<U32"),
        )
        native_records.append(
            {
                "sequence_id": index,
                "source_file": str(source),
                "benchmark_file": str(output),
                "rows": len(frame),
                "anomaly_points": int(labels_native.sum()),
                "changepoints": int(changepoints.sum()),
            }
        )
    native_manifest = Path(paths["results"]) / "data" / "skab_native_manifest.csv"
    atomic_write_dataframe(pd.DataFrame(native_records), native_manifest)
    raw_summary = skab_summary(paths["raw_skab"])
    raw_summary_path = Path(paths["results"]) / "data" / "skab_raw_file_summary.csv"
    atomic_write_dataframe(raw_summary, raw_summary_path)
    summary = pd.DataFrame(
        [
            {"dataset": "skab", "split": "train", "rows": len(raw_train), "channels": len(SKAB_SENSOR_COLUMNS)},
            {"dataset": "skab", "split": "validation", "rows": len(raw_validation), "channels": len(SKAB_SENSOR_COLUMNS)},
            {"dataset": "skab", "split": "controlled_test", "rows": len(raw_controlled), "channels": len(SKAB_SENSOR_COLUMNS)},
            {"dataset": "skab", "split": "native_sequences", "rows": sum(len(frame) for frame in anomaly_sequences), "channels": len(SKAB_SENSOR_COLUMNS)},
        ]
    )
    summary["main_anomaly_ratio"] = float(labels.mean())
    summary_path = Path(paths["results"]) / "data" / "skab_preparation_summary.csv"
    atomic_write_dataframe(summary, summary_path)
    return {
        "benchmark": benchmark_path,
        "scenario_manifest": scenario_path,
        "scaler": scaler_path,
        "event_manifest": event_path,
        "native_manifest": native_manifest,
        "raw_summary": raw_summary_path,
        "summary": summary_path,
    }


def prepare_all_benchmarks(config: dict[str, Any], seed: int = 42) -> dict[str, dict[str, Path]]:
    output = {
        "gas": prepare_gas_benchmark(config, seed=seed),
        "skab": prepare_skab_benchmark(config, seed=seed),
    }
    save_json(Path(config["paths"]["results"]) / "data" / "preparation_outputs.json", output)
    return output
