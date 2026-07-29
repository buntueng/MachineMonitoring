from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

GAS_NAMES = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene",
}

SKAB_SENSOR_COLUMNS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "RateRMS",
]

# SKAB files are not fully consistent with the column name documented in the
# repository README. In particular, the official CSV files commonly use
# ``Volume Flow RateRMS`` while the documentation and many downstream examples
# use ``RateRMS``. All accepted variants are converted to the canonical names
# above so the remaining project code can use one stable schema.
SKAB_COLUMN_ALIASES = {
    "datetime": ("datetime", "date time", "timestamp", "time"),
    "Accelerometer1RMS": (
        "Accelerometer1RMS",
        "Accelerometer 1 RMS",
        "Accelerometer_1_RMS",
    ),
    "Accelerometer2RMS": (
        "Accelerometer2RMS",
        "Accelerometer 2 RMS",
        "Accelerometer_2_RMS",
    ),
    "Current": ("Current",),
    "Pressure": ("Pressure",),
    "Temperature": ("Temperature",),
    "Thermocouple": ("Thermocouple",),
    "Voltage": ("Voltage",),
    "RateRMS": (
        "RateRMS",
        "Volume Flow RateRMS",
        "VolumeFlowRateRMS",
        "Volume_Flow_RateRMS",
        "Flow RateRMS",
        "FlowRateRMS",
    ),
    "anomaly": ("anomaly", "is_anomaly", "label"),
    "changepoint": ("changepoint", "change_point", "is_changepoint"),
}


def _normalise_column_name(name: object) -> str:
    """Return a punctuation-insensitive representation of a column name."""
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lstrip("\ufeff").lower())


_SKAB_ALIAS_LOOKUP = {
    _normalise_column_name(alias): canonical
    for canonical, aliases in SKAB_COLUMN_ALIASES.items()
    for alias in aliases
}


def _canonicalise_skab_columns(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Map known SKAB header variants to the project's canonical schema.

    The function intentionally raises an error when two source columns would
    map to the same canonical name, because silently choosing one could corrupt
    an experiment.
    """
    rename_map: dict[object, str] = {}
    target_to_source: dict[str, object] = {}

    for source in frame.columns:
        canonical = _SKAB_ALIAS_LOOKUP.get(_normalise_column_name(source))
        if canonical is None:
            continue
        previous = target_to_source.get(canonical)
        if previous is not None and previous != source:
            raise ValueError(
                f"Ambiguous SKAB columns in {path}: {previous!r} and {source!r} "
                f"both map to {canonical!r}."
            )
        target_to_source[canonical] = source
        if source != canonical:
            rename_map[source] = canonical

    return frame.rename(columns=rename_map)


def _find_gas_files(root: str | Path) -> list[Path]:
    paths = sorted(Path(root).rglob("batch*.dat"), key=lambda p: int(re.search(r"batch(\d+)", p.name.lower()).group(1)))
    if not paths:
        raise FileNotFoundError(
            f"No batch*.dat files were found under {root}. Run scripts/download_datasets.py --gas first."
        )
    return paths


def parse_gas_batch(path: str | Path, expected_features: int = 128) -> pd.DataFrame:
    """Parse one UCI gas-drift batch file.

    Each row starts with ``gas_code;concentration`` followed by sparse ``index:value`` features.
    """
    path = Path(path)
    batch_match = re.search(r"batch(\d+)", path.name.lower())
    if batch_match is None:
        raise ValueError(f"Cannot infer batch number from {path.name}")
    batch = int(batch_match.group(1))
    rows: list[dict[str, float | int | str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for row_index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            tokens = stripped.split()
            try:
                gas_token, concentration_token = tokens[0].split(";", maxsplit=1)
                gas_code = int(gas_token)
                concentration = float(concentration_token)
            except Exception as exc:
                raise ValueError(f"Invalid label/concentration token in {path} line {row_index + 1}") from exc
            features = np.zeros(expected_features, dtype=np.float32)
            for token in tokens[1:]:
                if ":" not in token:
                    continue
                index_token, value_token = token.split(":", maxsplit=1)
                feature_index = int(index_token) - 1
                if 0 <= feature_index < expected_features:
                    features[feature_index] = float(value_token)
            row: dict[str, float | int | str] = {
                "batch": batch,
                "row_in_batch": row_index,
                "gas_code": gas_code,
                "gas_name": GAS_NAMES.get(gas_code, f"Gas_{gas_code}"),
                "concentration": concentration,
            }
            row.update({f"f{index + 1:03d}": float(value) for index, value in enumerate(features)})
            rows.append(row)
    return pd.DataFrame(rows)


def load_gas_dataset(root: str | Path) -> pd.DataFrame:
    frames = [parse_gas_batch(path) for path in _find_gas_files(root)]
    return pd.concat(frames, ignore_index=True)


def _read_skab_csv(path: Path) -> pd.DataFrame:
    """Read SKAB CSV files with either semicolon or comma separators."""
    frame = pd.read_csv(path, sep=";")
    if frame.shape[1] == 1:
        frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lstrip("\ufeff") for column in frame.columns]
    frame = _canonicalise_skab_columns(frame, path)
    unnamed = [column for column in frame.columns if column.lower().startswith("unnamed")]
    if unnamed and "datetime" not in frame.columns:
        frame = frame.rename(columns={unnamed[0]: "datetime"})
    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    for column in SKAB_SENSOR_COLUMNS + ["anomaly", "changepoint"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    available = [column for column in SKAB_SENSOR_COLUMNS if column in frame.columns]
    if len(available) != len(SKAB_SENSOR_COLUMNS):
        missing = sorted(set(SKAB_SENSOR_COLUMNS) - set(available))
        raise ValueError(f"Missing expected SKAB sensor columns in {path}: {missing}")
    if "anomaly" not in frame.columns:
        frame["anomaly"] = 0
    if "changepoint" not in frame.columns:
        frame["changepoint"] = 0
    frame = frame.dropna(subset=SKAB_SENSOR_COLUMNS).reset_index(drop=True)
    frame["source_file"] = str(path)
    return frame


def discover_skab_files(root: str | Path) -> tuple[list[Path], list[Path]]:
    all_csv = sorted(Path(root).rglob("*.csv"))
    if not all_csv:
        raise FileNotFoundError(
            f"No SKAB CSV files were found under {root}. Run scripts/download_datasets.py --skab first."
        )
    normal_files: list[Path] = []
    anomaly_files: list[Path] = []
    for path in all_csv:
        lowered = str(path).lower().replace("_", "-")
        if "anomaly-free" in lowered or "anomalyfree" in lowered:
            normal_files.append(path)
        else:
            anomaly_files.append(path)
    return normal_files, anomaly_files


def load_skab_sequences(root: str | Path) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    normal_paths, anomaly_paths = discover_skab_files(root)
    normal_sequences = [_read_skab_csv(path) for path in normal_paths]
    anomaly_sequences = [_read_skab_csv(path) for path in anomaly_paths]
    return normal_sequences, anomaly_sequences


def skab_summary(root: str | Path) -> pd.DataFrame:
    normal_paths, anomaly_paths = discover_skab_files(root)
    records = []
    for split, paths in [("normal", normal_paths), ("anomaly", anomaly_paths)]:
        for path in paths:
            frame = _read_skab_csv(path)
            records.append(
                {
                    "split": split,
                    "source_file": str(path),
                    "n_rows": len(frame),
                    "anomaly_points": int(frame["anomaly"].fillna(0).sum()),
                    "changepoints": int(frame["changepoint"].fillna(0).sum()),
                }
            )
    return pd.DataFrame(records)
