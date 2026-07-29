from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_logger(name: str, log_path: str | Path) -> logging.Logger:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def append_row_csv(path: str | Path, row: dict[str, Any]) -> None:
    """Append a dictionary row while keeping a stable union of columns."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        current = pd.read_csv(target)
        updated = pd.concat([current, pd.DataFrame([row])], ignore_index=True, sort=False)
    else:
        updated = pd.DataFrame([row])
    atomic_write_dataframe(updated, target)


def atomic_write_dataframe(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if str(target).endswith(".csv.gz") else target.suffix
    with tempfile.NamedTemporaryFile(delete=False, dir=target.parent, suffix=suffix) as handle:
        temp_path = Path(handle.name)
    try:
        compression = "gzip" if str(target).endswith(".gz") else None
        df.to_csv(temp_path, index=False, compression=compression, **kwargs)
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def serialized_model_size_mb(model: torch.nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(model.state_dict(), temp_path)
        return temp_path.stat().st_size / (1024.0**2)
    finally:
        temp_path.unlink(missing_ok=True)


def safe_npz(path: str | Path, **arrays: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
