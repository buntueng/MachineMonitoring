from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root inferred from this source file."""
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration and attach absolute project paths."""
    root = project_root()
    config_path = Path(path) if path is not None else root / "configs" / "default.yaml"
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["paths"] = {
        "root": root,
        "raw_gas": root / "data" / "raw" / "gas_sensor",
        "raw_skab": root / "data" / "raw" / "skab",
        "processed": root / "data" / "processed",
        "benchmarks": root / "data" / "processed" / "benchmarks",
        "results": root / "results",
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
    }
    for value in config["paths"].values():
        Path(value).mkdir(parents=True, exist_ok=True)
    return config
