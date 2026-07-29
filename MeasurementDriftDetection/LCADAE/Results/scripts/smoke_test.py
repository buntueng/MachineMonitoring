#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drift_detection.fault_injection import apply_fault_scenario, build_scenario_protocol, inject_faults
from drift_detection.metrics import classification_metrics, localization_metrics, threshold_from_validation
from drift_detection.models import DenseAutoencoder
from drift_detection.proposed import LCADAutoencoder
from drift_detection.training import (
    lcad_anomaly_scores,
    reconstruction_scores,
    train_lcad_model,
    train_reconstruction_model,
)
from drift_detection.windows import make_windows


def synthetic_series(seed: int, length: int, channels: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 30, length)
    base = np.stack([np.sin(time * (0.15 + index * 0.01)) for index in range(channels)], axis=1)
    return (base + rng.normal(0, 0.08, size=base.shape)).astype(np.float32)


def main() -> None:
    train = synthetic_series(1, 500, 8)
    validation = synthetic_series(2, 200, 8)
    clean_test = synthetic_series(3, 300, 8)
    test, labels, faults, event_ids, channel_labels, _ = inject_faults(
        clean_test,
        seed=42,
        event_count=4,
        min_duration=12,
        max_duration=20,
    )
    train_w = make_windows(train, 24, 4)["windows"]
    val_w = make_windows(validation, 24, 4)["windows"]
    test_bundle = make_windows(
        test,
        24,
        4,
        labels=labels,
        fault_types=faults,
        event_ids=event_ids,
        channel_labels=channel_labels,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        dense = DenseAutoencoder(24, 8, hidden_dim=24, latent_dim=6)
        train_reconstruction_model(
            dense,
            train_w,
            val_w,
            device,
            epochs=1,
            batch_size=32,
            learning_rate=1e-3,
            weight_decay=1e-5,
            patience=1,
            checkpoint_path=temp / "dense.pt",
            history_path=temp / "dense.csv",
            resume=False,
        )
        val_scores = reconstruction_scores(dense, val_w, device)
        test_scores = reconstruction_scores(dense, test_bundle["windows"], device)
        threshold = threshold_from_validation(val_scores, 0.98)
        dense_metrics = classification_metrics(test_bundle["labels"], test_scores, threshold, test_bundle["event_ids"])

        proposed = LCADAutoencoder(8, hidden_dim=16, dilation_rates=(1, 2), kernel_size=3, dropout=0.0)
        train_lcad_model(
            proposed,
            train_w,
            val_w,
            device,
            epochs=1,
            batch_size=32,
            learning_rate=1e-3,
            weight_decay=1e-5,
            patience=1,
            checkpoint_path=temp / "proposed.pt",
            history_path=temp / "proposed.csv",
            loss_weights={"reconstruction": 1.0, "statistics": 0.2, "correlation": 0.1},
        )
        val_scores, _ = lcad_anomaly_scores(
            proposed,
            val_w,
            device,
            {"reconstruction": 1.0, "statistics": 0.3, "correlation": 0.1},
        )
        test_scores, channel_scores = lcad_anomaly_scores(
            proposed,
            test_bundle["windows"],
            device,
            {"reconstruction": 1.0, "statistics": 0.3, "correlation": 0.1},
        )
        threshold = threshold_from_validation(val_scores, 0.98)
        proposed_metrics = classification_metrics(test_bundle["labels"], test_scores, threshold, test_bundle["event_ids"])
        proposed_metrics.update(localization_metrics(test_bundle["channel_labels"], channel_scores))

    protocol_config = {
        "experiment": {"profile": "quick"},
        "data": {
            "fault_injection": {
                "enabled_faults": ["additive", "proportional", "gradual", "noise", "stuck", "common_mode"],
                "scenario_grid": {
                    "quick": {"severities": [1.0], "duration_windows": [1], "replicates": 1},
                    "channel_fractions": [0.25],
                    "normalized_start_min": 0.1,
                    "normalized_start_max": 0.8,
                },
            }
        },
    }
    scenario = build_scenario_protocol(protocol_config, profile="quick").iloc[0]
    injected = apply_fault_scenario(clean_test, scenario, window_size=24)
    assert injected[4].shape == clean_test.shape

    print("Smoke test completed successfully.")
    print(f"DenseAE F1: {dense_metrics['f1']:.3f}")
    print(f"LCAD-AE F1: {proposed_metrics['f1']:.3f}")
    print(f"LCAD-AE channel AUPRC: {proposed_metrics['channel_auprc']:.3f}")


if __name__ == "__main__":
    main()
