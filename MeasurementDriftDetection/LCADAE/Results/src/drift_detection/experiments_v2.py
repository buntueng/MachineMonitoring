from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from .classical import IsolationForestModel, PCASPEModel
from .evaluation_v2 import (
    evaluate_gas_natural,
    evaluate_mixed_benchmark,
    evaluate_scenario_grid,
    evaluate_skab_native,
    load_npz,
)
from .io_utils import (
    atomic_write_dataframe,
    configure_logger,
    count_parameters,
    get_device,
    serialized_model_size_mb,
    utc_now_iso,
)
from .metrics import threshold_candidates
from .models import build_baseline_model
from .proposed import LCADAutoencoder
from .reproducibility import set_global_seed
from .training import (
    inference_latency_ms,
    lcad_anomaly_scores,
    reconstruction_scores,
    train_lcad_model,
    train_reconstruction_model,
    train_usad_model,
    usad_scores,
)
from .windows import make_windows

ScoreWindows = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray | None]]
ScoreSeries = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class TrainedAdapter:
    model: Any
    score_windows: ScoreWindows
    validation_scores: np.ndarray
    parameter_count: float
    model_size_mb: float
    inference_ms_per_window: float
    score_series: ScoreSeries | None = None


def manuscript_result_root(config: dict[str, Any]) -> Path:
    root = Path(config["paths"]["results"]) / str(config["project"]["experiment_version"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def reset_manuscript_results(config: dict[str, Any], keep_checkpoints: bool = True) -> None:
    """Delete only manuscript_v2 result CSVs, never the raw data or old results."""
    root = manuscript_result_root(config)
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    root.mkdir(parents=True, exist_ok=True)
    if not keep_checkpoints:
        checkpoint_root = Path(config["paths"]["checkpoints"]) / str(config["project"]["experiment_version"])
        if checkpoint_root.exists():
            for path in sorted(checkpoint_root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()


def _stable_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _config_fingerprint(config: dict[str, Any]) -> str:
    relevant = {
        "experiment_version": config["project"].get("experiment_version"),
        "profile": config.get("experiment", {}).get("profile"),
        "data": {
            "window_size": config["data"].get("window_size"),
            "stride": config["data"].get("stride"),
            "fault_injection": config["data"].get("fault_injection"),
        },
        "training": config.get("training"),
        "baselines": config.get("baselines"),
        "proposed": config.get("proposed"),
    }
    return _stable_key(relevant)


def _existing_completed(path: Path, experiment_key: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "experiment_key" not in frame.columns:
        return pd.DataFrame()
    return frame[(frame["experiment_key"] == experiment_key) & (frame["status"] == "completed")].tail(1)


def _replace_experiment_rows(path: Path, new_rows: pd.DataFrame, experiment_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        if "experiment_key" in existing.columns:
            existing = existing[existing["experiment_key"] != experiment_key]
        combined = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    else:
        combined = new_rows.copy()
    atomic_write_dataframe(combined, path)


def _align_series_scores(scores: np.ndarray, end_indices: np.ndarray, original_length: int) -> np.ndarray:
    values = np.asarray(scores, dtype=float).reshape(-1)
    if len(values) == original_length:
        return values[end_indices]
    if len(values) == len(end_indices):
        return values
    if len(values) > int(end_indices.max()):
        return values[end_indices]
    raise ValueError(f"Cannot align {len(values)} scores to a series of length {original_length}")


def _deepod_class(name: str):
    try:
        from deepod.models.time_series import AnomalyTransformer, DCdetector, TranAD
    except Exception as exc:
        raise ImportError("DeepOD is required. Install requirements-sota.txt before enabling SOTA models.") from exc
    return {"TranAD": TranAD, "AnomalyTransformer": AnomalyTransformer, "DCdetector": DCdetector}[name]


def _fit_baseline_adapter(
    model_name: str,
    bundle: dict[str, np.ndarray],
    config: dict[str, Any],
    seed: int,
    run_id: str,
    result_dir: Path,
) -> TrainedAdapter:
    window_size = int(config["data"]["window_size"])
    stride = int(config["data"]["stride"])
    train_windows = make_windows(bundle["train_series"], window_size, stride)["windows"]
    validation_bundle = make_windows(bundle["validation_series"], window_size, stride)
    validation_windows = validation_bundle["windows"]
    device = get_device()

    if model_name == "PCA_SPE":
        model = PCASPEModel(int(config["baselines"]["pca_components"]), random_state=seed).fit(train_windows)
        score_windows = lambda windows: (model.score(windows), None)
        validation_scores, _ = score_windows(validation_windows)
        size_mb = len(pickle.dumps(model)) / (1024.0**2)
        latency = inference_latency_ms(lambda value: model.score(value), validation_windows)
        return TrainedAdapter(model, score_windows, validation_scores, float("nan"), size_mb, latency)

    if model_name == "IsolationForest":
        model = IsolationForestModel(
            int(config["baselines"]["isolation_forest_estimators"]), random_state=seed
        ).fit(train_windows)
        score_windows = lambda windows: (model.score(windows), None)
        validation_scores, _ = score_windows(validation_windows)
        size_mb = len(pickle.dumps(model)) / (1024.0**2)
        latency = inference_latency_ms(lambda value: model.score(value), validation_windows)
        return TrainedAdapter(model, score_windows, validation_scores, float("nan"), size_mb, latency)

    if model_name in {"DenseAE", "LSTMAE", "ConvAE", "USAD"}:
        model = build_baseline_model(
            model_name,
            window_size,
            train_windows.shape[2],
            hidden_dim=int(config["baselines"]["deep_hidden_dim"]),
            latent_dim=int(config["baselines"]["deep_latent_dim"]),
        )
        checkpoint = Path(config["paths"]["checkpoints"]) / config["project"]["experiment_version"] / f"{run_id}.pt"
        history = result_dir / "histories" / f"{run_id}.csv"
        if model_name == "USAD":
            train_usad_model(
                model,
                train_windows,
                validation_windows,
                device,
                int(config["training"]["epochs"]),
                int(config["training"]["batch_size"]),
                float(config["training"]["learning_rate"]),
                float(config["training"]["weight_decay"]),
                int(config["training"]["patience"]),
                checkpoint,
                history,
                int(config["training"]["num_workers"]),
            )
            score_windows = lambda windows: (usad_scores(model, windows, device), None)
        else:
            train_reconstruction_model(
                model,
                train_windows,
                validation_windows,
                device,
                int(config["training"]["epochs"]),
                int(config["training"]["batch_size"]),
                float(config["training"]["learning_rate"]),
                float(config["training"]["weight_decay"]),
                int(config["training"]["patience"]),
                checkpoint,
                history,
                int(config["training"]["num_workers"]),
            )
            score_windows = lambda windows: (reconstruction_scores(model, windows, device), None)
        validation_scores, _ = score_windows(validation_windows)
        return TrainedAdapter(
            model=model,
            score_windows=score_windows,
            validation_scores=validation_scores,
            parameter_count=float(count_parameters(model)),
            model_size_mb=serialized_model_size_mb(model),
            inference_ms_per_window=inference_latency_ms(lambda value: score_windows(value)[0], validation_windows),
        )

    if model_name in {"TranAD", "AnomalyTransformer", "DCdetector"}:
        cls = _deepod_class(model_name)
        common = {
            "seq_len": window_size,
            "stride": stride,
            "epochs": int(config["baselines"]["sota_epochs"]),
            "epoch_steps": int(config["baselines"]["sota_epoch_steps"]),
            "batch_size": int(config["training"]["batch_size"]),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "verbose": 1,
            "random_state": seed,
        }
        if model_name == "DCdetector":
            common.update({"d_model": 64, "e_layers": 2, "n_heads": 1, "patch_size": [4, 8]})
        model = cls(**common)
        model.fit(bundle["train_series"])

        def score_series(series: np.ndarray, end_indices: np.ndarray) -> np.ndarray:
            raw = model.decision_function(series)
            return _align_series_scores(raw, end_indices, len(series))

        validation_scores = score_series(bundle["validation_series"], validation_bundle["end_indices"])
        net = getattr(model, "net", None)
        parameter_count = float(count_parameters(net)) if isinstance(net, torch.nn.Module) else float("nan")
        size_mb = serialized_model_size_mb(net) if isinstance(net, torch.nn.Module) else float("nan")

        def unavailable_score_windows(_: np.ndarray) -> tuple[np.ndarray, None]:
            raise RuntimeError("This adapter must score full series")

        started = time.perf_counter()
        _ = score_series(bundle["validation_series"], validation_bundle["end_indices"])
        latency = (time.perf_counter() - started) * 1000.0 / max(1, len(validation_scores))
        return TrainedAdapter(
            model,
            unavailable_score_windows,
            validation_scores,
            parameter_count,
            size_mb,
            float(latency),
            score_series=score_series,
        )
    raise KeyError(f"Unknown baseline model: {model_name}")


def _lcad_weights(config: dict[str, Any], variant: str) -> tuple[dict[str, float], dict[str, float]]:
    settings = config["proposed"]
    if variant in settings.get("ablations", {}):
        source = settings["ablations"][variant]
    else:
        source = settings
    loss_weights = {
        "reconstruction": float(source["reconstruction_weight"]),
        "statistics": float(source["statistics_weight"]),
        "correlation": float(source["correlation_weight"]),
    }
    score_weights = {
        "reconstruction": float(source["score_reconstruction_weight"]),
        "statistics": float(source["score_statistics_weight"]),
        "correlation": float(source["score_correlation_weight"]),
    }
    return loss_weights, score_weights


def _fit_lcad_adapter(
    bundle: dict[str, np.ndarray],
    config: dict[str, Any],
    seed: int,
    run_id: str,
    result_dir: Path,
    variant: str = "full",
    train_fraction: float = 1.0,
    transferred_blocks: dict[str, torch.Tensor] | None = None,
    reuse_existing_checkpoint: bool = False,
) -> TrainedAdapter:
    window_size = int(config["data"]["window_size"])
    stride = int(config["data"]["stride"])
    train_windows_all = make_windows(bundle["train_series"], window_size, stride)["windows"]
    validation_windows = make_windows(bundle["validation_series"], window_size, stride)["windows"]
    minimum = int(config["proposed"]["transfer"].get("minimum_target_windows", 24))
    use_count = min(len(train_windows_all), max(minimum, int(np.ceil(len(train_windows_all) * train_fraction))))
    train_windows = train_windows_all[:use_count]
    settings = config["proposed"]
    model = LCADAutoencoder(
        n_features=train_windows.shape[2],
        hidden_dim=int(settings["hidden_dim"]),
        dilation_rates=tuple(int(value) for value in settings["dilation_rates"]),
        kernel_size=int(settings["kernel_size"]),
        dropout=float(settings["dropout"]),
    )
    if transferred_blocks is not None:
        model.blocks.load_state_dict(transferred_blocks, strict=True)
    device = get_device()
    checkpoint = Path(config["paths"]["checkpoints"]) / config["project"]["experiment_version"] / f"{run_id}.pt"
    history = result_dir / "histories" / f"{run_id}.csv"
    loss_weights, score_weights = _lcad_weights(config, variant)
    if reuse_existing_checkpoint and checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        model.to(device)
    else:
        train_lcad_model(
            model,
            train_windows,
            validation_windows,
            device,
            int(config["training"]["epochs"]),
            int(config["training"]["batch_size"]),
            float(config["training"]["learning_rate"]),
            float(config["training"]["weight_decay"]),
            int(config["training"]["patience"]),
            checkpoint,
            history,
            loss_weights,
            int(config["training"]["num_workers"]),
        )

    def score_windows(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return lcad_anomaly_scores(model, windows, device, score_weights)

    validation_scores, _ = score_windows(validation_windows)
    adapter = TrainedAdapter(
        model=model,
        score_windows=score_windows,
        validation_scores=validation_scores,
        parameter_count=float(count_parameters(model)),
        model_size_mb=serialized_model_size_mb(model),
        inference_ms_per_window=inference_latency_ms(lambda value: score_windows(value)[0], validation_windows),
    )
    setattr(adapter, "actual_train_windows", use_count)
    setattr(adapter, "available_train_windows", len(train_windows_all))
    return adapter


def _dataset_auxiliary_paths(config: dict[str, Any], dataset: str) -> dict[str, Path]:
    if dataset == "gas_sensor":
        return {"natural": Path(config["paths"]["benchmarks"]) / "gas_natural_later_batches.npz"}
    if dataset == "skab":
        return {"native_manifest": Path(config["paths"]["results"]) / "data" / "skab_native_manifest.csv"}
    return {}


def _evaluate_adapter(
    adapter: TrainedAdapter,
    bundle: dict[str, np.ndarray],
    scenario_manifest: pd.DataFrame,
    dataset: str,
    model_name: str,
    seed: int,
    run_id: str,
    experiment_key: str,
    family: str,
    variant: str,
    config: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
    evaluate_auxiliary: bool = True,
) -> dict[str, Any]:
    root = manuscript_result_root(config)
    result_dir = root / family
    window_size = int(config["data"]["window_size"])
    stride = int(config["data"]["stride"])
    thresholds = threshold_candidates(adapter.validation_scores, config)
    threshold = float(thresholds["primary"])

    mixed_metrics, mixed_scores, mixed_channel_scores = evaluate_mixed_benchmark(
        bundle,
        adapter.score_windows,
        threshold,
        window_size,
        stride,
        score_series=adapter.score_series,
    )
    scenario_summary, scenario_metrics, sensitivity, representatives = evaluate_scenario_grid(
        bundle["clean_test_series"],
        scenario_manifest,
        adapter.score_windows,
        thresholds,
        window_size,
        stride,
        score_series=adapter.score_series,
        representative_scores_per_fault=bool(config["experiment"].get("save_representative_scores", True)),
    )
    metadata = {
        "experiment_key": experiment_key,
        "run_id": run_id,
        "dataset": dataset,
        "model": model_name,
        "variant": variant,
        "seed": seed,
        "family": family,
        "experiment_version": config["project"]["experiment_version"],
        "config_fingerprint": _config_fingerprint(config),
        "scenario_profile": config["experiment"]["profile"],
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    for key, value in metadata.items():
        scenario_metrics[key] = value
        if not sensitivity.empty:
            sensitivity[key] = value
        if not representatives.empty:
            representatives[key] = value
    _replace_experiment_rows(result_dir / "scenario_metrics.csv", scenario_metrics, experiment_key)
    if not sensitivity.empty:
        _replace_experiment_rows(result_dir / "threshold_sensitivity.csv", sensitivity, experiment_key)
    if not representatives.empty:
        atomic_write_dataframe(
            representatives, result_dir / "representative_scores" / f"{run_id}.csv.gz"
        )
    mixed_scores = mixed_scores.assign(**metadata)
    atomic_write_dataframe(mixed_scores, result_dir / "mixed_scores" / f"{run_id}.csv.gz")
    if mixed_channel_scores is not None:
        channel_frame = pd.DataFrame(mixed_channel_scores, columns=[str(value) for value in bundle["channel_names"]])
        channel_frame.insert(0, "window_index", np.arange(len(channel_frame)))
        for key, value in reversed(list(metadata.items())):
            channel_frame.insert(0, key, value)
        atomic_write_dataframe(
            channel_frame, result_dir / "mixed_channel_scores" / f"{run_id}.csv.gz"
        )

    fault_summary = (
        scenario_metrics.groupby("fault_type", as_index=False)[
            [
                column
                for column in [
                    "precision",
                    "recall",
                    "f1",
                    "auroc",
                    "auprc",
                    "event_recall",
                    "mean_detection_delay_windows",
                    "false_positive_rate",
                    "channel_auprc",
                    "top1_localization_accuracy",
                    "topk_channel_recall",
                ]
                if column in scenario_metrics.columns
            ]
        ]
        .mean(numeric_only=True)
    )
    for key, value in metadata.items():
        fault_summary[key] = value
    _replace_experiment_rows(result_dir / "fault_summary.csv", fault_summary, experiment_key)

    auxiliary = _dataset_auxiliary_paths(config, dataset)
    native_summary: dict[str, float] = {}
    if evaluate_auxiliary and dataset == "gas_sensor" and config["experiment"].get("evaluate_natural_gas", True):
        natural_path = auxiliary.get("natural")
        if natural_path and natural_path.exists():
            natural = evaluate_gas_natural(
                natural_path,
                adapter.score_windows,
                threshold,
                adapter.validation_scores,
                window_size,
                stride,
                score_series=adapter.score_series,
            )
            for key, value in metadata.items():
                natural[key] = value
            _replace_experiment_rows(result_dir / "gas_natural_metrics.csv", natural, experiment_key)
            if not natural.empty:
                native_summary["natural_max_exceedance_rate"] = float(natural["threshold_exceedance_rate"].max())
                native_summary["natural_max_robust_shift"] = float(natural["robust_shift_from_validation"].max())
    if evaluate_auxiliary and dataset == "skab" and config["experiment"].get("evaluate_native_skab", True):
        native_manifest = auxiliary.get("native_manifest")
        if native_manifest and native_manifest.exists():
            native = evaluate_skab_native(
                native_manifest,
                adapter.score_windows,
                threshold,
                window_size,
                stride,
                score_series=adapter.score_series,
            )
            for key, value in metadata.items():
                native[key] = value
            _replace_experiment_rows(result_dir / "skab_native_metrics.csv", native, experiment_key)
            if not native.empty:
                for metric in ["precision", "recall", "f1", "auroc", "auprc", "event_recall", "false_positive_rate"]:
                    if metric in native.columns:
                        native_summary[f"native_{metric}"] = float(pd.to_numeric(native[metric], errors="coerce").mean())
                native_summary["native_sequence_count"] = float(len(native))

    return {
        **scenario_summary,
        **{f"mixed_{key}": value for key, value in mixed_metrics.items()},
        **native_summary,
        "threshold": threshold,
        "threshold_method": config["training"]["threshold"]["primary_method"],
        "validation_score_median": float(np.median(adapter.validation_scores)),
        "validation_score_p99": float(np.quantile(adapter.validation_scores, 0.99)),
    }


def _run_row(
    *,
    experiment_key: str,
    run_id: str,
    dataset: str,
    model: str,
    variant: str,
    seed: int,
    family: str,
    status: str,
    error_message: str,
    elapsed: float,
    adapter: TrainedAdapter | None,
    metrics: dict[str, Any],
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "timestamp_utc": utc_now_iso(),
        "experiment_key": experiment_key,
        "run_id": run_id,
        "dataset": dataset,
        "model": model,
        "variant": variant,
        "seed": seed,
        "family": family,
        "evaluation_set": "scenario_grid",
        "experiment_version": config["project"]["experiment_version"],
        "config_fingerprint": _config_fingerprint(config),
        "scenario_profile": config["experiment"]["profile"],
        "status": status,
        "error_message": error_message,
        "training_and_evaluation_seconds": elapsed,
        "parameter_count": adapter.parameter_count if adapter else float("nan"),
        "model_size_mb": adapter.model_size_mb if adapter else float("nan"),
        "inference_ms_per_window": adapter.inference_ms_per_window if adapter else float("nan"),
        "window_size": int(config["data"]["window_size"]),
        "stride": int(config["data"]["stride"]),
        **metrics,
    }
    if extra:
        row.update(extra)
    return row


def run_baseline_manuscript_suite(
    benchmark_path: str | Path,
    dataset: str,
    config: dict[str, Any],
    seeds: list[int] | None = None,
    run_optional_sota: bool = False,
    force: bool | None = None,
) -> pd.DataFrame:
    bundle = load_npz(benchmark_path)
    manifest_path = Path(config["paths"]["benchmarks"]) / f"{dataset}_scenario_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Scenario manifest not found: {manifest_path}. Re-run Notebook 01.")
    manifest = pd.read_csv(manifest_path)
    result_dir = manuscript_result_root(config) / "baselines"
    run_csv = result_dir / "runs.csv"
    logger = configure_logger(
        f"manuscript_baseline_{dataset}", Path(config["paths"]["logs"]) / f"manuscript_baseline_{dataset}.log"
    )
    models = list(config["baselines"]["core_models"])
    if run_optional_sota:
        models.extend(config["baselines"]["optional_sota_models"])
    seed_values = list(seeds or config["project"]["seed_list"])
    force_run = bool(config["experiment"].get("force_rerun", False) if force is None else force)
    rows: list[dict[str, Any]] = []
    for seed in seed_values:
        for model_name in models:
            key_payload = {
                "version": config["project"]["experiment_version"],
                "config_fingerprint": _config_fingerprint(config),
                "family": "baselines",
                "dataset": dataset,
                "model": model_name,
                "seed": int(seed),
                "profile": config["experiment"]["profile"],
            }
            experiment_key = _stable_key(key_payload)
            existing = _existing_completed(run_csv, experiment_key)
            if not force_run and not existing.empty:
                rows.append(existing.iloc[0].to_dict())
                logger.info("RUN_SKIPPED_EXISTING | dataset=%s model=%s seed=%s", dataset, model_name, seed)
                continue
            set_global_seed(int(seed), bool(config["project"].get("deterministic", True)))
            run_id = f"{dataset}_{model_name}_{seed}_{experiment_key[:8]}"
            started = time.perf_counter()
            adapter: TrainedAdapter | None = None
            metrics: dict[str, Any] = {}
            status = "completed"
            error = ""
            try:
                logger.info("RUN_STARTED | dataset=%s model=%s seed=%s", dataset, model_name, seed)
                adapter = _fit_baseline_adapter(model_name, bundle, config, int(seed), run_id, result_dir)
                metrics = _evaluate_adapter(
                    adapter,
                    bundle,
                    manifest,
                    dataset,
                    model_name,
                    int(seed),
                    run_id,
                    experiment_key,
                    "baselines",
                    "published_baseline",
                    config,
                )
            except Exception as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                logger.exception("RUN_FAILED | %s", error)
            row = _run_row(
                experiment_key=experiment_key,
                run_id=run_id,
                dataset=dataset,
                model=model_name,
                variant="published_baseline",
                seed=int(seed),
                family="baselines",
                status=status,
                error_message=error,
                elapsed=time.perf_counter() - started,
                adapter=adapter,
                metrics=metrics,
                config=config,
            )
            _replace_experiment_rows(run_csv, pd.DataFrame([row]), experiment_key)
            rows.append(row)
            logger.info("RUN_FINISHED | status=%s", status)
    return pd.DataFrame(rows)


def run_proposed_manuscript_suite(
    benchmark_path: str | Path,
    dataset: str,
    config: dict[str, Any],
    seeds: list[int] | None = None,
    variants: list[str] | None = None,
    force: bool | None = None,
) -> pd.DataFrame:
    bundle = load_npz(benchmark_path)
    manifest_path = Path(config["paths"]["benchmarks"]) / f"{dataset}_scenario_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Scenario manifest not found: {manifest_path}. Re-run Notebook 01.")
    manifest = pd.read_csv(manifest_path)
    result_dir = manuscript_result_root(config) / "proposed"
    run_csv = result_dir / "runs.csv"
    logger = configure_logger(
        f"manuscript_proposed_{dataset}", Path(config["paths"]["logs"]) / f"manuscript_proposed_{dataset}.log"
    )
    seed_values = list(seeds or config["project"]["seed_list"])
    variant_values = variants or ["full"]
    force_run = bool(config["experiment"].get("force_rerun", False) if force is None else force)
    rows: list[dict[str, Any]] = []
    for seed in seed_values:
        for variant in variant_values:
            model_name = str(config["proposed"]["name"])
            key_payload = {
                "version": config["project"]["experiment_version"],
                "config_fingerprint": _config_fingerprint(config),
                "family": "proposed",
                "dataset": dataset,
                "model": model_name,
                "variant": variant,
                "seed": int(seed),
                "profile": config["experiment"]["profile"],
            }
            experiment_key = _stable_key(key_payload)
            existing = _existing_completed(run_csv, experiment_key)
            if not force_run and not existing.empty:
                rows.append(existing.iloc[0].to_dict())
                logger.info("RUN_SKIPPED_EXISTING | dataset=%s variant=%s seed=%s", dataset, variant, seed)
                continue
            set_global_seed(int(seed), bool(config["project"].get("deterministic", True)))
            run_id = f"{dataset}_{model_name}_{variant}_{seed}_{experiment_key[:8]}"
            started = time.perf_counter()
            adapter: TrainedAdapter | None = None
            metrics: dict[str, Any] = {}
            status = "completed"
            error = ""
            try:
                logger.info("RUN_STARTED | dataset=%s variant=%s seed=%s", dataset, variant, seed)
                adapter = _fit_lcad_adapter(bundle, config, int(seed), run_id, result_dir, variant=variant)
                metrics = _evaluate_adapter(
                    adapter,
                    bundle,
                    manifest,
                    dataset,
                    model_name,
                    int(seed),
                    run_id,
                    experiment_key,
                    "proposed",
                    variant,
                    config,
                )
            except Exception as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                logger.exception("RUN_FAILED | %s", error)
            row = _run_row(
                experiment_key=experiment_key,
                run_id=run_id,
                dataset=dataset,
                model=model_name,
                variant=variant,
                seed=int(seed),
                family="proposed",
                status=status,
                error_message=error,
                elapsed=time.perf_counter() - started,
                adapter=adapter,
                metrics=metrics,
                config=config,
            )
            _replace_experiment_rows(run_csv, pd.DataFrame([row]), experiment_key)
            rows.append(row)
            logger.info("RUN_FINISHED | status=%s", status)
    return pd.DataFrame(rows)


def run_cross_domain_transfer_suite(
    gas_benchmark_path: str | Path,
    skab_benchmark_path: str | Path,
    config: dict[str, Any],
    seeds: list[int] | None = None,
    fractions: list[float] | None = None,
    directions: list[str] | None = None,
    force: bool | None = None,
) -> pd.DataFrame:
    bundles = {"gas_sensor": load_npz(gas_benchmark_path), "skab": load_npz(skab_benchmark_path)}
    manifests = {
        "gas_sensor": pd.read_csv(Path(config["paths"]["benchmarks"]) / "gas_sensor_scenario_manifest.csv"),
        "skab": pd.read_csv(Path(config["paths"]["benchmarks"]) / "skab_scenario_manifest.csv"),
    }
    direction_map = {"gas_to_skab": ("gas_sensor", "skab"), "skab_to_gas": ("skab", "gas_sensor")}
    transfer_settings = config["proposed"]["transfer"]
    direction_values = directions or list(transfer_settings["directions"])
    fraction_values = [float(value) for value in (fractions or transfer_settings["target_normal_fractions"])]
    seed_values = list(seeds or config["project"]["seed_list"])
    result_dir = manuscript_result_root(config) / "transfer"
    run_csv = result_dir / "runs.csv"
    logger = configure_logger("cross_domain_transfer", Path(config["paths"]["logs"]) / "cross_domain_transfer.log")
    force_run = bool(config["experiment"].get("force_rerun", False) if force is None else force)
    rows: list[dict[str, Any]] = []

    for seed in seed_values:
        for direction in direction_values:
            if direction not in direction_map:
                raise KeyError(f"Unknown transfer direction: {direction}")
            source_name, target_name = direction_map[direction]
            source_bundle = bundles[source_name]
            target_bundle = bundles[target_name]
            source_run_id = f"pretrain_{source_name}_{seed}_{_config_fingerprint(config)[:8]}"
            source_adapter = _fit_lcad_adapter(
                source_bundle,
                config,
                int(seed),
                source_run_id,
                result_dir / "source_pretraining",
                variant="full",
                train_fraction=1.0,
                reuse_existing_checkpoint=not force_run,
            )
            transferred_blocks = {
                key: value.detach().cpu().clone() for key, value in source_adapter.model.blocks.state_dict().items()
            }
            for fraction in fraction_values:
                for initialization in ["scratch", "transfer"]:
                    key_payload = {
                        "version": config["project"]["experiment_version"],
                        "config_fingerprint": _config_fingerprint(config),
                        "family": "transfer",
                        "direction": direction,
                        "target": target_name,
                        "seed": int(seed),
                        "target_fraction": float(fraction),
                        "initialization": initialization,
                        "profile": config["experiment"]["profile"],
                    }
                    experiment_key = _stable_key(key_payload)
                    existing = _existing_completed(run_csv, experiment_key)
                    if not force_run and not existing.empty:
                        rows.append(existing.iloc[0].to_dict())
                        continue
                    set_global_seed(int(seed), bool(config["project"].get("deterministic", True)))
                    run_id = f"{direction}_{initialization}_{fraction:g}_{seed}_{experiment_key[:8]}"
                    started = time.perf_counter()
                    adapter: TrainedAdapter | None = None
                    metrics: dict[str, Any] = {}
                    status = "completed"
                    error = ""
                    try:
                        adapter = _fit_lcad_adapter(
                            target_bundle,
                            config,
                            int(seed),
                            run_id,
                            result_dir,
                            variant="full",
                            train_fraction=float(fraction),
                            transferred_blocks=transferred_blocks if initialization == "transfer" else None,
                        )
                        actual_fraction = float(
                            getattr(adapter, "actual_train_windows") / max(1, getattr(adapter, "available_train_windows"))
                        )
                        metrics = _evaluate_adapter(
                            adapter,
                            target_bundle,
                            manifests[target_name],
                            target_name,
                            f"LCAD_AE_{initialization}",
                            int(seed),
                            run_id,
                            experiment_key,
                            "transfer",
                            "full",
                            config,
                            extra_metadata={
                                "direction": direction,
                                "source_domain": source_name,
                                "target_domain": target_name,
                                "requested_target_fraction": float(fraction),
                                "actual_target_fraction": actual_fraction,
                                "initialization": initialization,
                            },
                            evaluate_auxiliary=False,
                        )
                    except Exception as exc:
                        status = "failed"
                        error = f"{type(exc).__name__}: {exc}"
                        logger.exception("TRANSFER_FAILED | %s", error)
                    extra = {
                        "direction": direction,
                        "source_domain": source_name,
                        "target_domain": target_name,
                        "requested_target_fraction": float(fraction),
                        "initialization": initialization,
                        "actual_target_fraction": (
                            float(getattr(adapter, "actual_train_windows") / max(1, getattr(adapter, "available_train_windows")))
                            if adapter is not None
                            else float("nan")
                        ),
                        "target_train_windows": float(getattr(adapter, "actual_train_windows", float("nan"))) if adapter else float("nan"),
                    }
                    row = _run_row(
                        experiment_key=experiment_key,
                        run_id=run_id,
                        dataset=target_name,
                        model=f"LCAD_AE_{initialization}",
                        variant="full",
                        seed=int(seed),
                        family="transfer",
                        status=status,
                        error_message=error,
                        elapsed=time.perf_counter() - started,
                        adapter=adapter,
                        metrics=metrics,
                        config=config,
                        extra=extra,
                    )
                    _replace_experiment_rows(run_csv, pd.DataFrame([row]), experiment_key)
                    rows.append(row)
    return pd.DataFrame(rows)
