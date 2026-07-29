from __future__ import annotations

import pickle
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from .classical import IsolationForestModel, PCASPEModel
from .io_utils import (
    append_row_csv,
    atomic_write_dataframe,
    configure_logger,
    count_parameters,
    get_device,
    serialized_model_size_mb,
    utc_now_iso,
)
from .metrics import classification_metrics, per_fault_metrics, threshold_from_validation
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


def load_benchmark(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _window_bundle(bundle: dict[str, np.ndarray], window_size: int, stride: int) -> dict[str, dict[str, np.ndarray]]:
    train = make_windows(bundle["train_series"], window_size, stride)
    validation = make_windows(bundle["validation_series"], window_size, stride)
    test = make_windows(
        bundle["test_series"],
        window_size,
        stride,
        labels=bundle["test_labels"],
        fault_types=bundle["test_fault_types"],
        event_ids=bundle["test_event_ids"],
    )
    return {"train": train, "validation": validation, "test": test}


def _score_frame(
    dataset: str,
    model: str,
    seed: int,
    run_id: str,
    test_windows: dict[str, np.ndarray],
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "run_id": run_id,
            "window_index": np.arange(len(scores)),
            "start_index": test_windows["start_indices"],
            "end_index": test_windows["end_indices"],
            "label": test_windows["labels"],
            "event_id": test_windows["event_ids"],
            "fault_type": test_windows["fault_types"],
            "score": scores,
            "threshold": threshold,
            "prediction": (scores >= threshold).astype(int),
        }
    )


def _fit_classical(name: str, train_windows: np.ndarray, config: dict[str, Any], seed: int):
    if name == "PCA_SPE":
        model = PCASPEModel(config["baselines"]["pca_components"], random_state=seed)
    elif name == "IsolationForest":
        model = IsolationForestModel(config["baselines"]["isolation_forest_estimators"], random_state=seed)
    else:
        raise KeyError(name)
    model.fit(train_windows)
    return model


def _deepod_class(name: str):
    try:
        from deepod.models.time_series import AnomalyTransformer, DCdetector, TranAD
    except Exception as exc:
        raise ImportError(
            "DeepOD is required for optional SOTA baselines. Install requirements-sota.txt."
        ) from exc
    return {"TranAD": TranAD, "AnomalyTransformer": AnomalyTransformer, "DCdetector": DCdetector}[name]


def _align_series_scores(scores: np.ndarray, end_indices: np.ndarray, original_length: int) -> np.ndarray:
    values = np.asarray(scores, dtype=float).reshape(-1)
    if len(values) == original_length:
        return values[end_indices]
    if len(values) == len(end_indices):
        return values
    if len(values) > int(end_indices.max()):
        return values[end_indices]
    raise ValueError(f"Cannot align {len(values)} scores to a series of length {original_length}")


def _run_deepod(
    name: str,
    bundle: dict[str, np.ndarray],
    windows: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
    seed: int,
):
    cls = _deepod_class(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    common = {
        "seq_len": int(config["data"]["window_size"]),
        "stride": int(config["data"]["stride"]),
        "epochs": int(config["baselines"]["sota_epochs"]),
        "epoch_steps": int(config["baselines"]["sota_epoch_steps"]),
        "batch_size": int(config["training"]["batch_size"]),
        "device": device,
        "verbose": 1,
        "random_state": seed,
    }
    if name == "DCdetector":
        common.update({"d_model": 64, "e_layers": 2, "n_heads": 1, "patch_size": [4, 8]})
    model = cls(**common)
    model.fit(bundle["train_series"])
    validation_raw = model.decision_function(bundle["validation_series"])
    test_raw = model.decision_function(bundle["test_series"])
    validation_scores = _align_series_scores(
        validation_raw,
        windows["validation"]["end_indices"],
        len(bundle["validation_series"]),
    )
    test_scores = _align_series_scores(test_raw, windows["test"]["end_indices"], len(bundle["test_series"]))
    parameter_count = float("nan")
    size_mb = float("nan")
    net = getattr(model, "net", None)
    if isinstance(net, torch.nn.Module):
        parameter_count = count_parameters(net)
        size_mb = serialized_model_size_mb(net)
    return model, validation_scores, test_scores, parameter_count, size_mb


def run_baseline_suite(
    benchmark_path: str | Path,
    dataset: str,
    config: dict[str, Any],
    run_optional_sota: bool = False,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    bundle = load_benchmark(benchmark_path)
    window_size = int(config["data"]["window_size"])
    stride = int(config["data"]["stride"])
    windows = _window_bundle(bundle, window_size, stride)
    result_dir = Path(config["paths"]["results"]) / "baselines"
    result_dir.mkdir(parents=True, exist_ok=True)
    run_csv = result_dir / "baseline_runs.csv"
    fault_csv = result_dir / "baseline_fault_metrics.csv"
    logger = configure_logger(f"baseline_{dataset}", Path(config["paths"]["logs"]) / f"baseline_{dataset}.log")
    model_names = list(config["baselines"]["core_models"])
    if run_optional_sota:
        model_names.extend(config["baselines"]["optional_sota_models"])
    seed_values = seeds or list(config["project"]["seed_list"])
    collected_rows: list[dict[str, Any]] = []
    for seed in seed_values:
        set_global_seed(seed, bool(config["project"].get("deterministic", True)))
        for model_name in model_names:
            run_id = f"{dataset}_{model_name}_{seed}_{uuid.uuid4().hex[:8]}"
            logger.info("RUN_STARTED | dataset=%s model=%s seed=%s run_id=%s", dataset, model_name, seed, run_id)
            started = time.perf_counter()
            status = "completed"
            error_message = ""
            parameter_count = float("nan")
            size_mb = float("nan")
            latency_ms = float("nan")
            try:
                if model_name in {"PCA_SPE", "IsolationForest"}:
                    model = _fit_classical(model_name, windows["train"]["windows"], config, seed)
                    validation_scores = model.score(windows["validation"]["windows"])
                    test_scores = model.score(windows["test"]["windows"])
                    size_mb = len(pickle.dumps(model)) / (1024.0**2)
                    latency_ms = inference_latency_ms(model.score, windows["test"]["windows"])
                elif model_name in {"DenseAE", "LSTMAE", "ConvAE", "USAD"}:
                    model = build_baseline_model(
                        model_name,
                        window_size,
                        windows["train"]["windows"].shape[2],
                        hidden_dim=int(config["baselines"]["deep_hidden_dim"]),
                        latent_dim=int(config["baselines"]["deep_latent_dim"]),
                    )
                    checkpoint = Path(config["paths"]["checkpoints"]) / f"{run_id}.pt"
                    history = result_dir / "histories" / f"{run_id}.csv"
                    device = get_device()
                    if model_name == "USAD":
                        train_usad_model(
                            model,
                            windows["train"]["windows"],
                            windows["validation"]["windows"],
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
                        scoring = lambda value: usad_scores(model, value, device)
                    else:
                        train_reconstruction_model(
                            model,
                            windows["train"]["windows"],
                            windows["validation"]["windows"],
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
                        scoring = lambda value: reconstruction_scores(model, value, device)
                    validation_scores = scoring(windows["validation"]["windows"])
                    test_scores = scoring(windows["test"]["windows"])
                    parameter_count = count_parameters(model)
                    size_mb = serialized_model_size_mb(model)
                    latency_ms = inference_latency_ms(scoring, windows["test"]["windows"])
                elif model_name in {"TranAD", "AnomalyTransformer", "DCdetector"}:
                    model, validation_scores, test_scores, parameter_count, size_mb = _run_deepod(
                        model_name, bundle, windows, config, seed
                    )
                    latency_ms = float("nan")
                else:
                    raise KeyError(f"Unknown model {model_name}")
                threshold = threshold_from_validation(
                    validation_scores,
                    float(config["training"]["validation_threshold_quantile"]),
                )
                metrics = classification_metrics(
                    windows["test"]["labels"],
                    test_scores,
                    threshold,
                    event_ids=windows["test"]["event_ids"],
                )
                score_frame = _score_frame(dataset, model_name, seed, run_id, windows["test"], test_scores, threshold)
                atomic_write_dataframe(score_frame, result_dir / "scores" / f"{run_id}.csv.gz")
                for fault_row in per_fault_metrics(
                    windows["test"]["labels"],
                    test_scores,
                    threshold,
                    windows["test"]["fault_types"],
                ):
                    fault_row.update({"dataset": dataset, "model": model_name, "seed": seed, "run_id": run_id})
                    append_row_csv(fault_csv, fault_row)
            except Exception as exc:
                status = "failed"
                error_message = f"{type(exc).__name__}: {exc}"
                metrics = {}
                logger.exception("RUN_FAILED | %s", error_message)
            elapsed = time.perf_counter() - started
            row = {
                "timestamp_utc": utc_now_iso(),
                "dataset": dataset,
                "model": model_name,
                "seed": seed,
                "run_id": run_id,
                "status": status,
                "error_message": error_message,
                "training_and_evaluation_seconds": elapsed,
                "parameter_count": parameter_count,
                "model_size_mb": size_mb,
                "inference_ms_per_window": latency_ms,
                "window_size": window_size,
                "stride": stride,
                **metrics,
            }
            append_row_csv(run_csv, row)
            collected_rows.append(row)
            logger.info("RUN_FINISHED | status=%s elapsed=%.2f", status, elapsed)
    return pd.DataFrame(collected_rows)


def run_proposed_suite(
    benchmark_path: str | Path,
    dataset: str,
    config: dict[str, Any],
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    bundle = load_benchmark(benchmark_path)
    window_size = int(config["data"]["window_size"])
    stride = int(config["data"]["stride"])
    windows = _window_bundle(bundle, window_size, stride)
    result_dir = Path(config["paths"]["results"]) / "proposed"
    run_csv = result_dir / "proposed_runs.csv"
    fault_csv = result_dir / "proposed_fault_metrics.csv"
    logger = configure_logger(f"proposed_{dataset}", Path(config["paths"]["logs"]) / f"proposed_{dataset}.log")
    seed_values = seeds or list(config["project"]["seed_list"])
    rows = []
    settings = config["proposed"]
    for seed in seed_values:
        set_global_seed(seed, bool(config["project"].get("deterministic", True)))
        run_id = f"{dataset}_{settings['name']}_{seed}_{uuid.uuid4().hex[:8]}"
        started = time.perf_counter()
        status = "completed"
        error_message = ""
        metrics: dict[str, Any] = {}
        logger.info("RUN_STARTED | dataset=%s model=%s seed=%s", dataset, settings["name"], seed)
        try:
            model = LCADAutoencoder(
                n_features=windows["train"]["windows"].shape[2],
                hidden_dim=int(settings["hidden_dim"]),
                dilation_rates=tuple(int(value) for value in settings["dilation_rates"]),
                kernel_size=int(settings["kernel_size"]),
                dropout=float(settings["dropout"]),
            )
            device = get_device()
            checkpoint = Path(config["paths"]["checkpoints"]) / f"{run_id}.pt"
            history_path = result_dir / "histories" / f"{run_id}.csv"
            train_lcad_model(
                model,
                windows["train"]["windows"],
                windows["validation"]["windows"],
                device,
                int(config["training"]["epochs"]),
                int(config["training"]["batch_size"]),
                float(config["training"]["learning_rate"]),
                float(config["training"]["weight_decay"]),
                int(config["training"]["patience"]),
                checkpoint,
                history_path,
                loss_weights={
                    "reconstruction": float(settings["reconstruction_weight"]),
                    "statistics": float(settings["statistics_weight"]),
                    "correlation": float(settings["correlation_weight"]),
                },
                num_workers=int(config["training"]["num_workers"]),
            )
            score_weights = {
                "reconstruction": float(settings["score_reconstruction_weight"]),
                "statistics": float(settings["score_statistics_weight"]),
                "correlation": float(settings["score_correlation_weight"]),
            }
            scoring = lambda value: lcad_anomaly_scores(model, value, device, score_weights)[0]
            validation_scores, _ = lcad_anomaly_scores(model, windows["validation"]["windows"], device, score_weights)
            test_scores, channel_scores = lcad_anomaly_scores(model, windows["test"]["windows"], device, score_weights)
            threshold = threshold_from_validation(
                validation_scores,
                float(config["training"]["validation_threshold_quantile"]),
            )
            metrics = classification_metrics(
                windows["test"]["labels"],
                test_scores,
                threshold,
                event_ids=windows["test"]["event_ids"],
            )
            score_frame = _score_frame(dataset, settings["name"], seed, run_id, windows["test"], test_scores, threshold)
            atomic_write_dataframe(score_frame, result_dir / "scores" / f"{run_id}.csv.gz")
            channel_frame = pd.DataFrame(channel_scores, columns=[str(value) for value in bundle["channel_names"]])
            channel_frame.insert(0, "window_index", np.arange(len(channel_frame)))
            channel_frame.insert(0, "run_id", run_id)
            atomic_write_dataframe(channel_frame, result_dir / "channel_scores" / f"{run_id}.csv.gz")
            for fault_row in per_fault_metrics(
                windows["test"]["labels"], test_scores, threshold, windows["test"]["fault_types"]
            ):
                fault_row.update({"dataset": dataset, "model": settings["name"], "seed": seed, "run_id": run_id})
                append_row_csv(fault_csv, fault_row)
            parameter_count = count_parameters(model)
            size_mb = serialized_model_size_mb(model)
            latency_ms = inference_latency_ms(scoring, windows["test"]["windows"])
        except Exception as exc:
            status = "failed"
            error_message = f"{type(exc).__name__}: {exc}"
            parameter_count = float("nan")
            size_mb = float("nan")
            latency_ms = float("nan")
            logger.exception("RUN_FAILED | %s", error_message)
        row = {
            "timestamp_utc": utc_now_iso(),
            "dataset": dataset,
            "model": settings["name"],
            "seed": seed,
            "run_id": run_id,
            "status": status,
            "error_message": error_message,
            "training_and_evaluation_seconds": time.perf_counter() - started,
            "parameter_count": parameter_count,
            "model_size_mb": size_mb,
            "inference_ms_per_window": latency_ms,
            "window_size": window_size,
            "stride": stride,
            **metrics,
        }
        append_row_csv(run_csv, row)
        rows.append(row)
        logger.info("RUN_FINISHED | status=%s", status)
    return pd.DataFrame(rows)


def aggregate_results(config: dict[str, Any]) -> dict[str, Path]:
    result_root = Path(config["paths"]["results"])
    input_paths = [
        result_root / "baselines" / "baseline_runs.csv",
        result_root / "proposed" / "proposed_runs.csv",
    ]
    frames = [pd.read_csv(path) for path in input_paths if path.exists()]
    if not frames:
        raise FileNotFoundError("No baseline or proposed run CSV files were found")
    runs = pd.concat(frames, ignore_index=True, sort=False)
    completed = runs[runs["status"] == "completed"].copy()
    numeric_metrics = [
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "event_recall",
        "mean_detection_delay_windows",
        "false_positive_rate",
        "parameter_count",
        "model_size_mb",
        "inference_ms_per_window",
        "training_and_evaluation_seconds",
    ]
    available_metrics = [metric for metric in numeric_metrics if metric in completed.columns]
    summary = completed.groupby(["dataset", "model"])[available_metrics].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    combined_dir = result_root / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    runs_path = combined_dir / "all_runs.csv"
    summary_path = combined_dir / "model_summary.csv"
    ranking_path = combined_dir / "model_ranking.csv"
    atomic_write_dataframe(runs, runs_path)
    atomic_write_dataframe(summary, summary_path)
    ranking = summary.sort_values(["dataset", "auprc_mean", "f1_mean"], ascending=[True, False, False])
    atomic_write_dataframe(ranking, ranking_path)
    return {"all_runs": runs_path, "summary": summary_path, "ranking": ranking_path}
