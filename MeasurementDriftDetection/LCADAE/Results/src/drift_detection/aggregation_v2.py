from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .experiments_v2 import manuscript_result_root
from .io_utils import atomic_write_dataframe


def _read_existing(paths: list[Path]) -> list[pd.DataFrame]:
    return [pd.read_csv(path) for path in paths if path.exists()]


def _bootstrap_ci(values: np.ndarray, repetitions: int, confidence: float, seed: int = 2026) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return float(clean[0]), float(clean[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(int(repetitions), len(clean)), replace=True).mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def aggregate_manuscript_results(config: dict[str, Any]) -> dict[str, Path]:
    root = manuscript_result_root(config)
    combined = root / "combined"
    combined.mkdir(parents=True, exist_ok=True)
    families = ["baselines", "proposed", "transfer"]

    run_frames = _read_existing([root / family / "runs.csv" for family in families])
    if not run_frames:
        raise FileNotFoundError("No manuscript_v2 run records were found")
    all_runs = pd.concat(run_frames, ignore_index=True, sort=False)
    if "experiment_key" in all_runs.columns:
        all_runs = all_runs.sort_values("timestamp_utc").drop_duplicates("experiment_key", keep="last")
    completed = all_runs[all_runs["status"] == "completed"].copy()

    metrics = [
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
        "mixed_f1",
        "native_f1",
        "parameter_count",
        "model_size_mb",
        "inference_ms_per_window",
        "training_and_evaluation_seconds",
    ]
    available = [metric for metric in metrics if metric in completed.columns]
    group_columns = ["dataset", "family", "model", "variant"]
    summary = completed.groupby(group_columns, dropna=False)[available].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    seed_counts = (
        completed.groupby(group_columns, dropna=False)["seed"].nunique().rename("distinct_seed_count").reset_index()
    )
    summary = summary.merge(seed_counts, on=group_columns, how="left")

    scenario_frames = _read_existing([root / family / "scenario_metrics.csv" for family in families])
    scenarios = pd.concat(scenario_frames, ignore_index=True, sort=False) if scenario_frames else pd.DataFrame()
    if not scenarios.empty:
        scenarios = scenarios.sort_values("run_id").drop_duplicates(["experiment_key", "scenario_id"], keep="last")

    bootstrap_rows: list[dict[str, Any]] = []
    repetitions = int(config["experiment"].get("bootstrap_repetitions", 2000))
    confidence = float(config["experiment"].get("confidence_level", 0.95))
    if not scenarios.empty:
        scenario_groups = ["dataset", "family", "model", "variant"]
        for keys, frame in scenarios.groupby(scenario_groups, dropna=False):
            row = dict(zip(scenario_groups, keys))
            row["scenario_observation_count"] = len(frame)
            row["distinct_seed_count"] = int(frame["seed"].nunique())
            for metric in ["f1", "auprc", "auroc", "event_recall", "false_positive_rate", "topk_channel_recall"]:
                if metric not in frame.columns:
                    continue
                values = pd.to_numeric(frame[metric], errors="coerce").to_numpy()
                finite = values[np.isfinite(values)]
                low, high = _bootstrap_ci(values, repetitions, confidence)
                row[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)
    if bootstrap.empty:
        bootstrap = pd.DataFrame(columns=[
            "dataset", "family", "model", "variant", "scenario_observation_count",
            "distinct_seed_count", "f1_mean", "f1_ci_low", "f1_ci_high",
            "auprc_mean", "auprc_ci_low", "auprc_ci_high",
        ])

    pairwise_rows: list[dict[str, Any]] = []
    if not scenarios.empty:
        proposed = scenarios[(scenarios["family"] == "proposed") & (scenarios["variant"] == "full")]
        baselines = scenarios[scenarios["family"] == "baselines"]
        for dataset in sorted(set(proposed["dataset"]) & set(baselines["dataset"])):
            proposed_dataset = proposed[proposed["dataset"] == dataset]
            for model_name, baseline_frame in baselines[baselines["dataset"] == dataset].groupby("model"):
                merged = proposed_dataset.merge(
                    baseline_frame,
                    on=["dataset", "seed", "scenario_id"],
                    suffixes=("_proposed", "_baseline"),
                )
                if len(merged) < 5:
                    continue
                differences = merged["f1_proposed"].to_numpy(float) - merged["f1_baseline"].to_numpy(float)
                if np.allclose(differences, 0):
                    statistic, p_value = 0.0, 1.0
                else:
                    statistic, p_value = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
                pairwise_rows.append(
                    {
                        "dataset": dataset,
                        "proposed_model": "LCAD_AE[full]",
                        "baseline_model": model_name,
                        "paired_scenario_count": len(merged),
                        "mean_f1_difference": float(np.mean(differences)),
                        "median_f1_difference": float(np.median(differences)),
                        "win_rate": float(np.mean(differences > 0)),
                        "wilcoxon_statistic": float(statistic),
                        "p_value": float(p_value),
                    }
                )
    pairwise_columns = [
        "dataset", "proposed_model", "baseline_model", "paired_scenario_count",
        "mean_f1_difference", "median_f1_difference", "win_rate",
        "wilcoxon_statistic", "p_value", "p_value_holm",
    ]
    pairwise = pd.DataFrame(pairwise_rows)
    if not pairwise.empty:
        pairwise["p_value_holm"] = _holm_adjust(pairwise["p_value"].tolist())
        pairwise = pairwise.reindex(columns=pairwise_columns)
    else:
        pairwise = pd.DataFrame(columns=pairwise_columns)

    standard = completed[completed["family"].isin(["baselines", "proposed"])].copy()
    standard = standard[(standard["family"] == "baselines") | (standard["variant"] == "full")]
    cross_rows: list[dict[str, Any]] = []
    for (family, model, variant), frame in standard.groupby(["family", "model", "variant"], dropna=False):
        datasets = set(frame["dataset"])
        if not {"gas_sensor", "skab"}.issubset(datasets):
            continue
        row = {"family": family, "model": model, "variant": variant}
        for metric in ["f1", "auprc", "auroc", "event_recall", "false_positive_rate"]:
            if metric in frame.columns:
                domain_means = frame.groupby("dataset")[metric].mean()
                row[metric] = float(domain_means.mean())
                row[f"{metric}_worst_domain"] = float(domain_means.min())
        row["distinct_seed_count"] = int(frame["seed"].nunique())
        cross_rows.append(row)
    cross_domain = pd.DataFrame(cross_rows)
    if not cross_domain.empty:
        cross_domain = cross_domain.sort_values(["auprc", "f1"], ascending=False)
    else:
        cross_domain = pd.DataFrame(columns=[
            "family", "model", "variant", "f1", "f1_worst_domain",
            "auprc", "auprc_worst_domain", "auroc", "auroc_worst_domain",
            "event_recall", "event_recall_worst_domain",
            "false_positive_rate", "false_positive_rate_worst_domain",
            "distinct_seed_count",
        ])

    transfer = completed[completed["family"] == "transfer"].copy()
    transfer_summary = pd.DataFrame(columns=[
        "direction", "target_domain", "requested_target_fraction", "initialization",
        "f1_mean", "f1_std", "f1_count", "auprc_mean", "auprc_std", "auprc_count",
        "auroc_mean", "auroc_std", "auroc_count", "event_recall_mean",
        "event_recall_std", "event_recall_count", "false_positive_rate_mean",
        "false_positive_rate_std", "false_positive_rate_count",
    ])
    if not transfer.empty:
        transfer_summary = (
            transfer.groupby(
                ["direction", "target_domain", "requested_target_fraction", "initialization"], as_index=False
            )[[metric for metric in ["f1", "auprc", "auroc", "event_recall", "false_positive_rate"] if metric in transfer.columns]]
            .agg(["mean", "std", "count"])
        )
        transfer_summary.columns = [
            *["direction", "target_domain", "requested_target_fraction", "initialization"],
            *[f"{metric}_{stat}" for metric in ["f1", "auprc", "auroc", "event_recall", "false_positive_rate"] if metric in transfer.columns for stat in ["mean", "std", "count"]],
        ]

    ablation = completed[(completed["family"] == "proposed")].copy()
    ablation_summary = pd.DataFrame(columns=[
        "dataset", "variant", "f1", "auprc", "auroc", "event_recall",
        "false_positive_rate", "parameter_count",
    ])
    if not ablation.empty:
        ablation_summary = (
            ablation.groupby(["dataset", "variant"], as_index=False)[
                [metric for metric in ["f1", "auprc", "auroc", "event_recall", "false_positive_rate", "parameter_count"] if metric in ablation.columns]
            ]
            .mean(numeric_only=True)
        )

    completed_standard = completed[completed["family"].isin(["baselines", "proposed"])]
    distinct_seeds = int(completed_standard["seed"].nunique()) if not completed_standard.empty else 0
    full_proposed = completed[(completed["family"] == "proposed") & (completed["variant"] == "full")]
    sota_completed = sorted(set(completed[completed["model"].isin(config["baselines"]["optional_sota_models"])]["model"]))
    transfer_pairs = int(len(transfer_summary)) if not transfer_summary.empty else 0
    scenario_count = int(scenarios["scenario_id"].nunique()) if not scenarios.empty else 0
    checklist = pd.DataFrame(
        [
            {"requirement": "At least three distinct random seeds", "status": "pass" if distinct_seeds >= 3 else "fail", "evidence": distinct_seeds},
            {"requirement": "Balanced severity-duration scenario grid", "status": "pass" if scenario_count >= 24 else "fail", "evidence": scenario_count},
            {"requirement": "Full proposed model evaluated in both domains", "status": "pass" if set(full_proposed["dataset"]) >= {"gas_sensor", "skab"} else "fail", "evidence": ",".join(sorted(set(full_proposed["dataset"])))},
            {"requirement": "At least two modern SOTA comparators", "status": "pass" if len(sota_completed) >= 2 else "pending", "evidence": ",".join(sota_completed)},
            {"requirement": "Proposed-model ablation study", "status": "pass" if ablation["variant"].nunique() >= 4 else "pending", "evidence": int(ablation["variant"].nunique()) if not ablation.empty else 0},
            {"requirement": "Cross-domain transfer/adaptation", "status": "pass" if transfer_pairs > 0 else "pending", "evidence": transfer_pairs},
            {"requirement": "Paired statistical comparison", "status": "pass" if not pairwise.empty else "pending", "evidence": len(pairwise)},
            {"requirement": "Sensor localization metrics", "status": "pass" if "topk_channel_recall" in scenarios.columns and scenarios["topk_channel_recall"].notna().any() else "pending", "evidence": int(scenarios.get("topk_channel_recall", pd.Series(dtype=float)).notna().sum())},
        ]
    )

    outputs = {
        "all_runs": combined / "all_runs.csv",
        "model_summary": combined / "model_summary.csv",
        "scenario_metrics": combined / "scenario_metrics.csv",
        "bootstrap_confidence_intervals": combined / "bootstrap_confidence_intervals.csv",
        "paired_statistical_tests": combined / "paired_statistical_tests.csv",
        "cross_domain_summary": combined / "cross_domain_summary.csv",
        "transfer_summary": combined / "transfer_summary.csv",
        "ablation_summary": combined / "ablation_summary.csv",
        "manuscript_readiness": combined / "manuscript_readiness_checklist.csv",
    }
    atomic_write_dataframe(all_runs, outputs["all_runs"])
    atomic_write_dataframe(summary, outputs["model_summary"])
    atomic_write_dataframe(scenarios, outputs["scenario_metrics"])
    atomic_write_dataframe(bootstrap, outputs["bootstrap_confidence_intervals"])
    atomic_write_dataframe(pairwise, outputs["paired_statistical_tests"])
    atomic_write_dataframe(cross_domain, outputs["cross_domain_summary"])
    atomic_write_dataframe(transfer_summary, outputs["transfer_summary"])
    atomic_write_dataframe(ablation_summary, outputs["ablation_summary"])
    atomic_write_dataframe(checklist, outputs["manuscript_readiness"])
    return outputs
