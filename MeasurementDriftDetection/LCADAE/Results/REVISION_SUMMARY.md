# Experiment Revision Summary — Manuscript V2

## Decision addressed

The preliminary notebooks demonstrated that the pipeline ran, but they were not sufficient for final manuscript claims because repeated executions of seed 42 were counted as multiple runs, SOTA comparators were disabled, anomaly prevalence was high, fault-specific evidence was sparse, and no true cross-domain transfer experiment was performed.

## Revised experimental design

### Independent repetitions

The default seed list is now:

```text
42, 52, 62, 72, 82
```

Every run receives a stable experiment key based on the model, dataset, seed, scenario profile, experiment version, and relevant configuration. A completed key is skipped rather than appended again.

### Clean result namespace

All revised records are saved under:

```text
results/manuscript_v2/
```

The old preliminary CSV files are not read by the revised aggregation code.

### Sparse mixed-fault benchmark

The mixed test now targets approximately 10% point-level anomaly prevalence. It includes six fault families and exact point-by-channel ground-truth masks.

### Balanced cross-domain scenario grid

The manuscript profile creates 288 independent scenarios per domain:

```text
6 fault types × 4 severities × 4 durations × 3 replicates
```

Gas Sensor and SKAB use the same scenario IDs and factor settings. Each scenario is injected independently into clean held-out data, preventing fault overlap and allowing paired comparison.

### Threshold calibration

The primary threshold is obtained only from normal validation scores. The code also records sensitivity for:

- quantiles 0.95, 0.975, 0.99, and 0.995;
- MAD multipliers 4, 6, and 8.

Test labels are never used to choose the primary threshold.

### Natural and native evaluation

- Gas Sensor later batches are evaluated as observational natural shift using threshold-exceedance rates and robust score shift.
- SKAB native files are evaluated against their anomaly labels and are reported as general industrial anomalies, not automatically as measurement drift.

### Sensor localization

The controlled benchmark stores a time-by-channel fault mask. LCAD-AE reports:

- channel AUPRC;
- channel AUROC;
- top-1 localization accuracy;
- top-k channel recall.

### Proposed-model ablation

Four variants are included:

1. reconstruction only;
2. reconstruction plus statistics;
3. reconstruction plus correlation;
4. full LCAD-AE.

### Cross-domain transfer

Only the channel-independent temporal blocks are transferred. Input projection, reconstruction head, and statistics head remain domain-specific. Both directions are tested:

- Gas Sensor to SKAB;
- SKAB to Gas Sensor.

Scratch and transferred initialization are paired at target normal-data fractions of 1%, 5%, 10%, 25%, and 100%.

### Statistical analysis

Notebook 04 generates:

- scenario-level bootstrap confidence intervals;
- paired Wilcoxon signed-rank tests;
- Holm-adjusted p-values;
- cross-domain average and worst-domain results;
- ablation and transfer summaries;
- a manuscript-readiness checklist.

## Revised files

```text
notebooks/01_data_preparation.ipynb
notebooks/02_baseline_models.ipynb
notebooks/03_proposed_model.ipynb
notebooks/04_results_aggregation.ipynb
configs/default.yaml
src/drift_detection/fault_injection.py
src/drift_detection/benchmark.py
src/drift_detection/windows.py
src/drift_detection/metrics.py
src/drift_detection/evaluation_v2.py
src/drift_detection/experiments_v2.py
src/drift_detection/aggregation_v2.py
scripts/smoke_test.py
README.md
METHODOLOGY_UPGRADE_PLAN.md
```

## Validation performed

- All Python modules compile successfully.
- All notebook code cells parse successfully.
- Notebook structures pass `nbformat` validation.
- DenseAE and LCAD-AE smoke tests complete successfully.
- Synthetic end-to-end integration tests complete for baseline, proposed, aggregation, duplicate prevention, and cross-domain transfer.

The real datasets were not included in the archive and the final experiments have not been rerun inside this packaging environment. Notebook 01 must be executed first on the user's downloaded Gas Sensor and SKAB data.
