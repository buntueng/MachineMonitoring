# Cross-Domain Systematic Measurement Drift Detection

Research project for:

**Cross-Domain Validation of Deep Learning-Based Systematic Measurement Drift Detection Using Industrial Sensor Data**

The revised project evaluates Gas Sensor Array Drift at Different Concentrations and SKAB using a common, auditable protocol. All user-facing notebook text, logs, tables, and saved records are in English.

## Why the experiments were revised

The preliminary run was useful as a smoke test, but it used only seed 42, appended repeated executions as separate runs, used a high anomaly ratio, disabled the optional SOTA models, and averaged two independently trained domains without a true transfer experiment.

The `manuscript_v2` protocol corrects these limitations:

- five independent seeds: 42, 52, 62, 72, and 82;
- stable experiment keys and duplicate-run prevention;
- a sparse mixed-fault test near 10% point prevalence;
- a balanced scenario grid shared across both domains;
- fixed severity, duration, replicate, start, and channel-fraction factors;
- validation-only quantile and MAD threshold sensitivity;
- exact point-by-channel ground-truth masks;
- proposed-model component ablations;
- natural Gas Sensor and native SKAB evaluation;
- cross-domain shared-temporal-encoder transfer;
- bootstrap confidence intervals and paired Wilcoxon tests with Holm correction.

Old preliminary files remain outside `results/manuscript_v2/` and are not included in revised aggregation.

## Project structure

```text
cross_domain_drift_detection/
├── configs/default.yaml
├── data/
│   ├── raw/
│   └── processed/benchmarks/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_proposed_model.ipynb
│   └── 04_results_aggregation.ipynb
├── scripts/
│   ├── download_datasets.py
│   └── smoke_test.py
├── src/drift_detection/
│   ├── benchmark.py
│   ├── fault_injection.py
│   ├── evaluation_v2.py
│   ├── experiments_v2.py
│   ├── aggregation_v2.py
│   └── ...
└── results/manuscript_v2/
```

## Installation

Create an environment and install the core dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install published SOTA comparators before the final manuscript run:

```bash
python -m pip install -r requirements-sota.txt
```

Use a CUDA-enabled PyTorch installation appropriate for the local GPU when GPU acceleration is required.

## Download datasets

```bash
python scripts/download_datasets.py --all
```

Notebook 01 can also download the datasets when `DOWNLOAD_DATASETS = True`.

## Validate the installation

```bash
python scripts/smoke_test.py
```

## Run the study

Open JupyterLab:

```bash
jupyter lab
```

Run the notebooks in order:

1. `01_data_preparation.ipynb`
2. `02_baseline_models.ipynb`
3. `03_proposed_model.ipynb`
4. `04_results_aggregation.ipynb`

For an uninterrupted command-line execution:

```bash
bash run_all_notebooks.sh
```

## Quick and manuscript profiles

The default configuration uses:

```yaml
experiment:
  profile: manuscript
```

The manuscript profile generates 288 independent scenarios in each domain:

- 6 fault families;
- 4 severity levels;
- 4 duration levels;
- 3 replicates.

Use `quick` only to validate code paths. Results from the quick profile must not be reported in the manuscript.

## Baselines

Core models:

- PCA-SPE;
- Isolation Forest;
- Dense Autoencoder;
- LSTM Autoencoder;
- Convolutional Autoencoder;
- USAD.

Optional SOTA models through DeepOD:

- TranAD;
- Anomaly Transformer;
- DCdetector.

Notebook 04 marks the SOTA requirement as pending until at least two of these models complete successfully.

## Proposed model

**LCAD-AE: Lightweight Context-Aware Dual-Scale Autoencoder** combines:

- depthwise-separable temporal convolution;
- local reconstruction error;
- window-level mean and variance consistency;
- cross-channel correlation consistency;
- per-channel anomaly contributions.

Ablations evaluate reconstruction only, reconstruction plus statistics, reconstruction plus correlation, and the full model.

## Cross-domain experiment

Gas Sensor and SKAB have different channel dimensions. Raw input layers therefore cannot be transferred directly. The revised experiment transfers only the channel-independent temporal blocks and keeps domain-specific input, reconstruction, and statistics heads.

For each direction, the target model is trained with 1%, 5%, 10%, 25%, and 100% of the available target normal windows. Scratch and transferred initialization use the same target data and evaluation scenarios.

## Main output files

```text
results/manuscript_v2/combined/
├── all_runs.csv
├── model_summary.csv
├── scenario_metrics.csv
├── bootstrap_confidence_intervals.csv
├── paired_statistical_tests.csv
├── cross_domain_summary.csv
├── transfer_summary.csv
├── ablation_summary.csv
└── manuscript_readiness_checklist.csv
```

Run-level, threshold-sensitivity, fault-specific, natural/native, and representative-score records remain available under the `baselines`, `proposed`, and `transfer` subdirectories.

## Interpretation boundary

Controlled injected scenarios provide exact fault type, onset, duration, severity, and affected-channel ground truth. Natural later-batch change in Gas Sensor and native SKAB labels provide complementary observational evidence and must be reported separately. Native SKAB anomalies must not all be described as measurement drift.
