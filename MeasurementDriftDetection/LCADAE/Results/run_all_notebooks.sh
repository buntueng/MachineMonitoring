#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

for notebook in 01_data_preparation 02_baseline_models 03_proposed_model 04_results_aggregation; do
  echo "[$(date -Is)] Starting ${notebook}.ipynb" | tee -a logs/run_all_notebooks.log
  jupyter nbconvert --to notebook --execute "notebooks/${notebook}.ipynb" \
    --ExecutePreprocessor.timeout=-1 \
    --output "${notebook}.executed.ipynb" \
    --output-dir notebooks 2>&1 | tee -a logs/run_all_notebooks.log
  echo "[$(date -Is)] Finished ${notebook}.ipynb" | tee -a logs/run_all_notebooks.log
done
