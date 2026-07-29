# Revised Manuscript Experiment Plan

## Primary claim

The study evaluates whether a lightweight deep model can detect systematic measurement drift across heterogeneous industrial sensor domains and whether temporal representations transfer between those domains when target normal data are limited.

## Evidence layers

### Controlled sparse mixed-fault test

One event from each fault family is inserted into a held-out normal series. The total point-level anomaly prevalence is targeted near 10%. This test supports operational metrics and representative score plots.

### Balanced scenario grid

Every model is trained once per dataset and seed, then evaluated on independent scenarios sharing the same protocol in Gas Sensor and SKAB. The manuscript profile includes 288 scenarios per domain: six fault types, four severities, four durations, and three replicates. Scenario-macro AUPRC and F1 are the primary ranking measures.

### Natural and native evaluation

Gas Sensor later batches are summarized by threshold exceedance and robust score shift because they do not provide exact instrument-fault labels. SKAB native sequences are evaluated against their published anomaly labels and reported as general industrial anomaly detection, not automatically as sensor drift.

## Independent repetitions

All final experiments use five distinct seeds. Stable experiment keys prevent the same seed and configuration from being counted more than once.

## Threshold calibration

The primary threshold is derived only from normal validation scores. Quantile and robust MAD thresholds are evaluated in a sensitivity analysis. Test labels are never used to select the primary threshold.

## Sensor localization

Controlled scenarios store an exact time-by-channel mask. The proposed model is evaluated with channel AUPRC, channel AUROC, top-1 localization accuracy, and top-k channel recall.

## Ablation

Four LCAD-AE variants isolate the contribution of the statistics and correlation terms:

1. reconstruction only;
2. reconstruction plus statistics;
3. reconstruction plus correlation;
4. full LCAD-AE.

## Cross-domain transfer

The temporal blocks are pretrained in one domain and transferred to the other. Domain-specific projections and heads are newly initialized. Scratch and transfer conditions are paired at target normal fractions of 1%, 5%, 10%, 25%, and 100%.

## Statistical analysis

Scenario observations are paired by dataset, seed, and scenario ID. The aggregation notebook generates bootstrap confidence intervals and Wilcoxon signed-rank tests with Holm correction.

## Submission gate

Final Results and Discussion should be written only after the readiness checklist confirms:

- at least three distinct seeds, preferably all five;
- full proposed-model results in both domains;
- completed ablations;
- completed transfer experiments;
- at least two modern SOTA comparators;
- paired statistical tests;
- sensor localization results.
