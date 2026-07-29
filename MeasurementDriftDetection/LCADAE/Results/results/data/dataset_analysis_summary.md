# Dataset Analysis Summary

## Gas Sensor domain

- The dataset contains 13910 observations, 128 engineered channels, and 10 temporal batches.
- After gas identity and concentration adjustment, the mean absolute residual changes from 0.4918 in Batch 1 to 0.9097 in Batch 10.
- Batch 4 has the largest mean absolute residual (1.4384) and should be treated as the strongest natural-shift period, not automatically as a labeled fault period.
- The batch-composition table must be examined together with the residual drift index because gas-class imbalance can otherwise be mistaken for instrumental drift.

## SKAB domain

- The parsed SKAB collection contains 46806 observations across 1 normal files and 34 files containing operating disturbances or faults.
- Native labels contain 13067 anomaly points and 129 changepoints.
- The sequences with the largest correlation departure from the normal reference are:
  - `9.csv`: correlation distance 0.6284, anomaly ratio 0.3505.
  - `13.csv`: correlation distance 0.5609, anomaly ratio 0.2871.
  - `14.csv`: correlation distance 0.5493, anomaly ratio 0.3337.

## Cross-domain benchmark

- gas: 128 channels, 415 training windows, 311 controlled-test windows, and an anomaly-point ratio of 0.0975.
- skab: 8 channels, 1403 training windows, 463 controlled-test windows, and an anomaly-point ratio of 0.0925.

## Interpretation boundary

Natural later-batch shift in the Gas Sensor dataset and native SKAB anomalies are observational evidence. Controlled injected faults provide exact onset, type, duration, severity, and affected-channel ground truth. These two evidence sources must be reported separately in the manuscript.
