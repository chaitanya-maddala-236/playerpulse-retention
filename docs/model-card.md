# Model card

## Identity and intended use

The model is a synthetic-data portfolio demonstration for game-retention engineering. Version, training date, dependencies, seed, feature schema, threshold, and evaluation data are written to `artifacts/models/playerpulse-synthetic-v1/metadata.json` and `metrics.json`. It is not validated on real players and should not drive interventions.

## Training and evaluation

Candidate pipelines include a dummy prior, logistic regression, class-weighted logistic regression, random forest, class-weighted random forest, and histogram gradient boosting. Imputation/scaling are inside each training pipeline. A later chronological validation window chooses model by PR-AUC and classification threshold by F1. The test set remains held out until final evaluation. The chosen model is refit on train+validation, then measured on the test partition once. Confusion matrix, PR-AUC, ROC-AUC, precision, recall, F1, Brier score, calibration, and validation permutation importance are recorded.

## Features and explanations

Inputs aggregate only the prior 7 and 30 days of sessions and playtime plus last-activity age. Player IDs and future information are excluded. Local explanations replace one feature at a time with its training median and report the probability difference. These are associations under a perturbation, not causal attribution.

## Limitations and fairness

Synthetic telemetry has simplified, hand-chosen activity rates. There is no real-world predictive validity, subgroup data, fairness analysis, calibration evidence for real populations, or intervention study. Protected characteristics are neither inferred nor included. Do not target or manipulate players with these scores.

## Monitoring and release review

Compare feature drift, prediction distribution, service health, and only matured outcome labels. A drift or performance signal prompts review; it does not prove model failure and never automatically replaces the model. Before promoting a candidate, use a licensed representative dataset, review label coverage/leakage and relevant subgroups, tune on validation, evaluate on a future test set, check artifact and API compatibility, and approve a monitored release.
