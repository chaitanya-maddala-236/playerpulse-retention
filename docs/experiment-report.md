# Synthetic experiment report

## Scope

This report records the reproducible run from `python -m playerpulse.train` with seed `42` and generator `synthetic-events-v1`. It is a software-pipeline check on generated data, not evidence about real players. The generator produced 3,000 unique player snapshots with a 14-day future inactivity label and five past-only activity features.

The chronological, player-disjoint split contains 2,100 training, 450 validation, and 450 test rows. Churn prevalence is 27.0% in training, 25.3% in validation, and 23.8% in test. Validation PR-AUC selected the class-weighted logistic regression candidate; validation F1 selected the 0.4275 threshold. The test partition was not used for model or threshold selection.

## Validation comparison

| Candidate | PR-AUC | ROC-AUC | F1 |
| --- | ---: | ---: | ---: |
| Dummy prior | 0.2533 | 0.5000 | 0.4043 |
| Logistic regression | 0.6321 | 0.8839 | 0.7152 |
| Class-weighted logistic regression | 0.6322 | 0.8840 | 0.7171 |
| Random forest | 0.6139 | 0.8746 | 0.7211 |
| Class-weighted random forest | 0.6228 | 0.8772 | 0.7206 |
| Histogram gradient boosting | 0.5943 | 0.8684 | 0.7197 |

The selected candidate had slightly higher PR-AUC than the alternatives. The unweighted random forest had a marginally higher validation F1, but PR-AUC is the predeclared primary selection metric for this imbalanced task.

## Held-out test

| Metric | Result |
| --- | ---: |
| Samples | 450 |
| Churn prevalence | 23.8% |
| PR-AUC | 0.5802 |
| ROC-AUC | 0.8789 |
| Precision | 0.4929 |
| Recall | 0.9720 |
| F1 | 0.6541 |
| Brier score | 0.1544 |
| Threshold selected on validation | 0.4275 |

Confusion matrix at the selected threshold (rows are actual `[no churn, churn]`, columns are predicted `[no churn, churn]`): `[[236, 107], [3, 104]]`.

Validation permutation importance identifies `sessions_30d` as the largest positive contributor (mean PR-AUC decrease `0.2858`). Other importance estimates are small or negative in this run; permutation estimates can vary and are not causal explanations.

## Reproduction and artifacts

Run `python -m playerpulse.train --seed 42 --players 3000 --output-dir artifacts/models/playerpulse-synthetic-v1`. The command writes the fitted pipeline, schema, threshold, drift reference, metadata, validation and test metrics, calibration points, and candidate comparison to the versioned artifact directory. Metrics are calculated by scikit-learn against the generated validation and held-out partitions.

The model card documents intended use and limitations. No generalization, subgroup fairness, calibration, or business-impact claim follows from this synthetic experiment.
