from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from playerpulse import __version__
from playerpulse.synthetic_data import (
    FEATURE_COLUMNS,
    GENERATOR_VERSION,
    TARGET_COLUMN,
    feature_frame,
    generate_dataset,
    split_chronologically,
)

MODEL_VERSION = "playerpulse-synthetic-v1"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models" / MODEL_VERSION


def _pipeline(estimator, scale_features: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def candidate_models(seed: int = 42) -> dict[str, Pipeline]:
    return {
        "dummy_prior": _pipeline(DummyClassifier(strategy="prior")),
        "logistic_regression": _pipeline(
            LogisticRegression(max_iter=1_000, random_state=seed), scale_features=True
        ),
        "logistic_regression_balanced": _pipeline(
            LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=seed),
            scale_features=True,
        ),
        "random_forest": _pipeline(
            RandomForestClassifier(
                n_estimators=160,
                max_depth=9,
                min_samples_leaf=4,
                random_state=seed,
                n_jobs=-1,
            )
        ),
        "random_forest_balanced": _pipeline(
            RandomForestClassifier(
                n_estimators=160,
                max_depth=9,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )
        ),
        "hist_gradient_boosting": _pipeline(
            HistGradientBoostingClassifier(
                max_iter=140,
                learning_rate=0.08,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=seed,
            )
        ),
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    _, _, thresholds = precision_recall_curve(labels, probabilities)
    candidates = np.unique(np.append(thresholds, 0.5))
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        score = f1_score(labels, probabilities >= threshold, zero_division=0)
        if score > best_f1 or (np.isclose(score, best_f1) and threshold > best_threshold):
            best_threshold = float(threshold)
            best_f1 = float(score)
    return best_threshold, best_f1


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = probabilities >= threshold
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    precision, recall, pr_thresholds = precision_recall_curve(labels, probabilities)
    calibration_observed, calibration_predicted = calibration_curve(
        labels, probabilities, n_bins=10, strategy="uniform"
    )
    return {
        "n_samples": int(len(labels)),
        "n_churned": int(np.asarray(labels).sum()),
        "churn_prevalence": float(np.asarray(labels).mean()),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
        "precision_recall_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
        "calibration": {
            "predicted_probability": calibration_predicted.tolist(),
            "observed_prevalence": calibration_observed.tolist(),
        },
    }


def _drift_reference(features: pd.DataFrame) -> dict:
    reference = {}
    for column in FEATURE_COLUMNS:
        values = features[column].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))[1:-1]
        counts, _ = np.histogram(values, bins=np.concatenate(([-np.inf], edges, [np.inf])))
        reference[column] = {
            "edges": edges.tolist(),
            "proportions": (counts / counts.sum()).tolist(),
        }
    return reference


def _installed_versions() -> dict[str, str]:
    installed = {}
    for package in ("numpy", "pandas", "scikit-learn", "joblib"):
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            installed[package] = "not-installed"
    return installed


def train_model(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    n_players: int = 3_000,
) -> tuple[dict, dict]:
    dataset = generate_dataset(n_players=n_players, seed=seed)
    training, validation, test = split_chronologically(dataset)
    x_training = feature_frame(training)
    x_validation = feature_frame(validation)
    x_test = feature_frame(test)
    y_training = training[TARGET_COLUMN].to_numpy()
    y_validation = validation[TARGET_COLUMN].to_numpy()
    y_test = test[TARGET_COLUMN].to_numpy()

    candidate_results = []
    fitted_candidates = {}
    for name, model in candidate_models(seed).items():
        model.fit(x_training, y_training)
        probabilities = model.predict_proba(x_validation)[:, 1]
        threshold, threshold_f1 = select_threshold(y_validation, probabilities)
        candidate_results.append(
            {
                "model": name,
                "validation": {
                    "n_samples": len(validation),
                    "churn_prevalence": float(y_validation.mean()),
                    "pr_auc": float(average_precision_score(y_validation, probabilities)),
                    "roc_auc": float(roc_auc_score(y_validation, probabilities)),
                    "precision": float(
                        precision_score(y_validation, probabilities >= threshold, zero_division=0)
                    ),
                    "recall": float(
                        recall_score(y_validation, probabilities >= threshold, zero_division=0)
                    ),
                    "f1": threshold_f1,
                    "selected_threshold": threshold,
                },
            }
        )
        fitted_candidates[name] = model

    selected_result = max(
        candidate_results,
        key=lambda result: (result["validation"]["pr_auc"], result["validation"]["f1"]),
    )
    selected_name = selected_result["model"]
    selected_threshold = selected_result["validation"]["selected_threshold"]
    selected_validation_probabilities = fitted_candidates[selected_name].predict_proba(
        x_validation
    )[:, 1]
    validation_metrics = classification_metrics(
        y_validation, selected_validation_probabilities, selected_threshold
    )
    validation_importance = permutation_importance(
        fitted_candidates[selected_name],
        x_validation,
        y_validation,
        scoring="average_precision",
        n_repeats=3,
        random_state=seed,
        n_jobs=-1,
    )
    validation_importance_report = sorted(
        [
            {
                "feature": feature,
                "mean_pr_auc_decrease": float(mean),
                "std_pr_auc_decrease": float(std),
            }
            for feature, mean, std in zip(
                FEATURE_COLUMNS,
                validation_importance.importances_mean,
                validation_importance.importances_std,
                strict=True,
            )
        ],
        key=lambda entry: entry["mean_pr_auc_decrease"],
        reverse=True,
    )

    final_model = clone(candidate_models(seed)[selected_name])
    final_model.fit(
        pd.concat([x_training, x_validation], ignore_index=True),
        np.concatenate([y_training, y_validation]),
    )
    test_probabilities = final_model.predict_proba(x_test)[:, 1]
    test_metrics = classification_metrics(y_test, test_probabilities, selected_threshold)
    metrics = {
        "synthetic_demo": True,
        "dataset_version": GENERATOR_VERSION,
        "random_seed": seed,
        "split": {
            "strategy": "chronological; one snapshot per player",
            "n_train": len(training),
            "n_validation": len(validation),
            "n_test": len(test),
            "train_end": training["prediction_at"].max().isoformat(),
            "validation_end": validation["prediction_at"].max().isoformat(),
            "test_start": test["prediction_at"].min().isoformat(),
            "train_churn_prevalence": float(y_training.mean()),
            "validation_churn_prevalence": float(y_validation.mean()),
            "test_churn_prevalence": float(y_test.mean()),
        },
        "candidates": candidate_results,
        "selected_model": selected_name,
        "threshold_selected_on_validation": selected_threshold,
        "validation": validation_metrics,
        "test": test_metrics,
        "feature_importance_validation": validation_importance_report,
        "test_evaluated_at_utc": datetime.now(UTC).isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": final_model,
        "model_version": MODEL_VERSION,
        "features": list(FEATURE_COLUMNS),
        "threshold": selected_threshold,
        "training_feature_medians": {
            name: float(x_training[name].median()) for name in FEATURE_COLUMNS
        },
        "metrics": metrics,
        "drift_reference": _drift_reference(x_training),
    }
    joblib.dump(artifact, output_dir / "model.joblib", compress=3)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
    metadata = {
        "model_version": MODEL_VERSION,
        "application_version": __version__,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_version": GENERATOR_VERSION,
        "dataset_is_synthetic": True,
        "random_seed": seed,
        "n_players": len(dataset),
        "prediction_horizon_days": 14,
        "feature_window_days": 30,
        "selected_model": selected_name,
        "threshold": selected_threshold,
        "features": list(FEATURE_COLUMNS),
        "training_feature_reference": artifact["drift_reference"],
        "evaluation_file": "metrics.json",
        "python_version": platform.python_version(),
        "dependencies": _installed_versions(),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
    return metadata, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the synthetic PlayerPulse model."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--players", type=int, default=3_000)
    arguments = parser.parse_args()
    metadata, metrics = train_model(arguments.output_dir, arguments.seed, arguments.players)
    print(
        json.dumps(
            {
                "artifact_dir": str(arguments.output_dir.resolve()),
                "model_version": metadata["model_version"],
                "selected_model": metrics["selected_model"],
                "synthetic_test_metrics": {
                    metric: metrics["test"][metric]
                    for metric in (
                        "n_samples",
                        "n_churned",
                        "churn_prevalence",
                        "pr_auc",
                        "roc_auc",
                        "precision",
                        "recall",
                        "f1",
                        "brier_score",
                        "threshold",
                        "confusion_matrix",
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
