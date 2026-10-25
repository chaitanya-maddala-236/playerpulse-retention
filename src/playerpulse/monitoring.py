from __future__ import annotations

import numpy as np


def population_stability_index(expected: np.ndarray, observed: np.ndarray) -> float:
    expected = np.asarray(expected, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if expected.shape != observed.shape or expected.size == 0:
        raise ValueError("Expected and observed distributions must have matching non-empty shapes.")
    expected = np.clip(expected, 1e-6, None)
    observed = np.clip(observed, 1e-6, None)
    expected = expected / expected.sum()
    observed = observed / observed.sum()
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def feature_psi(values: list[float], reference: dict) -> float | None:
    if len(values) < 20:
        return None
    edges = np.asarray(reference["edges"], dtype=float)
    bins = np.concatenate(([-np.inf], edges, [np.inf]))
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=bins)
    return population_stability_index(np.asarray(reference["proportions"]), counts)


def risk_band(probability: float) -> str:
    if probability >= 0.65:
        return "high"
    if probability >= 0.35:
        return "medium"
    return "low"


def activity_segment(features: dict[str, float]) -> str:
    sessions = features["sessions_30d"]
    if sessions == 0:
        return "no_recent_activity"
    if sessions < 5:
        return "low_activity"
    if sessions < 15:
        return "moderate_activity"
    return "high_activity"
