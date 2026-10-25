import numpy as np

from playerpulse.monitoring import feature_psi, population_stability_index, risk_band


def test_psi_returns_none_until_sample_is_large_enough():
    reference = {"edges": [1.0, 3.0], "proportions": [0.25, 0.5, 0.25]}

    assert feature_psi([0.5] * 19, reference) is None


def test_psi_detects_large_distribution_shift():
    reference = {"edges": [1.0, 3.0], "proportions": [0.25, 0.5, 0.25]}
    value = feature_psi([10.0] * 40, reference)

    assert value is not None
    assert value > 0.2


def test_population_stability_index_requires_matching_bins():
    assert np.isclose(population_stability_index(np.array([0.5, 0.5]), np.array([0.5, 0.5])), 0)


def test_risk_bands_have_readable_text_buckets():
    assert risk_band(0.2) == "low"
    assert risk_band(0.5) == "medium"
    assert risk_band(0.9) == "high"
