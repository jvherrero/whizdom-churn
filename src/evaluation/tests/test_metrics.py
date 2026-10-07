"""Unit tests of src/evaluation/metrics.py on synthetic data with known answers.

    .venv/bin/python -m pytest src/evaluation/tests -q      (make test)
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from lifelines import KaplanMeierFitter
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.metrics import (  # noqa: E402
    bootstrap_ci, brier_contributions, c_index, classification_metrics, expected_calibration_error,
    integrated_brier_score, ipcw_weights, one_calibration, time_dependent_auc, top_decile_precision,
)

RNG = np.random.default_rng(0)


def _no_censoring_km(n):
    """Censoring curve of a sample with no censoring: G(t) = 1 everywhere."""
    return KaplanMeierFitter().fit(np.full(n, 1000.0), event_observed=np.zeros(n))


# ---------------------------------------------------------------- classification

def test_ece_is_small_for_calibrated_probabilities():
    p = RNG.uniform(0, 1, 50_000)
    y = RNG.uniform(0, 1, p.size) < p
    assert expected_calibration_error(y, p) < 0.01


def test_ece_measures_a_constant_offset():
    p = RNG.uniform(0.2, 0.6, 50_000)
    y = RNG.uniform(0, 1, p.size) < p
    assert expected_calibration_error(y, p + 0.1) == pytest.approx(0.1, abs=0.01)


def test_top_decile_precision():
    y = np.array([1] * 10 + [0] * 90)
    assert top_decile_precision(y, np.arange(100)[::-1]) == 1.0  # churners ranked first
    assert top_decile_precision(y, np.arange(100)) == 0.0         # churners ranked last


def test_classification_metrics_lift():
    y = np.array([1] * 10 + [0] * 90)
    m = classification_metrics(y, np.linspace(0.99, 0.01, 100))
    assert m["roc_auc"] == 1.0
    assert m["top_decile_precision"] == 1.0
    assert m["top_decile_lift"] == pytest.approx(10.0)
    assert m["churn_rate"] == pytest.approx(0.1)


# ---------------------------------------------------------------- survival

def test_c_index_perfect_reversed_and_random():
    d = np.arange(1, 201, dtype=float)
    e = np.ones_like(d)
    assert c_index(d, -d, e) == 1.0   # higher risk = earlier churn
    assert c_index(d, d, e) == 0.0
    assert c_index(d, RNG.normal(size=d.size), e) == pytest.approx(0.5, abs=0.1)


def test_time_dependent_auc_without_censoring_is_the_plain_auc():
    n, t = 5000, 30
    d = RNG.integers(0, 90, n).astype(float)
    e = np.ones(n)
    risk = -d + RNG.normal(0, 25, n)
    expected = roc_auc_score(d <= t, risk)  # cases churned by t, controls still active after t
    weights = ipcw_weights(d, _no_censoring_km(n))
    assert np.allclose(weights, 1.0)
    assert time_dependent_auc(d, e, risk, t, weights) == pytest.approx(expected, abs=1e-9)


def test_time_dependent_auc_is_nan_without_cases():
    d = np.full(10, 50.0)
    assert np.isnan(time_dependent_auc(d, np.ones(10), RNG.normal(size=10), 7, np.ones(10)))


def test_one_calibration_accepts_the_true_probability():
    t, n = 30, 20_000
    rate = RNG.choice([0.005, 0.02, 0.05], n)  # three risk groups with known hazards
    d_true = RNG.exponential(1 / rate)
    d, e = np.floor(np.minimum(d_true, 60)), (d_true <= 60).astype(int)
    truth = 1 - np.exp(-rate * (t + 1))  # floor(T) <= t  <=>  T < t + 1
    result = one_calibration(d, e, truth, t)
    assert result["mean_abs_gap"] < 0.02
    assert result["p_value"] > 0.01


def test_one_calibration_rejects_a_halved_probability():
    t, n = 30, 20_000
    rate = RNG.choice([0.005, 0.02, 0.05], n)
    d_true = RNG.exponential(1 / rate)
    d, e = np.floor(np.minimum(d_true, 60)), (d_true <= 60).astype(int)
    truth = 1 - np.exp(-rate * (t + 1))
    result = one_calibration(d, e, truth / 2, t)
    assert result["mean_abs_gap"] > 0.1
    assert result["p_value"] < 1e-6


def test_integrated_brier_score_perfect_and_coin_flip():
    n = 1000
    d = RNG.integers(0, 30, n).astype(float)
    e = np.ones(n)
    grid = np.arange(0, 31)
    km = _no_censoring_km(n)
    perfect = (grid[None, :] < d[:, None]).astype(float)  # S(t) = 1 until the churn day, then 0
    assert integrated_brier_score(brier_contributions(perfect, grid, d, e, km), grid) == pytest.approx(0.0)
    coin = np.full((n, grid.size), 0.5)
    assert integrated_brier_score(brier_contributions(coin, grid, d, e, km), grid) == pytest.approx(0.25)


# ---------------------------------------------------------------- uncertainty

def test_bootstrap_ci_contains_the_estimate_and_is_reproducible():
    x = RNG.normal(10, 2, 2000)
    low, high = bootstrap_ci(x.size, lambda i: x[i].mean(), n_resamples=300)
    assert low < x.mean() < high
    assert high - low == pytest.approx(2 * 1.96 * 2 / np.sqrt(x.size), rel=0.25)
    assert (low, high) == bootstrap_ci(x.size, lambda i: x[i].mean(), n_resamples=300)
