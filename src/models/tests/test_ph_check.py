"""Proportional hazards check (P1 7) on simulated survival data.

    .venv/bin/python -m pytest src/models/tests/test_ph_check.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ph_check  # noqa: E402
from train import DURATION, EVENT  # noqa: E402


def simulate(early_only: bool, n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """x doubles the hazard. With early_only, only until day 10: after it, x has no effect."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    base = 0.02
    t = rng.exponential(1 / (base * np.exp(0.7 * x)))
    if early_only:
        late = t > 10                                  # survived the early period: memoryless restart, no effect
        t[late] = 10 + rng.exponential(1 / base, late.sum())
    t = np.floor(t)
    observed = (t <= 200).astype(int)
    return pd.DataFrame({"x": x, DURATION: np.minimum(t, 200), EVENT: observed})


def fitted(frame):
    return CoxPHFitter().fit(frame, duration_col=DURATION, event_col=EVENT)


def test_proportional_effect_is_stable():
    frame = simulate(early_only=False)
    beta_t, events, _ = ph_check.beta_by_window(fitted(frame), frame)
    assert events.sum() == frame[EVENT].sum()
    assert ph_check.verdict(beta_t.loc["x"]) == "stable"


def test_effect_that_fades_is_flagged():
    frame = simulate(early_only=True)
    beta_t, _, rho = ph_check.beta_by_window(fitted(frame), frame)
    assert ph_check.verdict(beta_t.loc["x"]) in {"changes size", "sign changes"}
    assert beta_t.loc["x", "over 90"] < beta_t.loc["x", "day 0"] and rho["x"] < 0


def test_verdict_rules():
    labels = [w[2] for w in ph_check.WINDOWS]
    v = lambda values: ph_check.verdict(pd.Series(values, index=labels))  # noqa: E731
    assert v([0.01, -0.02, 0.0, 0.03, -0.01]) == "negligible"
    assert v([0.25, -0.22, -0.02, 0.0, 0.04]) == "sign changes"
    assert v([-0.50, -0.33, -0.25, -0.26, -0.20]) == "changes size"
    assert v([-0.62, -0.65, -0.50, -0.52, -0.50]) == "stable"
