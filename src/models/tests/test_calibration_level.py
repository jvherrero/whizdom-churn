"""Calibration level (calibration_level.py) on hand-made data: no S3, no MLflow.

    .venv/bin/python -m pytest src/models/tests -q      (make test)
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calibration_level as cl  # noqa: E402
from train import BRAND, TARGET  # noqa: E402

D = dt.date


def test_cutoffs_matured_and_early():
    # On 2026-03-01: 2026-01-01 + 60 days is 2026-03-02, so January is early (59 days seen), not matured.
    assert cl.cutoffs(D(2026, 3, 1)) == ([D(2025, 10, 1), D(2025, 11, 1), D(2025, 12, 1)], [D(2026, 1, 1), D(2026, 2, 1)])
    assert cl.cutoffs(D(2026, 10, 6)) == ([D(2026, 6, 1), D(2026, 7, 1), D(2026, 8, 1)], [D(2026, 9, 1)])
    # 2026-10-01 has 9 days seen on 2026-10-10: fewer than MIN_EARLY_DAYS, left out.
    assert cl.cutoffs(D(2026, 10, 10))[1] == [D(2026, 9, 1)]


def test_needed_shift_sets_the_mean_and_keeps_the_order():
    p = np.random.default_rng(0).uniform(0.05, 0.9, 2000)
    shift = cl.needed_shift(p, 0.6)
    shifted = cl.apply(p, np.full(len(p), 64), {64: shift})
    assert shifted.mean() == pytest.approx(0.6, abs=1e-6)
    assert (np.argsort(p, kind="stable") == np.argsort(shifted, kind="stable")).all()
    assert cl.needed_shift(p, 0.0) is None and cl.needed_shift(p, 1.0) is None


def test_apply_leaves_a_brand_without_shift_unchanged():
    p = np.array([0.2, 0.5, 0.8])
    out = cl.apply(p, [64, 14, 14], {14: 1.0})
    assert out[0] == pytest.approx(0.2) and (out[1:] > p[1:]).all()


def test_fit_median_ignores_one_odd_month():
    rng = np.random.default_rng(1)
    parts = []
    for i, rate in enumerate([0.40, 0.41, 0.70]):  # the last month is a peak
        p = rng.uniform(0.2, 0.6, 5000)
        parts.append(pd.DataFrame({"cutoff_date": D(2026, 1 + i, 1), BRAND: 64, "p": p,
                                   TARGET: rng.binomial(1, np.clip(p * rate / p.mean(), 0, 1))}))
    matured = pd.concat(parts, ignore_index=True)
    shift = cl.fit(lambda rows: rows["p"].to_numpy(), matured, None, D(2026, 6, 1))[64]
    assert abs(shift) < 0.15  # the median follows the two ordinary months, not the peak (about +1.2)


def test_early_rate(monkeypatch):
    # Complete cohort on 2026-01-01: 10 players; on 2026-03-01 (k = 28 days for the early cohort of 2026-02-01):
    # 2 bet within 28 days, 2 more between 28 and 60 days, 6 never. P(no bet in 60 | none in 28) = 6 / 8.
    matured = pd.DataFrame({"cutoff_date": D(2026, 1, 1), "tenant_id": "t", "player_id": range(10)})
    early = pd.DataFrame({"cutoff_date": D(2026, 2, 1), "tenant_id": "t", "player_id": range(100, 104)})
    bets = [(0, 5), (1, 20), (2, 40), (3, 55),          # days after 2026-01-01
            (100, 31 + 3), (101, 31 + 10)]              # early cohort: 2 of 4 bet again by 2026-03-01
    act = pd.DataFrame([{"tenant_id": "t", "player_id": p, "day": D(2026, 1, 1) + dt.timedelta(days=d)} for p, d in bets])
    monkeypatch.setattr(cl.daily_cache, "read", lambda *a, **k: act.copy())
    assert cl.early_rate(early, matured, D(2026, 3, 1), 64) == pytest.approx(0.5 * 6 / 8)
