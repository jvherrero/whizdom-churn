"""Survival data card (P1 10), on synthetic rows.

    .venv/bin/python -m pytest src/models/tests/test_survival_card.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import survival_card  # noqa: E402


def rows() -> pd.DataFrame:
    # player 1: churn on the cutoff; 2: bets until day 20, then churns; 3: censored (last bet on day 100)
    return pd.DataFrame({"cutoff_date": ["2026-01-01"] * 3, "player_id": [1, 2, 3],
                         "days_since_first_bet": np.log1p([10, 200, 150]),
                         "event_60d": [1, 0, 0], "duration_days": [0, 20, 100], "event_observed": [1, 1, 0]})


def test_summary_counts_censoring_events_and_cap():
    s = survival_card.summary(rows())
    assert np.isclose(s["censored"], 1 / 3) and s["event_at_0"] == 0.5
    assert s["later_q"][1] == 20 and s["censored_median"] == 100
    assert np.isclose(s["tenure_capped"], 2 / 3) and s["consistent"] == 1.0


def test_label_check_catches_a_mismatch():
    bad = rows()
    bad.loc[1, "event_60d"] = 1                      # churn label says day 0, survival says day 20
    assert np.isclose(survival_card.summary(bad)["consistent"], 2 / 3)
