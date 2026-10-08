"""Churn labels (churn_labels._targets) on hand-made activity: day 0 is the cutoff.

    .venv/bin/python -m pytest src/features/tests -q      (make test)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import churn_labels  # noqa: E402

KEY = ["tenant_id", "player_id"]


def targets(active_days: dict[int, list[int]], horizon: int = 200) -> pd.DataFrame:
    players = pd.DataFrame({"tenant_id": "t", "player_id": list(active_days)})
    active = pd.DataFrame([{"tenant_id": "t", "player_id": p, "day": d} for p, days in active_days.items() for d in days],
                          columns=KEY + ["day"])
    return churn_labels._targets(active, players, horizon).set_index("player_id")


def test_labels_by_player():
    t = targets({
        1: [],              # no bet after the cutoff: churn starts now
        2: [5],             # last bet on day 5: churn starts on day 5
        3: [20, 70],        # 50-day gap, then plays on to day 70: churn starts on day 70
        4: [10, 100],       # 90-day gap from day 10: churn starts on day 10
    })
    assert t.loc[1, ["event_60d", "duration_days", "churn_within_7d", "churn_within_30d"]].tolist() == [1, 0, 1, 1]
    assert t.loc[2, ["event_60d", "duration_days", "churn_within_7d", "churn_within_14d"]].tolist() == [0, 5, 1, 1]
    assert t.loc[3, ["duration_days", "churn_within_30d"]].tolist() == [70, 0]
    assert t.loc[4, ["duration_days", "churn_within_7d", "churn_within_14d", "churn_within_30d"]].tolist() == [10, 0, 1, 1]
    assert (t["event_observed"] == 1).all()


def test_within_labels_never_decrease_with_the_horizon():
    t = targets({p: [p * 3] for p in range(40)})
    w = t[["event_60d", "churn_within_7d", "churn_within_14d", "churn_within_30d"]]
    assert (w.diff(axis=1).iloc[:, 1:] >= 0).all().all()
