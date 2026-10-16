"""Active days from the activity cache: the definition of an active day and its windows.

    .venv/bin/python -m pytest src/features/tests/test_active_days.py -q
"""

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_features  # noqa: E402

CUTOFF = dt.date(2026, 4, 1)


def activity(days_before: list[int], player: int = 1) -> pd.DataFrame:
    """One row per day with a bet, as daily_cache "activity" stores it."""
    return pd.DataFrame({"tenant_id": "t", "player_id": player,
                         "day": [pd.Timestamp(CUTOFF) - pd.Timedelta(d, unit="D") for d in days_before]})


def test_windows_end_on_the_cutoff_day_included():
    # Bets on the cutoff day, 6 days before (last day of the 7-day window) and 7 days before (outside it).
    out = build_features.active_days(activity([0, 6, 7, 13, 14, 29, 30, 89, 90]), CUTOFF).loc[("t", 1)]
    assert out["active_days_l7d"] == 2           # days 0 and 6
    assert out["prior_active_days_l7d"] == 2     # days 7 and 13 (days 8 to 14 before the cutoff, as in the source)
    assert out["active_days_l30d"] == 6          # days 0 to 29
    assert out["active_days_l90d"] == 8          # days 0 to 89


def test_days_after_the_cutoff_and_repeated_rows_do_not_count():
    act = pd.concat([activity([0, 0, 3]), activity([-1, -5])])   # the same day twice, and two days after the cutoff
    out = build_features.active_days(act, CUTOFF).loc[("t", 1)]
    assert out["active_days_l7d"] == 2


def test_a_player_without_bets_in_a_window_is_absent_here_and_zero_in_the_features():
    out = build_features.active_days(pd.concat([activity([0], 1), activity([40], 2)]), CUTOFF)
    assert out.loc[("t", 2), "active_days_l90d"] == 1
    assert pd.isna(out.loc[("t", 2), "active_days_l7d"])   # history_signals fills it with 0
