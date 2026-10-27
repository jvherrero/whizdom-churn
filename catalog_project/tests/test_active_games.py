import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from active_games import active_set, add_factors  # noqa: E402


def games(ggr, rounds=None):
    return pd.DataFrame({"game_uid": [f"g{i}" for i in range(len(ggr))], "ggr_eur_6m": ggr,
                         "rounds_6m": rounds if rounds is not None else [1] * len(ggr)})


def test_cut_keeps_the_game_that_crosses_the_limit():
    # 60 + 30 = 90% < 95%, so the third game (7) is needed and crosses the limit: the last 2 are left out
    result = active_set(games([60, 30, 7, 2, 1]), total_ggr=100, coverage=0.95)
    assert list(result["game_uid"]) == ["g0", "g1", "g2"]


def test_games_with_ggr_zero_or_below_never_enter():
    result = active_set(games([90, -5, 0, 10]), total_ggr=100, coverage=1.0)
    assert list(result["game_uid"]) == ["g0", "g3"]


def test_unresolved_ggr_counts_in_the_total():
    # 10 of the 100 do not resolve: 90% of the total is only reached with both games
    result = active_set(games([50, 40]), total_ggr=100, coverage=0.9)
    assert len(result) == 2


def test_sorted_by_ggr_then_rounds_then_game_uid():
    result = active_set(games([5, 5, 90], rounds=[1, 9, 1]), total_ggr=100, coverage=0.95)
    assert list(result["game_uid"]) == ["g2", "g1"]
    tie = active_set(games([5, 5]), total_ggr=10, coverage=0.9)
    assert list(tie["game_uid"]) == ["g0", "g1"]


def test_factors():
    df = pd.DataFrame({"daily_players_30d": [300, 0], "unique_players_30d": [100, 0],
                       "daily_players_90d": [900, 90], "unique_players_90d": [100, 30],
                       "release_date": ["2026-10-01", None], "first_seen": [None, "2026-10-21"]})
    out = add_factors(df, pd.Timestamp("2026-10-27"), "2022-01-01", "2025-04-01")
    assert np.allclose(out["stickiness_30d"].iloc[0], 0.1) and np.isnan(out["stickiness_30d"].iloc[1])   # 300/30/100
    assert np.allclose(out["stickiness_90d"], [0.1, 1 / 30])
    assert out["days_since_launch"].tolist() == [26, 6]


def test_launch_unknown_for_the_placeholder_and_the_first_day_of_the_data():
    df = pd.DataFrame({"daily_players_30d": [1] * 3, "unique_players_30d": [1] * 3, "daily_players_90d": [1] * 3,
                       "unique_players_90d": [1] * 3, "release_date": ["2022-01-01", "2022-01-01", "2022-01-01"],
                       "first_seen": ["2025-04-01", "2025-04-02", None]})
    out = add_factors(df, pd.Timestamp("2026-10-27"), "2022-01-01", "2025-04-01")
    assert np.isnan(out["days_since_launch"].iloc[0]) and np.isnan(out["days_since_launch"].iloc[2])
    assert out["days_since_launch"].iloc[1] == 573

