import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from missing_games import active_set  # noqa: E402


def games(rounds):
    return pd.DataFrame({"game_uid": [f"g{i}" for i in range(len(rounds))], "rounds_6m": rounds})


def test_cut_keeps_the_game_that_crosses_the_limit():
    # 60 + 30 = 90% < 95%, so the third game (7) is needed and crosses the limit: the last 3 are left out
    result = active_set(games([60, 30, 7, 2, 1]), total_rounds=100, coverage=0.95)
    assert list(result["game_uid"]) == ["g0", "g1", "g2"]


def test_unresolved_rounds_count_in_the_total():
    # 10 of the 100 rounds do not resolve: 90% of the total is only reached with both games
    result = active_set(games([50, 40]), total_rounds=100, coverage=0.9)
    assert len(result) == 2


def test_sorted_and_ties_by_game_uid():
    result = active_set(games([5, 5, 90]), total_rounds=100, coverage=0.5)
    assert list(result["game_uid"]) == ["g2"]
    tie = active_set(games([5, 5]), total_rounds=10, coverage=0.9)
    assert list(tie["game_uid"]) == ["g0", "g1"]


def test_split_batches():
    from generate_batches import split
    games = pd.DataFrame({"game_uid": [f"g{i}" for i in range(10)], "rounds_6m": range(10, 0, -1)})
    batches = split(games, ["a", "b", "c"])
    sizes = sorted(len(b) for b in batches.values())
    assert sizes == [3, 3, 4]
    uids = [u for b in batches.values() for u in b["game_uid"]]
    assert sorted(uids) == sorted(games["game_uid"])          # no overlap, nothing lost
    assert list(batches["a"]["game_uid"]) == ["g0", "g3", "g6", "g9"]
