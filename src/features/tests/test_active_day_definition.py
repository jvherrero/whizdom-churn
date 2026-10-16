"""Active day (measure M053 of gld-schemas): a stake from real money or released bonus funds.
Stakes only from playable bonus funds do not make a day active.

    .venv/bin/python -m pytest src/features/tests/test_active_day_definition.py -q
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import daily_cache  # noqa: E402

DAY = dt.date(2025, 6, 1)
# player: (real, released bonus, playable bonus)
STAKES = {1: (10.0, 0.0, 0.0), 2: (0.0, 5.0, 0.0), 3: (0.0, 0.0, 20.0), 4: (1.0, 0.0, 30.0), 5: (None, None, 8.0)}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    folder = tmp_path / "lake" / "gld_player_gaming_daily" / f"{DAY:%Y/%m/%d}"
    folder.mkdir(parents=True)
    pd.DataFrame({"tenant_id": "t1", "brand_id": np.int32(64), "player_id": np.array(list(STAKES), dtype="int32"),
                  "bets": np.uint32(3), "rounds": np.uint32(3), "turnover_eur": 1.0, "ggr_eur": 0.5,
                  "wagered_real": [s[0] for s in STAKES.values()],
                  "wagered_released_bonus": [s[1] for s in STAKES.values()],
                  "wagered_playable_bonus": [s[2] for s in STAKES.values()]}).to_parquet(folder / "data.parquet")
    monkeypatch.setenv("DATALAKE_ROOT", str(tmp_path / "lake"))
    monkeypatch.setattr(daily_cache, "CACHE_ROOT", tmp_path / "cache")
    return tmp_path / "cache"


def test_playable_bonus_alone_is_not_an_active_day(cache):
    daily_cache.build("activity", start=DAY, end=DAY)
    every_bet = daily_cache.read("activity", 64)
    active = daily_cache.read("activity", 64, active_only=True)
    assert set(every_bet["player_id"]) == set(STAKES)            # bets and turnover keep every stake
    assert set(active["player_id"]) == {1, 2, 4}                  # real, released, or real mixed with playable


def test_days_cached_before_the_active_column_are_rebuilt(cache):
    daily_cache.build("activity", start=DAY, end=DAY)
    path = daily_cache.cache_dir("activity") / f"{DAY}.parquet"
    pd.read_parquet(path).drop(columns="active").to_parquet(path)   # a day cached with the old definition
    with pytest.raises(FileNotFoundError, match="make cache"):
        daily_cache.read("activity", 64, active_only=True)
    assert daily_cache.build("activity", start=DAY, end=DAY) == [DAY]
    assert set(daily_cache.read("activity", 64, active_only=True)["player_id"]) == {1, 2, 4}
