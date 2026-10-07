"""Churn targets from the gold activity cache (docs/churn_definition_v0.md, section 4).

    labels = churn_targets(population, cutoff=dt.date(2026, 5, 1), brand_id=64, data_end=dt.date(2026, 10, 6))

`population`: (tenant_id, player_id) of the players of a cutoff. Only activity strictly after the
cutoff and up to `data_end` is read. Targets, day 0 = the cutoff:

    event_60d       1 = no bet in (cutoff, cutoff + 60]; empty when cutoff + 60 > data_end (not observable)
    duration_days   first day t (the cutoff or a later active day) followed by 60 days without a bet;
                    censored players: their last active day
    event_observed  1 = that 60-day silence fits in the data, 0 = censored
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gold_cache  # noqa: E402

HORIZON_DAYS = 60
KEY = ["tenant_id", "player_id"]


def _targets(active: pd.DataFrame, players: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """event_60d, duration_days, event_observed from the active-day offsets (1..horizon_days) after
    the cutoff. `active`: KEY + day (offset); `players`: KEY."""
    gap = HORIZON_DAYS
    days = (pd.concat([players.assign(day=0), active.merge(players, on=KEY)], ignore_index=True)
            .drop_duplicates().sort_values(KEY + ["day"]))
    next_day = days.groupby(KEY)["day"].shift(-1)
    # Day t starts a churn when no bet follows in (t, t + 60] and that whole window is in the data.
    starts = (next_day - days["day"] > gap) | (next_day.isna() & (days["day"] + gap <= horizon_days))
    churn_day = days[starts].groupby(KEY)["day"].min().rename("churn_day")
    last_day = days.groupby(KEY)["day"].max().rename("last_day")
    first_return = active.groupby(KEY)["day"].min().rename("first_return")
    out = players.merge(churn_day, on=KEY, how="left").merge(last_day, on=KEY, how="left") \
                 .merge(first_return, on=KEY, how="left")
    out["event_60d"] = (~(out["first_return"] <= gap)).astype(int)
    out["event_observed"] = out["churn_day"].notna().astype(int)
    out["duration_days"] = out["churn_day"].fillna(out["last_day"]).astype(int)
    return out[KEY + ["event_60d", "duration_days", "event_observed"]]


def churn_targets(population: pd.DataFrame, cutoff: dt.date, brand_id: int, data_end: dt.date) -> pd.DataFrame:
    players = population[KEY].drop_duplicates()
    active = gold_cache.read("activity", brand_id, start=cutoff + dt.timedelta(days=1), end=data_end,
                             columns="tenant_id, player_id, day")
    active["day"] = (pd.to_datetime(active["day"]) - pd.Timestamp(cutoff)).dt.days
    out = _targets(active[KEY + ["day"]], players, (data_end - cutoff).days)
    if cutoff + dt.timedelta(days=HORIZON_DAYS) > data_end:
        out["event_60d"] = pd.array([pd.NA] * len(out), dtype="Int64")  # the 60 days are not all in the data yet
    return out.assign(cutoff_date=cutoff)


def population(cutoff: dt.date, brand_id: int | str, lookback_days: int = 30) -> pd.DataFrame:
    """Players of the brand ("basel" = every brand) with a bet in (cutoff - lookback_days, cutoff],
    with their brand_id."""
    rows = gold_cache.read("activity", brand_id, start=cutoff - dt.timedelta(days=lookback_days - 1), end=cutoff,
                           columns="DISTINCT tenant_id, brand_id, player_id")
    return rows.assign(cutoff_date=cutoff)
