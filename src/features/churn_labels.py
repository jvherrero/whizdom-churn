"""Churn targets from the daily activity cache (docs/churn_definition_v0.md, section 4).

    labels = churn_targets(population, cutoff=dt.date(2026, 5, 1), brand_id=64, data_end=dt.date(2026, 10, 6))

`population`: (tenant_id, player_id) of the players of a cutoff. Only activity strictly after the
cutoff and up to `data_end` is read. Targets, day 0 = the cutoff:

    event_60d       1 = no bet in (cutoff, cutoff + 60]: the churn starts now; empty when cutoff + 60 > data_end
    duration_days   first day t (the cutoff or a later active day) followed by 60 days without a bet;
                    censored players: their last active day
    event_observed  1 = that 60-day silence fits in the data, 0 = censored
    churn_within_{h}d  (h = 7, 14, 30) 1 = the churn starts within h days of the cutoff (duration_days <= h),
                    0 = it does not; empty when cutoff + h + 60 > data_end (a start at day h is not known yet).
                    event_60d is churn_within_0d. They are the plan's 7 / 14 / 30-day labels for the 60-day
                    definition, and what p_churn_7d/14d/30d of player_scores predict.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import daily_cache  # noqa: E402

HORIZON_DAYS = 60
WITHIN_DAYS = (7, 14, 30)
KEY = ["tenant_id", "player_id"]
# Every label column: lead features, never a model input.
LABEL_COLUMNS = ["event_60d", "duration_days", "event_observed", *[f"churn_within_{h}d" for h in WITHIN_DAYS]]


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
    for h in WITHIN_DAYS:
        out[f"churn_within_{h}d"] = (out["churn_day"] <= h).astype(int)  # no churn start found: 0
    return out[KEY + LABEL_COLUMNS]


def churn_targets(population: pd.DataFrame, cutoff: dt.date, brand_id: int, data_end: dt.date) -> pd.DataFrame:
    players = population[KEY].drop_duplicates()
    active = daily_cache.read("activity", brand_id, start=cutoff + dt.timedelta(days=1), end=data_end,
                             columns="tenant_id, player_id, day")
    active["day"] = (pd.to_datetime(active["day"]) - pd.Timestamp(cutoff)).dt.days
    out = _targets(active[KEY + ["day"]], players, (data_end - cutoff).days)
    # A label whose window is not all in the data yet is unknown, not 0.
    for column, days in [("event_60d", 0), *[(f"churn_within_{h}d", h) for h in WITHIN_DAYS]]:
        out[column] = out[column].astype("Int64")
        if cutoff + dt.timedelta(days=days + HORIZON_DAYS) > data_end:
            out[column] = pd.array([pd.NA] * len(out), dtype="Int64")
    return out.assign(cutoff_date=cutoff)


def population(cutoff: dt.date, brand_id: int | str, lookback_days: int = 30) -> pd.DataFrame:
    """Players of the brand ("basel" = every brand) with a bet in (cutoff - lookback_days, cutoff],
    with their brand_id."""
    rows = daily_cache.read("activity", brand_id, start=cutoff - dt.timedelta(days=lookback_days - 1), end=cutoff,
                           columns="DISTINCT tenant_id, brand_id, player_id")
    return rows.assign(cutoff_date=cutoff)
