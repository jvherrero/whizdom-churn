"""Churn labels from the daily activity cache (docs/churn_definition_v0.md).

A "bet" below is an active day: a stake from real money or released bonus funds (daily_cache).

    labels = churn_targets(population, cutoff=dt.date(2026, 5, 1), brand_id=64, data_end=dt.date(2026, 10, 6))

Labels for each player of a cutoff (day 0 is the cutoff):

    event_60d          1 if there is no bet in (cutoff, cutoff + 60]
    duration_days      first day followed by 60 days without a bet, or the last active day if censored
    event_observed     1 if that 60-day silence is inside the data, 0 if censored
    churn_within_{h}d  1 if the churn starts within h days (h = 7, 14, 30)

A label is empty when its window goes past data_end.

The threshold is 60 days. CHURN_DAYS=N in the environment (`make pipeline CHURN_DAYS=90`) changes it for a
sensitivity run; the column names keep "60d", and every output of that run carries VARIANT (see below).
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import daily_cache  # noqa: E402

DEFAULT_HORIZON_DAYS = 60
WITHIN_DAYS = (7, 14, 30)


def _horizon_days() -> int:
    value = int(os.environ.get("CHURN_DAYS") or DEFAULT_HORIZON_DAYS)
    if value <= max(WITHIN_DAYS):
        raise ValueError(f"CHURN_DAYS must be more than {max(WITHIN_DAYS)} days, got {value}")
    return value


HORIZON_DAYS = _horizon_days()
# Outputs of another threshold carry this suffix (models, datasets, scores, reports), so a sensitivity
# run never replaces the official 60-day ones.
VARIANT = "" if HORIZON_DAYS == DEFAULT_HORIZON_DAYS else f"_churn{HORIZON_DAYS}d"


def same_threshold(runs: pd.DataFrame) -> pd.DataFrame:
    """MLflow runs trained on the current threshold (runs without the tag are 60-day runs)."""
    if runs.empty:
        return runs
    days = pd.to_numeric(runs.get("tags.churn_days", pd.Series(index=runs.index, dtype=float)), errors="coerce")
    return runs[days.fillna(DEFAULT_HORIZON_DAYS) == HORIZON_DAYS]
KEY = ["tenant_id", "player_id"]
# Label columns. They must never be used as model inputs.
LABEL_COLUMNS = ["event_60d", "duration_days", "event_observed", *[f"churn_within_{h}d" for h in WITHIN_DAYS]]


def _targets(active: pd.DataFrame, players: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Compute the labels from the active days after the cutoff (day = days since the cutoff)."""
    gap = HORIZON_DAYS
    days = (pd.concat([players.assign(day=0), active.merge(players, on=KEY)], ignore_index=True)
            .drop_duplicates().sort_values(KEY + ["day"]))
    next_day = days.groupby(KEY)["day"].shift(-1)
    # A churn starts on day t if there is no bet in (t, t + 60] and the window is inside the data.
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
        out[f"churn_within_{h}d"] = (out["churn_day"] <= h).astype(int)
    return out[KEY + LABEL_COLUMNS]


def churn_targets(population: pd.DataFrame, cutoff: dt.date, brand_id: int, data_end: dt.date) -> pd.DataFrame:
    players = population[KEY].drop_duplicates()
    active = daily_cache.read("activity", brand_id, start=cutoff + dt.timedelta(days=1), end=data_end,
                             columns="tenant_id, player_id, day", active_only=True)
    active["day"] = (pd.to_datetime(active["day"]) - pd.Timestamp(cutoff)).dt.days
    out = _targets(active[KEY + ["day"]], players, (data_end - cutoff).days)
    # If the window is not complete yet, the label is unknown, not 0.
    for column, days in [("event_60d", 0), *[(f"churn_within_{h}d", h) for h in WITHIN_DAYS]]:
        out[column] = out[column].astype("Int64")
        if cutoff + dt.timedelta(days=days + HORIZON_DAYS) > data_end:
            out[column] = pd.array([pd.NA] * len(out), dtype="Int64")
    return out.assign(cutoff_date=cutoff)


def population(cutoff: dt.date, brand_id: int | str, lookback_days: int = 30) -> pd.DataFrame:
    """Players with an active day in (cutoff - lookback_days, cutoff]. "basel" means all brands."""
    rows = daily_cache.read("activity", brand_id, start=cutoff - dt.timedelta(days=lookback_days - 1), end=cutoff,
                           columns="DISTINCT tenant_id, brand_id, player_id", active_only=True)
    return rows.assign(cutoff_date=cutoff)
