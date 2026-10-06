"""Churn label builder for a given cutoff, the churn definition of docs/churn_definition_v0.md.

Call signature:

    df_labels = build_labels(cutoff_date="2026-07-25", brand_id=64, party_ids=some_list)

`party_ids` is normally the partyId column from a build_feature_store() call
for the same cutoff_date and brand_id, the population the label is for.

Only data strictly after cutoff_date, through DATA_AVAILABLE_THROUGH, is ever
read here, so this never overlaps with what build_feature_store() reads for
the same cutoff. One scan gives every target:

event_60d (classification):
  1 (churned)  - zero bet rows in (cutoff_date, cutoff_date + 60d]
  0 (returned) - at least one bet row in that window

duration_days + event_observed (survival, Cox PH). Day 0 is cutoff_date:
  the churn day is the first day t (the cutoff or an active day after it) followed by
  60 days with no bet. event_observed=1 when that silence fits inside the data, and
  duration_days=t. Otherwise the player is censored (event_observed=0) at the last
  active day. duration_days=0 with event_observed=1 is the same as event_60d=1.
  A churn day t can only be seen when t + 60 <= DATA_AVAILABLE_THROUGH, so recent
  cutoffs only show churn days close to the cutoff.

Whether event_60d is confirmable yet is build_feature_store()'s label_available_60d
column, not this module's job. The survival columns are valid either way, censoring
already handles the end of the data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import (
    DATA_AVAILABLE_THROUGH, LABEL_HORIZON_DAYS, LANDING_TABLES,
    _daily_paths, _date_range, _landing_paths, _resolve_operators_for_brand, _s3_duckdb,
)

LABEL_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data/02_intermediate/label_cache"
# Bump when the label logic changes, so old cached labels are never reused.
LABEL_VERSION = 3
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data/processed"
LABEL_COLUMNS = ["event_60d", "duration_days", "event_observed"]


def _label_cache_path(brand_id: int, cutoff: dt.date, data_end: dt.date) -> Path:
    # The data end is in the name: more data changes the survival targets.
    return LABEL_CACHE_DIR / f"{brand_id}_{cutoff}_through{data_end}_v{LABEL_VERSION}.parquet"


def _active_days_after(con, operator: str, brand_id: int, cutoff: dt.date, data_end: dt.date) -> pd.DataFrame:
    """(partyId, activity_date) for every day with at least one bet row in (cutoff, data_end]
    (any row, same "active" definition build_features.py uses, not just GAME_BET/GAME_WIN)."""
    # The cutoff folder holds local cutoff+1 00:00 to 02:00, so it is read too.
    days = _date_range(cutoff, data_end)
    daily = _daily_paths("bet", operator, days)
    if daily is not None:  # the daily tables (src/features/daily_tables.py): same rows, no S3
        df = con.sql(f"""
            SELECT DISTINCT partyId, activity_date FROM read_parquet({daily})
            WHERE brandId = {brand_id} AND n_rows > 0
        """).df()
        df["activity_date"] = pd.to_datetime(df["activity_date"]).dt.date
        return df[(df["activity_date"] > cutoff) & (df["activity_date"] <= data_end)]
    paths = _landing_paths(operator, LANDING_TABLES["bet"], days)
    df = con.sql(f"""
        SELECT DISTINCT partyId, CAST(dateTime AS DATE) AS activity_date
        FROM read_parquet({paths}, union_by_name=True)
        WHERE brandId = {brand_id} AND partyId IS NOT NULL AND dateTime IS NOT NULL
    """).df()
    df["activity_date"] = pd.to_datetime(df["activity_date"]).dt.date
    # Same timezone-spillover clamp build_features.py uses.
    return df[(df["activity_date"] > cutoff) & (df["activity_date"] <= data_end)]


def _churn_targets(active_days: pd.DataFrame, party_ids, horizon_days: int) -> pd.DataFrame:
    """event_60d, duration_days and event_observed from active day offsets (1..horizon_days)."""
    gap = LABEL_HORIZON_DAYS
    players = pd.DataFrame({"partyId": pd.unique(np.asarray(list(party_ids)))})
    days = pd.concat(
        [players.assign(day=0), active_days[active_days["partyId"].isin(players["partyId"])]],
        ignore_index=True,
    ).drop_duplicates().sort_values(["partyId", "day"])
    next_day = days.groupby("partyId")["day"].shift(-1)
    # Day t starts a churn when no bet follows in (t, t + 60], and that whole window is in the data.
    churn_starts = (next_day - days["day"] > gap) | (next_day.isna() & (days["day"] + gap <= horizon_days))
    churn_day = days[churn_starts].groupby("partyId")["day"].min()
    last_active_day = days.groupby("partyId")["day"].max()
    first_return_day = active_days.groupby("partyId")["day"].min()

    players["event_60d"] = (~(players["partyId"].map(first_return_day) <= gap)).astype(int)
    players["event_observed"] = players["partyId"].isin(churn_day.index).astype(int)
    players["duration_days"] = (
        players["partyId"].map(churn_day).fillna(players["partyId"].map(last_active_day)).astype(int)
    )
    return players


def build_labels(
    cutoff_date: str | dt.date, brand_id: int, party_ids, use_cache: bool = True,
    data_end: str | dt.date | None = None,
) -> pd.DataFrame:
    """One row per partyId in `party_ids`, with every churn target for `cutoff_date`.
    `data_end` is the last day the labels may look at (default DATA_AVAILABLE_THROUGH); a run
    "as of" an earlier date passes that date, so it never sees later data."""
    cutoff = pd.to_datetime(cutoff_date).date() if isinstance(cutoff_date, str) else cutoff_date
    data_end = pd.to_datetime(data_end).date() if data_end else DATA_AVAILABLE_THROUGH
    if data_end > DATA_AVAILABLE_THROUGH:
        raise ValueError(f"data_end {data_end} is after DATA_AVAILABLE_THROUGH {DATA_AVAILABLE_THROUGH}")

    cache_path = _label_cache_path(brand_id, cutoff, data_end)
    if use_cache and cache_path.exists():
        active = pd.read_parquet(cache_path)
    else:
        operators = _resolve_operators_for_brand(brand_id, cutoff)
        con = _s3_duckdb()
        try:
            active = pd.concat(
                [_active_days_after(con, op, brand_id, cutoff, data_end) for op in operators], ignore_index=True
            )
        finally:
            con.close()
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            active.to_parquet(cache_path, index=False)

    active = active.assign(day=(pd.to_datetime(active["activity_date"]) - pd.Timestamp(cutoff)).dt.days)
    labels = _churn_targets(active[["partyId", "day"]], party_ids, (data_end - cutoff).days)
    labels.insert(0, "cutoff_date", cutoff)
    labels.insert(1, "brandId", brand_id)
    return labels


def build_training_labels(features_path: str | Path | None = None, data_end: str | dt.date | None = None) -> Path:
    """Every churn target for every (cutoff_date, partyId) in a train_features_base file,
    saved next to it as train_labels_<same suffix>.parquet."""
    if features_path is None:
        features_path = max(PROCESSED_DIR.glob("train_features_base_*_*_*.parquet"), key=lambda p: p.stat().st_mtime)
    features_path = Path(features_path)
    features = pd.read_parquet(features_path, columns=["cutoff_date", "brandId", "partyId", "label_available_60d"])

    labels = pd.concat(
        [build_labels(cutoff, int(brand_id), grp["partyId"], data_end=data_end)
         for (cutoff, brand_id), grp in features.groupby(["cutoff_date", "brandId"])],
        ignore_index=True,
    )
    labels = labels.merge(
        features[["cutoff_date", "brandId", "partyId", "label_available_60d"]],
        on=["cutoff_date", "brandId", "partyId"], how="left",
    )
    # A label whose 60-day window runs past the data we have (or past data_end) is not a real 0/1.
    # The survival columns stay: censoring already covers the end of the data.
    end = pd.to_datetime(data_end).date() if data_end else DATA_AVAILABLE_THROUGH
    window_fits = labels["cutoff_date"].map(lambda c: c + dt.timedelta(days=LABEL_HORIZON_DAYS) <= end)
    labels["label_available_60d"] = labels["label_available_60d"] & window_fits
    labels["event_60d"] = labels["event_60d"].astype("Int64").where(labels["label_available_60d"])

    output_path = PROCESSED_DIR / features_path.name.replace("train_features_base", "train_labels")
    labels.to_parquet(output_path, index=False)
    print(f"read {features_path.relative_to(PROCESSED_DIR.parents[1])}")
    print(f"saved {output_path.relative_to(PROCESSED_DIR.parents[1])} ({len(labels):,} rows)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn targets for every row of a train_features_base file.")
    parser.add_argument("--features-path", help="default: the latest train_features_base file")
    build_training_labels(parser.parse_args().features_path)
