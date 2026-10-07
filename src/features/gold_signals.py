"""Snapshots of gld_player_signals_daily for one brand, cached locally per day.

    frame = snapshots([dt.date(2026, 5, 1)], brand_id=64)     # builds the missing days first

gld_player_signals_daily has one row per player ever seen and as-of day (UTC), with the windows
already computed (l7d, l30d, l90d, prior periods, tenure, recency, platform scores). Its brand_id is
empty before July 2026, so the brand's players come from the activity cache (gold_cache.py).

A day's cache keeps the brand's players with a bet in (day - 30, day + 30]: the population of a
cutoff at that day (bet in the 30 days up to it) and of cutoffs 7 and 30 days later, which read
this day as their "7 / 30 days before" snapshot for trends. Keeping a row is not using its future:
every value in it is as of that day.
Cache: data/02_intermediate/gold_cache/signals/brand{id}/{day}.parquet.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gold_cache  # noqa: E402
from gold import daily_files, gold_duckdb, gold_filesystem, uri  # noqa: E402

TABLE = "gld_player_signals_daily"
WINDOW_DAYS = 30
# The signal columns kept (gold names). Money is cast to DOUBLE (EUR); the rest keeps its type.
MONEY = [
    "wagered_eur_l7d", "wagered_eur_l30d", "wagered_eur_l90d", "ggr_eur_l7d", "ggr_eur_l30d", "ggr_eur_l90d",
    "ngr_eur_l30d", "won_eur_l7d", "won_eur_l30d", "deposits_eur_l7d", "deposits_eur_l30d", "deposits_eur_l90d",
    "withdrawals_eur_l7d", "withdrawals_eur_l30d", "net_cash_eur_l30d", "bonus_ggr_eur_l30d",
    "wagered_bonus_eur_l30d", "wagered_real_eur_l30d", "bonus_granted_eur_l30d", "bonus_wagered_eur_l30d",
    "prior_ggr_eur_l7d", "prior_ggr_eur_l30d", "prior_wagered_eur_l7d", "prior_wagered_eur_l30d",
    "prior_deposits_eur_l7d", "prior_deposits_eur_l30d", "value_score_eur", "avg_bet_eur_l7d", "avg_bet_eur_l30d",
]
OTHER = [
    "tenure_days", "days_since_bet", "current_active_streak_days", "gap_days", "active_days_l7d", "active_days_l30d",
    "active_days_l90d", "prior_active_days_l7d", "prior_active_days_l30d", "sessions_l7d", "sessions_l30d",
    "bets_l7d", "bets_l30d", "prior_bets_l7d", "prior_bets_l30d", "play_secs_l7d", "play_secs_l30d",
    "games_breadth_l30d", "sessions_with_bonus_l30d", "engagement_score", "rg_loss_chasing_30d",
    "rg_deposit_velocity", "net_loss_velocity", "deposit_frequency_score", "deposit_recency_score",
    "risk_score", "churn_score", "churn_band", "lifecycle_stage", "tp_night_index", "tp_weekend_index",
    "tp_avg_session_duration_secs",
]


def cache_dir(brand_id: int) -> Path:
    return gold_cache.CACHE_ROOT / "signals" / f"brand{int(brand_id)}"


def _sql(day: dt.date, source: str, activity_files: list[str], brand_id: int) -> str:
    cols = ", ".join(["s.tenant_id", "s.player_id", "s.snapshot_date"] + [f"s.{c}::DOUBLE AS {c}" for c in MONEY]
                     + [f"s.{c}" for c in OTHER])
    lo, hi = day - dt.timedelta(days=WINDOW_DAYS), day + dt.timedelta(days=WINDOW_DAYS)
    return f"""
        SELECT {cols} FROM {source} s
        SEMI JOIN (SELECT DISTINCT tenant_id, player_id FROM read_parquet({activity_files})
                   WHERE brand_id = {int(brand_id)} AND day > DATE '{lo}' AND day <= DATE '{hi}') a
        USING (tenant_id, player_id)"""


def build(days: list[dt.date], brand_id: int, workers: int = 4) -> list[dt.date]:
    """Cache the snapshots of `days` that are missing. A day whose +30-day activity window is not
    complete yet in the activity cache is rebuilt on every call (new players may still join it)."""
    out_dir = cache_dir(brand_id)
    activity_days = set(gold_cache.cached_days("activity"))
    last_activity = max(activity_days) if activity_days else None
    todo = sorted(d for d in set(days) if not (out_dir / f"{d}.parquet").exists()
                  or last_activity is None or d + dt.timedelta(days=WINDOW_DAYS) > last_activity)
    if not todo:
        return []
    pfs, base = gold_filesystem()
    files = daily_files(pfs, base, TABLE)
    missing = [d for d in todo if d not in files]
    if missing:
        raise FileNotFoundError(f"no {TABLE} file for {missing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    con = gold_duckdb()
    act_dir = gold_cache.cache_dir("activity")

    def one(day):
        window = [str(act_dir / f"{d}.parquet") for d in sorted(activity_days)
                  if day - dt.timedelta(days=WINDOW_DAYS) < d <= day + dt.timedelta(days=WINDOW_DAYS)]
        tmp = out_dir / f"{day}.parquet.tmp"
        cursor = con.cursor()
        try:  # written by DuckDB: typed columns even when no row matches
            cursor.sql(_sql(day, f"read_parquet('{uri(pfs, files[day])}')", window, brand_id)).write_parquet(str(tmp))
        finally:
            cursor.close()
        os.replace(tmp, out_dir / f"{day}.parquet")
        return pq.ParquetFile(out_dir / f"{day}.parquet").metadata.num_rows

    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(one, todo))
    con.close()
    print(f"gold signals brand {brand_id}: {len(todo)} snapshot(s) ({sum(rows):,} rows) in {time.time() - t0:.0f}s")
    return todo


def snapshots(days: list[dt.date], brand_id: int, build_missing: bool = True) -> pd.DataFrame:
    """The cached signal rows of `days` (snapshot_date = the day)."""
    if build_missing:
        build(days, brand_id)
    files = [str(cache_dir(brand_id) / f"{d}.parquet") for d in sorted(set(days))]
    frame = duckdb.sql(f"SELECT * FROM read_parquet({files}, union_by_name=true)").df()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"]).dt.date
    return frame
