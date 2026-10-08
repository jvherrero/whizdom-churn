"""Local daily caches of the source tables: small per-day extracts that every later step reads.

    .venv/bin/python src/features/daily_cache.py                         # activity, every missing day
    .venv/bin/python src/features/daily_cache.py --cache financial payments --brand-id 64
    rows = read("activity", brand_id=64, start=..., end=...)

Caches (data/02_intermediate/daily_cache/{cache}/[brand{id}/]{day}.parquet):

    activity   every brand: one row per (tenant_id, brand_id, player_id, day) with at least one bet,
               from gld_player_gaming_daily (bets, rounds, turnover and GGR in EUR). The churn event.
    financial  one brand: one row per player and day with money moving (wagered, won, GGR, deposits,
               withdrawals, bonus play in EUR), from gld_player_financial_daily. That table has no
               brand_id, so the brand's players come from the activity cache.
    payments   one brand: one row per player and processed day with payments (completed and failed
               deposits, withdrawals), from gld_player_payments_daily (only from 2025-09-18, 40 days missing).

Only missing days are read, in parallel, plus the last RESTATEMENT_DAYS days, which the source reprocesses
every day. Brand-scoped caches need the activity cache first (it is built automatically).
"""

from __future__ import annotations

import argparse
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
from datalake import RESTATEMENT_DAYS, daily_files, s3_duckdb, s3_filesystem, last_closed_day, uri  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = Path(os.environ.get("DAILY_CACHE_DIR", PROJECT_ROOT / "data/02_intermediate/daily_cache"))

CACHES = {
    "activity": {
        "table": "gld_player_gaming_daily", "brand_scoped": False,
        "sql": """
            SELECT tenant_id, brand_id, player_id, DATE '{day}' AS day,
                   SUM(bets)::BIGINT AS bets, SUM(rounds)::BIGINT AS rounds,
                   SUM(turnover_eur)::DOUBLE AS turnover_eur, SUM(ggr_eur)::DOUBLE AS ggr_eur
            FROM {source} GROUP BY ALL HAVING SUM(bets) > 0""",
    },
    "financial": {
        "table": "gld_player_financial_daily", "brand_scoped": True,
        "sql": """
            SELECT f.tenant_id, f.player_id, DATE '{day}' AS day,
                   wagered_eur_today::DOUBLE AS wagered_eur, won_eur_today::DOUBLE AS won_eur,
                   ggr_eur_today::DOUBLE AS ggr_eur, deposits_eur_today::DOUBLE AS deposits_eur,
                   withdrawals_eur_today::DOUBLE AS withdrawals_eur, bonus_ggr_eur_today::DOUBLE AS bonus_ggr_eur,
                   wagered_bonus_eur_today::DOUBLE AS wagered_bonus_eur
            FROM {source} f SEMI JOIN brand_players b USING (tenant_id, player_id)
            WHERE wagered_eur_today <> 0 OR deposits_eur_today <> 0 OR withdrawals_eur_today <> 0""",
    },
    "payments": {
        "table": "gld_player_payments_daily", "brand_scoped": True,
        "sql": """
            SELECT tenant_id, player_id, DATE '{day}' AS day,
                   SUM(deposit_count)::BIGINT AS deposit_count, SUM(deposit_attempts)::BIGINT AS deposit_attempts,
                   SUM(failed_deposit_count)::BIGINT AS failed_deposit_count,
                   SUM(withdraw_count)::BIGINT AS withdraw_count, SUM(failed_withdraw_count)::BIGINT AS failed_withdraw_count,
                   SUM(deposit_amount_eur)::DOUBLE AS deposit_eur, SUM(withdraw_amount_eur)::DOUBLE AS withdraw_eur
            FROM {source} WHERE brand_id = {brand} GROUP BY ALL""",
    },
}


def cache_dir(cache: str, brand_id: int | None = None) -> Path:
    if CACHES[cache]["brand_scoped"]:
        if brand_id is None:
            raise ValueError(f"the {cache} cache is per brand: pass brand_id")
        return CACHE_ROOT / cache / f"brand{int(brand_id)}"
    return CACHE_ROOT / cache


def cached_days(cache: str, brand_id: int | None = None) -> list[dt.date]:
    return sorted(dt.date.fromisoformat(p.stem) for p in cache_dir(cache, brand_id).glob("*.parquet"))


def build(cache: str, brand_id: int | None = None, start: dt.date | None = None, end: dt.date | None = None,
          workers: int = 8) -> list[dt.date]:
    """Cache every closed day in [start, end] that is missing, and re-read the last RESTATEMENT_DAYS
    days available (the source may have changed them). Returns the days written."""
    spec = CACHES[cache]
    if spec["brand_scoped"] and not cached_days("activity"):
        build("activity", end=end, workers=workers)
    end = min(end or last_closed_day(), last_closed_day())
    pfs, base = s3_filesystem()
    files = {d: p for d, p in daily_files(pfs, base, spec["table"]).items() if d <= end and (not start or d >= start)}
    if not files:
        return []
    out_dir = cache_dir(cache, brand_id)
    restated = {d for d in files if d > max(files) - dt.timedelta(days=RESTATEMENT_DAYS)}
    todo = sorted(d for d in files if d in restated or not (out_dir / f"{d}.parquet").exists())
    if not todo:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    con = s3_duckdb()
    if cache == "financial":  # the brand's players: financial_daily has no brand_id
        con.sql(f"CREATE TABLE brand_players AS SELECT DISTINCT tenant_id, player_id "
                f"FROM read_parquet('{cache_dir('activity')}/*.parquet') WHERE brand_id = {int(brand_id)}")

    def one(day):
        tmp = out_dir / f"{day}.parquet.tmp"  # never leave half a file
        cursor = con.cursor()  # one DuckDB cursor per thread (same database: sees brand_players)
        try:
            # Written by DuckDB, not pandas: a day with no rows still gets typed columns.
            cursor.sql(spec["sql"].format(day=day, brand=brand_id,
                                          source=f"read_parquet('{uri(pfs, files[day])}')")).write_parquet(str(tmp))
        finally:
            cursor.close()
        os.replace(tmp, out_dir / f"{day}.parquet")
        return pq.ParquetFile(out_dir / f"{day}.parquet").metadata.num_rows

    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(one, todo))
    con.close()
    scope = f" brand {brand_id}" if spec["brand_scoped"] else ""
    print(f"daily cache {cache}{scope}: {len(todo)} day(s) ({todo[0]} to {todo[-1]}, {sum(rows):,} rows) "
          f"in {time.time() - t0:.0f}s")
    return todo


def top_up(end: dt.date, brand_id: int | str) -> None:
    """Every cache the features and labels read, up to `end`: activity (every brand), then financial
    and payments of `brand_id` ("basel" = every brand with a bet on `end`)."""
    build("activity", end=end)
    brands = ([int(b) for b in read("activity", None, start=end, end=end, columns="DISTINCT brand_id")["brand_id"]]
              if brand_id == "basel" else [int(brand_id)])
    for b in brands:
        build("financial", b, end=end)
        build("payments", b, end=end)


def read(cache: str, brand_id: int | str | None = None, start: dt.date | None = None, end: dt.date | None = None,
         columns: str = "*") -> pd.DataFrame:
    """Rows of a cache. activity: brand_id None or "basel" = every brand."""
    scoped = CACHES[cache]["brand_scoped"]
    folder = cache_dir(cache, brand_id if scoped else None)
    days = [d for d in cached_days(cache, brand_id if scoped else None)
            if (not start or d >= start) and (not end or d <= end)]
    if not days:
        raise FileNotFoundError(f"no {cache} cache in {folder} for these days: run `make cache` first")
    files = [str(folder / f"{d}.parquet") for d in days]
    where = f" WHERE brand_id = {int(brand_id)}" if not scoped and brand_id not in (None, "basel") else ""
    return duckdb.sql(f"SELECT {columns} FROM read_parquet({files}, union_by_name=true){where}").df()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the local daily caches of the source tables.")
    parser.add_argument("--cache", nargs="+", default=["activity"], choices=list(CACHES))
    parser.add_argument("--brand-id", type=int, default=64, help="for the brand-scoped caches")
    parser.add_argument("--start", type=dt.date.fromisoformat)
    parser.add_argument("--end", type=dt.date.fromisoformat, help="default: yesterday (UTC)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for name in args.cache:
        build(name, args.brand_id, args.start, args.end, args.workers)
        days = cached_days(name, args.brand_id if CACHES[name]["brand_scoped"] else None)
        print(f"{name}: {len(days)} day(s) cached" + (f", {days[0]} to {days[-1]}" if days else ""))
