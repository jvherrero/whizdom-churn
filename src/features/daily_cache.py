"""Local daily caches of the source tables, one Parquet file per day.

    .venv/bin/python src/features/daily_cache.py
    .venv/bin/python src/features/daily_cache.py --cache financial payments --brand-id 64
    rows = read("activity", brand_id=64, start=..., end=...)

Caches in data/02_intermediate/daily_cache/{cache}/[brand{id}/]{day}.parquet:

    activity   all brands, one row per player and day with at least one bet (gld_player_gaming_daily);
               `active` marks the active days (read them with active_only=True)
    financial  one brand, money per player and day (gld_player_financial_daily)
    payments   one brand, deposits and withdrawals per player and day (gld_player_payments_daily)

Only missing days are read, plus the last RESTATEMENT_DAYS days, because the source changes them, and
the days cached before a required column was added.

Active day (measure M053 of gld-schemas): the player staked a non-zero amount from real money or
released bonus funds. A day with stakes only from playable bonus funds is not active: playable funds
become released only when the bonus conditions are met. The churn label, the population, recency,
tenure and the active_days_* features all use this definition.
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
from datalake import (  # noqa: E402
    RESTATEMENT_DAYS,
    daily_files,
    last_closed_day,
    s3_duckdb,
    s3_filesystem,
    uri,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = Path(os.environ.get("DAILY_CACHE_DIR", PROJECT_ROOT / "data/02_intermediate/daily_cache"))

CACHES = {
    "activity": {
        "table": "gld_player_gaming_daily", "brand_scoped": False, "required": ["active"],
        "sql": """
            SELECT tenant_id, brand_id, player_id, DATE '{day}' AS day,
                   SUM(bets)::BIGINT AS bets, SUM(rounds)::BIGINT AS rounds,
                   SUM(turnover_eur)::DOUBLE AS turnover_eur, SUM(ggr_eur)::DOUBLE AS ggr_eur,
                   SUM(COALESCE(wagered_real, 0) + COALESCE(wagered_released_bonus, 0)) > 0 AS active
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


def _complete(path: Path, spec: dict) -> bool:
    """The day is cached and has every column the cache needs now."""
    return path.exists() and set(spec.get("required", [])) <= set(pq.read_schema(path).names)


def build(cache: str, brand_id: int | None = None, start: dt.date | None = None, end: dt.date | None = None,
          workers: int = 8) -> list[dt.date]:
    """Cache the missing days in [start, end] and refresh the last RESTATEMENT_DAYS days."""
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
    todo = sorted(d for d in files if d in restated or not _complete(out_dir / f"{d}.parquet", spec))
    if not todo:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    con = s3_duckdb()
    if cache == "financial":  # financial_daily has no brand_id
        con.sql(f"CREATE TABLE brand_players AS SELECT DISTINCT tenant_id, player_id "
                f"FROM read_parquet('{cache_dir('activity')}/*.parquet') WHERE brand_id = {int(brand_id)}")

    def one(day):
        tmp = out_dir / f"{day}.parquet.tmp"
        cursor = con.cursor()  # one cursor per thread
        try:
            # DuckDB keeps the column types even when a day has no rows.
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
    """Update every cache needed by the features and labels up to `end`."""
    build("activity", end=end)
    brands = ([int(b) for b in read("activity", None, start=end, end=end, columns="DISTINCT brand_id")["brand_id"]]
              if brand_id == "basel" else [int(brand_id)])
    for b in brands:
        build("financial", b, end=end)
        build("payments", b, end=end)


def read(cache: str, brand_id: int | str | None = None, start: dt.date | None = None, end: dt.date | None = None,
         columns: str = "*", active_only: bool = False) -> pd.DataFrame:
    """Read a cache. For activity, brand_id None or "basel" means all brands, and active_only keeps
    the active days only (module docstring)."""
    scoped = CACHES[cache]["brand_scoped"]
    folder = cache_dir(cache, brand_id if scoped else None)
    days = [d for d in cached_days(cache, brand_id if scoped else None)
            if (not start or d >= start) and (not end or d <= end)]
    if not days:
        raise FileNotFoundError(f"no {cache} cache in {folder} for these days: run `make cache` first")
    files = [str(folder / f"{d}.parquet") for d in days]
    conditions = [f"brand_id = {int(brand_id)}"] if not scoped and brand_id not in (None, "basel") else []
    if active_only:
        stale = [d for d in days if not _complete(folder / f"{d}.parquet", CACHES[cache])]
        if stale:
            raise FileNotFoundError(f"{len(stale)} {cache} day(s) cached before the `active` column "
                                    f"({stale[0]} to {stale[-1]}): run `make cache` first")
        conditions.append("active")
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
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
