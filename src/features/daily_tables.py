"""Daily tables: the S3 landing data summarised once per landing day, every brand at once, and kept
up to date day by day (only the days not yet built are read).

    .venv/bin/python src/features/daily_tables.py                        # HISTORY_START .. the last complete day in S3
    .venv/bin/python src/features/daily_tables.py --start 2026-10-01 --end 2026-10-04 --workers 32

For each (operator, table, landing day):
  1. every S3 file of that day is read in parallel (thread pool), and only the columns the project
     uses: parquet lets you fetch column chunks alone, and they are ~7% of the bytes, so nothing is
     fully downloaded and the personal data columns (names, emails, addresses) never leave S3;
  2. the rows are aggregated in memory to one row per (brand, player, local day) and written as a
     small parquet in data/02_intermediate/daily_tables/ (no raw file ever touches the disk);
  3. the per-brand daily profile of the data alerts (rows, nulls, totals, categories) is computed in
     the same pass and cached where src/features/alerts.py looks for it.

build_features, build_labels and the alerts read these tables when every day they need is built,
and S3 otherwise, so the results do not change; only the time does.

Tables (activity_date is the local Europe/Madrid day, like CAST(... AS DATE) everywhere else):
  bet          brandId, partyId, activity_date, currency, n_rows, n_game, stake_local, win_local
  transaction  brandId, partyId, activity_date, currency, n_deposit, deposit_local (completed deposits)
  player       brandId, partyId, regdate
  bonus        partyId, activity_date, n_events, n_active, active_amount (bonus has no brandId)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.config import Config
from pyarrow import fs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import (
    AWS_PROFILE, BUCKET, DAILY_DIR, HISTORY_START, LANDING_LEVEL, LANDING_TABLES,
    OPERATORS, TIMEZONE, _date_range, daily_path, latest_s3_complete_day,
)
import alerts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGION = "eu-central-1"
DEFAULT_WORKERS = 64       # parallel S3 file reads
DEFAULT_UNIT_WORKERS = 8   # (operator, table, day) units at once; each bet day holds ~1 GB in memory

# Columns read from S3: exactly what the daily tables (and so the features and labels) need. The
# same list is what the data alerts check (configs/eda_alerts.yaml: used_columns), nothing else.
READ_COLUMNS = alerts.load_config()["used_columns"]

# One aggregation per table, over the rows of one landing day (`events`). They keep exactly what the
# feature, label and activity queries need, so reading them gives the same rows as reading S3.
DAILY_SQL = {
    "bet": """
        SELECT brandId, partyId, CAST(dateTime AS DATE) AS activity_date, currency,
               COUNT(*) AS n_rows,
               COUNT(*) FILTER (WHERE rolledBack = False AND tranType IN ('GAME_BET', 'GAME_WIN')) AS n_game,
               -SUM(CASE WHEN rolledBack = False AND tranType = 'GAME_BET' THEN amountReal ELSE 0 END) AS stake_local,
               SUM(CASE WHEN rolledBack = False AND tranType = 'GAME_WIN' THEN amountReal ELSE 0 END) AS win_local
        FROM events WHERE partyId IS NOT NULL AND dateTime IS NOT NULL
        GROUP BY ALL""",
    "transaction": """
        SELECT brandId, partyId, CAST(requestDate AS DATE) AS activity_date, {currency} AS currency,
               COUNT(*) AS n_deposit, SUM(processedAmount) AS deposit_local
        FROM events
        WHERE transactionType = 'DEPOSIT' AND status = 'COMPLETED' AND partyId IS NOT NULL AND requestDate IS NOT NULL
        GROUP BY ALL""",
    "player": """
        SELECT DISTINCT brandId, partyId, CAST(regdate AS DATE) AS regdate FROM events""",
    "bonus": """
        SELECT partyId, CAST(changeStatusTimestamp AS DATE) AS activity_date, COUNT(*) AS n_events,
               COUNT(*) FILTER (WHERE status = 'ACTIVE') AS n_active,
               SUM(amount) FILTER (WHERE status = 'ACTIVE') AS active_amount
        FROM events WHERE partyId IS NOT NULL AND changeStatusTimestamp IS NOT NULL
        GROUP BY ALL""",
}


def _clients():
    session = boto3.Session(profile_name=AWS_PROFILE)
    creds = session.get_credentials().get_frozen_credentials()
    s3 = session.client("s3", config=Config(max_pool_connections=64))
    s3fs = fs.S3FileSystem(access_key=creds.access_key, secret_key=creds.secret_key,
                           session_token=creds.token, region=REGION)
    return s3, s3fs


def _list_keys(s3, operator: str, table: str, day: dt.date) -> list[str]:
    prefix = f"{LANDING_LEVEL}/{operator}/{LANDING_TABLES[table]}/{day.year}/{day.month:02d}/{day.day:02d}/"
    return [o["Key"] for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix)
            for o in page.get("Contents", []) if o["Key"].endswith(".parquet")]


def _read_columns(s3fs, key: str, columns: list[str]) -> pa.Table:
    """Only the wanted columns of one S3 parquet file (ranged reads), whatever columns it has."""
    f = pq.ParquetFile(f"{BUCKET}/{key}", filesystem=s3fs)
    return f.read(columns=[c for c in columns if c in f.schema_arrow.names])


def _alert_profiles(con, operator: str, table: str, day: dt.date, cfg: dict) -> int:
    """The data-alert daily profile of every brand (bonus: the whole operator), written where
    alerts.daily_profile() caches it. Only the columns read here are profiled. Profiles that
    already exist (e.g. from a full-column S3 read) are kept."""
    columns = [c for c in con.sql("DESCRIBE events").df()["column_name"]]
    by_brand = TABLE_BRAND_SCOPE[table]
    totals = alerts.TABLE_SPECS[table]["totals"]
    selects = (["COUNT(*) AS n_rows"] + [f'COUNT("{c}") AS "nn__{c}"' for c in columns]
               + [f'{expr} AS "total__{name}"' for name, expr in totals.items()])
    group = "brandId" if by_brand else "NULL"
    stats = con.sql(f"SELECT {group} AS scope, {', '.join(selects)} FROM events "
                    f"{'WHERE brandId IS NOT NULL' if by_brand else ''} GROUP BY ALL").df()
    cat_cols = [c for c in cfg["new_categories"]["columns"].get(table, []) if c in columns]
    max_values = cfg["new_categories"]["max_values_per_column"]
    # Distinct values of each categorical column for every scope in one query (not one per brand).
    values = {}
    for c in cat_cols:
        lists = con.sql(f'SELECT {group} AS scope, list(DISTINCT CAST("{c}" AS VARCHAR)) AS v FROM events '
                        f'{"WHERE brandId IS NOT NULL" if by_brand else ""} GROUP BY ALL').df()
        for scope_value, vals in zip(lists["scope"], lists["v"]):
            key = int(scope_value) if by_brand else None
            values.setdefault(key, {})[c] = sorted(v for v in vals if v is not None)[:max_values]
    written = 0
    for row in stats.itertuples(index=False):
        row = row._asdict()
        scope_value = row["scope"]
        scope = f"brand{int(scope_value)}" if by_brand else "operator"
        path = alerts.PROFILE_DIR / operator / table / scope / f"{day}_v{alerts.PROFILE_VERSION}.json"
        if path.exists():
            continue
        n = int(row["n_rows"])
        profile = {
            "day": str(day), "rows": n, "missing_folder": False,
            "null_rate": {c: 1 - int(row[f"nn__{c}"]) / n for c in columns},
            "totals": {"rows": n, **{k: (None if pd.isna(row[f"total__{k}"]) else float(row[f"total__{k}"]))
                                     for k in totals}},
            "categories": values.get(int(scope_value) if by_brand else None, {}),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile))
        written += 1
    return written


TABLE_BRAND_SCOPE = {"bet": True, "transaction": True, "player": True, "bonus": False}


def build_day(s3, s3fs, pool: ThreadPoolExecutor, con, operator: str, table: str, day: dt.date,
              cfg: dict, overwrite: bool = False) -> dict:
    """One (operator, table, landing day): parallel column reads -> aggregation -> small parquet."""
    path = daily_path(table, operator, day)
    row = {"operator": operator, "table": table, "day": day, "files": 0, "source_rows": 0, "daily_rows": 0,
           "kb_on_disk": 0.0, "read_s": 0.0, "aggregate_s": 0.0, "status": "already built"}
    if path.exists() and not overwrite:
        return row
    t0 = time.perf_counter()
    keys = _list_keys(s3, operator, table, day)
    if not keys:  # no data that day: nothing written, readers fall back to S3 as before
        return {**row, "status": "no files in S3", "read_s": time.perf_counter() - t0}
    events = pa.concat_tables(list(pool.map(lambda k: _read_columns(s3fs, k, READ_COLUMNS[table]), keys)),
                              promote_options="permissive")
    t1 = time.perf_counter()
    con.register("events", events)
    try:
        currency = "currency" if "currency" in events.column_names else "NULL::VARCHAR"
        daily = con.sql(DAILY_SQL[table].format(currency=currency)).fetch_arrow_table()
        _alert_profiles(con, operator, table, day, cfg)
    finally:
        con.unregister("events")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")  # never leave half a file if the run is interrupted
    pq.write_table(daily, tmp, compression="zstd")
    os.replace(tmp, path)
    return {**row, "files": len(keys), "source_rows": events.num_rows, "daily_rows": daily.num_rows,
            "kb_on_disk": path.stat().st_size / 1e3, "read_s": t1 - t0, "aggregate_s": time.perf_counter() - t1,
            "status": "built"}


def build(start: dt.date = HISTORY_START, end: dt.date | None = None, operators=OPERATORS,
          tables=tuple(DAILY_SQL), workers: int = DEFAULT_WORKERS, unit_workers: int = DEFAULT_UNIT_WORKERS,
          overwrite: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Build every missing (operator, table, day) in [start, end], `unit_workers` units at a time
    (each reading its files with the shared pool of `workers` threads). Returns one row per unit."""
    # Never past the last complete day in S3: the newest folder may still be filling up, and a day
    # built from half its files would stay wrong in the table.
    complete = latest_s3_complete_day()
    if end is None or end > complete:
        if end is not None and verbose:
            print(f"daily tables: {end} is not complete in S3 yet, building up to {complete}")
        end = complete
    s3, s3fs = _clients()
    cfg = alerts.load_config()
    con = duckdb.connect()
    con.sql(f"SET TimeZone='{TIMEZONE}'")  # activity_date = the local day, as in every other query
    units = [(day, op, table) for day in _date_range(start, end) for op in operators for table in tables]
    rows, t0 = [], time.perf_counter()

    def run(unit):
        day, operator, table = unit
        cursor = con.cursor()  # one DuckDB cursor per thread: a connection is not shared across threads
        cursor.sql(f"SET TimeZone='{TIMEZONE}'")
        try:
            return build_day(s3, s3fs, file_pool, cursor, operator, table, day, cfg, overwrite)
        finally:
            cursor.close()

    try:
        with ThreadPoolExecutor(workers) as file_pool, ThreadPoolExecutor(unit_workers) as unit_pool:
            futures = [unit_pool.submit(run, u) for u in units]
            for i, future in enumerate(as_completed(futures), 1):
                rows.append(future.result())
                if verbose and (i % (len(operators) * len(tables)) == 0 or i == len(units)):
                    print(f"[{i:4d}/{len(units)} units] elapsed {(time.perf_counter() - t0) / 60:5.1f} min")
    finally:
        con.close()
    out = pd.DataFrame(rows).sort_values(["day", "operator", "table"]).reset_index(drop=True)
    if verbose:
        built = out[out["status"] == "built"]
        print(f"built {len(built)} of {len(out)} (operator, table, day) units in {(time.perf_counter() - t0) / 60:.1f} min: "
              f"{built['source_rows'].sum():,} source rows -> {built['daily_rows'].sum():,} daily rows, "
              f"{built['kb_on_disk'].sum() / 1e3:.1f} MB in {os.path.relpath(DAILY_DIR, PROJECT_ROOT)}/")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the daily tables from S3 (only the missing days).")
    parser.add_argument("--start", type=dt.date.fromisoformat, default=HISTORY_START)
    parser.add_argument("--end", type=dt.date.fromisoformat, help="default: the last complete day in S3")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--unit-workers", type=int, default=DEFAULT_UNIT_WORKERS)
    parser.add_argument("--overwrite", action="store_true", help="rebuild days that already exist")
    args = parser.parse_args()
    build(args.start, args.end, workers=args.workers, unit_workers=args.unit_workers, overwrite=args.overwrite)
