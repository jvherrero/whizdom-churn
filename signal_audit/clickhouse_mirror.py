"""A local copy of a ClickHouse signals table, so the audit reads it as gld_player_signals_daily.

One Parquet file per day, laid out like the lake (runs/<run>/mirror/gld_player_signals_daily/YYYY/MM/DD/),
with the days the run needs only; settings.py points LOCAL_TABLES at it. Read only: each day is read with
FINAL (ReplacingMergeTree: the latest version of each key), block by block, so memory holds one block.
The rows stay in the local copy (outside git; delete runs/<run>/mirror/ when the run is done); the audit
only takes aggregates out of it. Host, user and password come from .env and are never printed.

    clickhouse_mirror.copy(clickhouse_mirror.days_needed())
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pyarrow import fs

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(PROJECT_ROOT / "eda")]
import settings  # noqa: E402


def client():
    """Read-only client from .env (CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD)."""
    import clickhouse_connect
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    return clickhouse_connect.get_client(host=os.environ["CLICKHOUSE_HOST"], port=int(os.getenv("CLICKHOUSE_PORT", "28123")),
                                         username=os.environ["CLICKHOUSE_USER"],
                                         password=os.environ.get("CLICKHOUSE_PASSWORD", ""), secure=False)


def days_needed() -> list[dt.date]:
    """The period, the long period, each snapshot with the days its window and lag checks read before it, the
    evidence days and the monthly snapshots."""
    import formula_check as fc
    ks = fc.checks(fc.load())
    depth = int(max(ks["days"].max() if "days" in ks else 0, ks["lag_days"].max() if "lag_days" in ks else 0))
    days = set(pd.date_range(settings.PERIOD["start"], settings.PERIOD["end"]).date)
    days |= set(pd.date_range(settings.LONG["start"], settings.LONG["end"]).date)
    for s in settings.SNAPSHOTS:
        days |= {s - dt.timedelta(days=k) for k in range(depth + 1)}
    return sorted(days | set(settings.EVIDENCE_DAYS) | set(settings.MONTH_STARTS))


def path(day: dt.date) -> Path:
    return settings.MIRROR / settings.TABLE / f"{day:%Y/%m/%d}" / "part-0.parquet"


def copy(days: list[dt.date], cl=None) -> list[dt.date]:
    """Copy the days that are not copied yet; a day with no rows in ClickHouse gets no file. Returns the
    days copied."""
    todo = [d for d in sorted(set(days)) if not path(d).exists()]
    if not todo:
        return []
    cl = cl or client()
    table, date = settings.RUN["table"], settings.RUN.get("date_column", "snapshot_date")
    done = []
    for day in todo:
        out = path(day)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp, writer, rows = out.with_suffix(".tmp"), None, 0
        query = f"SELECT * FROM {table} FINAL WHERE {date} = {{day:Date}}"
        with cl.query_arrow_stream(query, parameters={"day": day}, use_strings=True) as stream:
            for block in stream:
                block = pa.Table.from_batches([block]) if isinstance(block, pa.RecordBatch) else block
                if date != "snapshot_date":
                    block = block.rename_columns(["snapshot_date" if c == date else c for c in block.column_names])
                writer = writer or pq.ParquetWriter(tmp, block.schema)
                writer.write_table(block)
                rows += block.num_rows
        if writer is None:
            print(f"  {day}: no rows in {table}")
            continue
        writer.close()
        os.replace(tmp, out)
        done.append(day)
        print(f"  {day}: {rows:,} rows")
    return done


def schema_differences() -> dict[str, list[str]]:
    """Columns of the contract of the run (the lake table by default) missing from the copy, and columns of
    the copy not in it (from the schema of the last copied day)."""
    files = sorted(settings.MIRROR.glob(f"{settings.TABLE}/*/*/*/part-0.parquet"))
    if not files:
        return {"missing": [], "extra": []}
    date = settings.RUN.get("date_column", "snapshot_date")   # renamed to snapshot_date in the copy
    have = {date if n == "snapshot_date" else n for n in pq.read_schema(files[-1]).names}
    want = {c["name"] for c in yaml.safe_load(settings.CONTRACT.read_text(encoding="utf-8"))["columns"]}
    return {"missing": sorted(want - have), "extra": sorted(have - want)}


def write_footers() -> Path:
    """The Parquet metadata of every copied day (rows, nulls, min and max per column), like the footers of
    eda/profile_tables.py for the lake, so the history figures read the copy."""
    from profile_tables import footer
    files = sorted(settings.MIRROR.glob(f"{settings.TABLE}/*/*/*/part-0.parquet"))
    parts = [footer(fs.LocalFileSystem(), str(f), dt.date(*map(int, f.parts[-4:-1]))) for f in files]
    out = settings.FOOTERS / f"{settings.TABLE}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
    return out
