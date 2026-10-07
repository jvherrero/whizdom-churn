"""Access to the S3 gold layer (org/40-gold), the model's data source.

Every gold table the project reads is partitioned by day: {table}/YYYY/MM/DD/data.parquet. Dates
are UTC days. GOLD_ROOT=<local folder> reads a local copy with the same layout instead of S3 (tests).
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import duckdb
from pyarrow import fs

AWS_PROFILE = "javier-whizdom-prod-ds"
REGION = "eu-central-1"
BUCKET = "whizdomai-eu-central-1-650177547431-datalake-gmntc-prod"
GOLD_PREFIX = "org/40-gold"
# Gold reprocesses its last 7 days every day (data card): a day younger than this can still change.
RESTATEMENT_DAYS = 7


def _credentials():
    import boto3

    return boto3.Session(profile_name=AWS_PROFILE).get_credentials().get_frozen_credentials()


def gold_filesystem() -> tuple[fs.FileSystem, str]:
    """(filesystem, base path) of the gold layer."""
    root = os.environ.get("GOLD_ROOT")
    if root:
        return fs.LocalFileSystem(), str(Path(root).resolve())
    c = _credentials()
    return (fs.S3FileSystem(access_key=c.access_key, secret_key=c.secret_key, session_token=c.token, region=REGION),
            f"{BUCKET}/{GOLD_PREFIX}")


def gold_duckdb(threads: int | None = None) -> duckdb.DuckDBPyConnection:
    """DuckDB connection that can read the gold files (S3 credentials loaded, or local)."""
    con = duckdb.connect()
    if threads:
        con.sql(f"SET threads={threads}")
    if not os.environ.get("GOLD_ROOT"):
        c = _credentials()
        con.sql("INSTALL httpfs; LOAD httpfs;")
        con.sql(f"CREATE SECRET (TYPE S3, KEY_ID '{c.access_key}', SECRET '{c.secret_key}', "
                f"SESSION_TOKEN '{c.token}', REGION '{REGION}')")
    return con


def daily_files(pfs: fs.FileSystem, base: str, table: str) -> dict[dt.date, str]:
    """{day: path} of a gold table's daily files."""
    out = {}
    for info in pfs.get_file_info(fs.FileSelector(f"{base}/{table}", recursive=True, allow_not_found=True)):
        if info.path.endswith(".parquet"):
            y, m, d = info.path.split("/")[-4:-1]
            out[dt.date(int(y), int(m), int(d))] = info.path
    return out


def uri(pfs: fs.FileSystem, path: str) -> str:
    """The path as DuckDB reads it."""
    return path if isinstance(pfs, fs.LocalFileSystem) else f"s3://{path}"


def last_closed_day() -> dt.date:
    """Yesterday (UTC): today's gold files are still being written."""
    return dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
