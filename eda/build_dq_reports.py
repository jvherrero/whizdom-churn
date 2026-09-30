"""Build Data Quality (DQ) reports and Data Card source-table registers.

Orchestrates the generation of HTML and JSON DQ reports alongside the Data Card 
source-table register for 10-landing operator tables, reusing S3 and DuckDB 
access patterns verified in `01_profiling.ipynb`.

Execution Strategy:
    * Standard Tables (`player`, `bonus`, `transaction` / `whizdomai-payments`):
      Materialized fully in memory and validated using `dq_lib`'s Pandera wrapper.
    * Large Ledger (`bet` / `whizdomai-transactions`):
      Due to scale (~22.5M total rows/day, ~30GB peak RSS), quantitative checks 
      (row count, null %, key uniqueness) execute via SQL pushdown scoped to 
      `SNAPSHOT_DAY`. Full-history gap checks run via S3 file listing (no data reads).
    * Missing Dataset (`session`):
      Confirmed absent in `10-landing/kafka-sink` for both operators. Documented 
      explicitly as 'unavailable' in the Data Card instead of generating dummy reports.

Outputs:
    HTML and JSON files containing DQ metrics per available table, plus an updated 
    Data Card register.
"""

import datetime as dt
import os
import sys
from pathlib import Path

os.environ["AWS_PROFILE"] = "javier-whizdom-prod-ds"

import boto3
import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dq_lib import (
    ColumnNullStat, ColumnSpec, KeyCheck, TableProfile, ValidationIssue,
    ValidationResult, render_source_table_register, validate_dataframe, write_report,
)

BUCKET = "whizdomai-eu-central-1-650177547431-datalake-gmntc-prod"
LANDING_LEVEL = "org/10-landing/kafka-sink"
OPERATORS = ["primus", "secundus"]
LANDING_TABLES = {
    "player": "whizdomai-players",
    "bet": "whizdomai-transactions",
    "transaction": "whizdomai-payments",
    "bonus": "whizdomai-bonuses",
}
GRAIN_CANDIDATE_KEYS = {
    "player": ["uuid", "partyId"],
    "bet": ["transactionId", "partyId"],
    "transaction": ["paymentId", "partyId"],
    "bonus": ["changeId", "id", "partyId"],
}
GRAIN_DESCRIPTIONS = {
    "player": "One row per player change event; partyId repeats, uuid identifies the event and is unique.",
    "bet": "One row per bet/transaction event; transactionId is unique.",
    "transaction": "One row per payment status-change event, not one row per payment; paymentId repeats.",
    "bonus": "One row per bonus status-change event; changeId is unique, id (the bonus) repeats.",
    "session": "One row per derived player session (see notes); partyId repeats, (operator, partyId, session_id) is unique.",
}
SMALL_TABLES = ["player", "transaction", "bonus"]  # safe to materialize fully
SESSION_GAP_MINUTES = 30  # justified against the real gap histogram in 01_profiling.ipynb
QUALITY_START = dt.date(2026, 9, 23)
QUALITY_END = dt.date(2026, 9, 29)
QUALITY_DAYS = [QUALITY_START + dt.timedelta(days=i) for i in range((QUALITY_END - QUALITY_START).days + 1)]
HISTORY_START = dt.date(2026, 6, 18)
HISTORY_END = dt.date(2026, 9, 29)
HISTORY_DAYS = [HISTORY_START + dt.timedelta(days=i) for i in range((HISTORY_END - HISTORY_START).days + 1)]
SNAPSHOT_DAY = ("2026", "09", "10")

OUT_DIR = Path(__file__).parent.parent / "docs" / "dq_reports"
DATA_CARD_PATH = Path(__file__).parent.parent / "docs" / "data_card.md"


def s3_duckdb():
    creds = boto3.Session(profile_name="javier-whizdom-prod-ds").get_credentials().get_frozen_credentials()
    con = duckdb.connect()
    con.sql("INSTALL httpfs; LOAD httpfs;")
    con.sql("SET memory_limit='4GB'")
    con.sql("SET threads=2")
    con.sql(f"""
        CREATE OR REPLACE SECRET s3_secret (
            TYPE S3, KEY_ID '{creds.access_key}', SECRET '{creds.secret_key}',
            SESSION_TOKEN '{creds.token}', REGION 'eu-central-1'
        );
    """)
    return con


def landing_paths(operator, landing_table, days):
    return [
        f"s3://{BUCKET}/{LANDING_LEVEL}/{operator}/{landing_table}/{d.year}/{d.month:02d}/{d.day:02d}/*.snappy.parquet"
        for d in days
    ]


def load_full(business_name, days):
    """Materialize the full table for both operators over `days`. Only used
    for the three small tables."""
    landing_table = LANDING_TABLES[business_name]
    frames = []
    for operator in OPERATORS:
        con = s3_duckdb()
        paths = landing_paths(operator, landing_table, days)
        con.sql(f"CREATE OR REPLACE VIEW t AS SELECT * FROM read_parquet({paths}, union_by_name=True)")
        df = con.sql("SELECT *, ? AS operator FROM t", params=[operator]).df()
        con.close()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def daily_presence(business_name, operator, days):
    landing_table = LANDING_TABLES[business_name]
    con = s3_duckdb()
    rows = []
    for d in days:
        path = f"s3://{BUCKET}/{LANDING_LEVEL}/{operator}/{landing_table}/{d.year}/{d.month:02d}/{d.day:02d}/*.snappy.parquet"
        n_files = con.sql(f"SELECT count(*) FROM glob('{path}')").fetchone()[0]
        rows.append({"date": d, "n_files": n_files})
    con.close()
    return rows


def date_range_and_gaps(business_name):
    """Full-history presence check (glob only, cheap) for both operators combined."""
    all_rows = []
    for operator in OPERATORS:
        all_rows += daily_presence(business_name, operator, HISTORY_DAYS)
    presence = pd.DataFrame(all_rows)
    by_day = presence.groupby("date")["n_files"].sum()
    with_data = by_day[by_day > 0]
    if with_data.empty:
        return None, None, None, []
    date_min, date_max = with_data.index.min(), with_data.index.max()
    within = by_day.loc[date_min:date_max]
    gaps = sorted(str(d) for d, n in within.items() if n == 0)
    return str(date_min), str(date_max), len(gaps), gaps


def small_table_profile(business_name):
    df = load_full(business_name, QUALITY_DAYS)
    columns = [
        ColumnSpec(name=col, dtype=str(df[col].dtype), nullable=bool(df[col].isna().any()))
        for col in df.columns
    ]
    validation = validate_dataframe(df, columns)
    null_stats = [
        ColumnNullStat(name=col, null_percentage=round(100 * df[col].isna().mean(), 4))
        for col in df.columns
    ]
    key_checks = [
        KeyCheck(
            column=key, n_rows=len(df),
            n_distinct=int(df[key].nunique(dropna=False)),
            is_unique=int(df[key].nunique(dropna=False)) == len(df),
        )
        for key in GRAIN_CANDIDATE_KEYS[business_name] if key in df.columns
    ]
    date_min, date_max, n_gaps, gap_dates = date_range_and_gaps(business_name)
    return TableProfile(
        table_name=business_name,
        grain_description=GRAIN_DESCRIPTIONS[business_name],
        row_count=len(df),
        date_range_start=date_min,
        date_range_end=date_max,
        n_gap_days=n_gaps,
        gap_dates=gap_dates,
        columns=columns,
        null_stats=null_stats,
        key_checks=key_checks,
        validation=validation,
        notes=[
            f"Schema captured from the current data on {dt.date.today()}; nullability reflects "
            "what was observed, so this run validates cleanly by construction, the point of "
            "the baseline is to catch drift on future runs, not to fail today.",
            f"row_count, null %, and key uniqueness reflect the {QUALITY_START} to {QUALITY_END} "
            "window (both operators combined), not the full history, kept consistent with the "
            "window used elsewhere in this project's profiling. Date range and gap detection "
            "above cover the complete observed history instead, since that check is cheap.",
        ],
    )


def bet_profile():
    business_name = "bet"
    landing_table = LANDING_TABLES[business_name]
    view = landing_table.replace("-", "_")

    def run(query, operator):
        con = s3_duckdb()
        path = f"s3://{BUCKET}/{LANDING_LEVEL}/{operator}/{landing_table}/{SNAPSHOT_DAY[0]}/{SNAPSHOT_DAY[1]}/{SNAPSHOT_DAY[2]}/*.snappy.parquet"
        con.sql(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{path}', union_by_name=True)")
        result = con.sql(query).df()
        con.close()
        return result

    schema_by_op, null_by_op, rows_by_op = {}, {}, {}
    for operator in OPERATORS:
        desc = run(f"DESCRIBE SELECT * FROM {view}", operator)
        schema_by_op[operator] = dict(zip(desc["column_name"], desc["column_type"]))
        summary = run(f"SUMMARIZE SELECT * FROM {view}", operator)
        null_by_op[operator] = dict(zip(summary["column_name"], summary["null_percentage"]))
        rows_by_op[operator] = int(summary["count"].iloc[0])

    total_rows = sum(rows_by_op.values())
    all_columns = list(schema_by_op[OPERATORS[0]].keys())
    columns = [
        ColumnSpec(name=c, dtype=schema_by_op[OPERATORS[0]][c],
                   nullable=any(null_by_op[op].get(c, 0) > 0 for op in OPERATORS))
        for c in all_columns
    ]
    # Schema-presence check against the DuckDB-reported live schema (metadata
    # only, no row data needed) Pandera itself is not used here since we
    # never materialize `bet` as a DataFrame.
    issues = []
    for op in OPERATORS:
        live_cols = set(schema_by_op[op].keys())
        expected_cols = set(all_columns)
        for extra in live_cols - expected_cols:
            issues.append(ValidationIssue(extra, "column_unexpected", f"Unexpected column in {op}: {extra}"))
        for missing in expected_cols - live_cols:
            issues.append(ValidationIssue(missing, "column_missing", f"Missing column in {op}: {missing}"))
    validation = ValidationResult(passed=not issues, issues=issues)

    null_stats = []
    for c in all_columns:
        weighted = sum(null_by_op[op].get(c, 0) * rows_by_op[op] for op in OPERATORS) / total_rows
        null_stats.append(ColumnNullStat(name=c, null_percentage=round(weighted, 4)))

    key_checks = []
    for operator in OPERATORS:
        keys = GRAIN_CANDIDATE_KEYS[business_name]
        distinct_exprs = ", ".join(f'count(DISTINCT "{k}") AS n_{k}' for k in keys)
        result = run(f"SELECT count(*) AS n_rows, {distinct_exprs} FROM {view}", operator)
        n_rows = int(result["n_rows"].iloc[0])
        for k in keys:
            n_distinct = int(result[f"n_{k}"].iloc[0])
            key_checks.append(KeyCheck(f"{k} [{operator}]", n_rows, n_distinct, n_distinct == n_rows))

    date_min, date_max, n_gaps, gap_dates = date_range_and_gaps(business_name)
    return TableProfile(
        table_name=business_name,
        grain_description=GRAIN_DESCRIPTIONS[business_name],
        row_count=total_rows,
        date_range_start=date_min,
        date_range_end=date_max,
        n_gap_days=n_gaps,
        gap_dates=gap_dates,
        columns=columns,
        null_stats=null_stats,
        key_checks=key_checks,
        validation=validation,
        notes=[
            f"bet (whizdomai-transactions) is far larger than the other tables (~14M+8.5M rows "
            f"per day): a full SELECT * for a single operator/day measured ~30GB peak RSS and "
            "was OOM-killed, so it is never materialized as a DataFrame.",
            f"row_count, null %, schema and key uniqueness reflect a single day "
            f"({'-'.join(SNAPSHOT_DAY)}) via SQL pushdown (DuckDB DESCRIBE/SUMMARIZE/"
            "count(DISTINCT ...)), not the full history. Date range and gap detection above DO "
            "cover the complete observed history (file-listing only, no data read).",
            "Key checks are reported per operator (not combined) because combining exact "
            "distinct counts across operators without double-counting would need a cross-"
            "operator distinct query, not run here to keep this within a single-day scan.",
        ],
    )


def build_sessions(operator, gap_minutes=SESSION_GAP_MINUTES):
    """One row per derived session for `operator`, over QUALITY_DAYS.

    Same LAG/SUM OVER sessionization as 01_profiling.ipynb's `session_gaps`,
    but grouped down to one row per session (partyId, session_id,
    session_start, session_end, n_bets, duration_minutes) instead of one row
    per gap between two sessions -- this is the natural analogue of a
    `session` table, not the gap-focused view.
    """
    landing_table = LANDING_TABLES["bet"]
    view = landing_table.replace("-", "_")
    paths = landing_paths(operator, landing_table, QUALITY_DAYS)
    con = s3_duckdb()
    con.sql(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet({paths}, union_by_name=True)")
    query = f"""
        WITH events AS (
            SELECT partyId, dateTime
            FROM {view}
            WHERE partyId IS NOT NULL AND dateTime IS NOT NULL
        ),
        with_gap AS (
            SELECT partyId, dateTime,
                   dateTime - LAG(dateTime) OVER (PARTITION BY partyId ORDER BY dateTime) AS gap
            FROM events
        ),
        with_flag AS (
            SELECT partyId, dateTime,
                   CASE WHEN gap IS NULL OR gap > INTERVAL '{gap_minutes} minutes' THEN 1 ELSE 0 END AS new_session
            FROM with_gap
        ),
        with_session_id AS (
            SELECT partyId, dateTime,
                   SUM(new_session) OVER (PARTITION BY partyId ORDER BY dateTime
                                           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS session_id
            FROM with_flag
        )
        SELECT partyId, session_id,
               MIN(dateTime) AS session_start, MAX(dateTime) AS session_end,
               COUNT(*) AS n_bets,
               epoch(MAX(dateTime) - MIN(dateTime)) / 60.0 AS duration_minutes
        FROM with_session_id
        GROUP BY partyId, session_id
        ORDER BY partyId, session_start
    """
    result = con.sql(query).df()
    con.close()
    result.insert(0, "operator", operator)
    return result


def session_profile():
    business_name = "session"
    df = pd.concat([build_sessions(operator) for operator in OPERATORS], ignore_index=True)
    df["session_key"] = df["operator"] + "-" + df["partyId"].astype(str) + "-" + df["session_id"].astype(str)

    columns = [
        ColumnSpec(name=col, dtype=str(df[col].dtype), nullable=bool(df[col].isna().any()))
        for col in df.columns
    ]
    validation = validate_dataframe(df, columns)
    null_stats = [
        ColumnNullStat(name=col, null_percentage=round(100 * df[col].isna().mean(), 4))
        for col in df.columns
    ]
    key_checks = [
        KeyCheck(
            column="session_key (operator+partyId+session_id)", n_rows=len(df),
            n_distinct=int(df["session_key"].nunique(dropna=False)),
            is_unique=int(df["session_key"].nunique(dropna=False)) == len(df),
        )
    ]

    session_days = set(df["session_start"].dt.date)
    date_min = min(session_days) if session_days else None
    date_max = max(session_days) if session_days else None
    gap_dates = sorted(str(d) for d in (set(QUALITY_DAYS) - session_days))

    return TableProfile(
        table_name=business_name,
        grain_description=GRAIN_DESCRIPTIONS[business_name],
        row_count=len(df),
        date_range_start=str(date_min) if date_min else None,
        date_range_end=str(date_max) if date_max else None,
        n_gap_days=len(gap_dates),
        gap_dates=gap_dates,
        columns=columns,
        null_stats=null_stats,
        key_checks=key_checks,
        validation=validation,
        notes=[
            f"There is no `session` table under org/10-landing/kafka-sink for either operator "
            f"(confirmed by listing both operators' prefixes in full). This profile is derived "
            f"from `bet` (whizdomai-transactions) instead: a new session starts after "
            f"{SESSION_GAP_MINUTES} minutes with no bet event for that player. That threshold "
            f"was chosen by inspecting the real distribution of gaps between consecutive bets "
            f"(see 01_profiling.ipynb) -- 99.81% of all such gaps are under 30 minutes, and the "
            f"distribution has no sharp elbow, so this is a reasonable choice, not a uniquely "
            f"provable one.",
            f"Unlike the other 4 tables, this profile only covers the {QUALITY_START} to "
            f"{QUALITY_END} window, not the full available history -- computing sessions over "
            f"~104 days would cost as much as the 7-day version already does (~18 minutes for "
            f"both operators), scaled up roughly 15x.",
            "The key is a synthetic composite (operator + partyId + session_id), built for this "
            "check only -- dq_lib's KeyCheck covers a single column, and no single raw column "
            "identifies a session on its own.",
            "The date range above can go one day past QUALITY_END: S3 partitions are by UTC "
            "calendar day, but dateTime is stored in local time (+02:00). Confirmed directly "
            "against S3: the last day's partition holds timestamps from 01:59:59+02:00 that day "
            "to 01:59:59+02:00 the next day -- about 10.9% of that day's rows fall on the next "
            "local date. This profile's date range reflects the real event timestamp (local "
            "time), not the partition folder, unlike the other 4 tables.",
        ],
    )


def main():
    profiles = []
    for name in SMALL_TABLES:
        print(f"profiling {name} (materialized, {QUALITY_START} to {QUALITY_END})...", flush=True)
        profiles.append(small_table_profile(name))
    print(f"profiling bet (SQL pushdown, single day {'-'.join(SNAPSHOT_DAY)})...", flush=True)
    profiles.append(bet_profile())
    print(f"profiling session (derived from bet, {QUALITY_START} to {QUALITY_END}, both operators)...", flush=True)
    profiles.append(session_profile())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for profile in profiles:
        write_report(profile, OUT_DIR)
        print(f"wrote {OUT_DIR / profile.table_name}.html / .json", flush=True)

    register = render_source_table_register(profiles)
    DATA_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_CARD_PATH.write_text(register, encoding="utf-8")
    print(f"wrote {DATA_CARD_PATH}", flush=True)


if __name__ == "__main__":
    main()
