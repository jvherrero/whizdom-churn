"""EDA stage 1 (data-quality checks) for the source tables the model reads: T2 of the delivery plan.

    .venv/bin/python eda/profile_tables.py                          # every table, every closed day
    .venv/bin/python eda/profile_tables.py --tables gld_player_signals_daily --start 2026-01-01
    .venv/bin/python eda/profile_tables.py --skip-keys              # parquet metadata only (seconds)

Two passes, both aggregate-only (no player rows are loaded, nothing is sampled):

1. Parquet metadata of every daily file (no data read): rows, column types, null counts and
   min/max per column, processing time. Gives row counts, date range and gaps, schema drift,
   null rates, value ranges (e.g. a negative stake) and which days were backfilled at once.
2. Keys, one SQL aggregation per day over the key columns only: grain uniqueness, rows that repeat
   a key with another _version (the tables come from ClickHouse ReplacingMergeTree, which keeps
   old versions until it merges: "READ WITH FINAL" in gld-schemas.xlsx), and players of the daily
   activity tables missing from that day's signals (referential integrity).

Both passes are cached per (table, day) in data/02_intermediate/profile_tables/, so a rerun only reads
the new days. Outputs: docs/dq_reports/{table}.html/.json (dq_lib) and the source-table
register in docs/data_card.md. DATALAKE_ROOT=<local folder> reads a local copy with the same layout
instead of S3 (tests).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from pyarrow import fs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from dq_lib import (  # noqa: E402
    ColumnNullStat, ColumnSpec, KeyCheck, TableProfile, ValidationIssue, ValidationResult,
    render_source_table_register, write_report,
)

sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
from datalake import daily_files, s3_duckdb, s3_filesystem, last_closed_day  # noqa: E402
from datalake import uri as _uri  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "data/02_intermediate/profile_tables"
OUT_DIR = PROJECT_ROOT / "docs/dq_reports"
DATA_CARD_PATH = PROJECT_ROOT / "docs/data_card.md"
REGISTER_START, REGISTER_END = "<!-- source-register:start -->", "<!-- source-register:end -->"
SIGNALS = "gld_player_signals_daily"

# The source tables the churn model reads (names as deployed in S3; gld-schemas.xlsx describes the rebuild).
TABLES = {
    SIGNALS: {
        "date": "snapshot_date", "key": ["tenant_id", "player_id", "snapshot_date"],
        "grain": "One row per player ever seen and as-of day (snapshot_date, UTC); the precomputed "
                 "signals: l7d/l30d/l90d/l180d windows, prior windows, tenure, recency, platform scores.",
        "use": "features (T6)",
    },
    "gld_player_engagement_daily": {
        "date": "snapshot_date", "key": ["tenant_id", "player_id", "snapshot_date"],
        "grain": "One row per player ever seen and day: sessions, bets and active day that day.",
        "use": "labels (T7): active days after the cutoff",
    },
    "gld_player_financial_daily": {
        "date": "snapshot_date", "key": ["tenant_id", "player_id", "snapshot_date"],
        "grain": "One row per player ever seen and day: EUR wagered, GGR, NGR, deposits, withdrawals that day.",
        "use": "labels and value (T7, value at risk)",
    },
    "gld_player_gaming_daily": {
        "date": "event_date",
        "key": ["tenant_id", "brand_id", "player_id", "game_id", "provider_code", "event_date", "currency",
                "source_system"],
        "grain": "One row per player, game, provider, day and currency with play that day (sparse).",
        "use": "signal explorer (T4): game breadth, wins",
    },
    "gld_player_payments_daily": {
        "date": "event_date", "key": ["tenant_id", "brand_id", "player_id", "event_date", "currency", "source_system"],
        "grain": "One row per player, processed day and currency with payments that day (sparse).",
        "use": "signal explorer (T4): failed deposits, withdrawals",
    },
    "gld_player_bonus_daily": {
        "date": "event_date",
        "key": ["tenant_id", "brand_id", "player_id", "event_date", "award_type", "currency", "source_system"],
        "grain": "One row per player, day, award type and currency with bonus activity that day (sparse).",
        "use": "signal explorer (T4): bonus dependence",
    },
}
SPARSE = [t for t in TABLES if t != SIGNALS and TABLES[t]["key"][1] == "brand_id"]  # checked against signals

# Columns that can never be negative (amounts paid or played, counts, durations); a negative minimum
# is a data-quality issue. Net figures (GGR, NGR, net cash), deltas and model scores can be negative.
NON_NEGATIVE = re.compile(r"(wagered|deposit|withdraw|won|turnover|payout|granted|released|expired|forfeited|"
                          r"cost|rounds|bets|wins|_count|attempts|sessions|play_secs|active_days|tenure|days_since)")
SIGNED = re.compile(r"(ggr|ngr|net_|delta|momentum|acceleration|_z|score|escalation|position|velocity|"
                    r"propensity|fee|tax)")


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def footer(pfs: fs.FileSystem, path: str, day: dt.date) -> pd.DataFrame:
    """One row per top-level column of one daily file, from its parquet metadata only."""
    md = pq.ParquetFile(pfs.open_input_file(path)).metadata
    schema = md.schema.to_arrow_schema()
    cols = {f.name: {"type": str(f.type), "nulls": 0, "min": None, "max": None} for f in schema}
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        for i in range(rg.num_columns):
            chunk = rg.column(i)
            name = chunk.path_in_schema
            if name not in cols:  # a leaf of a nested (list) column: not profiled
                continue
            s = chunk.statistics
            if s is None:
                continue
            if s.has_null_count:
                cols[name]["nulls"] += s.null_count
            if s.has_min_max:
                lo, hi = cols[name]["min"], cols[name]["max"]
                cols[name]["min"] = s.min if lo is None else min(lo, s.min)
                cols[name]["max"] = s.max if hi is None else max(hi, s.max)
    signature = "|".join(f"{n}:{c['type']}" for n, c in cols.items())
    frame = pd.DataFrame([{
        "day": day, "column": n, "type": c["type"], "nulls": c["nulls"],
        "min_num": _number(c["min"]), "max_num": _number(c["max"]),
        "min_str": None if c["min"] is None else str(c["min"]), "max_str": None if c["max"] is None else str(c["max"]),
    } for n, c in cols.items()])
    processed = cols.get("_processed_at", {}).get("max")
    return frame.assign(n_rows=md.num_rows, processed_at=None if processed is None else str(processed),
                        schema_hash=hashlib.md5(signature.encode()).hexdigest()[:10])


def key_stats(con, pfs, table: str, path: str, day: dt.date, signals_path: str | None) -> dict:
    """Grain uniqueness and _version duplicates of one day, plus (sparse tables) players missing from
    that day's signals. Key columns only."""
    key = TABLES[table]["key"]
    source = f"read_parquet('{_uri(pfs, path)}')"
    row = con.sql(f"""
        SELECT COUNT(*) AS n_rows, COUNT(DISTINCT ({', '.join(key)})) AS n_keys,
               COUNT(DISTINCT ({', '.join(key)}, _version)) AS n_key_versions
        FROM {source}""").fetchone()
    out = {"day": day, "n_rows": row[0], "n_keys": row[1], "n_key_versions": row[2], "n_orphan_players": None}
    if signals_path is not None:
        out["n_orphan_players"] = con.sql(f"""
            SELECT COUNT(*) FROM (SELECT DISTINCT tenant_id, player_id FROM {source}) d
            ANTI JOIN (SELECT tenant_id, player_id FROM read_parquet('{_uri(pfs, signals_path)}')) s
            USING (tenant_id, player_id)""").fetchone()[0]
    return out


def _cached(path: Path, frame_days: list[dt.date], compute, workers: int, label: str) -> pd.DataFrame:
    """Read the cache, compute the missing days in parallel, write it back."""
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    done = set(pd.to_datetime(old["day"]).dt.date) if len(old) else set()
    todo = [d for d in frame_days if d not in done]
    if todo:
        t0 = time.time()
        with ThreadPoolExecutor(workers) as pool:
            parts = list(pool.map(compute, todo))
        new = pd.concat([p if isinstance(p, pd.DataFrame) else pd.DataFrame([p]) for p in parts], ignore_index=True)
        old = pd.concat([old, new], ignore_index=True) if len(old) else new
        path.parent.mkdir(parents=True, exist_ok=True)
        old.to_parquet(path, index=False)
        print(f"  {label}: {len(todo)} new day(s) in {time.time() - t0:.0f}s")
    old["day"] = pd.to_datetime(old["day"]).dt.date
    return old[old["day"].isin(frame_days)]


def profile_table(table: str, foot: pd.DataFrame, keys: pd.DataFrame | None, end: dt.date) -> TableProfile:
    spec = TABLES[table]
    per_day = foot.drop_duplicates("day").sort_values("day")
    days = list(per_day["day"])
    gaps = sorted(set(pd.date_range(days[0], days[-1]).date) - set(days))
    rows_total = int(per_day["n_rows"].sum())
    latest = foot[foot["day"] == days[-1]]

    by_col = foot.groupby("column").agg(nulls=("nulls", "sum"), rows=("n_rows", "sum"), first_day=("day", "min"),
                                        min_num=("min_num", "min"), max_num=("max_num", "max"))
    columns = [ColumnSpec(r.column, r.type, bool(by_col.loc[r.column, "nulls"] > 0)) for r in latest.itertuples()]
    null_stats = [ColumnNullStat(c, round(100 * r.nulls / r.rows, 4) if r.rows else 0.0) for c, r in by_col.iterrows()]

    issues, notes = [], []
    # Schema drift: columns that appear, disappear or change type over the history.
    if per_day["schema_hash"].nunique() > 1:
        for c, r in by_col.iterrows():
            if r.first_day > days[0]:
                issues.append(ValidationIssue(c, "schema_drift", f"column first appears on {r.first_day}"))
        gone = sorted(set(by_col.index) - set(latest["column"]))
        for c in gone:
            issues.append(ValidationIssue(c, "schema_drift", f"column absent from the latest file ({days[-1]})"))
        types = foot.groupby("column")["type"].nunique()
        for c in types[types > 1].index:
            issues.append(ValidationIssue(c, "schema_drift", f"type changes over time: {sorted(foot.loc[foot['column'] == c, 'type'].unique())}"))
    # Value ranges: amounts, counts and durations below zero.
    for c, r in by_col.iterrows():
        if NON_NEGATIVE.search(c) and not SIGNED.search(c) and r.min_num is not None and r.min_num < 0:
            n_days = int((foot.loc[foot["column"] == c, "min_num"] < 0).sum())
            issues.append(ValidationIssue(c, "negative_value", f"minimum {r.min_num:,.4f}, on {n_days} day(s)"))
    # One as-of / event date per daily file, equal to its folder.
    dates = foot[foot["column"] == spec["date"]]
    wrong = dates[(dates["min_str"] != dates["max_str"]) | (dates["min_str"] != dates["day"].astype(str))]
    if len(wrong):
        issues.append(ValidationIssue(spec["date"], "date_partition",
                                      f"{len(wrong)} file(s) hold another date than their folder, e.g. {wrong['day'].iloc[0]}: "
                                      f"{wrong['min_str'].iloc[0]} to {wrong['max_str'].iloc[0]}"))

    key_checks = []
    if keys is not None and len(keys):
        n_rows, n_keys = int(keys["n_rows"].sum()), int(keys["n_keys"].sum())
        key_checks.append(KeyCheck("+".join(spec["key"]), n_rows, n_keys, n_rows == n_keys))
        dup = keys[keys["n_rows"] > keys["n_keys"]]
        if len(dup):
            extra = int((dup["n_rows"] - dup["n_keys"]).sum())
            versions = int((dup["n_key_versions"] - dup["n_keys"]).sum())
            issues.append(ValidationIssue("+".join(spec["key"]), "duplicate_keys",
                                          f"{extra:,} rows repeat a key on {len(dup)} of {len(keys)} day(s) "
                                          f"({extra / n_rows:.2%} of the rows); {versions:,} of them are older _version rows "
                                          "of the same key: keep the latest _version per key (ClickHouse FINAL)"))
        if keys["n_orphan_players"].notna().any():
            orphans = int(keys["n_orphan_players"].fillna(0).sum())
            if orphans:
                issues.append(ValidationIssue("player_id", "referential_integrity",
                                              f"{orphans:,} player-days are missing from that day's {SIGNALS}"))
        notes.append(f"Keys checked on {len(keys)} of {len(days)} day(s) (key columns only, SQL aggregation).")
    else:
        notes.append("Keys not checked in this run (--skip-keys).")

    processed = pd.to_datetime(per_day["processed_at"], utc=True, errors="coerce")
    batch = processed.dt.date.value_counts()
    backfill = batch[batch > 1]
    if len(backfill):
        notes.append("Backfilled history: " + "; ".join(f"{n} day(s) processed on {d}" for d, n in backfill.sort_index().items())
                     + ". Point-in-time correctness of those days must be confirmed with the data team.")
    notes.append(f"Rows per day: first {int(per_day['n_rows'].iloc[0]):,}, median {int(per_day['n_rows'].median()):,}, "
                 f"last {int(per_day['n_rows'].iloc[-1]):,}. Days up to {end} (the last closed day).")
    notes.append(f"Used for: {spec['use']}.")
    return TableProfile(
        table_name=table, grain_description=spec["grain"], row_count=rows_total,
        date_range_start=str(days[0]), date_range_end=str(days[-1]), n_gap_days=len(gaps),
        gap_dates=[str(d) for d in gaps], columns=columns, null_stats=null_stats, key_checks=key_checks,
        validation=ValidationResult(passed=not issues, issues=issues), notes=notes,
    )


def write_register(profiles: list[TableProfile], path: Path) -> None:
    """The source-table register, kept between markers at the top of the data card."""
    register = render_source_table_register(profiles).replace(
        "## Source-table register", "## Source-table register", 1)
    block = f"{REGISTER_START}\n{register}{REGISTER_END}\n"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if REGISTER_START in text:
        text = text[:text.index(REGISTER_START)] + block + text[text.index(REGISTER_END) + len(REGISTER_END) + 1:]
    else:
        text = block + text
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- semantic checks

# Signal columns and how to recompute them from the local caches (src/features/daily_cache.py):
# (cache, column or "days", aggregate, window in days). "recency" = days since the last bet.
SEMANTIC_CHECKS = {
    "days_since_bet": ("activity", None, "recency", None),
    "active_days_l7d": ("activity", "day", "days", 7), "active_days_l30d": ("activity", "day", "days", 30),
    "bets_l7d": ("activity", "bets", "sum", 7), "bets_l30d": ("activity", "bets", "sum", 30),
    "wagered_eur_l7d": ("activity", "turnover_eur", "sum", 7), "wagered_eur_l30d": ("activity", "turnover_eur", "sum", 30),
    "ggr_eur_l7d": ("activity", "ggr_eur", "sum", 7), "ggr_eur_l30d": ("activity", "ggr_eur", "sum", 30),
    "deposits_eur_l30d": ("payments", "deposit_eur", "sum", 30), "withdrawals_eur_l30d": ("payments", "withdraw_eur", "sum", 30),
}
MATCH_TOLERANCE = 0.02   # relative; plus 0.5 absolute (one day, one bet, half a euro)
MIN_MATCH = 0.95         # share of players that must match for the column to be OK


def semantic_checks(brand_id: int, days: list[dt.date]) -> pd.DataFrame:
    """Each signal column of SEMANTIC_CHECKS against the same quantity recomputed from the daily
    caches, for the brand's players with a bet in the 30 days up to each day. One row per (day, column)."""
    import numpy as np

    import daily_cache
    import signal_snapshots

    snaps = signal_snapshots.snapshots(days, brand_id)
    rows = []
    for day in days:
        ts = pd.Timestamp(day)
        act = daily_cache.read("activity", brand_id, start=day - dt.timedelta(days=29), end=day)
        act["day"] = pd.to_datetime(act["day"])
        players = act[["tenant_id", "player_id"]].drop_duplicates()
        snap = snaps[snaps["snapshot_date"] == day].merge(players, on=["tenant_id", "player_id"])
        try:
            pay = daily_cache.read("payments", brand_id, start=day - dt.timedelta(days=29), end=day)
            pay["day"] = pd.to_datetime(pay["day"])
        except FileNotFoundError:
            pay = None
        for column, (cache, source, how, window) in SEMANTIC_CHECKS.items():
            data = act if cache == "activity" else pay
            if data is None or column not in snap:
                continue
            key = ["tenant_id", "player_id"]
            if how == "recency":
                mine = (ts - data.groupby(key)["day"].max()).dt.days
            else:
                part = data[data["day"] > ts - pd.Timedelta(window, unit="D")]
                mine = part.groupby(key)["day"].nunique() if how == "days" else part.groupby(key)[source].sum()
            both = snap.set_index(key)[[column]].join(mine.rename("recomputed")).fillna({"recomputed": 0})
            g, r = both[column].astype(float), both["recomputed"].astype(float)
            match = float(np.isclose(g, r, rtol=MATCH_TOLERANCE, atol=0.5).mean())
            rows.append({"day": day, "column": column, "recomputed_from": f"{cache}.{source or 'day'}", "players": len(both),
                         "match_share": round(match, 4), "corr": round(float(g.corr(r)), 4) if g.std() and r.std() else None,
                         "table_sum": round(float(g.sum()), 2), "recomputed_sum": round(float(r.sum()), 2),
                         "status": "OK" if match >= MIN_MATCH else "MISMATCH"})
    # tenure_days must grow by the calendar days between two snapshots for the same player.
    for prev, day in zip(days, days[1:]):
        a, b = snaps[snaps["snapshot_date"] == prev], snaps[snaps["snapshot_date"] == day]
        both = a.merge(b, on=["tenant_id", "player_id"], suffixes=("_a", "_b"))
        if both.empty:
            continue
        growth = both["tenure_days_b"].astype(float) - both["tenure_days_a"].astype(float)
        match = float(((growth - (day - prev).days).abs() <= 1).mean())
        rows.append({"day": day, "column": "tenure_days", "recomputed_from": f"tenure_days on {prev} + {(day - prev).days} days",
                     "players": len(both), "match_share": round(match, 4), "corr": None,
                     "table_sum": round(float(growth.median()), 2), "recomputed_sum": float((day - prev).days),
                     "status": "OK" if match >= MIN_MATCH else "MISMATCH"})
    return pd.DataFrame(rows)


def deposit_completeness(brand_id: int) -> pd.DataFrame:
    """Completed deposits per month against the players with a bet that month (local caches only).
    Agreement between source tables cannot show that deposits are missing everywhere; this can: a month
    whose share of depositing players falls far below the brand's usual level is incomplete."""
    import daily_cache

    act = daily_cache.read("activity", brand_id, columns="tenant_id, player_id, day")
    pay = daily_cache.read("payments", brand_id, columns="tenant_id, player_id, day, deposit_count, failed_deposit_count")
    for frame in (act, pay):
        frame["month"] = pd.to_datetime(frame["day"]).dt.to_period("M")
    active = act.groupby("month")["player_id"].nunique().rename("players_betting")
    deposits = pay[pay["deposit_count"] > 0].groupby("month").agg(
        depositors=("player_id", "nunique"), completed_deposits=("deposit_count", "sum"))
    failed = pay.groupby("month")["failed_deposit_count"].sum().rename("failed_deposits")
    out = pd.concat([active, deposits, failed], axis=1).fillna(0)
    out["depositor_share"] = out["depositors"] / out["players_betting"]
    usual = out["depositor_share"].quantile(0.75)  # the brand's usual level: robust to a run of empty months
    out["status"] = ["OK" if r >= 0.5 * usual else "INCOMPLETE" for r in out["depositor_share"]]
    return out.reset_index().assign(month=lambda d: d["month"].astype(str))


def write_semantic_report(table: pd.DataFrame, brand_id: int, out_dir: Path,
                          completeness: pd.DataFrame | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"semantic_checks_brand{brand_id}.csv", index=False)
    summary = table.groupby("column").agg(days=("day", "size"), min_match=("match_share", "min"),
                                          median_match=("match_share", "median"),
                                          mismatched_days=("status", lambda s: int((s == "MISMATCH").sum())))
    bad = table[table["status"] == "MISMATCH"].groupby("column")["day"].agg(lambda d: f"{min(d)} to {max(d)}")
    lines = [f"# Semantic checks of gld_player_signals_daily (brand {brand_id})", "",
             "Each signal column against the same quantity recomputed from the daily caches "
             "(`src/features/daily_cache.py`), for the brand's players with a bet in the 30 days up to each day. "
             f"A column matches a player within {MATCH_TOLERANCE:.0%} (or 0.5 absolute); it is OK on a day when "
             f"at least {MIN_MATCH:.0%} of the players match.", "",
             "| Column | Days checked | Median match | Worst match | Days not OK | Not OK from / to |", "|---|---|---|---|---|---|"]
    for col, r in summary.iterrows():
        lines.append(f"| `{col}` | {int(r.days)} | {r.median_match:.1%} | {r.min_match:.1%} | {int(r.mismatched_days)} | {bad.get(col, '')} |")
    if completeness is not None:
        lines += ["", "## Completeness of completed deposits", "",
                  "Players with a completed deposit / players with a bet, per month (gld_player_payments_daily). "
                  "A month below half the brand's usual share (75th percentile of the months) is INCOMPLETE: deposits are missing there, even "
                  "where the source tables agree with each other.", "",
                  "| Month | Players betting | Depositors | Depositor share | Completed deposits | Failed deposits | Status |",
                  "|---|---|---|---|---|---|---|"]
        for r in completeness.itertuples():
            lines.append(f"| {r.month} | {int(r.players_betting):,} | {int(r.depositors):,} | {r.depositor_share:.1%} | "
                         f"{int(r.completed_deposits):,} | {int(r.failed_deposits):,} | {r.status} |")
    path = out_dir / f"semantic_checks_brand{brand_id}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA stage 1 (data quality) for the source tables.")
    parser.add_argument("--tables", nargs="+", default=list(TABLES), choices=list(TABLES))
    parser.add_argument("--start", type=dt.date.fromisoformat)
    parser.add_argument("--end", type=dt.date.fromisoformat, help="default: yesterday (UTC), the last closed day")
    parser.add_argument("--skip-keys", action="store_true", help="parquet metadata only")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--data-card", type=Path, default=DATA_CARD_PATH)
    parser.add_argument("--semantic", action="store_true",
                        help="also check the signal columns against the daily caches, first day of every month")
    parser.add_argument("--brand-id", type=int, default=64, help="brand of the semantic checks")
    parser.add_argument("--only-semantic", action="store_true", help="skip the table profiles, run the semantic checks only")
    args = parser.parse_args()
    end = args.end or last_closed_day()

    pfs, base = s3_filesystem()
    con = s3_duckdb() if not args.skip_keys else None
    signal_files = daily_files(pfs, base, SIGNALS)
    profiles = []
    for table in ([] if args.only_semantic else args.tables):
        print(f"{table}:")
        files = {d: p for d, p in daily_files(pfs, base, table).items() if d <= end and (not args.start or d >= args.start)}
        if not files:
            print("  no files")
            continue
        days = sorted(files)
        foot = _cached(args.cache_dir / "footers" / f"{table}.parquet", days,
                       lambda d: footer(pfs, files[d], d), args.workers, "parquet metadata")
        keys = None
        if con is not None:
            def compute(d):
                cursor = con.cursor()  # one DuckDB cursor per thread
                try:
                    return key_stats(cursor, pfs, table, files[d], d, signal_files.get(d) if table in SPARSE else None)
                finally:
                    cursor.close()
            keys = _cached(args.cache_dir / "keys" / f"{table}.parquet", days, compute, max(1, args.workers // 4), "keys")
        profile = profile_table(table, foot, keys, end)
        write_report(profile, args.out_dir)
        profiles.append(profile)
        status = "PASS" if profile.validation.passed else f"{len(profile.validation.issues)} issue(s)"
        print(f"  {profile.date_range_start} to {profile.date_range_end}, {profile.row_count:,} rows, "
              f"{profile.n_gap_days} gap day(s): {status}")
    if profiles:
        write_register(profiles, args.data_card)
        print(f"wrote {os.path.relpath(args.out_dir, PROJECT_ROOT)}/*.html|json and the register in "
              f"{os.path.relpath(args.data_card, PROJECT_ROOT)}")
    if args.semantic or args.only_semantic:
        import daily_cache

        daily_cache.build("activity", end=end)
        daily_cache.build("payments", args.brand_id, end=end)
        first = daily_cache.cached_days("activity")[0] + dt.timedelta(days=30)
        months = [d.date() for d in pd.date_range(args.start or first, end, freq="MS")]
        table = semantic_checks(args.brand_id, months)
        completeness = deposit_completeness(args.brand_id)
        completeness.to_csv(args.out_dir / f"deposit_completeness_brand{args.brand_id}.csv", index=False)
        path = write_semantic_report(table, args.brand_id, args.out_dir, completeness)
        incomplete = completeness.loc[completeness["status"] == "INCOMPLETE", "month"].tolist()
        if incomplete:
            print(f"deposits INCOMPLETE in {incomplete[0]} to {incomplete[-1]} ({len(incomplete)} month(s))")
        worst = table[table["status"] == "MISMATCH"]["column"].unique()
        print(f"semantic checks, brand {args.brand_id}, {len(months)} day(s): "
              + (f"MISMATCH in {sorted(worst)}" if len(worst) else "every column OK")
              + f" -> {os.path.relpath(path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
