"""Reconciliation of the lake tables (phase 2): repeated or lost records, with SQL counts only.

    make reconcile START=2026-05-29 END=2026-05-31 [CHECKS="kafka same_keys totals"]
    .venv/bin/python src/quality/reconcile.py --start 2026-05-29 --end 2026-05-31 --checks kafka

Checks, set in the "reconcile" section of configs/quality.yaml:
    kafka      bronze messages: rows repeated under a business key in the day and the day before,
               offset gaps per Kafka partition over the period, delay from the event to Kafka to the lake
    same_keys  tables that should hold the same keys every day: keys only in the reference table or
               only in the other table, and values that differ for the same key ("filters" keeps a
               subset of a table, e.g. players with some activity)
    totals     an atomic table summed by key against its daily table: keys on one side only, the same
               keys still missing one day before and after (a day shift, not a loss), and measures
               that differ ("informative" measures are shown without a status)
Every number comes from DuckDB over all the rows of each day: no sampling, and no player rows leave
DuckDB. Results are cached per check and day.
Output: docs/dq_reports/reconciliation_<start>_<end>[_<checks>].md, one file per run, and MLflow.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import contracts  # noqa: E402
import validate  # noqa: E402
from datalake import s3_duckdb, s3_filesystem, uri  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs/quality.yaml"
CACHE_DIR = PROJECT_ROOT / "data/02_intermediate/quality/reconcile"
REPORT_DIR = PROJECT_ROOT / "docs/dq_reports"
MLFLOW_EXPERIMENT = "whizdom-churn-quality"
CACHE_VERSION = 4
CHECKS = ("kafka", "same_keys", "totals")


def q(column: str) -> str:
    return f'"{column}"'


def diff(column: str) -> str:
    """|a - b| as DOUBLE: unsigned columns (uint64) cannot hold a negative difference."""
    return f"abs(CAST(a.{q(column)} AS DOUBLE) - CAST(b.{q(column)} AS DOUBLE))"


def same(key: list[str], a: str = "a", b: str = "b") -> str:
    """Join condition on the key; IS NOT DISTINCT FROM so that null keys still match."""
    return " AND ".join(f"{a}.{q(k)} IS NOT DISTINCT FROM {b}.{q(k)}" for k in key)


class Lake:
    """Daily Parquet files of the lake tables, read through DuckDB."""

    def __init__(self, con):
        self.con = con
        self.tables = contracts.load()
        self.fs, self.files = {}, {}

    def day_files(self, table: str) -> dict:
        if table not in self.files:
            pfs, base = s3_filesystem(self.tables[table].layer)
            self.fs[table] = pfs
            self.files[table] = validate.day_files(pfs, base, table)
        return self.files[table]

    def source(self, table: str, day: dt.date) -> str | None:
        files = self.day_files(table).get(day)
        if not files:
            return None
        paths = ", ".join(f"'{uri(self.fs[table], p)}'" for p, _ in files)
        return f"read_parquet([{paths}], union_by_name = true)"

    def signature(self, tables: list[str], days: list[dt.date]) -> list:
        return [[t, str(d), [[p.rsplit("/", 1)[-1], s] for p, s in self.day_files(t).get(d, [])]]
                for t in tables for d in days]

    def one(self, sql: str) -> dict:
        frame = self.con.sql(sql).df()
        return {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                for k, v in frame.iloc[0].items()}


def row(check: str, name: str, day, item: str, n, problems, extra: str = "", info: bool = False) -> dict:
    return {"check": check, "name": name, "day": str(day), "item": item, "n": n, "problems": problems,
            "extra": extra, "info": info}


# ---------------------------------------------------------------------------- kafka

def kafka_day(lake: Lake, cfg: dict, day: dt.date) -> list[dict]:
    table = cfg["table"]
    src, prev = lake.source(table, day), lake.source(table, day - dt.timedelta(days=1))
    if src is None:
        return [row("kafka", table, day, "no files", None, None)]
    out = []
    for name, key in cfg["business_keys"].items():
        cols = ", ".join(q(k) for k in key)
        r = lake.one(f"SELECT count(*) AS n, count(DISTINCT row({cols})) AS k, "
                     f"count(*) FILTER (WHERE {' AND '.join(f'{q(k)} IS NULL' for k in key)}) AS nulls "
                     f"FROM {src}")
        out.append(row("kafka", table, day, f"rows repeated in the day ({name})", r["n"], r["n"] - r["k"]))
        out.append(row("kafka", table, day, f"rows with an empty key ({name})", r["n"], r["nulls"]))
        if prev:
            r2 = lake.one(f"WITH a AS (SELECT DISTINCT {cols} FROM {src}), b AS (SELECT DISTINCT {cols} FROM {prev}) "
                          f"SELECT (SELECT count(*) FROM a) AS k, count(*) AS again FROM a SEMI JOIN b ON {same(key)}")
            out.append(row("kafka", table, day, f"keys already seen the day before ({name})", r2["k"], r2["again"]))
    ev, kt, it = (q(cfg[k]) for k in ("event_time", "kafka_time", "ingest_time"))
    # Which time the day folders follow: rows whose event or Kafka day is not the folder day.
    r = lake.one(f"SELECT count(*) AS n, count(*) FILTER (WHERE CAST({ev} AS DATE) <> DATE '{day}') AS ev, "
                 f"count(*) FILTER (WHERE CAST({kt} AS DATE) <> DATE '{day}') AS kt FROM {src}")
    out.append(row("kafka", table, day, "rows whose event day is not the folder day", r["n"], r["ev"], info=True))
    out.append(row("kafka", table, day, "rows whose Kafka day is not the folder day", r["n"], r["kt"], info=True))
    for label, a, b in (("delay event to Kafka", ev, kt), ("delay Kafka to lake", kt, it)):
        r = lake.one(f"SELECT quantile_cont(epoch({b}) - epoch({a}), 0.5) AS p50, "
                     f"quantile_cont(epoch({b}) - epoch({a}), 0.99) AS p99, max(epoch({b}) - epoch({a})) AS mx, "
                     f"count(*) FILTER (WHERE {b} < {a}) AS negative, count(*) AS n FROM {src}")
        extra = f"p50 {r['p50']:,.0f} s, p99 {r['p99']:,.0f} s, max {r['mx']:,.0f} s" if r["p50"] is not None else ""
        out.append(row("kafka", table, day, f"{label}: later time before the earlier one", r["n"], r["negative"],
                       extra))
    return out


def kafka_offsets(lake: Lake, cfg: dict, days: list[dt.date]) -> list[dict]:
    """Offset gaps per partition over the whole period, hole by hole. A hole of 1 offset is usually a Kafka
    transaction marker, not a lost message. Small holes (2 to 1,000) can be lost messages. Large jumps are
    another stretch of the topic, e.g. messages filed under other days, and are only reported."""
    table = cfg["table"]
    sources = [s for s in (lake.source(table, d) for d in days) if s]
    if not sources:
        return []
    union = " UNION ALL ".join(f"SELECT kafka_topic, kafka_partition, kafka_offset FROM {s}" for s in sources)
    frame = lake.con.sql(
        f"WITH o AS (SELECT DISTINCT kafka_topic, kafka_partition, kafka_offset FROM ({union})), "
        f"h AS (SELECT kafka_topic, kafka_offset - lag(kafka_offset) OVER (PARTITION BY kafka_topic, kafka_partition "
        f"ORDER BY kafka_offset) - 1 AS miss FROM o) "
        f"SELECT kafka_topic, count(*) AS n, count(*) FILTER (WHERE miss = 1) AS holes_1, "
        f"count(*) FILTER (WHERE miss BETWEEN 2 AND 1000) AS holes_small, "
        f"coalesce(sum(miss) FILTER (WHERE miss BETWEEN 2 AND 1000), 0) AS missing_small, "
        f"count(*) FILTER (WHERE miss > 1000) AS jumps, max(miss) AS largest FROM h GROUP BY ALL").df()
    period = f"{days[0]} to {days[-1]}"
    out = []
    for r in frame.itertuples():
        n = int(r.n)
        out.append(row("kafka", table, period, f"holes of 1 offset in topic {r.kafka_topic} (transaction markers)",
                       n, int(r.holes_1), info=True))
        out.append(row("kafka", table, period, f"offsets missing in holes of 2 to 1,000 in topic {r.kafka_topic}",
                       n, int(r.missing_small), f"{int(r.holes_small):,} holes"))
        out.append(row("kafka", table, period, f"jumps over 1,000 offsets in topic {r.kafka_topic}", n, int(r.jumps),
                       f"largest {int(r.largest or 0):,} offsets", info=True))
    return out


# ---------------------------------------------------------------------------- same keys

def same_keys_day(lake: Lake, name: str, cfg: dict, tol: float, day: dt.date) -> list[dict]:
    key, compare = cfg["key"], cfg.get("compare", [])
    ref, others = cfg["tables"][0], cfg["tables"][1:]
    src_ref = lake.source(ref, day)
    if src_ref is None:
        return [row("same_keys", name, day, f"no files in {ref}", None, None)]
    out = []
    cols = ", ".join(q(c) for c in key + compare)
    filters = cfg.get("filters", {})

    def where(table: str) -> str:
        return f" WHERE {filters[table]}" if table in filters else ""
    for other in others:
        src = lake.source(other, day)
        if src is None:
            out.append(row("same_keys", name, day, f"no files in {other}", None, None))
            continue
        diffs = ", ".join(f"count(*) FILTER (WHERE a._in AND b._in AND ({diff(c)} > {tol} "
                          f"OR (a.{q(c)} IS NULL) <> (b.{q(c)} IS NULL))) AS {q('diff_' + c)}" for c in compare)
        r = lake.one(f"WITH a AS (SELECT {cols}, true AS _in FROM {src_ref}{where(ref)}), "
                     f"b AS (SELECT {cols}, true AS _in FROM {src}{where(other)}) "
                     f"SELECT count(a._in) AS n_ref, count(b._in) AS n_other, "
                     f"count(*) FILTER (WHERE b._in IS NULL) AS only_ref, "
                     f"count(*) FILTER (WHERE a._in IS NULL) AS only_other, "
                     f"count(*) FILTER (WHERE a._in AND b._in) AS both{', ' + diffs if diffs else ''} "
                     f"FROM a FULL OUTER JOIN b ON {same(key)}")
        out.append(row("same_keys", name, day, f"keys only in {ref} (not in {other})", r["n_ref"], r["only_ref"]))
        out.append(row("same_keys", name, day, f"keys only in {other} (not in {ref})", r["n_other"], r["only_other"]))
        for c in compare:
            out.append(row("same_keys", name, day, f"{c} differs ({ref} vs {other})", r["both"], r[f"diff_{c}"]))
    return out


# ---------------------------------------------------------------------------- totals

def totals_day(lake: Lake, name: str, cfg: dict, tol: float, day: dt.date) -> list[dict]:
    key, measures = cfg["key"], cfg["measures"]
    src, tgt = lake.source(cfg["source"], day), lake.source(cfg["target"], day)
    missing = [t for t, s in ((cfg["source"], src), (cfg["target"], tgt)) if s is None]
    if missing:
        return [row("totals", name, day, f"no files in {t}", None, None) for t in missing]
    keys = ", ".join(q(k) for k in key)
    agg = ", ".join(f"{sql} AS {q(m)}" for m, sql in measures.items())
    # the target is also summed by the key: with a key shorter than its grain it has several rows per key
    picked = ", ".join(f"sum({q(m)}) AS {q(m)}" for m in measures)
    checks = ", ".join(
        f"count(*) FILTER (WHERE a._in AND b._in AND abs(coalesce(CAST(a.{q(m)} AS DOUBLE), 0) - coalesce(CAST(b.{q(m)} AS DOUBLE), 0)) > {tol}) "
        f"AS {q('diff_' + m)}, sum(a.{q(m)}) AS {q('sa_' + m)}, sum(b.{q(m)}) AS {q('sb_' + m)}" for m in measures)
    r = lake.one(f"WITH a AS (SELECT {keys}, {agg}, true AS _in FROM {src} GROUP BY ALL), "
                 f"b AS (SELECT {keys}, {picked}, true AS _in FROM {tgt} GROUP BY ALL) "
                 f"SELECT count(a._in) AS n_a, count(b._in) AS n_b, "
                 f"count(*) FILTER (WHERE b._in IS NULL) AS only_a, count(*) FILTER (WHERE a._in IS NULL) AS only_b, "
                 f"count(*) FILTER (WHERE a._in AND b._in) AS both, {checks} "
                 f"FROM a FULL OUTER JOIN b ON {same(key)}")
    out = [row("totals", name, day, f"keys only in {cfg['source']} (not in {cfg['target']})", r["n_a"], r["only_a"]),
           row("totals", name, day, f"keys only in {cfg['target']} (not in {cfg['source']})", r["n_b"], r["only_b"])]
    if cfg.get("date_column"):
        out += shifted_keys(lake, name, cfg, day, keys)
    for m in measures:
        sa, sb = r[f"sa_{m}"] or 0, r[f"sb_{m}"] or 0
        out.append(row("totals", name, day, f"{m} differs", r["both"], r[f"diff_{m}"],
                       f"sum over all keys: {cfg['source']} {sa:,.2f}, {cfg['target']} {sb:,.2f}",
                       info=m in cfg.get("informative", [])))
    return out


def shifted_keys(lake: Lake, name: str, cfg: dict, day: dt.date, keys: str) -> list[dict]:
    """Keys on one side only that are not on the other side one day before or after either, comparing
    the key without the date column. These are the records really missing, not a day shift."""
    key, date = cfg["key"], cfg["date_column"]
    rest = [k for k in key if k != date]
    rest_cols = ", ".join(q(k) for k in rest)
    out = []
    for this, other in ((cfg["source"], cfg["target"]), (cfg["target"], cfg["source"])):
        near = [lake.source(other, day + dt.timedelta(days=d)) for d in (-1, 0, 1)]
        union = " UNION ALL ".join(f"SELECT {rest_cols} FROM {s}" for s in near if s)
        r = lake.one(f"WITH a AS (SELECT DISTINCT {keys} FROM {lake.source(this, day)}), "
                     f"b AS (SELECT DISTINCT {keys} FROM {lake.source(other, day)}), "
                     f"near AS (SELECT DISTINCT {rest_cols} FROM ({union})), "
                     f"only_a AS (SELECT * FROM a ANTI JOIN b ON {same(key)}) "
                     f"SELECT count(*) AS n, (SELECT count(*) FROM a) AS total "
                     f"FROM only_a ANTI JOIN near ON {same(rest, 'only_a', 'near')}")
        out.append(row("totals", name, day, f"keys only in {this}, not in {other} one day before or after either",
                       r["total"], r["n"]))
    return out


# ---------------------------------------------------------------------------- run

def cached(lake: Lake, check: str, name: str, cfg: dict, tables: list[str], days: list[dt.date], day: dt.date,
           compute) -> list[dict]:
    signature = [CACHE_VERSION, hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12],
                 lake.signature(tables, days)]
    path = CACHE_DIR / f"{check}_{name}" / f"{day}.json"
    if path.exists():
        item = json.loads(path.read_text())
        if item["signature"] == signature:
            return item["rows"]
    t0 = time.time()
    rows = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"signature": signature, "rows": rows}, default=str))
    print(f"  {check} {name} {day} ({time.time() - t0:.0f}s)")
    return rows


def status(share, cfg: dict) -> str:
    if share is None or pd.isna(share):
        return "info"
    if share > cfg["critical_above"]:
        return "critical"
    return "warn" if share > cfg["warn_above"] else "ok"


def frame(rows: list[dict], cfg: dict) -> pd.DataFrame:
    out = pd.DataFrame(rows, columns=["check", "name", "day", "item", "n", "problems", "extra", "info"])
    n = pd.to_numeric(out["n"], errors="coerce")
    problems = pd.to_numeric(out["problems"], errors="coerce")
    out["share"] = (problems / n.where(n > 0)).where(problems.notna())
    out["status"] = [status(s, cfg["status"]) if not str(i).startswith("no files") else "critical"
                     for s, i in zip(out["share"], out["item"])]
    out.loc[out["item"].str.startswith("delay"), "status"] = out.loc[out["item"].str.startswith("delay"), "status"] \
        .replace({"warn": "info", "critical": "warn"})
    out.loc[out["info"].fillna(False).astype(bool), "status"] = "info"
    return out


def report_path(start: dt.date, end: dt.date, checks: list[str]) -> Path:
    suffix = "" if set(checks) == set(CHECKS) else "_" + "-".join(checks)
    return REPORT_DIR / f"reconciliation_{start}_{end}{suffix}.md"


def write_report(results: pd.DataFrame, start: dt.date, end: dt.date, path: Path) -> Path:
    order = {"critical": 0, "warn": 1, "info": 2, "ok": 3}
    lines = ["# Reconciliation", "",
             f"Generated by `src/quality/reconcile.py` on {dt.date.today()}, for {start} to {end}. Counts from SQL "
             "over every row of each day; no player rows. Status: share of rows or keys with a problem, "
             "against `reconcile.status` in `configs/quality.yaml`.", "",
             "## Summary", "",
             "| Check | Name | Item | Days | Days with a problem | Problems | Max share | Status | Detail |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (check, name, item), g in results.groupby(["check", "name", "item"], sort=False):
        problems = pd.to_numeric(g["problems"], errors="coerce")
        worst = min(g["status"], key=order.get)
        share = g["share"].max()
        detail = g["extra"].replace("", pd.NA).dropna()
        lines.append(f"| {check} | {name} | {item} | {g['day'].nunique()} | {int((problems > 0).sum())} | "
                     f"{int(problems.sum()) if problems.notna().any() else ''} | "
                     f"{'' if pd.isna(share) else f'{100 * share:.4f}%'} | {worst} | "
                     f"{detail.iloc[-1] if len(detail) else ''} |")
    bad = results[results["status"].isin(["critical", "warn"])]
    lines += ["", "## Days with a problem", ""]
    if bad.empty:
        lines.append("None.")
    else:
        lines += ["| Day | Check | Name | Item | Rows or keys | Problems | Share | Status |", "|---|---|---|---|---|---|---|---|"]
        for r in bad.sort_values(["check", "name", "item", "day"]).itertuples():
            n = "" if pd.isna(r.n) else f"{int(r.n):,}"
            p = "" if pd.isna(r.problems) else f"{int(r.problems):,}"
            s = "" if pd.isna(r.share) else f"{100 * r.share:.4f}%"
            lines.append(f"| {r.day} | {r.check} | {r.name} | {r.item} | {n} | {p} | {s} | {r.status} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def log_mlflow(results: pd.DataFrame, report: Path, start: dt.date, end: dt.date, checks: list[str]) -> None:
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"reconcile_{start}_{end}_{int(time.time())}"):
        mlflow.set_tags({"source_script": "src/quality/reconcile.py"})
        mlflow.log_params({"start": str(start), "end": str(end), "checks": ",".join(checks)})
        mlflow.log_metrics({f"n_{s}": int((results["status"] == s).sum()) for s in ("ok", "warn", "critical")})
        mlflow.log_artifact(str(report))


def run(start: dt.date, end: dt.date, checks: list[str] | None = None, mlflow_log: bool = True) -> Path:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())["reconcile"]
    checks = checks or list(CHECKS)
    days = list(pd.date_range(start, end).date)
    con = s3_duckdb()
    con.sql("SET TimeZone = 'UTC'")
    lake = Lake(con)
    tol = cfg["tolerance"]
    rows = []
    for day in days:
        if "kafka" in checks:
            k = cfg["kafka"]
            prev = day - dt.timedelta(days=1)
            rows += cached(lake, "kafka", k["table"], k, [k["table"]], [day, prev], day,
                           lambda: kafka_day(lake, k, day))
        if "same_keys" in checks:
            for name, c in cfg["same_keys"].items():
                rows += cached(lake, "same_keys", name, c | {"tolerance": tol}, c["tables"], [day], day,
                               lambda: same_keys_day(lake, name, c, tol, day))
        if "totals" in checks:
            for name, c in cfg["totals"].items():
                near = [day + dt.timedelta(days=d) for d in (-1, 0, 1)] if c.get("date_column") else [day]
                rows += cached(lake, "totals", name, c | {"tolerance": tol}, [c["source"], c["target"]], near, day,
                               lambda: totals_day(lake, name, c, tol, day))
    if "kafka" in checks:
        rows += kafka_offsets(lake, cfg["kafka"], days)
    results = frame(rows, cfg)
    path = write_report(results, start, end, report_path(start, end, checks))
    print(f"saved {path}")
    if mlflow_log:
        log_mlflow(results, path, start, end, checks)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconciliation of the lake tables (counts only).")
    parser.add_argument("--start", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--end", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--checks", nargs="+", choices=CHECKS)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    run(args.start, args.end, args.checks, not args.no_mlflow)
