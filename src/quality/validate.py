"""Validate lake tables day by day with the Great Expectations suites of gx/expectations/.

    make validate-gx TABLES="gld_player_payments transactions" START=2026-10-01 END=2026-10-03
    .venv/bin/python src/quality/validate.py --tables gld_player_payments --start 2026-10-01 --end 2026-10-03

For each table and day, all the files of the day are read with DuckDB, only the columns that the
expectations use (never the pii columns), and validated as one pandas batch: every row is checked, no
sampling. The column set and the types are compared with the Parquet footer of the day. Results are
cached by table and day, with the suite hash and the file sizes, so a rerun only validates new or
changed days.

Output:
    data/03_output/gx_docs/index.html     GE Data Docs (local only: they show example values)
    docs/dq_reports/gx_summary.md         counts and percentages only
    MLflow experiment whizdom-churn-quality
"""

from __future__ import annotations

import os

os.environ.setdefault("GX_ANALYTICS_ENABLED", "false")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import great_expectations as gx  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import contracts  # noqa: E402
import layers  # noqa: E402
import suites  # noqa: E402
from datalake import s3_duckdb, s3_filesystem, uri  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "data/02_intermediate/quality/gx"
DOCS_DIR = PROJECT_ROOT / "data/03_output/gx_docs"
SUMMARY_PATH = PROJECT_ROOT / "docs/dq_reports/gx_summary.md"
MLFLOW_EXPERIMENT = "whizdom-churn-quality"
CACHE_VERSION = 3  # change it when the cached fields change


def day_files(pfs, base: str, table: str) -> dict[dt.date, list[tuple[str, int]]]:
    """Parquet files of each day, as (path, size)."""
    files = layers.list_files(pfs, base, table)
    files = files[(files["suffix"] == ".parquet") & files["day"].notna()]
    return {day: sorted(zip(g["path"], g["size"])) for day, g in files.groupby("day")}


def schema_rows(pfs, path: str, table: contracts.TableDef) -> list[dict]:
    """Column set and types of the file against the contract."""
    schema = pq.ParquetFile(pfs.open_input_file(path)).schema_arrow
    actual = {f.name: str(f.type) for f in schema}
    expected = {c.name: c.type for c in table.columns}
    rows = []
    for name in sorted(set(expected) - set(actual)):
        rows.append({"rule": "schema", "check": "missing column", "column": name})
    for name in sorted(set(actual) - set(expected)):
        rows.append({"rule": "schema", "check": "extra column", "column": name})
    for name in sorted(set(actual) & set(expected)):
        if actual[name] != expected[name]:
            rows.append({"rule": "schema", "check": f"type {expected[name]} -> {actual[name]}", "column": name})
    return [r | {"success": False, "unexpected_count": None, "unexpected_percent": None} for r in rows]


def load_day(con, pfs, files: list[tuple[str, int]], raw: dict, names: set[str]) -> pd.DataFrame:
    day_col = raw["meta"].get("batch_day_column")
    columns = suites.columns_used(raw)
    if "_version" in names and "_version" not in columns:
        columns.append("_version")   # to tell unmerged versions from true duplicates of the grain
    exprs = [f'"{c}"' for c in columns]
    if day_col:
        exprs.append(f"strftime(CAST(\"{day_col}\" AS DATE), '%Y-%m-%d') AS \"{suites.BATCH_DAY}\"")
    paths = ", ".join(f"'{uri(pfs, p)}'" for p, _ in files)
    return con.sql(f"SELECT {', '.join(exprs)} FROM read_parquet([{paths}], union_by_name = true)").df()


class Runner:
    """One ephemeral GE context; one suite, validation definition and checkpoint per table."""

    def __init__(self, docs_dir: Path | None = None):
        self.ctx = gx.get_context(mode="ephemeral")
        self.ctx.variables.progress_bars = {"globally": False}
        self.ctx.add_data_docs_site(site_name="local", site_config={
            "class_name": "SiteBuilder", "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
            "store_backend": {"class_name": "TupleFilesystemStoreBackend", "base_directory": str(docs_dir or DOCS_DIR)}})
        self.source = self.ctx.data_sources.add_pandas(name="lake")
        self.checkpoints = {}

    def checkpoint(self, name: str, raw: dict):
        if name not in self.checkpoints:
            batch = self.source.add_dataframe_asset(name=name).add_batch_definition_whole_dataframe(name="day")
            suite = self.ctx.suites.add(gx.ExpectationSuite(name=name, expectations=raw["expectations"],
                                                            meta=raw["meta"]))
            vd = self.ctx.validation_definitions.add(gx.ValidationDefinition(name=name, data=batch, suite=suite))
            self.checkpoints[name] = self.ctx.checkpoints.add(gx.Checkpoint(
                name=name, validation_definitions=[vd], result_format={"result_format": "BASIC"},
                actions=[gx.checkpoint.UpdateDataDocsAction(name="docs", site_names=["local"])]))
        return self.checkpoints[name]

    def validate(self, name: str, raw: dict, df: pd.DataFrame, day: dt.date) -> list[dict]:
        result = self.checkpoint(name, raw).run(
            batch_parameters={"dataframe": df}, expectation_parameters={"batch_days": [str(day)]},
            run_id=gx.core.run_identifier.RunIdentifier(run_name=f"{name} {day}"))
        rows = []
        for validation in result.run_results.values():
            for r in validation.results:
                cfg = r.expectation_config
                kw = cfg.kwargs
                error = r.exception_info.get("exception_message") if r.exception_info.get("raised_exception") else None
                rows.append({"rule": cfg.meta.get("rule"), "check": cfg.type,
                             "column": kw.get("column") or "+".join(kw.get("column_list", [])),
                             "success": bool(r.success), "error": error,
                             # counts only: the example values (partial_unexpected_list) stay in the Data Docs
                             "unexpected_count": r.result.get("unexpected_count"),
                             "unexpected_percent": r.result.get("unexpected_percent")})
        return rows


def validate_table(runner: Runner, con, pfs, base: str, table: contracts.TableDef, days: list[dt.date]) -> list[dict]:
    raw = suites.load(table.name)
    key = suites.suite_hash(raw)
    files = day_files(pfs, base, table.name)
    out = []
    for day in days:
        if day not in files:
            out.append({"table": table.name, "day": str(day), "n_rows": 0, "rows": [], "status": "no files"})
            continue
        signature = [CACHE_VERSION, key, [[p.rsplit("/", 1)[-1], s] for p, s in files[day]]]
        cache = CACHE_DIR / table.name / f"{day}.json"
        if cache.exists() and json.loads(cache.read_text())["signature"] == signature:
            out.append(json.loads(cache.read_text()))
            continue
        t0 = time.time()
        df = load_day(con, pfs, files[day], raw, {c.name for c in table.columns})
        rows = schema_rows(pfs, files[day][-1][0], table) + runner.validate(table.name, raw, df, day)
        for r in rows:
            if not r["success"] and r["rule"] == "grain" and "_version" in df:
                r["left_after_latest_version"] = left_after_latest_version(df, table.grain)
            # observed range of the failed value checks: aggregates only, enough to pick the right bounds
            if not r["success"] and r["rule"] in ("range", "non_negative") and r["column"] in df:
                values = pd.to_numeric(df[r["column"]], errors="coerce")
                r |= {"min": float(values.min()), "max": float(values.max())}
        item = {"table": table.name, "day": str(day), "n_rows": len(df), "rows": rows, "status": "validated",
                "signature": signature}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(item, default=str))
        failed = sum(not r["success"] for r in rows)
        print(f"  {table.name} {day}: {len(df):,} rows, {failed} failed checks ({time.time() - t0:.0f}s)")
        out.append(item)
    return out


def left_after_latest_version(df: pd.DataFrame, grain: list[str]) -> int:
    """Rows still in a repeated grain after keeping the latest _version of each grain. If 0, the
    duplicates are older versions that ClickHouse had not merged yet (ReplacingMergeTree)."""
    latest = df[df["_version"] == df.groupby(grain, dropna=False)["_version"].transform("max")]
    return int(latest.duplicated(grain, keep=False).sum())


def summarise(items: list[dict]) -> pd.DataFrame:
    rows = [{"table": i["table"], "day": i["day"], **r} for i in items for r in i["rows"]]
    columns = ["table", "day", "rule", "check", "column", "success", "error", "unexpected_count",
               "unexpected_percent", "min", "max", "left_after_latest_version"]
    return pd.DataFrame(rows, columns=columns)


def _num(x) -> str:
    return "" if pd.isna(x) else f"{x:,.4g}"


def write_summary(items: list[dict], start: dt.date, end: dt.date, path: Path | None = None) -> Path:
    path = path or SUMMARY_PATH
    checks = summarise(items)
    lines = ["# Great Expectations Summary", "",
             f"Generated by `src/quality/validate.py` on {dt.date.today()}, for {start} to {end}. Every row of "
             "each day is checked. This file has counts and percentages only; the Data Docs in "
             "`data/03_output/gx_docs/` (local, not versioned) also show example values.", "",
             "## Tables", "",
             "| Table | Days validated | Days without files | Rows | Failed checks | Days with a failure | Grain unique every day |",
             "|---|---|---|---|---|---|---|"]
    for table in dict.fromkeys(i["table"] for i in items):
        part = [i for i in items if i["table"] == table]
        done = [i for i in part if i["status"] == "validated"]
        c = checks[checks["table"] == table]
        failed = c[~c["success"]]
        grain = c[c["rule"] == "grain"]
        grain_ok = "yes" if len(grain) and grain["success"].all() else ("no" if len(grain) else "")
        lines.append(f"| `{table}` | {len(done)} | {len(part) - len(done)} | {sum(i['n_rows'] for i in done):,} | "
                     f"{len(failed)} | {failed['day'].nunique()} | {grain_ok} |")
    failed = checks[~checks["success"]]
    lines += ["", "## Failed checks", ""]
    if failed.empty:
        lines.append("None.")
    else:
        lines += ["| Table | Rule | Check | Column | Days failed | Unexpected rows | Max % of rows | "
                  "Observed min | Observed max | Left after latest _version | Error |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        group = failed.groupby(["table", "rule", "check", "column"], dropna=False)
        for (table, rule, check, column), g in group:
            n = pd.to_numeric(g["unexpected_count"], errors="coerce")
            pct = pd.to_numeric(g["unexpected_percent"], errors="coerce")
            error = g["error"].dropna().iloc[0][:80] if g["error"].notna().any() else ""
            lines.append(f"| `{table}` | {rule} | {check.replace('expect_', '')} | `{column}` | {g['day'].nunique()} | "
                         f"{int(n.sum()) if n.notna().any() else ''} | "
                         f"{f'{pct.max():.3f}' if pct.notna().any() else ''} | {_num(g['min'].min())} | "
                         f"{_num(g['max'].max())} | {_num(g['left_after_latest_version'].sum(min_count=1))} | {error} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def log_mlflow(items: list[dict], summary: Path, tables: list[str], start: dt.date, end: dt.date) -> None:
    import mlflow

    checks = summarise(items)
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"gx_{start}_{end}_{int(time.time())}"):
        mlflow.set_tags({"source_script": "src/quality/validate.py"})
        mlflow.log_params({"start": str(start), "end": str(end), "tables": ",".join(tables)})
        mlflow.log_metrics({"n_tables": len(tables), "n_days": len({i["day"] for i in items}),
                            "n_checks": len(checks), "n_failed": int((~checks["success"]).sum()) if len(checks) else 0})
        mlflow.log_artifact(str(summary))


def run(tables: list[str], start: dt.date, end: dt.date, mlflow_log: bool = True) -> Path:
    lake = contracts.load()
    unknown = sorted(set(tables) - set(lake))
    if unknown:
        raise SystemExit(f"no lake contract for {unknown}")
    days = list(pd.date_range(start, end).date)
    runner = Runner()
    con = s3_duckdb()
    con.sql("SET TimeZone = 'UTC'")
    items = []
    for name in tables:
        table = lake[name]
        pfs, base = s3_filesystem(table.layer)
        print(f"{name} ({table.layer}), {len(days)} day(s)")
        items += validate_table(runner, con, pfs, base, table, days)
    path = write_summary(items, start, end)
    print(f"saved {path}")
    if mlflow_log:
        log_mlflow(items, path, tables, start, end)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate lake tables day by day with Great Expectations.")
    parser.add_argument("--tables", nargs="+", required=True)
    parser.add_argument("--start", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--end", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    run(args.tables, args.start, args.end, not args.no_mlflow)
