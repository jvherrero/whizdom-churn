"""Distribution of the gold metrics and the right histogram bins for each one (phase 3).

    make distributions START=2026-09-01 END=2026-09-30 [TABLES="gld_player_signals_daily"]
    .venv/bin/python src/quality/distributions.py --start 2026-09-01 --end 2026-09-30

For every numeric metric of the lake contracts of the layer in configs/quality.yaml (section
"distributions"), one SQL pass over all the rows of the period gives: rows, nulls, zeros, negatives,
min, max, mean, standard deviation, percentiles, skewness, kurtosis and an approximate count of
distinct values. Percentiles and distinct values use approx_quantile and approx_count_distinct: they
read every row (no sampling) but their result is approximate, which is enough to choose bins.

Each metric gets a type and a bin proposal (thresholds in the config):
    few_values     one bin per value
    zero_inflated  a bin for 0 and 9 quantile bins of the non-zero values
    signed         bins around 0 (signed log if the tails are heavy)
    heavy_tail     log(1 + x) scale
    skewed         quantile bins
    symmetric      equal width, Freedman and Diaconis rule, capped at max_bins
It also reports the effective bins of plain deciles (what alerts.psi uses today: repeated edges are
dropped) and of the proposal, and flags extreme values: the largest absolute value far above the
p99 of the non-zero absolute values (with many zeros, the plain p99 is near 0 and flags everything).
Output: docs/dq_reports/distributions_<start>_<end>.md and MLflow. Aggregates only, no player rows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
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
CACHE_DIR = PROJECT_ROOT / "data/02_intermediate/quality/distributions"
REPORT_DIR = PROJECT_ROOT / "docs/dq_reports"
MLFLOW_EXPERIMENT = "whizdom-churn-quality"
CACHE_VERSION = 2
NUMERIC = ("int", "uint", "float", "double", "decimal")
PERCENTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
DECILES = [i / 10 for i in range(11)]
NINTHS = [i / 9 for i in range(10)]   # 9 bins for the non-zero values, plus the bin for 0: 10 in all


def metrics(table: contracts.TableDef, exclude: str) -> list[str]:
    skip = re.compile(exclude)
    return [c.name for c in table.columns if c.type.startswith(NUMERIC) and not c.key and c.group != "meta"
            and not c.pii and not skip.search(c.name)]


def source(table: contracts.TableDef, days: list[dt.date]) -> tuple[str | None, list]:
    """read_parquet over every file of the period, and the file signature for the cache."""
    pfs, base = s3_filesystem(table.layer)
    files = validate.day_files(pfs, base, table.name)
    picked = [(p, s) for d in days for p, s in files.get(d, [])]
    if not picked:
        return None, []
    paths = ", ".join(f"'{uri(pfs, p)}'" for p, _ in picked)
    return f"read_parquet([{paths}], union_by_name = true)", [[p.rsplit("/", 4)[-4:], s] for p, s in picked]


def stats_sql(columns: list[str], src: str) -> str:
    parts = ["count(*) AS rows"]
    for i, c in enumerate(columns):
        x = f'CAST("{c}" AS DOUBLE)'
        parts += [f"count({x}) AS n_{i}", f"count(*) FILTER (WHERE {x} = 0) AS zero_{i}",
                  f"count(*) FILTER (WHERE {x} < 0) AS neg_{i}", f"min({x}) AS min_{i}", f"max({x}) AS max_{i}",
                  f"avg({x}) AS mean_{i}", f"stddev_samp({x}) AS sd_{i}", f"skewness({x}) AS skew_{i}",
                  f"kurtosis({x}) AS kurt_{i}", f"approx_count_distinct({x}) AS distinct_{i}",
                  f"approx_quantile({x}, {PERCENTILES}) AS pct_{i}", f"approx_quantile({x}, {DECILES}) AS dec_{i}",
                  f"approx_quantile({x}, {NINTHS}) FILTER (WHERE {x} > 0) AS pos_{i}",
                  f"approx_quantile(abs({x}), 0.99) FILTER (WHERE {x} <> 0) AS p99nz_{i}"]
    return f"SELECT {', '.join(parts)} FROM {src}"


def number(v):
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else v


def table_stats(con, columns: list[str], src: str, block: int) -> list[dict]:
    out = []
    for start in range(0, len(columns), block):
        chunk = columns[start:start + block]
        rel = con.sql(stats_sql(chunk, src))
        r = dict(zip(rel.columns, rel.fetchone()))
        for i, c in enumerate(chunk):
            pct = r[f"pct_{i}"] or [None] * len(PERCENTILES)
            out.append({
                "metric": c, "rows": r["rows"], "n": r[f"n_{i}"], "zeros": r[f"zero_{i}"], "negatives": r[f"neg_{i}"],
                "min": number(r[f"min_{i}"]), "max": number(r[f"max_{i}"]), "mean": number(r[f"mean_{i}"]),
                "sd": number(r[f"sd_{i}"]), "skewness": number(r[f"skew_{i}"]), "kurtosis": number(r[f"kurt_{i}"]),
                "distinct": r[f"distinct_{i}"],
                **{f"p{round(100 * p)}": number(v) for p, v in zip(PERCENTILES, pct)},
                "deciles": [number(v) for v in (r[f"dec_{i}"] or [])],
                "ninths_positive": [number(v) for v in (r[f"pos_{i}"] or [])],
                "p99_nonzero": number(r[f"p99nz_{i}"])})
    return out


def effective_bins(edges: list) -> int:
    """Bins left when repeated quantile edges are dropped, as in alerts.psi (np.unique of the edges)."""
    return max(len({e for e in edges if e is not None}) - 1, 0)


def classify(s: dict, cfg: dict) -> dict:
    """Type of the metric, bin proposal, effective bins and flags."""
    t, n = cfg["thresholds"], s["n"] or 0
    zero_share = s["zeros"] / n if n else 0.0
    neg_share = s["negatives"] / n if n else 0.0
    skew, kurt = s["skewness"] or 0.0, s["kurtosis"] or 0.0
    deciles = effective_bins(s["deciles"])
    flags = []
    largest = max(abs(s["max"] or 0), abs(s["min"] or 0))
    if s["p99_nonzero"] and largest / s["p99_nonzero"] > t["extreme_ratio"]:
        flags.append(f"extreme values (largest {largest / s['p99_nonzero']:,.0f} x p99 of non-zero values)")
    if n == 0:
        kind, proposal, bins = "empty", "no values", 0
    elif (s["distinct"] or 0) <= t["few_values"]:
        kind, proposal, bins = "few_values", "one bin per value", int(s["distinct"] or 0)
    elif zero_share > t["zero_share"]:
        kind, proposal = "zero_inflated", "a bin for 0, 9 quantile bins for the non-zero values"
        bins = 1 + effective_bins(s["ninths_positive"])
    elif neg_share > t["negative_share"]:
        kind, proposal, bins = "signed", "bins around 0" + (", signed log" if kurt > t["kurtosis"] else ""), 10
    elif kurt > t["kurtosis"]:
        kind, proposal, bins = "heavy_tail", "log(1 + x), equal width on that scale", 10
    elif skew > t["skewness"]:
        kind, proposal, bins = "skewed", "deciles", deciles
    else:
        iqr = (s["p75"] or 0) - (s["p25"] or 0)
        width = 2 * iqr * n ** (-1 / 3) if iqr > 0 else 0
        span = (s["p99"] or 0) - (s["p1"] or 0)
        bins = min(cfg["max_bins"], math.ceil(span / width)) if width > 0 else cfg["max_bins"]
        kind, proposal = "symmetric", "equal width (Freedman and Diaconis)"
    if deciles < 10 and kind not in ("few_values", "empty"):
        flags.append(f"deciles keep {deciles} of 10 bins")
    return {"type": kind, "proposal": proposal, "bins": bins, "decile_bins": deciles,
            "null_share": 1 - n / s["rows"] if s["rows"] else None, "zero_share": zero_share,
            "negative_share": neg_share, "flags": flags}


def fmt(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{v:,.{digits}g}" if abs(v) < 1e6 else f"{v:,.3e}"


def pct(v) -> str:
    return "" if v is None else f"{100 * v:.1f}"


def cfg_ratio() -> str:
    return f"{yaml.safe_load(CONFIG_PATH.read_text())['distributions']['thresholds']['extreme_ratio']:g}"


def write_report(results: dict[str, list[dict]], start: dt.date, end: dt.date) -> Path:
    rows = [r | {"table": t} for t, items in results.items() for r in items]
    frame = pd.DataFrame(rows)
    lines = ["# Distribution of the metrics", "",
             f"Generated by `src/quality/distributions.py` on {dt.date.today()}, for {start} to {end}: every row "
             "of the period; percentiles and distinct values are approximate (DuckDB approx_quantile and "
             "approx_count_distinct). Thresholds in `distributions` of `configs/quality.yaml`. Aggregates only.", ""]
    if not frame.empty:
        lines += ["## Summary", "", f"{len(frame)} metrics in {frame['table'].nunique()} tables.", "",
                  "| Type | Metrics | Bins proposed |", "|---|---|---|"]
        for kind, g in frame.groupby("type"):
            lines.append(f"| {kind} | {len(g)} | {g['proposal'].iloc[0]} |")
        lose = frame[(frame["decile_bins"] < 10) & ~frame["type"].isin(["few_values", "empty"])]
        lines += ["", f"Plain deciles, as `alerts.psi` uses them, keep fewer than 10 bins in **{len(lose)} of "
                  f"{len(frame)}** metrics, because of repeated values (mostly zeros).", ""]
        extreme = frame[frame["flags"].map(lambda f: any(x.startswith("extreme") for x in f))]
        if len(extreme):
            lines += ["### Extreme values", "",
                      "Largest absolute value over the p99 of the non-zero absolute values, above "
                      f"{cfg_ratio()}.", "",
                      "| Table | Metric | p99 of non-zero | Min | Max | Ratio |", "|---|---|---|---|---|---|"]
            largest = pd.concat([extreme["max"].abs(), extreme["min"].abs()], axis=1).max(axis=1)
            for r in extreme.assign(ratio=largest / extreme["p99_nonzero"]).sort_values("ratio", ascending=False) \
                    .itertuples():
                lines.append(f"| `{r.table}` | `{r.metric}` | {fmt(r.p99_nonzero)} | {fmt(r.min)} | {fmt(r.max)} | "
                             f"{r.ratio:,.0f} |")
            lines.append("")
    for table, items in results.items():
        lines += [f"## {table}", "",
                  "| Metric | Rows | % null | % zero | % neg | p1 | p50 | p99 | Max | Skew | Kurt | Type | Bins | "
                  "Decile bins | Flags |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in items:
            lines.append(f"| `{r['metric']}` | {r['rows']:,} | {pct(r['null_share'])} | {pct(r['zero_share'])} | "
                         f"{pct(r['negative_share'])} | {fmt(r['p1'])} | {fmt(r['p50'])} | {fmt(r['p99'])} | "
                         f"{fmt(r['max'])} | {fmt(r['skewness'], 2)} | {fmt(r['kurtosis'], 2)} | {r['type']} | "
                         f"{r['bins']} | {r['decile_bins']} | {'; '.join(r['flags'])} |")
        lines.append("")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"distributions_{start}_{end}.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def log_mlflow(results: dict, report: Path, start: dt.date, end: dt.date) -> None:
    import mlflow

    rows = [r for items in results.values() for r in items]
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"distributions_{start}_{end}_{int(time.time())}"):
        mlflow.set_tags({"source_script": "src/quality/distributions.py"})
        mlflow.log_params({"start": str(start), "end": str(end), "tables": len(results)})
        mlflow.log_metrics({"n_metrics": len(rows),
                            "n_decile_bins_lost": sum(r["decile_bins"] < 10 for r in rows),
                            "n_extreme": sum(any(f.startswith("extreme") for f in r["flags"]) for r in rows)})
        mlflow.log_artifact(str(report))


def run(start: dt.date, end: dt.date, tables: list[str] | None = None, mlflow_log: bool = True) -> Path:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())["distributions"]
    lake = contracts.load()
    names = tables or sorted(n for n, t in lake.items() if t.layer == cfg["layer"])
    days = list(pd.date_range(start, end).date)
    con = s3_duckdb()
    con.sql("SET TimeZone = 'UTC'")
    key = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    results = {}
    for name in names:
        table = lake[name]
        columns = metrics(table, cfg["exclude_columns"])
        src, files = source(table, days)
        if not columns or src is None:
            print(f"{name}: no metrics or no files in the period")
            continue
        cache = CACHE_DIR / f"{name}_{start}_{end}.json"
        signature = [CACHE_VERSION, key, columns, files]
        if cache.exists() and json.loads(cache.read_text())["signature"] == signature:
            stats = json.loads(cache.read_text())["stats"]
        else:
            t0 = time.time()
            stats = table_stats(con, columns, src, cfg["columns_per_query"])
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"signature": signature, "stats": stats}))
            print(f"{name}: {len(columns)} metrics ({time.time() - t0:.0f}s)")
        results[name] = [s | classify(s, cfg) for s in stats]
    path = write_report(results, start, end)
    print(f"saved {path}")
    if mlflow_log:
        log_mlflow(results, path, start, end)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distribution of the gold metrics and bin proposals.")
    parser.add_argument("--start", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--end", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--tables", nargs="+")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    run(args.start, args.end, args.tables, not args.no_mlflow)
