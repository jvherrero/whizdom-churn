"""Data alerts (EDA stage 3): checks of the source data and of the features against a benchmark.

    .venv/bin/python src/features/alerts.py --as-of 2026-10-06
    .venv/bin/python src/features/alerts.py --as-of 2026-10-06 --stage source --brand-id basel

The source checks run on the local daily caches. The benchmark is the period used to build the
training features, and the current window is the last days up to AS_OF.

    expectations   pandera suites (eda/expectations/)
    row_count      rows per day compared with the benchmark mean
    null_rate      null share per column compared with the benchmark
    daily_total    daily totals compared with a rolling median +- n * MAD

The feature checks compare the scoring snapshot with the training dataset (pandera, players, PSI).
Thresholds are in configs/eda_alerts.yaml.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import os
import socket
import sys
import time
from pathlib import Path

import duckdb
import mlflow
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import daily_cache  # noqa: E402
from build_features import ALL_BRANDS, FEATURES, LOOKBACK_DAYS, parse_brand_id  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "eda"))
import expectations  # noqa: E402  (pandera suites, eda/expectations/)
from dq_lib import validate_dataframe  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs/eda_alerts.yaml"
ALERTS_DIR = PROJECT_ROOT / "data/03_output/alerts"
MLFLOW_EXPERIMENT = "whizdom-churn-alerts"

ALERT_COLUMNS = ["stage", "check", "severity", "operator", "brandId", "table", "column",
                 "value", "benchmark", "threshold", "message"]


class CriticalAlert(RuntimeError):
    """Raised when a run has a critical alert. The pipeline stops."""


# Feature drift only sets drift_flag on the scores. Any other critical alert stops the run.
NON_BLOCKING_CHECKS = {"feature_drift"}


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _alert(check, severity, message, **context) -> dict:
    return {"check": check, "severity": severity, "message": message, **context}


def _graded(value: float, warn: float, critical: float | None) -> str | None:
    """'critical' / 'warn' / None for a value that is bad when it is large."""
    if critical is not None and value > critical:
        return "critical"
    return "warn" if value > warn else None


def _days(start: dt.date, end: dt.date) -> list[dt.date]:
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]


# Daily caches: profiles

def daily_profiles(cache: str, brand_id: int, days: list[dt.date], totals: list[str]) -> dict[dt.date, dict]:
    """Rows, null shares and daily totals of each day of a cache, in one SQL query."""
    scoped = daily_cache.CACHES[cache]["brand_scoped"]
    folder = daily_cache.cache_dir(cache, brand_id if scoped else None)
    cached = set(daily_cache.cached_days(cache, brand_id if scoped else None))
    files = [str(folder / f"{d}.parquet") for d in days if d in cached]
    empty = {"rows": 0, "null_rate": {}, "totals": {}}
    if not files:
        return {d: dict(empty, day=str(d)) for d in days}
    source = f"read_parquet({files}, union_by_name=true)"
    columns = [c for c in duckdb.sql(f"DESCRIBE SELECT * FROM {source}").df()["column_name"] if c != "day"]
    sums = [t for t in totals if t != "rows"]
    where = "" if scoped else f"WHERE brand_id = {int(brand_id)}"
    frame = duckdb.sql(
        "SELECT CAST(day AS DATE) AS day, COUNT(*) AS n_rows, "
        + ", ".join([f'COUNT("{c}") AS "nn__{c}"' for c in columns] + [f'SUM("{t}") AS "total__{t}"' for t in sums])
        + f" FROM {source} {where} GROUP BY 1").df()
    frame.index = pd.to_datetime(frame.pop("day")).dt.date
    out = {}
    for d in days:
        if d not in frame.index:
            out[d] = dict(empty, day=str(d))
            continue
        row, n = frame.loc[d], int(frame.loc[d, "n_rows"])
        out[d] = {"day": str(d), "rows": n,
                  "null_rate": {c: 1 - row[f"nn__{c}"] / n for c in columns} if n else {},
                  "totals": {"rows": n, **{t: float(row[f"total__{t}"]) for t in sums if pd.notna(row[f"total__{t}"])}}}
    return out


# Daily caches: checks

def check_row_count(bench: list[dict], current: list[dict], rule: dict, **ctx) -> list[dict]:
    """Compare the rows of each current day with the benchmark mean. A day with no rows is critical."""
    out = [_alert("row_count", "critical", f"no rows on {p['day']}", value=0,
                  threshold=rule["critical_relative_change"], **ctx) for p in current if p["rows"] == 0]
    base = np.mean([p["rows"] for p in bench]) if bench else 0
    for p in [p for p in current if p["rows"]]:
        change = p["rows"] / base - 1 if base else np.inf
        severity = _graded(abs(change), rule["warn_relative_change"], rule.get("critical_relative_change"))
        if severity:
            out.append(_alert("row_count", severity, f"{p['day']}: {p['rows']:,} rows, {change:+.0%} vs the benchmark "
                              f"daily mean {base:,.0f}", value=p["rows"], benchmark=base,
                              threshold=rule["warn_relative_change"], **ctx))
    return out


def check_null_rate(bench: list[dict], current: list[dict], rule: dict, **ctx) -> list[dict]:
    def pooled(profiles):
        """Null share per column, weighted by rows."""
        rows = sum(p["rows"] for p in profiles)
        columns = {c for p in profiles for c in p["null_rate"]}
        return rows, {c: sum(p["null_rate"].get(c, 0) * p["rows"] for p in profiles) / rows for c in columns} if rows else {}

    _, base = pooled(bench)
    n_cur, cur = pooled(current)
    if n_cur < rule.get("min_rows", 0):
        return []
    out = []
    for col, rate in cur.items():
        if col not in base:
            continue  # new columns are reported by the expectations
        increase = (rate - base[col]) * 100
        severity = _graded(increase, rule["warn_increase_pts"], rule.get("critical_increase_pts"))
        if severity:
            out.append(_alert("null_rate", severity, f"null share {rate:.1%} vs {base[col]:.1%} in the benchmark "
                              f"(+{increase:.1f} pts)", column=col, value=rate, benchmark=base[col],
                              threshold=rule["warn_increase_pts"], **ctx))
    return out


def check_daily_totals(profiles: dict, current_days: list[dt.date], rule: dict, metrics: list[str], **ctx) -> list[dict]:
    """Compare each current day with the median +- n * MAD of the previous days."""
    out = []
    for day in current_days:
        for metric in metrics:
            value = profiles[day]["totals"].get(metric)
            window = [profiles[day - dt.timedelta(days=i)]["totals"].get(metric)
                      for i in range(1, rule["window_days"] + 1) if day - dt.timedelta(days=i) in profiles]
            window = np.array([w for w in window if w is not None], dtype=float)
            if value is None or len(window) < rule["min_periods"]:
                continue
            median = np.median(window)
            mad = 1.4826 * np.median(np.abs(window - median))
            z = abs(value - median) / mad if mad > 0 else (0.0 if value == median else np.inf)
            severity = _graded(z, rule["warn_n_mad"], rule.get("critical_n_mad"))
            if severity:
                out.append(_alert("daily_total", severity, f"{day}: {metric} = {value:,.0f}, outside the rolling band "
                                  f"{median:,.0f} +- {rule['warn_n_mad']} MAD ({z:.1f} MAD away)", column=metric,
                                  value=value, benchmark=median, threshold=rule["warn_n_mad"], **ctx))
    return out


def check_expectations(df: pd.DataFrame, suite: str, rule: dict, columns: list[str] | None = None,
                       **ctx) -> list[dict]:
    """Pandera suite, limited to `columns` if given."""
    module = getattr(expectations, suite)
    result = (module.validate(df) if columns is None
              else validate_dataframe(df, [c for c in module.COLUMNS if c.name in columns]))
    return [_alert("expectations", rule["severity"], f"{suite}: {i.message}", column=i.column, **ctx)
            for i in result.issues]


def benchmark_days(cutoffs: list[str]) -> list[dt.date]:
    """Days used to build the training features."""
    dates = sorted(dt.date.fromisoformat(str(c)) for c in cutoffs)
    return _days(dates[0] - dt.timedelta(days=LOOKBACK_DAYS - 1), dates[-1])


def source_alerts(brand_id: int | str, as_of: str | dt.date, bench_days: list[dt.date],
                config: dict | None = None) -> pd.DataFrame:
    """Run all the daily-cache checks for a brand ("basel" means all brands)."""
    cfg = config or load_config()
    as_of = dt.date.fromisoformat(str(as_of))
    current_days = _days(as_of - dt.timedelta(days=cfg["current_window_days"] - 1), as_of)
    band = cfg["daily_total"]
    brands = ([int(b) for b in daily_cache.read("activity", None, start=as_of, end=as_of,
                                               columns="DISTINCT brand_id")["brand_id"]]
              if brand_id == ALL_BRANDS else [int(brand_id)])
    alerts = []
    for brand in brands:
        for cache, spec in cfg["tables"].items():
            ctx = {"brandId": brand, "table": cache}
            # Payments are only complete from March 2026, so the benchmark starts there.
            valid_from = dt.date.fromisoformat(str(spec.get("valid_from", "1900-01-01")))
            bench_used = [d for d in bench_days if d >= valid_from]
            days = sorted(set(bench_used) | set(_days(current_days[0] - dt.timedelta(days=band["window_days"]),
                                                       current_days[-1])))
            profiles = daily_profiles(cache, brand, days, spec["totals"])
            bench, current = [profiles[d] for d in bench_used], [profiles[d] for d in current_days]
            alerts += check_row_count(bench, current, cfg["row_count"], **ctx)
            alerts += check_null_rate(bench, current, cfg["null_rate"], **ctx)
            alerts += check_daily_totals({d: p for d, p in profiles.items() if d >= valid_from}, current_days,
                                         band, spec["totals"], **ctx)
            if any(p["rows"] for p in current):
                rows = daily_cache.read(cache, brand, start=current_days[0], end=current_days[-1])
                alerts += check_expectations(rows, cache, cfg["expectations"], **ctx)
    return pd.DataFrame([{"stage": "source", **a} for a in alerts], columns=ALERT_COLUMNS)


# Features

def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """PSI of `actual` against `expected`, with quantile bins (one bin per value if there are few)."""
    expected, actual = pd.Series(expected).dropna(), pd.Series(actual).dropna()
    if expected.nunique() <= bins:
        e = expected.value_counts(normalize=True)
        a = actual.value_counts(normalize=True)
        index = e.index.union(a.index)
        e, a = e.reindex(index, fill_value=0).to_numpy(), a.reindex(index, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_alerts(benchmark: pd.DataFrame, snapshot: pd.DataFrame, config: dict | None = None,
                   model_features: set[str] | None = None) -> pd.DataFrame:
    """Compare the scoring features with the training dataset, per brand.
    If `model_features` is given, only those features are checked for drift."""
    cfg = config or load_config()
    alerts = [{"table": "feature_snapshot", "brandId": "all", **a}
              for a in check_expectations(snapshot, "feature_snapshot", cfg["expectations"])]
    rule, drift = cfg["row_count"], cfg["feature_drift"]
    for brand, snap in snapshot.groupby("brandId"):
        bench = benchmark[benchmark["brandId"] == brand]
        ctx = {"brandId": int(brand), "table": "feature_snapshot"}
        if bench.empty:
            alerts.append(_alert("new_categories", "critical", "brand not in the training dataset",
                                 column="brandId", value=int(brand), **ctx))
            continue
        base = bench.groupby("cutoff_date").size().mean()
        change = len(snap) / base - 1
        severity = _graded(abs(change), rule["warn_relative_change"], rule.get("critical_relative_change"))
        if severity:
            alerts.append(_alert("row_count", severity, f"{len(snap):,} players, {change:+.0%} vs {base:,.0f} per "
                                 f"training cutoff", value=len(snap), benchmark=base,
                                 threshold=rule["warn_relative_change"], **ctx))
        checked = [f for f in FEATURES if f in snap and f in bench
                   and (model_features is None or f in model_features)]
        drifting = []
        for feature in checked:
            value = psi(bench[feature], snap[feature], drift["bins"])
            severity = _graded(value, drift["warn_psi"], drift.get("critical_psi"))
            if severity:
                drifting.append(feature)
                alerts.append(_alert("feature_drift", severity, f"PSI {value:.3f} vs the training dataset",
                                     column=feature, value=value, threshold=drift["warn_psi"], **ctx))
        if len(drifting) >= drift["critical_n_features"]:  # many small shifts together
            alerts.append(_alert("feature_drift", "critical", f"{len(drifting)} features with PSI > "
                                 f"{drift['warn_psi']}: {drifting}", value=len(drifting),
                                 threshold=drift["critical_n_features"], **ctx))
    return pd.DataFrame([{"stage": "features", **a} for a in alerts], columns=ALERT_COLUMNS)


# Report

MAX_ALERT_TAGS = 50


def _where(a) -> str:
    """Short name of an alert, e.g. table.column (brand X)."""
    target = ".".join(str(v) for v in (a.table, a.column) if isinstance(v, str) and v)
    return f"{target} (brand {a.brandId}{', ' + a.operator if isinstance(a.operator, str) else ''})"


def _one_line(a) -> str:
    return f"{a.severity.upper()} | {a.check} | {_where(a)} | {a.message}"


def _alerts_markdown(alerts: pd.DataFrame, stage: str, brand: str, as_of: str) -> str:
    counts = alerts["severity"].value_counts()
    lines = [f"**Alerts, {stage} checks** (brand {brand}, as of {as_of}): "
             f"{int(counts.get('critical', 0))} critical, {int(counts.get('warn', 0))} warn. "
             "Critical stops the pipeline (feature drift only flags); thresholds in `configs/eda_alerts.yaml`.", "",
             "| severity | check | where | what |", "|---|---|---|---|"]
    lines += [f"| {a.severity} | {a.check} | {_where(a)} | {str(a.message).replace('|', '/')} |"
              for a in alerts.itertuples(index=False)]
    return "\n".join(lines)


def acknowledge(alerts: pd.DataFrame, stage: str, as_of: str | dt.date, config: dict | None = None) -> pd.DataFrame:
    """Turn into warnings the critical alerts that a person reviewed (`acknowledged` in the config).

    An entry covers one stage, check, brand and table, from `from` to `until` (both included), and only
    while the alert value stays at or below `max_value`: a bigger jump, or the same one after `until`,
    stops the pipeline again. The alert stays in the report, with the reason in its message."""
    entries = (load_config() if config is None else config).get("acknowledged") or []
    if alerts.empty or not entries:
        return alerts
    day = dt.date.fromisoformat(str(as_of))
    alerts = alerts.copy()
    for e in entries:
        if e["stage"] != stage or not (dt.date.fromisoformat(str(e["from"])) <= day <= dt.date.fromisoformat(str(e["until"]))):
            continue
        hit = ((alerts["severity"] == "critical") & (alerts["check"] == e["check"]) & (alerts["table"] == e["table"])
               & (alerts["brandId"].astype(str) == str(e["brandId"]))
               & (pd.to_numeric(alerts["value"], errors="coerce") <= e["max_value"]))
        alerts.loc[hit, "severity"] = "warn"
        alerts.loc[hit, "message"] = alerts.loc[hit, "message"] + f" [acknowledged until {e['until']}: {e['reason']}]"
    return alerts


def report(alerts: pd.DataFrame, stage: str, brand: str, as_of: str, stop_on_critical: bool = True) -> Path:
    """Save the alerts to CSV and MLflow, print them and raise CriticalAlert if needed. Reviewed alerts
    (`acknowledged` in the config) are reported as warnings."""
    alerts = acknowledge(alerts, stage, as_of)
    timestamp_unix = int(time.time())
    name = f"alerts_{stage}_brand{brand}_{as_of}_{timestamp_unix}"
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ALERTS_DIR / f"{name}.csv"
    order = alerts["severity"].map({"critical": 0, "warn": 1})
    alerts = alerts.assign(_order=order).sort_values(["_order", "check"]).drop(columns="_order")
    alerts.to_csv(path, index=False)
    counts = alerts["severity"].value_counts()
    n_warn, n_critical = int(counts.get("warn", 0)), int(counts.get("critical", 0))
    n_blocking = int(((alerts["severity"] == "critical") & ~alerts["check"].isin(NON_BLOCKING_CHECKS)).sum())

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"timestamp_unix": timestamp_unix, "user": getpass.getuser(), "host": socket.gethostname(),
                         "source_script": "src/features/alerts.py", "brand_id": brand, "stage": stage})
        mlflow.log_params({"as_of": as_of, "config": os.path.relpath(CONFIG_PATH, PROJECT_ROOT)})
        mlflow.log_metrics({"n_alerts": len(alerts), "n_warn": n_warn, "n_critical": n_critical})
        # Counts per type, e.g. n_warn_null_rate.
        for (severity, check), n in alerts.groupby(["severity", "check"]).size().items():
            mlflow.log_metric(f"n_{severity}_{check}", int(n))

        if len(alerts):
            mlflow.set_tag("mlflow.note.content", _alerts_markdown(alerts, stage, brand, as_of))
            for i, a in enumerate(alerts.head(MAX_ALERT_TAGS).itertuples(index=False), 1):
                mlflow.set_tag(f"alert_{i:02d}", _one_line(a))
            mlflow.log_table(alerts.astype(str), "alerts_table.json")
        else:
            mlflow.set_tag("mlflow.note.content", f"No alerts in the **{stage}** checks (brand {brand}, as of {as_of}).")
        mlflow.log_artifact(str(path))
        mlflow.log_artifact(str(CONFIG_PATH))

    print(f"alerts [{stage}] brand {brand} as of {as_of}: {n_critical} critical, {n_warn} warn "
          f"-> {os.path.relpath(path, PROJECT_ROOT)}")
    if len(alerts):
        with pd.option_context("display.max_colwidth", 90, "display.width", 200):
            print(alerts[["severity", "check", "table", "brandId", "column", "message"]].to_string(index=False))
    if n_blocking and stop_on_critical:
        raise CriticalAlert(f"{n_blocking} critical alert(s) in the {stage} checks, see {path.name}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Data alerts as of one date (thresholds: configs/eda_alerts.yaml).")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--brand-id", type=parse_brand_id, default=64, help="brandId, or 'basel' for every brand")
    parser.add_argument("--stage", choices=["source", "features", "all"], default="all")
    args = parser.parse_args()

    from build_features import build_feature_store
    from run_pipeline import training_cutoffs

    label = str(args.brand_id)
    cutoffs = training_cutoffs(dt.date.fromisoformat(args.as_of))
    failed = False
    if args.stage in ("source", "all"):
        alerts = source_alerts(args.brand_id, args.as_of, benchmark_days(cutoffs))
        failed |= _report_no_raise(alerts, "source", label, args.as_of)
    if args.stage in ("features", "all"):
        dataset = max((PROJECT_ROOT / "data/processed").glob(f"train_dataset_{label}_*.parquet"),
                      key=lambda p: p.stat().st_mtime)
        snapshot = build_feature_store(args.as_of, args.brand_id)
        # Only the features used by the latest models are checked for drift.
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
        from train import latest_run, run_info
        try:
            used = {f for m in ("lightgbm_classifier", "cox_ph") for f in run_info(latest_run(m, label).run_id)["features"]}
        except LookupError:
            used = None
        failed |= _report_no_raise(feature_alerts(pd.read_parquet(dataset), snapshot, model_features=used),
                                   "features", label, args.as_of)
    sys.exit(1 if failed else 0)


def _report_no_raise(alerts, stage, brand, as_of) -> bool:
    try:
        report(alerts, stage, brand, as_of)
    except CriticalAlert as e:
        print(f"CRITICAL: {e}")
        return True
    return False


if __name__ == "__main__":
    main()
