"""
Data alerts: checks of the source data the pipeline reads and of the feature snapshot against a
benchmark, with the thresholds of configs/eda_alerts.yaml (EDA stage 3).

    .venv/bin/python src/features/alerts.py --as-of 2026-10-06                  # source + features
    .venv/bin/python src/features/alerts.py --as-of 2026-10-06 --stage source --brand-id basel

Source checks run on the local daily caches (src/features/daily_cache.py: activity, financial and
payments of the brand), the exact rows the features and labels are built from, with one SQL
aggregation per cache and no S3 read. The benchmark is the days the training features were built
from; the current window is the last current_window_days up to AS_OF.
    expectations     a pandera suite failing (eda/expectations/{activity,financial,payments}.py), on the current days
    row_count        rows per day (players with a bet, with money moving, with a payment) outside
                     +-N% of the benchmark's daily mean; a day with no rows is critical
    null_rate        null share of a cache column up more than N points vs the benchmark
    daily_total      a daily total (bets, EUR, deposits) outside the rolling band (median +- n * MAD
                     of the previous days)
Feature checks: the scoring snapshot vs the training dataset (expectations, players, PSI drift).
New categorical values are not checked: the caches have no categorical column (the source tables already map
games, currencies and payment types; money is in EUR).
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
    """Raised when a run has at least one critical alert that blocks it: the pipeline stops."""


# Feature drift never stops a run: the delivery plan writes the scores and flags them (drift_flag).
# Every other critical alert (expectations, row count, null rate) stops it before scoring.
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


# ---------------------------------------------------------------- daily caches: profiles

def daily_profiles(cache: str, brand_id: int, days: list[dt.date], totals: list[str]) -> dict[dt.date, dict]:
    """Rows, null share per column and the `totals` (sums of cache columns, "rows" = row count) of
    every day of one cache and brand, in one SQL aggregation. A day without a cache file has 0 rows."""
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


# ---------------------------------------------------------------- daily caches: checks

def check_row_count(bench: list[dict], current: list[dict], rule: dict, **ctx) -> list[dict]:
    """Rows per current day vs the benchmark's daily mean; a day with no rows is critical."""
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
        """Null share per column over the profiles, weighted by their rows."""
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
            continue  # a new column is a schema change: the expectations report it
        increase = (rate - base[col]) * 100
        severity = _graded(increase, rule["warn_increase_pts"], rule.get("critical_increase_pts"))
        if severity:
            out.append(_alert("null_rate", severity, f"null share {rate:.1%} vs {base[col]:.1%} in the benchmark "
                              f"(+{increase:.1f} pts)", column=col, value=rate, benchmark=base[col],
                              threshold=rule["warn_increase_pts"], **ctx))
    return out


def check_daily_totals(profiles: dict, current_days: list[dt.date], rule: dict, metrics: list[str], **ctx) -> list[dict]:
    """Each current day vs the median +- n * MAD of the `window_days` days before it."""
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
    """The pandera suite, restricted to `columns` (the used ones) when given."""
    module = getattr(expectations, suite)
    result = (module.validate(df) if columns is None
              else validate_dataframe(df, [c for c in module.COLUMNS if c.name in columns]))
    return [_alert("expectations", rule["severity"], f"{suite}: {i.message}", column=i.column, **ctx)
            for i in result.issues]


def benchmark_days(cutoffs: list[str]) -> list[dt.date]:
    """The days the training features were built from: (first cutoff - lookback, last cutoff]."""
    dates = sorted(dt.date.fromisoformat(str(c)) for c in cutoffs)
    return _days(dates[0] - dt.timedelta(days=LOOKBACK_DAYS - 1), dates[-1])


def source_alerts(brand_id: int | str, as_of: str | dt.date, bench_days: list[dt.date],
                config: dict | None = None) -> pd.DataFrame:
    """Every daily-cache check for `brand_id` ("basel" = every brand with a bet on `as_of`): the last
    current_window_days up to `as_of` vs the benchmark days. Local caches only (daily_cache.top_up first)."""
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
            # A cache whose data is only complete from valid_from (payments: completed deposits from
            # March 2026) is benchmarked on those days only.
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
            if any(p["rows"] for p in current):  # every row of the current days, no sample
                rows = daily_cache.read(cache, brand, start=current_days[0], end=current_days[-1])
                alerts += check_expectations(rows, cache, cfg["expectations"], **ctx)
    return pd.DataFrame([{"stage": "source", **a} for a in alerts], columns=ALERT_COLUMNS)


# ---------------------------------------------------------------- features

def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index of `actual` vs `expected`: quantile bins of the benchmark, or one
    bin per value for a feature with few distinct values (flags, counts)."""
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
    """The feature snapshot being scored vs the training dataset, brand by brand: expectations,
    player count and PSI per feature. With `model_features`, only the features the models use are
    checked for drift: drift in an unused one cannot hurt the scores."""
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
        if len(drifting) >= drift["critical_n_features"]:  # many small shifts together are critical too
            alerts.append(_alert("feature_drift", "critical", f"{len(drifting)} features with PSI > "
                                 f"{drift['warn_psi']}: {drifting}", value=len(drifting),
                                 threshold=drift["critical_n_features"], **ctx))
    return pd.DataFrame([{"stage": "features", **a} for a in alerts], columns=ALERT_COLUMNS)


# ---------------------------------------------------------------- report

MAX_ALERT_TAGS = 50


def _where(a) -> str:
    """table.column (brand X / operator) of one alert."""
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


def report(alerts: pd.DataFrame, stage: str, brand: str, as_of: str, stop_on_critical: bool = True) -> Path:
    """Save the alert list (CSV + MLflow run), print it, and raise CriticalAlert if any is critical."""
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
        # How many of each kind, e.g. n_warn_null_rate, n_critical_row_count.
        for (severity, check), n in alerts.groupby(["severity", "check"]).size().items():
            mlflow.log_metric(f"n_{severity}_{check}", int(n))
        # Every alert readable without opening a file: the run description (Overview tab) is a
        # table of them, one tag per alert (alert_01, ...) and an MLflow table (Artifacts tab).
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
        # The features the latest trained models use: only those can raise a critical drift alert.
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
