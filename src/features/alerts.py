"""
Data alerts: checks of the landing data and of the feature snapshot against a benchmark, with
the thresholds of configs/eda_alerts.yaml. 

    .venv/bin/python src/features/alerts.py --as-of 2026-10-04                  # landing + features
    .venv/bin/python src/features/alerts.py --as-of 2026-10-04 --stage landing --brand-id basel

Checks
    expectations     any pandera suite (eda/expectations/) failing
    null_rate        null share of a column up more than N points vs the benchmark
    row_count        rows per day outside +-N% of the benchmark's daily mean
    new_categories   a value never seen in the benchmark (e.g. a new country or tranType)
    daily_total      a daily total outside the rolling band (median +- n * MAD of the previous days)
    feature_drift    a feature whose distribution shifts (PSI) vs the training dataset

"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
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
from build_features import (
    ALL_BRANDS, BASELINE_FEATURES, HISTORY_START, LANDING_TABLES, LOOKBACK_DAYS,
    _date_range, _discover_brand_ids, _landing_paths, _resolve_operators_for_brand, _s3_duckdb,
    parse_brand_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "eda"))
import expectations  # noqa: E402  (pandera suites, eda/expectations/)

CONFIG_PATH = PROJECT_ROOT / "configs/eda_alerts.yaml"
PROFILE_DIR = PROJECT_ROOT / "data/02_intermediate/daily_profiles"
ALERTS_DIR = PROJECT_ROOT / "data/03_output/alerts"
PROFILE_VERSION = 1
MLFLOW_EXPERIMENT = "whizdom-churn-alerts"

# Daily totals per table (SQL over one day). `bonus` has no brandId: it is checked per operator.
TABLE_SPECS = {
    "bet": {"brand_filter": True, "totals": {
        "n_players": "COUNT(DISTINCT partyId)",
        "stake_local": "-SUM(CASE WHEN tranType = 'GAME_BET' AND rolledBack = False THEN amountReal ELSE 0 END)",
    }},
    "transaction": {"brand_filter": True, "totals": {
        "n_players": "COUNT(DISTINCT partyId)",
        "deposits_local": "SUM(CASE WHEN transactionType = 'DEPOSIT' AND status = 'COMPLETED' "
                          "THEN processedAmount ELSE 0 END)",
    }},
    "player": {"brand_filter": True, "totals": {"n_players": "COUNT(DISTINCT partyId)"}},
    "bonus": {"brand_filter": False, "totals": {
        "n_players": "COUNT(DISTINCT partyId)",
        "bonus_amount": "SUM(CASE WHEN status = 'ACTIVE' THEN amount ELSE 0 END)",
    }},
}
ALERT_COLUMNS = ["stage", "check", "severity", "operator", "brandId", "table", "column",
                 "value", "benchmark", "threshold", "message"]


class CriticalAlert(RuntimeError):
    """Raised when a run has at least one critical alert: the pipeline stops."""


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _alert(check, severity, message, **context) -> dict:
    return {"check": check, "severity": severity, "message": message, **context}


def _graded(value: float, warn: float, critical: float | None) -> str | None:
    """'critical' / 'warn' / None for a value that is bad when it is large."""
    if critical is not None and value > critical:
        return "critical"
    return "warn" if value > warn else None


# ---------------------------------------------------------------- landing: daily profiles

def daily_profile(con, operator: str, table: str, brand_id: int | None, day: dt.date,
                  categorical_columns: list[str], max_values: int, use_cache: bool = True) -> dict:
    """Rows, null share per column, daily totals and distinct categorical values of one table,
    one day, one brand (None = the whole operator), from SQL aggregations over every row."""
    scope = f"brand{brand_id}" if brand_id is not None else "operator"
    cache = PROFILE_DIR / operator / table / scope / f"{day}_v{PROFILE_VERSION}.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text())

    source = f"read_parquet({_landing_paths(operator, LANDING_TABLES[table], [day])}, union_by_name=True)"
    try:
        columns = con.sql(f"DESCRIBE SELECT * FROM {source}").df()["column_name"].tolist()
    except duckdb.Error:  # no file for that day
        profile = {"day": str(day), "rows": 0, "missing_folder": True, "null_rate": {}, "totals": {}, "categories": {}}
    else:
        where = f"WHERE brandId = {brand_id}" if brand_id is not None else ""
        totals = TABLE_SPECS[table]["totals"]
        selects = (["COUNT(*) AS n_rows"] + [f'COUNT("{c}") AS "nn__{c}"' for c in columns]
                   + [f'{expr} AS "total__{name}"' for name, expr in totals.items()])
        row = con.sql(f"SELECT {', '.join(selects)} FROM {source} {where}").df().iloc[0]
        n = int(row["n_rows"])
        categories = {}
        for col in [c for c in categorical_columns if c in columns]:
            values = con.sql(f'SELECT DISTINCT CAST("{col}" AS VARCHAR) AS v FROM {source} {where} '
                             f'LIMIT {max_values}').df()["v"].dropna()
            categories[col] = sorted(values.tolist())
        profile = {
            "day": str(day), "rows": n, "missing_folder": False,
            "null_rate": {c: (1 - int(row[f"nn__{c}"]) / n) if n else None for c in columns},
            "totals": {"rows": n, **{k: (None if pd.isna(row[f"total__{k}"]) else float(row[f"total__{k}"]))
                                     for k in totals}},
            "categories": categories,
        }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(profile))
    return profile


# ---------------------------------------------------------------- landing: checks

def check_row_count(bench: list[dict], current: list[dict], rule: dict, **ctx) -> list[dict]:
    base = np.mean([p["rows"] for p in bench]) if bench else 0
    out = []
    for p in current:
        if p["rows"] == 0:
            out.append(_alert("row_count", "critical", f"no rows on {p['day']}", value=0, benchmark=base,
                              threshold=rule["critical_relative_change"], **ctx))
            continue
        change = p["rows"] / base - 1 if base else np.inf
        severity = _graded(abs(change), rule["warn_relative_change"], rule.get("critical_relative_change"))
        if severity:
            out.append(_alert("row_count", severity, f"{p['day']}: {p['rows']:,} rows, {change:+.0%} vs the "
                              f"benchmark daily mean {base:,.0f}", value=p["rows"], benchmark=base,
                              threshold=rule["warn_relative_change"], **ctx))
    return out


def check_null_rate(bench: list[dict], current: list[dict], rule: dict, used: set[str] | None = None,
                    **ctx) -> list[dict]:
    """`used`: the columns the pipeline reads; a null jump elsewhere is capped at warn."""
    def pooled(profiles):
        rows = sum(p["rows"] for p in profiles)
        cols = {c for p in profiles for c in p["null_rate"]}
        return rows, {c: sum((p["null_rate"].get(c) or 0) * p["rows"] for p in profiles) / rows for c in cols} if rows else {}

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
        unused = used is not None and col not in used
        if severity:
            if unused:
                severity = "warn"
            out.append(_alert("null_rate", severity, f"null share {rate:.1%} vs {base[col]:.1%} in the benchmark "
                              f"(+{increase:.1f} pts)" + (" (not used by the pipeline)" if unused else ""), column=col, value=rate, benchmark=base[col],
                              threshold=rule["warn_increase_pts"], **ctx))
    return out


def check_new_categories(bench: list[dict], current: list[dict], rule: dict, **ctx) -> list[dict]:
    out = []
    for col in {c for p in current for c in p["categories"]}:
        seen = {v for p in bench for v in p["categories"].get(col, [])}
        new = sorted({v for p in current for v in p["categories"].get(col, [])} - seen)
        if new:
            out.append(_alert("new_categories", rule["severity"], f"{len(new)} value(s) never seen in the benchmark: "
                              f"{new[:10]}", column=col, value=len(new), benchmark=len(seen), **ctx))
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


def check_expectations(df: pd.DataFrame, suite: str, rule: dict, **ctx) -> list[dict]:
    result = getattr(expectations, suite).validate(df)
    return [_alert("expectations", rule["severity"], f"{suite}: {i.message}", column=i.column, **ctx)
            for i in result.issues]


def benchmark_days(cutoffs: list[str]) -> list[dt.date]:
    """The landing days the training features were built from: (first cutoff - lookback, last cutoff]."""
    dates = sorted(dt.date.fromisoformat(str(c)) for c in cutoffs)
    return _date_range(dates[0] - dt.timedelta(days=LOOKBACK_DAYS - 1), dates[-1])


def landing_alerts(brand_id: int | str, as_of: str | dt.date, bench_days: list[dt.date],
                   config: dict | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Every landing check for `brand_id` ("basel" = every brand): the last current_window_days up
    to `as_of` vs the benchmark days."""
    cfg = config or load_config()
    as_of = dt.date.fromisoformat(str(as_of))
    current_days = _date_range(as_of - dt.timedelta(days=cfg["current_window_days"] - 1), as_of)
    band_days = _date_range(current_days[0] - dt.timedelta(days=cfg["daily_total"]["window_days"]), current_days[-1])
    all_days = sorted(d for d in set(bench_days) | set(current_days) | set(band_days) if d >= HISTORY_START)
    pairs = (_discover_brand_ids(as_of) if brand_id == ALL_BRANDS
             else [(op, int(brand_id)) for op in _resolve_operators_for_brand(int(brand_id), as_of)])
    cats = cfg["new_categories"]["columns"]

    alerts, done = [], set()
    con = _s3_duckdb()
    try:
        for operator, brand in pairs:
            for table in cfg["tables"]:
                scope = brand if TABLE_SPECS[table]["brand_filter"] else None
                if (operator, table, scope) in done:
                    continue  # bonus: once per operator
                done.add((operator, table, scope))
                ctx = {"operator": operator, "brandId": scope if scope is not None else "all", "table": table}
                profiles = {d: daily_profile(con, operator, table, scope, d, cats.get(table, []),
                                             cfg["new_categories"]["max_values_per_column"], use_cache) for d in all_days}
                bench = [profiles[d] for d in bench_days if d in profiles]
                current = [profiles[d] for d in current_days]
                row_rule = {**cfg["row_count"], **cfg["row_count"].get("per_table", {}).get(table, {})}
                alerts += check_row_count(bench, current, row_rule, **ctx)
                used = cfg["null_rate"].get("used_columns", {}).get(table)
                alerts += check_null_rate(bench, current, cfg["null_rate"], set(used) if used else None, **ctx)
                alerts += check_new_categories(bench, current, cfg["new_categories"], **ctx)
                alerts += check_daily_totals(profiles, current_days, cfg["daily_total"],
                                             cfg["daily_total"]["totals"].get(table, []), **ctx)
                # Expectations on every row of the current days (no sample).
                where = f"WHERE brandId = {scope}" if scope is not None else ""
                paths = _landing_paths(operator, LANDING_TABLES[table], current_days)
                try:
                    rows = con.sql(f"SELECT * FROM read_parquet({paths}, union_by_name=True) {where}").df()
                except duckdb.Error:
                    rows = None  # no file: row_count already says it, as critical
                if rows is not None and len(rows):
                    # Some suites expect the operator column the profiling loader adds; add it only for those.
                    expects_operator = any(c.name == "operator" for c in getattr(expectations, table).COLUMNS)
                    if expects_operator and "operator" not in rows:
                        rows = rows.assign(operator=operator)
                    alerts += check_expectations(rows, table, cfg["expectations"], **ctx)
    finally:
        con.close()
    return pd.DataFrame([{"stage": "landing", **a} for a in alerts], columns=ALERT_COLUMNS)


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
    player count and PSI per feature. With `model_features`, only a feature the models use can be
    critical: drift in an unused one cannot hurt the scores, so it is capped at warn."""
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
        for feature in [f for f in BASELINE_FEATURES if f in snap and f in bench]:
            value = psi(bench[feature], snap[feature], drift["bins"])
            severity = _graded(value, drift["warn_psi"], drift.get("critical_psi"))
            unused = model_features is not None and feature not in model_features
            if severity:
                if unused:
                    severity = "warn"
                alerts.append(_alert("feature_drift", severity, f"PSI {value:.3f} vs the training dataset"
                                     + (" (not used by the models)" if unused else ""),
                                     column=feature, value=value, threshold=drift["warn_psi"], **ctx))
    return pd.DataFrame([{"stage": "features", **a} for a in alerts], columns=ALERT_COLUMNS)


# ---------------------------------------------------------------- report

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

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"timestamp_unix": timestamp_unix, "user": getpass.getuser(), "host": socket.gethostname(),
                         "source_script": "src/features/alerts.py", "brand_id": brand, "stage": stage})
        mlflow.log_params({"as_of": as_of, "config": os.path.relpath(CONFIG_PATH, PROJECT_ROOT)})
        mlflow.log_metrics({"n_alerts": len(alerts), "n_warn": n_warn, "n_critical": n_critical})
        mlflow.log_artifact(str(path))
        mlflow.log_artifact(str(CONFIG_PATH))

    print(f"alerts [{stage}] brand {brand} as of {as_of}: {n_critical} critical, {n_warn} warn "
          f"-> {os.path.relpath(path, PROJECT_ROOT)}")
    if len(alerts):
        with pd.option_context("display.max_colwidth", 90, "display.width", 200):
            print(alerts[["severity", "check", "table", "brandId", "column", "message"]].to_string(index=False))
    if n_critical and stop_on_critical:
        raise CriticalAlert(f"{n_critical} critical alert(s) in the {stage} checks, see {path.name}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Data alerts as of one date (thresholds: configs/eda_alerts.yaml).")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--brand-id", type=parse_brand_id, default=64, help="brandId, or 'basel' for every brand")
    parser.add_argument("--stage", choices=["landing", "features", "all"], default="all")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    from build_features import build_feature_store
    from run_pipeline import training_cutoffs

    label = str(args.brand_id)
    cutoffs = training_cutoffs(dt.date.fromisoformat(args.as_of))
    failed = False
    if args.stage in ("landing", "all"):
        alerts = landing_alerts(args.brand_id, args.as_of, benchmark_days(cutoffs), use_cache=not args.no_cache)
        failed |= _report_no_raise(alerts, "landing", label, args.as_of)
    if args.stage in ("features", "all"):
        dataset = max((PROJECT_ROOT / "data/processed").glob(f"train_dataset_{label}_*.parquet"),
                      key=lambda p: p.stat().st_mtime)
        snapshot = build_feature_store(args.as_of, args.brand_id)
        # The features the latest trained models use: only those can raise a critical drift alert.
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
        from train import latest_run, run_info
        try:
            used = {f for m in ("lightgbm_classifier", "cox_ph") for f in run_info(latest_run(m).run_id)["features"]}
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
