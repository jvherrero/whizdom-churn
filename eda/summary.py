"""EDA stage 4: a training snapshot next to its benchmark (T7).

    .venv/bin/python eda/summary.py                                    # latest dataset vs the one before it
    .venv/bin/python eda/summary.py --dataset data/processed/train_dataset_64_...parquet --benchmark <path>
    make eda-summary [DATASET=path] [BENCHMARK=path]

One table, one row per statistic: metric, current, benchmark, delta, status. The benchmark is the
previous training dataset of the same brand (the previous run), or the dataset given with --benchmark
(for example a frozen training snapshot). Statistics:
    snapshot      rows, players, cutoffs, rows per cutoff
    labels        60-day churn rate, churn within 7 / 14 / 30 days, event and censoring rates
    durations     distribution of the churn day (duration_days) of the observed events
    features      null share, median (real units) and PSI against the benchmark, per feature
Status uses the thresholds of configs/eda_alerts.yaml (row count, null rate, PSI); a label rate that
moves more than RATE_WARN_PTS points is a warning. Outputs: docs/eda_summary_brand{id}.md, the table
and figures in data/03_output/eda_summary/brand{id}/, and the table in the dataset's MLflow run.
The pipeline runs it after building the dataset.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
from alerts import load_config, psi  # noqa: E402
from build_features import FEATURES, _unscale, brand_label  # noqa: E402
from build_survival_dataset import MLFLOW_EXPERIMENT, PROCESSED_DIR, WITHIN_COLUMNS  # noqa: E402

DOC_PATH = PROJECT_ROOT / "docs/eda_summary_brand{brand}.md"
OUTPUT_DIR = PROJECT_ROOT / "data/03_output/eda_summary/brand{brand}"
RATE_WARN_PTS = 5.0  # a label rate (churn, censoring) that moves more than this many points
DURATION_QUANTILES = (0.5, 0.75, 0.9)


def profile(df: pd.DataFrame) -> dict[str, float]:
    """Every statistic of one dataset, by name."""
    events = df["event_observed"] == 1
    durations = df.loc[events, "duration_days"]
    features = [f for f in FEATURES if f in df]  # an older dataset may lack a newer feature
    real = _unscale(df[features])
    out = {
        "rows": len(df), "players": len(df[["tenant_id", "player_id"]].drop_duplicates()),
        "cutoffs": df["cutoff_date"].nunique(), "rows_per_cutoff": len(df) / df["cutoff_date"].nunique(),
        "churn_rate_60d": df["event_60d"].mean(),
        **{f"{c}_rate": df[c].mean() for c in WITHIN_COLUMNS if c in df and df[c].notna().any()},
        "event_rate": events.mean(), "censoring_rate": 1 - events.mean(),
        "duration_day0_share": (durations == 0).mean() if len(durations) else np.nan,
        **{f"duration_p{int(q * 100)}_days": durations.quantile(q) for q in DURATION_QUANTILES},
        "duration_max_days": durations.max(),
    }
    for f in features:
        out[f"null_share__{f}"] = df[f].isna().mean()
        out[f"median__{f}"] = real[f].median()
    return out


def _status(metric: str, current: float, benchmark: float, cfg: dict) -> str:
    if pd.isna(benchmark):
        return "new"
    if metric in ("rows", "players", "rows_per_cutoff"):
        change = abs(current / benchmark - 1) if benchmark else np.inf
        rule = cfg["row_count"]
        return "critical" if change > rule["critical_relative_change"] else "warn" if change > rule["warn_relative_change"] else "ok"
    if metric.startswith("null_share__"):
        rise = (current - benchmark) * 100
        rule = cfg["null_rate"]
        return "critical" if rise > rule["critical_increase_pts"] else "warn" if rise > rule["warn_increase_pts"] else "ok"
    if metric.startswith("psi__"):
        rule = cfg["feature_drift"]
        return "critical" if current > rule["critical_psi"] else "warn" if current > rule["warn_psi"] else "ok"
    if "_rate" in metric:  # churn, churn within h days, event and censoring rates
        return "warn" if abs(current - benchmark) * 100 > RATE_WARN_PTS else "ok"
    return "info"  # durations, medians, cutoffs: reported, not judged


def summary_table(current: pd.DataFrame, benchmark: pd.DataFrame | None) -> pd.DataFrame:
    cfg = load_config()
    cur = profile(current)
    ben = profile(benchmark) if benchmark is not None else {}
    rows = [{"metric": m, "current": v, "benchmark": ben.get(m, np.nan)} for m, v in cur.items()]
    if benchmark is not None:
        rows += [{"metric": f"psi__{f}", "current": psi(benchmark[f], current[f], cfg["feature_drift"]["bins"]),
                  "benchmark": 0.0} for f in FEATURES
                 if f in current and f in benchmark and current[f].notna().any() and benchmark[f].notna().any()]
    table = pd.DataFrame(rows)
    table["delta"] = table["current"] - table["benchmark"]
    table["status"] = [_status(r.metric, r.current, r.benchmark, cfg) for r in table.itertuples()]
    return table


def by_cutoff(df: pd.DataFrame) -> pd.DataFrame:
    """Rows, churn, censoring and median churn day per cutoff: the censoring grows for recent cutoffs."""
    g = df.groupby("cutoff_date")
    out = pd.DataFrame({
        "rows": g.size(), "churn_rate_60d": g["event_60d"].mean(),
        **{f"{c}_rate": g[c].mean() for c in WITHIN_COLUMNS if c in df},
        "censoring_rate": 1 - g["event_observed"].mean(),
        "median_churn_day": df[df["event_observed"] == 1].groupby("cutoff_date")["duration_days"].median(),
    })
    return out.reset_index()


def _figures(df: pd.DataFrame, cut: pd.DataFrame, fig_dir: Path) -> dict[str, Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = {"durations": fig_dir / "duration_distribution.png", "censoring": fig_dir / "censoring_by_cutoff.png"}
    d = df.loc[df["event_observed"] == 1, "duration_days"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(d, bins=np.arange(0, d.max() + 8, 7), color="#1e6b57")
    ax.set_yscale("log")
    ax.set_xlabel("churn day after the cutoff (duration_days, observed events, 7-day bins)")
    ax.set_ylabel("players (log scale)")
    ax.set_title(f"Duration distribution: {(d == 0).mean():.0%} churn on day 0, median {d.median():.0f} days")
    fig.tight_layout(); fig.savefig(paths["durations"], dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = cut["cutoff_date"].astype(str)
    ax.bar(x, cut["censoring_rate"], color="#9aa6a1", label="censored (churn day not known yet)")
    ax.plot(x, cut["churn_rate_60d"], color="#1e6b57", marker="o", label="60-day churn rate")
    ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=45); ax.legend(fontsize=8)
    ax.set_title("Censoring and churn by cutoff")
    fig.tight_layout(); fig.savefig(paths["censoring"], dpi=120); plt.close(fig)
    return paths


def _fmt(v) -> str:
    if pd.isna(v):
        return ""
    return f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:.3f}"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_fmt(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |"
              for r in df.itertuples(index=False)]
    return "\n".join(lines)


def run(dataset_path: str | Path | None = None, benchmark_path: str | Path | None = None) -> Path:
    files = sorted(PROCESSED_DIR.glob("train_dataset_*.parquet"), key=lambda p: p.stat().st_mtime)
    dataset_path = Path(dataset_path).resolve() if dataset_path else files[-1].resolve()
    current = pd.read_parquet(dataset_path)
    label = brand_label(current["brandId"].unique())
    if benchmark_path is None:  # the previous dataset of the same brand
        earlier = [p for p in files if p.resolve() != dataset_path and p.stat().st_mtime < dataset_path.stat().st_mtime
                   and p.name.startswith(f"train_dataset_{label}_")]
        benchmark_path = earlier[-1] if earlier else None
    benchmark = pd.read_parquet(benchmark_path) if benchmark_path else None
    table = summary_table(current, benchmark)
    cut = by_cutoff(current)

    out_dir = Path(str(OUTPUT_DIR).format(brand=label))
    doc = Path(str(DOC_PATH).format(brand=label))
    figs = _figures(current, cut, out_dir)
    table.to_csv(out_dir / "summary.csv", index=False)
    cut.to_csv(out_dir / "by_cutoff.csv", index=False)

    counts = table["status"].value_counts()
    key = table[~table["metric"].str.contains("__")]
    flagged = table[table["metric"].str.contains("__") & table["status"].isin(["warn", "critical"])]
    feats = pd.DataFrame({
        "feature": FEATURES,
        "null share": [table.set_index("metric")["current"].get(f"null_share__{f}") for f in FEATURES],
        "null share, benchmark": [table.set_index("metric")["benchmark"].get(f"null_share__{f}") for f in FEATURES],
        "median": [table.set_index("metric")["current"].get(f"median__{f}") for f in FEATURES],
        "PSI": [table.set_index("metric")["current"].get(f"psi__{f}", np.nan) for f in FEATURES],
    })
    rel = lambda p: os.path.relpath(p, doc.parent)
    bench_name = os.path.relpath(benchmark_path, PROJECT_ROOT) if benchmark_path else "none (first dataset)"
    text = f"""# EDA Summary vs Benchmark: Training Snapshot (brandId {label})

EDA stage 4 of the delivery plan: every statistic of the training snapshot next to its benchmark.

| | |
|---|---|
| Snapshot | `{os.path.relpath(dataset_path, PROJECT_ROOT)}` |
| Benchmark | `{bench_name}` (the previous dataset of the brand) |
| Status | {int(counts.get('critical', 0))} critical, {int(counts.get('warn', 0))} warn, {int(counts.get('ok', 0))} ok |

Thresholds: row counts and null shares as in `configs/eda_alerts.yaml` (row count +-20% warn, +-50% critical; null share +5 / +20 points; PSI 0.10 / 0.25); a label rate that moves more than {RATE_WARN_PTS:.0f} points is a warning. Medians are in real units (days, EUR, counts).

## 1. Snapshot, Labels and Durations

{_md(key[["metric", "current", "benchmark", "delta", "status"]])}

## 2. Censoring and Duration Distribution

`duration_days` is the day the churn starts after the cutoff (0 = no bet in the next 60 days); a player is censored when the 60-day silence that would confirm it does not fit in the data yet. Censoring grows for recent cutoffs, which have less data after them.

![Duration distribution]({rel(figs['durations'])})

![Censoring by cutoff]({rel(figs['censoring'])})

{_md(cut)}

## 3. Features

{"Flagged against the benchmark:" if len(flagged) else "No feature flagged against the benchmark."}

{_md(flagged[["metric", "current", "benchmark", "delta", "status"]]) if len(flagged) else ""}

{_md(feats)}

Generated by `eda/summary.py` (`make eda-summary`; the pipeline runs it after building the dataset). Table: `{os.path.relpath(out_dir / 'summary.csv', PROJECT_ROOT)}`.
"""
    doc.write_text(text, encoding="utf-8")

    # Into the dataset's own MLflow run, when there is one.
    runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT],
                              filter_string=f"params.output_path = '{os.path.relpath(dataset_path, PROJECT_ROOT)}'")
    if not runs.empty:
        with mlflow.start_run(run_id=runs.iloc[0].run_id):
            for p in (out_dir / "summary.csv", out_dir / "by_cutoff.csv", *figs.values(), doc):
                mlflow.log_artifact(str(p), artifact_path="eda_summary")
            mlflow.log_metrics({f"eda_summary_n_{s}": int(counts.get(s, 0)) for s in ("critical", "warn", "ok")})
    print(f"EDA summary: {int(counts.get('critical', 0))} critical, {int(counts.get('warn', 0))} warn "
          f"vs {bench_name} -> {os.path.relpath(doc, PROJECT_ROOT)}")
    return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA stage 4: training snapshot vs benchmark.")
    parser.add_argument("--dataset", help="default: the latest train_dataset file")
    parser.add_argument("--benchmark", help="default: the previous dataset of the same brand")
    args = parser.parse_args()
    run(args.dataset, args.benchmark)
