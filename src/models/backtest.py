"""Backtest (walk-forward): the training, validation and test periods are moved forward a few days at
a time, so the models are tested on several later periods instead of one. It is the temporal
robustness check of the training spec (point 8): if the optimisation gain does not hold on every
scenario, the simpler (reference) configuration is the safer choice.

    .venv/bin/python src/models/backtest.py --as-of 2026-10-04                     # brand 64, every 3 days
    .venv/bin/python src/models/backtest.py --as-of 2026-10-04 --step-days 4 --train-window 2 --n-trials 30

Cutoffs go from MIN_CUTOFF to AS_OF - 60 days (the last one whose 60-day label fits), one every
`step_days`. Scenario k trains on the `train_window` cutoffs before cutoff k (train/validation by
player, as always) and tests on cutoff k:

    train c1, c2 -> test c3      train c2, c3 -> test c4      ...

In each scenario three runs are trained exactly like the normal training (train.py): the reference
LightGBM (optimisation step 1), the optimised LightGBM (every step) and Cox PH (with the optimised
LightGBM's features). Features and labels are built once for every cutoff (from the daily tables).

Output: docs/backtest_brand{id}.md, a figure and a table in data/03_output/backtest/, and the runs
in MLflow tagged with the backtest id and scenario.

Limits: the 60-day label windows of nearby scenarios overlap and the players repeat, so the
scenarios are not independent; the backtest gains weight as the history grows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import DATA_AVAILABLE_THROUGH, LABEL_HORIZON_DAYS, MIN_CUTOFF, brand_label, parse_brand_id
from build_labels import build_training_labels
from build_training_features import build_training_features
from create_dataset import create_dataset
from train import run_info, train

DEFAULT_STEP_DAYS = 3
DEFAULT_TRAIN_WINDOW = 2
SCENARIO_DIR = PROJECT_ROOT / "data/02_intermediate/backtest"
OUTPUT_DIR = PROJECT_ROOT / "data/03_output/backtest"
DOC_PATH = PROJECT_ROOT / "docs/backtest_brand{brand}.md"
MLFLOW_EXPERIMENT = "whizdom-churn-backtest"
METRICS = {  # what each scenario reports, from the training runs
    "lgbm_reference": ["test_calibrated_roc_auc", "test_calibrated_log_loss", "test_calibrated_ece"],
    "lgbm_optimised": ["test_calibrated_roc_auc", "test_calibrated_log_loss", "test_calibrated_ece"],
    "cox": ["test_c_index"],
}


def backtest_cutoffs(as_of: dt.date, step_days: int) -> list[str]:
    last = as_of - dt.timedelta(days=LABEL_HORIZON_DAYS)
    out, day = [], MIN_CUTOFF
    while day <= last:
        out.append(str(day))
        day += dt.timedelta(days=step_days)
    return out


def run(as_of: str, brand_id: int | str = 64, step_days: int = DEFAULT_STEP_DAYS,
        train_window: int = DEFAULT_TRAIN_WINDOW, n_trials: int | None = None) -> Path:
    start = time.time()
    as_of_date = dt.date.fromisoformat(as_of)
    if as_of_date > DATA_AVAILABLE_THROUGH:
        raise ValueError(f"AS_OF {as_of} is after DATA_AVAILABLE_THROUGH ({DATA_AVAILABLE_THROUGH})")
    cutoffs = backtest_cutoffs(as_of_date, step_days)
    if len(cutoffs) <= train_window:
        raise ValueError(f"{len(cutoffs)} cutoffs ({cutoffs}): not enough for a {train_window}-cutoff train window "
                         "plus a test cutoff; lower --step-days or --train-window")
    backtest_id = f"backtest_brand{brand_id}_{as_of}_step{step_days}_{int(start)}"
    print(f"{backtest_id}: cutoffs {cutoffs}, {len(cutoffs) - train_window} scenario(s)")

    # Features and labels of every cutoff, once (daily tables + caches make this fast).
    features_path = build_training_features(cutoffs, brand_id)
    build_training_labels(features_path, data_end=as_of)
    create_dataset(features_path)
    dataset_path = features_path.with_name(features_path.name.replace("train_features_base", "train_dataset"))
    data = pd.read_parquet(dataset_path)

    scenario_dir = SCENARIO_DIR / backtest_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(train_window, len(cutoffs)):
        train_cutoffs, test_cutoff = cutoffs[k - train_window:k], cutoffs[k]
        number = k - train_window + 1
        subset = data[data["cutoff_date"].astype(str).isin(train_cutoffs + [test_cutoff])]
        path = scenario_dir / f"scenario{number}_test{test_cutoff}.parquet"
        subset.to_parquet(path, index=False)
        tags = {"backtest_id": backtest_id, "scenario": number, "test_cutoff": test_cutoff,
                "train_cutoffs": ",".join(train_cutoffs)}
        print(f"\n== scenario {number}: train {train_cutoffs} -> test {test_cutoff} ({len(subset):,} rows)")
        runs = {
            "lgbm_reference": train("lightgbm_classifier", path, reference=True, tags=tags),
            "lgbm_optimised": train("lightgbm_classifier", path, n_trials=n_trials, tags=tags),
        }
        runs["cox"] = train("cox_ph", path, features_from_run=runs["lgbm_optimised"], tags=tags)
        row = {"scenario": number, "train_cutoffs": ", ".join(train_cutoffs), "test_cutoff": test_cutoff,
               "test_rows": int((subset["cutoff_date"].astype(str) == test_cutoff).sum()),
               "test_churn_rate": float(subset.loc[subset["cutoff_date"].astype(str) == test_cutoff, "event_60d"].mean())}
        for model, metric_names in METRICS.items():
            metrics = run_info(runs[model])["metrics"]
            for m in metric_names:
                row[f"{model}__{m.removeprefix('test_').removeprefix('calibrated_')}"] = metrics.get(m, np.nan)
            row[f"{model}__run"] = run_info(runs[model])["run_name"]
        rows.append(row)

    table = pd.DataFrame(rows)
    doc = _report(table, backtest_id, brand_id, data, step_days, train_window, as_of, time.time() - start)
    print(f"\nsaved {os.path.relpath(doc, PROJECT_ROOT)} | {(time.time() - start) / 60:.1f} min")
    return doc


def _report(table: pd.DataFrame, backtest_id: str, brand_id, data, step_days, train_window, as_of, seconds) -> Path:
    label = brand_label(data["brandId"].unique())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = OUTPUT_DIR / f"{backtest_id}.csv"
    figure_path = OUTPUT_DIR / f"{backtest_id}.png"
    table.to_csv(table_path, index=False)

    gain = table["lgbm_optimised__roc_auc"] - table["lgbm_reference__roc_auc"]
    ll_gain = table["lgbm_reference__log_loss"] - table["lgbm_optimised__log_loss"]  # positive = optimised better
    holds = bool((gain >= 0).all() and (ll_gain >= 0).all())

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    x = table["test_cutoff"]
    for ax, (metric, title, models) in zip(axes, [
        ("roc_auc", "LightGBM AUC (higher is better)", ["lgbm_reference", "lgbm_optimised"]),
        ("log_loss", "LightGBM log-loss (lower is better)", ["lgbm_reference", "lgbm_optimised"]),
        ("ece", "LightGBM ECE (lower is better)", ["lgbm_reference", "lgbm_optimised"]),
        ("c_index", "Cox PH c-index (higher is better)", ["cox"]),
    ]):
        for model in models:
            ax.plot(x, table[f"{model}__{metric}"], marker="o", label=model.replace("lgbm_", ""))
        ax.set_title(title); ax.set_xlabel("test cutoff"); ax.tick_params(axis="x", rotation=30); ax.legend()
    fig.suptitle(f"Backtest {label}: one point per scenario (train on the previous {train_window} cutoffs)")
    fig.tight_layout(); fig.savefig(figure_path, dpi=120); plt.close(fig)

    def stats(col):
        s = table[col]
        return f"{s.mean():.4f} ± {s.std(ddof=1):.4f} (min {s.min():.4f}, max {s.max():.4f})" if len(s) > 1 else f"{s.iloc[0]:.4f}"

    cols = ["scenario", "train_cutoffs", "test_cutoff", "test_rows", "test_churn_rate",
            "lgbm_reference__roc_auc", "lgbm_optimised__roc_auc", "lgbm_reference__log_loss", "lgbm_optimised__log_loss",
            "lgbm_reference__ece", "lgbm_optimised__ece", "cox__c_index"]
    header = ["#", "train cutoffs", "test", "test rows", "churn", "AUC ref", "AUC opt", "log-loss ref", "log-loss opt",
              "ECE ref", "ECE opt", "C-index"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in table[cols].itertuples(index=False):
        cells = [str(r[0]), r[1], r[2], f"{r[3]:,}", f"{r[4]:.1%}"] + [f"{v:.4f}" for v in r[5:]]
        lines.append("| " + " | ".join(cells) + " |")

    text = f"""# Backtest (walk-forward), brandId {label}

The training, validation and test periods move forward {step_days} days at a time: each scenario trains on the previous {train_window} cutoffs (train / validation split by player) and tests on the next one. Three models per scenario, trained exactly as in normal training: the **reference** LightGBM (optimisation step 1), the **optimised** LightGBM (every optimisation step) and Cox PH (with the optimised LightGBM's features). This is the temporal robustness check of the training spec (point 8).

Backtest id `{backtest_id}` (data as of {as_of}, {len(table)} scenarios, {seconds / 60:.1f} min). Every run is in MLflow with the tags `backtest_id` and `scenario`.

## Results per scenario

{chr(10).join(lines)}

![Backtest]({os.path.relpath(figure_path, Path(str(DOC_PATH).format(brand=label)).parent)})

## Summary (mean ± standard deviation over the scenarios)

| Metric | Reference | Optimised |
|---|---|---|
| LightGBM AUC | {stats('lgbm_reference__roc_auc')} | {stats('lgbm_optimised__roc_auc')} |
| LightGBM log-loss | {stats('lgbm_reference__log_loss')} | {stats('lgbm_optimised__log_loss')} |
| LightGBM ECE | {stats('lgbm_reference__ece')} | {stats('lgbm_optimised__ece')} |
| Cox PH c-index | | {stats('cox__c_index')} |

**Does the optimisation gain hold on every scenario?** {"Yes" if holds else "No"}: the optimised model has a higher or equal AUC in {int((gain >= 0).sum())} of {len(table)} scenarios (mean change {gain.mean():+.4f}) and a lower or equal log-loss in {int((ll_gain >= 0).sum())} of {len(table)} (mean change {-ll_gain.mean():+.4f}). {"The optimised configuration can stay." if holds else "By the spec's rule (point 8), the simpler reference configuration is the safer choice until the gain holds on every scenario."}

## How to read it

- The spread between scenarios (the ± above) is the variation of the metrics from one period to another: a difference between two models smaller than that is not a real difference.
- The scenarios are **not independent**: the 60-day label windows of nearby cutoffs overlap and the players repeat, so the results look more stable than they would be on truly separate periods. A fully out-of-time test needs training cutoffs at least 60 days before the test cutoff, which the history does not allow yet. Each new week of data adds a cutoff, so this backtest gains weight with time.
- The winsorisation caps come from the configuration of the main pipeline (they may have seen some test cutoffs' features, not their labels).

Generated by `src/models/backtest.py` (`make backtest`). Table: `{os.path.relpath(table_path, PROJECT_ROOT)}`.
"""
    doc = Path(str(DOC_PATH).format(brand=label))
    doc.write_text(text, encoding="utf-8")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=backtest_id):
        mlflow.set_tags({"brand_id": label, "backtest_id": backtest_id})
        mlflow.log_params({"as_of": as_of, "step_days": step_days, "train_window": train_window,
                           "n_scenarios": len(table), "test_cutoffs": list(table["test_cutoff"])})
        for model, names in METRICS.items():
            for m in names:
                col = f"{model}__{m.removeprefix('test_').removeprefix('calibrated_')}"
                mlflow.log_metric(f"{col}_mean", float(table[col].mean()))
                if len(table) > 1:
                    mlflow.log_metric(f"{col}_std", float(table[col].std(ddof=1)))
        mlflow.log_metric("optimisation_gain_holds", int(holds))
        mlflow.log_metric("execution_time_s", seconds)
        for path in (table_path, figure_path, doc):
            mlflow.log_artifact(str(path))
    return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the reference and optimised models.")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--brand-id", type=parse_brand_id, default=64)
    parser.add_argument("--step-days", type=int, default=DEFAULT_STEP_DAYS)
    parser.add_argument("--train-window", type=int, default=DEFAULT_TRAIN_WINDOW)
    parser.add_argument("--n-trials", type=int, help="Optuna trials per optimised run (default: the catalog's)")
    args = parser.parse_args()
    run(args.as_of, args.brand_id, args.step_days, args.train_window, args.n_trials)
