"""Backtest (walk-forward): the training, validation and test periods are moved forward a few days at
a time, so the models are tested on several later periods instead of one. It is the temporal
robustness check of the training spec (point 8): if the optimisation gain does not hold on every
scenario, the simpler (reference) configuration is the safer choice.

    .venv/bin/python src/models/backtest.py --as-of 2026-10-04                     # brand 64, every 3 days
    .venv/bin/python src/models/backtest.py --as-of 2026-10-04 --step-days 4 --train-window 2 --n-trials 30
    .venv/bin/python src/models/backtest.py --as-of 2026-10-04 --seeds 42 7 2026   # each scenario x 3 seeds

Cutoffs go from MIN_CUTOFF to AS_OF - 60 days (the last one whose 60-day label fits), one every
`step_days`. Scenario k trains on the `train_window` cutoffs before cutoff k (train/validation by
player, as always) and tests on cutoff k:

    train c1, c2 -> test c3      train c2, c3 -> test c4      ...

In each scenario three runs are trained exactly like the normal training (train.py): the reference
LightGBM (optimisation step 1), the optimised LightGBM (every step) and Cox PH (with the optimised
LightGBM's features). Features and labels are built once for every cutoff (from the daily tables).

Every scenario is repeated with each seed (train/validation split, LightGBM sampling, Optuna), which
separates two kinds of variation: between periods (does the model hold over time?) and between seeds
(is a result luck?).

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
from build_features import HISTORY_START, LABEL_HORIZON_DAYS, data_available_through, MIN_CUTOFF, brand_label, parse_brand_id
from build_labels import build_training_labels
from build_training_features import build_training_features
from create_dataset import create_dataset
from train import run_info, train

DEFAULT_STEP_DAYS = 3
DEFAULT_TRAIN_WINDOW = 2
DEFAULT_SEEDS = (42, 7, 2026)
DEFAULT_SEED_MODE = "per_window"
BASE_SEED = 42  # per_window mode: the window seeds are drawn from this one, so a backtest is reproducible
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
        train_window: int = DEFAULT_TRAIN_WINDOW, n_trials: int | None = None, seeds=DEFAULT_SEEDS,
        seed_mode: str = DEFAULT_SEED_MODE) -> Path:
    """seed_mode "per_window": each window gets its own random seed (one training per window, fast;
    the spread between windows then mixes period and seed). "grid": each window is trained with every
    seed in `seeds` (slower; separates the variation between periods from the one between seeds)."""
    start = time.time()
    if seed_mode not in ("per_window", "grid"):
        raise ValueError(f"seed_mode must be 'per_window' or 'grid', not {seed_mode!r}")
    as_of_date = dt.date.fromisoformat(as_of)
    import daily_tables  # top up the daily tables with the new complete days first
    daily_tables.build(HISTORY_START, as_of_date, verbose=False)
    available = data_available_through()
    if as_of_date > available:
        raise ValueError(f"AS_OF {as_of} is after the last complete data day ({available})")
    cutoffs = backtest_cutoffs(as_of_date, step_days)
    if len(cutoffs) <= train_window:
        raise ValueError(f"{len(cutoffs)} cutoffs ({cutoffs}): not enough for a {train_window}-cutoff train window "
                         "plus a test cutoff; lower --step-days or --train-window")
    backtest_id = f"backtest_brand{brand_id}_{as_of}_step{step_days}_{int(start)}"
    n_scenarios = len(cutoffs) - train_window
    if seed_mode == "per_window":
        window_seeds = [[int(x)] for x in np.random.default_rng(BASE_SEED).integers(0, 10_000, n_scenarios)]
    else:
        window_seeds = [list(seeds)] * n_scenarios
    seeds = sorted({x for ws in window_seeds for x in ws})
    print(f"{backtest_id}: cutoffs {cutoffs}, {n_scenarios} scenario(s), seed mode {seed_mode}: {window_seeds}")

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
        for seed in window_seeds[number - 1]:
            tags = {"backtest_id": backtest_id, "scenario": number, "seed": seed, "test_cutoff": test_cutoff,
                    "train_cutoffs": ",".join(train_cutoffs)}
            print(f"\n== scenario {number}/{n_scenarios}, seed {seed}: train {train_cutoffs} -> test {test_cutoff}")
            runs = {
                "lgbm_reference": train("lightgbm_classifier", path, reference=True, tags=tags, seed=seed),
                "lgbm_optimised": train("lightgbm_classifier", path, n_trials=n_trials, tags=tags, seed=seed),
            }
            runs["cox"] = train("cox_ph", path, features_from_run=runs["lgbm_optimised"], tags=tags, seed=seed)
            row = {"scenario": number, "seed": seed, "train_cutoffs": ", ".join(train_cutoffs), "test_cutoff": test_cutoff,
                   "test_rows": int((subset["cutoff_date"].astype(str) == test_cutoff).sum()),
                   "test_churn_rate": float(subset.loc[subset["cutoff_date"].astype(str) == test_cutoff, "event_60d"].mean())}
            for model, metric_names in METRICS.items():
                info = run_info(runs[model])
                for m in metric_names:
                    row[f"{model}__{_short(m)}"] = info["metrics"].get(m, np.nan)
                row[f"{model}__run"] = info["run_name"]
            rows.append(row)

    table = pd.DataFrame(rows)
    doc = _report(table, backtest_id, brand_id, data, step_days, train_window, as_of, time.time() - start, seeds,
                  seed_mode)
    print(f"\nsaved {os.path.relpath(doc, PROJECT_ROOT)} | {(time.time() - start) / 60:.1f} min")
    return doc


def _short(metric: str) -> str:
    return metric.removeprefix("test_").removeprefix("calibrated_")


def _report(table: pd.DataFrame, backtest_id: str, brand_id, data, step_days, train_window, as_of, seconds,
            seeds, seed_mode: str = "grid") -> Path:
    grid = seed_mode == "grid" and len(seeds) > 1  # only then is there a spread between seeds per window
    label = brand_label(data["brandId"].unique())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = OUTPUT_DIR / f"{backtest_id}.csv"
    figure_path = OUTPUT_DIR / f"{backtest_id}.png"
    table.to_csv(table_path, index=False)

    metric_cols = [f"{m}__{_short(x)}" for m, xs in METRICS.items() for x in xs]
    by_scenario = table.groupby(["scenario", "test_cutoff"])[metric_cols].agg(["mean", "std"])
    mean = by_scenario.xs("mean", axis=1, level=1).reset_index()
    seed_std = by_scenario.xs("std", axis=1, level=1).reset_index()

    gain = mean["lgbm_optimised__roc_auc"] - mean["lgbm_reference__roc_auc"]
    ll_gain = mean["lgbm_reference__log_loss"] - mean["lgbm_optimised__log_loss"]  # positive = optimised better
    holds = bool((gain >= 0).all() and (ll_gain >= 0).all())

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    x = mean["test_cutoff"]
    for ax, (metric, title, models) in zip(axes, [
        ("roc_auc", "LightGBM AUC (higher is better)", ["lgbm_reference", "lgbm_optimised"]),
        ("log_loss", "LightGBM log-loss (lower is better)", ["lgbm_reference", "lgbm_optimised"]),
        ("ece", "LightGBM ECE (lower is better)", ["lgbm_reference", "lgbm_optimised"]),
        ("c_index", "Cox PH c-index (higher is better)", ["cox"]),
    ]):
        for model in models:
            col = f"{model}__{metric}"
            ax.errorbar(x, mean[col], yerr=seed_std[col].fillna(0), marker="o", capsize=4, label=model.replace("lgbm_", ""))
        ax.set_title(title); ax.set_xlabel("test cutoff"); ax.tick_params(axis="x", rotation=30); ax.legend()
    fig.suptitle(f"Backtest {label}: " + (f"mean over {len(seeds)} seeds per scenario, bars = spread between seeds"
                                          if grid else "one seed per window (period and seed vary together)"))
    fig.tight_layout(); fig.savefig(figure_path, dpi=120); plt.close(fig)

    def spread(col):
        """Mean, variation between periods (std of the scenario means) and between seeds (mean std)."""
        between_periods = mean[col].std(ddof=1) if len(mean) > 1 else np.nan
        if not grid:
            return f"{mean[col].mean():.4f} / ± {between_periods:.4f}"
        return f"{mean[col].mean():.4f} / ± {between_periods:.4f} / ± {seed_std[col].mean():.4f}"

    header = ["#", "test", "test rows", "churn", "AUC ref", "AUC opt", "log-loss ref", "log-loss opt", "ECE ref", "ECE opt", "C-index"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    first = table.drop_duplicates("scenario").set_index("scenario")
    for r in mean.itertuples(index=False):
        sc = r.scenario
        cells = [str(sc), r.test_cutoff, f"{first.loc[sc, 'test_rows']:,}", f"{first.loc[sc, 'test_churn_rate']:.1%}"]
        for col in ["lgbm_reference__roc_auc", "lgbm_optimised__roc_auc", "lgbm_reference__log_loss",
                    "lgbm_optimised__log_loss", "lgbm_reference__ece", "lgbm_optimised__ece", "cox__c_index"]:
            sd = seed_std.loc[seed_std["scenario"] == sc, col].iloc[0]
            cells.append(f"{getattr(r, col):.4f}" + (f" ± {sd:.4f}" if grid else ""))
        lines.append("| " + " | ".join(cells) + " |")
    doc = Path(str(DOC_PATH).format(brand=label))
    spread_head = "mean / ± periods / ± seeds" if grid else "mean / ± between windows (period and seed)"
    summary_intro = (
        "Mean over the scenarios, then the two kinds of variation: **between periods** (standard deviation of the "
        "scenario means: does the model hold over time?) and **between seeds** (mean standard deviation across seeds "
        "within a scenario: how much is luck?)." if grid else
        "Mean over the windows and its standard deviation. Each window has its own seed, so this spread mixes the "
        "change of period and the change of seed: if it is small, the model is robust to both. To tell them apart, "
        "run with `SEED_MODE=grid`.")

    text = f"""# Backtest (walk-forward), brandId {label}

The training, validation and test periods move forward {step_days} days at a time: each scenario trains on the previous {train_window} cutoffs (train / validation split by player) and tests on the next one. Three models per scenario, trained exactly as in normal training: the **reference** LightGBM (optimisation step 1), the **optimised** LightGBM (every optimisation step) and Cox PH (with the optimised LightGBM's features). Seed mode **{seed_mode}**: {"each scenario is repeated with every seed" if grid else "each window has its own random seed"} ({", ".join(map(str, seeds))}); the seed changes the train / validation split, the LightGBM row and column sampling and the Optuna search. This is the temporal robustness check of the training spec (point 8).

Backtest id `{backtest_id}` (data as of {as_of}, {len(mean)} scenarios x {len(seeds)} seeds = {len(table)} trainings per model, {seconds / 60:.1f} min). Every run is in MLflow with the tags `backtest_id`, `scenario` and `seed`; the summary run (experiment `{MLFLOW_EXPERIMENT}`) has one chart per metric across scenarios.

## Results per scenario (test month; mean ± spread between seeds)

{chr(10).join(lines)}

![Backtest]({os.path.relpath(figure_path, doc.parent)})

## Summary

{summary_intro}

| Metric | Reference: {spread_head} | Optimised: {spread_head} |
|---|---|---|
| LightGBM AUC | {spread('lgbm_reference__roc_auc')} | {spread('lgbm_optimised__roc_auc')} |
| LightGBM log-loss | {spread('lgbm_reference__log_loss')} | {spread('lgbm_optimised__log_loss')} |
| LightGBM ECE | {spread('lgbm_reference__ece')} | {spread('lgbm_optimised__ece')} |
| Cox PH c-index | | {spread('cox__c_index')} |

**Does the optimisation gain hold on every scenario?** {"Yes" if holds else "No"}: averaged over the seeds, the optimised model has a higher or equal AUC in {int((gain >= 0).sum())} of {len(mean)} scenarios (mean change {gain.mean():+.4f}) and a lower or equal log-loss in {int((ll_gain >= 0).sum())} of {len(mean)} (mean change {-ll_gain.mean():+.4f}). {"The optimised configuration can stay." if holds else "By the spec's rule (point 8), the simpler reference configuration is the safer choice until the gain holds on every scenario."}

## How to read it

- A difference between two models smaller than the spread between seeds is luck, not a real difference; one smaller than the spread between periods may not hold next month.
- The scenarios are **not independent**: the 60-day label windows of nearby cutoffs overlap and the players repeat, so the results look more stable than they would be on truly separate periods. A fully out-of-time test needs training cutoffs at least 60 days before the test cutoff, which the history does not allow yet. Each new week of data adds a cutoff, so this backtest gains weight with time.
- The winsorisation caps come from the configuration of the main pipeline (they may have seen some test cutoffs' features, not their labels).

Generated by `src/models/backtest.py` (`make backtest`). Table (one row per scenario and seed): `{os.path.relpath(table_path, PROJECT_ROOT)}`.
"""
    doc.write_text(text, encoding="utf-8")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=backtest_id):
        mlflow.set_tags({"brand_id": label, "backtest_id": backtest_id})
        mlflow.log_params({"as_of": as_of, "step_days": step_days, "train_window": train_window, "seed_mode": seed_mode,
                           "n_scenarios": len(mean), "seeds": seeds, "test_cutoffs": list(mean["test_cutoff"])})
        # One point per scenario (step = scenario number): MLflow draws each metric as a line chart
        # across the windows in Model metrics. The mean over seeds, and each seed on its own.
        for r_mean, r_std in zip(mean.itertuples(index=False), seed_std.itertuples(index=False)):
            step = int(r_mean.scenario)
            for col in metric_cols:
                name = col.replace("__", "_")
                mlflow.log_metric(name, float(getattr(r_mean, col)), step=step)
                if grid:
                    mlflow.log_metric(f"{name}_seed_std", float(getattr(r_std, col)), step=step)
        for r in table.itertuples(index=False):
            for col in metric_cols:
                mlflow.log_metric(f"{col.replace('__', '_')}_seed{r.seed}", float(getattr(r, col)), step=int(r.scenario))
        for col in metric_cols:  # summary numbers (no step)
            name = col.replace("__", "_")
            mlflow.log_metric(f"summary_{name}_mean", float(mean[col].mean()))
            if len(mean) > 1:
                mlflow.log_metric(f"summary_{name}_std_between_periods", float(mean[col].std(ddof=1)))
            if grid:
                mlflow.log_metric(f"summary_{name}_std_between_seeds", float(seed_std[col].mean()))
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
    parser.add_argument("--seed-mode", choices=["per_window", "grid"], default=DEFAULT_SEED_MODE,
                        help="per_window: one random seed per window (fast); grid: every window x every --seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS), help="grid mode: the seeds")
    args = parser.parse_args()
    run(args.as_of, args.brand_id, args.step_days, args.train_window, args.n_trials, args.seeds, args.seed_mode)
