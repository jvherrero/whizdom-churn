"""Temporal robustness check (training spec, point 8): the optimisation is repeated on earlier windows
of the same monthly cutoffs. If its gain (optimised vs reference LightGBM) does not hold on every
window, the simpler reference configuration is the safer choice.

Window k takes the first k cutoffs of the training dataset and splits them like train.split_rows
(train = all but the last 3, validation = the 3rd last, test = the last 2). With 12 cutoffs and the
default 2 windows: train on months 1-8 (validation month 8) and test on 9-10, then train on 1-9 and
test on 10-11, as in the plan; the full dataset (test on 11-12) is the main training run.

    .venv/bin/python src/models/robustness.py                                      # latest dataset
    .venv/bin/python src/models/robustness.py --dataset-path data/processed/train_dataset_...parquet --n-windows 3
    .venv/bin/python src/models/robustness.py --seed-mode grid --seeds 42 7 2026   # each window x 3 seeds

The rolling-origin backtest of T10 (src/models/backtest.py) is a separate check: does the model predict.
"""

from __future__ import annotations

import argparse
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
from build_features import brand_label
from train import N_TEST_CUTOFFS, N_VALID_CUTOFFS, PROCESSED_DIR, run_info, train

DEFAULT_WINDOWS = 2
DEFAULT_SEEDS = (42, 7, 2026)
DEFAULT_SEED_MODE = "per_window"
BASE_SEED = 42  # per_window mode: the window seeds are drawn from this one, so a backtest is reproducible
SCENARIO_DIR = PROJECT_ROOT / "data/02_intermediate/robustness"
OUTPUT_DIR = PROJECT_ROOT / "data/03_output/robustness"
DOC_PATH = PROJECT_ROOT / "docs/temporal_robustness_brand{brand}.md"
MLFLOW_EXPERIMENT = "whizdom-churn-robustness"
METRICS = {  # what each window reports, from the training runs
    "lgbm_reference": ["test_calibrated_roc_auc", "test_calibrated_log_loss", "test_calibrated_ece"],
    "lgbm_optimised": ["test_calibrated_roc_auc", "test_calibrated_log_loss", "test_calibrated_ece"],
    "cox": ["test_c_index"],
}


def run(dataset_path: str | Path | None = None, n_windows: int = DEFAULT_WINDOWS, n_trials: int | None = None,
        seeds=DEFAULT_SEEDS, seed_mode: str = DEFAULT_SEED_MODE) -> Path:
    """seed_mode "per_window": each window gets its own random seed (one training per window, fast;
    the spread between windows then mixes period and seed). "grid": each window is trained with every
    seed in `seeds` (slower; separates the variation between periods from the one between seeds)."""
    start = time.time()
    if seed_mode not in ("per_window", "grid"):
        raise ValueError(f"seed_mode must be 'per_window' or 'grid', not {seed_mode!r}")
    if dataset_path is None:
        dataset_path = max(PROCESSED_DIR.glob("train_dataset_*.parquet"), key=lambda p: p.stat().st_mtime)
    data = pd.read_parquet(dataset_path)
    cutoffs = sorted(str(c) for c in data["cutoff_date"].unique())
    held_out = N_VALID_CUTOFFS + N_TEST_CUTOFFS
    if len(cutoffs) - n_windows <= held_out:
        raise ValueError(f"{len(cutoffs)} cutoffs: not enough for {n_windows} earlier window(s) with at least one "
                         f"train month, {N_VALID_CUTOFFS} validation and {N_TEST_CUTOFFS} test months")
    label = brand_label(data["brandId"].unique())
    backtest_id = f"robustness_brand{label}_{cutoffs[-1]}_{n_windows}windows_{int(start)}"
    if seed_mode == "per_window":
        window_seeds = [[int(x)] for x in np.random.default_rng(BASE_SEED).integers(0, 10_000, n_windows)]
    else:
        window_seeds = [list(seeds)] * n_windows
    seeds = sorted({x for ws in window_seeds for x in ws})
    print(f"{backtest_id}: {n_windows} window(s), seed mode {seed_mode}: {window_seeds}")

    scenario_dir = SCENARIO_DIR / backtest_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, k in enumerate(range(len(cutoffs) - n_windows, len(cutoffs)), 1):
        window = cutoffs[:k]
        train_cutoffs, test_cutoffs = window[:-held_out], window[-N_TEST_CUTOFFS:]
        subset = data[data["cutoff_date"].astype(str).isin(window)]
        test_rows = subset["cutoff_date"].astype(str).isin(test_cutoffs)
        path = scenario_dir / f"window{number}_test{test_cutoffs[0]}.parquet"
        subset.to_parquet(path, index=False)
        for seed in window_seeds[number - 1]:
            tags = {"backtest_id": backtest_id, "scenario": number, "seed": seed, "test_cutoffs": ",".join(test_cutoffs),
                    "train_cutoffs": ",".join(train_cutoffs)}
            print(f"\n== window {number}/{n_windows}, seed {seed}: train {train_cutoffs[0]}..{train_cutoffs[-1]}, "
                  f"validation {window[-held_out]} -> test {test_cutoffs}")
            runs = {
                "lgbm_reference": train("lightgbm_classifier", path, reference=True, tags=tags, seed=seed),
                "lgbm_optimised": train("lightgbm_classifier", path, n_trials=n_trials, tags=tags, seed=seed),
            }
            runs["cox"] = train("cox_ph", path, features_from_run=runs["lgbm_optimised"], tags=tags, seed=seed)
            row = {"scenario": number, "seed": seed, "train_cutoffs": f"{train_cutoffs[0]} .. {train_cutoffs[-1]}",
                   "test_cutoff": ", ".join(test_cutoffs), "test_rows": int(test_rows.sum()),
                   "test_churn_rate": float(subset.loc[test_rows, "event_60d"].mean())}
            for model, metric_names in METRICS.items():
                info = run_info(runs[model])
                for m in metric_names:
                    row[f"{model}__{_short(m)}"] = info["metrics"].get(m, np.nan)
                row[f"{model}__run"] = info["run_name"]
            rows.append(row)

    table = pd.DataFrame(rows)
    doc = _report(table, backtest_id, label, os.path.relpath(Path(dataset_path).resolve(), PROJECT_ROOT),
                  time.time() - start, seeds, seed_mode)
    print(f"\nsaved {os.path.relpath(doc, PROJECT_ROOT)} | {(time.time() - start) / 60:.1f} min")
    return doc


def _short(metric: str) -> str:
    return metric.removeprefix("test_").removeprefix("calibrated_")


def _report(table: pd.DataFrame, backtest_id: str, label: str, dataset: str, seconds, seeds,
            seed_mode: str = "grid") -> Path:
    grid = seed_mode == "grid" and len(seeds) > 1  # only then is there a spread between seeds per window
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
        ax.set_title(title); ax.set_xlabel("test months"); ax.tick_params(axis="x", rotation=30); ax.legend()
    fig.suptitle(f"Backtest {label}: " + (f"mean over {len(seeds)} seeds per scenario, bars = spread between seeds"
                                          if grid else "one seed per window (period and seed vary together)"))
    fig.tight_layout(); fig.savefig(figure_path, dpi=120); plt.close(fig)

    def spread(col):
        """Mean, variation between periods (std of the scenario means) and between seeds (mean std)."""
        between_periods = mean[col].std(ddof=1) if len(mean) > 1 else np.nan
        if not grid:
            return f"{mean[col].mean():.4f} / ± {between_periods:.4f}"
        return f"{mean[col].mean():.4f} / ± {between_periods:.4f} / ± {seed_std[col].mean():.4f}"

    header = ["#", "test months", "test rows", "churn", "AUC ref", "AUC opt", "log-loss ref", "log-loss opt", "ECE ref", "ECE opt", "C-index"]
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

    text = f"""# Temporal Robustness (training spec, point 8), brandId {label}

The optimisation is repeated on earlier windows of the monthly cutoffs of `{dataset}`: window k takes the first cutoffs up to its test months and splits them as in normal training (train = the earlier months, validation = the month before the test months, test = the last 2 months). Three models per window, trained exactly as in normal training: the **reference** LightGBM (optimisation step 1), the **optimised** LightGBM (every optimisation step) and Cox PH (with the optimised LightGBM's features). Seed mode **{seed_mode}**: {"each window is repeated with every seed" if grid else "each window has its own random seed"} ({", ".join(map(str, seeds))}); the seed changes the LightGBM row and column sampling and the Optuna search.

Backtest id `{backtest_id}` ({f"{len(mean)} windows x {len(seeds)} seeds" if grid else f"{len(mean)} windows, one seed each"} = {len(table)} trainings per model, {seconds / 60:.1f} min). Every run is in MLflow with the tags `backtest_id`, `scenario` and `seed`; the summary run (experiment `{MLFLOW_EXPERIMENT}`) has one chart per metric across windows.

## Results per window (test months; mean ± spread between seeds)

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

**Does the optimisation gain hold on every window?** {"Yes" if holds else "No"}: averaged over the seeds, the optimised model has a higher or equal AUC in {int((gain >= 0).sum())} of {len(mean)} windows (mean change {gain.mean():+.4f}) and a lower or equal log-loss in {int((ll_gain >= 0).sum())} of {len(mean)} (mean change {-ll_gain.mean():+.4f}). {"The optimised configuration can stay." if holds else "By the spec's rule (point 8), the simpler reference configuration is the safer choice until the gain holds on every window."}

## How to read it

- A difference between two models smaller than the spread between seeds is luck, not a real difference; one smaller than the spread between periods may not hold next month.
- The windows are **not independent**: they share their earlier months, and the players repeat across cutoffs.
- The winsorisation caps come from the configuration of the main pipeline (they may have seen some test months' features, not their labels).

Generated by `src/models/robustness.py` (`make robustness`). Table (one row per scenario and seed): `{os.path.relpath(table_path, PROJECT_ROOT)}`.
"""
    doc.write_text(text, encoding="utf-8")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=backtest_id):
        mlflow.set_tags({"brand_id": label, "backtest_id": backtest_id})
        mlflow.log_params({"dataset_path": dataset, "seed_mode": seed_mode, "n_windows": len(mean), "seeds": seeds,
                           "test_cutoffs": list(mean["test_cutoff"])})
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
    parser = argparse.ArgumentParser(description="Temporal robustness of the reference and optimised models.")
    parser.add_argument("--dataset-path", help="default: the latest train_dataset file")
    parser.add_argument("--n-windows", type=int, default=DEFAULT_WINDOWS, help="earlier windows to train and test")
    parser.add_argument("--n-trials", type=int, help="Optuna trials per optimised run (default: the catalog's)")
    parser.add_argument("--seed-mode", choices=["per_window", "grid"], default=DEFAULT_SEED_MODE,
                        help="per_window: one random seed per window (fast); grid: every window x every --seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS), help="grid mode: the seeds")
    args = parser.parse_args()
    run(args.dataset_path, args.n_windows, args.n_trials, args.seeds, args.seed_mode)
