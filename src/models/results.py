"""First results table: the reference LightGBM (optimisation step 1: catalog defaults, preferred class
weights, early stopping; no imbalance comparison, search, regularisation or pruning) and the reference
Cox PH, both on the temporal split, with 95% bootstrap confidence intervals.

    .venv/bin/python src/models/results.py                   # latest train_dataset
    .venv/bin/python src/models/results.py --dataset-path data/processed/train_dataset_...parquet

Two MLflow runs (role=reference, with config, seed and metrics, the CI metrics included) and
docs/results_v0_brand{id}.md with AUC, C-index, IBS and ECE.

    AUC       LightGBM: ranks churners above non-churners (event_60d)
    ECE       LightGBM: expected calibration error of the delivered (calibrated) probability
    C-index   Cox PH: ranks who churns earlier
    IBS       Cox PH: integrated Brier score of the survival curve over days 0..horizon, with inverse
              probability of censoring weights (IPCW); lower is better, 0.25 is a coin flip
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.metrics import log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import brand_label
from segments import add_segment_features, load_segment_model
from train import (
    BRAND, DURATION, EVENT, PROCESSED_DIR, SEED, TARGET, churn_probability, cox_input,
    expected_calibration_error, load_calibrator, model_input, run_info, split_rows, train,
)

N_BOOTSTRAP = 1000
DOC_PATH = PROJECT_ROOT / "docs/results_v0_brand{brand}.md"


def _ci(values) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def bootstrap(n_rows: int, statistic, seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval of `statistic(indices)` over N_BOOTSTRAP resamples of the rows.
    One row is one player here (a split holds one cutoff per player in the test month)."""
    rng = np.random.default_rng(seed)
    return _ci([statistic(rng.integers(0, n_rows, n_rows)) for _ in range(N_BOOTSTRAP)])


def brier_contributions(surv: np.ndarray, grid: np.ndarray, durations, events, censoring: KaplanMeierFitter) -> np.ndarray:
    """Per-player IPCW Brier terms (players x days). At day t, a player who churned by t
    (event, T <= t) should have S(t) = 0, weighted by 1 / G(T-); a player still active (T > t)
    should have S(t) = 1, weighted by 1 / G(t). Censored before t: weight 0. G = censoring KM."""
    d, e = np.asarray(durations, dtype=float), np.asarray(events)
    g = lambda t: np.clip(censoring.survival_function_at_times(t).to_numpy(), 1e-6, None)
    g_before = np.where(d > 0, g(np.maximum(d - 1, 0)), 1.0)  # G(T-): days are integers
    out = np.zeros_like(surv)
    for j, t in enumerate(grid):
        churned = (d <= t) & (e == 1)
        active = d > t
        out[:, j] = churned * surv[:, j] ** 2 / g_before + active * (1 - surv[:, j]) ** 2 / g(t)[0]
    return out


def integrated(contrib: np.ndarray, grid: np.ndarray, rows=None) -> float:
    curve = (contrib if rows is None else contrib[rows]).mean(axis=0)
    return float(curve[0]) if len(grid) == 1 else float(np.trapezoid(curve, grid) / (grid[-1] - grid[0]))


def lightgbm_results(info: dict, data: pd.DataFrame, split: pd.Series) -> dict:
    model = mlflow.lightgbm.load_model(f"runs:/{info['run_id']}/model")
    calibrator = load_calibrator(info["run_id"])
    out = {}
    for name in ("valid", "test"):
        part = data[split == name].reset_index(drop=True)
        y = part[TARGET].to_numpy()
        raw = model.predict_proba(model_input(part, info["features"]))[:, 1]
        p = churn_probability(model, calibrator, part, info["features"])
        out[name] = {
            "auc": (roc_auc_score(y, p), bootstrap(len(y), lambda i: roc_auc_score(y[i], p[i]))),
            "ece": (expected_calibration_error(y, p), bootstrap(len(y), lambda i: expected_calibration_error(y[i], p[i]))),
            "ece_raw": (expected_calibration_error(y, raw), None),
            "log_loss": (log_loss(y, p), bootstrap(len(y), lambda i: log_loss(y[i], p[i], labels=[0, 1]))),
            "churn_rate": float(y.mean()), "n": len(y),
        }
    return out


def cox_results(info: dict, data: pd.DataFrame, split: pd.Series) -> dict:
    model = joblib.load(mlflow.artifacts.download_artifacts(f"runs:/{info['run_id']}/model/model.joblib"))
    train_rows = data[split == "train"]
    censoring = KaplanMeierFitter().fit(train_rows[DURATION], event_observed=1 - train_rows[EVENT])
    horizon = int(info["metrics"]["train_max_event_day"])
    grid = np.arange(0, horizon + 1)
    out = {}
    for name in ("valid", "test"):
        part = data[split == name].reset_index(drop=True)
        d, e = part[DURATION].to_numpy(), part[EVENT].to_numpy()
        risk = np.asarray(model.predict_partial_hazard(cox_input(model, part)))
        surv = model.predict_survival_function(cox_input(model, part), times=grid).T.to_numpy()
        contrib = brier_contributions(surv, grid, d, e, censoring)
        out[name] = {
            "c_index": (concordance_index(d, -risk, e), bootstrap(len(d), lambda i: concordance_index(d[i], -risk[i], e[i]))),
            "ibs": (integrated(contrib, grid), bootstrap(len(d), lambda i: integrated(contrib, grid, i))),
            "event_rate": float(e.mean()), "n": len(d), "horizon": horizon,
        }
    return out


def _log(info: dict, results: dict):
    with mlflow.start_run(run_id=info["run_id"]):
        for split_name, metrics in results.items():
            for key, value in metrics.items():
                if isinstance(value, tuple):
                    point, ci = value
                    mlflow.log_metric(f"results_{split_name}_{key}", point)
                    if ci:
                        mlflow.log_metric(f"results_{split_name}_{key}_ci_low", ci[0])
                        mlflow.log_metric(f"results_{split_name}_{key}_ci_high", ci[1])
        mlflow.log_param("results_bootstrap", f"{N_BOOTSTRAP} resamples of players, 95% percentile interval")


def _fmt(value, digits=3) -> str:
    point, ci = value
    return f"{point:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]" if ci else f"{point:.{digits}f}"


def write_doc(lgbm: dict, cox: dict, lgbm_res: dict, cox_res: dict, data: pd.DataFrame, split: pd.Series,
              dataset_path: Path) -> Path:
    label = brand_label(data[BRAND].unique())
    path = Path(str(DOC_PATH).format(brand=label))
    test_cutoff = ", ".join(sorted(str(c) for c in data.loc[split == "test", "cutoff_date"].unique()))
    train_cutoffs = ", ".join(sorted(str(c) for c in data.loc[split != "test", "cutoff_date"].unique()))
    p = lgbm["params"]
    text = f"""# Results v0: reference models (brandId {label})

The first results table: the reference LightGBM and Cox PH, trained once on the temporal split. "Reference" means optimisation step 1 only: the catalog defaults, binary log-loss with the preferred class weights and early stopping on the validation rows. No imbalance comparison, hyperparameter search, regularisation check or final pruning. The optimised models are measured against this table.

## Setup

| | |
|---|---|
| Dataset | `{os.path.relpath(dataset_path, PROJECT_ROOT)}` |
| Training months | {train_cutoffs} (train / validation split by player, 85 / 15) |
| Test month (out of time) | {test_cutoff} |
| Rows | train {int((split == 'train').sum()):,}, validation {int((split == 'valid').sum()):,}, test {int((split == 'test').sum()):,} |
| Seed | {SEED} (split, models, bootstrap) |
| LightGBM run | `{lgbm['run_name']}`: {len(lgbm['features']) - 1} features + `brandId`, `learning_rate` {p.get('learning_rate')}, `scale_pos_weight` {p.get('scale_pos_weight')}, {p.get('n_estimators')} trees max (early stopping) |
| Cox PH run | `{cox['run_name']}`: {len(cox['features'])} features, stratified by brand, `penalizer` {cox['params'].get('penalizer')} |
| Config | `config/catalog_entry.json` in each run |

Confidence intervals: 95% percentile intervals over {N_BOOTSTRAP} bootstrap resamples of the players of each split. They show the sampling noise of the evaluation rows, not the variation between training runs.

## Results

| Model | Metric | Validation | Test (out of time) |
|---|---|---|---|
| LightGBM | AUC | {_fmt(lgbm_res['valid']['auc'])} | **{_fmt(lgbm_res['test']['auc'])}** |
| LightGBM | ECE (calibrated) | {_fmt(lgbm_res['valid']['ece'])} | **{_fmt(lgbm_res['test']['ece'])}** |
| LightGBM | ECE (raw model output) | {_fmt(lgbm_res['valid']['ece_raw'])} | {_fmt(lgbm_res['test']['ece_raw'])} |
| LightGBM | Log-loss (calibrated) | {_fmt(lgbm_res['valid']['log_loss'])} | {_fmt(lgbm_res['test']['log_loss'])} |
| Cox PH | C-index | {_fmt(cox_res['valid']['c_index'])} | **{_fmt(cox_res['test']['c_index'])}** |
| Cox PH | IBS (days 0 to {cox_res['test']['horizon']}) | {_fmt(cox_res['valid']['ibs'])} | **{_fmt(cox_res['test']['ibs'])}** |

Churn in 60 days: validation {lgbm_res['valid']['churn_rate']:.1%}, test {lgbm_res['test']['churn_rate']:.1%}.

## How to Read It

- **AUC** (0.5 = random, 1 = perfect): how well LightGBM ranks players by churn risk.
- **ECE** (lower is better): the average gap between the predicted probability and the observed churn rate. The calibrated value is what scoring delivers; the raw one shows what calibration fixes (the class weights push the raw probabilities up).
- **C-index** (0.5 = random, 1 = perfect): how well Cox PH ranks who churns earlier.
- **IBS** (lower is better, 0.25 = a coin flip): the squared error of the predicted survival curve, averaged over days 0 to {cox_res['test']['horizon']}, with censored players weighted by the inverse probability of being still observed (IPCW, censoring curve from the train rows).
- **The test month only shows churn on day 0** (a churn day after the cutoff can be confirmed only up to 60 days before the data ends). So the Cox test metrics mostly measure "who is already gone", not "which day".
- **The validation numbers are slightly optimistic**: early stopping and calibration use those rows. The test month is the honest check.

Generated by `src/models/results.py` (`make results`). The same metrics, with their intervals, are logged in both MLflow runs as `results_*`.
"""
    path.write_text(text, encoding="utf-8")
    return path


def run(dataset_path: str | Path | None = None) -> Path:
    if dataset_path is None:
        dataset_path = max(PROCESSED_DIR.glob("train_dataset_*.parquet"), key=lambda p: p.stat().st_mtime)
    dataset_path = Path(dataset_path).resolve()
    lgbm_id = train("lightgbm_classifier", dataset_path, reference=True)
    cox_id = train("cox_ph", dataset_path, reference=True, features_from_run=lgbm_id)
    lgbm, cox = run_info(lgbm_id), run_info(cox_id)

    data = pd.read_parquet(dataset_path)
    split = split_rows(data)
    segment_model = load_segment_model(lgbm_id)
    if segment_model is not None:
        data = add_segment_features(data, segment_model)

    lgbm_res, cox_res = lightgbm_results(lgbm, data, split), cox_results(cox, data, split)
    _log(lgbm, lgbm_res)
    _log(cox, cox_res)
    doc = write_doc(lgbm, cox, lgbm_res, cox_res, data, split, dataset_path)
    print(f"saved {os.path.relpath(doc, PROJECT_ROOT)} | runs {lgbm['run_name']}, {cox['run_name']}")
    return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reference LightGBM + Cox PH and the first results table.")
    parser.add_argument("--dataset-path", help="default: the latest train_dataset file")
    run(parser.parse_args().dataset_path)
