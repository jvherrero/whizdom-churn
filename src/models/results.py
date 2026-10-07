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

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import brand_label  # noqa: E402
from evaluation.evaluate import cox_results, lightgbm_results, log_results, run_data  # noqa: E402
from evaluation.metrics import N_BOOTSTRAP  # noqa: E402
from train import BRAND, PROCESSED_DIR, SEED, run_info, train  # noqa: E402

DOC_PATH = PROJECT_ROOT / "docs/results_v0_brand{brand}.md"


def _fmt(value, digits=3) -> str:
    point, ci = value
    if point != point:  # NaN: not measurable on this split
        return "n/a"
    return f"{point:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]" if ci else f"{point:.{digits}f}"


def _onecal(res: dict, h) -> str:
    if f"one_calibration_{h}d_gap" not in res:
        return "n/a"
    return f"{res[f'one_calibration_{h}d_gap'][0]:.3f} ({res[f'one_calibration_{h}d_p_value'][0]:.3g})"


def write_doc(lgbm: dict, cox: dict, lgbm_res: dict, cox_res: dict, data: pd.DataFrame, split: pd.Series,
              dataset_path: Path) -> Path:
    label = brand_label(data[BRAND].unique())
    path = Path(str(DOC_PATH).format(brand=label))
    cutoffs = {s: ", ".join(sorted(str(c) for c in data.loc[split == s, "cutoff_date"].unique()))
               for s in ("train", "valid", "test")}
    test_events = data.loc[(split == "test") & (data["event_observed"] == 1), "duration_days"]
    p = lgbm["params"]
    horizons = [k.removeprefix("td_auc_").removesuffix("d") for k in cox_res["test"] if k.startswith("td_auc_")]
    survival_rows = "".join(
        f"| Cox PH | Time-dependent AUC, day {h} | {_fmt(cox_res['valid'].get(f'td_auc_{h}d', (float('nan'), None)))} "
        f"| {_fmt(cox_res['test'][f'td_auc_{h}d'])} |\n"
        f"| Cox PH | One-calibration, day {h}: gap (p-value) | {_onecal(cox_res['valid'], h)} | {_onecal(cox_res['test'], h)} |\n"
        for h in horizons)
    text = f"""# Results v0: reference models (brandId {label})

The first results table: the reference LightGBM and Cox PH, trained once on the temporal split. "Reference" means optimisation step 1 only: the catalog defaults, binary log-loss with the preferred class weights and early stopping on the validation rows. No imbalance comparison, hyperparameter search, regularisation check or final pruning. The optimised models are measured against this table.

## Setup

| | |
|---|---|
| Dataset | `{os.path.relpath(dataset_path, PROJECT_ROOT)}` |
| Train months | {cutoffs['train']} |
| Validation month | {cutoffs['valid']} (early stopping and calibration) |
| Test months (out of time) | {cutoffs['test']} |
| Rows | train {int((split == 'train').sum()):,}, validation {int((split == 'valid').sum()):,}, test {int((split == 'test').sum()):,} |
| Seed | {SEED} (models, bootstrap) |
| LightGBM run | `{lgbm['run_name']}`: {len(lgbm['features']) - 1} features + `brandId`, `learning_rate` {p.get('learning_rate')}, `scale_pos_weight` {p.get('scale_pos_weight')}, {p.get('n_estimators')} trees max (early stopping) |
| Cox PH run | `{cox['run_name']}`: {len(cox['features'])} features, stratified by brand, `penalizer` {cox['params'].get('penalizer')} |
| Config | `config/catalog_entry.json` in each run |

Confidence intervals: 95% percentile intervals over {N_BOOTSTRAP} bootstrap resamples of the rows (player-cutoffs) of each split. They show the sampling noise of the evaluation rows, not the variation between training runs.

## Results

| Model | Metric | Validation | Test (out of time) |
|---|---|---|---|
| LightGBM | AUC | {_fmt(lgbm_res['valid']['auc'])} | **{_fmt(lgbm_res['test']['auc'])}** |
| LightGBM | ECE (calibrated) | {_fmt(lgbm_res['valid']['ece'])} | **{_fmt(lgbm_res['test']['ece'])}** |
| LightGBM | ECE (raw model output) | {_fmt(lgbm_res['valid']['ece_raw'])} | {_fmt(lgbm_res['test']['ece_raw'])} |
| LightGBM | Log-loss (calibrated) | {_fmt(lgbm_res['valid']['log_loss'])} | {_fmt(lgbm_res['test']['log_loss'])} |
| LightGBM | Top-decile precision (lift) | {_fmt(lgbm_res['valid']['top_decile_precision'])} ({lgbm_res['valid']['top_decile_lift'][0]:.2f}x) | {_fmt(lgbm_res['test']['top_decile_precision'])} ({lgbm_res['test']['top_decile_lift'][0]:.2f}x) |
| Cox PH | C-index | {_fmt(cox_res['valid']['c_index'])} | **{_fmt(cox_res['test']['c_index'])}** |
| Cox PH | IBS (days 0 to {cox_res['test']['horizon']}) | {_fmt(cox_res['valid']['ibs'])} | **{_fmt(cox_res['test']['ibs'])}** |
{survival_rows}
Churn in 60 days: validation {lgbm_res['valid']['churn_rate']:.1%}, test {lgbm_res['test']['churn_rate']:.1%}.

## How to Read It

- **AUC** (0.5 = random, 1 = perfect): how well LightGBM ranks players by churn risk.
- **ECE** (lower is better): the average gap between the predicted probability and the observed churn rate. The calibrated value is what scoring delivers; the raw one shows what calibration fixes (the class weights push the raw probabilities up). Its bootstrap interval can sit above the point value: resampling adds noise to every bin, and an absolute gap only grows with noise, so ECE intervals are biased upwards on small splits.
- **Top-decile precision**: the churn rate among the 10% of players with the highest score, the ones a retention campaign would contact; the lift is that rate over the overall churn rate.
- **C-index** (0.5 = random, 1 = perfect): how well Cox PH ranks who churns earlier.
- **Time-dependent AUC at day t**: how well Cox PH separates players who churned by day t from players still active after t (IPCW). Only the days up to the last training churn day are shown.
- **One-calibration at day t**: players in 10 groups of predicted P(churn by t); the gap is the mean difference between the predicted and the Kaplan-Meier probability, the p-value a Hosmer-Lemeshow-type test (small = miscalibrated). The Cox survival curve is not calibrated; the probability the business uses comes from the calibrated LightGBM.
- **IBS** (lower is better, 0.25 = a coin flip): the squared error of the predicted survival curve, averaged over days 0 to {cox_res['test']['horizon']}, with censored players weighted by the inverse probability of being still observed (IPCW, censoring curve from the train rows).
- **The test months only show churn on days 0 to {int(test_events.max()) if len(test_events) else 0}** (a churn day can be confirmed only when the 60 days after it are in the data). So the Cox test metrics mostly measure "who is already gone", not "which day".
- **The validation numbers are slightly optimistic**: early stopping uses that month (the calibrated probability is cross-fitted, so calibration does not). The test months are the honest check.

Generated by `src/models/results.py` (`make results`), with the metrics of `src/evaluation/`. The same metrics, with their intervals, are logged in both MLflow runs as `results_*`.
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
    data, split = run_data(lgbm)  # the same dataset, seed and segments for both runs
    lgbm_res, cox_res = lightgbm_results(lgbm, data, split), cox_results(cox, data, split)
    log_results(lgbm, lgbm_res)
    log_results(cox, cox_res)
    doc = write_doc(lgbm, cox, lgbm_res, cox_res, data, split, dataset_path)
    print(f"saved {os.path.relpath(doc, PROJECT_ROOT)} | runs {lgbm['run_name']}, {cox['run_name']}")
    return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reference LightGBM + Cox PH and the first results table.")
    parser.add_argument("--dataset-path", help="default: the latest train_dataset file")
    run(parser.parse_args().dataset_path)
