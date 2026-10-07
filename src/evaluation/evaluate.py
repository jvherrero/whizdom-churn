"""Evaluate a logged training run on its own temporal split, with 95% bootstrap intervals.

    .venv/bin/python src/evaluation/evaluate.py --run-id <MLflow run id>

The run's model (LightGBM or Cox PH), calibrator, features and dataset are read from MLflow,
so the split and the segments are exactly the ones it was trained with. Metrics on the validation
month and the out-of-time test months:
    LightGBM   AUC, ECE (calibrated and raw), log-loss, top-decile precision and lift
    Cox PH     C-index, IBS (IPCW, days 0 to the last training event day), and at each horizon in
               HORIZONS that fits in it: time-dependent AUC and one-calibration
On the validation rows the calibrated probability is cross-fitted (each fold calibrated on the
others), as in training: the calibrator was fitted on those rows, so in-sample it looks perfect.
They are logged back into the run as results_{split}_{metric}[_ci_low|_ci_high].
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sklearn.metrics import log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
from evaluation.metrics import (  # noqa: E402
    N_BOOTSTRAP, bootstrap_ci, brier_contributions, c_index, expected_calibration_error, integrated_brier_score,
    ipcw_weights, one_calibration, time_dependent_auc, top_decile_precision,
)
from segments import add_segment_features, load_segment_model  # noqa: E402
from train import (  # noqa: E402
    DURATION, EVENT, TARGET, churn_probability, cox_input, cross_fitted_calibration, load_calibrator, model_input,
    run_info, split_rows,
)

HORIZONS = (7, 14, 30, 60)  # days: time-dependent AUC and one-calibration of the Cox survival curve


def lightgbm_results(info: dict, data: pd.DataFrame, split: pd.Series) -> dict:
    model = mlflow.lightgbm.load_model(f"runs:/{info['run_id']}/model")
    calibrator = load_calibrator(info["run_id"])
    out = {}
    for name in ("valid", "test"):
        part = data[split == name].reset_index(drop=True)
        y = part[TARGET].to_numpy()
        raw = model.predict_proba(model_input(part, info["features"]))[:, 1]
        p = (cross_fitted_calibration(raw, part, calibrator.method) if name == "valid" and calibrator is not None
             else churn_probability(model, calibrator, part, info["features"]))
        top = top_decile_precision(y, p)
        out[name] = {
            "auc": (roc_auc_score(y, p), bootstrap_ci(len(y), lambda i: roc_auc_score(y[i], p[i]))),
            "ece": (expected_calibration_error(y, p), bootstrap_ci(len(y), lambda i: expected_calibration_error(y[i], p[i]))),
            "ece_raw": (expected_calibration_error(y, raw), None),
            "log_loss": (log_loss(y, p), bootstrap_ci(len(y), lambda i: log_loss(y[i], p[i], labels=[0, 1]))),
            "top_decile_precision": (top, bootstrap_ci(len(y), lambda i: top_decile_precision(y[i], p[i]))),
            "top_decile_lift": (top / y.mean(), None),
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
            "c_index": (c_index(d, risk, e), bootstrap_ci(len(d), lambda i: c_index(d[i], risk[i], e[i]))),
            "ibs": (integrated_brier_score(contrib, grid),
                    bootstrap_ci(len(d), lambda i: integrated_brier_score(contrib, grid, i))),
            "event_rate": float(e.mean()), "n": len(d), "horizon": horizon,
        }
        weights = ipcw_weights(d, censoring)
        for t in [h for h in HORIZONS if h <= horizon]:
            auc_t = time_dependent_auc(d, e, risk, t, weights)
            if np.isnan(auc_t):
                continue  # no churner by t, or nobody still active after t, in this split
            out[name][f"td_auc_{t}d"] = (auc_t, bootstrap_ci(
                len(d), lambda i: time_dependent_auc(d[i], e[i], risk[i], t, weights[i])))
            calibration = one_calibration(d, e, 1 - surv[:, t], t)
            out[name][f"one_calibration_{t}d_gap"] = (calibration["mean_abs_gap"], None)
            out[name][f"one_calibration_{t}d_p_value"] = (calibration["p_value"], None)
    return out


def log_results(info: dict, results: dict) -> None:
    """Every (point, interval) metric into the run, as results_{split}_{metric}."""
    with mlflow.start_run(run_id=info["run_id"]):
        for split_name, metrics in results.items():
            for key, value in metrics.items():
                if isinstance(value, tuple):
                    point, ci = value
                    mlflow.log_metric(f"results_{split_name}_{key}", point)
                    if ci:
                        mlflow.log_metric(f"results_{split_name}_{key}_ci_low", ci[0])
                        mlflow.log_metric(f"results_{split_name}_{key}_ci_high", ci[1])
        mlflow.log_param("results_bootstrap", f"{N_BOOTSTRAP} resamples of the rows (player-cutoffs), 95% percentile interval")


def run_data(info: dict) -> tuple[pd.DataFrame, pd.Series]:
    """The run's dataset (with its segment features, if any) and its train/valid/test split."""
    data = pd.read_parquet(PROJECT_ROOT / info["dataset_path"])
    split = split_rows(data)
    segment_model = load_segment_model(info["run_id"])
    if segment_model is not None:
        data = add_segment_features(data, segment_model)
    return data, split


def evaluate(run_id: str, log: bool = True) -> dict:
    info = run_info(run_id)
    data, split = run_data(info)
    kind = {"lightgbm_classifier": lightgbm_results, "cox_ph": cox_results}.get(info["model_id"])
    if kind is None:
        raise ValueError(f"no evaluation for model {info['model_id']!r} (lightgbm_classifier or cox_ph)")
    results = kind(info, data, split)
    if log:
        log_results(info, results)
    return results


def _fmt(value) -> str:
    point, ci = value
    return f"{point:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else f"{point:.4f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a training run with bootstrap intervals.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--no-log", action="store_true", help="print only, do not log into the run")
    args = parser.parse_args()
    for split_name, metrics in evaluate(args.run_id, log=not args.no_log).items():
        print(split_name, {k: _fmt(v) for k, v in metrics.items() if isinstance(v, tuple)})
