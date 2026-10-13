"""Evaluation metrics. They take arrays and return numbers.

Classification: expected_calibration_error, top_decile_precision, classification_metrics.
Survival: c_index, ipcw_weights, time_dependent_auc, one_calibration, integrated_brier_score.
Uncertainty: bootstrap_ci (95% interval over resampled rows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ECE_BINS = 10
TOP_SHARE = 0.10  # top decile
CALIBRATION_GROUPS = 10
N_BOOTSTRAP = 200
SEED = 42


def expected_calibration_error(y, p, n_bins: int = ECE_BINS) -> float:
    """Mean |observed rate - mean prediction| over equal-size bins."""
    bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    frame = pd.DataFrame({"y": np.asarray(y), "p": p, "bin": bins})
    per_bin = frame.groupby("bin").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((per_bin["n"] * (per_bin["y"] - per_bin["p"]).abs()).sum() / per_bin["n"].sum())


def top_decile_precision(y, score, share: float = TOP_SHARE) -> float:
    """Share of churners among the highest `share` of scores."""
    y, score = np.asarray(y), np.asarray(score)
    n_top = max(1, int(round(len(y) * share)))
    return float(y[np.argsort(-score, kind="stable")[:n_top]].mean())


def classification_metrics(y, p) -> dict:
    precision = top_decile_precision(y, p)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "top_decile_precision": precision,
        "top_decile_lift": precision / float(np.mean(y)) if np.mean(y) else float("nan"),
        "mean_prediction": float(np.mean(p)),
        "churn_rate": float(np.mean(y)),
    }


def c_index(durations, risk, events) -> float:
    """Harrell's c-index. Higher risk means earlier churn."""
    return concordance_index(durations, -np.asarray(risk), events)


def _censoring_weight(censoring: KaplanMeierFitter, t) -> np.ndarray:
    """G(t): probability of still being observed at day t (Kaplan-Meier of the censoring)."""
    return np.clip(censoring.survival_function_at_times(np.atleast_1d(t)).to_numpy(), 1e-6, None)


def ipcw_weights(durations, censoring: KaplanMeierFitter) -> np.ndarray:
    """Inverse probability of censoring weight, 1 / G(T-), per player."""
    d = np.asarray(durations, dtype=float)
    return 1 / np.where(d > 0, _censoring_weight(censoring, np.maximum(d - 1, 0)), 1.0)


def time_dependent_auc(durations, events, risk, t: float, weights) -> float:
    """AUC at day t (Uno's IPCW estimator): players who churned by t against players still active
    after t. NaN if one of the groups is empty."""
    d, e, r = np.asarray(durations, dtype=float), np.asarray(events), np.asarray(risk, dtype=float)
    cases, controls = (d <= t) & (e == 1), d > t
    if not cases.any() or not controls.any():
        return float("nan")
    w = np.asarray(weights, dtype=float)[cases]
    control_risk = np.sort(r[controls])
    below = np.searchsorted(control_risk, r[cases], side="left")
    ties = np.searchsorted(control_risk, r[cases], side="right") - below
    concordant = (below + 0.5 * ties) / len(control_risk)
    return float(np.sum(w * concordant) / np.sum(w))


def one_calibration(durations, events, churn_by_t, t: float, n_groups: int = CALIBRATION_GROUPS) -> dict:
    """One-calibration at day t (D'Agostino-Nam): compare the predicted and Kaplan-Meier churn
    probability in groups of risk. Returns the statistic, p-value and mean absolute gap."""
    d, e, p = np.asarray(durations, dtype=float), np.asarray(events), np.asarray(churn_by_t, dtype=float)
    groups = pd.qcut(p, n_groups, labels=False, duplicates="drop")
    statistic, gaps, sizes = 0.0, [], []
    for g in np.unique(groups):
        rows = groups == g
        km = KaplanMeierFitter().fit(d[rows], event_observed=e[rows])
        observed = 1 - float(km.survival_function_at_times(t).iloc[0])
        expected = float(p[rows].mean())
        statistic += rows.sum() * (observed - expected) ** 2 / max(expected * (1 - expected), 1e-12)
        gaps.append(abs(observed - expected))
        sizes.append(rows.sum())
    df = max(len(gaps) - 1, 1)
    return {"statistic": float(statistic), "p_value": float(stats.chi2.sf(statistic, df)),
            "mean_abs_gap": float(np.average(gaps, weights=sizes))}


def brier_contributions(surv: np.ndarray, grid: np.ndarray, durations, events, censoring: KaplanMeierFitter) -> np.ndarray:
    """IPCW Brier terms per player and day."""
    d, e = np.asarray(durations, dtype=float), np.asarray(events)
    w_before = ipcw_weights(d, censoring)  # 1 / G(T-)
    out = np.zeros_like(surv)
    for j, t in enumerate(grid):
        churned = (d <= t) & (e == 1)
        active = d > t
        out[:, j] = churned * surv[:, j] ** 2 * w_before + active * (1 - surv[:, j]) ** 2 / _censoring_weight(censoring, t)[0]
    return out


def integrated_brier_score(contrib: np.ndarray, grid: np.ndarray, rows=None) -> float:
    """Integrated Brier score over the day grid."""
    curve = (contrib if rows is None else contrib[rows]).mean(axis=0)
    return float(curve[0]) if len(grid) == 1 else float(np.trapezoid(curve, grid) / (grid[-1] - grid[0]))


def bootstrap_ci(n_rows: int, statistic, n_resamples: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval of `statistic` over resampled rows."""
    rng = np.random.default_rng(seed)
    values = [statistic(rng.integers(0, n_rows, n_rows)) for _ in range(n_resamples)]
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
