"""Evaluation metrics, model-agnostic: arrays in, numbers out.

Classification (the churn probability):
    expected_calibration_error   mean |observed rate - mean prediction| over equal-size bins
    top_decile_precision         churn rate among the 10% highest scores (and its lift over the base rate)
    classification_metrics       ROC AUC, PR AUC, Brier, log-loss, ECE, top-decile precision, ...
Survival (time to churn):
    c_index                      Harrell's concordance of a risk score (higher = earlier churn)
    ipcw_weights                 1 / G(T-) per player, the inverse probability of censoring weight
    time_dependent_auc           cumulative/dynamic AUC at day t: churned by t vs still active after t (IPCW)
    one_calibration              predicted vs Kaplan-Meier churn probability by day t, per risk decile
    brier_contributions          per-player IPCW Brier terms over a grid of days
    integrated_brier_score       their mean, integrated over the grid (0.25 = a coin flip)
Uncertainty:
    bootstrap_ci                 95% percentile interval of a statistic over resampled rows
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
N_BOOTSTRAP = 200  # delivery plan: bootstrap 95% CI, 200 resamples
SEED = 42


def expected_calibration_error(y, p, n_bins: int = ECE_BINS) -> float:
    """Mean |observed rate - mean prediction| over equal-size bins of the prediction, weighted by size."""
    bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    frame = pd.DataFrame({"y": np.asarray(y), "p": p, "bin": bins})
    per_bin = frame.groupby("bin").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((per_bin["n"] * (per_bin["y"] - per_bin["p"]).abs()).sum() / per_bin["n"].sum())


def top_decile_precision(y, score, share: float = TOP_SHARE) -> float:
    """Share of churners among the `share` highest scores: who a retention campaign would contact."""
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
    """Concordance of `risk` (e.g. the Cox partial hazard: higher means an earlier churn)."""
    return concordance_index(durations, -np.asarray(risk), events)


def _censoring_weight(censoring: KaplanMeierFitter, t) -> np.ndarray:
    """G(t), the probability of still being observed at day t (censoring KM), floored to avoid 1/0."""
    return np.clip(censoring.survival_function_at_times(np.atleast_1d(t)).to_numpy(), 1e-6, None)


def ipcw_weights(durations, censoring: KaplanMeierFitter) -> np.ndarray:
    """1 / G(T-) per player: the inverse probability of still being observed just before the churn
    day (G = censoring KM; days are integers, so T- = T - 1; G(T-) = 1 for T = 0)."""
    d = np.asarray(durations, dtype=float)
    return 1 / np.where(d > 0, _censoring_weight(censoring, np.maximum(d - 1, 0)), 1.0)


def time_dependent_auc(durations, events, risk, t: float, weights) -> float:
    """Cumulative/dynamic AUC at day t (Uno's IPCW estimator): the probability that a player who
    churned by day t has a higher risk than a player still active after t. Each case is weighted by
    `weights` (ipcw_weights()), so players censored early do not bias it. NaN if t has no cases or
    no controls."""
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
    """One-calibration at day t (D'Agostino-Nam): players in `n_groups` groups of predicted P(churn by t);
    in each, the mean prediction vs the Kaplan-Meier churn probability by t (censoring-aware).
    Returns the Hosmer-Lemeshow-type statistic, its chi-square p-value (df = groups - 1; a small
    p-value means miscalibrated) and the mean absolute gap between predicted and observed."""
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
    """Per-player IPCW Brier terms (players x days). At day t, a player who churned by t
    (event, T <= t) should have S(t) = 0, weighted by 1 / G(T-); a player still active (T > t)
    should have S(t) = 1, weighted by 1 / G(t). Censored before t: weight 0. G = censoring KM."""
    d, e = np.asarray(durations, dtype=float), np.asarray(events)
    w_before = ipcw_weights(d, censoring)  # 1 / G(T-)
    out = np.zeros_like(surv)
    for j, t in enumerate(grid):
        churned = (d <= t) & (e == 1)
        active = d > t
        out[:, j] = churned * surv[:, j] ** 2 * w_before + active * (1 - surv[:, j]) ** 2 / _censoring_weight(censoring, t)[0]
    return out


def integrated_brier_score(contrib: np.ndarray, grid: np.ndarray, rows=None) -> float:
    """Mean Brier term per day (over `rows`, default all), integrated over the grid."""
    curve = (contrib if rows is None else contrib[rows]).mean(axis=0)
    return float(curve[0]) if len(grid) == 1 else float(np.trapezoid(curve, grid) / (grid[-1] - grid[0]))


def bootstrap_ci(n_rows: int, statistic, n_resamples: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval of `statistic(indices)` over resamples of the rows (one row = one player)."""
    rng = np.random.default_rng(seed)
    values = [statistic(rng.integers(0, n_rows, n_rows)) for _ in range(n_resamples)]
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
