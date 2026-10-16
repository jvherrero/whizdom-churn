"""Calibration level, estimated again on the scoring date.

The calibration is fitted on the validation month, which is 2 to 3 months older than the scoring
date because the label needs 60 days. If the churn rate changes in between, all probabilities move
by about the same amount. The shape is still right, but the level is not, so we shift it.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import daily_cache  # noqa: E402
from build_features import LABEL_HORIZON_DAYS, MIN_CUTOFF, finalise, raw_features  # noqa: E402
from build_survival_dataset import with_labels  # noqa: E402
from segments import add_segment_features  # noqa: E402
from train import BRAND, CALIBRATION_BOUND, TARGET, churn_probability  # noqa: E402

N_MATURED = 3
MIN_EARLY_DAYS = 14
MAX_SHIFT = 2.0  # the backtest never needed more than about 0.9
KEY = ["tenant_id", "player_id"]


def cutoffs(as_of: dt.date, n_matured: int = N_MATURED) -> tuple[list[dt.date], list[dt.date]]:
    """Return the last months with a complete label and the later months with at least MIN_EARLY_DAYS."""
    months = [d.date() for d in pd.date_range(MIN_CUTOFF, as_of, freq="MS")]
    matured = [c for c in months if c + dt.timedelta(days=LABEL_HORIZON_DAYS) <= as_of][-n_matured:]
    early = [c for c in months if (not matured or c > matured[-1]) and (as_of - c).days >= MIN_EARLY_DAYS
             and c + dt.timedelta(days=LABEL_HORIZON_DAYS) > as_of]
    return matured, early


def predictor(model, calibrator, segment_model, features: list[str]) -> Callable[[pd.DataFrame], np.ndarray]:
    """Function that gives the calibrated churn probability, without the shift."""
    def predict(rows: pd.DataFrame) -> np.ndarray:
        X = add_segment_features(rows, segment_model) if segment_model is not None else rows
        return churn_probability(model, calibrator, X, features)
    return predict


def needed_shift(p: np.ndarray, rate: float) -> float | None:
    """Log-odds shift that makes mean(p) equal to `rate`, or None."""
    if not 0 < rate < 1 or len(p) == 0:
        return None
    z = logit(np.clip(np.asarray(p, dtype=float), CALIBRATION_BOUND, 1 - CALIBRATION_BOUND))
    try:
        return float(brentq(lambda d: expit(z + d).mean() - rate, -10, 10))
    except ValueError:
        return None


def apply(p: np.ndarray, brands, shifts: dict[int, float]) -> np.ndarray:
    """Apply each brand's shift to the probabilities."""
    delta = pd.Series(np.asarray(brands)).map(shifts).fillna(0).to_numpy()
    z = logit(np.clip(np.asarray(p, dtype=float), CALIBRATION_BOUND, 1 - CALIBRATION_BOUND))
    return np.clip(expit(z + delta), CALIBRATION_BOUND, 1 - CALIBRATION_BOUND)

# Estimates the churn rate of a recent month: the share of players with no bet yet, multiplied by
# the historical probability that they stay silent for 60 days.
def early_rate(early: pd.DataFrame, matured: pd.DataFrame, as_of: dt.date, brand_id: int) -> float:
    """Estimated churn rate of a recent cutoff, using a complete cutoff as reference."""
    c_early, c_matured = early["cutoff_date"].iloc[0], matured["cutoff_date"].iloc[0]
    k = (as_of - c_early).days
    act = daily_cache.read("activity", brand_id, start=c_matured + dt.timedelta(days=1), end=as_of,
                           columns="tenant_id, player_id, day", active_only=True)
    act["day"] = pd.to_datetime(act["day"])

    def first_bet(rows: pd.DataFrame, cutoff: dt.date) -> pd.Series:
        """Days from the cutoff to each player's next bet (NaN if none yet)."""
        c = pd.Timestamp(cutoff)
        days = (act[act["day"] > c].groupby(KEY)["day"].min() - c).dt.days
        return days.reindex(pd.MultiIndex.from_frame(rows[KEY])).reset_index(drop=True)

    silent_early = ~(first_bet(early, c_early) <= k)
    t = first_bet(matured, c_matured)
    silent_k, silent_60 = ~(t <= k), ~(t <= LABEL_HORIZON_DAYS)
    return float(silent_early.mean() * silent_60.sum() / max(silent_k.sum(), 1))


def fit(predict: Callable[[pd.DataFrame], np.ndarray], matured: pd.DataFrame, early: pd.DataFrame | None,
        as_of: dt.date) -> dict[int, float]:
    """One log-odds shift per brand: the median of the shifts needed on the recent months."""
    shifts = {}
    for brand, rows in matured.groupby(BRAND):
        needed = [needed_shift(predict(g), g[TARGET].mean()) for _, g in rows.groupby("cutoff_date")]
        if early is not None:
            last = rows[rows["cutoff_date"] == rows["cutoff_date"].max()]
            for _, g in early[early[BRAND] == brand].groupby("cutoff_date"):
                needed.append(needed_shift(predict(g), early_rate(g, last, as_of, int(brand))))
        needed = [s for s in needed if s is not None]
        shifts[int(brand)] = float(np.clip(np.median(needed), -MAX_SHIFT, MAX_SHIFT)) if needed else 0.0
    return shifts


def frames(as_of: dt.date, brand_id: int | str, dataset: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Rows of the complete and recent months, from the training dataset or the daily caches."""
    matured_cutoffs, early_cutoffs = cutoffs(as_of)
    have = set(pd.to_datetime(dataset["cutoff_date"]).dt.date) if dataset is not None else set()
    parts = []
    for c in matured_cutoffs:
        if c in have:
            rows = dataset[pd.to_datetime(dataset["cutoff_date"]).dt.date == c].copy()
            rows["cutoff_date"] = c
        else:
            rows, _ = with_labels(finalise(raw_features(c, brand_id)), as_of)
        parts.append(rows)
    matured = pd.concat(parts, ignore_index=True)
    early = pd.concat([finalise(raw_features(c, brand_id)) for c in early_cutoffs], ignore_index=True) if early_cutoffs else None
    return matured, early
