"""Calibration level, re-estimated on the scoring date.

The isotonic calibration of a training run (train.py) is fitted on its validation month: in use, the
latest month with a complete 60-day label is 2 to 3 months before the date the model scores. The churn
rate moves in between (around 50% from November to January, 30 to 40% the rest of the year) and every
probability is then off by about the same amount: the shape of the calibration holds, its level does not.
In the backtest, setting only the level right would have kept every month's ECE under 0.06.
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
MAX_SHIFT = 2.0  # safety bound: the largest shift the backtest needed was about 0.9
KEY = ["tenant_id", "player_id"]


def cutoffs(as_of: dt.date, n_matured: int = N_MATURED) -> tuple[list[dt.date], list[dt.date]]:
    """(the last n_matured month starts with a complete label on as_of, the later ones with at least
    MIN_EARLY_DAYS days seen)."""
    months = [d.date() for d in pd.date_range(MIN_CUTOFF, as_of, freq="MS")]
    matured = [c for c in months if c + dt.timedelta(days=LABEL_HORIZON_DAYS) <= as_of][-n_matured:]
    early = [c for c in months if (not matured or c > matured[-1]) and (as_of - c).days >= MIN_EARLY_DAYS
             and c + dt.timedelta(days=LABEL_HORIZON_DAYS) > as_of]
    return matured, early


def predictor(model, calibrator, segment_model, features: list[str]) -> Callable[[pd.DataFrame], np.ndarray]:
    """Rows of the feature table -> the run's calibrated churn probability (no level shift)."""
    def predict(rows: pd.DataFrame) -> np.ndarray:
        X = add_segment_features(rows, segment_model) if segment_model is not None else rows
        return churn_probability(model, calibrator, X, features)
    return predict


def needed_shift(p: np.ndarray, rate: float) -> float | None:
    """The log-odds shift that makes mean(p) equal `rate` (None when it cannot)."""
    if not 0 < rate < 1 or len(p) == 0:
        return None
    z = logit(np.clip(np.asarray(p, dtype=float), CALIBRATION_BOUND, 1 - CALIBRATION_BOUND))
    try:
        return float(brentq(lambda d: expit(z + d).mean() - rate, -10, 10))
    except ValueError:
        return None


def apply(p: np.ndarray, brands, shifts: dict[int, float]) -> np.ndarray:
    """The probabilities with each brand's shift (0 for a brand without one)."""
    delta = pd.Series(np.asarray(brands)).map(shifts).fillna(0).to_numpy()
    z = logit(np.clip(np.asarray(p, dtype=float), CALIBRATION_BOUND, 1 - CALIBRATION_BOUND))
    return np.clip(expit(z + delta), CALIBRATION_BOUND, 1 - CALIBRATION_BOUND)

#estimates the future churn rate. It does so by multiplying the proportion of current players
#  who have not yet placed a bet by the historical conditional probability.
def early_rate(early: pd.DataFrame, matured: pd.DataFrame, as_of: dt.date, brand_id: int) -> float:
    """The estimated churn rate of an early cohort (one cutoff, rows of one brand) on as_of: share with no
    bet yet x P(no bet in 60 days | no bet in the first k days) on `matured` (one complete cutoff)."""
    c_early, c_matured = early["cutoff_date"].iloc[0], matured["cutoff_date"].iloc[0]
    k = (as_of - c_early).days
    act = daily_cache.read("activity", brand_id, start=c_matured + dt.timedelta(days=1), end=as_of,
                           columns="tenant_id, player_id, day")
    act["day"] = pd.to_datetime(act["day"])

    def first_bet(rows: pd.DataFrame, cutoff: dt.date) -> pd.Series:
        """Days from the cutoff to each player's next bet seen (NaN: none up to as_of)."""
        c = pd.Timestamp(cutoff)
        days = (act[act["day"] > c].groupby(KEY)["day"].min() - c).dt.days
        return days.reindex(pd.MultiIndex.from_frame(rows[KEY])).reset_index(drop=True)

    silent_early = ~(first_bet(early, c_early) <= k)
    t = first_bet(matured, c_matured)
    silent_k, silent_60 = ~(t <= k), ~(t <= LABEL_HORIZON_DAYS)
    return float(silent_early.mean() * silent_60.sum() / max(silent_k.sum(), 1))


def fit(predict: Callable[[pd.DataFrame], np.ndarray], matured: pd.DataFrame, early: pd.DataFrame | None,
        as_of: dt.date) -> dict[int, float]:
    """One log-odds shift per brand. `matured`: rows of the last cutoffs with a complete label (with TARGET);
    `early`: rows of the later cutoffs (no label needed), or None."""
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
    """(matured rows with their label, early rows) for a scoring date: from the training dataset when it
    has the cutoff (its labels are complete there), otherwise rebuilt from the daily caches."""
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
