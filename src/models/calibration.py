"""Probability calibration per brand, fitted on the validation rows.

    isotonic   monotone step function, needs more rows
    platt      logistic curve on the log-odds of the raw score

Each brand with at least MIN_ROWS rows of both classes gets its own calibration. Other brands use
one calibration fitted on all brands. It is a separate module so the pickled object loads the same
way everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

MIN_ROWS = 500
METHODS = ("isotonic", "platt")


class _Platt:
    """Platt scaling: logistic regression on the log-odds of the raw score."""

    def __init__(self, bound: float):
        self.bound = bound

    @staticmethod
    def _logit(raw: np.ndarray) -> np.ndarray:
        p = np.clip(raw, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw, y) -> "_Platt":
        self.model_ = LogisticRegression(C=1e6).fit(self._logit(raw), y)  # almost no penalty
        return self

    def predict(self, raw) -> np.ndarray:
        return np.clip(self.model_.predict_proba(self._logit(raw))[:, 1], self.bound, 1 - self.bound)


class BrandCalibrator:
    def __init__(self, bound: float, min_rows: int = MIN_ROWS, method: str = "isotonic"):
        if method not in METHODS:
            raise ValueError(f"calibration method {method!r} not in {METHODS}")
        # Never return exactly 0 or 1.
        self.bound, self.min_rows, self.method = bound, min_rows, method

    def _new(self):
        if self.method == "platt":
            return _Platt(self.bound)
        return IsotonicRegression(y_min=self.bound, y_max=1 - self.bound, out_of_bounds="clip")

    def fit(self, raw: np.ndarray, y, brands) -> "BrandCalibrator":
        raw, y, brands = np.asarray(raw), np.asarray(y), np.asarray(brands)
        self.overall_ = self._new().fit(raw, y)
        self.per_brand_ = {}
        for brand in np.unique(brands):
            rows = brands == brand
            if rows.sum() >= self.min_rows and len(np.unique(y[rows])) == 2:
                self.per_brand_[int(brand)] = self._new().fit(raw[rows], y[rows])
        return self

    def predict(self, raw: np.ndarray, brands) -> np.ndarray:
        raw, brands = np.asarray(raw), np.asarray(brands)
        out = self.overall_.predict(raw)
        for brand, calibrator in self.per_brand_.items():
            rows = brands == brand
            if rows.any():
                out[rows] = calibrator.predict(raw[rows])
        return out

    def describe(self) -> pd.DataFrame:
        """Brands with their own calibration."""
        return pd.DataFrame({"brandId": sorted(self.per_brand_), "calibration": f"own {self.method}"})
