"""Isotonic calibration per brand, with a calibration over every brand as the fallback.

Brands have different churn rates, so one calibration for all of them would be wrong for each
brand on its own. A brand gets its own isotonic regression when it has at least MIN_ROWS
validation rows with both classes; otherwise (and for a brand never seen in training) the one
fitted on every brand is used.

It lives in its own module (not in train.py) so the pickled object loads the same way from
train.py, score.py and feature_importance.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

MIN_ROWS = 500


class BrandCalibrator:
    def __init__(self, bound: float, min_rows: int = MIN_ROWS):
        # Isotonic regression gives exactly 0 or 1 at the tails; bound it to [bound, 1 - bound].
        self.bound, self.min_rows = bound, min_rows

    def _isotonic(self) -> IsotonicRegression:
        return IsotonicRegression(y_min=self.bound, y_max=1 - self.bound, out_of_bounds="clip")

    def fit(self, raw: np.ndarray, y, brands) -> "BrandCalibrator":
        raw, y, brands = np.asarray(raw), np.asarray(y), np.asarray(brands)
        self.overall_ = self._isotonic().fit(raw, y)
        self.per_brand_ = {}
        for brand in np.unique(brands):
            rows = brands == brand
            if rows.sum() >= self.min_rows and len(np.unique(y[rows])) == 2:
                self.per_brand_[int(brand)] = self._isotonic().fit(raw[rows], y[rows])
        return self

    def predict(self, raw: np.ndarray, brands) -> np.ndarray:
        raw, brands = np.asarray(raw), np.asarray(brands)
        out = self.overall_.predict(raw)
        for brand, iso in self.per_brand_.items():
            rows = brands == brand
            if rows.any():
                out[rows] = iso.predict(raw[rows])
        return out

    def describe(self) -> pd.DataFrame:
        """Which brands have their own calibration."""
        return pd.DataFrame({"brandId": sorted(self.per_brand_), "calibration": "own isotonic"})
