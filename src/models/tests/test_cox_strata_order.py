"""Cox survival predictions keep the order of the input rows, also with several strata.

    .venv/bin/python -m pytest src/models/tests/test_cox_strata_order.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import BRAND, DURATION, EVENT, HORIZONS, raw_horizon_probability  # noqa: E402


def test_horizon_probability_follows_the_rows_with_two_strata():
    rng = np.random.default_rng(0)
    n = 600
    rows = pd.DataFrame({"x": rng.normal(size=n), BRAND: rng.choice([14, 64], n)})
    rows[DURATION] = np.ceil(rng.exponential(20 / np.exp(rows["x"])))
    rows[EVENT] = 1
    model = CoxPHFitter().fit(rows[["x", BRAND, DURATION, EVENT]], DURATION, EVENT, strata=[BRAND])
    p = raw_horizon_probability(model, rows)
    # within a brand, a higher x means a higher hazard, so the probability must follow x
    for _, part in rows.groupby(BRAND):
        assert np.all(np.diff(p.loc[part.sort_values("x").index, HORIZONS[-1]].to_numpy()) >= -1e-12)
    # and each row gets the prediction made for it alone
    for i in rows.sample(20, random_state=1).index:
        alone = 1 - model.predict_survival_function(rows.loc[[i], ["x", BRAND]], times=[HORIZONS[-1]]).iloc[0, 0]
        assert np.isclose(p.loc[i, HORIZONS[-1]], alone)
