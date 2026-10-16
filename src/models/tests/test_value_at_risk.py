"""The value at risk handed off: expected 30-day GGR (0 when negative) x the LightGBM 60-day probability.

    .venv/bin/python -m pytest src/models/tests/test_value_at_risk.py -q
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score  # noqa: E402


def test_value_at_risk_uses_the_60_day_probability_and_no_negative_value():
    ggr = pd.Series([300.0, -50.0, 0.0])
    p60 = pd.Series([0.5, 0.9, 0.7])
    assert score.value_at_risk(ggr, p60).tolist() == [150.0, 0.0, 0.0]


def test_value_at_risk_is_the_scoring_column():
    assert "value_at_risk" in score.COLUMNS and "value_at_risk_30d" not in score.COLUMNS
