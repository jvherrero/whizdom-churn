"""CHURN_DAYS changes the churn threshold for a sensitivity run, and its outputs get a suffix.

    .venv/bin/python -m pytest src/features/tests/test_churn_days.py -q
"""

import os
import subprocess
import sys
from pathlib import Path

FEATURES = Path(__file__).resolve().parents[1]


def _threshold(env_value: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "CHURN_DAYS"}
    if env_value is not None:
        env["CHURN_DAYS"] = env_value
    code = ("import churn_labels as c, build_features as b; "
            "print(c.HORIZON_DAYS, b.LABEL_HORIZON_DAYS, repr(c.VARIANT))")
    out = subprocess.run([sys.executable, "-c", code], cwd=FEATURES, env=env, capture_output=True, text=True)
    return out.stdout.strip() or out.stderr.strip().splitlines()[-1]


def test_default_is_60_days_with_no_suffix():
    assert _threshold(None) == "60 60 ''"
    assert _threshold("") == "60 60 ''"


def test_another_threshold_changes_labels_and_features_and_adds_a_suffix():
    assert _threshold("90") == "90 90 '_churn90d'"


def test_a_threshold_not_above_the_cox_horizons_is_refused():
    assert "CHURN_DAYS must be more than 30" in _threshold("30")
