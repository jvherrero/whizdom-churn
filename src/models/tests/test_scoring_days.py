"""Scoring days of the sensitivity check: out of sample and with a closed 60-day label only.

    .venv/bin/python -m pytest src/models/tests/test_scoring_days.py -q
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scoring_day_sensitivity as s  # noqa: E402


def test_days_start_at_the_first_test_cutoff_and_need_a_closed_label():
    days = s.scoring_days(dt.date(2026, 7, 1), dt.date(2026, 10, 6))
    assert days == [dt.date(2026, 7, d) for d in (1, 8, 15, 22, 29)] + [dt.date(2026, 8, 1), dt.date(2026, 8, 8)][:1]
    assert all(d + dt.timedelta(days=60) <= dt.date(2026, 10, 6) for d in days)


def test_no_day_before_the_first_test_cutoff():
    days = s.scoring_days(dt.date(2026, 7, 10), dt.date(2026, 10, 6))
    assert min(days) == dt.date(2026, 7, 15)
