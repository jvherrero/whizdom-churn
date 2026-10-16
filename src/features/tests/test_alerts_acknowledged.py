"""Reviewed alerts (`acknowledged` in configs/eda_alerts.yaml) become warnings, only inside their limits.

    .venv/bin/python -m pytest src/features/tests/test_alerts_acknowledged.py -q
"""

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import alerts  # noqa: E402

ENTRY = {"stage": "source", "check": "row_count", "brandId": 72, "table": "payments", "from": dt.date(2026, 10, 6),
         "until": dt.date(2026, 11, 15), "max_value": 3500, "reason": "brand growth"}


def found(value=2531, brand=72, table="payments", check="row_count"):
    return pd.DataFrame([{"stage": "source", "check": check, "severity": "critical", "brandId": brand,
                          "table": table, "value": value, "message": "too many rows"}])


def severity(frame, as_of="2026-10-06", stage="source"):
    return alerts.acknowledge(frame, stage, as_of, {"acknowledged": [ENTRY]})["severity"].tolist()


def test_reviewed_alert_becomes_a_warning_with_its_reason():
    out = alerts.acknowledge(found(), "source", "2026-10-06", {"acknowledged": [ENTRY]})
    assert out["severity"].tolist() == ["warn"] and "acknowledged until 2026-11-15: brand growth" in out["message"][0]


def test_anything_outside_the_entry_stays_critical():
    assert severity(found(value=4000)) == ["critical"]             # a bigger jump
    assert severity(found(), as_of="2026-11-16") == ["critical"]   # after `until`
    assert severity(found(), as_of="2026-10-05") == ["critical"]   # before `from`
    assert severity(found(brand=64)) == ["critical"]               # another brand
    assert severity(found(table="activity")) == ["critical"]       # another table
    assert severity(found(check="null_rate")) == ["critical"]      # another check
    assert severity(found(), stage="features") == ["critical"]     # another stage


def test_no_entries_change_nothing():
    assert alerts.acknowledge(found(), "source", "2026-10-06", {})["severity"].tolist() == ["critical"]


def test_the_project_config_parses():
    entries = alerts.load_config()["acknowledged"]
    assert all({"stage", "check", "brandId", "table", "from", "until", "max_value", "reason"} <= set(e) for e in entries)
