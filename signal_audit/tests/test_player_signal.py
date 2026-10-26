"""A table with its own catalogue (player_signal_checks.yaml) on synthetic data with errors put in on purpose:
the recomputed checks find a column built with data of 7 days later, and the status of each column follows its
checks. No S3, no ClickHouse.

    .venv/bin/python -m pytest signal_audit/tests/test_player_signal.py -q
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_report  # noqa: E402
import clickhouse_mirror as cm  # noqa: E402
import formula_check as fc  # noqa: E402
import recompute_signal as rs  # noqa: E402

D = dt.date(2026, 9, 15)
CATALOGUE = Path(rs.__file__).parent / "player_signal_checks.yaml"
KEY = ["tenant_id", "player_id"]


def _recency(days: list[dt.date], end: dt.date) -> int:
    return (end - max(d for d in days if d <= end)).days


def test_recomputed_checks_find_data_from_the_future(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    players = range(10000, 10040)
    act, rows = [], []
    for p in players:
        days = sorted({D - dt.timedelta(days=int(k)) for k in rng.choice(60, 8, replace=False)} |
                      {D + dt.timedelta(days=int(k)) for k in rng.choice(np.arange(1, 10), 2, replace=False)})
        act += [{"tenant_id": "t", "brand_id": 14, "player_id": p, "day": pd.Timestamp(d), "bets": 2, "rounds": 2,
                 "turnover_eur": 10.0, "ggr_eur": 1.0, "active": True} for d in days]
        rows.append({"tenant_id": "t", "player_id": p, "brand_id": 14, "snapshot_date": D, "typical_bet_gap_days": 5.0,
                     "churn_propensity_raw": min(1, _recency(days, D + dt.timedelta(days=7)) / 90),   # 7 days later: wrong
                     "recency_score": min(1, _recency(days, D) / 5),                           # right
                     "deposit_recency_score": 1 - min(1, _recency(days, D) / 30)})             # right, no deposits
    monkeypatch.setattr(cm.settings, "MIRROR", tmp_path / "mirror")
    cm.path(D).parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(cm.path(D), index=False)
    pay = pd.DataFrame({"tenant_id": ["t"], "player_id": [1], "day": [pd.Timestamp(D)], "deposit_eur": [5.0]})
    m = pd.DataFrame(rs.match_day(14, D, pd.DataFrame(act), pay, fc.load(CATALOGUE)))
    s = rs.summary(m).set_index("column")
    assert s.loc["recency_score", "match at D %"] == 100 and s.loc["deposit_recency_score", "match at D %"] == 100
    assert s.loc["churn_propensity_raw", "match at D %"] < 50
    assert s.loc["churn_propensity_raw", "best shift"] == 7 and s.loc["churn_propensity_raw", "match at best %"] == 100
    assert "frequency_score" not in s.index   # not in the table: not compared


def test_the_status_of_each_column():
    ks = fc.checks({"columns": {
        "churn_score": [{"kind": "formula", "expr": "x", "measure": "M107"}],
        "recency_score": [{"kind": "bound", "expr": "x"}, {"kind": "recomputed", "expr": "x", "measure": "M132"}],
        "engagement_score": [{"kind": "bound", "expr": "x"}, {"kind": "none", "measure": "M116"}],
        "tilt_events": [{"kind": "bound", "expr": "x"}], "_version": [{"kind": "none"}]}})
    res = pd.DataFrame({"column": ["churn_score", "recency_score", "engagement_score", "tilt_events"], "n_check": [0, 0, 0, 0],
                        "pass %": [97.0, 100.0, 100.0, 100.0]})
    rec = pd.DataFrame({"column": ["recency_score"], "n_check": [1], "match at D %": [100.0]})
    s = check_report.statuses(ks, res, rec).set_index("column")["status"].to_dict()
    assert s == {"churn_score": "wrong", "recency_score": "correct", "engagement_score": "unconfirmed",
                 "tilt_events": "correct", "_version": "unchecked"}


def test_the_catalogue_names_only_known_inputs():
    ks = fc.checks(fc.load(CATALOGUE))
    known = set(rs.FUNCTIONS) | {"days_since_bet", "active_days_7", "active_days_30", "ggr_today", "ggr_7", "ggr_30", "ggr_90",
                                 "wagered_30", "avg_bet_7", "avg_bet_30", "deposits_today", "deposits_7", "deposits_30",
                                 "deposits_90", "ggr_prev", "active_streak", "gap_prev", "tenure", "typical_bet_gap_days"}
    import re
    for e in ks.loc[ks["kind"] == "recomputed", "expr"]:
        assert set(re.findall(r"[a-z_][a-z0-9_]*", e)) <= known, e


def test_every_check_of_the_catalogue_has_only_known_fields():
    """An unquoted comma in a YAML flow mapping splits a field (was_on_tilt IN (0, 1) became two keys)."""
    import yaml
    for column, checks in yaml.safe_load(CATALOGUE.read_text())["columns"].items():
        for k in checks:
            assert set(k) <= {"kind", "expr", "measure", "note"}, (column, k)
