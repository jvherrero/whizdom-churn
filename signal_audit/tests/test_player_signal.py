"""A table with its own catalogue (player_signal_checks.yaml) on synthetic data with errors put in on purpose:
the recomputed checks find a column built with data of 7 days later, and the status of each column follows its
checks. No S3, no ClickHouse.

    .venv/bin/python -m pytest signal_audit/tests/test_player_signal.py -q
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from signal_audit.checks import formula_check as fc
from signal_audit.checks import recompute_signal as rs
from signal_audit.checks import row_evidence as evidence
from signal_audit.data import audit
from signal_audit.data import clickhouse_mirror as cm
from signal_audit.report import check_report

D = dt.date(2026, 9, 15)
CATALOGUE = rs.settings.CONFIG / "player_signal_checks.yaml"
KEY = ["tenant_id", "player_id"]


def _recency(days: list[dt.date], end: dt.date) -> int:
    return (end - max(d for d in days if d <= end)).days


def _streak(days: list[dt.date], end: dt.date) -> int:
    n = 0
    while end - dt.timedelta(days=n) in days:
        n += 1
    return n


def test_active_day_and_later_data(tmp_path, monkeypatch):
    """recency_score built with the days of 7 days later: best on D+7. active_streak_days built with every day, also
    the days with only playable bonus: it matches with any wallet, not with the official active days."""
    rng = np.random.default_rng(0)
    act, rows = [], []
    for p in range(10000, 10040):
        days = sorted({D - dt.timedelta(days=int(k)) for k in rng.choice(20, 12, replace=False)} |
                      {D + dt.timedelta(days=int(k)) for k in rng.choice(np.arange(1, 10), 2, replace=False)})
        bonus_only = {days[-3]} if days[-3] <= D else set()      # a day before D with only playable bonus
        act += [{"tenant_id": "t", "player_id": p, "day": pd.Timestamp(d), "active": d not in bonus_only} for d in days]
        rows.append({"tenant_id": "t", "player_id": p, "brand_id": 14, "snapshot_date": D, "typical_bet_gap_days": 5.0,
                     "recency_score": min(1, _recency(days, D + dt.timedelta(days=7)) / 5),     # 7 days later: wrong
                     "active_streak_days": _streak(days, D)})                                   # any wallet
    monkeypatch.setattr(cm.settings, "MIRROR", tmp_path / "mirror")
    cm.path(D).parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(cm.path(D), index=False)
    m = pd.DataFrame(rs.match_day(D, pd.DataFrame(act), fc.load(CATALOGUE)))
    s = rs.summary(m).set_index("column")
    assert s.loc["recency_score", "best shift"] == 7 and s.loc["recency_score", "match at best %"] == 100
    assert s.loc["recency_score", "match at D %"] < 50
    assert s.loc["active_streak_days", "variant"] == "any wallet" and s.loc["active_streak_days", "any wallet %"] == 100
    assert s.loc["active_streak_days", "official %"] < 100
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
    known = set(rs.FUNCTIONS) | {"days_since_bet", "active_days_7", "active_days_30", "active_streak", "gap_prev", "typical_bet_gap_days"}
    import re
    for e in ks.loc[ks["kind"] == "recomputed", "expr"]:
        assert set(re.findall(r"[a-z_][a-z0-9_]*", e)) <= known, e


def test_every_check_of_the_catalogue_has_only_known_fields():
    """An unquoted comma in a YAML flow mapping splits a field (was_on_tilt IN (0, 1) became two keys)."""
    import yaml
    for column, checks in yaml.safe_load(CATALOGUE.read_text())["columns"].items():
        for k in checks:
            assert set(k) <= {"kind", "expr", "measure", "note", "label"}, (column, k)


@pytest.fixture
def local_table(tmp_path, monkeypatch):
    """A local copy of the table with its own brand_id, read as the lake reads its files (no S3, no lake activity)."""
    rng = np.random.default_rng(1)
    n = 1000
    z = rng.normal(0, 1, n)
    right = np.clip(0.5 + 0.5 * np.clip(z, -1, 1), 0, 1)
    f = pd.DataFrame({"tenant_id": "t", "player_id": range(n), "snapshot_date": D, "brand_id": [14] * 600 + [15] * 400, "_version": 1,
                      "bet_z_score": z, "risk_appetite_score": np.where(np.arange(n) < 500, right, 1.0),      # half follow the formula
                      "rg_loss_chasing": np.r_[rng.uniform(0, 1, 700), np.full(100, 1.0), rng.uniform(1.2, 2.0, 200)]})
    folder = tmp_path / "mirror" / "gld_player_signals_daily" / f"{D:%Y/%m/%d}"
    folder.mkdir(parents=True)
    f.to_parquet(folder / "part-0.parquet", index=False)
    monkeypatch.setenv("DATALAKE_ROOT", str(tmp_path / "lake"))
    monkeypatch.setenv("RUN_S3", "1")
    monkeypatch.setenv("LOCAL_TABLES", f"gld_player_signals_daily={tmp_path / 'mirror'}")
    monkeypatch.setattr(audit, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(audit, "_CON", None)
    monkeypatch.setattr(audit.settings, "CATALOGUE", True)
    cfg = fc.load(CATALOGUE)
    cfg["snapshots"] = [str(D)]
    return cfg


def test_a_table_with_its_own_catalogue_keeps_all_its_rows_and_its_brand(local_table):
    sql = fc.rows_sql(local_table, [D], ["risk_appetite_score"])
    brands = audit.connection().sql(f"SELECT brand, count(*) AS n FROM ({sql}) GROUP BY ALL ORDER BY ALL").df()
    assert brands.to_dict("list") == {"brand": [14, 15], "n": [600, 400]}


def test_formula_pairs_put_the_rows_that_follow_the_formula_on_the_diagonal(local_table):
    expr = "least(1, greatest(0, 0.5 + 0.5 * least(1, greatest(-1, bet_z_score))))"
    p = evidence.formula_pairs(local_table, "risk_appetite_score", expr)
    assert p["n"].sum() == 1000
    assert p.loc[p["x"] == p["y"], "n"].sum() >= 500   # the half that follows the formula, and some of the others


def test_the_value_of_a_limit_is_valid_and_the_values_beyond_it_are_counted(local_table):
    h = evidence.value_hist(local_table, "rg_loss_chasing", 0.0, 1.0)
    width = (h["b"].iloc[0] - h["a"].iloc[0]) / h["bars"].iloc[0]
    beyond = h.loc[h["a"] + h["bin"] * width >= 1.0 - 1e-6, "n"].sum()
    assert h["n"].sum() == 1000 and beyond == 200          # the 100 rows at exactly 1 are valid
    assert h["at_hi"].iloc[0] == pytest.approx(0.1) and h["at_lo"].iloc[0] == 0
