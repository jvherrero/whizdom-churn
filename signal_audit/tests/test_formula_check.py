"""Formula validation of the signals table on a small synthetic table (no S3).

    .venv/bin/python -m pytest signal_audit/tests/test_formula_check.py -q

Two players of brand 14 and one of brand 99 (out of scope), every day from 1 to 8 September 2026.
Player 1 follows every definition. Player 2 has a 7-day stake that is not the sum of its daily stake,
an active-day count above 7, and a frequency score that does not follow its formula.
"""

import datetime as dt

import pandas as pd
import pytest

from signal_audit import settings
from signal_audit.data import audit
from signal_audit.checks import formula_check as fc

DAYS = [dt.date(2026, 9, 1) + dt.timedelta(days=i) for i in range(8)]
SNAP = DAYS[-1]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    root = tmp_path / "lake"
    for i, d in enumerate(DAYS):
        rows = []
        for p, brand in ((1, 14), (2, 14), (3, 99)):
            today = 10.0 * p
            l7d = today * min(i + 1, 7) + (5.0 if p == 2 else 0.0)        # player 2: off by 5
            a7, a30 = (7, 10) if p != 2 else (9, 10)                        # player 2: 9 active days in 7
            freq = 100 * max(0, 1 - (a7 / 7) / (a30 / 30))
            rows.append({"tenant_id": "op", "player_id": p, "snapshot_date": d, "brand_id": None, "_version": 1,
                         "wagered_eur_today": today, "wagered_eur_l7d": l7d,
                         "prior_wagered_eur_l7d": 10.0 * p * min(i - 6, 7) if i >= 7 else 0.0,
                         "active_days_l7d": a7, "active_days_l30d": a30,
                         "frequency_decline": freq if p != 2 else freq + 50,
                         "avg_bet_eur_l30d": float(p), "bet_z_score": 0.0,
                         "churn_score": 20.0 * p, "churn_band": ["Low", "Medium", "High"][p - 1]})
        folder = root / "gld_player_signals_daily" / f"{d:%Y/%m/%d}"
        folder.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(folder / "data.parquet", index=False)
    activity = tmp_path / "activity"
    activity.mkdir()
    pd.DataFrame({"tenant_id": "op", "player_id": [1, 2, 3], "brand_id": [14, 14, 99]}).to_parquet(activity / "a.parquet")
    monkeypatch.setenv("DATALAKE_ROOT", str(root))
    monkeypatch.setenv("RUN_S3", "1")
    monkeypatch.setattr(audit, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(audit, "ACTIVITY", activity)
    monkeypatch.setattr(audit, "_CON", None)
    return {
        "table": "gld_player_signals_daily", "snapshots": [str(SNAP)], "brands": [14],
        "tolerance": {"atol": 0.01, "rtol": 0.001}, "verdict": {"good": 0.99, "fair": 0.95},
        "columns": {
            "wagered_eur_l7d": [{"kind": "window", "base": "wagered_eur_today", "days": 7, "note": ""}],
            "prior_wagered_eur_l7d": [{"kind": "lag", "source": "wagered_eur_l7d", "lag_days": 7, "note": ""}],
            "active_days_l7d": [{"kind": "bound", "expr": "active_days_l7d <= 7", "note": ""}],
            "frequency_decline": [{"kind": "formula", "note": "",
                                   "expr": "100 * greatest(0, 1 - (active_days_l7d / 7.0) / (active_days_l30d / 30.0))"}],
            "bet_z_score": [{"kind": "formula", "tenant_z": "avg_bet_eur_l30d", "note": ""}],
            "churn_band": [{"kind": "band", "score": "churn_score", "cutoffs": [25, 50, 75], "note": ""}],
        },
    }


def _row(res, column):
    return res.set_index("column").loc[column]


def test_only_the_brands_in_scope(cfg):
    r = fc.day_checks(cfg, SNAP)
    assert (r["rows"] == 2).all()                      # players 1 and 2, not player 3 of brand 99


def test_formula_bound_and_z_score(cfg):
    res = fc.verdicts(cfg, fc.day_checks(cfg, SNAP))
    f = _row(res, "frequency_decline")
    assert (f["n"], f["ok"], f["higher"]) == (2, 1, 1)
    b = _row(res, "active_days_l7d")
    assert (b["n"], b["ok"]) == (2, 1)
    z = _row(res, "bet_z_score")                       # stored 0 for both, but the z-scores are -1 and +1
    assert z["ok"] == 0


def test_window_and_lag(cfg):
    res = fc.verdicts(cfg, fc.history_checks(cfg, SNAP))
    w = _row(res, "wagered_eur_l7d")
    assert (w["n"], w["ok"], w["higher"]) == (2, 1, 1)   # player 2 is 5 above the sum of its 7 days
    lag = _row(res, "prior_wagered_eur_l7d")
    assert lag["n"] == 2


def test_band_ranges(cfg):
    b = fc.band_checks(cfg, SNAP)
    assert set(zip(b["band"], b["bucket"])) == {("Low", 0), ("Medium", 1)}


def test_the_real_catalogue_covers_every_column():
    import json
    inventory = settings.PROJECT_ROOT / "docs/dq_reports/layer_inventory.json"
    prod = [c for t in json.loads(inventory.read_text()) if t["table"] == "gld_player_signals_daily" for c, _ in t["schema"]]
    assert set(prod) <= set(fc.load()["columns"])


def test_a_bound_left_null_is_no_data(cfg):
    # 0 means no data: the rows are left out of the check instead of counted as failing
    cfg["columns"]["churn_score"] = [{"kind": "bound", "note": "",
                                      "expr": "CASE WHEN churn_score > 20 THEN churn_score <= 30 END"}]
    res = fc.verdicts(cfg, fc.day_checks(cfg, SNAP)).set_index("column").loc["churn_score"]
    assert (res["n"], res["ok"]) == (1, 0)          # player 1 (20) left out; player 2 (40) fails
