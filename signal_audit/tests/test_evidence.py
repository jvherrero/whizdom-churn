"""evidence.py on small synthetic data (no S3): repeated time-of-play days, and the session, index-limit
and hold checks on a small synthetic signals table.

    .venv/bin/python -m pytest signal_audit/tests -q
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from signal_audit.data import audit
from signal_audit.checks import evidence as ev

D = [dt.date(2026, 8, 20) + dt.timedelta(days=i) for i in range(4)]


def test_repeated_days_need_the_same_counts_and_a_changing_control():
    rows = []
    for i, d in enumerate(D):
        night = (100, 10) if i in (1, 2) else (200 + i, 20 + i)       # the third day repeats the second
        rows += [{"day": d, "brand_id": 14, "brand_filled": False, "metric": "night_index", "rows": night[0], "impossible": night[1]},
                 {"day": d, "brand_id": 14, "brand_filled": False, "metric": "active_days_l30d", "rows": 50 + i, "impossible": 0}]
    out = ev.repeated_days(pd.DataFrame(rows))
    assert out["day"].tolist() == [D[2]]
    assert out["active_days_l30d changes"].all()


DAY = dt.date(2026, 9, 1)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    n = 100
    rows = pd.DataFrame({
        "tenant_id": "op", "brand_id": np.repeat([14, 64], n // 2), "player_id": np.arange(n), "snapshot_date": DAY,
        "tp_avg_session_duration_secs": 600.0, "tp_median_session_duration_secs": 500.0, "tp_avg_session_gap_hours": 20.0,
        "tp_avg_sessions_per_active_day": 2.0, "tp_window_days": 30, "sessions_l7d": 4, "sessions_l30d": 10,
        "play_secs_today": 1200.0, "play_secs_l7d": 2400.0, "play_secs_l30d": 6000.0,   # 6000 / 10 = the mean
        "churn_score": 50.0, "value_decline": 20.0,
        "hold_pct_l30d": 7.0, "bets_l30d": 200, "wagered_eur_l30d": 100.0, "won_eur_l30d": 93.0, "ggr_eur_l30d": 7.0,
        "_version": 1})
    rows.loc[0, ["tp_avg_session_duration_secs", "tp_median_session_duration_secs"]] = 2 * 86400   # never closes
    rows.loc[1, "tp_avg_session_gap_hours"] = -5                                                    # overlap
    rows.loc[2, "play_secs_today"] = 90000                                                          # over a day
    rows.loc[3, "value_decline"] = 5e6                                                              # no clamp
    rows.loc[4:5, ["hold_pct_l30d", "bets_l30d", "won_eur_l30d"]] = [100.0, 1, 0.0]                 # lost the only bet
    rows.loc[6, ["hold_pct_l30d", "bets_l30d"]] = [-450.0, 1]                                       # a big win
    folder = tmp_path / "lake" / "gld_player_signals_daily" / f"{DAY:%Y/%m/%d}"
    folder.mkdir(parents=True)
    rows.to_parquet(folder / "part-0.parquet", index=False)
    monkeypatch.setenv("DATALAKE_ROOT", str(tmp_path / "lake"))
    monkeypatch.setenv("RUN_S3", "1")
    monkeypatch.setattr(audit, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(audit, "_CON", None)
    return {"period": {"start": DAY, "end": DAY}, "brands": [14, 64],
            "tables": {"gld_player_signals_daily": {"date": "snapshot_date", "key": ["tenant_id", "player_id", "snapshot_date"]}},
            "metrics": {}}


def test_session_checks(cfg):
    c = ev.session_checks(cfg).set_index(["check", "brand"])["holds %"]
    assert c[("mean = play_secs / sessions (30 days)", "all")] == pytest.approx(99)   # all but the session of 2 days
    assert c[("mean over 1 day", "14")] == pytest.approx(2)
    assert c[("gap between sessions < 0", "all")] == pytest.approx(1)
    assert c[("play_secs today > 24 h", "all")] == pytest.approx(1)


def test_index_limits(cfg):
    limits = {"churn_score": (0, 100, ""), "value_decline": (0, 100, ""), "hold_pct_l30d": (None, 100, "")}
    summary, hist = ev.index_limits(cfg, limits)
    s = summary.set_index(["column", "brand"])
    assert s.loc[("value_decline", "all"), "above %"] == pytest.approx(1)
    assert s.loc[("hold_pct_l30d", "all"), "at upper limit %"] == pytest.approx(2)
    assert hist.loc[hist["column"] == "value_decline", "bin"].max() == ev.BINS        # the value above the limit
    assert hist.groupby("column")["count"].sum().eq(100).all()


def test_hold_masses_by_bets(cfg):
    m = ev.hold_masses(cfg).set_index(["brand", "bets_30d"])
    assert m.loc[("all", "1 bet"), "exactly 100"] == 2 and m.loc[("all", "1 bet"), "below -100"] == 1
    assert m.loc[("all", "all"), "rows"] == 100 and m.loc[("all", "over 100"), "exactly 100"] == 0
    h = ev.hold_hist(cfg)
    assert h["count"].sum() == 100
