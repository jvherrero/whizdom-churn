"""gaming.py and gaming.yaml on a small synthetic gld_player_gaming_daily with errors put in on purpose (no S3).

    .venv/bin/python -m pytest signal_audit/tests -q
"""

import datetime as dt

import duckdb
import numpy as np
import pandas as pd
import pytest

from signal_audit.data import audit
from signal_audit.checks import gaming
from signal_audit.checks import stats

DAYS = [dt.date(2026, 9, 1), dt.date(2026, 9, 2)]
N = 200   # rows per brand and day


def _rows(day, brand, rng):
    n = N
    w = rng.lognormal(3, 1, n).round(2)
    won = (w * rng.uniform(0.5, 1.2, n)).round(2)
    t0 = pd.Timestamp(day, tz="UTC") + pd.to_timedelta(rng.integers(0, 40000, n), unit="s")
    f = pd.DataFrame({
        "tenant_id": "op", "brand_id": brand, "player_id": brand * 1000 + np.arange(n), "game_id": "g1", "provider_code": 1,
        "event_date": day, "currency": "EUR", "fx_rate_to_eur": 1.0, "fx_rate_date_used": day,
        "wagered_real": w, "wagered_real_eur": w, "wagered_released_bonus": 0.0, "wagered_released_bonus_eur": 0.0,
        "wagered_playable_bonus": 0.0, "wagered_playable_bonus_eur": 0.0,
        "won_real": won, "won_real_eur": won, "won_released_bonus": 0.0, "won_released_bonus_eur": 0.0,
        "won_playable_bonus": 0.0, "won_playable_bonus_eur": 0.0, "tax_real": 0.0, "tax_real_eur": 0.0,
        "tax_released_bonus_eur": 0.0, "tax_playable_bonus_eur": 0.0, "tax_eur": 0.0,
        "turnover_eur": w, "payout_eur": won, "ggr_eur": w - won, "real_ggr_eur": w - won, "bonus_ggr_eur": 0.0,
        "net_position_eur": won - w, "hold_pct": 100 * (w - won) / w, "rtp_actual": 100 * won / w,
        "bets": 10, "bet_count_real": 10, "bet_count_released_bonus": 0, "bet_count_playable_bonus": 0,
        "rounds": 10, "winning_rounds": 4, "big_win_rounds": 0, "win_rate": 0.4, "avg_bet_eur": w / 10,
        "event_count": 20, "max_win_eur": won / 4, "loyalty_points_delta": 0, "dq_status": "ok", "fx_is_provisional": False,
        "source_system": "s", "play_secs": 600.0, "first_played_at": t0, "last_played_at": t0 + pd.Timedelta(seconds=900),
        "_version": 1})
    return f


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    frames = []
    for d in DAYS:
        for b in (14, 64):
            frames.append(_rows(d, b, rng))
    f = pd.concat(frames, ignore_index=True)
    f.loc[0, ["won_real", "won_real_eur", "payout_eur"]] = -5.0          # a negative win
    f.loc[1, "play_secs"] = 90000.0                                       # more than a day in one row
    f.loc[2, "fx_rate_to_eur"] = 0.0                                      # a zero FX rate
    f.loc[3, "turnover_eur"] = f.loc[3, "wagered_real_eur"] + 100         # turnover not the sum of the stake
    f.loc[5, ["wagered_real", "wagered_real_eur", "turnover_eur"]] = 2_000_000.0   # big enough to overflow DECIMAL(18)
    late = pd.Timestamp(f.loc[6, "event_date"], tz="UTC") + pd.Timedelta(hours=23, minutes=30)
    f.loc[6, ["first_played_at", "last_played_at"]] = [late, late + pd.Timedelta(minutes=10)]   # late in the UTC day
    f.loc[7, "first_played_at"] = pd.Timestamp(f.loc[7, "event_date"], tz="UTC") - pd.Timedelta(hours=1)   # the day before
    old = f.iloc[[4]].assign(_version=0, turnover_eur=1e9)                # an older version of a row
    money = [c for c in f.columns if c.endswith("_eur") or c.startswith(("wagered_", "won_", "tax_"))]
    money = [c for c in money if c not in ("hold_pct", "rtp_actual", "fx_rate_to_eur")]
    for d, rows in pd.concat([f, old]).groupby("event_date"):
        folder = tmp_path / "lake" / "gld_player_gaming_daily" / f"{d:%Y/%m/%d}"
        folder.mkdir(parents=True)
        # the same types as the real table: money DECIMAL(18, 4), FX rate DECIMAL(18, 8)
        casts = ", ".join([f"CAST({c} AS DECIMAL(18, 4)) AS {c}" for c in money] + ["CAST(fx_rate_to_eur AS DECIMAL(18, 8)) AS fx_rate_to_eur"])
        duckdb.sql(f"COPY (SELECT * REPLACE ({casts}) FROM rows) TO '{folder / 'part-0.parquet'}' (FORMAT PARQUET)")
    monkeypatch.setenv("DATALAKE_ROOT", str(tmp_path / "lake"))
    monkeypatch.setenv("RUN_S3", "1")
    monkeypatch.setattr(audit, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(audit, "_CON", None)
    c = gaming.load_config()
    c["period"] = {"start": DAYS[0], "end": DAYS[-1]}
    c["brands"] = [14, 64]
    return c


def test_row_checks_find_the_errors_put_in(cfg):
    s = gaming.checks_summary(gaming.row_checks(cfg)).set_index("check")
    n = 2 * N * len(DAYS)
    assert s.loc["won_real >= 0", "pass %"] == pytest.approx(100 * (n - 1) / n)
    assert s.loc["play_secs <= 86400", "pass %"] == pytest.approx(100 * (n - 1) / n)
    assert s.loc["fx_rate_to_eur > 0", "pass %"] == pytest.approx(100 * (n - 1) / n)
    turnover = s.loc["turnover_eur = wagered_real_eur + wagered_released_bonus_eur + wagered_playable_bonus_eur"]
    assert turnover["rows"] == n and turnover["higher %"] == pytest.approx(100 / n)   # the old version was dropped
    assert s.loc["wagered_real_eur = wagered_real * fx_rate_to_eur", "verdict"] == "passes"   # no DECIMAL overflow
    assert s.loc["bets = bet_count_real + bet_count_released_bonus + bet_count_playable_bonus", "verdict"] == "passes"


def test_key_check_counts_the_older_version(cfg):
    k = gaming.key_check(cfg)
    assert k["keys_with_versions"].sum() == 1 and k["rows"].sum() == 2 * N * len(DAYS) + 1


def test_impossible_values_and_player_day_metrics(cfg):
    flags = audit.flags_by_day(cfg, "gaming_rows")
    s = audit.flags_summary(flags, cfg).set_index("metric")
    assert s.loc["row_play_secs", "impossible"] == 1 and s.loc["row_fx_rate", "impossible"] == 1
    prof, hist = audit.metric_data(cfg, "pd_play_secs")
    assert prof["n"] == 2 * N * len(DAYS)
    card = stats.brand_scorecard(cfg, ["row_turnover_eur"])
    assert set(card["brand"]) == {"14", "64"}


def test_schema_diff_reads_the_workbook():
    wb = gaming.workbook_schema()
    assert {"wagered_real_eur", "play_secs", "fx_rate_to_eur"} <= set(wb["column"])


def test_dates_are_utc_days(cfg):
    """A play at 23:30 UTC is on its UTC day, whatever the time zone of the machine."""
    s = gaming.checks_summary(gaming.row_checks(cfg)).set_index("check")
    n = 2 * N * len(DAYS)
    assert s.loc["CAST(first_played_at AS DATE) = event_date", "pass %"] == pytest.approx(100 * (n - 1) / n)
    h = gaming.first_play_hours(cfg)
    other = h[h["day_shift"] != 0]
    assert other["rows"].sum() == 1 and other["hour"].iloc[0] == 23 and other["day_shift"].iloc[0] == 1
    assert h["rows"].sum() == n


def test_a_failing_hypothesis_is_rejected_when_another_definition_passes():
    def check(label, src, group, ok):
        return {"day": DAYS[0], "brand_id": 14, "check": label, "column": "c", "kind": "identity", "source": src,
                "group": group, "note": "", "n": 100, "ok": ok, "higher": 100 - ok, "lower": 0}
    checks = pd.DataFrame([check("all wallets", "hypothesis", "c", 100), check("real only", "hypothesis", "c", 60),
                           check("workbook rule", "workbook", "c", 60), check("alone", "hypothesis", "d", 60),
                           check("almost alone", "hypothesis", "e", 97)])
    v = gaming.checks_summary(checks).set_index("check")["verdict"]
    assert v.to_dict() == {"all wallets": "passes", "real only": "rejected", "workbook rule": "fails", "alone": "rejected",
                           "almost alone": "almost"}


def test_anomalies_are_aggregates_of_the_bad_rows(cfg):
    cond = "play_secs > 86400"
    by = gaming.anomaly_by(cfg, cond, "brand_id")
    assert by["bad_rows"].sum() == 1 and by["rows"].sum() == 2 * N * len(DAYS) and "player_id" not in by
    p = gaming.anomaly_players(cfg, cond).iloc[0]
    assert p["players"] == 1 and p["top1_pct"] == 100 and p["players_1_day"] == 1
    v = gaming.anomaly_values(cfg, cond, "play_secs")
    assert v["value"].tolist() == [90000.0] and v["rows"].tolist() == [1]
    q = gaming.anomaly_quantiles(cfg, cond, {"secs": "play_secs"})
    assert q.loc[("secs", True), "p50"] == 90000 and q.loc[("secs", False), "p50"] == 600


def test_a_list_of_days_reads_only_those_days(cfg):
    one_day = {**cfg, "days": [DAYS[1]]}
    assert gaming.anomaly_by(one_day, "play_secs > 86400", "event_date")["rows"].sum() == 2 * N
