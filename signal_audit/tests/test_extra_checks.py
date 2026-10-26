"""extra_checks.py on a small synthetic gld_player_signals_daily with errors put in on purpose (no S3).

    .venv/bin/python -m pytest signal_audit/tests -q
"""

import datetime as dt

import pandas as pd
import pytest

from signal_audit.data import audit
from signal_audit.checks import extra_checks as x

DAY = dt.date(2026, 9, 1)
N = 50
TENANT = "73c25416-d9de-46fb-8a73-2e64f63d6330"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    t0 = pd.Timestamp(DAY + dt.timedelta(days=8), tz="UTC") + pd.Timedelta(hours=3)
    f = pd.DataFrame({
        "tenant_id": TENANT, "player_id": range(1000, 1000 + N), "snapshot_date": DAY, "brand_id": 14, "_version": 1,
        "_ingested_at": t0, "_processed_at": t0 + pd.Timedelta(minutes=5),
        "country": "TR", "currency": "TRY", "language": "tr", "lifecycle_stage": ["active"] * 40 + ["at_risk"] * 10,
        "value_tier": "bronze", "lifetime_value_tier": "bronze", "churn_band": "low", "risk_band": "low",
        "prev_lifecycle_stage": "active", "prev_value_tier": "bronze", "prev_churn_band": "low", "prev_risk_band": "low",
        "param_version": "v1", "days_since_bet": 1,
        **{f"{c}_{w}": v for w in ("today", "l90d", "l180d") for c, v in (("tax_eur", 1.0), ("ggr_eur", 10.0), ("wagered_eur", 100.0))}})
    f.loc[0, "country"] = "tr"                                          # mixed case
    f.loc[1, "tenant_id"] = TENANT.upper()                              # tenant id not in lower case
    f.loc[2, "_processed_at"] = f.loc[2, "_ingested_at"] + pd.Timedelta(minutes=20)   # slow processing
    f.loc[3, "tax_eur_today"] = 50.0                                    # tax above the GGR
    f["play_secs_today"] = 600.0
    f["tp_avg_session_duration_secs"] = 900.0
    f["tp_avg_session_gap_hours"] = 5.0
    f.loc[0:9, "play_secs_today"] = 100000.0                            # over 24 h: the sports players below
    f.loc[0:9, "tp_avg_session_duration_secs"] = 100000.0
    f["bonus_ggr_eur_today"] = 2.0
    f.loc[4, "bonus_ggr_eur_today"] = 3.0                               # not the sum of the gaming table
    g = pd.DataFrame({"tenant_id": f["tenant_id"], "player_id": f["player_id"], "brand_id": 14, "event_date": DAY,
                      "currency": "TRY", "provider_code": 1, "_version": 1, "bets": 1, "bonus_ggr_eur": 1.0})
    g = pd.concat([g.assign(game_id="a"), g.assign(game_id="b")])      # two games of 1.0 each: 2.0 per player-day
    g.loc[g["player_id"] < 1010, "provider_code"] = 58                  # players 0 to 9 bet on sports only
    gfolder = tmp_path / "lake" / "gld_player_gaming_daily" / f"{DAY:%Y/%m/%d}"
    gfolder.mkdir(parents=True)
    g.to_parquet(gfolder / "part-0.parquet", index=False)
    folder = tmp_path / "lake" / "gld_player_signals_daily" / f"{DAY:%Y/%m/%d}"
    folder.mkdir(parents=True)
    f.to_parquet(folder / "part-0.parquet", index=False)
    activity = tmp_path / "activity"
    activity.mkdir()
    f[["tenant_id", "player_id", "brand_id"]].to_parquet(activity / "2026-09-01.parquet", index=False)
    monkeypatch.setenv("DATALAKE_ROOT", str(tmp_path / "lake"))
    monkeypatch.setenv("RUN_S3", "1")
    monkeypatch.setattr(audit, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(audit, "ACTIVITY", activity)
    monkeypatch.setattr(audit, "_CON", None)
    c = audit.load_config()
    c["period"] = {"start": DAY, "end": DAY}
    c["brands"] = [14]
    return c


def test_load_times(cfg):
    t = x.load_times(cfg).iloc[0]
    assert t["rows"] == N and t["within_10_min"] == N - 1 and t["processed_before_ingested"] == 0
    assert t["process_minus_ingest_min"][1] == pytest.approx(5)
    assert t["ingest_after_day_end_h"][1] == pytest.approx(7 * 24 + 3)    # loaded on D + 8 at 03:00


def test_categories_and_formats(cfg):
    v = x.categories(cfg)
    f = x.formats(v).set_index("column")
    assert f.loc["country", "upper %"] == pytest.approx(100 * (N - 1) / N) and f.loc["country", "distinct"] == 2
    assert f.loc["language", "lower %"] == 100


def test_ids(cfg):
    i = x.ids(cfg).iloc[0]
    assert i["player_id_not_digits"] == 0 and i["player_id_min_len"] == 4 and i["player_id_max_len"] == 4
    assert i["tenant_id_not_uuid"] == 0 and i["tenant_id_not_lower"] == 1 and i["tenant_ids"] == 2
    assert x.id_lengths(cfg)["rows"].sum() == N


def test_tax(cfg):
    t = x.tax(cfg).set_index("window")
    assert t.loc["today", "above_ggr"] == 1 and t.loc["l90d", "above_ggr"] == 0
    assert t.loc["today", "tax_null"] == 0 and t.loc["today", "tax_zero"] == 0 and t.loc["today", "with_tax"] == N
    assert t.loc["l90d", "tax_over_ggr"][1] == pytest.approx(0.1) and t.loc["l90d", "tax_over_stake"][1] == pytest.approx(0.01)


def test_bonus_against_the_gaming_table(cfg):
    b = x.bonus_vs_gaming(cfg).iloc[0]
    assert b["rows"] == N and b["ok"] == N - 1 and b["higher"] == 1 and b["not_in_gaming"] == 0


def test_impossible_sessions_by_product(cfg):
    t = x.sessions_by_product(cfg).set_index(["rule", "grp"])
    assert t.loc[("play_secs_today > 24 h", "sports only"), "impossible"] == 10
    assert t.loc[("play_secs_today > 24 h", "casino only"), "impossible"] == 0
    assert t.loc[("mean session > 1 day", "bets on sports"), ["rows", "impossible"]].tolist() == [10, 10]
    assert t.loc[("mean session > 1 day", "casino only"), "impossible"] == 0
