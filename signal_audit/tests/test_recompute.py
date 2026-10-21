"""Sessions, night share and the player key, rebuilt from a small hand-made landing feed (no S3).

    .venv/bin/python -m pytest signal_audit/tests/test_recompute.py -q

Player 1 of brand 14 bets in TRY (local time UTC+3). Bets of 1 September 2026, in UTC:
    10:00, 10:10, 10:20     one session of 3 bets, 1,200 seconds
    11:00                   arrives one day late (folder of 2 September)
    12:00
    23:50 and 00:10 (2 Sep) one session across midnight; both bets are night in Istanbul (02:50, 03:10)
Also: a duplicated uuid, a rolled-back bet, a win, and a bet of brand 99 (out of scope).
"""

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit  # noqa: E402
import recompute  # noqa: E402

D1, D2 = dt.date(2026, 9, 1), dt.date(2026, 9, 2)


def _tx(uuid, ts, player=1, brand=14, kind="GAME_BET", stake=1.0, rolled=False):
    return {"uuid": uuid, "accountId": player, "brandId": brand, "dateTime": pd.Timestamp(ts, tz="UTC"),
            "tranType": kind, "currency": "TRY", "rolledBack": rolled, "amountReal": -stake,
            "amountReleasedBonus": 0.0, "amountPlayableBonus": 0.0,
            "firstname": "never read", "email": "never@read"}   # personal columns that must not be needed


def _folder(root: Path, day: dt.date, rows: list[dict]):
    folder = root / "org/10-landing/kafka-sink/primus/whizdomai-transactions" / f"{day:%Y/%m/%d}"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(folder / "part-0.parquet", index=False)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    root = tmp_path / "lake"
    _folder(root, D1, [
        _tx("a", "2026-09-01 10:00"), _tx("b", "2026-09-01 10:10"), _tx("c", "2026-09-01 10:20"),
        _tx("c", "2026-09-01 10:20"),                               # duplicated message
        _tx("d", "2026-09-01 12:00"), _tx("e", "2026-09-01 23:50", stake=2.0),
        _tx("x", "2026-09-01 10:05", rolled=True),                  # rolled back: not a bet
        _tx("w", "2026-09-01 10:06", kind="GAME_WIN"),              # a win: not a bet
        _tx("z", "2026-09-01 10:00", brand=99),                     # another brand
    ])
    _folder(root, D2, [_tx("f", "2026-09-02 00:10"), _tx("g", "2026-09-01 11:00")])   # 11:00 arrives late
    gold = root / "gld_player_gaming_daily" / f"{D1:%Y/%m/%d}"
    gold.mkdir(parents=True)
    pd.DataFrame({"tenant_id": "op", "brand_id": [14, 14], "player_id": [1, 2], "bets": [5, 3]}).to_parquet(
        gold / "data.parquet", index=False)
    monkeypatch.setenv("DATALAKE_ROOT", str(root))
    monkeypatch.setenv("RUN_S3", "1")
    monkeypatch.setattr(audit, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(audit, "_CON", None)
    c = audit.load_config(Path(recompute.__file__).parent / "metrics_recomputed.yaml")
    c["period"] = {"start": D1, "end": D2}
    c["brands"] = [14]
    recompute.build(c)
    return c


def _day(day):
    return pd.read_parquet(recompute.out_dir() / f"player_day_{day}.parquet").set_index("player_id").loc[1]


def test_sessions_with_a_30_minute_timeout(cfg):
    p = _day(D1)
    assert p["bets"] == 6                          # 10:00, 10:10, 10:20, 11:00, 12:00, 23:50
    assert p["sessions_30"] == 4                   # 10:00-10:20, 11:00, 12:00, 23:50-00:10
    assert p["single_bet_sessions_30"] == 2
    assert p["duration_sum_30"] == pytest.approx(2400)


def test_a_session_across_midnight_counts_once_on_its_first_day(cfg):
    p = _day(D2)
    assert p["bets"] == 1 and p["sessions_30"] == 0


def test_the_timeout_changes_the_sessions(cfg):
    p = _day(D1)
    assert p["sessions_60"] == 2                   # 10:00-12:00 and 23:50-00:10
    assert p["sessions_15"] == 4 and p["single_bet_sessions_15"] == 3


def test_night_stake_in_local_time(cfg):
    p = _day(D1)
    assert p["stake"] == pytest.approx(7) and p["night_stake"] == pytest.approx(2)   # 23:50 UTC is 02:50 in Istanbul


def test_player_key_check(cfg):
    k = recompute.key_check(cfg, D1).set_index("brand_id").loc[14]
    assert (k["landing_players"], k["gold_players"], k["in_both"]) == (1, 2, 1)


def test_recomputed_metrics_run_through_the_audit(cfg):
    prof, hist = audit.metric_data(cfg, "rc_sessions_per_active_day")
    assert prof["n"] == 1 and prof["max"] == 4
    prof, _ = audit.metric_data(cfg, "rc_session_gap_hours")
    assert prof["n"] == 3                          # three gaps between the four sessions of player 1
    prof, _ = audit.metric_data(cfg, "rc_night_share")
    assert prof["max"] == pytest.approx(3 / 8)     # night stake 2 + 1, stake 7 + 1


def test_time_of_play_weekend_entropy_and_night(cfg):
    import math
    recompute.build_time_of_play(cfg)
    # Istanbul is UTC+3: stake by local hour 13: 3, 14: 1 (the late bet), 15: 1, 02: 2, 03: 1 (2 Sep); total 8
    prof, _ = audit.metric_data(cfg, "rc_hourly_entropy")
    shares = [3 / 8, 1 / 8, 1 / 8, 2 / 8, 1 / 8]
    assert prof["max"] == pytest.approx(-sum(p * math.log(p) for p in shares) / math.log(24))
    prof, _ = audit.metric_data(cfg, "rc_weekend_share")
    assert prof["max"] == 0                        # 1 and 2 September 2026 are a Tuesday and a Wednesday
    prof, _ = audit.metric_data(cfg, "rc_night_share_tod")
    assert prof["max"] == pytest.approx(3 / 8)     # the same night share as from the player-day files
