"""Reconciliation (reconcile.py) on a small fake lake with planted errors, with no S3.

    .venv/bin/python -m pytest src/quality/tests/test_reconcile.py -q
"""

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contracts  # noqa: E402
import reconcile  # noqa: E402

CONFIG = {"reconcile": {
    "kafka": {"table": "transactions", "business_keys": {"platform_transaction": ["platform_id", "transaction_id"]},
              "event_time": "date_time", "kafka_time": "kafka_timestamp", "ingest_time": "_ingested_at"},
    "same_keys": {
        "player_daily": {"key": ["tenant_id", "player_id", "snapshot_date"],
                         "tables": ["gld_player_signals_daily", "gld_player_risk_daily"]},
        "player_snapshot": {"key": ["tenant_id", "player_id"], "tables": ["gld_players", "gld_player_lifetime"],
                            "compare": ["total_bets"]},
        "player_snapshot_busy": {"key": ["tenant_id", "player_id"], "tables": ["gld_players", "gld_player_lifetime"],
                                 "filters": {"gld_players": "total_bets > 5"}}},
    "totals": {"payments": {
        "source": "gld_player_payments", "target": "gld_player_payments_daily",
        "key": ["tenant_id", "player_id", "event_date"], "date_column": "event_date", "informative": ["deposit_count"],
        "measures": {"deposit_count": "count(*) FILTER (WHERE is_deposit AND is_completed)",
                     "deposit_amount": "sum(processed_amount) FILTER (WHERE is_deposit AND is_completed)"}}},
    "tolerance": 0.01, "status": {"warn_above": 0.0, "critical_above": 0.001}}}


def write(root: Path, layer: str, table: str, day: str, frame: pd.DataFrame):
    path = root / f"org/{layer}/{table}/2026/01/{day}/data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    contract = root / "contracts/lake" / f"{table}.yaml"
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text(yaml.safe_dump({"table": table, "layer": f"org/{layer}", "grain": [], "columns": []}))


@pytest.fixture
def lake(tmp_path, monkeypatch):
    t0 = pd.Timestamp("2026-01-01 10:00", tz="UTC")

    def tx(day, offsets, ids):
        times = [t0 + dt.timedelta(days=day - 1)] * len(ids)
        return pd.DataFrame({"kafka_topic": "tx", "kafka_partition": 0, "kafka_offset": offsets, "platform_id": 1,
                             "transaction_id": ids, "date_time": times,
                             "kafka_timestamp": [t + dt.timedelta(seconds=1) for t in times],
                             "_ingested_at": [t + dt.timedelta(seconds=60) for t in times]})
    write(tmp_path, "20-bronze", "transactions", "01", tx(1, [0, 1, 2], [10, 11, 12]))
    # day 02: offset 4 never arrived, 12 came the day before, 13 comes twice
    write(tmp_path, "20-bronze", "transactions", "02", tx(2, [3, 5, 6], [12, 13, 13]))

    def players(ids, **extra):
        return pd.DataFrame({"tenant_id": "t", "player_id": ids,
                             "snapshot_date": pd.Timestamp("2026-01-01").date(), **extra})
    write(tmp_path, "40-gold", "gld_player_signals_daily", "01", players([1, 2, 3]))
    write(tmp_path, "40-gold", "gld_player_risk_daily", "01", players([1, 2]))              # player 3 missing
    bets = pd.array([5, 6], dtype="uint64")                    # uint64: 6 - 7 must not overflow
    write(tmp_path, "40-gold", "gld_players", "01", players([1, 2], total_bets=bets))
    write(tmp_path, "40-gold", "gld_player_lifetime", "01",
          players([1, 2], total_bets=pd.array([5, 7], dtype="uint64")))

    day1, day2 = pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-01-02").date()
    write(tmp_path, "40-gold", "gld_player_payments", "01", pd.DataFrame({
        "tenant_id": "t", "player_id": [1, 1, 2], "event_date": day1, "is_deposit": True,
        "is_completed": [True, True, False], "processed_amount": [10.0, 5.0, 3.0]}))
    # player 3 pays on day 02 in the atomic table but on day 01 in the daily one: a day shift, not a loss
    write(tmp_path, "40-gold", "gld_player_payments", "02", pd.DataFrame({
        "tenant_id": "t", "player_id": [3], "event_date": day2, "is_deposit": True, "is_completed": [True],
        "processed_amount": [1.0]}))
    write(tmp_path, "40-gold", "gld_player_payments_daily", "01", pd.DataFrame({
        "tenant_id": "t", "player_id": [1, 1, 2, 3, 4], "event_date": day1, "deposit_count": [1, 1, 0, 1, 1],
        "deposit_amount": [10.0, 4.0, 0.0, 1.0, 2.0]}))  # player 1 in two rows (two sources): 14 instead of 15;
    #                                                   player 4 has no payment at all

    config = tmp_path / "quality.yaml"
    config.write_text(yaml.safe_dump(CONFIG))
    monkeypatch.setenv("DATALAKE_ROOT", str(tmp_path))
    monkeypatch.setattr(contracts, "CONTRACTS_DIR", tmp_path / "contracts")
    monkeypatch.setattr(reconcile, "CONFIG_PATH", config)
    monkeypatch.setattr(reconcile, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(reconcile, "REPORT_DIR", tmp_path)
    return tmp_path


def problems(rows, item_start: str, day: str | None = None) -> int:
    match = [r for r in rows if r["item"].startswith(item_start) and (day is None or r["day"] == day)]
    assert match, item_start
    return sum(r["problems"] or 0 for r in match)


def test_planted_errors_are_found(lake, monkeypatch):
    captured = []
    original = reconcile.frame
    monkeypatch.setattr(reconcile, "frame", lambda rows, cfg: captured.extend(rows) or original(rows, cfg))
    reconcile.run(dt.date(2026, 1, 1), dt.date(2026, 1, 2), mlflow_log=False)

    assert problems(captured, "rows repeated in the day", "2026-01-02") == 1
    assert problems(captured, "rows repeated in the day", "2026-01-01") == 0
    assert problems(captured, "keys already seen the day before", "2026-01-02") == 1
    assert problems(captured, "offsets missing in holes of 2 to 1,000") == 0
    assert problems(captured, "holes of 1 offset") == 1                     # offset 4
    assert problems(captured, "rows whose event day is not the folder day") == 0
    assert problems(captured, "delay Kafka to lake") == 0

    assert problems(captured, "keys only in gld_player_signals_daily") == 1
    assert problems(captured, "keys only in gld_player_risk_daily") == 0
    assert problems(captured, "total_bets differs") == 1

    assert problems(captured, "keys only in gld_player_payments_daily (not", "2026-01-01") == 2   # players 3, 4
    assert problems(captured, "keys only in gld_player_payments_daily, not in", "2026-01-01") == 1  # only 4
    assert problems(captured, "deposit_amount differs", "2026-01-01") == 1
    assert problems(captured, "deposit_count differs", "2026-01-01") == 0
    busy = [r for r in captured if r["name"] == "player_snapshot_busy" and r["item"].startswith("keys only in gld_player_l")]
    assert busy[0]["problems"] == 1                                    # player 1 (5 bets) is filtered out

    report = (lake / "reconciliation_2026-01-01_2026-01-02.md").read_text()
    assert "| totals | payments | deposit_amount differs |" in report and "critical" in report
    assert [line for line in report.splitlines() if "| deposit_count differs |" in line][0].count("| info |") == 1


def test_rerun_uses_the_cache(lake, monkeypatch):
    reconcile.run(dt.date(2026, 1, 1), dt.date(2026, 1, 2), mlflow_log=False)
    for name in ("kafka_day", "same_keys_day", "totals_day"):
        monkeypatch.setattr(reconcile, name, lambda *a: pytest.fail("computed again"))
    reconcile.run(dt.date(2026, 1, 1), dt.date(2026, 1, 2), mlflow_log=False)
