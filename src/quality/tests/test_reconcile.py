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
    "tolerance": 0.01, "relative_tolerance": 0.001, "status": {"warn_above": 0.0, "critical_above": 0.001}}}
PAY = ["kafka-sink/primus/pay", "kafka-sink/secundus/pay"]
CONFIG["reconcile"]["duplicates"] = {"landing_payments": {"tables": PAY, "keys": {"payment_status": ["paymentId", "status"]}}}
CONFIG["reconcile"]["period_totals"] = {"landing_to_gold": {
    "source": PAY, "target": "gld_brand_payments", "key": ["brand_id"], "source_columns": {"brand_id": "brandId"},
    "source_filter": "status = 'COMPLETED'", "measures": {"deposits": "count(DISTINCT paymentId)"},
    "target_measures": {"deposits": "sum(completed_deposits)"}}}
CONFIG["reconcile"]["period_totals"]["late_events"] = {
    "source": "transactions", "target": "gld_player_gaming_daily", "key": ["platform_id"],
    "source_columns": {"platform_id": "platform_id"}, "event_date": "CAST(date_time AS DATE)", "late_days": 1,
    "measures": {"bets": "count(*)"}}
CONFIG["reconcile"]["totals"]["bronze_to_gold"] = {
    "source": "transactions", "target": "gld_player_gaming_daily", "key": ["platform_id", "event_date"],
    "source_columns": {"event_date": "CAST(date_time AS DATE)"}, "source_filter": "transaction_id <> 11",
    "measures": {"bets": "count(*)", "small_gap": "count(*) * 1000000.0", "large_gap": "count(*) * 1000000.0"}}


def write(root: Path, layer: str, table: str, day: str, frame: pd.DataFrame):
    path = root / f"org/{layer}/{table}/2026/01/{day}/data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    contract = root / "contracts/lake" / f"{table.replace('/', '__')}.yaml"
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

    # gold has 2 bets on day 01; bronze has 3 rows, one filtered out by source_filter: they match
    write(tmp_path, "40-gold", "gld_player_gaming_daily", "01", pd.DataFrame({
        "platform_id": [1], "event_date": [pd.Timestamp("2026-01-01").date()], "bets": [2],
        "small_gap": [2_001_000.0], "large_gap": [2_004_000.0]}))    # 0.05% and 0.2% over 2,000,000

    # landing: one folder per operator, read together. Payment 1 comes as PENDING and then COMPLETED
    # (not a duplicate); on day 02 payment 3 comes COMPLETED twice (a duplicate).
    def pay(ids, statuses, brand):
        return pd.DataFrame({"paymentId": ids, "status": statuses, "brandId": brand})
    write(tmp_path, "10-landing", PAY[0], "01", pay([1, 1, 2], ["PENDING", "COMPLETED", "COMPLETED"], 14))
    write(tmp_path, "10-landing", PAY[1], "02", pay([3, 3], ["COMPLETED", "COMPLETED"], 23))
    # gold has both deposits of brand 14 but none of brand 23
    for day, brand, n in (("01", 14, 2), ("02", 23, 0)):
        write(tmp_path, "40-gold", "gld_brand_payments", day, pd.DataFrame({"brand_id": [brand], "completed_deposits": [n]}))

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


def problems(rows, item_start: str, day: str | None = None, check: str | None = None) -> int:
    match = [r for r in rows if r["item"].startswith(item_start) and (day is None or r["day"] == day)
             and (check is None or r["check"] == check)]
    assert match, item_start
    return sum(r["problems"] or 0 for r in match)


def test_planted_errors_are_found(lake, monkeypatch):
    captured = []
    original = reconcile.frame
    monkeypatch.setattr(reconcile, "frame", lambda rows, cfg: captured.extend(rows) or original(rows, cfg))
    reconcile.run(dt.date(2026, 1, 1), dt.date(2026, 1, 2), mlflow_log=False)

    assert problems(captured, "rows repeated in the day", "2026-01-02", "kafka") == 1
    assert problems(captured, "rows repeated in the day", "2026-01-01", "kafka") == 0
    assert problems(captured, "keys already seen the day before", "2026-01-02", "kafka") == 1
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
    gaming = [r for r in captured if r["name"] == "bronze_to_gold" and r["day"] == "2026-01-01"]
    found = {r["item"]: r["problems"] for r in gaming}
    assert found["bets differs"] == 0
    assert found["small_gap differs"] == 0 and found["large_gap differs"] == 1   # relative tolerance 0.1%
    dup = {(r["day"], r["item"]): r["problems"] for r in captured if r["check"] == "duplicates"}
    assert dup[("2026-01-01", "rows repeated in the day (payment_status)")] == 0
    assert dup[("2026-01-02", "rows repeated in the day (payment_status)")] == 1
    period = {r["item"]: r["problems"] for r in captured if r["check"] == "period_totals"}
    assert period["deposits differs"] == 1                       # brand 23: 1 in landing, 0 in gold
    late = {r["item"]: r["extra"] for r in captured if r["name"] == "late_events"}
    assert "source 6.00" in late["bets differs"]   # both days are in the period; day 03 (late) has no folder
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


def test_summed_detail_adds_up_every_day():
    detail = pd.Series(["sum over all keys: source 10.00, target 11.00 (+10.000%)",
                        "sum over all keys: gld_a 30.00, gld_b 29.00"])        # the old wording is read too
    assert reconcile.summed_detail(detail) == "sum over all days and keys: source 40.00, target 40.00 (+0.000%)"
    assert reconcile.summed_detail(pd.Series(["p50 1 s", "p50 2 s"])) == "p50 2 s (last of 2 days)"
