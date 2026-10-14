"""Great Expectations suites and day-by-day validation on a small fake lake, with no S3.

    .venv/bin/python -m pytest src/quality/tests/test_validate.py -q
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contracts  # noqa: E402
import suites  # noqa: E402
import validate  # noqa: E402

CONTRACT = """table: gld_player_payments
layer: org/40-gold
grain: [tenant_id, payment_id]
grain_checked: false
partition_by: YYYY/MM/DD folders
engine: null
columns:
  - {name: tenant_id, type: string, key: true, nullable: false}
  - {name: payment_id, type: string, key: true, nullable: false}
  - {name: deposit_amount, type: double, nullable: true}
  - {name: score, type: double, range: [0.0, 1.0], nullable: true}
  - {name: event_date, type: 'date32[day]', nullable: true}
  - {name: email, type: string, nullable: true, pii: true}
"""


@pytest.fixture
def lake(tmp_path, monkeypatch):
    """Day 01 is clean. Day 02 has a duplicated payment, a negative deposit, a score above 1 and a row
    of another day. Day 03 has no file."""
    base = dict(tenant_id=["t"] * 3, email=["x@y.z"] * 3)
    days = {
        "01": pd.DataFrame(base | dict(payment_id=["a", "b", "c"], deposit_amount=[1.0, 2.0, 0.0],
                                       score=[0.1, 0.5, 1.0], event_date=pd.to_datetime(["2026-01-01"] * 3).date)),
        "02": pd.DataFrame(base | dict(payment_id=["d", "d", "e"], deposit_amount=[1.0, -2.0, None],
                                       score=[0.1, 1.5, None],
                                       event_date=pd.to_datetime(["2026-01-02"] * 2 + ["2026-01-01"]).date)),
    }
    for day, frame in days.items():
        path = tmp_path / f"org/40-gold/gld_player_payments/2026/01/{day}/data.parquet"
        path.parent.mkdir(parents=True)
        frame.to_parquet(path, index=False)
    (tmp_path / "contracts/lake").mkdir(parents=True)
    (tmp_path / "contracts/lake/gld_player_payments.yaml").write_text(CONTRACT)
    monkeypatch.setenv("DATALAKE_ROOT", str(tmp_path))
    monkeypatch.setattr(contracts, "CONTRACTS_DIR", tmp_path / "contracts")
    monkeypatch.setattr(suites, "SUITES_DIR", tmp_path / "gx")
    monkeypatch.setattr(validate, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(validate, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr(validate, "SUMMARY_PATH", tmp_path / "gx_summary.md")
    return tmp_path


def test_suite_rules(lake):
    suites.run()
    raw = suites.load("gld_player_payments")
    rules = {(e["meta"]["rule"], e["kwargs"].get("column") or tuple(e["kwargs"].get("column_list", [])))
             for e in raw["expectations"]}
    assert rules == {("grain", ("tenant_id", "payment_id")), ("not_null", "tenant_id"),
                     ("not_null", "payment_id"), ("non_negative", "deposit_amount"), ("range", "score"),
                     ("batch_day", "_batch_day")}               # a text copy of event_date
    assert "email" not in suites.columns_used(raw)          # pii: never loaded
    assert "id" not in raw and all("id" not in e for e in raw["expectations"])   # stable diffs


def test_validate_days(lake, monkeypatch):
    suites.run()
    validate.run(["gld_player_payments"], pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-01-03").date(),
                 mlflow_log=False)
    checks = validate.summarise([json.loads(p.read_text()) for p in sorted((lake / "cache").rglob("*.json"))])
    failed = checks[~checks["success"]]
    assert set(failed["day"]) == {"2026-01-02"}
    assert set(failed["rule"]) == {"grain", "non_negative", "range", "batch_day"}
    assert failed.set_index("rule").loc["grain", "unexpected_count"] == 2
    assert failed["error"].isna().all()
    summary = (lake / "gx_summary.md").read_text()
    assert "| `gld_player_payments` | 2 | 1 | 6 | 4 | 1 | no |" in summary
    assert "| -2 | 1 |" in summary                                  # observed min and max of deposit_amount
    assert "x@y.z" not in summary and (lake / "docs/index.html").exists()

    # A rerun reads every day from the cache.
    monkeypatch.setattr(validate, "load_day", lambda *a: pytest.fail("day loaded again"))
    validate.run(["gld_player_payments"], pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-01-03").date(),
                 mlflow_log=False)


HOURLY = """table: gld_player_signals_hourly
layer: org/40-gold
grain: [player_id, event_hour]
columns:
  - {name: player_id, type: int32, key: true, nullable: false}
  - {name: event_hour, type: 'timestamp[ms, tz=UTC]', key: true, nullable: false}
  - {name: _version, type: uint64, group: meta, nullable: true}
"""


def test_hourly_grain_and_versions(lake):
    """event_hour is both in the grain and the batch day column: hours of one day must stay distinct.
    Player 2 at 01:00 has two versions, which is not a true duplicate."""
    hours = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 01:00"], utc=True)
    frame = pd.DataFrame({"player_id": [1, 2, 2], "event_hour": hours, "_version": [1, 1, 2]})
    path = lake / "org/40-gold/gld_player_signals_hourly/2026/01/01/data.parquet"
    path.parent.mkdir(parents=True)
    frame.to_parquet(path, index=False)
    (lake / "contracts/lake/gld_player_signals_hourly.yaml").write_text(HOURLY)
    suites.run(["gld_player_signals_hourly"])
    day = pd.Timestamp("2026-01-01").date()
    validate.run(["gld_player_signals_hourly"], day, day, mlflow_log=False)
    item = json.loads((lake / "cache/gld_player_signals_hourly/2026-01-01.json").read_text())
    rows = {r["rule"]: r for r in item["rows"]}
    assert rows["batch_day"]["success"]
    assert rows["grain"]["unexpected_count"] == 2                 # player 2 at 01:00, not every row
    assert rows["grain"]["left_after_latest_version"] == 0         # only an unmerged older version
