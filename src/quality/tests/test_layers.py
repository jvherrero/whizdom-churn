"""Layer inventory (layers.py) on a small fake lake with no S3, and the table contracts.

    .venv/bin/python -m pytest src/quality/tests -q
"""

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contracts  # noqa: E402
import inventory_to_contracts  # noqa: E402
import layers  # noqa: E402


def write(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


@pytest.fixture
def lake(tmp_path, monkeypatch):
    """Bronze from a Kafka sink (dt=/hour= folders), silver and gold by YYYY/MM/DD, one gold day missing."""
    pay = pd.DataFrame({"tenant_id": ["t"] * 3, "brand_id": [64] * 3, "payment_id": ["a", "b", "c"],
                        "amount": [1.0, 2.0, 3.0]})
    kafka = pay.assign(_kafka_offset=[1, 2, 3], _ingested_at=pd.Timestamp("2026-01-01", tz="UTC"))
    for hour in ("00", "01"):
        write(tmp_path / f"org/20-bronze/kafka-sink/payments/dt=2026-01-01/hour={hour}/part-0.parquet", kafka)
    write(tmp_path / "org/30-silver/slv_payments/2026/01/01/data.parquet", pay)
    for day in ("01", "03"):
        write(tmp_path / f"org/40-gold/gld_payment/2026/01/{day}/data.parquet", pay)
    write(tmp_path / "org/40-gold/gld_player/data.parquet", pd.DataFrame({"tenant_id": ["t"], "player_id": [1]}))
    monkeypatch.setenv("DATALAKE_ROOT", str(tmp_path))
    monkeypatch.setattr(layers, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(layers, "CACHE_DIR", tmp_path / "cache")
    return tmp_path


def test_file_day_and_layout():
    assert layers.file_day("2026/01/03/data.parquet") == dt.date(2026, 1, 3)
    assert layers.file_day("dt=2026-01-03/hour=05/part-0.parquet") == dt.date(2026, 1, 3)
    assert layers.file_day("data.parquet") is None
    assert layers.layout("2026/01/03/data.parquet") == "YYYY/MM/DD"
    assert layers.layout("dt=2026-01-03/hour=05/part-0.parquet") == "dt=YYYY-MM-DD/hour=NN"


def test_inventory(lake):
    layers.run(["org/20-bronze", "org/30-silver", "org/40-gold"], rows=True, workers=2)
    results = {r["table"]: r for r in pd.read_json(lake / "out/layer_inventory.json").to_dict("records")}
    assert set(results) == {"kafka-sink/payments", "slv_payments", "gld_payment", "gld_player"}

    bronze = results["kafka-sink/payments"]
    assert bronze["n_files"] == 2 and bronze["n_rows"] == 6 and bronze["layout"] == "dt=YYYY-MM-DD/hour=NN"
    assert bronze["meta_columns"] == ["_kafka_offset", "_ingested_at"]

    gold = results["gld_payment"]
    assert gold["missing_days"] == 1 and gold["missing_examples"] == ["2026-01-02"]
    assert gold["contract"]["contract"] == "gld_payment" and gold["contract"]["same_name"]
    assert "payment_id" not in gold["contract"]["missing"]

    assert pd.isna(results["gld_player"].get("first_day"))   # no date in the path: an SCD table


def test_other_layers_keep_the_main_inventory(lake):
    layers.run(["org/20-bronze", "org/30-silver", "org/40-gold"], rows=False, workers=2)
    layers.run(["org/40-gold"], rows=False, workers=2)
    assert (lake / "out/layer_inventory.json").exists() and (lake / "out/layer_inventory_40-gold.json").exists()
    assert "kafka-sink/payments" in (lake / "out/layer_inventory.json").read_text()


def test_rerun_reads_only_new_files(lake, monkeypatch):
    layers.run(["org/40-gold"], rows=True, workers=2)
    calls = []
    original = layers.parquet_meta
    monkeypatch.setattr(layers, "parquet_meta", lambda pfs, p: calls.append(p) or original(pfs, p))
    layers.run(["org/40-gold"], rows=True, workers=2)
    # Every footer comes from the cache: no file is opened again.
    assert calls == []


def test_contracts():
    tables = contracts.load("design")
    assert len(tables) == 21
    signal = tables["gld_player_signal"]
    assert signal.grain == ["tenant_id", "brand_id", "event_date", "player_id"]
    assert len(signal.columns) == 44
    streak = signal.column("active_streak_days")
    assert streak.nullable and streak.base_type == "UInt16" and streak.unit == "days"
    assert signal.column("engagement_score").range == (0.0, 100.0)
    assert tables["gld_payment"].grain == ["tenant_id", "brand_id", "payment_id"]


def test_column_types():
    assert not contracts.ColumnDef("x", "string", nullable=False).nullable      # lake: explicit
    low_null = contracts.ColumnDef("x", "LowCardinality(Nullable(String))")
    assert low_null.nullable and low_null.base_type == "String"
    array = contracts.ColumnDef("x", "Array(Nullable(String))")
    assert not array.nullable and array.base_type == "Array(Nullable(String))"
    assert contracts.ColumnDef("x", "Nullable(Decimal(18, 4))").base_type == "Decimal(18, 4)"


def test_lake_contracts(tmp_path):
    inventory = [
        {"layer": "org/40-gold", "table": "gld_player_payments", "layout": "YYYY/MM/DD",
         "schema": [["tenant_id", "string"], ["brand_id", "int32"], ["payment_id", "string"],
                    ["processed_amount", "decimal128(18, 4)"], ["_version", "uint64"]],
         "contract": {"contract": "gld_payment"}},
        {"layer": "org/40-gold", "table": "gld_new_table", "layout": "YYYY/MM/DD",
         "schema": [["tenant_id", "string"], ["player_id", "int32"], ["snapshot_date", "date32[day]"],
                    ["email", "string"]]},
        {"layer": "org/30-silver", "table": "medallion/pii", "layout": None},   # empty: no contract
    ]
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory))
    written = inventory_to_contracts.run(tmp_path / "lake", path)
    assert written == ["gld_player_payments", "gld_new_table"]

    tables = contracts.load(directory=tmp_path / "lake")
    pay = tables["gld_player_payments"]
    assert pay.grain == ["tenant_id", "brand_id", "payment_id"] and pay.grain_checked   # in grains_checked
    amount = pay.column("processed_amount")
    assert amount.unit == "currency" and amount.additive and amount.nullable
    assert pay.column("_version").group == "meta"
    assert not pay.column("payment_id").nullable

    new = tables["gld_new_table"]
    assert new.grain == ["tenant_id", "player_id", "snapshot_date"] and not new.grain_checked   # proposal
    assert new.column("email").pii


def test_sink_offsets():
    """Partition 0: files at offsets 0 (10 rows), 10 (10 rows) and 25: offsets 20 to 24 are missing.
    Partition 1: files at 0 (10 rows) and 5: offsets 5 to 9 are written twice."""
    names = ["t+0+0.parquet", "t+0+10.parquet", "t+0+25.parquet", "t+1+0.parquet", "t+1+5.parquet"]
    files = pd.DataFrame({"relative": [f"2026/01/0{i + 1}/{n}" for i, n in enumerate(names)],
                          "n_rows": [10, 10, 10, 10, 10], "day": [dt.date(2026, 1, i + 1) for i in range(5)]})
    o = layers.sink_offsets(files)
    assert (o["gaps"], o["offsets_missing"], o["overlaps"], o["offsets_twice"]) == (1, 5, 1, 5)
    assert o["days"] == ["2026-01-03", "2026-01-05"]
    assert layers.sink_offsets(files.assign(relative="2026/01/01/data.parquet")) is None   # not a sink table
