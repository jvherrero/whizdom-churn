"""The automatic audit without S3 or ClickHouse: the local copy of a table, the copy from a fake ClickHouse
client, the verdicts of the points and the status of the columns.

    .venv/bin/python -m pytest signal_audit/tests/test_run_audit.py -q
"""

import contextlib
import datetime as dt
import math
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from signal_audit.data import audit
from signal_audit.data import clickhouse_mirror as cm
from signal_audit.checks import evidence as ev
from signal_audit.checks import formula_check as fc
from signal_audit.report import report

DAY = dt.date(2026, 9, 1)


def _write(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_a_table_in_local_tables_is_read_from_its_copy(tmp_path, monkeypatch):
    lake, mirror = tmp_path / "lake", tmp_path / "mirror"
    _write(lake / "gld_player_signals_daily" / "2026/09/01" / "part-0.parquet", pd.DataFrame({"x": [1]}))
    _write(lake / "gld_player_gaming_daily" / "2026/09/01" / "part-0.parquet", pd.DataFrame({"x": [1]}))
    _write(mirror / "gld_player_signals_daily" / "2026/09/02" / "part-0.parquet", pd.DataFrame({"x": [2]}))
    monkeypatch.setenv("DATALAKE_ROOT", str(lake))
    monkeypatch.setenv("LOCAL_TABLES", f"gld_player_signals_daily={mirror}")
    pfs, base = audit.s3_filesystem()
    signals = audit.daily_files(pfs, base, "gld_player_signals_daily")
    assert list(signals) == [dt.date(2026, 9, 2)] and signals[dt.date(2026, 9, 2)].startswith(str(mirror))
    assert list(audit.daily_files(pfs, base, "gld_player_gaming_daily")) == [DAY]   # the other tables: the lake
    assert audit.uri(pfs, signals[dt.date(2026, 9, 2)]) == signals[dt.date(2026, 9, 2)]


class FakeClient:
    """Two blocks for 1 September, nothing for the 2nd; the date column is called event_date."""
    def __init__(self):
        self.queries = []

    def query_arrow_stream(self, query, parameters=None, use_strings=None):
        self.queries.append((query, parameters))
        blocks = [] if parameters["day"] != DAY else [
            pa.table({"tenant_id": ["t"] * 2, "player_id": [1, 2], "event_date": [DAY] * 2, "days_since_bet": [0, 3]}),
            pa.record_batch({"tenant_id": ["t"], "player_id": [3], "event_date": [DAY], "days_since_bet": [9]})]   # what ClickHouse sends
        return contextlib.nullcontext(iter(blocks))


def test_the_copy_from_clickhouse(tmp_path, monkeypatch):
    monkeypatch.setattr(cm.settings, "MIRROR", tmp_path / "mirror")
    monkeypatch.setattr(cm.settings, "FOOTERS", tmp_path / "footers")
    monkeypatch.setattr(cm.settings, "RUN", {"table": "db.signals", "date_column": "event_date"})
    client = FakeClient()
    assert cm.copy([DAY, DAY + dt.timedelta(days=1)], client) == [DAY]          # no rows on the 2nd: no file
    assert "FINAL" in client.queries[0][0] and "db.signals" in client.queries[0][0]
    rows = pd.read_parquet(cm.path(DAY))
    assert len(rows) == 3 and "snapshot_date" in rows and "event_date" not in rows
    assert cm.copy([DAY], client) == []                                         # copied days are kept
    diff = cm.schema_differences()
    assert "churn_score" in diff["missing"] and diff["extra"] == ["event_date"]   # the date keeps its ClickHouse name
    foot = pd.read_parquet(cm.write_footers())
    assert set(foot["column"]) == {"tenant_id", "player_id", "snapshot_date", "days_since_bet"} and foot["n_rows"].max() == 3


def test_the_days_the_copy_needs(monkeypatch):
    s = cm.settings
    monkeypatch.setattr(s, "PERIOD", {"start": DAY, "end": dt.date(2026, 9, 30)})
    monkeypatch.setattr(s, "LONG", {"start": DAY, "end": dt.date(2026, 9, 30)})
    monkeypatch.setattr(s, "SNAPSHOTS", [dt.date(2026, 9, 15)])
    monkeypatch.setattr(s, "EVIDENCE_DAYS", [DAY])
    monkeypatch.setattr(s, "MONTH_STARTS", [dt.date(2026, 6, 1), DAY])
    days = cm.days_needed()
    assert days[0] == dt.date(2026, 6, 1) and dt.date(2026, 8, 16) in days and days[-1] == dt.date(2026, 9, 30)
    assert dt.date(2026, 7, 1) not in days


PASSING = {"rec_match0": 100, "ad_off_lo": 100, "vd_above": 0, "tp_frozen_n": 0, "tp_first": DAY, "long_start": DAY,
           "se_over_day": 0, "gap_neg": 0, "ps_over_day": 0, "night_above": 0, "rgn_match": 100, "weekend_above": 0,
           "ent_above": 0, "deprec_formula": 50, "ten_min": 100, "rp_at100": 10, "brand_null": 0, "code_country": 0,
           "code_language": 0, "code_currency_hi": 0, "hold_rows": 0, "peak_hour0": 4, "peak_day0": 14, "dep_empty": 0,
           "units_max": 1, "tax_zero": 0, "tax_above": 0}


def test_a_correct_table_passes_every_point():
    v = report.verdicts(PASSING)
    assert {k for k, s in v.items() if s != "pass"} == {"it:ngr", "it:prior", "it:z"}
    assert {v[k] for k in ("it:ngr", "it:prior", "it:z")} == {"unconfirmed"}


def test_the_rules_find_the_errors():
    v = report.verdicts(PASSING | {"rec_match0": 6.4, "vd_above": 0.84, "tp_first": dt.date(2026, 9, 5)})
    assert {k for k, s in v.items() if s == "fail"} == {"it:recency", "it:valuedecline", "it:churnscore", "it:tpdays"}
    assert report.verdicts(PASSING | {"ten_min": math.nan})["it:tenure"] == "no data"


def test_fill_writes_the_numbers():
    assert report.fill("<<a:.1f>>\\% and <<b>>", {"a": 6.43, "b": 7}) == "6.4\\% and 7"
    assert report.fill("<<a:.1f>>", {"a": math.nan}) == "--"


def test_the_status_of_the_columns_follows_the_points_that_fail(monkeypatch):
    columns = {"days_since_bet": [{"kind": "bound", "expr": "days_since_bet >= 0"}],
               "churn_score": [{"kind": "formula", "expr": "0.5 * recency_pressure"}],
               "ngr_eur_today": [{"kind": "identity", "expr": "ggr_eur_today"}], "player_id": [{"kind": "none"}]}
    monkeypatch.setattr(fc, "load", lambda: {"columns": columns})
    monkeypatch.setattr(fc, "run", lambda cfg: pd.DataFrame({"column": ["days_since_bet", "churn_score", "ngr_eur_today"],
                                                             "n_check": [0, 0, 0], "pass %": [100.0, 100.0, 90.0]}))
    s = ev.column_status({"it:recency"}).set_index("column")["status"]
    assert s.to_dict() == {"days_since_bet": "wrong", "churn_score": "correct", "ngr_eur_today": "unconfirmed",
                           "player_id": "unchecked"}
    assert ev.column_status({"it:churnscore"}).set_index("column").loc["churn_score", "status"] == "affected"
    assert ev.column_status().set_index("column").loc["days_since_bet", "status"] == "wrong"   # None: every point fails


@pytest.mark.parametrize("point", report.POINTS["points"], ids=lambda p: p["id"])
def test_every_point_has_its_texts_and_known_figures(point):
    for field in ("name", "rule", "result", "definition", "fix"):
        assert set(point[field]) == {"es", "en"}
    assert all(n in report.POINTS["figures"] for n in point["figures"])
    es, en = ({n.removesuffix(f"_{lang}") for n, _ in report.FIELD.findall(point["result"][lang])} for lang in ("es", "en"))
    assert es == en   # the same numbers in both languages (dates have one name per language)
