import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "check_games_matched_final"))
import completeness as c  # noqa: E402


def small():
    games = pd.DataFrame({
        "game_uid": ["a", "b", "c", "d"], "overall_status": ["matched", "matched", "fuzzy", "unmatched"], "rank": ["1", "2", "3", "4"],
        "game_name": ["A", "B", "C", "D"], "provider_name": ["P"] * 4,
        "attr_rtp": ["96", None, None, "N/A"], "attr_rtp_conflict": ["no", "yes", "no", "no"],
        "sc_rtp": ["96", "96.5", "95", "-"], "research_rtp": [None, None, "96%", "94"], "own_studio": ["S"] * 4}).set_index("game_uid", drop=False)
    records = pd.DataFrame({"game_uid": ["a", "c", "c"], "overall_status": ["matched", "fuzzy", "fuzzy"], "rtp_percent": ["96", "97", None],
                            "name": ["A", "C", "C"]})
    fields = {"fields": {"rtp": {"attr": "attr_rtp", "sources": ["sc_rtp", "research_rtp"], "records": "rtp_percent", "kind": "number",
                                 "order": "o", "normalize": "n", "conflict": "k"},
                         "studio": {"sources": ["own_studio"], "kind": "category", "order": "o", "normalize": "n", "conflict": "k"}},
              "ignore": ["game_uid", "overall_status", "rank", "game_name", "provider_name", "name"]}
    return games, records, fields


def test_is_filled_and_pattern():
    series = pd.Series(["96", "", None, "N/A", "n-a", " - ", "0", "None", "x5000.00", "[]"])
    assert c.is_filled(series).tolist() == [True, False, False, False, False, False, True, False, True, False]    # 0 is a value
    assert c.pattern("x5000.00") == "a9.9" and c.pattern("28/03/2024") == "9/9/9" and c.pattern("$0.2 - $100") == "$9.9 - $9"
    assert c.level(85) == "sufficient" and c.level(60) == "partial" and c.level(10) == "needs enrichment" and c.level(0) == "empty"


def test_field_coverage():
    games, records, fields = small()
    row = c.field_coverage(games, records, fields).set_index("field").loc["rtp"]
    assert row["attr_matched_%"] == 50 and row["attr_fuzzy_%"] == 0 and row["attr_unmatched_%"] == 0      # N/A is missing
    assert row["union_matched_%"] == 100 and row["union_fuzzy_%"] == 100 and row["union_usable_%"] == 100  # b: sc, c: research, records
    assert row["only_outside_attr"] == 2 and row["conflicts"] == 1                                         # b and c; conflict yes on b
    assert row["level"] == "sufficient" and "records:rtp_percent 66.7" in row["by_source_%"]
    studio = c.field_coverage(games, records, fields).set_index("field").loc["studio"]
    assert studio["only_outside_attr"] is None or pd.isna(studio["only_outside_attr"])                      # no column of Navid


def test_dispersion_and_examples():
    games, records, fields = small()
    d = c.dispersion(games, records, fields).set_index("place")
    assert d.loc["sc_rtp", ["attr_empty_matched", "attr_empty_fuzzy", "attr_empty_unmatched"]].tolist() == [1, 1, 0]
    assert d.loc["research_rtp", "attr_empty_fuzzy"] == 1 and d.loc["records:rtp_percent", "attr_empty_fuzzy"] == 1
    ex = c.dispersion_examples(games, records, fields)
    assert ex["game_uid"].tolist() == ["b"] and "sc_rtp: 96.5" in ex.loc[0, "values_found"]            # only matched games


def test_value_formats_and_column_coverage():
    games, records, fields = small()
    formats = c.value_formats(games, records, fields)
    rtp = formats[(formats["field"] == "rtp") & (formats["column"] == "sc_rtp")].set_index("value_or_shape")["count"].to_dict()
    assert rtp == {"9": 2, "9.9": 1}                                                       # '-' is missing; shapes of the number
    coverage = c.column_coverage(games, records, fields).set_index(["table", "column"])
    row = coverage.loc[("games", "attr_rtp")]
    assert row["field"] == "rtp" and row["written_as_NA"] == 1 and row["filled_matched_%"] == 50
    assert coverage.loc[("games", "attr_rtp_conflict"), "field"] == "rtp" and coverage.loc[("crawler_records", "rtp_percent"), "field"] == "rtp"


def test_uncovered_columns_and_rules():
    games, records, fields = small()
    assert c.uncovered_columns(games, records, fields) == {"games": [], "crawler_records": []}
    games["new_column"] = 1
    assert c.uncovered_columns(games, records, fields)["games"] == ["new_column"]
    coverage = c.field_coverage(games, records, fields)
    rules = c.unification_rules(fields, coverage).set_index("field")
    assert rules.loc["rtp", ["order", "level"]].tolist() == ["o", "sufficient"] and "fuzzy (marked" in rules.loc["rtp", "rows_used"]


REAL = HERE / "games_matched_final.xlsx"


@pytest.mark.skipif(not REAL.exists(), reason="the file of Navid is not in the folder")
def test_every_column_of_the_real_file_is_in_fields_yaml():
    games, records = c.load(REAL)
    assert c.uncovered_columns(games, records, c.load_fields()) == {"games": [], "crawler_records": []}
