from dataclasses import asdict, replace
import json

import pandas as pd
import pytest

from dq_lib import (
    ColumnSpec, ColumnNullStat, KeyCheck, TableProfile, ValidationIssue,
    ValidationResult, build_schema, validate_dataframe, key_check,
    render_html_report, render_json_report, write_report,
    render_source_table_register,
)


@pytest.fixture
def columns():
    return [ColumnSpec("id", "int64", False), ColumnSpec("name", "object", False),
            ColumnSpec("amount", "float64", True), ColumnSpec("active", "bool", False),
            ColumnSpec("created_at", "datetime64[ns]", False)]


@pytest.fixture
def frame():
    return pd.DataFrame({"id": [1, 2], "name": pd.Series(["row-1", "row-2"], dtype="object"),
                         "amount": [42.0, None], "active": [True, False],
                         "created_at": pd.to_datetime(["2024-01-01", "2024-01-03"]).astype("datetime64[ns]")})


def test_valid_schema(frame, columns):
    pd.testing.assert_frame_equal(build_schema(columns).validate(frame), frame)
    assert validate_dataframe(frame, columns) == ValidationResult(True, [])


@pytest.mark.parametrize("case,column,check", [
    ("dtype", "id", "dtype"), ("null", "name", "nullable"),
    ("missing", "name", "column_missing"), ("extra", "extra", "column_unexpected"),
])
def test_validation_failures(frame, columns, case, column, check):
    if case == "dtype":
        frame["id"] = frame["id"].astype(str)
    elif case == "null":
        frame.loc[0, "name"] = None
    elif case == "missing":
        frame = frame.drop(columns=["name"])
    else:
        frame["extra"] = 1
    original = frame.copy(deep=True)
    result = validate_dataframe(frame, columns)
    assert not result.passed
    assert len(result.issues) == 1
    assert result.issues[0].column == column
    assert result.issues[0].check == check
    assert result.issues[0].message
    pd.testing.assert_frame_equal(frame, original)


def test_all_failures_collected(frame, columns):
    frame["id"] = frame["id"].astype(str)
    frame.loc[0, "name"] = None
    frame = frame.drop(columns=["active"])
    frame["extra"] = 1
    result = validate_dataframe(frame, columns)
    assert {issue.check for issue in result.issues} == {
        "dtype", "nullable", "column_missing", "column_unexpected"}


@pytest.mark.parametrize("values,distinct,unique", [
    ([1, 2], 2, True), ([1, 1], 1, False), ([1, None], 2, True),
    ([None, None], 1, False), ([], 0, True),
])
def test_keys(values, distinct, unique):
    assert key_check(pd.DataFrame({"id": values}), "id") == KeyCheck(
        "id", len(values), distinct, unique)


@pytest.fixture
def profile():
    return TableProfile(
        "example", "One row per generic event", 2, "2024-01-01", "2024-01-03", 1,
        ["2024-01-02"], [ColumnSpec("id", "int64", False)],
        [ColumnNullStat("id", 0), ColumnNullStat("amount", 50),
         ColumnNullStat("name", 25)], [KeyCheck("id", 2, 1, False)],
        ValidationResult(False, [ValidationIssue("id", "dtype", "Expected int64")]),
        ["Synthetic data only"],
    )


@pytest.fixture
def minimal():
    return TableProfile("empty", "One row per generic event", 0, None, None, None,
                        [], [], [], [], ValidationResult(True, []), [])


@pytest.mark.parametrize("fixture_name", ["profile", "minimal"])
def test_json_report(request, fixture_name):
    profile = request.getfixturevalue(fixture_name)
    result = render_json_report(profile)
    assert result == asdict(profile)
    assert json.loads(json.dumps(result)) == result


def test_html_normal(profile):
    html = render_html_report(profile)
    assert html.startswith("<!DOCTYPE html>")
    assert "<style>" in html
    for text in ["example", "One row per generic event", "2024-01-01",
                 "2024-01-03", "FAIL", "Expected int64", "Synthetic data only"]:
        assert text in html
    assert html.index("<td>amount</td>") < html.index("<td>name</td>") < html.index("<td>id</td>")
    assert "<script" not in html


def test_html_empty(minimal):
    html = render_html_report(minimal)
    assert "PASS" in html
    assert "no issues found" in html
    assert "No key checks" in html
    assert "Unknown" in html
    assert html.endswith("</html>")


def test_html_escaping(profile):
    profile.table_name = "<example>"
    profile.notes = ['<script>alert("fake")</script>']
    html = render_html_report(profile)
    assert "&lt;example&gt;" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


@pytest.mark.parametrize("use_string", [False, True])
def test_write_report(profile, tmp_path, use_string):
    target = tmp_path / "nested" / "reports"
    write_report(profile, str(target) if use_string else target)
    assert (target / "example.html").read_text() == render_html_report(profile)
    assert json.loads((target / "example.json").read_text()) == render_json_report(profile)


def test_register(profile):
    result = render_source_table_register([profile])
    assert result.startswith("## Source-table register\n")
    assert len(result.splitlines()) == 5
    assert "| example | One row per generic event |" in result
    assert "2024-01-01 to 2024-01-03" in result
    assert "amount (50%), name (25%)" in result
    assert "1 (2024-01-02)" in result


@pytest.mark.parametrize("count", [10, 11])
def test_register_gaps(profile, count):
    profile.n_gap_days = count
    profile.gap_dates = [f"2024-02-{day:02}" for day in range(1, count + 1)]
    result = render_source_table_register([profile])
    if count == 10:
        assert "2024-02-10" in result
        assert "see JSON" not in result
    else:
        assert "11 (see JSON report for the full list)" in result
        assert "2024-02-01" not in result


def test_register_empty_and_order(minimal):
    minimal.null_stats = [ColumnNullStat("id", 0)]
    result = render_source_table_register([minimal, replace(minimal, table_name="second")])
    assert "none" in result
    assert "Unknown" in result
    assert result.index("| empty |") < result.index("| second |")
    assert len(render_source_table_register([]).splitlines()) == 4


def test_register_top_three_and_escaping(profile):
    profile.null_stats = [ColumnNullStat(f"column-{n}", n) for n in range(1, 5)]
    profile.grain_description = "generic | event\nnext line"
    result = render_source_table_register([profile])
    assert "column-4 (4%), column-3 (3%), column-2 (2%)" in result
    assert "column-1" not in result
    assert "generic &#124; event<br>next line" in result
