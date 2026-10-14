"""Great Expectations suites from the lake contracts (configs/contracts/lake/), one per table.

    make gx-suites                                  # writes gx/expectations/<table>.json
    .venv/bin/python src/quality/suites.py --tables gld_player_payments transactions

Rules, each one tagged in the expectation meta so the summary can group them:
    grain         the grain columns are unique together
    not_null      columns with nullable: false (today, the grain columns)
    range         columns with a range in the contract
    non_negative  numeric columns named like a non-negative amount or count (NON_NEGATIVE and not
                  SIGNED, as in eda/profile_tables.py); unsigned types are skipped, they cannot be < 0.
                  Gold only: in bronze, amounts are raw movements and can have either sign.
    batch_day     the date column holds the day of the batch (a suite parameter set at run time). The
                  check reads BATCH_DAY, a copy of that column as YYYY-MM-DD text made by validate.py,
                  so a timestamp column such as event_hour keeps its real values for the grain check.
Columns with personal data (pii: true) get no checks on their values. The column set and the types
are not GE expectations: validate.py compares them with the Parquet schema of each day.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import great_expectations as gx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "eda"))
import contracts  # noqa: E402
from profile_tables import NON_NEGATIVE, SIGNED  # noqa: E402

SUITES_DIR = PROJECT_ROOT / "gx/expectations"
# Date columns that should hold the day of the file, in order of preference.
BATCH_DAY_COLUMNS = ["snapshot_date", "event_date", "event_hour", "window_end_date"]
NUMERIC = ("int", "uint", "float", "double", "decimal")
BATCH_DAY = "_batch_day"


def batch_day_column(table: contracts.TableDef) -> str | None:
    names = {c.name for c in table.columns}
    return next((c for c in BATCH_DAY_COLUMNS if c in names), None)


def expectations(table: contracts.TableDef) -> list:
    out = []
    grain = table.grain
    if len(grain) > 1:
        out.append(gx.expectations.ExpectCompoundColumnsToBeUnique(column_list=grain, meta={"rule": "grain"}))
    elif grain:
        out.append(gx.expectations.ExpectColumnValuesToBeUnique(column=grain[0], meta={"rule": "grain"}))
    by_name = table.layer == "org/40-gold"
    for c in table.columns:
        if c.pii:
            continue
        if not c.nullable:
            out.append(gx.expectations.ExpectColumnValuesToNotBeNull(column=c.name, meta={"rule": "not_null"}))
        if c.range:
            out.append(gx.expectations.ExpectColumnValuesToBeBetween(
                column=c.name, min_value=c.range[0], max_value=c.range[1], meta={"rule": "range"}))
        elif (by_name and c.type.startswith(NUMERIC) and not c.type.startswith("uint")
              and NON_NEGATIVE.search(c.name) and not SIGNED.search(c.name)):
            out.append(gx.expectations.ExpectColumnValuesToBeBetween(
                column=c.name, min_value=0, meta={"rule": "non_negative"}))
    day = batch_day_column(table)
    if day:
        out.append(gx.expectations.ExpectColumnValuesToBeInSet(
            column=BATCH_DAY, value_set={"$PARAMETER": "batch_days"}, meta={"rule": "batch_day", "column": day}))
    return out


def suite_dict(table: contracts.TableDef) -> dict:
    """The suite as JSON, without GE's random ids, so the file only changes when a rule changes."""
    suite = gx.ExpectationSuite(name=table.name, expectations=expectations(table),
                                meta={"contract": f"configs/contracts/lake/{table.name}.yaml",
                                      "batch_day_column": batch_day_column(table)})
    raw = suite.to_json_dict()
    raw.pop("id", None)
    for e in raw["expectations"]:
        e.pop("id", None)
    return raw


def suite_hash(raw: dict) -> str:
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:12]


def load(name: str) -> dict:
    return json.loads((SUITES_DIR / f"{name}.json").read_text())


def columns_used(raw: dict) -> list[str]:
    """Columns that the expectations read, in table order of first use."""
    out = []
    for e in raw["expectations"]:
        kw = e["kwargs"]
        for c in kw.get("column_list", []) + ([kw["column"]] if "column" in kw else []):
            if c != BATCH_DAY and c not in out:
                out.append(c)
    return out


def run(tables: list[str] | None = None) -> list[Path]:
    lake = contracts.load()
    SUITES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in tables or sorted(lake):
        raw = suite_dict(lake[name])
        path = SUITES_DIR / f"{name}.json"
        path.write_text(json.dumps(raw, indent=1, default=str) + "\n")
        rules = {}
        for e in raw["expectations"]:
            rules[e["meta"]["rule"]] = rules.get(e["meta"]["rule"], 0) + 1
        print(f"{name}: {len(raw['expectations'])} expectations {rules}")
        paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Great Expectations suites from the lake contracts.")
    parser.add_argument("--tables", nargs="+", help="default: every lake contract")
    run(parser.parse_args().tables)
