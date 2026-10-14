"""One-off conversion of the gold schema workbook (gld-schemas.xlsx) into the table contracts of
configs/contracts/design/. I ran it once on 2026-10-14. Since then the YAML files are the source of truth and
I edit them by hand, so running it again overwrites those edits. I keep it to show where each field of
the contracts comes from. No other code reads the workbook.

    uv run --with openpyxl python src/quality/excel_to_contracts.py --out /tmp/contracts
    diff -r /tmp/contracts configs/contracts/design

openpyxl is not a project dependency: --with installs it only for this run.

Where each field comes from:
    table, columns        one sheet per table, found by its header row (TABLE_HEADER)
    grain, partition_by   "Table Summary" sheet: ORDER BY and PARTITION BY
    engine                "Table Summary" sheet: engine name without its parameters
    type, key, group      table sheet: "Data type", "Key?" and "Group"
    unit, additive        "Measures" sheet, by the measure id of the column: "Unit", "Can it be added up?"
    range                 clamp(lo, hi, ...) in the column rule, otherwise [0, 1] when the unit is 0-1
    status                table sheet: "Source status", as a short code (STATUS)
The "Score Specification" sheet is not used: its M### ids do not match the ones of "Measures".
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import openpyxl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXCEL_PATH = PROJECT_ROOT / "gld-schemas.xlsx"
OUT_DIR = PROJECT_ROOT / "configs/contracts/design"
TABLE_HEADER = ["#", "Column", "Data type", "Group", "Key?", "Description", "Business rule / formula",
                "Source status", "Notes (Decision / Review id)", "Measure / Dim ID"]
CLAMP = re.compile(r"clamp\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)")
# Source status in the workbook -> short code. Any other value (one cell has a note) gives no status.
STATUS = {"EXISTS on prod": "exists", "RENAMED": "renamed", "NEW / CHANGED": "new", "NEW TABLE": "new",
          "NEW": "new", "NEW — derivable from our history": "new", "AVAILABLE — internal to churn model": "new",
          "NEEDS PIPELINE — bronze only": "needs_pipeline", "NEEDS SOURCE — not shipped upstream": "needs_source",
          "AVAILABLE — slv, not in gold": "in_silver", "AVAILABLE — slv_players": "in_silver"}


def text(value) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def rows(sheet) -> list[tuple]:
    return [tuple(text(c) for c in row) for row in sheet.iter_rows(values_only=True)]


def table_summary(sheet) -> dict[str, dict]:
    """Grain, partition and engine of each table. The grain is the ORDER BY, e.g. (tenant_id, brand_id)."""
    out = {}
    for row in rows(sheet):
        name, order_by = row[0], row[1] if len(row) > 1 else None
        if not name or not order_by or not order_by.startswith("("):
            continue
        out[name] = {"grain": [c.strip() for c in order_by.strip("()").split(",") if c.strip()],
                     "partition_by": None if row[2] in (None, "(none)") else row[2],
                     "engine": (row[5] or "").split("(")[0] or None}
    return out


def measures(sheet) -> dict[str, dict]:
    """Unit and additivity of each measure id (M###)."""
    data = rows(sheet)
    header_at = next(i for i, r in enumerate(data) if r and r[0] == "ID")
    header = list(data[header_at])
    unit_at, add_at = header.index("Unit"), header.index("Can it be added up?")
    return {r[0]: {"unit": r[unit_at], "additive": str(r[add_at]).startswith("Yes")}
            for r in data[header_at + 1:] if r and r[0] and r[0].startswith("M")}


def column(r: tuple, measure: dict) -> tuple[dict, str]:
    """One column as a contract entry, plus a YAML comment when the workbook contradicts itself."""
    ch_type, rule, unit = r[2], r[6], measure.get("unit")
    entry = {"name": r[1], "type": ch_type}
    if (r[4] or "").upper().startswith("YES"):
        entry["key"] = True
    if r[3]:
        entry["group"] = r[3]
    if unit:
        entry["unit"] = unit
    note = ""
    clamp = CLAMP.search(rule or "")
    if clamp:
        entry["range"] = [float(clamp.group(1)), float(clamp.group(2))]
        if unit == "0-1":
            note = "  # rule clamps to 0..100 but the unit is 0-1: check on prod"
    elif unit == "0-1":
        entry["range"] = [0.0, 1.0]
    if "additive" in measure:
        entry["additive"] = measure["additive"]
    if STATUS.get(r[7]):
        entry["status"] = STATUS[r[7]]
    return entry, note


def contract(name: str, data: list[tuple], summary: dict, units: dict) -> str:
    header_at = next(i for i, r in enumerate(data) if list(r[:len(TABLE_HEADER)]) == TABLE_HEADER)
    columns = [r for r in data[header_at + 1:] if r[1] and r[2]]
    info = summary.get(name, {})
    grain = info.get("grain") or [r[1] for r in columns if (r[4] or "").upper().startswith("YES")]
    head = {"table": name, "grain": grain, "partition_by": info.get("partition_by"), "engine": info.get("engine")}
    lines = [f"# {name}: target design of the gold rebuild (2026-09-01), used as the contract of the data checks.",
             "# See configs/contracts/README.md for the fields.",
             yaml.safe_dump(head, default_flow_style=None, sort_keys=False, width=200).rstrip(),
             "columns:"]
    for r in columns:
        entry, note = column(r, units.get(r[9] or "", {}))
        lines.append(f"  - {yaml.safe_dump(entry, default_flow_style=True, sort_keys=False, width=1000).strip()}{note}")
    return "\n".join(lines) + "\n"


def run(excel: Path = EXCEL_PATH, out_dir: Path = OUT_DIR) -> None:
    workbook = openpyxl.load_workbook(excel, read_only=True, data_only=True)
    summary = table_summary(workbook["Table Summary"])
    units = measures(workbook["Measures"])
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in workbook.sheetnames:
        data = rows(workbook[name])
        if any(list(r[:len(TABLE_HEADER)]) == TABLE_HEADER for r in data[:5]):
            (out_dir / f"{name}.yaml").write_text(contract(name, data, summary, units))
            n += 1
    workbook.close()
    print(f"{n} contracts saved in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert gld-schemas.xlsx into configs/contracts/design/*.yaml.")
    parser.add_argument("--excel", type=Path, default=EXCEL_PATH)
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="write here instead, e.g. to diff")
    args = parser.parse_args()
    run(args.excel, args.out)
