"""Contracts of the tables that exist in the lake, from the layer inventory (make layers ROWS=1).

    .venv/bin/python src/quality/inventory_to_contracts.py            # writes configs/contracts/lake/
    .venv/bin/python src/quality/inventory_to_contracts.py --out /tmp/lake

Where each field comes from:
    columns, type     schema of the last Parquet file of the table (Arrow types)
    grain             "grains" in configs/quality.yaml if set; otherwise the design grain when the table
                      has a design contract with the same name and all its grain columns; otherwise a
                      proposal from the id and date columns. grain_checked is true only for the tables
                      listed in "grains_checked", once phase 1 shows the grain is unique.
    group, unit,      the design column with the same name, first in the design table that the
    range, additive   inventory matched, then in any design table (only if all of them agree)
    range             "ranges" in configs/quality.yaml wins over the design range
    nullable          false for the grain columns, true for the rest (the data decides, not the design)
    pii               the column name matches "pii_columns" in configs/quality.yaml
Running it again overwrites manual edits: change configs/quality.yaml instead, or compare with --out.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contracts  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INVENTORY = PROJECT_ROOT / "docs/dq_reports/layer_inventory.json"
CONFIG_PATH = PROJECT_ROOT / "configs/quality.yaml"
OUT_DIR = PROJECT_ROOT / "configs/contracts/lake"
ID_COLUMNS = ["tenant_id", "brand_id", "player_id", "game_id", "platform_id", "payment_id", "bonus_id",
              "change_id", "entity_id", "segment_kind", "segment_band"]
DATE_COLUMNS = ["snapshot_date", "event_date", "event_hour", "window_end_date", "cohort_date"]
ENRICHED = ("group", "unit", "range", "additive")


def design_index(design: dict) -> dict[str, list[contracts.ColumnDef]]:
    index: dict[str, list[contracts.ColumnDef]] = {}
    for table in design.values():
        for c in table.columns:
            index.setdefault(c.name, []).append(c)
    return index


def enrich(name: str, matched: contracts.TableDef | None, index: dict) -> dict:
    """group, unit, range and additive of the design column with the same name."""
    source = matched.column(name) if matched else None
    if source is None:
        candidates = index.get(name, [])
        values = {tuple(getattr(c, f) for f in ENRICHED) for c in candidates}
        source = candidates[0] if len(values) == 1 else None
    if source is None:
        return {}
    out = {f: getattr(source, f) for f in ENRICHED if getattr(source, f) is not None}
    if "range" in out:
        out["range"] = list(out["range"])
    return out


def proposed_grain(columns: list[str]) -> list[str]:
    """Id columns plus the first date column. Brand level tables (no player or game) add currency."""
    grain = [c for c in ID_COLUMNS if c in columns]
    date = next((c for c in DATE_COLUMNS if c in columns), None)
    if date:
        grain.append(date)
    if not {"player_id", "game_id", "entity_id"} & set(grain) and "currency" in columns:
        grain.append("currency")
    return grain


def contract(item: dict, design: dict, index: dict, cfg: dict) -> tuple[str, str]:
    name = item["table"].split("/")[-1]
    columns = [c for c, _ in item["schema"]]
    matched = design.get((item.get("contract") or {}).get("contract") or "")
    pii = re.compile(cfg["pii_columns"])
    meta = re.compile(cfg["meta_columns"])
    same_name = design.get(name)

    if name in cfg.get("grains", {}):
        grain, origin = cfg["grains"][name], "set by hand in configs/quality.yaml"
    elif same_name and set(same_name.grain) <= set(columns):
        grain, origin = same_name.grain, f"design contract {name}"
    else:
        grain, origin = proposed_grain(columns), "proposed from the id and date columns"
    checked = name in (cfg.get("grains_checked") or [])

    head = {"table": name, "layer": item["layer"], "grain": grain, "grain_checked": checked,
            "partition_by": f"{item['layout']} folders" if item.get("layout") else None,
            "engine": None}
    lines = [f"# {name}: table in {item['layer']}/{item['table']}, from the layer inventory of "
             f"{dt.date.today()}.",
             f"# Grain: {origin}." + ("" if checked else " Not checked on the data yet."),
             f"# Design contract: {matched.name if matched else 'none'}. See configs/contracts/README.md.",
             yaml.safe_dump(head, default_flow_style=None, sort_keys=False, width=200).rstrip(),
             "columns:"]
    for col, arrow_type in item["schema"]:
        entry = {"name": col, "type": arrow_type}
        if col in grain:
            entry["key"] = True
        entry |= enrich(col, matched, index)
        if col in cfg.get("ranges", {}).get(name, {}):
            entry["range"] = [float(x) for x in cfg["ranges"][name][col]]
            if entry.get("unit") == "0-1":   # the design unit no longer holds
                if entry["range"] == [0.0, 100.0]:
                    entry["unit"] = "0-100"
                else:
                    del entry["unit"]
        if meta.match(col):
            entry["group"] = "meta"
        entry["nullable"] = col not in grain
        if pii.match(col):
            entry["pii"] = True
        lines.append(f"  - {yaml.safe_dump(entry, default_flow_style=True, sort_keys=False, width=1000).strip()}")
    return name, "\n".join(lines) + "\n"


def run(out_dir: Path = OUT_DIR, inventory: Path = INVENTORY) -> list[str]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    design = contracts.load("design")
    index = design_index(design)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for item in json.loads(inventory.read_text()):
        if not item.get("schema"):
            continue
        name, text = contract(item, design, index, cfg)
        (out_dir / f"{name}.yaml").write_text(text)
        written.append(name)
    print(f"{len(written)} contracts saved in {out_dir}")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lake table contracts from the layer inventory.")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="write here instead, e.g. to diff")
    args = parser.parse_args()
    run(args.out)
