"""Table contracts for the data checks, one YAML file per table.

    configs/contracts/lake/     tables that exist in the lake today (from the layer inventory)
    configs/contracts/design/   gold tables of the rebuild design (from the design workbook)

    tables = load()                      # lake contracts: {table name: TableDef}
    design = load("design")
    design["gld_player_signal"].grain    # ['tenant_id', 'brand_id', 'event_date', 'player_id']

The fields are explained in configs/contracts/README.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = PROJECT_ROOT / "configs/contracts"
WRAPPERS = ("Nullable(", "LowCardinality(")


@dataclass(frozen=True)
class ColumnDef:
    name: str
    type: str                                  # ClickHouse type (design) or Arrow type (lake)
    key: bool = False                          # part of the grain
    group: str | None = None
    unit: str | None = None                    # currency, days, 0-1, ...
    range: tuple[float, float] | None = None   # allowed values, both ends included
    additive: bool | None = None               # can be summed across days and players
    status: str | None = None                  # design only: exists, renamed, new, in_silver, ...
    nullable: bool | None = None               # None: taken from a ClickHouse Nullable(...) type
    pii: bool = False                          # personal data: never print its values

    def __post_init__(self):
        if self.nullable is None:
            t = self.type
            if t.startswith("LowCardinality("):
                t = t[len("LowCardinality("):-1]
            object.__setattr__(self, "nullable", t.startswith("Nullable("))

    @property
    def base_type(self) -> str:
        """Type without Nullable(...) and LowCardinality(...), e.g. UInt16 or Decimal(18, 4)."""
        t = self.type
        while t.startswith(WRAPPERS):
            t = t[t.index("(") + 1:-1]
        return t


@dataclass
class TableDef:
    name: str
    grain: list[str]          # ORDER BY columns, or the proposed unique key
    partition_by: str | None
    engine: str | None
    columns: list[ColumnDef] = field(default_factory=list)
    layer: str | None = None  # lake contracts: e.g. org/40-gold
    grain_checked: bool = True  # False until the grain is shown to be unique on real data

    def column(self, name: str) -> ColumnDef | None:
        return next((c for c in self.columns if c.name == name), None)


def _column(raw: dict) -> ColumnDef:
    if "range" in raw:
        raw = raw | {"range": tuple(raw["range"])}
    return ColumnDef(**raw)


def load(which: str = "lake", directory: Path | None = None) -> dict[str, TableDef]:
    """One TableDef per contract file of configs/contracts/<which>/."""
    tables = {}
    for path in sorted((directory or CONTRACTS_DIR / which).glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        columns = [_column(c) for c in raw["columns"]]
        tables[raw["table"]] = TableDef(raw["table"], raw["grain"], raw.get("partition_by"),
                                        raw.get("engine"), columns, raw.get("layer"),
                                        raw.get("grain_checked", True))
    return tables
