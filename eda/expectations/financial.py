"""Pandera suite for the financial cache: one row per player and day with money moving.
Keys are never empty. Values can be, and the null-rate alert checks them."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dq_lib import ColumnSpec, build_schema, validate_dataframe

COLUMNS = [
    ColumnSpec("tenant_id", "object", False),
    ColumnSpec("player_id", "int32", False),
    ColumnSpec("day", "datetime64[us]", False),
    ColumnSpec("wagered_eur", "float64", True),
    ColumnSpec("won_eur", "float64", True),
    ColumnSpec("ggr_eur", "float64", True),
    ColumnSpec("deposits_eur", "float64", True),
    ColumnSpec("withdrawals_eur", "float64", True),
    ColumnSpec("bonus_ggr_eur", "float64", True),
    ColumnSpec("wagered_bonus_eur", "float64", True),
]

SCHEMA = build_schema(COLUMNS)


def validate(df):
    return validate_dataframe(df, COLUMNS)
