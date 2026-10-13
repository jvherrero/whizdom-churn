"""Pandera suite for the payments cache: one row per player and day with payments.
Keys are never empty. Values can be, and the null-rate alert checks them."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dq_lib import ColumnSpec, build_schema, validate_dataframe

COLUMNS = [
    ColumnSpec("tenant_id", "object", False),
    ColumnSpec("player_id", "int32", False),
    ColumnSpec("day", "datetime64[us]", False),
    ColumnSpec("deposit_count", "int64", True),
    ColumnSpec("deposit_attempts", "int64", True),
    ColumnSpec("failed_deposit_count", "int64", True),
    ColumnSpec("withdraw_count", "int64", True),
    ColumnSpec("failed_withdraw_count", "int64", True),
    ColumnSpec("deposit_eur", "float64", True),
    ColumnSpec("withdraw_eur", "float64", True),
]

SCHEMA = build_schema(COLUMNS)


def validate(df):
    return validate_dataframe(df, COLUMNS)
