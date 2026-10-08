"""Pandera suite for the local daily cache `payments` (src/features/daily_cache.py):
gld_player_payments_daily, one brand, summed per player and processed day.
One row per (tenant_id, player_id, day). Keys are never empty; a value may be (the null-rate alert
watches its share)."""

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
