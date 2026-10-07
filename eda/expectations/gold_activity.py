"""Pandera suite for the local gold cache `activity` (src/features/gold_cache.py):
gld_player_gaming_daily, summed per player and day (bets > 0).
One row per (tenant_id, player_id, day). Keys are never empty; a value may be (the null-rate alert
watches its share)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dq_lib import ColumnSpec, build_schema, validate_dataframe

COLUMNS = [
    ColumnSpec("tenant_id", "object", False),
    ColumnSpec("brand_id", "int32", False),
    ColumnSpec("player_id", "int32", False),
    ColumnSpec("day", "datetime64[us]", False),
    ColumnSpec("bets", "int64", True),
    ColumnSpec("rounds", "int64", True),
    ColumnSpec("turnover_eur", "float64", True),
    ColumnSpec("ggr_eur", "float64", True),
]

SCHEMA = build_schema(COLUMNS)


def validate(df):
    return validate_dataframe(df, COLUMNS)
