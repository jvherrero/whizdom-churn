"""Pandera suite for the final features, one row per player and cutoff.

All features are float64. Many can be empty (for example deposits before March 2026), but the ones
every active player has cannot.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "features"))
from dq_lib import ColumnSpec, build_schema, validate_dataframe
from build_features import FEATURES

ID_COLUMNS = [
    ColumnSpec("cutoff_date", "object", False),
    ColumnSpec("tenant_id", "object", False),
    ColumnSpec("brandId", "int64", False),
    ColumnSpec("player_id", "int32", False),
]
ALWAYS_PRESENT = {"days_since_last_bet", "days_since_first_bet"}

COLUMNS = ID_COLUMNS + [ColumnSpec(name, "float64", name not in ALWAYS_PRESENT) for name in FEATURES]

SCHEMA = build_schema(COLUMNS)


def validate(df):
    return validate_dataframe(df, COLUMNS)
