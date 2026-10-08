"""Pandera suite for `build_feature_store()`'s output (`ID_COLUMNS + FEATURES`,
src/features/build_features.py), one row per player per cutoff_date.

Every feature is float64. Most may be empty by design (deposit features before March 2026, ratios
over 0, a player missing from a signal snapshot); the ones every player of the population must have
(a bet in the 30 days up to the cutoff) are not nullable.
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
