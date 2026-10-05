"""Pandera suite for `build_feature_store()`'s default output (`ID_COLUMNS +
BASELINE_FEATURES`, src/features/build_features.py), one row per player per
cutoff_date.

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dq_lib import ColumnSpec, build_schema, validate_dataframe

ID_COLUMNS = [
    ColumnSpec("cutoff_date", "object", False),
    ColumnSpec("operator", "object", False),
    ColumnSpec("brandId", "int64", False),
    ColumnSpec("partyId", "int32", False),
    ColumnSpec("label_available_60d", "bool", False),
]

_FLOAT_FEATURES = [
    "days_since_last_active", "days_since_last_deposit", "days_since_last_bonus",
    "n_active_days_last_7d", "n_active_days_last_30d",
    "n_deposit_last_7d", "n_deposit_last_30d",
    "n_bonus_last_7d",
    "stake_last_7d", "stake_last_30d",
    "win_last_7d", "win_last_30d",
    "net_loss_last_7d", "net_loss_last_30d",
    "deposit_amount_last_7d", "deposit_amount_last_30d",
    "bonus_amount_last_7d",
    "freq_drop_7_vs_prior7", "deposit_drop_7_vs_prior7", "bonus_drop_7_vs_prior7",
    "stake_trend_7", "win_trend_7", "net_loss_trend_7", "deposit_amount_trend_7", "bonus_amount_trend_7",
    "tenure_days", "max_prior_gap_days",
    "rtp_last_7d",
]

COLUMNS = (
    ID_COLUMNS
    + [ColumnSpec(name, "float64", False) for name in _FLOAT_FEATURES]
    + [ColumnSpec("already_dormant_7", "int64", False)]
)

SCHEMA = build_schema(COLUMNS)


def validate(df):
    return validate_dataframe(df, COLUMNS)
