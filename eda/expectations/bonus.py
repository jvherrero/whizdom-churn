"""Pandera suite for `bonus` (`whizdomai-bonuses`), one row per bonus
status-change event, `changeId` unique, `id` (the bonus itself) repeats.
Columns and dtypes come straight from the verified full scan in
`docs/dq_reports/bonus.json`, not guessed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dq_lib import ColumnSpec, build_schema, validate_dataframe

COLUMNS = [
    ColumnSpec("changeId", "int64", False),
    ColumnSpec("changeStatusTimestamp", "datetime64[us, Europe/Madrid]", False),
    ColumnSpec("id", "int32", False),
    ColumnSpec("partyId", "int32", False),
    ColumnSpec("bonusPlanId", "int32", False),
    ColumnSpec("triggerDate", "datetime64[us, Europe/Madrid]", False),
    ColumnSpec("expiryDate", "datetime64[us, Europe/Madrid]", False),
    ColumnSpec("releaseDate", "datetime64[us, Europe/Madrid]", True),
    ColumnSpec("status", "object", False),
    ColumnSpec("amountWagered", "float64", False),
    ColumnSpec("wagerRequirement", "float64", False),
    ColumnSpec("amount", "float64", False),
    ColumnSpec("releasedBonusWinnings", "float64", False),
    ColumnSpec("releasedBonus", "float64", False),
    ColumnSpec("releasedBonusWinningsAmount", "float64", False),
    ColumnSpec("releasedBonusAmount", "float64", False),
    ColumnSpec("playableBonusWinnings", "float64", False),
    ColumnSpec("playableBonus", "float64", False),
    ColumnSpec("operator", "object", False),
]

SCHEMA = build_schema(COLUMNS)


def validate(df):
    return validate_dataframe(df, COLUMNS)
