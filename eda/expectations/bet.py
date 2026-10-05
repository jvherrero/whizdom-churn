"""Pandera suite for `bet` (`whizdomai-transactions`), one row per bet or
transaction event, `transactionId` unique.

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dq_lib import ColumnSpec, build_schema, validate_dataframe

COLUMNS = [
    ColumnSpec("uuid", "object", False),
    ColumnSpec("transactionId", "int64", False),
    ColumnSpec("platformId", "int64", False),
    ColumnSpec("accountId", "int32", False),
    ColumnSpec("partyId", "int32", False),
    ColumnSpec("firstname", "object", True),
    ColumnSpec("lastname", "object", True),
    ColumnSpec("address", "object", True),
    ColumnSpec("email", "object", False),
    ColumnSpec("nickname", "object", True),
    ColumnSpec("ip", "object", False),
    ColumnSpec("mobile", "bool", False),
    ColumnSpec("os", "object", False),
    ColumnSpec("ua", "object", False),
    ColumnSpec("device", "object", False),
    ColumnSpec("brandId", "int32", False),
    ColumnSpec("dateTime", "datetime64[us, Europe/Madrid]", False),
    ColumnSpec("tranType", "object", False),
    ColumnSpec("currency", "object", False),
    ColumnSpec("amountReal", "float64", False),
    ColumnSpec("amountReleasedBonus", "float64", False),
    ColumnSpec("amountPlayableBonus", "float64", False),
    ColumnSpec("amountRawLoyalty", "int64", False),
    ColumnSpec("balanceRawLoyalty", "int64", False),
    ColumnSpec("balanceReal", "float64", False),
    ColumnSpec("balanceReleasedBonus", "float64", False),
    ColumnSpec("balancePlayableBonus", "float64", False),
    ColumnSpec("platformTranId", "object", True),
    ColumnSpec("gameTranId", "object", True),
    ColumnSpec("gameId", "object", True),
    ColumnSpec("rollbackTranId", "Int64", True),
    ColumnSpec("rolledBack", "bool", False),
]

SCHEMA = build_schema(COLUMNS)


def validate(df):
    return validate_dataframe(df, COLUMNS)
