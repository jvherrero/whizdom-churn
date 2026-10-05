"""Pandera suite for `transaction` (`whizdomai-payments`), one row per payment
status-change event, not one row per payment, `paymentId` repeats. """

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dq_lib import ColumnSpec, build_schema, validate_dataframe

COLUMNS = [
    ColumnSpec("paymentId", "int32", False),
    ColumnSpec("partyId", "int32", False),
    ColumnSpec("userId", "object", False),
    ColumnSpec("brandId", "int32", False),
    ColumnSpec("transactionType", "object", False),
    ColumnSpec("transactionMethod", "object", False),
    ColumnSpec("accountNumber", "int32", False),
    ColumnSpec("requestedAmount", "float64", True),
    ColumnSpec("processedAmount", "float64", False),
    ColumnSpec("fee", "float64", False),
    ColumnSpec("status", "object", False),
    ColumnSpec("requestDate", "datetime64[us, Europe/Madrid]", False),
    ColumnSpec("processDate", "datetime64[us, Europe/Madrid]", True),
    ColumnSpec("currency", "object", False),
    ColumnSpec("operator", "object", False),
]

SCHEMA = build_schema(COLUMNS)


def validate(df):
    return validate_dataframe(df, COLUMNS)
