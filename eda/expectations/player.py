"""Pandera suite for `player` (`whizdomai-players`), one row per player change
event, `uuid` unique, `partyId` repeats. """

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dq_lib import ColumnSpec, build_schema, validate_dataframe

COLUMNS = [
    ColumnSpec("uuid", "object", False),
    ColumnSpec("partyId", "int32", False),
    ColumnSpec("brandId", "int32", False),
    ColumnSpec("isActive", "bool", False),
    ColumnSpec("lockedStatus", "object", False),
    ColumnSpec("lockedSince", "datetime64[us, Europe/Madrid]", True),
    ColumnSpec("lockedUntil", "datetime64[us, Europe/Madrid]", True),
    ColumnSpec("loginAttempts", "int32", False),
    ColumnSpec("firstname", "object", True),
    ColumnSpec("lastname", "object", True),
    ColumnSpec("birthdate", "datetime64[us, Europe/Madrid]", True),
    ColumnSpec("regdate", "datetime64[us, Europe/Madrid]", False),
    ColumnSpec("country", "object", True),
    ColumnSpec("city", "object", True),
    ColumnSpec("province", "object", True),
    ColumnSpec("address", "object", True),
    ColumnSpec("postalCode", "object", True),
    ColumnSpec("building", "object", True),
    ColumnSpec("unit", "object", True),
    ColumnSpec("language", "object", False),
    ColumnSpec("currency", "object", False),
    ColumnSpec("nationalRegNumber", "object", True),
    ColumnSpec("birthCountry", "object", True),
    ColumnSpec("birthCity", "object", True),
    ColumnSpec("profession", "object", True),
    ColumnSpec("nationality", "object", True),
    ColumnSpec("phone", "object", True),
    ColumnSpec("email", "object", True),
    ColumnSpec("nickname", "object", False),
    ColumnSpec("iban", "object", True),
    ColumnSpec("documentNumber", "object", True),
    ColumnSpec("documentType", "object", True),
    ColumnSpec("kycStatus", "object", False),
    ColumnSpec("lastKycRequestedDate", "datetime64[us, Europe/Madrid]", True),
    ColumnSpec("lastDepositDate", "datetime64[us, Europe/Madrid]", True),
    ColumnSpec("lastDepositAmount", "float64", True),
    ColumnSpec("lastDepositType", "object", True),
    ColumnSpec("lastWithdrawDate", "datetime64[us, Europe/Madrid]", True),
    ColumnSpec("lastWithdrawAmount", "float64", True),
    ColumnSpec("trackingCodes", "object", False),
    ColumnSpec("operator", "object", False),
]

SCHEMA = build_schema(COLUMNS)


def validate(df):
    return validate_dataframe(df, COLUMNS)
