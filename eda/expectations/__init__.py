"""Pandera suites, one per daily cache and one for the feature snapshot.

    from expectations import SUITES
    SUITES["activity"].validate(df)

    from expectations import activity
    result = activity.validate(df)  # dq_lib.ValidationResult
"""

from . import feature_snapshot, activity, financial, payments

SUITES = {
    "activity": activity.SCHEMA,
    "financial": financial.SCHEMA,
    "payments": payments.SCHEMA,
    "feature_snapshot": feature_snapshot.SCHEMA,
}

__all__ = ["SUITES", "activity", "financial", "payments", "feature_snapshot"]
