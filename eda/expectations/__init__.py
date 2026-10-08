"""Pandera suites checked into version control, one per source table (the local daily caches the
pipeline reads, src/features/daily_cache.py) plus one for the feature snapshot `build_feature_store()`
produces.

    from expectations import SUITES
    SUITES["activity"].validate(some_activity_dataframe)

or, for the issue list this project's own report format expects:

    from expectations import activity
    result = activity.validate(some_activity_dataframe)  # dq_lib.ValidationResult
"""

from . import feature_snapshot, activity, financial, payments

SUITES = {
    "activity": activity.SCHEMA,
    "financial": financial.SCHEMA,
    "payments": payments.SCHEMA,
    "feature_snapshot": feature_snapshot.SCHEMA,
}

__all__ = ["SUITES", "activity", "financial", "payments", "feature_snapshot"]
