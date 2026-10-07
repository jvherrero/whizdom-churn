"""Pandera suites checked into version control, one per source table (the local gold caches the
pipeline reads, src/features/gold_cache.py) plus one for the feature snapshot `build_feature_store()`
produces.

    from expectations import SUITES
    SUITES["gold_activity"].validate(some_activity_dataframe)

or, for the issue list this project's own report format expects:

    from expectations import gold_activity
    result = gold_activity.validate(some_activity_dataframe)  # dq_lib.ValidationResult
"""

from . import feature_snapshot, gold_activity, gold_financial, gold_payments

SUITES = {
    "gold_activity": gold_activity.SCHEMA,
    "gold_financial": gold_financial.SCHEMA,
    "gold_payments": gold_payments.SCHEMA,
    "feature_snapshot": feature_snapshot.SCHEMA,
}

__all__ = ["SUITES", "gold_activity", "gold_financial", "gold_payments", "feature_snapshot"]
