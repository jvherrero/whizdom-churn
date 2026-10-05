"""Pandera suites checked into version control, one per source table plus one
for the feature snapshot `build_feature_store()` produces.

    from expectations import SUITES
    SUITES["bet"].validate(some_bet_dataframe)

or, for the issue list this project's own report format expects:

    from expectations import bet
    result = bet.validate(some_bet_dataframe)  # dq_lib.ValidationResult
"""

from . import bet, bonus, feature_snapshot, player, transaction

SUITES = {
    "player": player.SCHEMA,
    "bet": bet.SCHEMA,
    "transaction": transaction.SCHEMA,
    "bonus": bonus.SCHEMA,
    "feature_snapshot": feature_snapshot.SCHEMA,
}

__all__ = ["SUITES", "player", "bet", "transaction", "bonus", "feature_snapshot"]
