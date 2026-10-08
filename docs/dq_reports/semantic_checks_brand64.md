# Semantic checks of gld_player_signals_daily (brand 64)

Each signal column against the same quantity recomputed from the daily caches (`src/features/daily_cache.py`), for the brand's players with a bet in the 30 days up to each day. A column matches a player within 2% (or 0.5 absolute); it is OK on a day when at least 95% of the players match.

| Column | Days checked | Median match | Worst match | Days not OK | Not OK from / to |
|---|---|---|---|---|---|
| `active_days_l30d` | 18 | 98.4% | 95.6% | 0 |  |
| `active_days_l7d` | 18 | 99.5% | 98.1% | 0 |  |
| `bets_l30d` | 18 | 100.0% | 99.7% | 0 |  |
| `bets_l7d` | 18 | 100.0% | 100.0% | 0 |  |
| `days_since_bet` | 18 | 16.0% | 3.5% | 18 | 2025-05-01 to 2026-10-01 |
| `deposits_eur_l30d` | 13 | 100.0% | 99.4% | 0 |  |
| `ggr_eur_l30d` | 18 | 99.8% | 97.5% | 0 |  |
| `ggr_eur_l7d` | 18 | 100.0% | 99.9% | 0 |  |
| `wagered_eur_l30d` | 18 | 100.0% | 99.1% | 0 |  |
| `wagered_eur_l7d` | 18 | 100.0% | 99.9% | 0 |  |
| `withdrawals_eur_l30d` | 13 | 100.0% | 100.0% | 0 |  |

## Completeness of completed deposits

Players with a completed deposit / players with a bet, per month (gld_player_payments_daily). A month below half the brand's usual share (75th percentile of the months) is INCOMPLETE: deposits are missing there, even where the source tables agree with each other.

| Month | Players betting | Depositors | Depositor share | Completed deposits | Failed deposits | Status |
|---|---|---|---|---|---|---|
| 2025-04 | 30,600 | 0 | 0.0% | 0 | 0 | INCOMPLETE |
| 2025-05 | 35,603 | 0 | 0.0% | 0 | 0 | INCOMPLETE |
| 2025-06 | 27,606 | 0 | 0.0% | 0 | 0 | INCOMPLETE |
| 2025-07 | 19,444 | 0 | 0.0% | 0 | 0 | INCOMPLETE |
| 2025-08 | 18,583 | 0 | 0.0% | 0 | 0 | INCOMPLETE |
| 2025-09 | 19,331 | 0 | 0.0% | 0 | 0 | INCOMPLETE |
| 2025-10 | 22,252 | 0 | 0.0% | 0 | 0 | INCOMPLETE |
| 2025-11 | 17,981 | 0 | 0.0% | 0 | 0 | INCOMPLETE |
| 2025-12 | 14,708 | 180 | 1.2% | 213 | 84,249 | INCOMPLETE |
| 2026-01 | 12,115 | 925 | 7.6% | 1,155 | 168,677 | INCOMPLETE |
| 2026-02 | 12,269 | 4,201 | 34.2% | 17,990 | 172,332 | OK |
| 2026-03 | 13,394 | 8,782 | 65.6% | 102,508 | 73,494 | OK |
| 2026-04 | 12,709 | 8,905 | 70.1% | 101,555 | 55,237 | OK |
| 2026-05 | 13,395 | 8,333 | 62.2% | 105,571 | 56,495 | OK |
| 2026-06 | 10,450 | 7,275 | 69.6% | 142,098 | 89,747 | OK |
| 2026-07 | 13,985 | 9,427 | 67.4% | 100,387 | 69,707 | OK |
| 2026-08 | 14,290 | 10,684 | 74.8% | 111,335 | 90,420 | OK |
| 2026-09 | 15,642 | 10,761 | 68.8% | 124,717 | 74,414 | OK |
| 2026-10 | 7,166 | 5,009 | 69.9% | 25,689 | 21,095 | OK |
