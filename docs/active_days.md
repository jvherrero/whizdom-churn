# Active days: definition, check against the source and rebuild (review item P0 2)

## 1. Definition

An **active day** is a calendar day (the `event_date` of `gld_player_gaming_daily`) on which the player staked a non-zero amount from real money or released bonus funds; stakes only from playable bonus funds do not count (measure M053 of `gld-schemas`). It is the same event as the churn label: churn is 60 days without an active day. The canonical source is the daily activity cache (`daily_cache` "activity" read with `active_only=True`: one row per player and day with `wagered_real + wagered_released_bonus > 0` in `gld_player_gaming_daily`), whose grain was validated with Great Expectations and whose bet counts match the raw Kafka transactions (`docs/data_quality/data_quality.pdf`).

| Feature | Window (days before the cutoff) |
|---|---|
| `active_days_l7d` | 0 to 6 (the cutoff day is day 0) |
| `active_days_l30d` | 0 to 29 (the cutoff day is day 0) |
| `active_days_l90d` | 0 to 89 (the cutoff day is day 0) |
| `prior_active_days_l7d` | 7 to 13 (the cutoff day is day 0) |

## 2. The signals table against the canonical count

Share of players with the same value, per brand and feature, over every cutoff of the last backtest. When the source differs, it is almost always higher: it counts the days with stakes only from playable bonus, which are not active days, and some days without a bet. It is lower in at most 0.01% of the players.

| Brand | Feature | Cutoffs | Player-cutoffs | Equal | Source higher | Source lower | Worst cutoff (equal) |
|---|---|---|---|---|---|---|---|
| 14 | `active_days_l30d` | 14 | 170,389 | 47.3% | 52.7% | 0.00% | 2026-04-01 (35.4%) |
| 14 | `active_days_l7d` | 14 | 170,389 | 86.4% | 13.6% | 0.00% | 2025-10-01 (66.5%) |
| 14 | `active_days_l90d` | 14 | 170,389 | 39.9% | 60.1% | 0.01% | 2026-04-01 (30.4%) |
| 14 | `prior_active_days_l7d` | 14 | 170,389 | 86.2% | 13.8% | 0.00% | 2026-04-01 (71.0%) |
| 23 | `active_days_l30d` | 14 | 477,839 | 49.0% | 51.0% | 0.00% | 2026-01-01 (33.9%) |
| 23 | `active_days_l7d` | 14 | 477,839 | 81.2% | 18.8% | 0.00% | 2025-09-01 (69.7%) |
| 23 | `active_days_l90d` | 14 | 477,839 | 34.0% | 66.0% | 0.00% | 2026-01-01 (25.0%) |
| 23 | `prior_active_days_l7d` | 14 | 477,839 | 82.1% | 17.9% | 0.00% | 2026-01-01 (69.6%) |
| 64 | `active_days_l30d` | 14 | 226,669 | 64.1% | 35.9% | 0.00% | 2026-06-01 (52.9%) |
| 64 | `active_days_l7d` | 14 | 226,669 | 90.7% | 9.3% | 0.00% | 2026-06-01 (85.2%) |
| 64 | `active_days_l90d` | 14 | 226,669 | 53.7% | 46.3% | 0.00% | 2025-07-01 (42.6%) |
| 64 | `prior_active_days_l7d` | 14 | 226,669 | 89.9% | 10.1% | 0.00% | 2026-01-01 (81.3%) |
| 72 | `active_days_l30d` | 14 | 258,704 | 53.7% | 46.3% | 0.00% | 2026-01-01 (37.5%) |
| 72 | `active_days_l7d` | 14 | 258,704 | 85.8% | 14.2% | 0.00% | 2026-04-01 (76.0%) |
| 72 | `active_days_l90d` | 14 | 258,704 | 48.3% | 51.7% | 0.00% | 2026-01-01 (34.8%) |
| 72 | `prior_active_days_l7d` | 14 | 258,704 | 85.7% | 14.3% | 0.00% | 2026-01-01 (72.6%) |
| 73 | `active_days_l30d` | 14 | 454,409 | 62.5% | 37.5% | 0.00% | 2026-01-01 (51.6%) |
| 73 | `active_days_l7d` | 14 | 454,409 | 84.7% | 15.3% | 0.00% | 2026-01-01 (74.9%) |
| 73 | `active_days_l90d` | 14 | 454,409 | 51.5% | 48.5% | 0.00% | 2026-01-01 (43.2%) |
| 73 | `prior_active_days_l7d` | 14 | 454,409 | 88.0% | 12.0% | 0.00% | 2026-01-01 (76.1%) |

**Decision.** The source cannot be validated independently: its definition of an active day is not documented and in some months it counts days with no bet. The features are rebuilt from the canonical activity (`src/features/build_features.py`, `active_days`), and the models and backtests are run again.

## 3. Backtest before and after the rebuild

Before: the source features and a bet from any wallet as activity. After: the features rebuilt from the official active days, which also change the population and the label, so the two rows of a brand are not scored on exactly the same players.

| Brand | Version | Mean AUC | Min AUC | Max ECE | Months ECE > 0.08 | Top-decile precision | Ratio vs recency | GGR at risk captured |
|---|---|---|---|---|---|---|---|---|
| 14 | before | 0.870 | 0.821 | 0.112 | 2 | 0.875 | 1.19 | 69.2% |
| 14 | after | 0.887 | 0.877 | 0.094 | 1 | 0.869 | 1.36 | 79.8% |
| 23 | before | 0.912 | 0.876 | 0.066 | 0 | 0.871 | 1.29 | 61.8% |
| 23 | after | 0.886 | 0.863 | 0.101 | 1 | 0.796 | 1.40 | 61.1% |
| 64 | before | 0.875 | 0.853 | 0.087 | 1 | 0.861 | 1.21 | 65.7% |
| 64 | after | 0.869 | 0.831 | 0.118 | 2 | 0.829 | 1.30 | 67.6% |
| 72 | before | 0.870 | 0.846 | 0.099 | 1 | 0.932 | 1.16 | 64.3% |
| 72 | after | 0.876 | 0.857 | 0.102 | 1 | 0.935 | 1.19 | 68.0% |
| 73 | before | 0.887 | 0.863 | 0.042 | 0 | 0.920 | 1.25 | 60.3% |
| 73 | after | 0.887 | 0.867 | 0.073 | 0 | 0.914 | 1.26 | 59.2% |
