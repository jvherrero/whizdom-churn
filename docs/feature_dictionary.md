# Feature Dictionary v1 (brandId=64)

The features T6 builds from the S3 data lake (`build_features --as-of DATE`), one row per player and cutoff. Every signal promoted in T4 (`docs/churn_signals_v0.md`) is in it; the rest are standard RFM and lag features, marked "not tested", which T6's selection keeps or drops.

## 0. How to Read It

- **Cutoff**: the "today" of a row. A feature uses only data **up to and including the cutoff day** (UTC days, as in the source tables); nothing after it.
- **Population**: players of the brand with a bet in the 30 days up to the cutoff.
- **Window**: how many days before the cutoff the feature looks at. **Lag**: how far back the window ends (0 = it ends on the cutoff day; 7 d = days 8 to 14 before it).
- **T4**: the result in the signal explorer (promoted / rejected / not tested), with the single-signal AUC for 60-day churn.
- **Sources.** `gld_player_signals_daily` is read on the cutoff day (its windows are already computed). The other sources are the local daily caches of `src/features/daily_cache.py`. Money is in EUR (converted in the source tables).
- **Scaling.** Monetary features are winsorised at p99.5 (p0.5 too for signed ones) per brand, then sign-log scaled, as in v0.

## 1. Leakage Rules

- Features: data up to the cutoff only. `gld_player_signals_daily` at `snapshot_date = cutoff` includes the cutoff day itself, which is closed (as of end of day).
- Labels (section 3): data strictly after the cutoff only. Never an input.
- **Backfilled history.** The source tables were recomputed over their whole history in July and August 2026. That each past `snapshot_date` uses only data up to that day is assumed, and must be confirmed with the data team.
- **The last 7 days are reprocessed daily in the source.** A feature computed at scoring time can change slightly later; the training data always uses days older than that.
- A test (pytest) checks that no feature changes when data after the cutoff changes (T6).

## 2. Features (39 plus the brand)

### Recency (2)

| Feature | Definition | Source | Window | Lag | T4 | Notes |
|---|---|---|---|---|---|---|
| `days_since_last_bet` | Days from the last day with a bet to the cutoff | activity cache (`gld_player_gaming_daily`) | all history | 0 | promoted (AUC 0.758) | Replaces `days_since_bet` of `gld_player_signals_daily`, which is broken. |
| `days_since_last_deposit` | Days from the last completed deposit to the cutoff, capped at 31 (none in 30 days) | payments cache (`gld_player_payments_daily`) | 30 d | 0 | promoted (AUC 0.774) |  Deposit data: empty before its window is fully after 2026-03-01. |

### Frequency (9)

| Feature | Definition | Source | Window | Lag | T4 | Notes |
|---|---|---|---|---|---|---|
| `active_days_l7d` | Days with play in the last 7 days | `gld_player_signals_daily` `active_days_l7d` | 7 d | 0 | not tested | Matches the recomputation for 98% to 100% of players. |
| `active_days_l30d` | Days with play in the last 30 days | `gld_player_signals_daily` `active_days_l30d` | 30 d | 0 | not tested | Validated (95% to 100%). |
| `active_days_l90d` | Days with play in the last 90 days | `gld_player_signals_daily` `active_days_l90d` | 90 d | 0 | not tested |  |
| `bets_l7d` | Bets in the last 7 days | `gld_player_signals_daily` `bets_l7d` | 7 d | 0 | not tested | Validated (100%). |
| `bets_l30d` | Bets in the last 30 days | `gld_player_signals_daily` `bets_l30d` | 30 d | 0 | not tested | Validated (100%). |
| `sessions_l7d` | Sessions in the last 7 days | `gld_player_signals_daily` `sessions_l7d` | 7 d | 0 | not tested | Not validated (no session table cached). |
| `sessions_l30d` | Sessions in the last 30 days | `gld_player_signals_daily` `sessions_l30d` | 30 d | 0 | not tested | Not validated. |
| `n_deposit_days_7d` | Days with a completed deposit in the last 7 days | payments cache (`gld_player_payments_daily`) | 7 d | 0 | promoted (AUC 0.300) |  Deposit data: empty before its window is fully after 2026-03-01. |
| `n_deposit_days_30d` | Days with a completed deposit in the last 30 days | payments cache (`gld_player_payments_daily`) | 30 d | 0 | promoted (AUC 0.211) |  Deposit data: empty before its window is fully after 2026-03-01. |

### Monetary (7)

| Feature | Definition | Source | Window | Lag | T4 | Notes |
|---|---|---|---|---|---|---|
| `wagered_eur_l7d` | Stake, all wallets, EUR, last 7 days | `gld_player_signals_daily` `wagered_eur_l7d` | 7 d | 0 | not tested | Validated (100%). Winsorised at p99.5. |
| `wagered_eur_l30d` | Stake, EUR, last 30 days | `gld_player_signals_daily` `wagered_eur_l30d` | 30 d | 0 | not tested | Validated. Winsorised. |
| `wagered_eur_l90d` | Stake, EUR, last 90 days | `gld_player_signals_daily` `wagered_eur_l90d` | 90 d | 0 | not tested | Winsorised. |
| `net_loss_7d` | Player net loss (GGR), EUR, last 7 days; negative = the player won | `gld_player_signals_daily` `ggr_eur_l7d` | 7 d | 0 | promoted (AUC 0.358) | Validated. Signed; winsorised both ends. Activity proxy (T4). |
| `ggr_eur_l30d` | Player net loss (GGR), EUR, last 30 days | `gld_player_signals_daily` `ggr_eur_l30d` | 30 d | 0 | not tested | Validated. Signed; winsorised. |
| `deposits_eur_l30d` | Completed deposits, EUR, last 30 days | `gld_player_signals_daily` `deposits_eur_l30d` | 30 d | 0 | not tested |  Deposit data: empty before its window is fully after 2026-03-01. |
| `withdrawals_30d_share` | Withdrawals / deposits, EUR, last 30 days | payments cache (`gld_player_payments_daily`) | 30 d | 0 | promoted (AUC 0.371) |  Deposit data: empty before its window is fully after 2026-03-01. Activity proxy (T4). |

### Lag (3)

| Feature | Definition | Source | Window | Lag | T4 | Notes |
|---|---|---|---|---|---|---|
| `prior_wagered_eur_l7d` | Stake, EUR, in days 8 to 14 before the cutoff | `gld_player_signals_daily` `prior_wagered_eur_l7d` | 7 d | 7 d | not tested |  |
| `prior_active_days_l7d` | Days with play in days 8 to 14 before the cutoff | `gld_player_signals_daily` `prior_active_days_l7d` | 7 d | 7 d | not tested |  |
| `prior_ggr_eur_l30d` | GGR, EUR, in days 31 to 60 before the cutoff | `gld_player_signals_daily` `prior_ggr_eur_l30d` | 30 d | 30 d | not tested | Signed. |

### Trend (3)

| Feature | Definition | Source | Window | Lag | T4 | Notes |
|---|---|---|---|---|---|---|
| `wagered_trend_7` | (stake last 7 days + 1) / (stake days 8 to 14 + 1) | derived | 7 d vs 7 d | 0 / 7 d | not tested | From the two columns above. |
| `heavy_loss_multiple` | Net loss last 7 days / weekly average net loss of the last 90 | `gld_player_signals_daily` `ggr_eur_l7d`, `ggr_eur_l90d` | 7 d vs 90 d | 0 | promoted (AUC 0.388) | Empty when the 90-day net loss is not positive. Activity proxy (T4). |
| `deposits_30_vs_prior30` | Deposits last 30 days / deposits days 31 to 60 | `gld_player_signals_daily` `deposits_eur_l30d`, `prior_deposits_eur_l30d` | 30 d vs 30 d | 0 / 30 d | promoted (AUC 0.344) |  Deposit data: empty before its window is fully after 2026-03-01. |

### Tenure and lifecycle (3)

| Feature | Definition | Source | Window | Lag | T4 | Notes |
|---|---|---|---|---|---|---|
| `days_since_first_bet` | Days since the first bet seen in the activity, capped at 150 | activity cache (`gld_player_gaming_daily`) | since 2025-04-01, capped at 150 d | 0 | replaces `tenure_days` (AUC 0.68 to 0.80 per cutoff) | `tenure_days` of the signals table is not consistent across days in the backfilled history (see section 4). Capped so it means the same at every cutoff: each has at least 153 days of history. |
| `prior_dormancy_spells_14d` | Returns after 14 or more silent days, last 180 days | activity cache (`gld_player_gaming_daily`) | 180 d | 0 | promoted (AUC 0.354) | Activity proxy (T4). |
| `days_since_last_return` | Days since the last return after 14 or more silent days, capped at 181 | activity cache (`gld_player_gaming_daily`) | 180 d | 0 | promoted (AUC 0.605) |  |

### Mix and behaviour (7)

| Feature | Definition | Source | Window | Lag | T4 | Notes |
|---|---|---|---|---|---|---|
| `engagement_score` | Composite of active days, sessions and play time, 0 to 100 | `gld_player_signals_daily` `engagement_score` | 30 d | 0 | promoted (AUC 0.181) | Source formula: 0.4 active days + 0.3 sessions + 0.3 play time (gld-schemas.xlsx). |
| `games_breadth_30d` | Distinct games played, last 30 days | `gld_player_signals_daily` `games_breadth_l30d` | 30 d | 0 | promoted (AUC 0.287) |  |
| `games_breadth_ratio` | (distinct games last 30 days + 1) / (same 30 days before) | `gld_player_signals_daily` `games_breadth_l30d` at the cutoff and 30 days before | 30 d vs 30 d | 0 / 30 d | promoted (AUC 0.663) | High for new players (no previous month). |
| `bonus_stake_share_30d` | Share of the stake from bonus money, last 30 days | `gld_player_signals_daily` `wagered_bonus_eur_l30d` / `wagered_eur_l30d` | 30 d | 0 | promoted (AUC 0.639) |  |
| `losing_streak` | Consecutive latest playing days with a net loss (last 90 days) | financial cache (`gld_player_financial_daily`) | 90 d | 0 | promoted (AUC 0.322) | Activity proxy (T4). |
| `avg_bet_eur_l30d` | Average bet, EUR, last 30 days | `gld_player_signals_daily` `avg_bet_eur_l30d` | 30 d | 0 | not tested | Winsorised. |
| `night_play_index` | Share of play at night | `gld_player_signals_daily` `tp_night_index` | source window | 0 | not tested |  |

### Deposit and outcome (5)

| Feature | Definition | Source | Window | Lag | T4 | Notes |
|---|---|---|---|---|---|---|
| `deposited_within_3d` | A completed deposit in the last 3 days | payments cache (`gld_player_payments_daily`) | 3 d | 0 | promoted (AUC 0.355) |  Deposit data: empty before its window is fully after 2026-03-01. |
| `deposited_within_7d` | A completed deposit in the last 7 days | payments cache (`gld_player_payments_daily`) | 7 d | 0 | promoted (AUC 0.309) |  Deposit data: empty before its window is fully after 2026-03-01. |
| `deposited_within_14d` | A completed deposit in the last 14 days | payments cache (`gld_player_payments_daily`) | 14 d | 0 | promoted (AUC 0.286) |  Deposit data: empty before its window is fully after 2026-03-01. |
| `deposit_frequency_score` | Deposits last 7 days against the last 30, 0 to 100 (source score) | `gld_player_signals_daily` `deposit_frequency_score` | 7 d vs 30 d | 0 | promoted (AUC 0.320) |  Deposit data: empty before its window is fully after 2026-03-01. |
| `failed_deposits_14d` | Failed deposit attempts, last 14 days | payments cache (`gld_player_payments_daily`) | 14 d | 0 | promoted (AUC 0.341) |  Deposit data: empty before its window is fully after 2026-03-01. Activity proxy (T4). |


| Feature | Definition | Source | Window | Lag | T4 | Notes |
|---|---|---|---|---|---|---|

### Identity (1)

| Feature | Definition | Source | Window | Lag | T4 | Notes |
|---|---|---|---|---|---|---|
| `brandId` | Brand, as a categorical input (one model for several brands) | population |  |  | not tested | Not a behaviour; lets per-brand levels differ. |

## 3. Lead Features: Label Only, Never an Input

| Column | Definition | Source |
|---|---|---|
| `event_60d` | 1 = no bet in (cutoff, cutoff + 60 days] | activity cache, after the cutoff |
| `duration_days` | Days to the start of the first 60-day silence (Cox PH) | same |
| `event_observed` | 1 = that silence fits in the data, 0 = censored | same |
| `churn_within_7d`, `_14d`, `_30d` | 1 = the churn starts within 7 / 14 / 30 days of the cutoff (`duration_days` <= h); empty until cutoff + h + 60 days is in the data | same |
| activity and GGR after the cutoff | For value at risk (T11) | same |

## 4. Not Used, and Why

| Column or signal | Why |
|---|---|
| `days_since_bet` (signals table) | Broken: 0 for most players (`docs/dq_reports/semantic_checks_brand64.md`). Replaced by `days_since_last_bet`. |
| `churn_score`, `churn_band` (signals table) | The platform's current rule: kept as the **comparator** of the backtest (T10), not as an input, so the comparison stays fair. Also built on the broken recency. |
| `lifecycle_stage` (signals table) | Built from the broken `days_since_bet`. |
| `tenure_days` (signals table) | Not consistent across days in the backfilled history: between two monthly snapshots it grows by the elapsed days for only 11% to 22% of the players (99% in the live period from August 2026). Replaced by `days_since_first_bet`. |
| `cutoff_day_of_week` (calendar) | With monthly training cutoffs it is constant inside a cutoff, so it only identifies the month: the adversarial validation of the selection separates train and test months with it alone (AUC 0.998). |
| Deposits from `gld_player_financial_daily` | Only filled from June 2026 (always 0 before); deposits come from `gld_player_payments_daily`, which has them from March 2026. |
| Week-on-week session and active-day ratios | Rejected in T4: the effect flips side at 2 of 12 cutoffs. |
| Loss chasing, heavy-loss flag, bonus granted, the source's deposit recency score | Rejected in T4: AUC within 0.05 of 0.5. |
| Big win then withdrawal, withdrawal without redeposit | Rejected in T4: too rare to separate (AUC near 0.5), although strong where present. |

## 5. Data Limits That Shape the Features

- **Deposit features exist only from March 2026.** Before it the source has almost no completed deposits. They are empty (unknown, not 0) at earlier cutoffs. LightGBM handles empty values; for Cox PH they need an explicit choice in T6 (missing indicator, or train Cox on the cutoffs from April 2026).
- **`sessions_*` pass every plausibility check, `tenure_days` is wrong in the backfilled history** (`docs/signal_validation.md`): the pipeline uses `days_since_first_bet` instead.
- **Several risk signals measure activity volume** (losing streak, net loss, failed deposits, prior dormancy, withdrawal share): they predict churn because they go with playing more. They stay as candidates; T6's selection decides whether they add anything beyond the activity features.

## 6. Feature Store (Feast)

The features of every training cutoff are registered in Feast, so training rows can be fetched point in time and later served from the same definitions.

- **Definitions** (the versioned registry): `src/features/feature_repo/definitions.py`. Two entities (`tenant_id`, `player_id`), one source, one feature view `player_features` (`brandId` + the 39 features of section 2) and one feature service `churn_features`. Configuration: `src/features/feature_repo/feature_store.yaml`.
- **Offline store**: `data/02_intermediate/feature_store/player_features/brand{id}.parquet`, the final features (winsorised + sign-log) of every cutoff ever built, one file per brand. A rebuilt cutoff replaces its old rows. The registry file that `feast apply` builds from the definitions is `data/02_intermediate/feature_store/registry.db`.
- **Point in time**: `event_timestamp` is the cutoff (UTC) and the feature view's ttl is 1 day. `get_historical_features` gives a row the features of its own cutoff, and nothing before the first cutoff or older than a day. Every publish checks it on the latest cutoff (the store must give back exactly what was written), and `src/features/tests/test_feature_store.py` checks it on hand-made rows.
- **Local, not S3 and Redis**: the plan's S3 offline store and Redis online store need write access to AWS, which the sprint does not have. The online store is a local SQLite file, empty until Week 2. Moving to S3 and Redis changes `feature_store.yaml` and the source path, not the definitions.
- **Commands**: `make pipeline` fills it as step 3b; `make feature-store [DATASET=path]` fills it from each brand's latest training dataset.

```python
from feature_store import historical, store
rows = pd.DataFrame({"tenant_id": [...], "player_id": [...], "event_timestamp": pd.Timestamp("2026-08-01", tz="UTC")})
features = historical(store(), rows)      # the 39 features of each player as of 2026-08-01
```
