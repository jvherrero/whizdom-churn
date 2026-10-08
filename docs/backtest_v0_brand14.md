# Backtest v0: rolling-origin (brandId 14)

Does the model predict the way it will be used: trained on the past, scored on a date, judged on what the players did next, repeated over a year. Backtest id `backtest_brand14_2026-10-06_1791436485`, data as of 2026-10-06, 11 test cutoffs, 34 min. Config: `configs/backtest.yaml`.

**Verdict: FAIL** (a model that fails is not promoted, whatever its single-split scores).

| Pass condition (delivery plan) | Candidate (`lgbm_optimised`) | Result |
|---|---|---|
| mean top-decile precision >= 1.5x the recency rule | 0.880 vs 0.738 (1.19x) | FAIL |
| AUC never below 0.7 | min 0.821 (2026-03-01) | pass |
| ECE never above 0.08 | max 0.132 (2026-05-01) | FAIL |

![Backtest](../data/03_output/backtest/backtest_brand14_2026-10-06_1791436485.png)

## How It Runs

- **Test cutoffs**: the first day of each month, 2025-10-01, 2025-11-01, 2025-12-01, 2026-01-01, 2026-02-01, 2026-03-01, 2026-04-01, 2026-05-01, 2026-06-01, 2026-07-01, 2026-08-01. Skipped (fewer than 2 earlier months with a known label): 2025-09-01.
- **Training at each cutoff C**: every earlier monthly cutoff whose 60-day label was already known on C (cutoff + 60 days <= C), labels computed from data up to C only; the latest 1 of them are the validation months (early stopping, tuning, calibration). So the training data grows from 2 to 12 months, and the models never see a label C did not know.
- **Scored**: every player with a bet in the 30 days up to C, judged on the 60 days after it (data up to 2026-10-06).
- **Models**: trained exactly as in normal training, with 20 Optuna trials (not 100) and at most 500 trees, as the plan allows for the backtest. Each training run is in MLflow with the tag `backtest_id`.
- **Comparators**: `lgbm_reference` (optimisation step 1), `cox` (partial hazard), `recency` (days since the last bet alone) and `incumbent` (the platform's `churn_score` in `gld_player_signals_daily`, a weighted heuristic: 0.5 recency + 0.3 frequency + 0.2 value). The plan's 7/14/30-day labels are replaced by the 60-day churn of `docs/churn_definition_v0.md`.

## Results Across Cutoffs

Mean, 95% bootstrap CI of the mean (cutoffs resampled, 200 times) and range. Log-loss, ECE and Brier only for the calibrated probabilities.

| comparator | auc | c_index | top_decile_precision | top_decile_recall | log_loss | ece | brier |
|---|---|---|---|---|---|---|---|
| cox | 0.864 [0.845, 0.883] (range 0.815 to 0.917) | 0.839 [0.818, 0.860] (range 0.792 to 0.911) | 0.868 [0.834, 0.899] (range 0.769 to 0.956) | 0.156 [0.144, 0.169] (range 0.133 to 0.192) | n/a | n/a | n/a |
| incumbent | 0.612 [0.574, 0.645] (range 0.427 to 0.671) | 0.603 [0.570, 0.627] (range 0.450 to 0.654) | 0.540 [0.476, 0.599] (range 0.403 to 0.697) | 0.096 [0.089, 0.105] (range 0.078 to 0.124) | n/a | n/a | n/a |
| lgbm_optimised | 0.871 [0.854, 0.889] (range 0.821 to 0.919) | 0.844 [0.825, 0.863] (range 0.797 to 0.913) | 0.880 [0.847, 0.908] (range 0.780 to 0.961) | 0.158 [0.147, 0.170] (range 0.136 to 0.197) | 0.429 [0.402, 0.457] (range 0.349 to 0.536) | 0.059 [0.040, 0.076] (range 0.017 to 0.132) | 0.136 [0.125, 0.146] (range 0.107 to 0.176) |
| lgbm_reference | 0.870 [0.854, 0.887] (range 0.823 to 0.918) | 0.844 [0.825, 0.862] (range 0.800 to 0.913) | 0.875 [0.842, 0.903] (range 0.779 to 0.954) | 0.157 [0.147, 0.168] (range 0.135 to 0.196) | 0.431 [0.404, 0.457] (range 0.349 to 0.532) | 0.059 [0.040, 0.077] (range 0.018 to 0.132) | 0.136 [0.126, 0.147] (range 0.107 to 0.174) |
| recency | 0.743 [0.718, 0.772] (range 0.669 to 0.806) | 0.724 [0.702, 0.745] (range 0.657 to 0.782) | 0.738 [0.697, 0.777] (range 0.601 to 0.844) | 0.133 [0.124, 0.143] (range 0.106 to 0.164) | n/a | n/a | n/a |

## AUC per Cutoff

| cutoff | n_players | churn_rate | n_train_cutoffs | lgbm_optimised | lgbm_reference | cox | recency | incumbent | ECE lgbm_optimised |
|---|---|---|---|---|---|---|---|---|---|
| 2025-10-01 | 14849 | 0.591 | 2 | 0.893 | 0.893 | 0.870 | 0.669 | 0.427 | 0.066 |
| 2025-11-01 | 15810 | 0.580 | 3 | 0.840 | 0.839 | 0.824 | 0.753 | 0.620 | 0.018 |
| 2025-12-01 | 17699 | 0.653 | 4 | 0.842 | 0.840 | 0.843 | 0.775 | 0.651 | 0.058 |
| 2026-01-01 | 10564 | 0.497 | 5 | 0.845 | 0.845 | 0.839 | 0.718 | 0.594 | 0.039 |
| 2026-02-01 | 8525 | 0.461 | 6 | 0.874 | 0.872 | 0.876 | 0.806 | 0.671 | 0.054 |
| 2026-03-01 | 11293 | 0.505 | 6 | 0.821 | 0.823 | 0.815 | 0.731 | 0.628 | 0.101 |
| 2026-04-01 | 13727 | 0.628 | 7 | 0.864 | 0.864 | 0.842 | 0.680 | 0.587 | 0.071 |
| 2026-05-01 | 14753 | 0.685 | 9 | 0.898 | 0.896 | 0.900 | 0.805 | 0.670 | 0.132 |
| 2026-06-01 | 10857 | 0.573 | 10 | 0.890 | 0.886 | 0.889 | 0.756 | 0.632 | 0.045 |
| 2026-07-01 | 8360 | 0.456 | 11 | 0.894 | 0.894 | 0.890 | 0.775 | 0.620 | 0.052 |
| 2026-08-01 | 10535 | 0.563 | 12 | 0.919 | 0.918 | 0.917 | 0.711 | 0.638 | 0.017 |

## Targeting Replay (would it have paid?)

A campaign that contacts the top **1,000** players at every cutoff (N to agree with the CRM team). The candidate ranks by **value at risk** = trailing-90-day mean daily GGR x 30 x its churn probability; `recency` and `incumbent` rank by their own score; `random` is the expected value of a random pick. **Realised GGR at risk** = the 30-day value of the players who did churn. No treatment was applied, so this measures targeting quality, not uplift.

| ranking | churn_captured | ggr_at_risk_captured | ggr_at_risk_per_1000 |
|---|---|---|---|
| incumbent | 0.081 [0.063, 0.101] (range 0.044 to 0.132) | 0.102 [0.069, 0.136] (range 0.039 to 0.248) | 26,286.420 [17,104.503, 35,638.633] (range 6,203.256 to 61,547.986) |
| lgbm_optimised | 0.035 [0.026, 0.047] (range 0.012 to 0.072) | 0.691 [0.624, 0.744] (range 0.497 to 0.816) | 178,985.549 [141,685.455, 207,328.446] (range 70,614.902 to 273,166.749) |
| random | 0.085 [0.074, 0.097] (range 0.057 to 0.120) | 0.085 [0.074, 0.097] (range 0.057 to 0.120) | 21,582.550 [17,142.777, 25,449.421] (range 8,253.770 to 31,788.822) |
| recency | 0.114 [0.091, 0.141] (range 0.073 to 0.193) | 0.130 [0.086, 0.179] (range 0.044 to 0.258) | 34,145.221 [22,283.530, 51,335.910] (range 5,797.606 to 86,424.296) |

## How to Read It

- **Top-decile precision condition**: precision cannot exceed 1, so 1.5x the recency rule is only reachable if the recency rule's precision is under 0.67. Here the recency rule reaches 0.738, so the most any model could reach is 1.35x. With a 60-day churn rate around 56%, recency alone already finds the players who left; the comparison of top-decile *recall* and of the targeting replay says more.
- **Calibration lags the churn rate**: the calibration month is the latest one whose 60-day label was known on C, 2 to 3 months before it. When the monthly churn rate moves between them (around 50% from November to January, 30 to 40% the rest of the year), the probabilities are off by the same amount, so the ECE rises in those months while the AUC does not move. Calibrating on 3 validation months instead of 1 (`valid_cutoffs: 3`, tried on 2026-10-07) lowered the worst month from 0.104 to 0.086 but did not remove it, so 1 month is kept; the delayed-label monitoring of Week 2 is what catches it.
- **The replay trades churners for value**: the candidate ranks by value at risk, so it contacts fewer players who churn than recency does, but the ones it contacts carry most of the GGR at risk. Ranking by churn probability alone is the top-decile columns above.
- **The cutoffs are not independent**: the players repeat and the training months overlap, so the CIs are optimistic.
- **The winsorisation caps** come from the main pipeline's configuration (they may have seen later months' features, never labels).
- **Early cutoffs train on few months**, and on no deposit data before March 2026 (the deposit features are empty then). Training cutoffs before 2025-09 have under 150 days of history, so `days_since_first_bet` (capped at 150) is lower there for the same players.

Generated by `src/models/backtest.py` (`make backtest`). Tables: `data/03_output/backtest/backtest_brand14_2026-10-06_1791436485_metrics.csv`, `data/03_output/backtest/backtest_brand14_2026-10-06_1791436485_replay.csv`.
