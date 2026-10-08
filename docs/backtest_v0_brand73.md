# Backtest v0: rolling-origin (brandId 73)

Does the model predict the way it will be used: trained on the past, scored on a date, judged on what the players did next, repeated over a year. Backtest id `backtest_brand73_2026-10-06_1791438871`, data as of 2026-10-06, 11 test cutoffs, 96 min. Config: `configs/backtest.yaml`.

**Verdict: FAIL** (a model that fails is not promoted, whatever its single-split scores).

| Pass condition (delivery plan) | Candidate (`lgbm_optimised`) | Result |
|---|---|---|
| mean top-decile precision >= 1.5x the recency rule | 0.919 vs 0.740 (1.24x) | FAIL |
| AUC never below 0.7 | min 0.863 (2026-01-01) | pass |
| ECE never above 0.08 | max 0.054 (2026-03-01) | pass |

![Backtest](../data/03_output/backtest/backtest_brand73_2026-10-06_1791438871.png)

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
| cox | 0.881 [0.875, 0.889] (range 0.861 to 0.897) | 0.827 [0.816, 0.844] (range 0.801 to 0.875) | 0.920 [0.913, 0.926] (range 0.894 to 0.932) | 0.216 [0.206, 0.226] (range 0.187 to 0.242) | n/a | n/a | n/a |
| incumbent | 0.636 [0.615, 0.662] (range 0.590 to 0.745) | 0.600 [0.579, 0.629] (range 0.565 to 0.727) | 0.422 [0.380, 0.479] (range 0.311 to 0.624) | 0.100 [0.087, 0.116] (range 0.079 to 0.162) | n/a | n/a | n/a |
| lgbm_optimised | 0.887 [0.880, 0.894] (range 0.863 to 0.904) | 0.828 [0.816, 0.846] (range 0.798 to 0.879) | 0.919 [0.913, 0.925] (range 0.896 to 0.931) | 0.216 [0.206, 0.226] (range 0.187 to 0.241) | 0.419 [0.405, 0.432] (range 0.386 to 0.459) | 0.033 [0.023, 0.040] (range 0.012 to 0.054) | 0.136 [0.130, 0.141] (range 0.123 to 0.151) |
| lgbm_reference | 0.888 [0.881, 0.896] (range 0.863 to 0.906) | 0.828 [0.817, 0.847] (range 0.797 to 0.882) | 0.921 [0.915, 0.927] (range 0.897 to 0.938) | 0.217 [0.206, 0.227] (range 0.187 to 0.244) | 0.418 [0.402, 0.432] (range 0.384 to 0.459) | 0.032 [0.023, 0.039] (range 0.009 to 0.051) | 0.135 [0.130, 0.141] (range 0.122 to 0.151) |
| recency | 0.774 [0.763, 0.782] (range 0.733 to 0.793) | 0.717 [0.704, 0.730] (range 0.679 to 0.765) | 0.740 [0.710, 0.770] (range 0.664 to 0.826) | 0.174 [0.166, 0.182] (range 0.153 to 0.200) | n/a | n/a | n/a |

## AUC per Cutoff

| cutoff | n_players | churn_rate | n_train_cutoffs | lgbm_optimised | lgbm_reference | cox | recency | incumbent | ECE lgbm_optimised |
|---|---|---|---|---|---|---|---|---|---|
| 2025-10-01 | 41016 | 0.467 | 2 | 0.878 | 0.878 | 0.868 | 0.776 | 0.637 | 0.048 |
| 2025-11-01 | 40036 | 0.469 | 3 | 0.888 | 0.889 | 0.885 | 0.783 | 0.628 | 0.012 |
| 2025-12-01 | 34378 | 0.491 | 4 | 0.868 | 0.868 | 0.861 | 0.793 | 0.660 | 0.043 |
| 2026-01-01 | 27604 | 0.440 | 5 | 0.863 | 0.863 | 0.861 | 0.733 | 0.596 | 0.029 |
| 2026-02-01 | 25439 | 0.393 | 6 | 0.889 | 0.892 | 0.885 | 0.761 | 0.605 | 0.045 |
| 2026-03-01 | 30050 | 0.461 | 6 | 0.898 | 0.898 | 0.897 | 0.756 | 0.590 | 0.054 |
| 2026-04-01 | 25309 | 0.393 | 7 | 0.886 | 0.886 | 0.877 | 0.785 | 0.631 | 0.047 |
| 2026-05-01 | 26425 | 0.416 | 9 | 0.888 | 0.890 | 0.884 | 0.773 | 0.624 | 0.023 |
| 2026-06-01 | 25110 | 0.404 | 10 | 0.900 | 0.900 | 0.897 | 0.778 | 0.620 | 0.013 |
| 2026-07-01 | 25925 | 0.390 | 11 | 0.904 | 0.906 | 0.895 | 0.783 | 0.655 | 0.034 |
| 2026-08-01 | 25616 | 0.385 | 12 | 0.894 | 0.898 | 0.886 | 0.788 | 0.745 | 0.012 |

## Targeting Replay (would it have paid?)

A campaign that contacts the top **1,000** players at every cutoff (N to agree with the CRM team). The candidate ranks by **value at risk** = trailing-90-day mean daily GGR x 30 x its churn probability; `recency` and `incumbent` rank by their own score; `random` is the expected value of a random pick. **Realised GGR at risk** = the 30-day value of the players who did churn. No treatment was applied, so this measures targeting quality, not uplift.

| ranking | churn_captured | ggr_at_risk_captured | ggr_at_risk_per_1000 |
|---|---|---|---|
| incumbent | 0.020 [0.010, 0.034] (range 0.005 to 0.065) | 0.019 [0.009, 0.032] (range 0.001 to 0.055) | 11,566.815 [5,608.469, 19,185.964] (range 482.433 to 30,229.610) |
| lgbm_optimised | 0.028 [0.024, 0.031] (range 0.020 to 0.033) | 0.602 [0.572, 0.632] (range 0.486 to 0.673) | 419,652.248 [373,014.492, 468,978.067] (range 285,882.136 to 613,819.728) |
| random | 0.035 [0.031, 0.038] (range 0.024 to 0.040) | 0.035 [0.031, 0.038] (range 0.024 to 0.040) | 23,882.245 [21,464.340, 26,550.421] (range 18,095.165 to 33,543.413) |
| recency | 0.062 [0.054, 0.070] (range 0.040 to 0.082) | 0.060 [0.046, 0.081] (range 0.028 to 0.143) | 39,882.681 [31,221.764, 51,760.377] (range 21,087.335 to 84,553.517) |

## How to Read It

- **Top-decile precision condition**: precision cannot exceed 1, so 1.5x the recency rule is only reachable if the recency rule's precision is under 0.67. Here the recency rule reaches 0.740, so the most any model could reach is 1.35x. With a 60-day churn rate around 43%, recency alone already finds the players who left; the comparison of top-decile *recall* and of the targeting replay says more.
- **Calibration lags the churn rate**: the calibration month is the latest one whose 60-day label was known on C, 2 to 3 months before it. When the monthly churn rate moves between them (around 50% from November to January, 30 to 40% the rest of the year), the probabilities are off by the same amount, so the ECE rises in those months while the AUC does not move. Calibrating on 3 validation months instead of 1 (`valid_cutoffs: 3`, tried on 2026-10-07) lowered the worst month from 0.104 to 0.086 but did not remove it, so 1 month is kept; the delayed-label monitoring of Week 2 is what catches it.
- **The replay trades churners for value**: the candidate ranks by value at risk, so it contacts fewer players who churn than recency does, but the ones it contacts carry most of the GGR at risk. Ranking by churn probability alone is the top-decile columns above.
- **The cutoffs are not independent**: the players repeat and the training months overlap, so the CIs are optimistic.
- **The winsorisation caps** come from the main pipeline's configuration (they may have seen later months' features, never labels).
- **Early cutoffs train on few months**, and on no deposit data before March 2026 (the deposit features are empty then). Training cutoffs before 2025-09 have under 150 days of history, so `days_since_first_bet` (capped at 150) is lower there for the same players.

Generated by `src/models/backtest.py` (`make backtest`). Tables: `data/03_output/backtest/backtest_brand73_2026-10-06_1791438871_metrics.csv`, `data/03_output/backtest/backtest_brand73_2026-10-06_1791438871_replay.csv`.
