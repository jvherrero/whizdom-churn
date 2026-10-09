# Backtest v0: rolling-origin (brandId 73)

Does the model predict the way it will be used: trained on the past, scored on a date, judged on what the players did next, repeated over a year. Backtest id `backtest_brand73_2026-10-06_1791547886`, data as of 2026-10-06, 11 test cutoffs, 113 min. Config: `configs/backtest.yaml`.

**Verdict: PASS** (a model that fails is not promoted, whatever its single-split scores).

| Pass condition (delivery plan) | Candidate (`lgbm_optimised`) | Result |
|---|---|---|
| mean top-decile precision >= 1.10x the recency rule | 0.918 vs 0.740: 1.24x [1.19, 1.29] | pass |
| AUC never below 0.7 | min 0.863 (2026-01-01) | pass |
| ECE never above 0.08 | max 0.044 (2025-12-01) | pass |

![Backtest](../data/03_output/backtest/backtest_brand73_2026-10-06_1791547886.png)

## How It Runs

- **Test cutoffs**: the first day of each month, 2025-10-01, 2025-11-01, 2025-12-01, 2026-01-01, 2026-02-01, 2026-03-01, 2026-04-01, 2026-05-01, 2026-06-01, 2026-07-01, 2026-08-01. Skipped (fewer than 2 earlier months with a known label): 2025-09-01.
- **Training at each cutoff C**: every earlier monthly cutoff whose 60-day label was already known on C (cutoff + 60 days <= C), labels computed from data up to C only; the latest 1 of them are the validation months (early stopping, tuning, calibration). So the training data grows from 2 to 12 months, and the models never see a label C did not know.
- **Scored**: every player with a bet in the 30 days up to C, judged on the 60 days after it (data up to 2026-10-06).
- **Calibration level on C**, as scoring sets it (`src/models/calibration_level.py`): one log-odds shift per brand, the median of the shifts that set the mean probability right on the last 3 months with a known label and on the later months' early signal (who has bet again by C). It changes no ranking, only the ECE, log-loss and Brier; the ECE without it is in the per-cutoff table.
- **Models**: trained exactly as in normal training, with 20 Optuna trials (not 100) and at most 500 trees, as the plan allows for the backtest. Each training run is in MLflow with the tag `backtest_id`.
- **Comparators**: `lgbm_reference` (optimisation step 1), `cox` (partial hazard), `recency` (days since the last bet alone) and `incumbent` (the platform's `churn_score` in `gld_player_signals_daily`, a weighted heuristic: 0.5 recency + 0.3 frequency + 0.2 value). The plan's 7/14/30-day labels are replaced by the 60-day churn of `docs/churn_definition_v0.md`.

## Results Across Cutoffs

Mean, 95% bootstrap CI of the mean (cutoffs resampled, 200 times) and range. Log-loss, ECE and Brier only for the calibrated probabilities.

| comparator | auc | c_index | top_decile_precision | top_decile_recall | log_loss | ece | brier |
|---|---|---|---|---|---|---|---|
| cox | 0.880 [0.873, 0.889] (range 0.857 to 0.897) | 0.827 [0.816, 0.844] (range 0.801 to 0.875) | 0.916 [0.905, 0.924] (range 0.885 to 0.932) | 0.216 [0.205, 0.226] (range 0.186 to 0.241) | n/a | n/a | n/a |
| incumbent | 0.636 [0.615, 0.662] (range 0.590 to 0.745) | 0.600 [0.579, 0.629] (range 0.565 to 0.727) | 0.419 [0.376, 0.476] (range 0.308 to 0.616) | 0.099 [0.086, 0.116] (range 0.078 to 0.160) | n/a | n/a | n/a |
| lgbm_optimised | 0.887 [0.880, 0.894] (range 0.863 to 0.905) | 0.827 [0.816, 0.846] (range 0.797 to 0.879) | 0.918 [0.909, 0.924] (range 0.896 to 0.932) | 0.216 [0.205, 0.226] (range 0.187 to 0.242) | 0.418 [0.403, 0.432] (range 0.383 to 0.462) | 0.027 [0.021, 0.032] (range 0.009 to 0.044) | 0.136 [0.130, 0.141] (range 0.122 to 0.151) |
| lgbm_reference | 0.888 [0.881, 0.896] (range 0.864 to 0.908) | 0.828 [0.816, 0.847] (range 0.798 to 0.881) | 0.918 [0.908, 0.926] (range 0.893 to 0.933) | 0.216 [0.205, 0.227] (range 0.187 to 0.242) | 0.416 [0.400, 0.431] (range 0.378 to 0.462) | 0.027 [0.021, 0.032] (range 0.007 to 0.045) | 0.135 [0.129, 0.140] (range 0.120 to 0.151) |
| recency | 0.774 [0.763, 0.782] (range 0.733 to 0.793) | 0.717 [0.704, 0.730] (range 0.679 to 0.765) | 0.740 [0.710, 0.770] (range 0.666 to 0.826) | 0.174 [0.166, 0.182] (range 0.153 to 0.200) | n/a | n/a | n/a |

## AUC per Cutoff

| cutoff | n_players | churn_rate | n_train_cutoffs | lgbm_optimised | lgbm_reference | cox | recency | incumbent | ECE lgbm_optimised | ECE without level | level shift |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-10-01 | 41016 | 0.467 | 2 | 0.878 | 0.878 | 0.868 | 0.776 | 0.637 | 0.035 | 0.049 | 0.098 |
| 2025-11-01 | 40036 | 0.469 | 3 | 0.888 | 0.889 | 0.885 | 0.783 | 0.628 | 0.025 | 0.013 | -0.097 |
| 2025-12-01 | 34378 | 0.491 | 4 | 0.866 | 0.867 | 0.857 | 0.793 | 0.660 | 0.044 | 0.042 | -0.015 |
| 2026-01-01 | 27604 | 0.440 | 5 | 0.863 | 0.864 | 0.861 | 0.733 | 0.596 | 0.030 | 0.027 | -0.032 |
| 2026-02-01 | 25439 | 0.393 | 6 | 0.889 | 0.891 | 0.885 | 0.761 | 0.605 | 0.031 | 0.042 | -0.085 |
| 2026-03-01 | 30050 | 0.461 | 6 | 0.901 | 0.902 | 0.897 | 0.756 | 0.590 | 0.033 | 0.051 | -0.165 |
| 2026-04-01 | 25309 | 0.393 | 7 | 0.886 | 0.887 | 0.877 | 0.785 | 0.631 | 0.034 | 0.047 | -0.092 |
| 2026-05-01 | 26425 | 0.416 | 9 | 0.883 | 0.883 | 0.873 | 0.773 | 0.624 | 0.019 | 0.019 | 0.025 |
| 2026-06-01 | 25110 | 0.404 | 10 | 0.900 | 0.900 | 0.897 | 0.778 | 0.620 | 0.014 | 0.013 | 0.035 |
| 2026-07-01 | 25925 | 0.390 | 11 | 0.905 | 0.908 | 0.897 | 0.783 | 0.655 | 0.025 | 0.033 | -0.107 |
| 2026-08-01 | 25616 | 0.385 | 12 | 0.894 | 0.896 | 0.886 | 0.788 | 0.745 | 0.009 | 0.011 | 0.033 |

## Targeting Replay (would it have paid?)

A campaign that contacts the top **1,000** players at every cutoff (N to agree with the CRM team). The candidate ranks by **value at risk** = trailing-90-day mean daily GGR x 30 x its churn probability; `recency` and `incumbent` rank by their own score; `random` is the expected value of a random pick. **Realised GGR at risk** = the 30-day value of the players who did churn. No treatment was applied, so this measures targeting quality, not uplift.

| ranking | churn_captured | ggr_at_risk_captured | ggr_at_risk_per_1000 |
|---|---|---|---|
| incumbent | 0.020 [0.010, 0.035] (range 0.005 to 0.069) | 0.021 [0.009, 0.037] (range 0.001 to 0.070) | 12,596.631 [5,853.357, 21,560.086] (range 930.783 to 37,164.788) |
| lgbm_optimised | 0.028 [0.025, 0.031] (range 0.020 to 0.033) | 0.603 [0.572, 0.633] (range 0.480 to 0.676) | 419,922.941 [373,363.605, 470,109.169] (range 285,818.115 to 614,790.453) |
| random | 0.035 [0.031, 0.038] (range 0.024 to 0.040) | 0.035 [0.031, 0.038] (range 0.024 to 0.040) | 23,882.245 [21,464.340, 26,550.421] (range 18,095.165 to 33,543.413) |
| recency | 0.062 [0.054, 0.070] (range 0.039 to 0.082) | 0.056 [0.047, 0.066] (range 0.028 to 0.082) | 37,740.883 [32,129.177, 44,747.887] (range 20,566.772 to 56,116.648) |

## How to Read It

- **Top-decile condition at 1.10x, not the plan's 1.5x** (`docs/decision_top_decile_condition.md`): precision cannot exceed 1 and the recency rule already reaches 0.740, so the most any model could reach is 1.35x, and this ceiling falls in the months with more churn. 1.10x is the strictest round value the model meets with 95% confidence in brands 14, 64 and 73, and at today's precisions it is the plan's own demand (1.5 times the churners found per wasted contact). Interval: 95%, cutoffs resampled 200 times.
- **Calibration lags the churn rate**: the calibration month is the latest one whose 60-day label was known on C, 2 to 3 months before it. When the monthly churn rate moves between them (around 50% from November to January, 30 to 40% the rest of the year), every probability is off by about the same amount: the ECE rises while the AUC does not move. The level shift on C removes most of it. What is left comes from the 60 days after C (a holiday season, a sports calendar), which no data before C shows: in the analysis of 2026-10-09 even the exact level of the month before C would not have kept every month of brand 14 under 0.08 (`eda/07_calibration_seasonality.ipynb`).
- **The replay trades churners for value**: the candidate ranks by value at risk, so it contacts fewer players who churn than recency does, but the ones it contacts carry most of the GGR at risk. Ranking by churn probability alone is the top-decile columns above.
- **The cutoffs are not independent**: the players repeat and the training months overlap, so the CIs are optimistic.
- **The winsorisation caps** come from the main pipeline's configuration (they may have seen later months' features, never labels).
- **Early cutoffs train on few months**, and on no deposit data before March 2026 (the deposit features are empty then). Training cutoffs before 2025-09 have under 150 days of history, so `days_since_first_bet` (capped at 150) is lower there for the same players.

Generated by `src/models/backtest.py` (`make backtest`). Tables: `data/03_output/backtest/backtest_brand73_2026-10-06_1791547886_metrics.csv`, `data/03_output/backtest/backtest_brand73_2026-10-06_1791547886_replay.csv`.
