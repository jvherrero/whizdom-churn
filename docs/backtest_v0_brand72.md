# Backtest v0: rolling-origin (brandId 72)

Does the model predict the way it will be used: trained on the past, scored on a date, judged on what the players did next, repeated over a year. Backtest id `backtest_brand72_2026-10-06_1791890715`, data as of 2026-10-06, 11 test cutoffs, 52 min. Config: `configs/backtest.yaml`.

**Verdict: FAIL** (a model that fails is not promoted, whatever its single-split scores).

| Pass condition (delivery plan) | Candidate (`lgbm_optimised`) | Result |
|---|---|---|
| mean top-decile precision >= 1.10x the recency rule | 0.932 vs 0.806: 1.16x [1.12, 1.20] | pass |
| AUC never below 0.7 | min 0.846 (2025-10-01) | pass |
| ECE never above 0.08 | max 0.099 (2025-12-01) | FAIL |

![Backtest](../data/03_output/backtest/backtest_brand72_2026-10-06_1791890715.png)

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
| cox | 0.865 [0.856, 0.876] (range 0.843 to 0.911) | 0.832 [0.819, 0.844] (range 0.807 to 0.879) | 0.937 [0.921, 0.954] (range 0.868 to 0.985) | 0.161 [0.148, 0.174] (range 0.130 to 0.205) | n/a | n/a | n/a |
| incumbent | 0.610 [0.585, 0.634] (range 0.527 to 0.702) | 0.588 [0.565, 0.615] (range 0.525 to 0.692) | 0.577 [0.529, 0.628] (range 0.458 to 0.726) | 0.098 [0.091, 0.107] (range 0.082 to 0.129) | n/a | n/a | n/a |
| lgbm_optimised | 0.870 [0.861, 0.880] (range 0.846 to 0.911) | 0.833 [0.823, 0.845] (range 0.807 to 0.876) | 0.932 [0.922, 0.948] (range 0.900 to 0.981) | 0.160 [0.148, 0.174] (range 0.132 to 0.213) | 0.434 [0.406, 0.450] (range 0.336 to 0.492) | 0.052 [0.038, 0.065] (range 0.011 to 0.099) | 0.139 [0.128, 0.145] (range 0.103 to 0.159) |
| lgbm_reference | 0.871 [0.862, 0.881] (range 0.852 to 0.910) | 0.835 [0.825, 0.845] (range 0.813 to 0.876) | 0.937 [0.926, 0.950] (range 0.899 to 0.976) | 0.161 [0.149, 0.175] (range 0.131 to 0.212) | 0.430 [0.405, 0.446] (range 0.336 to 0.491) | 0.049 [0.035, 0.063] (range 0.012 to 0.100) | 0.138 [0.128, 0.144] (range 0.103 to 0.158) |
| recency | 0.738 [0.718, 0.754] (range 0.678 to 0.772) | 0.703 [0.688, 0.715] (range 0.662 to 0.731) | 0.806 [0.769, 0.837] (range 0.673 to 0.931) | 0.137 [0.127, 0.145] (range 0.109 to 0.159) | n/a | n/a | n/a |

## AUC per Cutoff

| cutoff | n_players | churn_rate | n_train_cutoffs | lgbm_optimised | lgbm_reference | cox | recency | incumbent | ECE lgbm_optimised | ECE without level | level shift |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-10-01 | 26741 | 0.657 | 2 | 0.846 | 0.852 | 0.844 | 0.703 | 0.586 | 0.037 | 0.037 | 0.000 |
| 2025-11-01 | 29996 | 0.712 | 3 | 0.867 | 0.866 | 0.861 | 0.678 | 0.527 | 0.065 | 0.081 | 0.110 |
| 2025-12-01 | 23018 | 0.728 | 4 | 0.852 | 0.853 | 0.843 | 0.760 | 0.617 | 0.099 | 0.079 | -0.146 |
| 2026-01-01 | 19595 | 0.739 | 5 | 0.911 | 0.910 | 0.911 | 0.751 | 0.641 | 0.063 | 0.047 | -0.135 |
| 2026-02-01 | 11613 | 0.527 | 6 | 0.867 | 0.869 | 0.866 | 0.755 | 0.607 | 0.060 | 0.086 | -0.194 |
| 2026-03-01 | 13658 | 0.558 | 6 | 0.865 | 0.869 | 0.859 | 0.737 | 0.595 | 0.073 | 0.102 | -0.235 |
| 2026-04-01 | 13830 | 0.523 | 7 | 0.858 | 0.858 | 0.857 | 0.735 | 0.592 | 0.078 | 0.129 | -0.427 |
| 2026-05-01 | 15748 | 0.583 | 9 | 0.877 | 0.877 | 0.873 | 0.772 | 0.629 | 0.011 | 0.016 | 0.064 |
| 2026-06-01 | 13054 | 0.539 | 10 | 0.893 | 0.894 | 0.888 | 0.766 | 0.597 | 0.023 | 0.040 | 0.178 |
| 2026-07-01 | 11452 | 0.424 | 11 | 0.866 | 0.868 | 0.851 | 0.736 | 0.616 | 0.039 | 0.045 | -0.053 |
| 2026-08-01 | 17248 | 0.563 | 12 | 0.863 | 0.867 | 0.864 | 0.722 | 0.702 | 0.023 | 0.018 | -0.147 |

## Targeting Replay (would it have paid?)

A campaign that contacts the top **1,000** players at every cutoff (N to agree with the CRM team). The candidate ranks by **value at risk** = trailing-90-day mean daily GGR x 30 x its churn probability; `recency` and `incumbent` rank by their own score; `random` is the expected value of a random pick. **Realised GGR at risk** = the 30-day value of the players who did churn. No treatment was applied, so this measures targeting quality, not uplift.

| ranking | churn_captured | ggr_at_risk_captured | ggr_at_risk_per_1000 |
|---|---|---|---|
| incumbent | 0.056 [0.041, 0.073] (range 0.018 to 0.111) | 0.044 [0.029, 0.060] (range 0.008 to 0.092) | 11,078.372 [7,619.032, 15,087.334] (range 2,248.335 to 25,881.265) |
| lgbm_optimised | 0.046 [0.036, 0.058] (range 0.022 to 0.082) | 0.643 [0.601, 0.688] (range 0.556 to 0.777) | 176,528.582 [152,376.203, 202,331.790] (range 114,159.106 to 254,661.750) |
| random | 0.062 [0.050, 0.072] (range 0.033 to 0.087) | 0.062 [0.050, 0.072] (range 0.033 to 0.087) | 16,358.816 [13,743.288, 18,815.412] (range 10,738.576 to 24,245.260) |
| recency | 0.087 [0.067, 0.108] (range 0.037 to 0.140) | 0.102 [0.062, 0.141] (range 0.019 to 0.249) | 25,451.026 [16,836.324, 36,521.428] (range 7,907.895 to 70,248.672) |

## How to Read It

- **Top-decile condition at 1.10x, not the plan's 1.5x** (`docs/decision_top_decile_condition.md`): precision cannot exceed 1 and the recency rule already reaches 0.806, so the most any model could reach is 1.24x, and this ceiling falls in the months with more churn. 1.10x is the strictest round value the model meets with 95% confidence in brands 14, 64 and 73, and at today's precisions it is the plan's own demand (1.5 times the churners found per wasted contact). Interval: 95%, cutoffs resampled 200 times.
- **Calibration lags the churn rate**: the calibration month is the latest one whose 60-day label was known on C, 2 to 3 months before it. When the monthly churn rate moves between them (around 50% from November to January, 30 to 40% the rest of the year), every probability is off by about the same amount: the ECE rises while the AUC does not move. The level shift on C removes most of it. What is left comes from the 60 days after C (a holiday season, a sports calendar), which no data before C shows: in the analysis of 2026-10-09 even the exact level of the month before C would not have kept every month of brand 14 under 0.08 (`eda/07_calibration_seasonality.ipynb`).
- **The replay trades churners for value**: the candidate ranks by value at risk, so it contacts fewer players who churn than recency does, but the ones it contacts carry most of the GGR at risk. Ranking by churn probability alone is the top-decile columns above.
- **The cutoffs are not independent**: the players repeat and the training months overlap, so the CIs are optimistic.
- **The winsorisation caps** come from the main pipeline's configuration (they may have seen later months' features, never labels).
- **Early cutoffs train on few months**, and on no deposit data before March 2026 (the deposit features are empty then). Training cutoffs before 2025-09 have under 150 days of history, so `days_since_first_bet` (capped at 150) is lower there for the same players.

Generated by `src/models/backtest.py` (`make backtest`). Tables: `data/03_output/backtest/backtest_brand72_2026-10-06_1791890715_metrics.csv`, `data/03_output/backtest/backtest_brand72_2026-10-06_1791890715_replay.csv`.
