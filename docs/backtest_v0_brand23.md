# Backtest v0: rolling-origin (brandId 23)

Does the model predict the way it will be used: trained on the past, scored on a date, judged on what the players did next, repeated over a year. Backtest id `backtest_brand23_2026-10-06_1792074259`, data as of 2026-10-06, 11 test cutoffs, 233 min. Config: `configs/backtest.yaml`.

**Verdict: FAIL** (a model that fails is not promoted, whatever its single-split scores).

| Pass condition (delivery plan) | Candidate (`lgbm_optimised`) | Result |
|---|---|---|
| mean top-decile precision >= 1.10x the recency rule | 0.796 vs 0.572: 1.39x [1.32, 1.46] | pass |
| AUC never below 0.7 | min 0.863 (2026-04-01) | pass |
| ECE never above 0.08 | max 0.101 (2025-12-01) | FAIL |

![Backtest](../data/03_output/backtest/backtest_brand23_2026-10-06_1792074259.png)

## How It Runs

- **Test cutoffs**: the first day of each month, 2025-10-01, 2025-11-01, 2025-12-01, 2026-01-01, 2026-02-01, 2026-03-01, 2026-04-01, 2026-05-01, 2026-06-01, 2026-07-01, 2026-08-01. Skipped (fewer than 2 earlier months with a known label): 2025-09-01.
- **Training at each cutoff C**: every earlier monthly cutoff whose 60-day label was already known on C (cutoff + 60 days <= C), labels computed from data up to C only; the latest 1 of them are the validation months (early stopping, tuning, calibration). So the training data grows from 2 to 12 months, and the models never see a label C did not know.
- **Scored**: every player with a bet in the 30 days up to C, judged on the 60 days after it (data up to 2026-10-06).
- **Calibration level on C**, as scoring sets it (`src/models/calibration_level.py`): one log-odds shift per brand, the median of the shifts that set the mean probability right on the last 3 months with a known label and on the later months' early signal (who has bet again by C). It changes no ranking, only the ECE, log-loss and Brier; the ECE without it is in the per-cutoff table.
- **Models**: trained exactly as in normal training, with 20 Optuna trials (not 100) and at most 500 trees, as the plan allows for the backtest. Each training run is in MLflow with the tag `backtest_id`.
- **Comparators**: `lgbm_reference` (optimisation step 1), `cox` (partial hazard), `recency` (days since the last bet alone) and `incumbent` (the platform's `churn_score` in `gld_player_signals_daily`, a weighted heuristic: 0.5 recency + 0.3 frequency + 0.2 value). The plan's 7/14/30-day labels are replaced by the 60-day churn of `docs/churn_definition_v0.md`.

## Results Across Cutoffs

Mean, 95% CI of the mean, median and range. The CI is a circular block bootstrap: each of the 2000 resamples joins blocks of 3 consecutive cutoffs, because neighbouring months move together (season, players who repeat). Log-loss, ECE and Brier only for the calibrated probabilities.

| comparator | auc | c_index | top_decile_precision | top_decile_recall | log_loss | ece | brier |
|---|---|---|---|---|---|---|---|
| cox | 0.877 [0.867, 0.886] (median 0.880, range 0.851 to 0.896) | 0.802 [0.779, 0.829] (median 0.796, range 0.768 to 0.877) | 0.788 [0.739, 0.847] (median 0.761, range 0.663 to 0.943) | 0.335 [0.289, 0.372] (median 0.351, range 0.238 to 0.426) | n/a | n/a | n/a |
| incumbent | 0.640 [0.623, 0.664] (median 0.632, range 0.603 to 0.749) | 0.595 [0.576, 0.626] (median 0.580, range 0.562 to 0.731) | 0.176 [0.131, 0.222] (median 0.134, range 0.121 to 0.334) | 0.075 [0.061, 0.098] (median 0.064, range 0.054 to 0.198) | n/a | n/a | n/a |
| lgbm_optimised | 0.886 [0.877, 0.894] (median 0.892, range 0.863 to 0.902) | 0.802 [0.778, 0.829] (median 0.793, range 0.765 to 0.881) | 0.796 [0.750, 0.853] (median 0.754, range 0.668 to 0.946) | 0.338 [0.291, 0.377] (median 0.356, range 0.238 to 0.431) | 0.358 [0.324, 0.401] (median 0.346, range 0.290 to 0.488) | 0.044 [0.026, 0.061] (median 0.031, range 0.010 to 0.101) | 0.115 [0.103, 0.130] (median 0.111, range 0.091 to 0.158) |
| lgbm_reference | 0.887 [0.878, 0.896] (median 0.895, range 0.863 to 0.904) | 0.803 [0.779, 0.830] (median 0.794, range 0.764 to 0.881) | 0.801 [0.757, 0.857] (median 0.781, range 0.667 to 0.951) | 0.340 [0.292, 0.379] (median 0.360, range 0.240 to 0.430) | 0.356 [0.322, 0.399] (median 0.338, range 0.289 to 0.486) | 0.043 [0.026, 0.060] (median 0.030, range 0.012 to 0.101) | 0.114 [0.102, 0.129] (median 0.107, range 0.091 to 0.158) |
| recency | 0.785 [0.778, 0.792] (median 0.786, range 0.763 to 0.803) | 0.708 [0.690, 0.730] (median 0.699, range 0.677 to 0.776) | 0.572 [0.526, 0.644] (median 0.517, range 0.469 to 0.758) | 0.240 [0.215, 0.260] (median 0.253, range 0.191 to 0.278) | n/a | n/a | n/a |
| recency_prob | 0.785 [0.778, 0.792] (median 0.787, range 0.763 to 0.803) | 0.708 [0.690, 0.730] (median 0.699, range 0.677 to 0.776) | 0.572 [0.526, 0.643] (median 0.517, range 0.469 to 0.758) | 0.240 [0.215, 0.260] (median 0.253, range 0.191 to 0.278) | n/a | n/a | n/a |
| simple_rft | 0.881 [0.874, 0.889] (median 0.885, range 0.860 to 0.901) | 0.801 [0.779, 0.825] (median 0.799, range 0.765 to 0.872) | 0.803 [0.761, 0.861] (median 0.803, range 0.693 to 0.945) | 0.341 [0.294, 0.377] (median 0.354, range 0.238 to 0.425) | n/a | n/a | n/a |

## Month by Month against Recency (review item P1 11)

The cutoffs are not independent, so the interval alone can look too sure. This table does not depend on any resampling: the median and the worst month of each comparator, and the months where it beats the recency rule (AUC above recency; top-decile ratio at or above 1.10x).

| comparator | months | auc_median | auc_worst | auc_worst_month | months_auc_above_recency | ratio_median | ratio_worst | ratio_worst_month | months_ratio_at_threshold |
|---|---|---|---|---|---|---|---|---|---|
| lgbm_optimised | 11 | 0.892 | 0.863 | 2026-04-01 | 11 | 1.421 | 1.247 | 2025-12-01 | 11 |
| lgbm_reference | 11 | 0.895 | 0.863 | 2026-01-01 | 11 | 1.427 | 1.254 | 2025-12-01 | 11 |
| cox | 11 | 0.880 | 0.851 | 2026-04-01 | 11 | 1.395 | 1.244 | 2025-12-01 | 11 |
| incumbent | 11 | 0.632 | 0.603 | 2026-05-01 | 0 | 0.245 | 0.213 | 2026-06-01 | 0 |
| recency_prob | 11 | 0.787 | 0.763 | 2026-05-01 | 5 | 1.000 | 0.989 | 2025-10-01 | 0 |
| simple_rft | 11 | 0.885 | 0.860 | 2025-12-01 | 11 | 1.419 | 1.246 | 2025-12-01 | 11 |

## AUC per Cutoff

| cutoff | n_players | churn_rate | n_train_cutoffs | lgbm_optimised | lgbm_reference | cox | recency | incumbent | recency_prob | simple_rft | ECE lgbm_optimised | ECE without level | level shift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-10-01 | 33646 | 0.312 | 2 | 0.898 | 0.900 | 0.891 | 0.800 | 0.632 | 0.799 | 0.901 | 0.022 | 0.027 | 0.044 |
| 2025-11-01 | 33608 | 0.340 | 3 | 0.896 | 0.895 | 0.887 | 0.785 | 0.612 | 0.784 | 0.897 | 0.053 | 0.050 | -0.026 |
| 2025-12-01 | 31038 | 0.397 | 4 | 0.863 | 0.864 | 0.858 | 0.766 | 0.626 | 0.765 | 0.860 | 0.101 | 0.094 | -0.061 |
| 2026-01-01 | 20054 | 0.247 | 5 | 0.863 | 0.863 | 0.859 | 0.774 | 0.644 | 0.774 | 0.861 | 0.021 | 0.011 | -0.185 |
| 2026-02-01 | 19720 | 0.189 | 6 | 0.888 | 0.888 | 0.870 | 0.786 | 0.631 | 0.787 | 0.880 | 0.074 | 0.105 | -0.253 |
| 2026-03-01 | 22149 | 0.208 | 6 | 0.892 | 0.895 | 0.880 | 0.803 | 0.643 | 0.803 | 0.894 | 0.073 | 0.116 | -0.346 |
| 2026-04-01 | 21103 | 0.200 | 7 | 0.863 | 0.864 | 0.851 | 0.781 | 0.625 | 0.781 | 0.863 | 0.053 | 0.053 | -0.004 |
| 2026-05-01 | 22053 | 0.227 | 9 | 0.889 | 0.890 | 0.879 | 0.763 | 0.603 | 0.763 | 0.880 | 0.017 | 0.023 | 0.089 |
| 2026-06-01 | 21421 | 0.235 | 10 | 0.902 | 0.904 | 0.896 | 0.795 | 0.633 | 0.795 | 0.889 | 0.024 | 0.022 | -0.027 |
| 2026-07-01 | 19785 | 0.187 | 11 | 0.896 | 0.900 | 0.885 | 0.794 | 0.640 | 0.794 | 0.886 | 0.010 | 0.011 | -0.050 |
| 2026-08-01 | 19444 | 0.168 | 12 | 0.895 | 0.896 | 0.890 | 0.792 | 0.749 | 0.792 | 0.885 | 0.031 | 0.035 | -0.040 |

## Targeting Replay (would it have paid?)

A campaign that contacts the top **1,000** players at every cutoff (N to agree with the CRM team). The candidate ranks by **value at risk** = trailing-90-day mean daily GGR x 30 x its churn probability; `recency` and `incumbent` rank by their own score; `random` is the expected value of a random pick. **Realised GGR at risk** = the 30-day value of the players who did churn. No treatment was applied, so this measures targeting quality, not uplift.

| ranking | churn_captured | ggr_at_risk_captured | ggr_at_risk_per_1000 |
|---|---|---|---|
| incumbent | 0.022 [0.010, 0.040] (median 0.013, range 0.007 to 0.107) | 0.012 [0.002, 0.026] (median 0.003, range 0.001 to 0.077) | 5,301.540 [1,485.655, 10,895.702] (median 1,898.781, range 455.426 to 29,712.332) |
| lgbm_optimised | 0.053 [0.041, 0.064] (median 0.055, range 0.027 to 0.088) | 0.611 [0.568, 0.650] (median 0.609, range 0.524 to 0.717) | 336,918.340 [280,094.149, 410,443.703] (median 318,444.225, range 198,737.339 to 639,591.843) |
| random | 0.044 [0.037, 0.048] (median 0.047, range 0.030 to 0.051) | 0.044 [0.037, 0.048] (median 0.047, range 0.030 to 0.051) | 23,633.554 [19,547.131, 28,038.872] (median 22,278.217, range 10,977.258 to 37,202.770) |
| recency | 0.112 [0.084, 0.134] (median 0.121, range 0.058 to 0.151) | 0.100 [0.073, 0.125] (median 0.089, range 0.042 to 0.156) | 53,166.809 [43,526.608, 61,582.186] (median 53,500.066, range 15,362.019 to 79,774.111) |

## Value-Matched Policies (review items P0 4 and P0 5)

Every risk score ranks the same players with the same 1,000 contacts, alone (value-blind) and multiplied by the expected 30-day GGR (value-aware). `recency` is the days since the last bet turned into a probability, `simple_rft` a logistic regression on recency, frequency and tenure (both fitted on the training months of each cutoff), `lgbm_60d` the candidate, and `cox_30d` the `p_churn_30d` that scoring writes, so `cox_30d` value-aware is the `value_at_risk_30d` of `player_scores`. Mean over the cutoffs.

| Score | Churners caught, value-blind | GGR at risk caught, value-blind | Churners caught, value-aware | GGR at risk caught, value-aware | Months value-aware beats `simple_rft` value-aware |
|---|---|---|---|---|---|
| `recency` | 11.2% | 10.1% | 2.6% | 54.0% | 0 of 11 |
| `simple_rft` | 16.8% | 2.9% | 5.2% | 59.5% |  |
| `lgbm_60d` | 16.4% | 2.3% | 5.3% | 61.1% | 9 of 11 |
| `cox_30d` | 16.4% | 2.2% | 3.0% | 55.3% | 1 of 11 |

Split of the GGR at risk caught by the final policy (`lgbm_60d` value-aware) against value-blind recency:

- **Predictive uplift** (`lgbm_60d` against `simple_rft`, both value-aware): +1.6 points.
- **Value-weighting uplift** (`lgbm_60d` value-aware against value-blind): +58.8 points.
- **Combined** (`lgbm_60d` value-aware against `recency` value-blind): +51.0 points.

### Cox 30-day against LightGBM 60-day value at risk

Share of the top 1,000 players in both lists: mean 66% (range 51% to 75%). Spearman correlation of the two values at risk: 0.878; of the two probabilities: 0.972.

## Calibration Level: the Early Churn Estimate (review item P0 3)

On a scoring date the calibration level estimates the 60-day churn rate of the recent cohorts, whose label is not complete yet, from data up to that date only: the share of the cohort with no bet so far, times the share of the last complete cohort that, silent after the same number of days, stayed silent for 60 days (`calibration_level.early_rate`). Here the estimate is replayed for every monthly cohort 14, 21, 30, 45 days after its cutoff, with data up to that day only, and set against the rate known today and against simply using the rate of the last complete cohort. Mean absolute error, in points of churn rate.

| Days observed | Cohorts | Mean realised rate | Error of the early estimate | Error of the last complete month |
|---|---|---|---|---|
| 14 days | 12 | 25.2% | 2.67 | 5.62 |
| 21 days | 12 | 25.2% | 2.35 | 5.62 |
| 30 days | 13 | 25.4% | 1.09 | 3.83 |
| 45 days | 13 | 25.4% | 0.48 | 3.68 |

The early estimate is closer to the realised rate in 44 of 50 cohorts.

## Cox PH per Horizon (review item P1 7)

The calibrated Cox probability of churn starting within h days, against the h-day label, on the players whose label is known. Mean over the cutoffs (range).

| Horizon | Players per cutoff | Churn rate | AUC | Brier | ECE |
|---|---|---|---|---|---|
| 7 days | 24,458 | 0.283 | 0.866 (0.841 to 0.886) | 0.142 | 0.096 (0.058 to 0.184) |
| 14 days | 24,458 | 0.309 | 0.858 (0.836 to 0.878) | 0.148 | 0.094 (0.054 to 0.187) |
| 30 days | 24,458 | 0.359 | 0.848 (0.825 to 0.868) | 0.158 | 0.092 (0.043 to 0.191) |


## How to Read It

- **Top-decile condition at 1.10x, not the plan's 1.5x** (`docs/decision_top_decile_condition.md`): precision cannot exceed 1 and the recency rule already reaches 0.572, so the most any model could reach is 1.75x, and this ceiling falls in the months with more churn. 1.10x is the strictest round value the model meets with 95% confidence in brands 14, 64 and 73, and at today's precisions it is the plan's own demand (1.5 times the churners found per wasted contact). Interval: 95%, circular block bootstrap of 3 consecutive cutoffs, 2000 resamples.
- **Calibration lags the churn rate**: the calibration month is the latest one whose 60-day label was known on C, 2 to 3 months before it. When the monthly churn rate moves between them (around 50% from November to January, 30 to 40% the rest of the year), every probability is off by about the same amount: the ECE rises while the AUC does not move. The level shift on C removes most of it. What is left comes from the 60 days after C (a holiday season, a sports calendar), which no data before C shows: in the analysis of 2026-10-09 even the exact level of the month before C would not have kept every month of brand 14 under 0.08 (`eda/07_calibration_seasonality.ipynb`).
- **The replay trades churners for value**: the candidate ranks by value at risk, so it contacts fewer players who churn than recency does, but the ones it contacts carry most of the GGR at risk. Ranking by churn probability alone is the top-decile columns above.
- **The cutoffs are not independent**: the players repeat and the training months overlap. The CIs resample blocks of consecutive months, which keeps the dependence between neighbouring months, but not players, so they can still be a little optimistic; the month-by-month counts do not depend on any resampling.
- **The winsorisation caps** come from the main pipeline's configuration (they may have seen later months' features, never labels).
- **Early cutoffs train on few months**, and on no deposit data before March 2026 (the deposit features are empty then). Training cutoffs before 2025-09 have under 150 days of history, so `days_since_first_bet` (capped at 150) is lower there for the same players.

Generated by `src/models/backtest.py` (`make backtest`). Tables: `data/03_output/backtest/backtest_brand23_2026-10-06_1792074259_metrics.csv`, `data/03_output/backtest/backtest_brand23_2026-10-06_1792074259_replay.csv`.
