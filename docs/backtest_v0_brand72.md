# Backtest v0: rolling-origin (brandId 72)

Does the model predict the way it will be used: trained on the past, scored on a date, judged on what the players did next, repeated over a year. Backtest id `backtest_brand72_2026-10-06_1792078504`, data as of 2026-10-06, 11 test cutoffs, 140 min. Config: `configs/backtest.yaml`.

**Verdict: FAIL** (a model that fails is not promoted, whatever its single-split scores).

| Pass condition (delivery plan) | Candidate (`lgbm_optimised`) | Result |
|---|---|---|
| mean top-decile precision >= 1.10x the recency rule | 0.935 vs 0.784: 1.19x [1.15, 1.23] | pass |
| AUC never below 0.7 | min 0.857 (2025-10-01) | pass |
| ECE never above 0.08 | max 0.102 (2025-12-01) | FAIL |

![Backtest](../data/03_output/backtest/backtest_brand72_2026-10-06_1792078504.png)

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
| cox | 0.869 [0.860, 0.878] (median 0.860, range 0.853 to 0.892) | 0.824 [0.811, 0.840] (median 0.820, range 0.796 to 0.859) | 0.933 [0.919, 0.948] (median 0.932, range 0.893 to 0.972) | 0.185 [0.166, 0.200] (median 0.189, range 0.147 to 0.217) | n/a | n/a | n/a |
| incumbent | 0.630 [0.617, 0.647] (median 0.625, range 0.606 to 0.714) | 0.599 [0.583, 0.622] (median 0.596, range 0.562 to 0.700) | 0.468 [0.393, 0.537] (median 0.475, range 0.296 to 0.660) | 0.092 [0.080, 0.108] (median 0.084, range 0.070 to 0.144) | n/a | n/a | n/a |
| lgbm_optimised | 0.876 [0.868, 0.884] (median 0.873, range 0.857 to 0.897) | 0.825 [0.813, 0.840] (median 0.824, range 0.799 to 0.861) | 0.935 [0.922, 0.950] (median 0.930, range 0.904 to 0.970) | 0.186 [0.167, 0.200] (median 0.188, range 0.148 to 0.217) | 0.441 [0.426, 0.454] (median 0.448, range 0.404 to 0.468) | 0.042 [0.024, 0.060] (median 0.025, range 0.011 to 0.102) | 0.143 [0.138, 0.148] (median 0.145, range 0.129 to 0.153) |
| lgbm_reference | 0.877 [0.868, 0.885] (median 0.875, range 0.857 to 0.897) | 0.826 [0.814, 0.841] (median 0.822, range 0.801 to 0.864) | 0.933 [0.918, 0.950] (median 0.931, range 0.893 to 0.974) | 0.185 [0.167, 0.200] (median 0.189, range 0.148 to 0.215) | 0.439 [0.424, 0.452] (median 0.446, range 0.403 to 0.471) | 0.041 [0.023, 0.059] (median 0.027, range 0.012 to 0.102) | 0.142 [0.137, 0.147] (median 0.145, range 0.129 to 0.153) |
| recency | 0.760 [0.752, 0.768] (median 0.762, range 0.726 to 0.789) | 0.711 [0.702, 0.720] (median 0.710, range 0.685 to 0.739) | 0.784 [0.754, 0.820] (median 0.785, range 0.699 to 0.872) | 0.155 [0.144, 0.164] (median 0.158, range 0.133 to 0.174) | n/a | n/a | n/a |
| recency_prob | 0.760 [0.752, 0.768] (median 0.762, range 0.726 to 0.789) | 0.711 [0.702, 0.720] (median 0.710, range 0.685 to 0.739) | 0.782 [0.753, 0.819] (median 0.783, range 0.713 to 0.872) | 0.155 [0.144, 0.163] (median 0.158, range 0.133 to 0.172) | n/a | n/a | n/a |
| simple_rft | 0.870 [0.864, 0.877] (median 0.864, range 0.858 to 0.889) | 0.822 [0.810, 0.835] (median 0.823, range 0.793 to 0.852) | 0.928 [0.915, 0.944] (median 0.928, range 0.895 to 0.969) | 0.185 [0.166, 0.199] (median 0.187, range 0.148 to 0.216) | n/a | n/a | n/a |

## Month by Month against Recency (review item P1 11)

The cutoffs are not independent, so the interval alone can look too sure. This table does not depend on any resampling: the median and the worst month of each comparator, and the months where it beats the recency rule (AUC above recency; top-decile ratio at or above 1.10x).

| comparator | months | auc_median | auc_worst | auc_worst_month | months_auc_above_recency | ratio_median | ratio_worst | ratio_worst_month | months_ratio_at_threshold |
|---|---|---|---|---|---|---|---|---|---|
| lgbm_optimised | 11 | 0.873 | 0.857 | 2025-10-01 | 11 | 1.183 | 1.112 | 2025-12-01 | 11 |
| lgbm_reference | 11 | 0.875 | 0.857 | 2025-10-01 | 11 | 1.182 | 1.108 | 2025-10-01 | 11 |
| cox | 11 | 0.860 | 0.853 | 2025-10-01 | 11 | 1.184 | 1.108 | 2025-12-01 | 11 |
| incumbent | 11 | 0.625 | 0.606 | 2026-04-01 | 0 | 0.606 | 0.407 | 2026-04-01 | 0 |
| recency_prob | 11 | 0.762 | 0.726 | 2026-08-01 | 4 | 0.999 | 0.983 | 2026-02-01 | 0 |
| simple_rft | 11 | 0.864 | 0.858 | 2025-10-01 | 11 | 1.181 | 1.108 | 2025-11-01 | 11 |

## AUC per Cutoff

| cutoff | n_players | churn_rate | n_train_cutoffs | lgbm_optimised | lgbm_reference | cox | recency | incumbent | recency_prob | simple_rft | ECE lgbm_optimised | ECE without level | level shift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-10-01 | 15458 | 0.552 | 2 | 0.857 | 0.857 | 0.853 | 0.752 | 0.613 | 0.751 | 0.858 | 0.025 | 0.023 | -0.012 |
| 2025-11-01 | 16183 | 0.608 | 3 | 0.870 | 0.869 | 0.859 | 0.763 | 0.618 | 0.763 | 0.867 | 0.073 | 0.068 | -0.037 |
| 2025-12-01 | 14709 | 0.656 | 4 | 0.861 | 0.859 | 0.855 | 0.762 | 0.626 | 0.762 | 0.860 | 0.102 | 0.091 | -0.076 |
| 2026-01-01 | 9381 | 0.566 | 5 | 0.890 | 0.893 | 0.886 | 0.764 | 0.631 | 0.763 | 0.886 | 0.021 | 0.022 | -0.186 |
| 2026-02-01 | 8366 | 0.452 | 6 | 0.873 | 0.875 | 0.859 | 0.760 | 0.607 | 0.760 | 0.864 | 0.049 | 0.089 | -0.266 |
| 2026-03-01 | 9969 | 0.496 | 6 | 0.869 | 0.871 | 0.860 | 0.761 | 0.617 | 0.761 | 0.861 | 0.055 | 0.097 | -0.295 |
| 2026-04-01 | 8485 | 0.421 | 7 | 0.867 | 0.868 | 0.859 | 0.747 | 0.606 | 0.747 | 0.858 | 0.071 | 0.071 | -0.001 |
| 2026-05-01 | 10530 | 0.505 | 9 | 0.892 | 0.893 | 0.887 | 0.775 | 0.631 | 0.775 | 0.885 | 0.011 | 0.013 | 0.032 |
| 2026-06-01 | 10036 | 0.486 | 10 | 0.897 | 0.897 | 0.892 | 0.789 | 0.639 | 0.789 | 0.889 | 0.017 | 0.026 | 0.090 |
| 2026-07-01 | 8989 | 0.425 | 11 | 0.884 | 0.885 | 0.879 | 0.763 | 0.625 | 0.763 | 0.877 | 0.013 | 0.015 | -0.029 |
| 2026-08-01 | 10307 | 0.459 | 12 | 0.873 | 0.876 | 0.868 | 0.726 | 0.714 | 0.726 | 0.862 | 0.021 | 0.021 | -0.039 |

## Targeting Replay (would it have paid?)

A campaign that contacts the top **1,000** players at every cutoff (N to agree with the CRM team). The candidate ranks by **value at risk** = trailing-90-day mean daily GGR x 30 x its churn probability; `recency` and `incumbent` rank by their own score; `random` is the expected value of a random pick. **Realised GGR at risk** = the 30-day value of the players who did churn. No treatment was applied, so this measures targeting quality, not uplift.

| ranking | churn_captured | ggr_at_risk_captured | ggr_at_risk_per_1000 |
|---|---|---|---|
| incumbent | 0.089 [0.066, 0.110] (median 0.097, range 0.039 to 0.143) | 0.071 [0.050, 0.090] (median 0.075, range 0.021 to 0.123) | 16,543.813 [12,479.125, 20,268.551] (median 15,189.229, range 4,685.494 to 26,445.683) |
| lgbm_optimised | 0.080 [0.064, 0.092] (median 0.082, range 0.046 to 0.109) | 0.680 [0.632, 0.710] (median 0.672, range 0.575 to 0.827) | 167,223.909 [147,221.114, 191,271.096] (median 157,100.070, range 126,057.902 to 319,104.718) |
| random | 0.095 [0.078, 0.107] (median 0.100, range 0.062 to 0.120) | 0.095 [0.078, 0.107] (median 0.100, range 0.062 to 0.120) | 22,668.451 [20,365.075, 24,907.977] (median 22,744.941, range 14,774.196 to 37,421.639) |
| recency | 0.148 [0.116, 0.175] (median 0.157, range 0.086 to 0.202) | 0.141 [0.091, 0.190] (median 0.137, range 0.052 to 0.311) | 31,691.926 [22,735.847, 40,272.333] (median 28,691.246, range 15,719.632 to 68,454.404) |

## Value-Matched Policies (review items P0 4 and P0 5)

Every risk score ranks the same players with the same 1,000 contacts, alone (value-blind) and multiplied by the expected 30-day GGR (value-aware). `recency` is the days since the last bet turned into a probability, `simple_rft` a logistic regression on recency, frequency and tenure (both fitted on the training months of each cutoff), `lgbm_60d` the candidate, and `cox_30d` the `p_churn_30d` that scoring writes, so `cox_30d` value-aware is the `value_at_risk_30d` of `player_scores`. Mean over the cutoffs.

| Score | Churners caught, value-blind | GGR at risk caught, value-blind | Churners caught, value-aware | GGR at risk caught, value-aware | Months value-aware beats `simple_rft` value-aware |
|---|---|---|---|---|---|
| `recency` | 14.9% | 14.4% | 4.0% | 61.6% | 0 of 11 |
| `simple_rft` | 17.8% | 4.9% | 7.9% | 68.2% |  |
| `lgbm_60d` | 17.9% | 4.5% | 8.0% | 68.0% | 5 of 11 |
| `cox_30d` | 17.9% | 4.7% | 4.9% | 64.0% | 0 of 11 |

Split of the GGR at risk caught by the final policy (`lgbm_60d` value-aware) against value-blind recency:

- **Predictive uplift** (`lgbm_60d` against `simple_rft`, both value-aware): -0.2 points.
- **Value-weighting uplift** (`lgbm_60d` value-aware against value-blind): +63.5 points.
- **Combined** (`lgbm_60d` value-aware against `recency` value-blind): +53.6 points.

### Cox 30-day against LightGBM 60-day value at risk

Share of the top 1,000 players in both lists: mean 68% (range 52% to 78%). Spearman correlation of the two values at risk: 0.902; of the two probabilities: 0.984.

## Calibration Level: the Early Churn Estimate (review item P0 3)

On a scoring date the calibration level estimates the 60-day churn rate of the recent cohorts, whose label is not complete yet, from data up to that date only: the share of the cohort with no bet so far, times the share of the last complete cohort that, silent after the same number of days, stayed silent for 60 days (`calibration_level.early_rate`). Here the estimate is replayed for every monthly cohort 14, 21, 30, 45 days after its cutoff, with data up to that day only, and set against the rate known today and against simply using the rate of the last complete cohort. Mean absolute error, in points of churn rate.

| Days observed | Cohorts | Mean realised rate | Error of the early estimate | Error of the last complete month |
|---|---|---|---|---|
| 14 days | 12 | 51.6% | 2.40 | 5.66 |
| 21 days | 12 | 51.6% | 1.61 | 5.66 |
| 30 days | 13 | 51.9% | 1.15 | 5.30 |
| 45 days | 13 | 51.9% | 0.57 | 5.09 |

The early estimate is closer to the realised rate in 45 of 50 cohorts.

## Cox PH per Horizon (review item P1 7)

The calibrated Cox probability of churn starting within h days, against the h-day label, on the players whose label is known. Mean over the cutoffs (range).

| Horizon | Players per cutoff | Churn rate | AUC | Brier | ECE |
|---|---|---|---|---|---|
| 7 days | 11,211 | 0.559 | 0.868 (0.852 to 0.889) | 0.173 | 0.150 (0.077 to 0.247) |
| 14 days | 11,211 | 0.588 | 0.867 (0.853 to 0.888) | 0.166 | 0.135 (0.077 to 0.221) |
| 30 days | 11,211 | 0.641 | 0.867 (0.846 to 0.887) | 0.154 | 0.114 (0.064 to 0.223) |


## How to Read It

- **Top-decile condition at 1.10x, not the plan's 1.5x** (`docs/decision_top_decile_condition.md`): precision cannot exceed 1 and the recency rule already reaches 0.784, so the most any model could reach is 1.27x, and this ceiling falls in the months with more churn. 1.10x is the strictest round value the model meets with 95% confidence in brands 14, 64 and 73, and at today's precisions it is the plan's own demand (1.5 times the churners found per wasted contact). Interval: 95%, circular block bootstrap of 3 consecutive cutoffs, 2000 resamples.
- **Calibration lags the churn rate**: the calibration month is the latest one whose 60-day label was known on C, 2 to 3 months before it. When the monthly churn rate moves between them (around 50% from November to January, 30 to 40% the rest of the year), every probability is off by about the same amount: the ECE rises while the AUC does not move. The level shift on C removes most of it. What is left comes from the 60 days after C (a holiday season, a sports calendar), which no data before C shows: in the analysis of 2026-10-09 even the exact level of the month before C would not have kept every month of brand 14 under 0.08 (`eda/07_calibration_seasonality.ipynb`).
- **The replay trades churners for value**: the candidate ranks by value at risk, so it contacts fewer players who churn than recency does, but the ones it contacts carry most of the GGR at risk. Ranking by churn probability alone is the top-decile columns above.
- **The cutoffs are not independent**: the players repeat and the training months overlap. The CIs resample blocks of consecutive months, which keeps the dependence between neighbouring months, but not players, so they can still be a little optimistic; the month-by-month counts do not depend on any resampling.
- **The winsorisation caps** come from the main pipeline's configuration (they may have seen later months' features, never labels).
- **Early cutoffs train on few months**, and on no deposit data before March 2026 (the deposit features are empty then). Training cutoffs before 2025-09 have under 150 days of history, so `days_since_first_bet` (capped at 150) is lower there for the same players.

Generated by `src/models/backtest.py` (`make backtest`). Tables: `data/03_output/backtest/backtest_brand72_2026-10-06_1792078504_metrics.csv`, `data/03_output/backtest/backtest_brand72_2026-10-06_1792078504_replay.csv`.
