# Backtest v0: rolling-origin (brandId 14)

Does the model predict the way it will be used: trained on the past, scored on a date, judged on what the players did next, repeated over a year. Backtest id `backtest_brand14_2026-10-06_1792074259`, data as of 2026-10-06, 11 test cutoffs, 70 min. Config: `configs/backtest.yaml`.

**Verdict: FAIL** (a model that fails is not promoted, whatever its single-split scores).

| Pass condition (delivery plan) | Candidate (`lgbm_optimised`) | Result |
|---|---|---|
| mean top-decile precision >= 1.10x the recency rule | 0.869 vs 0.642: 1.35x [1.30, 1.41] | pass |
| AUC never below 0.7 | min 0.877 (2026-02-01) | pass |
| ECE never above 0.08 | max 0.094 (2025-12-01) | FAIL |

![Backtest](../data/03_output/backtest/backtest_brand14_2026-10-06_1792074259.png)

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
| cox | 0.883 [0.879, 0.887] (median 0.880, range 0.875 to 0.893) | 0.820 [0.799, 0.843] (median 0.811, range 0.785 to 0.875) | 0.863 [0.843, 0.881] (median 0.863, range 0.817 to 0.923) | 0.257 [0.240, 0.277] (median 0.261, range 0.220 to 0.312) | n/a | n/a | n/a |
| incumbent | 0.648 [0.638, 0.662] (median 0.638, range 0.594 to 0.708) | 0.613 [0.599, 0.633] (median 0.602, range 0.563 to 0.697) | 0.311 [0.249, 0.382] (median 0.246, range 0.192 to 0.562) | 0.090 [0.079, 0.107] (median 0.080, range 0.073 to 0.160) | n/a | n/a | n/a |
| lgbm_optimised | 0.887 [0.883, 0.890] (median 0.888, range 0.877 to 0.894) | 0.821 [0.800, 0.845] (median 0.813, range 0.786 to 0.884) | 0.869 [0.847, 0.886] (median 0.882, range 0.812 to 0.923) | 0.258 [0.241, 0.278] (median 0.262, range 0.218 to 0.310) | 0.400 [0.390, 0.411] (median 0.397, range 0.377 to 0.458) | 0.034 [0.024, 0.044] (median 0.029, range 0.014 to 0.094) | 0.130 [0.126, 0.133] (median 0.128, range 0.121 to 0.149) |
| lgbm_reference | 0.887 [0.884, 0.891] (median 0.888, range 0.877 to 0.896) | 0.821 [0.800, 0.845] (median 0.811, range 0.784 to 0.886) | 0.870 [0.848, 0.888] (median 0.884, range 0.814 to 0.917) | 0.259 [0.241, 0.278] (median 0.262, range 0.219 to 0.311) | 0.400 [0.390, 0.411] (median 0.394, range 0.377 to 0.456) | 0.033 [0.022, 0.043] (median 0.029, range 0.010 to 0.094) | 0.129 [0.126, 0.133] (median 0.128, range 0.121 to 0.148) |
| recency | 0.775 [0.769, 0.782] (median 0.776, range 0.752 to 0.802) | 0.718 [0.704, 0.735] (median 0.712, range 0.688 to 0.760) | 0.642 [0.601, 0.677] (median 0.670, range 0.569 to 0.712) | 0.190 [0.182, 0.198] (median 0.190, range 0.169 to 0.221) | n/a | n/a | n/a |
| recency_prob | 0.775 [0.768, 0.782] (median 0.776, range 0.753 to 0.802) | 0.718 [0.704, 0.735] (median 0.711, range 0.688 to 0.759) | 0.643 [0.602, 0.683] (median 0.656, range 0.560 to 0.721) | 0.190 [0.183, 0.198] (median 0.190, range 0.171 to 0.222) | n/a | n/a | n/a |
| simple_rft | 0.879 [0.873, 0.884] (median 0.879, range 0.868 to 0.890) | 0.819 [0.798, 0.842] (median 0.809, range 0.787 to 0.875) | 0.852 [0.825, 0.879] (median 0.851, range 0.779 to 0.901) | 0.253 [0.239, 0.271] (median 0.250, range 0.213 to 0.298) | n/a | n/a | n/a |

## Month by Month against Recency (review item P1 11)

The cutoffs are not independent, so the interval alone can look too sure. This table does not depend on any resampling: the median and the worst month of each comparator, and the months where it beats the recency rule (AUC above recency; top-decile ratio at or above 1.10x).

| comparator | months | auc_median | auc_worst | auc_worst_month | months_auc_above_recency | ratio_median | ratio_worst | ratio_worst_month | months_ratio_at_threshold |
|---|---|---|---|---|---|---|---|---|---|
| lgbm_optimised | 11 | 0.888 | 0.877 | 2026-02-01 | 11 | 1.336 | 1.220 | 2025-11-01 | 11 |
| lgbm_reference | 11 | 0.888 | 0.877 | 2026-02-01 | 11 | 1.347 | 1.220 | 2025-11-01 | 11 |
| cox | 11 | 0.880 | 0.875 | 2026-02-01 | 11 | 1.338 | 1.214 | 2025-11-01 | 11 |
| incumbent | 11 | 0.638 | 0.594 | 2025-10-01 | 0 | 0.416 | 0.331 | 2026-02-01 | 0 |
| recency_prob | 11 | 0.776 | 0.753 | 2026-01-01 | 3 | 1.005 | 0.954 | 2025-11-01 | 0 |
| simple_rft | 11 | 0.879 | 0.868 | 2025-12-01 | 11 | 1.336 | 1.238 | 2025-11-01 | 11 |

## AUC per Cutoff

| cutoff | n_players | churn_rate | n_train_cutoffs | lgbm_optimised | lgbm_reference | cox | recency | incumbent | recency_prob | simple_rft | ECE lgbm_optimised | ECE without level | level shift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-10-01 | 7756 | 0.331 | 2 | 0.883 | 0.882 | 0.879 | 0.759 | 0.594 | 0.759 | 0.878 | 0.022 | 0.022 | -0.002 |
| 2025-11-01 | 7195 | 0.320 | 3 | 0.883 | 0.884 | 0.880 | 0.793 | 0.634 | 0.792 | 0.881 | 0.014 | 0.014 | -0.005 |
| 2025-12-01 | 8174 | 0.419 | 4 | 0.884 | 0.885 | 0.880 | 0.793 | 0.679 | 0.791 | 0.868 | 0.094 | 0.093 | -0.011 |
| 2026-01-01 | 5812 | 0.306 | 5 | 0.889 | 0.889 | 0.880 | 0.752 | 0.631 | 0.753 | 0.871 | 0.037 | 0.033 | -0.033 |
| 2026-02-01 | 5159 | 0.262 | 6 | 0.877 | 0.877 | 0.875 | 0.767 | 0.638 | 0.767 | 0.871 | 0.014 | 0.044 | -0.323 |
| 2026-03-01 | 5885 | 0.301 | 6 | 0.882 | 0.884 | 0.877 | 0.778 | 0.660 | 0.777 | 0.872 | 0.045 | 0.077 | -0.245 |
| 2026-04-01 | 5624 | 0.295 | 7 | 0.890 | 0.888 | 0.886 | 0.761 | 0.639 | 0.761 | 0.879 | 0.029 | 0.040 | -0.091 |
| 2026-05-01 | 6843 | 0.395 | 9 | 0.888 | 0.889 | 0.887 | 0.776 | 0.635 | 0.776 | 0.884 | 0.040 | 0.054 | 0.106 |
| 2026-06-01 | 7158 | 0.408 | 10 | 0.890 | 0.891 | 0.887 | 0.784 | 0.633 | 0.784 | 0.885 | 0.039 | 0.052 | 0.094 |
| 2026-07-01 | 6662 | 0.365 | 11 | 0.894 | 0.893 | 0.893 | 0.802 | 0.675 | 0.802 | 0.890 | 0.020 | 0.014 | -0.187 |
| 2026-08-01 | 6507 | 0.352 | 12 | 0.894 | 0.896 | 0.884 | 0.761 | 0.708 | 0.761 | 0.885 | 0.017 | 0.016 | -0.007 |

## Targeting Replay (would it have paid?)

A campaign that contacts the top **1,000** players at every cutoff (N to agree with the CRM team). The candidate ranks by **value at risk** = trailing-90-day mean daily GGR x 30 x its churn probability; `recency` and `incumbent` rank by their own score; `random` is the expected value of a random pick. **Realised GGR at risk** = the 30-day value of the players who did churn. No treatment was applied, so this measures targeting quality, not uplift.

| ranking | churn_captured | ggr_at_risk_captured | ggr_at_risk_per_1000 |
|---|---|---|---|
| incumbent | 0.181 [0.155, 0.208] (median 0.197, range 0.118 to 0.255) | 0.173 [0.148, 0.197] (median 0.185, range 0.070 to 0.228) | 43,518.306 [36,490.315, 50,639.719] (median 46,632.270, range 8,614.806 to 66,693.256) |
| lgbm_optimised | 0.107 [0.091, 0.126] (median 0.106, range 0.069 to 0.159) | 0.798 [0.770, 0.821] (median 0.814, range 0.677 to 0.855) | 195,434.825 [177,407.433, 217,483.376] (median 198,323.883, range 83,614.744 to 260,878.655) |
| random | 0.154 [0.140, 0.170] (median 0.150, range 0.122 to 0.194) | 0.154 [0.140, 0.170] (median 0.150, range 0.122 to 0.194) | 37,135.809 [32,750.478, 41,579.447] (median 38,309.339, range 17,178.435 to 51,044.639) |
| recency | 0.291 [0.261, 0.331] (median 0.284, range 0.212 to 0.415) | 0.284 [0.217, 0.354] (median 0.310, range 0.141 to 0.500) | 71,163.222 [52,166.527, 90,247.786] (median 66,801.391, range 21,135.677 to 131,541.177) |

## Value-Matched Policies (review items P0 4 and P0 5)

Every risk score ranks the same players with the same 1,000 contacts, alone (value-blind) and multiplied by the expected 30-day GGR (value-aware). `recency` is the days since the last bet turned into a probability, `simple_rft` a logistic regression on recency, frequency and tenure (both fitted on the training months of each cutoff), `lgbm_60d` the candidate, and `cox_30d` the `p_churn_30d` that scoring writes, so `cox_30d` value-aware is the `value_at_risk_30d` of `player_scores`. Mean over the cutoffs.

| Score | Churners caught, value-blind | GGR at risk caught, value-blind | Churners caught, value-aware | GGR at risk caught, value-aware | Months value-aware beats `simple_rft` value-aware |
|---|---|---|---|---|---|
| `recency` | 29.0% | 31.0% | 4.8% | 73.9% | 0 of 11 |
| `simple_rft` | 37.6% | 3.7% | 9.6% | 79.9% |  |
| `lgbm_60d` | 38.4% | 3.8% | 10.7% | 79.8% | 7 of 11 |
| `cox_30d` | 38.3% | 3.7% | 5.4% | 76.0% | 0 of 11 |

Split of the GGR at risk caught by the final policy (`lgbm_60d` value-aware) against value-blind recency:

- **Predictive uplift** (`lgbm_60d` against `simple_rft`, both value-aware): -0.1 points.
- **Value-weighting uplift** (`lgbm_60d` value-aware against value-blind): +76.0 points.
- **Combined** (`lgbm_60d` value-aware against `recency` value-blind): +48.8 points.

### Cox 30-day against LightGBM 60-day value at risk

Share of the top 1,000 players in both lists: mean 66% (range 59% to 76%). Spearman correlation of the two values at risk: 0.892; of the two probabilities: 0.983.

## Calibration Level: the Early Churn Estimate (review item P0 3)

On a scoring date the calibration level estimates the 60-day churn rate of the recent cohorts, whose label is not complete yet, from data up to that date only: the share of the cohort with no bet so far, times the share of the last complete cohort that, silent after the same number of days, stayed silent for 60 days (`calibration_level.early_rate`). Here the estimate is replayed for every monthly cohort 14, 21, 30, 45 days after its cutoff, with data up to that day only, and set against the rate known today and against simply using the rate of the last complete cohort. Mean absolute error, in points of churn rate.

| Days observed | Cohorts | Mean realised rate | Error of the early estimate | Error of the last complete month |
|---|---|---|---|---|
| 14 days | 12 | 34.1% | 2.19 | 5.17 |
| 21 days | 12 | 34.1% | 1.89 | 5.17 |
| 30 days | 13 | 34.1% | 1.26 | 3.55 |
| 45 days | 13 | 34.1% | 0.60 | 3.81 |

The early estimate is closer to the realised rate in 33 of 50 cohorts.

## Cox PH per Horizon (review item P1 7)

The calibrated Cox probability of churn starting within h days, against the h-day label, on the players whose label is known. Mean over the cutoffs (range).

| Horizon | Players per cutoff | Churn rate | AUC | Brier | ECE |
|---|---|---|---|---|---|
| 7 days | 6,627 | 0.382 | 0.880 (0.870 to 0.890) | 0.153 | 0.107 (0.052 to 0.171) |
| 14 days | 6,627 | 0.402 | 0.877 (0.866 to 0.894) | 0.153 | 0.101 (0.046 to 0.155) |
| 30 days | 6,627 | 0.450 | 0.873 (0.862 to 0.895) | 0.155 | 0.094 (0.034 to 0.147) |


## How to Read It

- **Top-decile condition at 1.10x, not the plan's 1.5x** (`docs/decision_top_decile_condition.md`): precision cannot exceed 1 and the recency rule already reaches 0.642, so the most any model could reach is 1.56x, and this ceiling falls in the months with more churn. 1.10x is the strictest round value the model meets with 95% confidence in brands 14, 64 and 73, and at today's precisions it is the plan's own demand (1.5 times the churners found per wasted contact). Interval: 95%, circular block bootstrap of 3 consecutive cutoffs, 2000 resamples.
- **Calibration lags the churn rate**: the calibration month is the latest one whose 60-day label was known on C, 2 to 3 months before it. When the monthly churn rate moves between them (around 50% from November to January, 30 to 40% the rest of the year), every probability is off by about the same amount: the ECE rises while the AUC does not move. The level shift on C removes most of it. What is left comes from the 60 days after C (a holiday season, a sports calendar), which no data before C shows: in the analysis of 2026-10-09 even the exact level of the month before C would not have kept every month of brand 14 under 0.08 (`eda/07_calibration_seasonality.ipynb`).
- **The replay trades churners for value**: the candidate ranks by value at risk, so it contacts fewer players who churn than recency does, but the ones it contacts carry most of the GGR at risk. Ranking by churn probability alone is the top-decile columns above.
- **The cutoffs are not independent**: the players repeat and the training months overlap. The CIs resample blocks of consecutive months, which keeps the dependence between neighbouring months, but not players, so they can still be a little optimistic; the month-by-month counts do not depend on any resampling.
- **The winsorisation caps** come from the main pipeline's configuration (they may have seen later months' features, never labels).
- **Early cutoffs train on few months**, and on no deposit data before March 2026 (the deposit features are empty then). Training cutoffs before 2025-09 have under 150 days of history, so `days_since_first_bet` (capped at 150) is lower there for the same players.

Generated by `src/models/backtest.py` (`make backtest`). Tables: `data/03_output/backtest/backtest_brand14_2026-10-06_1792074259_metrics.csv`, `data/03_output/backtest/backtest_brand14_2026-10-06_1792074259_replay.csv`.
