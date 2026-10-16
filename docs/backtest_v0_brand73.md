# Backtest v0: rolling-origin (brandId 73)

Does the model predict the way it will be used: trained on the past, scored on a date, judged on what the players did next, repeated over a year. Backtest id `backtest_brand73_2026-10-06_1792081564`, data as of 2026-10-06, 11 test cutoffs, 152 min. Config: `configs/backtest.yaml`.

**Verdict: PASS** (a model that fails is not promoted, whatever its single-split scores).

| Pass condition (delivery plan) | Candidate (`lgbm_optimised`) | Result |
|---|---|---|
| mean top-decile precision >= 1.10x the recency rule | 0.914 vs 0.730: 1.25x [1.20, 1.29] | pass |
| AUC never below 0.7 | min 0.867 (2025-12-01) | pass |
| ECE never above 0.08 | max 0.073 (2025-12-01) | pass |

![Backtest](../data/03_output/backtest/backtest_brand73_2026-10-06_1792081564.png)

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
| cox | 0.879 [0.871, 0.887] (median 0.877, range 0.857 to 0.897) | 0.820 [0.806, 0.836] (median 0.812, range 0.794 to 0.862) | 0.912 [0.897, 0.923] (median 0.915, range 0.865 to 0.952) | 0.230 [0.211, 0.244] (median 0.238, range 0.185 to 0.261) | n/a | n/a | n/a |
| incumbent | 0.648 [0.634, 0.670] (median 0.645, range 0.612 to 0.736) | 0.606 [0.590, 0.631] (median 0.599, range 0.569 to 0.715) | 0.373 [0.319, 0.433] (median 0.338, range 0.270 to 0.558) | 0.095 [0.080, 0.116] (median 0.083, range 0.072 to 0.169) | n/a | n/a | n/a |
| lgbm_optimised | 0.887 [0.880, 0.894] (median 0.885, range 0.867 to 0.903) | 0.820 [0.807, 0.836] (median 0.809, range 0.799 to 0.866) | 0.914 [0.899, 0.925] (median 0.913, range 0.862 to 0.958) | 0.231 [0.211, 0.245] (median 0.238, range 0.186 to 0.260) | 0.412 [0.397, 0.431] (median 0.405, range 0.383 to 0.470) | 0.025 [0.015, 0.038] (median 0.015, range 0.008 to 0.073) | 0.134 [0.128, 0.141] (median 0.131, range 0.123 to 0.156) |
| lgbm_reference | 0.888 [0.881, 0.894] (median 0.886, range 0.868 to 0.904) | 0.821 [0.808, 0.837] (median 0.811, range 0.799 to 0.868) | 0.915 [0.901, 0.926] (median 0.914, range 0.867 to 0.957) | 0.231 [0.212, 0.246] (median 0.239, range 0.186 to 0.262) | 0.410 [0.395, 0.430] (median 0.402, range 0.382 to 0.468) | 0.024 [0.014, 0.038] (median 0.016, range 0.008 to 0.074) | 0.133 [0.127, 0.141] (median 0.130, range 0.122 to 0.155) |
| recency | 0.779 [0.775, 0.782] (median 0.780, range 0.765 to 0.798) | 0.714 [0.706, 0.722] (median 0.715, range 0.691 to 0.739) | 0.730 [0.699, 0.763] (median 0.735, range 0.649 to 0.832) | 0.184 [0.174, 0.191] (median 0.182, range 0.162 to 0.204) | n/a | n/a | n/a |
| recency_prob | 0.779 [0.775, 0.782] (median 0.780, range 0.765 to 0.798) | 0.714 [0.706, 0.722] (median 0.715, range 0.691 to 0.739) | 0.730 [0.699, 0.763] (median 0.735, range 0.649 to 0.832) | 0.184 [0.174, 0.191] (median 0.182, range 0.162 to 0.204) | n/a | n/a | n/a |
| simple_rft | 0.881 [0.876, 0.886] (median 0.882, range 0.864 to 0.896) | 0.817 [0.807, 0.831] (median 0.808, range 0.799 to 0.853) | 0.915 [0.903, 0.927] (median 0.917, range 0.867 to 0.954) | 0.231 [0.213, 0.245] (median 0.237, range 0.185 to 0.262) | n/a | n/a | n/a |

## Month by Month against Recency (review item P1 11)

The cutoffs are not independent, so the interval alone can look too sure. This table does not depend on any resampling: the median and the worst month of each comparator, and the months where it beats the recency rule (AUC above recency; top-decile ratio at or above 1.10x).

| comparator | months | auc_median | auc_worst | auc_worst_month | months_auc_above_recency | ratio_median | ratio_worst | ratio_worst_month | months_ratio_at_threshold |
|---|---|---|---|---|---|---|---|---|---|
| lgbm_optimised | 11 | 0.885 | 0.867 | 2025-12-01 | 11 | 1.236 | 1.152 | 2025-12-01 | 11 |
| lgbm_reference | 11 | 0.886 | 0.868 | 2025-12-01 | 11 | 1.240 | 1.151 | 2025-12-01 | 11 |
| cox | 11 | 0.877 | 0.857 | 2025-12-01 | 11 | 1.231 | 1.144 | 2025-12-01 | 11 |
| incumbent | 11 | 0.645 | 0.612 | 2026-02-01 | 0 | 0.485 | 0.367 | 2026-04-01 | 0 |
| recency_prob | 11 | 0.780 | 0.765 | 2026-02-01 | 2 | 1.000 | 1.000 | 2025-11-01 | 0 |
| simple_rft | 11 | 0.882 | 0.864 | 2025-12-01 | 11 | 1.233 | 1.146 | 2025-12-01 | 11 |

## AUC per Cutoff

| cutoff | n_players | churn_rate | n_train_cutoffs | lgbm_optimised | lgbm_reference | cox | recency | incumbent | recency_prob | simple_rft | ECE lgbm_optimised | ECE without level | level shift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-10-01 | 33579 | 0.434 | 2 | 0.880 | 0.880 | 0.866 | 0.791 | 0.649 | 0.791 | 0.882 | 0.026 | 0.028 | 0.022 |
| 2025-11-01 | 33229 | 0.459 | 3 | 0.884 | 0.884 | 0.874 | 0.772 | 0.622 | 0.772 | 0.881 | 0.040 | 0.029 | -0.098 |
| 2025-12-01 | 29815 | 0.515 | 4 | 0.867 | 0.868 | 0.857 | 0.787 | 0.656 | 0.787 | 0.864 | 0.073 | 0.078 | 0.030 |
| 2026-01-01 | 20395 | 0.389 | 5 | 0.882 | 0.883 | 0.877 | 0.767 | 0.636 | 0.767 | 0.875 | 0.008 | 0.023 | -0.105 |
| 2026-02-01 | 21728 | 0.380 | 6 | 0.893 | 0.894 | 0.889 | 0.765 | 0.612 | 0.765 | 0.886 | 0.029 | 0.073 | -0.336 |
| 2026-03-01 | 24222 | 0.412 | 6 | 0.894 | 0.895 | 0.889 | 0.798 | 0.645 | 0.798 | 0.891 | 0.029 | 0.084 | -0.454 |
| 2026-04-01 | 20472 | 0.360 | 7 | 0.885 | 0.885 | 0.874 | 0.783 | 0.641 | 0.783 | 0.875 | 0.011 | 0.011 | -0.002 |
| 2026-05-01 | 21925 | 0.387 | 9 | 0.893 | 0.894 | 0.887 | 0.768 | 0.625 | 0.768 | 0.886 | 0.010 | 0.014 | 0.057 |
| 2026-06-01 | 21001 | 0.385 | 10 | 0.903 | 0.904 | 0.897 | 0.789 | 0.647 | 0.789 | 0.896 | 0.015 | 0.015 | -0.014 |
| 2026-07-01 | 21212 | 0.356 | 11 | 0.895 | 0.896 | 0.884 | 0.780 | 0.659 | 0.780 | 0.883 | 0.014 | 0.018 | -0.042 |
| 2026-08-01 | 20034 | 0.331 | 12 | 0.884 | 0.886 | 0.876 | 0.766 | 0.736 | 0.766 | 0.869 | 0.015 | 0.015 | 0.001 |

## Targeting Replay (would it have paid?)

A campaign that contacts the top **1,000** players at every cutoff (N to agree with the CRM team). The candidate ranks by **value at risk** = trailing-90-day mean daily GGR x 30 x its churn probability; `recency` and `incumbent` rank by their own score; `random` is the expected value of a random pick. **Realised GGR at risk** = the 30-day value of the players who did churn. No treatment was applied, so this measures targeting quality, not uplift.

| ranking | churn_captured | ggr_at_risk_captured | ggr_at_risk_per_1000 |
|---|---|---|---|
| incumbent | 0.026 [0.010, 0.047] (median 0.013, range 0.007 to 0.085) | 0.024 [0.008, 0.042] (median 0.017, range 0.001 to 0.079) | 14,161.378 [5,819.381, 24,542.616] (median 10,557.180, range 943.730 to 48,594.196) |
| lgbm_optimised | 0.036 [0.030, 0.041] (median 0.038, range 0.024 to 0.049) | 0.592 [0.556, 0.620] (median 0.592, range 0.412 to 0.686) | 398,450.149 [345,104.938, 456,139.264] (median 401,302.517, range 302,017.399 to 601,256.392) |
| random | 0.043 [0.037, 0.047] (median 0.046, range 0.030 to 0.050) | 0.043 [0.037, 0.047] (median 0.046, range 0.030 to 0.050) | 28,355.432 [25,938.620, 31,757.005] (median 25,929.609, range 22,197.841 to 42,956.303) |
| recency | 0.081 [0.066, 0.093] (median 0.087, range 0.051 to 0.105) | 0.075 [0.059, 0.089] (median 0.073, range 0.033 to 0.162) | 47,819.577 [43,075.779, 52,596.582] (median 43,931.478, range 24,602.075 to 91,042.660) |

## Value-Matched Policies (review items P0 4 and P0 5)

Every risk score ranks the same players with the same 1,000 contacts, alone (value-blind) and multiplied by the expected 30-day GGR (value-aware). `recency` is the days since the last bet turned into a probability, `simple_rft` a logistic regression on recency, frequency and tenure (both fitted on the training months of each cutoff), `lgbm_60d` the candidate, and `cox_30d` the `p_churn_30d` that scoring writes, so `cox_30d` value-aware is the `value_at_risk_30d` of `player_scores`. Mean over the cutoffs.

| Score | Churners caught, value-blind | GGR at risk caught, value-blind | Churners caught, value-aware | GGR at risk caught, value-aware | Months value-aware beats `simple_rft` value-aware |
|---|---|---|---|---|---|
| `recency` | 8.1% | 7.2% | 1.8% | 52.5% | 0 of 11 |
| `simple_rft` | 10.2% | 1.5% | 3.5% | 57.7% |  |
| `lgbm_60d` | 10.3% | 1.4% | 3.6% | 59.2% | 9 of 11 |
| `cox_30d` | 10.3% | 1.4% | 2.0% | 53.0% | 0 of 11 |

Split of the GGR at risk caught by the final policy (`lgbm_60d` value-aware) against value-blind recency:

- **Predictive uplift** (`lgbm_60d` against `simple_rft`, both value-aware): +1.5 points.
- **Value-weighting uplift** (`lgbm_60d` value-aware against value-blind): +57.8 points.
- **Combined** (`lgbm_60d` value-aware against `recency` value-blind): +52.0 points.

### Cox 30-day against LightGBM 60-day value at risk

Share of the top 1,000 players in both lists: mean 63% (range 54% to 73%). Spearman correlation of the two values at risk: 0.868; of the two probabilities: 0.982.

## Calibration Level: the Early Churn Estimate (review item P0 3)

On a scoring date the calibration level estimates the 60-day churn rate of the recent cohorts, whose label is not complete yet, from data up to that date only: the share of the cohort with no bet so far, times the share of the last complete cohort that, silent after the same number of days, stayed silent for 60 days (`calibration_level.early_rate`). Here the estimate is replayed for every monthly cohort 14, 21, 30, 45 days after its cutoff, with data up to that day only, and set against the rate known today and against simply using the rate of the last complete cohort. Mean absolute error, in points of churn rate.

| Days observed | Cohorts | Mean realised rate | Error of the early estimate | Error of the last complete month |
|---|---|---|---|---|
| 14 days | 12 | 40.5% | 1.62 | 4.13 |
| 21 days | 12 | 40.5% | 1.38 | 4.13 |
| 30 days | 13 | 40.6% | 0.89 | 3.69 |
| 45 days | 13 | 40.6% | 0.33 | 3.75 |

The early estimate is closer to the realised rate in 44 of 50 cohorts.

## Cox PH per Horizon (review item P1 7)

The calibrated Cox probability of churn starting within h days, against the h-day label, on the players whose label is known. Mean over the cutoffs (range).

| Horizon | Players per cutoff | Churn rate | AUC | Brier | ECE |
|---|---|---|---|---|---|
| 7 days | 24,758 | 0.447 | 0.874 (0.855 to 0.892) | 0.162 | 0.114 (0.015 to 0.190) |
| 14 days | 24,758 | 0.476 | 0.870 (0.853 to 0.889) | 0.161 | 0.104 (0.014 to 0.174) |
| 30 days | 24,758 | 0.531 | 0.866 (0.851 to 0.884) | 0.159 | 0.088 (0.014 to 0.192) |


## How to Read It

- **Top-decile condition at 1.10x, not the plan's 1.5x** (`docs/decision_top_decile_condition.md`): precision cannot exceed 1 and the recency rule already reaches 0.730, so the most any model could reach is 1.37x, and this ceiling falls in the months with more churn. 1.10x is the strictest round value the model meets with 95% confidence in brands 14, 64 and 73, and at today's precisions it is the plan's own demand (1.5 times the churners found per wasted contact). Interval: 95%, circular block bootstrap of 3 consecutive cutoffs, 2000 resamples.
- **Calibration lags the churn rate**: the calibration month is the latest one whose 60-day label was known on C, 2 to 3 months before it. When the monthly churn rate moves between them (around 50% from November to January, 30 to 40% the rest of the year), every probability is off by about the same amount: the ECE rises while the AUC does not move. The level shift on C removes most of it. What is left comes from the 60 days after C (a holiday season, a sports calendar), which no data before C shows: in the analysis of 2026-10-09 even the exact level of the month before C would not have kept every month of brand 14 under 0.08 (`eda/07_calibration_seasonality.ipynb`).
- **The replay trades churners for value**: the candidate ranks by value at risk, so it contacts fewer players who churn than recency does, but the ones it contacts carry most of the GGR at risk. Ranking by churn probability alone is the top-decile columns above.
- **The cutoffs are not independent**: the players repeat and the training months overlap. The CIs resample blocks of consecutive months, which keeps the dependence between neighbouring months, but not players, so they can still be a little optimistic; the month-by-month counts do not depend on any resampling.
- **The winsorisation caps** come from the main pipeline's configuration (they may have seen later months' features, never labels).
- **Early cutoffs train on few months**, and on no deposit data before March 2026 (the deposit features are empty then). Training cutoffs before 2025-09 have under 150 days of history, so `days_since_first_bet` (capped at 150) is lower there for the same players.

Generated by `src/models/backtest.py` (`make backtest`). Tables: `data/03_output/backtest/backtest_brand73_2026-10-06_1792081564_metrics.csv`, `data/03_output/backtest/backtest_brand73_2026-10-06_1792081564_replay.csv`.
