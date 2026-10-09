# Backtest v0: rolling-origin (brandId 23)

Does the model predict the way it will be used: trained on the past, scored on a date, judged on what the players did next, repeated over a year. Backtest id `backtest_brand23_2026-10-06_1791554664`, data as of 2026-10-06, 11 test cutoffs, 127 min. Config: `configs/backtest.yaml`.

**Verdict: PASS** (a model that fails is not promoted, whatever its single-split scores).

| Pass condition (delivery plan) | Candidate (`lgbm_optimised`) | Result |
|---|---|---|
| mean top-decile precision >= 1.10x the recency rule | 0.871 vs 0.684: 1.27x [1.21, 1.36] | pass |
| AUC never below 0.7 | min 0.876 (2026-01-01) | pass |
| ECE never above 0.08 | max 0.066 (2025-12-01) | pass |

![Backtest](../data/03_output/backtest/backtest_brand23_2026-10-06_1791554664.png)

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
| cox | 0.905 [0.895, 0.913] (range 0.871 to 0.932) | 0.841 [0.829, 0.859] (range 0.807 to 0.906) | 0.867 [0.838, 0.891] (range 0.771 to 0.971) | 0.313 [0.280, 0.352] (range 0.221 to 0.420) | n/a | n/a | n/a |
| incumbent | 0.640 [0.616, 0.668] (range 0.574 to 0.718) | 0.603 [0.582, 0.632] (range 0.553 to 0.706) | 0.237 [0.196, 0.274] (range 0.151 to 0.375) | 0.084 [0.072, 0.104] (range 0.064 to 0.162) | n/a | n/a | n/a |
| lgbm_optimised | 0.912 [0.903, 0.921] (range 0.876 to 0.938) | 0.845 [0.830, 0.866] (range 0.806 to 0.914) | 0.871 [0.846, 0.893] (range 0.802 to 0.965) | 0.316 [0.280, 0.355] (range 0.221 to 0.422) | 0.338 [0.306, 0.366] (range 0.273 to 0.408) | 0.036 [0.026, 0.046] (range 0.010 to 0.066) | 0.106 [0.096, 0.116] (range 0.085 to 0.130) |
| lgbm_reference | 0.913 [0.904, 0.922] (range 0.875 to 0.938) | 0.846 [0.831, 0.866] (range 0.805 to 0.915) | 0.876 [0.851, 0.899] (range 0.799 to 0.964) | 0.318 [0.281, 0.357] (range 0.221 to 0.424) | 0.336 [0.304, 0.365] (range 0.271 to 0.409) | 0.036 [0.025, 0.045] (range 0.009 to 0.067) | 0.106 [0.095, 0.116] (range 0.084 to 0.130) |
| recency | 0.797 [0.782, 0.809] (range 0.758 to 0.825) | 0.734 [0.720, 0.751] (range 0.700 to 0.788) | 0.684 [0.635, 0.728] (range 0.554 to 0.791) | 0.245 [0.221, 0.274] (range 0.189 to 0.301) | n/a | n/a | n/a |

## AUC per Cutoff

| cutoff | n_players | churn_rate | n_train_cutoffs | lgbm_optimised | lgbm_reference | cox | recency | incumbent | ECE lgbm_optimised | ECE without level | level shift |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-10-01 | 46997 | 0.354 | 2 | 0.907 | 0.906 | 0.904 | 0.786 | 0.617 | 0.040 | 0.040 | -0.003 |
| 2025-11-01 | 48223 | 0.402 | 3 | 0.900 | 0.901 | 0.899 | 0.759 | 0.593 | 0.060 | 0.057 | -0.022 |
| 2025-12-01 | 42188 | 0.417 | 4 | 0.913 | 0.913 | 0.909 | 0.802 | 0.638 | 0.066 | 0.050 | -0.145 |
| 2026-01-01 | 31250 | 0.303 | 5 | 0.876 | 0.875 | 0.871 | 0.797 | 0.665 | 0.040 | 0.025 | -0.163 |
| 2026-02-01 | 27218 | 0.296 | 6 | 0.904 | 0.908 | 0.899 | 0.824 | 0.675 | 0.021 | 0.023 | -0.079 |
| 2026-03-01 | 29448 | 0.319 | 6 | 0.938 | 0.938 | 0.932 | 0.758 | 0.574 | 0.041 | 0.048 | -0.085 |
| 2026-04-01 | 26294 | 0.236 | 7 | 0.916 | 0.924 | 0.906 | 0.825 | 0.657 | 0.057 | 0.057 | -0.002 |
| 2026-05-01 | 26616 | 0.234 | 9 | 0.910 | 0.911 | 0.897 | 0.789 | 0.589 | 0.018 | 0.023 | 0.089 |
| 2026-06-01 | 26268 | 0.253 | 10 | 0.930 | 0.931 | 0.920 | 0.807 | 0.642 | 0.022 | 0.032 | 0.129 |
| 2026-07-01 | 23899 | 0.193 | 11 | 0.916 | 0.916 | 0.901 | 0.821 | 0.672 | 0.010 | 0.012 | -0.020 |
| 2026-08-01 | 23919 | 0.196 | 12 | 0.924 | 0.925 | 0.915 | 0.800 | 0.718 | 0.024 | 0.038 | -0.162 |

## Targeting Replay (would it have paid?)

A campaign that contacts the top **1,000** players at every cutoff (N to agree with the CRM team). The candidate ranks by **value at risk** = trailing-90-day mean daily GGR x 30 x its churn probability; `recency` and `incumbent` rank by their own score; `random` is the expected value of a random pick. **Realised GGR at risk** = the 30-day value of the players who did churn. No treatment was applied, so this measures targeting quality, not uplift.

| ranking | churn_captured | ggr_at_risk_captured | ggr_at_risk_per_1000 |
|---|---|---|---|
| incumbent | 0.015 [0.005, 0.030] (range 0.002 to 0.078) | 0.016 [0.004, 0.034] (range 0.001 to 0.088) | 6,871.566 [1,868.298, 13,860.298] (range 373.465 to 27,392.633) |
| lgbm_optimised | 0.034 [0.027, 0.042] (range 0.018 to 0.058) | 0.618 [0.581, 0.651] (range 0.534 to 0.720) | 315,462.060 [252,832.800, 394,329.101] (range 178,001.755 to 732,859.946) |
| random | 0.033 [0.029, 0.038] (range 0.021 to 0.042) | 0.033 [0.029, 0.038] (range 0.021 to 0.042) | 16,117.315 [13,382.028, 18,365.244] (range 7,087.748 to 26,840.039) |
| recency | 0.088 [0.070, 0.107] (range 0.038 to 0.138) | 0.090 [0.062, 0.120] (range 0.013 to 0.166) | 42,655.639 [30,158.678, 56,218.161] (range 6,108.189 to 73,610.569) |

## How to Read It

- **Top-decile condition at 1.10x, not the plan's 1.5x** (`docs/decision_top_decile_condition.md`): precision cannot exceed 1 and the recency rule already reaches 0.684, so the most any model could reach is 1.46x, and this ceiling falls in the months with more churn. 1.10x is the strictest round value the model meets with 95% confidence in brands 14, 64 and 73, and at today's precisions it is the plan's own demand (1.5 times the churners found per wasted contact). Interval: 95%, cutoffs resampled 200 times.
- **Calibration lags the churn rate**: the calibration month is the latest one whose 60-day label was known on C, 2 to 3 months before it. When the monthly churn rate moves between them (around 50% from November to January, 30 to 40% the rest of the year), every probability is off by about the same amount: the ECE rises while the AUC does not move. The level shift on C removes most of it. What is left comes from the 60 days after C (a holiday season, a sports calendar), which no data before C shows: in the analysis of 2026-10-09 even the exact level of the month before C would not have kept every month of brand 14 under 0.08 (`eda/07_calibration_seasonality.ipynb`).
- **The replay trades churners for value**: the candidate ranks by value at risk, so it contacts fewer players who churn than recency does, but the ones it contacts carry most of the GGR at risk. Ranking by churn probability alone is the top-decile columns above.
- **The cutoffs are not independent**: the players repeat and the training months overlap, so the CIs are optimistic.
- **The winsorisation caps** come from the main pipeline's configuration (they may have seen later months' features, never labels).
- **Early cutoffs train on few months**, and on no deposit data before March 2026 (the deposit features are empty then). Training cutoffs before 2025-09 have under 150 days of history, so `days_since_first_bet` (capped at 150) is lower there for the same players.

Generated by `src/models/backtest.py` (`make backtest`). Tables: `data/03_output/backtest/backtest_brand23_2026-10-06_1791554664_metrics.csv`, `data/03_output/backtest/backtest_brand23_2026-10-06_1791554664_replay.csv`.
