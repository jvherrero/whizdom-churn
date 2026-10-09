# Backtest v0: rolling-origin (brandId 64)

Does the model predict the way it will be used: trained on the past, scored on a date, judged on what the players did next, repeated over a year. Backtest id `backtest_brand64_2026-10-06_1791536375`, data as of 2026-10-06, 11 test cutoffs, 49 min. Config: `configs/backtest.yaml`.

**Verdict: FAIL** (a model that fails is not promoted, whatever its single-split scores).

| Pass condition (delivery plan) | Candidate (`lgbm_optimised`) | Result |
|---|---|---|
| mean top-decile precision >= 1.10x the recency rule | 0.861 vs 0.713: 1.21x [1.17, 1.25] | pass |
| AUC never below 0.7 | min 0.853 (2025-12-01) | pass |
| ECE never above 0.08 | max 0.087 (2025-12-01) | FAIL |

![Backtest](../data/03_output/backtest/backtest_brand64_2026-10-06_1791536375.png)

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
| cox | 0.868 [0.859, 0.877] (range 0.839 to 0.892) | 0.815 [0.802, 0.834] (range 0.788 to 0.885) | 0.859 [0.833, 0.886] (range 0.789 to 0.964) | 0.217 [0.201, 0.232] (range 0.161 to 0.280) | n/a | n/a | n/a |
| incumbent | 0.624 [0.605, 0.646] (range 0.564 to 0.705) | 0.596 [0.579, 0.621] (range 0.550 to 0.689) | 0.380 [0.336, 0.441] (range 0.270 to 0.611) | 0.095 [0.084, 0.111] (range 0.076 to 0.147) | n/a | n/a | n/a |
| lgbm_optimised | 0.875 [0.867, 0.882] (range 0.853 to 0.896) | 0.818 [0.805, 0.837] (range 0.796 to 0.887) | 0.861 [0.834, 0.887] (range 0.789 to 0.957) | 0.217 [0.203, 0.232] (range 0.174 to 0.278) | 0.436 [0.420, 0.451] (range 0.388 to 0.506) | 0.042 [0.029, 0.053] (range 0.011 to 0.087) | 0.142 [0.136, 0.147] (range 0.125 to 0.166) |
| lgbm_reference | 0.875 [0.868, 0.882] (range 0.852 to 0.896) | 0.818 [0.805, 0.838] (range 0.796 to 0.887) | 0.861 [0.831, 0.885] (range 0.777 to 0.956) | 0.217 [0.202, 0.232] (range 0.174 to 0.277) | 0.436 [0.420, 0.452] (range 0.386 to 0.507) | 0.041 [0.029, 0.052] (range 0.011 to 0.085) | 0.142 [0.136, 0.147] (range 0.125 to 0.166) |
| recency | 0.752 [0.733, 0.767] (range 0.669 to 0.791) | 0.705 [0.689, 0.721] (range 0.643 to 0.740) | 0.713 [0.678, 0.740] (range 0.577 to 0.772) | 0.179 [0.170, 0.188] (range 0.150 to 0.197) | n/a | n/a | n/a |

## AUC per Cutoff

| cutoff | n_players | churn_rate | n_train_cutoffs | lgbm_optimised | lgbm_reference | cox | recency | incumbent | ECE lgbm_optimised | ECE without level | level shift |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-10-01 | 19279 | 0.392 | 2 | 0.890 | 0.889 | 0.886 | 0.791 | 0.634 | 0.025 | 0.025 | -0.001 |
| 2025-11-01 | 22163 | 0.470 | 3 | 0.879 | 0.880 | 0.877 | 0.762 | 0.618 | 0.037 | 0.055 | 0.128 |
| 2025-12-01 | 17727 | 0.506 | 4 | 0.853 | 0.852 | 0.848 | 0.730 | 0.597 | 0.087 | 0.096 | 0.066 |
| 2026-01-01 | 14569 | 0.503 | 5 | 0.865 | 0.866 | 0.839 | 0.669 | 0.564 | 0.011 | 0.019 | -0.139 |
| 2026-02-01 | 11878 | 0.379 | 6 | 0.881 | 0.882 | 0.878 | 0.778 | 0.650 | 0.060 | 0.102 | -0.302 |
| 2026-03-01 | 12622 | 0.367 | 6 | 0.874 | 0.875 | 0.866 | 0.756 | 0.619 | 0.048 | 0.108 | -0.428 |
| 2026-04-01 | 13187 | 0.373 | 7 | 0.866 | 0.865 | 0.865 | 0.758 | 0.610 | 0.065 | 0.065 | -0.001 |
| 2026-05-01 | 12823 | 0.356 | 9 | 0.855 | 0.855 | 0.852 | 0.751 | 0.600 | 0.032 | 0.031 | -0.011 |
| 2026-06-01 | 12982 | 0.381 | 10 | 0.876 | 0.876 | 0.872 | 0.773 | 0.627 | 0.025 | 0.025 | 0.000 |
| 2026-07-01 | 10463 | 0.295 | 11 | 0.886 | 0.887 | 0.878 | 0.748 | 0.638 | 0.033 | 0.033 | 0.005 |
| 2026-08-01 | 13996 | 0.416 | 12 | 0.896 | 0.896 | 0.892 | 0.755 | 0.705 | 0.034 | 0.023 | -0.185 |

## Targeting Replay (would it have paid?)

A campaign that contacts the top **1,000** players at every cutoff (N to agree with the CRM team). The candidate ranks by **value at risk** = trailing-90-day mean daily GGR x 30 x its churn probability; `recency` and `incumbent` rank by their own score; `random` is the expected value of a random pick. **Realised GGR at risk** = the 30-day value of the players who did churn. No treatment was applied, so this measures targeting quality, not uplift.

| ranking | churn_captured | ggr_at_risk_captured | ggr_at_risk_per_1000 |
|---|---|---|---|
| incumbent | 0.060 [0.041, 0.081] (range 0.009 to 0.139) | 0.087 [0.046, 0.130] (range 0.002 to 0.231) | 19,746.961 [9,361.512, 30,524.863] (range 615.737 to 73,265.653) |
| lgbm_optimised | 0.052 [0.045, 0.058] (range 0.030 to 0.072) | 0.657 [0.623, 0.696] (range 0.509 to 0.757) | 168,800.129 [137,929.051, 197,315.657] (range 67,863.222 to 255,724.741) |
| random | 0.071 [0.061, 0.079] (range 0.045 to 0.096) | 0.071 [0.061, 0.079] (range 0.045 to 0.096) | 17,506.948 [14,912.610, 20,364.693] (range 10,258.922 to 28,325.681) |
| recency | 0.130 [0.110, 0.150] (range 0.073 to 0.189) | 0.112 [0.079, 0.155] (range 0.029 to 0.239) | 25,159.227 [18,938.017, 31,366.985] (range 10,044.331 to 48,441.110) |

## How to Read It

- **Top-decile condition at 1.10x, not the plan's 1.5x** (`docs/decision_top_decile_condition.md`): precision cannot exceed 1 and the recency rule already reaches 0.713, so the most any model could reach is 1.40x, and this ceiling falls in the months with more churn. 1.10x is the strictest round value the model meets with 95% confidence in brands 14, 64 and 73, and at today's precisions it is the plan's own demand (1.5 times the churners found per wasted contact). Interval: 95%, cutoffs resampled 200 times.
- **Calibration lags the churn rate**: the calibration month is the latest one whose 60-day label was known on C, 2 to 3 months before it. When the monthly churn rate moves between them (around 50% from November to January, 30 to 40% the rest of the year), every probability is off by about the same amount: the ECE rises while the AUC does not move. The level shift on C removes most of it. What is left comes from the 60 days after C (a holiday season, a sports calendar), which no data before C shows: in the analysis of 2026-10-09 even the exact level of the month before C would not have kept every month of brand 14 under 0.08 (`eda/07_calibration_seasonality.ipynb`).
- **The replay trades churners for value**: the candidate ranks by value at risk, so it contacts fewer players who churn than recency does, but the ones it contacts carry most of the GGR at risk. Ranking by churn probability alone is the top-decile columns above.
- **The cutoffs are not independent**: the players repeat and the training months overlap, so the CIs are optimistic.
- **The winsorisation caps** come from the main pipeline's configuration (they may have seen later months' features, never labels).
- **Early cutoffs train on few months**, and on no deposit data before March 2026 (the deposit features are empty then). Training cutoffs before 2025-09 have under 150 days of history, so `days_since_first_bet` (capped at 150) is lower there for the same players.

Generated by `src/models/backtest.py` (`make backtest`). Tables: `data/03_output/backtest/backtest_brand64_2026-10-06_1791536375_metrics.csv`, `data/03_output/backtest/backtest_brand64_2026-10-06_1791536375_replay.csv`.
