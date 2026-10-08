# Backtest v0: rolling-origin (brandId 64)

Does the model predict the way it will be used: trained on the past, scored on a date, judged on what the players did next, repeated over a year. Backtest id `backtest_brand64_2026-10-06_1791391493`, data as of 2026-10-06, 11 test cutoffs, 41 min. Config: `configs/backtest.yaml`.

**Verdict: FAIL** (a model that fails is not promoted, whatever its single-split scores).

| Pass condition (delivery plan) | Candidate (`lgbm_optimised`) | Result |
|---|---|---|
| mean top-decile precision >= 1.5x the recency rule | 0.866 vs 0.714 (1.21x) | FAIL |
| AUC never below 0.7 | min 0.851 (2025-12-01) | pass |
| ECE never above 0.08 | max 0.104 (2026-03-01) | FAIL |

![Backtest](../data/03_output/backtest/backtest_brand64_2026-10-06_1791391493.png)

## How It Runs

- **Test cutoffs**: the first day of each month, 2025-10-01, 2025-11-01, 2025-12-01, 2026-01-01, 2026-02-01, 2026-03-01, 2026-04-01, 2026-05-01, 2026-06-01, 2026-07-01, 2026-08-01. Skipped (fewer than 2 earlier months with a known label): 2025-09-01.
- **Training at each cutoff C**: every earlier monthly cutoff whose 60-day label was already known on C (cutoff + 60 days <= C), labels computed from data up to C only; the latest of them is the validation month (early stopping, tuning, calibration). So the training data grows from 2 to 12 months, and the models never see a label C did not know.
- **Scored**: every player with a bet in the 30 days up to C, judged on the 60 days after it (data up to 2026-10-06).
- **Models**: trained exactly as in normal training, with 20 Optuna trials (not 100) and at most 500 trees, as the plan allows for the backtest. Each training run is in MLflow with the tag `backtest_id`.
- **Comparators**: `lgbm_reference` (optimisation step 1), `cox` (partial hazard), `recency` (days since the last bet alone) and `incumbent` (the platform's `churn_score` in `gld_player_signals_daily`, a weighted heuristic: 0.5 recency + 0.3 frequency + 0.2 value). The plan's 7/14/30-day labels are replaced by the 60-day churn of `docs/churn_definition_v0.md`.

## Results Across Cutoffs

Mean, 95% bootstrap CI of the mean (cutoffs resampled, 200 times) and range. Log-loss, ECE and Brier only for the calibrated probabilities.

| comparator | auc | c_index | top_decile_precision | top_decile_recall | log_loss | ece | brier |
|---|---|---|---|---|---|---|---|
| cox | 0.869 [0.860, 0.878] (range 0.839 to 0.891) | 0.816 [0.803, 0.834] (range 0.788 to 0.884) | 0.858 [0.831, 0.886] (range 0.788 to 0.964) | 0.217 [0.201, 0.232] (range 0.159 to 0.280) | n/a | n/a | n/a |
| incumbent | 0.624 [0.605, 0.646] (range 0.564 to 0.705) | 0.596 [0.579, 0.621] (range 0.550 to 0.689) | 0.380 [0.334, 0.443] (range 0.269 to 0.607) | 0.095 [0.084, 0.111] (range 0.076 to 0.146) | n/a | n/a | n/a |
| lgbm_optimised | 0.875 [0.868, 0.882] (range 0.851 to 0.896) | 0.818 [0.805, 0.837] (range 0.795 to 0.887) | 0.866 [0.840, 0.892] (range 0.782 to 0.953) | 0.219 [0.204, 0.234] (range 0.176 to 0.283) | 0.440 [0.423, 0.460] (range 0.387 to 0.509) | 0.051 [0.033, 0.071] (range 0.019 to 0.104) | 0.143 [0.137, 0.151] (range 0.125 to 0.168) |
| lgbm_reference | 0.875 [0.867, 0.882] (range 0.845 to 0.897) | 0.818 [0.805, 0.838] (range 0.791 to 0.887) | 0.865 [0.835, 0.893] (range 0.783 to 0.955) | 0.218 [0.204, 0.233] (range 0.177 to 0.283) | 0.442 [0.423, 0.464] (range 0.385 to 0.524) | 0.052 [0.034, 0.072] (range 0.023 to 0.108) | 0.144 [0.137, 0.152] (range 0.124 to 0.174) |
| recency | 0.752 [0.733, 0.767] (range 0.669 to 0.791) | 0.705 [0.689, 0.721] (range 0.643 to 0.740) | 0.714 [0.680, 0.742] (range 0.578 to 0.779) | 0.179 [0.170, 0.189] (range 0.149 to 0.198) | n/a | n/a | n/a |

## AUC per Cutoff

| cutoff | n_players | churn_rate | n_train_cutoffs | lgbm_optimised | lgbm_reference | cox | recency | incumbent | ECE lgbm_optimised |
|---|---|---|---|---|---|---|---|---|---|
| 2025-10-01 | 19279 | 0.392 | 2 | 0.892 | 0.891 | 0.887 | 0.791 | 0.634 | 0.024 |
| 2025-11-01 | 22163 | 0.470 | 3 | 0.884 | 0.882 | 0.882 | 0.762 | 0.618 | 0.050 |
| 2025-12-01 | 17727 | 0.506 | 4 | 0.851 | 0.845 | 0.848 | 0.730 | 0.597 | 0.092 |
| 2026-01-01 | 14569 | 0.503 | 5 | 0.864 | 0.865 | 0.839 | 0.669 | 0.564 | 0.019 |
| 2026-02-01 | 11878 | 0.379 | 6 | 0.881 | 0.882 | 0.879 | 0.778 | 0.650 | 0.102 |
| 2026-03-01 | 12622 | 0.367 | 6 | 0.877 | 0.878 | 0.868 | 0.756 | 0.619 | 0.104 |
| 2026-04-01 | 13187 | 0.373 | 7 | 0.865 | 0.864 | 0.865 | 0.758 | 0.610 | 0.065 |
| 2026-05-01 | 12823 | 0.356 | 9 | 0.855 | 0.856 | 0.851 | 0.751 | 0.600 | 0.030 |
| 2026-06-01 | 12982 | 0.381 | 10 | 0.876 | 0.877 | 0.872 | 0.773 | 0.627 | 0.026 |
| 2026-07-01 | 10463 | 0.295 | 11 | 0.886 | 0.887 | 0.878 | 0.748 | 0.638 | 0.034 |
| 2026-08-01 | 13996 | 0.416 | 12 | 0.896 | 0.897 | 0.891 | 0.755 | 0.705 | 0.021 |

## Targeting Replay (would it have paid?)

A campaign that contacts the top **1,000** players at every cutoff (N to agree with the CRM team). The candidate ranks by **value at risk** = trailing-90-day mean daily GGR x 30 x its churn probability; `recency` and `incumbent` rank by their own score; `random` is the expected value of a random pick. **Realised GGR at risk** = the 30-day value of the players who did churn. No treatment was applied, so this measures targeting quality, not uplift.

| ranking | churn_captured | ggr_at_risk_captured | ggr_at_risk_per_1000 |
|---|---|---|---|
| incumbent | 0.060 [0.041, 0.082] (range 0.009 to 0.136) | 0.080 [0.040, 0.124] (range 0.003 to 0.218) | 18,109.129 [8,721.930, 28,905.330] (range 759.564 to 69,349.551) |
| lgbm_optimised | 0.052 [0.045, 0.058] (range 0.034 to 0.074) | 0.658 [0.620, 0.696] (range 0.512 to 0.770) | 168,743.462 [138,015.607, 197,627.601] (range 67,807.277 to 257,338.011) |
| random | 0.071 [0.061, 0.079] (range 0.045 to 0.096) | 0.071 [0.061, 0.079] (range 0.045 to 0.096) | 17,506.948 [14,912.610, 20,364.693] (range 10,258.922 to 28,325.681) |
| recency | 0.130 [0.111, 0.150] (range 0.074 to 0.188) | 0.123 [0.082, 0.171] (range 0.030 to 0.257) | 28,810.832 [18,968.080, 40,733.286] (range 10,229.920 to 81,538.538) |

## How to Read It

- **Top-decile precision condition**: precision cannot exceed 1, so 1.5x the recency rule is only reachable if the recency rule's precision is under 0.67. Here the recency rule reaches 0.714, so the most any model could reach is 1.40x. With a 60-day churn rate around 40%, recency alone already finds the players who left; the comparison of top-decile *recall* and of the targeting replay says more.
- **Calibration lags the churn rate**: the calibration month is the latest one whose 60-day label was known on C, 2 to 3 months before it. When the monthly churn rate moves between them (around 50% from November to January, 30 to 40% the rest of the year), the probabilities are off by the same amount, so the ECE rises in those months while the AUC does not move. Calibrating on 3 validation months instead of 1 (`valid_cutoffs: 3`, tried on 2026-10-07) lowered the worst month from 0.104 to 0.086 but did not remove it, so 1 month is kept; the delayed-label monitoring of Week 2 is what catches it.
- **The replay trades churners for value**: the candidate ranks by value at risk, so it contacts fewer players who churn than recency does, but the ones it contacts carry most of the GGR at risk. Ranking by churn probability alone is the top-decile columns above.
- **The cutoffs are not independent**: the players repeat and the training months overlap, so the CIs are optimistic.
- **The winsorisation caps** come from the main pipeline's configuration (they may have seen later months' features, never labels).
- **Early cutoffs train on few months**, and on no deposit data before March 2026 (the deposit features are empty then). Training cutoffs before 2025-09 have under 150 days of history, so `days_since_first_bet` (capped at 150) is lower there for the same players.

Generated by `src/models/backtest.py` (`make backtest`). Tables: `data/03_output/backtest/backtest_brand64_2026-10-06_1791391493_metrics.csv`, `data/03_output/backtest/backtest_brand64_2026-10-06_1791391493_replay.csv`.
