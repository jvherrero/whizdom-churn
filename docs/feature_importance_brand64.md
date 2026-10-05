# Feature Importance v0 (brandId=64)

This document reports three importance measures together for the two baseline models. Gain-based importance alone overstates features with many distinct values and features that are correlated with others, so I do not use it here.

| Measure | Question it answers |
|---|---|
| Permutation importance | How much do the model's AUC and calibration fall when this feature is shuffled? |
| SHAP values | What does this feature do to each player's score, and in which direction? |
| Ablation by family | What is a whole family of features worth to the model? |

## 0. Models and Data

| | LightGBM | Cox PH |
|---|---|---|
| MLflow run | `lightgbm_classifier_1791203911` | `cox_ph_1791203962` |
| Target | `event_60d` (churn in the next 60 days) | `duration_days` + `event_observed` (churn day) |
| Features (after Stage 2 selection) | 7 | 4 |
| Test score (last cutoff) | AUC 0.8739 | c-index 0.8614 |

- Dataset: `data/processed/train_dataset_64_2026-07-18_2026-07-27_2026-08-05_1791191780.parquet`.
- Rows: train 22,634, validation 4,007, test 14,687. The test month is 2026-08-05.
- With only 3 cutoffs there is no separate validation month. The validation rows are 15% of the players of the training months, never seen in training. Permutation importance and SHAP use them.
- Every number here comes from the models logged in those two runs. The script checks that they reproduce the logged test score before it measures anything.

## 1. Permutation Importance

I shuffle one feature at a time on the validation rows, 10 times, and measure how much the score changes. The interval is a 95% t-interval over the 10 shuffles, so it shows how stable the shuffle is, not the sampling error of the validation set.

- **AUC drop**: positive means the feature helps the model rank players.
- **Brier rise** and **ECE rise**: positive means the probabilities get less reliable. ECE is the expected calibration error over 10 equal-size bins.
- The probabilities are the ones the model delivers: isotonic-calibrated.

### LightGBM

| feature | family | AUC drop [95% CI] | Brier rise [95% CI] | ECE rise [95% CI] |
|---|---|---|---|---|
| n_active_days_last_30d | frequency | 0.0982 [0.0945, 0.1019] | 0.0450 [0.0434, 0.0465] | 0.0391 [0.0353, 0.0430] |
| stake_last_30d | monetary | 0.0359 [0.0333, 0.0384] | 0.0270 [0.0255, 0.0284] | 0.0679 [0.0651, 0.0707] |
| tenure_days_missing | tenure | 0.0225 [0.0210, 0.0240] | 0.0138 [0.0130, 0.0145] | 0.0267 [0.0241, 0.0293] |
| max_prior_gap_days | tenure | 0.0173 [0.0159, 0.0186] | 0.0115 [0.0109, 0.0121] | 0.0251 [0.0224, 0.0278] |
| days_since_last_deposit | recency | 0.0154 [0.0144, 0.0163] | 0.0090 [0.0085, 0.0096] | 0.0138 [0.0113, 0.0162] |
| n_bonus_last_7d | frequency | 0.0137 [0.0125, 0.0149] | 0.0093 [0.0086, 0.0100] | 0.0297 [0.0279, 0.0314] |
| bonus_amount_trend_7 | lag_trend | 0.0080 [0.0069, 0.0091] | 0.0052 [0.0044, 0.0059] | 0.0204 [0.0175, 0.0233] |

![Permutation importance, LightGBM](../data/03_output/feature_importance/brand64/permutation_lightgbm.png)

### Cox PH

The Cox model gives a risk ranking, not a probability, so only the c-index is measured.

| feature | family | c-index drop [95% CI] |
|---|---|---|
| n_active_days_last_30d | frequency | 0.0671 [0.0637, 0.0705] |
| stake_last_30d | monetary | 0.0356 [0.0339, 0.0373] |
| max_prior_gap_days | tenure | 0.0347 [0.0328, 0.0366] |
| tenure_days_missing | tenure | 0.0113 [0.0102, 0.0124] |

![Permutation importance, Cox PH](../data/03_output/feature_importance/brand64/permutation_cox.png)

## 2. SHAP Values

SHAP splits each player's score into one part per feature. The parts add up to the score.

- **LightGBM**: exact TreeSHAP values from LightGBM itself (`pred_contrib=True`, the same algorithm as `shap.TreeExplainer`), in log-odds of churn.
- **Cox PH**: the model is linear, so the exact SHAP value is `beta * (x - training mean)`, in log-hazard. Positive means a higher risk of churning sooner.
- **Direction** is the sign of the rank correlation (`rank_corr`) between the feature value and its SHAP value. Under 0.5 in absolute value one direction would mislead, so it says "non-monotonic" and the dependence plot shows the real shape.

### LightGBM

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| n_active_days_last_30d | frequency | 1.1356 | -0.9633 | higher value, lower churn risk |
| days_since_last_deposit | recency | 0.3127 | 0.7778 | higher value, higher churn risk |
| stake_last_30d | monetary | 0.2650 | -0.8510 | higher value, lower churn risk |
| tenure_days_missing | tenure | 0.2591 | -0.7884 | higher value, lower churn risk |
| max_prior_gap_days | tenure | 0.2487 | -0.8499 | higher value, lower churn risk |
| n_bonus_last_7d | frequency | 0.1693 | -0.9250 | higher value, lower churn risk |
| bonus_amount_trend_7 | lag_trend | 0.0363 | -0.0638 | non-monotonic, see the plot |

![Global SHAP, LightGBM](../data/03_output/feature_importance/brand64/shap_global_lightgbm.png)

Dependence plots for the top 7 features (all the features the model uses). The x axis is in real units (days, EUR, counts), not on the sign-log scale. Each dot is one validation player.

![SHAP dependence, LightGBM](../data/03_output/feature_importance/brand64/shap_dependence_lightgbm.png)

### Cox PH

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| n_active_days_last_30d | frequency | 0.3954 | -1.0000 | higher value, lower churn risk |
| stake_last_30d | monetary | 0.3074 | -1.0000 | higher value, lower churn risk |
| max_prior_gap_days | tenure | 0.3030 | -1.0000 | higher value, lower churn risk |
| tenure_days_missing | tenure | 0.1706 | -1.0000 | higher value, lower churn risk |

![Global SHAP, Cox PH](../data/03_output/feature_importance/brand64/shap_global_cox.png)

A dependence plot for a linear model is a straight line with slope `beta`, so I do not draw it. The hazard ratios are in `hazard_ratios.csv` in the Cox MLflow run.

## 3. Ablation by Family

I retrain the model without every feature of one family and compare it with the full model (same features otherwise, same parameters, same rows). A negative delta means the family helps. A family with no feature in the model is left empty.

Full model: LightGBM AUC valid 0.8800, test 0.8739. Cox c-index valid 0.8487, test 0.8614.

| family | LightGBM features | LightGBM AUC delta, valid | LightGBM AUC delta, test | Cox features | Cox c-index delta, valid | Cox c-index delta, test |
|---|---|---|---|---|---|---|
| recency | days_since_last_deposit | -0.0034 | -0.0029 |  |  |  |
| frequency | n_active_days_last_30d, n_bonus_last_7d | -0.0073 | -0.0043 | n_active_days_last_30d | -0.0095 | -0.0091 |
| monetary | stake_last_30d | -0.0007 | -0.0048 | stake_last_30d | -0.0133 | -0.0136 |
| lag_trend | bonus_amount_trend_7 | -0.0030 | -0.0023 |  |  |  |
| tenure | tenure_days_missing, max_prior_gap_days | -0.0109 | -0.0194 | max_prior_gap_days, tenure_days_missing | -0.0125 | -0.0208 |
| mix |  |  |  |  |  |  |
| segment |  |  |  |  |  |  |

The most valuable family on the test month is **tenure** for LightGBM (AUC -0.0194 without it) and **tenure** for Cox (c-index -0.0208 without it).

## 4. Known Limits

- **The Cox test month only has churn on day 0.** With data through 2026-10-04, a churn day after the cutoff can only be confirmed up to 60 days before the data ends, which is day 0 for the last cutoff. So on the test month the c-index measures "who is already gone", not "which day".
- **The intervals only cover the shuffle.** A different validation sample would move the numbers more than the intervals show.
- **Correlated features share credit.** Stage 2 already removed pairs over 0.95, but features from the same family (for example `n_active_days_last_30d` and `max_prior_gap_days`) can still hide each other in permutation importance. Ablation by family shows the joint value.

Generated by `src/models/feature_importance.py` (`make importance`). The figures and tables are in `data/03_output/feature_importance/brand64/` and in the `feature_importance/` folder of both MLflow runs.
