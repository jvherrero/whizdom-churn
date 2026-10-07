# Feature Importance v0 (brandId 64)

This document reports three importance measures together for the two baseline models. Gain-based importance alone overstates features with many distinct values and features that are correlated with others, so I do not use it here.

| Measure | Question it answers |
|---|---|
| Permutation importance | How much do the model's AUC and calibration fall when this feature is shuffled? |
| SHAP values | What does this feature do to each player's score, and in which direction? |
| Ablation by family | What is a whole family of features worth to the model? |

## 0. Models and Data

| | LightGBM | Cox PH |
|---|---|---|
| MLflow run | `lightgbm_classifier_brand64_1791355570` | `cox_ph_brand64_1791355643` |
| Target | `event_60d` (churn in the next 60 days) | `duration_days` + `event_observed` (churn day) |
| Features (after Stage 2 selection) | 8 | 7 |
| Test score (last cutoff) | AUC 0.8767 | c-index 0.8677 |

- Dataset: `data/processed/train_dataset_64_2026-07-20_2026-07-29_2026-08-07_1791355499.parquet`.
- Rows: train 23,039, validation 4,082, test 14,894. The test month is 2026-08-07.
- With only 3 cutoffs there is no separate validation month. The validation rows are 15% of the players of the training months, never seen in training. Permutation importance and SHAP use them.
- Brands: LightGBM gets `brandId` as a categorical feature and Cox PH is stratified by brand (one baseline per brand), so `brandId` never appears in the Cox tables. With one brand it is constant and its importance is 0. Ablation does not drop it: the model always adds it.
- Every number here comes from the models logged in those two runs. The script checks that they reproduce the logged test score before it measures anything.

## 1. Permutation Importance

I shuffle one feature at a time on the validation rows, 10 times, and measure how much the score changes. The interval is a 95% t-interval over the 10 shuffles, so it shows how stable the shuffle is, not the sampling error of the validation set.

- **AUC drop**: positive means the feature helps the model rank players.
- **Brier rise** and **ECE rise**: positive means the probabilities get less reliable. ECE is the expected calibration error over 10 equal-size bins.
- The probabilities are the ones the model delivers: isotonic-calibrated.

### LightGBM

| feature | family | AUC drop [95% CI] | Brier rise [95% CI] | ECE rise [95% CI] |
|---|---|---|---|---|
| n_active_days_last_30d | frequency | 0.0797 [0.0763, 0.0830] | 0.0408 [0.0392, 0.0424] | 0.0269 [0.0240, 0.0298] |
| tenure_days_missing | tenure | 0.0230 [0.0215, 0.0245] | 0.0140 [0.0132, 0.0149] | 0.0217 [0.0192, 0.0242] |
| days_since_last_deposit | recency | 0.0192 [0.0177, 0.0206] | 0.0106 [0.0098, 0.0115] | 0.0090 [0.0080, 0.0100] |
| stake_last_30d | monetary | 0.0141 [0.0124, 0.0158] | 0.0094 [0.0085, 0.0104] | 0.0238 [0.0216, 0.0261] |
| max_prior_gap_days | tenure | 0.0120 [0.0110, 0.0130] | 0.0076 [0.0071, 0.0080] | 0.0112 [0.0088, 0.0135] |
| n_bonus_last_7d | frequency | 0.0093 [0.0079, 0.0108] | 0.0052 [0.0047, 0.0057] | 0.0102 [0.0081, 0.0124] |
| bonus_amount_trend_7 | lag_trend | 0.0040 [0.0034, 0.0045] | 0.0016 [0.0014, 0.0019] | 0.0054 [0.0039, 0.0068] |
| brandId | other | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

![Permutation importance, LightGBM](../data/03_output/feature_importance/brand64/permutation_lightgbm.png)

### Cox PH

The Cox model gives a risk ranking, not a probability, so only the c-index is measured.

| feature | family | c-index drop [95% CI] |
|---|---|---|
| n_active_days_last_30d | frequency | 0.0864 [0.0836, 0.0892] |
| max_prior_gap_days | tenure | 0.0176 [0.0162, 0.0189] |
| tenure_days_missing | tenure | 0.0168 [0.0156, 0.0179] |
| stake_last_30d | monetary | 0.0135 [0.0120, 0.0150] |
| n_bonus_last_7d | frequency | 0.0063 [0.0051, 0.0075] |
| days_since_last_deposit | recency | 0.0059 [0.0047, 0.0071] |
| bonus_amount_trend_7 | lag_trend | 0.0000 [-0.0003, 0.0003] |

![Permutation importance, Cox PH](../data/03_output/feature_importance/brand64/permutation_cox.png)

## 2. SHAP Values

SHAP splits each player's score into one part per feature. The parts add up to the score.

- **LightGBM**: exact TreeSHAP values from LightGBM itself (`pred_contrib=True`, the same algorithm as `shap.TreeExplainer`), in log-odds of churn.
- **Cox PH**: the model is linear, so the exact SHAP value is `beta * (x - training mean)`, in log-hazard. Positive means a higher risk of churning sooner.
- **Direction** is the sign of the rank correlation (`rank_corr`) between the feature value and its SHAP value. Under 0.5 in absolute value one direction would mislead, so it says "non-monotonic" and the dependence plot shows the real shape.

### LightGBM

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| n_active_days_last_30d | frequency | 1.0764 | -0.9648 | higher value, lower churn risk |
| days_since_last_deposit | recency | 0.4437 | 0.8288 | higher value, higher churn risk |
| max_prior_gap_days | tenure | 0.3582 | -0.8266 | higher value, lower churn risk |
| tenure_days_missing | tenure | 0.3345 | -0.7946 | higher value, lower churn risk |
| stake_last_30d | monetary | 0.2711 | -0.8851 | higher value, lower churn risk |
| n_bonus_last_7d | frequency | 0.2312 | -0.9332 | higher value, lower churn risk |
| bonus_amount_trend_7 | lag_trend | 0.0770 | 0.2509 | non-monotonic, see the plot |
| brandId | other | 0.0000 |  | categorical (brand) |

![Global SHAP, LightGBM](../data/03_output/feature_importance/brand64/shap_global_lightgbm.png)

Dependence plots for the top 8 features (all the features the model uses). The x axis is in real units (days, EUR, counts), not on the sign-log scale. Each dot is one validation player.

![SHAP dependence, LightGBM](../data/03_output/feature_importance/brand64/shap_dependence_lightgbm.png)

### Cox PH

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| n_active_days_last_30d | frequency | 0.5278 | -1.0000 | higher value, lower churn risk |
| max_prior_gap_days | tenure | 0.2371 | -1.0000 | higher value, lower churn risk |
| stake_last_30d | monetary | 0.2359 | -1.0000 | higher value, lower churn risk |
| tenure_days_missing | tenure | 0.2192 | -1.0000 | higher value, lower churn risk |
| days_since_last_deposit | recency | 0.1437 | 1.0000 | higher value, higher churn risk |
| n_bonus_last_7d | frequency | 0.1368 | -1.0000 | higher value, lower churn risk |
| bonus_amount_trend_7 | lag_trend | 0.0172 | 1.0000 | higher value, higher churn risk |

![Global SHAP, Cox PH](../data/03_output/feature_importance/brand64/shap_global_cox.png)

A dependence plot for a linear model is a straight line with slope `beta`, so I do not draw it. The hazard ratios are in `hazard_ratios.csv` in the Cox MLflow run.

## 3. Importance by Family

How much each group of features matters, with two measures side by side. **SHAP share**: the family's part of the model's total mean |SHAP| on the validation rows, so how much it weighs in the predictions. **Score lost**: what the model loses on the test month when the whole family is removed and the model is retrained, so what it adds that the other families cannot replace. A family can weigh a lot and still lose little when removed, if other families carry the same information.

| family | lightgbm_shap_share | cox_shap_share | lightgbm_auc_lost | cox_c_index_lost |
|---|---|---|---|---|
| frequency | 0.4683 | 0.4379 | 0.0052 | 0.0069 |
| tenure | 0.2481 | 0.3007 | 0.0240 | 0.0212 |
| recency | 0.1589 | 0.0947 | 0.0052 | 0.0028 |
| monetary | 0.0971 | 0.1554 | 0.0036 | 0.0016 |
| lag_trend | 0.0276 | 0.0113 | 0.0007 | 0.0000 |

![Importance by family](../data/03_output/feature_importance/brand64/importance_by_family.png)

## 4. Ablation by Family

I retrain the model without every feature of one family and compare it with the full model (same features otherwise, same parameters, same rows). A negative delta means the family helps. A family with no feature in the model is left empty.

Full model: LightGBM AUC valid 0.8776, test 0.8767. Cox c-index valid 0.8491, test 0.8677.

| family | LightGBM features | LightGBM AUC delta, valid | LightGBM AUC delta, test | Cox features | Cox c-index delta, valid | Cox c-index delta, test |
|---|---|---|---|---|---|---|
| recency | days_since_last_deposit | -0.0050 | -0.0052 | days_since_last_deposit | -0.0028 | -0.0028 |
| frequency | n_active_days_last_30d, n_bonus_last_7d | -0.0043 | -0.0052 | n_active_days_last_30d, n_bonus_last_7d | -0.0098 | -0.0069 |
| monetary | stake_last_30d | -0.0010 | -0.0036 | stake_last_30d | -0.0011 | -0.0016 |
| lag_trend | bonus_amount_trend_7 | -0.0027 | -0.0007 | bonus_amount_trend_7 | 0.0003 | -0.0000 |
| tenure | tenure_days_missing, max_prior_gap_days | -0.0171 | -0.0240 | tenure_days_missing, max_prior_gap_days | -0.0162 | -0.0212 |
| mix |  |  |  |  |  |  |
| segment |  |  |  |  |  |  |

The most valuable family on the test month is **tenure** for LightGBM (AUC -0.0240 without it) and **tenure** for Cox (c-index -0.0212 without it).

## 5. Known Limits

- **The Cox test month only has churn on day 0.** With data through 2026-10-04, a churn day after the cutoff can only be confirmed up to 60 days before the data ends, which is day 0 for the last cutoff. So on the test month the c-index measures "who is already gone", not "which day".
- **The intervals only cover the shuffle.** A different validation sample would move the numbers more than the intervals show.
- **Correlated features share credit.** Stage 2 already removed pairs over 0.95, but features from the same family (for example `n_active_days_last_30d` and `max_prior_gap_days`) can still hide each other in permutation importance. Ablation by family shows the joint value.

Generated by `src/models/feature_importance.py` (`make importance`). The figures and tables are in `data/03_output/feature_importance/brand64/` and in the `feature_importance/` folder of both MLflow runs.
