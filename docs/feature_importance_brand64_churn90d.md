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
| MLflow run | `lightgbm_classifier_brand64_1792151357` | `cox_ph_brand64_1792151389` |
| Target | `event_60d` (churn in the next 60 days) | `duration_days` + `event_observed` (churn day) |
| Features (after Stage 2 selection) | 6 | 5 |
| Test score (test months) | AUC 0.8805 | c-index 0.8428 |

- Dataset: `data/processed/churn90d/train_dataset_64_2025-08-01_2025-09-01_2025-10-01_2025-11-01_2025-12-01_2026-01-01_2026-02-01_2026-03-01_2026-04-01_2026-05-01_2026-06-01_2026-07-01_1792151122.parquet`.
- Rows: train 118,558, validation 9,546, test 17,058. Validation month: 2026-05-01. Test months: 2026-06-01, 2026-07-01.
- Permutation importance and SHAP use the validation month, which the models were never fitted on (it is only used for early stopping, tuning and calibration).
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
| active_days_l90d | frequency | 0.1318 [0.1288, 0.1347] | 0.0465 [0.0454, 0.0476] | 0.0568 [0.0542, 0.0594] |
| days_since_last_bet | recency | 0.0335 [0.0323, 0.0348] | 0.0110 [0.0106, 0.0114] | 0.0070 [0.0057, 0.0083] |
| days_since_last_return | tenure | 0.0155 [0.0148, 0.0161] | 0.0082 [0.0079, 0.0086] | 0.0073 [0.0053, 0.0094] |
| wagered_eur_l7d | monetary | 0.0121 [0.0110, 0.0132] | 0.0053 [0.0048, 0.0058] | 0.0108 [0.0100, 0.0117] |
| engagement_score | mix | 0.0052 [0.0047, 0.0057] | 0.0027 [0.0025, 0.0029] | 0.0085 [0.0077, 0.0094] |
| brandId | other | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

The delivery plan keeps a feature in the served model only if its permutation-importance interval excludes zero. Every LightGBM feature passes this rule.

![Permutation importance, LightGBM](../data/03_output/feature_importance/brand64_churn90d/permutation_lightgbm.png)

### Cox PH

The Cox model gives a risk ranking, not a probability, so only the c-index is measured.

| feature | family | c-index drop [95% CI] |
|---|---|---|
| active_days_l90d | frequency | 0.2195 [0.2174, 0.2217] |
| days_since_last_return | tenure | 0.0111 [0.0104, 0.0117] |
| wagered_eur_l7d | monetary | 0.0065 [0.0060, 0.0069] |
| engagement_score | mix | 0.0036 [0.0031, 0.0042] |
| days_since_last_bet | recency | 0.0022 [0.0019, 0.0025] |

![Permutation importance, Cox PH](../data/03_output/feature_importance/brand64_churn90d/permutation_cox.png)

## 2. SHAP Values

SHAP splits each player's score into one part per feature. The parts add up to the score.

- **LightGBM**: exact TreeSHAP values from LightGBM itself (`pred_contrib=True`, the same algorithm as `shap.TreeExplainer`), in log-odds of churn.
- **Cox PH**: the model is linear, so the exact SHAP value is `beta * (x - training mean)`, in log-hazard. Positive means a higher risk of churning sooner.
- **Direction** is the sign of the rank correlation (`rank_corr`) between the feature value and its SHAP value. Under 0.5 in absolute value one direction would mislead, so it says "non-monotonic" and the dependence plot shows the real shape.

### LightGBM

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| active_days_l90d | frequency | 1.1727 | -0.9882 | higher value, lower churn risk |
| days_since_last_bet | recency | 0.5809 | 0.9546 | higher value, higher churn risk |
| wagered_eur_l7d | monetary | 0.4052 | -0.9113 | higher value, lower churn risk |
| days_since_last_return | tenure | 0.2472 | 0.6066 | higher value, higher churn risk |
| engagement_score | mix | 0.1901 | -0.8352 | higher value, lower churn risk |
| brandId | other | 0.0000 |  | categorical (brand) |

![Global SHAP, LightGBM](../data/03_output/feature_importance/brand64_churn90d/shap_global_lightgbm.png)

Dependence plots for the top 6 features (all the features the model uses). The x axis is in real units (days, EUR, counts), not on the sign-log scale. Each dot is one validation player.

![SHAP dependence, LightGBM](../data/03_output/feature_importance/brand64_churn90d/shap_dependence_lightgbm.png)

### Cox PH

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| active_days_l90d | frequency | 0.6040 | -1.0000 | higher value, lower churn risk |
| days_since_last_return | tenure | 0.1487 | 1.0000 | higher value, higher churn risk |
| wagered_eur_l7d | monetary | 0.1173 | -1.0000 | higher value, lower churn risk |
| engagement_score | mix | 0.0970 | -1.0000 | higher value, lower churn risk |
| days_since_last_bet | recency | 0.0565 | 1.0000 | higher value, higher churn risk |

![Global SHAP, Cox PH](../data/03_output/feature_importance/brand64_churn90d/shap_global_cox.png)

A dependence plot for a linear model is a straight line with slope `beta`, so I do not draw it. The hazard ratios are in `hazard_ratios.csv` in the Cox MLflow run.

## 3. Importance by Family

How much each group of features matters, with two measures side by side. **SHAP share**: the family's part of the model's total mean |SHAP| on the validation rows, so how much it weighs in the predictions. **Score lost**: what the model loses on the test months when the whole family is removed and the model is retrained, so what it adds that the other families cannot replace. A family can weigh a lot and still lose little when removed, if other families carry the same information.

| family | lightgbm_shap_share | cox_shap_share | lightgbm_auc_lost | cox_c_index_lost |
|---|---|---|---|---|
| frequency | 0.4517 | 0.5902 | 0.0172 | 0.0521 |
| recency | 0.2238 | 0.0552 | 0.0053 | 0.0007 |
| monetary | 0.1561 | 0.1146 | 0.0009 | 0.0011 |
| tenure | 0.0952 | 0.1453 | 0.0049 | 0.0024 |
| mix | 0.0732 | 0.0948 | 0.0018 | 0.0001 |

![Importance by family](../data/03_output/feature_importance/brand64_churn90d/importance_by_family.png)

## 4. Ablation by Family

I retrain the model without every feature of one family and compare it with the full model (same features otherwise, same parameters, same rows). A negative delta means the family helps. A family with no feature in the model is left empty.

Full model: LightGBM AUC valid 0.8706, test 0.8805. Cox c-index valid 0.8062, test 0.8428.

| family | LightGBM features | LightGBM AUC delta, valid | LightGBM AUC delta, test | Cox features | Cox c-index delta, valid | Cox c-index delta, test |
|---|---|---|---|---|---|---|
| recency | days_since_last_bet | -0.0067 | -0.0053 | days_since_last_bet | -0.0008 | -0.0007 |
| frequency | active_days_l90d | -0.0183 | -0.0172 | active_days_l90d | -0.0462 | -0.0521 |
| monetary | wagered_eur_l7d | -0.0001 | -0.0009 | wagered_eur_l7d | -0.0007 | -0.0011 |
| lag |  |  |  |  |  |  |
| trend |  |  |  |  |  |  |
| tenure | days_since_last_return | -0.0057 | -0.0049 | days_since_last_return | -0.0034 | -0.0024 |
| mix | engagement_score | -0.0013 | -0.0018 | engagement_score | 0.0005 | -0.0001 |
| deposit |  |  |  |  |  |  |
| segment |  |  |  |  |  |  |

The most valuable family on the test months is **frequency** for LightGBM (AUC -0.0172 without it) and **frequency** for Cox (c-index -0.0521 without it).

## 5. Known Limits

- **The Cox test months only have churn on days 0 to 37.** A churn day can only be confirmed when the 60 days after it are in the data, so on the test months the c-index mostly measures "who is already gone", not "which day".
- **The intervals only cover the shuffle.** A different validation sample would move the numbers more than the intervals show.
- **Correlated features share credit.** Stage 2 already removed pairs over 0.95, but features from the same family (for example `active_days_l30d` and `active_days_l90d`) can still hide each other in permutation importance. Ablation by family shows the joint value.

Generated by `src/models/feature_importance.py` (`make importance`). The figures and tables are in `data/03_output/feature_importance/brand64_churn90d/` and in the `feature_importance/` folder of both MLflow runs.
