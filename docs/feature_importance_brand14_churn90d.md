# Feature Importance v0 (brandId 14)

This document reports three importance measures together for the two baseline models. Gain-based importance alone overstates features with many distinct values and features that are correlated with others, so I do not use it here.

| Measure | Question it answers |
|---|---|
| Permutation importance | How much do the model's AUC and calibration fall when this feature is shuffled? |
| SHAP values | What does this feature do to each player's score, and in which direction? |
| Ablation by family | What is a whole family of features worth to the model? |

## 0. Models and Data

| | LightGBM | Cox PH |
|---|---|---|
| MLflow run | `lightgbm_classifier_brand14_1792151817` | `cox_ph_brand14_1792151834` |
| Target | `event_60d` (churn in the next 60 days) | `duration_days` + `event_observed` (churn day) |
| Features (after Stage 2 selection) | 4 | 3 |
| Test score (test months) | AUC 0.8923 | c-index 0.8723 |

- Dataset: `data/processed/churn90d/train_dataset_14_2025-08-01_2025-09-01_2025-10-01_2025-11-01_2025-12-01_2026-01-01_2026-02-01_2026-03-01_2026-04-01_2026-05-01_2026-06-01_2026-07-01_1792151724.parquet`.
- Rows: train 60,091, validation 6,843, test 13,820. Validation month: 2026-05-01. Test months: 2026-06-01, 2026-07-01.
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
| active_days_l90d | frequency | 0.2281 [0.2254, 0.2309] | 0.0952 [0.0942, 0.0963] | 0.1218 [0.1195, 0.1241] |
| days_since_last_bet | recency | 0.0587 [0.0565, 0.0609] | 0.0324 [0.0312, 0.0336] | 0.0565 [0.0530, 0.0599] |
| prior_dormancy_spells_14d | tenure | 0.0238 [0.0225, 0.0251] | 0.0152 [0.0144, 0.0159] | 0.0293 [0.0271, 0.0314] |
| brandId | other | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

The delivery plan keeps a feature in the served model only if its permutation-importance interval excludes zero. Every LightGBM feature passes this rule.

![Permutation importance, LightGBM](../data/03_output/feature_importance/brand14_churn90d/permutation_lightgbm.png)

### Cox PH

The Cox model gives a risk ranking, not a probability, so only the c-index is measured.

| feature | family | c-index drop [95% CI] |
|---|---|---|
| active_days_l90d | frequency | 0.2486 [0.2450, 0.2522] |
| days_since_last_bet | recency | 0.0229 [0.0212, 0.0247] |
| prior_dormancy_spells_14d | tenure | 0.0138 [0.0129, 0.0147] |

![Permutation importance, Cox PH](../data/03_output/feature_importance/brand14_churn90d/permutation_cox.png)

## 2. SHAP Values

SHAP splits each player's score into one part per feature. The parts add up to the score.

- **LightGBM**: exact TreeSHAP values from LightGBM itself (`pred_contrib=True`, the same algorithm as `shap.TreeExplainer`), in log-odds of churn.
- **Cox PH**: the model is linear, so the exact SHAP value is `beta * (x - training mean)`, in log-hazard. Positive means a higher risk of churning sooner.
- **Direction** is the sign of the rank correlation (`rank_corr`) between the feature value and its SHAP value. Under 0.5 in absolute value one direction would mislead, so it says "non-monotonic" and the dependence plot shows the real shape.

### LightGBM

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| active_days_l90d | frequency | 0.8689 | -0.9731 | higher value, lower churn risk |
| days_since_last_bet | recency | 0.3643 | 0.9508 | higher value, higher churn risk |
| prior_dormancy_spells_14d | tenure | 0.1089 | -0.8071 | higher value, lower churn risk |
| brandId | other | 0.0000 |  | categorical (brand) |

![Global SHAP, LightGBM](../data/03_output/feature_importance/brand14_churn90d/shap_global_lightgbm.png)

Dependence plots for the top 4 features (all the features the model uses). The x axis is in real units (days, EUR, counts), not on the sign-log scale. Each dot is one validation player.

![SHAP dependence, LightGBM](../data/03_output/feature_importance/brand14_churn90d/shap_dependence_lightgbm.png)

### Cox PH

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| active_days_l90d | frequency | 0.7095 | -1.0000 | higher value, lower churn risk |
| days_since_last_bet | recency | 0.2246 | 1.0000 | higher value, higher churn risk |
| prior_dormancy_spells_14d | tenure | 0.1590 | -1.0000 | higher value, lower churn risk |

![Global SHAP, Cox PH](../data/03_output/feature_importance/brand14_churn90d/shap_global_cox.png)

A dependence plot for a linear model is a straight line with slope `beta`, so I do not draw it. The hazard ratios are in `hazard_ratios.csv` in the Cox MLflow run.

## 3. Importance by Family

How much each group of features matters, with two measures side by side. **SHAP share**: the family's part of the model's total mean |SHAP| on the validation rows, so how much it weighs in the predictions. **Score lost**: what the model loses on the test months when the whole family is removed and the model is retrained, so what it adds that the other families cannot replace. A family can weigh a lot and still lose little when removed, if other families carry the same information.

| family | lightgbm_shap_share | cox_shap_share | lightgbm_auc_lost | cox_c_index_lost |
|---|---|---|---|---|
| frequency | 0.6474 | 0.6491 | 0.0538 | 0.0704 |
| recency | 0.2715 | 0.2054 | 0.0185 | 0.0123 |
| tenure | 0.0812 | 0.1454 | 0.0123 | 0.0107 |

![Importance by family](../data/03_output/feature_importance/brand14_churn90d/importance_by_family.png)

## 4. Ablation by Family

I retrain the model without every feature of one family and compare it with the full model (same features otherwise, same parameters, same rows). A negative delta means the family helps. A family with no feature in the model is left empty.

Full model: LightGBM AUC valid 0.8834, test 0.8923. Cox c-index valid 0.8406, test 0.8723.

| family | LightGBM features | LightGBM AUC delta, valid | LightGBM AUC delta, test | Cox features | Cox c-index delta, valid | Cox c-index delta, test |
|---|---|---|---|---|---|---|
| recency | days_since_last_bet | -0.0197 | -0.0185 | days_since_last_bet | -0.0102 | -0.0123 |
| frequency | active_days_l90d | -0.0661 | -0.0538 | active_days_l90d | -0.0856 | -0.0704 |
| monetary |  |  |  |  |  |  |
| lag |  |  |  |  |  |  |
| trend |  |  |  |  |  |  |
| tenure | prior_dormancy_spells_14d | -0.0068 | -0.0123 | prior_dormancy_spells_14d | -0.0067 | -0.0107 |
| mix |  |  |  |  |  |  |
| deposit |  |  |  |  |  |  |
| segment |  |  |  |  |  |  |

The most valuable family on the test months is **frequency** for LightGBM (AUC -0.0538 without it) and **frequency** for Cox (c-index -0.0704 without it).

## 5. Known Limits

- **The Cox test months only have churn on days 0 to 37.** A churn day can only be confirmed when the 60 days after it are in the data, so on the test months the c-index mostly measures "who is already gone", not "which day".
- **The intervals only cover the shuffle.** A different validation sample would move the numbers more than the intervals show.
- **Correlated features share credit.** Stage 2 already removed pairs over 0.95, but features from the same family (for example `active_days_l30d` and `active_days_l90d`) can still hide each other in permutation importance. Ablation by family shows the joint value.

Generated by `src/models/feature_importance.py` (`make importance`). The figures and tables are in `data/03_output/feature_importance/brand14_churn90d/` and in the `feature_importance/` folder of both MLflow runs.
