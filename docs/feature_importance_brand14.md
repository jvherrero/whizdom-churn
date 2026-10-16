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
| MLflow run | `lightgbm_classifier_brand14_1792127565` | `cox_ph_brand14_1792127579` |
| Target | `event_60d` (churn in the next 60 days) | `duration_days` + `event_observed` (churn day) |
| Features (after Stage 2 selection) | 5 | 4 |
| Test score (test months) | AUC 0.8958 | c-index 0.8659 |

- Dataset: `data/processed/train_dataset_14_2025-09-01_2025-10-01_2025-11-01_2025-12-01_2026-01-01_2026-02-01_2026-03-01_2026-04-01_2026-05-01_2026-06-01_2026-07-01_2026-08-01_1792127495.parquet`.
- Rows: train 59,779, validation 7,158, test 13,169. Validation month: 2026-06-01. Test months: 2026-07-01, 2026-08-01.
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
| active_days_l90d | frequency | 0.1299 [0.1273, 0.1325] | 0.0667 [0.0653, 0.0680] | 0.0549 [0.0516, 0.0581] |
| days_since_first_bet | tenure | 0.0248 [0.0237, 0.0259] | 0.0159 [0.0152, 0.0165] | 0.0222 [0.0210, 0.0234] |
| wagered_eur_l7d | monetary | 0.0222 [0.0213, 0.0232] | 0.0124 [0.0118, 0.0129] | 0.0259 [0.0241, 0.0276] |
| days_since_last_bet | recency | 0.0182 [0.0174, 0.0190] | 0.0106 [0.0100, 0.0112] | 0.0107 [0.0090, 0.0125] |
| brandId | other | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

The delivery plan keeps a feature in the served model only if its permutation-importance interval excludes zero. Every LightGBM feature passes this rule.

![Permutation importance, LightGBM](../data/03_output/feature_importance/brand14/permutation_lightgbm.png)

### Cox PH

The Cox model gives a risk ranking, not a probability, so only the c-index is measured.

| feature | family | c-index drop [95% CI] |
|---|---|---|
| active_days_l90d | frequency | 0.1711 [0.1682, 0.1740] |
| days_since_first_bet | tenure | 0.0181 [0.0172, 0.0190] |
| wagered_eur_l7d | monetary | 0.0159 [0.0152, 0.0165] |
| days_since_last_bet | recency | 0.0073 [0.0069, 0.0077] |

![Permutation importance, Cox PH](../data/03_output/feature_importance/brand14/permutation_cox.png)

## 2. SHAP Values

SHAP splits each player's score into one part per feature. The parts add up to the score.

- **LightGBM**: exact TreeSHAP values from LightGBM itself (`pred_contrib=True`, the same algorithm as `shap.TreeExplainer`), in log-odds of churn.
- **Cox PH**: the model is linear, so the exact SHAP value is `beta * (x - training mean)`, in log-hazard. Positive means a higher risk of churning sooner.
- **Direction** is the sign of the rank correlation (`rank_corr`) between the feature value and its SHAP value. Under 0.5 in absolute value one direction would mislead, so it says "non-monotonic" and the dependence plot shows the real shape.

### LightGBM

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| active_days_l90d | frequency | 1.2432 | -0.9846 | higher value, lower churn risk |
| wagered_eur_l7d | monetary | 0.6058 | -0.8636 | higher value, lower churn risk |
| days_since_last_bet | recency | 0.3010 | 0.9413 | higher value, higher churn risk |
| days_since_first_bet | tenure | 0.2875 | -0.7686 | higher value, lower churn risk |
| brandId | other | 0.0000 |  | categorical (brand) |

![Global SHAP, LightGBM](../data/03_output/feature_importance/brand14/shap_global_lightgbm.png)

Dependence plots for the top 5 features (all the features the model uses). The x axis is in real units (days, EUR, counts), not on the sign-log scale. Each dot is one validation player.

![SHAP dependence, LightGBM](../data/03_output/feature_importance/brand14/shap_dependence_lightgbm.png)

### Cox PH

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| active_days_l90d | frequency | 0.5922 | -1.0000 | higher value, lower churn risk |
| wagered_eur_l7d | monetary | 0.2248 | -1.0000 | higher value, lower churn risk |
| days_since_first_bet | tenure | 0.1955 | -1.0000 | higher value, lower churn risk |
| days_since_last_bet | recency | 0.1326 | 1.0000 | higher value, higher churn risk |

![Global SHAP, Cox PH](../data/03_output/feature_importance/brand14/shap_global_cox.png)

A dependence plot for a linear model is a straight line with slope `beta`, so I do not draw it. The hazard ratios are in `hazard_ratios.csv` in the Cox MLflow run.

## 3. Importance by Family

How much each group of features matters, with two measures side by side. **SHAP share**: the family's part of the model's total mean |SHAP| on the validation rows, so how much it weighs in the predictions. **Score lost**: what the model loses on the test months when the whole family is removed and the model is retrained, so what it adds that the other families cannot replace. A family can weigh a lot and still lose little when removed, if other families carry the same information.

| family | lightgbm_shap_share | cox_shap_share | lightgbm_auc_lost | cox_c_index_lost |
|---|---|---|---|---|
| frequency | 0.5100 | 0.5171 | 0.0293 | 0.0306 |
| monetary | 0.2485 | 0.1963 | 0.0015 | 0.0006 |
| recency | 0.1235 | 0.1158 | 0.0041 | 0.0023 |
| tenure | 0.1179 | 0.1708 | 0.0106 | 0.0070 |

![Importance by family](../data/03_output/feature_importance/brand14/importance_by_family.png)

## 4. Ablation by Family

I retrain the model without every feature of one family and compare it with the full model (same features otherwise, same parameters, same rows). A negative delta means the family helps. A family with no feature in the model is left empty.

Full model: LightGBM AUC valid 0.8933, test 0.8958. Cox c-index valid 0.8512, test 0.8659.

| family | LightGBM features | LightGBM AUC delta, valid | LightGBM AUC delta, test | Cox features | Cox c-index delta, valid | Cox c-index delta, test |
|---|---|---|---|---|---|---|
| recency | days_since_last_bet | -0.0037 | -0.0041 | days_since_last_bet | -0.0028 | -0.0023 |
| frequency | active_days_l90d | -0.0212 | -0.0293 | active_days_l90d | -0.0264 | -0.0306 |
| monetary | wagered_eur_l7d | -0.0023 | -0.0015 | wagered_eur_l7d | 0.0012 | -0.0006 |
| lag |  |  |  |  |  |  |
| trend |  |  |  |  |  |  |
| tenure | days_since_first_bet | -0.0090 | -0.0106 | days_since_first_bet | -0.0051 | -0.0070 |
| mix |  |  |  |  |  |  |
| deposit |  |  |  |  |  |  |
| segment |  |  |  |  |  |  |

The most valuable family on the test months is **frequency** for LightGBM (AUC -0.0293 without it) and **frequency** for Cox (c-index -0.0306 without it).

## 5. Known Limits

- **The Cox test months only have churn on days 0 to 37.** A churn day can only be confirmed when the 60 days after it are in the data, so on the test months the c-index mostly measures "who is already gone", not "which day".
- **The intervals only cover the shuffle.** A different validation sample would move the numbers more than the intervals show.
- **Correlated features share credit.** Stage 2 already removed pairs over 0.95, but features from the same family (for example `active_days_l30d` and `active_days_l90d`) can still hide each other in permutation importance. Ablation by family shows the joint value.

Generated by `src/models/feature_importance.py` (`make importance`). The figures and tables are in `data/03_output/feature_importance/brand14/` and in the `feature_importance/` folder of both MLflow runs.
