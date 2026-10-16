# Feature Importance v0 (brandId 72)

This document reports three importance measures together for the two baseline models. Gain-based importance alone overstates features with many distinct values and features that are correlated with others, so I do not use it here.

| Measure | Question it answers |
|---|---|
| Permutation importance | How much do the model's AUC and calibration fall when this feature is shuffled? |
| SHAP values | What does this feature do to each player's score, and in which direction? |
| Ablation by family | What is a whole family of features worth to the model? |

## 0. Models and Data

| | LightGBM | Cox PH |
|---|---|---|
| MLflow run | `lightgbm_classifier_brand72_1792152939` | `cox_ph_brand72_1792152965` |
| Target | `event_60d` (churn in the next 60 days) | `duration_days` + `event_observed` (churn day) |
| Features (after Stage 2 selection) | 8 | 7 |
| Test score (test months) | AUC 0.8915 | c-index 0.8659 |

- Dataset: `data/processed/churn90d/train_dataset_72_2025-08-01_2025-09-01_2025-10-01_2025-11-01_2025-12-01_2026-01-01_2026-02-01_2026-03-01_2026-04-01_2026-05-01_2026-06-01_2026-07-01_1792152794.parquet`.
- Rows: train 112,600, validation 10,530, test 19,025. Validation month: 2026-05-01. Test months: 2026-06-01, 2026-07-01.
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
| active_days_l90d | frequency | 0.0920 [0.0901, 0.0939] | 0.0509 [0.0499, 0.0519] | 0.0352 [0.0332, 0.0372] |
| days_since_last_bet | recency | 0.0367 [0.0352, 0.0381] | 0.0219 [0.0211, 0.0227] | 0.0232 [0.0213, 0.0250] |
| prior_dormancy_spells_14d | tenure | 0.0183 [0.0172, 0.0194] | 0.0112 [0.0106, 0.0118] | 0.0140 [0.0121, 0.0159] |
| wagered_eur_l7d | monetary | 0.0086 [0.0082, 0.0091] | 0.0056 [0.0052, 0.0059] | 0.0124 [0.0116, 0.0132] |
| bets_l30d | frequency | 0.0069 [0.0066, 0.0072] | 0.0038 [0.0037, 0.0040] | 0.0123 [0.0117, 0.0129] |
| losing_streak | mix | 0.0066 [0.0061, 0.0071] | 0.0045 [0.0043, 0.0048] | 0.0197 [0.0190, 0.0204] |
| days_since_last_return | tenure | 0.0066 [0.0063, 0.0068] | 0.0041 [0.0039, 0.0042] | 0.0117 [0.0107, 0.0127] |
| brandId | other | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

The delivery plan keeps a feature in the served model only if its permutation-importance interval excludes zero. Every LightGBM feature passes this rule.

![Permutation importance, LightGBM](../data/03_output/feature_importance/brand72_churn90d/permutation_lightgbm.png)

### Cox PH

The Cox model gives a risk ranking, not a probability, so only the c-index is measured.

| feature | family | c-index drop [95% CI] |
|---|---|---|
| active_days_l90d | frequency | 0.1329 [0.1310, 0.1348] |
| prior_dormancy_spells_14d | tenure | 0.0559 [0.0546, 0.0572] |
| days_since_last_bet | recency | 0.0146 [0.0141, 0.0151] |
| bets_l30d | frequency | 0.0080 [0.0075, 0.0084] |
| wagered_eur_l7d | monetary | 0.0048 [0.0046, 0.0050] |
| days_since_last_return | tenure | 0.0018 [0.0016, 0.0020] |
| losing_streak | mix | -0.0000 [-0.0001, 0.0000] |

![Permutation importance, Cox PH](../data/03_output/feature_importance/brand72_churn90d/permutation_cox.png)

## 2. SHAP Values

SHAP splits each player's score into one part per feature. The parts add up to the score.

- **LightGBM**: exact TreeSHAP values from LightGBM itself (`pred_contrib=True`, the same algorithm as `shap.TreeExplainer`), in log-odds of churn.
- **Cox PH**: the model is linear, so the exact SHAP value is `beta * (x - training mean)`, in log-hazard. Positive means a higher risk of churning sooner.
- **Direction** is the sign of the rank correlation (`rank_corr`) between the feature value and its SHAP value. Under 0.5 in absolute value one direction would mislead, so it says "non-monotonic" and the dependence plot shows the real shape.

### LightGBM

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| active_days_l90d | frequency | 1.0276 | -0.9774 | higher value, lower churn risk |
| days_since_last_bet | recency | 0.5450 | 0.9717 | higher value, higher churn risk |
| wagered_eur_l7d | monetary | 0.3123 | -0.9092 | higher value, lower churn risk |
| prior_dormancy_spells_14d | tenure | 0.2594 | -0.9126 | higher value, lower churn risk |
| bets_l30d | frequency | 0.2359 | -0.9194 | higher value, lower churn risk |
| days_since_last_return | tenure | 0.1504 | 0.6625 | higher value, higher churn risk |
| losing_streak | mix | 0.1149 | -0.4317 | non-monotonic, see the plot |
| brandId | other | 0.0000 |  | categorical (brand) |

![Global SHAP, LightGBM](../data/03_output/feature_importance/brand72_churn90d/shap_global_lightgbm.png)

Dependence plots for the top 8 features (all the features the model uses). The x axis is in real units (days, EUR, counts), not on the sign-log scale. Each dot is one validation player.

![SHAP dependence, LightGBM](../data/03_output/feature_importance/brand72_churn90d/shap_dependence_lightgbm.png)

### Cox PH

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| active_days_l90d | frequency | 0.4737 | -1.0000 | higher value, lower churn risk |
| prior_dormancy_spells_14d | tenure | 0.3249 | -1.0000 | higher value, lower churn risk |
| days_since_last_bet | recency | 0.1418 | 1.0000 | higher value, higher churn risk |
| bets_l30d | frequency | 0.1265 | -1.0000 | higher value, lower churn risk |
| wagered_eur_l7d | monetary | 0.0902 | -1.0000 | higher value, lower churn risk |
| days_since_last_return | tenure | 0.0625 | -1.0000 | higher value, lower churn risk |
| losing_streak | mix | 0.0088 | -1.0000 | higher value, lower churn risk |

![Global SHAP, Cox PH](../data/03_output/feature_importance/brand72_churn90d/shap_global_cox.png)

A dependence plot for a linear model is a straight line with slope `beta`, so I do not draw it. The hazard ratios are in `hazard_ratios.csv` in the Cox MLflow run.

## 3. Importance by Family

How much each group of features matters, with two measures side by side. **SHAP share**: the family's part of the model's total mean |SHAP| on the validation rows, so how much it weighs in the predictions. **Score lost**: what the model loses on the test months when the whole family is removed and the model is retrained, so what it adds that the other families cannot replace. A family can weigh a lot and still lose little when removed, if other families carry the same information.

| family | lightgbm_shap_share | cox_shap_share | lightgbm_auc_lost | cox_c_index_lost |
|---|---|---|---|---|
| frequency | 0.4776 | 0.4886 | 0.0155 | 0.0317 |
| recency | 0.2060 | 0.1154 | 0.0050 | 0.0026 |
| tenure | 0.1549 | 0.3153 | 0.0141 | 0.0160 |
| monetary | 0.1180 | 0.0735 | 0.0002 | 0.0007 |
| mix | 0.0434 | 0.0072 | 0.0009 | -0.0001 |

![Importance by family](../data/03_output/feature_importance/brand72_churn90d/importance_by_family.png)

## 4. Ablation by Family

I retrain the model without every feature of one family and compare it with the full model (same features otherwise, same parameters, same rows). A negative delta means the family helps. A family with no feature in the model is left empty.

Full model: LightGBM AUC valid 0.8875, test 0.8915. Cox c-index valid 0.8441, test 0.8659.

| family | LightGBM features | LightGBM AUC delta, valid | LightGBM AUC delta, test | Cox features | Cox c-index delta, valid | Cox c-index delta, test |
|---|---|---|---|---|---|---|
| recency | days_since_last_bet | -0.0052 | -0.0050 | days_since_last_bet | -0.0030 | -0.0026 |
| frequency | active_days_l90d, bets_l30d | -0.0105 | -0.0155 | active_days_l90d, bets_l30d | -0.0282 | -0.0317 |
| monetary | wagered_eur_l7d | -0.0006 | -0.0002 | wagered_eur_l7d | -0.0004 | -0.0007 |
| lag |  |  |  |  |  |  |
| trend |  |  |  |  |  |  |
| tenure | prior_dormancy_spells_14d, days_since_last_return | -0.0125 | -0.0141 | prior_dormancy_spells_14d, days_since_last_return | -0.0136 | -0.0160 |
| mix | losing_streak | -0.0015 | -0.0009 | losing_streak | -0.0000 | 0.0001 |
| deposit |  |  |  |  |  |  |
| segment |  |  |  |  |  |  |

The most valuable family on the test months is **frequency** for LightGBM (AUC -0.0155 without it) and **frequency** for Cox (c-index -0.0317 without it).

## 5. Known Limits

- **The Cox test months only have churn on days 0 to 37.** A churn day can only be confirmed when the 60 days after it are in the data, so on the test months the c-index mostly measures "who is already gone", not "which day".
- **The intervals only cover the shuffle.** A different validation sample would move the numbers more than the intervals show.
- **Correlated features share credit.** Stage 2 already removed pairs over 0.95, but features from the same family (for example `active_days_l30d` and `active_days_l90d`) can still hide each other in permutation importance. Ablation by family shows the joint value.

Generated by `src/models/feature_importance.py` (`make importance`). The figures and tables are in `data/03_output/feature_importance/brand72_churn90d/` and in the `feature_importance/` folder of both MLflow runs.
