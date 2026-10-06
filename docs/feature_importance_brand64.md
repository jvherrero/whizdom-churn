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
| MLflow run | `lightgbm_classifier_brand64_1791273593` | `cox_ph_brand64_1791273654` |
| Target | `event_60d` (churn in the next 60 days) | `duration_days` + `event_observed` (churn day) |
| Features (after Stage 2 selection) | 8 | 7 |
| Test score (last cutoff) | AUC 0.8738 | c-index 0.8651 |

- Dataset: `data/processed/train_dataset_64_2026-07-18_2026-07-27_2026-08-05_1791191780.parquet`.
- Rows: train 22,634, validation 4,007, test 14,687. The test month is 2026-08-05.
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
| n_active_days_last_30d | frequency | 0.0760 [0.0732, 0.0788] | 0.0367 [0.0354, 0.0381] | 0.0135 [0.0113, 0.0157] |
| stake_last_30d | monetary | 0.0364 [0.0336, 0.0392] | 0.0236 [0.0221, 0.0252] | 0.0416 [0.0372, 0.0459] |
| tenure_days_missing | tenure | 0.0217 [0.0203, 0.0231] | 0.0123 [0.0116, 0.0131] | 0.0104 [0.0085, 0.0124] |
| days_since_last_deposit | recency | 0.0149 [0.0134, 0.0164] | 0.0076 [0.0067, 0.0085] | -0.0043 [-0.0060, -0.0027] |
| max_prior_gap_days | tenure | 0.0145 [0.0129, 0.0160] | 0.0086 [0.0079, 0.0094] | 0.0043 [0.0022, 0.0065] |
| n_bonus_last_7d | frequency | 0.0133 [0.0121, 0.0144] | 0.0074 [0.0068, 0.0081] | 0.0098 [0.0080, 0.0116] |
| bonus_amount_trend_7 | lag_trend | 0.0081 [0.0072, 0.0090] | 0.0039 [0.0034, 0.0043] | 0.0010 [-0.0013, 0.0032] |
| brandId | other | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

![Permutation importance, LightGBM](../data/03_output/feature_importance/brand64/permutation_lightgbm.png)

### Cox PH

The Cox model gives a risk ranking, not a probability, so only the c-index is measured.

| feature | family | c-index drop [95% CI] |
|---|---|---|
| n_active_days_last_30d | frequency | 0.1114 [0.1069, 0.1158] |
| tenure_days_missing | tenure | 0.0142 [0.0130, 0.0154] |
| stake_last_30d | monetary | 0.0141 [0.0127, 0.0156] |
| max_prior_gap_days | tenure | 0.0113 [0.0100, 0.0126] |
| days_since_last_deposit | recency | 0.0065 [0.0052, 0.0078] |
| n_bonus_last_7d | frequency | 0.0049 [0.0042, 0.0056] |
| bonus_amount_trend_7 | lag_trend | 0.0009 [0.0005, 0.0012] |

![Permutation importance, Cox PH](../data/03_output/feature_importance/brand64/permutation_cox.png)

## 2. SHAP Values

SHAP splits each player's score into one part per feature. The parts add up to the score.

- **LightGBM**: exact TreeSHAP values from LightGBM itself (`pred_contrib=True`, the same algorithm as `shap.TreeExplainer`), in log-odds of churn.
- **Cox PH**: the model is linear, so the exact SHAP value is `beta * (x - training mean)`, in log-hazard. Positive means a higher risk of churning sooner.
- **Direction** is the sign of the rank correlation (`rank_corr`) between the feature value and its SHAP value. Under 0.5 in absolute value one direction would mislead, so it says "non-monotonic" and the dependence plot shows the real shape.

### LightGBM

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| n_active_days_last_30d | frequency | 1.1166 | -0.9648 | higher value, lower churn risk |
| days_since_last_deposit | recency | 0.4053 | 0.7697 | higher value, higher churn risk |
| max_prior_gap_days | tenure | 0.3456 | -0.8303 | higher value, lower churn risk |
| stake_last_30d | monetary | 0.3309 | -0.8804 | higher value, lower churn risk |
| tenure_days_missing | tenure | 0.3249 | -0.7884 | higher value, lower churn risk |
| n_bonus_last_7d | frequency | 0.2744 | -0.9336 | higher value, lower churn risk |
| bonus_amount_trend_7 | lag_trend | 0.1193 | 0.1139 | non-monotonic, see the plot |
| brandId | other | 0.0000 |  | categorical (brand) |

![Global SHAP, LightGBM](../data/03_output/feature_importance/brand64/shap_global_lightgbm.png)

Dependence plots for the top 8 features (all the features the model uses). The x axis is in real units (days, EUR, counts), not on the sign-log scale. Each dot is one validation player.

![SHAP dependence, LightGBM](../data/03_output/feature_importance/brand64/shap_dependence_lightgbm.png)

### Cox PH

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| n_active_days_last_30d | frequency | 0.6307 | -1.0000 | higher value, lower churn risk |
| stake_last_30d | monetary | 0.2232 | -1.0000 | higher value, lower churn risk |
| tenure_days_missing | tenure | 0.2106 | -1.0000 | higher value, lower churn risk |
| max_prior_gap_days | tenure | 0.2038 | -1.0000 | higher value, lower churn risk |
| days_since_last_deposit | recency | 0.1583 | 1.0000 | higher value, higher churn risk |
| n_bonus_last_7d | frequency | 0.1421 | -1.0000 | higher value, lower churn risk |
| bonus_amount_trend_7 | lag_trend | 0.0184 | 1.0000 | higher value, higher churn risk |

![Global SHAP, Cox PH](../data/03_output/feature_importance/brand64/shap_global_cox.png)

A dependence plot for a linear model is a straight line with slope `beta`, so I do not draw it. The hazard ratios are in `hazard_ratios.csv` in the Cox MLflow run.

## 3. Importance by Family

How much each group of features matters, with two measures side by side. **SHAP share**: the family's part of the model's total mean |SHAP| on the validation rows, so how much it weighs in the predictions. **Score lost**: what the model loses on the test month when the whole family is removed and the model is retrained, so what it adds that the other families cannot replace. A family can weigh a lot and still lose little when removed, if other families carry the same information.

| family | lightgbm_shap_share | cox_shap_share | lightgbm_auc_lost | cox_c_index_lost |
|---|---|---|---|---|
| frequency | 0.4769 | 0.4869 | 0.0035 | 0.0072 |
| tenure | 0.2299 | 0.2611 | 0.0198 | 0.0184 |
| recency | 0.1389 | 0.0997 | 0.0032 | 0.0022 |
| monetary | 0.1135 | 0.1406 | 0.0057 | 0.0025 |
| lag_trend | 0.0409 | 0.0116 | 0.0004 | -0.0002 |

![Importance by family](../data/03_output/feature_importance/brand64/importance_by_family.png)

## 4. Ablation by Family

I retrain the model without every feature of one family and compare it with the full model (same features otherwise, same parameters, same rows). A negative delta means the family helps. A family with no feature in the model is left empty.

Full model: LightGBM AUC valid 0.8822, test 0.8738. Cox c-index valid 0.8518, test 0.8651.

| family | LightGBM features | LightGBM AUC delta, valid | LightGBM AUC delta, test | Cox features | Cox c-index delta, valid | Cox c-index delta, test |
|---|---|---|---|---|---|---|
| recency | days_since_last_deposit | -0.0043 | -0.0032 | days_since_last_deposit | -0.0010 | -0.0022 |
| frequency | n_active_days_last_30d, n_bonus_last_7d | -0.0093 | -0.0035 | n_active_days_last_30d, n_bonus_last_7d | -0.0107 | -0.0072 |
| monetary | stake_last_30d | -0.0036 | -0.0057 | stake_last_30d | -0.0039 | -0.0025 |
| lag_trend | bonus_amount_trend_7 | -0.0053 | -0.0004 | bonus_amount_trend_7 | -0.0001 | 0.0002 |
| tenure | tenure_days_missing, max_prior_gap_days | -0.0117 | -0.0198 | tenure_days_missing, max_prior_gap_days | -0.0106 | -0.0184 |
| mix |  |  |  |  |  |  |
| segment |  |  |  |  |  |  |

The most valuable family on the test month is **tenure** for LightGBM (AUC -0.0198 without it) and **tenure** for Cox (c-index -0.0184 without it).

## 5. Known Limits

- **The Cox test month only has churn on day 0.** With data through 2026-10-04, a churn day after the cutoff can only be confirmed up to 60 days before the data ends, which is day 0 for the last cutoff. So on the test month the c-index measures "who is already gone", not "which day".
- **The intervals only cover the shuffle.** A different validation sample would move the numbers more than the intervals show.
- **Correlated features share credit.** Stage 2 already removed pairs over 0.95, but features from the same family (for example `n_active_days_last_30d` and `max_prior_gap_days`) can still hide each other in permutation importance. Ablation by family shows the joint value.

Generated by `src/models/feature_importance.py` (`make importance`). The figures and tables are in `data/03_output/feature_importance/brand64/` and in the `feature_importance/` folder of both MLflow runs.
