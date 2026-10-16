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
| MLflow run | `lightgbm_classifier_brand72_1792128858` | `cox_ph_brand72_1792128882` |
| Target | `event_60d` (churn in the next 60 days) | `duration_days` + `event_observed` (churn day) |
| Features (after Stage 2 selection) | 7 | 6 |
| Test score (test months) | AUC 0.8783 | c-index 0.8523 |

- Dataset: `data/processed/train_dataset_72_2025-09-01_2025-10-01_2025-11-01_2025-12-01_2026-01-01_2026-02-01_2026-03-01_2026-04-01_2026-05-01_2026-06-01_2026-07-01_2026-08-01_1792128721.parquet`.
- Rows: train 108,937, validation 10,036, test 19,296. Validation month: 2026-06-01. Test months: 2026-07-01, 2026-08-01.
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
| active_days_l90d | frequency | 0.0801 [0.0783, 0.0819] | 0.0478 [0.0468, 0.0488] | 0.0323 [0.0314, 0.0333] |
| prior_dormancy_spells_14d | tenure | 0.0314 [0.0305, 0.0323] | 0.0185 [0.0180, 0.0189] | 0.0313 [0.0294, 0.0332] |
| days_since_last_bet | recency | 0.0242 [0.0233, 0.0251] | 0.0151 [0.0146, 0.0157] | 0.0157 [0.0139, 0.0175] |
| wagered_eur_l7d | monetary | 0.0187 [0.0181, 0.0194] | 0.0121 [0.0117, 0.0125] | 0.0124 [0.0114, 0.0134] |
| prior_wagered_eur_l7d | lag | 0.0071 [0.0067, 0.0075] | 0.0051 [0.0048, 0.0054] | 0.0120 [0.0102, 0.0138] |
| prior_ggr_eur_l30d | lag | 0.0049 [0.0045, 0.0053] | 0.0033 [0.0031, 0.0034] | 0.0178 [0.0170, 0.0185] |
| brandId | other | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

The delivery plan keeps a feature in the served model only if its permutation-importance interval excludes zero. Every LightGBM feature passes this rule.

![Permutation importance, LightGBM](../data/03_output/feature_importance/brand72/permutation_lightgbm.png)

### Cox PH

The Cox model gives a risk ranking, not a probability, so only the c-index is measured.

| feature | family | c-index drop [95% CI] |
|---|---|---|
| active_days_l90d | frequency | 0.1357 [0.1332, 0.1382] |
| prior_dormancy_spells_14d | tenure | 0.0256 [0.0246, 0.0266] |
| wagered_eur_l7d | monetary | 0.0147 [0.0142, 0.0152] |
| days_since_last_bet | recency | 0.0082 [0.0078, 0.0087] |
| prior_wagered_eur_l7d | lag | 0.0061 [0.0058, 0.0064] |
| prior_ggr_eur_l30d | lag | -0.0000 [-0.0002, 0.0001] |

![Permutation importance, Cox PH](../data/03_output/feature_importance/brand72/permutation_cox.png)

## 2. SHAP Values

SHAP splits each player's score into one part per feature. The parts add up to the score.

- **LightGBM**: exact TreeSHAP values from LightGBM itself (`pred_contrib=True`, the same algorithm as `shap.TreeExplainer`), in log-odds of churn.
- **Cox PH**: the model is linear, so the exact SHAP value is `beta * (x - training mean)`, in log-hazard. Positive means a higher risk of churning sooner.
- **Direction** is the sign of the rank correlation (`rank_corr`) between the feature value and its SHAP value. Under 0.5 in absolute value one direction would mislead, so it says "non-monotonic" and the dependence plot shows the real shape.

### LightGBM

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| active_days_l90d | frequency | 0.9829 | -0.9842 | higher value, lower churn risk |
| days_since_last_bet | recency | 0.4823 | 0.9451 | higher value, higher churn risk |
| wagered_eur_l7d | monetary | 0.4460 | -0.9105 | higher value, lower churn risk |
| prior_dormancy_spells_14d | tenure | 0.3688 | -0.9158 | higher value, lower churn risk |
| prior_wagered_eur_l7d | lag | 0.2539 | -0.8897 | higher value, lower churn risk |
| prior_ggr_eur_l30d | lag | 0.1120 | -0.6308 | higher value, lower churn risk |
| brandId | other | 0.0000 |  | categorical (brand) |

![Global SHAP, LightGBM](../data/03_output/feature_importance/brand72/shap_global_lightgbm.png)

Dependence plots for the top 7 features (all the features the model uses). The x axis is in real units (days, EUR, counts), not on the sign-log scale. Each dot is one validation player.

![SHAP dependence, LightGBM](../data/03_output/feature_importance/brand72/shap_dependence_lightgbm.png)

### Cox PH

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| active_days_l90d | frequency | 0.4469 | -1.0000 | higher value, lower churn risk |
| prior_dormancy_spells_14d | tenure | 0.2164 | -1.0000 | higher value, lower churn risk |
| wagered_eur_l7d | monetary | 0.1593 | -1.0000 | higher value, lower churn risk |
| prior_wagered_eur_l7d | lag | 0.1127 | -1.0000 | higher value, lower churn risk |
| days_since_last_bet | recency | 0.1111 | 1.0000 | higher value, higher churn risk |
| prior_ggr_eur_l30d | lag | 0.0309 | -1.0000 | higher value, lower churn risk |

![Global SHAP, Cox PH](../data/03_output/feature_importance/brand72/shap_global_cox.png)

A dependence plot for a linear model is a straight line with slope `beta`, so I do not draw it. The hazard ratios are in `hazard_ratios.csv` in the Cox MLflow run.

## 3. Importance by Family

How much each group of features matters, with two measures side by side. **SHAP share**: the family's part of the model's total mean |SHAP| on the validation rows, so how much it weighs in the predictions. **Score lost**: what the model loses on the test months when the whole family is removed and the model is retrained, so what it adds that the other families cannot replace. A family can weigh a lot and still lose little when removed, if other families carry the same information.

| family | lightgbm_shap_share | cox_shap_share | lightgbm_auc_lost | cox_c_index_lost |
|---|---|---|---|---|
| frequency | 0.3715 | 0.4148 | 0.0111 | 0.0330 |
| recency | 0.1823 | 0.1032 | 0.0048 | 0.0022 |
| monetary | 0.1686 | 0.1479 | 0.0020 | 0.0021 |
| tenure | 0.1394 | 0.2009 | 0.0085 | 0.0099 |
| lag | 0.1383 | 0.1333 | 0.0021 | 0.0011 |

![Importance by family](../data/03_output/feature_importance/brand72/importance_by_family.png)

## 4. Ablation by Family

I retrain the model without every feature of one family and compare it with the full model (same features otherwise, same parameters, same rows). A negative delta means the family helps. A family with no feature in the model is left empty.

Full model: LightGBM AUC valid 0.8960, test 0.8783. Cox c-index valid 0.8499, test 0.8523.

| family | LightGBM features | LightGBM AUC delta, valid | LightGBM AUC delta, test | Cox features | Cox c-index delta, valid | Cox c-index delta, test |
|---|---|---|---|---|---|---|
| recency | days_since_last_bet | -0.0032 | -0.0048 | days_since_last_bet | -0.0015 | -0.0022 |
| frequency | active_days_l90d | -0.0093 | -0.0111 | active_days_l90d | -0.0260 | -0.0330 |
| monetary | wagered_eur_l7d | -0.0020 | -0.0020 | wagered_eur_l7d | -0.0020 | -0.0021 |
| lag | prior_ggr_eur_l30d, prior_wagered_eur_l7d | -0.0009 | -0.0021 | prior_ggr_eur_l30d, prior_wagered_eur_l7d | -0.0019 | -0.0011 |
| trend |  |  |  |  |  |  |
| tenure | prior_dormancy_spells_14d | -0.0105 | -0.0085 | prior_dormancy_spells_14d | -0.0104 | -0.0099 |
| mix |  |  |  |  |  |  |
| deposit |  |  |  |  |  |  |
| segment |  |  |  |  |  |  |

The most valuable family on the test months is **frequency** for LightGBM (AUC -0.0111 without it) and **frequency** for Cox (c-index -0.0330 without it).

## 5. Known Limits

- **The Cox test months only have churn on days 0 to 37.** A churn day can only be confirmed when the 60 days after it are in the data, so on the test months the c-index mostly measures "who is already gone", not "which day".
- **The intervals only cover the shuffle.** A different validation sample would move the numbers more than the intervals show.
- **Correlated features share credit.** Stage 2 already removed pairs over 0.95, but features from the same family (for example `active_days_l30d` and `active_days_l90d`) can still hide each other in permutation importance. Ablation by family shows the joint value.

Generated by `src/models/feature_importance.py` (`make importance`). The figures and tables are in `data/03_output/feature_importance/brand72/` and in the `feature_importance/` folder of both MLflow runs.
