# Feature Importance v0 (brandId 23)

This document reports three importance measures together for the two baseline models. Gain-based importance alone overstates features with many distinct values and features that are correlated with others, so I do not use it here.

| Measure | Question it answers |
|---|---|
| Permutation importance | How much do the model's AUC and calibration fall when this feature is shuffled? |
| SHAP values | What does this feature do to each player's score, and in which direction? |
| Ablation by family | What is a whole family of features worth to the model? |

## 0. Models and Data

| | LightGBM | Cox PH |
|---|---|---|
| MLflow run | `lightgbm_classifier_brand23_1792128061` | `cox_ph_brand23_1792128092` |
| Target | `event_60d` (churn in the next 60 days) | `duration_days` + `event_observed` (churn day) |
| Features (after Stage 2 selection) | 8 | 7 |
| Test score (test months) | AUC 0.8931 | c-index 0.8399 |

- Dataset: `data/processed/train_dataset_23_2025-09-01_2025-10-01_2025-11-01_2025-12-01_2026-01-01_2026-02-01_2026-03-01_2026-04-01_2026-05-01_2026-06-01_2026-07-01_2026-08-01_1792127749.parquet`.
- Rows: train 235,552, validation 21,421, test 39,229. Validation month: 2026-06-01. Test months: 2026-07-01, 2026-08-01.
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
| active_days_l30d | frequency | 0.0646 [0.0630, 0.0662] | 0.0231 [0.0227, 0.0236] | 0.0207 [0.0198, 0.0217] |
| days_since_last_bet | recency | 0.0250 [0.0240, 0.0260] | 0.0086 [0.0083, 0.0090] | 0.0085 [0.0079, 0.0092] |
| prior_dormancy_spells_14d | tenure | 0.0249 [0.0237, 0.0261] | 0.0154 [0.0147, 0.0160] | 0.0103 [0.0095, 0.0112] |
| prior_ggr_eur_l30d | lag | 0.0143 [0.0139, 0.0148] | 0.0086 [0.0084, 0.0089] | 0.0210 [0.0208, 0.0213] |
| bets_l7d | frequency | 0.0142 [0.0136, 0.0148] | 0.0083 [0.0079, 0.0086] | 0.0215 [0.0212, 0.0218] |
| prior_wagered_eur_l7d | lag | 0.0097 [0.0093, 0.0100] | 0.0047 [0.0044, 0.0049] | 0.0105 [0.0101, 0.0110] |
| losing_streak | mix | 0.0072 [0.0068, 0.0075] | 0.0047 [0.0045, 0.0049] | 0.0151 [0.0149, 0.0152] |
| brandId | other | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

The delivery plan keeps a feature in the served model only if its permutation-importance interval excludes zero. Every LightGBM feature passes this rule.

![Permutation importance, LightGBM](../data/03_output/feature_importance/brand23/permutation_lightgbm.png)

### Cox PH

The Cox model gives a risk ranking, not a probability, so only the c-index is measured.

| feature | family | c-index drop [95% CI] |
|---|---|---|
| active_days_l30d | frequency | 0.1250 [0.1229, 0.1271] |
| prior_dormancy_spells_14d | tenure | 0.0255 [0.0244, 0.0266] |
| bets_l7d | frequency | 0.0147 [0.0141, 0.0152] |
| prior_wagered_eur_l7d | lag | 0.0127 [0.0122, 0.0131] |
| losing_streak | mix | 0.0045 [0.0043, 0.0048] |
| prior_ggr_eur_l30d | lag | 0.0039 [0.0035, 0.0042] |
| days_since_last_bet | recency | 0.0015 [0.0013, 0.0017] |

![Permutation importance, Cox PH](../data/03_output/feature_importance/brand23/permutation_cox.png)

## 2. SHAP Values

SHAP splits each player's score into one part per feature. The parts add up to the score.

- **LightGBM**: exact TreeSHAP values from LightGBM itself (`pred_contrib=True`, the same algorithm as `shap.TreeExplainer`), in log-odds of churn.
- **Cox PH**: the model is linear, so the exact SHAP value is `beta * (x - training mean)`, in log-hazard. Positive means a higher risk of churning sooner.
- **Direction** is the sign of the rank correlation (`rank_corr`) between the feature value and its SHAP value. Under 0.5 in absolute value one direction would mislead, so it says "non-monotonic" and the dependence plot shows the real shape.

### LightGBM

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| active_days_l30d | frequency | 1.1175 | -0.9843 | higher value, lower churn risk |
| days_since_last_bet | recency | 0.6207 | 0.9111 | higher value, higher churn risk |
| prior_dormancy_spells_14d | tenure | 0.3610 | -0.9232 | higher value, lower churn risk |
| bets_l7d | frequency | 0.3144 | -0.9240 | higher value, lower churn risk |
| prior_ggr_eur_l30d | lag | 0.2660 | -0.5671 | higher value, lower churn risk |
| prior_wagered_eur_l7d | lag | 0.1804 | -0.7886 | higher value, lower churn risk |
| losing_streak | mix | 0.1039 | -0.4139 | non-monotonic, see the plot |
| brandId | other | 0.0000 |  | categorical (brand) |

![Global SHAP, LightGBM](../data/03_output/feature_importance/brand23/shap_global_lightgbm.png)

Dependence plots for the top 8 features (all the features the model uses). The x axis is in real units (days, EUR, counts), not on the sign-log scale. Each dot is one validation player.

![SHAP dependence, LightGBM](../data/03_output/feature_importance/brand23/shap_dependence_lightgbm.png)

### Cox PH

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| active_days_l30d | frequency | 0.4471 | -1.0000 | higher value, lower churn risk |
| prior_dormancy_spells_14d | tenure | 0.2372 | -1.0000 | higher value, lower churn risk |
| bets_l7d | frequency | 0.1829 | -1.0000 | higher value, lower churn risk |
| prior_wagered_eur_l7d | lag | 0.1587 | -1.0000 | higher value, lower churn risk |
| losing_streak | mix | 0.1078 | -1.0000 | higher value, lower churn risk |
| prior_ggr_eur_l30d | lag | 0.1026 | -1.0000 | higher value, lower churn risk |
| days_since_last_bet | recency | 0.0539 | 1.0000 | higher value, higher churn risk |

![Global SHAP, Cox PH](../data/03_output/feature_importance/brand23/shap_global_cox.png)

A dependence plot for a linear model is a straight line with slope `beta`, so I do not draw it. The hazard ratios are in `hazard_ratios.csv` in the Cox MLflow run.

## 3. Importance by Family

How much each group of features matters, with two measures side by side. **SHAP share**: the family's part of the model's total mean |SHAP| on the validation rows, so how much it weighs in the predictions. **Score lost**: what the model loses on the test months when the whole family is removed and the model is retrained, so what it adds that the other families cannot replace. A family can weigh a lot and still lose little when removed, if other families carry the same information.

| family | lightgbm_shap_share | cox_shap_share | lightgbm_auc_lost | cox_c_index_lost |
|---|---|---|---|---|
| frequency | 0.4831 | 0.4883 | 0.0081 | 0.0280 |
| recency | 0.2094 | 0.0418 | 0.0038 | 0.0007 |
| lag | 0.1506 | 0.2026 | 0.0069 | 0.0038 |
| tenure | 0.1218 | 0.1839 | 0.0053 | 0.0117 |
| mix | 0.0350 | 0.0835 | 0.0014 | 0.0013 |

![Importance by family](../data/03_output/feature_importance/brand23/importance_by_family.png)

## 4. Ablation by Family

I retrain the model without every feature of one family and compare it with the full model (same features otherwise, same parameters, same rows). A negative delta means the family helps. A family with no feature in the model is left empty.

Full model: LightGBM AUC valid 0.9013, test 0.8931. Cox c-index valid 0.8215, test 0.8399.

| family | LightGBM features | LightGBM AUC delta, valid | LightGBM AUC delta, test | Cox features | Cox c-index delta, valid | Cox c-index delta, test |
|---|---|---|---|---|---|---|
| recency | days_since_last_bet | -0.0032 | -0.0038 | days_since_last_bet | -0.0004 | -0.0007 |
| frequency | active_days_l30d, bets_l7d | -0.0111 | -0.0081 | active_days_l30d, bets_l7d | -0.0269 | -0.0280 |
| monetary |  |  |  |  |  |  |
| lag | prior_ggr_eur_l30d, prior_wagered_eur_l7d | -0.0043 | -0.0069 | prior_ggr_eur_l30d, prior_wagered_eur_l7d | -0.0051 | -0.0038 |
| trend |  |  |  |  |  |  |
| tenure | prior_dormancy_spells_14d | -0.0061 | -0.0053 | prior_dormancy_spells_14d | -0.0098 | -0.0117 |
| mix | losing_streak | -0.0021 | -0.0014 | losing_streak | -0.0016 | -0.0013 |
| deposit |  |  |  |  |  |  |
| segment |  |  |  |  |  |  |

The most valuable family on the test months is **frequency** for LightGBM (AUC -0.0081 without it) and **frequency** for Cox (c-index -0.0280 without it).

## 5. Known Limits

- **The Cox test months only have churn on days 0 to 37.** A churn day can only be confirmed when the 60 days after it are in the data, so on the test months the c-index mostly measures "who is already gone", not "which day".
- **The intervals only cover the shuffle.** A different validation sample would move the numbers more than the intervals show.
- **Correlated features share credit.** Stage 2 already removed pairs over 0.95, but features from the same family (for example `active_days_l30d` and `active_days_l90d`) can still hide each other in permutation importance. Ablation by family shows the joint value.

Generated by `src/models/feature_importance.py` (`make importance`). The figures and tables are in `data/03_output/feature_importance/brand23/` and in the `feature_importance/` folder of both MLflow runs.
