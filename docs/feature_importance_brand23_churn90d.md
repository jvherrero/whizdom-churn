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
| MLflow run | `lightgbm_classifier_brand23_1792152477` | `cox_ph_brand23_1792152518` |
| Target | `event_60d` (churn in the next 60 days) | `duration_days` + `event_observed` (churn day) |
| Features (after Stage 2 selection) | 10 | 9 |
| Test score (test months) | AUC 0.9095 | c-index 0.8684 |

- Dataset: `data/processed/churn90d/train_dataset_23_2025-08-01_2025-09-01_2025-10-01_2025-11-01_2025-12-01_2026-01-01_2026-02-01_2026-03-01_2026-04-01_2026-05-01_2026-06-01_2026-07-01_1792152022.parquet`.
- Rows: train 242,205, validation 22,053, test 41,206. Validation month: 2026-05-01. Test months: 2026-06-01, 2026-07-01.
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
| active_days_l90d | frequency | 0.0527 [0.0519, 0.0534] | 0.0229 [0.0226, 0.0232] | 0.0326 [0.0324, 0.0327] |
| days_since_last_bet | recency | 0.0236 [0.0230, 0.0243] | 0.0080 [0.0079, 0.0082] | 0.0143 [0.0138, 0.0148] |
| prior_dormancy_spells_14d | tenure | 0.0174 [0.0169, 0.0179] | 0.0104 [0.0101, 0.0106] | 0.0060 [0.0054, 0.0066] |
| bets_l7d | frequency | 0.0103 [0.0098, 0.0109] | 0.0049 [0.0047, 0.0052] | 0.0150 [0.0147, 0.0153] |
| engagement_score | mix | 0.0087 [0.0084, 0.0090] | 0.0039 [0.0038, 0.0040] | 0.0123 [0.0117, 0.0130] |
| prior_wagered_eur_l7d | lag | 0.0050 [0.0047, 0.0053] | 0.0023 [0.0021, 0.0024] | 0.0097 [0.0093, 0.0100] |
| losing_streak | mix | 0.0047 [0.0044, 0.0051] | 0.0040 [0.0038, 0.0042] | 0.0185 [0.0183, 0.0187] |
| bonus_stake_share_30d | mix | 0.0043 [0.0041, 0.0046] | 0.0023 [0.0022, 0.0024] | 0.0076 [0.0072, 0.0081] |
| days_since_last_return | tenure | 0.0025 [0.0023, 0.0026] | 0.0016 [0.0015, 0.0017] | 0.0062 [0.0058, 0.0065] |
| brandId | other | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

The delivery plan keeps a feature in the served model only if its permutation-importance interval excludes zero. Every LightGBM feature passes this rule.

![Permutation importance, LightGBM](../data/03_output/feature_importance/brand23_churn90d/permutation_lightgbm.png)

### Cox PH

The Cox model gives a risk ranking, not a probability, so only the c-index is measured.

| feature | family | c-index drop [95% CI] |
|---|---|---|
| active_days_l90d | frequency | 0.1756 [0.1744, 0.1768] |
| prior_dormancy_spells_14d | tenure | 0.0183 [0.0177, 0.0188] |
| bets_l7d | frequency | 0.0103 [0.0100, 0.0106] |
| prior_wagered_eur_l7d | lag | 0.0045 [0.0043, 0.0048] |
| days_since_last_bet | recency | 0.0019 [0.0018, 0.0021] |
| bonus_stake_share_30d | mix | 0.0011 [0.0010, 0.0012] |
| losing_streak | mix | 0.0004 [0.0003, 0.0005] |
| engagement_score | mix | 0.0003 [0.0002, 0.0005] |
| days_since_last_return | tenure | -0.0000 [-0.0000, -0.0000] |

![Permutation importance, Cox PH](../data/03_output/feature_importance/brand23_churn90d/permutation_cox.png)

## 2. SHAP Values

SHAP splits each player's score into one part per feature. The parts add up to the score.

- **LightGBM**: exact TreeSHAP values from LightGBM itself (`pred_contrib=True`, the same algorithm as `shap.TreeExplainer`), in log-odds of churn.
- **Cox PH**: the model is linear, so the exact SHAP value is `beta * (x - training mean)`, in log-hazard. Positive means a higher risk of churning sooner.
- **Direction** is the sign of the rank correlation (`rank_corr`) between the feature value and its SHAP value. Under 0.5 in absolute value one direction would mislead, so it says "non-monotonic" and the dependence plot shows the real shape.

### LightGBM

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| active_days_l90d | frequency | 0.8023 | -0.9817 | higher value, lower churn risk |
| days_since_last_bet | recency | 0.5434 | 0.9658 | higher value, higher churn risk |
| bets_l7d | frequency | 0.3159 | -0.9616 | higher value, lower churn risk |
| engagement_score | mix | 0.2601 | -0.9445 | higher value, lower churn risk |
| prior_dormancy_spells_14d | tenure | 0.2400 | -0.9446 | higher value, lower churn risk |
| prior_wagered_eur_l7d | lag | 0.1919 | -0.9086 | higher value, lower churn risk |
| bonus_stake_share_30d | mix | 0.1282 | 0.6051 | higher value, higher churn risk |
| days_since_last_return | tenure | 0.0712 | 0.2898 | non-monotonic, see the plot |
| losing_streak | mix | 0.0677 | -0.4694 | non-monotonic, see the plot |
| brandId | other | 0.0000 |  | categorical (brand) |

![Global SHAP, LightGBM](../data/03_output/feature_importance/brand23_churn90d/shap_global_lightgbm.png)

Dependence plots for the top 10 features (all the features the model uses). The x axis is in real units (days, EUR, counts), not on the sign-log scale. Each dot is one validation player.

![SHAP dependence, LightGBM](../data/03_output/feature_importance/brand23_churn90d/shap_dependence_lightgbm.png)

### Cox PH

| feature | family | mean_abs_shap | rank_corr | direction |
|---|---|---|---|---|
| active_days_l90d | frequency | 0.6242 | -1.0000 | higher value, lower churn risk |
| prior_dormancy_spells_14d | tenure | 0.2422 | -1.0000 | higher value, lower churn risk |
| bets_l7d | frequency | 0.1731 | -1.0000 | higher value, lower churn risk |
| prior_wagered_eur_l7d | lag | 0.1290 | -1.0000 | higher value, lower churn risk |
| days_since_last_bet | recency | 0.0652 | 1.0000 | higher value, higher churn risk |
| engagement_score | mix | 0.0576 | -1.0000 | higher value, lower churn risk |
| bonus_stake_share_30d | mix | 0.0413 | 1.0000 | higher value, higher churn risk |
| losing_streak | mix | 0.0376 | -1.0000 | higher value, lower churn risk |
| days_since_last_return | tenure | 0.0046 | 1.0000 | higher value, higher churn risk |

![Global SHAP, Cox PH](../data/03_output/feature_importance/brand23_churn90d/shap_global_cox.png)

A dependence plot for a linear model is a straight line with slope `beta`, so I do not draw it. The hazard ratios are in `hazard_ratios.csv` in the Cox MLflow run.

## 3. Importance by Family

How much each group of features matters, with two measures side by side. **SHAP share**: the family's part of the model's total mean |SHAP| on the validation rows, so how much it weighs in the predictions. **Score lost**: what the model loses on the test months when the whole family is removed and the model is retrained, so what it adds that the other families cannot replace. A family can weigh a lot and still lose little when removed, if other families carry the same information.

| family | lightgbm_shap_share | cox_shap_share | lightgbm_auc_lost | cox_c_index_lost |
|---|---|---|---|---|
| frequency | 0.4267 | 0.5800 | 0.0078 | 0.0221 |
| recency | 0.2073 | 0.0474 | 0.0042 | 0.0008 |
| mix | 0.1740 | 0.0993 | 0.0045 | 0.0008 |
| tenure | 0.1187 | 0.1795 | 0.0075 | 0.0101 |
| lag | 0.0732 | 0.0939 | 0.0026 | 0.0039 |

![Importance by family](../data/03_output/feature_importance/brand23_churn90d/importance_by_family.png)

## 4. Ablation by Family

I retrain the model without every feature of one family and compare it with the full model (same features otherwise, same parameters, same rows). A negative delta means the family helps. A family with no feature in the model is left empty.

Full model: LightGBM AUC valid 0.8989, test 0.9095. Cox c-index valid 0.8233, test 0.8684.

| family | LightGBM features | LightGBM AUC delta, valid | LightGBM AUC delta, test | Cox features | Cox c-index delta, valid | Cox c-index delta, test |
|---|---|---|---|---|---|---|
| recency | days_since_last_bet | -0.0035 | -0.0042 | days_since_last_bet | -0.0006 | -0.0008 |
| frequency | active_days_l90d, bets_l7d | -0.0104 | -0.0078 | active_days_l90d, bets_l7d | -0.0279 | -0.0221 |
| monetary |  |  |  |  |  |  |
| lag | prior_wagered_eur_l7d | -0.0020 | -0.0026 | prior_wagered_eur_l7d | -0.0020 | -0.0039 |
| trend |  |  |  |  |  |  |
| tenure | prior_dormancy_spells_14d, days_since_last_return | -0.0079 | -0.0075 | prior_dormancy_spells_14d, days_since_last_return | -0.0070 | -0.0101 |
| mix | engagement_score, losing_streak, bonus_stake_share_30d | -0.0033 | -0.0045 | engagement_score, losing_streak, bonus_stake_share_30d | -0.0003 | -0.0008 |
| deposit |  |  |  |  |  |  |
| segment |  |  |  |  |  |  |

The most valuable family on the test months is **frequency** for LightGBM (AUC -0.0078 without it) and **frequency** for Cox (c-index -0.0221 without it).

## 5. Known Limits

- **The Cox test months only have churn on days 0 to 37.** A churn day can only be confirmed when the 60 days after it are in the data, so on the test months the c-index mostly measures "who is already gone", not "which day".
- **The intervals only cover the shuffle.** A different validation sample would move the numbers more than the intervals show.
- **Correlated features share credit.** Stage 2 already removed pairs over 0.95, but features from the same family (for example `active_days_l30d` and `active_days_l90d`) can still hide each other in permutation importance. Ablation by family shows the joint value.

Generated by `src/models/feature_importance.py` (`make importance`). The figures and tables are in `data/03_output/feature_importance/brand23_churn90d/` and in the `feature_importance/` folder of both MLflow runs.
