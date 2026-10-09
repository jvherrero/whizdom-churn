# player_scores Schema v0
`player_scores` holds one row per active player, brand and run date: the churn scores the business reads. Consumers read this table; they never call the model.

## 1. How It Is Written

- **Command**: `make score AS_OF=YYYY-MM-DD [BRAND=64] [MODEL_VERSION=n|alias]` (`src/models/score.py`). `make pipeline` runs it as its last step, with the models it has just trained.
- **Rows**: every player of the brand with a bet in the 30 days up to the run date. The command checks that the row count equals the number of active players and stops otherwise.
- **Storage**: `data/03_output/player_scores/run_date=YYYY-MM-DD/brand_id=N.parquet`, one partition per run date and brand. A rerun for the same date overwrites its partition, so the command is idempotent and can be rerun for any date. In production the same rows go to ClickHouse, partitioned by `run_date` (delete-then-insert).
- **Models**: `MODEL_VERSION` is a version (number or alias) of the registered model `churn_lightgbm_classifier_brand{id}` in MLflow. The Cox PH companion is the run trained with that LightGBM run's features. Without it, the latest optimised run is used.
- **Features**: rebuilt as of the run date through `build_features` (T6), the same code as training.
- **Calibration level**: the run's calibration was fitted on its validation month, 2 to 3 months before the run date, and the churn rate moves with the seasons. So the level of `p_churn_60d` is re-estimated on the run date (`src/models/calibration_level.py`): one log-odds shift per brand, the median of the shifts that set the mean probability right on the last 3 months with a known label and on the later months' early signal (who has bet again by the run date). It changes no ranking. `--no-level` (`make score NO_LEVEL=1`) keeps the training calibration.
- **Checks before writing**: the feature snapshot is compared with the training dataset (`configs/eda_alerts.yaml`). A critical data-quality alert (expectations, player count) stops the run and writes no scores. Critical feature drift does not stop it: it sets `drift_flag`.

## 2. Columns

| Column | Type | Meaning |
|---|---|---|
| `run_date` | string (YYYY-MM-DD) | The date the players are scored as of. Features use data up to and including this day (UTC). |
| `model_version` | string | The registered model version, `churn_lightgbm_classifier_brand{id}/v{n}` (the run name if it is not registered). |
| `tenant_id` | string | Tenant of the player. |
| `brand_id` | int64 | Brand of the player. |
| `player_id` | int32 | Player id, unique within a tenant. Key: (`run_date`, `tenant_id`, `player_id`). |
| `p_churn_7d` | float64 | Probability that the player's churn starts within 7 days: their last bet before a 60-day silence falls in the next 7 days. Cox PH survival curve, 1 - S(7), calibrated with isotonic regression on the validation month. |
| `p_churn_14d` | float64 | The same within 14 days. Its label in the training dataset is `churn_within_14d` (likewise 7 and 30). |
| `p_churn_30d` | float64 | The same within 30 days. The three are non-decreasing (7 <= 14 <= 30) and never below `p_churn_60d`. |
| `p_churn_60d` | float64 | **The main score.** Probability of no bet in the next 60 days (churn starting now, `docs/churn_definition_v0.md`). Calibrated LightGBM, with the calibration level of the run date (section 1). |
| `median_survival_days` | float64 | Cox PH: days until the player's survival curve falls to 0.5. 0 = more likely than not already gone. Empty when `median_beyond_horizon`. |
| `median_beyond_horizon` | bool | True when the curve never falls to 0.5 within the days the model has seen churn on: the churn day is later than that. |
| `expected_ggr_30d` | float64 | What the player brings in 30 days at the current pace: trailing-90-day mean daily GGR x 30, in EUR. Negative when the player won. |
| `value_at_risk_30d` | float64 | `expected_ggr_30d` (0 when negative) x `p_churn_30d`, in EUR: the GGR the brand expects to lose in 30 days. The ranking for a retention campaign. |
| `risk_band` | int64 | 1 to 10: the decile of `p_churn_60d` within the run date and brand (10 = the riskiest 10%), of equal size: ties (the calibration is a step function) are broken by `p_churn_30d`. No threshold is built in: the CRM team picks the bands they act on. |
| `top_drivers` | string | The 3 features that move the player's `p_churn_60d` the most (TreeSHAP, log-odds), with their sign: + pushes towards churn. Example: `days_since_last_bet (+1.20); active_days_l90d (-0.85); wagered_eur_l7d (-0.31)`. |
| `segment` | int64 | k-means player type (`docs/player_segments_brand{id}.md`). |
| `drift_flag` | int64 | 1 when the run had a critical feature drift alert for the brand (PSI > 0.25 on one feature used by the models, or > 0.10 on 5 or more). The scores are written but should be reviewed before acting on them. |
| `lower`, `upper` | float64 | Reserved for prediction intervals (conformal survival intervals, RO5 work). Empty in v0. |

## 3. Differences from the Delivery Plan

- **Churn is 60 days of silence** (agreed in T3), not 30. So the main score is `p_churn_60d`, and the plan's `p_churn_7d/14d/30d` mean "the 60-day churn starts within 7 / 14 / 30 days". They come from the Cox PH companion, calibrated, so the three horizons are consistent with each other. In the test months of the reference run, calibration lowered their expected calibration error from about 0.10 to about 0.03.
- **`risk_band` is the decile of `p_churn_60d`**, the calibrated LightGBM score the business consumes, as in the plan's architecture.
- **Value at risk uses `p_churn_30d`**, as in the plan's formula. The backtest's targeting replay (T10) ranks by `p_churn_60d` instead, the score it evaluates.
- **Extra columns**: `tenant_id` and `brand_id` (the player key of the source tables), `p_churn_60d`, `median_beyond_horizon`, `top_drivers` (the plan's SHAP top-3 drivers) and `segment`.

Generated by hand; the code is the reference: `COLUMNS` in `src/models/score.py`.
