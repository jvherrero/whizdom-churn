# Week 1 Summary: Baseline Model and End-to-End Pipeline (brandId 64)

Written on 2026-10-07, updated on 2026-10-14. Data: S3 data lake (`org/40-gold`), 2025-04-01 to 2026-10-06. Brand 64 is the main example; brands 14, 23, 72 and 73 have their own models and backtests.

## 1. What Runs

`make pipeline AS_OF=DATE [BACKTEST=1]` runs, as of one date: daily caches and data alerts, 39 features at 12 monthly cutoffs, the survival dataset, LightGBM and Cox PH, evaluation with bootstrap CIs, segment and importance reports, the rolling-origin backtest and the `player_scores` table. Every run is logged to MLflow. From a clean clone (T12, done on 2026-10-14), `make setup`, `make pipeline AS_OF=2026-10-06` (11 minutes with S3 access) and `make backtest AS_OF=2026-10-06` run end to end, and the metrics and the backtest are close to the ones below. `make test` passes 28 project tests and 23 tests of `eda/dq_lib`; the 2 API tests that need a model are skipped until the first `make pipeline` and `make promote`.

| Deliverable | Status |
|---|---|
| D1 Repo, `make setup`, `make pipeline`, pre-commit, tests | Done |
| D2 Churn definition with KM curves and return rates | Done: `docs/churn_definition_v0.md` |
| D3 Signal explorer report | Done: `docs/churn_signals_v0.md` |
| D4 Feature dictionary, pipeline, leakage test | Done (39 features). Feast registry and offline store, local (`src/features/feature_repo/`, point-in-time check); S3 and Redis need write access |
| D5 Survival dataset and data card | Done: 7/14/30-day labels, EDA stage 4 summary (`docs/eda_summary_brand64.md`). Versioned in MLflow, **not yet as a snapshot in S3** |
| D6 Evaluation module with CIs and calibration plots | Done: `src/evaluation/` |
| D7 Reference LightGBM and Cox PH with results table | Done: `docs/results_v0_brand64.md`; model card `docs/model_card_v0_brand64.html` |
| D8 Backtest: rolling-origin and targeting replay | Done: `docs/backtest_v0_brand64.md` (verdict FAIL, see 3) |
| D9 Scoring command and schema | Done: `docs/schema_player_scores.md` |
| Week 2, started early: churn API | Done locally: FastAPI service (`serving/`, `make serve`, `/health`, `/score/batch`, `/score/{brand_id}/{player_id}`), Docker image `whizdom-churn-serve`, model loaded by the alias `Production` (`make promote`). Not deployed: ECS needs write access to AWS |

## 2. Results

- **Churn**: 60 days without a bet. After 60 silent days, 13.0% of players come back within the next 60 days.
- **LightGBM** (optimised, 5 features + brand): test AUC **0.894** [0.891, 0.899], calibrated ECE **0.021**, top-decile precision 0.880 (2.4x the base rate). Days since the last bet alone: AUC 0.747.
- **Cox PH**: test C-index **0.869** [0.865, 0.874], IBS 0.036. Its 7/14/30-day probabilities are calibrated on the validation month (ECE 0.10 to 0.03).
- **Backtest, 11 monthly cutoffs**: AUC 0.851 to 0.896, above recency by 0.12 on average (at least 0.10 every month). In the riskiest 10% of players, 86% churn with the model against 71% with recency. A campaign on the top 1,000 players by value at risk would have reached 66% of the GGR at risk (11% by recency, 7% at random). Most of this gain comes from using money: a simple value x recency rule reaches 94% of what the model reaches, and the model adds 6% on top (better in 10 of 11 months).
- **Most useful features**: activity frequency (`active_days_l90d`), then recency and tenure.

## 3. Open Issues

1. **Backtest pass conditions**.
   - **Top-decile condition, decided on 2026-10-09** (`docs/decision_top_decile_condition.md`): "precision >= 1.5x the recency rule" cannot be met (recency is 68% to 81% precise, so the ceiling is 1.24x to 1.46x). The KPI stays and its threshold is lowered to 1.10x, the strictest round value the model meets with 95% confidence in every brand. All five brands pass it (1.16x to 1.27x).
   - **ECE <= 0.08 every month**: the calibration month lags 2 to 3 months behind the scored date, and the churn rate peaks around 50% from November to January. The calibration level is now re-estimated on the scoring date (`src/models/calibration_level.py`). In the new backtests of the five brands, the months over 0.08 go from 9 to 4: December 2025 in brands 14, 64 and 72, and May 2026 in brand 14. No data before the scoring date could have anticipated them; one more year of history will allow a year-on-year seasonal term.
2. **Optimisation adds nothing**: the reference and the optimised LightGBM score the same in the backtest and on 2 earlier windows. By the plan's rule, the simpler reference configuration is the safer choice.
3. **Source data, for the data team**: completed deposits only exist from March 2026 (deposit features are 82% empty and are dropped); `days_since_bet` and `tenure_days` in `gld_player_signals_daily` are inconsistent (recomputed from activity instead); `brand_id` is empty before July 2026; payments have 40 missing days. Is the 2026-08-08 backfill point-in-time correct?
4. **To agree**: the number of players the CRM team can contact (N in the targeting replay, 1,000 for now).
5. **Not done yet**: the versioned snapshot and the Feast offline store in S3, and the API on ECS (the sprint's AWS access is read-only); the rest of the Week 2 work (Airflow DAGs, monitoring and drift).

## 4. Deviations from the Plan

Churn is 60 days, not 30, so the main score is `p_churn_60d` and `p_churn_7d/14d/30d` mean "the churn starts within 7 / 14 / 30 days". The calibration method is chosen on cross-fitted validation ECE, not on the test months. The code lives under `src/` (`features`, `models`, `evaluation`).
