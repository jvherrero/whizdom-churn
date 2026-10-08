# Whizdom Churn Prediction Pipeline

Player churn prediction for one brand (brandId 64), built on the S3 data lake (`org/40-gold`). One command runs the whole chain as of a date: source data, features, survival dataset, LightGBM churn classifier and Cox PH survival companion, evaluation with confidence intervals, reports, backtest and a daily `player_scores` table.

A player has **churned** when 60 days pass without a bet (`docs/churn_definition_v0.md`).

## 1. Requirements

- [`uv`](https://docs.astral.sh/uv/) (it installs Python 3.13 and the pinned packages from `uv.lock`).
- Read access to the S3 data lake (`org/40-gold`) through an AWS SSO profile. The default profile is `javier-whizdom-prod-ds`; set `DATALAKE_AWS_PROFILE` to use another one. The session expires: log in again when a run fails with `TokenRetrievalError`.

```bash
aws sso login --profile javier-whizdom-prod-ds
```

## 2. Setup

```bash
make setup      # uv sync + pre-commit hooks
make test       # unit tests: evaluation metrics, the feature leakage test, dq_lib
```

## 3. Running It

```bash
make pipeline AS_OF=2026-10-06               # everything as of one date
make pipeline AS_OF=2026-10-06 BACKTEST=1    # the same plus the rolling-origin backtest (about 45 min more)
make mlflow-ui                               # the runs, models and artifacts, on http://localhost:5000
```

`AS_OF` is the "today" of the run: nothing after it is read. The steps, in order:

1. **Daily caches**: the local daily extracts of the source tables are topped up to `AS_OF` (only the missing days, plus the last 7, which the source reprocesses). The first run on a new machine downloads the whole history, so it takes much longer.
2. **Source alerts**: the caches on `AS_OF` against the training days (row counts, nulls, daily totals, pandera suites).
3. **Features and dataset**: 12 monthly cutoffs, the 39 features of `docs/feature_dictionary.md`, the winsorisation caps and the 60-day churn labels.
4. **Models**: LightGBM (Stage 2 feature selection, class balance, Optuna, regularisation, pruning, calibration) and Cox PH with the same features. Split by month: train 1 to 9, validation 10, test 11 and 12.
5. **Evaluation**: AUC, ECE, top-decile precision, C-index, IBS, time-dependent AUC and one-calibration, with bootstrap CIs, logged into each MLflow run.
6. **Reports**: player segments and feature importance (`docs/`).
7. **Backtest** (with `BACKTEST=1`): rolling-origin over the last 12 months, `docs/backtest_v0_brand64.md`.
8. **Scores**: every active player as of `AS_OF`, after the feature alerts, in `data/03_output/player_scores/run_date=AS_OF/` (`docs/schema_player_scores.md`).

A critical data alert stops the run before any score is written. Feature drift only sets `drift_flag`.

## 4. Other Commands

`make help` lists all of them. The main ones:

| Command | What it does |
|---|---|
| `make score AS_OF=... [MODEL_VERSION=n]` | Scores for one date with a registered model version (T11) |
| `make backtest AS_OF=...` | Rolling-origin backtest (T10), `configs/backtest.yaml` |
| `make results` | Reference LightGBM + Cox PH and the first results table (T9) |
| `make evaluate RUN_ID=...` | Metrics with bootstrap CIs of one MLflow run (T8) |
| `make robustness` | Training spec point 8: the optimisation on 2 earlier windows |
| `make profile` | Data quality of the source tables, `docs/dq_reports/` and the data card (T2) |
| `make cache` | Build or top up the local daily caches only |
| `make alerts AS_OF=...` | Data alerts on their own (`configs/eda_alerts.yaml`) |
| `make anomalies-daily`, `make anomalies-features` | EDA stage 2: outliers and anomalies |

`BRAND=64` is the default; `BRAND=basel` means every brand.

## 5. Repository Layout

| Path | Content |
|---|---|
| `src/features/` | S3 access and daily caches, features (`build_features.py`), labels, survival dataset, alerts, the pipeline (`run_pipeline.py`) |
| `src/models/` | Training (`train.py`), feature selection, calibration, segments, importance, results, backtest, robustness, scoring |
| `src/evaluation/` | Metrics and `evaluate --run-id` |
| `eda/` | Source-table profiling, signal explorer, anomaly study, pandera suites (`expectations/`), notebooks |
| `configs/` | Alert thresholds, backtest settings, winsorisation caps |
| `catalog/` | Model catalog: every model's class, parameters and training settings |
| `docs/` | Churn definition, signals, feature dictionary, data card, results, backtest, scores schema, Week 1 summary |
| `data/` | Local data, not in git (player-level): caches, datasets, scores. Only report figures are versioned |
| `infra/` | Terraform and Docker Compose for the AWS deployment (Week 2) |

Every training, dataset, alert and backtest run is logged to MLflow (`mlflow.db` + `mlruns/`, local).

## 6. Types of Tasks

All the work in this project is divided into five types of tasks:

1. **EDA:** data profiling, activity and value distributions, sensitivity analysis of the churn definition, and censoring checks.
2. **Feature Engineering:** the player-day feature table: recency, frequency, monetary (RFM), session and bet aggregates, bonus exposure, tenure and cross-vertical mix.
3. **Model Selection:** LightGBM churn classifier (the chosen architecture), Cox PH survival companion, optimisation, calibration and model card.
4. **Model Serving:** FastAPI service image with online and batch endpoints, scoring schema, `player_scores` table and model loading by registry alias.
5. **MLOps:** repository structure, config-driven runs, experiment tracking (MLflow), orchestration with Airflow, feature store (Feast), deployment on ECS, monitoring, drift detection, alerts and runbook.
