# Whizdom Churn Prediction Pipeline

Repository for the base churn prediction model and the end-to-end pipeline.

## Environment Setup

This project uses:
-  `uv` for dependency management.
- `pre-commit` to ensure code quality.

To set up the environment on a clean machine, run:

```bash
make setup
```

## Types of Tasks

All the work involved in this project is divided into the following five main types of tasks:

1. **EDA:**

    Data profiling, activity and value distributions, sensitivity analysis of the churn definition, and checks for censoring.
2. **Feature Engineering:**

    Creation of the player-day feature table: recency, frequency, monetary (RFM), session- and bet-level aggregates, bonus exposure, tenure and cross-vertical mix.
3. **Model Selection:**

    LightGBM churn classifier (chosen architecture), complementary Cox PH survival model, optimisation, calibration and model card.
4. **Model Serving:**

    FastAPI service image with online and batch endpoints, scoring scheme, `player_scores` table and model loading via registration aliases.
5. **MLOps**

    Repository structure, configuration-based runs, experiment tracking (MLflow), orchestration with Airflow, feature store (Feast), deployment on ECS, monitoring, drift detection, alerts and operations manual (runbook).

