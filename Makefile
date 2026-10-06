# Every step of the project, from the repo root. `make help` lists them.
#
#   make pipeline AS_OF=2026-10-04                everything, as of one date: data, models, reports, scores
#   make data-pipeline                            only the training data steps (default or CUTOFFS="...")
#   make features AS_OF=2026-08-28                final feature vector for one date
#   make features AS_OF=2026-08-28 BRAND=basel    same, every brand
#   make train-baseline                           LightGBM + Cox PH on the latest training dataset
#   make train MODEL=logistic_regression          any model id from catalog/model_library_catalog.json

PY := .venv/bin/python
BRAND ?= 64
AS_OF ?=
CUTOFFS ?=
FEATURES ?=
MODEL ?=
DATASET ?=
PARAMS ?=
MLFLOW_PORT ?= 5000

cutoffs_arg := $(if $(CUTOFFS),--cutoff-dates $(CUTOFFS))
features_arg := $(if $(FEATURES),--features-path $(FEATURES))
train_args := $(if $(DATASET),--dataset-path $(DATASET)) $(if $(PARAMS),--params '$(PARAMS)') $(if $(NO_SELECTION),--no-selection) $(if $(NO_SEGMENTS),--no-segments) $(if $(NO_TUNING),--no-tuning) $(if $(N_TRIALS),--n-trials $(N_TRIALS))

.PHONY: help setup lint format \
        features anomalies-landing anomalies-features training-features labels dataset pipeline data-pipeline \
        segments train train-baseline results importance score alerts mlflow-ui clean-tmp

help:
	@echo "setup               install the environment and the pre-commit hooks"
	@echo "lint / format       ruff check / black + ruff --fix"
	@echo "features            final feature vector: AS_OF=YYYY-MM-DD [BRAND=64|basel]"
	@echo "anomalies-landing   anomaly study on raw landing data (EDA only)"
	@echo "anomalies-features  anomaly study on features, writes the winsorisation YAML [CUTOFFS=...]"
	@echo "training-features   feature snapshots for the training cutoffs [CUTOFFS=...]"
	@echo "labels              churn labels for a training features file [FEATURES=path]"
	@echo "dataset             join features and labels [FEATURES=path]"
	@echo "pipeline            EVERYTHING as of a date: AS_OF=YYYY-MM-DD [CUTOFFS=...] [SKIP_ANOMALIES=1] [SKIP_ALERTS=1]"
	@echo "                    data -> LightGBM + Cox -> segments + importance reports -> player scores"
	@echo "data-pipeline       only the training data steps [CUTOFFS=...] [SKIP_ANOMALIES=1]"
	@echo "segments            k-means player segments (k by silhouette) -> docs/player_segments_brand{id}.md"
	@echo "train               one catalog model: MODEL=<catalog id> [DATASET=path] [PARAMS='{json}'] [NO_SELECTION=1] [NO_SEGMENTS=1] [NO_TUNING=1] [N_TRIALS=n]"
	@echo "train-baseline      LightGBM (event_60d) + Cox PH (churn day) [DATASET=path]"
	@echo "results             reference LightGBM + Cox PH (optimisation step 1) -> docs/results_v0_brand{id}.md [DATASET=path]"
	@echo "importance          permutation, SHAP and family ablation of the latest baseline runs -> docs/feature_importance_brand{id}.md"
	@echo "score               churn probability + median survival days per player: AS_OF=YYYY-MM-DD [BRAND=64]"
	@echo "alerts              data alerts (configs/eda_alerts.yaml): AS_OF=YYYY-MM-DD [BRAND=64] [STAGE=landing|features|all]"
	@echo "mlflow-ui           open the MLflow UI on port $(MLFLOW_PORT)"
	@echo "clean-tmp           delete DuckDB spill files (.tmp/)"
	@echo "Default BRAND=$(BRAND). Without FEATURES, labels/dataset use the latest train_features_base file."

setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .

format:
	uv run black .
	uv run ruff check --fix .

features:
	@test -n "$(AS_OF)" || (echo "usage: make features AS_OF=YYYY-MM-DD [BRAND=64|basel]"; exit 1)
	$(PY) src/features/build_features.py --as-of $(AS_OF) --brand-id $(BRAND)

anomalies-landing:
	$(PY) eda/anomalies.py --source landing --brand-id $(BRAND)

anomalies-features:
	$(PY) eda/anomalies.py --source features --brand-id $(BRAND) $(cutoffs_arg)

training-features:
	$(PY) src/features/build_training_features.py --brand-id $(BRAND) $(cutoffs_arg)

labels:
	$(PY) src/features/build_labels.py $(features_arg)

dataset:
	$(PY) src/features/create_dataset.py $(features_arg)

pipeline:
	@test -n "$(AS_OF)" || (echo "usage: make pipeline AS_OF=YYYY-MM-DD [CUTOFFS=...] [SKIP_ANOMALIES=1]"; exit 1)
	$(PY) src/features/run_pipeline.py --as-of $(AS_OF) --brand-id $(BRAND) $(cutoffs_arg) $(if $(SKIP_ANOMALIES),--skip-anomalies) $(if $(SKIP_ALERTS),--skip-alerts)

data-pipeline:
	$(PY) src/features/run_pipeline.py --brand-id $(BRAND) $(cutoffs_arg) $(if $(SKIP_ANOMALIES),--skip-anomalies)

segments:
	$(PY) src/models/segments.py

train:
	@test -n "$(MODEL)" || (echo "usage: make train MODEL=<catalog id> [DATASET=path] [PARAMS='{json}']"; exit 1)
	$(PY) src/models/train.py --model-id $(MODEL) $(train_args)

train-baseline:
	$(PY) src/models/train.py --model-id lightgbm_classifier $(train_args)
	$(PY) src/models/train.py --model-id cox_ph $(train_args)

results:
	$(PY) src/models/results.py $(if $(DATASET),--dataset-path $(DATASET))

importance:
	$(PY) src/models/feature_importance.py

score:
	@test -n "$(AS_OF)" || (echo "usage: make score AS_OF=YYYY-MM-DD [BRAND=64]"; exit 1)
	$(PY) src/models/score.py --as-of $(AS_OF) --brand-id $(BRAND)

alerts:
	@test -n "$(AS_OF)" || (echo "usage: make alerts AS_OF=YYYY-MM-DD [BRAND=64] [STAGE=landing|features|all]"; exit 1)
	$(PY) src/features/alerts.py --as-of $(AS_OF) --brand-id $(BRAND) --stage $(or $(STAGE),all)

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port $(MLFLOW_PORT)

clean-tmp:
	rm -rf .tmp
