# Every step of the project, from the repo root. `make help` lists them.
#
#   make pipeline AS_OF=2026-10-06                everything, as of one date: data, models, evaluation, reports, scores
#   make pipeline AS_OF=2026-10-06 BACKTEST=1     the same plus the rolling-origin backtest (T10), before scoring
#   make data-pipeline                            only the training data steps (default or CUTOFFS="...")
#   make features AS_OF=2026-08-28                final feature vector for one date
#   make features AS_OF=2026-08-28 BRAND=basel    same, every brand
#   make train-baseline                           LightGBM + Cox PH on the latest training dataset, then importance
#   make train MODEL=logistic_regression          any model id from catalog/model_library_catalog.json

PY := .venv/bin/python
BRAND ?= 64
AS_OF ?=
CUTOFFS ?=
MODEL ?=
DATASET ?=
PARAMS ?=
MLFLOW_PORT ?= 5000

cutoffs_arg := $(if $(CUTOFFS),--cutoff-dates $(CUTOFFS))
train_args := $(if $(DATASET),--dataset-path $(DATASET)) $(if $(PARAMS),--params '$(PARAMS)') $(if $(NO_SELECTION),--no-selection) $(if $(NO_SEGMENTS),--no-segments) $(if $(NO_TUNING),--no-tuning) $(if $(N_TRIALS),--n-trials $(N_TRIALS))

.PHONY: help setup lint format test \
        profile cache features feature-store anomalies-daily anomalies-features validate-signals dataset eda-summary pipeline data-pipeline \
        segments train train-baseline evaluate results backtest backtest-report robustness importance model-card score alerts mlflow-ui clean-tmp

help:
	@echo "setup               install the environment and the pre-commit hooks"
	@echo "lint / format       ruff check / black + ruff --fix"
	@echo "test                unit tests: metrics, the feature leakage test, dq_lib"
	@echo "profile             T2 data quality of the source tables -> docs/dq_reports/ + data card [START=] [END=] [SKIP_KEYS=1] [SEMANTIC=1|ONLY_SEMANTIC=1]"
	@echo "cache               local daily caches of the source tables: activity (all brands), financial + payments (BRAND) [CACHES="activity financial payments"]"
	@echo "features            final feature vector: AS_OF=YYYY-MM-DD [BRAND=64|basel]"
	@echo "feature-store       Feast (T6): fill the offline store from each brand's latest training dataset, register, check [DATASET=path]"
	@echo "anomalies-daily     anomaly study on the player-day caches (EDA only)"
	@echo "anomalies-features  anomaly study on features (EDA), also rewrites the winsorisation YAML [CUTOFFS=...]"
	@echo "validate-signals    signals-table columns against the daily caches -> docs/signal_validation.md [BRANDS=\"14 64 73\"] [BUILD=1: missing snapshots from S3]"
	@echo "dataset             survival dataset: features + churn labels of the training cutoffs [CUTOFFS=...] [DATA_END=YYYY-MM-DD]"
	@echo "eda-summary         EDA stage 4: training snapshot vs benchmark -> docs/eda_summary_brand{id}.md [DATASET=path] [BENCHMARK=path]"
	@echo "pipeline            EVERYTHING as of a date: AS_OF=YYYY-MM-DD [CUTOFFS=...] [SKIP_ANOMALIES=1] [SKIP_ALERTS=1] [BACKTEST=1]"
	@echo "                    daily caches -> features -> dataset -> LightGBM + Cox -> evaluate -> reports [-> backtest] -> player scores"
	@echo "data-pipeline       only the training data steps [CUTOFFS=...] [SKIP_ANOMALIES=1]"
	@echo "segments            k-means player segments (k by silhouette) -> docs/player_segments_brand{id}.md"
	@echo "train               one catalog model: MODEL=<catalog id> [DATASET=path] [PARAMS='{json}'] [NO_SELECTION=1] [NO_SEGMENTS=1] [NO_TUNING=1] [N_TRIALS=n]"
	@echo "train-baseline      LightGBM (event_60d) + Cox PH (churn day) + feature importance report [DATASET=path]"
	@echo "evaluate            metrics of one MLflow training run with 95% bootstrap CIs, logged into it: RUN_ID=<id>"
	@echo "results             reference LightGBM + Cox PH (optimisation step 1) -> docs/results_v0_brand{id}.md [DATASET=path]"
	@echo "backtest            T10 rolling-origin backtest over the last 12 monthly cutoffs -> docs/backtest_v0_brand{id}.md: AS_OF=YYYY-MM-DD [CONFIG=configs/backtest.yaml] [TOP_UP=1]"
	@echo "backtest-report     rewrite a finished backtest's report with the current pass conditions, no training: ID=backtest_brand{id}_..."
	@echo "robustness          training spec point 8: earlier monthly windows of a dataset -> docs/temporal_robustness_brand{id}.md [DATASET=path] [N_WINDOWS=2] [N_TRIALS=n] [SEED_MODE=per_window|grid] [SEEDS=\"42 7 2026\"]"
	@echo "model-card          one-screen model card from the MLflow runs (+ latest backtest) -> docs/model_card_v0_brand{id}.html [BRAND=64]"
	@echo "importance          permutation, SHAP and family ablation of the latest baseline runs -> docs/feature_importance_brand{id}.md"
	@echo "score               T11 player_scores of every active player: AS_OF=YYYY-MM-DD [BRAND=64] [MODEL_VERSION=n|alias] [NO_LEVEL=1]"
	@echo "alerts              data alerts (configs/eda_alerts.yaml): AS_OF=YYYY-MM-DD [BRAND=64] [STAGE=source|features|all]"
	@echo "mlflow-ui           open the MLflow UI on port $(MLFLOW_PORT)"
	@echo "clean-tmp           delete DuckDB spill files (.tmp/)"
	@echo "Default BRAND=$(BRAND)."

setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .

format:
	uv run black .
	uv run ruff check --fix .

test:
	$(PY) -m pytest src/evaluation/tests src/features/tests src/models/tests -q
	cd eda && ../$(PY) -m pytest dq_lib/tests -q

features:
	@test -n "$(AS_OF)" || (echo "usage: make features AS_OF=YYYY-MM-DD [BRAND=64|basel]"; exit 1)
	$(PY) src/features/build_features.py --as-of $(AS_OF) --brand-id $(BRAND)

feature-store:
	$(PY) src/features/feature_store.py $(if $(DATASET),--dataset $(DATASET))

anomalies-daily:
	$(PY) eda/anomalies.py --source daily --brand-id $(BRAND)

anomalies-features:
	$(PY) eda/anomalies.py --source features --brand-id $(BRAND) $(cutoffs_arg)

validate-signals:
	$(PY) eda/signal_validation.py $(if $(BRANDS),--brands $(BRANDS)) $(if $(BUILD),--build-missing) $(if $(REPORT_ONLY),--report-only)

dataset:
	$(PY) src/features/build_survival_dataset.py --brand-id $(BRAND) $(cutoffs_arg) $(if $(DATA_END),--data-end $(DATA_END))

eda-summary:
	$(PY) eda/summary.py $(if $(DATASET),--dataset $(DATASET)) $(if $(BENCHMARK),--benchmark $(BENCHMARK))

pipeline:
	@test -n "$(AS_OF)" || (echo "usage: make pipeline AS_OF=YYYY-MM-DD [CUTOFFS=...] [SKIP_ANOMALIES=1]"; exit 1)
	$(PY) src/features/run_pipeline.py --as-of $(AS_OF) --brand-id $(BRAND) $(cutoffs_arg) $(if $(SKIP_ANOMALIES),--skip-anomalies) $(if $(SKIP_ALERTS),--skip-alerts) $(if $(BACKTEST),--backtest)

profile:
	$(PY) eda/profile_tables.py $(if $(START),--start $(START)) $(if $(END),--end $(END)) $(if $(SKIP_KEYS),--skip-keys) $(if $(SEMANTIC),--semantic --brand-id $(BRAND)) $(if $(ONLY_SEMANTIC),--only-semantic --brand-id $(BRAND))

cache:
	$(PY) src/features/daily_cache.py --cache $(or $(CACHES),activity financial payments) --brand-id $(BRAND) $(if $(START),--start $(START)) $(if $(END),--end $(END))

data-pipeline:
	$(PY) src/features/run_pipeline.py --brand-id $(BRAND) $(cutoffs_arg) $(if $(SKIP_ANOMALIES),--skip-anomalies)

segments:
	$(PY) src/models/segments.py

train:
	@test -n "$(MODEL)" || (echo "usage: make train MODEL=<catalog id> [DATASET=path] [PARAMS='{json}']"; exit 1)
	$(PY) src/models/train.py --model-id $(MODEL) $(train_args)

train-baseline:
	$(PY) src/models/train.py --model-id lightgbm_classifier --no-importance $(train_args)
	$(PY) src/models/train.py --model-id cox_ph $(train_args)

evaluate:
	@test -n "$(RUN_ID)" || (echo "usage: make evaluate RUN_ID=<MLflow run id>"; exit 1)
	$(PY) src/evaluation/evaluate.py --run-id $(RUN_ID)

results:
	$(PY) src/models/results.py $(if $(DATASET),--dataset-path $(DATASET))

backtest:
	@test -n "$(AS_OF)" || (echo "usage: make backtest AS_OF=YYYY-MM-DD [CONFIG=configs/backtest.yaml] [BRAND=64] [TOP_UP=1]"; exit 1)
	$(PY) src/models/backtest.py --as-of $(AS_OF) --brand-id $(BRAND) $(if $(CONFIG),--config $(CONFIG)) $(if $(TOP_UP),--top-up)

backtest-report:
	@test -n "$(ID)" || (echo "usage: make backtest-report ID=backtest_brand64_2026-10-06_... (data/03_output/backtest/)"; exit 1)
	$(PY) src/models/backtest.py --report-only $(ID) $(if $(CONFIG),--config $(CONFIG))

robustness:
	$(PY) src/models/robustness.py $(if $(DATASET),--dataset-path $(DATASET)) $(if $(N_WINDOWS),--n-windows $(N_WINDOWS)) $(if $(N_TRIALS),--n-trials $(N_TRIALS)) $(if $(SEEDS),--seeds $(SEEDS)) $(if $(SEED_MODE),--seed-mode $(SEED_MODE))

model-card:
	$(PY) src/models/model_card.py --brand-id $(BRAND)

importance:
	$(PY) src/models/feature_importance.py

score:
	@test -n "$(AS_OF)" || (echo "usage: make score AS_OF=YYYY-MM-DD [BRAND=64]"; exit 1)
	$(PY) src/models/score.py --as-of $(AS_OF) --brand-id $(BRAND) $(if $(MODEL_VERSION),--model-version $(MODEL_VERSION)) $(if $(NO_LEVEL),--no-level)

alerts:
	@test -n "$(AS_OF)" || (echo "usage: make alerts AS_OF=YYYY-MM-DD [BRAND=64] [STAGE=source|features|all]"; exit 1)
	$(PY) src/features/alerts.py --as-of $(AS_OF) --brand-id $(BRAND) --stage $(or $(STAGE),all)

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port $(MLFLOW_PORT)

clean-tmp:
	rm -rf .tmp
