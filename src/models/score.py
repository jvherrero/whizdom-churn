"""Scoring command (T11): every active player of a brand as of one date, written to the player_scores table.

    .venv/bin/python src/models/score.py --as-of 2026-10-06                       # latest registered version
    .venv/bin/python src/models/score.py --as-of 2026-10-06 --model-version 7     # or an alias, e.g. Production
    score(as_of, brand_id, lgbm_run_id, cox_run_id)                               # the pipeline: exact runs

The model version is a version of the registered LightGBM model churn_lightgbm_classifier_brand{id}
(MLflow Models tab); the Cox PH companion is the run trained with its features. Features are rebuilt
through build_features (T6), the same code as training. The schema of every column is in
docs/schema_player_scores.md. Output, one partition per run date and brand, overwritten on a rerun
(idempotent, re-runnable for any date):

    data/03_output/player_scores/run_date={as_of}/brand_id={id}.parquet

Before scoring, the feature snapshot is checked against the training dataset (configs/eda_alerts.yaml):
a critical data-quality alert stops the run with no scores written; critical feature drift only sets
drift_flag = 1 (delivery plan).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import churn_labels  # noqa: E402
from alerts import feature_alerts, report  # noqa: E402
from build_features import LOOKBACK_DAYS, brand_label, finalise, parse_brand_id, raw_features  # noqa: E402
from segments import add_segment_features, load_segment_model  # noqa: E402
from train import (  # noqa: E402
    BRAND, HORIZONS, churn_probability, cox_input, horizon_probability, latest_run, load_calibrator,
    load_horizon_calibrators, model_input, run_info,
)

OUTPUT_DIR = PROJECT_ROOT / "data/03_output/player_scores"
REGISTERED_MODEL = "churn_lightgbm_classifier_brand{brand}"
N_DRIVERS = 3
COLUMNS = [
    "run_date", "model_version", "tenant_id", "brand_id", "player_id",
    *[f"p_churn_{h}d" for h in HORIZONS], "p_churn_60d", "median_survival_days", "median_beyond_horizon",
    "expected_ggr_30d", "value_at_risk_30d", "risk_band", "top_drivers", "segment", "drift_flag", "lower", "upper",
]


def resolve_models(brand: str, model_version: str | None, lgbm_run_id: str | None,
                   cox_run_id: str | None) -> tuple[dict, dict, str]:
    """(LightGBM run, Cox run, model_version label). The LightGBM run is the given one, or the registered
    version `model_version` (a number or an alias), or the latest optimised run; the Cox run is the given
    one or the latest Cox run trained with that LightGBM run's features."""
    client = mlflow.MlflowClient()
    name = REGISTERED_MODEL.format(brand=brand)
    if lgbm_run_id is None and model_version:
        mv = (client.get_model_version(name, model_version) if str(model_version).isdigit()
              else client.get_model_version_by_alias(name, model_version))
        lgbm_run_id = mv.run_id
    lgbm = run_info(lgbm_run_id or latest_run("lightgbm_classifier", brand).run_id)
    if cox_run_id is None:
        runs = mlflow.search_runs(experiment_names=["whizdom-churn-training"],
                                  filter_string=f"params.model_id = 'cox_ph' and params.feature_source = '{lgbm['run_name']}'",
                                  order_by=["attributes.start_time DESC"], max_results=1)
        if runs.empty:
            raise LookupError(f"no Cox PH run trained with the features of {lgbm['run_name']}")
        cox_run_id = runs.iloc[0].run_id
    cox = run_info(cox_run_id)
    versions = client.search_model_versions(f"run_id = '{lgbm['run_id']}'")
    label = f"{name}/v{versions[0].version}" if versions else lgbm["run_name"]
    return lgbm, cox, label


def top_drivers(model, rows: pd.DataFrame, columns: list[str], n: int = N_DRIVERS) -> list[str]:
    """The `n` features that move each player's LightGBM score the most (TreeSHAP, log-odds), with their
    sign: + pushes towards churn. brandId is left out (constant within a brand)."""
    contrib = model.predict(model_input(rows, columns), pred_contrib=True)[:, :-1]
    keep = [i for i, c in enumerate(columns) if c != BRAND]
    contrib, names = contrib[:, keep], [columns[i] for i in keep]
    top = np.argsort(-np.abs(contrib), axis=1)[:, :n]
    return ["; ".join(f"{names[j]} ({contrib[r, j]:+.2f})" for j in top[r]) for r in range(len(rows))]


def score(as_of: str, brand_id: int | str = 64, lgbm_run_id: str | None = None, cox_run_id: str | None = None,
          model_version: str | None = None, check_alerts: bool = True) -> Path:
    start = time.time()
    brand = brand_label([brand_id]) if brand_id != "basel" else "basel"
    lgbm, cox, version = resolve_models(brand, model_version, lgbm_run_id, cox_run_id)
    lgbm_model = mlflow.lightgbm.load_model(f"runs:/{lgbm['run_id']}/model")
    calibrator = load_calibrator(lgbm["run_id"])
    cox_model = joblib.load(mlflow.artifacts.download_artifacts(f"runs:/{cox['run_id']}/model/model.joblib"))
    horizon_calibrators = load_horizon_calibrators(cox["run_id"])
    if calibrator is None:
        print(f"warning: {lgbm['run_name']} has no calibrator: p_churn_60d is the raw model output")
    raw_horizons = [h for h in HORIZONS if h not in (horizon_calibrators or {})]
    if raw_horizons:
        print(f"warning: {cox['run_name']} has no calibrator for {raw_horizons} days: those p_churn are raw")

    raw = raw_features(as_of, brand_id)
    features = finalise(raw)
    drifting = set()
    if check_alerts:
        # A critical data-quality alert raises here (no scores written); feature drift only flags.
        used = set(lgbm["features"]) | set(cox["features"])
        alerts = feature_alerts(pd.read_parquet(PROJECT_ROOT / lgbm["dataset_path"]), features, model_features=used)
        report(alerts, "features", str(brand_id), as_of)
        drifting = set(alerts.loc[(alerts["check"] == "feature_drift") & (alerts["severity"] == "critical"), "brandId"])

    scoring_start = time.perf_counter()  # inference only: feature building is timed apart
    segment_model = load_segment_model(lgbm["run_id"])
    X = add_segment_features(features, segment_model) if segment_model is not None else features
    scores = pd.DataFrame({"run_date": as_of, "model_version": version, "tenant_id": features["tenant_id"],
                           "brand_id": features[BRAND], "player_id": features["player_id"]})
    scores["p_churn_60d"] = churn_probability(lgbm_model, calibrator, X, lgbm["features"])
    cox_seen = X[BRAND].isin(set(cox["brands"] or X[BRAND].unique())).to_numpy()
    horizons = pd.DataFrame(np.nan, index=X.index, columns=list(HORIZONS))
    median = np.full(len(X), np.nan)
    if cox_seen.any():
        horizons.loc[cox_seen] = horizon_probability(cox_model, horizon_calibrators, X[cox_seen]).to_numpy()
        median[cox_seen] = cox_model.predict_median(cox_input(cox_model, X[cox_seen])).to_numpy()
    for h in HORIZONS:
        # Churning now (p_churn_60d: no bet in the next 60 days) is churning within h days too.
        scores[f"p_churn_{h}d"] = np.maximum(horizons[h].to_numpy(), scores["p_churn_60d"].to_numpy())
    scores["median_survival_days"] = np.where(np.isinf(median), np.nan, median)
    scores["median_beyond_horizon"] = np.isinf(median)
    # Value: what the player brings in 30 days at the trailing-90-day pace (negative = the player won).
    scores["expected_ggr_30d"] = (raw["ggr_eur_l90d"].fillna(0) / 90 * 30).to_numpy()
    scores["value_at_risk_30d"] = scores["expected_ggr_30d"].clip(lower=0) * scores["p_churn_30d"]
    # Deciles of equal size within the brand: ties in p_churn_60d (the isotonic calibration is a step
    # function) are broken by p_churn_30d, then by player, so the top band is exactly the riskiest 10%.
    ordered = scores.sort_values(["brand_id", "p_churn_60d", "p_churn_30d", "tenant_id", "player_id"])
    position = ordered.groupby("brand_id").cumcount() + 1
    size = ordered.groupby("brand_id")["player_id"].transform("size")
    scores["risk_band"] = np.ceil(position / size * 10).astype("int64")
    scores["top_drivers"] = top_drivers(lgbm_model, X, lgbm["features"])
    scores["segment"] = segment_model.assign(features) if segment_model is not None else pd.NA
    scoring_seconds = time.perf_counter() - scoring_start
    scores["drift_flag"] = scores["brand_id"].isin(drifting).astype("int64")
    scores["lower"], scores["upper"] = np.nan, np.nan  # reserved: conformal intervals (RO5)
    scores = scores[COLUMNS].sort_values(["brand_id", "tenant_id", "player_id"]).reset_index(drop=True)

    # Delivery plan: the row count equals the active players (a bet in the 30 days up to the date).
    active = len(churn_labels.population(pd.Timestamp(as_of).date(), brand_id, LOOKBACK_DAYS))
    if len(scores) != active:
        raise ValueError(f"{len(scores):,} scores for {active:,} active players")
    partition = OUTPUT_DIR / f"run_date={as_of}"
    partition.mkdir(parents=True, exist_ok=True)
    for b, rows in scores.groupby("brand_id"):
        rows.to_parquet(partition / f"brand_id={b}.parquet", index=False)  # overwrites: idempotent

    print(f"saved {os.path.relpath(partition, PROJECT_ROOT)}/ ({len(scores):,} players = active players, "
          f"model {version}, {time.time() - start:.1f}s)")
    print(f"inference: {scoring_seconds * 100_000 / max(len(scores), 1):.2f}s per 100k players")
    print(scores[[f"p_churn_{h}d" for h in HORIZONS] + ["p_churn_60d", "expected_ggr_30d", "value_at_risk_30d"]]
          .describe().loc[["mean", "50%", "max"]].round(3).to_string())
    print(f"drift_flag: {int(scores['drift_flag'].max())} | example drivers: {scores['top_drivers'].iloc[0]}")
    return partition


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score every active player as of one date (player_scores).")
    parser.add_argument("--as-of", required=True, help="the run date, YYYY-MM-DD")
    parser.add_argument("--brand-id", type=parse_brand_id, default=64, help="brandId, or 'basel' for every brand")
    parser.add_argument("--model-version", help="registered LightGBM version (number or alias); default: latest run")
    parser.add_argument("--lgbm-run-id", help="an exact LightGBM run instead of a version")
    parser.add_argument("--cox-run-id", help="default: the Cox run trained with the LightGBM run's features")
    parser.add_argument("--no-alerts", action="store_true", help="skip the feature checks (development only)")
    args = parser.parse_args()
    score(args.as_of, args.brand_id, args.lgbm_run_id, args.cox_run_id, args.model_version, not args.no_alerts)
