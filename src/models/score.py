"""Score every player of a brand as of one date with the latest baseline models.

    .venv/bin/python src/models/score.py --as-of 2026-08-05
    .venv/bin/python src/models/score.py --as-of 2026-08-05 --lgbm-run-id <id> --cox-run-id <id>

One row per player, saved to data/03_output/player_scores_{brand}_{as_of}_{unix_ts}.parquet:

    risk_band                       1-10: decile of churn_probability_60d within the day's population of the
                                    brand (10 = riskiest 10%); no threshold is baked in, the CRM team picks bands
    churn_probability_60d           LightGBM: probability of no bet in the next 60 days,
                                    isotonic-calibrated when the run has a calibrator
    median_survival_days            Cox PH: days after the cutoff until the predicted churn day,
                                    where the player's survival curve falls to 0.5. 0 means the
                                    player has more likely than not already stopped.
    median_beyond_horizon           True when the curve never falls to 0.5 inside the days the
                                    model has seen churn on (survival_horizon_days); then
                                    median_survival_days is NaN, i.e. "later than the horizon"
    survival_horizon_days           the last day after a cutoff where the Cox model saw a churn
    segment                         k-means player type (docs/player_segments_brand{id}.md), when the run has one
    brand_seen_in_training          False for a brand the models never saw (generic probability, no median)

The features come from build_feature_store(), the same function (winsorised, sign-log) the
training dataset was built with.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import build_feature_store, parse_brand_id
from segments import add_segment_features, load_segment_model
from train import BRAND, churn_probability, cox_input, latest_run, load_calibrator, run_info

OUTPUT_DIR = PROJECT_ROOT / "data/03_output"


def score(as_of: str, brand_id: int | str = 64, lgbm_run_id: str | None = None, cox_run_id: str | None = None) -> Path:
    start = time.time()
    lgbm = run_info(lgbm_run_id or latest_run("lightgbm_classifier").run_id)
    cox = run_info(cox_run_id or latest_run("cox_ph").run_id)
    lgbm_model = mlflow.lightgbm.load_model(f"runs:/{lgbm['run_id']}/model")
    calibrator = load_calibrator(lgbm["run_id"])
    if calibrator is None:
        print(f"warning: {lgbm['run_name']} has no calibrator, churn_probability_60d is the raw model output")
    cox_model = joblib.load(mlflow.artifacts.download_artifacts(f"runs:/{cox['run_id']}/model/model.joblib"))

    features = build_feature_store(as_of, brand_id)
    scoring_start = time.perf_counter()  # inference only: feature building is timed apart
    # Each run assigns players to the k-means segments it was trained with.
    lgbm_X, cox_X = features, features
    segment_model = load_segment_model(lgbm["run_id"])
    if segment_model is not None:
        lgbm_X = add_segment_features(features, segment_model)
    if (seg := load_segment_model(cox["run_id"])) is not None:
        cox_X = add_segment_features(features, seg)
    missing = sorted((set(lgbm["features"]) - set(lgbm_X.columns)) | (set(cox["features"]) - set(cox_X.columns)))
    if missing:
        raise ValueError(f"the feature store has no {missing}: the models were trained on other features")

    scores = features[["cutoff_date", "brandId", "partyId"]].copy()
    # A brand the models never saw: LightGBM treats its brandId as missing and the calibration
    # falls back to the all-brands one; the stratified Cox has no baseline for it, so no median.
    trained = set(lgbm["brands"] or features[BRAND].unique())
    seen = features[BRAND].isin(trained)
    scores["brand_seen_in_training"] = seen
    if not seen.all():
        print(f"warning: brandId {sorted(set(features.loc[~seen, BRAND]))} not in the training data "
              f"({(~seen).sum():,} players): generic probability, no median_survival_days")
    scores["churn_probability_60d"] = churn_probability(lgbm_model, calibrator, lgbm_X, lgbm["features"])
    scores["probability_calibrated"] = calibrator is not None
    # Threshold and band policy: no threshold is baked into the model. risk_band is the decile of the
    # churn probability within the day's population of each brand (10 = the riskiest 10%); the CRM
    # team picks the bands it acts on.
    pct = scores.groupby(["cutoff_date", "brandId"])["churn_probability_60d"].rank(pct=True, method="average")
    scores["risk_band"] = np.ceil(pct * 10).clip(1, 10).astype("int64")
    cox_seen = cox_X[BRAND].isin(set(cox["brands"] or cox_X[BRAND].unique()))
    median = np.full(len(cox_X), np.nan)
    if cox_seen.any():
        median[cox_seen.to_numpy()] = cox_model.predict_median(cox_input(cox_model, cox_X[cox_seen])).to_numpy()
    scores["median_survival_days"] = np.where(np.isinf(median), np.nan, median)
    scoring_seconds = time.perf_counter() - scoring_start
    scores["median_beyond_horizon"] = np.isinf(median)
    scores["survival_horizon_days"] = int(cox["metrics"]["train_max_event_day"])
    if segment_model is not None:
        scores["segment"] = segment_model.assign(features)  # see docs/player_segments_brand{id}.md
    scores["lgbm_run"] = lgbm["run_name"]
    scores["cox_run"] = cox["run_name"]

    output = OUTPUT_DIR / f"player_scores_{brand_id}_{as_of}_{int(time.time())}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(output, index=False)

    p = scores["churn_probability_60d"]
    print(f"saved {os.path.relpath(output, PROJECT_ROOT)} ({len(scores):,} players, {time.time() - start:.1f}s)")
    print(f"inference: {scoring_seconds:.2f}s for {len(scores):,} players = "
          f"{scoring_seconds * 100_000 / max(len(scores), 1):.2f}s per 100k players")
    print(f"churn_probability_60d: mean {p.mean():.3f} | > 0.5: {(p > 0.5).mean():.1%}")
    bands = scores.groupby("risk_band")["churn_probability_60d"].agg(["size", "min", "max"]).round(3)
    print("risk bands (decile of the day's population):")
    print(bands.to_string())
    print(f"median_survival_days: 0 days {(scores['median_survival_days'] == 0).mean():.1%} | "
          f"1-{scores['survival_horizon_days'].iloc[0]} days {(scores['median_survival_days'] > 0).mean():.1%} | "
          f"beyond the horizon {scores['median_beyond_horizon'].mean():.1%}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn probability and median survival days per player.")
    parser.add_argument("--as-of", required=True, help="cutoff date, YYYY-MM-DD")
    parser.add_argument("--brand-id", type=parse_brand_id, default=64, help="brandId, or 'basel' for every brand")
    parser.add_argument("--lgbm-run-id", help="default: the latest lightgbm_classifier run")
    parser.add_argument("--cox-run-id", help="default: the latest cox_ph run")
    args = parser.parse_args()
    score(args.as_of, args.brand_id, args.lgbm_run_id, args.cox_run_id)
