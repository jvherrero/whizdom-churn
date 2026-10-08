"""The whole project in series, "as of" one date:

    .venv/bin/python src/features/run_pipeline.py --as-of 2026-10-06
    .venv/bin/python src/features/run_pipeline.py --as-of 2026-10-06 --backtest         # + T10, about 45 min more
    .venv/bin/python src/features/run_pipeline.py --as-of 2026-10-06 --skip-anomalies   # keep the current caps
    .venv/bin/python src/features/run_pipeline.py --cutoff-dates 2026-07-01 2026-08-01  # data steps only

Source: the S3 data lake, org/40-gold (src/features/build_features.py). AS_OF is the "today" of the run: nothing
after it is read. The training cutoffs come from it: N_TRAINING_CUTOFFS monthly cutoffs (the first
day of each month), the last one being the latest whose 60-day label ends by AS_OF, never before
MIN_CUTOFF. train.py splits them by month: train = months 1-9, validation = month 10, test = 11-12.

Data
-1. daily caches topped up to AS_OF (activity, and financial + payments for the brand)
 0. source alerts: the caches on AS_OF vs the training days (rows, nulls, daily totals, expectations)
 1. raw features per cutoff (in EUR), not winsorised, not scaled
 2. winsorisation caps from them: configs/winsorisation_features_brand{id}.yaml (the full
    anomaly study is EDA: make anomalies-features)
 3. training features: the same raw rows winsorised + sign-log, checked with the pandera feature
    snapshot suite
 4. churn labels for exactly those rows, looking at data up to AS_OF only
 5. survival dataset: features joined with the labels (build_survival_dataset.py), then EDA stage 4:
    the snapshot vs the previous dataset of the brand (eda/summary.py, docs/eda_summary_brand{id}.md)
Models (only with --as-of)
 6. LightGBM (churn probability) and Cox PH (churn day), with segments, selection and calibration,
    then evaluated with 95% bootstrap intervals logged into each run (src/evaluation)
 7. reports: docs/player_segments_brand{id}.md and docs/feature_importance_brand{id}.md
 8. with --backtest: the rolling-origin backtest (T10), docs/backtest_v0_brand{id}.md; then the model
    card from the runs and the latest backtest of the brand (docs/model_card_v0_brand{id}.html)
 9. scores of every active player as of AS_OF (T11, src/models/score.py), after the feature alerts
    (snapshot vs training dataset): a critical data-quality alert stops the run with no scores written,
    critical feature drift sets drift_flag (delivery plan) -> data/03_output/player_scores/run_date=AS_OF/

Each step gets the exact file or MLflow run the previous one produced, never "the latest one".
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import daily_cache
from build_features import (
    LABEL_HORIZON_DAYS, MIN_CUTOFF, PLATFORM_SCORE, data_available_through, finalise,
    parse_brand_id, raw_features,
)
from alerts import benchmark_days, check_expectations, source_alerts, load_config, report
from build_survival_dataset import DEFAULT_CUTOFF_DATES, PROJECT_ROOT, build_survival_dataset
from winsorisation import write_feature_caps


N_TRAINING_CUTOFFS = 12
MIN_TRAINING_CUTOFFS = 4  # train.split_rows: at least 1 train month, 1 validation month, 2 test months


def training_cutoffs(as_of: dt.date) -> list[str]:
    """The N_TRAINING_CUTOFFS latest month starts whose 60-day label ends by `as_of`."""
    latest = as_of - dt.timedelta(days=LABEL_HORIZON_DAYS)
    last = latest.replace(day=1)
    months = pd.date_range(end=pd.Timestamp(last), periods=N_TRAINING_CUTOFFS, freq="MS")
    cutoffs = [m.date() for m in months if m.date() >= MIN_CUTOFF]
    if len(cutoffs) < MIN_TRAINING_CUTOFFS:
        raise ValueError(f"as of {as_of} there are {len(cutoffs)} training cutoff(s) between {MIN_CUTOFF} and "
                         f"{last}; the temporal split needs at least {MIN_TRAINING_CUTOFFS}")
    return [str(c) for c in cutoffs]


def build_training_data(
    cutoff_dates: list[str], brand_id: int | str, skip_anomalies: bool, data_end: str | None = None,
) -> Path:
    """Steps 1-5. Returns the training dataset path."""
    print("[1] raw features")
    raw = pd.concat([raw_features(c, brand_id) for c in cutoff_dates], ignore_index=True)
    if skip_anomalies:
        print("[2] skipped, using the current winsorisation YAML")
    else:
        print("[2] winsorisation caps (p99.5 in EUR), one YAML per brand")
        for path in write_feature_caps(raw):
            print(f"saved {path.relative_to(PROJECT_ROOT)}")

    print("[3] training features")
    features = finalise(raw)
    # Expectations on the training features: an alert list like the others, critical stops the run.
    snapshot_alerts = pd.DataFrame(
        [{"stage": "training_features", "table": "feature_snapshot", **a}
         for a in check_expectations(features, "feature_snapshot", load_config()["expectations"])],
        columns=["stage", "check", "severity", "operator", "brandId", "table", "column", "value", "benchmark",
                 "threshold", "message"],
    )
    report(snapshot_alerts, "training_features", str(brand_id), str(data_end or cutoff_dates[-1]))
    # The platform's churn score rides along as a reference column (finalise keeps the row order of raw).
    features[PLATFORM_SCORE] = raw["churn_score"].astype("float64").to_numpy()

    print("[4-5] churn labels + survival dataset")
    dataset = build_survival_dataset(features, data_end)
    sys.path.insert(0, str(PROJECT_ROOT / "eda"))
    import summary as eda_summary
    eda_summary.run(dataset)  # EDA stage 4: reported, never stops the run
    return dataset


def run_pipeline(
    as_of: str | None = None, cutoff_dates: list[str] | None = None, brand_id: int | str = 64,
    skip_anomalies: bool = False, skip_alerts: bool = False, backtest: bool = False,
) -> Path:
    """Without `as_of`: data steps 1-5 only. With it: every step, returns the player scores file.
    Data alerts (configs/eda_alerts.yaml): expectations on the training features and drift of the
    scoring snapshot before scoring; a critical alert stops the run (alerts.CriticalAlert), except
    feature drift, which is only reported. The source tables themselves are checked by `make profile` (T2)."""
    start = time.time()
    if as_of is None:
        dataset = build_training_data(cutoff_dates or DEFAULT_CUTOFF_DATES, brand_id, skip_anomalies)
        print(f"data pipeline done in {time.time() - start:.1f}s")
        return dataset

    as_of_date = dt.date.fromisoformat(as_of)
    # The daily caches are topped up first (only the missing days, plus the last 7, which the source reprocesses).
    print(f"[-1] daily caches up to {as_of}")
    daily_cache.top_up(as_of_date, brand_id)
    available = data_available_through()
    if as_of_date > available:
        raise ValueError(f"AS_OF {as_of} is after the last day with data ({available})")
    cutoffs = cutoff_dates or training_cutoffs(as_of_date)
    late = [c for c in cutoffs if dt.date.fromisoformat(c) + dt.timedelta(days=LABEL_HORIZON_DAYS) > as_of_date]
    if late:
        raise ValueError(f"the 60-day label of {late} does not fit before AS_OF {as_of}")
    print(f"as of {as_of}: training cutoffs {cutoffs}")
    if not skip_alerts:
        print(f"[0] source alerts: caches on {as_of} vs the training days")
        report(source_alerts(brand_id, as_of, benchmark_days(cutoffs)), "source", str(brand_id), as_of)

    dataset = build_training_data(cutoffs, brand_id, skip_anomalies, data_end=as_of)

    sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import feature_importance
    import score
    import segments
    from train import train

    print("[6] models")
    lgbm_run = train("lightgbm_classifier", dataset)
    cox_run = train("cox_ph", dataset, features_from_run=lgbm_run)  # same features as LightGBM
    from evaluation.evaluate import evaluate
    for run_id in (lgbm_run, cox_run):
        evaluate(run_id)  # results_* metrics with bootstrap CIs, in the run

    print("[7] reports")
    segments.run_report(dataset, lgbm_run)
    feature_importance.run(lgbm_run, cox_run)

    if backtest:
        # Optional (about 45 min): the rolling-origin backtest of T10, in the plan's order (before scoring).
        print(f"[8] rolling-origin backtest as of {as_of}")
        import backtest as rolling_origin
        rolling_origin.run(as_of=as_of, brand_id=brand_id)

    import model_card
    model_card.run(brand_id, lgbm_run, cox_run)

    print(f"[9] feature alerts + scores as of {as_of}")
    scores = score.score(as_of, brand_id, lgbm_run, cox_run, check_alerts=not skip_alerts)
    print(f"pipeline done in {(time.time() - start) / 60:.1f} min")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The whole project in series, as of one date.")
    parser.add_argument("--as-of", help="the run's 'today' (YYYY-MM-DD); without it, only the data steps run")
    parser.add_argument("--cutoff-dates", nargs="+", help="training cutoffs instead of the ones derived from --as-of")
    parser.add_argument("--brand-id", type=parse_brand_id, default=64, help="brandId, or 'basel' for every brand")
    parser.add_argument("--skip-anomalies", action="store_true")
    parser.add_argument("--skip-alerts", action="store_true", help="skip the data alerts (development only)")
    parser.add_argument("--backtest", action="store_true", help="also run the rolling-origin backtest (T10) at the end")
    args = parser.parse_args()
    run_pipeline(args.as_of, args.cutoff_dates, args.brand_id, args.skip_anomalies, args.skip_alerts, args.backtest)
