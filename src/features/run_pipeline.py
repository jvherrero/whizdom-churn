"""The full pipeline, run "as of" one date.

    .venv/bin/python src/features/run_pipeline.py --as-of 2026-10-06
    .venv/bin/python src/features/run_pipeline.py --as-of 2026-10-06 --backtest         # adds T10
    .venv/bin/python src/features/run_pipeline.py --as-of 2026-10-06 --skip-anomalies   # keep the current caps
    .venv/bin/python src/features/run_pipeline.py --cutoff-dates 2026-07-01 2026-08-01  # data steps only

AS_OF is the "today" of the run: no data after it is read. The training cutoffs are the first day
of the last N_TRAINING_CUTOFFS months whose 60-day label is complete by AS_OF.

Data
-1. update the daily caches up to AS_OF
 0. source alerts
 1. raw features per cutoff
 2. winsorisation caps (configs/winsorisation_features_brand{id}.yaml)
 3. final features, checked with pandera
 3b. Feast offline store and point-in-time check
 4. churn labels, using data up to AS_OF only
 5. survival dataset and EDA summary (docs/eda_summary_brand{id}.md)
Models (only with --as-of)
 6. LightGBM and Cox PH: training, calibration and evaluation
 7. segment and feature importance reports
 8. with --backtest: backtest (T10) and model card
 9. feature alerts and daily scores (T11) in data/03_output/player_scores/run_date=AS_OF/

Each step uses the exact file or MLflow run made by the step before.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import churn_labels
import daily_cache
from build_features import (
    LABEL_HORIZON_DAYS, MIN_CUTOFF, PLATFORM_SCORE, data_available_through, finalise,
    parse_brand_id, raw_features,
)
from alerts import benchmark_days, check_expectations, source_alerts, load_config, report
from build_survival_dataset import DEFAULT_CUTOFF_DATES, PROJECT_ROOT, build_survival_dataset
from winsorisation import write_feature_caps


N_TRAINING_CUTOFFS = 12
MIN_TRAINING_CUTOFFS = 4  # 1 train, 1 validation and 2 test months


def training_cutoffs(as_of: dt.date) -> list[str]:
    """Last N_TRAINING_CUTOFFS month starts whose 60-day label is complete by `as_of`."""
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
    # Pandera checks on the training features. A critical alert stops the run.
    snapshot_alerts = pd.DataFrame(
        [{"stage": "training_features", "table": "feature_snapshot", **a}
         for a in check_expectations(features, "feature_snapshot", load_config()["expectations"])],
        columns=["stage", "check", "severity", "operator", "brandId", "table", "column", "value", "benchmark",
                 "threshold", "message"],
    )
    report(snapshot_alerts, "training_features", str(brand_id), str(data_end or cutoff_dates[-1]))
    if churn_labels.VARIANT:
        # The feature store serves the official model: a sensitivity run must not replace its rows.
        print(f"[3b] skipped: feature store is for the official {churn_labels.DEFAULT_HORIZON_DAYS}-day run")
    else:
        print("[3b] feature store: offline snapshot + Feast registry")
        import feature_store
        feature_store.publish(features)
    # Keep the platform's churn score as a reference column (same row order as raw).
    features[PLATFORM_SCORE] = raw["churn_score"].astype("float64").to_numpy()

    print("[4-5] churn labels + survival dataset")
    dataset = build_survival_dataset(features, data_end)
    sys.path.insert(0, str(PROJECT_ROOT / "eda"))
    import summary as eda_summary
    eda_summary.run(dataset)  
    return dataset


def run_pipeline(
    as_of: str | None = None, cutoff_dates: list[str] | None = None, brand_id: int | str = 64,
    skip_anomalies: bool = False, skip_alerts: bool = False, backtest: bool = False,
) -> Path:
    """Run the pipeline. Without `as_of` only the data steps run; with it, all steps run and the
    scores file is returned. A critical alert stops the run, except feature drift."""
    start = time.time()
    if as_of is None:
        dataset = build_training_data(cutoff_dates or DEFAULT_CUTOFF_DATES, brand_id, skip_anomalies)
        print(f"data pipeline done in {time.time() - start:.1f}s")
        return dataset

    as_of_date = dt.date.fromisoformat(as_of)
    # Update the daily caches first.
    print(f"[-1] daily caches up to {as_of}")
    daily_cache.top_up(as_of_date, brand_id)
    available = data_available_through()
    if as_of_date > available:
        raise ValueError(f"AS_OF {as_of} is after the last day with data ({available})")
    cutoffs = cutoff_dates or training_cutoffs(as_of_date)
    late = [c for c in cutoffs if dt.date.fromisoformat(c) + dt.timedelta(days=LABEL_HORIZON_DAYS) > as_of_date]
    if late:
        raise ValueError(f"the {LABEL_HORIZON_DAYS}-day label of {late} does not fit before AS_OF {as_of}")
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
        evaluate(run_id)  # metrics with bootstrap CIs

    print("[7] reports")
    segments.run_report(dataset, lgbm_run)
    feature_importance.run(lgbm_run, cox_run)

    if backtest:
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
