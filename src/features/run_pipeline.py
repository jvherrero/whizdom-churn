"""The whole project in series, "as of" one date:

    .venv/bin/python src/features/run_pipeline.py --as-of 2026-10-04
    .venv/bin/python src/features/run_pipeline.py --as-of 2026-10-04 --skip-anomalies   # keep the current caps
    .venv/bin/python src/features/run_pipeline.py --cutoff-dates 2026-07-18 2026-08-05  # data steps only

AS_OF is the "today" of the run: nothing after it is read. The training cutoffs come from it:
the last one is AS_OF - 60 days (the last cutoff whose 60-day label fits), then one every
CUTOFF_STEP_DAYS days back, N_TRAINING_CUTOFFS in total, never before MIN_CUTOFF.

Alerts (configs/eda_alerts.yaml; critical stops the run)
 0. landing alerts: the AS_OF day vs the days the training features are built from
Data
 1. raw features per cutoff (S3 to EUR, cached), not winsorised, not scaled
 2. anomaly study on them: anomaly table + configs/winsorisation_features_brand{id}.yaml
 3. training features: winsorised + sign-log, checked with the pandera feature snapshot suite
 4. churn labels for exactly those rows, looking at data up to AS_OF only
 5. training dataset (features joined with labels)
Models (only with --as-of)
 6. LightGBM (churn probability) and Cox PH (churn day), with segments, selection and calibration
 7. reports: docs/player_segments_brand{id}.md and docs/feature_importance_brand{id}.md
 8. feature alerts: the snapshot as of AS_OF vs the training dataset (PSI, players, expectations)
 9. scores of every player as of AS_OF: data/03_output/player_scores_*.parquet

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
from anomalies import FEATURES_SPEC, detect_anomalies, save_anomaly_outputs
from build_features import (
    DATA_AVAILABLE_THROUGH, HISTORY_START, LABEL_HORIZON_DAYS, MIN_CUTOFF, build_feature_store, parse_brand_id,
)
from alerts import benchmark_days, check_expectations, feature_alerts, landing_alerts, load_config, report
from build_labels import build_training_labels
from build_training_features import DEFAULT_CUTOFF_DATES, PROJECT_ROOT, build_training_features
from create_dataset import create_dataset

sys.path.insert(0, str(PROJECT_ROOT / "eda"))

N_TRAINING_CUTOFFS = 3
CUTOFF_STEP_DAYS = 9


def training_cutoffs(as_of: dt.date) -> list[str]:
    """The latest cutoffs whose 60-day label fits before `as_of`, CUTOFF_STEP_DAYS apart."""
    last = as_of - dt.timedelta(days=LABEL_HORIZON_DAYS)
    cutoffs = [last - dt.timedelta(days=CUTOFF_STEP_DAYS * i) for i in range(N_TRAINING_CUTOFFS)]
    cutoffs = sorted(c for c in cutoffs if c >= MIN_CUTOFF)
    if len(cutoffs) < 2:
        raise ValueError(f"as of {as_of} there are {len(cutoffs)} training cutoff(s) between {MIN_CUTOFF} and "
                         f"{last}; the out-of-time test needs at least 2")
    return [str(c) for c in cutoffs]


def build_training_data(
    cutoff_dates: list[str], brand_id: int | str, skip_anomalies: bool, use_cache: bool, data_end: str | None = None,
) -> Path:
    """Steps 1-5. Returns the training dataset path."""
    if skip_anomalies:
        print("[1-2] skipped, using the current winsorisation YAML")
    else:
        print("[1] raw features")
        raw = pd.concat(
            [build_feature_store(c, brand_id, use_cache=use_cache, full=True, scaled=False, winsorise=False)
             for c in cutoff_dates],
            ignore_index=True,
        )
        print("[2] anomaly study, one per brand")
        # Caps (p99.5) and outlier thresholds are brand-specific: never pooled across brands.
        for brand, rows in raw.groupby("brandId"):
            table, flagged, config = detect_anomalies(rows, FEATURES_SPEC, brand_id=int(brand))
            for path in save_anomaly_outputs(table, flagged, config, "features", int(brand)).values():
                print(f"saved {path.relative_to(PROJECT_ROOT)}")
            structural = table[table["method"].str.startswith("structural") & (table["n_flagged"] > 0)]
            if not structural.empty:
                print(f"brandId {brand}:")
                print(structural[["column", "method", "n_flagged"]].to_string(index=False))

    print("[3] training features")
    features_path = build_training_features(cutoff_dates, brand_id)
    # Expectations on the training features: an alert list like the others, critical stops the run.
    training_features = pd.read_parquet(features_path)
    snapshot_alerts = pd.DataFrame(
        [{"stage": "training_features", "table": "feature_snapshot", **a}
         for a in check_expectations(training_features, "feature_snapshot", load_config()["expectations"])],
        columns=["stage", "check", "severity", "operator", "brandId", "table", "column", "value", "benchmark",
                 "threshold", "message"],
    )
    report(snapshot_alerts, "training_features", str(brand_id), str(data_end or cutoff_dates[-1]))

    print("[4] labels")
    build_training_labels(features_path, data_end=data_end)

    print("[5] training dataset")
    create_dataset(features_path)
    return features_path.with_name(features_path.name.replace("train_features_base", "train_dataset"))


def run_pipeline(
    as_of: str | None = None, cutoff_dates: list[str] | None = None, brand_id: int | str = 64,
    skip_anomalies: bool = False, use_cache: bool = True, skip_alerts: bool = False,
) -> Path:
    """Without `as_of`: data steps 1-5 only. With it: every step, returns the player scores file.
    Data alerts (configs/eda_alerts.yaml) run on the landing data first and on the scoring snapshot
    before scoring; a critical alert stops the run (alerts.CriticalAlert)."""
    start = time.time()
    if as_of is None:
        dataset = build_training_data(cutoff_dates or DEFAULT_CUTOFF_DATES, brand_id, skip_anomalies, use_cache)
        print(f"data pipeline done in {time.time() - start:.1f}s")
        return dataset

    as_of_date = dt.date.fromisoformat(as_of)
    if as_of_date > DATA_AVAILABLE_THROUGH:
        raise ValueError(f"AS_OF {as_of} is after DATA_AVAILABLE_THROUGH ({DATA_AVAILABLE_THROUGH}): "
                         "update it in src/features/build_features.py when new complete days have landed")
    cutoffs = cutoff_dates or training_cutoffs(as_of_date)
    late = [c for c in cutoffs if dt.date.fromisoformat(c) + dt.timedelta(days=LABEL_HORIZON_DAYS) > as_of_date]
    if late:
        raise ValueError(f"the 60-day label of {late} does not fit before AS_OF {as_of}")
    print(f"as of {as_of}: training cutoffs {cutoffs}")

    print(f"[-1] daily tables: S3 landing -> daily per-player tables, only the days not built yet (up to {as_of})")
    import daily_tables
    daily_tables.build(HISTORY_START, as_of_date, verbose=False)

    if skip_alerts:
        print("[0] landing alerts SKIPPED (--skip-alerts): use only while developing")
    else:
        print("[0] landing alerts: current days vs the days the training features are built from")
        report(landing_alerts(brand_id, as_of, benchmark_days(cutoffs), use_cache=use_cache),
               "landing", str(brand_id), as_of)

    dataset = build_training_data(cutoffs, brand_id, skip_anomalies, use_cache, data_end=as_of)

    sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
    import feature_importance
    import score
    import segments
    from train import train

    print("[6] models")
    lgbm_run = train("lightgbm_classifier", dataset)
    cox_run = train("cox_ph", dataset, features_from_run=lgbm_run)  # same features as LightGBM

    print("[7] reports")
    segments.run_report(dataset)
    feature_importance.run(lgbm_run, cox_run)

    if not skip_alerts:
        print(f"[8] feature alerts: snapshot as of {as_of} vs the training dataset")
        from train import run_info
        used = set(run_info(lgbm_run)["features"]) | set(run_info(cox_run)["features"])
        report(feature_alerts(pd.read_parquet(dataset), build_feature_store(as_of, brand_id, use_cache=use_cache),
                              model_features=used), "features", str(brand_id), as_of)

    print(f"[9] scores as of {as_of}")
    scores = score.score(as_of, brand_id, lgbm_run, cox_run)
    print(f"pipeline done in {(time.time() - start) / 60:.1f} min")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The whole project in series, as of one date.")
    parser.add_argument("--as-of", help="the run's 'today' (YYYY-MM-DD); without it, only the data steps run")
    parser.add_argument("--cutoff-dates", nargs="+", help="training cutoffs instead of the ones derived from --as-of")
    parser.add_argument("--brand-id", type=parse_brand_id, default=64, help="brandId, or 'basel' for every brand")
    parser.add_argument("--skip-anomalies", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--skip-alerts", action="store_true", help="skip the data alerts (development only)")
    args = parser.parse_args()
    run_pipeline(args.as_of, args.cutoff_dates, args.brand_id, args.skip_anomalies, not args.no_cache,
                 args.skip_alerts)
