"""Survival dataset (T7): the feature snapshots of several cutoffs joined with their churn targets.

    .venv/bin/python src/features/build_survival_dataset.py --cutoff-dates 2026-07-20 2026-07-29 2026-08-07
    path = build_survival_dataset(training_features(cutoffs, brand_id=64), data_end="2026-10-06")

One row per (cutoff_date, tenant_id, player_id): the final feature vector (build_feature_store) plus
    event_60d                       1 = no bet in the 60 days after the cutoff (the classifier target)
    duration_days, event_observed   time to churn and the right-censoring flag (Cox PH)
    churn_within_7d / 14d / 30d     1 = the churn starts within 7 / 14 / 30 days; empty when not known yet
Only rows whose 60-day label fits before `data_end` are kept. The dataset is saved as
data/processed/train_dataset_{brand}_{cutoffs}_{ts}.parquet and logged as one MLflow run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import (  
    ID_COLUMNS, LOOKBACK_DAYS, PLATFORM_SCORE, brand_label, build_feature_store, data_available_through,
    parse_brand_id,
)
import churn_labels  
from winsorisation import WINSOR_CONFIG_PATH 

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
# 12 monthly cutoffs, all with a complete 60-day label in the history (the default of the CLI).
DEFAULT_CUTOFF_DATES = [str(d.date()) for d in pd.date_range("2025-09-01", "2026-08-01", freq="MS")]
MLFLOW_EXPERIMENT = "whizdom-churn-training-features"
KEY = ["cutoff_date", "tenant_id", "player_id"]
LABEL_COLUMNS = churn_labels.LABEL_COLUMNS
TARGET = "event_60d"
WITHIN_COLUMNS = [f"churn_within_{h}d" for h in churn_labels.WITHIN_DAYS]
SURVIVAL_TARGETS = ["duration_days", "event_observed"]  # for the Cox PH model


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def training_features(cutoff_dates: list[str], brand_id: int | str = 64) -> pd.DataFrame:
    """The final feature vector at every cutoff, stacked."""
    return pd.concat([build_feature_store(c, brand_id) for c in cutoff_dates], ignore_index=True)


def _labels(features: pd.DataFrame, data_end: dt.date) -> pd.DataFrame:
    """Every churn target of every feature row, looking at data up to `data_end` only. A 60-day
    label whose window runs past `data_end` is not a real 0/1: churn_targets leaves event_60d empty."""
    return pd.concat(
        [churn_labels.churn_targets(rows[["tenant_id", "player_id"]], cutoff, int(brand), data_end)
         for (cutoff, brand), rows in features.groupby(["cutoff_date", "brandId"])],
        ignore_index=True,
    )


def with_labels(features: pd.DataFrame, data_end: dt.date) -> tuple[pd.DataFrame, int]:
    """`features` joined with their targets as seen on `data_end`, only the rows whose 60-day label is
    confirmable; and how many were not."""
    labels = _labels(features, data_end)
    dataset = features.merge(labels[KEY + LABEL_COLUMNS], on=KEY, how="left", validate="one_to_one",
                             indicator=True)
    n_without_label = int((dataset["_merge"] != "both").sum())
    if n_without_label:
        raise ValueError(f"{n_without_label:,} feature rows have no label")
    confirmed = dataset[TARGET].notna()
    dataset = dataset[confirmed].drop(columns="_merge")
    dataset[[TARGET, *SURVIVAL_TARGETS]] = dataset[[TARGET, *SURVIVAL_TARGETS]].astype("int64")
    dataset[WITHIN_COLUMNS] = dataset[WITHIN_COLUMNS].astype("Int64")  # empty where not known yet
    return dataset.reset_index(drop=True), int((~confirmed).sum())


def build_survival_dataset(features: pd.DataFrame, data_end: str | dt.date | None = None) -> Path:
    """Join `features` (training_features()) with their targets, keep the confirmable labels, save
    and log to MLflow. `data_end` is the last day the labels may look at (default: the last complete
    data day); a run "as of" an earlier date passes that date, so it never sees later data."""
    start_time = time.time()
    data_end = pd.to_datetime(data_end).date() if data_end else data_available_through()
    dataset, n_unconfirmed = with_labels(features, data_end)

    cutoff_dates = sorted(str(c) for c in features["cutoff_date"].unique())
    brand = brand_label(features["brandId"].unique())
    timestamp_unix = int(time.time())
    suffix = f"{brand}_{'_'.join(cutoff_dates)}_{timestamp_unix}"
    output_path = PROCESSED_DIR / f"train_dataset_{suffix}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    execution_time_s = round(time.time() - start_time, 1)
    print(f"saved {output_path.relative_to(PROJECT_ROOT)} ({len(dataset):,} rows, {dataset.shape[1]} columns)")
    print(f"dropped {n_unconfirmed:,} rows whose label is not confirmable yet")
    _log_run(dataset, features, output_path, f"train_dataset_brand{suffix}", cutoff_dates, data_end,
             n_unconfirmed, timestamp_unix, execution_time_s)
    return output_path


def _log_run(dataset, features, output_path, run_name, cutoff_dates, data_end, n_unconfirmed, timestamp_unix,
             execution_time_s) -> None:
    feature_columns = [c for c in features.columns if c not in ID_COLUMNS and c != PLATFORM_SCORE]
    relative = str(output_path.relative_to(PROJECT_ROOT))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "timestamp_unix": timestamp_unix,
            "timestamp_iso": datetime.fromtimestamp(timestamp_unix, tz=timezone.utc).isoformat(),
            "git_commit": git_commit(), "user": getpass.getuser(), "host": socket.gethostname(),
            "source_script": "src/features/build_survival_dataset.py",
            "brand_id": brand_label(dataset["brandId"].unique()),
        })
        mlflow.log_params({
            "cutoff_dates": cutoff_dates,
            "brand_id": sorted(dataset["brandId"].unique().tolist()),
            "tenants": sorted(features["tenant_id"].unique().tolist()),
            "data_end": str(data_end),
            "output_path": relative,
            "lookback_days": LOOKBACK_DAYS,
            "winsorisation_config": [str(Path(str(WINSOR_CONFIG_PATH).format(brand_id=b)).relative_to(PROJECT_ROOT))
                                     for b in sorted(features["brandId"].unique())],
            "target": TARGET,
            "survival_targets": SURVIVAL_TARGETS,
            "split_group": "tenant_id + player_id",
            "feature_columns": feature_columns,
        })
        churn_by_cutoff = dataset.groupby("cutoff_date")[TARGET].mean()
        events = dataset["event_observed"] == 1
        mlflow.log_metrics({
            "n_cutoffs": len(cutoff_dates),
            "n_rows": len(dataset),
            "n_features": len(feature_columns),
            "n_dropped_unconfirmed_label": n_unconfirmed,
            "n_players": int(dataset[["tenant_id", "player_id"]].drop_duplicates().shape[0]),
            "n_players_multi_cutoff": int((dataset.groupby(["tenant_id", "player_id"])["cutoff_date"].nunique() > 1).sum()),
            "n_nulls": int(dataset.isna().sum().sum()),
            "n_duplicate_player_cutoff": int(dataset.duplicated(subset=KEY).sum()),
            "churn_rate": round(float(dataset[TARGET].mean()), 4),
            "survival_event_rate": round(float(events.mean()), 4),
            "censoring_rate": round(float(1 - events.mean()), 4),
            "survival_max_event_day": int(dataset.loc[events, "duration_days"].max()) if events.any() else 0,
            # Churn starting within 7 / 14 / 30 days, over the rows where it is known; and how many are known.
            **{f"{c}_rate": round(float(dataset[c].mean()), 4) for c in WITHIN_COLUMNS if dataset[c].notna().any()},
            **{f"{c}_known_share": round(float(dataset[c].notna().mean()), 4) for c in WITHIN_COLUMNS},
            "file_size_mb": round(output_path.stat().st_size / (1024 * 1024), 2),
            "execution_time_s": execution_time_s,
            **{f"churn_rate_{c}": round(float(r), 4) for c, r in churn_by_cutoff.items()},
            **{f"rows_{c}": int(n) for c, n in dataset.groupby("cutoff_date").size().items()},
            # Null share of every feature: the deposit features are empty before March 2026 by design.
            **{f"null_share__{c}": round(float(dataset[c].isna().mean()), 4) for c in feature_columns},
        })
        mlflow.log_input(mlflow.data.from_pandas(dataset, source=relative, targets=TARGET, name=run_name),
                         context="training")
        mlflow.log_artifact(str(output_path))
        print(f"MLflow run '{run_name}' logged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature snapshots of several cutoffs joined with their churn targets.")
    parser.add_argument("--cutoff-dates", nargs="+", default=DEFAULT_CUTOFF_DATES)
    parser.add_argument("--brand-id", type=parse_brand_id, default=64, help="brandId, or 'basel' for every brand")
    parser.add_argument("--data-end", help="last day the labels may look at (default: the last complete data day)")
    args = parser.parse_args()
    build_survival_dataset(training_features(args.cutoff_dates, args.brand_id), args.data_end)
