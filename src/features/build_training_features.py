"""
Call:
    output_path = build_training_features(cutoff_dates=["2026-07-18", "2026-07-27", "2026-08-05"], brand_id=64)

    

    MLflow: .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db


"""

from __future__ import annotations

import argparse
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
    FEATURE_VERSION, ID_COLUMNS, LOOKBACK_DAYS, WINSOR_CONFIG_PATH, build_feature_store, parse_brand_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# MIN_CUTOFF (2026-07-18) to the last cutoff whose 60-day label fits in the data (2026-08-05).
DEFAULT_CUTOFF_DATES = ["2026-07-18", "2026-07-27", "2026-08-05"]
MLFLOW_EXPERIMENT = "whizdom-churn-training-features"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def build_training_features(cutoff_dates: list[str] = DEFAULT_CUTOFF_DATES, brand_id: int | str = 64) -> Path:
    start_time = time.time()
    snapshots = [build_feature_store(cutoff_date=cutoff_date, brand_id=brand_id) for cutoff_date in cutoff_dates]
    full_dataset = pd.concat(snapshots, ignore_index=True)
    execution_time_s = round(time.time() - start_time, 1)


    cutoffs_str = "_".join(cutoff_dates)
    timestamp_unix = int(time.time())
    run_name = f"train_features_base_brand{brand_id}_{cutoffs_str}_{timestamp_unix}"

    output_path = PROJECT_ROOT / "data/processed" / f"train_features_base_{brand_id}_{cutoffs_str}_{timestamp_unix}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_dataset.to_parquet(output_path, index=False)
    print(f"saved {output_path.relative_to(PROJECT_ROOT)} ({len(full_dataset):,} rows, {full_dataset.shape[1]} columns)")
    print(f"execution time: {execution_time_s}s")


    feature_columns = [c for c in full_dataset.columns if c not in ID_COLUMNS]
    n_nulls = int(full_dataset.isna().sum().sum())
    n_duplicate_player_cutoff = int(full_dataset.duplicated(subset=["cutoff_date", "brandId", "partyId"]).sum())
    file_size_mb = round(output_path.stat().st_size / (1024 * 1024), 2)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("timestamp_unix", timestamp_unix)
        mlflow.set_tag("timestamp_iso", datetime.fromtimestamp(timestamp_unix, tz=timezone.utc).isoformat())
        mlflow.set_tag("git_commit", _git_commit())
        mlflow.set_tag("user", getpass.getuser())
        mlflow.set_tag("host", socket.gethostname())
        mlflow.set_tag("source_script", "src/features/build_training_features.py")
        mlflow.set_tag("brand_id", brand_id)


        mlflow.log_param("cutoff_dates", cutoff_dates)
        mlflow.log_param("brand_id", brand_id)
        mlflow.log_param("operator", sorted(full_dataset["operator"].unique().tolist()))
        mlflow.log_param("output_path", str(output_path.relative_to(PROJECT_ROOT)))
        mlflow.log_param("feature_columns", feature_columns)
        mlflow.log_param("lookback_days", LOOKBACK_DAYS)
        mlflow.log_param("feature_version", FEATURE_VERSION)
        mlflow.log_param("winsorisation_config", [  # one caps file per brand
            str(Path(str(WINSOR_CONFIG_PATH).format(brand_id=b)).relative_to(PROJECT_ROOT))
            for b in sorted(full_dataset["brandId"].unique())
        ])

        mlflow.log_metric("n_cutoffs", len(cutoff_dates))
        mlflow.log_metric("n_rows", len(full_dataset))
        mlflow.log_metric("n_columns", full_dataset.shape[1])
        mlflow.log_metric("n_features", len(feature_columns))
        mlflow.log_metric("n_nulls", n_nulls)
        mlflow.log_metric("n_duplicate_player_cutoff", n_duplicate_player_cutoff)
        mlflow.log_metric("file_size_mb", file_size_mb)
        mlflow.log_metric("execution_time_s", execution_time_s)

        dataset = mlflow.data.from_pandas(
            full_dataset, source=str(output_path.relative_to(PROJECT_ROOT)), name=run_name
        )
        mlflow.log_input(dataset, context="training")
        # Copies the actual parquet into MLflow's artifact store (local
        # mlruns/.../artifacts/ folder, already in .gitignore)
        mlflow.log_artifact(str(output_path))
        print(f"MLflow run '{run_name}' logged (timestamp_unix={timestamp_unix})")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature snapshots for several past cutoffs, logged to MLflow.")
    parser.add_argument("--cutoff-dates", nargs="+", default=DEFAULT_CUTOFF_DATES)
    parser.add_argument("--brand-id", type=parse_brand_id, default=64, help="brandId, or 'basel' for every brand")
    args = parser.parse_args()
    build_training_features(args.cutoff_dates, args.brand_id)
