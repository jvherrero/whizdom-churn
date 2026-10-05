"""Training dataset: a train_features_base file joined with its train_labels file.

    .venv/bin/python src/features/create_dataset.py
    dataset = create_dataset()                      # latest train_features_base file
    dataset = create_dataset("data/processed/train_features_base_64_..._<ts>.parquet")

"""

from __future__ import annotations

import argparse
import getpass
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import brand_label
from build_labels import LABEL_COLUMNS
from build_training_features import MLFLOW_EXPERIMENT, PROJECT_ROOT, _git_commit

PROCESSED_DIR = PROJECT_ROOT / "data/processed"
KEY = ["cutoff_date", "brandId", "partyId"]
TARGET = "event_60d"
SURVIVAL_TARGETS = ["duration_days", "event_observed"]  # for the Cox PH model


def create_dataset(features_path: str | Path | None = None) -> pd.DataFrame:
    """Join features and labels on (cutoff_date, brandId, partyId), keep confirmable labels only."""
    start_time = time.time()
    if features_path is None:
        features_path = max(PROCESSED_DIR.glob("train_features_base_*_*_*.parquet"), key=lambda p: p.stat().st_mtime)
    features_path = Path(features_path)
    labels_path = features_path.with_name(features_path.name.replace("train_features_base", "train_labels"))
    output_path = features_path.with_name(features_path.name.replace("train_features_base", "train_dataset"))

    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path, columns=KEY + LABEL_COLUMNS)
    dataset = features.merge(labels, on=KEY, how="left", validate="one_to_one", indicator=True)

    n_without_label = int((dataset["_merge"] != "both").sum())
    if n_without_label:
        raise ValueError(f"{n_without_label:,} feature rows have no label: {labels_path.name} is out of sync")

    # The labels file says "not confirmable" with an empty event_60d (e.g. a run with an earlier data_end).
    confirmed = dataset["label_available_60d"] & dataset[TARGET].notna()
    n_unconfirmed = int((~confirmed).sum())
    dataset = dataset[confirmed].drop(columns=["_merge", "label_available_60d"])
    dataset[LABEL_COLUMNS] = dataset[LABEL_COLUMNS].astype("int64")
    dataset = dataset.reset_index(drop=True)

    dataset.to_parquet(output_path, index=False)
    execution_time_s = round(time.time() - start_time, 1)
    print(f"saved {output_path.relative_to(PROJECT_ROOT)} ({len(dataset):,} rows, {dataset.shape[1]} columns)")
    print(f"dropped {n_unconfirmed:,} rows whose label is not confirmable yet")

    feature_columns = [c for c in dataset.columns if c not in KEY + ["operator"] + LABEL_COLUMNS]
    churn_by_cutoff = dataset.groupby("cutoff_date")[TARGET].mean()

    n_players_multi_cutoff = int((dataset.groupby(["brandId", "partyId"])["cutoff_date"].nunique() > 1).sum())
    timestamp_unix = int(time.time())
    # train_dataset_brand{brand}_{cutoffs}_{ts}: the brand ("basel" for every brand) in the name.
    run_name = f"train_dataset_brand{output_path.stem.removeprefix('train_dataset_')}"

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("timestamp_unix", timestamp_unix)
        mlflow.set_tag("timestamp_iso", datetime.fromtimestamp(timestamp_unix, tz=timezone.utc).isoformat())
        mlflow.set_tag("git_commit", _git_commit())
        mlflow.set_tag("user", getpass.getuser())
        mlflow.set_tag("host", socket.gethostname())
        mlflow.set_tag("source_script", "src/features/create_dataset.py")
        mlflow.set_tag("brand_id", brand_label(dataset["brandId"].unique()))

        mlflow.log_param("features_path", str(features_path.relative_to(PROJECT_ROOT)))
        mlflow.log_param("labels_path", str(labels_path.relative_to(PROJECT_ROOT)))
        mlflow.log_param("output_path", str(output_path.relative_to(PROJECT_ROOT)))
        mlflow.log_param("cutoff_dates", sorted(str(c) for c in dataset["cutoff_date"].unique()))
        mlflow.log_param("brand_id", sorted(dataset["brandId"].unique().tolist()))
        mlflow.log_param("target", TARGET)
        mlflow.log_param("survival_targets", SURVIVAL_TARGETS)
        mlflow.log_param("split_group", "partyId")
        mlflow.log_param("feature_columns", feature_columns)

        mlflow.log_metric("n_rows", len(dataset))
        mlflow.log_metric("n_features", len(feature_columns))
        mlflow.log_metric("n_dropped_unconfirmed_label", n_unconfirmed)
        mlflow.log_metric("n_players", int(dataset[["brandId", "partyId"]].drop_duplicates().shape[0]))
        mlflow.log_metric("n_players_multi_cutoff", n_players_multi_cutoff)
        mlflow.log_metric("n_nulls", int(dataset.isna().sum().sum()))
        mlflow.log_metric("churn_rate", round(float(dataset[TARGET].mean()), 4))
        mlflow.log_metric("survival_event_rate", round(float(dataset["event_observed"].mean()), 4))
        mlflow.log_metric("survival_max_event_day", int(dataset.loc[dataset["event_observed"] == 1, "duration_days"].max()))
        for cutoff, rate in churn_by_cutoff.items():
            mlflow.log_metric(f"churn_rate_{cutoff}", round(float(rate), 4))
        mlflow.log_metric("execution_time_s", execution_time_s)

        mlflow.log_input(
            mlflow.data.from_pandas(dataset, source=str(output_path.relative_to(PROJECT_ROOT)), targets=TARGET,
                                    name=run_name),
            context="training",
        )
        mlflow.log_artifact(str(output_path))
        print(f"MLflow run '{run_name}' logged")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join a train_features_base file with its train_labels file.")
    parser.add_argument("--features-path", help="default: the latest train_features_base file")
    create_dataset(parser.parse_args().features_path)
