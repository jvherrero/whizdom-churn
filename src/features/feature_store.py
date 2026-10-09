"""Feast feature store (T6, D4): the offline store of the player features and their registry.

    .venv/bin/python src/features/feature_store.py                  # fill from each brand's latest training dataset
    .venv/bin/python src/features/feature_store.py --dataset PATH   # fill from one training dataset
    make feature-store

`make pipeline` writes the features of its training cutoffs here (step 3b) and registers the definitions
(feature_repo/definitions.py, the versioned registry). The offline store keeps every cutoff ever built,
one Parquet file per brand; a rebuilt cutoff replaces the old rows. Training rows can then be fetched
point in time: get_historical_features(rows with tenant_id, player_id, event_timestamp) returns the
features of the row's own cutoff, never later ones. Each publish checks it on the latest cutoff: the
store must give back exactly the features that were written, and nothing on the day before the first
cutoff.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("FEAST_USAGE", "False")  # no usage telemetry
import pandas as pd  # noqa: E402
from feast import FeatureStore  # noqa: E402

REPO = Path(__file__).resolve().parent / "feature_repo"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))
import definitions  # noqa: E402
from build_features import FEATURES  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data/processed"
KEY = ["tenant_id", "player_id"]


def offline_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Feature rows (cutoff_date, tenant_id, player_id, brandId, FEATURES) in the offline store's layout."""
    out = features[["tenant_id", "player_id", "brandId"] + FEATURES].copy()
    out = out.astype({"tenant_id": str, "player_id": "int64", "brandId": "int64"} | {f: "float64" for f in FEATURES})
    out.insert(0, "event_timestamp", pd.to_datetime(features["cutoff_date"]).dt.tz_localize("UTC"))
    out["created_timestamp"] = pd.Timestamp.now(tz="UTC")
    return out


def write_offline(features: pd.DataFrame, offline_dir: Path = definitions.OFFLINE_PATH) -> list[Path]:
    """Add the features of these cutoffs to each brand's file, replacing the rows of the same cutoffs."""
    offline_dir.mkdir(parents=True, exist_ok=True)
    rows = offline_rows(features)
    paths = []
    for brand, new in rows.groupby("brandId"):
        path = offline_dir / f"brand{brand}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            new = pd.concat([old[~old["event_timestamp"].isin(new["event_timestamp"].unique())], new], ignore_index=True)
        new.sort_values(["event_timestamp"] + KEY).to_parquet(path, index=False)
        paths.append(path)
    return paths


def store() -> FeatureStore:
    """The project's feature store with the current definitions registered (feast apply)."""
    fs = FeatureStore(repo_path=str(REPO))
    fs.apply(definitions.objects(definitions.OFFLINE_PATH))
    return fs


def historical(fs: FeatureStore, entities: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time features for rows with tenant_id, player_id and event_timestamp (UTC)."""
    return fs.get_historical_features(entity_df=entities, features=fs.get_feature_service("churn_features")).to_df()


def check(fs: FeatureStore, features: pd.DataFrame) -> int:
    """The store gives back exactly the features written for the latest cutoff, and nothing the day before
    the first one. Returns the rows checked; raises when a value differs."""
    written = offline_rows(features)
    last, first = written["event_timestamp"].max(), written["event_timestamp"].min()
    expected = written[written["event_timestamp"] == last]
    got = historical(fs, expected[KEY + ["event_timestamp"]]).merge(expected, on=KEY, suffixes=("", "_written"))
    if len(got) != len(expected):
        raise AssertionError(f"feature store: {len(got):,} rows back for {len(expected):,} written on {last.date()}")
    for f in ["brandId"] + FEATURES:
        a, b = got[f].astype("float64"), got[f"{f}_written"].astype("float64")
        if not ((a - b).abs().fillna(0).le(1e-9) & (a.isna() == b.isna())).all():
            raise AssertionError(f"feature store: {f} differs from what was written on {last.date()}")
    before = expected[KEY].assign(event_timestamp=first - pd.Timedelta(1, unit="D"))
    if historical(fs, before)[FEATURES].notna().any().any():
        raise AssertionError(f"feature store: features found before the first cutoff {first.date()} (point in time)")
    return len(expected)


def publish(features: pd.DataFrame) -> list[Path]:
    """Write the features to the offline store, register the definitions and check the round trip."""
    paths = write_offline(features)
    n = check(store(), features)
    cutoffs = sorted(str(c) for c in pd.to_datetime(features["cutoff_date"]).dt.date.unique())
    print(f"feature store: {len(features):,} rows, cutoffs {cutoffs[0]} to {cutoffs[-1]} -> "
          f"{', '.join(os.path.relpath(p, PROJECT_ROOT) for p in paths)}; point-in-time check on {n:,} rows OK")
    return paths


def latest_datasets() -> list[Path]:
    """The latest training dataset of each brand (data/processed/train_dataset_{brand}_..._{timestamp}.parquet)."""
    by_brand = {}
    for path in sorted(DATASET_DIR.glob("train_dataset_*.parquet"), key=lambda p: p.stem.rsplit("_", 1)[-1]):
        by_brand[path.stem.split("_")[2]] = path
    return [by_brand[b] for b in sorted(by_brand)]


def run(datasets: list[Path] | None = None) -> None:
    for path in datasets or latest_datasets():
        print(f"from {os.path.relpath(path, PROJECT_ROOT)}")
        publish(pd.read_parquet(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill the Feast offline store from training datasets and register it.")
    parser.add_argument("--dataset", type=Path, nargs="+", help="training datasets (default: each brand's latest)")
    run(parser.parse_args().dataset)
