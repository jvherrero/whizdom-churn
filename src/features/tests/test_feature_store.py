"""Feast feature store (feature_store.py) on hand-made rows and a temporary store: no S3, no real data.

    .venv/bin/python -m pytest src/features/tests -q      (make test)
"""

import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("FEAST_USAGE", "False")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import feature_store as fs  # noqa: E402
from build_features import FEATURES  # noqa: E402
from feast import FeatureStore  # noqa: E402
from feast.repo_config import RepoConfig  # noqa: E402

D = dt.date


def features(cutoff: dt.date, value: float) -> pd.DataFrame:
    """3 players of brand 64 on one cutoff; feature i of every player = value + i; one value missing."""
    rows = pd.DataFrame({"cutoff_date": cutoff, "tenant_id": "t", "player_id": [1, 2, 3], "brandId": 64})
    for i, f in enumerate(FEATURES):
        rows[f] = value + i
    rows.loc[2, FEATURES[0]] = np.nan
    return rows


def at(day: str, hours: int = 0) -> pd.DataFrame:
    """Entity rows: the 3 players at `day` + `hours` (UTC)."""
    return pd.DataFrame({"tenant_id": "t", "player_id": [1, 2, 3],
                         "event_timestamp": pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours, unit="h")})


@pytest.fixture
def store(tmp_path):
    config = RepoConfig(project="test", provider="local", registry=str(tmp_path / "registry.db"),
                        online_store={"type": "sqlite", "path": str(tmp_path / "online.db")},
                        offline_store={"type": "file"}, entity_key_serialization_version=3)
    s = FeatureStore(config=config)
    s.apply(fs.definitions.objects(tmp_path / "offline"))
    return s, tmp_path / "offline"


def test_point_in_time(store):
    s, offline = store
    fs.write_offline(pd.concat([features(D(2026, 1, 1), 10.0), features(D(2026, 2, 1), 20.0)]), offline)
    got = fs.historical(s, at("2026-02-01")).set_index("player_id")
    assert got.loc[1, FEATURES[1]] == 21.0                                   # the row's own cutoff
    assert np.isnan(got.loc[3, FEATURES[0]])                                 # a missing value stays missing
    assert fs.historical(s, at("2026-02-01", hours=12)).set_index("player_id").loc[1, FEATURES[1]] == 21.0
    assert fs.historical(s, at("2026-01-31"))[FEATURES].isna().all().all()   # the January row is older than the ttl
    assert fs.historical(s, at("2025-12-31"))[FEATURES].isna().all().all()   # nothing before the first cutoff


def test_rebuilt_cutoff_replaces_the_old_rows(store):
    s, offline = store
    fs.write_offline(features(D(2026, 2, 1), 20.0), offline)
    fs.write_offline(features(D(2026, 2, 1), 50.0), offline)
    assert len(pd.read_parquet(offline / "brand64.parquet")) == 3
    assert fs.historical(s, at("2026-02-01")).set_index("player_id").loc[1, FEATURES[1]] == 51.0


def test_check_passes_on_what_was_written(store):
    s, offline = store
    rows = pd.concat([features(D(2026, 1, 1), 10.0), features(D(2026, 2, 1), 20.0)])
    fs.write_offline(rows, offline)
    assert fs.check(s, rows) == 3
    with pytest.raises(AssertionError):
        fs.check(s, pd.concat([features(D(2026, 1, 1), 10.0), features(D(2026, 2, 1), 99.0)]))
