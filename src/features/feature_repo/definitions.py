"""Feast definitions for the player features (T6).

The offline store is local Parquet and the online store is a local SQLite file, because we only
have read access to AWS. A feature row is valid only on its cutoff day (ttl of 1 day), so a
training row never gets features from a later date.

    make feature-store
"""

import sys
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float64, Int64
from feast.value_type import ValueType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_features import FEATURE_COLUMNS, FEATURES  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OFFLINE_PATH = PROJECT_ROOT / "data/02_intermediate/feature_store/player_features"
TTL = timedelta(days=1)


def objects(offline_path: Path = OFFLINE_PATH) -> list:
    """Build all Feast objects for the offline store in `offline_path`."""
    tenant = Entity(name="tenant", join_keys=["tenant_id"], value_type=ValueType.STRING,
                    description="Tenant of the player (player ids are unique within a tenant)")
    player = Entity(name="player", join_keys=["player_id"], value_type=ValueType.INT64, description="Player id")
    source = FileSource(
        name="player_features_source",
        path=str(offline_path),
        timestamp_field="event_timestamp",             # cutoff date
        created_timestamp_column="created_timestamp",  # if a cutoff is rebuilt, the newest row wins
        description="Final feature vector (winsorised + sign-log) of every training cutoff, one file per brand",
    )
    view = FeatureView(
        name="player_features",
        entities=[tenant, player],
        ttl=TTL,
        schema=[Field(name="brandId", dtype=Int64)] + [Field(name=f, dtype=Float64) for f in FEATURES],
        source=source,
        online=True,
        description="docs/feature_dictionary.md; families: " + ", ".join(FEATURE_COLUMNS),
    )
    # All candidate features. Each model keeps its own subset (saved in MLflow).
    service = FeatureService(name="churn_features", features=[view], description="Every candidate feature of the churn models")
    return [tenant, player, source, view, service]


# Needed by `feast apply` when it runs from this folder.
tenant, player, player_features_source, player_features, churn_features = objects()
