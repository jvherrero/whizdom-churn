"""Feast feature definitions (T6): the player features of build_features.py, one row per player per cutoff.

The offline store is local Parquet (data/02_intermediate/feature_store/player_features/, one file per
brand, written by feature_store.py): the plan's S3 offline store and Redis online store need write access
to AWS, which the sprint does not have, so the online store is a local SQLite file until Week 2.
Point-in-time joins: a feature row is valid on the day of its cutoff (ttl 1 day), so a training row never
gets features computed after its date.

    make feature-store          # registers these definitions (feast apply) and checks the store
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
    """Every Feast object, reading the offline store at `offline_path` (a folder of Parquet files)."""
    tenant = Entity(name="tenant", join_keys=["tenant_id"], value_type=ValueType.STRING,
                    description="Tenant of the player (player ids are unique within a tenant)")
    player = Entity(name="player", join_keys=["player_id"], value_type=ValueType.INT64, description="Player id")
    source = FileSource(
        name="player_features_source",
        path=str(offline_path),
        timestamp_field="event_timestamp",             # the cutoff: features use data up to that day
        created_timestamp_column="created_timestamp",  # when the row was written: the latest wins on a rebuild
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
    # Every feature: the models pick their own set from it (the pruned list is in each MLflow run).
    service = FeatureService(name="churn_features", features=[view], description="Every candidate feature of the churn models")
    return [tenant, player, source, view, service]


# Module-level objects: what `feast apply` finds when run from this folder.
tenant, player, player_features_source, player_features, churn_features = objects()
