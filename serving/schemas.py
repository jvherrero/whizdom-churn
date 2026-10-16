"""Request and response models of the churn API.

Each class is the shape of a JSON message. FastAPI uses them to check the input and to build the
documentation at /docs. Column meanings are in docs/schema_player_scores.md.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Answer of GET /health."""

    status: str                 
    model_version: str          


class BatchRequest(BaseModel):
    """Body of POST /score/batch: which date and brand to score."""

    as_of: dt.date              # scoring date, e.g. 2026-10-06
    brand_id: int = 64


class BatchResponse(BaseModel):
    """Answer of POST /score/batch: a summary, not the scores themselves."""

    as_of: dt.date
    brand_id: int
    model_version: str
    n_players: int              # rows written to player_scores
    output_path: str            # the Parquet file that was written
    seconds: float              # time the request took


class PlayerScore(BaseModel):
    """Answer of GET /score/{brand_id}/{player_id}: the score of one player."""

    run_date: dt.date
    model_version: str
    feature_version: str        # which feature data was used (e.g. the cutoff date)
    brand_id: int
    player_id: int
    p_churn_60d: float = Field(ge=0, le=1)     # main score
    p_churn_7d: float = Field(ge=0, le=1)
    p_churn_14d: float = Field(ge=0, le=1)
    p_churn_30d: float = Field(ge=0, le=1)
    value_at_risk: float        # EUR: expected_ggr_30d x p_churn_60d
    risk_band: int = Field(ge=1, le=10)        # 10 = riskiest 10%
    top_drivers: str            # e.g. "days_since_last_bet (+1.20); ..."
    drift_flag: int             # 1 = review before acting
