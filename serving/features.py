"""Where the API gets a player's data.

Today it reads the latest scores written by the daily batch (src/models/score.py).
When there is an online store (Redis in the plan), only this file has to change.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SCORES_DIR = Path(__file__).resolve().parents[1] / "data/03_output/player_scores"


def latest_scores_file(brand_id: int) -> Path | None:
    """The most recent scores file of a brand, or None if the brand was never scored."""
    files = sorted(SCORES_DIR.glob(f"run_date=*/brand_id={brand_id}.parquet"))
    return files[-1] if files else None


def player_row(brand_id: int, player_id: int) -> pd.Series | None:
    """The latest scores of one player, or None if the player is not there."""
    path = latest_scores_file(brand_id)
    if path is None:
        return None
    scores = pd.read_parquet(path)
    row = scores[scores["player_id"] == player_id]
    return None if row.empty else row.iloc[0]
