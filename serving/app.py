"""Churn API: the daily batch scoring and the score of one player.

    uvicorn serving.app:app --port 8000        (from the repo root)
    open http://localhost:8000/docs

The batch endpoint runs the same code as `make score` (src/models/score.py). The player endpoint
reads the latest scores written by the batch, so both always give the same number for a player.
"""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# serving/ must be on the path first: there is also a src/features folder with the same name.
from features import latest_scores_file, player_row  # noqa: E402
import score as batch  # noqa: E402
from schemas import BatchRequest, BatchResponse, HealthResponse, PlayerScore  # noqa: E402

MODEL_VERSION = "Production"  # alias of the registered model in MLflow
BRAND_ID = 64

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Find the model once at start-up. If it fails, /health says so.
    try:
        _, _, version = batch.resolve_models(str(BRAND_ID), MODEL_VERSION, None, None)
        app.state.model_version = version
    except Exception as error:
        app.state.model_version = None
        print(f"model not loaded: {error}")
    yield


app = FastAPI(title="Whizdom churn API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if app.state.model_version is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return HealthResponse(status="ok", model_version=app.state.model_version)


@app.post("/score/batch", response_model=BatchResponse)
def score_batch(request: BatchRequest) -> BatchResponse:
    start = time.time()
    partition = batch.score(str(request.as_of), request.brand_id, model_version=MODEL_VERSION)
    path = partition / f"brand_id={request.brand_id}.parquet"
    n_players = len(pd.read_parquet(path, columns=["player_id"]))
    return BatchResponse(as_of=request.as_of, brand_id=request.brand_id, model_version=app.state.model_version,
                         n_players=n_players, output_path=str(path.relative_to(PROJECT_ROOT)),
                         seconds=round(time.time() - start, 1))


@app.get("/score/{brand_id}/{player_id}", response_model=PlayerScore)
def score_player(brand_id: int, player_id: int) -> PlayerScore:
    if latest_scores_file(brand_id) is None:
        raise HTTPException(status_code=404, detail=f"no scores for brand {brand_id}: run /score/batch first")
    r = player_row(brand_id, player_id)
    if r is None:
        raise HTTPException(status_code=404, detail=f"player {player_id} is not active in brand {brand_id}")
    return PlayerScore(
        run_date=r["run_date"], model_version=r["model_version"], feature_version=f"cutoff {r['run_date']}",
        brand_id=brand_id, player_id=player_id,
        p_churn_60d=r["p_churn_60d"], p_churn_7d=r["p_churn_7d"], p_churn_14d=r["p_churn_14d"],
        p_churn_30d=r["p_churn_30d"], value_at_risk=r["value_at_risk"], risk_band=r["risk_band"],
        top_drivers=r["top_drivers"], drift_flag=r["drift_flag"],
    )

