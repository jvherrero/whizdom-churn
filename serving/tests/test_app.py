"""Tests of the churn API with the FastAPI test client (no server needed).

    .venv/bin/python -m pytest serving/tests -q
"""

import sys
from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# Import the app the same way uvicorn does (serving.app), so a broken import fails here too.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import serving.app as api  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


def a_scored_player(brand_id: int = 64) -> int:
    """Any player of the latest scores, to test the player endpoint."""
    files = sorted(api.batch.OUTPUT_DIR.glob(f"run_date=*/brand_id={brand_id}.parquet"))
    return int(pd.read_parquet(files[-1], columns=["player_id"]).player_id.iloc[0])

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_version"].startswith("churn_lightgbm_classifier_brand64")


def test_known_player(client):
    r = client.get(f"/score/64/{a_scored_player()}")
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["p_churn_60d"] <= 1
    assert 1 <= body["risk_band"] <= 10

def test_unknown_player(client):
    assert client.get("/score/64/1").status_code == 404


def test_bad_date(client):
    # FastAPI rejects it before scoring, so this test is fast.
    assert client.post("/score/batch", json={"as_of": "yesterday"}).status_code == 422