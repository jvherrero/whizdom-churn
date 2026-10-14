"""Tests of the churn API with the FastAPI test client (no server needed).

    .venv/bin/python -m pytest serving/tests -q

Two tests need a trained model with the alias Production and the scores of one day. On a fresh
clone, before `make pipeline`, they are skipped, not failed.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Import the app the same way uvicorn does (serving.app), so a broken import fails here too.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import serving.app as api  # noqa: E402

BRAND_ID = 64


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


def scored_player(brand_id: int = BRAND_ID) -> int | None:
    """Any player of the latest scores, or None if the brand has no scores yet."""
    path = api.latest_scores_file(brand_id)
    if path is None:
        return None
    return int(pd.read_parquet(path, columns=["player_id"]).player_id.iloc[0])


def test_health(client):
    if api.app.state.model_version is None:
        pytest.skip("no model with the alias Production yet: run make pipeline and set the alias")
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_version"].startswith(f"churn_lightgbm_classifier_brand{BRAND_ID}")


def test_known_player(client):
    player = scored_player()
    if player is None:
        pytest.skip("no scores yet: run make pipeline first")
    r = client.get(f"/score/{BRAND_ID}/{player}")
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["p_churn_60d"] <= 1
    assert 1 <= body["risk_band"] <= 10


def test_unknown_player(client):
    assert client.get(f"/score/{BRAND_ID}/1").status_code == 404


def test_bad_date(client):
    # FastAPI rejects it before scoring, so this test is fast.
    assert client.post("/score/batch", json={"as_of": "yesterday"}).status_code == 422
