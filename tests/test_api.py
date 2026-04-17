from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.routes import analytics as analytics_routes
from app.backend.routes import health as health_routes
from app.backend.routes import models as model_routes
from app.backend.routes import movies as movie_routes


def sample_df() -> pd.DataFrame:
    rows = []
    for idx in range(320):
        rows.append(
            {
                "tconst": f"tt{idx:05d}",
                "titleType": "movie" if idx % 2 == 0 else "tvSeries",
                "primaryTitle": f"Title {idx}",
                "originalTitle": f"Title {idx}",
                "isAdult": 0,
                "startYear": 1980 + (idx % 40),
                "runtimeMinutes": 80 + (idx % 60),
                "genres": "Drama,Action" if idx % 3 == 0 else "Comedy,Romance",
                "averageRating": 7.8 if idx % 4 == 0 else 6.1,
                "numVotes": 30000 if idx % 4 == 0 else 5000 + idx,
            }
        )
    return pd.DataFrame(rows)


def test_health_details_and_movie_pagination(monkeypatch):
    df = sample_df()
    monkeypatch.setattr(movie_routes, "get_dataset", lambda: df)
    monkeypatch.setattr(health_routes, "get_dataset", lambda: df)
    client = TestClient(app)
    health_response = client.get("/health/details")
    assert health_response.status_code == 200
    movie_response = client.get("/movies", params={"page": 2, "limit": 20, "sort_by": "numVotes"})
    assert movie_response.status_code == 200
    payload = movie_response.json()
    assert payload["page"] == 2
    assert len(payload["items"]) == 20


def test_analytics_export_and_key_findings(monkeypatch):
    df = sample_df()
    monkeypatch.setattr(analytics_routes, "get_dataset", lambda: df)
    client = TestClient(app)
    response = client.get("/analytics/summary")
    assert response.status_code == 200
    assert len(response.json()["key_findings"]) >= 5
    csv_response = client.get("/analytics/genres", params={"export_format": "csv"})
    assert csv_response.status_code == 200
    assert "genre" in csv_response.text.lower()


def test_models_compare_fast_mode(monkeypatch):
    df = sample_df()
    monkeypatch.setattr(model_routes, "get_dataset", lambda: df)
    client = TestClient(app)
    response = client.get("/models/compare", params={"mode": "fast", "min_votes": 0})
    assert response.status_code == 200
    assert response.json()["best_model"]["model_name"]
