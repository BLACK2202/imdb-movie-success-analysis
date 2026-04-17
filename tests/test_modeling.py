from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.services import modeling
from app.services.modeling import ModelingConfig, compare_models, load_cached_comparison


def test_compare_models_returns_multiple_models():
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
    df = pd.DataFrame(rows)
    result = compare_models(
        df,
        ModelingConfig(
            min_votes=0,
            year_start=1980,
            year_end=2025,
            success_rating=7.0,
            success_votes=25000,
            cv_folds=3,
        ),
        persist=False,
    )
    assert len(result["comparison"]) >= 5
    assert result["best_model"]["model_name"]


def test_cached_model_loading_roundtrip(monkeypatch):
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
    df = pd.DataFrame(rows)
    config = ModelingConfig(min_votes=0, year_start=1980, year_end=2025, success_rating=7.0, success_votes=25000, cv_folds=3, mode="fast", sample_size=320)
    base = Path("tests") / "_tmp_artifacts"
    base.mkdir(parents=True, exist_ok=True)

    def fake_artifact_paths(_config):
        return {
            "model": base / "model.joblib",
            "summary": base / "summary.json",
            "latest_model": base / "latest_model.joblib",
            "latest_summary": base / "latest_summary.json",
        }

    monkeypatch.setattr(modeling, "_artifact_paths", fake_artifact_paths)
    compare_models(df, config, persist=True, force_retrain=True)
    cached = load_cached_comparison(config)
    assert cached is not None
    assert cached["best_model"]["model_name"]
