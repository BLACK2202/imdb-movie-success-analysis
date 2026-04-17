from __future__ import annotations

import pandas as pd

from app.services.data_loader import prepare_live_rows
from app.services.data_quality import distribution_summary, quality_assessment


def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "tconst": "tt001",
                "titleType": "movie",
                "primaryTitle": "Alpha",
                "originalTitle": "Alpha",
                "isAdult": 0,
                "startYear": 2020,
                "runtimeMinutes": 100,
                "genres": "Drama,Action",
                "averageRating": 7.6,
                "numVotes": 10000,
                "success_score": 7.6 * 4.0000434,
            },
            {
                "tconst": "tt002",
                "titleType": "tvSeries",
                "primaryTitle": "Beta",
                "originalTitle": "Beta",
                "isAdult": 0,
                "startYear": 2021,
                "runtimeMinutes": 45,
                "genres": "Drama",
                "averageRating": 8.1,
                "numVotes": 5000,
                "success_score": 8.1 * 3.6990569,
            },
        ]
    )


def test_quality_assessment_returns_summary():
    assessment = quality_assessment(sample_df())
    assert "overall_quality" in assessment
    assert "missing_values" in assessment


def test_distribution_summary_contains_shape_stats():
    summary = distribution_summary(sample_df(), "averageRating")
    assert set(summary.keys()) == {"mean", "median", "skewness", "kurtosis"}


def test_prepare_live_rows_adds_success_score():
    incoming = pd.DataFrame(
        [
            {
                "tconst": "tt_live_1",
                "titleType": "movie",
                "startYear": 2025,
                "genres": "Drama",
                "primaryTitle": "Live Title",
                "averageRating": 8.0,
                "numVotes": 1200,
            }
        ]
    )
    prepared = prepare_live_rows(incoming)
    assert "success_score" in prepared.columns
    assert len(prepared) == 1
