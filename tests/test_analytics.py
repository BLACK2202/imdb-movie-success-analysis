from __future__ import annotations

import pandas as pd

from app.services.analytics import genre_breakdown, key_findings, summary_metrics, yearly_trends


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
            },
        ]
    )


def test_summary_metrics_counts_titles():
    metrics = summary_metrics(sample_df())
    assert metrics["total_titles"] == 2
    assert metrics["content_types"] == 2


def test_genre_breakdown_explodes_multi_genre_rows():
    genre_df = genre_breakdown(sample_df(), top_n=10)
    drama_row = genre_df.loc[genre_df["genre"] == "Drama"].iloc[0]
    assert int(drama_row["titles"]) == 2


def test_yearly_trends_preserves_years():
    trends = yearly_trends(sample_df())
    assert trends["startYear"].tolist() == [2020, 2021]


def test_key_findings_returns_multiple_insights():
    findings = key_findings(sample_df())
    assert len(findings) >= 5
