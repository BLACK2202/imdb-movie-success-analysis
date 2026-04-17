from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

from app.backend.dependencies import get_dataset
from app.services.analytics import (
    content_type_comparison,
    genre_breakdown,
    key_findings,
    rating_vote_correlation,
    recommendation_insights,
    summary_metrics,
    yearly_trends,
)
from app.services.data_loader import apply_filters, dataframe_to_records

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _filtered_dataset(
    min_rating: float | None = None,
    min_votes: int | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
    title_types: list[str] | None = None,
    genres: list[str] | None = None,
    search: str | None = None,
):
    dataset = get_dataset()
    year_range = (year_start, year_end) if year_start is not None and year_end is not None else None
    return apply_filters(
        dataset,
        year_range=year_range,
        min_rating=min_rating,
        min_votes=min_votes,
        title_types=title_types,
        genres=genres,
        search=search,
    )


@router.get("/summary")
def analytics_summary(
    min_rating: float | None = Query(default=None),
    min_votes: int | None = Query(default=None),
    year_start: int | None = Query(default=None),
    year_end: int | None = Query(default=None),
    title_types: list[str] | None = Query(default=None),
    genres: list[str] | None = Query(default=None),
    search: str | None = Query(default=None),
    export_format: str = Query(default="json"),
):
    filtered = _filtered_dataset(min_rating, min_votes, year_start, year_end, title_types, genres, search)
    payload = {
        "summary": summary_metrics(filtered),
        "content_type_comparison": dataframe_to_records(content_type_comparison(filtered)),
        "rating_vote_correlation": round(rating_vote_correlation(filtered), 4),
        "recommendations": dataframe_to_records(recommendation_insights(filtered)),
        "key_findings": key_findings(filtered),
    }
    if export_format.lower() == "csv":
        summary_df = filtered.describe(include="all").transpose().reset_index()
        return PlainTextResponse(summary_df.to_csv(index=False), media_type="text/csv")
    return payload


@router.get("/genres")
def analytics_genres(
    top_n: int = Query(default=15, le=50),
    min_rating: float | None = Query(default=None),
    min_votes: int | None = Query(default=None),
    year_start: int | None = Query(default=None),
    year_end: int | None = Query(default=None),
    title_types: list[str] | None = Query(default=None),
    genres: list[str] | None = Query(default=None),
    search: str | None = Query(default=None),
    export_format: str = Query(default="json"),
):
    genre_df = genre_breakdown(_filtered_dataset(min_rating, min_votes, year_start, year_end, title_types, genres, search), top_n=top_n)
    if export_format.lower() == "csv":
        return PlainTextResponse(genre_df.to_csv(index=False), media_type="text/csv")
    return {"items": dataframe_to_records(genre_df)}


@router.get("/trends")
def analytics_trends(
    min_rating: float | None = Query(default=None),
    min_votes: int | None = Query(default=None),
    year_start: int | None = Query(default=None),
    year_end: int | None = Query(default=None),
    title_types: list[str] | None = Query(default=None),
    genres: list[str] | None = Query(default=None),
    search: str | None = Query(default=None),
    export_format: str = Query(default="json"),
):
    trend_df = yearly_trends(_filtered_dataset(min_rating, min_votes, year_start, year_end, title_types, genres, search))
    if export_format.lower() == "csv":
        return PlainTextResponse(trend_df.to_csv(index=False), media_type="text/csv")
    return {"items": dataframe_to_records(trend_df)}
