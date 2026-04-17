from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from app.backend.dependencies import get_dataset
from app.services.data_loader import apply_filters, dataframe_to_records

router = APIRouter(prefix="/movies", tags=["movies"])


@router.get("")
def list_movies(
    min_rating: float | None = Query(default=None),
    min_votes: int | None = Query(default=None),
    search: str | None = Query(default=None),
    title_types: list[str] | None = Query(default=None),
    genres: list[str] | None = Query(default=None),
    year_start: int | None = Query(default=None),
    year_end: int | None = Query(default=None),
    limit: int = Query(default=50, le=500),
    page: int = Query(default=1, ge=1),
    sort_by: str = Query(default="averageRating"),
    sort_order: str = Query(default="desc"),
    export_format: str = Query(default="json"),
):
    dataset = get_dataset()
    year_range = (year_start, year_end) if year_start is not None and year_end is not None else None
    filtered = apply_filters(
        dataset,
        year_range=year_range,
        min_rating=min_rating,
        min_votes=min_votes,
        title_types=title_types,
        genres=genres,
        search=search,
    )
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=sort_order.lower() == "asc")
    start = (page - 1) * limit
    paged = filtered.iloc[start : start + limit]
    if export_format.lower() == "csv":
        return PlainTextResponse(filtered.to_csv(index=False), media_type="text/csv")
    return {
        "count": int(len(filtered)),
        "page": page,
        "page_size": limit,
        "total_pages": max(1, (len(filtered) + limit - 1) // limit),
        "items": dataframe_to_records(paged),
    }


@router.get("/{movie_id}")
def get_movie(movie_id: str):
    dataset = get_dataset()
    movie = dataset[dataset["tconst"] == movie_id]
    if movie.empty:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found.")
    return movie.iloc[0].replace({float("nan"): None}).to_dict()
