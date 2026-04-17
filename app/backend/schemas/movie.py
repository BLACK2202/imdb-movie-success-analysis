from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MoviePredictionRequest(BaseModel):
    titleType: str = Field(default="movie")
    startYear: int
    runtimeMinutes: float | None = None
    isAdult: int = 0
    genres: list[str] = Field(default_factory=list)


class MovieQueryFilters(BaseModel):
    min_rating: float | None = None
    min_votes: int | None = None
    search: str | None = None
    title_types: list[str] | None = None
    genres: list[str] | None = None
    year_start: int | None = None
    year_end: int | None = None
    limit: int = 50


class HealthResponse(BaseModel):
    status: str
    dataset_rows: int
    prepared_data_available: bool


class GenericRecordsResponse(BaseModel):
    items: list[dict[str, Any]]
