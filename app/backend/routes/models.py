from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.backend.dependencies import get_dataset
from app.services.modeling import ModelingConfig, compare_models

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/compare")
def compare_model_endpoint(
    min_votes: int = Query(default=500),
    year_start: int = Query(default=1920),
    year_end: int = Query(default=2025),
    success_rating: float = Query(default=7.0),
    success_votes: int = Query(default=25_000),
    mode: str = Query(default="full"),
):
    dataset = get_dataset()
    config = ModelingConfig(
        min_votes=min_votes,
        year_start=year_start,
        year_end=year_end,
        success_rating=success_rating,
        success_votes=success_votes,
        mode=mode,
        cv_folds=3 if mode == "fast" else 5,
        sample_size=50000 if mode == "fast" else None,
    )
    try:
        return compare_models(dataset, config=config, persist=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
