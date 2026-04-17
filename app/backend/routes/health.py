from __future__ import annotations

from fastapi import APIRouter

from app.backend.dependencies import get_dataset
from app.backend.schemas.movie import HealthResponse
from app.config.settings import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    dataset = get_dataset()
    return HealthResponse(
        status="ok",
        dataset_rows=len(dataset),
        prepared_data_available=settings.prepared_parquet_path.exists()
        or settings.prepared_pickle_path.exists(),
    )


@router.get("/health/details")
def health_details():
    dataset = get_dataset()
    return {
        "status": "ok",
        "dataset_rows": len(dataset),
        "processed_files": {
            "parquet": str(settings.prepared_parquet_path) if settings.prepared_parquet_path.exists() else None,
            "pickle": str(settings.prepared_pickle_path) if settings.prepared_pickle_path.exists() else None,
        },
        "model_artifacts": {
            "directory": str(settings.model_artifacts_dir),
            "available_files": sorted(path.name for path in settings.model_artifacts_dir.glob("*") if path.is_file()),
        },
    }
