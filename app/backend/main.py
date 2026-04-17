from __future__ import annotations

from fastapi import FastAPI

from app.backend.routes.analytics import router as analytics_router
from app.backend.routes.health import router as health_router
from app.backend.routes.models import router as models_router
from app.backend.routes.movies import router as movies_router
from app.backend.routes.predictions import router as predictions_router
from app.config.settings import settings

settings.ensure_directories()

app = FastAPI(
    title="IMDb Movie Success Analysis API",
    description="Backend services for IMDb analytics, model comparison, and prediction.",
    version="2.0.0",
)

app.include_router(health_router)
app.include_router(movies_router)
app.include_router(analytics_router)
app.include_router(models_router)
app.include_router(predictions_router)


@app.get("/")
def root():
    return {
        "message": "IMDb Movie Success Analysis API",
        "docs": "/docs",
        "health": "/health",
    }
