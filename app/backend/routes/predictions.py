from __future__ import annotations

from fastapi import APIRouter

from app.backend.dependencies import get_dataset
from app.backend.schemas.movie import MoviePredictionRequest
from app.services.modeling import ModelingConfig, compare_models, load_persisted_model, predict_success

router = APIRouter(tags=["predictions"])


@router.post("/predict")
def predict(payload: MoviePredictionRequest):
    model = load_persisted_model()
    if model is None:
        dataset = get_dataset()
        compare_models(dataset, ModelingConfig(), persist=True)
        model = load_persisted_model()
    return predict_success(payload.model_dump(), model)
