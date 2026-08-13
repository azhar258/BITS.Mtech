"""
REST API Module - Production Code
FastAPI application exposing the Iris classification model.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.inference import IrisClassifier

logger = logging.getLogger(__name__)

# ─── Pydantic Schemas ────────────────────────────────────────────────────────


class IrisFeatures(BaseModel):
    """Request schema for a single iris sample."""

    sepal_length_cm: float = Field(..., gt=0, lt=20, example=5.1)
    sepal_width_cm: float = Field(..., gt=0, lt=20, example=3.5)
    petal_length_cm: float = Field(..., gt=0, lt=20, example=1.4)
    petal_width_cm: float = Field(..., gt=0, lt=20, example=0.2)

    @field_validator(
        "sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"
    )
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Feature values must be positive.")
        return v


class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: Dict[str, float]


class BatchRequest(BaseModel):
    samples: List[IrisFeatures] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ─── App Lifecycle ────────────────────────────────────────────────────────────

classifier = IrisClassifier()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading model.")
    try:
        classifier.load()
        logger.info("Model loaded successfully on startup.")
    except Exception as exc:
        logger.warning("Model not loaded at startup: %s", exc)
    yield
    logger.info("Shutting down API.")


app = FastAPI(
    title="Iris Classifier API",
    description="REST API for Iris flower species classification. Group 146 | AIMLCZG546",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Return API health status and whether the model is loaded."""
    loaded = classifier._pipeline is not None
    logger.info("Health check called. Model loaded: %s.", loaded)
    return HealthResponse(status="ok", model_loaded=loaded)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
def predict(features: IrisFeatures):
    """
    Predict the Iris species for a single flower sample.

    Returns the predicted class name and class probabilities.
    """
    if classifier._pipeline is None:
        logger.error("Prediction requested but model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please load the model first.",
        )
    try:
        result = classifier.predict(features.model_dump())
        return PredictionResponse(**result)
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )


@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
def predict_batch(request: BatchRequest):
    """
    Predict Iris species for a batch of flower samples (up to 1000).

    Returns a list of predicted classes with probabilities.
    """
    if classifier._pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )
    try:
        samples = [s.model_dump() for s in request.samples]
        results = classifier.predict_batch(samples)
        return BatchResponse(predictions=[PredictionResponse(**r) for r in results])
    except Exception as exc:
        logger.error("Batch prediction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
