from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

try:
    from backend.utils.state import get_training_state
    from backend.utils.trainer import (
        evaluate_alert,
        get_metrics_payload,
        get_predictions_payload,
        get_recommendation_payload,
        predict_next_value,
        simulate_scaled_sequence,
        start_training_job,
    )
except ImportError:
    from utils.state import get_training_state
    from utils.trainer import (
        evaluate_alert,
        get_metrics_payload,
        get_predictions_payload,
        get_recommendation_payload,
        predict_next_value,
        simulate_scaled_sequence,
        start_training_job,
    )


router = APIRouter()


class PredictRequest(BaseModel):
    sequence: list[float] = Field(..., min_length=96, max_length=96)
    model: str = Field(default="hybrid")


class SimulateRequest(BaseModel):
    sequence: list[float] = Field(..., min_length=96, max_length=96)
    scale_factor: float = Field(default=1.0, gt=0)


class AlertRequest(BaseModel):
    prediction: float
    threshold: float


@router.get("/")
def root():
    return {"message": "Smart Grid Forecasting Dashboard API is running."}


@router.get("/status")
def get_status():
    return get_training_state()


@router.post("/train")
def train_models():
    started = start_training_job()
    if not started:
        raise HTTPException(status_code=409, detail="Training is already in progress.")

    return {"message": "Training started in the background."}


@router.get("/metrics")
def get_metrics():
    payload = get_metrics_payload()
    if payload is None:
        raise HTTPException(status_code=404, detail="Metrics are not available yet. Run training first.")
    return payload


@router.get("/predictions")
def get_predictions():
    payload = get_predictions_payload()
    if payload is None:
        raise HTTPException(status_code=404, detail="Predictions are not available yet. Run training first.")
    return payload


@router.post("/predict")
def predict(request: PredictRequest):
    try:
        predicted_value = predict_next_value(request.sequence, request.model.lower())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"model": request.model.lower(), "prediction": predicted_value}


@router.post("/simulate")
def simulate(request: SimulateRequest):
    try:
        return simulate_scaled_sequence(request.sequence, request.scale_factor, "hybrid")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/recommendation")
def recommendation():
    payload = get_recommendation_payload()
    if payload is None:
        raise HTTPException(status_code=404, detail="Recommendation is not available yet. Run training first.")
    return payload


@router.post("/alert")
def alert(request: AlertRequest):
    return evaluate_alert(request.prediction, request.threshold)
