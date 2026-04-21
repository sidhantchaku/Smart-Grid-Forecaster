import json
import os
import threading
from pathlib import Path

import numpy as np

try:
    from backend.models.forecast_models import MODEL_NAMES, build_model, load_model
    from backend.utils.data import APP_ROOT, WINDOW_SIZE, create_windows, load_series, normalize_series
    from backend.utils.metrics import calculate_metrics
    from backend.utils.state import get_training_state, set_training_state
except ImportError:
    from models.forecast_models import MODEL_NAMES, build_model, load_model
    from utils.data import APP_ROOT, WINDOW_SIZE, create_windows, load_series, normalize_series
    from utils.metrics import calculate_metrics
    from utils.state import get_training_state, set_training_state


IS_VERCEL = bool(os.getenv("VERCEL"))
SAVED_MODELS_DIR = Path("/tmp/smart-grid-saved-models") if IS_VERCEL else APP_ROOT / "backend" / "saved_models"
METRICS_PATH = SAVED_MODELS_DIR / "metrics.json"
PREDICTIONS_PATH = SAVED_MODELS_DIR / "predictions.json"
SCALER_PATH = SAVED_MODELS_DIR / "scaler.json"

MAX_WINDOWS = 1200 if IS_VERCEL else 2400


def _ensure_storage() -> None:
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_metrics_payload() -> dict | None:
    return _load_json(METRICS_PATH)


def get_predictions_payload() -> dict | None:
    return _load_json(PREDICTIONS_PATH)


def get_recommendation_payload() -> dict | None:
    metrics_payload = get_metrics_payload()
    if metrics_payload is None:
        return None

    models = metrics_payload.get("models", {})
    if not models:
        return None

    best_model, best_metrics = max(models.items(), key=lambda item: item[1].get("r2", float("-inf")))
    return {
        "best_model": best_model,
        "reason": f"Highest R2 score ({best_metrics.get('r2', 0)}) with the strongest overall forecast fit.",
        "metrics": best_metrics,
    }


def _prepare_datasets() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    raw_series, dataset_meta = load_series()
    normalized_series, mean, std = normalize_series(raw_series)
    features, targets = create_windows(normalized_series, WINDOW_SIZE)

    if len(features) == 0:
        raise ValueError("Dataset is too short for a 96-step sliding window.")

    if len(features) > MAX_WINDOWS:
        features = features[-MAX_WINDOWS:]
        targets = targets[-MAX_WINDOWS:]

    split_index = max(int(len(features) * 0.8), 1)
    split_index = min(split_index, len(features) - 1)

    x_train = features[:split_index]
    y_train = targets[:split_index]
    x_test = features[split_index:]
    y_test = targets[split_index:]

    meta = {
        **dataset_meta,
        "window_size": WINDOW_SIZE,
        "mean": mean,
        "std": std,
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
    }
    return x_train, y_train, x_test, y_test, meta


def _denormalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean


def _train_worker() -> None:
    try:
        _ensure_storage()

        x_train, y_train, x_test, y_test, meta = _prepare_datasets()
        metrics_payload = {
            "status": "completed",
            "dataset": meta,
            "models": {},
        }
        predictions_payload = {
            "model": "hybrid",
            "dataset": meta,
            "actual": [],
            "predicted": [],
        }

        actual_values = _denormalize(y_test, meta["mean"], meta["std"])

        for model_name in MODEL_NAMES:
            model = build_model(model_name)
            model.fit(x_train, y_train)

            predictions = model.predict(x_test).reshape(-1)
            predictions = _denormalize(predictions, meta["mean"], meta["std"])

            metrics_payload["models"][model_name] = calculate_metrics(actual_values, predictions)
            model.save(SAVED_MODELS_DIR / f"{model_name}.json")

            if model_name == "hybrid":
                predictions_payload["actual"] = np.round(actual_values, 4).tolist()
                predictions_payload["predicted"] = np.round(predictions, 4).tolist()

        with SCALER_PATH.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "mean": meta["mean"],
                    "std": meta["std"],
                    "window_size": WINDOW_SIZE,
                    "dataset_source": meta["source"],
                },
                handle,
                indent=2,
            )

        with METRICS_PATH.open("w", encoding="utf-8") as handle:
            json.dump(metrics_payload, handle, indent=2)

        with PREDICTIONS_PATH.open("w", encoding="utf-8") as handle:
            json.dump(predictions_payload, handle, indent=2)

        set_training_state("completed")
    except Exception as exc:
        set_training_state("failed", str(exc))


def start_training_job() -> bool:
    current_state = get_training_state()
    if current_state["status"] == "training":
        return False

    if IS_VERCEL:
        set_training_state("training")
        _train_worker()
        return True

    set_training_state("training")
    worker = threading.Thread(target=_train_worker, daemon=True)
    worker.start()
    return True


def predict_next_value(sequence: list[float], model_name: str) -> float:
    model_path = SAVED_MODELS_DIR / f"{model_name}.json"
    scaler = _load_json(SCALER_PATH)

    if scaler is None:
        raise FileNotFoundError("Scaler metadata is missing. Run training first.")

    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model '{model_name}'.")

    if not model_path.exists():
        raise FileNotFoundError(f"Model '{model_name}' has not been trained yet.")

    mean = float(scaler["mean"])
    std = float(scaler["std"])
    normalized_sequence = ((np.asarray(sequence, dtype=np.float32) - mean) / std).reshape(1, WINDOW_SIZE)

    model = load_model(model_path)
    prediction = float(model.predict(normalized_sequence).reshape(-1)[0])
    return round(prediction * std + mean, 4)


def simulate_scaled_sequence(sequence: list[float], scale_factor: float, model_name: str = "hybrid") -> dict:
    original_prediction = predict_next_value(sequence, model_name)
    scaled_sequence = [float(value) * scale_factor for value in sequence]
    simulated_prediction = predict_next_value(scaled_sequence, model_name)
    delta = round(simulated_prediction - original_prediction, 4)

    if delta > 0:
        insight = "Scaled demand increases the expected next-step load."
    elif delta < 0:
        insight = "Scaled demand reduces the expected next-step load."
    else:
        insight = "Scenario scaling has minimal impact on the next-step forecast."

    return {
        "original_prediction": original_prediction,
        "simulated_prediction": simulated_prediction,
        "delta": delta,
        "scale_factor": scale_factor,
        "insight": insight,
    }


def evaluate_alert(prediction: float, threshold: float) -> dict:
    alert = prediction > threshold
    margin = round(prediction - threshold, 4)

    if alert:
        message = "High load expected"
        severity = "high" if margin > threshold * 0.1 else "medium"
        recommendation = "Shift discretionary demand, prepare reserve supply, and monitor feeder stability."
    else:
        message = "Normal load"
        severity = "normal"
        recommendation = "Keep normal operating posture and continue monitoring."

    return {
        "alert": alert,
        "message": message,
        "severity": severity,
        "prediction": round(prediction, 4),
        "threshold": round(threshold, 4),
        "margin": margin,
        "recommendation": recommendation,
    }
