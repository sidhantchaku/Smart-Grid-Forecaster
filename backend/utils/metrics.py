import numpy as np


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.float32)

    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean(np.square(actual - predicted))))

    ss_res = float(np.sum(np.square(actual - predicted)))
    ss_tot = float(np.sum(np.square(actual - np.mean(actual))))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }
