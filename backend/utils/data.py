import math
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_SIZE = 96
APP_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = APP_ROOT.parent


def _generate_synthetic_series(length: int = 1600) -> np.ndarray:
    steps = np.arange(length, dtype=np.float32)
    seasonal = np.sin(steps / 18.0) + 0.45 * np.cos(steps / 7.5)
    trend = np.linspace(0.0, 1.0, length, dtype=np.float32)
    noise = 0.08 * np.sin(steps / math.pi)
    return (4.5 + seasonal + trend + noise).astype(np.float32)


def locate_dataset() -> tuple[Path | None, str]:
    candidates = [
        APP_ROOT / "data" / "ETTh1.csv",
        APP_ROOT / "data" / "sample_load.csv",
        WORKSPACE_ROOT / "REV-2" / "ETTh1 final.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate, candidate.name

    return None, "synthetic"


def load_series() -> tuple[np.ndarray, dict]:
    dataset_path, source_name = locate_dataset()

    if dataset_path is None:
        series = _generate_synthetic_series()
        return series, {"source": source_name, "points": len(series), "target_column": "synthetic_load"}

    frame = pd.read_csv(dataset_path)
    numeric_columns = frame.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_columns:
        raise ValueError(f"No numeric forecasting column found in dataset: {dataset_path}")

    preferred_columns = ["Load", "load", "OT", "value", "y"]
    target_column = next((col for col in preferred_columns if col in numeric_columns), numeric_columns[0])
    series = frame[target_column].astype("float32").to_numpy()

    return series, {"source": str(dataset_path), "points": len(series), "target_column": target_column}


def create_windows(series: np.ndarray, window_size: int = WINDOW_SIZE) -> tuple[np.ndarray, np.ndarray]:
    features = []
    targets = []

    for start in range(len(series) - window_size):
        end = start + window_size
        features.append(series[start:end])
        targets.append(series[end])

    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def normalize_series(series: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(series))
    std = float(np.std(series))
    std = std if std > 1e-8 else 1.0
    normalized = ((series - mean) / std).astype(np.float32)
    return normalized, mean, std
