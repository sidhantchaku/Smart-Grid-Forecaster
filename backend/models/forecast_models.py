import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _fit_ridge(features: np.ndarray, targets: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, float]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)

    x_bias = np.hstack([x, np.ones((x.shape[0], 1), dtype=np.float32)])
    identity = np.eye(x_bias.shape[1], dtype=np.float32)
    identity[-1, -1] = 0.0

    solution = np.linalg.solve(x_bias.T @ x_bias + alpha * identity, x_bias.T @ y)
    return solution[:-1], float(solution[-1])


def _apply_linear(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return np.asarray(features, dtype=np.float32) @ np.asarray(weights, dtype=np.float32) + bias


def _window_summaries(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    idx = np.arange(x.shape[1], dtype=np.float32)
    centered_idx = idx - np.mean(idx)
    slopes = ((x - x.mean(axis=1, keepdims=True)) * centered_idx).sum(axis=1) / (centered_idx**2).sum()

    features = np.column_stack(
        [
            x[:, -1],
            x[:, -3:].mean(axis=1),
            x[:, -6:].mean(axis=1),
            x[:, -12:].mean(axis=1),
            x[:, -24:].mean(axis=1),
            x[:, -48:].mean(axis=1),
            x.mean(axis=1),
            x.std(axis=1),
            x.min(axis=1),
            x.max(axis=1),
            slopes,
        ]
    )
    return features.astype(np.float32)


def _attention_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    recent = x[:, -24:]
    weights = np.linspace(1.0, 3.0, recent.shape[1], dtype=np.float32)
    weights = weights / weights.sum()
    weighted_recent = recent @ weights

    return np.column_stack(
        [
            weighted_recent,
            recent[:, -1],
            recent[:, -6:].mean(axis=1),
            recent[:, -12:].mean(axis=1),
            recent[:, -24:].std(axis=1),
            x[:, -24],
            x[:, -48],
            x[:, -72],
            x[:, -96],
        ]
    ).astype(np.float32)


def _trend_seasonal_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    last_day = x[:, -24:]
    prev_day = x[:, -48:-24]
    prev_two_day = x[:, -72:-48]
    prev_three_day = x[:, -96:-72]

    day_trend = last_day.mean(axis=1) - prev_day.mean(axis=1)
    cycle_strength = (
        0.5 * np.abs(last_day.mean(axis=1) - prev_day.mean(axis=1))
        + 0.3 * np.abs(prev_day.mean(axis=1) - prev_two_day.mean(axis=1))
        + 0.2 * np.abs(prev_two_day.mean(axis=1) - prev_three_day.mean(axis=1))
    )

    return np.column_stack(
        [
            x[:, -1],
            x[:, -24],
            x[:, -48],
            x[:, -72],
            last_day.mean(axis=1),
            prev_day.mean(axis=1),
            prev_two_day.mean(axis=1),
            day_trend,
            cycle_strength,
            x[:, -12:].mean(axis=1) - x[:, -24:].mean(axis=1),
        ]
    ).astype(np.float32)


@dataclass
class SerializableModel:
    name: str
    kind: str
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        def convert(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            return value

        return {
            "name": self.name,
            "kind": self.kind,
            "params": {key: convert(value) for key, value in self.params.items()},
        }

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


class LastValueModel(SerializableModel):
    def __init__(self, name: str = "baseline"):
        super().__init__(name=name, kind="last_value")

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        return None

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x[:, -1], dtype=np.float32)


class LinearLagModel(SerializableModel):
    def __init__(self, name: str, lag_count: int):
        super().__init__(name=name, kind="linear_lag", params={"lag_count": lag_count})

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        lag_count = int(self.params["lag_count"])
        weights, bias = _fit_ridge(x[:, -lag_count:], y, alpha=0.8)
        self.params["weights"] = weights
        self.params["bias"] = bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        lag_count = int(self.params["lag_count"])
        return _apply_linear(x[:, -lag_count:], self.params["weights"], self.params["bias"])


class SummaryRegressionModel(SerializableModel):
    def __init__(self, name: str = "mlp"):
        super().__init__(name=name, kind="summary_regression")

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        features = _window_summaries(x)
        weights, bias = _fit_ridge(features, y, alpha=0.5)
        self.params["weights"] = weights
        self.params["bias"] = bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        features = _window_summaries(x)
        return _apply_linear(features, self.params["weights"], self.params["bias"])


class AttentionRegressionModel(SerializableModel):
    def __init__(self, name: str = "attention"):
        super().__init__(name=name, kind="attention_regression")

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        features = _attention_features(x)
        weights, bias = _fit_ridge(features, y, alpha=0.7)
        self.params["weights"] = weights
        self.params["bias"] = bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        features = _attention_features(x)
        return _apply_linear(features, self.params["weights"], self.params["bias"])


class TrendSeasonalModel(SerializableModel):
    def __init__(self, name: str = "nbeats"):
        super().__init__(name=name, kind="trend_seasonal")

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        features = _trend_seasonal_features(x)
        weights, bias = _fit_ridge(features, y, alpha=0.6)
        self.params["weights"] = weights
        self.params["bias"] = bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        features = _trend_seasonal_features(x)
        return _apply_linear(features, self.params["weights"], self.params["bias"])


class HybridModel(SerializableModel):
    def __init__(self, name: str = "hybrid"):
        super().__init__(name=name, kind="hybrid")
        self.lag_model = LinearLagModel(name="hybrid_lag", lag_count=24)
        self.summary_model = SummaryRegressionModel(name="hybrid_summary")

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        split = max(int(len(x) * 0.8), 1)
        split = min(split, len(x) - 1)

        x_train, x_val = x[:split], x[split:]
        y_train, y_val = y[:split], y[split:]

        self.lag_model.fit(x_train, y_train)
        self.summary_model.fit(x_train, y_train)

        lag_pred = self.lag_model.predict(x_val)
        summary_pred = self.summary_model.predict(x_val)

        stacked = np.column_stack([lag_pred, summary_pred]).astype(np.float32)
        weights, bias = _fit_ridge(stacked, y_val, alpha=0.2)

        self.params["blend_weights"] = weights
        self.params["blend_bias"] = bias
        self.params["lag_model"] = self.lag_model.to_dict()
        self.params["summary_model"] = self.summary_model.to_dict()

    def predict(self, x: np.ndarray) -> np.ndarray:
        lag_pred = self.lag_model.predict(x)
        summary_pred = self.summary_model.predict(x)
        stacked = np.column_stack([lag_pred, summary_pred]).astype(np.float32)
        return _apply_linear(stacked, self.params["blend_weights"], self.params["blend_bias"])


def build_model(model_name: str):
    if model_name == "hybrid":
        return HybridModel()
    if model_name == "lstm":
        return LinearLagModel(name="lstm", lag_count=24)
    if model_name == "mlp":
        return SummaryRegressionModel()
    if model_name == "attention":
        return AttentionRegressionModel()
    if model_name == "nbeats":
        return TrendSeasonalModel()
    if model_name == "baseline":
        return LastValueModel()
    raise ValueError(f"Unsupported model '{model_name}'.")


def load_model(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    name = payload["name"]
    kind = payload["kind"]
    params = payload.get("params", {})

    model = build_model(name)
    if kind == "hybrid":
        model.params = {
            "blend_weights": np.asarray(params["blend_weights"], dtype=np.float32),
            "blend_bias": float(params["blend_bias"]),
            "lag_model": params["lag_model"],
            "summary_model": params["summary_model"],
        }
        model.lag_model.params = {
            "lag_count": int(params["lag_model"]["params"]["lag_count"]),
            "weights": np.asarray(params["lag_model"]["params"]["weights"], dtype=np.float32),
            "bias": float(params["lag_model"]["params"]["bias"]),
        }
        model.summary_model.params = {
            "weights": np.asarray(params["summary_model"]["params"]["weights"], dtype=np.float32),
            "bias": float(params["summary_model"]["params"]["bias"]),
        }
        return model

    if kind == "linear_lag":
        model.params = {
            "lag_count": int(params["lag_count"]),
            "weights": np.asarray(params["weights"], dtype=np.float32),
            "bias": float(params["bias"]),
        }
        return model

    if kind in {"summary_regression", "attention_regression", "trend_seasonal"}:
        model.params = {
            "weights": np.asarray(params["weights"], dtype=np.float32),
            "bias": float(params["bias"]),
        }
        return model

    if kind == "last_value":
        model.params = {}
        return model

    raise ValueError(f"Unsupported serialized model kind '{kind}'.")


MODEL_NAMES = ["hybrid", "lstm", "mlp", "attention", "nbeats", "baseline"]
