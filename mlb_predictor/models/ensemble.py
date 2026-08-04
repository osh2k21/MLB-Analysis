from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from ..calibration import IdentityCalibrator
from .common import ProbabilityModel, clamp_probability


@dataclass(frozen=True)
class ModelVote:
    model: str
    home_probability: float
    weight: float


@dataclass(frozen=True)
class EnsemblePrediction:
    raw_home_probability: float
    home_probability: float
    votes: tuple[ModelVote, ...]


class EnsembleBundle:
    """Serializable fitted ensemble. No model is optional at prediction time."""

    REQUIRED_MODELS = ("xgboost", "lightgbm", "logistic", "elo", "poisson")

    def __init__(
        self,
        models: Mapping[str, ProbabilityModel],
        weights: Mapping[str, float],
        feature_names: Sequence[str],
        calibrator: Any | None = None,
        feature_means: Sequence[float] | None = None,
        trained_through: str | None = None,
    ) -> None:
        missing = set(self.REQUIRED_MODELS) - set(models)
        if missing:
            raise ValueError(f"ensemble missing required models: {sorted(missing)}")
        if list(weights) and set(weights) != set(models):
            raise ValueError("model and weight keys differ")
        total = sum(float(weights.get(name, 0)) for name in models)
        if total <= 0:
            raise ValueError("ensemble weights must sum above zero")
        self.models = dict(models)
        self.weights = {name: float(weights[name]) / total for name in models}
        self.feature_names = list(feature_names)
        self.calibrator = calibrator or IdentityCalibrator()
        self.feature_means = list(feature_means or [0.0] * len(feature_names))
        self.trained_through = trained_through
        self.created_at = datetime.now(timezone.utc).isoformat()

    def predict(self, rows: Sequence[Sequence[float]]) -> list[EnsemblePrediction]:
        if not getattr(self.calibrator, "fitted", False):
            raise RuntimeError("ensemble calibrator is not fitted")
        model_probabilities = {name: model.predict_home_probability(rows) for name, model in self.models.items()}
        output = []
        for index in range(len(rows)):
            votes = tuple(
                ModelVote(name, clamp_probability(model_probabilities[name][index]), self.weights[name])
                for name in self.models
            )
            raw = sum(v.home_probability * v.weight for v in votes)
            calibrated = self.calibrator.transform([raw])[0]
            output.append(EnsemblePrediction(raw, clamp_probability(calibrated), votes))
        return output

