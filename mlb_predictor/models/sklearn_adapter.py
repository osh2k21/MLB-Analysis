from __future__ import annotations

from typing import Any, Sequence

from .common import clamp_probability


class SklearnProbabilityModel:
    def __init__(self, name: str, estimator: Any) -> None:
        self.name, self.estimator = name, estimator

    def predict_home_probability(self, rows: Sequence[Sequence[float]]) -> list[float]:
        predictions = self.estimator.predict_proba(rows)[:, 1]
        return [clamp_probability(x) for x in predictions]

