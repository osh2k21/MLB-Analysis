from __future__ import annotations

import math
from typing import Sequence


def _logit(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, float(p)))
    return math.log(p / (1 - p))


class IdentityCalibrator:
    name = "identity"
    fitted = True

    def fit(self, probabilities: Sequence[float], outcomes: Sequence[int]) -> "IdentityCalibrator":
        return self

    def transform(self, probabilities: Sequence[float]) -> list[float]:
        return [max(0.001, min(0.999, float(p))) for p in probabilities]


class IsotonicCalibrator:
    name = "isotonic"

    def __init__(self) -> None:
        self.model = None

    @property
    def fitted(self) -> bool:
        return self.model is not None

    def fit(self, probabilities: Sequence[float], outcomes: Sequence[int]) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression
        self.model = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
        self.model.fit(list(probabilities), list(outcomes))
        return self

    def transform(self, probabilities: Sequence[float]) -> list[float]:
        if self.model is None:
            raise RuntimeError("isotonic calibrator is not fitted")
        return [float(x) for x in self.model.predict(list(probabilities))]


class PlattCalibrator:
    name = "platt"

    def __init__(self) -> None:
        self.model = None

    @property
    def fitted(self) -> bool:
        return self.model is not None

    def fit(self, probabilities: Sequence[float], outcomes: Sequence[int]) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(C=1.0, solver="lbfgs")
        self.model.fit([[_logit(p)] for p in probabilities], list(outcomes))
        return self

    def transform(self, probabilities: Sequence[float]) -> list[float]:
        if self.model is None:
            raise RuntimeError("Platt calibrator is not fitted")
        return [float(x) for x in self.model.predict_proba([[_logit(p)] for p in probabilities])[:, 1]]

