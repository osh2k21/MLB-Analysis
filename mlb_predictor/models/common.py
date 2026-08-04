from __future__ import annotations

from typing import Protocol, Sequence


class ProbabilityModel(Protocol):
    name: str

    def predict_home_probability(self, rows: Sequence[Sequence[float]]) -> list[float]: ...


def clamp_probability(value: float) -> float:
    return max(0.001, min(0.999, float(value)))

