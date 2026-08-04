from __future__ import annotations

from typing import Sequence

from .common import clamp_probability


class EloModel:
    name = "elo"

    def __init__(self, elo_index: int, home_advantage_elo: float = 24.0) -> None:
        self.elo_index = elo_index
        self.home_advantage_elo = home_advantage_elo

    def predict_home_probability(self, rows: Sequence[Sequence[float]]) -> list[float]:
        out = []
        for row in rows:
            rating_diff = float(row[self.elo_index]) + self.home_advantage_elo
            out.append(clamp_probability(1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))))
        return out

