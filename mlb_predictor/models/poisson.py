from __future__ import annotations

import math
from typing import Sequence

from .common import clamp_probability


def _poisson_pmf(k: int, rate: float) -> float:
    return math.exp(-rate + k * math.log(rate) - math.lgamma(k + 1))


class PoissonModel:
    name = "poisson"

    def __init__(self, run_diff_index: int, total_index: int, max_runs: int = 20) -> None:
        self.run_diff_index, self.total_index, self.max_runs = run_diff_index, total_index, max_runs

    def predict_home_probability(self, rows: Sequence[Sequence[float]]) -> list[float]:
        results = []
        for row in rows:
            total = max(2.0, float(row[self.total_index]))
            diff = max(-total + 0.1, min(total - 0.1, float(row[self.run_diff_index])))
            home_rate, away_rate = max(0.05, (total + diff) / 2), max(0.05, (total - diff) / 2)
            home_win = tie = 0.0
            away_cdf = 0.0
            for home_runs in range(self.max_runs + 1):
                if home_runs > 0:
                    away_cdf += _poisson_pmf(home_runs - 1, away_rate)
                home_mass = _poisson_pmf(home_runs, home_rate)
                home_win += home_mass * away_cdf
                tie += home_mass * _poisson_pmf(home_runs, away_rate)
            # MLB cannot tie; split the extra-inning mass evenly absent a trained extras model.
            results.append(clamp_probability(home_win + 0.5 * tie))
        return results

