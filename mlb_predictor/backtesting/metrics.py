from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_probability: float
    win_rate: float


@dataclass(frozen=True)
class BacktestReport:
    samples: int
    win_rate: float
    brier_score: float
    log_loss: float
    roi: float | None
    calibration: tuple[CalibrationBin, ...]


def evaluate_predictions(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    decimal_odds: Sequence[float | None] | None = None,
    bins: int = 10,
) -> BacktestReport:
    if len(probabilities) != len(outcomes) or not probabilities:
        raise ValueError("probabilities and outcomes must have equal non-zero length")
    pairs = [(max(1e-15, min(1 - 1e-15, float(p))), int(y)) for p, y in zip(probabilities, outcomes)]
    if any(y not in (0, 1) for _, y in pairs):
        raise ValueError("outcomes must be binary")
    n = len(pairs)
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    loss = -sum(y * math.log(p) + (1 - y) * math.log(1 - p) for p, y in pairs) / n
    calibration = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        bucket = [(p, y) for p, y in pairs if lo <= p < hi or (i == bins - 1 and p == hi)]
        if bucket:
            calibration.append(CalibrationBin(lo, hi, len(bucket), sum(p for p, _ in bucket) / len(bucket), sum(y for _, y in bucket) / len(bucket)))
    roi = None
    if decimal_odds is not None:
        if len(decimal_odds) != n:
            raise ValueError("decimal_odds length differs")
        settled = [(y, odd) for (_, y), odd in zip(pairs, decimal_odds) if odd is not None and float(odd) > 1]
        if settled:
            profit = sum((float(odd) - 1.0) if y else -1.0 for y, odd in settled)
            roi = profit / len(settled)
    return BacktestReport(n, sum(y for _, y in pairs) / n, brier, loss, roi, tuple(calibration))

