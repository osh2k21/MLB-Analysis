from __future__ import annotations

from typing import Mapping, Sequence


def engineer_differences(home: Mapping[str, float], away: Mapping[str, float], outputs: Sequence[str]) -> dict[str, float]:
    engineered = {}
    for output in outputs:
        if not output.endswith("_diff"):
            continue
        raw_name = output[:-5]
        if raw_name in home and raw_name in away:
            engineered[output] = float(home[raw_name]) - float(away[raw_name])
    return engineered

