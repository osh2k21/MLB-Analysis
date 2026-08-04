from typing import Mapping

from .catalog import PITCHING
from .differential import engineer_differences


def engineer_pitching(home: Mapping[str, float], away: Mapping[str, float]) -> dict[str, float]:
    return engineer_differences(home, away, PITCHING)

