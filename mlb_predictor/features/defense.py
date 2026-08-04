from typing import Mapping

from .catalog import DEFENSE
from .differential import engineer_differences


def engineer_defense(home: Mapping[str, float], away: Mapping[str, float]) -> dict[str, float]:
    return engineer_differences(home, away, DEFENSE)

