from typing import Mapping

from .catalog import OFFENSE
from .differential import engineer_differences


def engineer_batting(home: Mapping[str, float], away: Mapping[str, float]) -> dict[str, float]:
    return engineer_differences(home, away, OFFENSE)

