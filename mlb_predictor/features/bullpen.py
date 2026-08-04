from typing import Mapping

from .catalog import BULLPEN
from .differential import engineer_differences


def engineer_bullpen(home: Mapping[str, float], away: Mapping[str, float]) -> dict[str, float]:
    return engineer_differences(home, away, BULLPEN)

