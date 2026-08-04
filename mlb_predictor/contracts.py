from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


class DataStatus(str, Enum):
    OK = "ok"
    MISSING = "missing"
    STALE = "stale"
    INVALID = "invalid"
    ERROR = "error"


@dataclass(frozen=True)
class Evidence:
    source: str
    status: DataStatus
    observed_at: datetime
    payload: Any = None
    detail: str = ""
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cache_hit: bool = False

    @property
    def age_seconds(self) -> float:
        now = datetime.now(timezone.utc)
        observed = self.observed_at if self.observed_at.tzinfo else self.observed_at.replace(tzinfo=timezone.utc)
        return max(0.0, (now - observed).total_seconds())

    @property
    def usable(self) -> bool:
        return self.status is DataStatus.OK and self.payload is not None


@dataclass(frozen=True)
class GameRef:
    game_pk: int
    game_date: str
    commence_time: datetime
    away_team_id: int
    home_team_id: int
    away_team: str
    home_team: str
    venue_id: int | None
    venue_name: str


@dataclass(frozen=True)
class FeatureSnapshot:
    game: GameRef
    values: Mapping[str, float]
    provenance: Mapping[str, str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def ordered(self, names: list[str]) -> list[float]:
        return [float(self.values[name]) for name in names]

