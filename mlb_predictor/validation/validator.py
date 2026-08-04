from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

from ..contracts import DataStatus, Evidence, GameRef


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    detail: str
    source: str
    age_seconds: float | None = None
    advisory: bool = False


@dataclass(frozen=True)
class ValidationReport:
    game_pk: int
    checks: tuple[ValidationCheck, ...]
    checked_at: datetime

    @property
    def ready(self) -> bool:
        return all(item.passed for item in self.checks if not item.advisory)

    @property
    def decision(self) -> str:
        return "Ready" if self.ready else "Prediction Withheld"

    @property
    def failures(self) -> tuple[ValidationCheck, ...]:
        return tuple(item for item in self.checks if not item.passed)


class GameValidator:
    """Eight hard gates. A single failure withholds the prediction."""

    MAX_AGES = {
        "starter_confirmed": 15 * 60,
        "lineup_confirmed": 15 * 60,
        "bullpen_updated": 60 * 60,
        "weather_available": 30 * 60,
        "odds_available": 10 * 60,
        "statcast_updated": 24 * 60 * 60,
        "injury_feed_updated": 60 * 60,
        "umpire_assigned": 60 * 60,
    }

    @staticmethod
    def _fresh(evidence: Evidence | None, max_age: float) -> tuple[bool, str, float | None]:
        if evidence is None:
            return False, "evidence not supplied", None
        if evidence.status is not DataStatus.OK or evidence.payload is None:
            return False, evidence.detail or evidence.status.value, evidence.age_seconds
        if evidence.age_seconds > max_age:
            return False, f"stale ({evidence.age_seconds / 60:.0f} minutes old)", evidence.age_seconds
        return True, "verified and fresh", evidence.age_seconds

    @staticmethod
    def _pregame_feed_flags(game: GameRef, evidence: Evidence | None) -> tuple[bool, bool, str]:
        if evidence is None or not evidence.usable:
            return False, False, "MLB live feed unavailable"
        data = evidence.payload
        probable = data.get("gameData", {}).get("probablePitchers", {})
        starter_ok = bool(probable.get("home", {}).get("id") and probable.get("away", {}).get("id"))
        box = data.get("liveData", {}).get("boxscore", {}).get("teams", {})
        home_order = box.get("home", {}).get("battingOrder") or []
        away_order = box.get("away", {}).get("battingOrder") or []
        lineup_ok = len(home_order) >= 9 and len(away_order) >= 9
        return starter_ok, lineup_ok, f"game {game.game_pk} MLB feed"

    def validate(self, game: GameRef, evidence: Mapping[str, Evidence]) -> ValidationReport:
        mlb = evidence.get("mlb_feed")
        starter_flag, lineup_flag, mlb_detail = self._pregame_feed_flags(game, mlb)
        mlb_fresh, mlb_fresh_detail, mlb_age = self._fresh(mlb, self.MAX_AGES["starter_confirmed"])
        checks = [
            ValidationCheck("Starter confirmed", mlb_fresh and starter_flag, mlb_detail if starter_flag else "both probable starters not confirmed", mlb.source if mlb else "MLB Stats API", mlb_age),
            ValidationCheck("Lineup confirmed", mlb_fresh and lineup_flag, mlb_detail if lineup_flag else "both nine-player batting orders not posted", mlb.source if mlb else "MLB Stats API", mlb_age),
        ]

        mapping = (
            ("Bullpen updated", "bullpen", "bullpen_updated", lambda x: isinstance(x, dict) and bool(x.get("home")) and bool(x.get("away")), False),
            ("Weather available", "weather", "weather_available", lambda x: isinstance(x, dict) and ("main" in x or "temperature_f" in x or "temperature_2m" in x), False),
            ("Odds available", "odds", "odds_available", lambda x: isinstance(x, dict) and bool(x.get("home")) and bool(x.get("away")), False),
            ("Statcast updated", "statcast", "statcast_updated", lambda x: isinstance(x, dict) and bool(x.get("home")) and bool(x.get("away")), False),
            ("Injury feed updated", "injuries", "injury_feed_updated", lambda x: isinstance(x, dict) and "home" in x and "away" in x, False),
            # MLB's free feed only publishes the umpire crew once a game goes live, so
            # this can never be "ready" pregame. Track and surface it, but don't withhold on it.
            ("Umpire assigned", "umpire", "umpire_assigned", lambda x: isinstance(x, dict) and bool(x.get("home_plate")), True),
        )
        for label, key, age_key, predicate, advisory in mapping:
            item = evidence.get(key)
            ok, detail, age = self._fresh(item, self.MAX_AGES[age_key])
            if ok and not predicate(item.payload):
                ok, detail = False, "payload missing required game fields"
            checks.append(ValidationCheck(label, ok, detail, item.source if item else key, age, advisory))
        return ValidationReport(game.game_pk, tuple(checks), datetime.now(timezone.utc))
