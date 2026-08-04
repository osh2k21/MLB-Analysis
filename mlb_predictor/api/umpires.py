from __future__ import annotations

from datetime import datetime, timezone

from ..contracts import DataStatus, Evidence

_OFFICIAL_TYPE_KEYS = {
    "Home Plate": "home_plate",
    "First Base": "first_base",
    "Second Base": "second_base",
    "Third Base": "third_base",
}


class UmpireClient:
    """Extracts the umpire crew from the MLB live game feed.

    MLB's public feed only publishes the umpire crew once a game goes live; it
    is empty for every pregame/scheduled game (confirmed against the real API).
    There is no free source for pregame umpire assignments, so absence here is
    a normal MISSING state, not an error, and the validator treats it as
    advisory rather than a hard gate.
    """

    def from_game_feed(self, feed: Evidence) -> Evidence:
        now = datetime.now(timezone.utc)
        if not feed.usable:
            return Evidence("MLB Stats API", DataStatus.MISSING, now, detail="game feed unavailable")
        payload = feed.payload
        officials = (
            payload.get("liveData", {}).get("boxscore", {}).get("officials")
            or payload.get("gameData", {}).get("officials")
            or []
        )
        crew: dict[str, str] = {}
        for entry in officials:
            key = _OFFICIAL_TYPE_KEYS.get(entry.get("officialType"))
            name = (entry.get("official") or {}).get("fullName")
            if key and name:
                crew[key] = name
        if not crew.get("home_plate"):
            return Evidence(
                "MLB Stats API", DataStatus.MISSING, feed.observed_at,
                detail="umpire crew not yet published by MLB (typically posted once the game goes live)",
            )
        return Evidence("MLB Stats API", DataStatus.OK, feed.observed_at, crew, cache_hit=feed.cache_hit)
