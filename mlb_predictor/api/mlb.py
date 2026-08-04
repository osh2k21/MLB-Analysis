from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..contracts import DataStatus, Evidence, GameRef
from .base import HttpClient, ProviderError


class MLBStatsClient:
    base_url = "https://statsapi.mlb.com/api/v1"

    def __init__(self, http: HttpClient) -> None:
        self.http = http

    def schedule(self, date: str) -> Evidence:
        try:
            result = self.http.get_json(
                f"{self.base_url}/schedule",
                {"sportId": 1, "date": date, "hydrate": "probablePitcher,lineups,weather,team,venue,status"},
                ttl_seconds=60,
                validator=lambda x: isinstance(x, dict) and "dates" in x,
            )
            games: list[dict[str, Any]] = []
            for date_block in result.data.get("dates", []):
                games.extend(date_block.get("games", []))
            return Evidence("MLB Stats API", DataStatus.OK, result.fetched_at, games, cache_hit=result.cache_hit)
        except ProviderError as exc:
            return Evidence("MLB Stats API", DataStatus.ERROR, datetime.now(timezone.utc), detail=str(exc))

    @staticmethod
    def game_ref(raw: dict[str, Any]) -> GameRef:
        teams, venue = raw["teams"], raw.get("venue", {})
        commence = datetime.fromisoformat(raw["gameDate"].replace("Z", "+00:00"))
        return GameRef(
            game_pk=int(raw["gamePk"]), game_date=raw.get("officialDate", commence.date().isoformat()),
            commence_time=commence, away_team_id=int(teams["away"]["team"]["id"]),
            home_team_id=int(teams["home"]["team"]["id"]), away_team=teams["away"]["team"]["name"],
            home_team=teams["home"]["team"]["name"], venue_id=venue.get("id"),
            venue_name=venue.get("name", ""),
        )

    def game_feed(self, game_pk: int) -> Evidence:
        try:
            result = self.http.get_json(
                f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
                ttl_seconds=45,
                validator=lambda x: isinstance(x, dict) and "gameData" in x and "liveData" in x,
            )
            stamp = result.data.get("metaData", {}).get("timeStamp")
            observed = result.fetched_at
            if stamp:
                try:
                    observed = datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            return Evidence("MLB Stats API", DataStatus.OK, observed, result.data, cache_hit=result.cache_hit)
        except ProviderError as exc:
            return Evidence("MLB Stats API", DataStatus.ERROR, datetime.now(timezone.utc), detail=str(exc))

    def team_stats(self, team_id: int, season: int, group: str) -> Evidence:
        try:
            result = self.http.get_json(
                f"{self.base_url}/teams/{team_id}/stats",
                {"stats": "season,seasonAdvanced", "group": group, "season": season},
                ttl_seconds=1800,
                validator=lambda x: isinstance(x, dict) and isinstance(x.get("stats"), list),
            )
            return Evidence("MLB Stats API", DataStatus.OK, result.fetched_at, result.data, cache_hit=result.cache_hit)
        except ProviderError as exc:
            return Evidence("MLB Stats API", DataStatus.ERROR, datetime.now(timezone.utc), detail=str(exc))

