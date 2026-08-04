from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..contracts import DataStatus, Evidence
from .base import HttpClient, ProviderError
from .mlb import MLBStatsClient

LEADERBOARD_URL = "https://baseballsavant.mlb.com/leaderboard/custom"
_BATTER_METRICS = ("barrel_batted_rate", "xwoba", "exit_velocity_avg", "hard_hit_percent")
_PITCHER_METRICS = ("xwoba", "exit_velocity_avg", "hard_hit_percent", "k_percent")


class StatcastClient:
    """Team-level Statcast composites from Baseball Savant's free public leaderboard.

    Savant's custom leaderboard CSV is free/no-key but player-level only, so
    this joins it against each team's active roster (also free, MLB Stats API)
    and aggregates weighted by plate appearances faced. This exists to give the
    validator's "Statcast updated" gate and the UI real data; the fitted model
    does not consume these values as training features (that would require a
    historical backfill and retrain, out of scope here).
    """

    def __init__(self, http: HttpClient, mlb: MLBStatsClient) -> None:
        self.http = http
        self.mlb = mlb

    def _roster_ids(self, team_id: int) -> set[int]:
        result = self.mlb.http.get_json(
            f"{self.mlb.base_url}/teams/{team_id}/roster",
            {"rosterType": "active"},
            ttl_seconds=3600,
            validator=lambda x: isinstance(x, dict) and isinstance(x.get("roster"), list),
        )
        return {
            int(entry["person"]["id"])
            for entry in result.data.get("roster", [])
            if entry.get("person", {}).get("id")
        }

    def _leaderboard(self, year: int, player_type: str, metrics: tuple[str, ...]) -> list[dict[str, Any]]:
        selections = ",".join(("player_id", "pa", *metrics))
        result = self.http.get_csv(
            LEADERBOARD_URL,
            {"year": year, "type": player_type, "min": 1, "selections": selections, "csv": "true"},
            ttl_seconds=6 * 60 * 60,
        )
        return result.data

    @staticmethod
    def _weighted(rows: list[dict[str, Any]], roster: set[int], metrics: tuple[str, ...]) -> dict[str, float]:
        totals = {metric: 0.0 for metric in metrics}
        weight_sum = 0.0
        for row in rows:
            try:
                player_id = int(row.get("player_id") or 0)
            except (TypeError, ValueError):
                continue
            if player_id not in roster:
                continue
            try:
                weight = float(row.get("pa") or 0)
            except (TypeError, ValueError):
                weight = 0.0
            if weight <= 0:
                continue
            for metric in metrics:
                try:
                    totals[metric] += weight * float(row.get(metric) or 0)
                except (TypeError, ValueError):
                    continue
            weight_sum += weight
        if weight_sum <= 0:
            return {}
        return {metric: totals[metric] / weight_sum for metric in metrics}

    def _team_composite(self, team_id: int, year: int) -> dict[str, float]:
        roster = self._roster_ids(team_id)
        batting = self._weighted(self._leaderboard(year, "batter", _BATTER_METRICS), roster, _BATTER_METRICS)
        pitching = self._weighted(self._leaderboard(year, "pitcher", _PITCHER_METRICS), roster, _PITCHER_METRICS)
        composite = {f"batting_{name}": value for name, value in batting.items()}
        composite.update({f"pitching_{name}": value for name, value in pitching.items()})
        return composite

    def fetch(self, home_team_id: int, away_team_id: int, season: int) -> Evidence:
        now = datetime.now(timezone.utc)
        try:
            payload = {
                "home": self._team_composite(home_team_id, season),
                "away": self._team_composite(away_team_id, season),
            }
            if not payload["home"] or not payload["away"]:
                return Evidence(
                    "Baseball Savant", DataStatus.MISSING, now,
                    detail="no roster-matched Statcast rows for this season yet",
                )
            return Evidence("Baseball Savant", DataStatus.OK, now, payload)
        except ProviderError as exc:
            return Evidence("Baseball Savant", DataStatus.ERROR, now, detail=str(exc))
