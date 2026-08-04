from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
from typing import Any

from ..contracts import DataStatus, Evidence
from .base import HttpClient, ProviderError

_IL_PLACED = re.compile(r"placed .* on the .*injured list", re.IGNORECASE)
_IL_ACTIVATED = re.compile(r"activated .* from the .*injured list", re.IGNORECASE)


class InjuryClient:
    """Derives each team's currently active injured list from official MLB transactions.

    Free, no key: statsapi.mlb.com publishes every roster transaction, including
    IL placements and activations. A player is "currently injured" if their most
    recent IL-related transaction within the lookback window is a placement with
    no later activation.
    """

    base_url = "https://statsapi.mlb.com/api/v1"

    def __init__(self, http: HttpClient, lookback_days: int = 60) -> None:
        self.http = http
        self.lookback_days = lookback_days

    def _team_active_il(self, team_id: int) -> list[dict[str, Any]]:
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=self.lookback_days)
        result = self.http.get_json(
            f"{self.base_url}/transactions",
            {"teamId": team_id, "startDate": start.isoformat(), "endDate": end.isoformat()},
            ttl_seconds=1800,
            validator=lambda x: isinstance(x, dict) and isinstance(x.get("transactions"), list),
        )
        active: dict[int, dict[str, Any]] = {}
        for txn in sorted(result.data.get("transactions", []), key=lambda t: t.get("date", "")):
            description = str(txn.get("description", ""))
            person = txn.get("person") or {}
            player_id = person.get("id")
            if player_id is None:
                continue
            if _IL_PLACED.search(description):
                active[player_id] = {
                    "player_id": player_id,
                    "name": person.get("fullName", ""),
                    "since": txn.get("date"),
                    "description": description,
                }
            elif _IL_ACTIVATED.search(description):
                active.pop(player_id, None)
        return list(active.values())

    def fetch(self, home_team_id: int, away_team_id: int) -> Evidence:
        now = datetime.now(timezone.utc)
        try:
            payload = {
                "home": self._team_active_il(home_team_id),
                "away": self._team_active_il(away_team_id),
            }
            return Evidence("MLB Stats API transactions", DataStatus.OK, now, payload)
        except ProviderError as exc:
            return Evidence("MLB Stats API transactions", DataStatus.ERROR, now, detail=str(exc))
