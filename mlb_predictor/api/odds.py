from __future__ import annotations

from datetime import datetime, timezone

from ..contracts import DataStatus, Evidence
from .base import HttpClient, ProviderError


class OddsClient:
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

    def __init__(self, http: HttpClient, api_key: str | None) -> None:
        self.http, self.api_key = http, api_key

    def games(self) -> Evidence:
        now = datetime.now(timezone.utc)
        if not self.api_key:
            return Evidence("The Odds API", DataStatus.MISSING, now, detail="ODDS_API_KEY is not configured")
        try:
            result = self.http.get_json(
                self.url,
                {"apiKey": self.api_key, "regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american"},
                ttl_seconds=180,
                validator=lambda x: isinstance(x, list),
            )
            return Evidence("The Odds API", DataStatus.OK, result.fetched_at, result.data, cache_hit=result.cache_hit)
        except ProviderError as exc:
            return Evidence("The Odds API", DataStatus.ERROR, now, detail=str(exc))

