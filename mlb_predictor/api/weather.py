from __future__ import annotations

from datetime import datetime, timezone

from ..contracts import DataStatus, Evidence
from .base import HttpClient, ProviderError


class WeatherClient:
    def __init__(self, http: HttpClient, api_key: str | None, base_url: str) -> None:
        self.http, self.api_key, self.base_url = http, api_key, base_url.rstrip("/")

    def current(self, latitude: float | None, longitude: float | None) -> Evidence:
        now = datetime.now(timezone.utc)
        if not self.api_key:
            return Evidence("Weather API", DataStatus.MISSING, now, detail="WEATHER_API_KEY is not configured")
        if latitude is None or longitude is None:
            return Evidence("Weather API", DataStatus.MISSING, now, detail="venue coordinates unavailable")
        try:
            result = self.http.get_json(
                f"{self.base_url}/weather",
                {"lat": latitude, "lon": longitude, "appid": self.api_key, "units": "imperial"},
                ttl_seconds=300,
                validator=lambda x: isinstance(x, dict) and "main" in x and "wind" in x,
            )
            observed = datetime.fromtimestamp(result.data.get("dt", now.timestamp()), tz=timezone.utc)
            return Evidence("Weather API", DataStatus.OK, observed, result.data, cache_hit=result.cache_hit)
        except ProviderError as exc:
            return Evidence("Weather API", DataStatus.ERROR, now, detail=str(exc))

    def forecast(self, latitude: float | None, longitude: float | None, at: datetime) -> Evidence:
        now = datetime.now(timezone.utc)
        if not self.api_key:
            return Evidence("Weather API", DataStatus.MISSING, now, detail="WEATHER_API_KEY is not configured")
        if latitude is None or longitude is None:
            return Evidence("Weather API", DataStatus.MISSING, now, detail="venue coordinates unavailable")
        try:
            result = self.http.get_json(
                f"{self.base_url}/forecast",
                {"lat": latitude, "lon": longitude, "appid": self.api_key, "units": "imperial"},
                ttl_seconds=600,
                validator=lambda x: isinstance(x, dict) and isinstance(x.get("list"), list),
            )
            target = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
            candidates = result.data.get("list", [])
            if not candidates:
                return Evidence("Weather API", DataStatus.MISSING, now, detail="forecast contains no time slots")
            slot = min(candidates, key=lambda x: abs(float(x.get("dt", 0)) - target.timestamp()))
            observed = datetime.fromtimestamp(float(slot.get("dt", 0)), tz=timezone.utc)
            if abs((observed - target).total_seconds()) > 3 * 60 * 60:
                return Evidence("Weather API", DataStatus.MISSING, now, detail="no forecast within three hours of first pitch")
            slot = dict(slot)
            slot["forecast_generated_at"] = result.fetched_at.isoformat()
            # Freshness is the request time; slot dt describes forecast validity, not data age.
            return Evidence("Weather API", DataStatus.OK, result.fetched_at, slot, cache_hit=result.cache_hit)
        except ProviderError as exc:
            return Evidence("Weather API", DataStatus.ERROR, now, detail=str(exc))
