from __future__ import annotations

from datetime import datetime, timezone

from ..contracts import DataStatus, Evidence
from .base import HttpClient, ProviderError


class WeatherClient:
    """Free Open-Meteo forecast client; no account or API key required."""

    def __init__(self, http: HttpClient, base_url: str = "https://api.open-meteo.com/v1") -> None:
        self.http, self.base_url = http, base_url.rstrip("/")

    def forecast(self, latitude: float | None, longitude: float | None, at: datetime) -> Evidence:
        now = datetime.now(timezone.utc)
        if latitude is None or longitude is None:
            return Evidence("Open-Meteo", DataStatus.MISSING, now, detail="venue coordinates unavailable")
        target = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
        try:
            result = self.http.get_json(
                f"{self.base_url}/forecast",
                {
                    "latitude": latitude, "longitude": longitude,
                    "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation_probability",
                    "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                    "timezone": "UTC", "forecast_days": 16,
                },
                ttl_seconds=600,
                validator=lambda x: isinstance(x, dict) and isinstance(x.get("hourly", {}).get("time"), list),
            )
            hourly = result.data["hourly"]
            times = [datetime.fromisoformat(value).replace(tzinfo=timezone.utc) for value in hourly["time"]]
            index = min(range(len(times)), key=lambda i: abs((times[i] - target).total_seconds()))
            if abs((times[index] - target).total_seconds()) > 90 * 60:
                return Evidence("Open-Meteo", DataStatus.MISSING, result.fetched_at, detail="no hourly forecast near first pitch")
            payload = {key: values[index] for key, values in hourly.items() if key != "time" and isinstance(values, list) and len(values) > index}
            payload["forecast_time"] = times[index].isoformat()
            return Evidence("Open-Meteo", DataStatus.OK, result.fetched_at, payload, cache_hit=result.cache_hit)
        except (ProviderError, ValueError) as exc:
            return Evidence("Open-Meteo", DataStatus.ERROR, now, detail=str(exc))

