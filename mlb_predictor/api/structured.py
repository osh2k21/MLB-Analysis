from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Any

from ..contracts import DataStatus, Evidence
from .base import HttpClient, ProviderError


class StructuredFeedClient:
    """Adapter for licensed/public JSON feeds where no stable free API exists."""

    def __init__(self, name: str, url: str | None, http: HttpClient, validator: Callable[[Any], bool] | None = None) -> None:
        self.name, self.url, self.http = name, url, http
        self.validator = validator or (lambda x: isinstance(x, (dict, list)))

    def fetch(self, ttl_seconds: int = 900) -> Evidence:
        now = datetime.now(timezone.utc)
        if not self.url:
            return Evidence(self.name, DataStatus.MISSING, now, detail=f"{self.name} feed URL is not configured")
        try:
            result = self.http.get_json(self.url, ttl_seconds=ttl_seconds, validator=self.validator)
            if not isinstance(result.data, dict):
                return Evidence(self.name, DataStatus.INVALID, now, detail="feed must be a timestamped JSON object")
            raw_stamp = result.data.get("updated_at") or result.data.get("timestamp")
            if not raw_stamp:
                return Evidence(self.name, DataStatus.INVALID, now, detail="feed has no updated_at/timestamp")
            try:
                observed = datetime.fromisoformat(str(raw_stamp).replace("Z", "+00:00"))
                if observed.tzinfo is None:
                    observed = observed.replace(tzinfo=timezone.utc)
                if observed > now.replace(microsecond=0):
                    return Evidence(self.name, DataStatus.INVALID, now, detail="feed timestamp is in the future")
            except ValueError:
                return Evidence(self.name, DataStatus.INVALID, now, detail="feed timestamp is invalid")
            return Evidence(self.name, DataStatus.OK, observed, result.data, cache_hit=result.cache_hit)
        except ProviderError as exc:
            return Evidence(self.name, DataStatus.ERROR, now, detail=str(exc))
