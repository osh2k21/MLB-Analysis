from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import threading
import time
from typing import Any, Callable, Mapping


class ProviderError(RuntimeError):
    pass


class RateLimiter:
    def __init__(self, requests_per_second: float = 3.0) -> None:
        self.interval = 1.0 / max(requests_per_second, 0.01)
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self.interval - (time.monotonic() - self._last)
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


class SQLiteCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS http_cache (cache_key TEXT PRIMARY KEY, body TEXT NOT NULL, "
                "stored_at TEXT NOT NULL, expires_at REAL NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def get(self, key: str) -> Any | None:
        with self._connect() as conn:
            row = conn.execute("SELECT body, expires_at FROM http_cache WHERE cache_key=?", (key,)).fetchone()
        if not row or row[1] < time.time():
            return None
        return json.loads(row[0])

    def put(self, key: str, value: Any, ttl_seconds: int) -> None:
        body = json.dumps(value, separators=(",", ":"), default=str)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO http_cache(cache_key, body, stored_at, expires_at) VALUES(?,?,?,?)",
                (key, body, datetime.now(timezone.utc).isoformat(), time.time() + ttl_seconds),
            )


@dataclass(frozen=True)
class HttpResult:
    data: Any
    fetched_at: datetime
    cache_hit: bool
    headers: Mapping[str, str]


class HttpClient:
    """GET-only JSON client with bounded retry, jitter, rate limiting and TTL cache."""

    def __init__(self, cache_path: Path, timeout: float = 12.0, attempts: int = 3, requests_per_second: float = 3.0) -> None:
        self.cache = SQLiteCache(cache_path)
        self.timeout = timeout
        self.attempts = max(1, attempts)
        self.limiter = RateLimiter(requests_per_second)

    @staticmethod
    def _key(url: str, params: Mapping[str, Any] | None) -> str:
        raw = json.dumps([url, sorted((params or {}).items())], separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        ttl_seconds: int,
        validator: Callable[[Any], bool] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResult:
        key = self._key(url, params)
        cached = self.cache.get(key)
        if cached is not None and (validator is None or validator(cached)):
            return HttpResult(cached, datetime.now(timezone.utc), True, {})

        try:
            import requests
        except ImportError as exc:
            raise ProviderError("The requests package is required; install requirements.txt") from exc

        last_error: Exception | None = None
        for attempt in range(self.attempts):
            self.limiter.wait()
            try:
                response = requests.get(url, params=params, headers=dict(headers or {}), timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    raise ProviderError(f"retryable HTTP {response.status_code}")
                response.raise_for_status()
                data = response.json()
                if validator is not None and not validator(data):
                    raise ProviderError("response failed schema validation")
                self.cache.put(key, data, ttl_seconds)
                return HttpResult(data, datetime.now(timezone.utc), False, dict(response.headers))
            except Exception as exc:  # retries network, HTTP, decoding, and schema failures
                last_error = exc
                if attempt + 1 < self.attempts:
                    time.sleep(min(4.0, (2 ** attempt) * 0.4 + random.random() * 0.2))
        raise ProviderError(f"request failed after {self.attempts} attempts: {last_error}")

