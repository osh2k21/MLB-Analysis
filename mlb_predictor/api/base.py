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
    """Concurrency note: with up to 20 games x dozens of calls each now running
    at once (Statcast alone adds a 6-worker sub-pool per game), this file sees
    far more simultaneous short-lived connections than it used to. SQLite's
    default rollback-journal mode serializes writers and can misbehave under
    that much concurrent contention; WAL mode (set once, persists in the file
    itself) lets readers and a writer proceed without blocking each other and
    is the standard fix for exactly this kind of workload."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._ensure_schema()

    def _create_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS http_cache (cache_key TEXT PRIMARY KEY, body TEXT NOT NULL, "
                "stored_at TEXT NOT NULL, expires_at REAL NOT NULL)"
            )
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        try:
            self._create_schema()
        except sqlite3.Error:
            # The file itself is corrupted (not just missing the table, which
            # CREATE TABLE IF NOT EXISTS alone would have fixed) -- delete and
            # start over. This is a disposable cache, never a source of truth,
            # so losing it just means the next few requests are cache misses
            # instead of every prediction on the page staying permanently broken.
            # Connections above are always explicitly closed (not just left to
            # `with`, which commits but does NOT close a sqlite3 connection) so
            # this unlink never fights an open handle.
            self.path.unlink(missing_ok=True)
            self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def get(self, key: str) -> Any | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT body, expires_at FROM http_cache WHERE cache_key=?", (key,)).fetchone()
        except sqlite3.Error:
            # Self-heals a missing/corrupted cache file instead of permanently
            # taking down every prediction that touches this cache.
            conn.close()
            self._ensure_schema()
            return None
        finally:
            conn.close()
        if not row or row[1] < time.time():
            return None
        return json.loads(row[0])

    def put(self, key: str, value: Any, ttl_seconds: int) -> None:
        body = json.dumps(value, separators=(",", ":"), default=str)
        params = (key, body, datetime.now(timezone.utc).isoformat(), time.time() + ttl_seconds)
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO http_cache(cache_key, body, stored_at, expires_at) VALUES(?,?,?,?)", params
            )
            conn.commit()
        except sqlite3.Error:
            conn.close()
            self._ensure_schema()
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO http_cache(cache_key, body, stored_at, expires_at) VALUES(?,?,?,?)", params
                )
                conn.commit()
            finally:
                conn.close()
        else:
            conn.close()


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

    def get_csv(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        ttl_seconds: int,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResult:
        """GET a public CSV feed, parsed to a list of row dicts, with the same cache/retry/rate-limit behavior as get_json."""
        key = self._key(url, params) + ":csv"
        cached = self.cache.get(key)
        if cached is not None:
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
                import csv
                import io
                text = response.text.lstrip("﻿")
                rows = list(csv.DictReader(io.StringIO(text)))
                self.cache.put(key, rows, ttl_seconds)
                return HttpResult(rows, datetime.now(timezone.utc), False, dict(response.headers))
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.attempts:
                    time.sleep(min(4.0, (2 ** attempt) * 0.4 + random.random() * 0.2))
        raise ProviderError(f"request failed after {self.attempts} attempts: {last_error}")

