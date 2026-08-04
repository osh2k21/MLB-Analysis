from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    cache_path: Path = field(default_factory=lambda: Path(os.getenv("MLB_CACHE_PATH", "data/http_cache.sqlite3")))
    database_path: Path = field(default_factory=lambda: Path(os.getenv("MLB_DATABASE_PATH", "data/mlb_predictor.sqlite3")))
    model_path: Path = field(default_factory=lambda: Path(os.getenv("MLB_MODEL_PATH", "artifacts/ensemble.joblib")))
    odds_api_key: str | None = field(default_factory=lambda: os.getenv("ODDS_API_KEY") or None)
    weather_api_key: str | None = field(default_factory=lambda: os.getenv("WEATHER_API_KEY") or None)
    weather_base_url: str = field(default_factory=lambda: os.getenv("WEATHER_BASE_URL", "https://api.openweathermap.org/data/2.5"))
    statcast_feed_url: str | None = field(default_factory=lambda: os.getenv("STATCAST_FEED_URL") or None)
    fangraphs_feed_url: str | None = field(default_factory=lambda: os.getenv("FANGRAPHS_FEED_URL") or None)
    retrosheet_feed_url: str | None = field(default_factory=lambda: os.getenv("RETROSHEET_FEED_URL") or None)
    injury_feed_url: str | None = field(default_factory=lambda: os.getenv("INJURY_FEED_URL") or None)
    umpire_feed_url: str | None = field(default_factory=lambda: os.getenv("UMPIRE_FEED_URL") or None)
    park_factor_feed_url: str | None = field(default_factory=lambda: os.getenv("PARK_FACTOR_FEED_URL") or None)
    request_timeout_seconds: float = 12.0
    max_attempts: int = 3
    requests_per_second: float = 3.0


def load_settings() -> Settings:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return Settings()

