from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from ..api import HttpClient, MLBStatsClient
from ..config import load_settings
from ..database import PredictionRepository


def settle_date(day: str) -> int:
    settings = load_settings()
    http = HttpClient(settings.cache_path, settings.request_timeout_seconds, settings.max_attempts, settings.requests_per_second)
    mlb = MLBStatsClient(http)
    repository = PredictionRepository(settings.database_path)
    schedule = mlb.schedule(day)
    if not schedule.usable:
        raise RuntimeError(schedule.detail)
    settled = 0
    for game in schedule.payload:
        status = game.get("status", {}).get("abstractGameState")
        if status != "Final":
            continue
        teams = game.get("teams", {})
        home_score, away_score = teams.get("home", {}).get("score"), teams.get("away", {}).get("score")
        if home_score is None or away_score is None or home_score == away_score:
            continue
        repository.settle(int(game["gamePk"]), home_score > away_score)
        settled += 1
    return settled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=(date.today() - timedelta(days=1)).isoformat())
    args = parser.parse_args()
    print(f"Settled {settle_date(args.date)} final game(s) for {args.date}")


if __name__ == "__main__":
    main()
