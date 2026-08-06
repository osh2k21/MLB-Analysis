from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from ..api.mlb import MLBStatsClient
from ..api.weather import WeatherClient
from ..contracts import GameRef
from ..features.catalog import FEATURE_NAMES
from ..features.free_live import FreeLiveFeatureBuilder

CHECKPOINT_PATH = Path("data/recent_games_checkpoint.json")
OUTPUT_PATH = Path("data/recent_training_games.csv")
CSV_COLUMNS = ["game_date", "home_team", "away_team", "site", "home_runs", "away_runs", *FEATURE_NAMES, "home_win"]


def load_resolved_games(track_db_path: Path) -> list[dict]:
    """Every game in the track-record log that has a real final score --
    the pool of candidates for the recent-games backfill."""
    conn = sqlite3.connect(track_db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT game_pk, game_date, away_team, home_team, away_score, home_score "
            "FROM ml_predictions WHERE resolved = 1 AND away_score IS NOT NULL AND home_score IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    return [
        {"game_pk": r[0], "game_date": r[1], "away_team": r[2], "home_team": r[3],
         "away_score": r[4], "home_score": r[5]}
        for r in rows
    ]


def _load_checkpoint(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_checkpoint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def build_recent_training_rows(
    games: list[dict], mlb: MLBStatsClient, weather: WeatherClient, bundle_metadata: dict,
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> list[dict]:
    """Backfills feature rows for games already resolved in the track-record
    log, using the SAME live-inference feature builder real predictions use
    (as-of each game's own commence time -- every internal window in
    FreeLiveFeatureBuilder is date-scoped to what's passed in, not
    datetime.now(), so this is leakage-safe for past dates too). This is
    deliberately NOT the Retrosheet pipeline: Retrosheet has no file for the
    current season yet, so the only source for "just played this week" games
    is the live MLB Stats API/Savant, queried as of their own game date.

    Checkpointed by game_pk so re-running only processes newly resolved
    games -- this is meant to be run before every retrain, growing the
    training set a little more each time."""
    checkpoint = _load_checkpoint(checkpoint_path)
    rows = []
    for game in games:
        key = str(game["game_pk"])
        if key in checkpoint:
            continue
        feed = mlb.game_feed(game["game_pk"])
        if not feed.usable:
            continue
        game_data = feed.payload.get("gameData", {})
        teams = game_data.get("teams", {})
        venue = game_data.get("venue", {})
        dt_str = game_data.get("datetime", {}).get("dateTime")
        if not (teams.get("home", {}).get("id") and teams.get("away", {}).get("id") and dt_str):
            checkpoint[key] = "skipped_incomplete"
            continue
        commence = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        ref = GameRef(
            game_pk=game["game_pk"], game_date=game["game_date"], commence_time=commence,
            away_team_id=teams["away"]["id"], home_team_id=teams["home"]["id"],
            away_team=game["away_team"], home_team=game["home_team"],
            venue_id=venue.get("id"), venue_name=venue.get("name"),
        )
        probable = game_data.get("probablePitchers", {})
        raw_game = {"teams": {
            "away": {"probablePitcher": probable.get("away", {})},
            "home": {"probablePitcher": probable.get("home", {})},
        }}
        coordinates = (venue.get("location", {}) or {}).get("defaultCoordinates", {}) or {}
        weather_evidence = weather.forecast(coordinates.get("latitude"), coordinates.get("longitude"), ref.commence_time)
        try:
            snapshot, _coverage = FreeLiveFeatureBuilder(mlb, bundle_metadata).build(ref, raw_game, feed, weather_evidence)
        except Exception as exc:
            checkpoint[key] = f"error:{exc}"
            continue
        row = dict(zip(FEATURE_NAMES, snapshot.ordered(FEATURE_NAMES)))
        row.update({
            "game_date": game["game_date"], "home_team": game["home_team"], "away_team": game["away_team"],
            # Not a real Retrosheet park code -- just a stable per-venue grouping
            # key so park-factor/Elo metadata rebuild in train_bundle still works.
            # Prefixed so it can never collide with an actual Retrosheet site code.
            "site": f"mlb_venue_{venue.get('id', 'unknown')}",
            "home_runs": float(game["home_score"]), "away_runs": float(game["away_score"]),
            "home_win": int(float(game["home_score"]) > float(game["away_score"])),
        })
        rows.append(row)
        checkpoint[key] = "done"
    _save_checkpoint(checkpoint_path, checkpoint)
    return rows


def append_rows_to_csv(rows: list[dict], output_path: Path = OUTPUT_PATH) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with open(output_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    import argparse
    from ..config import load_settings
    from ..api import HttpClient
    from ..pipeline import load_bundle

    parser = argparse.ArgumentParser(description="Backfill training rows from resolved track-record games")
    parser.add_argument("--track-db", type=Path, default=Path("mlb_track_record.db"))
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    settings = load_settings()
    http = HttpClient(settings.cache_path, timeout=settings.request_timeout_seconds, attempts=settings.max_attempts, requests_per_second=25.0)
    mlb = MLBStatsClient(http)
    weather = WeatherClient(http, settings.weather_base_url)
    bundle = load_bundle(settings.model_path)

    games = load_resolved_games(args.track_db)
    rows = build_recent_training_rows(games, mlb, weather, bundle.metadata, args.checkpoint)
    append_rows_to_csv(rows, args.output)
    print(f"Backfilled {len(rows)} new row(s) from {len(games)} resolved game(s) -> {args.output}")


if __name__ == "__main__":
    main()
