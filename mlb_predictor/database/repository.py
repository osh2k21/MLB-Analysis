from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any


class PredictionRepository:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def connection(self):
        conn = self.connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    game_pk INTEGER PRIMARY KEY, game_date TEXT NOT NULL, created_at TEXT NOT NULL,
                    model_version TEXT NOT NULL, raw_home_probability REAL NOT NULL,
                    home_probability REAL NOT NULL, feature_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL, votes_json TEXT NOT NULL,
                    explanation_json TEXT NOT NULL, home_win INTEGER, decimal_odds REAL,
                    settled_at TEXT
                );
                CREATE TABLE IF NOT EXISTS validation_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, game_pk INTEGER NOT NULL,
                    checked_at TEXT NOT NULL, ready INTEGER NOT NULL, checks_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
                    trained_through TEXT, rows INTEGER NOT NULL, metrics_json TEXT NOT NULL,
                    artifact_path TEXT NOT NULL
                );
                """
            )

    def record_validation(self, game_pk: int, ready: bool, checks: list[dict[str, Any]]) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO validation_audit(game_pk,checked_at,ready,checks_json) VALUES(?,?,?,?)",
                (game_pk, datetime.now(timezone.utc).isoformat(), int(ready), json.dumps(checks)),
            )

    def record_prediction(self, payload: dict[str, Any]) -> None:
        columns = (
            "game_pk", "game_date", "created_at", "model_version", "raw_home_probability",
            "home_probability", "feature_json", "provenance_json", "votes_json", "explanation_json", "decimal_odds"
        )
        with self.connection() as conn:
            conn.execute(
                f"INSERT OR IGNORE INTO predictions({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
                tuple(payload.get(key) for key in columns),
            )

    def settle(self, game_pk: int, home_win: bool) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE predictions SET home_win=?, settled_at=? WHERE game_pk=? AND home_win IS NULL",
                (int(home_win), datetime.now(timezone.utc).isoformat(), game_pk),
            )

    def settled(self) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return conn.execute("SELECT * FROM predictions WHERE home_win IS NOT NULL ORDER BY game_date").fetchall()
