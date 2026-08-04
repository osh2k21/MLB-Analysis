from __future__ import annotations

import csv
import json
from pathlib import Path

from ..database import PredictionRepository
from ..features.catalog import FEATURE_NAMES


def export_settled(repository: PredictionRepository, output: Path) -> int:
    rows = repository.settled()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["game_date", *FEATURE_NAMES, "home_win"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            features = json.loads(row["feature_json"])
            writer.writerow({
                "game_date": row["game_date"], **{name: features[name] for name in FEATURE_NAMES},
                "home_win": row["home_win"],
            })
    return len(rows)
