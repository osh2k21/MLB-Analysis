from __future__ import annotations

import argparse
from pathlib import Path

from .retrosheet_builder import build_training_frame, extract_season


def download_season(season: int, root: Path) -> Path:
    import requests
    zip_path = root / f"retrosheet_{season}.zip"
    season_dir = root / f"retrosheet_{season}"
    if not zip_path.exists():
        response = requests.get(f"https://www.retrosheet.org/downloads/{season}/{season}csvs.zip", timeout=120)
        response.raise_for_status()
        zip_path.write_bytes(response.content)
    if not season_dir.exists() or not list(season_dir.glob("*teamstats.csv")):
        extract_season(zip_path, season_dir)
    return season_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe free MLB training features")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/free_training_games.csv"))
    args = parser.parse_args()
    directories = [download_season(season, args.data_dir) for season in args.seasons]
    frame = build_training_frame(directories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame):,} rows and {len(frame.columns) - 2} features to {args.output}")


if __name__ == "__main__":
    main()
