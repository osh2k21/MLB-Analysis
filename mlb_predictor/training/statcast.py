"""Historical Statcast quality-of-contact backfill for the hard_hit_rate_diff /
hard_hit_rate_allowed_diff training features (see STATCAST_METRICS /
STATCAST_WINDOWS in features/catalog.py for the canonical feature contract),
sourced from Baseball Savant's free, no-key pitch-level search endpoint.

This module stays self-contained (no import from mlb_predictor.features) to
avoid a circular import with retrosheet_builder.py, which imports it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import json
import math
from pathlib import Path

# Savant team abbreviation -> Retrosheet code. Savant retroactively uses the
# current franchise abbreviation across its whole history (confirmed: "ATH"
# appears even for 2021 Athletics games, not "OAK"), so this is a single
# stable mapping with no season-dependent variation.
SAVANT_TO_RETROSHEET = {
    "ATH": "ATH", "ATL": "ATL", "AZ": "ARI", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHN", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "CWS": "CHA",
    "DET": "DET", "HOU": "HOU", "KC": "KCA", "LAA": "LAA", "LAD": "LAN",
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYM": "NYN", "NYY": "NYA",
    "PHI": "PHI", "PIT": "PIT", "SD": "SDN", "SEA": "SEA", "SF": "SFN",
    "STL": "SLN", "TB": "TBA", "TEX": "TEX", "TOR": "TOR", "WSH": "WAS",
}
# Retrosheet's own historical data may still use "OAK" for pre-2025 Athletics
# games even though Savant doesn't -- alias both so a lookup by either code finds it.
_ALIASES = {"OAK": "ATH"}

DailyRow = tuple[int, int, int, int, int, int]  # barrels, hard_hits, batted_balls, barrels_allowed, hard_hits_allowed, batted_balls_allowed


def _is_barrel(row: dict) -> bool:
    return row.get("launch_speed_angle") == "6"


def _is_hard_hit(row: dict) -> bool:
    try:
        return float(row["launch_speed"]) >= 95.0
    except (TypeError, ValueError, KeyError):
        return False


def aggregate_day(rows: list[dict]) -> dict[str, DailyRow]:
    """Pure: pitch-level Savant rows for one day -> per-Retrosheet-code
    (barrels, hard_hits, batted_balls, barrels_allowed, hard_hits_allowed,
    batted_balls_allowed) totals for that day. No network calls."""
    totals: dict[str, list[int]] = {}

    def bump(code: str, idx: int) -> None:
        if code not in totals:
            totals[code] = [0, 0, 0, 0, 0, 0]
        totals[code][idx] += 1

    for row in rows:
        if not row.get("launch_speed"):
            continue  # only batted-ball events carry exit velocity
        home = SAVANT_TO_RETROSHEET.get(row.get("home_team", ""))
        away = SAVANT_TO_RETROSHEET.get(row.get("away_team", ""))
        if not home or not away:
            continue
        top = row.get("inning_topbot") == "Top"
        batting_team = away if top else home
        pitching_team = home if top else away
        barrel = _is_barrel(row)
        hard_hit = _is_hard_hit(row)
        bump(batting_team, 2)
        bump(pitching_team, 5)
        if barrel:
            bump(batting_team, 0)
            bump(pitching_team, 3)
        if hard_hit:
            bump(batting_team, 1)
            bump(pitching_team, 4)

    return {code: tuple(values) for code, values in totals.items()}


def _season_ranges() -> list[tuple[date, date]]:
    return [
        (date(2021, 4, 1), date(2021, 10, 3)),
        (date(2022, 4, 7), date(2022, 10, 5)),
        (date(2023, 3, 30), date(2023, 10, 1)),
        (date(2024, 3, 28), date(2024, 9, 29)),
        (date(2025, 3, 27), date(2025, 9, 28)),
    ]


def _all_dates() -> list[date]:
    dates = []
    for start, end in _season_ranges():
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)
    return dates


def _fetch_day(day: date) -> dict[str, DailyRow]:
    import requests

    url = (
        "https://baseballsavant.mlb.com/statcast_search/csv"
        f"?all=true&hfGT=R%7C&game_date_gt={day.isoformat()}&game_date_lt={day.isoformat()}&type=details"
    )
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    response.raise_for_status()
    import csv
    import io

    text = response.text.lstrip("﻿")
    rows = list(csv.DictReader(io.StringIO(text)))
    return aggregate_day(rows)


def fetch_statcast_daily_aggregates(
    checkpoint_path: Path = Path("data/statcast_daily_cache.json"),
    max_workers: int = 6,
) -> dict[str, list[tuple[date, int, int, int, int, int, int]]]:
    """Bulk fetch across all of 2021-2025: Retrosheet code -> chronologically
    sorted per-day totals. Checkpoints progress to disk so an interrupted run
    can resume without re-downloading completed days -- this is a ~20-25
    minute, several-GB-of-transfer job even at 6x concurrency."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    else:
        raw = {}

    all_dates = _all_dates()
    pending = [d for d in all_dates if d.isoformat() not in raw]
    print(f"Statcast backfill: {len(all_dates)} days total, {len(pending)} remaining, {len(raw)} already cached.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_day, d): d for d in pending}
        completed = 0
        for future in as_completed(futures):
            day = futures[future]
            try:
                raw[day.isoformat()] = future.result()
            except Exception as exc:
                print(f"  {day.isoformat()} failed: {exc} -- will retry on next run")
            completed += 1
            if completed % 25 == 0 or completed == len(pending):
                checkpoint_path.write_text(json.dumps(raw), encoding="utf-8")
                print(f"  {completed}/{len(pending)} fetched this run, checkpoint saved.")

    checkpoint_path.write_text(json.dumps(raw), encoding="utf-8")

    timelines: dict[str, list[tuple[date, int, int, int, int, int, int]]] = {}
    for day_str, day_totals in raw.items():
        day = date.fromisoformat(day_str)
        for code, values in day_totals.items():
            timelines.setdefault(code, []).append((day, *values))
    for code in list(timelines):
        timelines[code].sort(key=lambda item: item[0])
    for alias, canonical in _ALIASES.items():
        if canonical in timelines:
            timelines[alias] = timelines[canonical]
    return timelines


def statcast_rates(
    daily_rows: list[tuple[date, int, int, int, int, int, int]], as_of: date, window_days: int
) -> dict[str, float]:
    """Pure: trailing-window barrel/hard-hit rates as of strictly before as_of.
    NaN when there are zero batted balls in the window (never a fabricated 0)."""
    cutoff = as_of - timedelta(days=window_days)
    barrels = hard_hits = batted_balls = 0
    barrels_allowed = hard_hits_allowed = batted_balls_allowed = 0
    for day, b, hh, bb, ba, hha, bba in daily_rows:
        if cutoff <= day < as_of:
            barrels += b
            hard_hits += hh
            batted_balls += bb
            barrels_allowed += ba
            hard_hits_allowed += hha
            batted_balls_allowed += bba
    return {
        "barrel_rate": barrels / batted_balls if batted_balls else math.nan,
        "hard_hit_rate": hard_hits / batted_balls if batted_balls else math.nan,
        "barrel_rate_allowed": barrels_allowed / batted_balls_allowed if batted_balls_allowed else math.nan,
        "hard_hit_rate_allowed": hard_hits_allowed / batted_balls_allowed if batted_balls_allowed else math.nan,
    }
