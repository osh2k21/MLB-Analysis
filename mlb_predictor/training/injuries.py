"""Historical injured-list backfill for the il_pitchers_diff / il_batters_diff
training features, sourced from the official MLB Stats API transactions feed
(free, no key -- the same data source used for live injury lookups).

Self-contained (no import from mlb_predictor.features) to avoid a circular
import with retrosheet_builder.py, which this module's caller lives in.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
import re

# Retrosheet team code -> current-franchise MLB Stats API team_id. Both "ATH"
# and "OAK" map to the same franchise id since the Athletics kept the same
# team_id across their Sacramento relocation; any Retrosheet code from the
# 2021-2025 training window not covered here (there shouldn't be any, since
# no franchise folded or expanded in that span) simply gets no IL timeline
# and the sweep below leaves its diff at a neutral 0 rather than failing.
RETROSHEET_TO_TEAM_ID = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHN": 112, "CHA": 145,
    "CIN": 113, "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KCA": 118,
    "LAA": 108, "LAN": 119, "MIA": 146, "MIL": 158, "MIN": 142, "NYN": 121,
    "NYA": 147, "ATH": 133, "OAK": 133, "PHI": 143, "PIT": 134, "SDN": 135,
    "SFN": 137, "SEA": 136, "SLN": 138, "TBA": 139, "TEX": 140, "TOR": 141,
    "WAS": 120,
}

_IL_PLACED = re.compile(r"placed .*\b([A-Z0-9]{1,3})\b [A-Z][\w.'-]*.* on the .*injured list")
_PITCHER_POSITIONS = {"P", "LHP", "RHP", "SP", "RP", "CL"}


def _classify(position: str) -> str:
    return "pitcher" if position in _PITCHER_POSITIONS else "batter"


def parse_il_events(transactions: list[dict]) -> list[tuple[date, int, int]]:
    """Pure parser: raw MLB Stats API transaction dicts -> chronologically sorted
    (date, pitcher_count, batter_count) snapshots of the team's active injured list
    immediately after that date's transactions are applied. No network calls.

    Tracks each player's own on/off-IL status by player id rather than blindly
    summing +1/-1 per matched phrase -- MLB's transaction log contains duplicate
    and retroactively-corrected entries for the same injury stint in practice
    (e.g. a player placed on IL twice for one stint), and a naive delta sum
    double-counts those, causing the running total to drift unboundedly over a
    multi-year window instead of staying in the realistic 0-8-ish range.
    """
    by_date: dict[date, list[dict]] = defaultdict(list)
    for txn in transactions:
        txn_date = txn.get("date")
        if not txn_date:
            continue
        try:
            event_date = datetime.strptime(txn_date, "%Y-%m-%d").date()
        except ValueError:
            continue
        by_date[event_date].append(txn)

    active_pitcher_by_player: dict[int, bool] = {}
    pitcher_count = batter_count = 0
    snapshots: list[tuple[date, int, int]] = []
    for event_date in sorted(by_date):
        changed = False
        for txn in by_date[event_date]:
            player_id = (txn.get("person") or {}).get("id")
            if player_id is None:
                continue
            description = str(txn.get("description", ""))
            placed = _IL_PLACED.search(description)
            if placed:
                if player_id not in active_pitcher_by_player:
                    is_pitcher = _classify(placed.group(1)) == "pitcher"
                    active_pitcher_by_player[player_id] = is_pitcher
                    pitcher_count += int(is_pitcher)
                    batter_count += int(not is_pitcher)
                    changed = True
            elif player_id in active_pitcher_by_player:
                # Any other transaction for a tracked player -- an explicit activation,
                # but also a trade, release, DFA, recall, or option -- ends this team's
                # tracking of that IL stint. Without this, a player who leaves the org
                # while hurt (common: trades, releases, DFAs) never gets an explicit
                # "activated from IL" transaction from *this* team and stays stuck as
                # permanently injured forever after, which is what caused the running
                # count to inflate to unrealistic levels (20+ "simultaneous" IL pitchers).
                is_pitcher = active_pitcher_by_player.pop(player_id)
                pitcher_count -= int(is_pitcher)
                batter_count -= int(not is_pitcher)
                changed = True
        if changed:
            snapshots.append((event_date, pitcher_count, batter_count))
    return snapshots


def active_il_counts(snapshots: list[tuple[date, int, int]], as_of: date) -> tuple[int, int]:
    """Pure: (pitcher_count, batter_count) as of strictly before as_of. Snapshots on or
    after as_of are excluded -- this is what keeps the feature leakage-safe."""
    pitchers = batters = 0
    for snap_date, snap_pitchers, snap_batters in snapshots:
        if snap_date >= as_of:
            break
        pitchers, batters = snap_pitchers, snap_batters
    return pitchers, batters


def _team_il_events(team_id: int, start: str, end: str) -> list[tuple[date, int, int]]:
    """Fetches a team's transactions and parses them into a chronological IL event list."""
    import requests

    response = requests.get(
        "https://statsapi.mlb.com/api/v1/transactions",
        params={"teamId": team_id, "startDate": start, "endDate": end},
        timeout=30,
    )
    response.raise_for_status()
    return parse_il_events(response.json().get("transactions", []))


def fetch_il_timelines(start: str = "2021-01-01", end: str | None = None) -> dict[str, list[tuple[date, int, int]]]:
    """One-time bulk fetch: Retrosheet team code -> chronological IL placement/activation events."""
    end = end or date.today().isoformat()
    timelines: dict[str, list[tuple[date, int, int]]] = {}
    for code, team_id in RETROSHEET_TO_TEAM_ID.items():
        timelines[code] = _team_il_events(team_id, start, end)
    return timelines
