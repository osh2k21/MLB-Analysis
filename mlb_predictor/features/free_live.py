from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import math
from typing import Any

from ..api.mlb import MLBStatsClient
from ..contracts import DataStatus, Evidence, FeatureSnapshot, GameRef
from .catalog import (
    FEATURE_NAMES, OFFENSE_METRICS, TEAM_PITCHING_METRICS, DEFENSE_METRICS, WINDOWS,
    STATCAST_METRICS, STATCAST_WINDOWS,
)
from .engine import FeatureError
from ..training.retrosheet_builder import RAW_PITCHER_COLUMNS, pitcher_rates, team_rates
from ..training.injuries import active_il_counts, parse_il_events
from ..training.statcast import aggregate_day, statcast_rates


TEAM_CODES = {
    "Arizona Diamondbacks":"ARI", "Atlanta Braves":"ATL", "Baltimore Orioles":"BAL", "Boston Red Sox":"BOS",
    "Chicago Cubs":"CHN", "Chicago White Sox":"CHA", "Cincinnati Reds":"CIN", "Cleveland Guardians":"CLE",
    "Colorado Rockies":"COL", "Detroit Tigers":"DET", "Houston Astros":"HOU", "Kansas City Royals":"KCA",
    "Los Angeles Angels":"LAA", "Los Angeles Dodgers":"LAN", "Miami Marlins":"MIA", "Milwaukee Brewers":"MIL",
    "Minnesota Twins":"MIN", "New York Mets":"NYN", "New York Yankees":"NYA", "Athletics":"ATH",
    "Oakland Athletics":"OAK", "Philadelphia Phillies":"PHI", "Pittsburgh Pirates":"PIT", "San Diego Padres":"SDN",
    "San Francisco Giants":"SFN", "Seattle Mariners":"SEA", "St. Louis Cardinals":"SLN", "Tampa Bay Rays":"TBA",
    "Texas Rangers":"TEX", "Toronto Blue Jays":"TOR", "Washington Nationals":"WAS",
}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _innings_outs(value: Any) -> float:
    text = str(value or "0.0")
    whole, _, partial = text.partition(".")
    return int(whole or 0) * 3 + min(2, int(partial or 0))


class FreeLiveFeatureBuilder:
    def __init__(self, mlb: MLBStatsClient, metadata: dict[str, Any] | None = None) -> None:
        self.mlb, self.metadata = mlb, metadata or {}

    def _stats(self, team_id: int, group: str, season: int, start: str | None = None, end: str | None = None) -> dict:
        stats_type = "byDateRange" if start and end else "season"
        params = {"stats": stats_type, "group": group, "season": season}
        if start and end:
            params.update({"startDate": start, "endDate": end})
        result = self.mlb.http.get_json(
            f"{self.mlb.base_url}/teams/{team_id}/stats", params, ttl_seconds=1800,
            validator=lambda x: isinstance(x, dict) and isinstance(x.get("stats"), list),
        )
        for block in result.data.get("stats", []):
            splits = block.get("splits") or []
            if splits:
                return splits[0].get("stat", {})
        return {}

    def _team_raw(self, team_id: int, season: int, start: str | None, end: str | None) -> dict[str, float]:
        hitting = self._stats(team_id, "hitting", season, start, end)
        pitching = self._stats(team_id, "pitching", season, start, end)
        fielding = self._stats(team_id, "fielding", season, start, end)
        games = max(_num(hitting.get("gamesPlayed")), _num(pitching.get("gamesPlayed")))
        return {
            "games": games, "wins": 0, "b_pa": _num(hitting.get("plateAppearances")), "b_ab": _num(hitting.get("atBats")),
            "b_r": _num(hitting.get("runs")), "b_h": _num(hitting.get("hits")), "b_d": _num(hitting.get("doubles")),
            "b_t": _num(hitting.get("triples")), "b_hr": _num(hitting.get("homeRuns")), "b_hbp": _num(hitting.get("hitByPitch")),
            "b_w": _num(hitting.get("baseOnBalls")), "b_k": _num(hitting.get("strikeOuts")), "b_sb": _num(hitting.get("stolenBases")),
            "b_cs": _num(hitting.get("caughtStealing")), "b_gdp": _num(hitting.get("groundIntoDoublePlay")), "lob": _num(hitting.get("leftOnBase")),
            "p_ipouts": _innings_outs(pitching.get("inningsPitched")), "p_bfp": _num(pitching.get("battersFaced")),
            "p_h": _num(pitching.get("hits")), "p_hr": _num(pitching.get("homeRuns")), "p_r": _num(pitching.get("runs")),
            "p_er": _num(pitching.get("earnedRuns")), "p_w": _num(pitching.get("baseOnBalls")), "p_k": _num(pitching.get("strikeOuts")),
            "d_po": _num(fielding.get("putOuts")), "d_a": _num(fielding.get("assists")), "d_e": _num(fielding.get("errors")),
            "d_dp": _num(fielding.get("doublePlays")),
        }

    def _team_windows(self, team_id: int, game_date: datetime) -> dict[str, float]:
        season, out = game_date.year, {}
        for window in WINDOWS:
            if window == "season":
                raw = self._team_raw(team_id, season, None, None)
            else:
                days = int(window[:-1])
                raw = self._team_raw(team_id, season, (game_date - timedelta(days=days)).date().isoformat(), (game_date - timedelta(days=1)).date().isoformat())
            out.update({f"{name}_{window}": value for name, value in team_rates(raw).items()})
        return out

    def _starter(self, pitcher_id: int | None, season: int) -> dict[str, float]:
        if not pitcher_id:
            return {}
        result = self.mlb.http.get_json(
            f"{self.mlb.base_url}/people/{pitcher_id}/stats", {"stats": "gameLog", "group": "pitching", "season": season},
            ttl_seconds=1800, validator=lambda x: isinstance(x, dict) and isinstance(x.get("stats"), list),
        )
        splits = (result.data.get("stats") or [{}])[0].get("splits") or []
        rows = []
        for split in splits:
            stat = split.get("stat", {})
            if not stat.get("gamesStarted"):
                continue
            rows.append({
                "games": 1, "p_ipouts": _innings_outs(stat.get("inningsPitched")), "p_bfp": _num(stat.get("battersFaced")),
                "p_h": _num(stat.get("hits")), "p_hr": _num(stat.get("homeRuns")), "p_er": _num(stat.get("earnedRuns")),
                "p_w": _num(stat.get("baseOnBalls")), "p_k": _num(stat.get("strikeOuts")),
            })
        def aggregate(selected):
            raw = {key: sum(row[key] for row in selected) for key in RAW_PITCHER_COLUMNS}
            return pitcher_rates(raw)
        output = {}
        for label, selected in (("last5", rows[-5:]), ("season", rows)):
            if selected:
                output.update({f"starter_{key}_{label}": value for key, value in aggregate(selected).items()})
        return output

    def _il_counts(self, team_id: int, game_date: datetime) -> tuple[float, float]:
        """Currently-active injured-list pitcher/batter counts, using the same parser and
        counting logic as the training-side backfill in training/injuries.py (90-day lookback
        is enough live since IL stints resolve within that window)."""
        start = (game_date - timedelta(days=90)).date().isoformat()
        end = game_date.date().isoformat()
        result = self.mlb.http.get_json(
            f"{self.mlb.base_url}/transactions", {"teamId": team_id, "startDate": start, "endDate": end},
            ttl_seconds=1800, validator=lambda x: isinstance(x, dict) and isinstance(x.get("transactions"), list),
        )
        events = parse_il_events(result.data.get("transactions", []))
        # active_il_counts excludes events on-or-after `as_of`; pass game_date's date plus one
        # day so a same-day transaction (known before first pitch) still counts as active.
        pitchers, batters = active_il_counts(events, game_date.date() + timedelta(days=1))
        return float(pitchers), float(batters)

    def _fetch_statcast_day(self, day) -> tuple:
        today = datetime.now(timezone.utc).date()
        ttl = 7 * 24 * 3600 if day < today else 3600
        result = self.mlb.http.get_csv(
            "https://baseballsavant.mlb.com/statcast_search/csv",
            {"all": "true", "hfGT": "R|", "game_date_gt": day.isoformat(), "game_date_lt": day.isoformat(), "type": "details"},
            ttl_seconds=ttl,
        )
        return day, aggregate_day(result.data)

    def _statcast_history(self, as_of: datetime) -> dict[str, list[tuple]]:
        """Last 30 days of team-level barrel/hard-hit aggregates, day-fetched (not one big
        range query) so each day gets its own cache entry through the shared HttpClient:
        fully-completed past days are immutable and cached for a week, while today's
        (possibly still in progress) date gets a short TTL. A full slate of games sharing
        this same HttpClient means only the first game's build() actually re-fetches a
        given day; the rest hit cache.

        Fetched with a small thread pool -- each day is a multi-MB pitch-level CSV, and
        30 of them sequentially takes minutes; concurrent fetches (still throttled by the
        shared HttpClient's own rate limiter) bring a cold cache down to tens of seconds."""
        days = [(as_of - timedelta(days=offset)).date() for offset in range(1, 31)]
        daily: dict[str, list[tuple]] = {}
        with ThreadPoolExecutor(max_workers=6) as executor:
            for day, totals in executor.map(self._fetch_statcast_day, days):
                for code, values in totals.items():
                    daily.setdefault(code, []).append((day, *values))
        return daily

    def _rest_travel(self, team_id: int, game_date: datetime, venue_name: str) -> tuple[float, float]:
        start = (game_date - timedelta(days=7)).date().isoformat(); end = (game_date - timedelta(days=1)).date().isoformat()
        result = self.mlb.http.get_json(
            f"{self.mlb.base_url}/schedule", {"sportId": 1, "teamId": team_id, "startDate": start, "endDate": end, "hydrate": "venue"},
            ttl_seconds=1800, validator=lambda x: isinstance(x, dict) and "dates" in x,
        )
        previous = [g for day in result.data.get("dates", []) for g in day.get("games", []) if g.get("status", {}).get("abstractGameState") == "Final"]
        if not previous:
            return math.nan, 0.0
        last = sorted(previous, key=lambda g: g.get("gameDate", ""))[-1]
        last_date = datetime.fromisoformat(last["gameDate"].replace("Z", "+00:00"))
        rest = max(0, (game_date.date() - last_date.date()).days)
        traveled = float(rest == 1 and last.get("venue", {}).get("name") != venue_name)
        return float(rest), traveled

    def _bullpen(self, team_id: int, game_date: datetime, days: int) -> tuple[dict[str, float], float]:
        start = (game_date - timedelta(days=days)).date().isoformat(); end = (game_date - timedelta(days=1)).date().isoformat()
        schedule = self.mlb.http.get_json(
            f"{self.mlb.base_url}/schedule", {"sportId": 1, "teamId": team_id, "startDate": start, "endDate": end},
            ttl_seconds=1800, validator=lambda x: isinstance(x, dict) and "dates" in x,
        )
        totals = {key: 0.0 for key in RAW_PITCHER_COLUMNS}
        for raw_game in [g for block in schedule.data.get("dates", []) for g in block.get("games", []) if g.get("status", {}).get("abstractGameState") == "Final"]:
            box = self.mlb.http.get_json(
                f"{self.mlb.base_url}/game/{raw_game['gamePk']}/boxscore", ttl_seconds=86400,
                validator=lambda x: isinstance(x, dict) and "teams" in x,
            ).data
            side = next((value for value in box.get("teams", {}).values() if value.get("team", {}).get("id") == team_id), None)
            if not side:
                continue
            pitcher_ids = side.get("pitchers") or []
            for pitcher_id in pitcher_ids[1:]:  # first pitcher used is the starter
                stat = side.get("players", {}).get(f"ID{pitcher_id}", {}).get("stats", {}).get("pitching", {})
                totals["games"] += 1
                totals["p_ipouts"] += _innings_outs(stat.get("inningsPitched"))
                totals["p_bfp"] += _num(stat.get("battersFaced")); totals["p_h"] += _num(stat.get("hits"))
                totals["p_hr"] += _num(stat.get("homeRuns")); totals["p_er"] += _num(stat.get("earnedRuns"))
                totals["p_w"] += _num(stat.get("baseOnBalls")); totals["p_k"] += _num(stat.get("strikeOuts"))
        return pitcher_rates(totals), totals["p_ipouts"] / 3.0

    def build(self, game: GameRef, raw_game: dict, feed: Evidence, weather: Evidence) -> tuple[FeatureSnapshot, float]:
        game_dt = game.commence_time.astimezone(timezone.utc)
        home_team = self._team_windows(game.home_team_id, game_dt)
        away_team = self._team_windows(game.away_team_id, game_dt)
        probable = feed.payload.get("gameData", {}).get("probablePitchers", {}) if feed.usable else {}
        if not probable.get("home", {}).get("id"):
            probable["home"] = raw_game.get("teams", {}).get("home", {}).get("probablePitcher", {})
        if not probable.get("away", {}).get("id"):
            probable["away"] = raw_game.get("teams", {}).get("away", {}).get("probablePitcher", {})
        home_starter = self._starter(probable.get("home", {}).get("id"), game_dt.year)
        away_starter = self._starter(probable.get("away", {}).get("id"), game_dt.year)
        values, provenance = {}, {}
        source = "MLB Stats API"
        for metric in (*OFFENSE_METRICS, *TEAM_PITCHING_METRICS, *DEFENSE_METRICS):
            for window in WINDOWS:
                name, raw_name = f"{metric}_{window}_diff", f"{metric}_{window}"
                values[name] = home_team.get(raw_name, math.nan) - away_team.get(raw_name, math.nan)
                provenance[name] = source
        for metric in ("era", "whip", "k_pct", "bb_pct", "hr_pct", "ip_per_start"):
            for window in ("last5", "season"):
                name, raw_name = f"starter_{metric}_{window}_diff", f"starter_{metric}_{window}"
                values[name] = home_starter.get(raw_name, math.nan) - away_starter.get(raw_name, math.nan)
                provenance[name] = source
        home_pen3, home_ip3 = self._bullpen(game.home_team_id, game_dt, 3)
        away_pen3, away_ip3 = self._bullpen(game.away_team_id, game_dt, 3)
        home_pen7, home_ip7 = self._bullpen(game.home_team_id, game_dt, 7)
        away_pen7, away_ip7 = self._bullpen(game.away_team_id, game_dt, 7)
        bullpen_values = {
            "bullpen_era_3d_diff": home_pen3["era"] - away_pen3["era"],
            "bullpen_era_7d_diff": home_pen7["era"] - away_pen7["era"],
            "bullpen_whip_3d_diff": home_pen3["whip"] - away_pen3["whip"],
            "bullpen_whip_7d_diff": home_pen7["whip"] - away_pen7["whip"],
            "bullpen_k_pct_7d_diff": home_pen7["k_pct"] - away_pen7["k_pct"],
            "bullpen_bb_pct_7d_diff": home_pen7["bb_pct"] - away_pen7["bb_pct"],
            "bullpen_ip_3d_diff": home_ip3 - away_ip3, "bullpen_ip_7d_diff": home_ip7 - away_ip7,
        }
        values.update(bullpen_values)
        provenance.update({name: "MLB Stats API recent relief boxscores" for name in bullpen_values})
        payload = weather.payload if weather.usable else {}
        roof_text = str(feed.payload.get("gameData", {}).get("venue", {}).get("fieldInfo", {}).get("roofType", "") if feed.usable else "").lower()
        values.update({
            "temperature_f": _num(payload.get("temperature_2m"), math.nan),
            "wind_speed_mph": _num(payload.get("wind_speed_10m"), math.nan),
            "roof_closed": float("closed" in roof_text or "dome" in roof_text),
            "day_game": float(str(feed.payload.get("gameData", {}).get("datetime", {}).get("dayNight", "")).lower() == "day") if feed.usable else math.nan,
            "park_run_factor": math.nan,
        })
        for name in ("temperature_f", "wind_speed_mph"):
            provenance[name] = "Open-Meteo"
        provenance["roof_closed"] = provenance["day_game"] = "MLB Stats API"
        provenance["park_run_factor"] = "Retrosheet historical park sample unavailable for MLB venue mapping"
        home_rest, home_travel = self._rest_travel(game.home_team_id, game_dt, game.venue_name)
        away_rest, away_travel = self._rest_travel(game.away_team_id, game_dt, game.venue_name)
        values["rest_days_diff"] = home_rest - away_rest
        values["traveled_yesterday_diff"] = home_travel - away_travel
        provenance["rest_days_diff"] = provenance["traveled_yesterday_diff"] = "MLB Stats API schedule"
        home_il_pitchers, home_il_batters = self._il_counts(game.home_team_id, game_dt)
        away_il_pitchers, away_il_batters = self._il_counts(game.away_team_id, game_dt)
        values["il_pitchers_diff"] = home_il_pitchers - away_il_pitchers
        values["il_batters_diff"] = home_il_batters - away_il_batters
        provenance["il_pitchers_diff"] = provenance["il_batters_diff"] = "MLB Stats API transactions"
        statcast_history = self._statcast_history(game_dt)
        home_statcast_days = statcast_history.get(TEAM_CODES.get(game.home_team), [])
        away_statcast_days = statcast_history.get(TEAM_CODES.get(game.away_team), [])
        for window in STATCAST_WINDOWS:
            window_days = int(window[:-1])
            home_rates = statcast_rates(home_statcast_days, game_dt.date(), window_days)
            away_rates = statcast_rates(away_statcast_days, game_dt.date(), window_days)
            for metric in STATCAST_METRICS:
                name = f"{metric}_{window}_diff"
                values[name] = home_rates[metric] - away_rates[metric]
                provenance[name] = "Baseball Savant"
        ratings = self.metadata.get("elo_ratings", {})
        values["team_elo_rating_diff"] = _num(ratings.get(TEAM_CODES.get(game.home_team)), 1500) - _num(ratings.get(TEAM_CODES.get(game.away_team)), 1500)
        provenance["team_elo_rating_diff"] = "Retrosheet historical Elo"
        valid = sum(math.isfinite(float(values.get(name, math.nan))) for name in FEATURE_NAMES)
        coverage = valid / len(FEATURE_NAMES)
        # Snapshot intentionally retains NaN. Fitted pipelines learned missingness handling.
        snapshot = FeatureSnapshot(game, values, provenance)
        return snapshot, coverage
