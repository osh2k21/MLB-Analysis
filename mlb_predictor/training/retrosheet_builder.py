from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import math
from typing import Iterable
import zipfile

from ..features.catalog import (
    FEATURE_NAMES, OFFENSE_METRICS, TEAM_PITCHING_METRICS, DEFENSE_METRICS, WINDOWS,
    STATCAST_METRICS, STATCAST_WINDOWS,
)
from .injuries import active_il_counts, fetch_il_timelines
from .statcast import fetch_statcast_daily_aggregates, statcast_rates


RAW_TEAM_COLUMNS = (
    "games", "wins", "b_pa", "b_ab", "b_r", "b_h", "b_d", "b_t", "b_hr",
    "b_hbp", "b_w", "b_k", "b_sb", "b_cs", "b_gdp", "lob", "p_ipouts",
    "p_bfp", "p_h", "p_hr", "p_r", "p_er", "p_w", "p_k", "d_po", "d_a",
    "d_e", "d_dp",
)
RAW_PITCHER_COLUMNS = ("games", "p_ipouts", "p_bfp", "p_h", "p_hr", "p_er", "p_w", "p_k")


def _safe(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else math.nan


def _sum_rows(rows: Iterable[dict], columns: Iterable[str]) -> dict[str, float]:
    return {column: sum(float(row.get(column, 0) or 0) for row in rows) for column in columns}


def team_rates(raw: dict[str, float]) -> dict[str, float]:
    games, pa, ab = raw["games"], raw["b_pa"], raw["b_ab"]
    singles = raw["b_h"] - raw["b_d"] - raw["b_t"] - raw["b_hr"]
    total_bases = singles + 2 * raw["b_d"] + 3 * raw["b_t"] + 4 * raw["b_hr"]
    avg = _safe(raw["b_h"], ab)
    obp = _safe(raw["b_h"] + raw["b_w"] + raw["b_hbp"], ab + raw["b_w"] + raw["b_hbp"])
    slg = _safe(total_bases, ab)
    babip = _safe(raw["b_h"] - raw["b_hr"], ab - raw["b_k"] - raw["b_hr"])
    innings = raw["p_ipouts"] / 3.0
    fielding_chances = raw["d_po"] + raw["d_a"] + raw["d_e"]
    return {
        "runs_pg": _safe(raw["b_r"], games), "avg": avg, "obp": obp, "slg": slg,
        "ops": obp + slg, "iso": slg - avg, "bb_pct": _safe(raw["b_w"], pa),
        "k_pct": _safe(raw["b_k"], pa), "hr_pa": _safe(raw["b_hr"], pa),
        "babip": babip, "sb_success": _safe(raw["b_sb"], raw["b_sb"] + raw["b_cs"]),
        "lob_pg": _safe(raw["lob"], games), "gdp_pa": _safe(raw["b_gdp"], pa),
        "runs_allowed_pg": _safe(raw["p_r"], games), "era": _safe(raw["p_er"] * 9, innings),
        "whip": _safe(raw["p_w"] + raw["p_h"], innings),
        "pitch_k_pct": _safe(raw["p_k"], raw["p_bfp"]), "pitch_bb_pct": _safe(raw["p_w"], raw["p_bfp"]),
        "pitch_k_minus_bb_pct": _safe(raw["p_k"] - raw["p_w"], raw["p_bfp"]),
        "pitch_hr_pct": _safe(raw["p_hr"], raw["p_bfp"]),
        "fielding_pct": _safe(raw["d_po"] + raw["d_a"], fielding_chances),
        "errors_pg": _safe(raw["d_e"], games), "double_plays_pg": _safe(raw["d_dp"], games),
    }


def pitcher_rates(raw: dict[str, float]) -> dict[str, float]:
    innings = raw["p_ipouts"] / 3.0
    return {
        "era": _safe(raw["p_er"] * 9, innings), "whip": _safe(raw["p_w"] + raw["p_h"], innings),
        "k_pct": _safe(raw["p_k"], raw["p_bfp"]), "bb_pct": _safe(raw["p_w"], raw["p_bfp"]),
        "hr_pct": _safe(raw["p_hr"], raw["p_bfp"]), "ip_per_start": _safe(innings, raw["games"]),
    }


def _read_csv(path: Path):
    import pandas as pd
    return pd.read_csv(path, low_memory=False)


def extract_season(zip_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        wanted = [name for name in archive.namelist() if name.endswith(("teamstats.csv", "pitching.csv", "gameinfo.csv"))]
        archive.extractall(output_dir, wanted)
    return output_dir


def build_training_frame(season_dirs: Iterable[Path]):
    """Build leakage-safe pregame rows from official Retrosheet daily logs."""
    import pandas as pd

    team_frames, pitching_frames, game_frames = [], [], []
    for directory in season_dirs:
        team_frames.append(_read_csv(next(directory.glob("*teamstats.csv"))))
        pitching_frames.append(_read_csv(next(directory.glob("*pitching.csv"))))
        game_frames.append(_read_csv(next(directory.glob("*gameinfo.csv"))))
    teams = pd.concat(team_frames, ignore_index=True)
    pitching = pd.concat(pitching_frames, ignore_index=True)
    games = pd.concat(game_frames, ignore_index=True)
    teams = teams[(teams["stattype"] == "value") & (teams["gametype"] == "regular")].copy()
    pitching = pitching[(pitching["stattype"] == "value") & (pitching["gametype"] == "regular")].copy()
    games = games[games["gametype"] == "regular"].copy()
    for frame in (teams, pitching, games):
        frame["date_dt"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d")

    numeric_team = [c for c in RAW_TEAM_COLUMNS if c not in ("games", "wins")]
    for column in numeric_team:
        teams[column] = pd.to_numeric(teams[column], errors="coerce").fillna(0)
    teams["games"], teams["wins"] = 1.0, pd.to_numeric(teams["win"], errors="coerce").fillna(0)

    starters = pitching[pd.to_numeric(pitching["p_seq"], errors="coerce") == 1].copy()
    relievers = pitching[pd.to_numeric(pitching["p_seq"], errors="coerce") > 1].copy()
    for frame in (starters, relievers):
        for column in RAW_PITCHER_COLUMNS[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        frame["games"] = 1.0

    team_history: dict[str, list[dict]] = defaultdict(list)
    starter_history: dict[str, list[dict]] = defaultdict(list)
    bullpen_history: dict[str, list[dict]] = defaultdict(list)
    pregame_team: dict[tuple[str, str], dict] = {}
    pregame_starter: dict[tuple[str, str], dict] = {}
    pregame_bullpen: dict[tuple[str, str], dict] = {}

    for row in teams.sort_values(["date_dt", "gid", "team"]).to_dict("records"):
        history = team_history[row["team"]]
        values = {}
        for window in WINDOWS:
            if window == "season":
                selected = [x for x in history if x["date_dt"].year == row["date_dt"].year]
            else:
                cutoff = row["date_dt"] - timedelta(days=int(window[:-1]))
                selected = [x for x in history if x["date_dt"] >= cutoff]
            rates = team_rates(_sum_rows(selected, RAW_TEAM_COLUMNS)) if selected else {}
            values.update({f"{name}_{window}": value for name, value in rates.items()})
        pregame_team[(row["gid"], row["team"])] = values
        history.append(row)

    for row in starters.sort_values(["date_dt", "gid", "id"]).to_dict("records"):
        history = starter_history[row["id"]]
        season_rows = [x for x in history if x["date_dt"].year == row["date_dt"].year]
        values = {}
        for label, selected in (("last5", history[-5:]), ("season", season_rows)):
            rates = pitcher_rates(_sum_rows(selected, RAW_PITCHER_COLUMNS)) if selected else {}
            values.update({f"starter_{name}_{label}": value for name, value in rates.items()})
        pregame_starter[(row["gid"], row["team"])] = values
        history.append(row)

    reliever_game_rows = []
    for (gid, team), frame in relievers.groupby(["gid", "team"], sort=False):
        first = frame.iloc[0]
        raw = {column: float(frame[column].sum()) for column in RAW_PITCHER_COLUMNS}
        raw.update({"gid": gid, "team": team, "date_dt": first["date_dt"]})
        reliever_game_rows.append(raw)
    for row in sorted(reliever_game_rows, key=lambda x: (x["date_dt"], x["gid"], x["team"])):
        history = bullpen_history[row["team"]]
        selected3 = [x for x in history if x["date_dt"] >= row["date_dt"] - timedelta(days=3)]
        selected7 = [x for x in history if x["date_dt"] >= row["date_dt"] - timedelta(days=7)]
        raw3, raw7 = _sum_rows(selected3, RAW_PITCHER_COLUMNS), _sum_rows(selected7, RAW_PITCHER_COLUMNS)
        rates3, rates7 = pitcher_rates(raw3), pitcher_rates(raw7)
        pregame_bullpen[(row["gid"], row["team"])] = {
            "bullpen_era_3d": rates3["era"], "bullpen_era_7d": rates7["era"],
            "bullpen_whip_3d": rates3["whip"], "bullpen_whip_7d": rates7["whip"],
            "bullpen_k_pct_7d": rates7["k_pct"], "bullpen_bb_pct_7d": rates7["bb_pct"],
            "bullpen_ip_3d": raw3["p_ipouts"] / 3, "bullpen_ip_7d": raw7["p_ipouts"] / 3,
        }
        history.append(row)

    elo = defaultdict(lambda: 1500.0)
    last_game: dict[str, tuple[datetime, str]] = {}
    park_runs, park_games, league_runs, league_games = defaultdict(float), defaultdict(float), 0.0, 0.0
    il_timelines = fetch_il_timelines()
    statcast_timelines = fetch_statcast_daily_aggregates()
    output = []
    for game in games.sort_values(["date_dt", "gid"]).to_dict("records"):
        home, away, gid = game["hometeam"], game["visteam"], game["gid"]
        home_team, away_team = pregame_team.get((gid, home), {}), pregame_team.get((gid, away), {})
        home_starter, away_starter = pregame_starter.get((gid, home), {}), pregame_starter.get((gid, away), {})
        home_pen, away_pen = pregame_bullpen.get((gid, home), {}), pregame_bullpen.get((gid, away), {})
        row = {
            "game_date": game["date_dt"].date().isoformat(), "home_win": int(float(game["hruns"]) > float(game["vruns"])),
            "home_team": home, "away_team": away, "site": game["site"],
            "home_runs": float(game["hruns"]), "away_runs": float(game["vruns"]),
        }
        for metric in (*OFFENSE_METRICS, *TEAM_PITCHING_METRICS, *DEFENSE_METRICS):
            for window in WINDOWS:
                row[f"{metric}_{window}_diff"] = home_team.get(f"{metric}_{window}", math.nan) - away_team.get(f"{metric}_{window}", math.nan)
        for metric in ("era", "whip", "k_pct", "bb_pct", "hr_pct", "ip_per_start"):
            for window in ("last5", "season"):
                row[f"starter_{metric}_{window}_diff"] = home_starter.get(f"starter_{metric}_{window}", math.nan) - away_starter.get(f"starter_{metric}_{window}", math.nan)
        for name in ("bullpen_era_3d", "bullpen_era_7d", "bullpen_whip_3d", "bullpen_whip_7d", "bullpen_k_pct_7d", "bullpen_bb_pct_7d", "bullpen_ip_3d", "bullpen_ip_7d"):
            row[f"{name}_diff"] = home_pen.get(name, math.nan) - away_pen.get(name, math.nan)
        roof_closed = str(game.get("sky", "")).lower() == "dome"
        row.update({
            "temperature_f": float(game.get("temp")) if str(game.get("temp", "")).replace(".", "", 1).isdigit() else math.nan,
            "wind_speed_mph": float(game.get("windspeed")) if str(game.get("windspeed", "")).replace(".", "", 1).isdigit() else math.nan,
            "roof_closed": float(roof_closed), "day_game": float(str(game.get("daynight", "")).lower() == "day"),
            "park_run_factor": (park_runs[game["site"]] / park_games[game["site"]]) / (league_runs / league_games) if park_games[game["site"]] >= 20 and league_games else math.nan,
            "team_elo_rating_diff": elo[home] - elo[away],
        })
        home_last, away_last = last_game.get(home), last_game.get(away)
        home_rest = (game["date_dt"] - home_last[0]).days if home_last else math.nan
        away_rest = (game["date_dt"] - away_last[0]).days if away_last else math.nan
        row["rest_days_diff"] = home_rest - away_rest if not (math.isnan(home_rest) or math.isnan(away_rest)) else math.nan
        home_travel = float(bool(home_last and home_rest == 1 and home_last[1] != game["site"]))
        away_travel = float(bool(away_last and away_rest == 1 and away_last[1] != game["site"]))
        row["traveled_yesterday_diff"] = home_travel - away_travel

        game_date = game["date_dt"].date()
        home_il_pitchers, home_il_batters = active_il_counts(il_timelines.get(home, []), game_date)
        away_il_pitchers, away_il_batters = active_il_counts(il_timelines.get(away, []), game_date)
        row["il_pitchers_diff"] = float(home_il_pitchers - away_il_pitchers)
        row["il_batters_diff"] = float(home_il_batters - away_il_batters)

        home_statcast_days, away_statcast_days = statcast_timelines.get(home, []), statcast_timelines.get(away, [])
        for window in STATCAST_WINDOWS:
            window_days = int(window[:-1])
            home_rates = statcast_rates(home_statcast_days, game_date, window_days)
            away_rates = statcast_rates(away_statcast_days, game_date, window_days)
            for metric in STATCAST_METRICS:
                row[f"{metric}_{window}_diff"] = home_rates[metric] - away_rates[metric]
        output.append(row)

        expected = 1 / (1 + 10 ** (-((elo[home] + 24) - elo[away]) / 400))
        elo[home] += 20 * (row["home_win"] - expected)
        elo[away] += 20 * ((1 - row["home_win"]) - (1 - expected))
        total_runs = float(game["hruns"]) + float(game["vruns"])
        park_runs[game["site"]] += total_runs; park_games[game["site"]] += 1
        league_runs += total_runs; league_games += 1
        last_game[home] = (game["date_dt"], game["site"]); last_game[away] = (game["date_dt"], game["site"])

    frame = pd.DataFrame(output)
    return frame[["game_date", "home_team", "away_team", "site", "home_runs", "away_runs", *FEATURE_NAMES, "home_win"]]
