"""Feature contract reproducible from free MLB, Retrosheet and weather data.

All rolling values are computed strictly before the predicted game. Market odds
are intentionally excluded from the model because comparable historical odds
are not available on the free Odds API plan; live odds are used after inference
to calculate no-vig edge.
"""

WINDOWS = ("7d", "14d", "30d", "season")

OFFENSE_METRICS = (
    "runs_pg", "avg", "obp", "slg", "ops", "iso", "bb_pct", "k_pct",
    "hr_pa", "babip", "sb_success", "lob_pg", "gdp_pa",
)
TEAM_PITCHING_METRICS = (
    "runs_allowed_pg", "era", "whip", "pitch_k_pct", "pitch_bb_pct",
    "pitch_k_minus_bb_pct", "pitch_hr_pct",
)
DEFENSE_METRICS = ("fielding_pct", "errors_pg", "double_plays_pg")

OFFENSE = [f"{metric}_{window}_diff" for metric in OFFENSE_METRICS for window in WINDOWS]
PITCHING = [f"{metric}_{window}_diff" for metric in TEAM_PITCHING_METRICS for window in WINDOWS]
DEFENSE = [f"{metric}_{window}_diff" for metric in DEFENSE_METRICS for window in WINDOWS]

STARTER_METRICS = ("era", "whip", "k_pct", "bb_pct", "hr_pct", "ip_per_start")
STARTER = [f"starter_{metric}_{window}_diff" for metric in STARTER_METRICS for window in ("last5", "season")]
PITCHING += STARTER

BULLPEN = [
    "bullpen_era_3d_diff", "bullpen_era_7d_diff", "bullpen_whip_3d_diff",
    "bullpen_whip_7d_diff", "bullpen_k_pct_7d_diff", "bullpen_bb_pct_7d_diff",
    "bullpen_ip_3d_diff", "bullpen_ip_7d_diff",
]

CONTEXT = [
    "temperature_f", "wind_speed_mph", "roof_closed", "day_game",
    "park_run_factor", "rest_days_diff", "traveled_yesterday_diff",
    "team_elo_rating_diff",
]

MARKET: list[str] = []  # post-model comparison, never a leaked training input

FEATURE_CATEGORIES = {
    "pitching": PITCHING,
    "bullpen": BULLPEN,
    "lineup": OFFENSE,
    "defense": DEFENSE,
    "weather": CONTEXT,
    "market": MARKET,
}
FEATURE_NAMES = [name for names in FEATURE_CATEGORIES.values() for name in names]
FEATURE_VERSION = "2.0.0-free"

assert len(FEATURE_NAMES) == 120, len(FEATURE_NAMES)
assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

