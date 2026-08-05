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
    "team_elo_rating_diff", "il_pitchers_diff", "il_batters_diff",
]

# Quality-of-contact from Baseball Savant. Deliberately no "season" window here
# (unlike WINDOWS above) -- a season-long Statcast rolling sum can't be computed
# cheaply at live-inference time without re-summing the whole season day by day,
# and the live and training feature sets must stay identical.
#
# barrel_rate was tried and dropped: validated via retrain, it made the model
# worse (Brier 0.24546 -> 0.24611) even though hard_hit_rate alone -- same
# windows, same both-sides structure -- measurably improved it (-> 0.24521).
# Barrel rate's narrower launch-angle+speed definition is noisier over these
# short windows on a ~12k-game training set; hard-hit rate (just >=95mph exit
# velocity) is the more stable of the two Savant metrics in practice here.
STATCAST_METRICS = ("hard_hit_rate", "hard_hit_rate_allowed")
STATCAST_WINDOWS = ("7d", "14d", "30d")
STATCAST = [f"{metric}_{window}_diff" for metric in STATCAST_METRICS for window in STATCAST_WINDOWS]

MARKET: list[str] = []  # post-model comparison, never a leaked training input

FEATURE_CATEGORIES = {
    "pitching": PITCHING,
    "bullpen": BULLPEN,
    "lineup": OFFENSE,
    "defense": DEFENSE,
    "weather": CONTEXT,
    "statcast": STATCAST,
    "market": MARKET,
}
FEATURE_NAMES = [name for names in FEATURE_CATEGORIES.values() for name in names]
FEATURE_VERSION = "2.2.0-free"

assert len(FEATURE_NAMES) == 128, len(FEATURE_NAMES)
assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

