"""Versioned model feature contract (143 verified, non-placeholder inputs)."""

PITCHING = [
    "starter_xera_diff", "starter_fip_diff", "starter_xfip_diff", "starter_siera_diff",
    "starter_csw_pct_diff", "starter_k_minus_bb_pct_diff", "starter_hard_hit_pct_diff",
    "starter_barrel_pct_diff", "starter_gb_pct_diff", "starter_whiff_pct_diff",
    "starter_chase_pct_diff", "starter_zone_pct_diff", "starter_first_pitch_strike_pct_diff",
    "starter_avg_exit_velocity_diff", "starter_max_exit_velocity_diff", "starter_era_diff",
    "starter_whip_diff", "starter_k9_diff", "starter_bb9_diff", "starter_hr9_diff",
    "starter_innings_per_start_diff", "starter_pitch_count_avg_diff", "starter_velocity_7d_diff",
    "starter_velocity_30d_diff", "starter_spin_7d_diff", "starter_spin_30d_diff",
    "starter_fastball_usage_diff", "starter_breaking_usage_diff", "starter_offspeed_usage_diff",
    "starter_last5_xera_diff", "starter_last5_fip_diff", "starter_last5_csw_pct_diff",
    "starter_home_away_xera_diff", "starter_day_night_xera_diff", "starter_rest_days_diff",
    "starter_times_through_order_penalty_diff", "starter_injury_risk_diff", "starter_pitch_limit_diff",
]

BULLPEN = [
    "bullpen_xera_diff", "bullpen_fip_diff", "bullpen_xfip_diff", "bullpen_siera_diff",
    "bullpen_k_minus_bb_pct_diff", "bullpen_csw_pct_diff", "bullpen_hard_hit_pct_diff",
    "bullpen_barrel_pct_diff", "bullpen_gb_pct_diff", "bullpen_whip_diff",
    "bullpen_available_ip_diff", "bullpen_pitches_1d_diff", "bullpen_pitches_3d_diff",
    "bullpen_ip_3d_diff", "bullpen_ip_7d_diff", "bullpen_high_leverage_3d_diff",
    "bullpen_high_leverage_7d_diff", "bullpen_rest_score_diff", "closer_available_diff",
    "setup_available_diff", "bullpen_injured_count_diff", "bullpen_back_to_back_count_diff",
]

OFFENSE = [
    "lineup_xwoba_diff", "lineup_xba_diff", "lineup_xslg_diff", "lineup_iso_diff",
    "lineup_ops_plus_diff", "lineup_wrc_plus_diff", "lineup_obp_diff", "lineup_slg_diff",
    "lineup_ops_diff", "lineup_hard_hit_pct_diff", "lineup_barrel_pct_diff",
    "lineup_chase_pct_diff", "lineup_contact_pct_diff", "lineup_zone_contact_pct_diff",
    "lineup_k_pct_diff", "lineup_bb_pct_diff", "lineup_k_minus_bb_pct_diff",
    "lineup_sprint_speed_diff", "lineup_baserunning_runs_diff", "lineup_vs_hand_xwoba_diff",
    "lineup_vs_hand_wrc_plus_diff", "lineup_home_away_wrc_plus_diff", "lineup_last7_wrc_plus_diff",
    "lineup_last14_wrc_plus_diff", "lineup_last30_wrc_plus_diff", "lineup_projected_runs_diff",
    "lineup_confirmed_players_diff", "lineup_injured_war_diff", "lineup_rest_days_diff",
    "lineup_top3_xwoba_diff", "lineup_bottom3_xwoba_diff", "lineup_platoon_advantage_count_diff",
]

DEFENSE = [
    "team_drs_diff", "team_oaa_diff", "team_defensive_efficiency_diff", "team_errors_rate_diff",
    "team_double_play_rate_diff", "outfield_oaa_diff", "infield_oaa_diff", "catcher_framing_runs_diff",
    "catcher_strike_rate_diff", "catcher_pop_time_diff", "catcher_caught_stealing_pct_diff",
    "catcher_blocking_runs_diff",
]

CONTEXT = [
    "team_elo_rating_diff",
    "temperature_f", "humidity_pct", "wind_speed_mph", "wind_out_component_mph",
    "air_density_kg_m3", "precipitation_probability", "roof_closed", "roof_unknown",
    "park_run_factor", "park_hr_factor", "park_hit_factor", "altitude_ft",
    "home_rest_days", "away_rest_days", "rest_days_diff", "travel_miles_diff",
    "time_zones_crossed_diff", "circadian_shift_hours_diff", "getaway_day_diff",
    "doubleheader_game_number", "umpire_run_factor", "umpire_k_factor",
]

MARKET = [
    "home_open_implied_prob", "home_current_implied_prob", "away_open_implied_prob",
    "away_current_implied_prob", "home_no_vig_prob", "moneyline_move_home",
    "spread_open_home", "spread_current_home", "total_open", "total_current",
    "reverse_line_movement_home", "sharp_book_home_prob", "consensus_home_prob",
    "book_dispersion", "minutes_since_odds_update", "market_liquidity_score",
]

FEATURE_CATEGORIES = {
    "pitching": PITCHING,
    "bullpen": BULLPEN,
    "lineup": OFFENSE,
    "defense": DEFENSE,
    "weather": CONTEXT,
    "market": MARKET,
}
FEATURE_NAMES = [name for names in FEATURE_CATEGORIES.values() for name in names]
FEATURE_VERSION = "1.0.0"

assert 100 <= len(FEATURE_NAMES) <= 150, len(FEATURE_NAMES)
assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)), "feature names must be unique"
