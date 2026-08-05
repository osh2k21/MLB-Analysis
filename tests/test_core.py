from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import math
import tempfile
import unittest
from pathlib import Path

from mlb_predictor.calibration import IdentityCalibrator
from mlb_predictor.contracts import DataStatus, Evidence, GameRef
from mlb_predictor.features import FEATURE_NAMES, FeatureEngine, FeatureError
from mlb_predictor.features.market import american_implied, engineer_market
from mlb_predictor.features.weather import air_density_kg_m3
from mlb_predictor.models.ensemble import EnsembleBundle
from mlb_predictor.database import PredictionRepository
from mlb_predictor.pipeline import PredictionPipeline
from mlb_predictor.training.injuries import active_il_counts, parse_il_events
from mlb_predictor.training.statcast import aggregate_day, statcast_rates
from mlb_predictor.validation import GameValidator


class ConstantModel:
    def __init__(self, name, value):
        self.name, self.value = name, value

    def predict_home_probability(self, rows):
        return [self.value] * len(rows)


def game():
    return GameRef(1, "2026-08-03", datetime.now(timezone.utc) + timedelta(hours=1), 10, 20, "Away", "Home", 30, "Park")


def complete_evidence():
    now = datetime.now(timezone.utc)
    feed = {"gameData": {"probablePitchers": {"home": {"id": 1}, "away": {"id": 2}}}, "liveData": {"boxscore": {"teams": {"home": {"battingOrder": list(range(9))}, "away": {"battingOrder": list(range(9))}}}}}
    return {
        "mlb_feed": Evidence("mlb", DataStatus.OK, now, feed),
        "bullpen": Evidence("retro", DataStatus.OK, now, {"home": {"available": 1}, "away": {"available": 1}}),
        "weather": Evidence("weather", DataStatus.OK, now, {"main": {}, "wind": {}}),
        "odds": Evidence("odds", DataStatus.OK, now, {"home": {"price": -110}, "away": {"price": 100}}),
        "statcast": Evidence("savant", DataStatus.OK, now, {"home": {"xwoba": .3}, "away": {"xwoba": .3}}),
        "injuries": Evidence("injury", DataStatus.OK, now, {"home": [], "away": []}),
        "umpire": Evidence("ump", DataStatus.OK, now, {"home_plate": "Ump"}),
    }


class CoreTests(unittest.TestCase):
    def test_feature_contract_has_requested_scale(self):
        self.assertGreaterEqual(len(FEATURE_NAMES), 100)
        self.assertLessEqual(len(FEATURE_NAMES), 150)
        self.assertEqual(len(FEATURE_NAMES), len(set(FEATURE_NAMES)))

    def test_features_never_impute(self):
        with self.assertRaises(FeatureError):
            FeatureEngine().build(game(), {}, {})

    def test_validator_is_fail_closed(self):
        report = GameValidator().validate(game(), {})
        self.assertFalse(report.ready)
        self.assertEqual(report.decision, "Prediction Withheld")
        self.assertEqual(len(report.checks), 8)

    def test_validator_passes_complete_fresh_evidence(self):
        self.assertTrue(GameValidator().validate(game(), complete_evidence()).ready)

    def test_ensemble_normalizes_votes(self):
        names = EnsembleBundle.REQUIRED_MODELS
        models = {name: ConstantModel(name, 0.6) for name in names}
        bundle = EnsembleBundle(models, {name: 1 for name in names}, ["x"], IdentityCalibrator(), [0])
        result = bundle.predict([[1]])[0]
        self.assertAlmostEqual(result.home_probability, 0.6)
        self.assertEqual(len(result.votes), 5)

    def test_market_removes_vig(self):
        values = engineer_market({"home_current_american": -110, "away_current_american": -110})
        self.assertAlmostEqual(values["home_no_vig_prob"], 0.5)
        self.assertAlmostEqual(american_implied(100), 0.5)

    def test_air_density_is_physical(self):
        density = air_density_kg_m3(70, 50, 1013.25)
        self.assertGreater(density, 1.1)
        self.assertLess(density, 1.3)

    def test_pipeline_persists_auditable_prediction(self):
        names = EnsembleBundle.REQUIRED_MODELS
        models = {name: ConstantModel(name, 0.6) for name in names}
        bundle = EnsembleBundle(models, {name: 1 for name in names}, FEATURE_NAMES, IdentityCalibrator(), [0] * len(FEATURE_NAMES))
        values = {name: 0.0 for name in FEATURE_NAMES}
        provenance = {name: "test fixture" for name in FEATURE_NAMES}
        snapshot = FeatureEngine().build(game(), values, provenance)
        with tempfile.TemporaryDirectory() as directory:
            repository = PredictionRepository(Path(directory) / "test.sqlite3")
            result = PredictionPipeline(bundle, repository).predict(game(), complete_evidence(), snapshot)
            self.assertAlmostEqual(result.prediction.home_probability, 0.6)
            self.assertEqual(len(repository.settled()), 0)
            with repository.connection() as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], 1)


class InjuriesFeatureTests(unittest.TestCase):
    def test_parses_pitcher_and_batter_placements(self):
        snapshots = parse_il_events([
            {"date": "2024-05-01", "person": {"id": 1}, "description": "New York Yankees placed LHP Carlos Rodón on the 15-day injured list."},
            {"date": "2024-05-02", "person": {"id": 2}, "description": "New York Yankees placed RF Aaron Judge on the 10-day injured list."},
            {"date": "2024-05-03", "person": {"id": 3}, "description": "New York Yankees placed 3B Ryan McMahon on the 10-day injured list."},
        ])
        self.assertEqual(snapshots, [
            (date(2024, 5, 1), 1, 0),
            (date(2024, 5, 2), 1, 1),
            (date(2024, 5, 3), 1, 2),
        ])

    def test_activation_removes_the_player(self):
        snapshots = parse_il_events([
            {"date": "2024-05-01", "person": {"id": 1}, "description": "New York Yankees placed LHP Carlos Rodón on the 15-day injured list."},
            {"date": "2024-05-10", "person": {"id": 1}, "description": "New York Yankees activated LHP Carlos Rodón from the 15-day injured list."},
        ])
        self.assertEqual(snapshots[-1], (date(2024, 5, 10), 0, 0))

    def test_unrelated_transactions_are_ignored(self):
        snapshots = parse_il_events([
            {"date": "2024-05-01", "person": {"id": 1}, "description": "New York Yankees signed RHP Michael Harpster."},
            {"date": "2024-05-02", "person": {"id": 2}, "description": "New York Yankees optioned 2B Oswald Peraza to Triple-A."},
        ])
        self.assertEqual(snapshots, [])

    def test_duplicate_placement_for_the_same_stint_does_not_double_count(self):
        # MLB's transaction log really does contain retroactive/corrected re-entries
        # for the same injury stint -- a second "placed" for a player already
        # tracked as active must be a no-op (no new snapshot), not another +1.
        snapshots = parse_il_events([
            {"date": "2024-05-01", "person": {"id": 1}, "description": "New York Yankees placed RF Aaron Judge on the 10-day injured list. Right rib stress fracture."},
            {"date": "2024-05-03", "person": {"id": 1}, "description": "New York Yankees placed RF Aaron Judge on the 10-day injured list retroactive to May 1, 2024. Right rib stress fracture."},
        ])
        self.assertEqual(snapshots, [(date(2024, 5, 1), 0, 1)])

    def test_placement_without_activation_counts_as_active(self):
        snapshots = [(date(2024, 5, 1), 1, 0)]
        self.assertEqual(active_il_counts(snapshots, date(2024, 5, 15)), (1, 0))

    def test_activation_clears_the_placement(self):
        snapshots = [(date(2024, 5, 1), 1, 0), (date(2024, 5, 10), 0, 0)]
        self.assertEqual(active_il_counts(snapshots, date(2024, 5, 15)), (0, 0))

    def test_snapshots_on_or_after_as_of_do_not_count(self):
        snapshots = [(date(2024, 5, 10), 0, 1)]
        self.assertEqual(active_il_counts(snapshots, date(2024, 5, 10)), (0, 0))
        self.assertEqual(active_il_counts(snapshots, date(2024, 5, 9)), (0, 0))
        self.assertEqual(active_il_counts(snapshots, date(2024, 5, 11)), (0, 1))


def _pitch(home_team, away_team, top, launch_speed=None, launch_speed_angle=None):
    return {
        "home_team": home_team, "away_team": away_team,
        "inning_topbot": "Top" if top else "Bot",
        "launch_speed": launch_speed, "launch_speed_angle": launch_speed_angle,
    }


class StatcastFeatureTests(unittest.TestCase):
    def test_barrel_credited_to_batter_and_allowed_to_opposing_pitcher(self):
        # Bot half: home team (LAA) is batting, away team (PHI) is pitching.
        rows = [_pitch("LAA", "PHI", top=False, launch_speed="102.5", launch_speed_angle="6")]
        totals = aggregate_day(rows)
        self.assertEqual(totals["LAA"], (1, 1, 1, 0, 0, 0))  # batting: 1 barrel, 1 hard-hit, 1 batted ball
        self.assertEqual(totals["PHI"], (0, 0, 0, 1, 1, 1))  # allowed: same ball counted against PHI's pitching

    def test_top_half_attributes_to_away_batting(self):
        # Top half: away team (PHI) is batting, home team (LAA) is pitching.
        rows = [_pitch("LAA", "PHI", top=True, launch_speed="96.0", launch_speed_angle="4")]
        totals = aggregate_day(rows)
        self.assertEqual(totals["PHI"][:3], (0, 1, 1))  # hard-hit (>=95) but not a barrel
        self.assertEqual(totals["LAA"][3:], (0, 1, 1))

    def test_non_batted_ball_rows_are_ignored(self):
        rows = [_pitch("LAA", "PHI", top=False, launch_speed=None)]
        self.assertEqual(aggregate_day(rows), {})

    def test_below_hard_hit_threshold_counts_as_batted_ball_only(self):
        rows = [_pitch("LAA", "PHI", top=False, launch_speed="80.0", launch_speed_angle="2")]
        totals = aggregate_day(rows)
        self.assertEqual(totals["LAA"], (0, 0, 1, 0, 0, 0))

    def test_unmapped_team_code_is_skipped_not_crashed(self):
        rows = [_pitch("XXX", "PHI", top=False, launch_speed="102.5", launch_speed_angle="6")]
        self.assertEqual(aggregate_day(rows), {})

    def test_statcast_rates_windows_and_nan_on_no_data(self):
        daily_rows = [
            (date(2024, 5, 1), 2, 3, 5, 1, 2, 4),  # barrels, hard_hits, bb, barrels_allowed, hh_allowed, bb_allowed
            (date(2024, 5, 10), 0, 0, 3, 0, 0, 2),
        ]
        # 7-day window from 2024-05-15 only reaches back to 2024-05-08, so only the 05-10 row counts.
        rates = statcast_rates(daily_rows, date(2024, 5, 15), 7)
        self.assertEqual(rates["barrel_rate"], 0 / 3)
        self.assertEqual(rates["barrel_rate_allowed"], 0 / 2)
        # 30-day window picks up both rows.
        rates_30 = statcast_rates(daily_rows, date(2024, 5, 15), 30)
        self.assertAlmostEqual(rates_30["barrel_rate"], 2 / 8)
        self.assertAlmostEqual(rates_30["hard_hit_rate_allowed"], 2 / 6)

    def test_statcast_rates_nan_with_zero_batted_balls_in_window(self):
        rates = statcast_rates([], date(2024, 5, 15), 7)
        self.assertTrue(math.isnan(rates["barrel_rate"]))
        self.assertTrue(math.isnan(rates["hard_hit_rate_allowed"]))


if __name__ == "__main__":
    unittest.main()
