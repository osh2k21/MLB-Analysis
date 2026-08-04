from __future__ import annotations

from datetime import datetime, timedelta, timezone
import tempfile
import unittest
from pathlib import Path

from mlb_predictor.api.injuries import InjuryClient
from mlb_predictor.calibration import IdentityCalibrator
from mlb_predictor.contracts import DataStatus, Evidence, GameRef
from mlb_predictor.features import FEATURE_NAMES, FeatureEngine, FeatureError
from mlb_predictor.features.market import american_implied, engineer_market
from mlb_predictor.features.parks import park_run_factor
from mlb_predictor.features.weather import air_density_kg_m3
from mlb_predictor.models.ensemble import EnsembleBundle
from mlb_predictor.database import PredictionRepository
from mlb_predictor.pipeline import PredictionPipeline
from mlb_predictor.validation import GameValidator
from mlb_predictor.validation.validator import ValidationCheck, ValidationReport


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


class FakeHttpResult:
    def __init__(self, data):
        self.data = data


class FakeTransactionsHttp:
    def __init__(self, transactions):
        self.transactions = transactions

    def get_json(self, url, params=None, *, ttl_seconds, validator=None):
        return FakeHttpResult({"transactions": self.transactions})


class InjuryClientTests(unittest.TestCase):
    def test_active_il_placement_without_activation(self):
        http = FakeTransactionsHttp([
            {"date": "2026-07-01", "person": {"id": 1, "fullName": "A"},
             "description": "New York Yankees placed RF A on the 10-day injured list."},
        ])
        active = InjuryClient(http)._team_active_il(147)
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["player_id"], 1)

    def test_activation_clears_the_placement(self):
        http = FakeTransactionsHttp([
            {"date": "2026-07-01", "person": {"id": 1, "fullName": "A"},
             "description": "New York Yankees placed RF A on the 10-day injured list."},
            {"date": "2026-07-15", "person": {"id": 1, "fullName": "A"},
             "description": "New York Yankees activated RF A from the 10-day injured list."},
        ])
        active = InjuryClient(http)._team_active_il(147)
        self.assertEqual(active, [])

    def test_fetch_returns_home_and_away_evidence(self):
        http = FakeTransactionsHttp([])
        evidence = InjuryClient(http).fetch(147, 111)
        self.assertEqual(evidence.status, DataStatus.OK)
        self.assertEqual(evidence.payload, {"home": [], "away": []})


class ParkFactorTests(unittest.TestCase):
    def test_unmapped_venue_is_neutral(self):
        self.assertEqual(park_run_factor("Some New Venue", {"NYC21": 1.2}), 1.0)

    def test_no_history_for_mapped_venue_is_neutral(self):
        self.assertEqual(park_run_factor("Yankee Stadium", {}), 1.0)

    def test_mapped_venue_returns_trained_factor(self):
        self.assertAlmostEqual(park_run_factor("Yankee Stadium", {"NYC21": 0.979}), 0.979)

    def test_sponsor_prefix_still_resolves(self):
        self.assertAlmostEqual(
            park_run_factor("UNIQLO Field at Dodger Stadium", {"LOS03": 1.009}), 1.009
        )


class ValidationAdvisoryTests(unittest.TestCase):
    def test_advisory_failure_alone_does_not_block(self):
        checks = (
            ValidationCheck("Weather available", True, "ok", "src"),
            ValidationCheck("Umpire assigned", False, "not yet published", "src", advisory=True),
        )
        report = ValidationReport(1, checks, datetime.now(timezone.utc))
        self.assertTrue(report.ready)
        self.assertEqual(len(report.failures), 1)

    def test_non_advisory_failure_still_blocks(self):
        checks = (
            ValidationCheck("Weather available", False, "missing", "src"),
            ValidationCheck("Umpire assigned", False, "not yet published", "src", advisory=True),
        )
        report = ValidationReport(1, checks, datetime.now(timezone.utc))
        self.assertFalse(report.ready)


if __name__ == "__main__":
    unittest.main()
