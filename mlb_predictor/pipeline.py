from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

from .contracts import Evidence, FeatureSnapshot, GameRef
from .database import PredictionRepository
from .explainability import Explanation, explain_prediction
from .features.catalog import FEATURE_VERSION
from .models.ensemble import EnsembleBundle, EnsemblePrediction
from .validation import GameValidator, ValidationReport


class PredictionWithheld(RuntimeError):
    def __init__(self, report: ValidationReport, extra_reason: str | None = None) -> None:
        self.report, self.extra_reason = report, extra_reason
        reasons = [item.detail for item in report.failures]
        if extra_reason:
            reasons.append(extra_reason)
        super().__init__("Prediction Withheld: " + "; ".join(reasons))


@dataclass(frozen=True)
class PredictionResult:
    prediction: EnsemblePrediction
    explanation: Explanation
    validation: ValidationReport


class PredictionPipeline:
    def __init__(self, bundle: EnsembleBundle, repository: PredictionRepository, validator: GameValidator | None = None) -> None:
        self.bundle, self.repository = bundle, repository
        self.validator = validator or GameValidator()

    def predict(self, game: GameRef, evidence: Mapping[str, Evidence], snapshot: FeatureSnapshot, decimal_odds: float | None = None) -> PredictionResult:
        report = self.validator.validate(game, evidence)
        self.repository.record_validation(game.game_pk, report.ready, [vars(x) for x in report.checks])
        if not report.ready:
            raise PredictionWithheld(report)
        if snapshot.game.game_pk != game.game_pk:
            raise PredictionWithheld(report, "feature snapshot belongs to a different game")
        try:
            artifact_created = datetime.fromisoformat(self.bundle.created_at.replace("Z", "+00:00"))
            artifact_age = datetime.now(timezone.utc) - artifact_created.astimezone(timezone.utc)
            if artifact_age.total_seconds() > 8 * 24 * 60 * 60:
                raise PredictionWithheld(report, f"model artifact is {artifact_age.days} days old; weekly retraining is overdue")
        except (AttributeError, ValueError):
            raise PredictionWithheld(report, "model artifact has no valid creation timestamp")
        row = snapshot.ordered(self.bundle.feature_names)
        prediction = self.bundle.predict([row])[0]
        explanation = explain_prediction(self.bundle, snapshot)
        now = datetime.now(timezone.utc).isoformat()
        self.repository.record_prediction({
            "game_pk": game.game_pk, "game_date": game.game_date, "created_at": now,
            "model_version": FEATURE_VERSION, "raw_home_probability": prediction.raw_home_probability,
            "home_probability": prediction.home_probability, "feature_json": json.dumps(snapshot.values),
            "provenance_json": json.dumps(snapshot.provenance),
            "votes_json": json.dumps([vars(v) for v in prediction.votes]),
            "explanation_json": json.dumps(explanation.category_points), "decimal_odds": decimal_odds,
        })
        return PredictionResult(prediction, explanation, report)


def load_bundle(path: Path) -> EnsembleBundle:
    if not path.exists():
        raise FileNotFoundError(f"trained ensemble not found at {path}")
    import joblib
    bundle = joblib.load(path)
    if not isinstance(bundle, EnsembleBundle):
        raise TypeError("model artifact is not an EnsembleBundle")
    return bundle
