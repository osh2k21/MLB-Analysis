from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ..backtesting import evaluate_predictions
from ..calibration import IsotonicCalibrator, PlattCalibrator
from ..features.catalog import FEATURE_NAMES
from ..models.elo import EloModel
from ..models.ensemble import EnsembleBundle
from ..models.poisson import PoissonModel
from ..models.sklearn_adapter import SklearnProbabilityModel


def _brier(probabilities, outcomes) -> float:
    return sum((float(p) - int(y)) ** 2 for p, y in zip(probabilities, outcomes)) / len(outcomes)


def train_bundle(csv_path: Path, artifact_path: Path) -> tuple[EnsembleBundle, dict]:
    import joblib
    import pandas as pd
    from ..models.logistic import build_logistic_model
    from ..models.xgboost import build_xgboost_model
    from ..models.lightgbm import build_lightgbm_model

    frame = pd.read_csv(csv_path)
    required = set(FEATURE_NAMES + ["game_date", "home_win"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"training file missing {len(missing)} columns: {sorted(missing)[:12]}")
    frame = frame.sort_values("game_date").reset_index(drop=True)
    if len(frame) < 500:
        raise ValueError("at least 500 chronologically ordered games are required")
    if frame[FEATURE_NAMES].isna().any().any():
        raise ValueError("training features contain missing values; no imputation is allowed")
    if not set(frame["home_win"].unique()).issubset({0, 1}):
        raise ValueError("home_win must be 0/1")

    train_end, calibration_end = int(len(frame) * 0.70), int(len(frame) * 0.85)
    train, calibration, test = frame.iloc[:train_end], frame.iloc[train_end:calibration_end], frame.iloc[calibration_end:]
    x_train, y_train = train[FEATURE_NAMES], train["home_win"].astype(int)
    x_cal, y_cal = calibration[FEATURE_NAMES], calibration["home_win"].astype(int)
    x_test, y_test = test[FEATURE_NAMES], test["home_win"].astype(int)

    estimators = {
        "logistic": build_logistic_model(),
        "xgboost": build_xgboost_model(),
        "lightgbm": build_lightgbm_model(),
    }
    models = {}
    for name, estimator in estimators.items():
        estimator.fit(x_train, y_train)
        models[name] = SklearnProbabilityModel(name, estimator)
    models["elo"] = EloModel(FEATURE_NAMES.index("team_elo_rating_diff"))
    models["poisson"] = PoissonModel(FEATURE_NAMES.index("lineup_projected_runs_diff"), FEATURE_NAMES.index("total_current"))

    cal_rows = x_cal.values.tolist()
    component_cal = {name: model.predict_home_probability(cal_rows) for name, model in models.items()}
    inverse_brier = {name: 1.0 / max(1e-6, _brier(probs, y_cal)) for name, probs in component_cal.items()}
    weight_total = sum(inverse_brier.values())
    weights = {name: score / weight_total for name, score in inverse_brier.items()}
    raw_cal = [sum(component_cal[name][i] * weights[name] for name in models) for i in range(len(calibration))]

    midpoint = max(50, len(calibration) // 2)
    candidates = [PlattCalibrator(), IsotonicCalibrator()]
    for candidate in candidates:
        candidate.fit(raw_cal[:midpoint], y_cal.iloc[:midpoint].tolist())
    selection_probs = [candidate.transform(raw_cal[midpoint:]) for candidate in candidates]
    selected = min(zip(candidates, selection_probs), key=lambda pair: _brier(pair[1], y_cal.iloc[midpoint:]))[0]
    selected.fit(raw_cal, y_cal.tolist())

    bundle = EnsembleBundle(models, weights, FEATURE_NAMES, selected, x_train.mean().tolist(), str(train["game_date"].max()))
    test_predictions = bundle.predict(x_test.values.tolist())
    test_probs = [item.home_probability for item in test_predictions]
    evaluation_probabilities = [max(p, 1 - p) for p in test_probs]
    evaluation_outcomes = [int(y) if p >= 0.5 else 1 - int(y) for p, y in zip(test_probs, y_test)]
    odds = None
    if "home_decimal_odds" in test and "away_decimal_odds" in test:
        odds = [float(h) if p >= 0.5 else float(a) for p, h, a in zip(test_probs, test["home_decimal_odds"], test["away_decimal_odds"])]
    report = evaluate_predictions(evaluation_probabilities, evaluation_outcomes, odds)
    metrics = asdict(report)
    importance = {name: 0.0 for name in FEATURE_NAMES}
    logistic_coef = abs(estimators["logistic"].named_steps["model"].coef_[0])
    component_importance = [logistic_coef, estimators["xgboost"].feature_importances_, estimators["lightgbm"].feature_importances_]
    for values in component_importance:
        total = float(sum(values)) or 1.0
        for name, value in zip(FEATURE_NAMES, values):
            importance[name] += float(value) / total / len(component_importance)
    metrics["feature_importance"] = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    metrics["ensemble_weights"] = weights
    metrics["calibrator"] = selected.name
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)
    artifact_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return bundle, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLB ensemble with chronological holdouts")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ensemble.joblib"))
    args = parser.parse_args()
    bundle, metrics = train_bundle(args.input, args.output)
    print(json.dumps({"artifact": str(args.output), "trained_through": bundle.trained_through, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
