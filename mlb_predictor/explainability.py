from __future__ import annotations

from dataclasses import dataclass

from .contracts import FeatureSnapshot
from .features.catalog import FEATURE_CATEGORIES
from .models.ensemble import EnsembleBundle


@dataclass(frozen=True)
class Explanation:
    category_points: dict[str, float]
    final_probability: float


def explain_prediction(bundle: EnsembleBundle, snapshot: FeatureSnapshot) -> Explanation:
    row = snapshot.ordered(bundle.feature_names)
    final = bundle.predict([row])[0].home_probability
    contributions: dict[str, float] = {}
    for category, feature_names in FEATURE_CATEGORIES.items():
        counterfactual = list(row)
        for name in feature_names:
            idx = bundle.feature_names.index(name)
            counterfactual[idx] = bundle.feature_means[idx]
        without_category = bundle.predict([counterfactual])[0].home_probability
        contributions[category] = (final - without_category) * 100.0
    return Explanation(contributions, final)
