from __future__ import annotations

import math
from typing import Mapping

from ..contracts import FeatureSnapshot, GameRef
from .catalog import FEATURE_NAMES


class FeatureError(ValueError):
    pass


class FeatureEngine:
    """Assembles only measured values; missing/NaN features are fatal."""

    def build(
        self,
        game: GameRef,
        values: Mapping[str, float],
        provenance: Mapping[str, str],
    ) -> FeatureSnapshot:
        missing = [name for name in FEATURE_NAMES if name not in values]
        missing_provenance = [name for name in FEATURE_NAMES if not provenance.get(name)]
        invalid = []
        normalized: dict[str, float] = {}
        for name in FEATURE_NAMES:
            if name not in values:
                continue
            try:
                value = float(values[name])
            except (TypeError, ValueError):
                invalid.append(name)
                continue
            if not math.isfinite(value):
                invalid.append(name)
            else:
                normalized[name] = value
        if missing or missing_provenance or invalid:
            parts = []
            if missing:
                parts.append(f"missing features: {', '.join(missing[:8])}" + ("..." if len(missing) > 8 else ""))
            if missing_provenance:
                parts.append(f"missing provenance for {len(missing_provenance)} feature(s)")
            if invalid:
                parts.append(f"invalid features: {', '.join(invalid[:8])}")
            raise FeatureError("; ".join(parts))
        return FeatureSnapshot(game, normalized, dict(provenance))
