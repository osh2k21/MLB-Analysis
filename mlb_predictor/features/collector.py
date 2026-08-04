from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Mapping

from ..contracts import DataStatus, Evidence, FeatureSnapshot, GameRef
from .engine import FeatureEngine
from .pitching import engineer_pitching
from .batting import engineer_batting
from .bullpen import engineer_bullpen
from .defense import engineer_defense
from .weather import engineer_context
from .market import engineer_market


def scope_game_evidence(evidence: Evidence, game_pk: int, section: str | None = None) -> Evidence:
    """Select one game's payload without changing the provider timestamp."""
    if not evidence.usable or not isinstance(evidence.payload, dict):
        return evidence
    games = evidence.payload.get("games")
    if not isinstance(games, dict):
        return evidence
    entry = games.get(str(game_pk)) or games.get(game_pk)
    if section and isinstance(entry, dict):
        entry = entry.get(section)
    if entry is None:
        return Evidence(evidence.source, DataStatus.MISSING, evidence.observed_at, detail=f"game {game_pk} absent from feed", fetched_at=evidence.fetched_at, cache_hit=evidence.cache_hit)
    return Evidence(evidence.source, DataStatus.OK, evidence.observed_at, entry, fetched_at=evidence.fetched_at, cache_hit=evidence.cache_hit)


class FeatureCollector:
    """Merge provider-normalized features while preserving per-value provenance."""

    def __init__(self, engine: FeatureEngine | None = None) -> None:
        self.engine = engine or FeatureEngine()

    def collect(self, game: GameRef, evidence_items: Iterable[Evidence]) -> FeatureSnapshot:
        values: dict[str, float] = {}
        provenance: dict[str, str] = {}
        for evidence in evidence_items:
            if not evidence.usable or not isinstance(evidence.payload, dict):
                continue
            payload = evidence.payload
            games = payload.get("games")
            if isinstance(games, dict):
                payload = games.get(str(game.game_pk)) or games.get(game.game_pk) or {}
            features = payload.get("features", payload) if isinstance(payload, dict) else {}
            if not isinstance(features, dict):
                continue
            for name, raw in features.items():
                if isinstance(raw, dict) and "value" in raw:
                    value = raw["value"]
                    source = raw.get("source") or evidence.source
                else:
                    value, source = raw, evidence.source
                # Conflicts are fatal unless values match exactly; provider order cannot silently decide truth.
                if name in values and float(values[name]) != float(value):
                    raise ValueError(f"conflicting feature {name} from {provenance[name]} and {source}")
                values[name], provenance[name] = value, f"{source} @ {evidence.observed_at.isoformat()}"
            metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
            if isinstance(metrics, dict):
                builders = {
                    "pitching": lambda block: engineer_pitching(block.get("home", {}), block.get("away", {})),
                    "lineup": lambda block: engineer_batting(block.get("home", {}), block.get("away", {})),
                    "bullpen": lambda block: engineer_bullpen(block.get("home", {}), block.get("away", {})),
                    "defense": lambda block: engineer_defense(block.get("home", {}), block.get("away", {})),
                    "context": engineer_context,
                    "market": engineer_market,
                }
                for category, builder in builders.items():
                    block = metrics.get(category)
                    if not isinstance(block, dict):
                        continue
                    for name, value in builder(block).items():
                        if name in values and float(values[name]) != float(value):
                            raise ValueError(f"conflicting feature {name} from {provenance[name]} and {evidence.source}")
                        values[name] = value
                        provenance[name] = f"engineered from {evidence.source} @ {evidence.observed_at.isoformat()}"
        return self.engine.build(game, values, provenance)
