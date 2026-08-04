from __future__ import annotations

from typing import Mapping

from .catalog import MARKET


def american_implied(odds: float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    return (-odds) / ((-odds) + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def engineer_market(raw: Mapping[str, float]) -> dict[str, float]:
    out = {name: float(raw[name]) for name in MARKET if name in raw}
    if "home_current_implied_prob" not in out and "home_current_american" in raw:
        out["home_current_implied_prob"] = american_implied(raw["home_current_american"])
    if "away_current_implied_prob" not in out and "away_current_american" in raw:
        out["away_current_implied_prob"] = american_implied(raw["away_current_american"])
    if "home_no_vig_prob" not in out and all(k in out for k in ("home_current_implied_prob", "away_current_implied_prob")):
        total = out["home_current_implied_prob"] + out["away_current_implied_prob"]
        out["home_no_vig_prob"] = out["home_current_implied_prob"] / total
    return out
