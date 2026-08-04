from __future__ import annotations

import math
from typing import Mapping

from .catalog import CONTEXT


def air_density_kg_m3(temperature_f: float, humidity_pct: float, pressure_hpa: float) -> float:
    """Moist-air density from measured temperature, humidity and pressure."""
    temp_c = (float(temperature_f) - 32.0) * 5.0 / 9.0
    temp_k = temp_c + 273.15
    saturation_hpa = 6.1078 * 10 ** ((7.5 * temp_c) / (temp_c + 237.3))
    vapor_pa = max(0.0, min(100.0, float(humidity_pct))) / 100.0 * saturation_hpa * 100.0
    dry_pa = float(pressure_hpa) * 100.0 - vapor_pa
    return dry_pa / (287.058 * temp_k) + vapor_pa / (461.495 * temp_k)


def wind_out_component(speed_mph: float, wind_direction_degrees: float, center_field_bearing_degrees: float) -> float:
    angle = math.radians(float(wind_direction_degrees) - float(center_field_bearing_degrees))
    return float(speed_mph) * math.cos(angle)


def engineer_context(raw: Mapping[str, float]) -> dict[str, float]:
    out = {name: float(raw[name]) for name in CONTEXT if name in raw}
    if "air_density_kg_m3" not in out and all(k in raw for k in ("temperature_f", "humidity_pct", "pressure_hpa")):
        out["air_density_kg_m3"] = air_density_kg_m3(raw["temperature_f"], raw["humidity_pct"], raw["pressure_hpa"])
    if "wind_out_component_mph" not in out and all(k in raw for k in ("wind_speed_mph", "wind_direction_degrees", "center_field_bearing_degrees")):
        out["wind_out_component_mph"] = wind_out_component(raw["wind_speed_mph"], raw["wind_direction_degrees"], raw["center_field_bearing_degrees"])
    return out

