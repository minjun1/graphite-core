"""
graphite/scenarios/weathernext_forecast.py — Convert WeatherNext 2 forecasts to ScenarioShocks.

Transforms weather forecast fields (wind speed, precipitation, pressure)
into Graphite ScenarioShock objects with intensity proportional to
forecast severity.
"""
from typing import Any, Dict, List, Optional

from ..enums import SourceType
from ..scenario import ScenarioShock


# ── Hazard thresholds (based on Saffir-Simpson and NWS standards) ──
DEFAULT_THRESHOLDS = {
    "wind_speed_ms": {
        "tropical_storm": 17.5,     # 39 mph — TS threshold
        "hurricane_cat1": 33.0,     # 74 mph — Cat 1
        "hurricane_cat2": 43.0,     # 96 mph — Cat 2
        "hurricane_cat3": 50.0,     # 111 mph — Cat 3
    },
    "precipitation_mm": {
        "moderate": 50.0,           # 2 inches
        "heavy": 100.0,             # 4 inches
        "extreme": 200.0,           # 8 inches
    },
}


def compute_hazard_intensity(
    fields: Dict[str, float],
    thresholds: Optional[Dict] = None,
) -> float:
    """Compute a hazard intensity score (0.0–1.0) from forecast fields.

    Combines wind speed and precipitation signals. Higher values mean
    more severe hazard.

    Args:
        fields: Forecast fields dict with wind_speed_ms, precipitation_mm, etc.
        thresholds: Override default thresholds

    Returns:
        Intensity score between 0.0 and 1.0
    """
    t = thresholds or DEFAULT_THRESHOLDS

    intensity = 0.0
    components = 0

    # Wind component (0.0–1.0)
    wind = fields.get("wind_speed_ms", 0)
    wind_t = t.get("wind_speed_ms", DEFAULT_THRESHOLDS["wind_speed_ms"])
    if wind >= wind_t["hurricane_cat3"]:
        wind_score = 1.0
    elif wind >= wind_t["hurricane_cat2"]:
        wind_score = 0.9
    elif wind >= wind_t["hurricane_cat1"]:
        wind_score = 0.75
    elif wind >= wind_t["tropical_storm"]:
        wind_score = 0.5
    else:
        wind_score = wind / wind_t["tropical_storm"] * 0.5
    intensity += wind_score
    components += 1

    # Precipitation component (0.0–1.0)
    precip = fields.get("precipitation_mm", 0)
    precip_t = t.get("precipitation_mm", DEFAULT_THRESHOLDS["precipitation_mm"])
    if precip >= precip_t["extreme"]:
        precip_score = 1.0
    elif precip >= precip_t["heavy"]:
        precip_score = 0.75
    elif precip >= precip_t["moderate"]:
        precip_score = 0.5
    else:
        precip_score = precip / precip_t["moderate"] * 0.5
    intensity += precip_score
    components += 1

    return min(1.0, intensity / components) if components > 0 else 0.0


def forecast_to_scenario_shocks(
    forecasts: Dict[str, Dict[str, Any]],
    event_name: str,
    init_time: str,
    intensity_threshold: float = 0.4,
    thresholds: Optional[Dict] = None,
) -> List[ScenarioShock]:
    """Convert WeatherNext 2 forecasts into ScenarioShock objects.

    For each node with forecast data above the intensity threshold,
    creates a ScenarioShock. Multiple nodes can be shocked simultaneously.

    Args:
        forecasts: Dict of node_id → forecast point data
        event_name: Name for the scenario (e.g., "hurricane_beryl_2024")
        init_time: Forecast initialization time (ISO 8601)
        intensity_threshold: Minimum intensity to generate a shock
        thresholds: Override hazard thresholds

    Returns:
        List of ScenarioShock objects
    """
    shocks = []

    # Collect all nodes above threshold
    target_nodes = []
    max_intensity = 0.0
    evidence_parts = []

    for node_id, forecast in forecasts.items():
        fields = forecast.get("fields", {})
        intensity = compute_hazard_intensity(fields, thresholds)

        if intensity >= intensity_threshold:
            target_nodes.append(node_id)
            max_intensity = max(max_intensity, intensity)

            wind_mph = fields.get("wind_speed_ms", 0) * 2.237
            precip_in = fields.get("precipitation_mm", 0) / 25.4
            evidence_parts.append(
                f"{node_id}: wind {wind_mph:.0f}mph, rain {precip_in:.1f}in"
            )

    if target_nodes:
        shocks.append(ScenarioShock(
            shock_id=event_name,
            target_nodes=target_nodes,
            intensity=round(max_intensity, 3),
            observed_at=init_time,
            evidence=(
                f"WeatherNext 2 forecast (experimental, init {init_time}): "
                + "; ".join(evidence_parts[:5])
            ),
            source_type=SourceType.WEATHER_FORECAST,
        ))

    return shocks
