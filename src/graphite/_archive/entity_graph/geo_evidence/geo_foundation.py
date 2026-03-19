"""
graphite/geo_evidence/geo_foundation.py — Provenance for geospatial foundation model evidence.

Creates proper Provenance entries when attaching foundation model signals
(AlphaEarth embeddings, WeatherNext forecasts) to graph nodes.
"""

from ..enums import ConfidenceLevel, SourceType
from ..schemas import Provenance


def make_alphaearth_provenance(
    lat: float,
    lon: float,
    year: int,
    embedding_dim: int = 64,
    node_id: str = "",
) -> Provenance:
    """Create a Provenance entry for AlphaEarth embedding evidence.

    Args:
        lat: Latitude of the embedding sample point
        lon: Longitude of the embedding sample point
        year: Annual embedding year (2017–2025)
        embedding_dim: Dimensionality (always 64 for v1)
        node_id: Optional graph node ID for reference

    Returns:
        Provenance with EARTH_OBSERVATION source type and ISO 8601 dates
    """
    location_str = f"({lat:.4f}, {lon:.4f})"
    if node_id:
        location_str = f"{node_id} at {location_str}"

    return Provenance(
        source_id=f"alphaearth-v1-{year}-{lat:.4f}-{lon:.4f}",
        source_type=SourceType.EARTH_OBSERVATION,
        evidence_quote=(
            f"AlphaEarth Foundations v1 annual satellite embedding "
            f"({embedding_dim}-dim, multi-sensor: Sentinel-1/2, Landsat, climate) "
            f"for {location_str}, year {year}. "
            f"Source: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
        ),
        confidence=ConfidenceLevel.HIGH,
        observed_at=f"{year}-01-01",
        valid_from=f"{year}-01-01",
        valid_to=f"{year}-12-31",
        snapshot_id=f"alphaearth-v1-{year}",
    )


def make_weathernext_provenance(
    lat: float,
    lon: float,
    init_time: str,
    lead_hours: int = 72,
    node_id: str = "",
) -> Provenance:
    """Create a Provenance entry for WeatherNext forecast evidence.

    Args:
        lat: Latitude
        lon: Longitude
        init_time: Forecast initialization time (ISO 8601)
        lead_hours: Forecast lead time in hours
        node_id: Optional graph node ID

    Returns:
        Provenance with WEATHER_FORECAST source type
    """
    location_str = f"({lat:.4f}, {lon:.4f})"
    if node_id:
        location_str = f"{node_id} at {location_str}"

    return Provenance(
        source_id=f"weathernext2-{init_time}-{lat:.4f}-{lon:.4f}",
        source_type=SourceType.WEATHER_FORECAST,
        evidence_quote=(
            f"WeatherNext 2 ensemble forecast (0.25° resolution, 64 members) "
            f"for {location_str}, init {init_time}, lead {lead_hours}h. "
            f"Source: WeatherNext 2 experimental dataset"
        ),
        confidence=ConfidenceLevel.HIGH,
        observed_at=init_time,
        valid_from=init_time,
        snapshot_id=f"weathernext2-{init_time}",
    )
