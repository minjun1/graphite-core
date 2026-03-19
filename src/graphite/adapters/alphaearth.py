"""
graphite/adapters/alphaearth.py — Cache-first AlphaEarth embedding adapter.

Reads 64-dimensional AlphaEarth Foundations embeddings for geographic locations.
Primary path: local .npy cache files (deterministic, no network needed).
Optional: GCS COGs at gs://alphaearth_foundations/ (Requester Pays).

AlphaEarth Foundations:
  - 64-dim embeddings at ~10m/pixel resolution
  - Annual layers: 2017–2025
  - Multi-sensor: Sentinel-1/2, Landsat, climate sims, 3D laser
  - Earth Engine dataset: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
  - GCS bucket: gs://alphaearth_foundations (Requester Pays)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Constants ──
EMBEDDING_DIM = 64
GCS_BUCKET = "gs://alphaearth_foundations"
EE_DATASET = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
AVAILABLE_YEARS = list(range(2017, 2026))


class AlphaEarthAdapter:
    """Read AlphaEarth embeddings — local cache first, GCS optional.

    Primary path: cache/{year}/{node_id}.npy
      - Pre-fetched embeddings stored as numpy arrays
      - Deterministic: same result every run
      - No network dependency for demos/CI

    Optional path: GCS COGs (Requester Pays)
      - Requires billing_project to be set
      - Reads Cloud Optimized GeoTIFFs via rasterio
      - Results cached locally after first fetch

    Usage:
        adapter = AlphaEarthAdapter(cache_dir="cache/alphaearth")

        # From cache (fast, deterministic)
        emb = adapter.get_embedding(29.7355, -95.2690, year=2017)

        # With GCS fallback (requires rasterio + billing project)
        adapter = AlphaEarthAdapter(
            cache_dir="cache/alphaearth",
            billing_project="my-gcp-project",
        )
        emb = adapter.get_embedding(29.7355, -95.2690, year=2017)
    """

    def __init__(
        self,
        cache_dir: str = "cache/alphaearth",
        billing_project: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.billing_project = billing_project

    def get_embedding(
        self,
        lat: float,
        lon: float,
        year: int = 2017,
        node_id: Optional[str] = None,
    ) -> np.ndarray:
        """Return 64-dim embedding for a point location.

        Tries cache first (by node_id or lat/lon key), then GCS if configured.

        Args:
            lat: Latitude
            lon: Longitude
            year: Annual embedding year (2017–2025)
            node_id: Optional cache key (e.g., "PORT_HOUSTON")

        Returns:
            numpy array of shape (64,)
        """
        cache_key = node_id or f"{lat:.4f}_{lon:.4f}"

        # 1. Try cache
        cached = self._read_cache(cache_key, year)
        if cached is not None:
            return cached

        # 2. Try GCS (optional)
        if self.billing_project:
            embedding = self._fetch_from_gcs(lat, lon, year)
            if embedding is not None:
                self._write_cache(cache_key, year, embedding)
                return embedding

        # 3. No data available
        raise FileNotFoundError(
            f"No AlphaEarth embedding found for {cache_key} (year={year}). "
            f"Either pre-fetch to {self.cache_dir}/{year}/{cache_key}.npy "
            f"or set billing_project for GCS access."
        )

    def get_area_embedding(
        self,
        bbox: Tuple[float, float, float, float],
        year: int = 2017,
        node_id: Optional[str] = None,
    ) -> np.ndarray:
        """Return mean 64-dim embedding for a bounding box area.

        For cache mode, this falls back to get_embedding with the bbox centroid.
        For GCS mode, this would read and average the full bbox region.

        Args:
            bbox: (min_lat, min_lon, max_lat, max_lon)
            year: Annual embedding year
            node_id: Optional cache key

        Returns:
            numpy array of shape (64,)
        """
        cache_key = (
            node_id or f"bbox_{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}"
        )

        cached = self._read_cache(cache_key, year)
        if cached is not None:
            return cached

        # Fallback: centroid point embedding
        centroid_lat = (bbox[0] + bbox[2]) / 2
        centroid_lon = (bbox[1] + bbox[3]) / 2
        return self.get_embedding(centroid_lat, centroid_lon, year, node_id=cache_key)

    def get_embedding_safe(
        self,
        lat: float,
        lon: float,
        year: int = 2017,
        node_id: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Like get_embedding but returns None instead of raising."""
        try:
            return self.get_embedding(lat, lon, year, node_id)
        except FileNotFoundError:
            return None

    # ── Cache I/O ──

    def _cache_path(self, key: str, year: int) -> Path:
        return self.cache_dir / str(year) / f"{key}.npy"

    def _read_cache(self, key: str, year: int) -> Optional[np.ndarray]:
        path = self._cache_path(key, year)
        if path.exists():
            arr = np.load(path)
            if arr.shape == (EMBEDDING_DIM,):
                return arr
        return None

    def _write_cache(self, key: str, year: int, embedding: np.ndarray):
        path = self._cache_path(key, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embedding)

    # ── GCS fetch (optional, requires rasterio) ──

    def _fetch_from_gcs(
        self, lat: float, lon: float, year: int
    ) -> Optional[np.ndarray]:
        """Fetch embedding from GCS Cloud Optimized GeoTIFF.

        Requires:
          - rasterio package installed
          - GCP billing project with access to requester-pays buckets
        """
        try:
            import rasterio
            from rasterio.crs import CRS
        except ImportError:
            return None

        # GCS COGs are organized by year and UTM zone
        # For now, return None — actual implementation would:
        # 1. Determine UTM zone from lat/lon
        # 2. Construct GCS path: gs://alphaearth_foundations/{year}/{zone}/...
        # 3. Open COG with rasterio using GDAL_HTTP_HEADER_AUTH
        # 4. Sample at lat/lon → 64-dim array
        #
        # This is left as a stub because:
        # - GCS bucket is Requester Pays (needs billing project)
        # - Demo should work from cache without network
        # - Full implementation needs GDAL/rasterio + GCP auth setup
        return None

    # ── Batch operations ──

    def list_cached(self, year: int = 2017) -> List[str]:
        """List all cached embedding keys for a year."""
        year_dir = self.cache_dir / str(year)
        if not year_dir.exists():
            return []
        return [p.stem for p in year_dir.glob("*.npy")]

    def cache_stats(self) -> Dict[str, int]:
        """Return count of cached embeddings per year."""
        stats = {}
        for year in AVAILABLE_YEARS:
            count = len(self.list_cached(year))
            if count > 0:
                stats[str(year)] = count
        return stats
