"""
graphite/adapters/weathernext.py — Sample-first WeatherNext 2 forecast adapter.

Reads WeatherNext 2 ensemble forecast data for geographic locations.
Primary path: local forecast_snapshot.json (deterministic, no network needed).
Optional (--live): Earth Engine / BigQuery query (requires approved data request form).

WeatherNext 2:
  - 0.25° resolution, 64-member ensemble
  - Fields: temperature, wind, precipitation, humidity, pressure
  - Coverage: 2022-present, 6-hour init times, up to 15-day lead time
  - Access: EE/BigQuery (requires data request form)
  - Note: Experimental dataset, not validated for real-world use
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class WeatherNextAdapter:
    """Read WeatherNext 2 forecasts — sample snapshot first, live optional.

    Primary path: forecast_snapshot.json
      - Deterministic forecast data for demo nodes
      - No network dependency for demos/CI

    Optional path (live=True): Earth Engine / BigQuery
      - Requires approved data request form
      - Not implemented in v1

    Usage:
        adapter = WeatherNextAdapter(snapshot_path="forecast_snapshot.json")
        forecast = adapter.get_forecast("asset:PORT_HOUSTON")
    """

    def __init__(
        self,
        snapshot_path: Optional[str] = None,
        live: bool = False,
    ):
        self.live = live
        self._data = None
        self._snapshot_path = snapshot_path

        if snapshot_path:
            self._load_snapshot(snapshot_path)

    def _load_snapshot(self, path: str):
        """Load forecast data from a snapshot JSON file."""
        with open(path) as f:
            raw = json.load(f)

        self._meta = raw.get("meta", {})
        self._data = {}

        for point in raw.get("forecast_points", []):
            node_id = point.get("node_id", "")
            if node_id:
                self._data[node_id] = point

    @property
    def meta(self) -> Dict[str, Any]:
        """Return forecast metadata."""
        return self._meta if self._meta else {}

    def get_forecast(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return forecast fields for a node.

        Returns None if the node is not in the snapshot.
        """
        if self._data and node_id in self._data:
            return self._data[node_id]

        if self.live:
            return self._fetch_live(node_id)

        return None

    def get_all_forecasts(self) -> Dict[str, Dict[str, Any]]:
        """Return all forecast points from the snapshot."""
        return dict(self._data) if self._data else {}

    def list_nodes(self) -> List[str]:
        """List all node IDs with forecast data."""
        return list(self._data.keys()) if self._data else []

    def _fetch_live(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Fetch live forecast from Earth Engine / BigQuery.

        Not implemented in v1 — requires approved data request form.
        """
        # Stub: would use ee.ImageCollection or BigQuery SQL
        return None
