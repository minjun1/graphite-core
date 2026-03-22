"""tests/test_weathernext.py — Unit tests for graphite.adapters.weathernext."""

import os
import pytest
from graphite.adapters.weathernext import WeatherNextAdapter


FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "weathernext_snapshot.json")


class TestLoadSnapshot:
    def test_load_from_fixture(self):
        adapter = WeatherNextAdapter(snapshot_path=FIXTURE_PATH)
        assert adapter._data is not None
        assert len(adapter._data) == 2

    def test_no_snapshot(self):
        adapter = WeatherNextAdapter()
        assert adapter._data is None


class TestGetForecast:
    def test_known_node(self):
        adapter = WeatherNextAdapter(snapshot_path=FIXTURE_PATH)
        forecast = adapter.get_forecast("asset:PORT_HOUSTON")
        assert forecast is not None
        assert forecast["max_wind_kph"] == 145

    def test_unknown_node(self):
        adapter = WeatherNextAdapter(snapshot_path=FIXTURE_PATH)
        assert adapter.get_forecast("asset:NONEXISTENT") is None


class TestGetAllForecasts:
    def test_returns_all(self):
        adapter = WeatherNextAdapter(snapshot_path=FIXTURE_PATH)
        all_fc = adapter.get_all_forecasts()
        assert len(all_fc) == 2
        assert "asset:PORT_HOUSTON" in all_fc
        assert "facility:EXXON_BAYTOWN" in all_fc

    def test_empty_when_no_snapshot(self):
        adapter = WeatherNextAdapter()
        assert adapter.get_all_forecasts() == {}


class TestListNodes:
    def test_lists_all_ids(self):
        adapter = WeatherNextAdapter(snapshot_path=FIXTURE_PATH)
        nodes = adapter.list_nodes()
        assert set(nodes) == {"asset:PORT_HOUSTON", "facility:EXXON_BAYTOWN"}

    def test_empty_when_no_snapshot(self):
        adapter = WeatherNextAdapter()
        assert adapter.list_nodes() == []


class TestMeta:
    def test_returns_metadata(self):
        adapter = WeatherNextAdapter(snapshot_path=FIXTURE_PATH)
        meta = adapter.meta
        assert meta["model"] == "WeatherNext2"
        assert meta["resolution_deg"] == 0.25

    def test_empty_when_no_snapshot(self):
        adapter = WeatherNextAdapter()
        assert adapter.meta == {}
