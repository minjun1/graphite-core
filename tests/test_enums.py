"""tests/test_enums.py — Unit tests for graphite.enums."""

import pytest
from graphite.enums import (
    SourceType, EdgeType, NodeType, AssertionMode,
    ConfidenceLevel, EvidenceType,
)


class TestSourceType:
    def test_members_exist(self):
        expected = [
            "SEC_10K", "SEC_20F", "USGS_MCS", "PDF", "WEB", "MANUAL",
            "WEATHER_FORECAST", "EARTH_OBSERVATION", "PUBLIC_REPORT",
            "GEOSPATIAL_DATA",
        ]
        for name in expected:
            assert hasattr(SourceType, name)

    def test_string_roundtrip(self):
        assert SourceType("SEC_10K") == SourceType.SEC_10K
        assert SourceType("WEB") == SourceType.WEB

    def test_is_str(self):
        assert isinstance(SourceType.PDF, str)
        assert SourceType.PDF == "PDF"


class TestEdgeType:
    def test_members_exist(self):
        expected = [
            "PRODUCES", "REFINED_BY", "SUPPLIES_TO", "USED_BY",
            "LOCATED_IN", "DEPENDS_ON", "ADJACENT_TO", "EXPOSED_TO",
            "RISK_FLOWS_TO",
        ]
        for name in expected:
            assert hasattr(EdgeType, name)

    def test_string_roundtrip(self):
        assert EdgeType("PRODUCES") == EdgeType.PRODUCES


class TestNodeType:
    def test_members_exist(self):
        expected = [
            "COMPANY", "COUNTRY", "MINERAL", "REGION",
            "ASSET", "FACILITY", "CORRIDOR",
        ]
        for name in expected:
            assert hasattr(NodeType, name)

    def test_string_roundtrip(self):
        assert NodeType("COMPANY") == NodeType.COMPANY


class TestAssertionMode:
    def test_members_exist(self):
        for name in ["EXTRACTED", "INFERRED", "SEEDED"]:
            assert hasattr(AssertionMode, name)

    def test_string_value(self):
        assert AssertionMode.EXTRACTED == "EXTRACTED"


class TestConfidenceLevel:
    def test_members_exist(self):
        for name in ["HIGH", "MEDIUM", "LOW"]:
            assert hasattr(ConfidenceLevel, name)


class TestEvidenceType:
    def test_members_exist(self):
        for name in ["TEXT_QUOTE", "TABLE_CELL", "DERIVED", "MANUAL"]:
            assert hasattr(EvidenceType, name)

    def test_invalid_member_raises(self):
        with pytest.raises(ValueError):
            SourceType("NONEXISTENT")
