"""tests/test_enums.py — Unit tests for graphite.enums."""

import pytest
from graphite.enums import (
    SourceType, EdgeType, NodeType, AssertionMode,
    ConfidenceLevel, EvidenceType,
    ClaimType, ClaimStatus, ReviewState, ClaimOrigin, ClaimGranularity,
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


class TestClaimType:
    def test_members(self):
        assert ClaimType.RELATIONSHIP.value == "RELATIONSHIP"
        assert ClaimType.ATTRIBUTE.value == "ATTRIBUTE"
        assert ClaimType.RISK_ASSERTION.value == "RISK_ASSERTION"
        assert ClaimType.DEPENDENCY.value == "DEPENDENCY"

    def test_string_roundtrip(self):
        assert ClaimType("RELATIONSHIP") == ClaimType.RELATIONSHIP


class TestClaimStatus:
    def test_members(self):
        assert ClaimStatus.SUPPORTED.value == "SUPPORTED"
        assert ClaimStatus.WEAK.value == "WEAK"
        assert ClaimStatus.MIXED.value == "MIXED"
        assert ClaimStatus.UNSUPPORTED.value == "UNSUPPORTED"
        assert ClaimStatus.PENDING_REVIEW.value == "PENDING_REVIEW"

    def test_string_roundtrip(self):
        assert ClaimStatus("SUPPORTED") == ClaimStatus.SUPPORTED


class TestReviewState:
    def test_members(self):
        assert ReviewState.UNREVIEWED.value == "UNREVIEWED"
        assert ReviewState.APPROVED.value == "APPROVED"
        assert ReviewState.REJECTED.value == "REJECTED"
        assert ReviewState.NEEDS_FOLLOWUP.value == "NEEDS_FOLLOWUP"

    def test_string_roundtrip(self):
        assert ReviewState("APPROVED") == ReviewState.APPROVED


class TestClaimOrigin:
    def test_members(self):
        assert ClaimOrigin.EXTRACTOR.value == "EXTRACTOR"
        assert ClaimOrigin.AGENT.value == "AGENT"
        assert ClaimOrigin.RULE_ENGINE.value == "RULE_ENGINE"
        assert ClaimOrigin.ANALYST.value == "ANALYST"
        assert ClaimOrigin.IMPORTED.value == "IMPORTED"

    def test_string_roundtrip(self):
        assert ClaimOrigin("EXTRACTOR") == ClaimOrigin.EXTRACTOR


class TestClaimGranularity:
    def test_members(self):
        assert ClaimGranularity.ATOMIC.value == "ATOMIC"
        assert ClaimGranularity.SYNTHESIZED.value == "SYNTHESIZED"
        assert ClaimGranularity.THESIS.value == "THESIS"

    def test_string_roundtrip(self):
        assert ClaimGranularity("ATOMIC") == ClaimGranularity.ATOMIC
