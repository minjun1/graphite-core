"""tests/test_schemas.py — Unit tests for graphite.schemas."""

import pytest
from graphite.schemas import (
    NodeRef, Provenance, InferenceBasis, ExtractedEdge, ExtractionError,
)
from graphite.enums import (
    NodeType, SourceType, EvidenceType, ConfidenceLevel, AssertionMode,
)


class TestNodeRefFactories:
    def test_company(self):
        n = NodeRef.company("AAPL", label="Apple Inc.")
        assert n.node_id == "company:AAPL"
        assert n.node_type == NodeType.COMPANY
        assert n.label == "Apple Inc."

    def test_company_uppercases(self):
        n = NodeRef.company("aapl")
        assert n.node_id == "company:AAPL"

    def test_country(self):
        n = NodeRef.country("CD", label="Congo")
        assert n.node_id == "country:CD"
        assert n.node_type == NodeType.COUNTRY

    def test_mineral(self):
        n = NodeRef.mineral("cobalt")
        assert n.node_id == "mineral:COBALT"
        assert n.node_type == NodeType.MINERAL

    def test_region(self):
        n = NodeRef.region("houston_metro")
        assert n.node_id == "region:HOUSTON_METRO"
        assert n.node_type == NodeType.REGION

    def test_asset(self):
        n = NodeRef.asset("port_houston")
        assert n.node_id == "asset:PORT_HOUSTON"
        assert n.node_type == NodeType.ASSET

    def test_facility(self):
        n = NodeRef.facility("exxon_baytown")
        assert n.node_id == "facility:EXXON_BAYTOWN"
        assert n.node_type == NodeType.FACILITY

    def test_corridor(self):
        n = NodeRef.corridor("ship_channel")
        assert n.node_id == "corridor:SHIP_CHANNEL"
        assert n.node_type == NodeType.CORRIDOR


class TestProvenance:
    def test_construction(self):
        p = Provenance(
            source_id="acc-001",
            source_type=SourceType.SEC_10K,
            evidence_quote="Apple is a customer.",
            confidence=ConfidenceLevel.HIGH,
        )
        assert p.source_id == "acc-001"
        assert p.source_type == SourceType.SEC_10K
        assert p.confidence == ConfidenceLevel.HIGH
        assert p.evidence_type == EvidenceType.TEXT_QUOTE

    def test_defaults(self):
        p = Provenance(
            source_id="x",
            source_type=SourceType.WEB,
            evidence_quote="quote",
        )
        assert p.source_url == ""
        assert p.paragraph_index == -1
        assert p.extractor_version == "1.0"


class TestInferenceBasis:
    def test_construction(self):
        ib = InferenceBasis(
            method="customer_filing_reverse",
            reason="Inferred from TSLA 10-K",
            based_on_edges=["e1", "e2"],
            source_nodes=["company:TSLA"],
        )
        assert ib.method == "customer_filing_reverse"
        assert len(ib.based_on_edges) == 2
        assert ib.source_nodes == ["company:TSLA"]


class TestExtractedEdge:
    def _make_edge(self, provenance_levels=None):
        provs = []
        for level in (provenance_levels or [ConfidenceLevel.MEDIUM]):
            provs.append(Provenance(
                source_id=f"src-{level.value}",
                source_type=SourceType.SEC_10K,
                evidence_quote="quote",
                confidence=level,
            ))
        return ExtractedEdge(
            from_node=NodeRef.company("AAPL"),
            to_node=NodeRef.company("TSLA"),
            edge_type="SUPPLIES_TO",
            assertion_mode=AssertionMode.EXTRACTED,
            provenance=provs,
        )

    def test_edge_key_deterministic(self):
        e = self._make_edge()
        assert e.edge_key == "company:AAPL|company:TSLA|SUPPLIES_TO"

    def test_best_confidence_single(self):
        e = self._make_edge([ConfidenceLevel.LOW])
        assert e.best_confidence == ConfidenceLevel.LOW

    def test_best_confidence_multiple(self):
        e = self._make_edge([ConfidenceLevel.LOW, ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM])
        assert e.best_confidence == ConfidenceLevel.HIGH

    def test_best_confidence_no_provenance(self):
        e = ExtractedEdge(
            from_node=NodeRef.company("AAPL"),
            to_node=NodeRef.company("TSLA"),
            edge_type="SUPPLIES_TO",
            assertion_mode=AssertionMode.EXTRACTED,
        )
        assert e.best_confidence == ConfidenceLevel.LOW


class TestExtractionError:
    def test_construction(self):
        err = ExtractionError(
            entity_id="AAPL",
            source_type=SourceType.SEC_10K,
            error_type="parse_failed",
            message="Could not parse HTML",
        )
        assert err.entity_id == "AAPL"
        assert err.error_type == "parse_failed"
        assert err.timestamp
