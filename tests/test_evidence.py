"""tests/test_evidence.py — Unit tests for graphite.evidence."""

import pytest
from graphite.claim import Claim, ConfidenceResult  # noqa: F401 — resolves EvidencePacket forward refs
from graphite.evidence import (
    EvidenceData, ScoringData,
    CounterEvidence, EvidencePacket,
)
from graphite.rules import RuleResult

EvidencePacket.model_rebuild()


class TestEvidenceData:
    def test_construction(self):
        ed = EvidenceData(
            source_entity="company:AAPL",
            target_entity="company:TSLA",
            edge_type="SUPPLIES_TO",
            quote="Apple supplies components to Tesla",
            doc_url="https://example.com/doc.html",
            quote_hash="abc123",
        )
        assert ed.source_entity == "company:AAPL"
        assert ed.edge_type == "SUPPLIES_TO"
        assert ed.quote_type == "exact_quote"
        assert ed.temporal_status == "ACTIVE"

    def test_optional_fields_default(self):
        ed = EvidenceData(
            source_entity="a", target_entity="b", edge_type="X",
            quote="q", doc_url="u", quote_hash="h",
        )
        assert ed.source_id == ""
        assert ed.filing_type == ""
        assert ed.section == ""


class TestRuleResult:
    def test_construction(self):
        rr = RuleResult(
            rule_id="R01",
            rule_name="Sole Source",
            triggered=True,
            weight_delta=0.15,
            explanation="Single supplier detected",
            category="supply_risk",
        )
        assert rr.triggered is True
        assert rr.weight_delta == 0.15


class TestScoringData:
    def test_construction(self):
        sd = ScoringData(
            policy_id="minerals_v1",
            applied_rules=["R01", "R02"],
            calculated_weight=0.75,
            base_score=0.5,
            final_score=0.65,
        )
        assert sd.policy_id == "minerals_v1"
        assert len(sd.applied_rules) == 2

    def test_with_rule_details(self):
        rr = RuleResult(
            rule_id="R01", rule_name="test", triggered=True,
            weight_delta=0.1, explanation="x",
        )
        sd = ScoringData(
            policy_id="p1", applied_rules=["R01"],
            calculated_weight=0.5, rule_details=[rr],
        )
        assert len(sd.rule_details) == 1
        assert sd.rule_details[0].rule_id == "R01"


class TestCounterEvidence:
    def test_construction(self):
        ce = CounterEvidence(
            quote="Tesla has multiple suppliers",
            doc_url="https://example.com",
            impact="WEAKENS_MONOPOLY",
        )
        assert ce.impact == "WEAKENS_MONOPOLY"
        assert ce.source_filing == ""


class TestEvidencePacket:
    def test_construction_minimal(self):
        ep = EvidencePacket(
            claim_hash="hash123",
            status="SUPPORTED",
        )
        assert ep.graphite_version == "1.0"
        assert ep.status == "SUPPORTED"
        assert ep.evidence is None
        assert ep.counter_evidence == []

    def test_construction_full(self):
        ed = EvidenceData(
            source_entity="a", target_entity="b", edge_type="X",
            quote="q", doc_url="u", quote_hash="h",
        )
        ce = CounterEvidence(quote="counter", doc_url="u", impact="WEAKENS")
        ep = EvidencePacket(
            claim_hash="hash",
            status="MIXED_EVIDENCE",
            evidence=ed,
            counter_evidence=[ce],
            verdict_reason="Mixed signals",
        )
        assert ep.evidence.source_entity == "a"
        assert len(ep.counter_evidence) == 1

    def test_serialization_roundtrip(self):
        ep = EvidencePacket(claim_hash="h", status="SUPPORTED")
        data = ep.model_dump()
        restored = EvidencePacket.model_validate(data)
        assert restored.claim_hash == "h"
        assert restored.status == "SUPPORTED"
