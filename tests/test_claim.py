"""
tests/test_claim.py — Unit tests for Claim and related trust engine primitives.

Tests:
  - Claim identity (deterministic claim_id)
  - Claim serialization
  - Evidence linking (supporting + weakening)
  - Claim dependency (depends_on_claim_ids)
  - Status derivation (computed vs final, override)
  - Temporal scope
  - Granularity
  - Review workflow
  - ExtractedEdge claim_ids integration
"""
import pytest

from graphite.claim import (
    Claim,
    ClaimGranularity,
    ClaimOrigin,
    ClaimStatus,
    ClaimType,
    ConfidenceFactor,
    ConfidenceResult,
    ReviewState,
    _make_claim_id,
)
from graphite.enums import AssertionMode, ConfidenceLevel, SourceType, EvidenceType
from graphite.schemas import ExtractedEdge, NodeRef, Provenance


# ═══════════════════════════════════════
# claim_id
# ═══════════════════════════════════════

class TestClaimId:
    def test_deterministic(self):
        id1 = _make_claim_id(["company:NVDA"], "SUPPLIES_TO", ["company:TSLA"])
        id2 = _make_claim_id(["company:NVDA"], "SUPPLIES_TO", ["company:TSLA"])
        assert id1 == id2
        assert len(id1) == 16

    def test_order_invariant_within_group(self):
        id1 = _make_claim_id(["company:AAPL", "company:NVDA"], "SUPPLIES_TO", ["company:TSLA"])
        id2 = _make_claim_id(["company:NVDA", "company:AAPL"], "SUPPLIES_TO", ["company:TSLA"])
        assert id1 == id2

    def test_subject_object_never_mixed(self):
        id_ab = _make_claim_id(["company:AAPL"], "SUPPLIES_TO", ["company:TSLA"])
        id_ba = _make_claim_id(["company:TSLA"], "SUPPLIES_TO", ["company:AAPL"])
        assert id_ab != id_ba

    def test_case_insensitive(self):
        id1 = _make_claim_id(["company:nvda"], "supplies_to", ["company:TSLA"])
        id2 = _make_claim_id(["company:NVDA"], "SUPPLIES_TO", ["company:tsla"])
        assert id1 == id2

    def test_different_predicate_different_id(self):
        id1 = _make_claim_id(["company:NVDA"], "SUPPLIES_TO", ["company:TSLA"])
        id2 = _make_claim_id(["company:NVDA"], "DEPENDS_ON", ["company:TSLA"])
        assert id1 != id2


# ═══════════════════════════════════════
# Claim model
# ═══════════════════════════════════════

def _prov(quote="test quote", source_type=SourceType.SEC_10K):
    return Provenance(
        source_id="acc-001",
        source_type=source_type,
        evidence_quote=quote,
        confidence=ConfidenceLevel.HIGH,
    )


def _claim(**kwargs):
    defaults = dict(
        claim_text="test claim",
        claim_type=ClaimType.RELATIONSHIP,
        subject_entities=["company:A"],
        predicate="SUPPLIES_TO",
        object_entities=["company:B"],
    )
    defaults.update(kwargs)
    return Claim(**defaults)


class TestClaim:
    def test_auto_claim_id(self):
        claim = _claim()
        assert claim.claim_id
        assert len(claim.claim_id) == 16

    def test_explicit_claim_id_preserved(self):
        claim = _claim(claim_id="custom_id_12345")
        assert claim.claim_id == "custom_id_12345"

    def test_serialization_roundtrip(self):
        claim = _claim(supporting_evidence=[_prov()])
        data = claim.model_dump()
        restored = Claim.model_validate(data)
        assert restored.claim_id == claim.claim_id
        assert restored.claim_text == claim.claim_text
        assert len(restored.supporting_evidence) == 1

    def test_supporting_and_weakening(self):
        claim = _claim(
            supporting_evidence=[_prov("supports")],
            weakening_evidence=[_prov("contradicts")],
        )
        assert len(claim.supporting_evidence) == 1
        assert len(claim.weakening_evidence) == 1
        assert claim.evidence_count == 2

    def test_depends_on_claim_ids(self):
        parent = _claim(predicate="SUPPLIES_TO")
        child = _claim(
            predicate="DEPENDS_ON",
            depends_on_claim_ids=[parent.claim_id],
        )
        assert parent.claim_id in child.depends_on_claim_ids

    def test_all_claim_status_values(self):
        for status in ClaimStatus:
            claim = _claim(computed_status=status, final_status=status)
            assert claim.final_status == status

    def test_all_claim_origins(self):
        for origin in ClaimOrigin:
            claim = _claim(origin=origin)
            assert claim.origin == origin


# ═══════════════════════════════════════
# Granularity
# ═══════════════════════════════════════

class TestGranularity:
    def test_default_atomic(self):
        claim = _claim()
        assert claim.granularity == ClaimGranularity.ATOMIC

    def test_thesis_level(self):
        claim = _claim(granularity=ClaimGranularity.THESIS)
        assert claim.granularity == ClaimGranularity.THESIS

    def test_synthesized(self):
        claim = _claim(granularity=ClaimGranularity.SYNTHESIZED)
        assert claim.granularity == ClaimGranularity.SYNTHESIZED


# ═══════════════════════════════════════
# Temporal scope
# ═══════════════════════════════════════

class TestTemporalScope:
    def test_defaults_empty(self):
        claim = _claim()
        assert claim.as_of_date == ""
        assert claim.valid_from == ""
        assert claim.valid_to == ""

    def test_fiscal_year(self):
        claim = _claim(as_of_date="FY2024", valid_from="2024-01-01", valid_to="2024-12-31")
        assert claim.as_of_date == "FY2024"
        assert claim.valid_from == "2024-01-01"
        assert claim.valid_to == "2024-12-31"

    def test_open_ended(self):
        """valid_to empty = still valid."""
        claim = _claim(as_of_date="2025-03-15", valid_from="2025-03-15")
        assert claim.valid_to == ""

    def test_serialization_preserves_temporal(self):
        claim = _claim(as_of_date="Q1-2025", valid_from="2025-01-01")
        data = claim.model_dump()
        restored = Claim.model_validate(data)
        assert restored.as_of_date == "Q1-2025"
        assert restored.valid_from == "2025-01-01"


# ═══════════════════════════════════════
# computed_status / final_status / override
# ═══════════════════════════════════════

class TestStatusSplit:
    def test_defaults_pending(self):
        claim = _claim()
        assert claim.computed_status == ClaimStatus.PENDING_REVIEW
        assert claim.final_status == ClaimStatus.PENDING_REVIEW
        assert claim.override_reason == ""

    def test_compute_no_evidence_unsupported(self):
        claim = _claim()
        result = claim.compute_status()
        assert result == ClaimStatus.UNSUPPORTED
        assert claim.computed_status == ClaimStatus.UNSUPPORTED
        assert claim.final_status == ClaimStatus.UNSUPPORTED  # no override

    def test_compute_high_confidence_supported(self):
        claim = _claim(
            supporting_evidence=[_prov()],
            confidence=ConfidenceResult.from_score(0.8),
        )
        result = claim.compute_status()
        assert result == ClaimStatus.SUPPORTED
        assert claim.computed_status == ClaimStatus.SUPPORTED
        assert claim.final_status == ClaimStatus.SUPPORTED

    def test_compute_low_confidence(self):
        claim = _claim(
            supporting_evidence=[_prov()],
            confidence=ConfidenceResult.from_score(0.2),
        )
        assert claim.compute_status() == ClaimStatus.UNSUPPORTED

    def test_compute_mixed_evidence(self):
        claim = _claim(
            supporting_evidence=[_prov("s")],
            weakening_evidence=[_prov("w")],
        )
        assert claim.compute_status() == ClaimStatus.MIXED

    def test_no_confidence_pending(self):
        claim = _claim(supporting_evidence=[_prov()])
        assert claim.compute_status() == ClaimStatus.PENDING_REVIEW

    def test_override_preserves_computed(self):
        """Analyst override changes final_status but not computed_status."""
        claim = _claim(
            supporting_evidence=[_prov()],
            confidence=ConfidenceResult.from_score(0.3),
        )
        claim.compute_status()
        assert claim.computed_status == ClaimStatus.WEAK

        claim.override_status(
            ClaimStatus.SUPPORTED,
            reason="Analyst confirmed via direct call with supplier",
            reviewer="analyst_1",
        )
        assert claim.computed_status == ClaimStatus.WEAK  # unchanged
        assert claim.final_status == ClaimStatus.SUPPORTED  # overridden
        assert claim.override_reason == "Analyst confirmed via direct call with supplier"
        assert claim.reviewed_by == "analyst_1"
        assert claim.review_state == ReviewState.APPROVED

    def test_override_rejected(self):
        claim = _claim(
            supporting_evidence=[_prov()],
            confidence=ConfidenceResult.from_score(0.8),
        )
        claim.compute_status()
        assert claim.final_status == ClaimStatus.SUPPORTED

        claim.override_status(ClaimStatus.UNSUPPORTED, reason="Evidence is outdated")
        assert claim.final_status == ClaimStatus.UNSUPPORTED
        assert claim.review_state == ReviewState.REJECTED

    def test_recompute_does_not_overwrite_override(self):
        """After analyst override, recompute updates computed but not final."""
        claim = _claim(
            supporting_evidence=[_prov()],
            confidence=ConfidenceResult.from_score(0.8),
        )
        claim.compute_status()
        claim.override_status(ClaimStatus.WEAK, reason="Needs more data")

        # Recompute — should not touch final_status
        claim.compute_status()
        assert claim.computed_status == ClaimStatus.SUPPORTED
        assert claim.final_status == ClaimStatus.WEAK  # override preserved


# ═══════════════════════════════════════
# Review workflow
# ═══════════════════════════════════════

class TestReviewWorkflow:
    def test_default_unreviewed(self):
        claim = _claim()
        assert claim.review_state == ReviewState.UNREVIEWED
        assert not claim.is_reviewed

    def test_override_sets_review(self):
        claim = _claim()
        claim.override_status(ClaimStatus.SUPPORTED, "Good", "alice")
        assert claim.is_reviewed
        assert claim.review_state == ReviewState.APPROVED
        assert claim.reviewed_at is not None


# ═══════════════════════════════════════
# ExtractedEdge claim_ids
# ═══════════════════════════════════════

class TestEdgeClaimIds:
    def test_default_empty(self):
        edge = ExtractedEdge(
            from_node=NodeRef.company("NVDA"),
            to_node=NodeRef.company("TSLA"),
            edge_type="SUPPLIES_TO",
            assertion_mode=AssertionMode.EXTRACTED,
        )
        assert edge.claim_ids == []

    def test_with_claim_ids(self):
        edge = ExtractedEdge(
            from_node=NodeRef.company("NVDA"),
            to_node=NodeRef.company("TSLA"),
            edge_type="SUPPLIES_TO",
            assertion_mode=AssertionMode.EXTRACTED,
            claim_ids=["abc123", "def456"],
        )
        assert edge.claim_ids == ["abc123", "def456"]


# ═══════════════════════════════════════
# ConfidenceResult
# ═══════════════════════════════════════

class TestConfidenceResult:
    def test_from_score_high(self):
        r = ConfidenceResult.from_score(0.85)
        assert r.level == ConfidenceLevel.HIGH

    def test_from_score_medium(self):
        r = ConfidenceResult.from_score(0.55)
        assert r.level == ConfidenceLevel.MEDIUM

    def test_from_score_low(self):
        r = ConfidenceResult.from_score(0.2)
        assert r.level == ConfidenceLevel.LOW

    def test_clamps_to_bounds(self):
        assert ConfidenceResult.from_score(1.5).score == 1.0
        assert ConfidenceResult.from_score(-0.5).score == 0.0

    def test_with_factors(self):
        factors = [ConfidenceFactor(
            name="source_count",
            raw_value="3 sources",
            contribution=0.15,
            direction="POSITIVE",
            explanation="3 sources support this",
        )]
        r = ConfidenceResult.from_score(0.73, factors)
        assert len(r.factors) == 1
        assert r.factors[0].raw_value == "3 sources"
