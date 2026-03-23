"""tests/test_init.py — Smoke test for public symbol re-exports."""

import pytest


class TestPublicExports:
    def test_core_schemas(self):
        from graphite import ExtractedEdge, NodeRef, Provenance, InferenceBasis, ExtractionError
        assert all([ExtractedEdge, NodeRef, Provenance, InferenceBasis, ExtractionError])

    def test_enums(self):
        from graphite import EdgeType, NodeType, SourceType, ConfidenceLevel, AssertionMode, EvidenceType
        assert all([EdgeType, NodeType, SourceType, ConfidenceLevel, AssertionMode, EvidenceType])

    def test_evidence(self):
        from graphite import EvidencePacket, EvidenceData
        assert all([EvidencePacket, EvidenceData])

    def test_claim_types(self):
        from graphite import Claim, ClaimType, ClaimStatus, ClaimGranularity, ReviewState, ClaimOrigin
        assert all([Claim, ClaimType, ClaimStatus, ClaimGranularity, ReviewState, ClaimOrigin])

    def test_confidence(self):
        from graphite import ConfidenceFactor, ConfidenceResult, ConfidenceScorer
        assert all([ConfidenceFactor, ConfidenceResult, ConfidenceScorer])

    def test_store(self):
        from graphite import ClaimStore
        assert ClaimStore

    def test_domain(self):
        from graphite import (
            BaseFetcher, BaseExtractor, BasePipeline,
            DocumentContext, DomainSpec,
            register_domain, get_domain, list_domains,
        )
        assert all([BaseFetcher, BaseExtractor, BasePipeline, DocumentContext, DomainSpec])

    def test_rules(self):
        from graphite import BaseRuleEngine, RuleResult, ScoreBreakdown
        assert all([BaseRuleEngine, RuleResult, ScoreBreakdown])

    def test_verdict_types_importable(self):
        """Verdict types re-exported from root package."""
        from graphite import (
            VerdictEnum, ArgumentVerdictEnum, VerdictRationale,
            Verdict, ArgumentVerdict, VerificationReport,
        )
        assert VerdictEnum.SUPPORTED.value == "SUPPORTED"
        assert ArgumentVerdictEnum.GROUNDED.value == "GROUNDED"
        assert VerdictRationale is not None
        assert Verdict is not None
        assert ArgumentVerdict is not None
        assert VerificationReport is not None

    def test_null_handler_attached(self):
        """Graphite logger should have NullHandler for library silence by default."""
        import logging
        logger = logging.getLogger("graphite")
        assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)

    def test_version(self):
        from graphite import __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0
