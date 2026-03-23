"""tests/test_report.py — Unit tests for graphite.pipeline.report (mocked)."""

import pytest
from unittest.mock import patch, MagicMock

from graphite.pipeline.report import verify_agent_output, review_document
from graphite.claim import Claim, ClaimType, ClaimOrigin, ClaimGranularity
from graphite.pipeline.verdict import (
    Verdict, VerdictEnum, VerdictRationale,
    ArgumentVerdict, ArgumentVerdictEnum,
    VerificationReport,
)
from graphite.enums import AssertionMode


def _make_claim(text="test"):
    return Claim(
        claim_text=text,
        claim_type=ClaimType.RELATIONSHIP,
        subject_entities=["A"],
        predicate="P",
        object_entities=["B"],
        assertion_mode=AssertionMode.EXTRACTED,
        origin=ClaimOrigin.EXTRACTOR,
        granularity=ClaimGranularity.ATOMIC,
    )


def _make_verdict(claim, verdict_enum=VerdictEnum.SUPPORTED, needs_review=False):
    return Verdict(
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        verdict=verdict_enum,
        supporting_evidence_ids=[],
        conflicting_evidence_ids=[],
        rationale=VerdictRationale(text="rationale"),
        needs_human_review=needs_review,
        model_version="test",
        timestamp="2026-01-01T00:00:00Z",
    )


def _make_arg_verdict(verdict_enum=ArgumentVerdictEnum.GROUNDED):
    return ArgumentVerdict(
        text="conclusion",
        verdict=verdict_enum,
        rationale=VerdictRationale(text="ok"),
        needs_human_review=False,
    )


class TestVerifyAgentOutput:
    @patch("graphite.pipeline.report.ArgumentAnalyzer")
    @patch("graphite.pipeline.report.ClaimVerifier")
    @patch("graphite.pipeline.report.retrieve_evidence")
    @patch("graphite.pipeline.report.ClaimExtractor")
    def test_orchestrates_pipeline(self, MockExtractor, mock_retrieve, MockVerifier, MockAnalyzer):
        claim = _make_claim("Apple supplies Tesla")
        MockExtractor.return_value.extract_claims.return_value = [claim]
        mock_retrieve.return_value = {claim.claim_id: [{"text": "ev", "document_id": "d1"}]}
        MockVerifier.return_value.verify_claims.return_value = [_make_verdict(claim, VerdictEnum.SUPPORTED)]
        MockAnalyzer.return_value.analyze_argument_chain.return_value = [_make_arg_verdict(ArgumentVerdictEnum.GROUNDED)]

        corpus = [{"document_id": "d1", "text": "Apple supplies Tesla with chips."}]
        report = verify_agent_output("Apple supplies Tesla", corpus, api_key="test-key")

        assert isinstance(report, VerificationReport)
        assert report.total_claims == 1
        assert report.supported_count == 1
        assert report.conflicted_count == 0
        assert report.insufficient_count == 0
        assert report.grounded_argument_count == 1
        assert report.evidence_coverage_score == 1.0

    @patch("graphite.pipeline.report.ArgumentAnalyzer")
    @patch("graphite.pipeline.report.ClaimVerifier")
    @patch("graphite.pipeline.report.retrieve_evidence")
    @patch("graphite.pipeline.report.ClaimExtractor")
    def test_conflicted_claim_in_risky(self, MockExtractor, mock_retrieve, MockVerifier, MockAnalyzer):
        claim = _make_claim()
        MockExtractor.return_value.extract_claims.return_value = [claim]
        mock_retrieve.return_value = {claim.claim_id: []}
        MockVerifier.return_value.verify_claims.return_value = [_make_verdict(claim, VerdictEnum.CONFLICTED)]
        MockAnalyzer.return_value.analyze_argument_chain.return_value = []

        report = verify_agent_output("text", [{"document_id": "d1", "text": "x"}], api_key="test-key")
        assert report.conflicted_count == 1
        assert claim.claim_id in report.risky_claim_ids

    @patch("graphite.pipeline.report.ArgumentAnalyzer")
    @patch("graphite.pipeline.report.ClaimVerifier")
    @patch("graphite.pipeline.report.retrieve_evidence")
    @patch("graphite.pipeline.report.ClaimExtractor")
    def test_conclusion_jump_counted(self, MockExtractor, mock_retrieve, MockVerifier, MockAnalyzer):
        claim = _make_claim()
        MockExtractor.return_value.extract_claims.return_value = [claim]
        mock_retrieve.return_value = {}
        MockVerifier.return_value.verify_claims.return_value = [_make_verdict(claim)]
        MockAnalyzer.return_value.analyze_argument_chain.return_value = [_make_arg_verdict(ArgumentVerdictEnum.CONCLUSION_JUMP)]

        report = verify_agent_output("text", [], api_key="test-key")
        assert report.conclusion_jump_count == 1


class TestReviewDocumentAlias:
    def test_is_same_function(self):
        assert review_document is verify_agent_output


class TestVerificationReportGetVerdict:
    def test_get_existing_verdict(self):
        claim = _make_claim()
        v = _make_verdict(claim)
        report = VerificationReport(
            document_id="doc1",
            total_claims=1,
            verdicts=[v],
            argument_verdicts=[],
            model_metadata={},
        )
        result = report.get_verdict(claim.claim_id)
        assert result is not None
        assert result.claim_id == claim.claim_id

    def test_get_missing_verdict(self):
        report = VerificationReport(
            document_id="doc1",
            total_claims=0,
            verdicts=[],
            argument_verdicts=[],
            model_metadata={},
        )
        assert report.get_verdict("nonexistent") is None
