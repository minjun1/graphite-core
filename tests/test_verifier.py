"""tests/test_verifier.py — Unit tests for graphite.pipeline.verifier (mocked)."""

import pytest
from unittest.mock import patch, MagicMock

from graphite.pipeline.verifier import ClaimVerifier, verify_claims
from graphite.claim import Claim, ClaimType, ClaimOrigin, ClaimGranularity
from graphite.pipeline.verdict import VerdictEnum
from graphite.enums import AssertionMode


def _make_claim(claim_text="Test claim", subjects=None, objects=None):
    return Claim(
        claim_text=claim_text,
        claim_type=ClaimType.RELATIONSHIP,
        subject_entities=subjects or ["company:AAPL"],
        predicate="SUPPLIES_TO",
        object_entities=objects or ["company:TSLA"],
        assertion_mode=AssertionMode.EXTRACTED,
        origin=ClaimOrigin.EXTRACTOR,
        granularity=ClaimGranularity.ATOMIC,
    )


def _mock_verifier_json(verdict="SUPPORTED", rationale="Evidence supports"):
    return {
        "verdict": verdict,
        "rationale_text": rationale,
        "contradiction_type": None,
        "missing_evidence_reason": None,
        "temporal_alignment": None,
        "needs_human_review": False,
        "cited_span": "relevant quote",
        "supporting_evidence_indices": [0] if verdict == "SUPPORTED" else [],
        "conflicting_evidence_indices": [0] if verdict == "CONFLICTED" else [],
    }


class TestClaimVerifier:
    @patch("graphite.pipeline._client.create_llm_client")
    def test_supported_verdict(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat_json.return_value = _mock_verifier_json("SUPPORTED")

        verifier = ClaimVerifier(api_key="test-key")
        claim = _make_claim()
        evidence_map = {claim.claim_id: [{"text": "Apple supplies Tesla", "document_id": "doc1"}]}
        verdicts = verifier.verify_claims([claim], evidence_map)

        assert len(verdicts) == 1
        assert verdicts[0].verdict == VerdictEnum.SUPPORTED
        assert verdicts[0].claim_id == claim.claim_id

    @patch("graphite.pipeline._client.create_llm_client")
    def test_conflicted_verdict(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat_json.return_value = _mock_verifier_json("CONFLICTED")

        verifier = ClaimVerifier(api_key="test-key")
        claim = _make_claim()
        evidence_map = {claim.claim_id: [{"text": "Contradictory evidence", "document_id": "doc1"}]}
        verdicts = verifier.verify_claims([claim], evidence_map)

        assert verdicts[0].verdict == VerdictEnum.CONFLICTED

    @patch("graphite.pipeline._client.create_llm_client")
    def test_insufficient_verdict(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat_json.return_value = _mock_verifier_json("INSUFFICIENT")

        verifier = ClaimVerifier(api_key="test-key")
        claim = _make_claim()
        evidence_map = {claim.claim_id: []}
        verdicts = verifier.verify_claims([claim], evidence_map)

        assert verdicts[0].verdict == VerdictEnum.INSUFFICIENT

    @patch("graphite.pipeline._client.create_llm_client")
    def test_empty_evidence_map(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat_json.return_value = _mock_verifier_json("INSUFFICIENT")

        verifier = ClaimVerifier(api_key="test-key")
        claim = _make_claim()
        verdicts = verifier.verify_claims([claim], {})

        assert len(verdicts) == 1
        assert verdicts[0].verdict == VerdictEnum.INSUFFICIENT


class TestVerifyClaimsConvenience:
    @patch("graphite.pipeline._client.create_llm_client")
    def test_convenience_function(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat_json.return_value = _mock_verifier_json("SUPPORTED")

        claim = _make_claim()
        evidence_map = {claim.claim_id: [{"text": "evidence", "document_id": "doc1"}]}
        verdicts = verify_claims([claim], evidence_map, api_key="test-key")
        assert len(verdicts) == 1


class TestVerifyEmptyInput:
    @patch("graphite.pipeline._client.create_llm_client")
    def test_empty_claims_list_returns_empty_verdicts(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        verifier = ClaimVerifier(api_key="test-key")
        result = verifier.verify_claims([], {})
        assert result == []
        mock_client.chat_json.assert_not_called()


class TestVerifyUnexpectedVerdictString:
    @patch("graphite.pipeline._client.create_llm_client")
    def test_unexpected_verdict_string_defaults_to_insufficient(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat_json.return_value = _mock_verifier_json("MAYBE", "uncertain")

        verifier = ClaimVerifier(api_key="test-key")
        claim = _make_claim()
        evidence_map = {claim.claim_id: [{"text": "evidence", "document_id": "doc1"}]}
        verdicts = verifier.verify_claims([claim], evidence_map)

        assert len(verdicts) == 1
        assert verdicts[0].verdict == VerdictEnum.INSUFFICIENT


class TestVerifierErrors:
    @patch("graphite.pipeline._client.create_llm_client")
    def test_chat_json_error_propagates(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat_json.side_effect = ValueError("LLM returned invalid JSON")

        verifier = ClaimVerifier(api_key="test-key")
        claim = _make_claim()
        evidence_map = {claim.claim_id: [{"text": "evidence", "document_id": "doc1"}]}
        with pytest.raises(ValueError, match="LLM returned invalid JSON"):
            verifier.verify_claims([claim], evidence_map)


class TestClientInjection:
    def test_injected_client_is_used(self):
        from graphite.pipeline._client import LLMClient
        mock_client = MagicMock(spec=LLMClient)
        mock_client.chat_json.return_value = _mock_verifier_json("SUPPORTED")

        verifier = ClaimVerifier(client=mock_client)
        claim = _make_claim()
        evidence_map = {claim.claim_id: [{"text": "evidence", "document_id": "doc1"}]}
        verifier.verify_claims([claim], evidence_map)
        mock_client.chat_json.assert_called_once()


class TestCustomPrompt:
    @patch("graphite.pipeline._client.create_llm_client")
    def test_custom_system_prompt_is_used(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat_json.return_value = _mock_verifier_json("SUPPORTED")

        verifier = ClaimVerifier(api_key="test-key", system_prompt="Custom verifier prompt.")
        claim = _make_claim()
        evidence_map = {claim.claim_id: [{"text": "evidence", "document_id": "doc1"}]}
        verifier.verify_claims([claim], evidence_map)

        call_args = mock_client.chat_json.call_args
        assert call_args.kwargs["system_prompt"] == "Custom verifier prompt."
