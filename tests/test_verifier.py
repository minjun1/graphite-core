"""tests/test_verifier.py — Unit tests for graphite.pipeline.verifier (mocked)."""

import json
import pytest
from unittest.mock import patch, MagicMock

from graphite.pipeline.verifier import ClaimVerifier, verify_claims
from graphite.claim import Claim, ClaimType, ClaimOrigin, ClaimGranularity, VerdictEnum
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


def _mock_verifier_response(verdict="SUPPORTED", rationale="Evidence supports"):
    mock_message = MagicMock()
    mock_message.content = json.dumps({
        "verdict": verdict,
        "rationale_text": rationale,
        "contradiction_type": None,
        "missing_evidence_reason": None,
        "temporal_alignment": None,
        "needs_human_review": False,
        "cited_span": "relevant quote",
        "supporting_evidence_indices": [0] if verdict == "SUPPORTED" else [],
        "conflicting_evidence_indices": [0] if verdict == "CONFLICTED" else [],
    })
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


class TestClaimVerifier:
    @patch("graphite.pipeline._client.create_openai_client")
    def test_supported_verdict(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_verifier_response("SUPPORTED")

        verifier = ClaimVerifier(api_key="test-key")
        claim = _make_claim()
        evidence_map = {claim.claim_id: [{"text": "Apple supplies Tesla", "document_id": "doc1"}]}
        verdicts = verifier.verify_claims([claim], evidence_map)

        assert len(verdicts) == 1
        assert verdicts[0].verdict == VerdictEnum.SUPPORTED
        assert verdicts[0].claim_id == claim.claim_id

    @patch("graphite.pipeline._client.create_openai_client")
    def test_conflicted_verdict(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_verifier_response("CONFLICTED")

        verifier = ClaimVerifier(api_key="test-key")
        claim = _make_claim()
        evidence_map = {claim.claim_id: [{"text": "Contradictory evidence", "document_id": "doc1"}]}
        verdicts = verifier.verify_claims([claim], evidence_map)

        assert verdicts[0].verdict == VerdictEnum.CONFLICTED

    @patch("graphite.pipeline._client.create_openai_client")
    def test_insufficient_verdict(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_verifier_response("INSUFFICIENT")

        verifier = ClaimVerifier(api_key="test-key")
        claim = _make_claim()
        evidence_map = {claim.claim_id: []}
        verdicts = verifier.verify_claims([claim], evidence_map)

        assert verdicts[0].verdict == VerdictEnum.INSUFFICIENT

    @patch("graphite.pipeline._client.create_openai_client")
    def test_empty_evidence_map(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_verifier_response("INSUFFICIENT")

        verifier = ClaimVerifier(api_key="test-key")
        claim = _make_claim()
        verdicts = verifier.verify_claims([claim], {})

        assert len(verdicts) == 1
        assert verdicts[0].verdict == VerdictEnum.INSUFFICIENT


class TestVerifyClaimsConvenience:
    @patch("graphite.pipeline._client.create_openai_client")
    def test_convenience_function(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_verifier_response("SUPPORTED")

        claim = _make_claim()
        evidence_map = {claim.claim_id: [{"text": "evidence", "document_id": "doc1"}]}
        verdicts = verify_claims([claim], evidence_map, api_key="test-key")
        assert len(verdicts) == 1


class TestVerifyEmptyInput:
    @patch("graphite.pipeline._client.create_openai_client")
    def test_empty_claims_list_returns_empty_verdicts(self, MockOpenAI):
        """Empty claims list returns empty verdicts without calling LLM."""
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        verifier = ClaimVerifier(api_key="test-key")
        result = verifier.verify_claims([], {})
        assert result == []
        # Verify LLM was not called
        mock_client.chat.completions.create.assert_not_called()


class TestVerifyUnexpectedVerdictString:
    @patch("graphite.pipeline._client.create_openai_client")
    def test_unexpected_verdict_string_defaults_to_insufficient(self, MockOpenAI):
        """Unknown verdict string falls back to INSUFFICIENT."""
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = json.dumps({
            "verdict": "MAYBE",
            "rationale_text": "uncertain",
            "contradiction_type": None,
            "missing_evidence_reason": None,
            "temporal_alignment": None,
            "needs_human_review": False,
            "cited_span": None,
            "supporting_evidence_indices": [],
            "conflicting_evidence_indices": [],
        })
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        verifier = ClaimVerifier(api_key="test-key")
        claim = _make_claim()
        evidence_map = {claim.claim_id: [{"text": "evidence", "document_id": "doc1"}]}
        verdicts = verifier.verify_claims([claim], evidence_map)

        assert len(verdicts) == 1
        assert verdicts[0].verdict == VerdictEnum.INSUFFICIENT


class TestVerifierMalformedJSON:
    @patch("graphite.pipeline._client.create_openai_client")
    def test_malformed_json_raises_value_error(self, MockOpenAI):
        """Malformed LLM response raises ValueError, not JSONDecodeError."""
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = "not valid json {{"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        verifier = ClaimVerifier(api_key="test-key")
        claim = _make_claim()
        evidence_map = {claim.claim_id: [{"text": "evidence", "document_id": "doc1"}]}
        with pytest.raises(ValueError, match="LLM returned invalid JSON"):
            verifier.verify_claims([claim], evidence_map)


class TestCustomPrompt:
    @patch("graphite.pipeline._client.create_openai_client")
    def test_custom_system_prompt_is_used(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_verifier_response("SUPPORTED")

        verifier = ClaimVerifier(api_key="test-key", system_prompt="Custom verifier prompt.")
        claim = _make_claim()
        evidence_map = {claim.claim_id: [{"text": "evidence", "document_id": "doc1"}]}
        verifier.verify_claims([claim], evidence_map)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["content"] == "Custom verifier prompt."
