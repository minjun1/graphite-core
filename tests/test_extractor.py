"""tests/test_extractor.py — Unit tests for graphite.pipeline.extractor (mocked)."""

import pytest
from unittest.mock import patch, MagicMock

from graphite.pipeline.extractor import ClaimExtractor, extract_claims


class TestClaimExtractor:
    @patch("graphite.pipeline._client.create_llm_client")
    def test_extract_claims_parses_response(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        claims_data = [
            {
                "claim_text": "Apple supplies chips to Tesla",
                "subject_entities": ["Apple"],
                "predicate": "SUPPLIES_TO",
                "object_entities": ["Tesla"],
            },
        ]
        mock_client.chat_json.return_value = {"claims": claims_data}

        extractor = ClaimExtractor(api_key="test-key")
        result = extractor.extract_claims("Apple supplies chips to Tesla")

        assert len(result) == 1
        assert result[0].claim_text == "Apple supplies chips to Tesla"
        assert result[0].subject_entities == ["Apple"]
        assert result[0].predicate == "SUPPLIES_TO"
        assert result[0].object_entities == ["Tesla"]
        assert result[0].claim_id

    @patch("graphite.pipeline._client.create_llm_client")
    def test_extract_multiple_claims(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        claims_data = [
            {"claim_text": "C1", "subject_entities": ["A"], "predicate": "P1", "object_entities": ["B"]},
            {"claim_text": "C2", "subject_entities": ["C"], "predicate": "P2", "object_entities": ["D"]},
        ]
        mock_client.chat_json.return_value = {"claims": claims_data}

        extractor = ClaimExtractor(api_key="test-key")
        result = extractor.extract_claims("some document")
        assert len(result) == 2

    @patch("graphite.pipeline._client.create_llm_client")
    def test_empty_document_returns_empty(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat_json.return_value = {"claims": []}

        extractor = ClaimExtractor(api_key="test-key")
        result = extractor.extract_claims("")
        assert result == []


class TestExtractClaimsConvenience:
    @patch("graphite.pipeline._client.create_llm_client")
    def test_convenience_function(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat_json.return_value = {"claims": [
            {"claim_text": "X", "subject_entities": ["A"], "predicate": "P", "object_entities": ["B"]},
        ]}

        result = extract_claims("doc text", api_key="test-key")
        assert len(result) == 1


class TestExtractorErrors:
    @patch("graphite.pipeline._client.create_llm_client")
    def test_chat_json_error_propagates(self, mock_create):
        """Errors from chat_json propagate to caller."""
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat_json.side_effect = ValueError("LLM returned invalid JSON")

        extractor = ClaimExtractor(api_key="test-key")
        with pytest.raises(ValueError, match="LLM returned invalid JSON"):
            extractor.extract_claims("some document")


class TestClientInjection:
    def test_injected_client_is_used(self):
        """When client= is passed, it should be used instead of creating a new one."""
        from graphite.pipeline._client import LLMClient
        mock_client = MagicMock(spec=LLMClient)
        mock_client.chat_json.return_value = {"claims": []}

        extractor = ClaimExtractor(client=mock_client)
        extractor.extract_claims("doc")

        mock_client.chat_json.assert_called_once()


class TestCustomPrompt:
    @patch("graphite.pipeline._client.create_llm_client")
    def test_custom_system_prompt_is_used(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat_json.return_value = {"claims": []}

        extractor = ClaimExtractor(api_key="test-key", system_prompt="Custom extractor prompt.")
        extractor.extract_claims("doc")

        call_args = mock_client.chat_json.call_args
        assert call_args.kwargs["system_prompt"] == "Custom extractor prompt."
