"""tests/test_extractor.py — Unit tests for graphite.pipeline.extractor (mocked)."""

import json
import pytest
from unittest.mock import patch, MagicMock

from graphite.pipeline.extractor import ClaimExtractor, extract_claims


def _mock_openai_response(claims_data):
    mock_message = MagicMock()
    mock_message.content = json.dumps({"claims": claims_data})
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


class TestClaimExtractor:
    @patch("graphite.pipeline._client.create_openai_client")
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
        mock_client.chat.completions.create.return_value = _mock_openai_response(claims_data)

        extractor = ClaimExtractor(api_key="test-key")
        result = extractor.extract_claims("Apple supplies chips to Tesla")

        assert len(result) == 1
        assert result[0].claim_text == "Apple supplies chips to Tesla"
        assert result[0].subject_entities == ["Apple"]
        assert result[0].predicate == "SUPPLIES_TO"
        assert result[0].object_entities == ["Tesla"]
        assert result[0].claim_id

    @patch("graphite.pipeline._client.create_openai_client")
    def test_extract_multiple_claims(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        claims_data = [
            {"claim_text": "C1", "subject_entities": ["A"], "predicate": "P1", "object_entities": ["B"]},
            {"claim_text": "C2", "subject_entities": ["C"], "predicate": "P2", "object_entities": ["D"]},
        ]
        mock_client.chat.completions.create.return_value = _mock_openai_response(claims_data)

        extractor = ClaimExtractor(api_key="test-key")
        result = extractor.extract_claims("some document")
        assert len(result) == 2

    @patch("graphite.pipeline._client.create_openai_client")
    def test_empty_document_returns_empty(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response([])

        extractor = ClaimExtractor(api_key="test-key")
        result = extractor.extract_claims("")
        assert result == []


class TestExtractClaimsConvenience:
    @patch("graphite.pipeline._client.create_openai_client")
    def test_convenience_function(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        claims_data = [
            {"claim_text": "X", "subject_entities": ["A"], "predicate": "P", "object_entities": ["B"]},
        ]
        mock_client.chat.completions.create.return_value = _mock_openai_response(claims_data)

        result = extract_claims("doc text", api_key="test-key")
        assert len(result) == 1


class TestExtractorMalformedJSON:
    @patch("graphite.pipeline._client.create_openai_client")
    def test_malformed_json_raises_value_error(self, mock_create):
        """Malformed LLM response raises ValueError, not JSONDecodeError."""
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = "not valid json {{"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        extractor = ClaimExtractor(api_key="test-key")
        with pytest.raises(ValueError, match="LLM returned invalid JSON"):
            extractor.extract_claims("some document")
