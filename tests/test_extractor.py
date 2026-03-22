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
    @patch("openai.OpenAI")
    def test_extract_claims_parses_response(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

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

    @patch("openai.OpenAI")
    def test_extract_multiple_claims(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        claims_data = [
            {"claim_text": "C1", "subject_entities": ["A"], "predicate": "P1", "object_entities": ["B"]},
            {"claim_text": "C2", "subject_entities": ["C"], "predicate": "P2", "object_entities": ["D"]},
        ]
        mock_client.chat.completions.create.return_value = _mock_openai_response(claims_data)

        extractor = ClaimExtractor(api_key="test-key")
        result = extractor.extract_claims("some document")
        assert len(result) == 2

    @patch("openai.OpenAI")
    def test_empty_document_returns_empty(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response([])

        extractor = ClaimExtractor(api_key="test-key")
        result = extractor.extract_claims("")
        assert result == []


class TestExtractClaimsConvenience:
    @patch("openai.OpenAI")
    def test_convenience_function(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        claims_data = [
            {"claim_text": "X", "subject_entities": ["A"], "predicate": "P", "object_entities": ["B"]},
        ]
        mock_client.chat.completions.create.return_value = _mock_openai_response(claims_data)

        result = extract_claims("doc text", api_key="test-key")
        assert len(result) == 1
