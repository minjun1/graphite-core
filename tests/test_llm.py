"""tests/test_llm.py — Unit tests for graphite.llm (mocked)."""

import json
import pytest
from unittest.mock import patch, MagicMock

from pydantic import BaseModel

# Mock google.genai before importing llm module
_mock_genai = MagicMock()
_mock_types = MagicMock()


@pytest.fixture(autouse=True)
def _patch_genai_and_reset():
    """Patch google.genai in sys.modules and reset llm module state."""
    import graphite.llm as llm_mod
    mock_google = MagicMock()
    mock_google.genai = _mock_genai
    with patch.dict("sys.modules", {
        "google": mock_google,
        "google.genai": _mock_genai,
        "google.genai.types": _mock_types,
    }):
        llm_mod._client = None
        llm_mod._last_call = 0.0
        yield llm_mod


class SampleSchema(BaseModel):
    name: str
    value: int


class TestGeminiClient:
    def test_missing_api_key_raises(self, _patch_genai_and_reset):
        llm_mod = _patch_genai_and_reset
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False):
            with pytest.raises(RuntimeError, match="GEMINI_API_KEY not set"):
                llm_mod._init_client()

    def test_lazy_initialization(self, _patch_genai_and_reset):
        llm_mod = _patch_genai_and_reset
        mock_client = MagicMock()
        _mock_genai.Client.return_value = mock_client

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            result = llm_mod.get_gemini_client()
            assert result is mock_client
            # Second call returns cached client
            result2 = llm_mod.get_gemini_client()
            assert result2 is mock_client


class TestGeminiExtractStructured:
    def test_parses_response(self, _patch_genai_and_reset):
        llm_mod = _patch_genai_and_reset
        mock_response = MagicMock()
        mock_response.text = '{"name": "test", "value": 42}'

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        llm_mod._client = mock_client

        result = llm_mod.gemini_extract_structured(
            contents="test doc", system_prompt="extract", schema=SampleSchema,
        )
        assert result.name == "test"
        assert result.value == 42

    def test_strips_markdown_fences(self, _patch_genai_and_reset):
        llm_mod = _patch_genai_and_reset
        mock_response = MagicMock()
        mock_response.text = '```json\n{"name": "fenced", "value": 99}\n```'

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        llm_mod._client = mock_client

        result = llm_mod.gemini_extract_structured(
            contents="test", system_prompt="extract", schema=SampleSchema,
        )
        assert result.name == "fenced"
        assert result.value == 99

    def test_retry_on_failure(self, _patch_genai_and_reset):
        llm_mod = _patch_genai_and_reset
        good_response = MagicMock()
        good_response.text = '{"name": "retry", "value": 1}'

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            Exception("transient error"),
            good_response,
        ]
        llm_mod._client = mock_client

        with patch("time.sleep"):  # skip retry/rate-limit sleeps
            result = llm_mod.gemini_extract_structured(
                contents="test", system_prompt="extract",
                schema=SampleSchema, max_retries=3,
            )
            assert result.name == "retry"
            assert mock_client.models.generate_content.call_count == 2


class TestGeminiExtractJson:
    def test_returns_dict(self, _patch_genai_and_reset):
        llm_mod = _patch_genai_and_reset
        mock_response = MagicMock()
        mock_response.text = '{"key": "value", "num": 123}'

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        llm_mod._client = mock_client

        result = llm_mod.gemini_extract_json(contents="test", system_prompt="extract")
        assert result == {"key": "value", "num": 123}

    def test_strips_markdown_fences(self, _patch_genai_and_reset):
        llm_mod = _patch_genai_and_reset
        mock_response = MagicMock()
        mock_response.text = '```json\n{"fenced": true}\n```'

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        llm_mod._client = mock_client

        result = llm_mod.gemini_extract_json(contents="test", system_prompt="extract")
        assert result == {"fenced": True}
