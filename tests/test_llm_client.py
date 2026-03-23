"""tests/test_llm_client.py — Tests for the unified LLM client."""

import pytest
from unittest.mock import patch, MagicMock


class TestLLMClientProviderDetection:
    def test_claude_model_selects_anthropic(self):
        from graphite.pipeline._client import LLMClient
        client = LLMClient(api_key="test-key")
        assert client._is_anthropic("claude-sonnet-4-6") is True
        assert client._is_anthropic("claude-haiku-4-5-20251001") is True

    def test_gemini_model_selects_openai(self):
        from graphite.pipeline._client import LLMClient
        client = LLMClient(api_key="test-key")
        assert client._is_anthropic("gemini-2.5-flash") is False
        assert client._is_anthropic("gpt-4o") is False

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ant-key"}, clear=True)
    def test_anthropic_env_key_selects_anthropic_for_unknown_model(self):
        from graphite.pipeline._client import LLMClient
        client = LLMClient(api_key="test-key")
        assert client._is_anthropic("my-custom-model") is True

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ant-key"}, clear=True)
    def test_gemini_model_not_routed_to_anthropic(self):
        from graphite.pipeline._client import LLMClient
        client = LLMClient(api_key="test-key")
        assert client._is_anthropic("gemini-2.5-flash") is False
        assert client._is_anthropic("gpt-4o") is False


class TestLLMClientAPIKeyResolution:
    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises(self):
        from graphite.pipeline._client import create_llm_client
        with pytest.raises(RuntimeError, match="No API key found"):
            create_llm_client()

    @patch.dict("os.environ", {"GEMINI_API_KEY": "gem-key"}, clear=True)
    def test_gemini_key_resolves(self):
        from graphite.pipeline._client import create_llm_client
        client = create_llm_client()
        assert client.api_key == "gem-key"

    def test_explicit_key_takes_precedence(self):
        from graphite.pipeline._client import create_llm_client
        client = create_llm_client(api_key="explicit-key")
        assert client.api_key == "explicit-key"


class TestLLMClientChatJSON:
    @patch("graphite.pipeline._client.LLMClient._chat_openai")
    def test_chat_json_openai_path(self, mock_chat):
        from graphite.pipeline._client import LLMClient
        mock_chat.return_value = {"claims": []}
        client = LLMClient(api_key="test-key")
        result = client.chat_json("gemini-2.5-flash", "system", "user")
        assert result == {"claims": []}
        mock_chat.assert_called_once()

    @patch("graphite.pipeline._client.LLMClient._chat_anthropic")
    def test_chat_json_anthropic_path(self, mock_chat):
        from graphite.pipeline._client import LLMClient
        mock_chat.return_value = {"claims": []}
        client = LLMClient(api_key="test-key")
        result = client.chat_json("claude-sonnet-4-6", "system", "user")
        assert result == {"claims": []}
        mock_chat.assert_called_once()


class TestBackwardCompat:
    def test_create_llm_client_returns_llm_client(self):
        from graphite.pipeline._client import create_llm_client
        client = create_llm_client(api_key="test-key")
        assert hasattr(client, "chat_json")
