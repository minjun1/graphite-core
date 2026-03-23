"""tests/test_llm_client.py — Tests for the unified LLM client."""

import pytest
from unittest.mock import patch, MagicMock


class TestLLMClientProviderDetection:
    def test_claude_model_selects_anthropic(self):
        from graphite.pipeline._client import LLMClient
        client = LLMClient(api_key="test-key")
        assert client._is_anthropic("claude-sonnet-4-6") is True
        assert client._is_anthropic("claude-haiku-4-5-20251001") is True

    def test_non_claude_model_selects_openai(self):
        from graphite.pipeline._client import LLMClient
        client = LLMClient(api_key="test-key")
        assert client._is_anthropic("gemini-2.5-flash") is False
        assert client._is_anthropic("gpt-4o") is False
        assert client._is_anthropic("llama3") is False
        assert client._is_anthropic("my-custom-model") is False

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ant-key"}, clear=True)
    def test_env_key_does_not_affect_routing(self):
        """ANTHROPIC_API_KEY should NOT affect routing — only model name matters."""
        from graphite.pipeline._client import LLMClient
        client = LLMClient(api_key="test-key")
        assert client._is_anthropic("gemini-2.5-flash") is False
        assert client._is_anthropic("my-custom-model") is False
        assert client._is_anthropic("claude-sonnet-4-6") is True


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


class TestLLMClientRetry:
    @patch("graphite.pipeline._client.LLMClient._chat_openai")
    def test_retries_on_json_decode_error(self, mock_chat):
        """Transient JSONDecodeError should be retried."""
        import json as _json
        from graphite.pipeline._client import LLMClient
        mock_chat.side_effect = [
            _json.JSONDecodeError("bad", "", 0),
            {"claims": []},
        ]
        client = LLMClient(api_key="test-key")
        with patch("time.sleep"):
            result = client.chat_json("gemini-2.5-flash", "sys", "usr", max_retries=3)
        assert result == {"claims": []}
        assert mock_chat.call_count == 2

    @patch("graphite.pipeline._client.LLMClient._chat_openai")
    def test_does_not_retry_import_error(self, mock_chat):
        """ImportError (SDK missing) should NOT be retried."""
        from graphite.pipeline._client import LLMClient
        mock_chat.side_effect = ImportError("no module")
        client = LLMClient(api_key="test-key")
        with pytest.raises(ImportError):
            client.chat_json("gemini-2.5-flash", "sys", "usr", max_retries=3)
        assert mock_chat.call_count == 1

    @patch("graphite.pipeline._client.LLMClient._chat_openai")
    def test_does_not_retry_runtime_error(self, mock_chat):
        """RuntimeError (no API key) should NOT be retried."""
        from graphite.pipeline._client import LLMClient
        mock_chat.side_effect = RuntimeError("No API key")
        client = LLMClient(api_key="test-key")
        with pytest.raises(RuntimeError):
            client.chat_json("gemini-2.5-flash", "sys", "usr", max_retries=3)
        assert mock_chat.call_count == 1

    @patch("graphite.pipeline._client.LLMClient._chat_openai")
    def test_raises_after_max_retries_exhausted(self, mock_chat):
        """Should raise the last error after exhausting retries."""
        import json as _json
        from graphite.pipeline._client import LLMClient
        mock_chat.side_effect = _json.JSONDecodeError("bad", "", 0)
        client = LLMClient(api_key="test-key")
        with patch("time.sleep"):
            with pytest.raises(_json.JSONDecodeError):
                client.chat_json("gemini-2.5-flash", "sys", "usr", max_retries=2)
        assert mock_chat.call_count == 2

    def test_zero_retries_raises(self):
        """max_retries=0 should raise immediately."""
        from graphite.pipeline._client import LLMClient
        client = LLMClient(api_key="test-key")
        with pytest.raises(RuntimeError, match="max_retries must be >= 1"):
            client.chat_json("gemini-2.5-flash", "sys", "usr", max_retries=0)


class TestBackwardCompat:
    def test_create_llm_client_returns_llm_client(self):
        from graphite.pipeline._client import create_llm_client
        client = create_llm_client(api_key="test-key")
        assert hasattr(client, "chat_json")
