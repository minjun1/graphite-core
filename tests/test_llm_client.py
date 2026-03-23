"""tests/test_llm_client.py — Tests for the pipeline LLM client."""

import pytest
from unittest.mock import patch


class TestCreateOpenAIClientAPIKey:
    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises_runtime_error(self):
        """Should raise RuntimeError when no API key is available."""
        from graphite.pipeline._client import create_openai_client
        with pytest.raises(RuntimeError, match="No API key found"):
            create_openai_client()
