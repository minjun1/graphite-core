"""Unified LLM client for pipeline modules.

Supports OpenAI-compatible providers (Gemini, GPT, Ollama, vLLM)
and Anthropic (Claude) through a single chat_json() interface.
"""

import json
import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text


# Known non-Anthropic model prefixes
_NON_ANTHROPIC_PREFIXES = ("gemini", "gpt", "o1", "o3", "llama", "mistral", "qwen")


class LLMClient:
    """Unified LLM client — wraps OpenAI-compatible and Anthropic SDKs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url

    def _is_anthropic(self, model: str) -> bool:
        """Determine if a model should use the Anthropic SDK."""
        if model.startswith("claude"):
            return True
        if model.startswith(_NON_ANTHROPIC_PREFIXES):
            return False
        # For unknown model names, fall back to env var presence
        if os.environ.get("ANTHROPIC_API_KEY"):
            return True
        return False

    def chat_json(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
    ) -> dict:
        """Send a chat request and return parsed JSON response with retry."""
        if max_retries < 1:
            raise RuntimeError("chat_json: max_retries must be >= 1")
        for attempt in range(max_retries):
            try:
                if self._is_anthropic(model):
                    return self._chat_anthropic(model, system_prompt, user_prompt)
                return self._chat_openai(model, system_prompt, user_prompt)
            except (json.JSONDecodeError, ConnectionError, TimeoutError, OSError) as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "LLM attempt %d failed: %s. Retry in %ds...",
                    attempt + 1, e, wait,
                )
                time.sleep(wait)

    def _chat_openai(self, model: str, system_prompt: str, user_prompt: str) -> dict:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                'OpenAI client required. Install with: pip install "graphite-engine[llm]"'
            )

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
            or os.environ.get(
                "OPENAI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        raw = _strip_fences(response.choices[0].message.content)
        return json.loads(raw)

    def _chat_anthropic(self, model: str, system_prompt: str, user_prompt: str) -> dict:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                'Anthropic client required for Claude models. '
                'Install with: pip install "graphite-engine[llm]"'
            )

        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        kwargs = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        client = Anthropic(**kwargs)

        # Ensure system prompt asks for JSON output
        if "json" not in system_prompt.lower():
            system_prompt = system_prompt.rstrip() + "\n\nReturn your response as valid JSON."

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw = _strip_fences(response.content[0].text)
        return json.loads(raw)


def create_llm_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMClient:
    """Create an LLMClient, resolving API key from environment if needed.

    Key resolution order:
    1. Explicit api_key parameter
    2. ANTHROPIC_API_KEY
    3. GEMINI_API_KEY
    4. OPENAI_API_KEY
    """
    resolved_key = (
        api_key
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not resolved_key:
        raise RuntimeError(
            "No API key found. Set ANTHROPIC_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY, "
            "or pass api_key= explicitly."
        )
    return LLMClient(api_key=resolved_key, base_url=base_url)
