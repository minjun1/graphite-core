"""
graphite/llm.py — Shared Gemini LLM utilities (optional dependency).

Uses Gemini Pro by default.
Provides rate-limited, retried structured output with Pydantic validation.

Install: pip install graphite-engine[llm]
"""

import json
import os
import re
import threading
import time
from typing import Any, Type, TypeVar

from pydantic import BaseModel

# ── Markdown fence stripping ──
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)```\s*$", re.DOTALL)


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM response text."""
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text


# ── Model config ──
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
GEMINI_MIN_INTERVAL = float(os.environ.get("GEMINI_MIN_INTERVAL", "1.0"))

# ── Rate limiting ──
_gemini_lock = threading.Lock()
_last_call = 0.0

T = TypeVar("T", bound=BaseModel)

# ── Lazy client ──
_client = None
_client_lock = threading.Lock()


def _init_client():
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai is required for LLM features. "
            "Install with: pip install graphite-engine[llm]"
        )
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)


def get_gemini_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = _init_client()
    return _client


def gemini_extract_structured(
    contents: str,
    system_prompt: str,
    schema: Type[T],
    model: str = "",
    temperature: float = 0.0,
    max_retries: int = 3,
) -> T:
    """Call Gemini with structured output, returning a validated Pydantic model.

    Args:
        contents: Document text / context
        system_prompt: System instruction for extraction
        schema: Pydantic model class for structured output
        model: Override model name (default: GEMINI_MODEL)
        temperature: LLM temperature
        max_retries: Retry count

    Returns:
        Parsed and validated Pydantic model instance
    """
    from google.genai import types

    global _last_call
    client = get_gemini_client()
    model_name = model or GEMINI_MODEL

    for attempt in range(max_retries):
        with _gemini_lock:
            elapsed = time.time() - _last_call
            if elapsed < GEMINI_MIN_INTERVAL:
                time.sleep(GEMINI_MIN_INTERVAL - elapsed)
            _last_call = time.time()

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=temperature,
                ),
            )

            raw = strip_markdown_fences(response.text)
            data = json.loads(raw)
            return schema.model_validate(data)

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  ⚠️ Gemini attempt {attempt + 1} failed: {e}. Retry in {wait}s...")
            time.sleep(wait)

    # Guard: if max_retries=0, the loop never executes → silent None return
    raise RuntimeError("gemini_extract_structured: max_retries must be >= 1")


def gemini_extract_json(
    contents: str,
    system_prompt: str,
    model: str = "",
    temperature: float = 0.0,
) -> dict:
    """Call Gemini and return raw JSON dict (no schema validation)."""
    from google.genai import types

    global _last_call
    client = get_gemini_client()
    model_name = model or GEMINI_MODEL

    with _gemini_lock:
        elapsed = time.time() - _last_call
        if elapsed < GEMINI_MIN_INTERVAL:
            time.sleep(GEMINI_MIN_INTERVAL - elapsed)
        _last_call = time.time()

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            temperature=temperature,
        ),
    )

    raw = strip_markdown_fences(response.text)
    return json.loads(raw)
