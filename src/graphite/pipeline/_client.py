"""Shared OpenAI-compatible client factory for pipeline modules."""

import os
from typing import Optional


def create_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Create an OpenAI client with Graphite's default configuration.

    Priority: explicit args > GEMINI_API_KEY > OPENAI_API_KEY.
    Default base_url points to Google's OpenAI-compatible endpoint.

    Note: This will be replaced by LLMClient in Phase 3.
    """
    resolved_key = (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not resolved_key:
        raise RuntimeError(
            "No API key found. Set GEMINI_API_KEY or OPENAI_API_KEY, "
            "or pass api_key= explicitly."
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            'OpenAI client required for pipeline. Install with: pip install "graphite-engine[llm]"'
        )

    return OpenAI(
        api_key=resolved_key,
        base_url=base_url
        or os.environ.get(
            "OPENAI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
    )
