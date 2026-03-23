# Tech Debt Cleanup — Design Spec

**Date:** 2026-03-23
**Scope:** Remove legacy LLM module, add retry + client sharing + logging to pipeline, fix model routing.

---

## Overview

Clean up technical debt identified in the 2nd code review. Remove the unused `llm.py` legacy module and `google-genai` dependency, add retry logic and client sharing to the unified `LLMClient`, replace all `print()` with structured `logging`, and simplify the Anthropic model detection logic.

---

## 1. Remove `llm.py` and `google-genai` dependency

**Delete:**
- `src/graphite/llm.py`
- `tests/test_llm.py`

**Modify:** `pyproject.toml`
- Remove `google-genai>=1.0` from `[project.optional-dependencies] llm`
- Result: `llm = ["openai>=1.0", "anthropic>=0.40"]`

**Rationale:** No source module imports `llm.py` functions. The pipeline uses `LLMClient` exclusively. Gemini is already supported via the OpenAI-compatible endpoint in `LLMClient._chat_openai()`. The `google-genai` SDK is no longer needed.

---

## 2. Add retry logic to `LLMClient.chat_json()`

**File:** `src/graphite/pipeline/_client.py`

Add exponential backoff retry to `chat_json()`:

```python
def chat_json(
    self,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
) -> dict:
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
    raise RuntimeError("chat_json: max_retries must be >= 1")
```

The retry catches only transient/parse errors. `ImportError` (SDK missing), `RuntimeError` (no API key), and `KeyError`/`AttributeError` (response format bugs) are NOT retried — they fail immediately. SDK-specific API errors (`openai.APIError`, `anthropic.APIError`) inherit from `OSError` or similar, so `OSError` covers most transient network errors.

---

## 3. Pipeline client sharing

**Files:** `extractor.py`, `verifier.py`, `analyzer.py`, `report.py`

Each pipeline class currently creates its own `LLMClient` in `__init__`. Change to accept an optional pre-built client:

```python
class ClaimExtractor:
    def __init__(self, api_key=None, base_url=None, system_prompt=None, client=None):
        from graphite.pipeline._client import create_llm_client
        from graphite.pipeline.prompts import EXTRACTOR_SYSTEM_PROMPT
        self.client = client or create_llm_client(api_key=api_key, base_url=base_url)
        self.system_prompt = system_prompt or EXTRACTOR_SYSTEM_PROMPT
```

Same pattern for `ClaimVerifier` and `ArgumentAnalyzer`.

In `verify_agent_output()` (the Hero API), create one client and pass it to all stages:

```python
def verify_agent_output(text, corpus, model="gemini-2.5-flash", api_key=None, base_url=None, prompts=None):
    from graphite.pipeline._client import create_llm_client
    client = create_llm_client(api_key=api_key, base_url=base_url)

    extractor = ClaimExtractor(client=client, system_prompt=prompts.extractor)
    # ... verifier, analyzer also get client=client
```

**Backward compatibility:** Existing code that creates `ClaimExtractor(api_key="...")` still works — `client=None` triggers auto-creation.

---

## 4. Structured logging

**Pattern:** Standard Python `logging` with `NullHandler`.

Add to each module that currently uses `print()`:

```python
import logging
logger = logging.getLogger(__name__)
```

In `src/graphite/__init__.py`, add:
```python
import logging
logging.getLogger("graphite").addHandler(logging.NullHandler())
```

This ensures graphite is silent by default. Users who want logs do:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

**Files to update:**
- `src/graphite/pipeline/_client.py` — retry warnings
- `src/graphite/__init__.py` — NullHandler setup

No other files currently use `print()` in the pipeline path (the old `llm.py` print statements are deleted with the file).

---

## 5. Simplify `_is_anthropic()` model routing

**File:** `src/graphite/pipeline/_client.py`

Current logic:
```python
def _is_anthropic(self, model):
    if model.startswith("claude"): return True
    if model.startswith(known_prefixes): return False
    if os.environ.get("ANTHROPIC_API_KEY"): return True  # problematic
    return False
```

Change to:
```python
def _is_anthropic(self, model):
    return model.startswith("claude")
```

**Rationale:** The env-var fallback for unknown models causes surprising behavior with Ollama/vLLM custom model names. The rule should be simple: `claude*` → Anthropic, everything else → OpenAI-compatible. Users who need custom Anthropic-compatible endpoints with non-claude model names can inject a custom `LLMClient` subclass via the `client=` parameter added in Section 3.

> **Ordering dependency:** This simplification depends on Section 3 (client sharing) being completed first, since `client=` injection is the escape hatch for advanced routing.

---

## Breaking changes

- `from graphite.llm import gemini_extract_structured` stops working (no one uses it in src)
- `google-genai` no longer installed with `pip install graphite-engine[llm]`
- `_is_anthropic()` no longer auto-detects Anthropic from env var for unknown models
- Pipeline classes now accept `client=` parameter (additive, not breaking)

## Test impact

- `tests/test_llm.py` deleted (tests the removed module)
- `tests/test_llm_client.py` updated: retry tests, simplified `_is_anthropic` tests
- `tests/test_extractor.py`, `test_verifier.py`, `test_analyzer.py` updated: test client injection
- Net test count may decrease slightly (llm.py tests removed) but coverage of active code improves
