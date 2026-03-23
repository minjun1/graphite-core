# Tech Debt Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove legacy LLM module, add retry + client sharing + logging to pipeline, simplify model routing.

**Architecture:** Delete unused `llm.py` and `google-genai` dep. Add retry with narrow exception catching to `LLMClient.chat_json()`. Refactor pipeline classes to accept a shared `client=` parameter. Add Python `logging` with `NullHandler`. Simplify `_is_anthropic()` to `claude*`-only.

**Tech Stack:** Python 3.10+, Pydantic 2.x, pytest, OpenAI SDK, Anthropic SDK

**Spec:** `docs/superpowers/specs/2026-03-23-tech-debt-cleanup-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Delete | `src/graphite/llm.py` | Legacy Gemini module |
| Delete | `tests/test_llm.py` | Tests for deleted module |
| Modify | `pyproject.toml` | Remove google-genai |
| Modify | `src/graphite/pipeline/_client.py` | Retry, logging, simplified routing |
| Modify | `src/graphite/pipeline/extractor.py` | Accept `client=` param |
| Modify | `src/graphite/pipeline/verifier.py` | Accept `client=` param |
| Modify | `src/graphite/pipeline/analyzer.py` | Accept `client=` param |
| Modify | `src/graphite/pipeline/report.py` | Shared client, `base_url=` param |
| Modify | `src/graphite/__init__.py` | NullHandler logging setup |
| Modify | `tests/test_llm_client.py` | Retry tests, routing tests |
| Modify | `tests/test_extractor.py` | Client injection tests |

---

## Task 1: Remove `llm.py` and `google-genai` (Spec §1)

**Files:**
- Delete: `src/graphite/llm.py`
- Delete: `tests/test_llm.py`
- Modify: `pyproject.toml:29`

- [ ] **Step 1: Delete llm.py and its tests**

```bash
rm src/graphite/llm.py tests/test_llm.py
```

- [ ] **Step 2: Remove google-genai from pyproject.toml**

In `pyproject.toml`, change:
```toml
llm = ["openai>=1.0", "google-genai>=1.0", "anthropic>=0.40"]
```
to:
```toml
llm = ["openai>=1.0", "anthropic>=0.40"]
```

- [ ] **Step 3: Run tests to verify nothing breaks**

Run: `source venv/bin/activate && python -m pytest -v`
Expected: All PASS (test count decreases by ~11 — the deleted llm tests)

- [ ] **Step 4: Commit**

```bash
git rm src/graphite/llm.py tests/test_llm.py && git add pyproject.toml
git commit -m "refactor: remove legacy llm.py and google-genai dependency"
```

---

## Task 2: Add retry logic to `LLMClient.chat_json()` (Spec §2)

**Files:**
- Modify: `src/graphite/pipeline/_client.py:1-10,47-56`
- Test: `tests/test_llm_client.py`

- [ ] **Step 1: Write failing test for retry behavior**

Add to `tests/test_llm_client.py`:

```python
import time


class TestLLMClientRetry:
    @patch("graphite.pipeline._client.LLMClient._chat_openai")
    def test_retries_on_json_decode_error(self, mock_chat):
        """Transient JSONDecodeError should be retried."""
        import json
        from graphite.pipeline._client import LLMClient
        mock_chat.side_effect = [
            json.JSONDecodeError("bad", "", 0),
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
        import json
        from graphite.pipeline._client import LLMClient
        mock_chat.side_effect = json.JSONDecodeError("bad", "", 0)
        client = LLMClient(api_key="test-key")
        with patch("time.sleep"):
            with pytest.raises(json.JSONDecodeError):
                client.chat_json("gemini-2.5-flash", "sys", "usr", max_retries=2)
        assert mock_chat.call_count == 2

    def test_zero_retries_raises(self):
        """max_retries=0 should raise immediately."""
        from graphite.pipeline._client import LLMClient
        client = LLMClient(api_key="test-key")
        with pytest.raises(RuntimeError, match="max_retries must be >= 1"):
            client.chat_json("gemini-2.5-flash", "sys", "usr", max_retries=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source venv/bin/activate && python -m pytest tests/test_llm_client.py::TestLLMClientRetry -v`
Expected: FAIL — `chat_json()` doesn't accept `max_retries` yet

- [ ] **Step 3: Implement retry in `_client.py`**

Add `import time` and `import logging` at top of `_client.py`. Add logger. Update `chat_json()`:

```python
import json
import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ... _strip_fences, LLMClient class ...

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source venv/bin/activate && python -m pytest tests/test_llm_client.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/graphite/pipeline/_client.py tests/test_llm_client.py
git commit -m "feat: add retry with exponential backoff to LLMClient.chat_json()"
```

---

## Task 3: Pipeline client sharing (Spec §3)

**Files:**
- Modify: `src/graphite/pipeline/extractor.py:12-16`
- Modify: `src/graphite/pipeline/verifier.py:13-17`
- Modify: `src/graphite/pipeline/analyzer.py:17-21`
- Modify: `src/graphite/pipeline/report.py:16-22,34-50`
- Test: `tests/test_extractor.py`

- [ ] **Step 1: Write failing test for client injection**

Add to `tests/test_extractor.py`:

```python
class TestClientInjection:
    def test_injected_client_is_used(self):
        """When client= is passed, it should be used instead of creating a new one."""
        from graphite.pipeline._client import LLMClient
        mock_client = MagicMock(spec=LLMClient)
        mock_client.chat_json.return_value = {"claims": []}

        extractor = ClaimExtractor(client=mock_client)
        extractor.extract_claims("doc")

        mock_client.chat_json.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source venv/bin/activate && python -m pytest tests/test_extractor.py::TestClientInjection -v`
Expected: FAIL — `ClaimExtractor.__init__` doesn't accept `client=`

- [ ] **Step 3: Update extractor.py**

Change `__init__` in `src/graphite/pipeline/extractor.py`:

```python
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 system_prompt: Optional[str] = None, client=None):
        from graphite.pipeline._client import create_llm_client
        from graphite.pipeline.prompts import EXTRACTOR_SYSTEM_PROMPT
        self.client = client or create_llm_client(api_key=api_key, base_url=base_url)
        self.system_prompt = system_prompt or EXTRACTOR_SYSTEM_PROMPT
```

- [ ] **Step 4: Update verifier.py**

Same pattern for `ClaimVerifier.__init__` in `src/graphite/pipeline/verifier.py`:

```python
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 system_prompt: Optional[str] = None, client=None):
        from graphite.pipeline._client import create_llm_client
        from graphite.pipeline.prompts import VERIFIER_SYSTEM_PROMPT
        self.client = client or create_llm_client(api_key=api_key, base_url=base_url)
        self.system_prompt = system_prompt or VERIFIER_SYSTEM_PROMPT
```

- [ ] **Step 5: Update analyzer.py**

Same pattern for `ArgumentAnalyzer.__init__` in `src/graphite/pipeline/analyzer.py`:

```python
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 system_prompt: Optional[str] = None, client=None):
        from graphite.pipeline._client import create_llm_client
        from graphite.pipeline.prompts import ANALYZER_SYSTEM_PROMPT
        self.client = client or create_llm_client(api_key=api_key, base_url=base_url)
        self.system_prompt = system_prompt or ANALYZER_SYSTEM_PROMPT
```

- [ ] **Step 6: Update report.py — shared client + base_url**

Replace `verify_agent_output()` in `src/graphite/pipeline/report.py`:

```python
def verify_agent_output(
    text: str,
    corpus: List[Dict[str, str]],
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    prompts: Optional[PromptSet] = None,
) -> VerificationReport:
    """
    The Hero API: End-to-end verification pipeline.

    Args:
        text: The agent-generated markdown memo or assertions.
        corpus: List of evidence documents e.g. [{"document_id": "doc1", "text": "..."}]
        model: LLM model name to use across the pipeline.
        api_key: API key (or set GEMINI_API_KEY/OPENAI_API_KEY/ANTHROPIC_API_KEY).
        base_url: Custom endpoint URL for OpenAI-compatible or Anthropic proxies.
        prompts: Optional PromptSet for domain-specific prompts.

    Returns:
        VerificationReport: The aggregated results, ready for UI and audit logs.
    """
    from graphite.pipeline._client import create_llm_client

    prompts = prompts or DEFAULT_PROMPTS
    client = create_llm_client(api_key=api_key, base_url=base_url)
    document_id = hashlib.sha256(text.encode()).hexdigest()[:12]

    # 1. Extract
    extractor = ClaimExtractor(client=client, system_prompt=prompts.extractor)
    claims = extractor.extract_claims(text, model=model)

    # 2. Retrieve
    evidence_map = retrieve_evidence(claims, corpus)

    # 3. Verify
    verifier = ClaimVerifier(client=client, system_prompt=prompts.verifier)
    verdicts = verifier.verify_claims(claims, evidence_map, model=model)

    # 4. Analyze Arguments
    analyzer = ArgumentAnalyzer(client=client, system_prompt=prompts.analyzer)
    argument_verdicts = analyzer.analyze_argument_chain(text, verdicts, model=model)

    # 5. Build Report (unchanged from here)
```

Keep the rest of the function (lines 53-82) and the `review_document` alias unchanged.

- [ ] **Step 7: Add client injection tests to verifier and analyzer**

Add to `tests/test_verifier.py`:

```python
class TestClientInjection:
    def test_injected_client_is_used(self):
        from graphite.pipeline._client import LLMClient
        mock_client = MagicMock(spec=LLMClient)
        mock_client.chat_json.return_value = _mock_verifier_json("SUPPORTED")

        verifier = ClaimVerifier(client=mock_client)
        claim = _make_claim()
        evidence_map = {claim.claim_id: [{"text": "evidence", "document_id": "doc1"}]}
        verifier.verify_claims([claim], evidence_map)
        mock_client.chat_json.assert_called_once()
```

Add to `tests/test_analyzer.py`:

```python
class TestClientInjection:
    def test_injected_client_is_used(self):
        from graphite.pipeline._client import LLMClient
        mock_client = MagicMock(spec=LLMClient)
        mock_client.chat_json.return_value = _mock_analyzer_json([
            {"text": "ok", "verdict": "GROUNDED", "rationale_text": "fine",
             "contradiction_type": None, "needs_human_review": False},
        ])

        analyzer = ArgumentAnalyzer(client=mock_client)
        analyzer.analyze_argument_chain("memo", [_make_verdict()])
        mock_client.chat_json.assert_called_once()
```

> **Note:** Convenience functions (`extract_claims()`, `verify_claims()`, `analyze_argument_chain()`) intentionally do NOT accept `client=`. They always create a fresh client — this is by design for simple one-shot usage.

- [ ] **Step 8: Run all tests**

Run: `source venv/bin/activate && python -m pytest -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add src/graphite/pipeline/extractor.py src/graphite/pipeline/verifier.py \
        src/graphite/pipeline/analyzer.py src/graphite/pipeline/report.py \
        tests/test_extractor.py tests/test_verifier.py tests/test_analyzer.py
git commit -m "refactor: pipeline classes accept shared client= parameter"
```

---

## Task 4: Structured logging (Spec §4)

**Files:**
- Modify: `src/graphite/__init__.py`
- Modify: `src/graphite/pipeline/_client.py` (already has logger from Task 2)

- [ ] **Step 1: Add NullHandler to `__init__.py`**

Add at the end of `src/graphite/__init__.py`, after the `model_rebuild` call:

```python
# ── Logging ──
import logging
logging.getLogger("graphite").addHandler(logging.NullHandler())
```

- [ ] **Step 2: Verify _client.py already has logger from Task 2**

Check that `src/graphite/pipeline/_client.py` has `import logging` and `logger = logging.getLogger(__name__)` from the retry implementation.

- [ ] **Step 3: Add test for NullHandler**

Add to `tests/test_init.py` (or create if needed):

```python
def test_null_handler_attached():
    """Graphite logger should have NullHandler for library silence by default."""
    import logging
    logger = logging.getLogger("graphite")
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)
```

- [ ] **Step 4: Run tests**

Run: `source venv/bin/activate && python -m pytest -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/graphite/__init__.py tests/test_init.py
git commit -m "feat: add structured logging with NullHandler"
```

---

## Task 5: Simplify `_is_anthropic()` (Spec §5)

**Files:**
- Modify: `src/graphite/pipeline/_client.py:21-22,36-45`
- Modify: `tests/test_llm_client.py`

> **Ordering:** This task depends on Task 3 (client sharing) being complete — `client=` injection is the escape hatch.

- [ ] **Step 1: Update tests to match new behavior**

In `tests/test_llm_client.py`, replace `TestLLMClientProviderDetection`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source venv/bin/activate && python -m pytest tests/test_llm_client.py::TestLLMClientProviderDetection -v`
Expected: `test_env_key_does_not_affect_routing` and `test_non_claude_model_selects_openai` FAIL — `my-custom-model` currently returns True when ANTHROPIC_API_KEY is set

- [ ] **Step 3: Simplify `_is_anthropic()`**

In `src/graphite/pipeline/_client.py`, remove the `_NON_ANTHROPIC_PREFIXES` constant (line 22) and replace `_is_anthropic()`:

```python
    def _is_anthropic(self, model: str) -> bool:
        """Determine if a model should use the Anthropic SDK."""
        return model.startswith("claude")
```

- [ ] **Step 4: Run all tests**

Run: `source venv/bin/activate && python -m pytest -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/graphite/pipeline/_client.py tests/test_llm_client.py
git commit -m "refactor: simplify _is_anthropic() to claude-prefix only"
```

---

## Task 6: Final verification

- [ ] **Step 1: Run full test suite**

Run: `source venv/bin/activate && python -m pytest -v`
Expected: All PASS

- [ ] **Step 2: Run examples**

Run: `source venv/bin/activate && python examples/evidence_accumulation/run.py && python examples/lineage_override_demo/run.py`
Expected: Both complete without errors

- [ ] **Step 3: Import smoke test**

Run:
```bash
source venv/bin/activate && python -c "
from graphite import __version__
from graphite.pipeline._client import LLMClient, create_llm_client
print(f'Graphite v{__version__} — all imports OK')
# Verify llm.py is gone
try:
    import graphite.llm
    print('ERROR: graphite.llm should not exist')
except ImportError:
    print('graphite.llm correctly removed')
"
```
Expected: Imports OK, llm.py correctly removed

- [ ] **Step 4: Push to remote**

```bash
git push origin main
```
