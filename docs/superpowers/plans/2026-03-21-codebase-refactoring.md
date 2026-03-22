# Codebase Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor graphite into clean core/pipeline separation, eliminate duplication, harden error handling, and remove dead code.

**Architecture:** Three phases — (1) delete dead code + restructure types, (2) pipeline quality improvements, (3) polish. Each phase produces a working, tested codebase. All existing `from graphite import X` paths preserved via re-exports.

**Tech Stack:** Python, Pydantic, SQLite, pytest

**Spec:** `docs/superpowers/specs/2026-03-21-codebase-refactoring-design.md`

---

## Phase 1: Cleanup & Structure

### Task 1: Delete dead code

**Files:**
- Delete: `benchmark/` (entire directory)
- Delete: `src/graphite/_archive/` (entire directory)
- Delete: `examples/flood_replay_demo/` (entire directory)
- Delete: `examples/forecast_demo/` (entire directory)
- Delete: `examples/toy_battery_demo/` (entire directory)
- Delete: `examples/ablation/` (entire directory)

- [ ] **Step 1: Delete directories**

```bash
cd /Users/minjun/graf/graphite
rm -rf benchmark/
rm -rf src/graphite/_archive/
rm -rf examples/flood_replay_demo/
rm -rf examples/forecast_demo/
rm -rf examples/toy_battery_demo/
rm -rf examples/ablation/
```

- [ ] **Step 2: Verify tests still pass**

Run: `cd /Users/minjun/graf/graphite && pytest -q`
Expected: All 222 tests pass (none depended on deleted code)

- [ ] **Step 3: Remove _archive exclude from pyproject.toml**

In `pyproject.toml`, find and remove the `_archive` exclude pattern from `[tool.setuptools.packages.find]` if present.

- [ ] **Step 4: Commit**

```bash
cd /Users/minjun/graf/graphite
git add -A benchmark/ src/graphite/_archive/ examples/flood_replay_demo/ examples/forecast_demo/ examples/toy_battery_demo/ examples/ablation/ pyproject.toml
git commit -m "chore: remove dead code (benchmark, _archive, old examples)"
```

---

### Task 2: Move claim enums to enums.py

**Files:**
- Modify: `src/graphite/enums.py`
- Modify: `src/graphite/claim.py`
- Modify: `src/graphite/__init__.py`

- [ ] **Step 1: Add claim enums to enums.py**

Add these enum classes to `src/graphite/enums.py` (after the existing enums):

```python
class ClaimType(str, Enum):
    """What kind of assertion this claim makes."""
    RELATIONSHIP = "RELATIONSHIP"
    ATTRIBUTE = "ATTRIBUTE"
    RISK_ASSERTION = "RISK_ASSERTION"
    DEPENDENCY = "DEPENDENCY"


class ClaimStatus(str, Enum):
    """Trust verdict for a claim — typically computed, not manually set."""
    SUPPORTED = "SUPPORTED"
    WEAK = "WEAK"
    MIXED = "MIXED"
    UNSUPPORTED = "UNSUPPORTED"
    PENDING_REVIEW = "PENDING_REVIEW"


class ReviewState(str, Enum):
    """Analyst review workflow state."""
    UNREVIEWED = "UNREVIEWED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NEEDS_FOLLOWUP = "NEEDS_FOLLOWUP"


class ClaimOrigin(str, Enum):
    """How this claim was created."""
    EXTRACTOR = "EXTRACTOR"
    AGENT = "AGENT"
    RULE_ENGINE = "RULE_ENGINE"
    ANALYST = "ANALYST"
    IMPORTED = "IMPORTED"


class ClaimGranularity(str, Enum):
    """Abstraction level of a claim."""
    ATOMIC = "ATOMIC"
    SYNTHESIZED = "SYNTHESIZED"
    THESIS = "THESIS"
```

- [ ] **Step 2: Update claim.py to import from enums.py**

In `src/graphite/claim.py`:
- Remove the 5 enum class definitions (`ClaimType`, `ClaimStatus`, `ReviewState`, `ClaimOrigin`, `ClaimGranularity`) and the `# Claim Enums` section header
- Replace with imports at top:

```python
from .enums import (
    AssertionMode, ConfidenceLevel,
    ClaimType, ClaimStatus, ReviewState, ClaimOrigin, ClaimGranularity,
)
```

Remove the old import line `from .enums import AssertionMode, ConfidenceLevel`.

- [ ] **Step 3: Update __init__.py enum imports**

In `src/graphite/__init__.py`, change the enums import block to:

```python
from .enums import (
    EdgeType, NodeType, SourceType, ConfidenceLevel, AssertionMode,
    EvidenceType, ClaimType, ClaimStatus, ReviewState, ClaimOrigin,
    ClaimGranularity,
)
```

And remove `ClaimType, ClaimStatus, ClaimGranularity, ReviewState, ClaimOrigin` from the `from .claim import (...)` block, keeping only `Claim`.

- [ ] **Step 4: Run tests**

Run: `cd /Users/minjun/graf/graphite && pytest -q`
Expected: All tests pass. Imports like `from graphite.claim import ClaimStatus` still work because `claim.py` re-imports from `enums.py`.

- [ ] **Step 5: Commit**

```bash
cd /Users/minjun/graf/graphite
git add src/graphite/enums.py src/graphite/claim.py src/graphite/__init__.py
git commit -m "refactor: move claim enums to enums.py"
```

---

### Task 3: Create pipeline/verdict.py and move verdict types

**Files:**
- Create: `src/graphite/pipeline/verdict.py`
- Modify: `src/graphite/claim.py`
- Modify: `src/graphite/pipeline/__init__.py`
- Modify: `src/graphite/__init__.py`

- [ ] **Step 1: Create pipeline/verdict.py**

Create `src/graphite/pipeline/verdict.py` with the verdict types moved from `claim.py`:

```python
"""
graphite/pipeline/verdict.py — LLM-native verification verdict types.

These types represent the output of the verification pipeline:
claim-level verdicts, argument-level verdicts, and the aggregated report.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VerdictEnum(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONFLICTED = "CONFLICTED"
    INSUFFICIENT = "INSUFFICIENT"


class ArgumentVerdictEnum(str, Enum):
    GROUNDED = "GROUNDED"
    CONCLUSION_JUMP = "CONCLUSION_JUMP"
    OVERSTATED = "OVERSTATED"


class VerdictRationale(BaseModel):
    """Structured reasoning slots for meta-evaluations and transparent auditing."""
    contradiction_type: Optional[str] = Field(
        default=None, description="e.g., numeric mismatch, entity mismatch"
    )
    missing_evidence_reason: Optional[str] = Field(
        default=None, description="Why the evidence was insufficient or lacking"
    )
    temporal_alignment: Optional[str] = Field(
        default=None, description="e.g., stale evidence vs current claim timeline"
    )
    text: str = Field(description="Free-form rationale from the LLM judge")


class Verdict(BaseModel):
    """A claim-level judgment returned by the verifier pipeline."""
    claim_id: str
    claim_text: str
    verdict: VerdictEnum
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    conflicting_evidence_ids: List[str] = Field(default_factory=list)
    rationale: VerdictRationale
    needs_human_review: bool = Field(
        default=False,
        description="Flag for high-risk or low-confidence verdicts to route to human queues",
    )
    cited_span: Optional[str] = Field(
        default=None,
        description="The exact span from the evidence corpus cited to make the verdict",
    )
    model_version: str = Field(default="")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ArgumentVerdict(BaseModel):
    """An argument-level verification representing semantic logic jumps."""
    text: str = Field(description="The argument chain or conclusion being evaluated")
    verdict: ArgumentVerdictEnum
    rationale: VerdictRationale
    needs_human_review: bool = Field(default=False)


class VerificationReport(BaseModel):
    """Top-level review object aggregating the entire verification workflow."""
    document_id: str
    total_claims: int = 0
    supported_count: int = 0
    conflicted_count: int = 0
    insufficient_count: int = 0
    grounded_argument_count: int = 0
    conclusion_jump_count: int = 0
    risky_claim_ids: List[str] = Field(
        default_factory=list, description="Claim IDs flagged for human review"
    )
    evidence_coverage_score: float = 0.0
    verdicts: List[Verdict] = Field(default_factory=list)
    argument_verdicts: List[ArgumentVerdict] = Field(default_factory=list)
    model_metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_verdict(self, claim_id: str) -> Optional[Verdict]:
        for v in self.verdicts:
            if v.claim_id == claim_id:
                return v
        return None
```

- [ ] **Step 2: Remove verdict types from claim.py, add re-exports**

In `src/graphite/claim.py`:
- Remove the entire `# LLM-Native Verification Report Models` section (everything from line 382 to end: `VerdictEnum`, `ArgumentVerdictEnum`, `VerdictRationale`, `Verdict`, `ArgumentVerdict`, `VerificationReport`)
- Add re-exports at the bottom of the file:

```python
# ═══════════════════════════════════════
# Backward compat re-exports (to be removed in next major version)
# ═══════════════════════════════════════
from .pipeline.verdict import (  # noqa: E402, F401
    VerdictEnum,
    ArgumentVerdictEnum,
    VerdictRationale,
    Verdict,
    ArgumentVerdict,
    VerificationReport,
)
```

- [ ] **Step 3: Update pipeline/__init__.py**

In `src/graphite/pipeline/__init__.py`, add verdict type re-exports:

```python
"""
graphite/pipeline/__init__.py — Exposes the LLM-native verification pipeline.
"""

from .report import verify_agent_output, review_document
from .extractor import extract_claims
from .retriever import retrieve_evidence
from .verifier import verify_claims
from .analyzer import analyze_argument_chain
from .verdict import (
    VerdictEnum,
    ArgumentVerdictEnum,
    VerdictRationale,
    Verdict,
    ArgumentVerdict,
    VerificationReport,
)

__all__ = [
    "verify_agent_output",
    "review_document",
    "extract_claims",
    "retrieve_evidence",
    "verify_claims",
    "analyze_argument_chain",
    "VerdictEnum",
    "ArgumentVerdictEnum",
    "VerdictRationale",
    "Verdict",
    "ArgumentVerdict",
    "VerificationReport",
]
```

- [ ] **Step 4: Update root __init__.py**

In `src/graphite/__init__.py`, add verdict re-exports. Replace the existing `from .claim import (...)` block with:

```python
# ── Trust engine primitives ──
from .claim import Claim
from .claim import ConfidenceFactor, ConfidenceResult

# ── Pipeline verdict types (re-exported for convenience) ──
from .pipeline.verdict import (
    VerdictEnum, ArgumentVerdictEnum, VerdictRationale,
    Verdict, ArgumentVerdict, VerificationReport,
)
```

- [ ] **Step 5: Update pipeline internal imports**

In `src/graphite/pipeline/verifier.py`, change:
```python
from graphite.claim import Claim, Verdict, VerdictEnum, VerdictRationale
```
to:
```python
from graphite.claim import Claim
from graphite.pipeline.verdict import Verdict, VerdictEnum, VerdictRationale
```

In `src/graphite/pipeline/analyzer.py`, change:
```python
from graphite.claim import (
    Claim,
    Verdict,
    ArgumentVerdict,
    ArgumentVerdictEnum,
    VerdictRationale,
)
```
to:
```python
from graphite.pipeline.verdict import (
    Verdict,
    ArgumentVerdict,
    ArgumentVerdictEnum,
    VerdictRationale,
)
```

In `src/graphite/pipeline/report.py`, change:
```python
from graphite.claim import VerificationReport, VerdictEnum, ArgumentVerdictEnum
```
to:
```python
from graphite.pipeline.verdict import VerificationReport, VerdictEnum, ArgumentVerdictEnum
```

- [ ] **Step 6: Run tests**

Run: `cd /Users/minjun/graf/graphite && pytest -q`
Expected: All tests pass. Both `from graphite.claim import Verdict` and `from graphite.pipeline.verdict import Verdict` work.

- [ ] **Step 7: Commit**

```bash
cd /Users/minjun/graf/graphite
git add src/graphite/pipeline/verdict.py src/graphite/claim.py src/graphite/pipeline/__init__.py src/graphite/__init__.py src/graphite/pipeline/verifier.py src/graphite/pipeline/analyzer.py src/graphite/pipeline/report.py
git commit -m "refactor: move verdict types to pipeline/verdict.py"
```

---

## Phase 2: Pipeline Quality

### Task 4: Extract shared OpenAI client

**Files:**
- Create: `src/graphite/pipeline/_client.py`
- Modify: `src/graphite/pipeline/extractor.py`
- Modify: `src/graphite/pipeline/verifier.py`
- Modify: `src/graphite/pipeline/analyzer.py`

- [ ] **Step 1: Create pipeline/_client.py**

Create `src/graphite/pipeline/_client.py`:

```python
"""Shared OpenAI-compatible client factory for pipeline modules."""

import os
from typing import Optional


def create_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Create an OpenAI client with Graphite's default configuration.

    Priority: explicit args > GEMINI_API_KEY > OPENAI_API_KEY.
    Default base_url points to Google's OpenAI-compatible endpoint.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            'OpenAI client required for pipeline. Install with: pip install "graphite-engine[llm]"'
        )

    return OpenAI(
        api_key=api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENAI_API_KEY"),
        base_url=base_url
        or os.environ.get(
            "OPENAI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
    )
```

- [ ] **Step 2: Update extractor.py**

In `src/graphite/pipeline/extractor.py`, replace the `__init__` method of `ClaimExtractor`:

```python
class ClaimExtractor:
    """Extracts atomic claims from documents using an OpenAI-compatible LLM."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        from graphite.pipeline._client import create_openai_client
        self.client = create_openai_client(api_key=api_key, base_url=base_url)
```

Remove the `os` import if no longer used elsewhere in the file.

- [ ] **Step 3: Update verifier.py**

In `src/graphite/pipeline/verifier.py`, replace the `__init__` method of `ClaimVerifier`:

```python
class ClaimVerifier:
    """Evaluates claims against retrieved evidence spans."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        from graphite.pipeline._client import create_openai_client
        self.client = create_openai_client(api_key=api_key, base_url=base_url)
```

Remove the `os` import if no longer used elsewhere.

- [ ] **Step 4: Update analyzer.py**

In `src/graphite/pipeline/analyzer.py`, replace the `__init__` method of `ArgumentAnalyzer`:

```python
class ArgumentAnalyzer:
    """Analyzes the overall argument chain for unsupported conclusions."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        from graphite.pipeline._client import create_openai_client
        self.client = create_openai_client(api_key=api_key, base_url=base_url)
```

Remove the `os` import if no longer used elsewhere.

- [ ] **Step 5: Run tests**

Run: `cd /Users/minjun/graf/graphite && pytest -q`
Expected: All tests pass (pipeline tests mock the OpenAI client).

- [ ] **Step 6: Commit**

```bash
cd /Users/minjun/graf/graphite
git add src/graphite/pipeline/_client.py src/graphite/pipeline/extractor.py src/graphite/pipeline/verifier.py src/graphite/pipeline/analyzer.py
git commit -m "refactor: extract shared OpenAI client to pipeline/_client.py"
```

---

### Task 5: Fix llm.py JSON fence stripping

**Files:**
- Modify: `src/graphite/llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: Write failing tests for edge cases**

Add to `tests/test_llm.py` (in the appropriate test class):

```python
def test_strips_fence_with_trailing_newline(self, mock_genai):
    """Regression: trailing newline after closing fence."""
    mock_response = MagicMock()
    mock_response.text = '```json\n{"value": 42}\n```\n'
    mock_genai.return_value.models.generate_content.return_value = mock_response

    result = gemini_extract_json("test", "prompt")
    assert result == {"value": 42}

def test_strips_fence_without_json_tag(self, mock_genai):
    """Handles ``` without json tag."""
    mock_response = MagicMock()
    mock_response.text = '```\n{"value": 42}\n```'
    mock_genai.return_value.models.generate_content.return_value = mock_response

    result = gemini_extract_json("test", "prompt")
    assert result == {"value": 42}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/minjun/graf/graphite && pytest tests/test_llm.py -v -k "trailing_newline or without_json_tag"`
Expected: At least one FAIL (trailing newline case breaks with current slicing)

- [ ] **Step 3: Add strip_markdown_fences to llm.py**

In `src/graphite/llm.py`, add after the imports:

```python
import re

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)```\s*$", re.DOTALL)


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM response text."""
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text
```

Then replace the fence-stripping blocks in both `gemini_extract_structured()` and `gemini_extract_json()`.

In `gemini_extract_structured()` (around line 99-103), replace:
```python
raw = response.text.strip()
if raw.startswith("```json"):
    raw = raw[7:-3].strip()
elif raw.startswith("```"):
    raw = raw[3:-3].strip()
```
with:
```python
raw = strip_markdown_fences(response.text)
```

In `gemini_extract_json()` (around line 145-149), replace the same pattern with:
```python
raw = strip_markdown_fences(response.text)
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/minjun/graf/graphite && pytest tests/test_llm.py -v`
Expected: All tests pass including the new edge case tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/minjun/graf/graphite
git add src/graphite/llm.py tests/test_llm.py
git commit -m "fix: use regex for markdown fence stripping in llm.py"
```

---

### Task 6: Add JSON error handling in pipeline modules

**Files:**
- Modify: `src/graphite/pipeline/extractor.py`
- Modify: `src/graphite/pipeline/verifier.py`
- Modify: `src/graphite/pipeline/analyzer.py`
- Test: `tests/test_extractor.py`
- Test: `tests/test_verifier.py`
- Test: `tests/test_analyzer.py`

- [ ] **Step 1: Write failing tests for malformed JSON**

Add to `tests/test_extractor.py`:

```python
def test_malformed_json_raises_value_error(self):
    """Malformed LLM response raises ValueError, not JSONDecodeError."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "not valid json {{"

    with patch("graphite.pipeline._client.create_openai_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        extractor = ClaimExtractor()
        with pytest.raises(ValueError, match="LLM returned invalid JSON"):
            extractor.extract_claims("test document")
```

Add similar tests to `tests/test_verifier.py` and `tests/test_analyzer.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/minjun/graf/graphite && pytest -k "malformed_json" -v`
Expected: FAIL (currently raises `json.JSONDecodeError`, not `ValueError`)

- [ ] **Step 3: Add error handling to extractor.py**

In `src/graphite/pipeline/extractor.py`, in `ClaimExtractor.extract_claims()`, replace:
```python
raw_claims = json.loads(response.choices[0].message.content).get("claims", [])
```
with:
```python
try:
    raw_claims = json.loads(response.choices[0].message.content).get("claims", [])
except json.JSONDecodeError as e:
    raise ValueError(
        f"LLM returned invalid JSON: {e}. "
        f"Raw response: {response.choices[0].message.content[:200]}"
    ) from e
```

- [ ] **Step 4: Add error handling to verifier.py**

In `src/graphite/pipeline/verifier.py`, in `ClaimVerifier.verify_claims()`, replace:
```python
res = json.loads(response.choices[0].message.content)
```
with:
```python
try:
    res = json.loads(response.choices[0].message.content)
except json.JSONDecodeError as e:
    raise ValueError(
        f"LLM returned invalid JSON: {e}. "
        f"Raw response: {response.choices[0].message.content[:200]}"
    ) from e
```

- [ ] **Step 5: Add error handling to analyzer.py**

In `src/graphite/pipeline/analyzer.py`, in `ArgumentAnalyzer.analyze_argument_chain()`, replace:
```python
res_data = json.loads(response.choices[0].message.content)
```
with:
```python
try:
    res_data = json.loads(response.choices[0].message.content)
except json.JSONDecodeError as e:
    raise ValueError(
        f"LLM returned invalid JSON: {e}. "
        f"Raw response: {response.choices[0].message.content[:200]}"
    ) from e
```

- [ ] **Step 6: Run tests**

Run: `cd /Users/minjun/graf/graphite && pytest -q`
Expected: All tests pass including new malformed JSON tests.

- [ ] **Step 7: Commit**

```bash
cd /Users/minjun/graf/graphite
git add src/graphite/pipeline/extractor.py src/graphite/pipeline/verifier.py src/graphite/pipeline/analyzer.py tests/test_extractor.py tests/test_verifier.py tests/test_analyzer.py
git commit -m "fix: add JSON error handling in pipeline modules"
```

---

### Task 7: ClaimStore batch efficiency

**Files:**
- Modify: `src/graphite/claim_store.py`
- Test: `tests/test_claim_store.py`

- [ ] **Step 1: Refactor ClaimStore to thread connection**

In `src/graphite/claim_store.py`, refactor as follows:

1. Rename current `get_claim()` to `_get_claim_with_conn()` that takes a `conn` parameter:

```python
def _get_claim_with_conn(self, claim_id: str, conn: sqlite3.Connection) -> Optional[Claim]:
    """Retrieve a claim using provided connection."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT full_json FROM claims WHERE claim_id = ?", (claim_id,)
    )
    row = cursor.fetchone()
    if row:
        return Claim.model_validate_json(row[0])
    return None
```

2. Keep public `get_claim()` as wrapper:

```python
def get_claim(self, claim_id: str) -> Optional[Claim]:
    """Retrieve a claim by its exact ID."""
    with sqlite3.connect(self.db_path) as conn:
        return self._get_claim_with_conn(claim_id, conn)
```

3. Rename `_write_claim()` to `_write_claim_with_conn()` that takes a `conn` parameter (remove `with sqlite3.connect(...)` wrapper, just use cursor from conn):

```python
def _write_claim_with_conn(self, claim: Claim, conn: sqlite3.Connection) -> None:
    """Write a claim to SQLite using provided connection."""
    cursor = conn.cursor()
    subjects_str = json.dumps(claim.subject_entities)
    objects_str = json.dumps(claim.object_entities)
    score = claim.confidence.score if claim.confidence else 0.0
    full_json = claim.model_dump_json()
    cursor.execute(
        """
        INSERT INTO claims (
            claim_id, claim_text, claim_type, granularity,
            subject_entities, predicate, object_entities,
            as_of_date, computed_status, confidence_score, full_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(claim_id) DO UPDATE SET
            claim_text=excluded.claim_text,
            as_of_date=excluded.as_of_date,
            computed_status=excluded.computed_status,
            confidence_score=excluded.confidence_score,
            full_json=excluded.full_json
        """,
        (
            claim.claim_id, claim.claim_text, claim.claim_type.value,
            claim.granularity.value, subjects_str, claim.predicate,
            objects_str, claim.as_of_date, claim.computed_status.value,
            score, full_json,
        ),
    )
```

4. Create `_save_claim_with_conn()`:

```python
def _save_claim_with_conn(self, claim: Claim, conn: sqlite3.Connection) -> None:
    """Core save logic with evidence merge, using provided connection."""
    existing = self._get_claim_with_conn(claim.claim_id, conn)
    if existing is not None:
        # Merge evidence (same logic as current save_claim)
        existing.supporting_evidence = self._merge_evidence(
            existing.supporting_evidence, claim.supporting_evidence,
        )
        existing.weakening_evidence = self._merge_evidence(
            existing.weakening_evidence, claim.weakening_evidence,
        )
        existing.claim_text = claim.claim_text or existing.claim_text
        existing.as_of_date = claim.as_of_date or existing.as_of_date
        if claim.confidence is not None:
            existing.confidence = claim.confidence
        existing.computed_status = claim.computed_status
        if claim.is_overridden:
            existing.final_status = claim.final_status
            existing.override_reason = claim.override_reason
            existing.review_state = claim.review_state
            existing.reviewed_by = claim.reviewed_by
            existing.reviewed_at = claim.reviewed_at
            existing.reviewer_note = claim.reviewer_note
        if claim.generator_id:
            existing.generator_id = claim.generator_id
        if claim.generation_metadata:
            existing.generation_metadata = {
                **existing.generation_metadata, **claim.generation_metadata,
            }
        claim = existing
    self._write_claim_with_conn(claim, conn)
```

5. Update public methods:

```python
def save_claim(self, claim: Claim) -> None:
    """Save a single claim with evidence accumulation."""
    with sqlite3.connect(self.db_path) as conn:
        self._save_claim_with_conn(claim, conn)
        conn.commit()

def save_claims(self, claims: List[Claim]) -> None:
    """Batch save multiple claims with single connection."""
    with sqlite3.connect(self.db_path) as conn:
        for claim in claims:
            self._save_claim_with_conn(claim, conn)
        conn.commit()
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/minjun/graf/graphite && pytest tests/test_claim_store.py -v`
Expected: All tests pass (public API unchanged).

- [ ] **Step 3: Commit**

```bash
cd /Users/minjun/graf/graphite
git add src/graphite/claim_store.py
git commit -m "refactor: ClaimStore uses single connection for batch ops"
```

---

### Task 8: Consolidate RuleResult / RuleResultModel

**Files:**
- Modify: `src/graphite/rules.py`
- Modify: `src/graphite/evidence.py`
- Modify: `tests/test_evidence.py`
- Modify: `tests/test_rules.py`

- [ ] **Step 1: Convert RuleResult to Pydantic in rules.py**

In `src/graphite/rules.py`:

Replace:
```python
from dataclasses import dataclass, field
```
with:
```python
from dataclasses import field
from pydantic import BaseModel
```

Replace:
```python
@dataclass
class RuleResult:
    """Result from evaluating a single rule against an edge."""
    rule_id: str
    rule_name: str
    triggered: bool
    weight_delta: float
    explanation: str
    category: str = ""
```
with:
```python
class RuleResult(BaseModel):
    """Result from evaluating a single rule against an edge."""
    rule_id: str
    rule_name: str
    triggered: bool
    weight_delta: float
    explanation: str
    category: str = ""
```

Note: `ScoreBreakdown` stays as a dataclass (it uses `field(default_factory=list)` and `@property`). `ScoreBreakdown.rule_results` typed as `List[RuleResult]` is compatible since `RuleResult` is still constructable the same way.

- [ ] **Step 2: Remove RuleResultModel from evidence.py**

In `src/graphite/evidence.py`:

Remove the `RuleResultModel` class entirely (lines 40-48).

Add import:
```python
from .rules import RuleResult
```

Change `ScoringData.rule_details` type:
```python
rule_details: List[RuleResult] = []
```

- [ ] **Step 3: Update test_evidence.py**

In `tests/test_evidence.py`:

Change:
```python
from graphite.evidence import (
    EvidenceData, RuleResultModel, ScoringData,
    CounterEvidence, EvidencePacket,
)
```
to:
```python
from graphite.evidence import (
    EvidenceData, ScoringData,
    CounterEvidence, EvidencePacket,
)
from graphite.rules import RuleResult
```

In the test class, rename `TestRuleResultModel` to `TestRuleResult` and change all `RuleResultModel(...)` calls to `RuleResult(...)`.

- [ ] **Step 4: Update test_rules.py if needed**

Check if `test_rules.py` uses any dataclass-specific features like `asdict()`. If so, replace with `.model_dump()`. Current tests construct `RuleResult(...)` positionally — Pydantic `BaseModel` also supports keyword construction, so these should work. If any test uses positional args, switch to keyword args.

- [ ] **Step 5: Run tests**

Run: `cd /Users/minjun/graf/graphite && pytest tests/test_evidence.py tests/test_rules.py -v`
Expected: All tests pass.

Run: `cd /Users/minjun/graf/graphite && pytest -q`
Expected: Full suite passes.

- [ ] **Step 6: Commit**

```bash
cd /Users/minjun/graf/graphite
git add src/graphite/rules.py src/graphite/evidence.py tests/test_evidence.py tests/test_rules.py
git commit -m "refactor: consolidate RuleResult into single Pydantic model"
```

---

## Phase 3: Polish

### Task 9: SHA-1 → SHA-256 in text.py

**Files:**
- Modify: `src/graphite/text.py`
- Modify: `tests/test_text.py`

- [ ] **Step 1: Deprecate sha1_hex and switch internals**

In `src/graphite/text.py`:

Update `sha1_hex`:
```python
def sha1_hex(s: str) -> str:
    """Deprecated: use sha256_hex instead."""
    return hashlib.sha1(s.encode()).hexdigest()
```

In `find_best_paragraph_for_quote()`, replace both occurrences of `sha1_hex` with `sha256_hex`:
- Line 120: `return i, sha256_hex(p)[:12]`
- Line 131: `return best_idx, sha256_hex(paragraphs[best_idx])[:12]`

- [ ] **Step 2: Update test_text.py**

In `tests/test_text.py`, the `test_exact_substring_match` and `test_word_overlap_fallback` tests check `len(hash_val) == 12`. This still holds since `sha256_hex()[:12]` is also 12 chars. No change needed for those.

The `test_sha1_deterministic` test should remain (it tests the function still works even though deprecated).

- [ ] **Step 3: Run tests**

Run: `cd /Users/minjun/graf/graphite && pytest tests/test_text.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/minjun/graf/graphite
git add src/graphite/text.py tests/test_text.py
git commit -m "refactor: switch internal hashing from SHA-1 to SHA-256"
```

---

### Task 10: Testable recency in confidence.py

**Files:**
- Modify: `src/graphite/confidence.py`
- Modify: `tests/test_confidence.py`

- [ ] **Step 1: Write test using explicit `now`**

Add to `tests/test_confidence.py`:

```python
from datetime import datetime, timezone, timedelta

def test_recency_reproducible_with_now(self):
    """score() with explicit `now` produces identical results."""
    claim = self._make_claim_with_date(
        (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    )
    fixed_now = datetime(2026, 3, 21, tzinfo=timezone.utc)
    scorer = ConfidenceScorer()
    r1 = scorer.score(claim, now=fixed_now)
    r2 = scorer.score(claim, now=fixed_now)
    assert r1.score == r2.score
```

(The helper `_make_claim_with_date` may need to be created or adapted from existing test helpers.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/minjun/graf/graphite && pytest tests/test_confidence.py -k "reproducible_with_now" -v`
Expected: FAIL (`score()` doesn't accept `now` parameter)

- [ ] **Step 3: Add `now` parameter**

In `src/graphite/confidence.py`:

Add `Optional` and `datetime` to imports if not present.

Update `score()` signature:
```python
def score(self, claim: "Claim", now: Optional[datetime] = None) -> ConfidenceResult:
```

Pass `now` to `_compute_recency`:
```python
recency_score = self._compute_recency(all_evidence, now=now)
```

Update `_compute_recency()`:
```python
def _compute_recency(self, evidence: list, now: Optional[datetime] = None) -> float:
    """Score evidence recency (0.0 = very old, 1.0 = current)."""
    now = now or datetime.now(timezone.utc)
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/minjun/graf/graphite && pytest tests/test_confidence.py -v`
Expected: All pass including new test.

- [ ] **Step 5: Commit**

```bash
cd /Users/minjun/graf/graphite
git add src/graphite/confidence.py tests/test_confidence.py
git commit -m "refactor: add optional now parameter for reproducible recency scoring"
```

---

### Task 11: Fix EvidencePacket forward references

**Files:**
- Modify: `src/graphite/__init__.py`

- [ ] **Step 1: Add model_rebuild() to __init__.py**

In `src/graphite/__init__.py`, add at the very end of the file (after all imports):

```python
# Resolve forward references in EvidencePacket
# (must be after Claim and ConfidenceResult are imported)
EvidencePacket.model_rebuild()
```

- [ ] **Step 2: Verify test_evidence.py no longer needs manual model_rebuild**

Check if `tests/test_evidence.py` calls `EvidencePacket.model_rebuild()` — if so, it should now be optional (but safe to keep).

- [ ] **Step 3: Run tests**

Run: `cd /Users/minjun/graf/graphite && pytest tests/test_evidence.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/minjun/graf/graphite
git add src/graphite/__init__.py
git commit -m "fix: auto-resolve EvidencePacket forward refs in __init__.py"
```

---

### Task 12: Error path tests for pipeline

**Files:**
- Modify: `tests/test_extractor.py`
- Modify: `tests/test_verifier.py`
- Modify: `tests/test_analyzer.py`
- Modify: `tests/test_report.py`

- [ ] **Step 1: Add empty input tests**

Add to `tests/test_extractor.py`:
```python
def test_empty_document(self):
    """Empty string still calls LLM and returns empty claims list."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"claims": []}'
    with patch("graphite.pipeline._client.create_openai_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        extractor = ClaimExtractor()
        result = extractor.extract_claims("")
        assert result == []
```

Add to `tests/test_verifier.py`:
```python
def test_empty_claims_list(self):
    """Empty claims list returns empty verdicts."""
    with patch("graphite.pipeline._client.create_openai_client"):
        verifier = ClaimVerifier()
        result = verifier.verify_claims([], {})
        assert result == []
```

Add to `tests/test_analyzer.py`:
```python
def test_empty_verdicts_list(self):
    """Empty verdicts list is handled gracefully."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"argument_verdicts": []}'
    with patch("graphite.pipeline._client.create_openai_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        analyzer = ArgumentAnalyzer()
        result = analyzer.analyze_argument_chain("memo", [])
        assert result == []
```

- [ ] **Step 2: Add unexpected verdict string tests**

Add to `tests/test_verifier.py`:
```python
def test_unexpected_verdict_string_defaults_to_insufficient(self):
    """Unknown verdict string falls back to INSUFFICIENT."""
    # (mock LLM returning verdict: "MAYBE" instead of valid enum)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "verdict": "MAYBE",
        "rationale_text": "test",
        "supporting_evidence_indices": [],
        "conflicting_evidence_indices": [],
    })
    # ... (create claim, call verify_claims, assert verdict == VerdictEnum.INSUFFICIENT)
```

- [ ] **Step 3: Run all tests**

Run: `cd /Users/minjun/graf/graphite && pytest -q`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/minjun/graf/graphite
git add tests/test_extractor.py tests/test_verifier.py tests/test_analyzer.py tests/test_report.py
git commit -m "test: add error path tests for pipeline modules"
```

---

### Task 13: Update examples and README

**Files:**
- Modify: `examples/quickstart_verification/run.py`
- Modify: `README.md`

- [ ] **Step 1: Update quickstart example imports**

In `examples/quickstart_verification/run.py`, change line 17:
```python
from graphite.claim import ArgumentVerdictEnum, VerdictEnum
```
to:
```python
from graphite.pipeline.verdict import ArgumentVerdictEnum, VerdictEnum
```

- [ ] **Step 2: Update README.md code snippets**

In `README.md`, find any `from graphite.claim import` lines that reference verdict types and update to use `from graphite.pipeline.verdict import` or `from graphite import`.

- [ ] **Step 3: Commit**

```bash
cd /Users/minjun/graf/graphite
git add examples/quickstart_verification/run.py README.md
git commit -m "docs: update imports in examples and README"
```

---

### Task 14: Final verification

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/minjun/graf/graphite && pytest -v`
Expected: All tests pass.

- [ ] **Step 2: Verify backward compat imports**

Run: `cd /Users/minjun/graf/graphite && python -c "from graphite.claim import Verdict, VerdictEnum, ClaimStatus; print('OK')"`
Expected: `OK`

Run: `cd /Users/minjun/graf/graphite && python -c "from graphite import Verdict, VerificationReport, ClaimStatus; print('OK')"`
Expected: `OK`

Run: `cd /Users/minjun/graf/graphite && python -c "from graphite.pipeline.verdict import Verdict, VerdictEnum; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Verify no circular imports**

Run: `cd /Users/minjun/graf/graphite && python -c "import graphite; print(graphite.__version__)"`
Expected: `0.3.2` (no import errors)
