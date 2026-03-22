# Graphite Codebase Refactoring — Design Spec

**Date:** 2026-03-21
**Scope:** 19 issues (2 critical, 9 important, 8 suggestions) from comprehensive code review

## Goals

1. Clean separation between core primitives and pipeline layer
2. Eliminate code duplication across pipeline modules
3. Harden LLM response parsing
4. Improve DB efficiency and type consistency
5. Remove dead code (benchmark/, _archive/, deleted examples)

## Non-Goals

- New features or API additions
- Benchmark redesign (planned separately at larger scale)
- Performance optimization beyond DB connection pooling

---

## Phase 1: Cleanup & Structure

### 1.1 Delete dead code

Remove directories and files no longer needed:

- `benchmark/` — pilot-scale SEC benchmark, to be redesigned
- `src/graphite/_archive/` — old supply-chain graph code (preserved in git history)
- `examples/flood_replay_demo/`, `examples/forecast_demo/`, `examples/toy_battery_demo/` — supply-chain domain examples
- `examples/ablation/` — deleted in working tree

Keep: `evals/verify_eval.py` (pipeline regression test)

### 1.2 Move Verdict types to pipeline

**From** `claim.py` → **To** `pipeline/verdict.py`:
- `VerdictEnum`
- `ArgumentVerdictEnum`
- `VerdictRationale`
- `Verdict`
- `ArgumentVerdict`
- `VerificationReport`

**Import updates (pipeline internals):**
- `pipeline/verifier.py`: `from graphite.pipeline.verdict import ...`
- `pipeline/analyzer.py`: `from graphite.pipeline.verdict import ...`
- `pipeline/report.py`: `from graphite.pipeline.verdict import ...`
- `pipeline/__init__.py`: re-export verdict types (`Verdict`, `VerificationReport`, `VerdictEnum`, `ArgumentVerdictEnum`, `VerdictRationale`, `ArgumentVerdict`)

**Backward compatibility — two layers of re-export:**
1. `__init__.py` (root): `from .pipeline.verdict import ...` — preserves `from graphite import Verdict`
2. `claim.py`: add re-exports at bottom — preserves `from graphite.claim import Verdict` for existing code

```python
# claim.py — backward compat re-exports (to be removed in next major version)
from .pipeline.verdict import (
    VerdictEnum, ArgumentVerdictEnum, VerdictRationale,
    Verdict, ArgumentVerdict, VerificationReport,
)
```

**All files requiring import updates:**
- `pipeline/verifier.py` — imports `Verdict, VerdictEnum, VerdictRationale`
- `pipeline/analyzer.py` — imports `Verdict, ArgumentVerdict, ArgumentVerdictEnum, VerdictRationale`
- `pipeline/report.py` — imports `VerificationReport, VerdictEnum, ArgumentVerdictEnum`
- `tests/test_verifier.py` — imports `VerdictEnum` from `graphite.claim`
- `tests/test_analyzer.py` — imports `Verdict, VerdictEnum, VerdictRationale, ArgumentVerdictEnum`
- `tests/test_report.py` — imports `Verdict, VerdictEnum, VerdictRationale, ArgumentVerdict, ArgumentVerdictEnum, VerificationReport`
- `examples/quickstart_verification/run.py` — imports `ArgumentVerdictEnum, VerdictEnum`
- `README.md` — code snippet with `from graphite.claim import ArgumentVerdictEnum`

### 1.3 Consolidate enums

**Move from `claim.py` to `enums.py`:**
- `ClaimType`
- `ClaimStatus`
- `ReviewState`
- `ClaimOrigin`
- `ClaimGranularity`

**Move from `claim.py` to `pipeline/verdict.py`** (already done in 1.2):
- `VerdictEnum`
- `ArgumentVerdictEnum`

**Result:** `claim.py` imports all its enums from `enums.py`. No enum definitions remain in `claim.py`.

**`__init__.py` re-exports:** All existing public names remain importable from `graphite` — no breaking changes for `from graphite import ClaimStatus` etc.

### 1.4 Update `__init__.py`

```python
# Core enums (all from enums.py now)
from .enums import (
    EdgeType, NodeType, SourceType, ConfidenceLevel, AssertionMode,
    EvidenceType, ClaimType, ClaimStatus, ReviewState, ClaimOrigin,
    ClaimGranularity,
)

# Pipeline verdict types (re-exported for convenience)
from .pipeline.verdict import (
    VerdictEnum, ArgumentVerdictEnum, VerdictRationale,
    Verdict, ArgumentVerdict, VerificationReport,
)
```

---

## Phase 2: Pipeline Quality

### 2.1 Extract shared OpenAI client → `pipeline/_client.py`

Current state: `extractor.py`, `verifier.py`, `analyzer.py` each duplicate ~15 lines of identical client init logic.

**New file `pipeline/_client.py`:**

```python
"""Shared OpenAI-compatible client for pipeline modules."""
import os
from typing import Optional

def create_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    from openai import OpenAI
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

Each pipeline class calls `create_openai_client()` instead of duplicating the logic.

### 2.2 Fix `llm.py` JSON fence stripping

**Current (fragile):**
```python
if raw.startswith("```json"):
    raw = raw[7:-3].strip()
```

**Fixed (regex):**
```python
import re
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)```\s*$", re.DOTALL)

def strip_markdown_fences(text: str) -> str:
    m = _FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text.strip()
```

Applied in both `gemini_extract_structured()` and `gemini_extract_json()`.

### 2.3 Add JSON parsing error handling in pipeline

Wrap `json.loads(response.choices[0].message.content)` in all three pipeline modules:

```python
try:
    data = json.loads(response.choices[0].message.content)
except json.JSONDecodeError as e:
    raise ValueError(
        f"LLM returned invalid JSON: {e}. "
        f"Raw response: {response.choices[0].message.content[:200]}"
    ) from e
```

### 2.4 ClaimStore batch efficiency

**Current:** `save_claims()` loops `save_claim()`, each call opens 2 DB connections (1 for `get_claim()` read + 1 for `_write_claim()` write).

**Fixed:** Add `conn` parameter threading through both read and write paths:

```python
def save_claims(self, claims: List[Claim]) -> None:
    """Batch save with single connection."""
    with sqlite3.connect(self.db_path) as conn:
        for claim in claims:
            self._save_claim_with_conn(claim, conn)
        conn.commit()

def save_claim(self, claim: Claim) -> None:
    """Single claim save — opens own connection."""
    with sqlite3.connect(self.db_path) as conn:
        self._save_claim_with_conn(claim, conn)
        conn.commit()

def _save_claim_with_conn(self, claim: Claim, conn: sqlite3.Connection) -> None:
    """Core save logic using provided connection (handles merge + write)."""
    existing = self._get_claim_with_conn(claim.claim_id, conn)
    if existing is not None:
        # ... merge logic (unchanged) ...
        claim = existing
    self._write_claim_with_conn(claim, conn)
```

Both `get_claim()` and `_write_claim()` get `_with_conn` variants that accept a connection parameter. The public `get_claim()` and `save_claim()` still work standalone.

### 2.5 Consolidate `RuleResult` / `RuleResultModel`

`rules.py` has `RuleResult` (dataclass), `evidence.py` has `RuleResultModel` (Pydantic).

**Resolution:** Convert `RuleResult` in `rules.py` from `dataclass` to Pydantic `BaseModel`. This makes it the single canonical type usable in both the abstract rule engine interface and in `ScoringData.rule_details`. Remove `RuleResultModel` from `evidence.py`.

**Changes:**
- `rules.py`: `RuleResult` becomes `class RuleResult(BaseModel)` (fields unchanged)
- `evidence.py`: Remove `RuleResultModel`, change `ScoringData.rule_details` type to `List[RuleResult]` (import from `rules.py`)
- `tests/test_evidence.py`: Update imports (`RuleResultModel` → `RuleResult` from `graphite.rules`)
- `tests/test_rules.py`: Update for Pydantic-based `RuleResult` (construction unchanged, but `asdict()` → `.model_dump()`)

---

## Phase 3: Polish

### 3.1 `text.py` SHA-1 → SHA-256

- `sha1_hex()` → deprecate (keep for compatibility, add docstring warning)
- `find_best_paragraph_for_quote()` — change internal hash from `sha1_hex` to `sha256_hex`
- Any other internal callers of `sha1_hex` → switch to `sha256_hex`

### 3.2 `confidence.py` testable recency

Add optional `now` parameter to `_compute_recency()` and `score()`:

```python
def score(self, claim: "Claim", now: Optional[datetime] = None) -> ConfidenceResult:
    ...

def _compute_recency(self, evidence: list, now: Optional[datetime] = None) -> float:
    now = now or datetime.now(timezone.utc)
    ...
```

### 3.3 Fix `EvidencePacket` forward references

**Current:** Uses `TYPE_CHECKING` guard + requires explicit `model_rebuild()`.

**Fix:** Use string annotations in Field types and call `model_rebuild()` in `__init__.py` after all models are imported.

**Important:** Import order matters. `EvidencePacket.model_rebuild()` must come **after** `Claim` and `ConfidenceResult` are imported, since those are the forward-referenced types.

```python
# In __init__.py — order matters:
from .claim import Claim, ConfidenceFactor, ConfidenceResult  # must come first
from .evidence import EvidencePacket, EvidenceData
# ... other imports ...

# Resolve forward references (must be after Claim/ConfidenceResult imports)
EvidencePacket.model_rebuild()
```

This makes it transparent to users — no manual `model_rebuild()` needed.

### 3.4 Error path tests

Add tests for:
- Malformed JSON from LLM (→ `ValueError`)
- Unexpected verdict strings (→ fallback to default)
- Empty document/corpus inputs
- `ClaimStore` concurrent access patterns

These go in existing test files (`test_extractor.py`, `test_verifier.py`, etc.).

### 3.5 Commit working tree changes

After all refactoring, commit in logical chunks:
1. Dead code removal (benchmark, archive, old examples)
2. Structure changes (verdict.py, enum consolidation)
3. Pipeline quality (client extraction, error handling, DB efficiency)
4. Polish (SHA-256, recency, forward refs, tests)

---

## File Change Summary

| File | Action |
|------|--------|
| `benchmark/` | Delete |
| `src/graphite/_archive/` | Delete |
| `examples/flood_replay_demo/` | Delete |
| `examples/forecast_demo/` | Delete |
| `examples/toy_battery_demo/` | Delete |
| `examples/ablation/` | Delete |
| `pipeline/verdict.py` | **New** — Verdict types moved from claim.py |
| `pipeline/_client.py` | **New** — Shared OpenAI client factory |
| `claim.py` | Remove verdict types + enums (import from enums.py) |
| `enums.py` | Add ClaimType, ClaimStatus, ReviewState, ClaimOrigin, ClaimGranularity |
| `__init__.py` | Update imports, add model_rebuild() |
| `llm.py` | Regex-based fence stripping |
| `pipeline/extractor.py` | Use _client.py, add JSON error handling |
| `pipeline/verifier.py` | Use _client.py, add JSON error handling |
| `pipeline/analyzer.py` | Use _client.py, add JSON error handling |
| `pipeline/report.py` | Import from pipeline.verdict |
| `claim_store.py` | Batch save with single connection |
| `evidence.py` | Remove RuleResultModel, fix forward refs |
| `rules.py` | Convert `RuleResult` from dataclass to Pydantic BaseModel |
| `text.py` | SHA-1 deprecation, switch internals to SHA-256 |
| `confidence.py` | Optional `now` parameter |
| `tests/test_verifier.py` | Update verdict imports |
| `tests/test_analyzer.py` | Update verdict imports |
| `tests/test_report.py` | Update verdict imports |
| `tests/test_evidence.py` | `RuleResultModel` → `RuleResult` |
| `tests/test_rules.py` | Update for Pydantic `RuleResult` |
| `tests/*` | Error path tests |
| `examples/quickstart_verification/run.py` | Update verdict imports |
| `README.md` | Update code snippet imports |

## Risk Assessment

- **Breaking changes:** None at the public API level. All existing `from graphite import X` paths preserved via re-exports.
- **Test regression:** Run full suite after each phase.
- **Rollback:** Each phase is independently committable and revertable.
