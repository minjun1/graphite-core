# 100% Module Test Coverage for Graphite

**Date:** 2026-03-20
**Status:** Approved

## Goal

Every source module under `src/graphite/` (excluding `_archive/`) has a dedicated test file with meaningful tests for its public API. Tests follow the existing class-based grouping style (e.g., `class TestClaimId`). External dependencies (LLM APIs, geo services) are mocked via `unittest.mock.patch`.

## Scope

- 16 new test files across 3 layers
- Modules already covered (`claim.py`, `confidence.py`) are unchanged
- `_archive/` directory is excluded

## Layer 1 — Core Types

No external dependencies. Pure unit tests.

### `test_enums.py`
- All enum classes (`SourceType`, `EdgeType`, `NodeType`, `AssertionMode`, `ConfidenceLevel`, `EvidenceType`) exist with expected members
- String value roundtrip (e.g., `SourceType("sec_10k") == SourceType.SEC_10K`)
- Membership checks

### `test_schemas.py`
- `NodeRef` factory methods (`company()`, `mineral()`, `country()`, `region()`, `asset()`, `facility()`, `corridor()`) produce correct `node_id` prefix and `node_type`
- `ExtractedEdge.edge_key` returns deterministic dedup key
- `ExtractedEdge.best_confidence` returns highest confidence across provenances
- `Provenance` construction with all fields
- `InferenceBasis` construction
- `ExtractionError` construction

### `test_evidence.py`
- `EvidenceData` construction and field access
- `EvidencePacket` construction with nested `EvidenceData`, `ScoringData`, `CounterEvidence`
- `RuleResultModel` construction
- `ScoringData` with `applied_rules` and `rule_details`
- Serialization roundtrip (`model_dump()` / `model_validate()`)

### `test_text.py`
- `sha1_hex` and `sha256_hex` produce deterministic hex strings
- `normalize_text` collapses whitespace and strips
- `clip_quote` truncates at boundary with ellipsis
- `split_into_paragraphs` respects `min_len` and `max_paras`
- `score_paragraph` counts keyword occurrences
- `find_best_paragraph_for_quote` returns correct index
- Strategy registry: `register_strategy` adds callable, `build_context` dispatches correctly
- Built-in strategies (`default`, `usgs_country_mineral`, `sec_minerals`, `sec_generic`) are registered
- `build_context` raises `ValueError` for unknown strategy name

### `test_rules.py`
- `RuleResult` dataclass fields
- `ScoreBreakdown.triggered_rules` filters correctly
- `ScoreBreakdown.total_delta` returns `applied_delta`
- `ScoreBreakdown.to_dict()` includes all fields
- `BaseRuleEngine` cannot be instantiated directly (ABC)
- Concrete subclass can implement `compute_score`

### `test_domain.py`
- `register_domain` / `get_domain` / `list_domains` registry roundtrip
- `get_domain` returns `None` for unknown name
- `ExtractionResult` construction
- `DocumentContext` dataclass fields
- `BaseFetcher.fetch_batch` calls `fetch` per entity
- `BaseExtractor.extract_batch` calls `extract` per doc
- `BasePipeline` is abstract; concrete subclass can implement `run()`

### `test_cache.py`
- `make_key` returns deterministic 5-part key
- `content_hash` is deterministic for same input
- `put` / `get` roundtrip (using `tmp_path`)
- `has` returns True after `put`, False before
- `get` returns `None` for missing key
- `clear` removes all entries and returns count

## Layer 2 — Storage

### `test_claim_store.py`
- `save_claim` + `get_claim` roundtrip (SQLite in `tmp_path`)
- Evidence accumulation: saving same claim twice with different evidence merges both
- Evidence dedup: duplicate `(source_id, quote)` not added twice
- `save_claims` batch operation
- `search_claims` with `subject_contains`, `object_contains`, `predicate` filters
- `search_claims` with `as_of_date` filter
- `find_supporting_claims` returns claims with overlapping subjects/objects
- `find_potential_conflicts` returns claims with contradictory predicates
- Analyst override preservation: `save_claim` merges `override_reason`, `generator_id`, `generation_metadata` from incoming claim
- Empty store returns `None` / empty list

## Layer 3 — Pipeline + Adapters

All LLM-dependent modules mock the LLM client. Adapters mock file/network I/O.

### `test_llm.py`
- Mock `google.genai.Client` (lazily imported inside functions) via `unittest.mock.patch("google.genai.Client")`
- `gemini_extract_structured` parses mocked response into pydantic schema
- `gemini_extract_json` returns parsed dict from mocked response
- Markdown fence stripping: responses wrapped in triple-backtick JSON fences are cleaned
- Retry on transient failure in `gemini_extract_structured` (mock first call fails, second succeeds)
- Client lazy-initialization (not created until first call)
- Missing `GEMINI_API_KEY` raises `RuntimeError`

### `test_alphaearth.py`
- Cache write/read roundtrip with `tmp_path` and mock numpy arrays
- `get_embedding` returns cached array without fetching
- `get_embedding_safe` returns `None` on missing cache (no GCS configured)
- `list_cached` and `cache_stats` reflect cached entries
- `get_area_embedding` falls back to centroid point embedding

### `test_weathernext.py`
- Load from snapshot JSON fixture (small test fixture in `tests/fixtures/`)
- `get_forecast` returns forecast dict for known node
- `get_forecast` returns `None` for unknown node
- `get_all_forecasts` returns complete dict
- `list_nodes` returns all node IDs
- `meta` property returns metadata

### `test_retriever.py`
- Pure logic — no mocks needed
- `DocumentCorpus` chunks documents into paragraphs
- `EvidenceRetriever` class and `retrieve_evidence` convenience function
- `retrieve_evidence` returns ranked chunks by lexical overlap
- `top_k` parameter limits results
- Empty corpus returns empty results
- Claims with no keyword overlap fall back to `claim_text` tokens

### `test_extractor.py`
- Mock OpenAI client via `unittest.mock.patch("graphite.pipeline.extractor.OpenAI")`
- Test both `ClaimExtractor` class and `extract_claims` convenience function
- `extract_claims` parses mocked LLM JSON into `List[Claim]`
- Each returned `Claim` has valid `claim_id`, subjects, predicate, objects
- Empty document returns empty list

### `test_verifier.py`
- Mock OpenAI client via `unittest.mock.patch("graphite.pipeline.verifier.OpenAI")`
- Test both `ClaimVerifier` class and `verify_claims` convenience function
- `verify_claims` returns `List[Verdict]` with correct `verdict` values
- Mocked SUPPORTED, CONFLICTED, INSUFFICIENT responses all parse correctly
- Empty evidence map produces INSUFFICIENT verdicts

### `test_analyzer.py`
- Mock OpenAI client via `unittest.mock.patch("graphite.pipeline.analyzer.OpenAI")`
- Test both `ArgumentAnalyzer` class and `analyze_argument_chain` convenience function
- `analyze_argument_chain` returns `List[ArgumentVerdict]`
- Mocked GROUNDED, CONCLUSION_JUMP, OVERSTATED responses parse correctly
- Empty verdicts list handled gracefully

### `test_report.py`
- Mock all 4 pipeline sub-steps (`extract_claims`, `retrieve_evidence`, `verify_claims`, `analyze_argument_chain`)
- `verify_agent_output` orchestrates all steps and returns `VerificationReport`
- Report fields (`total_claims`, `supported_count`, etc.) match mocked data
- `review_document` is an alias for `verify_agent_output`
- `VerificationReport.get_verdict(claim_id)` returns correct `Verdict` or `None`

### `test_init.py`
- Smoke test: all expected public symbols importable from `graphite` (e.g., `from graphite import Claim, ClaimStore, ConfidenceScorer, ...`)

## Test Infrastructure

- All tests run with `pytest` — no new test dependencies
- SQLite and file-based tests use `tmp_path` fixture for isolation
- LLM mocks use `unittest.mock.patch` — no new mock libraries
- WeatherNext tests use a small JSON fixture at `tests/fixtures/weathernext_snapshot.json`
- No API keys required for any test

## Implementation Order

1. Layer 1 (core types) — 7 files, no dependencies between them
2. Layer 2 (claim_store) — 1 file, depends on Layer 1 types
3. Layer 3 (pipeline + adapters) — 7 files, depends on Layers 1-2
