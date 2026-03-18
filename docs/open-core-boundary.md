# Open Core Boundary: Graphite vs EdgarOS

## Graphite (open-source, Apache 2.0)

The engine. Domain-agnostic primitives for building claim verification systems.

| Module | Description |
|--------|-------------|
| `Claim` | Atomic trust primitive — structured assertion with provenance |
| `ClaimStore` | SQLite-backed claim registry with search |
| `Provenance` | First-class evidence source (document, quote, confidence) |
| `ConfidenceScorer` | Explainable confidence scoring with named factors |
| `GraphAssembler` | Graph construction with deduplication and provenance merge |
| `BaseFetcher` / `BaseExtractor` | Plugin interface for domain-specific extraction |
| `simulate.py` / `scenario.py` | Shock propagation (Dijkstra + Noisy-OR) |
| `rules.py` | Generic rule engine |
| `examples/` | Toy battery demo, flood replay, quickstart verification |
| `tests/` | Full test suite (122 tests) |

## EdgarOS (commercial, closed-source)

The first application built on Graphite. SEC-focused claim verification for AI-generated financial research.

| Module | Description |
|--------|-------------|
| `domains/memo/*` | Memo extractor, matcher, verifier — SEC-specific logic |
| `verification_config.py` | Evidence status, staleness thresholds, grounding rules |
| `entity_aliases.py` | Ticker alias expansion (e.g. SMCI → Super Micro) |
| `eval/` | Gold sets, eval runner, benchmark harness |
| `api/` | FastAPI endpoints for verification |
| `frontend/` | Next.js UI with evidence packet drawer |
| SEC claim registry | Production database of extracted claims from 10-K filings |
| Extraction prompts | LLM prompts for SEC claim extraction and matching |

## Decision Criteria

**Open it if** the capability is domain-agnostic and useful to anyone building a verification system.

**Keep it closed if** it contains SEC-specific calibration, competitive advantage, or production data.

## Examples

| Question | Answer |
|----------|--------|
| "Can I build my own claim verification system?" | ✅ Yes — use Graphite primitives |
| "Can I replicate EdgarOS's SEC accuracy?" | ❌ No — requires closed eval harness, gold sets, and prompt calibration |
| "Can I add a new domain (medical, legal)?" | ✅ Yes — subclass `BaseExtractor` and `BaseFetcher` |
| "Can I use the blast radius engine?" | ✅ Yes — `simulate.py` and `scenario.py` are open |
