# Changelog

All notable changes to Graphite will be documented in this file.

## [0.3.0] — 2026-03-17

### Repositioning: Claim Verification Engine

Graphite is now positioned as an **open-source claim verification engine** for high-stakes decisions, rather than a general-purpose graph/propagation engine.

### Added
- **Quickstart verification example** (`examples/quickstart_verification/`) — zero-dependency demo of Claim → ClaimStore → ConfidenceScorer
- **Benchmark section in README** — EdgarOS eval results on Graphite primitives (90.5% conservative precision, 0% false contradiction rate)
- **CI workflow** — GitHub Actions for Python 3.10/3.11/3.12 with pytest + example smoke tests

### Changed
- **README rewritten** — verification-first positioning; Claim, ClaimStore, Provenance as core primitives; propagation demoted to secondary capability
- **`pyproject.toml`** — description and keywords updated for verification positioning
- **`__init__.py`** — module docstring updated; verification pipeline as primary, propagation as "also included"
- **`CONTRIBUTING.md`** — removed legacy references (docker-compose, requirements.txt), updated architecture section

### Fixed
- **`Claim.is_overridden`** — now correctly includes `NEEDS_FOLLOWUP` review state, preventing `compute_status()` from silently overwriting analyst overrides for WEAK/MIXED verdicts
- **Test imports** — all test files migrated from legacy `core.*` to `graphite.*` package paths (3 files, 6 import blocks)

## [0.2.0] — 2026-03-14

- Initial claim engine: `Claim`, `ClaimStore`, `ConfidenceScorer`
- Scenario runner and shock propagation
- Geo-climate domain support (flood replay, AlphaEarth embeddings)

## [0.1.0] — 2026-02-27

- Core graph primitives: `ExtractedEdge`, `NodeRef`, `Provenance`
- Graph assembly with provenance merge and deduplication
- Dijkstra + Noisy-OR blast radius propagation
- Domain plugin interface (`BaseFetcher`, `BaseExtractor`, `DomainSpec`)
