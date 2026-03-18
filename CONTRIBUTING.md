# Contributing to Graphite

Thank you for your interest in contributing to Graphite! ⛏️

## Getting Started

1. **Fork** this repository
2. **Clone** your fork locally
3. **Set up** the development environment:
   ```bash
   git clone https://github.com/YOUR_USERNAME/graphite.git
   cd graphite
   python3 -m venv venv && source venv/bin/activate
   pip install -e ".[dev]"
   ```

## Running Tests

All tests run without external dependencies (no Neo4j, no LLM, no API keys).

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_propagation.py
```

### Test Structure

```
tests/
├── conftest.py              # Path setup (auto-loaded by pytest)
├── test_propagation.py      # Dijkstra + Noisy-OR propagation
├── test_rules.py            # Rule Engine
├── test_scenario.py         # Scenario runner
├── test_pipeline_core.py    # Graph assembler pipeline
└── test_graph_store.py      # Quote hashing, alias resolution
```

### Writing Tests

- **No external services required**: Tests use synthetic graphs and in-memory stores
- **Test determinism**: Same input → same output (critical for verification engines)
- **Test the math**: Verify confidence scoring, attenuation formulas, score bounds
- **Name clearly**: `test_claim_status_derives_from_confidence`, etc.

### PR Checklist

- [ ] All existing tests pass (`pytest`)
- [ ] New functionality has tests
- [ ] No hardcoded secrets or personal info

## How to Contribute

### Reporting Issues
- Use GitHub Issues for bug reports and feature requests
- Include reproduction steps for bugs
- Tag issues with appropriate labels (`bug`, `enhancement`, `domain-plugin`)

### Adding a New Domain Plugin
This is the most impactful way to contribute! A domain plugin needs:

1. **Fetcher** — subclass `BaseFetcher` to fetch documents
2. **Extractor** — subclass `BaseExtractor` to extract claims
3. **DomainSpec** — register your domain with `register_domain()`

See `src/graphite/domain.py` for the plugin interface.

### Code Style
- Python: Follow PEP 8, use type hints
- Write docstrings for public functions

### Pull Requests
1. Create a branch from `main`: `git checkout -b feature/your-feature`
2. Run tests: `pytest`
3. Commit with clear messages
4. Submit a PR with a description of what and why

## Architecture

```
src/graphite/
├── claim.py          # Claim: the atomic trust primitive
├── claim_store.py    # SQLite-backed claim registry
├── confidence.py     # Explainable confidence scoring
├── schemas.py        # ExtractedEdge, NodeRef, Provenance
├── assembler.py      # Graph assembly with provenance merge
├── domain.py         # Plugin interface (BaseFetcher, BaseExtractor)
├── simulate.py       # Dijkstra + Noisy-OR propagation
├── scenario.py       # Scenario runner (shock injection)
├── rules.py          # Generic rule engine
├── io.py             # Graph I/O (JSON, GraphML)
└── enums.py          # Edge types, source types, confidence levels
```

## Questions?

Open a Discussion on GitHub — we're happy to help!
