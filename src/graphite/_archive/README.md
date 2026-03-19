# Archived: Entity Graph / Supply Chain Layer

**Archived**: 2026-03-18
**Reason**: Graphite's product focus shifted to claim verification. The entity graph layer (supply chain modeling, shock propagation, blast radius) is no longer part of the core API.

## Contents

| File | What it did |
|---|---|
| `graph.py` | `GraphStore` — NetworkX DiGraph with optional Neo4j loading |
| `simulate.py` | Dijkstra + Noisy-OR blast radius propagation |
| `scenario.py` | `ScenarioRunner` — shock injection and propagation |
| `assembler.py` | `GraphAssembler` — edge dedup, provenance merge, graph stamping |
| `io.py` | Save/load (GraphML, JSON) + `push_to_neo4j()` |
| `features/` | AlphaEarth enricher, embedding similarity |
| `scenarios/` | WeatherNext forecast scenarios |
| `geo_evidence/` | Geospatial foundation model provenance |

## Future Reuse Path

The blast radius concept maps directly to **claim dependency propagation**:

- `simulate.py`'s Dijkstra + Noisy-OR → propagate confidence loss when a supporting claim breaks
- `scenario.py`'s shock injection → model "what if this evidence is retracted?"
- `assembler.py`'s dedup and provenance merge → already useful for claim evidence accumulation

When ready to implement claim dependency graphs, start by adapting `simulate.py` with claims as nodes and `depends_on_claim_ids` as edges.
