# 🔋 Toy Battery Supply Chain Demo

End-to-end Graphite pipeline: **Documents → Extract → Assemble → Simulate Shock**

## Run

```bash
python examples/toy_battery_demo/run.py
```

No Neo4j. No LLM. No API keys. Just Python.

## What it does

1. **Reads** 3 text documents (fake USGS report, CATL annual report, Tesla 10-K)
2. **Extracts** supply chain edges using regex (no LLM needed)
3. **Assembles** a NetworkX graph with provenance-backed edges
4. **Simulates** a Congo (cobalt) supply shock using Dijkstra + Noisy-OR

## Output

```
🔴 CONGO (Cobalt) Supply Shock — Blast Radius
============================================================
      HIGH | 30.0% | company:CATL
    MEDIUM |  9.0% | company:BMW
    MEDIUM |  9.0% | company:TSLA
============================================================
```

Congo disruption → CATL exposed (direct supplier) → Tesla and BMW exposed (downstream).

## Files

| File | Purpose |
|------|---------|
| `documents/*.txt` | Source documents (fake reports) |
| `extractor.py` | Deterministic regex extractor |
| `run.py` | End-to-end pipeline script |
| `expected_output.json` | Regression baseline |
