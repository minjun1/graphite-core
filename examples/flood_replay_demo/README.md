# 🌊 Hurricane Harvey Flood Replay Demo

**A historical event reconstruction using Graphite's evidence-backed graph engine.**

This demo replays the infrastructure cascade from Hurricane Harvey (August 2017) — reconstructing how port closure propagated through refineries and chemical plants using real public report data.

## What It Does

```
Real reports → Extract infrastructure edges → Assemble graph → Inject Harvey shock → Blast radius
```

1. **Extracts** infrastructure relationships from 3 real public reports (NWS, EIA, CSB)
2. **Assembles** an evidence-backed infrastructure dependency graph
3. **Injects** Hurricane Harvey as a `ScenarioShock` at Port of Houston
4. **Propagates** the shock through the graph using Graphite's Dijkstra + Noisy-OR engine
5. **Outputs** a blast radius showing downstream infrastructure exposure

## Run

```bash
cd graphite

# Basic demo
python examples/flood_replay_demo/run.py

# With AlphaEarth satellite embedding enrichment
python examples/flood_replay_demo/run.py --alphaearth
```

## Expected Output

```
🌊 HURRICANE HARVEY (2017) — Infrastructure Blast Radius
  Shock source: Port of Houston (closed Aug 25-31)
======================================================================
   EXTREME |  57.6% | facility:EXXON_BAYTOWN
   EXTREME |  54.0% | facility:MOTIVA_PORT_ARTHUR
   EXTREME |  50.4% | facility:VALERO_HOUSTON
      HIGH |  32.3% | corridor:HOUSTON_SHIP_CHANNEL
      HIGH |  30.0% | facility:LYONDELLBASELL_CHANNELVIEW
      HIGH |  27.6% | facility:DOW_FREEPORT
```

## Data Sources

All evidence comes from public reports with source attribution:

| Document | Source |
| --- | --- |
| `harvey_noaa_summary.txt` | NWS Corpus Christi — [weather.gov](https://www.weather.gov/crp/hurricane_harvey) |
| `houston_port_impact.txt` | EIA, Reuters, Port Houston Authority |
| `downstream_industry_impact.txt` | CSB [Arkema report](https://www.csb.gov/arkema-inc-chemical-plant-fire-/), PBS, EDF |

## Key Concepts Demonstrated

- **`ScenarioShock`** — hazards are runtime inputs, not graph nodes
- **`ScenarioRunner`** — thin orchestration: shock inject → propagate → blast radius
- **`AlphaEarthEnricher`** — attaches 64-dim satellite embeddings as node features
- **Semantic edge direction** — `DEPENDS_ON` means "A needs B to operate"
- **Temporal provenance** — all evidence tagged with ISO 8601 dates and snapshot ID
- **Evidence traceability** — every edge links back to a quote from a public report
- **Cache-first design** — AlphaEarth embeddings read from local .npy cache, no GCS dependency
