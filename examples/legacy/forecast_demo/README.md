# ⛈️ Hurricane Beryl (2024) Forecast Demo

**A forward-looking forecast scenario using Graphite Terra + WeatherNext 2.**

This demo shows how Graphite Terra translates weather forecast signals into
infrastructure blast radius graphs — using Hurricane Beryl (July 2024) as the scenario.

> ⚠️ **EXPERIMENTAL**: This demo uses [WeatherNext 2](https://developers.google.com/earth-engine/datasets/catalog/projects_gcp-public-data-weathernext_assets_weathernext_2_0_0)
> experimental data. WeatherNext 2 forecasts are not intended, validated, or approved
> for real-world decision-making.

## What It Does

```text
Forecast snapshot → Compute hazard → Generate ScenarioShock → Propagate → Blast radius
```

1. **Loads** a WeatherNext 2 forecast snapshot (0.25° resolution, 64-member ensemble)
2. **Computes** hazard intensity per infrastructure node (wind + precipitation)
3. **Converts** forecast fields into `ScenarioShock` objects via Saffir-Simpson thresholds
4. **Builds** an infrastructure graph from real Beryl 2024 NWS and port reports
5. **Propagates** forecast-driven shocks through the dependency graph
6. **Outputs** a forecast blast radius with experimental disclaimers

## Run

```bash
cd graphite
python examples/forecast_demo/run.py
```

## Expected Output

```text
⛈️  HURRICANE BERYL (2024) — Forecast-Driven Blast Radius
  WeatherNext 2 forecast init: 2024-07-07T12:00:00Z
  ⚠  EXPERIMENTAL — not validated for real-world decisions
==============================================================================
   EXTREME | 59.9% | asset:CENTERPOINT_GRID
   EXTREME | 59.9% | asset:PORT_HOUSTON
   EXTREME | 49.8% | corridor:HOUSTON_SHIP_CHANNEL
   EXTREME | 46.1% | region:HOUSTON_METRO
```

## Data Sources

| Source | Type |
| --- | --- |
| `beryl_nws_report.txt` | NWS Houston/Galveston post-tropical cyclone report |
| `beryl_port_energy_impact.txt` | Port Houston Authority, Reuters, Offshore Technology |
| `forecast_snapshot.json` | Simulated WeatherNext 2 snapshot (experimental) |

## Key Concepts Demonstrated

- **Forecast → Shock** — WeatherNext 2 fields converted to ScenarioShock via hazard thresholds
- **Multi-node shocking** — all forecast nodes above threshold are shocked simultaneously
- **Sample-first design** — deterministic snapshot by default, `--live` for real EE/BQ access
- **Harvey continuity** — same Houston asset graph, updated for 2024 event (+ Freeport LNG)
- **Experimental guardrails** — disclaimers in output and README per WeatherNext 2 terms

## Comparison with Harvey Demo

| | Demo A (Harvey) | Demo B (Beryl) |
| --- | --- | --- |
| **Event** | Cat 4, Aug 2017 | Cat 1, Jul 2024 |
| **Data** | NWS/EIA/CSB reports | NWS/Port Houston + WeatherNext 2 |
| **Shock** | Manual (single node) | Forecast-driven (multi-node) |
| **Signal** | AlphaEarth embeddings | WeatherNext 2 forecasts |
| **Message** | Historical replay + evidence | Forward-looking forecast risk |
