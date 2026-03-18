#!/usr/bin/env python3
"""
Hurricane Beryl (2024) Forecast Demo — WeatherNext-driven blast radius

Forward-looking demo: forecast snapshot → compute hazard → inject shocks → blast radius

This demo demonstrates Graphite Terra's forward-looking capability:
  1. Read a WeatherNext 2 forecast snapshot (deterministic, sample-first)
  2. Convert forecast fields to ScenarioShock objects via hazard thresholds
  3. Build Houston infrastructure dependency graph from real Beryl reports
  4. Propagate forecast-driven shocks through the graph
  5. Output a forecast blast radius with EXPERIMENTAL disclaimer

Run:
  python examples/forecast_demo/run.py                 # from snapshot (default)
  python examples/forecast_demo/run.py --alphaearth    # with AlphaEarth enrichment

Note: WeatherNext 2 is an experimental dataset.
      This demo uses a deterministic snapshot for reproducibility.
"""
import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graphite import GraphAssembler, save_graph, ScenarioRunner
from graphite.adapters.weathernext import WeatherNextAdapter
from graphite.scenarios.weathernext_forecast import (
    forecast_to_scenario_shocks, compute_hazard_intensity,
)
from extractor import get_beryl_edges


def parse_args():
    parser = argparse.ArgumentParser(description="Hurricane Beryl 2024 forecast demo")
    parser.add_argument(
        "--alphaearth", action="store_true",
        help="Enrich graph nodes with AlphaEarth satellite embeddings"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use live WeatherNext 2 data (requires approved access)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    snapshot_path = os.path.join(demo_dir, "forecast_snapshot.json")

    # ── Step 1: Load WeatherNext 2 forecast ──
    print("🌤️  Step 1: Loading WeatherNext 2 forecast snapshot...")
    adapter = WeatherNextAdapter(snapshot_path=snapshot_path, live=args.live)
    forecasts = adapter.get_all_forecasts()
    meta = adapter.meta

    print(f"   Model: {meta.get('model', 'unknown')}")
    print(f"   Init time: {meta.get('init_time', 'unknown')}")
    print(f"   Lead time: {meta.get('lead_hours', '?')}h")
    print(f"   → {len(forecasts)} forecast points loaded\n")

    # ── Step 2: Convert forecasts to hazard scores ──
    print("⚡ Step 2: Computing hazard intensities...")
    for node_id, fc in forecasts.items():
        fields = fc.get("fields", {})
        intensity = compute_hazard_intensity(fields)
        wind_mph = fields.get("wind_speed_ms", 0) * 2.237
        precip_in = fields.get("precipitation_mm", 0) / 25.4
        print(f"   {intensity:.0%} | {wind_mph:5.0f}mph | {precip_in:4.1f}in | {node_id}")
    print()

    # ── Step 3: Generate ScenarioShocks from forecasts ──
    print("💥 Step 3: Generating scenario shocks from forecast...")
    shocks = forecast_to_scenario_shocks(
        forecasts,
        event_name="hurricane_beryl_2024_forecast",
        init_time=meta.get("init_time", "2024-07-07T12:00:00Z"),
        intensity_threshold=0.4,
    )

    if not shocks:
        print("   ⚠  No shocks generated — forecast below threshold")
        return

    shock = shocks[0]
    print(f"   Shock ID: {shock.shock_id}")
    print(f"   Target nodes: {len(shock.target_nodes)}")
    print(f"   Intensity: {shock.intensity:.1%}\n")

    # ── Step 4: Build infrastructure graph from Beryl reports ──
    print("🔧 Step 4: Assembling Beryl infrastructure graph...")
    edges = get_beryl_edges()
    G = GraphAssembler().assemble(edges)
    print(f"   → {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Step 4b: Optional AlphaEarth enrichment + embedding-aware propagation ──
    embedding_aware = False
    if args.alphaearth:
        try:
            import numpy as np
            from graphite.features.alphaearth_enricher import AlphaEarthEnricher
            from graphite.features.embedding_similarity import (
                inject_edge_similarity, make_embedding_aware_alpha,
            )
            geom_path = os.path.join(demo_dir, "node_geometries.json")
            cache_dir = os.path.join(demo_dir, "cache", "alphaearth")
            if os.path.exists(geom_path):
                enricher = AlphaEarthEnricher(cache_dir=cache_dir)
                geometries = enricher.load_geometries(geom_path)
                stats = enricher.enrich(G, geometries, year=2024)
                print(f"   🛰️  AlphaEarth: {stats['enriched']}/{stats['total']} nodes enriched")

                # Level 2: inject edge similarity + modulate alpha
                n_ann = inject_edge_similarity(G)
                print(f"   📐 Edge similarity: {n_ann}/{G.number_of_edges()} edges annotated")
                embedding_aware = True
        except ImportError as e:
            print(f"   ⚠  AlphaEarth requires numpy: {e}")

    print()

    # ── Step 5: Save graph ──
    path = save_graph(G, os.path.join(demo_dir, "beryl_graph.json"), format="json")
    print(f"💾 Graph saved: {path}\n")

    # ── Step 6: Run forecast-driven scenario ──
    print("🌊 Step 6: Running forecast-driven blast radius...\n")

    def base_alpha_fn(edge_type, is_supply):
        alphas = {
            "RISK_FLOWS_TO": 0.8,
            "DEPENDS_ON": 0.7,
            "LOCATED_IN": 0.5,
            "EXPOSED_TO": 0.9,
            "ADJACENT_TO": 0.3,
        }
        return alphas.get(edge_type, 0.4)

    # Same parameters as Harvey: w=0.4, clamp [0.3, 1.7]
    if embedding_aware:
        from graphite.features.embedding_similarity import make_embedding_aware_alpha
        effective_alpha = make_embedding_aware_alpha(
            G, base_alpha_fn, embedding_weight=0.4  # same as Harvey
        )
        print("   📐 Using embedding-aware alpha (same params as Harvey: w=0.4)\n")
    else:
        effective_alpha = base_alpha_fn

    runner = ScenarioRunner()
    result = runner.run(
        G,
        shocks=shocks,
        max_hops=5,
        tau_stop=0.03,
        k=5,
        is_supply=True,
        alpha_fn=effective_alpha,
    )

    blast = result["blast_radius"]

    # ── Print results ──
    print(f"{'='*78}")
    print(f"  ⛈️  HURRICANE BERYL (2024) — Forecast-Driven Blast Radius")
    print(f"  WeatherNext 2 forecast init: {meta.get('init_time', '?')}")
    print(f"  ⚠  EXPERIMENTAL — not validated for real-world decisions")
    print(f"{'='*78}")

    if not blast:
        print("  ⚠  No downstream impact detected")
    else:
        for item in blast:
            tier = item["exposure_tier"]
            score = item["total_exposure"]
            entity = item["entity"]

            top_paths = item.get("top_paths", [])
            evidence_hint = ""
            if top_paths:
                ev_pack = top_paths[0].get("evidence_pack", [])
                if ev_pack:
                    evidence_hint = f"  ← {ev_pack[-1].get('evidence', '')[:50]}"

            print(f"  {tier:>8s} | {score:6.1%} | {entity}{evidence_hint}")

    print(f"{'='*78}")
    print(f"  ⚠  This forecast scenario demo uses WeatherNext 2 experimental data.")
    print(f"  ⚠  Not intended, validated, or approved for real-world use.\n")

    # ── Save output ──
    output = {
        "scenario_id": result["scenario_id"],
        "event": "Hurricane Beryl (Category 1, July 8 2024)",
        "forecast_model": meta.get("model", "WeatherNext 2"),
        "forecast_init": meta.get("init_time", "2024-07-07T12:00:00Z"),
        "disclaimer": "WeatherNext 2 is experimental. Not validated for real-world use.",
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "shocks": [{
            "shock_id": s.shock_id,
            "target_nodes": s.target_nodes,
            "intensity": s.intensity,
        } for s in shocks],
        "blast_radius": [
            {
                "entity": item["entity"],
                "exposure_tier": item["exposure_tier"],
                "total_exposure": round(item["total_exposure"], 6),
            }
            for item in blast
        ],
    }

    output_path = os.path.join(demo_dir, "output.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"📊 Results saved: {output_path}")


if __name__ == "__main__":
    main()
