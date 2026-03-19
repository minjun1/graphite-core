#!/usr/bin/env python3
"""
Hurricane Harvey Flood Replay Demo — End-to-End Graphite Pipeline

Real event reconstruction: documents → extract → assemble → scenario shock → blast radius
Optional: --alphaearth flag to enrich nodes with AlphaEarth foundation model embeddings

This demo shows Graphite handling geo-climate risk propagation using
real Hurricane Harvey (2017) data. Every edge is backed by evidence
from actual event reports.

Run:
  python examples/flood_replay_demo/run.py                    # basic
  python examples/flood_replay_demo/run.py --alphaearth       # with AlphaEarth enrichment
"""
import argparse
import os
import sys
import json

# Add example dir to path for local extractor import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graphite import GraphAssembler, save_graph, NodeRef, ScenarioShock, ScenarioRunner
from graphite.enums import SourceType
from extractor import extract_from_documents


def parse_args():
    parser = argparse.ArgumentParser(description="Hurricane Harvey flood replay demo")
    parser.add_argument(
        "--alphaearth", action="store_true",
        help="Enrich graph nodes with AlphaEarth satellite embeddings (64-dim)"
    )
    return parser.parse_args()


def enrich_with_alphaearth(G, demo_dir: str):
    """Attach AlphaEarth embeddings to graph nodes from local cache."""
    try:
        import numpy as np
        from graphite.features.alphaearth_enricher import AlphaEarthEnricher
    except ImportError as e:
        print(f"  ⚠  AlphaEarth enrichment requires numpy: {e}")
        return

    geom_path = os.path.join(demo_dir, "node_geometries.json")
    cache_dir = os.path.join(demo_dir, "cache", "alphaearth_real")

    if not os.path.exists(geom_path):
        print(f"  ⚠  Missing {geom_path} — skipping enrichment")
        return

    enricher = AlphaEarthEnricher(cache_dir=cache_dir)
    geometries = enricher.load_geometries(geom_path)
    stats = enricher.enrich(G, geometries, year=2017)

    print(f"  🛰️  AlphaEarth enrichment: {stats['enriched']}/{stats['total']} nodes")
    if stats["failed"] > 0:
        print(f"      ⚠  {stats['failed']} nodes missing cached embeddings")


def main():
    args = parse_args()
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    doc_dir = os.path.join(demo_dir, "documents")

    # ── Step 1: Extract edges from Hurricane Harvey documents ──
    print("🔬 Step 1: Extracting edges from Hurricane Harvey reports...")
    edges = extract_from_documents(doc_dir)
    print(f"   → {len(edges)} total edges extracted\n")

    # ── Step 2: Assemble into graph ──
    print("🔧 Step 2: Assembling infrastructure graph...")
    G = GraphAssembler().assemble(edges)
    print(f"   → {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Step 2b: Optional AlphaEarth enrichment ──
    if args.alphaearth:
        print("\n🛰️  Step 2b: Enriching with AlphaEarth satellite embeddings...")
        enrich_with_alphaearth(G, demo_dir)

    print()

    # ── Step 3: Save graph ──
    path = save_graph(G, os.path.join(demo_dir, "flood_graph.json"), format="json")
    print(f"💾 Graph saved: {path}\n")

    # ── Step 4: Run Hurricane Harvey scenario ──
    print("🌊 Step 4: Replaying Hurricane Harvey shock...")

    harvey = ScenarioShock(
        shock_id="hurricane_harvey_2017",
        target_nodes=["asset:PORT_HOUSTON"],  # Primary impact point
        intensity=0.90,
        observed_at="2017-08-25T22:00:00Z",
        evidence="Category 4 hurricane made landfall near Rockport, TX. "
                 "Port of Houston closed Aug 25-31. Houston Ship Channel shut for 6 days.",
        source_type=SourceType.WEATHER_FORECAST,
    )

    def alpha_fn(edge_type, is_supply):
        """Geo-climate edge attenuation function."""
        alphas = {
            "RISK_FLOWS_TO": 0.8,
            "DEPENDS_ON": 0.7,
            "LOCATED_IN": 0.5,
            "EXPOSED_TO": 0.9,
            "ADJACENT_TO": 0.3,
        }
        return alphas.get(edge_type, 0.4)

    runner = ScenarioRunner()
    result = runner.run(
        G,
        shocks=[harvey],
        max_hops=5,
        tau_stop=0.03,
        k=5,
        is_supply=True,
        alpha_fn=alpha_fn,
    )

    # ── Print blast radius results ──
    blast = result["blast_radius"]

    # ── Step 5: Optional embedding similarity adjustment ──
    if args.alphaearth and blast:
        try:
            from graphite.features.embedding_similarity import (
                compute_similarity_scores, adjust_blast_radius,
            )
            sim_scores = compute_similarity_scores(G, harvey.target_nodes)
            blast_adjusted = adjust_blast_radius(blast, sim_scores, weight=0.3)
            has_similarity = True
        except ImportError:
            blast_adjusted = blast
            has_similarity = False
    else:
        blast_adjusted = blast
        has_similarity = False

    header = "HURRICANE HARVEY (2017) — Infrastructure Blast Radius"
    if has_similarity:
        header += " (AlphaEarth similarity-adjusted)"
    elif args.alphaearth:
        header += " (AlphaEarth-enriched)"

    print(f"\n{'='*78}")
    print(f"  🌊 {header}")
    print(f"  Shock source: Port of Houston (closed Aug 25-31)")
    print(f"{'='*78}")

    display_blast = blast_adjusted if has_similarity else blast

    if not display_blast:
        print("  ⚠  No downstream impact detected — check graph connectivity")
    elif has_similarity:
        # Show enriched table with similarity delta
        print(f"  {'TIER':>8s} | {'ADJ':>6s} | {'BASE':>6s} | {'SIM':>5s} | {'Δ':>6s} | ENTITY")
        print(f"  {'-'*8} | {'-'*6} | {'-'*6} | {'-'*5} | {'-'*6} | {'-'*30}")
        for item in display_blast:
            tier = item["exposure_tier"]
            adj = item["total_exposure"]
            base = item.get("base_exposure", adj)
            sim = item.get("embedding_similarity", 0.5)
            delta = item.get("similarity_delta", 0.0)
            entity = item["entity"]
            delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
            print(f"  {tier:>8s} | {adj:5.1%} | {base:5.1%} | {sim:5.2f} | {delta_str:>6s} | {entity}")
    else:
        for item in display_blast:
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

    print(f"{'='*78}\n")

    if has_similarity:
        print("  📐 SIM = cosine similarity to shock source (AlphaEarth 64-dim embedding)")
        print("  📐 ADJ = 70% propagation + 30% similarity-weighted exposure\n")

    # ── Save results for regression testing ──
    output = {
        "scenario_id": result["scenario_id"],
        "shock_source": "asset:PORT_HOUSTON",
        "shock_event": "Hurricane Harvey (Category 4, Aug 25 2017)",
        "alphaearth_enriched": args.alphaearth,
        "similarity_adjusted": has_similarity,
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "blast_radius": [
            {
                "entity": item["entity"],
                "exposure_tier": item["exposure_tier"],
                "total_exposure": round(item["total_exposure"], 6),
                **({"base_exposure": round(item["base_exposure"], 6),
                    "embedding_similarity": item["embedding_similarity"],
                    "similarity_delta": item["similarity_delta"]}
                   if has_similarity and "base_exposure" in item else {}),
            }
            for item in display_blast
        ],
    }

    if args.alphaearth:
        enriched = sum(1 for n in G.nodes if "alphaearth_embedding" in G.nodes[n])
        output["alphaearth_nodes"] = enriched

    output_path = os.path.join(demo_dir, "output.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"📊 Results saved: {output_path}")


if __name__ == "__main__":
    main()

