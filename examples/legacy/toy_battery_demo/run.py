#!/usr/bin/env python3
"""
Toy Battery Supply Chain Demo — End-to-End Graphite Pipeline

Documents → Extract → Assemble → Simulate Shock → Print Results

Run: python examples/toy_battery_demo/run.py
"""
import os
import sys
import json

# Add example dir to path for local extractor import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graphite import GraphAssembler, save_graph
from graphite.simulate import top_k_paths_from_source, build_blast_radius
from graphite.graph import clamp
from extractor import extract_from_documents


def main():
    doc_dir = os.path.join(os.path.dirname(__file__), "documents")
    out_dir = os.path.dirname(__file__)

    # ── Step 1: Extract edges from documents ──
    print("🔬 Step 1: Extracting edges from documents...")
    edges = extract_from_documents(doc_dir)
    print(f"   → {len(edges)} total edges extracted\n")

    # ── Step 2: Assemble into graph ──
    print("🔧 Step 2: Assembling graph...")
    G = GraphAssembler().assemble(edges)
    print(f"   → {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    # ── Step 3: Save graph ──
    path = save_graph(G, os.path.join(out_dir, "battery_graph.json"), format="json")
    print(f"💾 Graph saved: {path}\n")

    # ── Step 4: Simulate a cobalt supply shock ──
    print("⚡ Step 3: Simulating CONGO supply shock...")

    def alpha_fn(edge_type, is_supply):
        """Simple attenuation: PRODUCES edges transmit more shock."""
        alphas = {"PRODUCES": 0.8, "SUPPLIES_TO": 0.6, "USED_BY": 0.4}
        return alphas.get(edge_type, 0.3)

    paths, hop_hist = top_k_paths_from_source(
        G, source="country:CD",
        max_hops=3, tau_stop=0.05, k=3,
        is_supply=True, alpha_fn=alpha_fn,
    )
    blast = build_blast_radius(paths, k=3, node_label="entity")

    # ── Print results ──
    print(f"\n{'='*60}")
    print(f"  🔴 CONGO (Cobalt) Supply Shock — Blast Radius")
    print(f"{'='*60}")
    for item in blast:
        tier = item["exposure_tier"]
        score = item["total_exposure"]
        entity = item["entity"]
        print(f"  {tier:>8s} | {score:.1%} | {entity}")
    print(f"{'='*60}\n")

    # ── Save results for regression testing ──
    output = {
        "shock_source": "country:CD",
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "blast_radius": [
            {
                "entity": item["entity"],
                "exposure_tier": item["exposure_tier"],
                "total_exposure": round(item["total_exposure"], 6),
            }
            for item in blast
        ],
    }

    output_path = os.path.join(out_dir, "output.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"📊 Results saved: {output_path}")


if __name__ == "__main__":
    main()
