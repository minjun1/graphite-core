#!/usr/bin/env python3
"""
AlphaEarth Ablation Study — Shows how embedding signals change blast radius.

Runs Harvey demo in 4 modes and compares results:
  A. Base propagation only (no AlphaEarth)
  B. + Level 1: Post-hoc similarity adjustment
  C. + Level 2: Structural edge weight modulation  ← NEW
  D. + Both levels combined

This proves AlphaEarth is a real signal, not cosmetic.
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "flood_replay_demo"))

from graphite import GraphAssembler, ScenarioShock, ScenarioRunner
from graphite.enums import SourceType
from graphite.features.alphaearth_enricher import AlphaEarthEnricher
from graphite.features.embedding_similarity import (
    compute_similarity_scores, adjust_blast_radius,
    inject_edge_similarity, make_embedding_aware_alpha,
)
from extractor import extract_from_documents


def base_alpha_fn(edge_type, is_supply):
    alphas = {
        "RISK_FLOWS_TO": 0.8,
        "DEPENDS_ON": 0.7,
        "LOCATED_IN": 0.5,
        "EXPOSED_TO": 0.9,
        "ADJACENT_TO": 0.3,
    }
    return alphas.get(edge_type, 0.4)


def run_scenario(G, alpha_fn):
    harvey = ScenarioShock(
        shock_id="hurricane_harvey_2017",
        target_nodes=["asset:PORT_HOUSTON"],
        intensity=0.90,
        observed_at="2017-08-25T22:00:00Z",
        evidence="Category 4 hurricane, Port of Houston closed Aug 25-31",
        source_type=SourceType.WEATHER_FORECAST,
    )
    runner = ScenarioRunner()
    return runner.run(G, shocks=[harvey], max_hops=5, tau_stop=0.03,
                      k=5, is_supply=True, alpha_fn=alpha_fn)


def print_row(label, blast, entity):
    for item in blast:
        if item["entity"] == entity:
            return f"{item['total_exposure']:6.1%}"
    return "   N/A"


def main():
    demo_dir = os.path.join(os.path.dirname(__file__), "..", "flood_replay_demo")
    doc_dir = os.path.join(demo_dir, "documents")
    geom_path = os.path.join(demo_dir, "node_geometries.json")
    cache_dir = os.path.join(demo_dir, "cache", "alphaearth")

    # Build graph
    edges = extract_from_documents(doc_dir)
    G = GraphAssembler().assemble(edges)

    # ═══════════════════════════════════════════════════════
    # Mode A: Base propagation (no AlphaEarth)
    # ═══════════════════════════════════════════════════════
    result_a = run_scenario(G, base_alpha_fn)
    blast_a = result_a["blast_radius"]

    # ═══════════════════════════════════════════════════════
    # Enrich with AlphaEarth
    # ═══════════════════════════════════════════════════════
    enricher = AlphaEarthEnricher(cache_dir=cache_dir)
    geometries = enricher.load_geometries(geom_path)
    enricher.enrich(G, geometries, year=2017)

    # ═══════════════════════════════════════════════════════
    # Mode B: + Level 1 post-hoc similarity adjustment
    # ═══════════════════════════════════════════════════════
    result_b_raw = run_scenario(G, base_alpha_fn)
    sim_scores = compute_similarity_scores(G, ["asset:PORT_HOUSTON"])
    blast_b = adjust_blast_radius(result_b_raw["blast_radius"], sim_scores)

    # ═══════════════════════════════════════════════════════
    # Mode C: + Level 2 structural edge modulation
    # ═══════════════════════════════════════════════════════
    n_annotated = inject_edge_similarity(G)
    emb_alpha = make_embedding_aware_alpha(G, base_alpha_fn, embedding_weight=0.4)
    result_c = run_scenario(G, emb_alpha)
    blast_c = result_c["blast_radius"]

    # ═══════════════════════════════════════════════════════
    # Mode D: Both levels combined
    # ═══════════════════════════════════════════════════════
    blast_d = adjust_blast_radius(blast_c, sim_scores, weight=0.3)

    # ═══════════════════════════════════════════════════════
    # Print ablation table
    # ═══════════════════════════════════════════════════════
    entities = [item["entity"] for item in blast_a]

    print()
    print("=" * 88)
    print("  📐 AlphaEarth Ablation Study — Hurricane Harvey 2017")
    print("  How much does AlphaEarth embedding signal change the blast radius?")
    print("=" * 88)
    print()
    print(f"  {'ENTITY':<42s} | {'(A)':>6s} | {'(B)':>6s} | {'(C)':>6s} | {'(D)':>6s} | {'Δ A→D':>7s}")
    print(f"  {'':-<42s} | {'':-<6s} | {'':-<6s} | {'':-<6s} | {'':-<6s} | {'':-<7s}")

    for entity in entities:
        a_val = next((i["total_exposure"] for i in blast_a if i["entity"] == entity), None)
        b_val = next((i["total_exposure"] for i in blast_b if i["entity"] == entity), None)
        c_val = next((i["total_exposure"] for i in blast_c if i["entity"] == entity), None)
        d_val = next((i["total_exposure"] for i in blast_d if i["entity"] == entity), None)

        if a_val is not None and d_val is not None:
            delta = d_val - a_val
            delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
        else:
            delta_str = "  N/A"

        a_s = f"{a_val:5.1%}" if a_val else "  N/A"
        b_s = f"{b_val:5.1%}" if b_val else "  N/A"
        c_s = f"{c_val:5.1%}" if c_val else "  N/A"
        d_s = f"{d_val:5.1%}" if d_val else "  N/A"

        print(f"  {entity:<42s} | {a_s:>6s} | {b_s:>6s} | {c_s:>6s} | {d_s:>6s} | {delta_str:>7s}")

    print()
    print("  Legend:")
    print("    (A) Base propagation only")
    print("    (B) + Level 1: post-hoc similarity re-weighting")
    print("    (C) + Level 2: structural edge modulation (embedding-aware alpha)")
    print("    (D) + Both levels combined")
    print(f"    Edges annotated with embedding similarity: {n_annotated}")
    print()

    # Save ablation results
    output = {
        "ablation": "alphaearth_embedding_impact",
        "modes": {
            "A_base": [{"entity": i["entity"], "exposure": round(i["total_exposure"], 4)} for i in blast_a],
            "B_posthoc": [{"entity": i["entity"], "exposure": round(i["total_exposure"], 4)} for i in blast_b],
            "C_structural": [{"entity": i["entity"], "exposure": round(i["total_exposure"], 4)} for i in blast_c],
            "D_combined": [{"entity": i["entity"], "exposure": round(i["total_exposure"], 4)} for i in blast_d],
        }
    }
    out_path = os.path.join(os.path.dirname(__file__), "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  📊 Saved: {out_path}")
    print()


if __name__ == "__main__":
    main()
