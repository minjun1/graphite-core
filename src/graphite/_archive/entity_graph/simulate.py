"""
graphite/simulate.py — Domain-agnostic shock propagation engine.

Implements Dijkstra's algorithm with Noisy-OR aggregation to simulate
how a disruption at one node cascades through a supply chain graph.

This is Graphite's core differentiator: mathematical blast radius simulation
with full evidence traceability.
"""

import math
import heapq
import networkx as nx
from typing import Dict, Any, List, Tuple, Callable, Optional

from .graph import clamp


# ── Default tier mapping (can be overridden by domain config) ──
DEFAULT_TIER_THRESHOLDS = {
    "EXTREME": 0.35,
    "HIGH": 0.20,
    "MEDIUM": 0.08,
}


def map_to_tier(score: float, thresholds: Dict[str, float] = None) -> str:
    """Map a float score to a human-readable tier."""
    t = thresholds or DEFAULT_TIER_THRESHOLDS
    if score >= t.get("EXTREME", 0.35):
        return "EXTREME"
    if score >= t.get("HIGH", 0.20):
        return "HIGH"
    if score >= t.get("MEDIUM", 0.08):
        return "MEDIUM"
    return "LOW"


def top_k_paths_from_source(
    G: nx.DiGraph,
    source: str,
    max_hops: int,
    tau_stop: float,
    k: int,
    is_supply: bool,
    alpha_fn: Callable[[str, bool], float],
    make_evidence_fn: Optional[Callable] = None,
    node_filter_fn: Optional[Callable] = None,
    edge_filter_fn: Optional[Callable] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[int, int]]:
    """
    Find top-K strongest paths from a source node using modified Dijkstra.

    This is the core blast radius algorithm. It works for any domain as long as:
    - Edges have a `bucket_weight` attribute (0.0 - 1.0)
    - A domain-specific alpha function maps edge types to attenuation factors

    Args:
        G: NetworkX DiGraph with weighted edges
        source: Starting node identifier
        max_hops: Maximum path length
        tau_stop: Minimum score threshold (prune paths below this)
        k: Number of alternative paths to keep per node
        is_supply: Direction flag (supply=forward, demand=reverse display)
        alpha_fn: Domain-specific function: (edge_type, is_supply) → attenuation float
        make_evidence_fn: Optional domain-specific function to build evidence pack entries
        node_filter_fn: Optional function: (node_id, node_data) → bool (False = skip)
        edge_filter_fn: Optional function: (src, tgt, edge_data) → bool (False = skip)

    Returns:
        (paths_by_node, hop_histogram)
    """
    source = source.strip()
    # Try exact match first, then uppercase (legacy compatibility)
    if source not in G:
        source_upper = source.upper()
        if source_upper in G:
            source = source_upper
    if source not in G:
        return {}, {1: 0, 2: 0, 3: 0}

    max_cost = -math.log(tau_stop) if tau_stop > 0 else float("inf")
    best_scores: Dict[str, List[float]] = {}
    results: Dict[str, List[Dict[str, Any]]] = {}
    hop_hist: Dict[int, int] = {i: 0 for i in range(1, max_hops + 1)}

    # Heap: (cost_so_far, hops, node, path_nodes, breakdown, evidence_pack)
    heap: List[Tuple[float, int, str, List[str], List[Dict], List[Dict]]] = []
    heapq.heappush(heap, (0.0, 0, source, [source], [], []))

    nodes_visited = 0

    while heap:
        nodes_visited += 1
        if nodes_visited > 10000:
            break

        cost_so_far, hops, node, path_nodes, breakdown, evidence_pack = heapq.heappop(
            heap
        )

        if hops > 0:
            score = math.exp(-cost_so_far)

            if score < tau_stop:
                continue

            arr = best_scores.setdefault(node, [])
            if len(arr) >= k and score <= min(arr):
                continue

            arr.append(score)
            arr.sort(reverse=True)
            if len(arr) > k:
                arr.pop()

            safe_path_nodes = (
                path_nodes if isinstance(path_nodes, list) else [source, node]
            )

            results.setdefault(node, []).append(
                {
                    "path_nodes": safe_path_nodes
                    if is_supply
                    else list(reversed(safe_path_nodes)),
                    "attenuated_score": float(score),
                    "path_score_breakdown": breakdown,
                    "evidence_pack": evidence_pack
                    if isinstance(evidence_pack, list)
                    else [],
                }
            )
            # Deterministic tie-breaking: highest score, shortest path, lexicographic
            results[node].sort(
                key=lambda x: (
                    -x["attenuated_score"],
                    len(x["path_nodes"]),
                    "".join(x["path_nodes"]),
                )
            )
            results[node] = results[node][:k]

            if hops in hop_hist:
                hop_hist[hops] += 1

        if hops == max_hops:
            continue
        if cost_so_far > max_cost:
            continue

        for nbr in G.successors(node):
            if nbr in path_nodes:
                continue  # cycle cut

            edge = G[node][nbr]

            # Domain-specific node filtering
            if node_filter_fn and not node_filter_fn(nbr, G.nodes.get(nbr, {})):
                continue

            # Domain-specific edge filtering
            if edge_filter_fn and not edge_filter_fn(node, nbr, edge):
                continue

            # Calculate W_e = W_b * alpha
            edge_type = edge.get("edge_type", edge.get("type", "UNKNOWN"))
            w_b = float(edge.get("bucket_weight", 0.2))
            alpha = alpha_fn(edge_type, is_supply)

            w_e = w_b * alpha
            w_e = clamp(w_e, 0.001, 0.999)
            c = -math.log(w_e)

            new_cost = cost_so_far + c
            if new_cost > max_cost:
                continue

            orig_s, orig_b = (node, nbr) if is_supply else (nbr, node)

            new_breakdown = breakdown + [
                {
                    "hop": hops + 1,
                    "target": nbr,
                    "bucket": round(w_b, 3),
                    "alpha": round(alpha, 3),
                    "attenuation": round(w_e, 3),
                }
            ]

            # Build evidence entry (domain-specific or generic)
            if make_evidence_fn:
                ev_entry = make_evidence_fn(orig_s, orig_b, edge)
            else:
                ev_entry = {
                    "source": orig_s,
                    "target": orig_b,
                    "edge_type": edge_type,
                    "weight": round(w_b, 3),
                    "evidence": edge.get("evidence", ""),
                    "doc_url": edge.get("doc_url", ""),
                }

            new_evidence = evidence_pack + [ev_entry]

            heapq.heappush(
                heap,
                (
                    new_cost,
                    hops + 1,
                    nbr,
                    path_nodes + [nbr],
                    new_breakdown,
                    new_evidence,
                ),
            )

    results.pop(source, None)
    return results, hop_hist


def build_blast_radius(
    paths_by_node: Dict[str, List[Dict[str, Any]]],
    k: int,
    tier_thresholds: Dict[str, float] = None,
    node_label: str = "entity",
) -> List[Dict[str, Any]]:
    """
    Aggregate multi-path exposure using Noisy-OR formula.

    Total_Exposure = 1 - ∏(1 - Score(P)) for all top paths P

    Args:
        paths_by_node: Output from top_k_paths_from_source
        k: Number of top paths to consider per node
        tier_thresholds: Custom tier mapping thresholds
        node_label: Key name for the node identifier in output (e.g., "ticker", "package")
    """
    blast_radius = []

    for node, path_objs in paths_by_node.items():
        if not isinstance(path_objs, list) or len(path_objs) == 0:
            continue

        top_paths = sorted(
            path_objs,
            key=lambda x: x.get("attenuated_score", 0.0),
            reverse=True,
        )[:k]

        # Noisy-OR aggregation
        survival = 1.0
        for p in top_paths:
            survival *= 1.0 - clamp(float(p.get("attenuated_score", 0.0)), 0.0, 0.999)
        total_exposure = 1.0 - survival

        blast_radius.append(
            {
                node_label: node,
                "exposure_tier": map_to_tier(total_exposure, tier_thresholds),
                "total_exposure": float(total_exposure),
                "top_paths": top_paths,
            }
        )

    # Sort: highest exposure first, then lexicographic
    blast_radius.sort(key=lambda x: (-x["total_exposure"], x[node_label]))
    return blast_radius
