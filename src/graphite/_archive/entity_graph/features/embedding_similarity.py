"""
graphite/features/embedding_similarity.py — Embedding-based risk scoring.

Two levels of embedding influence:

Level 1 (post-hoc): adjust_blast_radius — re-weights final scores
Level 2 (structural): make_embedding_aware_alpha — modulates edge weights
  DURING propagation so that environmentally similar connected nodes
  transmit risk more strongly.

Level 2 is the stronger effect: it changes which paths are explored
and how much risk flows through each edge. This is what makes
AlphaEarth a real signal, not cosmetic.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0.0 on degenerate inputs."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── Level 2: Structural embedding-aware propagation ──


def inject_edge_similarity(G: nx.DiGraph) -> int:
    """Pre-compute pairwise embedding similarity for all edges.

    For each edge (u, v) where both nodes have AlphaEarth embeddings,
    compute cosine similarity and store as edge attribute:
        G[u][v]["embedding_similarity"] = float (0.0 to 1.0)

    This is read by make_embedding_aware_alpha during propagation.

    Args:
        G: Graph with AlphaEarth-enriched nodes

    Returns:
        Number of edges annotated
    """
    annotated = 0
    for u, v in G.edges():
        u_emb = G.nodes[u].get("alphaearth_embedding")
        v_emb = G.nodes[v].get("alphaearth_embedding")

        if u_emb is not None and v_emb is not None:
            raw = cosine_similarity(np.array(u_emb), np.array(v_emb))
            G[u][v]["embedding_similarity"] = (raw + 1.0) / 2.0
            annotated += 1
        else:
            G[u][v]["embedding_similarity"] = 0.5

    return annotated


def make_embedding_aware_alpha(
    G: nx.DiGraph,
    base_alpha_fn: Callable[[str, bool], float],
    embedding_weight: float = 0.4,
) -> Callable[[str, bool], float]:
    """Create an alpha function that modulates edge weights by embedding similarity.

    The returned function wraps the base alpha_fn. Since the propagation engine
    only passes (edge_type, is_supply), we aggregate by edge type:

        For each edge_type, compute mean embedding_similarity across all edges.
        effective_alpha = base_alpha * (1 + weight * (mean_sim - 0.5) * 2)

    This means:
    - Edge types connecting similar environments: +20-40% stronger propagation
    - Edge types connecting dissimilar environments: -20-40% weaker propagation

    With weight=0.4 and mean_sim=0.75: alpha × 1.2 (+20%)
    With weight=0.4 and mean_sim=0.25: alpha × 0.8 (-20%)

    IMPORTANT: inject_edge_similarity must be called first.

    Args:
        G: Graph with embedding_similarity edge attributes
        base_alpha_fn: Original alpha function
        embedding_weight: How much similarity modulates alpha (0.0–1.0)

    Returns:
        New alpha function with same (edge_type, is_supply) signature
    """
    # Pre-compute edge-type → mean similarity
    type_similarities: Dict[str, List[float]] = {}
    for u, v, data in G.edges(data=True):
        etype = data.get("edge_type", data.get("type", "UNKNOWN"))
        sim = data.get("embedding_similarity", 0.5)
        type_similarities.setdefault(etype, []).append(sim)

    type_mean_sim = {
        etype: sum(sims) / len(sims)
        for etype, sims in type_similarities.items()
    }

    def embedding_alpha(edge_type: str, is_supply: bool) -> float:
        base = base_alpha_fn(edge_type, is_supply)
        mean_sim = type_mean_sim.get(edge_type, 0.5)

        # Modulate: sim > 0.5 → boost, sim < 0.5 → dampen
        modifier = 1.0 + embedding_weight * (mean_sim - 0.5) * 2
        modifier = max(0.3, min(1.7, modifier))

        return base * modifier

    return embedding_alpha


# ── Level 1: Post-hoc adjustment ──


def compute_similarity_scores(
    G: nx.DiGraph,
    source_nodes: List[str],
) -> Dict[str, float]:
    """Compute cosine similarity between source nodes and all other enriched nodes.

    Returns:
        Dict of node_id → similarity score (0.0 to 1.0).
        Source nodes get 1.0. Non-enriched nodes get 0.5 (neutral).
    """
    source_embeddings = []
    for sn in source_nodes:
        if sn in G and "alphaearth_embedding" in G.nodes[sn]:
            source_embeddings.append(np.array(G.nodes[sn]["alphaearth_embedding"]))

    if not source_embeddings:
        return {n: 0.5 for n in G.nodes}

    source_vec = np.mean(source_embeddings, axis=0)

    scores = {}
    for node in G.nodes:
        if node in source_nodes:
            scores[node] = 1.0
        elif "alphaearth_embedding" in G.nodes[node]:
            node_vec = np.array(G.nodes[node]["alphaearth_embedding"])
            raw_sim = cosine_similarity(source_vec, node_vec)
            scores[node] = (raw_sim + 1.0) / 2.0
        else:
            scores[node] = 0.5

    return scores


def adjust_blast_radius(
    blast_radius: List[Dict],
    similarity_scores: Dict[str, float],
    weight: float = 0.3,
) -> List[Dict]:
    """Adjust blast radius exposures using embedding similarity (Level 1).

    adjusted = (1 - weight) * original + weight * (original * similarity * 2)
    """
    adjusted = []
    for item in blast_radius:
        entity = item["entity"]
        original_exposure = item["total_exposure"]
        sim = similarity_scores.get(entity, 0.5)

        adjusted_exposure = (1 - weight) * original_exposure + weight * (original_exposure * sim * 2)
        adjusted_exposure = max(0.0, min(1.0, adjusted_exposure))

        new_item = dict(item)
        new_item["total_exposure"] = adjusted_exposure
        new_item["base_exposure"] = original_exposure
        new_item["embedding_similarity"] = round(sim, 4)
        new_item["similarity_delta"] = round(adjusted_exposure - original_exposure, 4)

        if adjusted_exposure >= 0.5:
            new_item["exposure_tier"] = "EXTREME"
        elif adjusted_exposure >= 0.3:
            new_item["exposure_tier"] = "HIGH"
        elif adjusted_exposure >= 0.15:
            new_item["exposure_tier"] = "MEDIUM"
        elif adjusted_exposure >= 0.05:
            new_item["exposure_tier"] = "LOW"
        else:
            new_item["exposure_tier"] = "MINIMAL"

        adjusted.append(new_item)

    adjusted.sort(key=lambda x: x["total_exposure"], reverse=True)
    return adjusted
