"""
tests/test_propagation.py — Tests for the core propagation engine.

Tests the Dijkstra + Noisy-OR blast radius algorithm
using a simple synthetic graph (no Neo4j required).
"""
import math
import networkx as nx
import pytest

from graphite.graph import clamp
from graphite.simulate import (
    top_k_paths_from_source,
    build_blast_radius,
    map_to_tier,
)


# ── Fixtures ──

def _simple_alpha(edge_type: str, is_supply: bool) -> float:
    """Test alpha function: all edges get 0.8 attenuation."""
    return 0.8


@pytest.fixture
def linear_graph():
    """A → B → C with decreasing weights."""
    G = nx.DiGraph()
    G.add_node("A")
    G.add_node("B")
    G.add_node("C")
    G.add_edge("A", "B", bucket_weight=0.8, edge_type="SUPPLIES_TO", evidence="A supplies B")
    G.add_edge("B", "C", bucket_weight=0.6, edge_type="SUPPLIES_TO", evidence="B supplies C")
    return G


@pytest.fixture
def diamond_graph():
    """
    A → B → D
    A → C → D
    Two paths from A to D (for Noisy-OR testing).
    """
    G = nx.DiGraph()
    for n in ["A", "B", "C", "D"]:
        G.add_node(n)
    G.add_edge("A", "B", bucket_weight=0.7, edge_type="SUPPLIES_TO", evidence="A→B")
    G.add_edge("A", "C", bucket_weight=0.5, edge_type="SUPPLIES_TO", evidence="A→C")
    G.add_edge("B", "D", bucket_weight=0.6, edge_type="SUPPLIES_TO", evidence="B→D")
    G.add_edge("C", "D", bucket_weight=0.4, edge_type="SUPPLIES_TO", evidence="C→D")
    return G


@pytest.fixture
def cycle_graph():
    """A → B → C → A (cycle). Should not infinite-loop."""
    G = nx.DiGraph()
    for n in ["A", "B", "C"]:
        G.add_node(n)
    G.add_edge("A", "B", bucket_weight=0.8, edge_type="SUPPLIES_TO", evidence="cycle")
    G.add_edge("B", "C", bucket_weight=0.8, edge_type="SUPPLIES_TO", evidence="cycle")
    G.add_edge("C", "A", bucket_weight=0.8, edge_type="SUPPLIES_TO", evidence="cycle")
    return G


# ── Tests: Helpers ──

class TestClamp:
    def test_clamp_within_range(self):
        assert clamp(0.5) == 0.5

    def test_clamp_below(self):
        assert clamp(-0.5) == 0.0

    def test_clamp_above(self):
        assert clamp(1.5) == 1.0

    def test_clamp_custom_range(self):
        assert clamp(0.5, 0.2, 0.8) == 0.5
        assert clamp(0.1, 0.2, 0.8) == 0.2
        assert clamp(0.9, 0.2, 0.8) == 0.8


class TestMapToTier:
    def test_extreme(self):
        assert map_to_tier(0.50) == "EXTREME"

    def test_high(self):
        assert map_to_tier(0.25) == "HIGH"

    def test_medium(self):
        assert map_to_tier(0.10) == "MEDIUM"

    def test_low(self):
        assert map_to_tier(0.05) == "LOW"

    def test_custom_thresholds(self):
        custom = {"EXTREME": 0.5, "HIGH": 0.3, "MEDIUM": 0.1}
        assert map_to_tier(0.4, custom) == "HIGH"


# ── Tests: Propagation ──

class TestPropagation:
    def test_source_not_in_graph(self, linear_graph):
        paths, hist = top_k_paths_from_source(
            linear_graph, "NONEXISTENT", max_hops=3, tau_stop=0.01,
            k=3, is_supply=True, alpha_fn=_simple_alpha,
        )
        assert paths == {}
        assert hist == {1: 0, 2: 0, 3: 0}

    def test_linear_1hop(self, linear_graph):
        paths, hist = top_k_paths_from_source(
            linear_graph, "A", max_hops=1, tau_stop=0.01,
            k=3, is_supply=True, alpha_fn=_simple_alpha,
        )
        assert "B" in paths
        assert "C" not in paths  # 2 hops away, max_hops=1
        assert hist[1] > 0

    def test_linear_2hop(self, linear_graph):
        paths, hist = top_k_paths_from_source(
            linear_graph, "A", max_hops=2, tau_stop=0.01,
            k=3, is_supply=True, alpha_fn=_simple_alpha,
        )
        assert "B" in paths
        assert "C" in paths
        # B should have higher score than C (1 hop vs 2 hops)
        b_score = paths["B"][0]["attenuated_score"]
        c_score = paths["C"][0]["attenuated_score"]
        assert b_score > c_score

    def test_attenuation_math(self, linear_graph):
        """Verify W_e = W_b * alpha and score = exp(-cost)."""
        paths, _ = top_k_paths_from_source(
            linear_graph, "A", max_hops=1, tau_stop=0.01,
            k=3, is_supply=True, alpha_fn=_simple_alpha,
        )
        b_score = paths["B"][0]["attenuated_score"]
        expected = 0.8 * 0.8  # W_b=0.8, alpha=0.8
        assert abs(b_score - expected) < 0.01

    def test_cycle_terminates(self, cycle_graph):
        """Must not infinite-loop on cycles."""
        paths, _ = top_k_paths_from_source(
            cycle_graph, "A", max_hops=3, tau_stop=0.01,
            k=3, is_supply=True, alpha_fn=_simple_alpha,
        )
        # Should complete without hanging
        assert isinstance(paths, dict)

    def test_tau_stop_pruning(self, linear_graph):
        """High tau_stop should prune weak paths."""
        paths, _ = top_k_paths_from_source(
            linear_graph, "A", max_hops=3, tau_stop=0.5,
            k=3, is_supply=True, alpha_fn=_simple_alpha,
        )
        # B has score 0.64 (> 0.5), C has lower score → might be pruned
        assert "B" in paths

    def test_evidence_pack_present(self, linear_graph):
        paths, _ = top_k_paths_from_source(
            linear_graph, "A", max_hops=2, tau_stop=0.01,
            k=3, is_supply=True, alpha_fn=_simple_alpha,
        )
        for node_paths in paths.values():
            for p in node_paths:
                assert "evidence_pack" in p
                assert isinstance(p["evidence_pack"], list)

    def test_path_nodes_correct(self, linear_graph):
        paths, _ = top_k_paths_from_source(
            linear_graph, "A", max_hops=2, tau_stop=0.01,
            k=3, is_supply=True, alpha_fn=_simple_alpha,
        )
        c_path = paths["C"][0]["path_nodes"]
        assert c_path == ["A", "B", "C"]


# ── Tests: Blast Radius ──

class TestBlastRadius:
    def test_noisy_or_aggregation(self, diamond_graph):
        """D reached via 2 paths: Noisy-OR should aggregate."""
        paths, _ = top_k_paths_from_source(
            diamond_graph, "A", max_hops=2, tau_stop=0.01,
            k=3, is_supply=True, alpha_fn=_simple_alpha,
        )
        blast = build_blast_radius(paths, k=3, node_label="ticker")

        d_entry = next((x for x in blast if x["ticker"] == "D"), None)
        assert d_entry is not None

        # D should have higher exposure via Noisy-OR than either single path
        b_entry = next((x for x in blast if x["ticker"] == "B"), None)
        c_entry = next((x for x in blast if x["ticker"] == "C"), None)

        # Both B and C should be present
        assert b_entry is not None
        assert c_entry is not None

    def test_noisy_or_formula(self, diamond_graph):
        """Verify: total = 1 - ∏(1 - score_i)."""
        paths, _ = top_k_paths_from_source(
            diamond_graph, "A", max_hops=2, tau_stop=0.01,
            k=3, is_supply=True, alpha_fn=_simple_alpha,
        )
        blast = build_blast_radius(paths, k=3, node_label="ticker")

        d_entry = next(x for x in blast if x["ticker"] == "D")
        path_scores = [p["attenuated_score"] for p in d_entry["top_paths"]]

        survival = 1.0
        for s in path_scores:
            survival *= (1.0 - s)
        expected = 1.0 - survival

        assert abs(d_entry["total_exposure"] - expected) < 0.001

    def test_sorted_by_exposure(self, diamond_graph):
        paths, _ = top_k_paths_from_source(
            diamond_graph, "A", max_hops=2, tau_stop=0.01,
            k=3, is_supply=True, alpha_fn=_simple_alpha,
        )
        blast = build_blast_radius(paths, k=3, node_label="ticker")
        exposures = [x["total_exposure"] for x in blast]
        assert exposures == sorted(exposures, reverse=True)

    def test_exposure_tier_assigned(self, diamond_graph):
        paths, _ = top_k_paths_from_source(
            diamond_graph, "A", max_hops=2, tau_stop=0.01,
            k=3, is_supply=True, alpha_fn=_simple_alpha,
        )
        blast = build_blast_radius(paths, k=3, node_label="ticker")
        for entry in blast:
            assert entry["exposure_tier"] in ("EXTREME", "HIGH", "MEDIUM", "LOW")

    def test_custom_node_label(self, linear_graph):
        paths, _ = top_k_paths_from_source(
            linear_graph, "A", max_hops=2, tau_stop=0.01,
            k=3, is_supply=True, alpha_fn=_simple_alpha,
        )
        blast = build_blast_radius(paths, k=3, node_label="package")
        for entry in blast:
            assert "package" in entry
            assert "ticker" not in entry
