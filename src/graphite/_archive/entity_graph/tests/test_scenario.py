"""
tests/test_scenario.py — Tests for the ScenarioRunner and ScenarioShock.

Verifies that the thin scenario orchestration layer correctly:
- Injects shocks at specified nodes
- Propagates through the graph
- Scales by shock intensity
- Produces valid blast radius output
"""
import math
import networkx as nx
import pytest

from graphite import ScenarioShock, ScenarioRunner
from graphite.enums import SourceType


# ── Fixtures ──

def _simple_alpha(edge_type: str, is_supply: bool) -> float:
    """Test alpha function: all edges get 0.7 attenuation."""
    return 0.7


@pytest.fixture
def infrastructure_graph():
    """
    Houston infrastructure graph for testing:

    region:HOUSTON → facility:PORT (RISK_FLOWS_TO, w=0.8)
    facility:PORT → facility:REFINERY (RISK_FLOWS_TO, w=0.6)
    facility:REFINERY → company:CHEMICAL_CO (RISK_FLOWS_TO, w=0.5)
    region:HOUSTON → facility:GRID_NODE (RISK_FLOWS_TO, w=0.7)
    """
    G = nx.DiGraph()

    G.add_node("region:HOUSTON", entity="region:HOUSTON")
    G.add_node("facility:PORT", entity="facility:PORT")
    G.add_node("facility:REFINERY", entity="facility:REFINERY")
    G.add_node("company:CHEMICAL_CO", entity="company:CHEMICAL_CO")
    G.add_node("facility:GRID_NODE", entity="facility:GRID_NODE")

    G.add_edge("region:HOUSTON", "facility:PORT",
               bucket_weight=0.8, edge_type="RISK_FLOWS_TO",
               cost=-math.log(0.8 * 0.7),
               evidence="Port shut for 6 days during Harvey")
    G.add_edge("facility:PORT", "facility:REFINERY",
               bucket_weight=0.6, edge_type="RISK_FLOWS_TO",
               cost=-math.log(0.6 * 0.7),
               evidence="Refinery depends on port for crude delivery")
    G.add_edge("facility:REFINERY", "company:CHEMICAL_CO",
               bucket_weight=0.5, edge_type="RISK_FLOWS_TO",
               cost=-math.log(0.5 * 0.7),
               evidence="Chemical Co sources feedstock from refinery")
    G.add_edge("region:HOUSTON", "facility:GRID_NODE",
               bucket_weight=0.7, edge_type="RISK_FLOWS_TO",
               cost=-math.log(0.7 * 0.7),
               evidence="Grid node located in Houston flood zone")

    return G


# ── Tests ──

class TestScenarioShock:
    def test_defaults(self):
        shock = ScenarioShock(
            shock_id="test-shock",
            target_nodes=["region:HOUSTON"],
        )
        assert shock.intensity == 0.85
        assert shock.source_type == SourceType.PUBLIC_REPORT
        assert shock.observed_at == ""

    def test_custom_values(self):
        shock = ScenarioShock(
            shock_id="hurricane_harvey_2017",
            target_nodes=["region:HOUSTON", "region:GALVESTON"],
            intensity=0.9,
            observed_at="2017-08-25T22:00:00Z",
            evidence="Category 4 hurricane made landfall near Rockport, TX",
            source_type=SourceType.WEATHER_FORECAST,
        )
        assert len(shock.target_nodes) == 2
        assert shock.intensity == 0.9


class TestScenarioRunner:
    def test_single_shock_propagation(self, infrastructure_graph):
        runner = ScenarioRunner()
        result = runner.run(
            infrastructure_graph,
            shocks=[ScenarioShock(
                shock_id="harvey",
                target_nodes=["region:HOUSTON"],
                intensity=0.85,
            )],
            max_hops=4,
            alpha_fn=_simple_alpha,
        )

        assert result["scenario_id"] == "harvey"
        assert len(result["blast_radius"]) > 0
        assert "paths_by_node" in result

        # PORT should be hit (1-hop)
        entities = [item["entity"] for item in result["blast_radius"]]
        assert "facility:PORT" in entities

    def test_multi_node_shock(self, infrastructure_graph):
        runner = ScenarioRunner()
        result = runner.run(
            infrastructure_graph,
            shocks=[ScenarioShock(
                shock_id="dual-shock",
                target_nodes=["region:HOUSTON"],
                intensity=0.9,
            )],
            max_hops=4,
            alpha_fn=_simple_alpha,
        )

        # Should reach downstream nodes
        entities = [item["entity"] for item in result["blast_radius"]]
        assert len(entities) >= 2  # at least PORT and GRID_NODE

    def test_intensity_scales_exposure(self, infrastructure_graph):
        runner = ScenarioRunner()

        # Run with high intensity
        result_high = runner.run(
            infrastructure_graph,
            shocks=[ScenarioShock(
                shock_id="strong",
                target_nodes=["region:HOUSTON"],
                intensity=1.0,
            )],
            max_hops=4,
            alpha_fn=_simple_alpha,
        )

        # Run with low intensity
        result_low = runner.run(
            infrastructure_graph,
            shocks=[ScenarioShock(
                shock_id="weak",
                target_nodes=["region:HOUSTON"],
                intensity=0.3,
            )],
            max_hops=4,
            alpha_fn=_simple_alpha,
        )

        # Higher intensity should produce higher exposure
        def get_exposure(result, entity):
            for item in result["blast_radius"]:
                if item["entity"] == entity:
                    return item["total_exposure"]
            return 0.0

        assert get_exposure(result_high, "facility:PORT") > get_exposure(result_low, "facility:PORT")

    def test_nonexistent_source_graceful(self, infrastructure_graph):
        runner = ScenarioRunner()
        result = runner.run(
            infrastructure_graph,
            shocks=[ScenarioShock(
                shock_id="missing",
                target_nodes=["region:NONEXISTENT"],
            )],
            max_hops=4,
            alpha_fn=_simple_alpha,
        )

        # Should return empty blast radius, not crash
        assert result["blast_radius"] == []

    def test_shock_metadata_preserved(self, infrastructure_graph):
        runner = ScenarioRunner()
        result = runner.run(
            infrastructure_graph,
            shocks=[ScenarioShock(
                shock_id="harvey",
                target_nodes=["region:HOUSTON"],
                intensity=0.85,
                observed_at="2017-08-25",
                evidence="Cat 4 hurricane",
            )],
            max_hops=4,
            alpha_fn=_simple_alpha,
        )

        assert len(result["shocks"]) == 1
        assert result["shocks"][0]["shock_id"] == "harvey"
        assert result["shocks"][0]["evidence"] == "Cat 4 hurricane"
        assert result["shocks"][0]["observed_at"] == "2017-08-25"
