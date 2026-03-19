"""
graphite/scenario.py — Thin scenario runner for shock propagation.

ScenarioShock is a runtime input, NOT a persistent graph node.
Hazards (hurricanes, floods, etc.) are injected as shocks at simulation time
and propagated through the graph to produce an impacted subgraph.

v1: shock inject → propagate → return blast radius. No branching, no comparison.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx

from .enums import SourceType
from .simulate import top_k_paths_from_source, build_blast_radius


@dataclass
class ScenarioShock:
    """A runtime shock injection — NOT a graph node.

    Hazards are modeled as scenario inputs, not permanent entities.
    This keeps the graph clean: only physical/economic entities live
    in the graph; shocks are injected at simulation time.

    Example:
        ScenarioShock(
            shock_id="hurricane_harvey_2017",
            target_nodes=["region:US-TX-HOUSTON"],
            intensity=0.85,
            observed_at="2017-08-25",
            evidence="Category 4 hurricane made landfall near Rockport, TX",
            source_type=SourceType.WEATHER_FORECAST,
        )
    """

    shock_id: str  # unique identifier, e.g. "hurricane_harvey_2017"
    target_nodes: List[str]  # graph nodes receiving the initial shock
    intensity: float = 0.85  # 0.0–1.0 shock strength
    observed_at: str = ""  # when the event occurred / was forecast
    evidence: str = ""  # source evidence backing this shock
    source_type: SourceType = SourceType.PUBLIC_REPORT
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScenarioRunner:
    """Run shock scenarios through an evidence graph.

    v1 is intentionally minimal — a thin orchestration wrapper around
    the existing propagation engine. It:
    1. Injects shocks as virtual source nodes
    2. Runs propagation from each shock source
    3. Merges and returns the impacted subgraph

    Usage:
        runner = ScenarioRunner()
        result = runner.run(
            G,
            shocks=[ScenarioShock(...)],
            max_hops=4,
            alpha_fn=my_alpha_fn,
        )
        for item in result["blast_radius"]:
            print(item["entity"], item["exposure_tier"])
    """

    def run(
        self,
        G: nx.DiGraph,
        shocks: List[ScenarioShock],
        max_hops: int = 4,
        tau_stop: float = 0.05,
        k: int = 5,
        is_supply: bool = True,
        alpha_fn: Optional[Callable[[str, bool], float]] = None,
        make_evidence_fn: Optional[Callable] = None,
        node_filter_fn: Optional[Callable] = None,
        edge_filter_fn: Optional[Callable] = None,
        tier_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Inject shocks → propagate → return impacted subgraph.

        Args:
            G: The evidence graph to propagate through
            shocks: List of ScenarioShock objects to inject
            max_hops: Maximum propagation depth
            tau_stop: Minimum score threshold for path pruning
            k: Number of alternative paths per node
            is_supply: Direction flag for propagation
            alpha_fn: Edge attenuation function (required)
            make_evidence_fn: Optional domain-specific evidence builder
            node_filter_fn: Optional node filter
            edge_filter_fn: Optional edge filter
            tier_thresholds: Optional custom tier mapping

        Returns:
            Dict with keys:
                - scenario_id: combined shock IDs
                - shocks: list of shock metadata
                - blast_radius: aggregated blast radius output
                - paths_by_node: raw paths per impacted node
        """
        if alpha_fn is None:
            alpha_fn = lambda edge_type, is_supply: 0.5

        # Merge paths from all shock sources
        all_paths: Dict[str, List[Dict[str, Any]]] = {}

        shock_meta = []
        for shock in shocks:
            shock_meta.append(
                {
                    "shock_id": shock.shock_id,
                    "target_nodes": shock.target_nodes,
                    "intensity": shock.intensity,
                    "observed_at": shock.observed_at,
                    "evidence": shock.evidence,
                }
            )

            for source_node in shock.target_nodes:
                if source_node not in G:
                    continue

                paths, _ = top_k_paths_from_source(
                    G,
                    source=source_node,
                    max_hops=max_hops,
                    tau_stop=tau_stop,
                    k=k,
                    is_supply=is_supply,
                    alpha_fn=alpha_fn,
                    make_evidence_fn=make_evidence_fn,
                    node_filter_fn=node_filter_fn,
                    edge_filter_fn=edge_filter_fn,
                )

                # Scale paths by shock intensity
                for node, node_paths in paths.items():
                    for p in node_paths:
                        p["attenuated_score"] *= shock.intensity
                    all_paths.setdefault(node, []).extend(node_paths)

        # Build unified blast radius
        blast = build_blast_radius(
            all_paths,
            k=k,
            tier_thresholds=tier_thresholds,
            node_label="entity",
        )

        scenario_id = "+".join(s.shock_id for s in shocks)

        return {
            "scenario_id": scenario_id,
            "shocks": shock_meta,
            "blast_radius": blast,
            "paths_by_node": all_paths,
        }
