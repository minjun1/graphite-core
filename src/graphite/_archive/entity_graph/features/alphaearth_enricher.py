"""
graphite/features/alphaearth_enricher.py — Attach AlphaEarth embeddings to graph nodes.

Takes a NetworkX graph + node geometry mapping, fetches embeddings
via AlphaEarthAdapter, and attaches them as node attributes.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np

from ..adapters.alphaearth import AlphaEarthAdapter, EMBEDDING_DIM
from ..geo_evidence.geo_foundation import make_alphaearth_provenance


class AlphaEarthEnricher:
    """Attach AlphaEarth embeddings to graph nodes as features.

    For each node with a geometry entry, fetches the 64-dim annual
    embedding and stores it as node attributes:
      - alphaearth_embedding: list[float] (64 values)
      - embedding_year: int
      - embedding_source: str
      - alphaearth_provenance: dict (serialized Provenance)

    Usage:
        enricher = AlphaEarthEnricher(cache_dir="cache/alphaearth")
        stats = enricher.enrich(G, node_geometries, year=2017)
        print(f"Enriched {stats['enriched']} / {stats['total']} nodes")
    """

    def __init__(
        self,
        cache_dir: str = "cache/alphaearth",
        billing_project: Optional[str] = None,
    ):
        self.adapter = AlphaEarthAdapter(
            cache_dir=cache_dir,
            billing_project=billing_project,
        )

    def enrich(
        self,
        G: nx.DiGraph,
        node_geometries: Dict[str, Dict[str, Any]],
        year: int = 2017,
    ) -> Dict[str, Any]:
        """Enrich graph nodes with AlphaEarth embeddings.

        Args:
            G: NetworkX graph to enrich (modified in place)
            node_geometries: Mapping of node_id → {"lat": float, "lon": float}
                             Optional: "bbox": [min_lat, min_lon, max_lat, max_lon]
            year: Annual embedding year (2017–2025)

        Returns:
            Stats dict: {"total": int, "enriched": int, "skipped": int, "failed": int}
        """
        stats = {"total": len(node_geometries), "enriched": 0, "skipped": 0, "failed": 0}

        for node_id, geom in node_geometries.items():
            if node_id not in G:
                stats["skipped"] += 1
                continue

            lat = geom.get("lat")
            lon = geom.get("lon")
            bbox = geom.get("bbox")

            if lat is None or lon is None:
                stats["skipped"] += 1
                continue

            # Extract a clean cache key from node_id
            cache_key = node_id.replace(":", "_").replace(" ", "_")

            try:
                if bbox:
                    embedding = self.adapter.get_area_embedding(
                        tuple(bbox), year=year, node_id=cache_key
                    )
                else:
                    embedding = self.adapter.get_embedding(
                        lat, lon, year=year, node_id=cache_key
                    )

                # Attach to graph node
                G.nodes[node_id]["alphaearth_embedding"] = embedding.tolist()
                G.nodes[node_id]["embedding_year"] = year
                G.nodes[node_id]["embedding_source"] = "alphaearth_foundations_v1"
                G.nodes[node_id]["embedding_dim"] = EMBEDDING_DIM

                # Create and attach provenance
                prov = make_alphaearth_provenance(
                    lat=lat, lon=lon, year=year, node_id=node_id
                )
                G.nodes[node_id]["alphaearth_provenance"] = {
                    "source_id": prov.source_id,
                    "source_type": prov.source_type.value,
                    "evidence_quote": prov.evidence_quote,
                    "observed_at": prov.observed_at,
                    "snapshot_id": prov.snapshot_id,
                }

                stats["enriched"] += 1

            except FileNotFoundError:
                stats["failed"] += 1

        return stats

    @staticmethod
    def load_geometries(path: str) -> Dict[str, Dict[str, Any]]:
        """Load node geometries from a JSON file.

        Expected format:
        {
            "asset:PORT_HOUSTON": {"lat": 29.7355, "lon": -95.2690},
            "corridor:SHIP_CHANNEL": {"lat": 29.74, "lon": -95.08, "bbox": [...]}
        }
        """
        with open(path) as f:
            return json.load(f)
