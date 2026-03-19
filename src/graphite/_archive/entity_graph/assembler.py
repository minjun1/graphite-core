"""
graphite/assembler.py — Graph assembly from extracted edges.

The assembler sits between extractors and writers:
  Extractor → ExtractedEdge[] → **Assembler** → nx.DiGraph → Writer

Responsibilities:
  - Deduplicate edges (merge provenance when same relationship from different sources)
  - Resolve conflicting attributes (merge policy with priority)
  - Normalize nodes
  - Collect extraction errors
  - Stamp graph metadata
"""

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import networkx as nx

from .enums import AssertionMode, ConfidenceLevel, SourceType, EvidenceType
from .schemas import ExtractedEdge, ExtractionError, Provenance
from .domain import DomainSpec


# ═══════════════════════════════════════
# Merge Policy
# ═══════════════════════════════════════

_EVIDENCE_PRIORITY = {
    EvidenceType.TABLE_CELL: 4,
    EvidenceType.TEXT_QUOTE: 3,
    EvidenceType.DERIVED: 2,
    EvidenceType.MANUAL: 1,
}

_SOURCE_PRIORITY = {
    SourceType.USGS_MCS: 4,
    SourceType.SEC_10K: 3,
    SourceType.SEC_20F: 3,
    SourceType.PDF: 2,
    SourceType.WEB: 1,
    SourceType.MANUAL: 1,
}

_ASSERTION_PRIORITY = {
    AssertionMode.EXTRACTED: 3,
    AssertionMode.INFERRED: 2,
    AssertionMode.SEEDED: 1,
}

_CONFIDENCE_PRIORITY = {
    ConfidenceLevel.HIGH: 3,
    ConfidenceLevel.MEDIUM: 2,
    ConfidenceLevel.LOW: 1,
}


def _provenance_score(p: Provenance) -> int:
    """Score a provenance for merge priority. Higher = prefer."""
    return (
        _EVIDENCE_PRIORITY.get(p.evidence_type, 0) * 100
        + _SOURCE_PRIORITY.get(p.source_type, 0) * 10
        + _CONFIDENCE_PRIORITY.get(p.confidence, 0)
    )


class GraphAssembler:
    """Assemble a NetworkX graph from normalized ExtractedEdge objects.

    Usage:
        assembler = GraphAssembler()
        G = assembler.assemble(edges)
    """

    def __init__(
        self,
        pipeline_version: str = "1.0",
        domain_spec: Optional[DomainSpec] = None,
        drop_zero_provenance: bool = True,
        drop_low_inferred: bool = False,
    ):
        self.pipeline_version = pipeline_version
        self.domain_spec = domain_spec
        self.drop_zero_provenance = drop_zero_provenance
        self.drop_low_inferred = drop_low_inferred
        self.errors: List[ExtractionError] = []

    def assemble(
        self,
        edges: List[ExtractedEdge],
        node_labels: Optional[Dict[str, str]] = None,
    ) -> nx.DiGraph:
        """Assemble edges into a NetworkX DiGraph.

        Args:
            edges: Normalized extracted edges
            node_labels: Optional node_id → display label mapping

        Returns:
            Assembled and stamped nx.DiGraph
        """
        # 1. Validate edge types against domain registry
        if self.domain_spec:
            edges = self._validate_edge_types(edges)

        # 2. Quality filters
        edges = self._quality_filter(edges)

        # 3. Deduplicate
        deduped = self.dedupe_edges(edges)

        # 4. Build graph
        G = nx.DiGraph()
        labels = node_labels or {}

        # Collect all nodes
        all_nodes: Dict[str, dict] = {}
        for edge in deduped:
            for nref in (edge.from_node, edge.to_node):
                if nref.node_id not in all_nodes:
                    all_nodes[nref.node_id] = {
                        "node_type": nref.node_type.value,
                        "name": nref.label or labels.get(nref.node_id, nref.node_id),
                    }

        for nid, attrs in all_nodes.items():
            G.add_node(nid, **attrs)

        # Add edges
        for edge in deduped:
            weight = edge.attributes.get("bucket_weight", 0.5)
            cost = -math.log(max(weight, 0.01))

            edge_attrs = {
                "edge_type": edge.edge_type,
                "assertion_mode": edge.assertion_mode.value,
                "bucket_weight": weight,
                "cost": round(cost, 6),
                "confidence": edge.best_confidence.value,
            }

            # Flatten domain attributes (but not nested structures)
            for k, v in edge.attributes.items():
                if k not in edge_attrs and not isinstance(v, (dict, list)):
                    edge_attrs[k] = v

            # Serialize provenance as JSON string (GraphML compatible)
            edge_attrs["provenance_json"] = json.dumps(
                [p.model_dump() for p in edge.provenance], default=str
            )
            edge_attrs["provenance_count"] = len(edge.provenance)
            edge_attrs["evidence"] = (
                edge.provenance[0].evidence_quote if edge.provenance else ""
            )
            edge_attrs["data_source"] = (
                edge.provenance[0].source_type.value if edge.provenance else ""
            )

            # Inference basis
            if edge.inference_basis:
                edge_attrs["inference_method"] = edge.inference_basis.method
                edge_attrs["inference_reason"] = edge.inference_basis.reason

            # Claim linkage (trust engine v1)
            if edge.claim_ids:
                edge_attrs["claim_ids"] = json.dumps(edge.claim_ids)

            G.add_edge(edge.from_node.node_id, edge.to_node.node_id, **edge_attrs)

        # 5. Stamp
        return self._stamp_graph(G)

    def dedupe_edges(self, edges: List[ExtractedEdge]) -> List[ExtractedEdge]:
        """Merge edges with same (from, to, type) — combine provenances."""
        by_key: Dict[str, ExtractedEdge] = {}

        for edge in edges:
            key = edge.edge_key
            if key in by_key:
                existing = by_key[key]
                merged = self._merge_edge_pair(existing, edge)
                by_key[key] = merged
            else:
                by_key[key] = edge

        return list(by_key.values())

    def _merge_edge_pair(self, a: ExtractedEdge, b: ExtractedEdge) -> ExtractedEdge:
        """Merge two edges with the same key."""
        # Merge provenance (dedupe by source_id)
        seen_sources = {p.source_id for p in a.provenance}
        merged_prov = list(a.provenance)
        for p in b.provenance:
            if p.source_id not in seen_sources:
                merged_prov.append(p)
                seen_sources.add(p.source_id)

        # Stronger assertion mode (EXTRACTED > INFERRED > SEEDED)
        mode = (
            a.assertion_mode
            if _ASSERTION_PRIORITY.get(a.assertion_mode, 0)
            >= _ASSERTION_PRIORITY.get(b.assertion_mode, 0)
            else b.assertion_mode
        )

        # Merge attributes with conflict tracking
        merged_attrs = dict(a.attributes)
        for k, v in b.attributes.items():
            if k in merged_attrs and merged_attrs[k] != v:
                rv_key = f"{k}_reported_values"
                existing_rv = merged_attrs.get(rv_key, [])
                if not existing_rv:
                    a_prov = a.provenance[0] if a.provenance else None
                    existing_rv.append(
                        {
                            "value": merged_attrs[k],
                            "source": a_prov.source_type.value if a_prov else "unknown",
                            "confidence": a_prov.confidence.value if a_prov else "LOW",
                        }
                    )
                b_prov = b.provenance[0] if b.provenance else None
                existing_rv.append(
                    {
                        "value": v,
                        "source": b_prov.source_type.value if b_prov else "unknown",
                        "confidence": b_prov.confidence.value if b_prov else "LOW",
                    }
                )
                merged_attrs[rv_key] = existing_rv

                a_score = max((_provenance_score(p) for p in a.provenance), default=0)
                b_score = max((_provenance_score(p) for p in b.provenance), default=0)
                if b_score > a_score:
                    merged_attrs[k] = v
            else:
                merged_attrs[k] = v

        basis = a.inference_basis or b.inference_basis

        return ExtractedEdge(
            from_node=a.from_node,
            to_node=a.to_node,
            edge_type=a.edge_type,
            assertion_mode=mode,
            attributes=merged_attrs,
            provenance=merged_prov,
            inference_basis=basis,
        )

    def _validate_edge_types(self, edges: List[ExtractedEdge]) -> List[ExtractedEdge]:
        """Check edge types against domain registry."""
        allowed = (
            set(self.domain_spec.allowed_edge_types) if self.domain_spec else set()
        )
        if not allowed:
            return edges

        valid = []
        for edge in edges:
            if edge.edge_type in allowed:
                valid.append(edge)
            else:
                self.errors.append(
                    ExtractionError(
                        entity_id=edge.from_node.node_id,
                        source_type=edge.provenance[0].source_type
                        if edge.provenance
                        else SourceType.MANUAL,
                        error_type="validation_failed",
                        message=f"Edge type '{edge.edge_type}' not in domain allowed types: {allowed}",
                    )
                )
        return valid

    def _quality_filter(self, edges: List[ExtractedEdge]) -> List[ExtractedEdge]:
        """Apply quality filters."""
        filtered = []
        for edge in edges:
            if self.drop_zero_provenance and not edge.provenance:
                self.errors.append(
                    ExtractionError(
                        entity_id=edge.from_node.node_id,
                        source_type=SourceType.MANUAL,
                        error_type="no_edges",
                        message=f"Edge {edge.edge_key} dropped: zero provenance",
                    )
                )
                continue
            if (
                self.drop_low_inferred
                and edge.assertion_mode == AssertionMode.INFERRED
                and edge.best_confidence == ConfidenceLevel.LOW
            ):
                continue
            filtered.append(edge)
        return filtered

    def _stamp_graph(self, G: nx.DiGraph) -> nx.DiGraph:
        """Add metadata to graph."""
        edge_types = defaultdict(int)
        assertion_modes = defaultdict(int)
        for _, _, d in G.edges(data=True):
            edge_types[d.get("edge_type", "?")] += 1
            assertion_modes[d.get("assertion_mode", "?")] += 1

        G.graph["built_at"] = datetime.now(timezone.utc).isoformat()
        G.graph["pipeline_version"] = self.pipeline_version
        G.graph["node_count"] = G.number_of_nodes()
        G.graph["edge_count"] = G.number_of_edges()
        G.graph["edge_types"] = json.dumps(dict(edge_types))
        G.graph["assertion_modes"] = json.dumps(dict(assertion_modes))
        if self.domain_spec:
            G.graph["domain"] = self.domain_spec.name

        return G
