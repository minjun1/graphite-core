"""
graphite/schemas.py — Core data models for the Graphite engine.

These are the canonical types that cross module boundaries.
Domain-specific schemas live in their own packages and get
converted to these types before reaching the assembler.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .enums import (
    AssertionMode, ConfidenceLevel, EdgeType, EvidenceType,
    NodeType, SourceType,
)


# ═══════════════════════════════════════
# Node Identity
# ═══════════════════════════════════════

class NodeRef(BaseModel):
    """Canonical node identity.

    Prevents 'Apple' vs 'AAPL' vs 'Apple Inc.' chaos by enforcing
    typed, prefixed IDs like "company:AAPL", "country:CD", "mineral:COBALT".
    """
    node_id: str = Field(description="Canonical ID: 'company:AAPL', 'country:CD', 'mineral:COBALT'")
    node_type: NodeType
    label: str = Field(default="", description="Display name, filled at assembly if blank")

    @classmethod
    def company(cls, ticker: str, label: str = "") -> "NodeRef":
        return cls(node_id=f"company:{ticker.upper()}", node_type=NodeType.COMPANY, label=label)

    @classmethod
    def country(cls, iso: str, label: str = "") -> "NodeRef":
        return cls(node_id=f"country:{iso.upper()}", node_type=NodeType.COUNTRY, label=label)

    @classmethod
    def mineral(cls, name: str, label: str = "") -> "NodeRef":
        return cls(node_id=f"mineral:{name.upper()}", node_type=NodeType.MINERAL, label=label)

    @classmethod
    def region(cls, code: str, label: str = "") -> "NodeRef":
        return cls(node_id=f"region:{code.upper()}", node_type=NodeType.REGION, label=label)

    @classmethod
    def asset(cls, asset_id: str, label: str = "") -> "NodeRef":
        return cls(node_id=f"asset:{asset_id.upper()}", node_type=NodeType.ASSET, label=label)

    @classmethod
    def facility(cls, fac_id: str, label: str = "") -> "NodeRef":
        return cls(node_id=f"facility:{fac_id.upper()}", node_type=NodeType.FACILITY, label=label)

    @classmethod
    def corridor(cls, corridor_id: str, label: str = "") -> "NodeRef":
        return cls(node_id=f"corridor:{corridor_id.upper()}", node_type=NodeType.CORRIDOR, label=label)


# ═══════════════════════════════════════
# Provenance (Evidence Source)
# ═══════════════════════════════════════

class Provenance(BaseModel):
    """First-class evidence source for an edge.

    A single edge may have MULTIPLE provenances when the same relationship
    is confirmed by different sources (e.g., USGS + SEC filing).
    """
    source_id: str = Field(description="Unique ID: accession_no, USGS URL, etc.")
    source_type: SourceType
    source_url: str = Field(default="")
    evidence_quote: str = Field(description="Retrieved candidate text span")
    cited_span: str = Field(default="", description="The specific exact quote used in the final verdict")
    evidence_type: EvidenceType = Field(default=EvidenceType.TEXT_QUOTE)
    paragraph_index: int = Field(default=-1)
    paragraph_hash: str = Field(default="")
    extracted_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    extractor_version: str = Field(default="1.0")
    pipeline_version: str = Field(default="1.0")
    prompt_version: str = Field(default="1.0")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    # Minimal temporal tagging (no full temporal engine — just snapshot markers)
    observed_at: str = Field(default="", description="When the evidence was observed")
    valid_from: str = Field(default="", description="Start of validity window")
    valid_to: str = Field(default="", description="End of validity window")
    snapshot_id: str = Field(default="", description="Links to a specific data snapshot")


# ═══════════════════════════════════════
# Inference Basis
# ═══════════════════════════════════════

class InferenceBasis(BaseModel):
    """Structured explanation for INFERRED edges.

    NOT a free dict — ensures UI/MCP can render inference explanations.
    """
    method: str = Field(description="e.g. 'customer_filing_reverse', 'industry_knowledge'")
    based_on_edges: List[str] = Field(
        default_factory=list,
        description="Edge keys that support this inference",
    )
    reason: str = Field(description="Human-readable explanation")
    source_nodes: List[str] = Field(
        default_factory=list,
        description="Node IDs that informed the inference",
    )


# ═══════════════════════════════════════
# Extracted Edge (Core Boundary Type)
# ═══════════════════════════════════════

class ExtractedEdge(BaseModel):
    """Normalized edge — the ONLY type that leaves any extractor.

    Domain-specific schemas (ProducesEdge, SupplyEdge, etc.) are internal
    to each domain's extractor. They get converted to ExtractedEdge before
    reaching the assembler or writer.
    """
    from_node: NodeRef
    to_node: NodeRef
    edge_type: str = Field(
        description="EdgeType enum value or domain-registered string"
    )
    assertion_mode: AssertionMode
    attributes: Dict[str, Any] = Field(default_factory=dict)
    provenance: List[Provenance] = Field(default_factory=list)
    inference_basis: Optional[InferenceBasis] = Field(default=None)
    claim_ids: List[str] = Field(
        default_factory=list,
        description="Claim IDs that this edge materializes. "
        "Empty for legacy edges; populated when extracted via claim-first pipeline.",
    )

    @property
    def edge_key(self) -> str:
        """Unique key for deduplication: (from, to, type)."""
        return f"{self.from_node.node_id}|{self.to_node.node_id}|{self.edge_type}"

    @property
    def best_confidence(self) -> ConfidenceLevel:
        """Best confidence across all provenances."""
        rank = {ConfidenceLevel.HIGH: 3, ConfidenceLevel.MEDIUM: 2, ConfidenceLevel.LOW: 1}
        best = max(
            self.provenance,
            key=lambda p: rank.get(p.confidence, 0),
            default=None,
        )
        return best.confidence if best else ConfidenceLevel.LOW


# ═══════════════════════════════════════
# Extraction Errors
# ═══════════════════════════════════════

class ExtractionError(BaseModel):
    """Tracks pipeline failures for quality monitoring."""
    entity_id: str
    source_type: SourceType
    error_type: str = Field(
        description="parse_failed | no_edges | validation_failed | low_confidence_only"
    )
    message: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
