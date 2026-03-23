"""
graphite/evidence.py — Evidence Packet specification.

Defines the standard evidence payload that Graphite returns for every
verified claim. This is the core "anti-hallucination" data structure:
every relationship in the graph is backed by a traceable evidence chain.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import TYPE_CHECKING, List, Optional, Literal

from .enums import ClaimStatus
from .rules import RuleResult

if TYPE_CHECKING:
    from .claim import Claim, ConfidenceResult


class EvidenceData(BaseModel):
    """Primary evidence supporting a graph edge."""

    source_entity: str
    target_entity: str
    edge_type: str
    quote: str  # Human-readable summary or exact quote
    exact_quote: str = ""  # Verbatim text from source document
    quote_type: Literal["exact_quote", "model_normalized", "extracted_claim"] = (
        "exact_quote"
    )
    doc_url: str  # Source document URL
    quote_hash: str  # Cryptographic hash for tamper detection
    # Domain-specific metadata (optional)
    source_id: str = ""  # e.g., CIK, DOI, package name
    filing_type: str = ""  # e.g., "10-K", "USGS Report"
    section: str = ""  # e.g., "Item 1A. Risk Factors"
    temporal_status: str = "ACTIVE"
    filing_date: str = ""
    fiscal_period: str = ""


class ScoringData(BaseModel):
    """Deterministic scoring breakdown — no LLM involved."""

    policy_id: str
    applied_rules: List[str]
    rule_details: List[RuleResult] = []
    calculated_weight: float
    base_score: float = 0.0
    final_score: float = 0.0
    total_delta: float = 0.0
    confidence: str = ""


class CounterEvidence(BaseModel):
    """Evidence that weakens or contradicts the primary claim."""

    quote: str
    doc_url: str
    impact: str  # e.g., "WEAKENS_MONOPOLY", "DIVERSIFIED_SUPPLY"
    source_filing: str = ""


class EvidencePacket(BaseModel):
    """
    The Graphite Evidence Packet — the core output format.

    Every verified claim returns this structure, providing:
    1. A deterministic verdict (SUPPORTED / WEAK / UNSUPPORTED / MIXED)
    2. The source evidence with document link and quote hash
    3. Rule engine scoring breakdown
    4. Counter-evidence (anti-cherry-picking safeguard)
    5. (v1+) Structured claim and explainable confidence
    """

    graphite_version: str = "1.0"
    claim_hash: str
    status: ClaimStatus  # SUPPORTED / WEAK / UNSUPPORTED / MIXED / PENDING_REVIEW
    verdict_reason: str = ""
    evidence: Optional[EvidenceData] = None
    scoring: Optional[ScoringData] = None
    score_breakdown: Optional[dict] = None
    counter_evidence: List[CounterEvidence] = []
    # Structured claim (v1 trust engine)
    claim: Optional[Claim] = Field(
        default=None,
        description="The structured Claim object, if available",
    )
    confidence_result: Optional[ConfidenceResult] = Field(
        default=None,
        description="Explainable confidence breakdown",
    )
    # Audit metadata
    graph_built_at: str = ""
    data_as_of: str = ""
    domain: str = ""  # Which domain plugin generated this
