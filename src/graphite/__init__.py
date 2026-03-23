"""
Graphite — Open-source claim verification engine for agent-generated
assertions in high-stakes domains.

Core primitives:
  - Claim: the atomic unit of trust — structured assertion with provenance
  - ClaimStore: evidence-accumulating registry (claims dedupe, evidence merges)
  - Provenance: first-class evidence source (document, quote, confidence)
  - ConfidenceScorer: explainable confidence scoring with named factors

Pipeline:
  Agent/Extractor → Claim[] → ClaimStore (accumulate) → Verify
"""

# ── Core schemas ──
from .schemas import ExtractedEdge, NodeRef, Provenance, InferenceBasis, ExtractionError
from .enums import (
    EdgeType, NodeType, SourceType, ConfidenceLevel, AssertionMode,
    EvidenceType, ClaimType, ClaimStatus, ReviewState, ClaimOrigin,
    ClaimGranularity,
)

# ── Trust engine primitives ──
from .claim import Claim
from .claim import ConfidenceFactor, ConfidenceResult
from .claim_store import ClaimStore
from .confidence import ConfidenceScorer

# ── Evidence (must come after Claim and ConfidenceResult for forward references) ──
from .evidence import EvidencePacket, EvidenceData

# ── Pipeline verdict types (re-exported for convenience) ──
from .pipeline.verdict import (
    VerdictEnum, ArgumentVerdictEnum, VerdictRationale,
    Verdict, ArgumentVerdict, VerificationReport,
)

# ── Domain plugin contracts ──
from .domain import (
    BaseFetcher,
    BaseExtractor,
    BasePipeline,
    DocumentContext,
    DomainSpec,
)
from .domain import register_domain, get_domain, list_domains

# ── Rules ──
from .rules import BaseRuleEngine, RuleResult, ScoreBreakdown

# ── Evaluation framework ──
from .eval import EvalCase, EvalResult, EvalRun, EvalRunner

__all__ = [
    # Schemas
    "ExtractedEdge", "NodeRef", "Provenance", "InferenceBasis", "ExtractionError",
    # Enums
    "EdgeType", "NodeType", "SourceType", "ConfidenceLevel", "AssertionMode",
    "EvidenceType", "ClaimType", "ClaimStatus", "ReviewState", "ClaimOrigin",
    "ClaimGranularity",
    # Trust engine
    "Claim", "ConfidenceFactor", "ConfidenceResult",
    "ClaimStore", "ConfidenceScorer",
    # Evidence
    "EvidencePacket", "EvidenceData",
    # Pipeline verdicts
    "VerdictEnum", "ArgumentVerdictEnum", "VerdictRationale",
    "Verdict", "ArgumentVerdict", "VerificationReport",
    # Domain plugins
    "BaseFetcher", "BaseExtractor", "BasePipeline",
    "DocumentContext", "DomainSpec",
    "register_domain", "get_domain", "list_domains",
    # Rules
    "BaseRuleEngine", "RuleResult", "ScoreBreakdown",
    # Eval
    "EvalCase", "EvalResult", "EvalRun", "EvalRunner",
]

__version__ = "0.3.2"

# Resolve forward references in EvidencePacket
# (must be after Claim and ConfidenceResult are imported)
EvidencePacket.model_rebuild(_types_namespace={"Claim": Claim, "ConfidenceResult": ConfidenceResult})

# ── Logging ──
import logging
logging.getLogger("graphite").addHandler(logging.NullHandler())
