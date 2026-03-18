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

Also included:
  Graph assembly (GraphAssembler) and shock propagation (simulate, scenario)
"""

# ── Core schemas ──
from .schemas import ExtractedEdge, NodeRef, Provenance, InferenceBasis, ExtractionError
from .enums import EdgeType, NodeType, SourceType, ConfidenceLevel, AssertionMode, EvidenceType
from .evidence import EvidencePacket, EvidenceData

# ── Trust engine primitives ──
from .claim import Claim, ClaimType, ClaimStatus, ClaimGranularity, ReviewState, ClaimOrigin
from .claim import ConfidenceFactor, ConfidenceResult
from .claim_store import ClaimStore
from .confidence import ConfidenceScorer

# ── Assembly ──
from .assembler import GraphAssembler

# ── Domain plugin contracts ──
from .domain import BaseFetcher, BaseExtractor, BasePipeline, DocumentContext, DomainSpec
from .domain import register_domain, get_domain, list_domains

# ── Rules ──
from .rules import BaseRuleEngine, RuleResult, ScoreBreakdown

# ── Simulation ──
from .simulate import top_k_paths_from_source, build_blast_radius, map_to_tier

# ── Scenario ──
from .scenario import ScenarioShock, ScenarioRunner

# ── I/O ──
from .io import save_graph, load_graph

__version__ = "0.3.0"
