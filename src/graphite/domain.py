"""
graphite/domain.py — Plugin contracts and domain registration.

Defines the interfaces that domain plugins implement (fetchers, extractors,
pipelines) and the DomainSpec registry for plugin discovery.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import networkx as nx
from pydantic import BaseModel, Field

from .enums import NodeType, SourceType
from .schemas import ExtractedEdge
from .claim import Claim


class ExtractionResult(BaseModel):
    """The complete result of a domain extraction pass."""
    claims: List[Claim] = Field(default_factory=list, description="Extracted claims")
    edges: List[ExtractedEdge] = Field(default_factory=list, description="Graph edges (materialized from claims)")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="Telemetry and metrics")
    unresolvable_count: int = Field(default=0, description="Claims discarded due to vagueness")



# ═══════════════════════════════════════
# Document Context
# ═══════════════════════════════════════

@dataclass
class DocumentContext:
    """A fetched document ready for extraction.

    This is the universal container passed from fetcher → extractor.
    The extractor chooses its own context strategy.
    """
    source_id: str              # accession_no, USGS URL, etc.
    source_type: SourceType     # enum, not string
    entity_id: str              # ticker, mineral name, etc.
    text_content: str           # full extracted text
    paragraphs: List[str]       # split paragraphs
    doc_url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════
# Base Interfaces
# ═══════════════════════════════════════

class BaseFetcher(ABC):
    """Fetch and parse documents into DocumentContext objects.

    Implementations:
      - UsgsFetcher: USGS PDF → text + tables
      - SecFetcher: SEC EDGAR API → 10-K HTML → text
      - CustomFetcher: Any new data source
    """

    @abstractmethod
    def fetch(self, entity_id: str, **kwargs) -> List[DocumentContext]:
        """Fetch document(s) for a given entity.

        Args:
            entity_id: Ticker, mineral name, country ISO, etc.

        Returns:
            List of DocumentContext objects ready for extraction.
        """

    def fetch_batch(self, entity_ids: List[str], **kwargs) -> List[DocumentContext]:
        """Fetch documents for multiple entities. Override for parallelism."""
        docs = []
        for eid in entity_ids:
            docs.extend(self.fetch(eid, **kwargs))
        return docs


class BaseExtractor(ABC):
    """Extract structured edges from documents.

    Implementations call build_context(doc, strategy="...") internally,
    then use an LLM (or deterministic parser) for extraction, then convert
    domain-specific schemas to normalized ExtractedEdge objects.
    """

    @abstractmethod
    def extract(self, doc: DocumentContext) -> ExtractionResult:
        """Extract graph edges from a document.

        Args:
            doc: DocumentContext from a fetcher

        Returns:
            ExtractionResult containing edges, claims, and diagnostics
        """

    def extract_batch(self, docs: List[DocumentContext]) -> ExtractionResult:
        """Extract from multiple documents. Override for parallelism."""
        all_claims = []
        all_edges = []
        unresolvable = 0
        diagnostics = {}
        for doc in docs:
            res = self.extract(doc)
            all_claims.extend(res.claims)
            all_edges.extend(res.edges)
            unresolvable += res.unresolvable_count
        return ExtractionResult(
            claims=all_claims,
            edges=all_edges,
            diagnostics=diagnostics,
            unresolvable_count=unresolvable
        )


class BasePipeline(ABC):
    """Orchestrates the full pipeline: fetch → extract → assemble → write.

    Subclass per domain. The build script calls `run()`.
    """

    @abstractmethod
    def run(self, entity_ids: List[str], output_path: str, **kwargs) -> nx.DiGraph:
        """Run the full pipeline.

        Args:
            entity_ids: List of entities to process
            output_path: Path to save the resulting graph

        Returns:
            The assembled NetworkX DiGraph
        """


# ═══════════════════════════════════════
# Domain Specification
# ═══════════════════════════════════════

class DomainSpec(BaseModel):
    """Everything needed to register a new supply chain domain.

    To add a new domain to Graphite:
    1. Create a package with fetcher + extractor
    2. Register a DomainSpec with allowed types and strategies
    3. Core handles assembly, propagation, evidence
    """
    name: str = Field(description="Domain name: 'minerals', 'sec', 'pharma'")
    allowed_edge_types: List[str] = Field(
        description="Validated at extraction time"
    )
    allowed_node_types: List[NodeType] = Field(
        description="NodeType enum values this domain uses"
    )
    context_strategies: List[str] = Field(
        default_factory=list,
        description="Registered strategy names for text processing",
    )
    propagation_alphas: Dict[str, float] = Field(
        default_factory=dict,
        description="edge_type → alpha for shock propagation (optional, future-proof)",
    )
    fetcher_class: Optional[Any] = Field(
        default=None,
        description="Type[BaseFetcher] — set at registration time",
    )
    extractor_class: Optional[Any] = Field(
        default=None,
        description="Type[BaseExtractor] — set at registration time",
    )

    model_config = {"arbitrary_types_allowed": True}


# ═══════════════════════════════════════
# Domain Registry
# ═══════════════════════════════════════

_domain_registry: Dict[str, DomainSpec] = {}


def register_domain(spec: DomainSpec) -> None:
    """Register a domain spec. Called at startup."""
    _domain_registry[spec.name] = spec


def get_domain(name: str) -> Optional[DomainSpec]:
    """Get a registered domain by name."""
    return _domain_registry.get(name)


def list_domains() -> List[str]:
    """List all registered domain names."""
    return list(_domain_registry.keys())
