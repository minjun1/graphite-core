"""
graphite/enums.py — Canonical enums for the Graphite engine.

These prevent string drift across domains. Domains can extend EdgeType
via string values, but must register them in DomainSpec.allowed_edge_types.
"""

from enum import Enum


class SourceType(str, Enum):
    """How a document was obtained."""

    SEC_10K = "SEC_10K"
    SEC_20F = "SEC_20F"
    USGS_MCS = "USGS_MCS"
    PDF = "PDF"
    WEB = "WEB"
    MANUAL = "MANUAL"
    # Geo-climate sources
    WEATHER_FORECAST = "WEATHER_FORECAST"
    EARTH_OBSERVATION = "EARTH_OBSERVATION"
    PUBLIC_REPORT = "PUBLIC_REPORT"
    GEOSPATIAL_DATA = "GEOSPATIAL_DATA"
    DOCUMENT = "DOCUMENT"


class EdgeType(str, Enum):
    """Core relationship types. Domains register additional types in DomainSpec."""

    PRODUCES = "PRODUCES"
    REFINED_BY = "REFINED_BY"
    SUPPLIES_TO = "SUPPLIES_TO"
    USED_BY = "USED_BY"
    # Geo-climate edge types (semantic direction — see docs/edge_direction.md)
    LOCATED_IN = "LOCATED_IN"
    DEPENDS_ON = "DEPENDS_ON"
    ADJACENT_TO = "ADJACENT_TO"
    EXPOSED_TO = "EXPOSED_TO"
    RISK_FLOWS_TO = "RISK_FLOWS_TO"


class NodeType(str, Enum):
    """Node classification. Used in NodeRef for type safety."""

    COMPANY = "COMPANY"
    COUNTRY = "COUNTRY"
    MINERAL = "MINERAL"
    # Geo-climate node types (see docs/node_taxonomy.md)
    REGION = "REGION"
    ASSET = "ASSET"
    FACILITY = "FACILITY"
    CORRIDOR = "CORRIDOR"


class AssertionMode(str, Enum):
    """How an edge was determined — critical for trust."""

    EXTRACTED = "EXTRACTED"  # Directly from document text
    INFERRED = "INFERRED"  # Derived from another entity's filing
    SEEDED = "SEEDED"  # Hardcoded baseline data


class ConfidenceLevel(str, Enum):
    """Edge confidence."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class EvidenceType(str, Enum):
    """What kind of evidence backs this edge."""

    TEXT_QUOTE = "TEXT_QUOTE"  # Verbatim prose from document
    TABLE_CELL = "TABLE_CELL"  # Value from a table/chart
    DERIVED = "DERIVED"  # Computed from multiple sources
    MANUAL = "MANUAL"  # Hand-entered by researcher


class ClaimType(str, Enum):
    """What kind of assertion this claim makes."""

    RELATIONSHIP = "RELATIONSHIP"
    ATTRIBUTE = "ATTRIBUTE"
    RISK_ASSERTION = "RISK_ASSERTION"
    DEPENDENCY = "DEPENDENCY"


class ClaimStatus(str, Enum):
    """Trust verdict for a claim — typically computed, not manually set."""

    SUPPORTED = "SUPPORTED"
    WEAK = "WEAK"
    MIXED = "MIXED"
    UNSUPPORTED = "UNSUPPORTED"
    PENDING_REVIEW = "PENDING_REVIEW"


class ReviewState(str, Enum):
    """Analyst review workflow state."""

    UNREVIEWED = "UNREVIEWED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NEEDS_FOLLOWUP = "NEEDS_FOLLOWUP"


class ClaimOrigin(str, Enum):
    """How this claim was created."""

    EXTRACTOR = "EXTRACTOR"
    AGENT = "AGENT"
    RULE_ENGINE = "RULE_ENGINE"
    ANALYST = "ANALYST"
    IMPORTED = "IMPORTED"


class ClaimGranularity(str, Enum):
    """Abstraction level of a claim."""

    ATOMIC = "ATOMIC"
    SYNTHESIZED = "SYNTHESIZED"
    THESIS = "THESIS"
