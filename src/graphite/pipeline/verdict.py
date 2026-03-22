"""
graphite/pipeline/verdict.py — LLM-native verification verdict types.

These types represent the output of the verification pipeline:
claim-level verdicts, argument-level verdicts, and the aggregated report.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VerdictEnum(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONFLICTED = "CONFLICTED"
    INSUFFICIENT = "INSUFFICIENT"


class ArgumentVerdictEnum(str, Enum):
    GROUNDED = "GROUNDED"
    CONCLUSION_JUMP = "CONCLUSION_JUMP"
    OVERSTATED = "OVERSTATED"


class VerdictRationale(BaseModel):
    """Structured reasoning slots for meta-evaluations and transparent auditing."""
    contradiction_type: Optional[str] = Field(
        default=None, description="e.g., numeric mismatch, entity mismatch"
    )
    missing_evidence_reason: Optional[str] = Field(
        default=None, description="Why the evidence was insufficient or lacking"
    )
    temporal_alignment: Optional[str] = Field(
        default=None, description="e.g., stale evidence vs current claim timeline"
    )
    text: str = Field(description="Free-form rationale from the LLM judge")


class Verdict(BaseModel):
    """A claim-level judgment returned by the verifier pipeline."""
    claim_id: str
    claim_text: str
    verdict: VerdictEnum
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    conflicting_evidence_ids: List[str] = Field(default_factory=list)
    rationale: VerdictRationale
    needs_human_review: bool = Field(
        default=False,
        description="Flag for high-risk or low-confidence verdicts to route to human queues",
    )
    cited_span: Optional[str] = Field(
        default=None,
        description="The exact span from the evidence corpus cited to make the verdict",
    )
    model_version: str = Field(default="")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ArgumentVerdict(BaseModel):
    """An argument-level verification representing semantic logic jumps."""
    text: str = Field(description="The argument chain or conclusion being evaluated")
    verdict: ArgumentVerdictEnum
    rationale: VerdictRationale
    needs_human_review: bool = Field(default=False)


class VerificationReport(BaseModel):
    """Top-level review object aggregating the entire verification workflow."""
    document_id: str
    total_claims: int = 0
    supported_count: int = 0
    conflicted_count: int = 0
    insufficient_count: int = 0
    grounded_argument_count: int = 0
    conclusion_jump_count: int = 0
    risky_claim_ids: List[str] = Field(
        default_factory=list, description="Claim IDs flagged for human review"
    )
    evidence_coverage_score: float = 0.0
    verdicts: List[Verdict] = Field(default_factory=list)
    argument_verdicts: List[ArgumentVerdict] = Field(default_factory=list)
    model_metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_verdict(self, claim_id: str) -> Optional[Verdict]:
        for v in self.verdicts:
            if v.claim_id == claim_id:
                return v
        return None
