"""Eval data types for structured verification benchmarking."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """A single evaluation test case."""

    id: str
    memo: str = Field(description="Agent-generated text to verify")
    corpus: List[Dict[str, str]] = Field(
        description='Evidence documents: [{"document_id": "...", "text": "..."}]'
    )
    expected_claim_verdicts: List[str] = Field(
        description="Expected claim-level verdicts (SUPPORTED/CONFLICTED/INSUFFICIENT)"
    )
    expected_argument_verdicts: List[str] = Field(
        default_factory=list,
        description="Expected argument-level verdicts (GROUNDED/CONCLUSION_JUMP/OVERSTATED)",
    )
    domain: str = Field(default="general", description="Domain tag for filtering")
    tags: List[str] = Field(default_factory=list, description="Arbitrary tags for filtering")


class EvalResult(BaseModel):
    """Result of evaluating a single case."""

    case_id: str
    expected_claim_verdicts: List[str]
    actual_claim_verdicts: List[str] = Field(default_factory=list)
    claim_verdict_pass: bool = False
    expected_argument_verdicts: List[str] = Field(default_factory=list)
    actual_argument_verdicts: List[str] = Field(default_factory=list)
    argument_verdict_pass: bool = True
    latency_ms: Optional[float] = None
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.claim_verdict_pass and self.argument_verdict_pass


class EvalRun(BaseModel):
    """Aggregated results of running an eval dataset."""

    model: str
    dataset: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    results: List[EvalResult] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def metrics(self) -> Dict[str, Any]:
        total = len(self.results)
        if total == 0:
            return {"total": 0}
        claim_pass = sum(1 for r in self.results if r.claim_verdict_pass)
        arg_pass = sum(1 for r in self.results if r.argument_verdict_pass)
        overall_pass = sum(1 for r in self.results if r.passed)
        latencies = [r.latency_ms for r in self.results if r.latency_ms is not None]
        return {
            "total": total,
            "claim_verdict_accuracy": claim_pass / total,
            "argument_verdict_accuracy": arg_pass / total,
            "overall_pass_rate": overall_pass / total,
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str) -> "EvalRun":
        with open(path) as f:
            return cls.model_validate_json(f.read())
