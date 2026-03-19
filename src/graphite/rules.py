"""
graphite/rules.py — Abstract interface for domain-specific scoring rules.

Each domain defines rules that adjust edge scores based on evidence quality.
The rule engine is deterministic: same input always produces same output.
No LLM is involved at scoring time.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RuleResult:
    """Result from evaluating a single rule against an edge."""

    rule_id: str
    rule_name: str
    triggered: bool
    weight_delta: float
    explanation: str
    category: str = ""


@dataclass
class ScoreBreakdown:
    """Complete scoring output for an edge."""

    base_score: float
    rule_results: List[RuleResult] = field(default_factory=list)
    final_score: float = 0.0
    raw_delta: float = 0.0
    applied_delta: float = 0.0
    category_summaries: List[Dict] = field(default_factory=list)
    confidence: str = ""
    verdict: str = ""
    verdict_reason: str = ""
    policy_version: str = "v1"

    @property
    def triggered_rules(self) -> List[RuleResult]:
        return [r for r in self.rule_results if r.triggered]

    @property
    def total_delta(self) -> float:
        return self.applied_delta

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_score": round(self.base_score, 4),
            "final_score": round(self.final_score, 4),
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "confidence": self.confidence,
            "raw_delta": round(self.raw_delta, 4),
            "applied_delta": round(self.applied_delta, 4),
            "policy_version": self.policy_version,
            "triggered_rules": [
                {
                    "rule_id": r.rule_id,
                    "rule_name": r.rule_name,
                    "weight_delta": round(r.weight_delta, 4),
                    "explanation": r.explanation,
                    "category": r.category,
                }
                for r in self.triggered_rules
            ],
            "category_summaries": self.category_summaries,
        }


class BaseRuleEngine(ABC):
    """
    Override this to define domain-specific scoring rules.

    Example for SEC supply chain:
        - R01: Sole source dependency → +0.15
        - R02: Revenue concentration → +0.12
        - R07: Undisclosed entity → -0.10

    Example for minerals:
        - M01: Single country source → +0.15
        - M02: China refining concentration → +0.10
        - M03: No substitute available → +0.12
    """

    @abstractmethod
    def compute_score(
        self,
        edge_data: Dict[str, Any],
        counter_signals: Optional[List[Dict]] = None,
    ) -> ScoreBreakdown:
        """
        Compute deterministic, explainable score for an edge.

        Args:
            edge_data: Edge attributes from the graph
            counter_signals: Optional list of counter-evidence

        Returns:
            ScoreBreakdown with full rule trace
        """
