"""
graphite/confidence.py — Explainable confidence scoring for claims.

Computes a 0.0-1.0 confidence score from evidence metadata.
Every score comes with a list of ConfidenceFactors explaining
why this particular score was assigned.

The scorer is deterministic: same input always produces same output.
No LLM is involved at scoring time.

Usage:
    scorer = ConfidenceScorer()
    result = scorer.score(claim)
    print(result.score)          # 0.73
    print(result.level)          # HIGH
    for f in result.factors:
        print(f.name, f.contribution, f.explanation)
"""

from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from .claim import ConfidenceFactor, ConfidenceResult

if TYPE_CHECKING:
    from .claim import Claim

from .enums import AssertionMode, ConfidenceLevel, SourceType


# ═══════════════════════════════════════
# Configuration
# ═══════════════════════════════════════

# Weights for each factor (sum roughly = 1.0)
_FACTOR_WEIGHTS = {
    "source_count": 0.20,
    "source_diversity": 0.15,
    "directness": 0.20,
    "recency": 0.15,
    "doc_quality": 0.15,
    "counter_penalty": 0.15,
}

# Source quality tiers
_SOURCE_QUALITY = {
    SourceType.SEC_10K: 1.0,
    SourceType.SEC_20F: 1.0,
    SourceType.USGS_MCS: 0.9,
    SourceType.PUBLIC_REPORT: 0.7,
    SourceType.PDF: 0.6,
    SourceType.EARTH_OBSERVATION: 0.8,
    SourceType.GEOSPATIAL_DATA: 0.8,
    SourceType.WEATHER_FORECAST: 0.7,
    SourceType.DOCUMENT: 0.6,
    SourceType.WEB: 0.4,
    SourceType.MANUAL: 0.3,
}

# Directness scoring
_ASSERTION_DIRECTNESS = {
    AssertionMode.EXTRACTED: 1.0,
    AssertionMode.INFERRED: 0.5,
    AssertionMode.SEEDED: 0.3,
}

# Recency: how many years before score starts decaying
_RECENCY_FULL_CREDIT_YEARS = 2
_RECENCY_HALF_LIFE_YEARS = 5


# ═══════════════════════════════════════
# Scorer
# ═══════════════════════════════════════


class ConfidenceScorer:
    """Compute explainable confidence from a Claim's evidence and metadata.

    Six factors:
    1. Source count     — more independent sources → higher confidence
    2. Source diversity — SEC + USGS > SEC + SEC
    3. Directness      — EXTRACTED > INFERRED > SEEDED
    4. Recency         — newer evidence → higher confidence
    5. Doc quality      — 10-K > web page
    6. Counter penalty  — weakening evidence reduces confidence
    """

    def score(self, claim: "Claim", now: Optional[datetime] = None) -> ConfidenceResult:
        """Compute confidence for a claim. Pure function, no side effects."""
        factors: List[ConfidenceFactor] = []
        all_evidence = claim.supporting_evidence

        # ── Factor 1: Source count ──
        n = len(all_evidence)
        if n == 0:
            return ConfidenceResult.from_score(
                0.0,
                [
                    ConfidenceFactor(
                        name="source_count",
                        raw_value="0 sources",
                        contribution=0.0,
                        direction="NEGATIVE",
                        explanation="No evidence supports this claim",
                    ),
                ],
            )

        # Diminishing returns: 1→0.4, 2→0.7, 3→0.85, 4+→0.95+
        count_score = min(1.0, 1.0 - (0.6**n))
        factors.append(
            ConfidenceFactor(
                name="source_count",
                raw_value=f"{n} source{'s' if n != 1 else ''}",
                contribution=round(count_score * _FACTOR_WEIGHTS["source_count"], 4),
                direction="POSITIVE",
                explanation=f"{n} evidence source{'s' if n != 1 else ''} support{'s' if n == 1 else ''} this claim",
            )
        )

        # ── Factor 2: Source diversity ──
        source_types = set(p.source_type for p in all_evidence)
        diversity = len(source_types)
        diversity_score = min(1.0, diversity / 3.0)  # 3+ distinct types = perfect
        factors.append(
            ConfidenceFactor(
                name="source_diversity",
                raw_value=f"{diversity} distinct source type{'s' if diversity != 1 else ''}",
                contribution=round(
                    diversity_score * _FACTOR_WEIGHTS["source_diversity"], 4
                ),
                direction="POSITIVE",
                explanation=f"Evidence from {', '.join(st.value for st in source_types)}",
            )
        )

        # ── Factor 3: Directness ──
        directness_score = _ASSERTION_DIRECTNESS.get(claim.assertion_mode, 0.3)
        factors.append(
            ConfidenceFactor(
                name="directness",
                raw_value=claim.assertion_mode.value,
                contribution=round(directness_score * _FACTOR_WEIGHTS["directness"], 4),
                direction="POSITIVE" if directness_score >= 0.5 else "NEGATIVE",
                explanation=f"Claim is {claim.assertion_mode.value.lower()} — "
                + (
                    "directly stated in source"
                    if directness_score >= 1.0
                    else "inferred from related evidence"
                    if directness_score >= 0.5
                    else "based on seeded/baseline data"
                ),
            )
        )

        # ── Factor 4: Recency ──
        recency_score = self._compute_recency(all_evidence, now=now)
        factors.append(
            ConfidenceFactor(
                name="recency",
                raw_value=self._newest_date(all_evidence),
                contribution=round(recency_score * _FACTOR_WEIGHTS["recency"], 4),
                direction="POSITIVE" if recency_score >= 0.5 else "NEGATIVE",
                explanation="Most recent evidence is "
                + (
                    "current"
                    if recency_score >= 0.8
                    else "recent"
                    if recency_score >= 0.5
                    else "aging — may be stale"
                ),
            )
        )

        # ── Factor 5: Document quality ──
        qualities = [_SOURCE_QUALITY.get(p.source_type, 0.3) for p in all_evidence]
        best_quality = max(qualities) if qualities else 0.3
        factors.append(
            ConfidenceFactor(
                name="doc_quality",
                raw_value=f"best: {all_evidence[qualities.index(best_quality)].source_type.value}"
                if all_evidence
                else "none",
                contribution=round(best_quality * _FACTOR_WEIGHTS["doc_quality"], 4),
                direction="POSITIVE" if best_quality >= 0.7 else "NEGATIVE",
                explanation=f"Highest-quality source is {all_evidence[qualities.index(best_quality)].source_type.value}"
                if all_evidence
                else "No sources",
            )
        )

        # ── Factor 6: Counter-evidence penalty ──
        n_weakening = len(claim.weakening_evidence)
        if n_weakening > 0:
            # Penalty scales: 1 counter → -0.3, 2 → -0.5, 3+ → -0.7
            penalty_raw = min(0.7, 0.15 + 0.15 * n_weakening)
            penalty = -penalty_raw
            factors.append(
                ConfidenceFactor(
                    name="counter_penalty",
                    raw_value=f"{n_weakening} weakening source{'s' if n_weakening != 1 else ''}",
                    contribution=round(penalty * _FACTOR_WEIGHTS["counter_penalty"], 4),
                    direction="NEGATIVE",
                    explanation=f"{n_weakening} piece{'s' if n_weakening != 1 else ''} of counter-evidence weaken{'s' if n_weakening == 1 else ''} this claim",
                )
            )
        else:
            factors.append(
                ConfidenceFactor(
                    name="counter_penalty",
                    raw_value="0 weakening sources",
                    contribution=0.0,
                    direction="POSITIVE",
                    explanation="No counter-evidence found",
                )
            )

        # ── Aggregate ──
        total = sum(f.contribution for f in factors)
        # Clamp to [0, 1]
        final_score = max(0.0, min(1.0, total))

        return ConfidenceResult.from_score(final_score, factors)

    def _compute_recency(self, evidence: list, now: Optional[datetime] = None) -> float:
        """Score evidence recency (0.0 = very old, 1.0 = current)."""
        now = now or datetime.now(timezone.utc)
        dates = []
        for p in evidence:
            for date_field in (p.extracted_at, p.observed_at, p.valid_from):
                if date_field:
                    try:
                        dt = datetime.fromisoformat(date_field.replace("Z", "+00:00"))
                        dates.append(dt)
                    except (ValueError, TypeError):
                        continue

        if not dates:
            return 0.5  # Unknown recency → neutral

        newest = max(dates)
        age_years = (now - newest).days / 365.25
        if age_years <= _RECENCY_FULL_CREDIT_YEARS:
            return 1.0
        # Exponential decay after full credit window
        decay = 0.5 ** (
            (age_years - _RECENCY_FULL_CREDIT_YEARS) / _RECENCY_HALF_LIFE_YEARS
        )
        return max(0.1, decay)

    def _newest_date(self, evidence: list) -> str:
        """Find the newest date string across all evidence."""
        dates = []
        for p in evidence:
            for date_field in (p.extracted_at, p.observed_at):
                if date_field:
                    dates.append(date_field)
        return max(dates) if dates else "unknown"
