"""
tests/test_confidence.py — Unit tests for the ConfidenceScorer.

Tests that the scorer produces explainable, deterministic confidence
scores from claim evidence metadata.
"""

import pytest
from datetime import datetime, timezone, timedelta

from graphite.claim import (
    Claim,
    ClaimType,
    ConfidenceResult,
)
from graphite.confidence import ConfidenceScorer
from graphite.enums import (
    AssertionMode,
    ConfidenceLevel,
    SourceType,
    EvidenceType,
)
from graphite.schemas import Provenance


def _prov(
    source_id="s1",
    source_type=SourceType.SEC_10K,
    quote="test evidence",
    extracted_at=None,
):
    return Provenance(
        source_id=source_id,
        source_type=source_type,
        evidence_quote=quote,
        extracted_at=extracted_at or datetime.now(timezone.utc).isoformat(),
    )


def _claim(**kwargs):
    defaults = dict(
        claim_text="test claim",
        claim_type=ClaimType.RELATIONSHIP,
        subject_entities=["company:A"],
        predicate="SUPPLIES_TO",
        object_entities=["company:B"],
        assertion_mode=AssertionMode.EXTRACTED,
    )
    defaults.update(kwargs)
    return Claim(**defaults)


class TestConfidenceScorer:
    def setup_method(self):
        self.scorer = ConfidenceScorer()

    def test_empty_claim_zero_confidence(self):
        """No evidence → score = 0.0."""
        claim = _claim()
        result = self.scorer.score(claim)
        assert result.score == 0.0
        assert result.level == ConfidenceLevel.LOW
        assert len(result.factors) >= 1

    def test_single_source_moderate(self):
        """1 high-quality source, extracted → moderate confidence."""
        claim = _claim(supporting_evidence=[_prov()])
        result = self.scorer.score(claim)
        assert 0.3 <= result.score <= 0.7
        assert len(result.factors) == 6  # all 6 factors

    def test_multi_source_higher(self):
        """3 sources → higher confidence than 1 source."""
        claim_1 = _claim(supporting_evidence=[_prov("s1")])
        claim_3 = _claim(
            supporting_evidence=[
                _prov("s1"),
                _prov("s2"),
                _prov("s3"),
            ]
        )
        r1 = self.scorer.score(claim_1)
        r3 = self.scorer.score(claim_3)
        assert r3.score > r1.score

    def test_diverse_sources_higher(self):
        """SEC + USGS > SEC + SEC."""
        claim_same = _claim(
            supporting_evidence=[
                _prov("s1", SourceType.SEC_10K),
                _prov("s2", SourceType.SEC_10K),
            ]
        )
        claim_diverse = _claim(
            supporting_evidence=[
                _prov("s1", SourceType.SEC_10K),
                _prov("s2", SourceType.USGS_MCS),
            ]
        )
        r_same = self.scorer.score(claim_same)
        r_diverse = self.scorer.score(claim_diverse)
        assert r_diverse.score > r_same.score

    def test_inferred_lower_than_extracted(self):
        """Inferred claims get lower confidence than extracted."""
        claim_ext = _claim(
            assertion_mode=AssertionMode.EXTRACTED,
            supporting_evidence=[_prov()],
        )
        claim_inf = _claim(
            assertion_mode=AssertionMode.INFERRED,
            supporting_evidence=[_prov()],
        )
        r_ext = self.scorer.score(claim_ext)
        r_inf = self.scorer.score(claim_inf)
        assert r_ext.score > r_inf.score

    def test_counter_evidence_penalty(self):
        """Weakening evidence reduces confidence."""
        claim_clean = _claim(supporting_evidence=[_prov()])
        claim_counter = _claim(
            supporting_evidence=[_prov("s1")],
            weakening_evidence=[_prov("w1")],
        )
        r_clean = self.scorer.score(claim_clean)
        r_counter = self.scorer.score(claim_counter)
        assert r_counter.score < r_clean.score

    def test_old_evidence_lower(self):
        """Evidence from 8 years ago → lower confidence."""
        old_date = (datetime.now(timezone.utc) - timedelta(days=8 * 365)).isoformat()
        new_date = datetime.now(timezone.utc).isoformat()

        claim_old = _claim(
            supporting_evidence=[_prov(extracted_at=old_date)],
        )
        claim_new = _claim(
            supporting_evidence=[_prov(extracted_at=new_date)],
        )
        r_old = self.scorer.score(claim_old)
        r_new = self.scorer.score(claim_new)
        assert r_new.score > r_old.score

    def test_confidence_level_derivation(self):
        """Score → HIGH/MEDIUM/LOW mapping."""
        r = ConfidenceResult.from_score(0.8)
        assert r.level == ConfidenceLevel.HIGH
        r = ConfidenceResult.from_score(0.5)
        assert r.level == ConfidenceLevel.MEDIUM
        r = ConfidenceResult.from_score(0.2)
        assert r.level == ConfidenceLevel.LOW

    def test_factors_all_explainable(self):
        """Every factor has a non-empty explanation."""
        claim = _claim(
            supporting_evidence=[_prov()],
            weakening_evidence=[_prov("w")],
        )
        result = self.scorer.score(claim)
        for factor in result.factors:
            assert factor.name
            assert factor.explanation
            assert factor.raw_value
            assert factor.direction in ("POSITIVE", "NEGATIVE")

    def test_web_source_lower_quality(self):
        """Web sources should score lower on doc_quality."""
        claim_sec = _claim(supporting_evidence=[_prov(source_type=SourceType.SEC_10K)])
        claim_web = _claim(supporting_evidence=[_prov(source_type=SourceType.WEB)])
        r_sec = self.scorer.score(claim_sec)
        r_web = self.scorer.score(claim_web)
        assert r_sec.score > r_web.score

    def test_document_source_type_has_quality(self):
        """SourceType.DOCUMENT should have an explicit quality tier, not default 0.3."""
        claim = _claim(supporting_evidence=[_prov(source_type=SourceType.DOCUMENT)])
        result = self.scorer.score(claim)
        doc_quality_factor = next(f for f in result.factors if f.name == "doc_quality")
        # DOCUMENT is 0.6 → contribution = 0.6 * 0.15 = 0.09, strictly better than default 0.3 * 0.15
        assert doc_quality_factor.contribution > 0.3 * 0.15

    def test_deterministic(self):
        """Same input always produces same output."""
        claim = _claim(supporting_evidence=[_prov()])
        r1 = self.scorer.score(claim)
        r2 = self.scorer.score(claim)
        assert r1.score == r2.score
        assert len(r1.factors) == len(r2.factors)

    def test_newest_date_uses_valid_from(self):
        """_newest_date should consider valid_from, not just extracted_at/observed_at."""
        scorer = ConfidenceScorer()
        prov = Provenance(
            source_id="src1",
            source_type=SourceType.SEC_10K,
            evidence_quote="test",
            extracted_at="",
            valid_from="2026-01-15",
        )
        result = scorer._newest_date([prov])
        assert result == "2026-01-15"

    def test_recency_reproducible_with_explicit_now(self):
        """score() with explicit now produces identical results."""
        fixed_now = datetime(2026, 3, 21, tzinfo=timezone.utc)
        # Create a claim with evidence from 1 year ago
        claim = _claim(
            supporting_evidence=[
                Provenance(
                    source_id="src1",
                    source_type=SourceType.SEC_10K,
                    evidence_quote="quote",
                    extracted_at=(fixed_now - timedelta(days=365)).isoformat(),
                )
            ],
        )
        scorer = ConfidenceScorer()
        r1 = scorer.score(claim, now=fixed_now)
        r2 = scorer.score(claim, now=fixed_now)
        assert r1.score == r2.score
