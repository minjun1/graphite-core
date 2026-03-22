"""tests/test_claim_store.py — Unit tests for graphite.claim_store."""

import pytest
from graphite.claim_store import ClaimStore
from graphite.claim import (
    Claim, ClaimType, ClaimOrigin, ClaimGranularity, ClaimStatus, ReviewState,
)
from graphite.enums import AssertionMode, SourceType, ConfidenceLevel
from graphite.schemas import Provenance


def _make_claim(subjects=None, predicate="SUPPLIES_TO", objects=None, **kwargs):
    return Claim(
        claim_text=kwargs.get("claim_text", "Test claim"),
        claim_type=ClaimType.RELATIONSHIP,
        subject_entities=subjects or ["company:AAPL"],
        predicate=predicate,
        object_entities=objects or ["company:TSLA"],
        assertion_mode=AssertionMode.EXTRACTED,
        origin=ClaimOrigin.EXTRACTOR,
        granularity=ClaimGranularity.ATOMIC,
        **{k: v for k, v in kwargs.items() if k != "claim_text"},
    )


def _make_provenance(source_id="src-1", quote="Evidence quote"):
    return Provenance(
        source_id=source_id,
        source_type=SourceType.SEC_10K,
        evidence_quote=quote,
        confidence=ConfidenceLevel.HIGH,
    )


class TestSaveAndGet:
    def test_roundtrip(self, tmp_path):
        store = ClaimStore(db_path=str(tmp_path / "claims.db"))
        claim = _make_claim()
        store.save_claim(claim)
        retrieved = store.get_claim(claim.claim_id)
        assert retrieved is not None
        assert retrieved.claim_id == claim.claim_id
        assert retrieved.predicate == "SUPPLIES_TO"

    def test_get_missing_returns_none(self, tmp_path):
        store = ClaimStore(db_path=str(tmp_path / "claims.db"))
        assert store.get_claim("nonexistent") is None


class TestEvidenceAccumulation:
    def test_merge_different_evidence(self, tmp_path):
        store = ClaimStore(db_path=str(tmp_path / "claims.db"))

        c1 = _make_claim(supporting_evidence=[_make_provenance("src-1", "quote A")])
        store.save_claim(c1)

        c2 = _make_claim(supporting_evidence=[_make_provenance("src-2", "quote B")])
        store.save_claim(c2)

        retrieved = store.get_claim(c1.claim_id)
        assert len(retrieved.supporting_evidence) == 2

    def test_dedup_same_source_same_quote(self, tmp_path):
        store = ClaimStore(db_path=str(tmp_path / "claims.db"))

        c1 = _make_claim(supporting_evidence=[_make_provenance("src-1", "same quote")])
        store.save_claim(c1)

        c2 = _make_claim(supporting_evidence=[_make_provenance("src-1", "same quote")])
        store.save_claim(c2)

        retrieved = store.get_claim(c1.claim_id)
        assert len(retrieved.supporting_evidence) == 1

    def test_append_same_source_different_quote(self, tmp_path):
        store = ClaimStore(db_path=str(tmp_path / "claims.db"))

        c1 = _make_claim(supporting_evidence=[_make_provenance("src-1", "quote A")])
        store.save_claim(c1)

        c2 = _make_claim(supporting_evidence=[_make_provenance("src-1", "quote B")])
        store.save_claim(c2)

        retrieved = store.get_claim(c1.claim_id)
        assert len(retrieved.supporting_evidence) == 2


class TestSaveClaims:
    def test_batch_save(self, tmp_path):
        store = ClaimStore(db_path=str(tmp_path / "claims.db"))
        c1 = _make_claim(subjects=["company:AAPL"], objects=["company:TSLA"])
        c2 = _make_claim(subjects=["company:NVDA"], objects=["company:TSLA"])
        store.save_claims([c1, c2])

        assert store.get_claim(c1.claim_id) is not None
        assert store.get_claim(c2.claim_id) is not None


class TestSearchClaims:
    def test_subject_contains(self, tmp_path):
        store = ClaimStore(db_path=str(tmp_path / "claims.db"))
        store.save_claim(_make_claim(subjects=["company:AAPL"]))
        store.save_claim(_make_claim(subjects=["company:NVDA"], objects=["company:AMD"]))

        results = store.search_claims(subject_contains="AAPL")
        assert len(results) == 1
        assert "company:AAPL" in results[0].subject_entities

    def test_object_contains(self, tmp_path):
        store = ClaimStore(db_path=str(tmp_path / "claims.db"))
        store.save_claim(_make_claim(objects=["company:TSLA"]))
        results = store.search_claims(object_contains="TSLA")
        assert len(results) == 1

    def test_predicate_filter(self, tmp_path):
        store = ClaimStore(db_path=str(tmp_path / "claims.db"))
        store.save_claim(_make_claim(predicate="SUPPLIES_TO"))
        store.save_claim(_make_claim(
            subjects=["company:X"], predicate="PRODUCES", objects=["mineral:COBALT"],
        ))
        results = store.search_claims(predicate="PRODUCES")
        assert len(results) == 1

    def test_as_of_date_filter(self, tmp_path):
        store = ClaimStore(db_path=str(tmp_path / "claims.db"))
        store.save_claim(_make_claim(as_of_date="2025-01-01"))
        store.save_claim(_make_claim(
            subjects=["company:X"], objects=["company:Y"], as_of_date="2024-06-01",
        ))
        results = store.search_claims(as_of_date="2025-01-01")
        assert len(results) == 1

    def test_empty_results(self, tmp_path):
        store = ClaimStore(db_path=str(tmp_path / "claims.db"))
        assert store.search_claims(subject_contains="NONEXISTENT") == []


class TestFindSupportingClaims:
    def test_finds_same_predicate_shared_entity(self, tmp_path):
        store = ClaimStore(db_path=str(tmp_path / "claims.db"))
        c1 = _make_claim(subjects=["company:AAPL"], predicate="SUPPLIES_TO", objects=["company:TSLA"])
        c2 = _make_claim(subjects=["company:AAPL"], predicate="SUPPLIES_TO", objects=["company:RIVN"])
        store.save_claims([c1, c2])

        supporting = store.find_supporting_claims(c1)
        assert len(supporting) == 1
        assert supporting[0].claim_id == c2.claim_id

    def test_excludes_self(self, tmp_path):
        store = ClaimStore(db_path=str(tmp_path / "claims.db"))
        c1 = _make_claim()
        store.save_claim(c1)
        supporting = store.find_supporting_claims(c1)
        assert len(supporting) == 0


class TestFindPotentialConflicts:
    def test_finds_different_predicate_shared_entity(self, tmp_path):
        store = ClaimStore(db_path=str(tmp_path / "claims.db"))
        c1 = _make_claim(subjects=["company:AAPL"], predicate="SUPPLIES_TO", objects=["company:TSLA"])
        c2 = _make_claim(subjects=["company:AAPL"], predicate="COMPETES_WITH", objects=["company:TSLA"])
        store.save_claims([c1, c2])

        conflicts = store.find_potential_conflicts(c1)
        assert len(conflicts) == 1
        assert conflicts[0].predicate == "COMPETES_WITH"


class TestAnalystOverridePreservation:
    def test_override_preserved_on_merge(self, tmp_path):
        store = ClaimStore(db_path=str(tmp_path / "claims.db"))
        c1 = _make_claim()
        store.save_claim(c1)

        c2 = _make_claim()
        c2.override_status(ClaimStatus.SUPPORTED, reason="Analyst confirmed", reviewer="analyst-1")
        store.save_claim(c2)

        retrieved = store.get_claim(c1.claim_id)
        assert retrieved.is_overridden
        assert retrieved.override_reason == "Analyst confirmed"

    def test_generator_id_preserved(self, tmp_path):
        store = ClaimStore(db_path=str(tmp_path / "claims.db"))
        c1 = _make_claim()
        store.save_claim(c1)

        c2 = _make_claim(generator_id="gemini-2.5-flash")
        store.save_claim(c2)

        retrieved = store.get_claim(c1.claim_id)
        assert retrieved.generator_id == "gemini-2.5-flash"

    def test_generation_metadata_merged(self, tmp_path):
        store = ClaimStore(db_path=str(tmp_path / "claims.db"))
        c1 = _make_claim(generation_metadata={"run": 1})
        store.save_claim(c1)

        c2 = _make_claim(generation_metadata={"run": 2, "extra": True})
        store.save_claim(c2)

        retrieved = store.get_claim(c1.claim_id)
        assert retrieved.generation_metadata["run"] == 2
        assert retrieved.generation_metadata["extra"] is True
