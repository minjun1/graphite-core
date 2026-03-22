"""tests/test_retriever.py — Unit tests for graphite.pipeline.retriever."""

import pytest
from graphite.pipeline.retriever import DocumentCorpus, EvidenceRetriever, retrieve_evidence
from graphite.claim import Claim, ClaimType, ClaimOrigin, ClaimGranularity
from graphite.enums import AssertionMode


def _make_claim(subjects=None, predicate="SUPPLIES_TO", objects=None, claim_text="test"):
    return Claim(
        claim_text=claim_text,
        claim_type=ClaimType.RELATIONSHIP,
        subject_entities=subjects or ["company:AAPL"],
        predicate=predicate,
        object_entities=objects or ["company:TSLA"],
        assertion_mode=AssertionMode.EXTRACTED,
        origin=ClaimOrigin.EXTRACTOR,
        granularity=ClaimGranularity.ATOMIC,
    )


class TestDocumentCorpus:
    def test_chunks_paragraphs(self):
        docs = [
            {"document_id": "doc1", "text": "First paragraph.\n\nSecond paragraph."},
        ]
        corpus = DocumentCorpus(docs)
        assert len(corpus.chunks) == 2
        assert corpus.chunks[0]["document_id"] == "doc1"
        assert corpus.chunks[0]["chunk_index"] == 0
        assert corpus.chunks[1]["text"] == "Second paragraph."

    def test_skips_empty_paragraphs(self):
        docs = [{"document_id": "doc1", "text": "Hello\n\n\n\nWorld"}]
        corpus = DocumentCorpus(docs)
        texts = [c["text"] for c in corpus.chunks]
        assert "" not in texts

    def test_multiple_documents(self):
        docs = [
            {"document_id": "d1", "text": "A\n\nB"},
            {"document_id": "d2", "text": "C\n\nD"},
        ]
        corpus = DocumentCorpus(docs)
        assert len(corpus.chunks) == 4


class TestEvidenceRetriever:
    def test_retrieve_ranked_by_overlap(self):
        docs = [
            {"document_id": "doc1", "text": "Apple supplies components to Tesla.\n\nUnrelated paragraph about weather."},
        ]
        corpus = DocumentCorpus(docs)
        retriever = EvidenceRetriever(corpus)

        claim = _make_claim(subjects=["company:APPLE"], objects=["company:TESLA"])
        results = retriever.retrieve_evidence([claim])

        chunks = results[claim.claim_id]
        assert len(chunks) >= 1
        assert "apple" in chunks[0]["text"].lower() or "tesla" in chunks[0]["text"].lower()

    def test_top_k_limits(self):
        text = "\n\n".join([f"Apple Tesla paragraph {i}" for i in range(10)])
        docs = [{"document_id": "doc1", "text": text}]
        corpus = DocumentCorpus(docs)
        retriever = EvidenceRetriever(corpus)

        claim = _make_claim(subjects=["company:APPLE"], objects=["company:TESLA"])
        results = retriever.retrieve_evidence([claim], top_k=2)
        assert len(results[claim.claim_id]) <= 2

    def test_empty_corpus(self):
        corpus = DocumentCorpus([])
        retriever = EvidenceRetriever(corpus)
        claim = _make_claim()
        results = retriever.retrieve_evidence([claim])
        assert results[claim.claim_id] == []

    def test_fallback_to_claim_text(self):
        docs = [
            {"document_id": "doc1", "text": "Cobalt mining in Congo is expanding rapidly."},
        ]
        corpus = DocumentCorpus(docs)
        retriever = EvidenceRetriever(corpus)

        # Build claim directly with truly empty entity lists to trigger claim_text fallback
        claim = Claim(
            claim_text="cobalt mining congo",
            claim_type=ClaimType.RELATIONSHIP,
            subject_entities=[],
            predicate="RELATED_TO",
            object_entities=[],
        )
        results = retriever.retrieve_evidence([claim])
        assert len(results[claim.claim_id]) >= 1


class TestRetrieveEvidenceConvenience:
    def test_convenience_function(self):
        docs = [{"document_id": "d1", "text": "Apple supplies Tesla with chips."}]
        claim = _make_claim(subjects=["company:APPLE"], objects=["company:TESLA"])
        results = retrieve_evidence([claim], docs)
        assert claim.claim_id in results
