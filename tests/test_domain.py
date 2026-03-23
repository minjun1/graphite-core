"""tests/test_domain.py — Unit tests for graphite.domain."""

import pytest
from graphite.domain import (
    ExtractionResult, DocumentContext, BaseFetcher, BaseExtractor,
    BasePipeline, DomainSpec, register_domain, get_domain, list_domains,
    _domain_registry,
)
from graphite.enums import NodeType, SourceType
from graphite.claim import Claim, ClaimType, ClaimGranularity, ClaimOrigin
from graphite.enums import AssertionMode


class TestExtractionResult:
    def test_construction_empty(self):
        er = ExtractionResult()
        assert er.claims == []
        assert er.edges == []
        assert er.diagnostics == {}
        assert er.unresolvable_count == 0

    def test_construction_with_claims(self):
        claim = Claim(
            claim_text="test",
            claim_type=ClaimType.RELATIONSHIP,
            subject_entities=["A"],
            predicate="SUPPLIES_TO",
            object_entities=["B"],
            assertion_mode=AssertionMode.EXTRACTED,
            origin=ClaimOrigin.EXTRACTOR,
            granularity=ClaimGranularity.ATOMIC,
        )
        er = ExtractionResult(claims=[claim])
        assert len(er.claims) == 1


class TestDocumentContext:
    def test_fields(self):
        dc = DocumentContext(
            source_id="acc-001",
            source_type=SourceType.SEC_10K,
            entity_id="AAPL",
            text_content="Full text here",
            paragraphs=["para1", "para2"],
            doc_url="https://example.com",
        )
        assert dc.source_id == "acc-001"
        assert dc.entity_id == "AAPL"
        assert len(dc.paragraphs) == 2

    def test_default_metadata(self):
        dc = DocumentContext(
            source_id="x", source_type=SourceType.WEB,
            entity_id="y", text_content="t", paragraphs=[],
        )
        assert dc.metadata == {}
        assert dc.doc_url == ""


class TestBaseFetcher:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseFetcher()

    def test_fetch_batch_calls_fetch(self):
        class FakeFetcher(BaseFetcher):
            def fetch(self, entity_id, **kwargs):
                return [DocumentContext(
                    source_id=entity_id, source_type=SourceType.WEB,
                    entity_id=entity_id, text_content="text",
                    paragraphs=["p"],
                )]

        f = FakeFetcher()
        docs = f.fetch_batch(["A", "B", "C"])
        assert len(docs) == 3
        assert docs[0].entity_id == "A"
        assert docs[2].entity_id == "C"


class TestBaseExtractor:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseExtractor()

    def test_extract_batch_calls_extract(self):
        class FakeExtractor(BaseExtractor):
            def extract(self, doc):
                claim = Claim(
                    claim_text=f"claim for {doc.entity_id}",
                    claim_type=ClaimType.RELATIONSHIP,
                    subject_entities=[doc.entity_id],
                    predicate="PRODUCES",
                    object_entities=["mineral:COBALT"],
                    assertion_mode=AssertionMode.EXTRACTED,
                    origin=ClaimOrigin.EXTRACTOR,
                    granularity=ClaimGranularity.ATOMIC,
                )
                return ExtractionResult(claims=[claim])

        ext = FakeExtractor()
        docs = [
            DocumentContext("s1", SourceType.SEC_10K, "AAPL", "t", []),
            DocumentContext("s2", SourceType.SEC_10K, "TSLA", "t", []),
        ]
        result = ext.extract_batch(docs)
        assert len(result.claims) == 2
        assert result.unresolvable_count == 0


class TestBasePipeline:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BasePipeline()

    def test_concrete_subclass(self):
        class FakePipeline(BasePipeline):
            def run(self, entity_ids, output_path, **kwargs):
                return {"entities": entity_ids}

        p = FakePipeline()
        result = p.run(["AAPL"], "/tmp/out.json")
        assert result == {"entities": ["AAPL"]}


class TestDomainRegistry:
    @pytest.fixture(autouse=True)
    def _save_restore_registry(self):
        saved = dict(_domain_registry)
        yield
        _domain_registry.clear()
        _domain_registry.update(saved)

    def test_register_and_get(self):
        spec = DomainSpec(
            name="_test_domain",
            allowed_edge_types=["PRODUCES"],
            allowed_node_types=[NodeType.COMPANY, NodeType.MINERAL],
        )
        register_domain(spec)
        retrieved = get_domain("_test_domain")
        assert retrieved is not None
        assert retrieved.name == "_test_domain"
        assert "PRODUCES" in retrieved.allowed_edge_types

    def test_get_unknown_returns_none(self):
        assert get_domain("nonexistent_domain_xyz") is None

    def test_list_domains_includes_registered(self):
        spec = DomainSpec(
            name="_test_domain_list",
            allowed_edge_types=[],
            allowed_node_types=[],
        )
        register_domain(spec)
        assert "_test_domain_list" in list_domains()
