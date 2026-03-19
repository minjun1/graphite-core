"""
tests/test_pipeline_core.py — Unit tests for the core pipeline primitives.

Tests: NodeRef, ExtractedEdge, Provenance, GraphAssembler (merge/dedupe/quality),
       text strategies, cache, graph_writer roundtrip, DomainSpec registry.
"""

import json
import os
import tempfile

import networkx as nx
import pytest

from graphite.enums import (
    AssertionMode,
    ConfidenceLevel,
    EdgeType,
    EvidenceType,
    NodeType,
    SourceType,
)
from graphite.schemas import (
    ExtractedEdge,
    ExtractionError,
    InferenceBasis,
    NodeRef,
    Provenance,
)
from graphite.domain import (
    DomainSpec,
    register_domain,
    get_domain,
    list_domains,
    DocumentContext,
)
from graphite.assembler import GraphAssembler
from graphite.cache import PipelineCache
from graphite.io import save_graph, load_graph
from graphite.text import (
    build_context,
    register_strategy,
    split_into_paragraphs,
    score_paragraph,
    find_best_paragraph_for_quote,
    clip_quote,
)


# ═══════════════════════════════════════
# NodeRef
# ═══════════════════════════════════════


class TestNodeRef:
    def test_company_factory(self):
        n = NodeRef.company("AAPL", label="Apple Inc.")
        assert n.node_id == "company:AAPL"
        assert n.node_type == NodeType.COMPANY
        assert n.label == "Apple Inc."

    def test_country_factory(self):
        n = NodeRef.country("CD", label="DR Congo")
        assert n.node_id == "country:CD"
        assert n.node_type == NodeType.COUNTRY

    def test_mineral_factory(self):
        n = NodeRef.mineral("COBALT")
        assert n.node_id == "mineral:COBALT"
        assert n.node_type == NodeType.MINERAL

    def test_case_normalization(self):
        n = NodeRef.company("aapl")
        assert n.node_id == "company:AAPL"

    # ── Geo-climate factories ──

    def test_region_factory(self):
        n = NodeRef.region("US-TX-HOUSTON", label="Houston Metro")
        assert n.node_id == "region:US-TX-HOUSTON"
        assert n.node_type == NodeType.REGION
        assert n.label == "Houston Metro"

    def test_asset_factory(self):
        n = NodeRef.asset("PORT_HOUSTON", label="Port of Houston")
        assert n.node_id == "asset:PORT_HOUSTON"
        assert n.node_type == NodeType.ASSET

    def test_facility_factory(self):
        n = NodeRef.facility("REFINERY_A", label="Motiva Refinery")
        assert n.node_id == "facility:REFINERY_A"
        assert n.node_type == NodeType.FACILITY

    def test_corridor_factory(self):
        n = NodeRef.corridor("HOUSTON_SHIP_CHANNEL")
        assert n.node_id == "corridor:HOUSTON_SHIP_CHANNEL"
        assert n.node_type == NodeType.CORRIDOR

    def test_region_case_normalization(self):
        n = NodeRef.region("us-tx-houston")
        assert n.node_id == "region:US-TX-HOUSTON"


# ═══════════════════════════════════════
# Provenance
# ═══════════════════════════════════════


class TestProvenance:
    def test_defaults(self):
        p = Provenance(
            source_id="12345",
            source_type=SourceType.USGS_MCS,
            evidence_quote="Congo produces 73% of world cobalt",
        )
        assert p.confidence == ConfidenceLevel.MEDIUM
        assert p.evidence_type == EvidenceType.TEXT_QUOTE
        assert p.extracted_at  # auto-set

    def test_table_cell_type(self):
        p = Provenance(
            source_id="usgs-cobalt",
            source_type=SourceType.USGS_MCS,
            evidence_quote="73%",
            evidence_type=EvidenceType.TABLE_CELL,
            confidence=ConfidenceLevel.HIGH,
        )
        assert p.evidence_type == EvidenceType.TABLE_CELL

    def test_temporal_fields_default(self):
        p = Provenance(
            source_id="test",
            source_type=SourceType.WEATHER_FORECAST,
            evidence_quote="Category 4 hurricane forecast",
        )
        assert p.observed_at == ""
        assert p.valid_from == ""
        assert p.valid_to == ""
        assert p.snapshot_id == ""

    def test_temporal_fields_set(self):
        p = Provenance(
            source_id="noaa-harvey",
            source_type=SourceType.WEATHER_FORECAST,
            evidence_quote="Category 4 hurricane made landfall",
            observed_at="2017-08-25T22:00:00Z",
            valid_from="2017-08-25",
            valid_to="2017-08-30",
            snapshot_id="harvey-2017-snapshot",
        )
        assert p.observed_at == "2017-08-25T22:00:00Z"
        assert p.snapshot_id == "harvey-2017-snapshot"


class TestExtractedEdge:
    def _make_edge(self, from_id="country:CD", to_id="mineral:COBALT", **kwargs):
        return ExtractedEdge(
            from_node=NodeRef(node_id=from_id, node_type=NodeType.COUNTRY),
            to_node=NodeRef(node_id=to_id, node_type=NodeType.MINERAL),
            edge_type=EdgeType.PRODUCES,
            assertion_mode=AssertionMode.EXTRACTED,
            provenance=[
                Provenance(
                    source_id="usgs-cobalt",
                    source_type=SourceType.USGS_MCS,
                    evidence_quote="Congo produces 73% of world cobalt",
                    confidence=ConfidenceLevel.HIGH,
                )
            ],
            **kwargs,
        )

    def test_edge_key(self):
        e = self._make_edge()
        assert e.edge_key == "country:CD|mineral:COBALT|PRODUCES"

    def test_best_confidence(self):
        e = self._make_edge()
        assert e.best_confidence == ConfidenceLevel.HIGH

    def test_inference_basis(self):
        e = ExtractedEdge(
            from_node=NodeRef(node_id="company:TSLA", node_type=NodeType.COMPANY),
            to_node=NodeRef(node_id="company:CATL", node_type=NodeType.COMPANY),
            edge_type=EdgeType.SUPPLIES_TO,
            assertion_mode=AssertionMode.INFERRED,
            inference_basis=InferenceBasis(
                method="customer_filing_reverse",
                reason="Tesla's 10-K mentions CATL as battery supplier",
                based_on_edges=["company:TSLA|company:CATL|SUPPLIES_TO"],
                source_nodes=["company:TSLA"],
            ),
            provenance=[
                Provenance(
                    source_id="tsla-10k",
                    source_type=SourceType.SEC_10K,
                    evidence_quote="CATL as primary battery cell supplier",
                    confidence=ConfidenceLevel.HIGH,
                )
            ],
        )
        assert e.inference_basis.method == "customer_filing_reverse"
        assert len(e.inference_basis.source_nodes) == 1


# ═══════════════════════════════════════
# GraphAssembler
# ═══════════════════════════════════════


class TestGraphAssembler:
    def _make_edge(
        self, from_id, to_id, edge_type, source_id, confidence, weight=0.5, **attrs
    ):
        return ExtractedEdge(
            from_node=NodeRef(node_id=from_id, node_type=NodeType.COUNTRY),
            to_node=NodeRef(node_id=to_id, node_type=NodeType.MINERAL),
            edge_type=edge_type,
            assertion_mode=AssertionMode.EXTRACTED,
            attributes={"bucket_weight": weight, **attrs},
            provenance=[
                Provenance(
                    source_id=source_id,
                    source_type=SourceType.USGS_MCS,
                    evidence_quote=f"Test quote for {from_id} → {to_id}",
                    confidence=confidence,
                )
            ],
        )

    def test_basic_assembly(self):
        edges = [
            self._make_edge(
                "country:CD",
                "mineral:COBALT",
                "PRODUCES",
                "usgs1",
                ConfidenceLevel.HIGH,
            ),
            self._make_edge(
                "country:AU",
                "mineral:LITHIUM",
                "PRODUCES",
                "usgs2",
                ConfidenceLevel.HIGH,
            ),
        ]
        asm = GraphAssembler()
        G = asm.assemble(edges)
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 2

    def test_dedupe_merges_provenance(self):
        e1 = self._make_edge(
            "country:CD", "mineral:COBALT", "PRODUCES", "src_usgs", ConfidenceLevel.HIGH
        )
        e2 = self._make_edge(
            "country:CD",
            "mineral:COBALT",
            "PRODUCES",
            "src_sec",
            ConfidenceLevel.MEDIUM,
        )

        asm = GraphAssembler()
        deduped = asm.dedupe_edges([e1, e2])
        assert len(deduped) == 1
        assert len(deduped[0].provenance) == 2

    def test_conflict_tracking(self):
        e1 = self._make_edge(
            "country:CD",
            "mineral:COBALT",
            "PRODUCES",
            "src1",
            ConfidenceLevel.HIGH,
            production_pct=73,
        )
        e2 = self._make_edge(
            "country:CD",
            "mineral:COBALT",
            "PRODUCES",
            "src2",
            ConfidenceLevel.MEDIUM,
            production_pct=71,
        )

        asm = GraphAssembler()
        deduped = asm.dedupe_edges([e1, e2])
        assert len(deduped) == 1
        assert "production_pct_reported_values" in deduped[0].attributes
        rv = deduped[0].attributes["production_pct_reported_values"]
        values = [r["value"] for r in rv]
        assert 73 in values and 71 in values

    def test_assertion_mode_upgrade(self):
        e1 = ExtractedEdge(
            from_node=NodeRef.country("CD"),
            to_node=NodeRef.mineral("COBALT"),
            edge_type="PRODUCES",
            assertion_mode=AssertionMode.SEEDED,
            provenance=[
                Provenance(
                    source_id="seed",
                    source_type=SourceType.MANUAL,
                    evidence_quote="seed data",
                    confidence=ConfidenceLevel.LOW,
                )
            ],
        )
        e2 = ExtractedEdge(
            from_node=NodeRef.country("CD"),
            to_node=NodeRef.mineral("COBALT"),
            edge_type="PRODUCES",
            assertion_mode=AssertionMode.EXTRACTED,
            provenance=[
                Provenance(
                    source_id="usgs",
                    source_type=SourceType.USGS_MCS,
                    evidence_quote="Congo produces 73%",
                    confidence=ConfidenceLevel.HIGH,
                )
            ],
        )
        asm = GraphAssembler()
        deduped = asm.dedupe_edges([e1, e2])
        assert deduped[0].assertion_mode == AssertionMode.EXTRACTED

    def test_drop_zero_provenance(self):
        e = ExtractedEdge(
            from_node=NodeRef.country("CD"),
            to_node=NodeRef.mineral("COBALT"),
            edge_type="PRODUCES",
            assertion_mode=AssertionMode.SEEDED,
            provenance=[],  # empty
        )
        asm = GraphAssembler(drop_zero_provenance=True)
        G = asm.assemble([e])
        assert G.number_of_edges() == 0
        assert len(asm.errors) == 1

    def test_edge_type_validation(self):
        spec = DomainSpec(
            name="minerals",
            allowed_edge_types=["PRODUCES", "REFINED_BY"],
            allowed_node_types=[NodeType.COUNTRY, NodeType.MINERAL],
        )
        e = self._make_edge(
            "country:CD", "mineral:COBALT", "INVALID_TYPE", "src", ConfidenceLevel.HIGH
        )
        asm = GraphAssembler(domain_spec=spec)
        G = asm.assemble([e])
        assert G.number_of_edges() == 0
        assert len(asm.errors) == 1

    def test_graph_stamping(self):
        edges = [
            self._make_edge(
                "country:CD",
                "mineral:COBALT",
                "PRODUCES",
                "usgs1",
                ConfidenceLevel.HIGH,
            )
        ]
        asm = GraphAssembler(pipeline_version="2.0")
        G = asm.assemble(edges)
        assert G.graph["pipeline_version"] == "2.0"
        assert G.graph["node_count"] == 2
        assert G.graph["edge_count"] == 1
        assert "built_at" in G.graph


# ═══════════════════════════════════════
# Text Strategies
# ═══════════════════════════════════════


class TestTextStrategies:
    def test_split_paragraphs(self):
        text = "Short.\n\n" + "A" * 100 + "\n\n" + "B" * 100
        paras = split_into_paragraphs(text, min_len=80)
        assert len(paras) == 2

    def test_score_paragraph(self):
        p = "The supplier delivers critical mineral cobalt from mines in DRC."
        score = score_paragraph(p, ["supplier", "cobalt", "mine", "DRC"])
        assert score >= 3

    def test_build_context_default(self):
        doc = DocumentContext(
            source_id="test",
            source_type=SourceType.USGS_MCS,
            entity_id="COBALT",
            text_content="text",
            paragraphs=["Supplier delivers cobalt from mines." * 5] * 10,
        )
        ctx = build_context(doc, strategy="default")
        assert len(ctx) > 0
        assert "[PARA_" in ctx

    def test_build_context_usgs(self):
        doc = DocumentContext(
            source_id="test",
            source_type=SourceType.USGS_MCS,
            entity_id="COBALT",
            text_content="text",
            paragraphs=["World mine production of cobalt in Congo." * 5] * 10,
        )
        ctx = build_context(doc, strategy="usgs_country_mineral")
        assert len(ctx) > 0

    def test_register_custom_strategy(self):
        def _custom(paragraphs, **kw):
            return "CUSTOM:" + "\n".join(paragraphs[:2])

        register_strategy("test_custom", _custom)
        doc = DocumentContext(
            source_id="t",
            source_type=SourceType.MANUAL,
            entity_id="X",
            text_content="",
            paragraphs=["A", "B", "C"],
        )
        ctx = build_context(doc, strategy="test_custom")
        assert ctx.startswith("CUSTOM:")

    def test_find_best_paragraph(self):
        paras = [
            "This paragraph discusses company revenue.",
            "The company sources cobalt from Congo supplying 73% of production.",
            "Financial results for the year ended December 31.",
        ]
        idx, h = find_best_paragraph_for_quote(paras, "cobalt from Congo supplying 73%")
        assert idx == 1

    def test_clip_quote(self):
        assert len(clip_quote("x" * 300, 280)) <= 281  # +1 for …


# ═══════════════════════════════════════
# Cache
# ═══════════════════════════════════════


class TestCache:
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            cache = PipelineCache(cache_dir=td)
            key = PipelineCache.make_key("src1", "hash1", "1.0", "1.0", "gemini-3.1")
            assert not cache.has(key)

            cache.put(key, {"edges": [{"from": "A", "to": "B"}]})
            assert cache.has(key)

            data = cache.get(key)
            assert data["edges"][0]["from"] == "A"

    def test_content_hash(self):
        h1 = PipelineCache.content_hash("hello world")
        h2 = PipelineCache.content_hash("hello world")
        h3 = PipelineCache.content_hash("hello world!")
        assert h1 == h2
        assert h1 != h3


# ═══════════════════════════════════════
# Graph Writer Roundtrip
# ═══════════════════════════════════════


class TestGraphWriter:
    def _make_graph(self):
        edges = [
            ExtractedEdge(
                from_node=NodeRef.country("CD"),
                to_node=NodeRef.mineral("COBALT"),
                edge_type="PRODUCES",
                assertion_mode=AssertionMode.EXTRACTED,
                attributes={"production_pct": 73, "bucket_weight": 0.8},
                provenance=[
                    Provenance(
                        source_id="usgs-cobalt",
                        source_type=SourceType.USGS_MCS,
                        evidence_quote="Congo produces 73% of world cobalt",
                        confidence=ConfidenceLevel.HIGH,
                    )
                ],
            ),
        ]
        asm = GraphAssembler()
        return asm.assemble(edges)

    def test_graphml_roundtrip(self):
        G = self._make_graph()
        with tempfile.TemporaryDirectory() as td:
            path = save_graph(G, os.path.join(td, "test.graphml"), format="graphml")
            G2 = load_graph(path)
            assert G2.number_of_nodes() == G.number_of_nodes()
            assert G2.number_of_edges() == G.number_of_edges()

            # Check provenance survived as JSON string
            for _, _, d in G2.edges(data=True):
                assert "provenance_json" in d
                assert "provenance_parsed" in d
                assert len(d["provenance_parsed"]) == 1

    def test_json_roundtrip(self):
        G = self._make_graph()
        with tempfile.TemporaryDirectory() as td:
            path = save_graph(G, os.path.join(td, "test.json"), format="json")
            G2 = load_graph(path)
            assert G2.number_of_nodes() == G.number_of_nodes()
            assert G2.number_of_edges() == G.number_of_edges()

    def test_json_has_nested_provenance(self):
        G = self._make_graph()
        with tempfile.TemporaryDirectory() as td:
            path = save_graph(G, os.path.join(td, "test.json"), format="json")
            with open(path) as f:
                data = json.load(f)
            assert "provenance" in data["edges"][0]
            assert isinstance(data["edges"][0]["provenance"], list)


# ═══════════════════════════════════════
# DomainSpec Registry
# ═══════════════════════════════════════


class TestDomainRegistry:
    def test_register_and_get(self):
        spec = DomainSpec(
            name="test_domain",
            allowed_edge_types=["PRODUCES", "SUPPLIES_TO"],
            allowed_node_types=[NodeType.COMPANY, NodeType.MINERAL],
        )
        register_domain(spec)
        assert get_domain("test_domain") is not None
        assert "test_domain" in list_domains()

    def test_propagation_alphas_optional(self):
        spec = DomainSpec(
            name="no_alphas",
            allowed_edge_types=["PRODUCES"],
            allowed_node_types=[NodeType.MINERAL],
        )
        assert spec.propagation_alphas == {}
