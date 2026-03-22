# 100% Module Test Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every source module under `src/graphite/` (excluding `_archive/`) has a dedicated test file with meaningful tests for its public API.

**Architecture:** Three-layer approach — core types first (no deps), then storage (SQLite), then pipeline+adapters (mocked LLM/IO). Tests follow existing class-based grouping style. All tests run with `pytest`, no API keys needed.

**Tech Stack:** pytest, unittest.mock, tmp_path fixture, pydantic, numpy (for adapter tests)

**Spec:** `docs/superpowers/specs/2026-03-20-100-percent-module-test-coverage-design.md`

---

## Layer 1 — Core Types

### Task 1: test_enums.py

**Files:**
- Create: `tests/test_enums.py`

- [ ] **Step 1: Write the test file**

```python
"""tests/test_enums.py — Unit tests for graphite.enums."""

import pytest
from graphite.enums import (
    SourceType, EdgeType, NodeType, AssertionMode,
    ConfidenceLevel, EvidenceType,
)


class TestSourceType:
    def test_members_exist(self):
        expected = [
            "SEC_10K", "SEC_20F", "USGS_MCS", "PDF", "WEB", "MANUAL",
            "WEATHER_FORECAST", "EARTH_OBSERVATION", "PUBLIC_REPORT",
            "GEOSPATIAL_DATA",
        ]
        for name in expected:
            assert hasattr(SourceType, name)

    def test_string_roundtrip(self):
        assert SourceType("SEC_10K") == SourceType.SEC_10K
        assert SourceType("WEB") == SourceType.WEB

    def test_is_str(self):
        assert isinstance(SourceType.PDF, str)
        assert SourceType.PDF == "PDF"


class TestEdgeType:
    def test_members_exist(self):
        expected = [
            "PRODUCES", "REFINED_BY", "SUPPLIES_TO", "USED_BY",
            "LOCATED_IN", "DEPENDS_ON", "ADJACENT_TO", "EXPOSED_TO",
            "RISK_FLOWS_TO",
        ]
        for name in expected:
            assert hasattr(EdgeType, name)

    def test_string_roundtrip(self):
        assert EdgeType("PRODUCES") == EdgeType.PRODUCES


class TestNodeType:
    def test_members_exist(self):
        expected = [
            "COMPANY", "COUNTRY", "MINERAL", "REGION",
            "ASSET", "FACILITY", "CORRIDOR",
        ]
        for name in expected:
            assert hasattr(NodeType, name)

    def test_string_roundtrip(self):
        assert NodeType("COMPANY") == NodeType.COMPANY


class TestAssertionMode:
    def test_members_exist(self):
        for name in ["EXTRACTED", "INFERRED", "SEEDED"]:
            assert hasattr(AssertionMode, name)

    def test_string_value(self):
        assert AssertionMode.EXTRACTED == "EXTRACTED"


class TestConfidenceLevel:
    def test_members_exist(self):
        for name in ["HIGH", "MEDIUM", "LOW"]:
            assert hasattr(ConfidenceLevel, name)


class TestEvidenceType:
    def test_members_exist(self):
        for name in ["TEXT_QUOTE", "TABLE_CELL", "DERIVED", "MANUAL"]:
            assert hasattr(EvidenceType, name)

    def test_invalid_member_raises(self):
        with pytest.raises(ValueError):
            SourceType("NONEXISTENT")
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/minjun/graf/graphite && python -m pytest tests/test_enums.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/minjun/graf/graphite
git add tests/test_enums.py
git commit -m "test: add test_enums.py — enum member existence, string roundtrip, membership"
```

---

### Task 2: test_schemas.py

**Files:**
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write the test file**

```python
"""tests/test_schemas.py — Unit tests for graphite.schemas."""

import pytest
from graphite.schemas import (
    NodeRef, Provenance, InferenceBasis, ExtractedEdge, ExtractionError,
)
from graphite.enums import (
    NodeType, SourceType, EvidenceType, ConfidenceLevel, AssertionMode,
)


class TestNodeRefFactories:
    def test_company(self):
        n = NodeRef.company("AAPL", label="Apple Inc.")
        assert n.node_id == "company:AAPL"
        assert n.node_type == NodeType.COMPANY
        assert n.label == "Apple Inc."

    def test_company_uppercases(self):
        n = NodeRef.company("aapl")
        assert n.node_id == "company:AAPL"

    def test_country(self):
        n = NodeRef.country("CD", label="Congo")
        assert n.node_id == "country:CD"
        assert n.node_type == NodeType.COUNTRY

    def test_mineral(self):
        n = NodeRef.mineral("cobalt")
        assert n.node_id == "mineral:COBALT"
        assert n.node_type == NodeType.MINERAL

    def test_region(self):
        n = NodeRef.region("houston_metro")
        assert n.node_id == "region:HOUSTON_METRO"
        assert n.node_type == NodeType.REGION

    def test_asset(self):
        n = NodeRef.asset("port_houston")
        assert n.node_id == "asset:PORT_HOUSTON"
        assert n.node_type == NodeType.ASSET

    def test_facility(self):
        n = NodeRef.facility("exxon_baytown")
        assert n.node_id == "facility:EXXON_BAYTOWN"
        assert n.node_type == NodeType.FACILITY

    def test_corridor(self):
        n = NodeRef.corridor("ship_channel")
        assert n.node_id == "corridor:SHIP_CHANNEL"
        assert n.node_type == NodeType.CORRIDOR


class TestProvenance:
    def test_construction(self):
        p = Provenance(
            source_id="acc-001",
            source_type=SourceType.SEC_10K,
            evidence_quote="Apple is a customer.",
            confidence=ConfidenceLevel.HIGH,
        )
        assert p.source_id == "acc-001"
        assert p.source_type == SourceType.SEC_10K
        assert p.confidence == ConfidenceLevel.HIGH
        assert p.evidence_type == EvidenceType.TEXT_QUOTE  # default

    def test_defaults(self):
        p = Provenance(
            source_id="x",
            source_type=SourceType.WEB,
            evidence_quote="quote",
        )
        assert p.source_url == ""
        assert p.paragraph_index == -1
        assert p.extractor_version == "1.0"


class TestInferenceBasis:
    def test_construction(self):
        ib = InferenceBasis(
            method="customer_filing_reverse",
            reason="Inferred from TSLA 10-K",
            based_on_edges=["e1", "e2"],
            source_nodes=["company:TSLA"],
        )
        assert ib.method == "customer_filing_reverse"
        assert len(ib.based_on_edges) == 2
        assert ib.source_nodes == ["company:TSLA"]


class TestExtractedEdge:
    def _make_edge(self, provenance_levels=None):
        provs = []
        for level in (provenance_levels or [ConfidenceLevel.MEDIUM]):
            provs.append(Provenance(
                source_id=f"src-{level.value}",
                source_type=SourceType.SEC_10K,
                evidence_quote="quote",
                confidence=level,
            ))
        return ExtractedEdge(
            from_node=NodeRef.company("AAPL"),
            to_node=NodeRef.company("TSLA"),
            edge_type="SUPPLIES_TO",
            assertion_mode=AssertionMode.EXTRACTED,
            provenance=provs,
        )

    def test_edge_key_deterministic(self):
        e = self._make_edge()
        assert e.edge_key == "company:AAPL|company:TSLA|SUPPLIES_TO"

    def test_best_confidence_single(self):
        e = self._make_edge([ConfidenceLevel.LOW])
        assert e.best_confidence == ConfidenceLevel.LOW

    def test_best_confidence_multiple(self):
        e = self._make_edge([ConfidenceLevel.LOW, ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM])
        assert e.best_confidence == ConfidenceLevel.HIGH

    def test_best_confidence_no_provenance(self):
        e = ExtractedEdge(
            from_node=NodeRef.company("AAPL"),
            to_node=NodeRef.company("TSLA"),
            edge_type="SUPPLIES_TO",
            assertion_mode=AssertionMode.EXTRACTED,
        )
        assert e.best_confidence == ConfidenceLevel.LOW


class TestExtractionError:
    def test_construction(self):
        err = ExtractionError(
            entity_id="AAPL",
            source_type=SourceType.SEC_10K,
            error_type="parse_failed",
            message="Could not parse HTML",
        )
        assert err.entity_id == "AAPL"
        assert err.error_type == "parse_failed"
        assert err.timestamp  # auto-filled
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/minjun/graf/graphite && python -m pytest tests/test_schemas.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/minjun/graf/graphite
git add tests/test_schemas.py
git commit -m "test: add test_schemas.py — NodeRef factories, ExtractedEdge properties, Provenance"
```

---

### Task 3: test_evidence.py

**Files:**
- Create: `tests/test_evidence.py`

- [ ] **Step 1: Write the test file**

```python
"""tests/test_evidence.py — Unit tests for graphite.evidence."""

import pytest
from graphite.evidence import (
    EvidenceData, RuleResultModel, ScoringData,
    CounterEvidence, EvidencePacket,
)


class TestEvidenceData:
    def test_construction(self):
        ed = EvidenceData(
            source_entity="company:AAPL",
            target_entity="company:TSLA",
            edge_type="SUPPLIES_TO",
            quote="Apple supplies components to Tesla",
            doc_url="https://example.com/doc.html",
            quote_hash="abc123",
        )
        assert ed.source_entity == "company:AAPL"
        assert ed.edge_type == "SUPPLIES_TO"
        assert ed.quote_type == "exact_quote"  # default
        assert ed.temporal_status == "ACTIVE"  # default

    def test_optional_fields_default(self):
        ed = EvidenceData(
            source_entity="a", target_entity="b", edge_type="X",
            quote="q", doc_url="u", quote_hash="h",
        )
        assert ed.source_id == ""
        assert ed.filing_type == ""
        assert ed.section == ""


class TestRuleResultModel:
    def test_construction(self):
        rr = RuleResultModel(
            rule_id="R01",
            rule_name="Sole Source",
            triggered=True,
            weight_delta=0.15,
            explanation="Single supplier detected",
            category="supply_risk",
        )
        assert rr.triggered is True
        assert rr.weight_delta == 0.15


class TestScoringData:
    def test_construction(self):
        sd = ScoringData(
            policy_id="minerals_v1",
            applied_rules=["R01", "R02"],
            calculated_weight=0.75,
            base_score=0.5,
            final_score=0.65,
        )
        assert sd.policy_id == "minerals_v1"
        assert len(sd.applied_rules) == 2

    def test_with_rule_details(self):
        rr = RuleResultModel(
            rule_id="R01", rule_name="test", triggered=True,
            weight_delta=0.1, explanation="x",
        )
        sd = ScoringData(
            policy_id="p1", applied_rules=["R01"],
            calculated_weight=0.5, rule_details=[rr],
        )
        assert len(sd.rule_details) == 1
        assert sd.rule_details[0].rule_id == "R01"


class TestCounterEvidence:
    def test_construction(self):
        ce = CounterEvidence(
            quote="Tesla has multiple suppliers",
            doc_url="https://example.com",
            impact="WEAKENS_MONOPOLY",
        )
        assert ce.impact == "WEAKENS_MONOPOLY"
        assert ce.source_filing == ""  # default


class TestEvidencePacket:
    def test_construction_minimal(self):
        ep = EvidencePacket(
            claim_hash="hash123",
            status="SUPPORTED",
        )
        assert ep.graphite_version == "1.0"
        assert ep.status == "SUPPORTED"
        assert ep.evidence is None
        assert ep.counter_evidence == []

    def test_construction_full(self):
        ed = EvidenceData(
            source_entity="a", target_entity="b", edge_type="X",
            quote="q", doc_url="u", quote_hash="h",
        )
        ce = CounterEvidence(quote="counter", doc_url="u", impact="WEAKENS")
        ep = EvidencePacket(
            claim_hash="hash",
            status="MIXED_EVIDENCE",
            evidence=ed,
            counter_evidence=[ce],
            verdict_reason="Mixed signals",
        )
        assert ep.evidence.source_entity == "a"
        assert len(ep.counter_evidence) == 1

    def test_serialization_roundtrip(self):
        ep = EvidencePacket(claim_hash="h", status="SUPPORTED")
        data = ep.model_dump()
        restored = EvidencePacket.model_validate(data)
        assert restored.claim_hash == "h"
        assert restored.status == "SUPPORTED"
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/minjun/graf/graphite && python -m pytest tests/test_evidence.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/minjun/graf/graphite
git add tests/test_evidence.py
git commit -m "test: add test_evidence.py — EvidenceData, EvidencePacket, ScoringData, CounterEvidence"
```

---

### Task 4: test_text.py

**Files:**
- Create: `tests/test_text.py`

- [ ] **Step 1: Write the test file**

```python
"""tests/test_text.py — Unit tests for graphite.text."""

import pytest
from graphite.text import (
    sha1_hex, sha256_hex, normalize_text, clip_quote,
    split_into_paragraphs, score_paragraph,
    find_best_paragraph_for_quote,
    register_strategy, build_context, _strategies,
)


class TestHashFunctions:
    def test_sha1_deterministic(self):
        assert sha1_hex("hello") == sha1_hex("hello")
        assert len(sha1_hex("hello")) == 40

    def test_sha256_deterministic(self):
        assert sha256_hex("hello") == sha256_hex("hello")
        assert len(sha256_hex("hello")) == 64

    def test_sha1_differs_from_sha256(self):
        assert sha1_hex("hello") != sha256_hex("hello")


class TestNormalizeText:
    def test_collapses_newlines(self):
        assert normalize_text("a\n\n\n\nb") == "a\n\nb"

    def test_collapses_spaces(self):
        assert normalize_text("a   b") == "a b"

    def test_strips(self):
        assert normalize_text("  hello  ") == "hello"


class TestClipQuote:
    def test_short_unchanged(self):
        assert clip_quote("short text") == "short text"

    def test_truncates_with_ellipsis(self):
        result = clip_quote("a" * 300, max_chars=10)
        assert len(result) == 11  # 10 chars + ellipsis
        assert result.endswith("…")

    def test_strips_input(self):
        assert clip_quote("  hello  ") == "hello"


class TestSplitIntoParagraphs:
    def test_splits_on_double_newline(self):
        text = ("a" * 100) + "\n\n" + ("b" * 100)
        result = split_into_paragraphs(text)
        assert len(result) == 2

    def test_min_len_filters(self):
        text = "short\n\n" + ("long" * 30)
        result = split_into_paragraphs(text, min_len=80)
        assert len(result) == 1  # "short" filtered out

    def test_max_paras_limits(self):
        text = "\n\n".join(["x" * 100 for _ in range(10)])
        result = split_into_paragraphs(text, max_paras=3)
        assert len(result) == 3

    def test_empty_text(self):
        assert split_into_paragraphs("") == []


class TestScoreParagraph:
    def test_counts_keywords(self):
        assert score_paragraph("cobalt mining in Congo", ["cobalt", "congo"]) == 2

    def test_case_insensitive(self):
        assert score_paragraph("COBALT Mining", ["cobalt"]) == 1

    def test_no_match(self):
        assert score_paragraph("hello world", ["cobalt"]) == 0


class TestFindBestParagraphForQuote:
    def test_exact_substring_match(self):
        paras = ["First paragraph about nothing.", "Apple is a key supplier of components."]
        idx, hash_val = find_best_paragraph_for_quote(paras, "Apple is a key supplier")
        assert idx == 1
        assert len(hash_val) == 12

    def test_word_overlap_fallback(self):
        paras = ["Cobalt mining operations in Congo.", "Lithium reserves in Chile."]
        idx, _ = find_best_paragraph_for_quote(paras, "Congo cobalt production")
        assert idx == 0

    def test_empty_quote(self):
        idx, hash_val = find_best_paragraph_for_quote(["para"], "")
        assert idx == -1
        assert hash_val == ""

    def test_empty_paragraphs(self):
        idx, hash_val = find_best_paragraph_for_quote([], "some quote")
        assert idx == -1


class TestStrategyRegistry:
    def test_builtin_strategies_registered(self):
        assert "default" in _strategies
        assert "usgs_country_mineral" in _strategies
        assert "sec_minerals" in _strategies
        assert "sec_generic" in _strategies

    def test_register_custom_strategy(self):
        def my_strategy(paragraphs, **kwargs):
            return "custom"
        register_strategy("_test_custom", my_strategy)
        assert "_test_custom" in _strategies

    def test_build_context_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown context strategy"):
            build_context(None, strategy="nonexistent_strategy_xyz")

    def test_build_context_dispatches(self):
        class FakeDoc:
            paragraphs = ["para one " * 20, "para two " * 20]

        # The default strategy should return something non-empty
        result = build_context(FakeDoc(), strategy="default")
        assert isinstance(result, str)
        assert len(result) > 0
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/minjun/graf/graphite && python -m pytest tests/test_text.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/minjun/graf/graphite
git add tests/test_text.py
git commit -m "test: add test_text.py — hash functions, text utils, strategy registry"
```

---

### Task 5: test_rules.py

**Files:**
- Create: `tests/test_rules.py`

- [ ] **Step 1: Write the test file**

```python
"""tests/test_rules.py — Unit tests for graphite.rules."""

import pytest
from graphite.rules import RuleResult, ScoreBreakdown, BaseRuleEngine


class TestRuleResult:
    def test_fields(self):
        rr = RuleResult(
            rule_id="R01", rule_name="Sole Source",
            triggered=True, weight_delta=0.15,
            explanation="Single supplier", category="supply_risk",
        )
        assert rr.rule_id == "R01"
        assert rr.triggered is True
        assert rr.weight_delta == 0.15
        assert rr.category == "supply_risk"

    def test_default_category(self):
        rr = RuleResult(
            rule_id="R01", rule_name="X",
            triggered=False, weight_delta=0.0, explanation="N/A",
        )
        assert rr.category == ""


class TestScoreBreakdown:
    def _make_breakdown(self):
        rules = [
            RuleResult("R01", "Sole Source", True, 0.15, "yes", "risk"),
            RuleResult("R02", "Diversified", False, 0.0, "no", "risk"),
            RuleResult("R03", "Revenue Conc", True, 0.12, "yes", "revenue"),
        ]
        return ScoreBreakdown(
            base_score=0.5,
            rule_results=rules,
            final_score=0.77,
            raw_delta=0.27,
            applied_delta=0.27,
            confidence="HIGH",
            verdict="SUPPORTED",
            verdict_reason="Strong evidence",
        )

    def test_triggered_rules(self):
        bd = self._make_breakdown()
        triggered = bd.triggered_rules
        assert len(triggered) == 2
        assert all(r.triggered for r in triggered)

    def test_total_delta_returns_applied_delta(self):
        bd = self._make_breakdown()
        assert bd.total_delta == bd.applied_delta
        assert bd.total_delta == 0.27

    def test_to_dict(self):
        bd = self._make_breakdown()
        d = bd.to_dict()
        assert d["base_score"] == 0.5
        assert d["final_score"] == 0.77
        assert d["verdict"] == "SUPPORTED"
        assert d["confidence"] == "HIGH"
        assert d["policy_version"] == "v1"
        assert len(d["triggered_rules"]) == 2
        assert d["triggered_rules"][0]["rule_id"] == "R01"

    def test_to_dict_rounds_values(self):
        bd = ScoreBreakdown(
            base_score=0.123456789,
            final_score=0.987654321,
            raw_delta=0.111111,
            applied_delta=0.111111,
        )
        d = bd.to_dict()
        assert d["base_score"] == 0.1235
        assert d["final_score"] == 0.9877


class TestBaseRuleEngine:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseRuleEngine()

    def test_concrete_subclass(self):
        class MyEngine(BaseRuleEngine):
            def compute_score(self, edge_data, counter_signals=None):
                return ScoreBreakdown(base_score=1.0, final_score=1.0)

        engine = MyEngine()
        result = engine.compute_score({"edge_type": "PRODUCES"})
        assert isinstance(result, ScoreBreakdown)
        assert result.final_score == 1.0
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/minjun/graf/graphite && python -m pytest tests/test_rules.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/minjun/graf/graphite
git add tests/test_rules.py
git commit -m "test: add test_rules.py — RuleResult, ScoreBreakdown, BaseRuleEngine ABC"
```

---

### Task 6: test_domain.py

**Files:**
- Create: `tests/test_domain.py`

- [ ] **Step 1: Write the test file**

```python
"""tests/test_domain.py — Unit tests for graphite.domain."""

import pytest
import networkx as nx
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
                return nx.DiGraph()

        p = FakePipeline()
        g = p.run(["AAPL"], "/tmp/out.json")
        assert isinstance(g, nx.DiGraph)


class TestDomainRegistry:
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
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/minjun/graf/graphite && python -m pytest tests/test_domain.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/minjun/graf/graphite
git add tests/test_domain.py
git commit -m "test: add test_domain.py — ExtractionResult, DocumentContext, ABCs, registry"
```

---

### Task 7: test_cache.py

**Files:**
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write the test file**

```python
"""tests/test_cache.py — Unit tests for graphite.cache."""

import pytest
from graphite.cache import PipelineCache


class TestMakeKey:
    def test_deterministic(self):
        k1 = PipelineCache.make_key("src1", "hash1", "v1", "p1", "gemini")
        k2 = PipelineCache.make_key("src1", "hash1", "v1", "p1", "gemini")
        assert k1 == k2

    def test_different_inputs_different_keys(self):
        k1 = PipelineCache.make_key("src1", "hash1", "v1", "p1", "gemini")
        k2 = PipelineCache.make_key("src2", "hash1", "v1", "p1", "gemini")
        assert k1 != k2

    def test_key_length(self):
        k = PipelineCache.make_key("a", "b", "c", "d", "e")
        assert len(k) == 24


class TestContentHash:
    def test_deterministic(self):
        h1 = PipelineCache.content_hash("hello world")
        h2 = PipelineCache.content_hash("hello world")
        assert h1 == h2

    def test_length(self):
        h = PipelineCache.content_hash("test")
        assert len(h) == 16


class TestCacheOperations:
    def test_put_get_roundtrip(self, tmp_path):
        cache = PipelineCache(cache_dir=str(tmp_path / "cache"))
        cache.put("key1", {"result": "data", "count": 42})
        result = cache.get("key1")
        assert result == {"result": "data", "count": 42}

    def test_has_after_put(self, tmp_path):
        cache = PipelineCache(cache_dir=str(tmp_path / "cache"))
        assert cache.has("key1") is False
        cache.put("key1", {"x": 1})
        assert cache.has("key1") is True

    def test_get_missing_returns_none(self, tmp_path):
        cache = PipelineCache(cache_dir=str(tmp_path / "cache"))
        assert cache.get("nonexistent") is None

    def test_clear(self, tmp_path):
        cache = PipelineCache(cache_dir=str(tmp_path / "cache"))
        cache.put("k1", {"a": 1})
        cache.put("k2", {"b": 2})
        count = cache.clear()
        assert count == 2
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_clear_empty(self, tmp_path):
        cache = PipelineCache(cache_dir=str(tmp_path / "cache"))
        count = cache.clear()
        assert count == 0
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/minjun/graf/graphite && python -m pytest tests/test_cache.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/minjun/graf/graphite
git add tests/test_cache.py
git commit -m "test: add test_cache.py — make_key, content_hash, put/get/has/clear"
```

---

## Layer 2 — Storage

### Task 8: test_claim_store.py

**Files:**
- Create: `tests/test_claim_store.py`

- [ ] **Step 1: Write the test file**

```python
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
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/minjun/graf/graphite && python -m pytest tests/test_claim_store.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/minjun/graf/graphite
git add tests/test_claim_store.py
git commit -m "test: add test_claim_store.py — CRUD, evidence accumulation, dedup, search, overrides"
```

---

## Layer 3 — Pipeline + Adapters

### Task 9: test_llm.py

**Files:**
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write the test file**

```python
"""tests/test_llm.py — Unit tests for graphite.llm (mocked)."""

import json
import pytest
from unittest.mock import patch, MagicMock

from pydantic import BaseModel

# Mock google.genai before importing llm module
_mock_genai = MagicMock()
_mock_types = MagicMock()


@pytest.fixture(autouse=True)
def _patch_genai_and_reset():
    """Patch google.genai in sys.modules and reset llm module state."""
    import graphite.llm as llm_mod
    with patch.dict("sys.modules", {
        "google": MagicMock(),
        "google.genai": _mock_genai,
        "google.genai.types": _mock_types,
    }):
        llm_mod._client = None
        llm_mod._last_call = 0.0
        yield llm_mod


class SampleSchema(BaseModel):
    name: str
    value: int


class TestGeminiClient:
    def test_missing_api_key_raises(self, _patch_genai_and_reset):
        llm_mod = _patch_genai_and_reset
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False):
            with pytest.raises(RuntimeError, match="GEMINI_API_KEY not set"):
                llm_mod._init_client()

    def test_lazy_initialization(self, _patch_genai_and_reset):
        llm_mod = _patch_genai_and_reset
        mock_client = MagicMock()
        _mock_genai.Client.return_value = mock_client

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            result = llm_mod.get_gemini_client()
            assert result is mock_client
            # Second call returns cached client
            result2 = llm_mod.get_gemini_client()
            assert result2 is mock_client


class TestGeminiExtractStructured:
    def test_parses_response(self, _patch_genai_and_reset):
        llm_mod = _patch_genai_and_reset
        mock_response = MagicMock()
        mock_response.text = '{"name": "test", "value": 42}'

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        llm_mod._client = mock_client

        result = llm_mod.gemini_extract_structured(
            contents="test doc", system_prompt="extract", schema=SampleSchema,
        )
        assert result.name == "test"
        assert result.value == 42

    def test_strips_markdown_fences(self, _patch_genai_and_reset):
        llm_mod = _patch_genai_and_reset
        mock_response = MagicMock()
        mock_response.text = '```json\n{"name": "fenced", "value": 99}\n```'

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        llm_mod._client = mock_client

        result = llm_mod.gemini_extract_structured(
            contents="test", system_prompt="extract", schema=SampleSchema,
        )
        assert result.name == "fenced"
        assert result.value == 99

    def test_retry_on_failure(self, _patch_genai_and_reset):
        llm_mod = _patch_genai_and_reset
        good_response = MagicMock()
        good_response.text = '{"name": "retry", "value": 1}'

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            Exception("transient error"),
            good_response,
        ]
        llm_mod._client = mock_client

        with patch("time.sleep"):  # skip retry/rate-limit sleeps
            result = llm_mod.gemini_extract_structured(
                contents="test", system_prompt="extract",
                schema=SampleSchema, max_retries=3,
            )
            assert result.name == "retry"


class TestGeminiExtractJson:
    def test_returns_dict(self, _patch_genai_and_reset):
        llm_mod = _patch_genai_and_reset
        mock_response = MagicMock()
        mock_response.text = '{"key": "value", "num": 123}'

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        llm_mod._client = mock_client

        result = llm_mod.gemini_extract_json(contents="test", system_prompt="extract")
        assert result == {"key": "value", "num": 123}

    def test_strips_markdown_fences(self, _patch_genai_and_reset):
        llm_mod = _patch_genai_and_reset
        mock_response = MagicMock()
        mock_response.text = '```json\n{"fenced": true}\n```'

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        llm_mod._client = mock_client

        result = llm_mod.gemini_extract_json(contents="test", system_prompt="extract")
        assert result == {"fenced": True}
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/minjun/graf/graphite && python -m pytest tests/test_llm.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/minjun/graf/graphite
git add tests/test_llm.py
git commit -m "test: add test_llm.py — mocked Gemini client, structured/json extraction, retries"
```

---

### Task 10: test_alphaearth.py

**Files:**
- Create: `tests/test_alphaearth.py`

- [ ] **Step 1: Write the test file**

```python
"""tests/test_alphaearth.py — Unit tests for graphite.adapters.alphaearth."""

import pytest
import numpy as np
from graphite.adapters.alphaearth import AlphaEarthAdapter, EMBEDDING_DIM


class TestCacheRoundtrip:
    def test_write_and_read(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)

        adapter._write_cache("test_node", 2017, emb)
        result = adapter._read_cache("test_node", 2017)

        assert result is not None
        np.testing.assert_array_almost_equal(result, emb)

    def test_read_missing_returns_none(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        assert adapter._read_cache("missing", 2017) is None


class TestGetEmbedding:
    def test_returns_cached(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        adapter._write_cache("PORT_HOUSTON", 2017, emb)

        result = adapter.get_embedding(29.7, -95.2, year=2017, node_id="PORT_HOUSTON")
        np.testing.assert_array_almost_equal(result, emb)

    def test_raises_when_not_cached(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        with pytest.raises(FileNotFoundError, match="No AlphaEarth embedding"):
            adapter.get_embedding(0.0, 0.0, year=2017)


class TestGetEmbeddingSafe:
    def test_returns_none_when_not_cached(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        result = adapter.get_embedding_safe(0.0, 0.0, year=2017)
        assert result is None

    def test_returns_embedding_when_cached(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        adapter._write_cache("test", 2017, emb)
        result = adapter.get_embedding_safe(0.0, 0.0, year=2017, node_id="test")
        assert result is not None


class TestGetAreaEmbedding:
    def test_returns_cached_bbox_embedding(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        bbox = (29.0, -96.0, 30.0, -95.0)
        cache_key = f"bbox_{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}"
        adapter._write_cache(cache_key, 2017, emb)

        result = adapter.get_area_embedding(bbox, year=2017)
        np.testing.assert_array_almost_equal(result, emb)

    def test_falls_back_to_get_embedding(self, tmp_path):
        """When bbox not cached, get_area_embedding calls get_embedding with bbox key."""
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        bbox = (29.0, -96.0, 30.0, -95.0)
        # No cache at all — should raise FileNotFoundError from get_embedding
        with pytest.raises(FileNotFoundError):
            adapter.get_area_embedding(bbox, year=2017)


class TestListCachedAndStats:
    def test_list_cached(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        adapter._write_cache("node_a", 2017, emb)
        adapter._write_cache("node_b", 2017, emb)

        cached = adapter.list_cached(2017)
        assert set(cached) == {"node_a", "node_b"}

    def test_list_cached_empty_year(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        assert adapter.list_cached(2020) == []

    def test_cache_stats(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        adapter._write_cache("a", 2017, emb)
        adapter._write_cache("b", 2017, emb)
        adapter._write_cache("c", 2018, emb)

        stats = adapter.cache_stats()
        assert stats["2017"] == 2
        assert stats["2018"] == 1
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/minjun/graf/graphite && python -m pytest tests/test_alphaearth.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/minjun/graf/graphite
git add tests/test_alphaearth.py
git commit -m "test: add test_alphaearth.py — cache roundtrip, embeddings, stats"
```

---

### Task 11: test_weathernext.py

**Files:**
- Modify: `src/graphite/adapters/weathernext.py` (add `self._meta = None` to `__init__`)
- Create: `tests/test_weathernext.py`
- Create: `tests/fixtures/weathernext_snapshot.json`

- [ ] **Step 0: Fix source bug — `_meta` not initialized when no snapshot**

In `src/graphite/adapters/weathernext.py`, add `self._meta = None` in `__init__` after `self._data = None`:

```python
self._data = None
self._meta = None  # <-- add this line
```

Without this, `adapter.meta` raises `AttributeError` when no snapshot is loaded.

- [ ] **Step 1: Create the fixture file**

```json
{
    "meta": {
        "model": "WeatherNext2",
        "init_time": "2024-07-06T00:00:00Z",
        "resolution_deg": 0.25
    },
    "forecast_points": [
        {
            "node_id": "asset:PORT_HOUSTON",
            "lat": 29.7355,
            "lon": -95.2690,
            "max_wind_kph": 145,
            "precip_mm_72h": 380,
            "storm_surge_m": 2.1
        },
        {
            "node_id": "facility:EXXON_BAYTOWN",
            "lat": 29.7356,
            "lon": -95.0138,
            "max_wind_kph": 130,
            "precip_mm_72h": 320,
            "storm_surge_m": 1.5
        }
    ]
}
```

- [ ] **Step 2: Write the test file**

```python
"""tests/test_weathernext.py — Unit tests for graphite.adapters.weathernext."""

import os
import pytest
from graphite.adapters.weathernext import WeatherNextAdapter


FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "weathernext_snapshot.json")


class TestLoadSnapshot:
    def test_load_from_fixture(self):
        adapter = WeatherNextAdapter(snapshot_path=FIXTURE_PATH)
        assert adapter._data is not None
        assert len(adapter._data) == 2

    def test_no_snapshot(self):
        adapter = WeatherNextAdapter()
        assert adapter._data is None


class TestGetForecast:
    def test_known_node(self):
        adapter = WeatherNextAdapter(snapshot_path=FIXTURE_PATH)
        forecast = adapter.get_forecast("asset:PORT_HOUSTON")
        assert forecast is not None
        assert forecast["max_wind_kph"] == 145

    def test_unknown_node(self):
        adapter = WeatherNextAdapter(snapshot_path=FIXTURE_PATH)
        assert adapter.get_forecast("asset:NONEXISTENT") is None


class TestGetAllForecasts:
    def test_returns_all(self):
        adapter = WeatherNextAdapter(snapshot_path=FIXTURE_PATH)
        all_fc = adapter.get_all_forecasts()
        assert len(all_fc) == 2
        assert "asset:PORT_HOUSTON" in all_fc
        assert "facility:EXXON_BAYTOWN" in all_fc

    def test_empty_when_no_snapshot(self):
        adapter = WeatherNextAdapter()
        assert adapter.get_all_forecasts() == {}


class TestListNodes:
    def test_lists_all_ids(self):
        adapter = WeatherNextAdapter(snapshot_path=FIXTURE_PATH)
        nodes = adapter.list_nodes()
        assert set(nodes) == {"asset:PORT_HOUSTON", "facility:EXXON_BAYTOWN"}

    def test_empty_when_no_snapshot(self):
        adapter = WeatherNextAdapter()
        assert adapter.list_nodes() == []


class TestMeta:
    def test_returns_metadata(self):
        adapter = WeatherNextAdapter(snapshot_path=FIXTURE_PATH)
        meta = adapter.meta
        assert meta["model"] == "WeatherNext2"
        assert meta["resolution_deg"] == 0.25

    def test_empty_when_no_snapshot(self):
        adapter = WeatherNextAdapter()
        assert adapter.meta == {}
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/minjun/graf/graphite && python -m pytest tests/test_weathernext.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
cd /Users/minjun/graf/graphite
git add tests/fixtures/weathernext_snapshot.json tests/test_weathernext.py
git commit -m "test: add test_weathernext.py — snapshot loading, forecast retrieval, metadata"
```

---

### Task 12: test_retriever.py

**Files:**
- Create: `tests/test_retriever.py`

- [ ] **Step 1: Write the test file**

```python
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
        # First chunk should mention apple/tesla
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

        # Claim with no entity prefixes — triggers claim_text fallback
        claim = _make_claim(
            subjects=[], objects=[],
            claim_text="cobalt mining congo",
        )
        results = retriever.retrieve_evidence([claim])
        assert len(results[claim.claim_id]) >= 1


class TestRetrieveEvidenceConvenience:
    def test_convenience_function(self):
        docs = [{"document_id": "d1", "text": "Apple supplies Tesla with chips."}]
        claim = _make_claim(subjects=["company:APPLE"], objects=["company:TESLA"])
        results = retrieve_evidence([claim], docs)
        assert claim.claim_id in results
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/minjun/graf/graphite && python -m pytest tests/test_retriever.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/minjun/graf/graphite
git add tests/test_retriever.py
git commit -m "test: add test_retriever.py — DocumentCorpus, EvidenceRetriever, lexical scoring"
```

---

### Task 13: test_extractor.py

**Files:**
- Create: `tests/test_extractor.py`

- [ ] **Step 1: Write the test file**

```python
"""tests/test_extractor.py — Unit tests for graphite.pipeline.extractor (mocked)."""

import json
import pytest
from unittest.mock import patch, MagicMock

from graphite.pipeline.extractor import ClaimExtractor, extract_claims


def _mock_openai_response(claims_data):
    """Build a mock OpenAI chat completion response."""
    mock_message = MagicMock()
    mock_message.content = json.dumps({"claims": claims_data})
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


class TestClaimExtractor:
    @patch("openai.OpenAI")
    def test_extract_claims_parses_response(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        claims_data = [
            {
                "claim_text": "Apple supplies chips to Tesla",
                "subject_entities": ["Apple"],
                "predicate": "SUPPLIES_TO",
                "object_entities": ["Tesla"],
            },
        ]
        mock_client.chat.completions.create.return_value = _mock_openai_response(claims_data)

        extractor = ClaimExtractor(api_key="test-key")
        result = extractor.extract_claims("Apple supplies chips to Tesla")

        assert len(result) == 1
        assert result[0].claim_text == "Apple supplies chips to Tesla"
        assert result[0].subject_entities == ["Apple"]
        assert result[0].predicate == "SUPPLIES_TO"
        assert result[0].object_entities == ["Tesla"]
        assert result[0].claim_id  # deterministic hash exists

    @patch("openai.OpenAI")
    def test_extract_multiple_claims(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        claims_data = [
            {"claim_text": "C1", "subject_entities": ["A"], "predicate": "P1", "object_entities": ["B"]},
            {"claim_text": "C2", "subject_entities": ["C"], "predicate": "P2", "object_entities": ["D"]},
        ]
        mock_client.chat.completions.create.return_value = _mock_openai_response(claims_data)

        extractor = ClaimExtractor(api_key="test-key")
        result = extractor.extract_claims("some document")
        assert len(result) == 2

    @patch("openai.OpenAI")
    def test_empty_document_returns_empty(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response([])

        extractor = ClaimExtractor(api_key="test-key")
        result = extractor.extract_claims("")
        assert result == []


class TestExtractClaimsConvenience:
    @patch("openai.OpenAI")
    def test_convenience_function(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        claims_data = [
            {"claim_text": "X", "subject_entities": ["A"], "predicate": "P", "object_entities": ["B"]},
        ]
        mock_client.chat.completions.create.return_value = _mock_openai_response(claims_data)

        result = extract_claims("doc text", api_key="test-key")
        assert len(result) == 1
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/minjun/graf/graphite && python -m pytest tests/test_extractor.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/minjun/graf/graphite
git add tests/test_extractor.py
git commit -m "test: add test_extractor.py — mocked LLM extraction, parsing, convenience function"
```

---

### Task 14: test_verifier.py

**Files:**
- Create: `tests/test_verifier.py`

- [ ] **Step 1: Write the test file**

```python
"""tests/test_verifier.py — Unit tests for graphite.pipeline.verifier (mocked)."""

import json
import pytest
from unittest.mock import patch, MagicMock

from graphite.pipeline.verifier import ClaimVerifier, verify_claims
from graphite.claim import Claim, ClaimType, ClaimOrigin, ClaimGranularity, VerdictEnum
from graphite.enums import AssertionMode


def _make_claim(claim_text="Test claim", subjects=None, objects=None):
    return Claim(
        claim_text=claim_text,
        claim_type=ClaimType.RELATIONSHIP,
        subject_entities=subjects or ["company:AAPL"],
        predicate="SUPPLIES_TO",
        object_entities=objects or ["company:TSLA"],
        assertion_mode=AssertionMode.EXTRACTED,
        origin=ClaimOrigin.EXTRACTOR,
        granularity=ClaimGranularity.ATOMIC,
    )


def _mock_verifier_response(verdict="SUPPORTED", rationale="Evidence supports"):
    mock_message = MagicMock()
    mock_message.content = json.dumps({
        "verdict": verdict,
        "rationale_text": rationale,
        "contradiction_type": None,
        "missing_evidence_reason": None,
        "temporal_alignment": None,
        "needs_human_review": False,
        "cited_span": "relevant quote",
        "supporting_evidence_indices": [0] if verdict == "SUPPORTED" else [],
        "conflicting_evidence_indices": [0] if verdict == "CONFLICTED" else [],
    })
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


class TestClaimVerifier:
    @patch("openai.OpenAI")
    def test_supported_verdict(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_verifier_response("SUPPORTED")

        verifier = ClaimVerifier(api_key="test-key")
        claim = _make_claim()
        evidence_map = {claim.claim_id: [{"text": "Apple supplies Tesla", "document_id": "doc1"}]}
        verdicts = verifier.verify_claims([claim], evidence_map)

        assert len(verdicts) == 1
        assert verdicts[0].verdict == VerdictEnum.SUPPORTED
        assert verdicts[0].claim_id == claim.claim_id

    @patch("openai.OpenAI")
    def test_conflicted_verdict(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_verifier_response("CONFLICTED")

        verifier = ClaimVerifier(api_key="test-key")
        claim = _make_claim()
        evidence_map = {claim.claim_id: [{"text": "Contradictory evidence", "document_id": "doc1"}]}
        verdicts = verifier.verify_claims([claim], evidence_map)

        assert verdicts[0].verdict == VerdictEnum.CONFLICTED

    @patch("openai.OpenAI")
    def test_insufficient_verdict(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_verifier_response("INSUFFICIENT")

        verifier = ClaimVerifier(api_key="test-key")
        claim = _make_claim()
        evidence_map = {claim.claim_id: []}
        verdicts = verifier.verify_claims([claim], evidence_map)

        assert verdicts[0].verdict == VerdictEnum.INSUFFICIENT

    @patch("openai.OpenAI")
    def test_empty_evidence_map(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_verifier_response("INSUFFICIENT")

        verifier = ClaimVerifier(api_key="test-key")
        claim = _make_claim()
        verdicts = verifier.verify_claims([claim], {})

        assert len(verdicts) == 1
        assert verdicts[0].verdict == VerdictEnum.INSUFFICIENT


class TestVerifyClaimsConvenience:
    @patch("openai.OpenAI")
    def test_convenience_function(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_verifier_response("SUPPORTED")

        claim = _make_claim()
        evidence_map = {claim.claim_id: [{"text": "evidence", "document_id": "doc1"}]}
        verdicts = verify_claims([claim], evidence_map, api_key="test-key")
        assert len(verdicts) == 1
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/minjun/graf/graphite && python -m pytest tests/test_verifier.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/minjun/graf/graphite
git add tests/test_verifier.py
git commit -m "test: add test_verifier.py — mocked SUPPORTED/CONFLICTED/INSUFFICIENT verdicts"
```

---

### Task 15: test_analyzer.py

**Files:**
- Create: `tests/test_analyzer.py`

- [ ] **Step 1: Write the test file**

```python
"""tests/test_analyzer.py — Unit tests for graphite.pipeline.analyzer (mocked)."""

import json
import pytest
from unittest.mock import patch, MagicMock

from graphite.pipeline.analyzer import ArgumentAnalyzer, analyze_argument_chain
from graphite.claim import (
    Verdict, VerdictEnum, VerdictRationale,
    ArgumentVerdictEnum,
)


def _make_verdict(claim_text="test claim", verdict=VerdictEnum.SUPPORTED):
    return Verdict(
        claim_id="claim-001",
        claim_text=claim_text,
        verdict=verdict,
        supporting_evidence_ids=[],
        conflicting_evidence_ids=[],
        rationale=VerdictRationale(text="test rationale"),
        needs_human_review=False,
        model_version="test",
        timestamp="2026-01-01T00:00:00Z",
    )


def _mock_analyzer_response(argument_verdicts):
    mock_message = MagicMock()
    mock_message.content = json.dumps({"argument_verdicts": argument_verdicts})
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


class TestArgumentAnalyzer:
    @patch("openai.OpenAI")
    def test_grounded_result(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_analyzer_response([
            {"text": "The conclusion follows", "verdict": "GROUNDED",
             "rationale_text": "Evidence supports", "contradiction_type": None,
             "needs_human_review": False},
        ])

        analyzer = ArgumentAnalyzer(api_key="test-key")
        verdicts = [_make_verdict()]
        results = analyzer.analyze_argument_chain("The memo text", verdicts)

        assert len(results) == 1
        assert results[0].verdict == ArgumentVerdictEnum.GROUNDED

    @patch("openai.OpenAI")
    def test_conclusion_jump(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_analyzer_response([
            {"text": "Unsupported conclusion", "verdict": "CONCLUSION_JUMP",
             "rationale_text": "Leap in logic", "contradiction_type": None,
             "needs_human_review": True},
        ])

        analyzer = ArgumentAnalyzer(api_key="test-key")
        results = analyzer.analyze_argument_chain("memo", [_make_verdict()])

        assert results[0].verdict == ArgumentVerdictEnum.CONCLUSION_JUMP
        assert results[0].needs_human_review is True

    @patch("openai.OpenAI")
    def test_overstated(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_analyzer_response([
            {"text": "Overstated claim", "verdict": "OVERSTATED",
             "rationale_text": "Exaggerated", "contradiction_type": None,
             "needs_human_review": False},
        ])

        analyzer = ArgumentAnalyzer(api_key="test-key")
        results = analyzer.analyze_argument_chain("memo", [_make_verdict()])

        assert results[0].verdict == ArgumentVerdictEnum.OVERSTATED

    @patch("openai.OpenAI")
    def test_empty_verdicts(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_analyzer_response([])

        analyzer = ArgumentAnalyzer(api_key="test-key")
        results = analyzer.analyze_argument_chain("memo", [])
        assert results == []


class TestAnalyzeConvenience:
    @patch("openai.OpenAI")
    def test_convenience_function(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_analyzer_response([
            {"text": "ok", "verdict": "GROUNDED", "rationale_text": "fine",
             "contradiction_type": None, "needs_human_review": False},
        ])

        results = analyze_argument_chain("memo", [_make_verdict()], api_key="test-key")
        assert len(results) == 1
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/minjun/graf/graphite && python -m pytest tests/test_analyzer.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/minjun/graf/graphite
git add tests/test_analyzer.py
git commit -m "test: add test_analyzer.py — mocked GROUNDED/CONCLUSION_JUMP/OVERSTATED analysis"
```

---

### Task 16: test_report.py

**Files:**
- Create: `tests/test_report.py`

- [ ] **Step 1: Write the test file**

```python
"""tests/test_report.py — Unit tests for graphite.pipeline.report (mocked)."""

import pytest
from unittest.mock import patch, MagicMock

from graphite.pipeline.report import verify_agent_output, review_document
from graphite.claim import (
    Claim, ClaimType, ClaimOrigin, ClaimGranularity,
    Verdict, VerdictEnum, VerdictRationale,
    ArgumentVerdict, ArgumentVerdictEnum,
    VerificationReport,
)
from graphite.enums import AssertionMode


def _make_claim(text="test"):
    return Claim(
        claim_text=text,
        claim_type=ClaimType.RELATIONSHIP,
        subject_entities=["A"],
        predicate="P",
        object_entities=["B"],
        assertion_mode=AssertionMode.EXTRACTED,
        origin=ClaimOrigin.EXTRACTOR,
        granularity=ClaimGranularity.ATOMIC,
    )


def _make_verdict(claim, verdict_enum=VerdictEnum.SUPPORTED, needs_review=False):
    return Verdict(
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        verdict=verdict_enum,
        supporting_evidence_ids=[],
        conflicting_evidence_ids=[],
        rationale=VerdictRationale(text="rationale"),
        needs_human_review=needs_review,
        model_version="test",
        timestamp="2026-01-01T00:00:00Z",
    )


def _make_arg_verdict(verdict_enum=ArgumentVerdictEnum.GROUNDED):
    return ArgumentVerdict(
        text="conclusion",
        verdict=verdict_enum,
        rationale=VerdictRationale(text="ok"),
        needs_human_review=False,
    )


class TestVerifyAgentOutput:
    @patch("graphite.pipeline.report.analyze_argument_chain")
    @patch("graphite.pipeline.report.verify_claims")
    @patch("graphite.pipeline.report.retrieve_evidence")
    @patch("graphite.pipeline.report.extract_claims")
    def test_orchestrates_pipeline(self, mock_extract, mock_retrieve, mock_verify, mock_analyze):
        claim = _make_claim("Apple supplies Tesla")
        mock_extract.return_value = [claim]
        mock_retrieve.return_value = {claim.claim_id: [{"text": "ev", "document_id": "d1"}]}
        mock_verify.return_value = [_make_verdict(claim, VerdictEnum.SUPPORTED)]
        mock_analyze.return_value = [_make_arg_verdict(ArgumentVerdictEnum.GROUNDED)]

        corpus = [{"document_id": "d1", "text": "Apple supplies Tesla with chips."}]
        report = verify_agent_output("Apple supplies Tesla", corpus, api_key="test-key")

        assert isinstance(report, VerificationReport)
        assert report.total_claims == 1
        assert report.supported_count == 1
        assert report.conflicted_count == 0
        assert report.insufficient_count == 0
        assert report.grounded_argument_count == 1
        assert report.evidence_coverage_score == 1.0

    @patch("graphite.pipeline.report.analyze_argument_chain")
    @patch("graphite.pipeline.report.verify_claims")
    @patch("graphite.pipeline.report.retrieve_evidence")
    @patch("graphite.pipeline.report.extract_claims")
    def test_conflicted_claim_in_risky(self, mock_extract, mock_retrieve, mock_verify, mock_analyze):
        claim = _make_claim()
        mock_extract.return_value = [claim]
        mock_retrieve.return_value = {claim.claim_id: []}
        mock_verify.return_value = [_make_verdict(claim, VerdictEnum.CONFLICTED)]
        mock_analyze.return_value = []

        report = verify_agent_output("text", [{"document_id": "d1", "text": "x"}], api_key="test-key")
        assert report.conflicted_count == 1
        assert claim.claim_id in report.risky_claim_ids

    @patch("graphite.pipeline.report.analyze_argument_chain")
    @patch("graphite.pipeline.report.verify_claims")
    @patch("graphite.pipeline.report.retrieve_evidence")
    @patch("graphite.pipeline.report.extract_claims")
    def test_conclusion_jump_counted(self, mock_extract, mock_retrieve, mock_verify, mock_analyze):
        claim = _make_claim()
        mock_extract.return_value = [claim]
        mock_retrieve.return_value = {}
        mock_verify.return_value = [_make_verdict(claim)]
        mock_analyze.return_value = [_make_arg_verdict(ArgumentVerdictEnum.CONCLUSION_JUMP)]

        report = verify_agent_output("text", [], api_key="test-key")
        assert report.conclusion_jump_count == 1


class TestReviewDocumentAlias:
    def test_is_same_function(self):
        assert review_document is verify_agent_output


class TestVerificationReportGetVerdict:
    def test_get_existing_verdict(self):
        claim = _make_claim()
        v = _make_verdict(claim)
        report = VerificationReport(
            document_id="doc1",
            total_claims=1,
            verdicts=[v],
            argument_verdicts=[],
            model_metadata={},
        )
        result = report.get_verdict(claim.claim_id)
        assert result is not None
        assert result.claim_id == claim.claim_id

    def test_get_missing_verdict(self):
        report = VerificationReport(
            document_id="doc1",
            total_claims=0,
            verdicts=[],
            argument_verdicts=[],
            model_metadata={},
        )
        assert report.get_verdict("nonexistent") is None
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/minjun/graf/graphite && python -m pytest tests/test_report.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/minjun/graf/graphite
git add tests/test_report.py
git commit -m "test: add test_report.py — mocked hero API, report aggregation, alias"
```

---

### Task 17: test_init.py

**Files:**
- Create: `tests/test_init.py`

- [ ] **Step 1: Write the test file**

```python
"""tests/test_init.py — Smoke test for public symbol re-exports."""

import pytest


class TestPublicExports:
    def test_core_schemas(self):
        from graphite import ExtractedEdge, NodeRef, Provenance, InferenceBasis, ExtractionError
        assert all([ExtractedEdge, NodeRef, Provenance, InferenceBasis, ExtractionError])

    def test_enums(self):
        from graphite import EdgeType, NodeType, SourceType, ConfidenceLevel, AssertionMode, EvidenceType
        assert all([EdgeType, NodeType, SourceType, ConfidenceLevel, AssertionMode, EvidenceType])

    def test_evidence(self):
        from graphite import EvidencePacket, EvidenceData
        assert all([EvidencePacket, EvidenceData])

    def test_claim_types(self):
        from graphite import Claim, ClaimType, ClaimStatus, ClaimGranularity, ReviewState, ClaimOrigin
        assert all([Claim, ClaimType, ClaimStatus, ClaimGranularity, ReviewState, ClaimOrigin])

    def test_confidence(self):
        from graphite import ConfidenceFactor, ConfidenceResult, ConfidenceScorer
        assert all([ConfidenceFactor, ConfidenceResult, ConfidenceScorer])

    def test_store(self):
        from graphite import ClaimStore
        assert ClaimStore

    def test_domain(self):
        from graphite import (
            BaseFetcher, BaseExtractor, BasePipeline,
            DocumentContext, DomainSpec,
            register_domain, get_domain, list_domains,
        )
        assert all([BaseFetcher, BaseExtractor, BasePipeline, DocumentContext, DomainSpec])

    def test_rules(self):
        from graphite import BaseRuleEngine, RuleResult, ScoreBreakdown
        assert all([BaseRuleEngine, RuleResult, ScoreBreakdown])

    def test_version(self):
        from graphite import __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/minjun/graf/graphite && python -m pytest tests/test_init.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/minjun/graf/graphite
git add tests/test_init.py
git commit -m "test: add test_init.py — smoke test for all public symbol re-exports"
```

---

## Final Validation

### Task 18: Run full test suite

- [ ] **Step 1: Run all tests**

Run: `cd /Users/minjun/graf/graphite && python -m pytest tests/ -v`
Expected: All PASS, 0 failures

- [ ] **Step 2: Verify all test files exist**

Expected test files (16 new + 2 existing):
- `tests/test_enums.py`
- `tests/test_schemas.py`
- `tests/test_evidence.py`
- `tests/test_text.py`
- `tests/test_rules.py`
- `tests/test_domain.py`
- `tests/test_cache.py`
- `tests/test_claim_store.py`
- `tests/test_llm.py`
- `tests/test_alphaearth.py`
- `tests/test_weathernext.py`
- `tests/test_retriever.py`
- `tests/test_extractor.py`
- `tests/test_verifier.py`
- `tests/test_analyzer.py`
- `tests/test_report.py`
- `tests/test_init.py`
- `tests/test_claim.py` (existing)
- `tests/test_confidence.py` (existing)
- `tests/fixtures/weathernext_snapshot.json`
