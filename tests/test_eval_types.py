"""Tests for graphite.eval.types."""

import pytest
from graphite.eval.types import EvalCase, EvalResult, EvalRun


class TestEvalCase:
    def test_construction(self):
        case = EvalCase(
            id="test_1",
            memo="Company X uses AWS.",
            corpus=[{"document_id": "doc1", "text": "X uses GCP."}],
            expected_claim_verdicts=["CONFLICTED"],
        )
        assert case.id == "test_1"
        assert len(case.corpus) == 1

    def test_optional_fields(self):
        case = EvalCase(
            id="t",
            memo="m",
            corpus=[],
            expected_claim_verdicts=["SUPPORTED"],
        )
        assert case.expected_argument_verdicts == []
        assert case.domain == "general"
        assert case.tags == []

    def test_serialization_roundtrip(self):
        case = EvalCase(
            id="t", memo="m", corpus=[{"document_id": "d", "text": "t"}],
            expected_claim_verdicts=["SUPPORTED"],
            expected_argument_verdicts=["GROUNDED"],
            domain="medical",
            tags=["drug_interaction"],
        )
        data = case.model_dump()
        restored = EvalCase.model_validate(data)
        assert restored == case


class TestEvalResult:
    def test_pass(self):
        result = EvalResult(
            case_id="t1",
            expected_claim_verdicts=["CONFLICTED"],
            actual_claim_verdicts=["CONFLICTED"],
            claim_verdict_pass=True,
            expected_argument_verdicts=["GROUNDED"],
            actual_argument_verdicts=["GROUNDED"],
            argument_verdict_pass=True,
        )
        assert result.passed is True

    def test_fail(self):
        result = EvalResult(
            case_id="t1",
            expected_claim_verdicts=["CONFLICTED"],
            actual_claim_verdicts=["SUPPORTED"],
            claim_verdict_pass=False,
        )
        assert result.passed is False

    def test_latency_optional(self):
        result = EvalResult(
            case_id="t1",
            expected_claim_verdicts=["X"],
            actual_claim_verdicts=["Y"],
            claim_verdict_pass=False,
            latency_ms=1234.5,
        )
        assert result.latency_ms == 1234.5


class TestEvalRun:
    def _make_results(self):
        return [
            EvalResult(case_id="t1", expected_claim_verdicts=["CONFLICTED"],
                       actual_claim_verdicts=["CONFLICTED"], claim_verdict_pass=True,
                       expected_argument_verdicts=["GROUNDED"],
                       actual_argument_verdicts=["GROUNDED"], argument_verdict_pass=True),
            EvalResult(case_id="t2", expected_claim_verdicts=["SUPPORTED"],
                       actual_claim_verdicts=["INSUFFICIENT"], claim_verdict_pass=False),
        ]

    def test_aggregate_metrics(self):
        run = EvalRun(
            model="gpt-4o",
            dataset="base",
            results=self._make_results(),
        )
        metrics = run.metrics()
        assert metrics["total"] == 2
        assert metrics["claim_verdict_accuracy"] == 0.5
        assert metrics["overall_pass_rate"] == 0.5

    def test_save_and_load(self, tmp_path):
        run = EvalRun(model="gpt-4o", dataset="base", results=self._make_results())
        path = tmp_path / "run.json"
        run.save(str(path))

        loaded = EvalRun.load(str(path))
        assert loaded.model == "gpt-4o"
        assert len(loaded.results) == 2
        assert loaded.metrics() == run.metrics()
