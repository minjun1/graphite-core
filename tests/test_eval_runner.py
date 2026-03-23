"""Tests for graphite.eval.runner."""

import json
import pytest
from unittest.mock import patch, MagicMock

from graphite.eval.runner import EvalRunner
from graphite.eval.types import EvalCase, EvalRun
from graphite.pipeline.verdict import (
    VerdictEnum, ArgumentVerdictEnum, VerdictRationale,
    Verdict, ArgumentVerdict, VerificationReport,
)


def _make_report(claim_verdict="CONFLICTED", arg_verdict="GROUNDED"):
    v = Verdict(
        claim_id="c1", claim_text="test",
        verdict=VerdictEnum(claim_verdict),
        rationale=VerdictRationale(text="test"),
        model_version="test", timestamp="2026-01-01T00:00:00Z",
    )
    av = ArgumentVerdict(
        text="test", verdict=ArgumentVerdictEnum(arg_verdict),
        rationale=VerdictRationale(text="test"),
    )
    return VerificationReport(
        document_id="d1", total_claims=1,
        verdicts=[v], argument_verdicts=[av],
    )


class TestEvalRunner:
    def test_load_dataset_from_list(self):
        cases = [
            EvalCase(id="t1", memo="m", corpus=[], expected_claim_verdicts=["SUPPORTED"]),
        ]
        runner = EvalRunner(cases=cases)
        assert len(runner.cases) == 1

    def test_load_dataset_from_json(self, tmp_path):
        data = [{"id": "t1", "memo": "m", "corpus": [],
                 "expected_claim_verdicts": ["SUPPORTED"]}]
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps(data))
        runner = EvalRunner.from_json(str(path))
        assert len(runner.cases) == 1

    @patch("graphite.eval.runner.verify_agent_output")
    def test_run_returns_eval_run(self, mock_verify):
        mock_verify.return_value = _make_report("CONFLICTED", "GROUNDED")

        cases = [
            EvalCase(id="t1", memo="memo", corpus=[{"document_id": "d", "text": "t"}],
                     expected_claim_verdicts=["CONFLICTED"],
                     expected_argument_verdicts=["GROUNDED"]),
        ]
        runner = EvalRunner(cases=cases)
        run = runner.run(model="test-model")

        assert isinstance(run, EvalRun)
        assert run.model == "test-model"
        assert len(run.results) == 1
        assert run.results[0].passed is True

    @patch("graphite.eval.runner.verify_agent_output")
    def test_run_detects_mismatch(self, mock_verify):
        mock_verify.return_value = _make_report("SUPPORTED", "GROUNDED")

        cases = [
            EvalCase(id="t1", memo="memo", corpus=[{"document_id": "d", "text": "t"}],
                     expected_claim_verdicts=["CONFLICTED"]),
        ]
        runner = EvalRunner(cases=cases)
        run = runner.run(model="test-model")

        assert run.results[0].claim_verdict_pass is False
        assert run.results[0].passed is False

    @patch("graphite.eval.runner.verify_agent_output")
    def test_run_saves_results(self, mock_verify, tmp_path):
        mock_verify.return_value = _make_report("CONFLICTED", "GROUNDED")

        cases = [
            EvalCase(id="t1", memo="m", corpus=[{"document_id": "d", "text": "t"}],
                     expected_claim_verdicts=["CONFLICTED"]),
        ]
        runner = EvalRunner(cases=cases)
        out_path = str(tmp_path / "results.json")
        run = runner.run(model="test-model", output_path=out_path)

        loaded = EvalRun.load(out_path)
        assert loaded.model == "test-model"
        assert len(loaded.results) == 1

    @patch("graphite.eval.runner.verify_agent_output")
    def test_run_with_domain_filter(self, mock_verify):
        mock_verify.return_value = _make_report("SUPPORTED", "GROUNDED")

        cases = [
            EvalCase(id="t1", memo="m", corpus=[], expected_claim_verdicts=["SUPPORTED"], domain="finance"),
            EvalCase(id="t2", memo="m", corpus=[], expected_claim_verdicts=["SUPPORTED"], domain="medical"),
        ]
        runner = EvalRunner(cases=cases)
        run = runner.run(model="test", domain="medical")

        assert len(run.results) == 1
        assert run.results[0].case_id == "t2"

    @patch("graphite.eval.runner.verify_agent_output")
    def test_pipeline_error_captured(self, mock_verify):
        mock_verify.side_effect = ValueError("LLM exploded")

        cases = [
            EvalCase(id="t1", memo="m", corpus=[], expected_claim_verdicts=["SUPPORTED"]),
        ]
        runner = EvalRunner(cases=cases)
        run = runner.run(model="test")

        assert run.results[0].error == "LLM exploded"
        assert run.results[0].passed is False
