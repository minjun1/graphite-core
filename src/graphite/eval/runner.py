"""Eval runner — orchestrates verification pipeline against eval datasets."""

import json
import time
from collections import Counter
from typing import List, Optional

from graphite.eval.types import EvalCase, EvalResult, EvalRun
from graphite.pipeline.report import verify_agent_output
from graphite.pipeline.prompts import PromptSet


class EvalRunner:
    """Run eval cases through the verification pipeline and collect results."""

    def __init__(self, cases: List[EvalCase]):
        self.cases = cases

    @classmethod
    def from_json(cls, path: str) -> "EvalRunner":
        with open(path) as f:
            data = json.load(f)
        cases = [EvalCase.model_validate(item) for item in data]
        return cls(cases=cases)

    def run(
        self,
        model: str = "gemini-3.1-flash",
        api_key: Optional[str] = None,
        prompts: Optional[PromptSet] = None,
        domain: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> EvalRun:
        cases = self.cases
        if domain:
            cases = [c for c in cases if c.domain == domain]

        results = []
        for case in cases:
            result = self._eval_case(case, model=model, api_key=api_key, prompts=prompts)
            results.append(result)

        run = EvalRun(model=model, dataset=domain or "all", results=results)

        if output_path:
            run.save(output_path)

        return run

    def _eval_case(
        self,
        case: EvalCase,
        model: str,
        api_key: Optional[str],
        prompts: Optional[PromptSet],
    ) -> EvalResult:
        start = time.monotonic()
        try:
            report = verify_agent_output(
                case.memo, case.corpus, model=model, api_key=api_key, prompts=prompts,
            )
        except Exception as e:
            return EvalResult(
                case_id=case.id,
                expected_claim_verdicts=case.expected_claim_verdicts,
                claim_verdict_pass=False,
                argument_verdict_pass=False,
                error=str(e),
                latency_ms=(time.monotonic() - start) * 1000,
            )

        elapsed = (time.monotonic() - start) * 1000

        actual_claim = [v.verdict.value for v in report.verdicts]
        actual_arg = [v.verdict.value for v in report.argument_verdicts]

        # Claim verdict pass: all expected verdicts appear in actual (with multiplicity)
        claim_pass = not (Counter(case.expected_claim_verdicts) - Counter(actual_claim))

        # Argument verdict pass: all expected appear in actual (or no expectation)
        arg_pass = True
        if case.expected_argument_verdicts:
            arg_pass = not (Counter(case.expected_argument_verdicts) - Counter(actual_arg))

        return EvalResult(
            case_id=case.id,
            expected_claim_verdicts=case.expected_claim_verdicts,
            actual_claim_verdicts=actual_claim,
            claim_verdict_pass=claim_pass,
            expected_argument_verdicts=case.expected_argument_verdicts,
            actual_argument_verdicts=actual_arg,
            argument_verdict_pass=arg_pass,
            latency_ms=elapsed,
        )
