"""graphite/evals/verify_eval.py — Run eval using the EvalRunner."""

import os
from graphite.eval import EvalRunner

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "graphite", "eval", "datasets", "base.json"
)


def run_evaluation(model: str = "gemini-2.5-flash"):
    runner = EvalRunner.from_json(DATASET_PATH)
    run = runner.run(model=model)

    for result in run.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.case_id}")
        if result.error:
            print(f"  Error: {result.error}")
        if not result.claim_verdict_pass:
            print(f"  Claim: expected {result.expected_claim_verdicts}, got {result.actual_claim_verdicts}")
        if not result.argument_verdict_pass:
            print(f"  Argument: expected {result.expected_argument_verdicts}, got {result.actual_argument_verdicts}")

    metrics = run.metrics()
    print(f"\nResults: {metrics['overall_pass_rate']:.0%} pass rate ({metrics['total']} cases)")
    if metrics.get("mean_latency_ms"):
        print(f"Mean latency: {metrics['mean_latency_ms']:.0f}ms")


if __name__ == "__main__":
    run_evaluation()
