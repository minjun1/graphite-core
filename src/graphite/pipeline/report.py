"""
graphite/pipeline/report.py — Output reporting and the Hero API `verify_agent_output`.
"""

import hashlib
from typing import List, Dict, Any, Optional

from graphite.claim import VerificationReport, VerdictEnum, ArgumentVerdictEnum
from graphite.pipeline.extractor import extract_claims
from graphite.pipeline.retriever import retrieve_evidence
from graphite.pipeline.verifier import verify_claims
from graphite.pipeline.analyzer import analyze_argument_chain


def verify_agent_output(
    text: str,
    corpus: List[Dict[str, str]],
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
) -> VerificationReport:
    """
    The Hero API: End-to-end verification pipeline.

    Args:
        text (str): The agent-generated markdown memo or assertions.
        corpus (List[Dict[str, str]]): List of evidence documents e.g. [{"document_id": "doc1", "text": "..."}]
        model (str): LLM model name to use across the pipeline.

    Returns:
        VerificationReport: The aggregated results, ready for UI and audit logs.
    """
    document_id = hashlib.sha256(text.encode()).hexdigest()[:12]

    # 1. Extract
    claims = extract_claims(text, model=model, api_key=api_key)

    # 2. Retrieve
    evidence_map = retrieve_evidence(claims, corpus)

    # 3. Verify
    verdicts = verify_claims(claims, evidence_map, model=model, api_key=api_key)

    # 4. Analyze Arguments
    argument_verdicts = analyze_argument_chain(
        text, verdicts, model=model, api_key=api_key
    )

    # 5. Build Report
    report = VerificationReport(
        document_id=document_id,
        total_claims=len(claims),
        verdicts=verdicts,
        argument_verdicts=argument_verdicts,
        model_metadata={"provider_model": model},
    )

    # Aggregate summaries
    for v in verdicts:
        if v.verdict == VerdictEnum.SUPPORTED:
            report.supported_count += 1
        elif v.verdict == VerdictEnum.CONFLICTED:
            report.conflicted_count += 1
        elif v.verdict == VerdictEnum.INSUFFICIENT:
            report.insufficient_count += 1

        if v.needs_human_review or v.verdict == VerdictEnum.CONFLICTED:
            report.risky_claim_ids.append(v.claim_id)

    for av in argument_verdicts:
        if av.verdict == ArgumentVerdictEnum.GROUNDED:
            report.grounded_argument_count += 1
        elif av.verdict == ArgumentVerdictEnum.CONCLUSION_JUMP:
            report.conclusion_jump_count += 1

    if report.total_claims > 0:
        report.evidence_coverage_score = report.supported_count / report.total_claims

    return report


# review_document is a semantic alias for verify_agent_output for different use cases.
review_document = verify_agent_output
