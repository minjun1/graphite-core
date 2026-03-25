"""
graphite/pipeline/report.py — Output reporting and the Hero API `verify_agent_output`.
"""

import hashlib
from typing import List, Dict, Any, Optional

from graphite.pipeline.verdict import VerificationReport, VerdictEnum, ArgumentVerdictEnum
from graphite.pipeline.extractor import ClaimExtractor
from graphite.pipeline.retriever import retrieve_evidence
from graphite.pipeline.verifier import ClaimVerifier
from graphite.pipeline.analyzer import ArgumentAnalyzer
from graphite.pipeline.prompts import PromptSet, DEFAULT_PROMPTS


def verify_agent_output(
    text: str,
    corpus: List[Dict[str, str]],
    model: str = "gemini-3.1-flash",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    prompts: Optional[PromptSet] = None,
) -> VerificationReport:
    """
    The Hero API: End-to-end verification pipeline.

    Args:
        text: The agent-generated markdown memo or assertions.
        corpus: List of evidence documents e.g. [{"document_id": "doc1", "text": "..."}]
        model: LLM model name to use across the pipeline.
        api_key: API key (or set GEMINI_API_KEY/OPENAI_API_KEY/ANTHROPIC_API_KEY).
        base_url: Custom endpoint URL for OpenAI-compatible or Anthropic proxies.
        prompts: Optional PromptSet for domain-specific prompts.

    Returns:
        VerificationReport: The aggregated results, ready for UI and audit logs.
    """
    from graphite.pipeline._client import create_llm_client

    prompts = prompts or DEFAULT_PROMPTS
    client = create_llm_client(api_key=api_key, base_url=base_url)
    document_id = hashlib.sha256(text.encode()).hexdigest()[:12]

    # 1. Extract
    extractor = ClaimExtractor(client=client, system_prompt=prompts.extractor)
    claims = extractor.extract_claims(text, model=model)

    # 2. Retrieve
    evidence_map = retrieve_evidence(claims, corpus)

    # 3. Verify
    verifier = ClaimVerifier(client=client, system_prompt=prompts.verifier)
    verdicts = verifier.verify_claims(claims, evidence_map, model=model)

    # 4. Analyze Arguments
    analyzer = ArgumentAnalyzer(client=client, system_prompt=prompts.analyzer)
    argument_verdicts = analyzer.analyze_argument_chain(text, verdicts, model=model)

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
