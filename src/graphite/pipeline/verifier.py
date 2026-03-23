"""graphite/pipeline/verifier.py — Claim-level verification using LLMs."""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from graphite.claim import Claim
from graphite.pipeline.verdict import Verdict, VerdictEnum, VerdictRationale


class ClaimVerifier:
    """Evaluates claims against retrieved evidence spans."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, system_prompt: Optional[str] = None):
        from graphite.pipeline._client import create_llm_client
        from graphite.pipeline.prompts import VERIFIER_SYSTEM_PROMPT
        self.client = create_llm_client(api_key=api_key, base_url=base_url)
        self.system_prompt = system_prompt or VERIFIER_SYSTEM_PROMPT

    def verify_claims(
        self,
        claims: List[Claim],
        evidence_map: Dict[str, List[Dict[str, Any]]],
        model: str = "gemini-2.5-flash",
    ) -> List[Verdict]:
        """Verify each claim against its retrieved evidence chunks."""
        verdicts = []

        for claim in claims:
            chunks = evidence_map.get(claim.claim_id, [])
            evidence_text = "\n".join(
                [f"[{i}] {c['text']}" for i, c in enumerate(chunks)]
            )

            user_prompt = f"CLAIM:\n{claim.claim_text}\n\nEVIDENCE:\n{evidence_text}"

            res = self.client.chat_json(
                model=model,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
            )

            supp_ids = [
                chunks[i]["document_id"]
                for i in res.get("supporting_evidence_indices", [])
                if i < len(chunks)
            ]
            conf_ids = [
                chunks[i]["document_id"]
                for i in res.get("conflicting_evidence_indices", [])
                if i < len(chunks)
            ]

            rationale = VerdictRationale(
                contradiction_type=res.get("contradiction_type"),
                missing_evidence_reason=res.get("missing_evidence_reason"),
                temporal_alignment=res.get("temporal_alignment"),
                text=res.get("rationale_text", ""),
            )

            v_str = res.get("verdict", "INSUFFICIENT").upper()
            try:
                verdict_enum = VerdictEnum(v_str)
            except ValueError:
                verdict_enum = VerdictEnum.INSUFFICIENT

            verdict = Verdict(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                verdict=verdict_enum,
                supporting_evidence_ids=list(set(supp_ids)),
                conflicting_evidence_ids=list(set(conf_ids)),
                rationale=rationale,
                needs_human_review=res.get("needs_human_review", False),
                cited_span=res.get("cited_span"),
                model_version=model,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            verdicts.append(verdict)

        return verdicts


def verify_claims(
    claims: List[Claim],
    evidence_map: Dict[str, List[Dict[str, Any]]],
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> List[Verdict]:
    """Convenience functional interface for claim verification."""
    verifier = ClaimVerifier(api_key=api_key, system_prompt=system_prompt)
    return verifier.verify_claims(claims, evidence_map, model=model)
