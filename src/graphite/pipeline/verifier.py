"""
graphite/pipeline/verifier.py — Claim-level verification using LLMs.
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from graphite.claim import Claim
from graphite.pipeline.verdict import Verdict, VerdictEnum, VerdictRationale


class ClaimVerifier:
    """Evaluates claims against retrieved evidence spans."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("OPENAI_API_KEY"),
                base_url=base_url
                or os.environ.get(
                    "OPENAI_BASE_URL",
                    "https://generativelanguage.googleapis.com/v1beta/openai/",
                ),
            )
        except ImportError:
            raise ImportError(
                'Please install the default verifier: pip install "graphite-engine[llm]"'
            )

    def verify_claims(
        self,
        claims: List[Claim],
        evidence_map: Dict[str, List[Dict[str, Any]]],
        model: str = "gemini-2.5-flash",
    ) -> List[Verdict]:
        """
        Verify each claim against its retrieved evidence chunks.
        evidence_map is a dict of claim_id -> list of chunk dicts.
        """
        verdicts = []

        system_prompt = (
            "You are an expert fact-checker. You will be given a CLAIM and a set of EVIDENCE chunks. "
            "Determine if the claim is SUPPORTED, CONFLICTED, or INSUFFICIENT based ONLY on the evidence. "
            "You must return JSON with the following keys: "
            "'verdict' (SUPPORTED|CONFLICTED|INSUFFICIENT), "
            "'rationale_text' (string explanation), "
            "'contradiction_type' (string or null, e.g. 'numeric mismatch'), "
            "'missing_evidence_reason' (string or null), "
            "'temporal_alignment' (string or null, e.g. 'stale evidence'), "
            "'needs_human_review' (boolean), "
            "'cited_span' (exact quote from evidence used, or null), "
            "'supporting_evidence_indices' (list of ints), "
            "'conflicting_evidence_indices' (list of ints)."
        )

        for claim in claims:
            chunks = evidence_map.get(claim.claim_id, [])
            evidence_text = "\n".join(
                [f"[{i}] {c['text']}" for i, c in enumerate(chunks)]
            )

            user_prompt = f"CLAIM:\n{claim.claim_text}\n\nEVIDENCE:\n{evidence_text}"

            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            res = json.loads(response.choices[0].message.content)

            # Extract evidence IDs based on returned indices
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

            # Map string to enum
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
) -> List[Verdict]:
    """Convenience functional interface for claim verification."""
    verifier = ClaimVerifier(api_key=api_key)
    return verifier.verify_claims(claims, evidence_map, model=model)
