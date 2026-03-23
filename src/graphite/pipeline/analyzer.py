"""
graphite/pipeline/analyzer.py — Argument-level analysis for detecting logic leaps.
"""

import json
from typing import List, Dict, Any, Optional

from graphite.pipeline.verdict import (
    Verdict,
    ArgumentVerdict,
    ArgumentVerdictEnum,
    VerdictRationale,
)


class ArgumentAnalyzer:
    """Analyzes the overall argument chain for unsupported conclusions (CONCLUSION_JUMP)."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, system_prompt: Optional[str] = None):
        from graphite.pipeline._client import create_openai_client
        from graphite.pipeline.prompts import ANALYZER_SYSTEM_PROMPT
        self.client = create_openai_client(api_key=api_key, base_url=base_url)
        self.system_prompt = system_prompt or ANALYZER_SYSTEM_PROMPT

    def analyze_argument_chain(
        self, memo_text: str, verdicts: List[Verdict], model: str = "gemini-2.5-flash"
    ) -> List[ArgumentVerdict]:
        """
        Evaluate if the document's overall conclusions logically follow from the supported claims.
        """
        verdicts_summary = ""
        for v in verdicts:
            verdicts_summary += f"- Claim: {v.claim_text}\n  Verdict: {v.verdict.value}\n  Rationale: {v.rationale.text}\n\n"

        user_prompt = f"MEMO TEXT:\n{memo_text}\n\nCLAIM VERDICTS:\n{verdicts_summary}"

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        try:
            res_data = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM returned invalid JSON: {e}. "
                f"Raw response: {response.choices[0].message.content[:200]}"
            ) from e
        results = []

        for item in res_data.get("argument_verdicts", []):
            try:
                v_enum = ArgumentVerdictEnum(item.get("verdict", "GROUNDED").upper())
            except ValueError:
                v_enum = ArgumentVerdictEnum.GROUNDED

            rationale = VerdictRationale(
                contradiction_type=item.get("contradiction_type"),
                text=item.get("rationale_text", ""),
            )

            results.append(
                ArgumentVerdict(
                    text=item.get("text", ""),
                    verdict=v_enum,
                    rationale=rationale,
                    needs_human_review=item.get("needs_human_review", False),
                )
            )

        return results


def analyze_argument_chain(
    memo_text: str,
    verdicts: List[Verdict],
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> List[ArgumentVerdict]:
    """Convenience functional interface for argument analysis."""
    analyzer = ArgumentAnalyzer(api_key=api_key, system_prompt=system_prompt)
    return analyzer.analyze_argument_chain(memo_text, verdicts, model=model)
