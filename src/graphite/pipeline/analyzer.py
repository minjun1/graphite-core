"""
graphite/pipeline/analyzer.py — Argument-level analysis for detecting logic leaps.
"""

import json
import os
from typing import List, Dict, Any, Optional

from graphite.pipeline.verdict import (
    Verdict,
    ArgumentVerdict,
    ArgumentVerdictEnum,
    VerdictRationale,
)


class ArgumentAnalyzer:
    """Analyzes the overall argument chain for unsupported conclusions (CONCLUSION_JUMP)."""

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
                'Please install the default analyzer: pip install "graphite-engine[llm]"'
            )

    def analyze_argument_chain(
        self, memo_text: str, verdicts: List[Verdict], model: str = "gemini-2.5-flash"
    ) -> List[ArgumentVerdict]:
        """
        Evaluate if the document's overall conclusions logically follow from the supported claims.
        """
        system_prompt = (
            "You are an expert logical analyzer. You will see an original MEMO containing arguments, "
            "and a list of VERDICTS for individual factual claims within that memo. "
            "Some claims may be SUPPORTED, others CONFLICTED. "
            "Identify any major 'conclusion leaps' where the memo makes a broad assertion or recommendation "
            "that is NOT justified by the supported facts, or is actively undermined by conflicted facts. "
            "Return JSON with an 'argument_verdicts' array. Each item should have: "
            "'text' (the conclusion text), 'verdict' (GROUNDED|CONCLUSION_JUMP|OVERSTATED), "
            "'rationale_text' (string), 'contradiction_type' (string or null), "
            "'needs_human_review' (boolean)."
        )

        verdicts_summary = ""
        for v in verdicts:
            verdicts_summary += f"- Claim: {v.claim_text}\n  Verdict: {v.verdict.value}\n  Rationale: {v.rationale.text}\n\n"

        user_prompt = f"MEMO TEXT:\n{memo_text}\n\nCLAIM VERDICTS:\n{verdicts_summary}"

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        res_data = json.loads(response.choices[0].message.content)
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
) -> List[ArgumentVerdict]:
    """Convenience functional interface for argument analysis."""
    analyzer = ArgumentAnalyzer(api_key=api_key)
    return analyzer.analyze_argument_chain(memo_text, verdicts, model=model)
