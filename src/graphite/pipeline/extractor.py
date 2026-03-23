"""graphite/pipeline/extractor.py — LLM-native claim extraction."""

from typing import List, Optional

from graphite.claim import Claim, ClaimType, ClaimOrigin, ClaimGranularity
from graphite.enums import AssertionMode


class ClaimExtractor:
    """Extracts atomic claims from documents using any supported LLM."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, system_prompt: Optional[str] = None):
        from graphite.pipeline._client import create_llm_client
        from graphite.pipeline.prompts import EXTRACTOR_SYSTEM_PROMPT
        self.client = create_llm_client(api_key=api_key, base_url=base_url)
        self.system_prompt = system_prompt or EXTRACTOR_SYSTEM_PROMPT

    def extract_claims(
        self, document: str, model: str = "gemini-2.5-flash"
    ) -> List[Claim]:
        """Parse raw text into discrete atomic claims."""
        data = self.client.chat_json(
            model=model,
            system_prompt=self.system_prompt,
            user_prompt=document,
        )

        parsed_claims = []
        for c in data.get("claims", []):
            claim = Claim(
                claim_text=c.get("claim_text", ""),
                claim_type=ClaimType.RELATIONSHIP,
                subject_entities=c.get("subject_entities", []),
                predicate=c.get("predicate", "RELATED_TO"),
                object_entities=c.get("object_entities", []),
                assertion_mode=AssertionMode.EXTRACTED,
                origin=ClaimOrigin.EXTRACTOR,
                generator_id=model,
                granularity=ClaimGranularity.ATOMIC,
            )
            parsed_claims.append(claim)

        return parsed_claims


def extract_claims(
    document: str, model: str = "gemini-2.5-flash", api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> List[Claim]:
    """Convenience function to extract claims using the default provider."""
    extractor = ClaimExtractor(api_key=api_key, system_prompt=system_prompt)
    return extractor.extract_claims(document, model=model)
