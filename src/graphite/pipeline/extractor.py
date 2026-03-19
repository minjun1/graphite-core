"""
graphite/pipeline/extractor.py — LLM-native claim extraction.
"""
import json
import os
from typing import List, Optional

from graphite.claim import Claim, ClaimType, ClaimOrigin, ClaimGranularity
from graphite.enums import AssertionMode

class ClaimExtractor:
    """Extracts atomic claims from documents using an OpenAI-compatible LLM."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                base_url=base_url or os.environ.get("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
            )
        except ImportError:
            raise ImportError("Please install the default extractor: pip install \"graphite-engine[llm]\"")
            
    def extract_claims(self, document: str, model: str = "gemini-2.5-flash") -> List[Claim]:
        """Parse raw text into discrete atomic claims."""
        
        system_prompt = (
            "You are an expert fact-extractor. Given a document, extract all distinct factual claims. "
            "For each claim, identify the subject entities, the predicate (relationship/action), and the object entities. "
            "Return JSON with a 'claims' array. Each item should have: "
            "'claim_text' (string), 'subject_entities' (list of str), 'predicate' (string), and 'object_entities' (list of str)."
        )
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": document}
            ],
            response_format={"type": "json_object"}
        )
        
        raw_claims = json.loads(response.choices[0].message.content).get("claims", [])
        
        parsed_claims = []
        for c in raw_claims:
            claim = Claim(
                claim_text=c.get("claim_text", ""),
                claim_type=ClaimType.RELATIONSHIP,
                subject_entities=c.get("subject_entities", []),
                predicate=c.get("predicate", "RELATED_TO"),
                object_entities=c.get("object_entities", []),
                assertion_mode=AssertionMode.EXTRACTED,
                origin=ClaimOrigin.EXTRACTOR,
                generator_id=model,
                granularity=ClaimGranularity.ATOMIC
            )
            parsed_claims.append(claim)
            
        return parsed_claims

def extract_claims(document: str, model: str = "gemini-2.5-flash", api_key: Optional[str] = None) -> List[Claim]:
    """Convenience function to extract claims using the default provider."""
    extractor = ClaimExtractor(api_key=api_key)
    return extractor.extract_claims(document, model=model)
