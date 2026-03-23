"""Default system prompts for the verification pipeline.

Each pipeline stage (extractor, verifier, analyzer) has a default prompt.
Domains can override any or all prompts via PromptSet.
"""

from pydantic import BaseModel, Field


EXTRACTOR_SYSTEM_PROMPT = (
    "You are an expert fact-extractor. Given a document, extract all distinct factual claims. "
    "For each claim, identify the subject entities, the predicate (relationship/action), and the object entities. "
    "Return JSON with a 'claims' array. Each item should have: "
    "'claim_text' (string), 'subject_entities' (list of str), 'predicate' (string), and 'object_entities' (list of str)."
)

VERIFIER_SYSTEM_PROMPT = (
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

ANALYZER_SYSTEM_PROMPT = (
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


class PromptSet(BaseModel):
    """A set of system prompts for the verification pipeline.

    Supply only the prompts you want to override; others default to built-ins.
    Tracks which fields were explicitly set for correct merge behavior.
    """

    extractor: str = Field(default=EXTRACTOR_SYSTEM_PROMPT)
    verifier: str = Field(default=VERIFIER_SYSTEM_PROMPT)
    analyzer: str = Field(default=ANALYZER_SYSTEM_PROMPT)

    _overridden: set = set()

    def __init__(self, **data):
        super().__init__(**data)
        self._overridden = set(data.keys()) & {"extractor", "verifier", "analyzer"}

    def merge(self, override: "PromptSet") -> "PromptSet":
        """Return a new PromptSet using override values where explicitly set."""
        return PromptSet(
            extractor=override.extractor if "extractor" in override._overridden else self.extractor,
            verifier=override.verifier if "verifier" in override._overridden else self.verifier,
            analyzer=override.analyzer if "analyzer" in override._overridden else self.analyzer,
        )


DEFAULT_PROMPTS = PromptSet()
