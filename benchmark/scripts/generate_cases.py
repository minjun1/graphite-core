"""
benchmark/scripts/generate_cases.py — Semi-synthetic SEC benchmark case generator.

Workflow:
  1. Load SEC source seed data (real excerpts from 10-K, 10-Q, earnings calls)
  2. For each seed, use an LLM to generate a memo with 4-6 claims
  3. Inject controlled failure modes into some claims
  4. Output structured JSONL matching benchmark/dataset_schema.json

Usage:
  python benchmark/scripts/generate_cases.py \
    --seeds benchmark/scripts/sec_seeds.json \
    --output benchmark/sec_cases_pilot.jsonl \
    --count 10

Environment:
  GEMINI_API_KEY must be set.
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ── Distribution targets ──
# 40% SUPPORTED, 30% CONFLICTED, 20% INSUFFICIENT, 10% CONCLUSION_JUMP
FAILURE_MODE_DISTRIBUTION = {
    "supported": 0.40,
    "numeric_mismatch": 0.10,
    "temporal_mismatch": 0.10,
    "paraphrased_contradiction": 0.10,
    "unsupported_extrapolation": 0.10,
    "missing_evidence": 0.10,
    "conclusion_jump": 0.10,
}

VERDICT_MAP = {
    "supported": ("SUPPORTED", "GROUNDED", None),
    "numeric_mismatch": ("CONFLICTED", "GROUNDED", "numeric_mismatch"),
    "temporal_mismatch": ("CONFLICTED", "GROUNDED", "temporal_mismatch"),
    "paraphrased_contradiction": ("CONFLICTED", "GROUNDED", "paraphrased_contradiction"),
    "unsupported_extrapolation": ("INSUFFICIENT", "CONCLUSION_JUMP", "unsupported_extrapolation"),
    "missing_evidence": ("INSUFFICIENT", "GROUNDED", "missing_evidence"),
    "conclusion_jump": ("SUPPORTED", "CONCLUSION_JUMP", "unsupported_extrapolation"),
}


def assign_failure_modes(n_claims: int) -> List[str]:
    """Assign failure modes to n claims following the target distribution."""
    modes = []
    for mode, ratio in FAILURE_MODE_DISTRIBUTION.items():
        count = max(1, round(n_claims * ratio))
        modes.extend([mode] * count)

    # Trim or pad to exactly n_claims
    random.shuffle(modes)
    if len(modes) > n_claims:
        modes = modes[:n_claims]
    while len(modes) < n_claims:
        modes.append("supported")

    random.shuffle(modes)
    return modes


MEMO_GENERATION_PROMPT = """You are a financial analyst writing a brief investment memo.

Given the following SEC source documents, write a SHORT investment memo (3-5 paragraphs) 
that makes {n_claims} specific factual claims about the company. Each claim should be 
a distinct, verifiable statement.

IMPORTANT INSTRUCTIONS FOR SPECIFIC CLAIMS:
{injection_instructions}

Source Documents:
{source_text}

Return JSON with the following structure:
{{
  "memo_text": "The full memo text",
  "claims": [
    {{
      "claim_text": "A specific claim",
      "injection_type": "supported|numeric_mismatch|temporal_mismatch|paraphrased_contradiction|unsupported_extrapolation|missing_evidence|conclusion_jump",
      "difficulty": "easy|medium|hard"
    }}
  ],
  "final_conclusion": "The memo's overall conclusion"
}}
"""

INJECTION_TEMPLATES = {
    "supported": "Claim {i}: Write a claim that is DIRECTLY and ACCURATELY supported by the source data. Use correct numbers, dates, and facts.",
    "numeric_mismatch": "Claim {i}: Write a claim that cites a WRONG NUMBER from the source. For example, if revenue was $35.1B, say $38.5B or $31.2B. Make it plausible but clearly incorrect.",
    "temporal_mismatch": "Claim {i}: Write a claim that gets the TIME PERIOD wrong. For example, attribute a Q3 result to Q4, or say 'current' about something that changed.",
    "paraphrased_contradiction": "Claim {i}: Write a claim that PARAPHRASES a fact from the source but REVERSES or DISTORTS its meaning. For example, if the source says 'migrating away from X', say 'exclusively uses X'.",
    "unsupported_extrapolation": "Claim {i}: Write a claim that takes a real data point and EXTRAPOLATES far beyond what the source supports. Make a prediction or causal claim the data doesn't warrant.",
    "missing_evidence": "Claim {i}: Write a claim about something the source documents DON'T COVER at all. Make it plausible for this company but unverifiable from the given sources.",
    "conclusion_jump": "Claim {i}: Write a claim where the underlying FACT is correct but the CONCLUSION drawn from it is a logical leap. The data point is real, but the inference is not warranted.",
}


def build_injection_instructions(modes: List[str]) -> str:
    """Build per-claim injection instructions for the LLM."""
    lines = []
    for i, mode in enumerate(modes, 1):
        template = INJECTION_TEMPLATES.get(mode, INJECTION_TEMPLATES["supported"])
        lines.append(template.format(i=i))
    return "\n".join(lines)


def generate_case_with_llm(
    seed: Dict[str, Any],
    case_index: int,
    n_claims: int = 5,
    model: str = "gemini-2.5-flash",
) -> Optional[Dict[str, Any]]:
    """Generate a single benchmark case from a seed using the LLM."""
    from graphite.llm import gemini_extract_json

    modes = assign_failure_modes(n_claims)
    injection_instructions = build_injection_instructions(modes)

    # Build source text for the prompt
    source_text = ""
    for doc in seed["source_docs"]:
        source_text += f"\n--- {doc['doc_type']}: {doc['doc_id']} ---\n{doc['text']}\n"

    prompt = MEMO_GENERATION_PROMPT.format(
        n_claims=n_claims,
        injection_instructions=injection_instructions,
        source_text=source_text,
    )

    system_prompt = (
        "You are a benchmark data generator for a financial claim verification system. "
        "Follow the injection instructions EXACTLY to create claims with specific error types. "
        "Return ONLY valid JSON matching the requested structure."
    )

    try:
        result = gemini_extract_json(
            contents=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=0.7,
        )
    except Exception as e:
        print(f"  ⚠️ LLM generation failed for seed {seed['ticker']}: {e}")
        return None

    # Build the structured case
    ticker = seed["ticker"]
    period = seed.get("fiscal_period", "FY2024")
    case_id = f"SEC-{ticker}-{period}-{case_index:03d}"

    claims = []
    generated_claims = result.get("claims", [])

    for j, (gen_claim, mode) in enumerate(zip(generated_claims, modes)):
        verdict, arg_verdict, failure_mode = VERDICT_MAP[mode]

        # Determine needs_human_review
        needs_review = verdict == "CONFLICTED" or arg_verdict == "CONCLUSION_JUMP"

        claim = {
            "claim_id": f"{case_id}-c{j+1}",
            "claim_text": gen_claim.get("claim_text", ""),
            "gold_verdict": verdict,
            "gold_argument_verdict": arg_verdict,
            "needs_human_review": needs_review,
            "gold_cited_span_ids": [],  # To be filled during manual annotation
            "failure_mode": failure_mode,
            "difficulty": gen_claim.get("difficulty", "medium"),
            "canonical_claim_id": "",
            "annotator_notes": f"Auto-generated with injection: {mode}",
        }
        claims.append(claim)

    case = {
        "case_id": case_id,
        "ticker": ticker,
        "filing_type": seed.get("filing_type", "mixed"),
        "fiscal_period": period,
        "generation_method": "semi_synthetic",
        "source_docs": seed["source_docs"],
        "memo_text": result.get("memo_text", ""),
        "claims": claims,
        "final_conclusion": result.get("final_conclusion", ""),
        "stateful_metadata": {
            "review_cycle": 1,
            "canonical_claims_introduced": [],
            "canonical_claims_repeated": [],
        },
        "metadata": {
            "created_by": "generate_cases.py",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": "0.1",
        },
    }

    return case


def generate_case_manual(
    seed: Dict[str, Any],
    case_index: int,
) -> Dict[str, Any]:
    """Generate a case using pre-written claims from the seed (no LLM needed).
    
    Use this when seeds already contain pre-written memo_text and claims.
    """
    ticker = seed["ticker"]
    period = seed.get("fiscal_period", "FY2024")
    case_id = f"SEC-{ticker}-{period}-{case_index:03d}"

    claims = []
    for j, claim_seed in enumerate(seed.get("claims", [])):
        claim = {
            "claim_id": f"{case_id}-c{j+1}",
            "claim_text": claim_seed["claim_text"],
            "gold_verdict": claim_seed["gold_verdict"],
            "gold_argument_verdict": claim_seed["gold_argument_verdict"],
            "needs_human_review": claim_seed.get("needs_human_review", False),
            "gold_cited_span_ids": claim_seed.get("gold_cited_span_ids", []),
            "failure_mode": claim_seed.get("failure_mode"),
            "difficulty": claim_seed.get("difficulty", "medium"),
            "canonical_claim_id": claim_seed.get("canonical_claim_id", ""),
            "annotator_notes": claim_seed.get("annotator_notes", ""),
        }
        claims.append(claim)

    return {
        "case_id": case_id,
        "ticker": ticker,
        "filing_type": seed.get("filing_type", "mixed"),
        "fiscal_period": period,
        "generation_method": seed.get("generation_method", "semi_synthetic"),
        "source_docs": seed["source_docs"],
        "memo_text": seed.get("memo_text", ""),
        "claims": claims,
        "final_conclusion": seed.get("final_conclusion", ""),
        "stateful_metadata": seed.get("stateful_metadata", {
            "review_cycle": 1,
            "canonical_claims_introduced": [],
            "canonical_claims_repeated": [],
        }),
        "metadata": {
            "created_by": "generate_cases.py",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": "0.1",
        },
    }


def load_seeds(path: str) -> List[Dict[str, Any]]:
    """Load seed data from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("seeds", [])


def print_distribution(cases: List[Dict[str, Any]]) -> None:
    """Print the verdict and failure mode distribution."""
    from collections import Counter

    verdicts = Counter()
    arg_verdicts = Counter()
    failure_modes = Counter()

    for case in cases:
        for claim in case["claims"]:
            verdicts[claim["gold_verdict"]] += 1
            arg_verdicts[claim["gold_argument_verdict"]] += 1
            failure_modes[str(claim["failure_mode"])] += 1

    total = sum(verdicts.values())
    print(f"\n📊 Distribution ({total} total claims across {len(cases)} cases):")
    print("\n  Claim Verdicts:")
    for v, count in sorted(verdicts.items()):
        print(f"    {v}: {count} ({count/total*100:.0f}%)")
    print("\n  Argument Verdicts:")
    for v, count in sorted(arg_verdicts.items()):
        print(f"    {v}: {count} ({count/total*100:.0f}%)")
    print("\n  Failure Modes:")
    for m, count in sorted(failure_modes.items()):
        print(f"    {m}: {count} ({count/total*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Generate SEC benchmark cases")
    parser.add_argument("--seeds", required=True, help="Path to seed data JSON")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--count", type=int, default=10, help="Number of cases to generate")
    parser.add_argument("--model", default="gemini-2.5-flash", help="LLM model for generation")
    parser.add_argument("--mode", choices=["llm", "manual"], default="manual",
                        help="Generation mode: 'llm' for LLM-generated, 'manual' for pre-written seeds")
    parser.add_argument("--claims-per-memo", type=int, default=5, help="Claims per memo (LLM mode)")
    args = parser.parse_args()

    print(f"📂 Loading seeds from {args.seeds}...")
    seeds = load_seeds(args.seeds)
    print(f"   Found {len(seeds)} seeds")

    count = min(args.count, len(seeds))
    cases = []

    for i, seed in enumerate(seeds[:count]):
        print(f"\n🔨 Generating case {i+1}/{count}: {seed['ticker']}...")

        if args.mode == "llm":
            case = generate_case_with_llm(seed, i + 1, n_claims=args.claims_per_memo, model=args.model)
        else:
            case = generate_case_manual(seed, i + 1)

        if case:
            cases.append(case)
            print(f"   ✓ {case['case_id']}: {len(case['claims'])} claims")
        else:
            print(f"   ✗ Failed to generate case for {seed['ticker']}")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"\n✅ Wrote {len(cases)} cases to {output_path}")
    print_distribution(cases)


if __name__ == "__main__":
    main()
