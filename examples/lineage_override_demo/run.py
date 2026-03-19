#!/usr/bin/env python3
"""
Lineage & Override Demo — Graphite ClaimStore

Shows graph-as-memory capabilities beyond basic accumulation:
  1. Initial verification verdict
  2. Analyst override (human-in-the-loop)
  3. New evidence arrives → re-review
  4. Prior decisions remain queryable (lineage)

Run: python examples/lineage_override_demo/run.py

No LLM. No API keys. Fully local.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from graphite import Claim, ClaimStore, ClaimType, ClaimOrigin, Provenance
from graphite.claim import ClaimStatus, ReviewState, ConfidenceResult, ConfidenceFactor
from graphite.enums import SourceType, ConfidenceLevel


def main():
    db_path = os.path.join(tempfile.gettempdir(), "graphite_lineage.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    store = ClaimStore(db_path=db_path)

    # ── Step 1: Initial claim with machine verdict ──
    print("=" * 60)
    print("  Graphite Lineage & Override Demo")
    print("=" * 60)
    print()
    print("📄 Step 1: Agent extracts claim → machine verdict")
    print()

    claim = Claim(
        subject_entities=["company:NVDA"],
        predicate="REVENUE_DROPS",
        object_entities=["metric:REVENUE"],
        claim_text="Nvidia's revenue will drop by 50% next quarter due to CoWoS constraints.",
        claim_type=ClaimType.RISK_ASSERTION,
        origin=ClaimOrigin.AGENT,
        generator_id="research-agent-v2",
        supporting_evidence=[
            Provenance(
                source_id="agent-memo-001",
                source_type=SourceType.WEB,
                evidence_quote="TSMC CoWoS capacity is constrained, therefore Nvidia revenue drops 50%.",
                confidence=ConfidenceLevel.MEDIUM,
            )
        ],
        weakening_evidence=[
            Provenance(
                source_id="nvda-10k-2024",
                source_type=SourceType.SEC_10K,
                evidence_quote="We have secured advance capacity and expect revenue to grow sequentially.",
                confidence=ConfidenceLevel.HIGH,
            )
        ],
    )

    # Score and compute machine verdict
    claim.confidence = ConfidenceResult.from_score(
        0.35,
        factors=[
            ConfidenceFactor(
                name="source_count",
                raw_value="1 supporting, 1 weakening",
                contribution=-0.1,
                direction="NEGATIVE",
                explanation="Conflicting evidence from higher-quality source",
            ),
            ConfidenceFactor(
                name="source_diversity",
                raw_value="agent memo vs SEC filing",
                contribution=-0.15,
                direction="NEGATIVE",
                explanation="SEC filing contradicts agent-generated claim",
            ),
        ],
    )
    claim.compute_status()
    store.save_claim(claim)

    saved = store.get_claim(claim.claim_id)
    print(f'   Claim: "{saved.claim_text}"')
    print(f"   Machine verdict: {saved.computed_status.value}")
    print(f"   Confidence: {saved.confidence.score} ({saved.confidence.level.value})")
    print(f"   Review state: {saved.review_state.value}")
    print()

    # ── Step 2: Analyst overrides the verdict ──
    print("👤 Step 2: Analyst reviews → overrides to UNSUPPORTED")
    print()

    saved.override_status(
        status=ClaimStatus.UNSUPPORTED,
        reason="50% revenue drop is speculative. SEC filing directly contradicts this with forward guidance.",
        reviewer="analyst-jane",
    )
    store.save_claim(saved)

    saved = store.get_claim(claim.claim_id)
    print(f"   Computed (machine): {saved.computed_status.value}")
    print(f"   Final (after override): {saved.final_status.value}")
    print(f'   Override reason: "{saved.override_reason}"')
    print(f"   Reviewed by: {saved.reviewed_by}")
    print(f"   Review state: {saved.review_state.value}")
    print()

    # ── Step 3: New evidence arrives → re-review ──
    print("📰 Step 3: New evidence arrives (earnings call) → re-review")
    print()

    updated_claim = Claim(
        subject_entities=["company:NVDA"],
        predicate="REVENUE_DROPS",
        object_entities=["metric:REVENUE"],
        claim_text="Nvidia's revenue will drop by 50% next quarter due to CoWoS constraints.",
        claim_type=ClaimType.RISK_ASSERTION,
        origin=ClaimOrigin.AGENT,
        weakening_evidence=[
            Provenance(
                source_id="nvda-earnings-q4-2024",
                source_type=SourceType.WEB,
                evidence_quote="Revenue grew 22% sequentially to $22.1B, exceeding guidance. CoWoS capacity expanded.",
                confidence=ConfidenceLevel.HIGH,
            )
        ],
    )
    store.save_claim(updated_claim)

    saved = store.get_claim(claim.claim_id)
    print(f"   Supporting evidence: {len(saved.supporting_evidence)}")
    print(f"   Weakening evidence: {len(saved.weakening_evidence)} ← accumulated!")
    for i, ev in enumerate(saved.weakening_evidence):
        print(f'     [{i + 1}] {ev.source_id}: "{ev.evidence_quote[:70]}..."')
    print()
    print(
        f"   Final status still: {saved.final_status.value} (analyst override preserved)"
    )
    print(f'   Override reason still: "{saved.override_reason}"')
    print()

    # ── Step 4: Query full lineage ──
    print("🔍 Step 4: Full claim lineage — every decision is queryable")
    print()
    print(f"   Claim ID: {saved.claim_id}")
    print(f"   Origin: {saved.origin.value}")
    print(f"   Generator: {saved.generator_id}")
    print(f"   Created at: {saved.extracted_at}")
    print(f"   Evidence trail:")
    print(f"     Supporting: {len(saved.supporting_evidence)} source(s)")
    for ev in saved.supporting_evidence:
        print(
            f"       └── {ev.source_id} ({ev.source_type.value}, confidence: {ev.confidence.value})"
        )
    print(f"     Weakening: {len(saved.weakening_evidence)} source(s)")
    for ev in saved.weakening_evidence:
        print(
            f"       └── {ev.source_id} ({ev.source_type.value}, confidence: {ev.confidence.value})"
        )
    print(f"   Machine verdict: {saved.computed_status.value}")
    print(
        f"   Analyst override: {saved.final_status.value} (by {saved.reviewed_by} at {saved.reviewed_at})"
    )
    print(f'   Override reason: "{saved.override_reason}"')
    print()

    # ── Step 5: Demonstrate analyst override survives recompute ──
    print("🔒 Step 5: Machine recompute does NOT overwrite analyst override")
    print()

    saved.compute_status()
    print(f"   Recomputed (machine): {saved.computed_status.value}")
    print(f"   Final (analyst): {saved.final_status.value} ← preserved!")
    print(f"   is_overridden: {saved.is_overridden}")
    print()

    # ── Done ──
    print("=" * 60)
    print("  ✅ Lineage & override demo complete!")
    print()
    print("  Key takeaways:")
    print("    1. Machine verdicts and analyst overrides coexist")
    print("    2. New evidence accumulates without losing history")
    print("    3. Analyst decisions persist through recomputation")
    print("    4. Every step is queryable — full provenance trail")
    print("=" * 60)

    os.remove(db_path)


if __name__ == "__main__":
    main()
