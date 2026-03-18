#!/usr/bin/env python3
"""
Evidence Accumulation Demo — Graphite ClaimStore

Shows how the ClaimStore builds a stronger fact base over time:
  1. Save a claim from one source
  2. Save the same claim from a different source → evidence accumulates
  3. Find supporting claims and potential conflicts

Run: python examples/evidence_accumulation/run.py

No LLM. No API keys. Fully local.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from graphite import Claim, ClaimStore, ClaimType, ClaimOrigin, Provenance
from graphite.enums import SourceType, ConfidenceLevel


def main():
    db_path = os.path.join(tempfile.gettempdir(), "graphite_accumulation.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    store = ClaimStore(db_path=db_path)

    # ── Step 1: Agent extracts a claim from TSMC's 10-K ──
    print("📄 Step 1: First extraction run (TSMC 10-K)")
    print()

    claim_v1 = Claim(
        subject_entities=["company:TSMC"],
        predicate="SUPPLIES_TO",
        object_entities=["company:NVDA"],
        claim_text="TSMC supplies advanced CoWoS packaging to Nvidia.",
        claim_type=ClaimType.RELATIONSHIP,
        origin=ClaimOrigin.AGENT,
        generator_id="sec-extractor-v2",
        supporting_evidence=[Provenance(
            source_id="tsmc-10k-2024",
            source_type=SourceType.SEC_10K,
            evidence_quote="The Company provides advanced packaging services including CoWoS.",
            confidence=ConfidenceLevel.HIGH,
        )],
    )
    store.save_claim(claim_v1)
    saved = store.get_claim(claim_v1.claim_id)
    print(f"   Evidence count: {len(saved.supporting_evidence)}")
    print(f"   Source: {saved.supporting_evidence[0].source_id}")
    print()

    # ── Step 2: Same claim found in Nvidia's 10-K → evidence accumulates ──
    print("📄 Step 2: Second extraction run (NVDA 10-K) — same claim, new evidence")
    print()

    claim_v2 = Claim(
        subject_entities=["company:TSMC"],
        predicate="SUPPLIES_TO",
        object_entities=["company:NVDA"],
        claim_text="TSMC supplies advanced CoWoS packaging to Nvidia.",
        claim_type=ClaimType.RELATIONSHIP,
        origin=ClaimOrigin.AGENT,
        generator_id="sec-extractor-v2",
        supporting_evidence=[Provenance(
            source_id="nvda-10k-2024",
            source_type=SourceType.SEC_10K,
            evidence_quote="We rely on TSMC for CoWoS advanced packaging of our H100 GPUs.",
            confidence=ConfidenceLevel.HIGH,
        )],
    )
    store.save_claim(claim_v2)
    saved = store.get_claim(claim_v2.claim_id)
    print(f"   Evidence count: {len(saved.supporting_evidence)} ← accumulated!")
    for i, ev in enumerate(saved.supporting_evidence):
        print(f"   [{i+1}] {ev.source_id}: \"{ev.evidence_quote[:60]}...\"")
    print()

    # ── Step 3: Exact duplicate evidence is NOT appended ──
    print("📄 Step 3: Saving exact duplicate → no change")
    print()

    store.save_claim(claim_v1)  # same source_id + same quote
    saved = store.get_claim(claim_v1.claim_id)
    print(f"   Evidence count: {len(saved.supporting_evidence)} ← still 2, duplicate skipped")
    print()

    # ── Step 4: Add a related claim, then query relationships ──
    print("🔍 Step 4: Finding supporting claims and potential conflicts")
    print()

    dependency_claim = Claim(
        subject_entities=["company:NVDA"],
        predicate="DEPENDS_ON",
        object_entities=["company:TSMC"],
        claim_text="Nvidia depends on TSMC for advanced chip packaging.",
        claim_type=ClaimType.DEPENDENCY,
        origin=ClaimOrigin.AGENT,
        generator_id="sec-extractor-v2",
        supporting_evidence=[Provenance(
            source_id="nvda-10k-2024",
            source_type=SourceType.SEC_10K,
            evidence_quote="We depend on TSMC as our primary packaging provider.",
            confidence=ConfidenceLevel.HIGH,
        )],
    )
    store.save_claim(dependency_claim)

    # Find claims that support the original supply claim
    supporting = store.find_supporting_claims(claim_v1)
    print(f"   Supporting claims for '{claim_v1.claim_text}':")
    if supporting:
        for s in supporting:
            print(f"     ✅ {s.claim_text}")
    else:
        print(f"     (none with same predicate)")

    # Find potential conflicts
    conflicts = store.find_potential_conflicts(claim_v1)
    print(f"\n   Potential conflicts:")
    for c in conflicts:
        print(f"     ⚠️  {c.claim_text} (predicate: {c.predicate})")
    print()

    # ── Done ──
    print("=" * 60)
    print("  ✅ Evidence accumulation demo complete!")
    print("  Key takeaway: claims dedupe, evidence accumulates.")
    print("=" * 60)

    os.remove(db_path)


if __name__ == "__main__":
    main()
