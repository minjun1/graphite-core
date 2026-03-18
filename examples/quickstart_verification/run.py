#!/usr/bin/env python3
"""
Quickstart: Claim Verification — End-to-End Graphite Demo

Create claims from documents → store in registry → query and verify

Run: python examples/quickstart_verification/run.py

No LLM. No Neo4j. No API keys. Runs fully local.
"""
import os
import sys
import tempfile

# Ensure graphite is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from graphite import (
    Claim, ClaimStore, ClaimType,
    Provenance, ConfidenceScorer,
)
from graphite.enums import SourceType, ConfidenceLevel


def main():
    # Use a temp file for the demo
    db_path = os.path.join(tempfile.gettempdir(), "graphite_quickstart.db")

    # Clean start
    if os.path.exists(db_path):
        os.remove(db_path)

    store = ClaimStore(db_path=db_path)

    # ── Step 1: Extract claims from "documents" ──
    print("📄 Step 1: Creating claims from documents...\n")

    claims = [
        Claim(
            subject_entities=["company:TSMC"],
            predicate="SUPPLIES_TO",
            object_entities=["company:NVDA"],
            claim_text="TSMC supplies advanced CoWoS packaging to Nvidia.",
            claim_type=ClaimType.RELATIONSHIP,
            as_of_date="2024-01-31",
            supporting_evidence=[Provenance(
                source_id="tsmc-10k-2024",
                source_type=SourceType.SEC_10K,
                evidence_quote="The Company provides advanced packaging services including CoWoS to major customers.",
                confidence=ConfidenceLevel.HIGH,
            )],
        ),
        Claim(
            subject_entities=["company:NVDA"],
            predicate="DEPENDS_ON",
            object_entities=["company:SKHYNIX"],
            claim_text="Nvidia depends on SK Hynix for HBM memory supply.",
            claim_type=ClaimType.DEPENDENCY,
            as_of_date="2024-01-28",
            supporting_evidence=[Provenance(
                source_id="nvda-10k-2024",
                source_type=SourceType.SEC_10K,
                evidence_quote="We source HBM3E memory from SK Hynix for our data center accelerators.",
                confidence=ConfidenceLevel.HIGH,
            )],
        ),
        Claim(
            subject_entities=["company:TSMC"],
            predicate="IS_CONSTRAINED",
            object_entities=["product:COWOS"],
            claim_text="TSMC is capacity-constrained in CoWoS packaging.",
            claim_type=ClaimType.ATTRIBUTE,
            as_of_date="2024-01-31",
            supporting_evidence=[Provenance(
                source_id="tsmc-10k-2024",
                source_type=SourceType.SEC_10K,
                evidence_quote="CoWoS advanced packaging capacity remains fully utilized.",
                confidence=ConfidenceLevel.MEDIUM,
            )],
        ),
    ]

    for c in claims:
        print(f"   → {c.claim_text}")
        ev = c.supporting_evidence[0]
        print(f"     source: {ev.source_id}  |  \"{ev.evidence_quote[:55]}...\"")
    print()

    # ── Step 2: Store claims ──
    print("💾 Step 2: Saving to ClaimStore...\n")
    store.save_claims(claims)
    print(f"   → {len(claims)} claims saved to {os.path.basename(db_path)}\n")

    # ── Step 3: Query the registry ──
    print("🔍 Step 3: Querying claims...\n")

    # Who supplies to Nvidia?
    nvidia_suppliers = store.search_claims(object_contains="NVDA")
    print(f"   Nvidia suppliers ({len(nvidia_suppliers)} found):")
    for c in nvidia_suppliers:
        print(f"     • {c.claim_text}")
    print()

    # What involves TSMC?
    tsmc_claims = store.search_claims(subject_contains="TSMC")
    print(f"   TSMC claims ({len(tsmc_claims)} found):")
    for c in tsmc_claims:
        print(f"     • [{c.claim_type.value}] {c.claim_text}")
    print()

    # ── Step 4: Compute confidence ──
    print("📊 Step 4: Scoring confidence...\n")
    scorer = ConfidenceScorer()
    for c in claims:
        result = scorer.score(c)
        c.confidence = result
        c.compute_status()
        icon = "✅" if c.final_status.value in ("SUPPORTED", "PENDING_REVIEW") else "⚠️"
        print(f"   {icon} {c.claim_text[:55]}...")
        print(f"      confidence: {result.score:.2f} ({result.level})  |  status: {c.final_status.value}")
        for f in result.factors:
            print(f"        {f.direction:>8s} {f.contribution:+.2f}  {f.name}: {f.explanation}")
        print()

    # ── Done ──
    print("=" * 60)
    print("  ✅ Quickstart complete!")
    print(f"  Claims stored: {len(claims)}")
    print(f"  Every claim is traceable back to its source document.")
    print("=" * 60)

    # Cleanup
    os.remove(db_path)


if __name__ == "__main__":
    main()
