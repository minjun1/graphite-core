#!/usr/bin/env python3
"""
Quickstart: Claim Verification — End-to-End Graphite Demo

Verify an AI-generated memo against a corpus of factual documents.
Extracts claims → retrieves evidence → verifies support → flags logic leaps

Run: python examples/quickstart_verification/run.py
"""

import os
import sys

# Ensure graphite is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from graphite.pipeline.verdict import ArgumentVerdictEnum, VerdictEnum
from graphite.pipeline import verify_agent_output


def main():
    print("🚀 Graphite Quickstart: LLM-Native Verification\n")

    # ── Step 1: The AI-generated memo (Agent Output) ──
    agent_memo = """
    # Investment Memo: TSMC & Nvidia
    
    TSMC supplies advanced CoWoS packaging to Nvidia. Because of this, 
    Nvidia is highly dependent on TSMC for its AI chips. 
    TSMC is currently capacity-constrained in CoWoS packaging, which means 
    Nvidia's revenue will drop by 50% next quarter.
    """

    print("📄 AGENT MEMO:")
    print(agent_memo.strip())
    print("-" * 60)

    # ── Step 2: The Evidence Corpus (e.g. SEC Filings) ──
    corpus = [
        {
            "document_id": "tsmc-10k-2024",
            "text": "The Company provides advanced packaging services including CoWoS to major customers. CoWoS advanced packaging capacity remains fully utilized and constrained for the near term.",
        },
        {
            "document_id": "nvda-10k-2024",
            "text": "We rely on third-party suppliers such as TSMC for wafer fabrication and CoWoS packaging. Supply constraints could impact our ability to meet demand, but we have secured advance capacity and expect revenue to grow sequentially.",
        },
    ]

    print("🔍 VERIFYING AGAINST CORPUS...")

    try:
        # ── Step 3: Run the Hero API Pipeline ──
        # Note: You need the OPENAI_API_KEY set for this to run in reality.
        report = verify_agent_output(agent_memo, corpus, model="gemini-2.5-flash")

        print("\n📊 VERIFICATION REPORT:")
        print(f"   Total Claims: {report.total_claims}")
        print(
            f"   Supported: {report.supported_count} | Conflicted: {report.conflicted_count} | Insufficient: {report.insufficient_count}"
        )

        print("\n✅ SUPPORTED CLAIMS:")
        supported_claims = [
            v for v in report.verdicts if v.verdict == VerdictEnum.SUPPORTED
        ]
        if not supported_claims:
            print("   (None)")
        for v in supported_claims:
            print(f"   [✓] {v.claim_text}")

        print("\n🚨 HUMAN REVIEW QUEUE:")
        if not report.risky_claim_ids and report.conclusion_jump_count == 0:
            print("   (No high-risk claims or logic leaps detected)")

        for claim_id in report.risky_claim_ids:
            v = report.get_verdict(claim_id)
            print(f"   [RISK] {v.claim_text}")
            reason_detail = (
                v.rationale.missing_evidence_reason
                or v.rationale.contradiction_type
                or v.rationale.text
            )
            print(f"          Reason: {reason_detail}")
            print(f"          Flags: Human Review Needed={v.needs_human_review}")

        for av in report.argument_verdicts:
            if av.verdict == ArgumentVerdictEnum.CONCLUSION_JUMP:
                print(f"   [LEAP] {av.text}")
                print(f"          Reason: {av.rationale.text}")

    except ImportError:
        print(
            '\n⚠️  [Dependencies Missing] Please install LLM support: `pip install "graphite-engine[llm]"`'
        )
    except Exception as e:
        print(f"\n⚠️  [Execution Error] Requires OPENAI_API_KEY in environment. ({e})")

    print("\n" + "=" * 60)
    print("  ✅ Quickstart complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
