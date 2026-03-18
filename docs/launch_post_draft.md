# Graphite: Open-Source Claim Verification Engine

*We're open-sourcing Graphite — the engine behind EdgarOS — as an Apache 2.0 claim verification primitive.*

---

## Why we changed the framing

When we started Graphite, we called it a "graph engine for evidence-backed relationships." That's technically accurate, but it doesn't answer the question people actually ask:

**"What does it do?"**

The honest answer: Graphite turns documents into structured claims with provenance, then lets you verify whether downstream assertions are actually grounded in evidence.

That's a **claim verification engine**, not a graph engine.

---

## What's in the box

Three primitives, zero dependencies beyond `networkx` and `pydantic`:

```python
from graphite import Claim, ClaimStore, Provenance
from graphite.enums import SourceType, ConfidenceLevel

claim = Claim(
    subject_entities=["company:TSMC"],
    predicate="SUPPLIES_TO",
    object_entities=["company:NVDA"],
    claim_text="TSMC supplies advanced CoWoS packaging to Nvidia.",
    claim_type=ClaimType.RELATIONSHIP,
    supporting_evidence=[Provenance(
        source_id="tsmc-10k-2024",
        source_type=SourceType.SEC_10K,
        evidence_quote="The Company provides advanced packaging services including CoWoS.",
        confidence=ConfidenceLevel.HIGH,
    )],
)

store = ClaimStore(db_path="claims.db")
store.save_claim(claim)
```

Every claim carries its source. Every source carries the exact quote. No black-box assertions.

---

## How we use it

**EdgarOS** — the first commercial app built on Graphite — verifies AI-generated financial research memos against SEC evidence.

In our benchmark (100 synthetic memos, 320 claims):

| Metric | Score |
|--------|-------|
| Conservative Precision | 90.5% |
| False Contradiction Rate | 0.0% |

The verification logic, gold sets, and prompts are proprietary. The claim primitives are open.

---

## Get started

```bash
pip install graphite-engine
```

```bash
git clone https://github.com/graf-research/graphite.git
cd graphite
pip install -e .
python examples/quickstart_verification/run.py
```

Apache 2.0. Two dependencies. 122 tests.

→ [github.com/graf-research/graphite](https://github.com/graf-research/graphite)
