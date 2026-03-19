# Graphite SEC Benchmark — Annotation Guide

> Version 0.1 · Last updated 2026-03-18

## Overview

You will review **AI-generated investment memos** alongside their **source documents** (SEC filings, earnings call excerpts, press releases). For each claim in a memo, you will assign:

1. A **claim verdict** — is the claim factually supported by the source?
2. An **argument verdict** — is the reasoning/logic sound?
3. A **needs_human_review** flag — would a professional analyst want to double-check this?
4. **Cited span IDs** — which parts of the source actually support or contradict the claim?
5. A **failure mode** label — what type of error is present (if any)?

---

## 1. Claim Verdicts

Each claim gets exactly one verdict.

### SUPPORTED

The claim is **directly and accurately supported** by the source documents.

**Criteria:**
- The core factual assertion matches the source
- Numbers, dates, and entities are correct
- No material omissions that would change the meaning

**Example:**
> **Claim:** "Apple reported $89.5B in revenue for Q4 FY2024."
> **Source:** "Net revenue for the quarter ended September 28, 2024, was $89.5 billion."
> **Verdict:** SUPPORTED

### CONFLICTED

The claim **contradicts** information in the source documents.

**Criteria:**
- A specific fact in the claim is directly contradicted by the source
- Numbers are wrong (not just rounded)
- Entities or attributions are incorrect
- Temporal claims are contradicted by more recent data

**Example:**
> **Claim:** "Tesla's gross margin improved to 22% in Q3 2024."
> **Source:** "Automotive gross margin was 17.1% in Q3 2024, down from 18.7% in Q2."
> **Verdict:** CONFLICTED

### INSUFFICIENT

The source documents **do not contain enough information** to confirm or deny the claim.

**Criteria:**
- The topic is mentioned but the specific assertion cannot be verified
- The claim makes a forward-looking statement the source doesn't address
- Relevant data is simply missing from the provided sources

**Example:**
> **Claim:** "NVIDIA's data center revenue will exceed $40B in FY2026."
> **Source:** (only contains FY2025 actual results, no forward guidance at that level)
> **Verdict:** INSUFFICIENT

### Edge Case: SUPPORTED vs INSUFFICIENT

If the source **partially** supports a claim but key details are missing:
- If the core assertion is verifiable → **SUPPORTED** (note partial coverage)
- If the missing details are material to the claim's meaning → **INSUFFICIENT**

### Edge Case: CONFLICTED vs INSUFFICIENT

If the source **seems** to contradict but the evidence is weak or indirect:
- If there is a **direct** factual contradiction → **CONFLICTED**
- If the contradiction only emerges through inference → **INSUFFICIENT** (not enough evidence to call it conflicted)

---

## 2. Argument Verdicts

Each claim also gets an argument-level verdict evaluating the **reasoning quality**.

### GROUNDED

The claim's reasoning is **logically valid** given the evidence.

**Criteria:**
- The conclusion follows from the cited facts
- Cause-and-effect relationships are reasonable and supported
- No logical leaps

### CONCLUSION_JUMP

The claim makes a **logical leap** that is not supported by the evidence.

**Criteria:**
- The claim takes a narrow fact and draws an overly broad conclusion
- Causal reasoning is unsupported ("A happened, therefore B will happen")
- The conclusion is plausible but not warranted by the evidence alone

**Example:**
> **Claim:** "Revenue increased 3% QoQ, proving that the company's AI strategy is paying off."
> **Source:** "Revenue rose 3% in Q3, driven primarily by seasonal demand in the consumer segment."
> **Verdict:** CONCLUSION_JUMP — the revenue increase is factual, but attributing it to AI strategy is not supported.

### OVERSTATED

The claim **exaggerates** what the evidence actually says.

**Criteria:**
- Directionally correct but magnitude is inflated
- Qualified language in the source is presented as absolute
- "Strong" evidence is called "definitive"; "growth" is called "explosive"

**Example:**
> **Claim:** "The company has achieved dominant market share in cloud computing."
> **Source:** "The company holds approximately 32% market share, ranking third behind AWS and Azure."
> **Verdict:** OVERSTATED — 32% third-place is not "dominant."

---

## 3. needs_human_review Flag

Set `needs_human_review: true` when:

| Condition | Example |
|---|---|
| Verdict confidence is low | Borderline SUPPORTED/INSUFFICIENT |
| Claim involves material financial figures | Revenue, earnings per share, guidance |
| Subtle contradiction that requires domain expertise | Accounting methodology changes |
| Forward-looking claim with regulatory implications | SEC disclosure-related assertions |
| Multiple sources partially contradict each other | 10-K says one thing, earnings call implies another |

**Default is `false`.** Only flag claims that genuinely warrant a second opinion.

---

## 4. Cited Span Selection

For each claim, identify which `source_doc` paragraphs are relevant.

**Rules:**
- Select the **minimal set** of spans that justify your verdict
- Include spans that **support** the claim (for SUPPORTED verdicts)
- Include spans that **contradict** the claim (for CONFLICTED verdicts)
- For INSUFFICIENT, include the **closest relevant** span and note what's missing
- Use the `span_id` values from the source documents

---

## 5. Failure Mode Taxonomy

For claims that are NOT `SUPPORTED + GROUNDED`, tag the specific failure mode:

| Failure Mode | Definition | Example |
|---|---|---|
| `numeric_mismatch` | A number in the claim doesn't match the source | $150M vs $15M |
| `temporal_mismatch` | A date, time period, or temporal relationship is wrong | "Current CEO" but person stepped down |
| `paraphrased_contradiction` | The claim rephrases a fact but reverses or distorts its meaning | "Exclusively uses AWS" when source says "migrating away from AWS" |
| `unsupported_extrapolation` | The claim extrapolates beyond what the data shows | "Q3 growth proves AI strategy works" |
| `missing_evidence` | The source simply doesn't contain info to verify the claim | Forward guidance not provided |
| `retrieval_noise` | The retrieved evidence is irrelevant to the claim | Evidence about a different company/period |

For `SUPPORTED + GROUNDED` claims, set `failure_mode: null`.

---

## 6. Annotation Workflow

1. **Read the source documents** first — understand context before looking at claims
2. **Read each claim** and decide the verdict
3. **Select cited spans** that justify your verdict
4. **Assign the argument verdict** — is the reasoning valid even if the facts check out?
5. **Tag failure mode** if applicable
6. **Set needs_human_review** if the case is borderline

**Time estimate:** ~5–8 minutes per memo case (4–6 claims), ~2.5–4 hours for 30 cases.

---

## 7. Disagreement Resolution

If two annotators disagree on a verdict:

1. Cases where both annotators agree → **accepted as gold**
2. Cases with disagreement → reviewed by the adjudicator (primary author)
3. Persistent ambiguity → document in `adjudication_notes.md` and assign best-judgment label
4. Systematic disagreement on a label definition → revise this guide

---

## Quick Reference Card

```
Claim Verdicts:     SUPPORTED | CONFLICTED | INSUFFICIENT
Argument Verdicts:  GROUNDED  | CONCLUSION_JUMP | OVERSTATED
Failure Modes:      numeric_mismatch | temporal_mismatch | paraphrased_contradiction
                    unsupported_extrapolation | missing_evidence | retrieval_noise
needs_human_review: true if borderline, material, or requires domain expertise
```
