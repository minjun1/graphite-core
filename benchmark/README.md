# Graphite SEC Benchmark

A claim-level verification benchmark for AI-generated investment memos, built on public SEC filings.

## Overview

| Metric | Value |
|---|---|
| Domain | SEC / Investment Memos |
| Cases | 30 memos |
| Claims | 120–180 (4–6 per memo) |
| Label types | Claim verdict, Argument verdict, Failure mode |
| Annotators | 3 (1 primary + 2 reviewers) |
| Generation | Semi-synthetic (real sources + controlled errors) |

## Benchmarks

### 1. Claim Verification

Evaluates claim-level fact-checking accuracy.

| System | Precision | Recall | F1 | Contradiction Recall |
|---|---|---|---|---|
| Baseline A (LLM judge) | — | — | — | — |
| Baseline B (stateless) | — | — | — | — |
| Graphite (stateful) | — | — | — | — |

### 2. Stateful Memory Ablation

Measures the value of graph memory across review cycles.

| Metric | Stateless | Stateful | Delta |
|---|---|---|---|
| Duplicate detection rate | — | — | — |
| Evidence reuse rate | — | — | — |
| Repeat review reduction | — | — | — |

### 3. Human Agreement

Agreement between Graphite verdicts and human reviewers (n=40–60 claims).

| Metric | Value |
|---|---|
| Claim verdict agreement | — |
| Cohen's kappa | — |
| needs_human_review precision | — |
| needs_human_review recall | — |

## Dataset

- [`annotation_guide.md`](annotation_guide.md) — label definitions and annotator instructions
- [`dataset_schema.json`](dataset_schema.json) — JSON Schema for case format
- `sec_cases.jsonl` — full benchmark dataset (generated in Week 3–4)
- `sec_cases_pilot.jsonl` — 10-case pilot set (generated in Week 1)

## Label Distribution (Target)

```
SUPPORTED:       40%  (~60–72 claims)
CONFLICTED:      30%  (~45–54 claims)
INSUFFICIENT:    20%  (~30–36 claims)
CONCLUSION_JUMP: 10%  (~15–18 claims)
```

Within CONFLICTED:
- `numeric_mismatch` — ~⅓
- `temporal_mismatch` — ~⅓
- `paraphrased_contradiction` — ~⅓

## Running Evaluations

```bash
# Claim verification benchmark
python benchmark/run_claim_eval.py --dataset benchmark/sec_cases.jsonl

# Stateful vs stateless ablation
python benchmark/run_stateful_ablation.py --dataset benchmark/sec_cases.jsonl

# Human agreement analysis
python benchmark/run_human_agreement.py \
  --predictions benchmark/results/predictions.jsonl \
  --annotations benchmark/results/human_annotations.jsonl
```

## Failure Mode Taxonomy

| Mode | Description |
|---|---|
| `numeric_mismatch` | Numbers in the claim don't match the source |
| `temporal_mismatch` | Dates or time periods are incorrect |
| `paraphrased_contradiction` | Claim rephrases a fact but distorts meaning |
| `unsupported_extrapolation` | Claim extends beyond what data supports |
| `missing_evidence` | Source doesn't contain relevant information |
| `retrieval_noise` | Retrieved evidence is irrelevant |

## Example Headline Results

> **SEC memo verification benchmark** (n=180 claims across 36 memos)
> - Claim-level F1: **X**
> - Conclusion-jump agreement with reviewers: **Y%**
> - Stateful memory reduced duplicate review load by **Z%**
