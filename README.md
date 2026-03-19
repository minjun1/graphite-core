  <div align="center">
    <h1>⛏️ Graphite</h1>
    <p><strong>Claim verification engine for AI agent outputs.</strong></p>
    <p><em>LLMs judge. Graphs remember.</em></p>
    <p>Graphite extracts claims from agent-generated text, retrieves evidence, verifies support and contradiction, flags unsupported reasoning leaps, and stores every verdict with a full provenance trail — building a verification memory that gets stronger with every review.</p>
    <p>
      <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
      <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.10%2B-brightgreen.svg" alt="Python"></a>
    </p>
  </div>

  > ⚠️ **v0.3.x — Experimental**. Usable and tested, but API may change before 1.0. Pin your version.

  ---

  ### How it works

  Graphite turns raw agent output into a structured verification report.

  ```python
  from graphite.pipeline import verify_agent_output

  report = verify_agent_output(
      text=agent_memo_markdown,
      corpus=sec_filings_corpus,
      model="gemini-2.5-flash"  # any OpenAI-compatible model works
  )

  print(f"Supported: {report.supported_count} | Conflicted: {report.conflicted_count}")
  print(f"Requires Human Review: {len(report.risky_claim_ids)} claims")
  ```

  This single API wraps a 5-step pipeline:

  1. **Extract**: Parses the document into atomic claims using LLMs.
  2. **Retrieve**: Finds candidate evidence spans across the corpus for each claim.
  3. **Verify**: Judges claims against the retrieved spans (Supported, Conflicted, Insufficient).
  4. **Analyze**: Flags argument-level reasoning leaps (`CONCLUSION_JUMP`).
  5. **Report**: Aggregates the findings into a `VerificationReport` with structured rationale, review flags, and full provenance.

  ---

  ### Handling Logic Leaps & Human Review

  Downstream UI and review workflows can be built directly on top of the structured output.

  ```python
  from graphite.claim import ArgumentVerdictEnum

  # Check for logic leaps (Argument-level verification)
  for argument in report.argument_verdicts:
      if argument.verdict == ArgumentVerdictEnum.CONCLUSION_JUMP:
          print(f"⚠️ LOGIC LEAP: {argument.text}")

  # Route high-risk factual claims to a Human-in-the-loop review queue
  for claim_id in report.risky_claim_ids:
      verdict = report.get_verdict(claim_id)
      if verdict.needs_human_review:
          print(f"🚨 REVIEW NEEDED: {verdict.claim_text}")
          print(f"   Reason: {verdict.rationale.missing_evidence_reason or verdict.rationale.contradiction_type}")
  ```

  ---

  ### Quickstart

  ```bash
  pip install "graphite-engine[llm]"
  export GEMINI_API_KEY="your-api-key-here"
  python examples/quickstart_verification/run.py
  ```

  Or from source:

  ```bash
  git clone https://github.com/graf-research/graphite.git
  cd graphite
  pip install -e ".[llm]"
  export GEMINI_API_KEY="your-api-key-here"
  python examples/quickstart_verification/run.py
  ```

  Graphite defaults to Gemini via the OpenAI-compatible endpoint, so any OpenAI-compatible provider also works — including local models via Ollama or vLLM, or hosted endpoints like Together and Groq. Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` to point at any compatible endpoint.

  ---

  ## Why a Graph?

  Most verification tools run once and forget. Graphite anchors every judgment into a persistent graph — turning disposable LLM outputs into a living verification memory.

  **Claims are first-class objects.** The same assertion can be identified, revisited, and re-evaluated across documents and time — not lost in prompt logs.

  **Evidence accumulates, not overwrites.** When a second source confirms (or contradicts) a claim, Graphite appends the new evidence to the existing node instead of starting from scratch.

  **Review history becomes lineage.** AI verdict → analyst override → re-evaluation with new data — every step is recorded as a relationship in the graph, not a flat log entry.

  **Cross-document deduplication.** When the same claim appears in TSMC's 10-K and Nvidia's 10-K, Graphite recognizes it as one canonical claim backed by two independent sources.

  **Reasoning structure, not just fact-checking.** Claims don't exist in isolation. Graphite can represent claim-to-conclusion relationships, enabling checks like `CONCLUSION_JUMP` when the logical link between premises and conclusion is unsupported.

  ---

  ## Stateful Verification Memory

  Unlike stateless evaluators that produce a score and discard context, Graphite's `ClaimStore` builds a persistent fact base that strengthens over time.

  ```
  Run 1: Extract "TSMC supplies CoWoS to Nvidia" from TSMC 10-K
          → 1 evidence source recorded

  Run 2: Same claim found in Nvidia 10-K
          → evidence accumulates → 2 independent sources

  Run 3: Exact duplicate from same source
          → deduplicated, no change

  Run 4: Related claim "Nvidia depends on TSMC" extracted
          → cross-claim linkage via shared entities
  ```

  What this looks like as a graph:

  ```
  Claim: "TSMC supplies CoWoS to Nvidia"
    ├── supported_by → TSMC 10-K (cited span)
    ├── supported_by → Nvidia 10-K (cited span)
    ├── reviewed_as → SUPPORTED (model verdict)
    └── contributes_to → "Nvidia depends on TSMC"
  ```

  Each claim is a deduplicated node. Evidence merges across extraction runs. Analyst overrides persist. The result is a verification memory where repeated reviews compound — not repeat.

  *Most verification tools forget. Graphite remembers — and gets stronger with every review.*

  *(See `examples/evidence_accumulation/` for a runnable demo — no API keys required.)*

  ---

  ## Graphite vs. Existing Tools

  *Evaluators grade your prompts. Graphite audits your agent's claims — and remembers every verdict.*

  These tools solve adjacent but different problems:

  | Dimension | Ragas / TruLens / DeepEval | Graphite |
  |-----------|---------------------------|----------|
  | **Purpose** | Prompt/model evaluation (CI/CD) | Production output verification (runtime) |
  | **State** | Stateless — each run is independent | Stateful — evidence accumulates across runs |
  | **Output** | Scores (faithfulness, relevance) | Structured `VerificationReport` with provenance |
  | **Logic Leaps** | Not addressed | `CONCLUSION_JUMP` / `OVERSTATED` detection |
  | **Human Review** | Manual review of score dashboards | `needs_human_review` routing with analyst override |
  | **Audit Trail** | Execution logs | Every verdict links to exact `cited_span` with full lineage |

  ---

  ## Evaluation Snapshot

  Representative verification cases from our golden test suite (current snapshot run on GPT-4o; `evals/verify_eval.py`):

  | Test Case | Type | Expected | Graphite Output |
  |-----------|------|----------|-----------------|
  | Paraphrased contradiction | Semantic | CONFLICTED | CONFLICTED |
  | Numeric mismatch (10× error) | Factual | CONFLICTED | CONFLICTED |
  | Temporal mismatch (stale CEO) | Temporal | CONFLICTED | CONFLICTED |
  | Unsupported revenue prediction | Reasoning Leap | CONCLUSION_JUMP | CONCLUSION_JUMP |

  - Claim-level verdict: correct in 3/3 factual cases
  - Argument-level verdict: correct in 1/1 reasoning case
  - These cases are intended as regression checks for key failure modes, not as a broad accuracy benchmark.

  > *This is a representative snapshot, not a comprehensive benchmark. See `evals/` for the full test suite and `examples/` for runnable demos. A larger-scale evaluation suite (100+ memos) is on the roadmap.*

  ---

  ## Core Primitives

  | Object | What it does |
  |-----------|-------------|
  | `VerificationReport` | Top-level summary of the entire review, ready for product UI integrations |
  | `Verdict` | Claim-level judgment (`SUPPORTED`, `CONFLICTED`, `INSUFFICIENT`) with structured rationale |
  | `ArgumentVerdict` | Argument-level judgment (`GROUNDED`, `CONCLUSION_JUMP`, `OVERSTATED`) |
  | `ClaimStore` | Persistent verification memory — deduplicates claims, merges evidence, and preserves review history across runs |

  ---

  ## Reference Applications

  Graphite is designed as the verification engine for high-stakes workflows across multiple domains:

  - **Compliance & Legal Review**: Checking internal policy documents or marketing copy against regulatory guidelines.
  - **Healthcare & Scientific Fact-checking**: Cross-referencing generated medical or scientific summaries against peer-reviewed journals.
  - **Investment & Research QA**: Verifying AI-generated analyst memos against SEC filings or earnings call transcripts.

  *(See `examples/quickstart_verification/` for end-to-end verification, `examples/evidence_accumulation/` for stateful memory, and `examples/lineage_override_demo/` for analyst override workflows.)*

  ---

  ## Optional extras

  **Core** (always included): `networkx` + `pydantic`

  ```bash
  pip install -e ".[llm]"     # LLM support (OpenAI-compatible providers)
  pip install -e ".[all]"     # Everything
  ```

  > Set `GEMINI_API_KEY` to get started. To use other providers, set `OPENAI_API_KEY` and `OPENAI_BASE_URL`.

  ---

  ## License

  Apache-2.0 — see [LICENSE](LICENSE).
