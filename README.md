  <div align="center">
    <h1>⛏️ Graphite</h1>
    <p><strong>Claim verification engine for AI agent outputs.</strong></p>
    <p><em>Extract. Verify. Remember.</em></p>
    <p>Graphite extracts claims from agent-generated text, retrieves evidence, verifies support and contradiction, flags unsupported reasoning leaps, and stores every verdict with a full provenance trail — building a verification memory that gets stronger with every review.</p>
    <p>
      <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
      <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.10%2B-brightgreen.svg" alt="Python"></a>
    </p>
  </div>

  > ⚠️ **v0.4.x — Experimental**. Usable and tested, but API may change before 1.0. Pin your version.

  ---

  ## Quickstart

  ```bash
  pip install "graphite-engine[llm]"
  export GEMINI_API_KEY="your-api-key-here"
  ```

  ```python
  from graphite.pipeline import verify_agent_output

  report = verify_agent_output(
      text=agent_memo,
      corpus=evidence_documents,
  )

  print(f"Supported: {report.supported_count} | Conflicted: {report.conflicted_count}")
  print(f"Requires Human Review: {len(report.risky_claim_ids)} claims")
  ```

  Works with any major LLM provider:

  ```python
  # Gemini (default)
  report = verify_agent_output(text, corpus)

  # Claude
  report = verify_agent_output(text, corpus, model="claude-opus-4-6")

  # OpenAI
  report = verify_agent_output(text, corpus, model="gpt-4.1")

  # Local / open-source (Ollama, vLLM)
  report = verify_agent_output(text, corpus, model="llama-4-scout")
  ```

  Set the API key for your provider:
  - `GEMINI_API_KEY` — Google Gemini (default)
  - `ANTHROPIC_API_KEY` — Anthropic Claude
  - `OPENAI_API_KEY` + `OPENAI_BASE_URL` — OpenAI or any compatible endpoint

  ---

  ## How It Works

  A single `verify_agent_output()` call runs a 5-step pipeline:

  1. **Extract** — Parses the document into atomic claims via LLM.
  2. **Retrieve** — Finds candidate evidence spans across the corpus.
  3. **Verify** — Judges each claim against retrieved evidence (Supported, Conflicted, Insufficient).
  4. **Analyze** — Flags argument-level reasoning leaps (`CONCLUSION_JUMP`, `OVERSTATED`).
  5. **Report** — Aggregates into a `VerificationReport` with structured rationale, review flags, and provenance.

  Domain-specific prompts? Pass a `PromptSet`:

  ```python
  from graphite.pipeline.prompts import PromptSet

  medical_prompts = PromptSet(
      extractor="You are a medical claim extractor. Extract clinical assertions...",
      verifier="You are a clinical evidence checker. Evaluate against guidelines...",
  )
  report = verify_agent_output(text, corpus, prompts=medical_prompts)
  ```

  ---

  ## Logic Leaps & Human Review

  ```python
  from graphite.pipeline.verdict import ArgumentVerdictEnum

  # Detect logic leaps
  for argument in report.argument_verdicts:
      if argument.verdict == ArgumentVerdictEnum.CONCLUSION_JUMP:
          print(f"LOGIC LEAP: {argument.text}")

  # Route risky claims to human review
  for claim_id in report.risky_claim_ids:
      verdict = report.get_verdict(claim_id)
      if verdict.needs_human_review:
          print(f"REVIEW NEEDED: {verdict.claim_text}")
          print(f"  Reason: {verdict.rationale.missing_evidence_reason or verdict.rationale.contradiction_type}")
  ```

  ---

  ## Verification Memory

  Most verification tools run once and forget. Graphite's `ClaimStore` builds a persistent fact base that strengthens over time.

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

  ```
  Claim: "TSMC supplies CoWoS to Nvidia"
    ├── supported_by → TSMC 10-K (cited span)
    ├── supported_by → Nvidia 10-K (cited span)
    ├── reviewed_as → SUPPORTED (model verdict)
    └── contributes_to → "Nvidia depends on TSMC"
  ```

  **Claims are first-class objects** — identified, revisited, and re-evaluated across documents and time.
  **Evidence accumulates** — new sources merge into existing claims, never overwrite.
  **Analyst overrides persist** — human decisions survive machine recomputation.
  **Cross-document dedup** — same claim from different filings = one canonical claim, multiple sources.

  *(See `examples/evidence_accumulation/` for a runnable demo — no API keys required.)*

  ---

  ## Graphite vs. Existing Tools

  | Dimension | Ragas / TruLens / DeepEval | Graphite |
  |-----------|---------------------------|----------|
  | **Purpose** | Prompt/model evaluation (CI/CD) | Production output verification (runtime) |
  | **State** | Stateless — each run is independent | Stateful — evidence accumulates across runs |
  | **Output** | Scores (faithfulness, relevance) | Structured `VerificationReport` with provenance |
  | **Logic Leaps** | Not addressed | `CONCLUSION_JUMP` / `OVERSTATED` detection |
  | **Human Review** | Manual review of dashboards | `needs_human_review` routing with analyst override |
  | **Audit Trail** | Execution logs | Every verdict links to exact `cited_span` with full lineage |

  ---

  ## Eval Framework

  Built-in benchmarking across models and domains:

  ```python
  from graphite.eval import EvalRunner

  runner = EvalRunner.from_json("src/graphite/eval/datasets/base.json")
  run = runner.run(model="gemini-2.5-pro", output_path="results.json")
  print(run.metrics())
  # {'total': 4, 'claim_verdict_accuracy': 0.75, 'overall_pass_rate': 0.5, ...}
  ```

  Supports domain filtering, custom prompts via `PromptSet`, latency tracking, and JSON result persistence.

  | Test Case | Type | Expected Claim | Expected Argument |
  |-----------|------|----------------|-------------------|
  | Paraphrased contradiction | Semantic | CONFLICTED | GROUNDED |
  | Numeric mismatch (10x error) | Factual | CONFLICTED | GROUNDED |
  | Temporal mismatch (stale CEO) | Temporal | CONFLICTED | GROUNDED |
  | Unsupported revenue prediction | Reasoning Leap | CONFLICTED | CONCLUSION_JUMP |

  ---

  ## Core Primitives

  | Object | What it does |
  |-----------|-------------|
  | `VerificationReport` | Top-level summary, ready for product UI integrations |
  | `Verdict` | Claim-level judgment (`SUPPORTED`, `CONFLICTED`, `INSUFFICIENT`) with structured rationale |
  | `ArgumentVerdict` | Argument-level judgment (`GROUNDED`, `CONCLUSION_JUMP`, `OVERSTATED`) |
  | `ClaimStore` | Persistent verification memory — deduplicates claims, merges evidence, preserves review history |
  | `PromptSet` | Configurable system prompts per pipeline stage |
  | `EvalRunner` | Structured benchmarking — datasets, metrics, model comparison |

  ---

  ## Install

  ```bash
  pip install graphite-engine             # core only (pydantic)
  pip install "graphite-engine[llm]"      # + openai, anthropic
  pip install "graphite-engine[all]"      # everything
  ```

  Or from source:

  ```bash
  git clone https://github.com/minjun1/graphite-core.git
  cd graphite-core
  pip install -e ".[llm]"
  ```

  ---

  ## License

  Apache-2.0 — see [LICENSE](LICENSE).
