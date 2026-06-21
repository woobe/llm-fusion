# llm-fusion

**Multi-scenario fusion pipeline for Hermes Agent** — dispatches 6 parallel LLM calls,
cleans and deduplicates responses, and synthesizes a single superior answer using a
judge model with chain-of-thought reasoning.

Supports 8 scenarios: `coding`, `bugfix`, `qa`, `plan_review`, `creative`,
`reasoning`, `document`, `general`. Each scenario has its own temperature profile,
token budget, cleaning rules, and judge strategy.

---

## Quick Start

```bash
# Install with one-liner
bash -c "$(curl -fsSL https://raw.githubusercontent.com/woobe/llm-fusion/main/install.sh)"
```

Scenarios are auto-detected. To force a specific scenario:
```
/llm-fusion coding: Write a Python function to sort a list
```

---

## Architecture Overview

```
user query
    │
    ▼
┌─────────────────────────────┐
│  1. Classifier              │  regex pre-classifier (+ optional LLM 2nd pass)
│     └► scenario_id          │  8 scenarios, auto-detected
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  2. Panel (6 parallel)      │  ThreadPoolExecutor — 3× deepseek + 3× mimo
│     ├── deepseek-v4-flash #1 │  temperature varies per instance for diversity
│     ├── deepseek-v4-flash #2 │
│     ├── deepseek-v4-flash #3 │
│     ├── mimo-v2.5 #1        │
│     ├── mimo-v2.5 #2        │
│     └── mimo-v2.5 #3        │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  3. Cleaner                  │  preamble strip, fence handling, dedup
│     └► cleaned_responses    │  scenario-specific thresholds
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  4. Judge                    │  single or two-stage synthesis
│     └► final answer          │  temp=0.0, high reasoning mode
└─────────────────────────────┘
    │
    ▼
  formatted output + metadata
```

---

## Design Decisions

### Panel vs. Judge (Why Two Roles?)

The pipeline separates **exploration** (panel) from **convergence** (judge).

- **Panel** — 6 diverse calls at varied temperatures. Each call is independent.
  The panel explores the answer space from different "angles" — some conservative
  (low temp), some creative (high temp), some via a different model entirely.
  The goal is breadth of coverage.

- **Judge** — a single deterministic (temp=0.0) call with high reasoning mode
  that reads all panel responses and synthesizes the best possible answer.
  The judge does not generate from scratch — it evaluates, compares, and
  synthesizes from the panel's raw material. This is fundamentally different
  from asking one model to answer directly, because the judge can:
  - Discard hallucinations from individual panel responses by cross-referencing
  - Combine the strongest parts of different responses
  - Fill gaps where one model covered something others missed
  - Catch contradictions and resolve them

**Why not just use one large model directly?** A single call gives one
"opinion". With 6 diverse responses feeding into a judge, the pipeline
gets a supermajority effect — errors and blindspots from any single model
are caught by the others. This is particularly effective for:
- **Coding**: one model's buggy solution is caught by the others
- **Factual QA**: hallucinated facts are contradicted by correct responses
- **Reasoning**: step-by-step errors are identified through cross-verification

### Temperature Diversity (Why 3× per model at different temps?)

Temperature controls the randomness of token sampling. Low temp (0.2–0.4)
produces conservative, predictable outputs; high temp (0.8–1.0) produces
more creative, varied outputs. The panel deliberately spans a range:

**Scenario temperature rationale:**

| Scenario | deepseek temp | mimo temps | Why |
|---|---|---|---|
| coding | 0.5 | [0.4,0.5,0.6] | Precision-focused, low creativity needed |
| bugfix | 0.4 | [0.4,0.5,0.6] | Very conservative — bugs need precise diagnosis |
| qa | 0.3 | [0.3,0.4,0.5] | Lowest temp — facts need accuracy, not creativity |
| plan_review | 0.6 | [0.6,0.7,0.8] | Moderate — needs analysis breadth + creativity |
| creative | 1.0 | [0.8,0.9,1.0] | Highest — wants originality and diversity |
| reasoning | 0.5 | [0.5,0.6,0.7] | Low — precision matters, but moderate diversity helps |
| document | 0.5 | [0.5,0.6,0.7] | Moderate — rewrite needs both precision and creativity |
| general | 0.75 | [0.6,0.7,0.8] | Default balance between accuracy and creativity |

For deepseek-v4-flash, all 3 copies use the same temp (diversity comes from
model stochasticity + prompt differences). For mimo-v2.5, each of the 3 copies
gets a different temp from the spread, explicitly forcing different "creative
modes". This asymmetry is deliberate — deepseek is the stronger model so
stochasticity alone provides enough variety; mimo benefits from explicit temp
variation to compensate for being a smaller model.

### Two-Stage Judge (When and Why)

Some scenarios use a **single-stage judge** (one call that produces the final
answer directly). Others use a **two-stage judge** (analysis first, then
synthesis):

| Judge strategy | Scenarios |
|---|---|
| Single-stage | coding, qa, creative, general |
| Two-stage | bugfix, plan_review, reasoning, document |

**Single-stage** is used when:
- The answer is relatively factual or creative (synthesis is straightforward)
- The panel responses can be consumed directly in one pass
- Latency matters more than depth (QA, creative)

**Two-stage** is used when:
- The task requires multi-step analysis (bug diagnosis, plan review)
- Step-by-step verification is needed (reasoning)
- A structured intermediate analysis improves the final output (document review)

Stage 1 produces a structured analysis (root cause, evidence, contradictions,
gaps). Stage 2 consumes this analysis alongside the raw responses to produce
the final answer. This decomposition works better than a single pass because:

1. Stage 1's reasoning budget isn't split between analysis and synthesis
2. The analysis can be verified independently before synthesis
3. Stage 2 operates on a distilled summary rather than raw 6-document text

Deep reasoning scenarios (reasoning, plan_review, document) use `reasoning_mode:
max` which enables the model's deepest chain-of-thought — this is expensive
(10k–16k completion tokens) but necessary for correctness.

### Model Selection Rationale

**deepseek-v4-flash** is the primary model used in both panel and judge roles:
- Fast inference (flash model) keeps pipeline latency acceptable
- Supports `reasoning_mode` parameter (none/low/high/max) for chain-of-thought
- Supports `max_completion_tokens` for precise token budgeting
- Strong coding and reasoning capabilities
- Competitive pricing vs. frontier models

**mimo-v2.5** is a secondary model used only in the panel:
- Smaller, cheaper model — adds diversity at marginal cost
- Panel-only role: it contributes candidate responses but never makes
  synthesis decisions
- Thinking disabled (`thinking.type: disabled`) to avoid excessive token use
- Uses `max_tokens` (not `max_completion_tokens`) — different token budget
  param reflecting API differences between providers

The pipeline is **not model-agnostic by design** — it specifically pairs a
strong primary model (deepseek) with a diverse secondary model (mimo). This
asymmetric pairing produces better results than two equally strong models
because the weaker model provides genuinely different "blindspots" that the
strong judge can identify and compensate for.

### opencode.go Focus

The pipeline targets **OpenCode Go** (`https://opencode.ai/zen/go/v1/chat/completions`)
as its primary LLM provider:

- **OpenAI-compatible API** — standard chat completions endpoint format
- **Supports reasoning_mode** — native chain-of-thought parameter that
  controls reasoning depth (none/low/high/max)
- **Dual token budget** — supports both `max_tokens` (legacy) and
  `max_completion_tokens` (modern, reasoning-aware)
- **Low-latency inference** — flash models respond in 1–5 seconds for
  moderate token budgets
- **Competitive pricing** — viable for 6x parallel panel calls + 1–2 judge
  calls per query

The pipeline uses stdlib `urllib` only — no requests, aiohttp, or httpx.
This means zero extra Python dependencies beyond PyYAML (for config).
The tradeoff: no async I/O, so parallelism is via `ThreadPoolExecutor`
rather than asyncio. For 6 calls this is perfectly adequate.

**Fallback to OpenRouter** is supported in `fallback.py` via
`call_with_fallback()`, but is not wired into the main pipeline by default.
The primary endpoint is the first-class path.

### Graceful Degradation

The pipeline never raises exceptions. Every module returns a well-structured
dict with a `success` bool. This feeds the graceful degradation chain:

1. **Panel fails** (< min_survivors=2 successful responses) → direct LLM call
   with moderate settings as fallback
2. **Cleaning leaves too few survivors** (< 2) → same direct fallback
3. **Judge fails** → direct fallback
4. **Everything fails** → error result with `success: false` and descriptive
   `error` field

When graceful degradation is disabled (`graceful_degradation: false`), any
failure halts the pipeline with an error — useful for testing and debugging.

The fallback uses a single `deepseek-v4-flash` call at temp=0.75 with 2000
max completion tokens — enough to answer most queries, but without the
quality benefits of the full fusion pipeline.

### Cleaning Pipeline (Why So Much Work on the Inputs?)

Raw LLM outputs are verbose and inconsistent. The cleaning pipeline transforms
them before the judge sees them:

1. **Preamble stripping** — removes "Here is my response:" / "Sure!" / "As an AI..."
   (scenario-specific regex patterns). This prevents the judge from wasting
   context on meta-commentary.

2. **Trailing meta stripping** — removes "Let me know if you have questions",
   "I hope this helps", etc. For some scenarios (plan_review, creative, document)
   trailing content is preserved because it may contain substantive material.

3. **Code fence handling** — for coding scenarios, fences are kept (they're
   structural). For others, fences are stripped but content preserved.

4. **Minimum word filtering** — responses below a scenario-specific threshold
   are discarded as too short to be useful:
   - QA: min 5 words (factual answers can be short)
   - Creative: min 10 words
   - Coding: min 15 words
   - Reasoning: min 20 words
   - Plan_review / Document: min 30 words (need depth)

5. **Deduplication via SequenceMatcher** — pairwise ratio comparison. If two
   responses exceed the dedup threshold (e.g., 0.85 for QA, 0.60 for creative),
   the shorter one is discarded. This prevents near-identical responses from
   dominating the judge's context window.

**Dedup threshold per scenario:**
- Creative: 0.60 (responses should be genuinely different)
- Coding: 0.70
- Bugfix / Reasoning: 0.75
- Plan_review / Document: 0.80
- QA / General: 0.85 (tight — factual answers are allowed to agree)

### Config Discovery (Portable, Zero-Config)

The config loader searches in this order:
1. `LLM_FUSION_CONFIG` environment variable
2. `./fusion_config.yaml` (cwd)
3. `./.llm-fusion/fusion_config.yaml` (hidden cwd dir)
4. `$XDG_CONFIG_HOME/llm-fusion/fusion_config.yaml` (XDG standard)
5. `~/.llm-fusion/fusion_config.yaml` (home hidden dir)
6. Bundled `assets/fusion_config.yaml.example` (defaults only)

This follows the XDG Base Directory specification and allows the pipeline to
work without any configuration file — defaults are compiled into the code
(via `_default_scenario_config()`).

### Rate Limiting

A token-bucket rate limiter (10 req/s, burst 20) is available in `fallback.py`
for fallback provider calls. The primary pipeline does not use rate limiting
because the 6 panel calls are made concurrently within a single pipeline
invocation, and the pipeline is typically run infrequently by a single user.
The rate limiter exists for scenarios where the pipeline might be called
programmatically in a loop.

---

## Requirements

- Python 3.10+
- PyYAML
- `OPENCODE_GO_API_KEY` in `~/.hermes/.env` or environment variable

---

## Configuration

See `assets/fusion_config.yaml` for the full configuration schema:

- **Panel temperatures and token budgets** per scenario
- **Judge settings** — model, temperature, reasoning mode, max tokens
- **Cleaning profiles** — preamble patterns, dedup thresholds, min word counts
- **API settings** — endpoint, timeouts, retry behavior
- **Pipeline settings** — max workers, min survivors, graceful degradation

Scenarios inherit from `default` and override specific values. This means
you only need to specify what changes per scenario.

---

## Project Structure

```
llm-fusion/
  README.md             # this file
  LICENSE               # MIT
  .gitignore

  skills/llm-fusion/    # tap-discoverable Hermes skill bundle
    SKILL.md            # Hermes Agent skill manifest
    scripts/
      pipeline.py       # orchestrator — runs the full pipeline
      panel.py          # parallel dispatch (6 calls, ThreadPoolExecutor)
      judge.py          # single & two-stage synthesis
      cleaner.py        # preamble stripping, dedup, min-word filter
      classifier.py     # regex + optional LLM scenario classification
      api_client.py     # urllib-based LLM API client
      config.py         # YAML config loader, scenario merging
      fallback.py       # rate limiter, OpenRouter fallback provider
      output.py         # chat formatting, JSON output saving
      skill_handler.py  # Hermes Agent trigger handler
      cli.py            # command-line entry point
      __main__.py       # python3 -m scripts
    assets/
      fusion_config.yaml          # full configuration
      fusion_config.yaml.example  # example (bundled fallback)

  tests/                # 87 unit tests
  local/                # dev notes, logs, plans
```

---

## For Developers

### Run from command line

```bash
PYTHONPATH=skills/llm-fusion python3 -m scripts --query "What is 2+2?" --verbose
```

### Run the pipeline programmatically

```python
import sys
sys.path.insert(0, "skills/llm-fusion")
from scripts.pipeline import run_pipeline
result = run_pipeline("What is 2+2?")
print(result['answer'])
```

### Run tests

```bash
python3 -m pytest tests/ -v
```

### Install from source (no Hermes required)

```bash
cd /path/to/llm-fusion
pip install pyyaml
export OPENCODE_GO_API_KEY=your_key_here
PYTHONPATH=skills/llm-fusion python3 -m scripts --query "What is 2+2?"
```

---

## Security

- API keys read from environment or `~/.hermes/.env` only
- Never commit `.env` files or API keys
- Config examples use placeholders, never real secrets
- All network calls go to configured API endpoints only

## License

MIT
