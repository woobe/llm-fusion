# llm-fusion

**Multi-scenario fusion pipeline for Hermes Agent** — dispatches parallel LLM calls
(tier-configurable panel), cleans and deduplicates responses, and synthesizes a single
superior answer using a judge model with chain-of-thought reasoning.

Supports 8 scenarios: `coding`, `bugfix`, `qa`, `plan_review`, `creative`,
`reasoning`, `document`, `general`. Each scenario has its own temperature profile,
token budget, cleaning rules, and judge strategy.

---

## Quick Start

```bash
# Install with one-liner
bash -c "$(curl -fsSL https://raw.githubusercontent.com/woobe/llm-fusion/main/install.sh)"
```

Then in chat (tier examples — min, low/default, medium, high):
```bash
llm-fusion min Explain recursion in one sentence
llm-fusion low Compare REST and GraphQL API design trade-offs
llm-fusion medium Write a Python function to merge overlapping intervals
llm-fusion high Analyse microservices vs monolith trade-offs for a startup
```

Scenarios are auto-detected. To force a specific scenario:
```bash
llm-fusion coding: Write a Python script that watches a directory and uploads new files to S3
llm-fusion bugfix: My pytest fixture resets state between tests — here's my conftest.py...
llm-fusion qa: What are the downsides of using Redis as a primary database?
llm-fusion plan_review: Review this migration plan from PostgreSQL to CockroachDB...
llm-fusion creative: Write a short story where a git merge conflict gains sentience
llm-fusion reasoning: If a bat and a ball cost $1.10 and the bat costs $1 more than the ball, how much does the ball cost?
llm-fusion document: Take this raw API response and format it as clean markdown docs
llm-fusion general: Explain the CAP theorem with real-world database examples
```

Combine tier and scenario:
```bash
llm-fusion medium coding: Build a Python rate-limited HTTP client with retries
llm-fusion high reasoning: Design a data pipeline for real-time Twitter sentiment analysis
llm-fusion min qa: Is MongoDB suitable as the primary store for an e-commerce product catalog?
llm-fusion high creative: Write a dialogue between a senior and junior dev about when to refactor
```

Or from the command line (same queries, no Hermes chat needed):
```bash
PYTHONPATH=skills/llm-fusion python3 -m scripts --query "Compare REST and GraphQL API trade-offs"
PYTHONPATH=skills/llm-fusion python3 -m scripts --tier medium --query "Write a Python function to merge overlapping intervals"
PYTHONPATH=skills/llm-fusion python3 -m scripts --tier high --query "Analyse microservices vs monolith trade-offs for a startup"
PYTHONPATH=skills/llm-fusion python3 -m scripts --tier min --dry-run --query "Explain recursion"
```

---

## Tiers

The panel uses a tier system to control how many models are called and which models participate. The default tier is **low** (4 calls).

| Tier | Default | Calls | Panel | Judge |
|---|---|---|---|---|---|
| min | | 3 | 2x deepseek-v4-flash + 1x mimo-v2.5 | deepseek-v4-flash |
| low | (default) | 4 | 2x deepseek-v4-flash + 2x mimo-v2.5 | deepseek-v4-flash |
| medium | | 3 | 1x deepseek-v4-flash + 1x mimo-v2.5 + 1x deepseek-v4-pro | deepseek-v4-flash |
| high | | 3 | 1x deepseek-v4-pro + 1x minimax-m3 + 1x qwen3.7-plus | deepseek-v4-flash |

- **min** — Fastest, lowest cost. Best for simple factual queries.
- **low** — Balanced speed and diversity. Good default for most use cases.
- **medium** — Adds deepseek-v4-pro for deeper reasoning. Good for coding, analysis.
- **high** — Premium panel with 3 different models. Best for complex reasoning, creative work, and important queries where quality matters most.

---

## Architecture Overview

```
user query
    |
    v
+-----------------------------+
|  1. Classifier              |  regex pre-classifier (+ optional LLM 2nd pass)
|     +> scenario_id          |  8 scenarios, auto-detected
+-----------------------------+
    |
    v
+-----------------------------+
|  2. Panel (tier-based)      |  ThreadPoolExecutor
|     +> min   — 2 deepseek   |  number of calls depends on tier:
|       + 1 mimo (3 total)    |     min    = 3 calls
|     +> low   — 2 deepseek   |     low    = 4 calls (default)
|       + 2 mimo (4 total)    |     medium = 3 calls (deepseek +
|     +> medium — 1 deepseek  |       mimo + deepseek-v4-pro)
|       + 1 mimo + 1 deepseek-v4-pro |     high  = 3 calls (deepseek-v4-pro
|       (3 total)             |       + minimax-m3 + qwen3.7-plus)
|     +> high — 1 deepseek-v4-pro |
|       + 1 minimax + 1 qwen  |
|       (3 total)             |
+-----------------------------+
    |
    v
+-----------------------------+
|  3. Cleaner                 |  preamble strip, fence handling, dedup
|     +> cleaned_responses    |  scenario-specific thresholds
+-----------------------------+
    |
    v
+-----------------------------+
|  4. Judge                   |  single or two-stage synthesis
|     +> final answer         |  temp=0.0, high reasoning mode
|   (unchanged by tier)       |  same judge config for all tiers
+-----------------------------+
    |
    v
  formatted output + metadata
```

---

## Changelog

### v0.2.4 (current)
- **DeepSeek V4 Pro** — validated and integrated into new tier structure
  - New `min`: 2 deepseek-v4-flash + 1 mimo-v2.5 (3 calls)
  - New `medium`: 1 deepseek-v4-flash + 1 mimo-v2.5 + 1 deepseek-v4-pro (3 calls)
  - New `high`: 1 deepseek-v4-pro + 1 minimax-m3 + 1 qwen3.7-plus (3 calls)
  - deepseek-v4-pro settings: temp=0.9, top_p=0.95, reasoning_mode=high, max_completion_tokens=2048
  - reasoning_mode now supported in panel extra_params and triggers 1.5x timeout multiplier
- CLI now supports `--tier high`
- 4 tiers: min (3), low (4), medium (3), high (3) — all with judge config unchanged
- Version bumped to 0.2.4

### v0.2.3
- **Qwen3.7 Plus** — validated and integrated into `medium` panel tier
  - New `medium`: 1 deepseek + 1 minimax-m3 + 1 qwen3.7-plus (3 calls)
  - Removed Mimo from `medium`; `min` and `low` still use Mimo
  - Qwen settings: temp=0.8, top_p=0.92, top_k=20, reasoning_effort=high
  - Uses max_tokens (not max_completion_tokens) and returns reasoning_content separately
- **MiniMax M3** — added as panel-only model in `medium` tier
  - Settings: temp=0.85, top_p=0.9, top_k=40, thinking.type=adaptive
  - Uses max_tokens (not max_completion_tokens)
  - Validated with test calls
- **Adaptive timeout** — timeout derived automatically per model from token budget
  - Formula: `max(floor, tokens / throughput + overhead)`
  - 1.5x multiplier for thinking models (adaptive/enabled modes)
  - Pipeline soft deadline (180s default, configurable)
  - Scales automatically when models are added or token budgets change
- Judge config unchanged by tier (all tiers use same judge config)
- Worker count adjusts dynamically to panel size
- CLI now supports `--tier min|low|medium`
- **Bug fixes:** API key HERMES_HOME fallback, output saved on all paths, panel responses in JSON, graceful variable scoping fix, output dir consistency

### v0.2
- Removed src/llm_fusion/ duplication, all code in scripts/ only
- Removed pyproject.toml (no pip package needed)
- Clean agentskills.io layout
- Comprehensive README covering architecture, scenarios, token budgets
- 87 tests pass

### v0.1
- Initial pipeline implementation
- 8 scenarios, 6 parallel panel calls, single/two-stage judge
- agentskills.io restructure
- 54 tests

---

## Design Decisions

### Panel vs. Judge (Why Two Roles?)

The pipeline separates **exploration** (panel) from **convergence** (judge).

- **Panel** -- 6 diverse calls at varied temperatures. Each call is independent.
  The panel explores the answer space from different "angles" -- some conservative
  (low temp), some creative (high temp), some via a different model entirely.
  The goal is breadth of coverage.

- **Judge** -- a single deterministic (temp=0.0) call with high reasoning mode
  that reads all panel responses and synthesizes the best possible answer.
  The judge does not generate from scratch -- it evaluates, compares, and
  synthesizes from the panel's raw material. This is fundamentally different
  from asking one model to answer directly, because the judge can:
  - Discard hallucinations from individual panel responses by cross-referencing
  - Combine the strongest parts of different responses
  - Fill gaps where one model covered something others missed
  - Catch contradictions and resolve them

### Temperature Diversity (Why multiple calls at different temps?)

Temperature controls the randomness of token sampling. Low temp (0.2-0.4)
produces conservative outputs; high temp (0.8-1.0) produces more creative,
varied outputs. The panel deliberately spans a range depending on scenario.

### Two-Stage Judge (When and Why)

Some scenarios use a **single-stage judge** (one call that produces the final
answer directly). Others use a **two-stage judge** (analysis first, then synthesis):

| Judge strategy | Scenarios |
|---|---|
| Single-stage | coding, qa, creative, general |
| Two-stage | bugfix, plan_review, reasoning, document |

Two-stage is used when multi-step analysis is needed (bug diagnosis, plan
review, reasoning) or a structured intermediate analysis improves the output
(document review).

### Model Selection

| Role | Model | temp | top_p | Extra | Token Param |
|---|---|---|---|---|---|
| Panel (min, low, medium) | deepseek-v4-flash | 0.75 | 0.9 | — | max_completion_tokens |
| Panel (min, low, medium) | mimo-v2.5 | 0.6/0.7/0.8 | 0.95 | thinking.type=disabled | max_tokens |
| Panel (medium, high) | deepseek-v4-pro | 0.9 | 0.95 | reasoning_mode=high | max_completion_tokens |
| Panel (high only) | minimax-m3 | 0.85 | 0.9 | top_k=40, thinking.type=adaptive | max_tokens |
| Panel (high only) | qwen3.7-plus | 0.8 | 0.92 | top_k=20, reasoning_effort=high | max_tokens |
| Judge (all tiers) | deepseek-v4-flash | 0.0 | 1.0 | reasoning_mode=high | max_completion_tokens |

- **deepseek-v4-flash** — primary model for panel and judge. Fast, supports
  reasoning_mode for chain-of-thought, strong coding and reasoning capabilities.
- **mimo-v2.5** — secondary panel model. Cheaper, adds diversity at marginal
  cost. Panel-only role (never makes synthesis decisions).
- **deepseek-v4-pro** — higher-capability deepseek model, active in `medium` and
  `high` tiers. Uses reasoning_mode=high with max_completion_tokens for deeper
  reasoning. Settings: temp=0.9, top_p=0.95.
- **minimax-m3** — panel-only model, active only in `high` tier. Provides
  additional diversity with adaptive thinking at temp=0.85.
- **qwen3.7-plus** — panel-only model, active only in `high` tier. Provides a
  high-reasoning response with separate `reasoning_content` and validated direct
  top-level params.

### opencode.go Focus

The pipeline targets OpenCode Go (https://opencode.ai/zen/go/v1) as its
primary LLM provider. May expand to more providers in future versions.

### Adaptive Timeout (Why per-model timeouts?)

Different models and scenarios have different response-time profiles. A
simple uniform timeout wastes time on fast models (waiting needlessly) or
causes premature failures on models with large token budgets. The pipeline
derives timeouts dynamically:

- **Formula**: `timeout = max(floor, token_budget / throughput + overhead)`
  where throughput and overhead are configurable per provider.
- **Thinking multiplier**: Models with `thinking.type=adaptive` or
  `thinking.type=enabled` or `reasoning_effort` set get a 1.5x multiplier
  because thinking/CoT responses take significantly longer per token.
- **Explicit override**: Any model entry can set a literal `timeout`
  field which bypasses the formula entirely.
- **Pipeline soft deadline**: An overall deadline (default 180s) guards
  the whole pipeline. If exceeded, the pipeline either errors or falls
  back to a direct single-call response depending on the
  `graceful_degradation` setting.

This design scales automatically when new models are added or token
budgets change — no manual timeout tuning per model.

---

## Requirements

- Python 3.10+
- PyYAML
- `OPENCODE_GO_API_KEY` in `~/.hermes/.env` or environment variable

---

## Project Structure

```
llm-fusion/
  README.md, LICENSE, .gitignore
  skills/llm-fusion/    # skill bundle
    SKILL.md, scripts/ (12 modules), assets/ (config)
  tests/                # 87 unit tests
  local/                # dev notes
```

---

## For Developers

### Run from command line

```bash
# Run with default (low) tier
PYTHONPATH=skills/llm-fusion python3 -m scripts --query "What is 2+2?" --verbose

# Choose a different tier (min / low / medium / high)
PYTHONPATH=skills/llm-fusion python3 -m scripts --query "What is 2+2?" --tier medium --verbose
PYTHONPATH=skills/llm-fusion python3 -m scripts --query "What is 2+2?" --tier high --verbose

# Dry-run to validate config and arguments
PYTHONPATH=skills/llm-fusion python3 -m scripts --dry-run --query "test" --tier min
```

### Run tests

```bash
python3 -m pytest tests/ -v
```

### Install from source

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

## License

MIT
