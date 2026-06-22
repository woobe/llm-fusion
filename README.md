# llm-fusion

**Multi-scenario fusion pipeline for Hermes Agent** — dispatches parallel LLM calls
(tier-configurable panel), cleans and deduplicates responses, and synthesizes a single
superior answer using a judge model with chain-of-thought reasoning.

Supports 8 scenarios: `coding`, `bugfix`, `qa`, `plan_review`, `creative`,
`reasoning`, `document`, `general`. Each scenario has its own temperature profile,
token budget, cleaning rules, and judge strategy.

---

## Example

```bash
llm-fusion low3 Explain Numerai's stake-weighted meta-model to a 5-yo in 5 lines
```

**Panel responses**:

- deepseek-v4-flash #1 — Guessing game about money movements. People put their own money behind guesses to show confidence. Meta-model mixes them, listens more to those who stake. Result is stronger and harder to trick.
- deepseek-v4-flash #2 — Big guessing game about stocks. Confident people put their own money in. Their guesses count more. Combined final guess is smarter and more reliable.
- deepseek-v4-flash #3 — Guessing game about stocks. People use tokens to show confidence. Meta-model pays more attention to bigger token holders. Everyone's guess mixed together is smarter.
- mimo-v2.5 #1 — Kids making guesses, some put toys on the line. The ones who risk their toys get heard more. Their clues mixed together make a super-smart guess.
- mimo-v2.5 #2 — Prediction contest. People put allowance money on their guesses. The more they bet, the more it counts. Extra weight for brave bettors.
- mimo-v2.5 #3 — Group project guessing stocks. People bet allowance on their predictions. Those who've been right before and bet more get more weight.

**Final answer**:

- Numerai is a big guessing game where people try to predict the stock market.
- Players put their own money or tokens on their guess to show how confident they are.
- A special "meta-model" takes everyone's guesses and mixes them into one final answer.
- It listens the most to the people who bet the most, because they are the most sure.
- This makes the team's final guess much stronger, smarter, and harder to trick than any single guess alone.

**Stats**:
─── Panel: 6/6 ok (3 ds4flash + 3 mimo) in 8.1s | Judge: single-stage in 33.9s | Total: 72s ───


---

## Quick Start

```bash
# Install with one-liner
bash -c "$(curl -fsSL https://raw.githubusercontent.com/woobe/llm-fusion/main/install.sh)"
```

Then in chat (tier examples — low1, low2/default, low3, medium, high):
```bash
llm-fusion low1 What is the capital of Australia?
llm-fusion low2 Write a bash one-liner to find the 5 largest files in a directory
llm-fusion medium Explain the difference between TCP and UDP with examples
llm-fusion high Compare SQLite and PostgreSQL for a mobile app
```

Scenarios are auto-detected. To force a specific scenario:
```bash
llm-fusion coding: Write a Python function to download a file from a URL with a progress bar
llm-fusion bugfix: My Python script raises a KeyError on dict access — here's the relevant code...
llm-fusion qa: What is the population of Japan?
llm-fusion plan_review: Review this simple blog database schema for potential issues
llm-fusion creative: Write a haiku about debugging
llm-fusion reasoning: A train leaves station A at 60 km/h. Station B is 120 km away. How long does the journey take?
llm-fusion document: Format this raw error log into a clear user-facing error message
llm-fusion general: What are the main differences between British and American English?
```

Combine tier and scenario:
```bash
llm-fusion medium coding: Write a Python function that retries an API call up to 3 times with exponential backoff
llm-fusion high reasoning: A store sells apples at $2 each and oranges at $3 each. If I buy 5 fruits for $12, how many of each did I buy?
llm-fusion low1 qa: Who founded SpaceX?
llm-fusion high creative: Write a short conversation between a senior and junior developer about code reviews
```

Or from the command line (same queries, no Hermes chat needed):
```bash
PYTHONPATH=skills/llm-fusion python3 -m scripts --query "What are the main differences between British and American English?"
PYTHONPATH=skills/llm-fusion python3 -m scripts --tier medium --query "Write a Python function that retries an API call up to 3 times with exponential backoff"
PYTHONPATH=skills/llm-fusion python3 -m scripts --tier high --query "Compare SQLite and PostgreSQL for a mobile app"
PYTHONPATH=skills/llm-fusion python3 -m scripts --tier low1 --dry-run --query "What is the capital of Australia?"
```

---

## Tiers

The panel uses a tier system to control how many models are called and which models participate. The default tier is **low2** (4 calls).

| Tier | Default | Calls | Panel | Judge |
|----|----|----|----|----|
| low1 | | 2 | 1x deepseek-v4-flash + 1x mimo-v2.5 | mimo-v2.5 |
| low2 | (default) | 4 | 2x deepseek-v4-flash + 2x mimo-v2.5 | mimo-v2.5 |
| low3 | | 6 | 3x deepseek-v4-flash + 3x mimo-v2.5 | mimo-v2.5 |
| medium | | 3 | 1x deepseek-v4-flash + 1x mimo-v2.5 + 1x deepseek-v4-pro | mimo-v2.5 |
| high | | 3 | 1x deepseek-v4-pro + 1x minimax-m3 + 1x qwen3.7-plus | mimo-v2.5 |

- **low1** — Fastest, lowest cost (2 calls). Best for simple factual queries.
- **low2** — Balanced speed and diversity. Good default for most use cases.
- **low3** — Higher capacity (6 calls). Best for comprehensive exploration.
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
|     +> low1  — 1 deepseek   |  number of calls depends on tier:
|       + 1 mimo (2 total)    |     low1   = 2 calls
|     +> low2  — 2 deepseek   |     low2   = 4 calls (default)
|       + 2 mimo (4 total)    |     low3   = 6 calls
|     +> low3  — 3 deepseek   |     medium = 3 calls (deepseek +
|       + 3 mimo (6 total)    |       mimo + deepseek-v4-pro)
|     +> medium — 1 deepseek  |     high  = 3 calls (deepseek-v4-pro
|       + 1 mimo + 1 deepseek-v4-pro |       + minimax-m3 + qwen3.7-plus)
|       (3 total)             |
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
|     +> final answer         |  temp=1.0, thinking.enabled
|   (unchanged by tier)       |  same judge config for all tiers
+-----------------------------+
    |
    v
  formatted output + metadata
```

---

## Changelog

### v0.2.8 (current)
- **Swapped judge model** — deepseek-v4-flash → mimo-v2.5 (temp=1.0, top_p=0.95, thinking.enabled, max_tokens=2048)
- **Improved judge latency** — validated avg 8.8s vs 15-35s for deepseek-v4-flash
- **Model-agnostic judge helpers** — `_merge_judge_call_config()`, `_build_judge_llm_kwargs()` support both Mimo and DeepSeek configs
- **api_client.py** — Mimo thinking now configurable via `extra_params`; hardcoded disabled only applies when no explicit thinking param passed
- **judge.py** — `_derive_judge_timeout()` handles both `max_tokens` and `max_completion_tokens`, plus `thinking.type` multiplier
- **Config defaults** — `config.py`, `fusion_config.yaml`, `fusion_config.yaml.example` all updated to Mimo judge
- **Pipeline metadata** — reports model-neutral judge config fields
- **Backward compatible** — existing configs using `deepseek-v4-flash` as judge still work
- 163 tests pass
- Version bumped to 0.2.8

### v0.2.7
- **Restructured tiers** — removed `min` and `low`, replaced with `low1` (1+1), `low2` (2+2), and `low3` (3+3)
- **New tier lineup:** low1 / low2 / low3 / medium / high (5 tiers)
- **Express path** now gated on `low1` instead of `min`
- **Default tier** changed from `low` to `low2`
- **Tier-aware retry** config updated for new tier names
- Fixed outdated default tier values in skill_handler.py and example config
- Version bumped to 0.2.7

### v0.2.6
- **Fixed express QA misclassification** — now gated on regex-only `qa` detection, preventing LLM-classified explanatory questions from bypassing panel+judge
- **Added explanatory keyword exclusion patterns** — `explain`, `compare`, `describe`, `difference between`, `how does`, `why does`, `with example` block express path for non-factual queries
- **Raised express token budget** 600→1200 for safety headroom
- **Improved LLM classifier** — returns structured JSON with `scenario`, `confidence`, `is_factual` for better routing decisions
- Version bumped to 0.2.6

### v0.2.5
- **Express direct path** for simple factual QA — short-circuits panel+judge when classifier detects short QA at high confidence, cutting pipeline time to a single direct LLM call
- **Reduced general judge reasoning** from `high` to `low` for low1/low2 tiers — directly attacks the 47-53s general scenario judge bottleneck
- **Lowered default judge token budgets** — default 8000→4000, coding 16000→8000 (no test case exceeded 3679 completion tokens)
- **Tighter timeouts matching real data** — panel 30→40s, judge 60→65s, max_timeout 300→90s, soft deadline 180→90s
- **Judge input truncation** — configurable max_panel_response_chars per scenario (qa=1200, general=1800)
- **Tier-aware retry policy** — low1 tier uses 0 retries, low2/low3/medium use 1 retry, high uses 2 retries
- All changes grounded in real latency test data (12-case benchmark across low1/low2 tiers)
- Version bumped to 0.2.5

### v0.2.4
- **DeepSeek V4 Pro** — validated and integrated into new tier structure
  - New `low1`: 1 deepseek-v4-flash + 1 mimo-v2.5 (2 calls)
  - New `low2`: 2 deepseek-v4-flash + 2 mimo-v2.5 (4 calls)
  - New `medium`: 1 deepseek-v4-flash + 1 mimo-v2.5 + 1 deepseek-v4-pro (3 calls)
  - New `high`: 1 deepseek-v4-pro + 1 minimax-m3 + 1 qwen3.7-plus (3 calls)
  - deepseek-v4-pro settings: temp=0.9, top_p=0.95, reasoning_mode=high, max_completion_tokens=2048
  - reasoning_mode now supported in panel extra_params and triggers 1.5x timeout multiplier
- CLI now supports `--tier high`
- 5 tiers: low1 (2), low2 (4), low3 (6), medium (3), high (3) — all with judge config unchanged
- Version bumped to 0.2.4

### v0.2.3
- **Qwen3.7 Plus** — validated and integrated into `medium` panel tier
  - New `medium`: 1 deepseek + 1 minimax-m3 + 1 qwen3.7-plus (3 calls)
  - Removed Mimo from `medium`; `low1`, `low2`, and `low3` still use Mimo
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
- CLI now supports `--tier min|low|medium` → CLI now supports `--tier low1|low2|low3|medium|high`
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
| Panel (low1, low2, low3, medium) | deepseek-v4-flash | 0.75 | 0.9 | — | max_completion_tokens |
| Panel (low1, low2, low3, medium) | mimo-v2.5 | 0.6/0.7/0.8 | 0.95 | thinking.type=disabled | max_tokens |
| Panel (medium, high) | deepseek-v4-pro | 0.9 | 0.95 | reasoning_mode=high | max_completion_tokens |
| Panel (high only) | minimax-m3 | 0.85 | 0.9 | top_k=40, thinking.type=adaptive | max_tokens |
| Panel (high only) | qwen3.7-plus | 0.8 | 0.92 | top_k=20, reasoning_effort=high | max_tokens |
| Judge (all tiers) | mimo-v2.5 | 1.0 | 0.95 | thinking.type=enabled | max_tokens |

- **deepseek-v4-flash** — primary panel model for low1/low2/low3/medium tiers. Fast, supports
  reasoning_mode for chain-of-thought, strong coding and reasoning capabilities.
- **mimo-v2.5** — secondary panel model (low1/low2/low3/medium) and **judge model (all tiers)**.
  Cheaper, uses internal thinking when enabled. Panel-only role for diversity; as judge it
  synthesizes the final answer with thinking.type=enabled, temp=1.0, top_p=0.95.
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
  tests/                # 143 unit tests
  local/                # dev notes
```

---

## For Developers

### Run from command line

```bash
# Run with default (low2) tier
PYTHONPATH=skills/llm-fusion python3 -m scripts --query "What is 2+2?" --verbose

# Choose a different tier (low1 / low2 / low3 / medium / high)
PYTHONPATH=skills/llm-fusion python3 -m scripts --query "What is 2+2?" --tier medium --verbose
PYTHONPATH=skills/llm-fusion python3 -m scripts --query "What is 2+2?" --tier high --verbose

# Dry-run to validate config and arguments
PYTHONPATH=skills/llm-fusion python3 -m scripts --dry-run --query "test" --tier low1
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
