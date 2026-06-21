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

Then in chat:
```
/llm-fusion What is the capital of France?
```

Scenarios are auto-detected. To force a specific scenario:
```
/llm-fusion coding: Write a Python function to sort a list
```

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
|  2. Panel (6 parallel)      |  ThreadPoolExecutor -- 3x deepseek + 3x mimo
|     +-- deepseek-v4-flash   |  temperature varies per instance for diversity
|     +-- deepseek-v4-flash   |
|     +-- deepseek-v4-flash   |
|     +-- mimo-v2.5           |
|     +-- mimo-v2.5           |
|     +-- mimo-v2.5           |
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
+-----------------------------+
    |
    v
  formatted output + metadata
```

---

## Changelog

### v0.2.1 (current)
- Fixed install.sh to use $HERMES_HOME for correct path resolution
- Version bump from 0.2.0

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

- **deepseek-v4-flash** -- primary model for panel and judge. Fast, supports
  reasoning_mode for chain-of-thought, strong coding and reasoning capabilities.
- **mimo-v2.5** -- secondary panel model. Cheaper, adds diversity at marginal
  cost. Panel-only role (never makes synthesis decisions).

### opencode.go Focus

The pipeline targets OpenCode Go (https://opencode.ai/zen/go/v1) as its
primary LLM provider. May expand to more providers in future versions.

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
PYTHONPATH=skills/llm-fusion python3 -m scripts --query "What is 2+2?" --verbose
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
