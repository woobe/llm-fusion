# llm-fusion

**Multi-scenario fusion pipeline for Hermes Agent** — dispatches parallel LLM calls
(tier-configurable panel), cleans and deduplicates responses, and synthesizes a single
superior answer using a judge model with chain-of-thought reasoning.

Supports 8 scenarios: `coding`, `bugfix`, `qa`, `plan_review`, `creative`,
`reasoning`, `document`, `general`. Each scenario has its own temperature profile,
token budget, cleaning rules, and judge strategy.

**5 tiers (effort)**
- **low1**: 1 x deepseek-v4-flash + 1 x mimo-v2.5
- **low2**: 2 x deepseek-v4-flash + 2 x mimo-v2.5
- **low3**: 3 x deepseek-v4-flash + 3 x mimo-v2.5
- **medium**: 1 x deepseek-v4-flash + 1 x mimo-v2.5 + 1 x deepseek-v4-pro (default tier)
- **high**: 1 x deepseek-v4-pro + 1 x minimax-m3 + 1 x qwen3.7-plus

---

## Quick Start

```bash
# Install with one-liner
bash -c "$(curl -fsSL https://raw.githubusercontent.com/woobe/llm-fusion/main/install.sh)"
```

Then in chat (see following examples).

---


## Example (Explain Something to a 5-yo)

```bash
llm-fusion: Explain Numerai's stake-weighted meta-model to a 5-yo in 5 lines
```

**Panel responses**:

- deepseek-v4-flash (416 tokens, 358 reasoning)

  > Numerai is a game where people guess how stocks will move.
  > The meta-model combines everyone's guesses into one smart guess.
  > But people who have more stake (like points or tokens) get more say in that combined guess.
  > This rewards good guesses and helps the whole team win.

- mimo-v2.5 (110 tokens, 22 reasoning)

  > Numerai is a game where lots of people try to guess how the stock market will move.
  > Each person puts their own money on their guesses to show they're really confident.
  > A smart computer then listens to everyone's guesses but trusts the people who put more money more.
  > It mixes all those guesses together into one super-smart prediction, like picking the best parts of every kid's crayon drawing to make one awesome picture.


- deepseek-v4-pro (577 tokens, 494 reasoning)

  > Think of Numerai as a big guessing game where lots of people try to predict the future.
  > Each player bets their own special coins on how sure they are about their guess.
  > The final guess is a blend of everyone's guesses, but guesses with more coins bet count more.
  > This way, players are careful because they can lose their coins if they're wrong, and win more if they're right.


**Judge synthesis**:

Judge synthesis (mimo-v2.5, single-stage, 333 tokens, 210 reasoning) blended all responses into the final answer with the special coins + best parts of every picture analogy — drawing from deepseek-v4-pro's "special coins" and mimo-v2.5's "crayon drawing" theme.

**Final Answer**:

  > Numerai is like a big stock-picking game where lots of people guess how the market will move.
  > Each player bets their own special coins to show how sure they are about their guess.
  > The game's smart computer makes a final guess by listening to everyone — but gives more importance to guesses from people who bet more coins.
  > This helps everyone try to guess carefully, because good guesses can win you more coins, and bad ones lose them.
  > By mixing all the guesses this way, the final prediction becomes extra smart — like taking the best parts of every picture to make one great drawing.

**Stats**:

Panel: 3/3 ok in 14s | Judge: single-stage in 14s | Total: 42s


---

## Example (Short Factual Question)


```bash
llm-fusion low1: What is the capital of Australia?
```

**Final answer**:

Canberra.

The pipeline classified this as a short factual question (qa, 0.85 confidence), so it short-circuited the panel+judge stage and answered directly.

Scenario: qa | Panel: 0/0 (short-circuited) | 2643ms total

---

## Example (Bash)


```bash
llm-fusion low2: Write a bash one-liner to find the 5 largest files in a directory
```

**Final answer**:

`find "${1:-.}" -maxdepth 1 -type f -printf '%s %p\n' 2>/dev/null | sort -rn | head -n 5`

Usage:
  - Defaults to current directory, or pass a target: ./script.sh /path/to/dir
  - For recursive search, drop -maxdepth 1
  - Handles spaces, permission errors, and empty directories

Pipeline notes:
  - Tier: low2 → 4 parallel calls (2× deepseek-v4-flash + 2× mimo-v2.5)
  - Classification: coding (LLM classifier, 95% confidence)
  - Panel: 4/4 succeeded in ~23s
  - Judge: single-stage synthesis in ~66s — merged best parts (mimo's -printf approach, deepseek's error handling) and documented edge cases
  - Total time: ~95s

The judge picked find -printf '%s %p' as the cleanest approach over alternatives like du/stat/ls -lS, noting it's the most robust with special characters and permission errors.


---

## Example (General QA)

```bash
llm-fusion medium: Explain the difference between TCP and UDP with examples
```

**Final answer**:

- TCP — connection-oriented. Three-way handshake, acknowledgments, retransmissions. Guarantees reliable, ordered delivery.
  - Examples: file transfers (FTP), email (SMTP), web browsing (HTTP/HTTPS)

- UDP — connectionless. No handshake, no confirmation. Prioritizes speed over reliability.
  - Examples: live video/audio streaming, VoIP, online gaming, DNS lookups

Bottom line: TCP for correctness (files, emails), UDP for speed (streaming, gaming).

Panel: 3/3 ok in 9.7s | Judge: single-stage in 15.8s | Total: 44s


---

## Example (Reasoning)

```bash
llm-fusion high: Compare SQLite and PostgreSQL for a mobile app
```

**Final answer**:

Core verdict: They're complementary, not competitors.

SQLite — embedded, serverless, ~600KB library, zero config, runs in-process. Built into iOS and Android. Perfect for on-device storage, offline-first, single-user. No network attack surface.

PostgreSQL — client-server daemon, hundreds of MB, needs configuration and tuning. Cannot run on iOS/Android (no official builds, no daemon support, killed by OS for resource usage). Excellent for the remote backend server.

Standard architecture: SQLite locally for offline data + PostgreSQL remotely with a sync/API layer between them. They serve different tiers — not an either/or choice.

Panel: 3/3 ok (deepseek-v4-pro + minimax-m3 + qwen3.7-plus) in 102s | Judge: two-stage in 152s | Total: 258s


---

# More Examples

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

The panel uses a tier system to control how many models are called and which models participate. The default tier is **medium** (3 calls).

| Tier | Default | Calls | Panel | Judge |
|----|----|----|----|----|
| low1 | | 2 | 1x deepseek-v4-flash + 1x mimo-v2.5 | mimo-v2.5 |
| low2 | | 4 | 2x deepseek-v4-flash + 2x mimo-v2.5 | mimo-v2.5 |
| low3 | | 6 | 3x deepseek-v4-flash + 3x mimo-v2.5 | mimo-v2.5 |
| medium | (default) | 3 | 1x deepseek-v4-flash + 1x mimo-v2.5 + 1x deepseek-v4-pro | mimo-v2.5 |
| high | | 3 | 1x deepseek-v4-pro + 1x minimax-m3 + 1x qwen3.7-plus | mimo-v2.5 |

- **low1** — Fastest, lowest cost (2 calls). Best for simple factual queries.
- **low2** — Balanced speed and diversity. Good for general use.
- **low3** — Higher capacity (6 calls). Best for comprehensive exploration.
- **medium** — Adds deepseek-v4-pro for deeper reasoning. Good for coding, analysis. (default)
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
|     +> low2  — 2 deepseek   |     low2   = 4 calls
|       + 2 mimo (4 total)    |     low3   = 6 calls
|     +> low3  — 3 deepseek   |     medium = 3 calls (default, deepseek +
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

### v0.2.16 (current)
- **Per-Tier Panel Quorum & Min Survivors** — tier-specific early-exit and failure thresholds
  - New `pipeline.quorum_by_tier` config: controls early-exit per tier
    - low1: 2 (all needed), low2/low3: 0 (wait for all), medium/high: 3 (all needed)
  - New `pipeline.min_survivors_by_tier` config: controls failure threshold per tier
    - low1: 2, low2: 3 (1 failure), low3: 5 (1 failure), medium/high: 3 (0 failures)
  - `_resolve_panel_quorum(config, total_calls, tier=None)` — checks `quorum_by_tier[tier]` first
  - New `_resolve_min_survivors(config, tier=None)` — checks `min_survivors_by_tier[tier]` first
  - Both resolvers fall back to `min_survivors` for backward compatibility
  - `dispatch_panel()` now passes tier to both resolvers
  - Pipeline post-cleaning survivor check uses tier-specific threshold
  - Quorum=0 disables early exit — panel runs until all calls finish or timeout
  - Future-proof: adding models to medium/high tiers works with quorum=3, min=3
  - Tests: per-tier quorum, per-tier min survivors, backward compatibility
  - 301 tests passing

### v0.2.15
- **Structured Error Categories + Observability** — typed error objects, per-call metadata, safe observability hooks
  - New `_categorize_error()` helper: auth_error, rate_limited, timeout, bad_request, network_error, empty_response
  - New `_build_call_metadata()` for structured per-call info
  - Per-call fields: `error_category`, `attempt_count`, `retryable`, `final_http_status`
  - Token counters: `prompt_tokens`, `completion_tokens` when available
  - No secrets, raw prompts, or request payloads in output
  - 29 new tests covering error categorization, metadata, and backward compatibility
  - 295 tests passing (was 266)

### v0.2.14
- **Status-Aware + Deadline-Aware Retries** — retry only retryable failures, respect pipeline soft deadline
  - New `_RETRYABLE_STATUSES` set: 408, 409, 425, 429, 5xx
  - New `_NON_RETRYABLE_STATUSES` set: 400, 401, 403, 404
  - `deadline_timestamp` parameter propagated through panel, judge, express QA, and direct fallback
  - Exponential backoff with jitter for 429/5xx
  - Config keys: `retry.retryable_statuses`, `retry.non_retryable_statuses`, `retry.backoff_base_seconds`, `retry.backoff_max_seconds`
  - 10 new tests (7 deadline-aware + 3 status tests)
  - 266 tests passing (was 259)

### v0.2.13
- **Prompt-Size Budgeting** — estimates input size before judge calls, trims by model priority + scenario boundaries
  - New `_apply_prompt_budget()` helper estimates total input before judge calls
  - New `_compact_sections()` performs section-compaction (not raw first-N chars)
  - Config keys: `prompt_budget.enabled`, `prompt_budget.strategy`, `prompt_budget.max_input_chars`, per-scenario budgets
  - Budget applied before single-stage, two-stage stage 1, and two-stage stage 2 judge calls
  - Metadata: `prompt_budget_applied`, `prompt_budget_strategy`, `prompt_budget_input_chars`, `prompt_budget_trimmed_chars`
  - Config added to all 8 scenarios in both bundled configs
  - 18 new tests in TestPromptBudget + 2 config-smoke tests
  - 259 tests passing (was 241)

### v0.2.12
- **Two-Stage Judge Token Reduction** — stage 2 now receives compact evidence bundle only (not raw panel responses)
  - New `_evidence_bundle_instructions()` helper structures stage 1 output (verdict, key_findings, contradictions, best_evidence, synthesis_plan)
  - `stage2_include_raw_responses` config flag (default: `false`) — opt in with `true` for full replay
  - `_build_responses_section()` supports `return_stats=True` for truncation counting
  - `max_panel_response_chars` defaults per two-stage scenario: bugfix=2000, plan_review=2500, reasoning=2000, document=3000
  - Metadata: `stage1_input_chars`, `stage2_input_chars`, `panel_response_truncated_count`, `panel_response_truncated_chars`
  - Saves ~3,000 tokens per two-stage judge call
  - Config keys added to both bundled configs
  - 8 new tests (7 judge + 1 pipeline metadata)
  - 241 tests passing (was 234)

### v0.2.11
- **Panel Early Quorum & Cancellation** — `dispatch_panel()` returns early once `min_survivors` responses collected
  - New `_resolve_panel_quorum()` helper derives quorum = min(total_calls, pipeline.min_survivors)
  - Pending futures cancelled, in-flight results discarded
  - No new config keys — uses existing `pipeline.min_survivors`
  - Executor lifecycle: explicit `shutdown(wait=False)` on early exit
  - 7 new metadata fields: `total_calls`, `quorum`, `quorum_reached`, `quorum_at_ms`, `cancelled_count`, `late_completed_count`, `panel_calls_early_exit`
  - 8 new unit tests in `TestPanelQuorum`, 1 pipeline integration test
  - 234 tests passing (was 226)

### v0.2.10
- **Config-driven direct fallback** — consolidated hardcoded fallback logic into one reusable helper
  - New `_apply_direct_fallback()` helper for all failure paths
  - Config section: `pipeline.direct_fallback` (model, temperature, top_p, max_tokens, timeout, retries, delays_seconds)
  - Consistent metadata: `fallback_reason`, `fallback_model`, `fallback_error`, `fallback_elapsed_ms`
  - Bug fix: insufficient survivors no longer falls through to judge when fallback fails
- **Fallback provider + rate limiter** — integrated existing fallback.py into main API path
  - Thread-safe `RateLimiter` with token bucket algorithm
  - Rate limiting applied to all outbound API calls
  - Status-aware retry: no retry on 401/403, retry on 429/5xx/None
  - Exponential backoff with jitter for retryable statuses
  - Optional provider fallback after primary exhaustion
  - Config keys: `api.rate_limit` and `api.fallback`
- Dead code removed: `call_with_fallback()` in fallback.py
- 226 tests passing (was 182)

### v0.2.9
- **Classifier optimization** — added `classification.enabled: false` config flag
  - LLM second-pass classifier now opt-in only (default: disabled)
  - Saves 1 API call per general query
  - Backward compatible: existing configs default to `enabled: false`
- 4 new unit tests added (disabled, missing-key, enabled, high-confidence bypass)
- 100 tests passing

### v0.2.8
- **Swapped judge model** — deepseek-v4-flash → mimo-v2.5 (temp=1.0, top_p=0.95, thinking.enabled, max_tokens=4096)
- **Improved judge latency** — validated avg 8.8s vs 15-35s for deepseek-v4-flash
- **Fixed empty-answer bug** — thinking tokens consumed entire 2048 budget; raised to 4096
- **Token budget tune** — comprehensive review across all models and scenarios (see details below)
- **Raised timeout ceilings** — `max_timeout` 90→360, `soft_deadline` 90→300, `judge_floor` 65→90, `panel_floor` 40→60, `overhead` 10→15
- **Added scenario-specific Mimo panel budgets** — qa 800, general 1200, coding/bugfix/reasoning 2000, plan_review/creative 2500, document 3000
- **Raised high-tier model defaults** — minimax-m3, qwen3.7-plus, deepseek-v4-pro: 2048→4096
- **Raised two-stage judge stages** — bugfix/plan_review/reasoning/document: 4096→6144
- **Config drift fixed** — active/example YAMLs now mirrored on timeouts and max_panel_response_chars
- **Code fallbacks aligned** — Mimo judge default 2048→4096, high-tier defaults 2048→4096
- **169 tests pass** (was 163)
- No version bump (config/tuning changes on v0.2.8)

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
  tests/                # 234 unit tests
  local/                # dev notes
```

---

## For Developers

### Run from command line

```bash
# Run with default (medium) tier
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

---

## License

MIT
