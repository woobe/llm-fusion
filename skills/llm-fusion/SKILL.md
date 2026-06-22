---
name: llm-fusion
description: Multi-scenario fusion pipeline with tier-based panel dispatch (min/low/medium/high) using deepseek-v4-flash, mimo-v2.5, minimax-m3, deepseek-v4-pro, and qwen3.7-plus — synthesizes a superior answer via a single or two-stage judge with chain-of-thought reasoning.
license: MIT
compatibility:
  - hermes-agent>=1.0.0
metadata:
  version: 2.0.0
  tags: [llm, fusion, ensemble, ai, opencode]
  triggers:
    - pattern: /llm-fusion
      handler: scripts.skill_handler.handle_fusion_trigger
      description: "Run the fusion pipeline. Usage: /llm-fusion <prompt>"
    - pattern: /fusion
      handler: scripts.skill_handler.handle_fusion_trigger
      description: Alias for /llm-fusion
  env_vars:
    - name: OPENCODE_GO_API_KEY
      description: Required. API key for OpenCode Go.
      required: true
---

# llm-fusion

Multi-model fusion skill for Hermes Agent. Dispatches parallel LLM calls across configurable tiers (min/low/medium/high) with deepseek-v4-flash, mimo-v2.5, minimax-m3, deepseek-v4-pro, and qwen3.7-plus, then cleans, deduplicates, and synthesizes a single superior answer using a judge model with chain-of-thought reasoning.

## Quick Start

```
/llm-fusion What is the capital of France?
/fusion Write a Python function to sort a list
```

## Usage

`/llm-fusion <prompt>` - auto-detects scenario, defaults to low tier
`/llm-fusion --tier medium <prompt>` - use medium tier (includes deepseek-v4-pro + mimo)
`/llm-fusion --tier high <prompt>` - use high tier (deepseek-v4-pro + minimax-m3 + qwen3.7-plus)
`/llm-fusion --tier min <prompt>` - use min tier (2 deepseek + 1 mimo)

## Scenarios

coding, bugfix, qa, plan_review, creative, reasoning, document, general

## Configuration

See assets/fusion_config.yaml for panel temps, token budgets, judge settings.

## License

MIT
