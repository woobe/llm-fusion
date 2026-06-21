---
name: llm-fusion
description: Multi-scenario fusion pipeline that dispatches 6 parallel LLM calls (3x deepseek-v4-flash + 3x mimo-v2.5) and synthesizes a superior answer via a single or two-stage judge with chain-of-thought reasoning.
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

Multi-model fusion skill for Hermes Agent. Dispatches 6 parallel LLM calls, cleans and deduplicates responses, then synthesizes a single superior answer using a judge model with chain-of-thought reasoning.

## Quick Start

```
/llm-fusion What is the capital of France?
/fusion Write a Python function to sort a list
```

## Usage

`/llm-fusion <prompt>` - auto-detects scenario
`/llm-fusion coding: <prompt>` - force coding scenario

## Scenarios

coding, bugfix, qa, plan_review, creative, reasoning, document, general

## Configuration

See assets/fusion_config.yaml for panel temps, token budgets, judge settings.

## License

MIT
