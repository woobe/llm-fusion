# llm-fusion

Multi-scenario fusion pipeline: dispatches 6 parallel LLM calls (3x deepseek-v4-flash + 3x mimo-v2.5), cleans and deduplicates responses, and synthesizes a single superior answer using a judge LLM with chain-of-thought reasoning.

Supports 8 scenarios: coding, bugfix, qa, plan_review, creative, reasoning, document, general.

## Quick Start

```bash
# Run the pipeline
python3 -c "from scripts.pipeline import run_pipeline; r = run_pipeline('What is 2+2?'); print(r['answer'])"
```

## Requirements

- Python 3.10+
- PyYAML (`pip install pyyaml`)
- API key: set `OPENCODE_GO_API_KEY` env var or add to `~/.hermes/.env`

## Hermes Skill Installation

```bash
hermes skills tap add woobe/llm-fusion
hermes skills install llm-fusion
```

Then use `/llm-fusion <prompt>` in chat.

## Project Structure

```
llm-fusion/
  SKILL.md        # agentskills.io manifest
  scripts/        # 12 Python modules (all code)
  assets/         # fusion_config.yaml
  tests/          # 87 unit tests
  references/     # architecture docs
  local/          # planning docs (gitignored)
```

## Configuration

See `assets/fusion_config.yaml` for all scenario settings.

## Tests

```bash
python3 -m pytest tests/ -v
```

## License

MIT
