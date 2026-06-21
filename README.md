# llm-fusion

Multi-scenario fusion pipeline for Hermes Agent. Dispatches 6 parallel LLM calls (3x deepseek-v4-flash + 3x mimo-v2.5), cleans and deduplicates responses, and synthesizes a single superior answer using a judge LLM with chain-of-thought reasoning.

Supports 8 scenarios: coding, bugfix, qa, plan_review, creative, reasoning, document, general.

## Install

```bash
hermes skills tap add woobe/llm-fusion
hermes skills install llm-fusion
```

## Usage

```
/llm-fusion What is the capital of France?
/llm-fusion Write a Python function to sort a list
/llm-fusion Fix this bug: [paste error and code]
/fusion Explain Numerai in two lines
```

Scenarios are auto-detected. To force a specific scenario:
```
/llm-fusion coding: Write a Python function to sort a list
```

## Requirements

- Python 3.10+
- PyYAML
- `OPENCODE_GO_API_KEY` in `~/.hermes/.env`

## Configuration

See `assets/fusion_config.yaml` for panel temperatures, token budgets, judge settings per scenario.

## Project Structure

```
llm-fusion/
  SKILL.md        # agentskills.io manifest
  scripts/        # 12 Python modules (pipeline code)
  assets/         # fusion_config.yaml
  tests/          # 87 unit tests
  references/     # architecture docs
```

---

## For Developers

### Run the pipeline from command line

```bash
python3 -c "from scripts.pipeline import run_pipeline; r = run_pipeline('What is 2+2?'); print(r['answer'])"
```

### Run tests

```bash
python3 -m pytest tests/ -v
```

### Install from source (no Hermes required)

```bash
cd llm-fusion
pip install pyyaml
export OPENCODE_GO_API_KEY=your_key_here
python3 -c "from scripts.pipeline import run_pipeline; r = run_pipeline('What is 2+2?'); print(r['answer'])"
```

## Security

- API keys read from environment or `~/.hermes/.env` only
- Never commit `.env` files or API keys
- Config examples use placeholders, never real secrets

## License

MIT
