# llm-fusion

A scenario-aware multi-model fusion pipeline: dispatches 6 parallel LLM
calls (3× deepseek-v4-flash + 3× mimo-v2.5), cleans and deduplicates the
responses, and synthesises a single superior answer using a judge LLM.

Supports 8 scenarios (coding, bugfix, qa, plan_review, creative,
reasoning, document, general) with scenario-specific panel temps, judge
prompts, cleaning profiles, and conciseness suffixes.

---

## Quick Start

```bash
# Install from source
python3 -m pip install -e .

# Dry-run to verify setup
python3 -m llm_fusion --dry-run --query "What is 2+2?"

# Full pipeline (requires API key)
python3 -m llm_fusion --query "What is 2+2?"
```

## Requirements

- Python 3.10+
- PyYAML (`pip install pyyaml`)
- Network access to OpenAI-compatible API endpoint
- API key: set `OPENCODE_GO_API_KEY` env var or add to `~/.hermes/.env`

## Usage

### CLI

```text
python3 -m llm_fusion [options]
python3 -m llm_fusion --query "Your query here"
python3 -m llm_fusion --dry-run --query "Dry run"    # no API calls
python3 -m llm_fusion --config /path/to/config.yaml --query "..."
python3 -m llm_fusion --output-dir /tmp/results --query "..."
python3 -m llm_fusion --version
```

After `pip install -e .`, the `llm-fusion` console script is also
available (hyphenated):

```text
llm-fusion --query "What is 2+2?"
```

### Hermes Skill (slash command)

Once installed, the skill is available as `/llm-fusion`:

```
/llm-fusion What is the strongest argument for scenario-aware model fusion?
```

### Agent Skills Install

```bash
# Via external Agent Skills directory:
mkdir -p ~/.agents/skills
cp -R .agents/skills/llm-fusion ~/.agents/skills/
# Then add ~/.agents/skills under skills.external_dirs in ~/.hermes/config.yaml
```

## Configuration

The pipeline uses a YAML config file with scenario-specific panel and
judge settings. See `.agents/skills/llm-fusion/references/configuration.md`
for the full reference.

Config discovery order (when `--config` is not specified):
1. `LLM_FUSION_CONFIG` environment variable
2. `./fusion_config.yaml` (current directory)
3. `./.llm-fusion/fusion_config.yaml`
4. `${XDG_CONFIG_HOME:-~/.config}/llm-fusion/fusion_config.yaml`
5. `~/.llm-fusion/fusion_config.yaml`
6. Bundled example (read-only defaults)

## Tests

```bash
python3 -m pytest -q
```

## Project Structure

```
llm-fusion/
├── .agents/skills/llm-fusion/    ← Agent Skills bundle
│   ├── SKILL.md
│   ├── scripts/llm_fusion.py     ← Portable wrapper
│   ├── references/                ← Documentation
│   └── assets/                    ← Example configs
├── src/llm_fusion/                ← Python package
├── tests/                         ← Test suite
├── docs/                          ← Developer docs
├── examples/                      ← Example configs
├── pyproject.toml
└── README.md
```

## Security

- API keys are read from environment variables or `~/.hermes/.env` only.
- Never commit `.env` files, API keys, or user-query result JSON.
- Saved output may contain prompt data — enable saving deliberately.
- Configuration examples use placeholders, never real secrets.

## License

MIT
