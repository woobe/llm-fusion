"""Deprecation shim: Hermes skill handler for llm-fusion.

Previously this module provided a Hermes Agent skill interface.
The Agent Skills bundle under ``.agents/skills/llm-fusion/`` supersedes it.
This shim delegates to the portable CLI for backward compatibility.

Never raises exceptions.
"""


def handle_fusion_trigger(query, config_path=None, verbose=True):
    """Handle a fusion pipeline trigger (backward-compatible shim).

    Delegates to ``llm_fusion.cli``.
    """
    import sys
    try:
        from llm_fusion.pipeline import run_pipeline
        from llm_fusion.output import format_for_chat

        result = run_pipeline(query, config_path=config_path, verbose=verbose)
        return format_for_chat(result, include_metadata=True)

    except Exception as exc:
        return f"[Fusion Skill Error] {exc}"


def get_skill_manifest():
    """Return the skill manifest (backward-compatible shim)."""
    return {
        "name": "llm-fusion",
        "version": "0.1.0",
        "description": "Multi-scenario fusion pipeline that dispatches 6 parallel LLM calls and synthesizes the best answer",
        "author": "snr-dev",
        "triggers": [
            {
                "pattern": r"(?i)\b(fuse|fusion|ensemble|combine)\b",
                "handler": "llm_fusion.skill_handler.handle_fusion_trigger",
                "description": "Trigger fusion pipeline with query containing 'fuse', 'fusion', 'ensemble', or 'combine'",
            },
        ],
        "config_schema": {
            "type": "object",
            "properties": {
                "config_path": {
                    "type": "string",
                    "description": "Path to fusion_config.yaml",
                    "default": "",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Enable verbose logging",
                    "default": False,
                },
            },
        },
    }
