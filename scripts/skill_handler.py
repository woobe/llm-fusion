"""Hermes skill handler for llm-fusion.

Provides the Hermes Agent skill interface for the fusion pipeline.
Never raises exceptions.
"""


def handle_fusion_trigger(query, config_path=None, verbose=True):
    """Handle a fusion pipeline trigger.

    Delegates to the pipeline and formats output for chat display.

    Parameters
    ----------
    query : str
        The user query to process.
    config_path : str or None
        Path to fusion_config.yaml.
    verbose : bool
        Enable verbose logging.

    Returns
    -------
    str
        Formatted chat output.
    """
    import sys
    try:
        from scripts.pipeline import run_pipeline
        from scripts.output import format_for_chat

        result = run_pipeline(query, config_path=config_path, verbose=verbose)
        return format_for_chat(result, include_metadata=True)

    except Exception as exc:
        return f"[Fusion Skill Error] {exc}"


def get_skill_manifest():
    """Return the skill manifest."""
    return {
        "name": "llm-fusion",
        "version": "0.1.0",
        "description": "Multi-scenario fusion pipeline that dispatches 6 parallel LLM calls and synthesizes the best answer",
        "author": "snr-dev",
        "triggers": [
            {
                "pattern": r"(?i)\b(fuse|fusion|ensemble|combine)\b",
                "handler": "scripts.skill_handler.handle_fusion_trigger",
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
