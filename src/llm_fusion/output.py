"""Output formatting for llm-fusion.

Provides functions to format pipeline results for chat display
and to save results as JSON files.

Never raises exceptions.
"""

import json
import os
import datetime


def format_for_chat(pipeline_result, include_metadata=True):
    """Format a pipeline result into a human-readable chat string.

    Parameters
    ----------
    pipeline_result : dict
        Result dict from run_pipeline().
    include_metadata : bool
        If True, append metadata (scenario, timing, etc.) as a comment block.

    Returns
    -------
    str
        Formatted output string. Returns error message if result is invalid.
    Never raises.
    """
    if not pipeline_result or not isinstance(pipeline_result, dict):
        return "[Error: Invalid pipeline result]"

    success = pipeline_result.get("success", False)
    answer = pipeline_result.get("answer")
    scenario = pipeline_result.get("scenario", "general")
    reasoning_content = pipeline_result.get("reasoning_content")
    metadata = pipeline_result.get("metadata", {})

    if not success:
        error = pipeline_result.get("error", "Unknown error")
        parts = ["[Pipeline Error]", f"Error: {error}", f"Scenario: {scenario}"]
        if include_metadata:
            timing = metadata.get("timing_ms", {})
            if timing:
                parts.append(f"Timing: {timing.get('total', '?')}ms total")
        return "\n".join(parts)

    # Build output
    output_parts = []

    # Include reasoning content if present (for two-stage/high reasoning_mode)
    if reasoning_content:
        output_parts.append(f"[Reasoning]\n{reasoning_content.strip()}\n")

    # Main answer
    if answer:
        output_parts.append(answer.strip())
    else:
        output_parts.append("[No answer content]")

    # Metadata block
    if include_metadata:
        lines = []
        timing = metadata.get("timing_ms", {})
        classification = metadata.get("classification", {})
        panel_info = metadata.get("panel", {})
        judge_info = metadata.get("judge", {})

        parts_list = []
        parts_list.append(f"Scenario: {scenario}")
        if classification:
            parts_list.append(
                f"Detected: {classification.get('scenario', '?')} "
                f"(confidence={classification.get('confidence', '?')}, "
                f"method={classification.get('detection_method', '?')})"
            )
        if panel_info:
            parts_list.append(
                f"Panel: {panel_info.get('models_succeeded', 0)}/"
                f"{panel_info.get('models_attempted', 0)} succeeded, "
                f"{panel_info.get('models_discarded', 0)} discarded"
            )
        if timing:
            parts_list.append(f"Timing: {timing.get('total', '?')}ms total")

        stages = judge_info.get("config", {}).get("stages", "single")
        parts_list.append(f"Judge: {stages}-stage")

        if parts_list:
            lines.append("\u2500\u2500\u2500 " + " | ".join(parts_list) + " \u2500\u2500\u2500")

        if lines:
            output_parts.append("\n" + "\n".join(lines))

    return "\n".join(output_parts)


def save_output(pipeline_result, output_dir=None, filename=None):
    """Save a pipeline result as a JSON file.

    Parameters
    ----------
    pipeline_result : dict
        Result dict from run_pipeline().
    output_dir : str or None
        Directory to save to. Default: './fusion_output/'
    filename : str or None
        Filename. Auto-generated with timestamp if None.

    Returns
    -------
    str or None
        Absolute path of the saved file, or None on failure.
    Never raises.
    """
    try:
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "fusion_output")
        os.makedirs(output_dir, exist_ok=True)

        if filename is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            scenario = pipeline_result.get("scenario", "general")
            filename = f"fusion_{scenario}_{ts}.json"

        filepath = os.path.join(output_dir, filename)

        # Create a serializable copy
        record = dict(pipeline_result)

        # Ensure elapsed is serializable
        if "elapsed" in record:
            record["elapsed"] = round(record["elapsed"], 3)

        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False)

        return os.path.abspath(filepath)

    except Exception as exc:
        print(f"[llm_fusion.output] Failed to save output: {exc}")
        return None
