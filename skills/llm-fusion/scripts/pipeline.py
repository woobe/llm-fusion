"""Pipeline orchestrator for llm-fusion.

Coordinates the full fusion pipeline:
1. Classify query
2. Resolve scenario config
3. Dispatch panel (parallel calls, tier-aware)
4. Clean responses
5. Judge (single or two-stage)
6. Format and return output

Never raises exceptions.
"""

import sys
import time

from scripts.config import load_config, get_scenario_config, normalize_tier
from scripts.classifier import classify_query
from scripts.panel import dispatch_panel
from scripts.cleaner import clean_panel_responses
from scripts.judge import judge_single_stage, judge_two_stage
from scripts.output import format_for_chat, save_output


def run_pipeline(query, config_path=None, output_dir=None, verbose=False, tier=None):
    """Run the full fusion pipeline on a user query.

    Parameters
    ----------
    query : str
        The user query to process.
    config_path : str or None
        Path to fusion_config.yaml. Auto-detected if None.
    output_dir : str or None
        Directory for saving output JSON. If None, no file is saved.
    verbose : bool
        Print progress information if True.
    tier : str or None
        Panel tier (``min``, ``low``, ``medium``, or ``None`` for default).

    Returns
    -------
    dict with keys:
        success : bool
        answer : str or None
        reasoning_content : str or None
        scenario : str
        metadata : dict
    Never raises.
    """
    start = time.monotonic()

    normalized_tier = normalize_tier(tier)



    result = {
        "success": False,
        "answer": None,
        "reasoning_content": None,
        "scenario": "general",
        "metadata": {
            "level": "low",
            "tier": normalized_tier,
            "pipeline_version": "2.0.0",
            "classification": {},
            "panel": {
                "models_attempted": 0,
                "models_succeeded": 0,
                "models_discarded": 0,
            },
            "judge": {},
            "reasoning": None,
            "timing_ms": {},
        },
    }

    if not query or not isinstance(query, str) or not query.strip():
        result["error"] = "Empty query"
        if verbose:
            print("[pipeline] Error: Empty query", file=sys.stderr)
        result["elapsed"] = time.monotonic() - start
        return result

    query = query.strip()

    if verbose:
        print("[pipeline] Loading config...")
    config = load_config(config_path)
    if not config:
        if verbose:
            print("[pipeline] Warning: Using default config (no config file loaded)", file=sys.stderr)

    # Pipeline soft deadline
    pipeline_cfg = config.get("pipeline", {}) if config else {}
    soft_deadline = pipeline_cfg.get("soft_deadline_seconds", 0)
    graceful = pipeline_cfg.get("graceful_degradation", True)

    # Resolve output directory
    resolved_output_dir = output_dir
    if resolved_output_dir is None and config:
        resolved_output_dir = config.get("pipeline", {}).get("output_dir", None)

    def _save(r):
        if resolved_output_dir:
            try:
                save_output(r, output_dir=resolved_output_dir)
            except Exception:
                pass
        return r

    def _check_deadline(phase):
        """Check if pipeline has exceeded the soft deadline.

        Returns the result dict (triggering early return with fallback)
        if the deadline is exceeded, or None to continue normally.

        When ``graceful_degradation`` is false, returns an error instead
        of attempting a fallback call.
        """
        if soft_deadline <= 0:
            return None
        elapsed = time.monotonic() - start
        if elapsed < soft_deadline:
            return None
        # Deadline exceeded — flag metadata
        result["metadata"]["deadline_exceeded"] = True
        if verbose:
            print(f"[pipeline] Soft deadline ({soft_deadline}s) exceeded during {phase}",
                  file=sys.stderr)
        if not graceful:
            result["error"] = f"Soft deadline ({soft_deadline}s) exceeded during {phase}"
            result["elapsed"] = elapsed
            return _save(result)
        if verbose:
            print(f"[pipeline] Falling back to direct call", file=sys.stderr)
        direct_result = _direct_fallback(query, config)
        if direct_result["success"]:
            result["success"] = True
            result["answer"] = direct_result.get("content")
            result["reasoning_content"] = direct_result.get("reasoning_content")
            result["metadata"]["level"] = "low"
            result["metadata"]["judge"] = {"mode": "direct_fallback"}
        else:
            result["error"] = f"Deadline exceeded during {phase}, fallback also failed"
            result["elapsed"] = elapsed
        return _save(result)

    # --- Step 1: Classify ---
    t0 = time.monotonic()
    classification = classify_query(query, config)
    t_class = time.monotonic() - t0
    scenario_id = classification.get("scenario", "general")
    result["scenario"] = scenario_id
    result["metadata"]["classification"] = classification
    if verbose:
        print(f"[pipeline] Classification: {scenario_id} "
              f"(confidence={classification.get('confidence')}, "
              f"method={classification.get('detection_method')}, "
              f"reason={classification.get('reason')})")

    deadline_check = _check_deadline("classification")
    if deadline_check:
        return deadline_check

    # --- Step 2: Resolve scenario config ---
    scenario_cfg = get_scenario_config(config, scenario_id, tier=normalized_tier)

    # --- Step 3: Dispatch panel ---
    t0 = time.monotonic()
    # Count expected parallel calls from scenario models
    panel_models = scenario_cfg.get("panel", {}).get("models", [])
    expected_calls = sum(m.get("count", 0) for m in panel_models)
    if verbose:
        print(f"[pipeline] Dispatching panel ({expected_calls} parallel calls, tier={normalized_tier})...")
    panel_result = dispatch_panel(query, scenario_id, config=config, tier=normalized_tier)
    t_panel = time.monotonic() - t0

    deadline_check = _check_deadline("panel")
    if deadline_check:
        return deadline_check

    panel_metrics = {
        "models_attempted": len(panel_result.get("responses", [])),
        "models_succeeded": sum(
            1 for r in panel_result.get("responses", []) if r.get("success")
        ),
    }
    result["metadata"]["panel"].update(panel_metrics)

    # Store raw panel responses for inspection
    result["panel_responses"] = []
    for r in panel_result.get("responses", []):
        result["panel_responses"].append({
            "label": r.get("label"),
            "model": r.get("model"),
            "success": r.get("success"),
            "content": r.get("content"),
            "reasoning_content": r.get("reasoning_content"),
            "usage": r.get("usage"),
            "error": r.get("error"),
            "timing_ms": r.get("timing_ms"),
        })

    if verbose:
        print(f"[pipeline] Panel complete: "
              f"{panel_metrics['models_succeeded']}/{panel_metrics['models_attempted']} succeeded "
              f"in {t_panel*1000:.0f}ms")

    # Handle panel failure
    if not panel_result.get("success"):
        error = "Panel dispatch failed: insufficient successful responses"
        if verbose:
            print(f"[pipeline] {error}", file=sys.stderr)
        if not graceful:
            result["metadata"]["timing_ms"] = {
                "classification": int(t_class * 1000),
                "panel": int(t_panel * 1000),
                "total": int((time.monotonic() - start) * 1000),
            }
            result["elapsed"] = time.monotonic() - start
            return _save(result)
        # Graceful degradation: try a direct LLM call
        if verbose:
            print("[pipeline] Graceful degradation: making direct LLM call...")
        direct_result = _direct_fallback(query, config)
        if direct_result["success"]:
            result["success"] = True
            result["answer"] = direct_result.get("content")
            result["reasoning_content"] = direct_result.get("reasoning_content")
            result["metadata"]["level"] = "low"
            result["metadata"]["judge"] = {"mode": "direct_fallback"}
            result["metadata"]["timing_ms"] = {
                "classification": int(t_class * 1000),
                "panel": int(t_panel * 1000),
                "total": int((time.monotonic() - start) * 1000),
            }
            result["elapsed"] = time.monotonic() - start
            return _save(result)
        else:
            result["error"] = "Panel and fallback both failed"
            result["elapsed"] = time.monotonic() - start
            return _save(result)

    # --- Step 4: Clean responses ---
    t0 = time.monotonic()
    cleaning_result = clean_panel_responses(panel_result, scenario_id, config=config)
    t_clean = time.monotonic() - t0

    panel_metrics["models_discarded"] = cleaning_result.get("discarded_count", 0)
    result["metadata"]["panel"].update(panel_metrics)

    cleaned_responses = cleaning_result.get("cleaned_responses", [])
    survived = cleaning_result.get("survived_count", 0)

    if verbose:
        print(f"[pipeline] Cleaning: {survived} survived, "
              f"{cleaning_result.get('discarded_count', 0)} discarded "
              f"in {t_clean*1000:.0f}ms")

    deadline_check = _check_deadline("cleaning")
    if deadline_check:
        return deadline_check

    if survived < pipeline_cfg.get("min_survivors", 2):
        if not graceful:
            result["error"] = f"Not enough survivors ({survived} < minimum)"
            result["elapsed"] = time.monotonic() - start
            return _save(result)
        if verbose:
            print(f"[pipeline] Not enough survivors ({survived}), using direct fallback...")
        direct_result = _direct_fallback(query, config)
        if direct_result["success"]:
            result["success"] = True
            result["answer"] = direct_result.get("content")
            result["reasoning_content"] = direct_result.get("reasoning_content")
            result["metadata"]["level"] = "low"
            result["metadata"]["judge"] = {"mode": "direct_fallback"}
            result["metadata"]["timing_ms"] = {
                "classification": int(t_class * 1000),
                "panel": int(t_panel * 1000),
                "cleaning": int(t_clean * 1000),
                "total": int((time.monotonic() - start) * 1000),
            }
            result["elapsed"] = time.monotonic() - start
            return _save(result)

    # --- Step 5: Judge ---
    judge_config = scenario_cfg.get("judge", {})
    judge_stages = judge_config.get("stages", "single")

    t0 = time.monotonic()
    if verbose:
        print(f"[pipeline] Running {'two' if judge_stages == 'two' else 'single'}-stage judge...")

    if judge_stages == "two":
        judge_result = judge_two_stage(
            query, cleaned_responses, scenario_id,
            config=config, judge_config=judge_config,
        )
    else:
        judge_result = judge_single_stage(
            query, cleaned_responses, scenario_id,
            config=config, judge_config=judge_config,
        )
    t_judge = time.monotonic() - t0

    if verbose:
        print(f"[pipeline] Judge {'succeeded' if judge_result.get('success') else 'failed'} "
              f"in {t_judge*1000:.0f}ms")

    if not judge_result.get("success"):
        if not graceful:
            result["error"] = f"Judge failed: {judge_result.get('error')}"
            result["elapsed"] = time.monotonic() - start
            return _save(result)
        if verbose:
            print("[pipeline] Judge failed, using direct fallback...")
        direct_result = _direct_fallback(query, config)
        if direct_result["success"]:
            result["success"] = True
            result["answer"] = direct_result.get("content")
            result["reasoning_content"] = direct_result.get("reasoning_content")
            result["metadata"]["level"] = "low"
            result["metadata"]["judge"] = {"mode": "direct_fallback"}
        else:
            result["error"] = "Judge and fallback both failed"
            result["elapsed"] = time.monotonic() - start
            return _save(result)
    else:
        result["success"] = True
        result["answer"] = judge_result.get("content")
        result["reasoning_content"] = judge_result.get("reasoning_content")

        # Build judge metadata
        if judge_stages == "two":
            result["metadata"]["judge"] = {
                "config": {
                    "stages": "two",
                    "stage1_model": judge_config.get("model", "deepseek-v4-flash"),
                    "stage1_reasoning_mode": judge_config.get("stage1", {}).get("reasoning_mode"),
                    "stage2_model": judge_config.get("model", "deepseek-v4-flash"),
                    "stage2_reasoning_mode": judge_config.get("stage2", {}).get("reasoning_mode"),
                },
                "stage1": {
                    "success": judge_result.get("stage1", {}).get("success"),
                    "usage": judge_result.get("stage1", {}).get("usage"),
                    "elapsed": judge_result.get("stage1", {}).get("elapsed", 0),
                },
                "stage2": {
                    "success": judge_result.get("stage2", {}).get("success"),
                    "usage": judge_result.get("stage2", {}).get("usage"),
                    "elapsed": judge_result.get("stage2", {}).get("elapsed", 0),
                },
            }
        else:
            result["metadata"]["judge"] = {
                "config": {
                    "stages": "single",
                    "model": judge_config.get("model", "deepseek-v4-flash"),
                    "reasoning_mode": judge_config.get("reasoning_mode"),
                },
                "usage": judge_result.get("usage"),
                "elapsed": judge_result.get("elapsed", 0),
            }

        # Set level based on judge success
        result["metadata"]["level"] = "high" if judge_result.get("success") else "low"

    deadline_check = _check_deadline("judge")
    if deadline_check:
        return deadline_check

    # --- Timing ---
    total_elapsed = time.monotonic() - start
    result["metadata"]["timing_ms"] = {
        "classification": int(t_class * 1000),
        "panel": int(t_panel * 1000),
        "cleaning": int(t_clean * 1000),
        "judge": int(t_judge * 1000),
        "total": int(total_elapsed * 1000),
    }
    result["elapsed"] = total_elapsed

    # --- Step 6: Save output ---
    if output_dir is None and config:
        output_dir = config.get("pipeline", {}).get("output_dir", None)
    if output_dir:
        saved_path = save_output(result, output_dir=output_dir)
        if verbose:
            print(f"[pipeline] Output saved to: {saved_path}")

    return _save(result)


def _direct_fallback(query, config):
    """Make a direct single LLM call as graceful degradation fallback.

    Returns call_llm result dict.
    Never raises.
    """
    from scripts.api_client import call_llm_with_retry

    api_cfg = {}
    endpoint = None
    if config and isinstance(config, dict):
        api_cfg = config.get("api", {}).get("primary", {})
        endpoint = api_cfg.get("endpoint")
        timeout_cfg = api_cfg.get("timeout", {})
        fb_timeout = timeout_cfg.get("judge_floor", 60)
    else:
        fb_timeout = 60

    return call_llm_with_retry(
        prompt=query,
        model="deepseek-v4-flash",
        temperature=0.75,
        top_p=0.9,
        max_completion_tokens=2000,
        timeout=fb_timeout,
        endpoint=endpoint,
        retries=1,
        delays=(2,),
    )
