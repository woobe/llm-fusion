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

import os
import sys
import time
import re

from scripts.config import load_config, get_scenario_config, normalize_tier
from scripts.classifier import classify_query
from scripts.panel import dispatch_panel, _resolve_min_survivors
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
        Panel tier (``low1``, ``low2``, ``low3``, ``medium`` (default), ``high``, or ``None`` for default).

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

    # Compute absolute deadline timestamp for deadline-aware retries
    deadline_timestamp = (start + soft_deadline) if soft_deadline > 0 else None

    # Resolve output directory once. If the process is running from an
    # installed skill directory, skip saving entirely to avoid cluttering
    # ~/.hermes/skills/llm-fusion or similar skill install paths.
    cwd = os.getcwd()
    cwd_for_guard = cwd.replace(os.sep, "/")
    if "/skills/" in cwd_for_guard:
        resolved_output_dir = None
    elif output_dir is not None:
        resolved_output_dir = output_dir
    else:
        config_output_dir = None
        if config:
            config_output_dir = config.get("pipeline", {}).get("output_dir")
        resolved_output_dir = config_output_dir or os.path.join(cwd, "fusion_output")

    def _save(r):
        if resolved_output_dir:
            saved_path = save_output(r, output_dir=resolved_output_dir)
            if saved_path:
                r["saved_path"] = saved_path
                if verbose:
                    print(f"[pipeline] Output saved to: {saved_path}")
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
        direct_result = _apply_direct_fallback(result, f"soft_deadline:{phase}", query, config, deadline_timestamp=deadline_timestamp)
        if not direct_result.get("success"):
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

    # --- Express QA path (short-circuit for simple factual questions) ---
    express_cfg = pipeline_cfg.get("express_qa", {})
    # Check exclusion patterns (skip express for explanatory/descriptive queries)
    exclude_patterns = express_cfg.get("exclude_patterns", [])
    is_excluded = any(re.search(pat, query, re.IGNORECASE) for pat in exclude_patterns) if exclude_patterns else False
    if (
        express_cfg.get("enabled", False)
        and scenario_id == "qa"
        and normalized_tier == "low1"
        and len(query) <= express_cfg.get("max_chars", 120)
        and classification.get("confidence", 0.0) >= express_cfg.get("min_confidence", 0.85)
        and classification.get("detection_method") == express_cfg.get("detection_method", "regex")
        and not is_excluded
    ):
        express_start = time.monotonic()
        if verbose:
            print("[pipeline] Express QA short-circuit triggered — skipping panel+judge...")
        from scripts.api_client import call_llm_with_retry
        api_cfg = config.get("api", {}).get("primary", {}) if config else {}
        endpoint = api_cfg.get("endpoint")
        timeout_cfg = api_cfg.get("timeout", {})
        fb_timeout = timeout_cfg.get("judge_floor", 60)
        direct = call_llm_with_retry(
            prompt=query,
            model=express_cfg.get("model", "deepseek-v4-flash"),
            temperature=express_cfg.get("temperature", 0.0),
            max_completion_tokens=express_cfg.get("max_completion_tokens", 600),
            timeout=fb_timeout,
            endpoint=endpoint,
            retries=1,
            delays=(2,),
            config=config,
            deadline_timestamp=deadline_timestamp,
        )
        if direct["success"]:
            result["success"] = True
            result["answer"] = direct.get("content")
            result["reasoning_content"] = direct.get("reasoning_content")
            result["metadata"]["level"] = "express"
            result["metadata"]["judge"] = {
                "mode": "express_direct",
                "model": express_cfg.get("model", "deepseek-v4-flash"),
                "error_category": direct.get("error_category"),
                "attempt_count": direct.get("attempt_count", 1),
                "retryable": direct.get("retryable", False),
                "final_http_status": direct.get("final_http_status"),
                "input_chars": direct.get("input_chars", 0),
                "output_chars": direct.get("output_chars", 0),
                "input_tokens": direct.get("input_tokens"),
                "output_tokens": direct.get("output_tokens"),
                "total_tokens": direct.get("total_tokens"),
            }
            result["elapsed"] = time.monotonic() - express_start
            result["metadata"]["timing_ms"] = {
                "classification": int(t_class * 1000),
                "judge": int((time.monotonic() - express_start) * 1000),
                "total": int((time.monotonic() - start) * 1000),
            }
            return _save(result)
        # Record express QA failure metadata before falling through
        # to normal panel+judge.
        result["metadata"]["express_qa_observability"] = {
            "error": direct.get("error"),
            "error_category": direct.get("error_category"),
            "attempt_count": direct.get("attempt_count", 0),
            "retryable": direct.get("retryable", False),
            "final_http_status": direct.get("final_http_status"),
            "elapsed_ms": int((time.monotonic() - express_start) * 1000),
            "output_chars": direct.get("output_chars", 0),
        }
        if verbose:
            print("[pipeline] Express QA call failed, falling through to normal panel+judge",
                  file=sys.stderr)

    # --- Step 2: Resolve scenario config ---
    scenario_cfg = get_scenario_config(config, scenario_id, tier=normalized_tier)

    # --- Step 3: Dispatch panel ---

    def _panel_progress(event):
        """Render panel progress events to stderr."""
        phase = event.get("phase")
        if phase == "panel_started":
            print(
                f"[llm-fusion] Panel progress: 0/{event['total']} complete "
                f"({event.get('max_workers', '?')} workers)",
                file=sys.stderr, flush=True,
            )
        elif phase == "panel_call_completed":
            status = "ok" if event["success"] else "FAILED"
            elapsed = event.get("elapsed", 0)
            if event["success"]:
                extra = f", {elapsed:.0f}ms"
            else:
                extra = f", error: {event.get('error', '?')}"
            print(
                f"[llm-fusion] Panel progress: {event['completed']}/{event['total']} complete "
                f"({event['label']} {status}{extra})",
                file=sys.stderr, flush=True,
            )
        elif phase == "panel_quorum_reached":
            print(
                f"[llm-fusion] Panel quorum reached: "
                f"{event.get('successful', '?')}/{event.get('total', '?')} successful "
                f"in {event.get('elapsed_ms', '?')}ms; "
                f"cancelled {event.get('cancelled_count', 0)} pending call(s)",
                file=sys.stderr, flush=True,
            )

    t0 = time.monotonic()
    # Count expected parallel calls from scenario models
    panel_models = scenario_cfg.get("panel", {}).get("models", [])
    expected_calls = sum(m.get("count", 0) for m in panel_models)
    if verbose:
        print(f"[pipeline] Dispatching panel ({expected_calls} parallel calls, tier={normalized_tier})...")
    panel_result = dispatch_panel(query, scenario_id, config=config, tier=normalized_tier,
                                  progress_callback=_panel_progress if verbose else None,
                                  deadline_timestamp=deadline_timestamp)
    t_panel = time.monotonic() - t0

    deadline_check = _check_deadline("panel")
    if deadline_check:
        return deadline_check

    panel_metrics = {
        "models_attempted": len(panel_result.get("responses", [])),
        "models_succeeded": sum(
            1 for r in panel_result.get("responses", []) if r.get("success")
        ),
        "models_submitted": panel_result.get("total_calls", expected_calls),
        "quorum": panel_result.get("quorum"),
        "quorum_reached": panel_result.get("quorum_reached", False),
        "quorum_at_ms": panel_result.get("quorum_at_ms"),
        "cancelled_count": panel_result.get("cancelled_count", 0),
        "late_completed_count": panel_result.get("late_completed_count", 0),
        "panel_calls_early_exit": panel_result.get("panel_calls_early_exit", False),
        "attempt_count_total": panel_result.get("attempt_count_total", 0),
        "retryable_error_count": panel_result.get("retryable_error_count", 0),
        "error_categories": panel_result.get("error_categories"),
        "http_statuses": panel_result.get("http_statuses"),
        "usage_totals": panel_result.get("usage_totals"),
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
            "attempt_count": r.get("attempt_count"),
            "retryable": r.get("retryable"),
            "final_http_status": r.get("final_http_status"),
            "http_status": r.get("http_status"),
            "error_category": r.get("error_category"),
            "retry_stopped_reason": r.get("retry_stopped_reason"),
            "input_chars": r.get("input_chars", 0),
            "output_chars": r.get("output_chars", 0),
            "reasoning_output_chars": r.get("reasoning_output_chars", 0),
            "input_tokens": r.get("input_tokens"),
            "output_tokens": r.get("output_tokens"),
            "total_tokens": r.get("total_tokens"),
            "from_fallback": r.get("from_fallback"),
            "fallback_provider": r.get("fallback_provider"),
            "fallback_error": r.get("fallback_error"),
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
        direct_result = _apply_direct_fallback(result, "panel_failure", query, config, deadline_timestamp=deadline_timestamp)
        if direct_result.get("success"):
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

    min_survivors = _resolve_min_survivors(config, tier=normalized_tier)
    if survived < min_survivors:
        if not graceful:
            result["error"] = f"Not enough survivors ({survived} < {min_survivors})"
            result["elapsed"] = time.monotonic() - start
            return _save(result)
        if verbose:
            print(f"[pipeline] Not enough survivors ({survived} < {min_survivors}), using direct fallback...")
        direct_result = _apply_direct_fallback(result, "insufficient_survivors", query, config, deadline_timestamp=deadline_timestamp)
        if direct_result.get("success"):
            result["metadata"]["timing_ms"] = {
                "classification": int(t_class * 1000),
                "panel": int(t_panel * 1000),
                "cleaning": int(t_clean * 1000),
                "total": int((time.monotonic() - start) * 1000),
            }
            result["elapsed"] = time.monotonic() - start
            return _save(result)
        # Fallback also failed — return early (was previously falling
        # through to judge with too few survivors).
        result["error"] = f"Not enough survivors ({survived} < {min_survivors}), fallback also failed"
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
            tier=normalized_tier,
            deadline_timestamp=deadline_timestamp,
        )
    else:
        judge_result = judge_single_stage(
            query, cleaned_responses, scenario_id,
            config=config, judge_config=judge_config,
            tier=normalized_tier,
            deadline_timestamp=deadline_timestamp,
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
        direct_result = _apply_direct_fallback(result, "judge_failure", query, config, deadline_timestamp=deadline_timestamp)
        if not direct_result.get("success"):
            result["error"] = "Judge and fallback both failed"
            result["elapsed"] = time.monotonic() - start
            return _save(result)
    else:
        result["success"] = True
        result["answer"] = judge_result.get("content")
        result["reasoning_content"] = judge_result.get("reasoning_content")

        # Build judge metadata
        if judge_stages == "two":
            stage1_cfg = judge_config.get("stage1", {})
            stage2_cfg = judge_config.get("stage2", {})
            result["metadata"]["judge"] = {
                "config": {
                    "stages": "two",
                    "stage1_model": judge_config.get("model", "mimo-v2.5"),
                    "stage1_max_tokens": stage1_cfg.get("max_tokens", judge_config.get("max_tokens")),
                    "stage1_max_completion_tokens": stage1_cfg.get(
                        "max_completion_tokens", judge_config.get("max_completion_tokens")
                    ),
                    "stage1_reasoning_mode": stage1_cfg.get(
                        "reasoning_mode", judge_config.get("reasoning_mode")
                    ),
                    "stage1_thinking": stage1_cfg.get("thinking", judge_config.get("thinking")),
                    "stage2_model": judge_config.get("model", "mimo-v2.5"),
                    "stage2_max_tokens": stage2_cfg.get("max_tokens", judge_config.get("max_tokens")),
                    "stage2_max_completion_tokens": stage2_cfg.get(
                        "max_completion_tokens", judge_config.get("max_completion_tokens")
                    ),
                    "stage2_reasoning_mode": stage2_cfg.get(
                        "reasoning_mode", judge_config.get("reasoning_mode")
                    ),
                    "stage2_thinking": stage2_cfg.get("thinking", judge_config.get("thinking")),
                    "stage2_include_raw_responses": judge_config.get(
                        "stage2_include_raw_responses", False
                    ),
                    "max_panel_response_chars": judge_config.get("max_panel_response_chars"),
                    "prompt_budget": {
                        "enabled": judge_result.get("prompt_budget_enabled"),
                        "chars_per_token": 4,
                        "strategy": judge_result.get("prompt_budget_strategy", "none"),
                        "warning": judge_result.get("prompt_budget_warning"),
                        "budget_chars": judge_result.get("prompt_budget_chars"),
                        "input_budget_chars": (judge_config or {}).get("prompt_budget", {}).get("input_budget_chars"),
                    },
                },
                "stage1": {
                    "success": judge_result.get("stage1", {}).get("success"),
                    "usage": judge_result.get("stage1", {}).get("usage"),
                    "elapsed": judge_result.get("stage1", {}).get("elapsed", 0),
                    "input_chars": judge_result.get("stage1_input_chars", 0),
                    "input_estimated_tokens": judge_result.get("stage1", {}).get("input_estimated_tokens", 0),
                    "budget_chars": judge_result.get("stage1", {}).get("budget_chars", 0),
                    "budget_compacted": judge_result.get("stage1", {}).get("budget_compacted", False),
                    "budget_warning": judge_result.get("stage1", {}).get("budget_warning", False),
                    "panel_response_truncated_count": judge_result.get(
                        "panel_response_truncated_count", 0
                    ),
                    "panel_response_truncated_chars": judge_result.get(
                        "panel_response_truncated_chars", 0
                    ),
                    "panel_response_compacted_count": judge_result.get("stage1", {}).get(
                        "panel_response_compacted_count", 0
                    ),
                    "panel_response_compacted_chars": judge_result.get("stage1", {}).get(
                        "panel_response_compacted_chars", 0
                    ),
                    "error_category": judge_result.get("stage1", {}).get("error_category"),
                    "attempt_count": judge_result.get("stage1", {}).get("attempt_count", 0),
                    "retryable": judge_result.get("stage1", {}).get("retryable", False),
                    "final_http_status": judge_result.get("stage1", {}).get("final_http_status"),
                },
                "stage2": {
                    "success": judge_result.get("stage2", {}).get("success"),
                    "usage": judge_result.get("stage2", {}).get("usage"),
                    "elapsed": judge_result.get("stage2", {}).get("elapsed", 0),
                    "input_chars": judge_result.get("stage2_input_chars", 0),
                    "input_estimated_tokens": max(1, int(judge_result.get("stage2_input_chars", 0) / 4)),
                    "include_raw_responses": judge_result.get(
                        "stage2_include_raw_responses", False
                    ),
                    "budget_compacted": judge_result.get("stage2", {}).get("budget_compacted", False),
                    "budget_warning": judge_result.get("stage2", {}).get("budget_warning", False),
                    "stage_content_compacted": judge_result.get("stage2", {}).get("stage_content_compacted", False),
                    "stage_content_compacted_chars": judge_result.get("stage2", {}).get("stage_content_compacted_chars", 0),
                    "error_category": judge_result.get("stage2", {}).get("error_category"),
                    "attempt_count": judge_result.get("stage2", {}).get("attempt_count", 0),
                    "retryable": judge_result.get("stage2", {}).get("retryable", False),
                    "final_http_status": judge_result.get("stage2", {}).get("final_http_status"),
                },
            }
        else:
            result["metadata"]["judge"] = {
                "config": {
                    "stages": "single",
                    "model": judge_config.get("model", "mimo-v2.5"),
                    "max_tokens": judge_config.get("max_tokens"),
                    "max_completion_tokens": judge_config.get("max_completion_tokens"),
                    "reasoning_mode": judge_config.get("reasoning_mode"),
                    "thinking": judge_config.get("thinking"),
                    "prompt_budget": {
                        "enabled": judge_result.get("prompt_budget_enabled"),
                        "chars_per_token": 4,
                        "strategy": judge_result.get("prompt_budget_strategy", "none"),
                        "warning": judge_result.get("prompt_budget_warning"),
                        "budget_chars": judge_result.get("prompt_budget_chars"),
                        "input_budget_chars": (judge_config or {}).get("prompt_budget", {}).get("input_budget_chars"),
                    },
                },
                "usage": judge_result.get("usage"),
                "elapsed": judge_result.get("elapsed", 0),
                "error_category": judge_result.get("error_category"),
                "attempt_count": judge_result.get("attempt_count", 0),
                "retryable": judge_result.get("retryable", False),
                "final_http_status": judge_result.get("final_http_status"),
                "prompt_budget": {
                    "enabled": judge_result.get("prompt_budget_enabled"),
                    "exceeded": judge_result.get("prompt_budget_exceeded"),
                    "estimated_input_tokens_before": judge_result.get("estimated_input_tokens_before"),
                    "estimated_input_tokens_after": judge_result.get("estimated_input_tokens_after"),
                    "estimated_input_chars_before": judge_result.get("estimated_input_chars_before"),
                    "estimated_input_chars_after": judge_result.get("estimated_input_chars_after"),
                    "compaction": judge_result.get("compaction_applied"),
                    "trimmed_response_count": judge_result.get("trimmed_response_count"),
                    "trimmed_chars": judge_result.get("trimmed_chars"),
                },
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

    return _save(result)


def _resolve_direct_fallback_config(config):
    """Resolve direct fallback settings from config with safe defaults.

    Reads ``pipeline.direct_fallback`` from *config* and returns a
    normalized dict with all keys needed by :func:`_direct_fallback`.
    When the config block is missing or incomplete, safe defaults
    matching the previous hardcoded behavior are used.

    Returns
    -------
    dict with keys: model, temperature, top_p, max_tokens, timeout,
    retries, delays_seconds, endpoint.
    Never raises.
    """
    defaults = {
        "model": "deepseek-v4-flash",
        "temperature": 0.75,
        "top_p": 0.9,
        "max_tokens": 2000,
        "timeout": 60,
        "retries": 1,
        "delays_seconds": (2,),
        "endpoint": None,
    }

    if not config or not isinstance(config, dict):
        return dict(defaults)

    fb_cfg = config.get("pipeline", {}).get("direct_fallback", {})
    if not isinstance(fb_cfg, dict):
        fb_cfg = {}

    resolved = dict(defaults)

    # Scalar overrides
    for key in ("model", "temperature", "top_p", "retries"):
        val = fb_cfg.get(key)
        if val is not None:
            resolved[key] = val

    # max_tokens is the user-facing config key; map to max_completion_tokens
    # in _direct_fallback's call_llm_with_retry call.
    if "max_tokens" in fb_cfg and fb_cfg["max_tokens"] is not None:
        resolved["max_tokens"] = int(fb_cfg["max_tokens"])

    # Timeout: explicit value > api.primary.timeout.judge_floor > 60
    explicit_timeout = fb_cfg.get("timeout")
    if explicit_timeout is not None:
        resolved["timeout"] = int(explicit_timeout)
    else:
        api_cfg = config.get("api", {}).get("primary", {})
        timeout_cfg = api_cfg.get("timeout", {})
        resolved["timeout"] = timeout_cfg.get("judge_floor", 60)

    # delays_seconds
    delays = fb_cfg.get("delays_seconds")
    if delays is not None and isinstance(delays, (list, tuple)):
        resolved["delays_seconds"] = tuple(delays)
    else:
        resolved["delays_seconds"] = (2,)

    # Endpoint from api.primary.endpoint
    api_cfg = config.get("api", {}).get("primary", {})
    resolved["endpoint"] = api_cfg.get("endpoint")

    return resolved


def _direct_fallback(query, config, deadline_timestamp=None):
    """Make a direct single LLM call as graceful degradation fallback.

    Uses config-driven settings from ``pipeline.direct_fallback``,
    falling back to hardcoded safe defaults when the config block is
    missing.

    Returns call_llm result dict.
    Never raises.
    """
    from scripts.api_client import call_llm_with_retry

    cfg = _resolve_direct_fallback_config(config)

    return call_llm_with_retry(
        prompt=query,
        model=cfg["model"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        max_completion_tokens=cfg["max_tokens"],
        timeout=cfg["timeout"],
        endpoint=cfg["endpoint"],
        retries=cfg["retries"],
        delays=cfg["delays_seconds"],
        config=config,
        deadline_timestamp=deadline_timestamp,
    )


def _apply_direct_fallback(result, reason, query, config, deadline_timestamp=None):
    """Apply direct fallback to *result*, setting fallback metadata.

    Parameters
    ----------
    result : dict
        Pipeline result dict (mutated in place).
    reason : str
        Fallback reason string (e.g. ``"panel_failure"``,
        ``"soft_deadline:panel"``).
    query : str
        Original user query.
    config : dict
        Fusion config dict.

    Returns
    -------
    dict
        The raw ``direct_result`` from :func:`_direct_fallback`.
        Callers can check ``direct_result.get("success")`` to
        determine whether the fallback call itself succeeded.
    Never raises.
    """
    fallback_cfg = _resolve_direct_fallback_config(config)
    fallback_model = fallback_cfg["model"]

    fb_start = time.monotonic()
    direct_result = _direct_fallback(query, config, deadline_timestamp=deadline_timestamp)
    fb_elapsed = time.monotonic() - fb_start

    # Always set fallback metadata
    result["metadata"]["fallback_reason"] = reason
    result["metadata"]["fallback_model"] = fallback_model
    result["metadata"]["fallback_elapsed_ms"] = int(fb_elapsed * 1000)
    result["metadata"]["fallback_error"] = (
        direct_result.get("error") or direct_result.get("fallback_error")
    )
    result["metadata"]["fallback_error_category"] = direct_result.get("error_category")
    result["metadata"]["fallback_attempt_count"] = direct_result.get("attempt_count", 0)
    result["metadata"]["fallback_final_http_status"] = direct_result.get("final_http_status")
    result["metadata"]["fallback_retryable"] = direct_result.get("retryable", False)
    result["metadata"]["fallback_usage"] = direct_result.get("usage")
    result["metadata"]["fallback_output_chars"] = direct_result.get("output_chars", 0)

    if direct_result.get("success"):
        result["success"] = True
        result["answer"] = direct_result.get("content")
        result["reasoning_content"] = direct_result.get("reasoning_content")
        result["metadata"]["level"] = "low"
        result["metadata"]["judge"] = {
            "mode": "direct_fallback",
            "model": fallback_model,
        }
        result.pop("error", None)
    # On failure leave success=False; caller sets the top-level error.

    return direct_result
