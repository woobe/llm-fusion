"""Panel dispatch for llm-fusion.

Dispatches parallel LLM calls with scenario-specific temperatures,
token budgets, and tier-based model counts.
Uses stdlib's concurrent.futures.ThreadPoolExecutor for parallelism.
Never raises exceptions.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from scripts.api_client import call_llm_with_retry
from scripts.config import load_config, get_scenario_config
from scripts.classifier import CONCISENESS_SUFFIXES


SYSTEM_PROMPT = "You are a knowledgeable assistant. Answer directly."


def _derive_timeout(model_entry, timeout_cfg):
    """Derive a per-model timeout from token budget using the config formula.

    Formula: timeout = max(floor, token_budget / throughput + overhead)
    - If model has thinking.type=adaptive/enabled or reasoning_effort set, multiply by 1.5x
    - If model has an explicit ``timeout`` field, use that directly
    - Falls back to ``timeout_cfg.panel_floor`` if no token budget found

    Parameters
    ----------
    model_entry : dict
        The model config entry (may contain max_tokens, max_completion_tokens,
        thinking, reasoning_effort, and an optional ``timeout`` field).
    timeout_cfg : dict
        Timeout configuration with keys: panel_floor, panel_throughput,
        overhead_seconds, max_timeout.

    Returns
    -------
    int
        Timeout in seconds.
    """
    # If model has an explicit timeout, use it directly
    explicit = model_entry.get("timeout")
    if explicit is not None and isinstance(explicit, (int, float)):
        return int(explicit)

    floor = timeout_cfg.get("panel_floor", 30)
    throughput = timeout_cfg.get("panel_throughput", 25)
    overhead = timeout_cfg.get("overhead_seconds", 10)
    max_timeout = timeout_cfg.get("max_timeout", 300)

    # Determine token budget
    token_budget = (
        model_entry.get("max_completion_tokens")
        or model_entry.get("max_tokens")
        or 0
    )

    if token_budget > 0:
        raw_timeout = token_budget / throughput + overhead
    else:
        raw_timeout = floor

    # Apply thinking multiplier if model uses adaptive/enabled thinking
    uses_thinking = False
    thinking = model_entry.get("thinking")
    if thinking and isinstance(thinking, dict):
        thinking_type = thinking.get("type", "")
        uses_thinking = thinking_type in ("adaptive", "enabled")

    if uses_thinking or model_entry.get("reasoning_effort") or model_entry.get("reasoning_mode"):
        raw_timeout *= 1.5

    timeout = max(floor, int(raw_timeout))
    return min(timeout, max_timeout)


def _build_call_specs(models_list, user_prompt, config):
    """Build a list of call specification dicts from model entries.

    Skips entries where ``count`` is None or ≤ 0.
    Passes ``top_k`` via ``extra_params`` if present.
    Never raises.

    Returns
    -------
    list[dict]
        Each dict has keys: label, model, temperature, top_p,
        max_tokens, max_completion_tokens, timeout, endpoint,
        and optionally extra_params.
    """
    api_cfg = config.get("api", {})
    primary_cfg = api_cfg.get("primary", {})
    timeout_cfg = primary_cfg.get("timeout", {})
    endpoint = primary_cfg.get("endpoint")

    call_specs = []
    for model_entry in models_list:
        model_name = model_entry.get("name", "deepseek-v4-flash")
        count = model_entry.get("count", 1)

        # Skip entries with count <= 0
        if count is None or count <= 0:
            continue

        temp = model_entry.get("temp")
        temps = model_entry.get("temps")
        top_p = model_entry.get("top_p", 0.9)
        max_tokens = model_entry.get("max_tokens")
        max_completion = model_entry.get("max_completion_tokens")
        thinking = model_entry.get("thinking")
        reasoning_effort = model_entry.get("reasoning_effort")
        reasoning_mode = model_entry.get("reasoning_mode")
        top_k = model_entry.get("top_k")

        # Derive per-model adaptive timeout
        model_timeout = _derive_timeout(model_entry, timeout_cfg)

        for i in range(count):
            label = f"{model_name} #{i + 1}"
            call_temp = temp
            if temps and isinstance(temps, list) and i < len(temps):
                call_temp = temps[i]
            elif call_temp is None:
                call_temp = 0.75  # default fallback

            spec = {
                "label": label,
                "model": model_name,
                "temperature": call_temp,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "max_completion_tokens": max_completion,
                "timeout": model_timeout,
                "endpoint": endpoint,
            }
            extra = {}
            if thinking:
                extra["thinking"] = thinking
            if reasoning_effort:
                extra["reasoning_effort"] = reasoning_effort
            if reasoning_mode:
                extra["reasoning_mode"] = reasoning_mode
            if top_k is not None:
                extra["top_k"] = top_k
            if extra:
                spec["extra_params"] = extra
            call_specs.append(spec)

    return call_specs


def _resolve_panel_quorum(config, total_calls):
    """Compute panel quorum from ``pipeline.min_survivors``, clamped to total_calls.

    Quorum is the number of successful panel responses needed to early-exit.
    Returns 0 when *total_calls* is 0 (no quorum needed — existing failure
    behavior applies). Never raises.

    Parameters
    ----------
    config : dict or None
        Full fusion config dict (may be None or empty).
    total_calls : int
        Number of submitted panel call specs.

    Returns
    -------
    int
        Quorum threshold, clamped to ``[0, total_calls]``.
    """
    if not isinstance(total_calls, int) or total_calls <= 0:
        return 0
    if not config or not isinstance(config, dict):
        min_survivors = 2
    else:
        min_survivors = config.get("pipeline", {}).get("min_survivors", 2)
    if not isinstance(min_survivors, (int, float)) or min_survivors < 0:
        min_survivors = 2
    min_survivors = int(min_survivors)
    return min(total_calls, min_survivors)


def dispatch_panel(query, scenario_id, config=None, max_workers=None, tier=None, progress_callback=None, deadline_timestamp=None):
    """Dispatch parallel panel calls for a given scenario.

    Parameters
    ----------
    query : str
        The user query.
    scenario_id : str
        One of the 8 scenario identifiers.
    config : dict or None
        Full fusion config dict. Loaded via load_config() if None.
    max_workers : int or None
        Thread pool size. If ``None``, computed as
        ``min(len(call_specs), config.pipeline.max_panel_workers)``
        with a minimum of 1.
    tier : str or None
        Panel tier (``low1``, ``low2``, ``low3``, ``medium`` (default), ``high``, or ``None`` for default).
    progress_callback : callable or None
        Optional callback invoked with structured event dicts:
        ``panel_started`` (total, max_workers) and
        ``panel_call_completed`` (completed, total, label, success, elapsed, error).
        Must never raise.

    Returns
    -------
    dict with keys:
        success : bool
        responses : list of dict, each with keys:
            label : str (e.g. 'deepseek-v4-flash #1')
            content : str
            model : str
            attempt : int
            success : bool
            error : str or None
            usage : dict or None
        config_used : dict
        elapsed : float
    Never raises.
    """
    start = time.monotonic()
    result = {
        "success": True,
        "responses": [],
        "config_used": {},
        "elapsed": 0.0,
        "total_calls": 0,
        "quorum": 0,
        "quorum_reached": False,
        "quorum_at_ms": None,
        "cancelled_count": 0,
        "late_completed_count": 0,
        "panel_calls_early_exit": False,
    }

    if config is None:
        config = load_config()

    if not config:
        config = {}

    scenario_cfg = get_scenario_config(config, scenario_id, tier=tier)
    result["config_used"] = scenario_cfg

    panel_cfg = scenario_cfg.get("panel", {})
    models_list = panel_cfg.get("models", [])

    if not models_list:
        result["success"] = False
        result["elapsed"] = time.monotonic() - start
        return result

    conciseness_suffix = scenario_cfg.get(
        "conciseness_suffix",
        CONCISENESS_SUFFIXES.get(scenario_id, CONCISENESS_SUFFIXES["general"]),
    )

    # Build the user prompt with conciseness suffix
    user_prompt = f"{query}\n\n{conciseness_suffix}"

    # Retrieve API config for retry/timeout settings — tier-aware
    api_cfg = config.get("api", {})
    primary_cfg = api_cfg.get("primary", {})
    # Try tier-specific retry config from pipeline.retry.<tier>
    tier_retry = {}
    if config:
        tier_retry = config.get("pipeline", {}).get("retry", {}).get(tier, {}) if tier else {}
    if tier_retry.get("max_retries") is not None:
        max_retries = tier_retry["max_retries"]
        delays = list(tier_retry.get("delays_seconds", []))
    else:
        # Fall back to api-level retry
        retry_cfg = primary_cfg.get("retry", {})
        max_retries = retry_cfg.get("max_retries", 2)
        delays = retry_cfg.get("delays_seconds", [1, 3])

    # Build the list of call specifications
    call_specs = _build_call_specs(models_list, user_prompt, config)

    if not call_specs:
        result["success"] = False
        result["elapsed"] = time.monotonic() - start
        return result

    # Determine worker count if not provided
    if max_workers is None:
        pipeline_cfg = config.get("pipeline", {})
        max_allowed = pipeline_cfg.get("max_panel_workers", 6)
        max_workers = min(len(call_specs), max_allowed)
        if max_workers < 1:
            max_workers = 1

    # Emit panel_started event before submitting
    if progress_callback:
        try:
            progress_callback({
                "phase": "panel_started",
                "total": len(call_specs),
                "max_workers": max_workers,
            })
        except Exception:
            pass

    # Execute calls in parallel via ThreadPoolExecutor
    def _do_call(spec):
        """Execute a single panel call."""
        try:
            resp = call_llm_with_retry(
                prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                model=spec["model"],
                temperature=spec["temperature"],
                top_p=spec["top_p"],
                max_tokens=spec.get("max_tokens"),
                max_completion_tokens=spec.get("max_completion_tokens"),
                timeout=spec.get("timeout", 30),
                endpoint=spec.get("endpoint"),
                extra_params=spec.get("extra_params"),
                retries=max_retries,
                delays=delays,
                config=config,
                deadline_timestamp=deadline_timestamp,
            )
            return {
                "label": spec["label"],
                "model": spec["model"],
                "success": resp["success"],
                "content": resp.get("content", ""),
                "reasoning_content": resp.get("reasoning_content"),
                "error": resp.get("error"),
                "usage": resp.get("usage"),
                "elapsed": resp.get("elapsed", 0),
            }
        except Exception as exc:
            return {
                "label": spec["label"],
                "model": spec["model"],
                "success": False,
                "content": "",
                "reasoning_content": None,
                "error": f"Panel dispatch exception: {exc}",
                "usage": None,
                "elapsed": 0,
            }

    total = len(call_specs)
    quorum = _resolve_panel_quorum(config, total)
    result["total_calls"] = total
    result["quorum"] = quorum

    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {}
    collected_futures = set()
    completed = 0
    success_count = 0
    early_exit = False

    try:
        for spec in call_specs:
            future = executor.submit(_do_call, spec)
            futures[future] = spec["label"]

        for future in as_completed(futures):
            collected_futures.add(future)
            try:
                response = future.result()
                result["responses"].append(response)
            except Exception as exc:
                label = futures.get(future, "unknown")
                response = {
                    "label": label,
                    "model": "unknown",
                    "success": False,
                    "content": "",
                    "reasoning_content": None,
                    "error": f"Future exception: {exc}",
                    "usage": None,
                    "elapsed": 0,
                }
                result["responses"].append(response)
            completed += 1
            if response.get("success"):
                success_count += 1
            if progress_callback:
                try:
                    progress_callback({
                        "phase": "panel_call_completed",
                        "completed": completed,
                        "total": total,
                        "label": response.get("label", "?"),
                        "success": response.get("success", False),
                        "elapsed": response.get("elapsed", 0),
                        "error": response.get("error"),
                    })
                except Exception:
                    pass  # callback must never raise

            # Check quorum for early exit
            if success_count >= quorum and quorum > 0:
                early_exit = True
                result["quorum_reached"] = True
                result["quorum_at_ms"] = int((time.monotonic() - start) * 1000)
                result["panel_calls_early_exit"] = True
                break
    finally:
        if early_exit:
            # Cancel pending futures (queued but not yet started)
            for f in futures:
                if not f.done() and f.cancel():
                    result["cancelled_count"] += 1
            # Shutdown without blocking on in-flight calls
            executor.shutdown(wait=False)
            # Best-effort count of late completions
            # Only count futures that completed after break (not cancelled ones)
            for f in futures:
                if f not in collected_futures and f.done() and not f.cancelled():
                    result["late_completed_count"] += 1
            # Emit quorum-reached progress event
            if progress_callback:
                try:
                    progress_callback({
                        "phase": "panel_quorum_reached",
                        "completed": completed,
                        "successful": success_count,
                        "total": total,
                        "quorum": quorum,
                        "elapsed_ms": result["quorum_at_ms"],
                        "cancelled_count": result["cancelled_count"],
                    })
                except Exception:
                    pass
        else:
            executor.shutdown()  # wait=True — existing behavior

    # Sort responses by label for consistent ordering
    result["responses"].sort(key=lambda r: r.get("label", ""))

    result["elapsed"] = time.monotonic() - start

    # Check if enough responses survived
    succeeded = sum(1 for r in result["responses"] if r.get("success"))
    min_survivors = config.get("pipeline", {}).get("min_survivors", 2)
    if succeeded < min_survivors:
        result["success"] = False

    return result
