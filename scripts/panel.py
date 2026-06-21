"""Panel dispatch for llm-fusion.

Dispatches 6 parallel LLM calls (3x deepseek-v4-flash, 3x mimo-v2.5)
with scenario-specific temperatures and token budgets.
Uses stdlib's concurrent.futures.ThreadPoolExecutor for parallelism.
Never raises exceptions.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from scripts.api_client import call_llm_with_retry
from scripts.config import load_config, get_scenario_config
from scripts.classifier import CONCISENESS_SUFFIXES


SYSTEM_PROMPT = "You are a knowledgeable assistant. Answer directly."


def dispatch_panel(query, scenario_id, config=None, max_workers=6):
    """Dispatch 6 parallel panel calls for a given scenario.

    Parameters
    ----------
    query : str
        The user query.
    scenario_id : str
        One of the 8 scenario identifiers.
    config : dict or None
        Full fusion config dict. Loaded via load_config() if None.
    max_workers : int
        Thread pool size (default 6).

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
    }

    if config is None:
        config = load_config()

    if not config:
        config = {}

    scenario_cfg = get_scenario_config(config, scenario_id)
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

    # Retrieve API config for retry/timeout settings
    api_cfg = config.get("api", {})
    primary_cfg = api_cfg.get("primary", {})
    retry_cfg = primary_cfg.get("retry", {})
    max_retries = retry_cfg.get("max_retries", 2)
    delays = retry_cfg.get("delays_seconds", [1, 3])
    timeout_panel = primary_cfg.get("timeout_panel", 30)
    endpoint = primary_cfg.get("endpoint")

    # Build the list of call specifications
    call_specs = []
    for model_entry in models_list:
        model_name = model_entry.get("name", "deepseek-v4-flash")
        count = model_entry.get("count", 1)
        temp = model_entry.get("temp")
        temps = model_entry.get("temps")
        top_p = model_entry.get("top_p", 0.9)
        max_tokens = model_entry.get("max_tokens")
        max_completion = model_entry.get("max_completion_tokens")
        thinking = model_entry.get("thinking")

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
                "timeout": timeout_panel,
                "endpoint": endpoint,
            }
            if thinking:
                spec["extra_params"] = {"thinking": thinking}
            call_specs.append(spec)

    # Execute calls in parallel via ThreadPoolExecutor
    futures = {}

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

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for spec in call_specs:
            future = executor.submit(_do_call, spec)
            futures[future] = spec["label"]

        for future in as_completed(futures):
            try:
                response = future.result()
                result["responses"].append(response)
            except Exception as exc:
                label = futures.get(future, "unknown")
                result["responses"].append({
                    "label": label,
                    "model": "unknown",
                    "success": False,
                    "content": "",
                    "reasoning_content": None,
                    "error": f"Future exception: {exc}",
                    "usage": None,
                    "elapsed": 0,
                })

    # Sort responses by label for consistent ordering
    result["responses"].sort(key=lambda r: r.get("label", ""))

    result["elapsed"] = time.monotonic() - start

    # Check if enough responses survived
    succeeded = sum(1 for r in result["responses"] if r.get("success"))
    min_survivors = config.get("pipeline", {}).get("min_survivors", 2)
    if succeeded < min_survivors:
        result["success"] = False

    return result
