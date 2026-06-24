"""Judge module for llm-fusion.

Single-stage judge: Synthesizes panel responses into a final answer in one pass.
Two-stage judge: Stage 1 produces analysis, Stage 2 produces the final answer.

Never raises exceptions.
"""

from scripts.api_client import call_llm_with_retry


# ---------------------------------------------------------------------------
# Judge system prompts — per scenario
# ---------------------------------------------------------------------------

# Single-stage scenario prompts
JUDGE_SYSTEM_PROMPTS_SINGLE = {
    "coding": (
        "You are a code synthesis expert. You receive multiple independent coding solutions "
        "to the same problem. Your job is to synthesize the BEST possible solution.\n\n"
        "INTERNAL ANALYSIS (use your reasoning for this):\n"
        "- Compare all solutions for correctness (does each handle edge cases?)\n"
        "- Evaluate algorithmic efficiency (time/space complexity)\n"
        "- Check code quality (readability, style, idiomatic patterns)\n"
        "- Identify the best parts of each solution\n"
        "- Detect and discard incorrect or buggy code\n\n"
        "FINAL ANSWER (write this as your visible output):\n"
        "Produce a single, complete, working solution that:\n"
        "- Is correct for all edge cases\n"
        "- Is well-structured and readable\n"
        "- Uses best practices for the language\n"
        "- Is the best synthesis of all submitted solutions\n"
        "- Include imports, docstrings, type hints where appropriate\n"
        "- Wrap code in a markdown code block with language annotation\n\n"
        "Your output should be BETTER than any individual solution."
    ),
    "qa": (
        "You are a factual synthesis expert. You receive multiple independent answers "
        "to the same factual question. Synthesize the single best answer.\n\n"
        "INTERNAL ANALYSIS (use your reasoning for this):\n"
        "- Identify the core facts that all responses agree on (high confidence)\n"
        "- Extract specific details that only some responses cover (fill gaps)\n"
        "- Detect and discard any factual inaccuracies\n"
        "- Resolve contradictions by determining which response is correct\n\n"
        "FINAL ANSWER (write this as your visible output):\n"
        "A single, concise, accurate answer that:\n"
        "- Covers all important facts\n"
        "- Is more complete than any individual response\n"
        "- Gets every detail right\n"
        "- 2-4 sentences, directly answering the question"
    ),
    "creative": (
        "You are a creative synthesis expert. You receive multiple independent creative "
        "responses to the same prompt. Synthesize the best possible creative output.\n\n"
        "INTERNAL ANALYSIS (use your reasoning for this):\n"
        "- Evaluate which responses are most original and compelling\n"
        "- Identify the strongest ideas, phrasings, and perspectives from each\n"
        "- Consider: voice, originality, persuasiveness, emotional impact, structure\n\n"
        "FINAL ANSWER (write this as your visible output):\n"
        "Produce a single creative response that:\n"
        "- Captures the best ideas from all responses\n"
        "- Has a consistent, compelling voice\n"
        "- Is more creative and well-crafted than any individual response\n"
        "- Does NOT feel like a mashup \u2014 it should read as a single coherent piece\n\n"
        "There is no length limit. Let the content dictate the length. Be creative."
    ),
    "general": (
        "You are a fusion synthesis expert. You receive multiple independent responses "
        "to the same query. Your job is to synthesize them into a single superior answer.\n\n"
        "INTERNAL ANALYSIS (use your reasoning for this):\n"
        "- Identify consensus points across all responses\n"
        "- Detect contradictions and resolve them\n"
        "- Extract the most specific, detailed, and accurate information from each\n"
        "- Fill gaps where one response covers something others missed\n"
        "- Discard incorrect or hallucinated content\n\n"
        "FINAL ANSWER (write this as your visible output):\n"
        "Produce a single, coherent answer that:\n"
        "- Covers everything important (comprehensive)\n"
        "- Gets facts right (accurate)\n"
        "- Has clear organization (structured)\n"
        "- Every sentence adds value (concise)"
    ),
}

def _evidence_bundle_instructions(scenario_id):
    """Return shared evidence-bundle formatting instructions for stage 1.

    Appended to the scenario-specific stage 1 system prompt to ensure the
    model produces a compact, structured bundle that stage 2 can consume
    without needing the raw panel responses.
    """
    scenario_fields = {
        "bugfix": (
            "Include these scenario-specific fields in your bundle:\n"
            "- root_cause: the definitive root cause\n"
            "- fix_strategy: the recommended approach\n"
            "- affected_area: which files/functions are affected\n"
            "- edge_cases: edge cases that must be handled"
        ),
        "plan_review": (
            "Include these scenario-specific fields in your bundle:\n"
            "- overall_assessment: summary verdict\n"
            "- critical_issues: prioritized issues\n"
            "- tradeoffs: design trade-offs noted\n"
            "- gaps: what the plan is missing\n"
            "- action_items: prioritized recommended changes"
        ),
        "reasoning": (
            "Include these scenario-specific fields in your bundle:\n"
            "- verified_steps: steps that are correct\n"
            "- wrong_steps: steps that contain errors\n"
            "- correct_path: the correct reasoning path\n"
            "- final_answer_constraints: constraints for the solution"
        ),
        "document": (
            "Include these scenario-specific fields in your bundle:\n"
            "- required_changes: changes needed\n"
            "- section_findings: findings per section\n"
            "- severity_groups: issues grouped by severity (Critical / Important / Minor)\n"
            "- rewrite_plan: ordered rewrite steps"
        ),
    }
    extra = scenario_fields.get(scenario_id, "")
    return (
        "\n\n"
        "Output your analysis as a compact EVIDENCE_BUNDLE with the following structure:\n"
        "EVIDENCE_BUNDLE\n"
        "verdict: <1-3 sentence synthesis direction>\n"
        "key_findings:\n"
        "- <up to 8 bullets; each includes response labels>\n"
        "contradictions:\n"
        "- <up to 5 bullets; include resolution or uncertainty>\n"
        "best_evidence:\n"
        "- <up to 6 short snippets or paraphrases; each <= 240 chars; include response label>\n"
        "missing_or_uncertain:\n"
        "- <up to 5 bullets>\n"
        "final_synthesis_plan:\n"
        "- <up to 6 ordered bullets for stage 2>\n"
        f"{extra}\n\n"
        "Be concise. Do not quote long raw passages — reference response labels. "
        "This bundle is the only default input to the synthesis stage."
    )


# Two-stage prompts (Stage 1 — Analysis)
JUDGE_SYSTEM_PROMPTS_STAGE1 = {
    "bugfix": (
        "You are a bug diagnosis expert. You receive multiple independent analyses of a "
        "software bug. Your job is to analyze all diagnoses and produce a definitive root cause analysis.\n\n"
        "Output a structured analysis covering:\n"
        "1. ROOT CAUSE: The definitive root cause of the bug\n"
        "2. EVIDENCE: What evidence supports this diagnosis (from error messages, stack traces, code)\n"
        "3. CONSENSUS: Where all panel responses agree\n"
        "4. CONTRADICTIONS: Where panel responses disagree \u2014 and which is correct\n"
        "5. MISSING INSIGHTS: Anything important that no panel response covered\n"
        "6. FIX STRATEGY: The recommended approach to fix (high-level, before code)\n\n"
        "Be thorough and precise. This analysis will be passed to a synthesis stage."
        + _evidence_bundle_instructions("bugfix")
        ),
        "plan_review": (
        "You are an architecture and plan review analyst. You receive multiple independent "
        "reviews of the same plan/design/architecture proposal. Produce a structured comparative analysis.\n\n"
        "Output structure:\n"
        "1. OVERALL ASSESSMENT: Summary verdict\n"
        "2. STRENGTHS: What the plan does well (with evidence from reviews)\n"
        "3. WEAKNESSES: Critical issues identified (with evidence)\n"
        "4. TRADEOFFS: Design decisions that have pros/cons\n"
        "5. GAPS: What the plan is missing\n"
        "6. ACTION ITEMS: Prioritized list of recommended changes\n\n"
        "For each point, note which panel responses contributed the insight. Be thorough \u2014 this "
        "analysis will be used to produce the final synthesized review."
        + _evidence_bundle_instructions("plan_review")
        ),
        "reasoning": (
        "You are a reasoning verification expert. You receive multiple independent solutions "
        "to a multi-step reasoning problem. Your job is to verify every step and identify errors.\n\n"
        "For each step in the problem:\n"
        "1. What step was taken?\n"
        "2. Is it correct? (Yes/No/Partially)\n"
        "3. Which panel responses got this step right/wrong?\n"
        "4. What is the correct reasoning for this step?\n\n"
        "Also identify:\n"
        "- Where panel responses diverge and which path is correct\n"
        "- Missing steps that no response covered\n"
        "- Alternative valid approaches\n\n"
        "Output a step-by-step verification table. Be precise and thorough."
        + _evidence_bundle_instructions("reasoning")
        ),
        "document": (
        "You are a document review analyst. You receive multiple independent reviews "
        "of the same document. Produce a structured comparative analysis.\n\n"
        "Cover these dimensions for each section of the document:\n"
        "1. CLARITY: Is the writing clear and understandable?\n"
        "2. COMPLETENESS: Are there missing sections or topics?\n"
        "3. CORRECTNESS: Are there factual or technical errors?\n"
        "4. STRUCTURE: Is the organization logical?\n"
        "5. TONE: Is the tone appropriate for the audience?\n"
        "6. SPECIFIC ISSUES: Line-level or section-level problems identified\n\n"
        "Group findings by severity: Critical / Important / Minor.\n"
        "Note which panel responses identified each issue."
        + _evidence_bundle_instructions("document")
        ),
}

# Two-stage prompts (Stage 2 — Synthesis)
JUDGE_SYSTEM_PROMPTS_STAGE2 = {
    "bugfix": (
        "You are a code fix synthesis expert. You receive:\n"
        "1. The original bug report\n"
        "2. Independent bug analyses from multiple models (or a compact evidence bundle)\n"
        "3. A definitive root cause analysis\n\n"
        "Your job is to produce the definitive fix.\n\n"
        "Produce:\n"
        "1. ROOT CAUSE SUMMARY (1-2 sentences)\n"
        "2. THE FIX: Complete, working code changes. Include the changed file/function.\n"
        "3. EXPLANATION: Why this fix works and what was wrong\n"
        "4. EDGE CASES: Any edge cases handled by this fix\n\n"
        "Wrap all code in markdown code blocks. Be precise and complete."
    ),
    "plan_review": (
        "You are a plan review synthesis expert. You receive:\n"
        "1. The original plan/proposal\n"
        "2. Multiple independent reviews from different models (or a compact evidence bundle)\n"
        "3. A structured comparative analysis\n\n"
        "Produce a definitive, well-organized review that:\n"
        "- Is comprehensive yet readable\n"
        "- Prioritizes the most important issues\n"
        "- Provides specific, actionable recommendations\n"
        "- Is structured with clear sections and priorities\n"
        "- Is BETTER than any individual review\n\n"
        "The final output should be a complete review document that the user can act on directly."
    ),
    "reasoning": (
        "You are a reasoning synthesis expert. You receive:\n"
        "1. The original multi-step problem\n"
        "2. Multiple independent solutions (or a compact evidence bundle)\n"
        "3. A step-by-step verification analysis\n\n"
        "Produce the definitive solution that:\n"
        "- Shows EVERY step with explicit reasoning\n"
        "- Is 100% correct (verified against the analysis)\n"
        "- Explains WHY each step is correct\n"
        "- Handles edge cases or special conditions\n"
        "- Is more complete and accurate than any individual solution\n\n"
        "Do NOT skip steps. Show all work."
    ),
    "document": (
        "You are a document improvement synthesis expert. You receive:\n"
        "1. The original document\n"
        "2. Multiple independent reviews (or a compact evidence bundle)\n"
        "3. A structured issues analysis\n\n"
        "Produce the definitive improved version of the document. Include:\n"
        "1. SUMMARY OF CHANGES: What you changed and why\n"
        "2. IMPROVED DOCUMENT: The full revised document\n"
        "3. KEY IMPROVEMENTS: The 3-5 most impactful changes made\n\n"
        "The improved document should be demonstrably better than the original.\n"
        "Use the original document as the base and apply the best suggestions from all reviews."
    ),
}

def _merge_judge_call_config(judge_config, stage_config=None):
    """Merge top-level judge config with an optional stage override.

    Stage configs inherit model, temp, top_p, token budget, and extra payload
    params from the top-level judge config. Nested stage dictionaries are
    removed from the merged per-call config.
    """
    merged = dict(judge_config or {})
    merged.pop("stage1", None)
    merged.pop("stage2", None)
    merged.update(stage_config or {})
    return merged


def _is_mimo_model(model):
    return "mimo" in (model or "").lower()


def _build_judge_llm_kwargs(call_config, default_max_tokens=4096, default_max_completion_tokens=None):
    """Build call_llm_with_retry kwargs from a merged judge call config.

    Supports both Mimo-style max_tokens/thinking and legacy DeepSeek-style
    max_completion_tokens/reasoning_mode.
    """
    call_config = call_config or {}
    model = call_config.get("model", "mimo-v2.5")
    is_mimo = _is_mimo_model(model)

    kwargs = {
        "model": model,
        "temperature": call_config.get("temp", call_config.get("temperature", 1.0 if is_mimo else 0.0)),
        "top_p": call_config.get("top_p", 0.95 if is_mimo else 1.0),
    }

    if call_config.get("max_tokens") is not None:
        kwargs["max_tokens"] = call_config.get("max_tokens")
    elif call_config.get("max_completion_tokens") is not None:
        kwargs["max_completion_tokens"] = call_config.get("max_completion_tokens")
    elif is_mimo:
        kwargs["max_tokens"] = default_max_tokens
    elif default_max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = default_max_completion_tokens

    # reasoning_mode is a DeepSeek-specific knob. Do not send it to Mimo.
    if call_config.get("reasoning_mode") and not is_mimo:
        kwargs["reasoning_mode"] = call_config.get("reasoning_mode")

    extra_params = {}
    for key in ("thinking", "reasoning_effort", "top_k"):
        if call_config.get(key) is not None:
            extra_params[key] = call_config.get(key)
    if extra_params:
        kwargs["extra_params"] = extra_params

    return kwargs


def _derive_judge_timeout(judge_config, api_cfg):
    """Derive judge timeout from max_tokens or max_completion_tokens.

    Computes timeout for each stage (single-stage config, stage1, stage2) and
    returns the maximum. Applies multipliers for expensive reasoning/thinking:
    - reasoning_mode='high' -> 1.5x
    - reasoning_mode='max' -> 2.0x
    - thinking.type in ('adaptive', 'enabled') -> at least 1.5x
    """
    timeout_cfg = api_cfg.get("timeout", {}) if api_cfg else {}
    floor = timeout_cfg.get("judge_floor", 90)
    throughput = timeout_cfg.get("judge_throughput", 20)
    overhead = timeout_cfg.get("overhead_seconds", 15)
    max_timeout = timeout_cfg.get("max_timeout", 480)

    def _compute(call_config):
        token_budget = (
            call_config.get("max_tokens")
            or call_config.get("max_completion_tokens")
            or 0
        )
        if token_budget > 0:
            raw = token_budget / throughput + overhead
        else:
            raw = floor

        multiplier = 1.0
        rm = (call_config.get("reasoning_mode") or "").lower()
        if rm == "max":
            multiplier = max(multiplier, 2.0)
        elif rm == "high":
            multiplier = max(multiplier, 1.5)

        thinking = call_config.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") in ("adaptive", "enabled"):
            multiplier = max(multiplier, 1.5)

        return raw * multiplier

    candidates = [_compute(_merge_judge_call_config(judge_config))]

    for stage_key in ("stage1", "stage2"):
        stage = (judge_config or {}).get(stage_key, {})
        if stage and isinstance(stage, dict):
            candidates.append(_compute(_merge_judge_call_config(judge_config, stage)))

    raw_timeout = max(candidates)
    timeout = max(floor, int(raw_timeout))
    return min(timeout, max_timeout)


def judge_single_stage(query, responses, scenario_id, config=None, judge_config=None, tier=None):
    """Run a single-stage judge synthesis.

    Takes the original query and cleaned panel responses, builds a prompt
    with the scenario-specific system prompt, and calls the LLM.

    Parameters
    ----------
    query : str
        Original user query.
    responses : list of dict
        Cleaned panel responses with 'label' and 'cleaned_content' keys.
    scenario_id : str
        Scenario identifier.
    config : dict or None
        Full fusion config (for API settings and tier-aware retry config).
    judge_config : dict or None
        Judge-specific config with keys: model, temp, top_p,
        max_completion_tokens, reasoning_mode,
        and optionally max_panel_response_chars.
        If None, extracted from config.
    tier : str or None
        Panel tier for tier-aware retry config. If None, defaults to api-level retry.

    Returns
    -------
    dict with keys:
        success : bool
        content : str or None (final answer)
        reasoning_content : str or None
        usage : dict or None
        error : str or None
        elapsed : float
    Never raises.
    """
    result = {
        "success": False,
        "content": None,
        "reasoning_content": None,
        "usage": None,
        "error": None,
        "elapsed": 0.0,
    }

    import time
    start = time.monotonic()

    if judge_config is None:
        from scripts.config import get_scenario_config
        scenario_cfg = get_scenario_config(config, scenario_id)
        judge_config = scenario_cfg.get("judge", {})

    system_prompt = JUDGE_SYSTEM_PROMPTS_SINGLE.get(
        scenario_id,
        JUDGE_SYSTEM_PROMPTS_SINGLE["general"],
    )

    # Build responses section with optional truncation
    max_chars = judge_config.get("max_panel_response_chars")
    responses_section = _build_responses_section(responses, max_chars=max_chars)

    user_prompt = (
        f"Original query: {query}\n\n"
        f"Below are the independent responses. Synthesize the best possible answer.\n\n"
        f"{responses_section}"
    )

    api_cfg = (config or {}).get("api", {}).get("primary", {})
    timeout = _derive_judge_timeout(judge_config, api_cfg)
    endpoint = api_cfg.get("endpoint")

    # Tier-aware retry: try pipeline.retry.<tier>, fallback to api retry, then defaults
    if config and tier:
        tier_retry = config.get("pipeline", {}).get("retry", {}).get(tier, {})
        judge_retries = tier_retry.get("max_retries")
        judge_delays = list(tier_retry.get("delays_seconds", []))
    else:
        judge_retries = None
        judge_delays = []
    if judge_retries is None:
        retry_cfg = api_cfg.get("retry", {})
        judge_retries = retry_cfg.get("max_retries", 2)
        judge_delays = retry_cfg.get("delays_seconds", [1, 3])

    judge_call_config = _merge_judge_call_config(judge_config)
    judge_kwargs = _build_judge_llm_kwargs(
        judge_call_config,
        default_max_tokens=4096,
        default_max_completion_tokens=8000,
    )

    llm_result = call_llm_with_retry(
        prompt=user_prompt,
        system_prompt=system_prompt,
        timeout=timeout,
        endpoint=endpoint,
        retries=judge_retries,
        delays=judge_delays,
        config=config,
        **judge_kwargs,
    )

    result["success"] = llm_result["success"]
    result["content"] = llm_result.get("content")
    result["reasoning_content"] = llm_result.get("reasoning_content")
    result["usage"] = llm_result.get("usage")
    result["error"] = llm_result.get("error")
    result["elapsed"] = time.monotonic() - start

    return result


def judge_two_stage(query, responses, scenario_id, config=None, judge_config=None, tier=None):
    """Run a two-stage judge (analysis then synthesis).

    Stage 1: Produce a structured analysis of all panel responses.
    Stage 2: Produce the final answer using the analysis from Stage 1.

    Parameters
    ----------
    query : str
        Original user query.
    responses : list of dict
        Cleaned panel responses.
    scenario_id : str
        Scenario identifier.
    config : dict or None
        Full fusion config.
    judge_config : dict or None
        Judge-specific config (must contain 'stage1' and 'stage2' sub-dicts).
    tier : str or None
        Panel tier for tier-aware retry config. If None, defaults to api-level retry.

    Returns
    -------
    dict with keys:
        success : bool
        content : str or None (final answer from stage 2)
        reasoning_content : str or None (from stage 2)
        stage1 : dict (stage 1 result)
        stage2 : dict (stage 2 result)
        usage : dict (combined)
        error : str or None
        elapsed : float
        stage1_input_chars : int (character count of stage 1 prompt)
        stage2_input_chars : int (character count of stage 2 prompt)
    Never raises.
    """
    result = {
        "success": False,
        "content": None,
        "reasoning_content": None,
        "stage1": {},
        "stage2": {},
        "usage": {},
        "error": None,
        "elapsed": 0.0,
        "stage1_input_chars": 0,
        "stage2_input_chars": 0,
    }

    import time
    start = time.monotonic()

    if judge_config is None:
        from scripts.config import get_scenario_config
        scenario_cfg = get_scenario_config(config, scenario_id)
        judge_config = scenario_cfg.get("judge", {})

    stage1_config = judge_config.get("stage1", {})
    stage2_config = judge_config.get("stage2", {})

    api_cfg = (config or {}).get("api", {}).get("primary", {})
    timeout = _derive_judge_timeout(judge_config, api_cfg)
    endpoint = api_cfg.get("endpoint")
    model = judge_config.get("model", "mimo-v2.5")

    # Tier-aware retry for judge stages
    if config and tier:
        tier_retry = config.get("pipeline", {}).get("retry", {}).get(tier, {})
        judge_retries = tier_retry.get("max_retries")
        judge_delays = list(tier_retry.get("delays_seconds", []))
    else:
        judge_retries = None
        judge_delays = []
    if judge_retries is None:
        retry_cfg = api_cfg.get("retry", {})
        judge_retries = retry_cfg.get("max_retries", 2)
        judge_delays = retry_cfg.get("delays_seconds", [1, 3])

    # Build responses section with optional truncation and stats
    max_chars = judge_config.get("max_panel_response_chars")
    responses_section, response_stats = _build_responses_section(
        responses, max_chars=max_chars, return_stats=True
    )
    include_responses_in_stage2 = judge_config.get("stage2_include_raw_responses", False)

    # Populate truncation metadata
    result["panel_response_truncated_count"] = response_stats["truncated_response_count"]
    result["panel_response_truncated_chars"] = response_stats["truncated_chars"]
    result["max_panel_response_chars"] = response_stats["max_panel_response_chars"]
    result["stage2_include_raw_responses"] = include_responses_in_stage2

    # --- Stage 1: Analysis ---
    stage1_system = JUDGE_SYSTEM_PROMPTS_STAGE1.get(
        scenario_id,
        "You are an analysis expert. Analyze the following responses in detail.",
    )

    stage1_prompt = (
        f"Original query: {query}\n\n"
        f"Below are {len(responses)} independent analyses from different models. "
        f"Produce a compact evidence bundle.\n\n"
        f"{responses_section}\n\n"
        f"Analyze these responses and produce a compact evidence bundle "
        f"for the synthesis stage. Be concise and structured."
    )

    result["stage1_input_chars"] = len(stage1_prompt)

    stage1_call_config = _merge_judge_call_config(judge_config, stage1_config)
    stage1_kwargs = _build_judge_llm_kwargs(
        stage1_call_config,
        default_max_tokens=4096,
        default_max_completion_tokens=10000,
    )

    stage1_result = call_llm_with_retry(
        prompt=stage1_prompt,
        system_prompt=stage1_system,
        timeout=timeout,
        endpoint=endpoint,
        retries=judge_retries,
        delays=judge_delays,
        config=config,
        **stage1_kwargs,
    )

    stage1_content = stage1_result.get("content", "")
    stage1_reasoning = stage1_result.get("reasoning_content")

    stage1_elapsed = time.monotonic() - start

    result["stage1"] = {
        "success": stage1_result["success"],
        "content": stage1_content,
        "reasoning_content": stage1_reasoning,
        "usage": stage1_result.get("usage"),
        "error": stage1_result.get("error"),
        "elapsed": stage1_elapsed,
    }

    if not stage1_result["success"]:
        result["error"] = f"Stage 1 failed: {stage1_result.get('error')}"
        result["elapsed"] = time.monotonic() - start
        return result

    # --- Stage 2: Synthesis ---
    stage2_system = JUDGE_SYSTEM_PROMPTS_STAGE2.get(
        scenario_id,
        "You are a synthesis expert. Synthesize the final answer based on the analysis provided.",
    )

    if include_responses_in_stage2:
        stage2_prompt = (
            f"Original query: {query}\n\n"
            f"Below are {len(responses)} independent responses from different models.\n\n"
            f"{responses_section}\n\n"
            f"Below is a structured analysis of these responses:\n\n"
            f"{stage1_content}\n\n"
            f"Synthesize the definitive final answer. Be thorough — this is the output the user will see."
        )
    else:
        stage2_prompt = (
            f"Original query: {query}\n\n"
            f"Below is a structured analysis of the responses:\n\n"
            f"{stage1_content}\n\n"
            f"Synthesize the definitive final answer. Be thorough — this is the output the user will see."
        )

    result["stage2_input_chars"] = len(stage2_prompt)

    stage2_call_config = _merge_judge_call_config(judge_config, stage2_config)
    stage2_kwargs = _build_judge_llm_kwargs(
        stage2_call_config,
        default_max_tokens=4096,
        default_max_completion_tokens=12000,
    )

    stage2_result = call_llm_with_retry(
        prompt=stage2_prompt,
        system_prompt=stage2_system,
        timeout=timeout,
        endpoint=endpoint,
        retries=judge_retries,
        delays=judge_delays,
        config=config,
        **stage2_kwargs,
    )

    stage2_content = stage2_result.get("content", "")
    stage2_reasoning = stage2_result.get("reasoning_content")

    total_elapsed = time.monotonic() - start

    result["stage2"] = {
        "success": stage2_result["success"],
        "content": stage2_content,
        "reasoning_content": stage2_reasoning,
        "usage": stage2_result.get("usage"),
        "error": stage2_result.get("error"),
        "elapsed": total_elapsed - stage1_elapsed,
    }

    result["success"] = stage2_result["success"]
    result["content"] = stage2_content
    result["reasoning_content"] = stage2_reasoning
    result["usage"] = {
        "stage1": stage1_result.get("usage"),
        "stage2": stage2_result.get("usage"),
    }
    result["error"] = stage2_result.get("error")
    result["elapsed"] = total_elapsed

    return result


def _build_responses_section(responses, max_chars=None, return_stats=False):
    """Build the '=== Label ===\nContent\n' section for judge prompts.

    If *max_chars* is set, each response's content is truncated to that many
    characters (with a truncation notice appended) to keep the judge prompt
    within a reasonable size.

    When *return_stats* is True, returns ``(section, stats)`` where *stats* is
    a dict with keys: *response_count*, *included_response_count*,
    *truncated_response_count*, *truncated_chars*, *max_panel_response_chars*.
    When *return_stats* is False (default), returns only the section string
    for backward compatibility.
    """
    parts = []
    response_count = len(responses)
    truncated_count = 0
    truncated_chars = 0
    for resp in responses:
        label = resp.get("label", "Response")
        content = resp.get("cleaned_content") or resp.get("content", "")
        if content:
            original_len = len(content)
            if max_chars is not None and original_len > max_chars:
                truncated_count += 1
                truncated_chars += original_len - max_chars
                content = content[:max_chars] + "\n\n[truncated to first {} chars]".format(max_chars)
            parts.append(f"=== {label} ===\n{content}")
    if not parts:
        parts.append("(no valid responses)")
    section = "\n\n".join(parts)
    if return_stats:
        stats = {
            "response_count": response_count,
            "included_response_count": len(parts) if parts != ["(no valid responses)"] else 0,
            "truncated_response_count": truncated_count,
            "truncated_chars": truncated_chars,
            "max_panel_response_chars": max_chars,
        }
        return section, stats
    return section
