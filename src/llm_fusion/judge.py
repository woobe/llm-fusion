"""Judge module for llm-fusion.

Single-stage judge: Synthesizes panel responses into a final answer in one pass.
Two-stage judge: Stage 1 produces analysis, Stage 2 produces the final answer.

Never raises exceptions.
"""

from llm_fusion.api_client import call_llm_with_retry


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
    ),
}

# Two-stage prompts (Stage 2 — Synthesis)
JUDGE_SYSTEM_PROMPTS_STAGE2 = {
    "bugfix": (
        "You are a code fix synthesis expert. You receive:\n"
        "1. The original bug report\n"
        "2. Independent bug analyses from multiple models\n"
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
        "2. Multiple independent reviews from different models\n"
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
        "2. Multiple independent solutions\n"
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
        "2. Multiple independent reviews\n"
        "3. A structured issues analysis\n\n"
        "Produce the definitive improved version of the document. Include:\n"
        "1. SUMMARY OF CHANGES: What you changed and why\n"
        "2. IMPROVED DOCUMENT: The full revised document\n"
        "3. KEY IMPROVEMENTS: The 3-5 most impactful changes made\n\n"
        "The improved document should be demonstrably better than the original.\n"
        "Use the original document as the base and apply the best suggestions from all reviews."
    ),
}


def judge_single_stage(query, responses, scenario_id, config=None, judge_config=None):
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
        Full fusion config (for API settings).
    judge_config : dict or None
        Judge-specific config with keys: model, temp, top_p,
        max_completion_tokens, reasoning_mode.
        If None, extracted from config.

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
        from llm_fusion.config import get_scenario_config
        scenario_cfg = get_scenario_config(config, scenario_id)
        judge_config = scenario_cfg.get("judge", {})

    system_prompt = JUDGE_SYSTEM_PROMPTS_SINGLE.get(
        scenario_id,
        JUDGE_SYSTEM_PROMPTS_SINGLE["general"],
    )

    # Build responses section
    responses_section = _build_responses_section(responses)

    user_prompt = (
        f"Original query: {query}\n\n"
        f"Below are the independent responses. Synthesize the best possible answer.\n\n"
        f"{responses_section}"
    )

    api_cfg = (config or {}).get("api", {}).get("primary", {})
    timeout = api_cfg.get("timeout_judge", 60)
    endpoint = api_cfg.get("endpoint")

    llm_result = call_llm_with_retry(
        prompt=user_prompt,
        system_prompt=system_prompt,
        model=judge_config.get("model", "deepseek-v4-flash"),
        temperature=judge_config.get("temp", 0.0),
        top_p=judge_config.get("top_p", 1.0),
        max_completion_tokens=judge_config.get("max_completion_tokens", 8000),
        reasoning_mode=judge_config.get("reasoning_mode"),
        timeout=timeout,
        endpoint=endpoint,
        retries=2,
        delays=(1, 3),
    )

    result["success"] = llm_result["success"]
    result["content"] = llm_result.get("content")
    result["reasoning_content"] = llm_result.get("reasoning_content")
    result["usage"] = llm_result.get("usage")
    result["error"] = llm_result.get("error")
    result["elapsed"] = time.monotonic() - start

    return result


def judge_two_stage(query, responses, scenario_id, config=None, judge_config=None):
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
    }

    import time
    start = time.monotonic()

    if judge_config is None:
        from llm_fusion.config import get_scenario_config
        scenario_cfg = get_scenario_config(config, scenario_id)
        judge_config = scenario_cfg.get("judge", {})

    stage1_config = judge_config.get("stage1", {})
    stage2_config = judge_config.get("stage2", {})

    api_cfg = (config or {}).get("api", {}).get("primary", {})
    timeout = api_cfg.get("timeout_judge", 60)
    endpoint = api_cfg.get("endpoint")
    model = judge_config.get("model", "deepseek-v4-flash")

    responses_section = _build_responses_section(responses)

    # --- Stage 1: Analysis ---
    stage1_system = JUDGE_SYSTEM_PROMPTS_STAGE1.get(
        scenario_id,
        "You are an analysis expert. Analyze the following responses in detail.",
    )

    stage1_prompt = (
        f"Original query: {query}\n\n"
        f"Below are {len(responses)} independent analyses from different models. "
        f"Produce a structured comparative analysis.\n\n"
        f"{responses_section}\n\n"
        f"Produce your structured analysis now. Be thorough and specific \u2014 "
        f"this will be used as input to the synthesis stage."
    )

    stage1_result = call_llm_with_retry(
        prompt=stage1_prompt,
        system_prompt=stage1_system,
        model=model,
        temperature=stage1_config.get("temp", 0.0),
        top_p=stage1_config.get("top_p", 1.0),
        max_completion_tokens=stage1_config.get("max_completion_tokens", 10000),
        reasoning_mode=stage1_config.get("reasoning_mode"),
        timeout=timeout,
        endpoint=endpoint,
        retries=2,
        delays=(1, 3),
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

    stage2_prompt = (
        f"Original query: {query}\n\n"
        f"Below are {len(responses)} independent responses from different models.\n\n"
        f"{responses_section}\n\n"
        f"Below is a structured analysis of these responses:\n\n"
        f"{stage1_content}\n\n"
        f"Synthesize the definitive final answer. Be thorough \u2014 this is the output the user will see."
    )

    stage2_result = call_llm_with_retry(
        prompt=stage2_prompt,
        system_prompt=stage2_system,
        model=model,
        temperature=stage2_config.get("temp", 0.0),
        top_p=stage2_config.get("top_p", 1.0),
        max_completion_tokens=stage2_config.get("max_completion_tokens", 12000),
        reasoning_mode=stage2_config.get("reasoning_mode"),
        timeout=timeout,
        endpoint=endpoint,
        retries=2,
        delays=(1, 3),
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


def _build_responses_section(responses):
    """Build the '=== Label ===\nContent\n' section for judge prompts."""
    parts = []
    for resp in responses:
        label = resp.get("label", "Response")
        content = resp.get("cleaned_content") or resp.get("content", "")
        if content:
            parts.append(f"=== {label} ===\n{content}")
    if not parts:
        parts.append("(no valid responses)")
    return "\n\n".join(parts)
