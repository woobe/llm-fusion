"""Scenario classifier for llm-fusion.

Two-stage detection:
1. Fast regex/keyword pre-classifier (O(1), no API call)
2. Optional LLM second-pass for low-confidence classifications

Never raises exceptions.
"""

import re


# ---------------------------------------------------------------------------
# Conciseness suffixes per scenario (Section 3.2 of the plan)
# ---------------------------------------------------------------------------
CONCISENESS_SUFFIXES = {
    "coding": (
        "Provide a complete, working solution. Include imports, edge cases, and a docstring. "
        "Put code in a markdown code block with the language specified. "
        "Do NOT include preamble like 'Here is the solution' or 'Sure!' — just the answer directly."
    ),
    "bugfix": (
        "Diagnose the root cause first, then provide the fix. Be thorough in your analysis. "
        "Include the fixed code in a markdown code block. "
        "Do NOT include conversational padding like 'I hope this helps' or 'Let me know if you have questions'."
    ),
    "qa": (
        "KEEP IT CONCISE — 2-4 informative sentences. No preamble, no meta-commentary, "
        "no opening phrases like 'Here's my response' or 'Sure!'. Just answer directly."
    ),
    "plan_review": (
        "Be thorough and detailed in your analysis. Do NOT shorten your response. "
        "Cover strengths, weaknesses, tradeoffs, and specific actionable recommendations. "
        "Remove only preamble like 'Here is my review' — everything else is valuable."
    ),
    "creative": (
        "Be creative, expressive, and original. There is no length limit — write as much "
        "as you need to express your ideas fully. Do not include meta-commentary about "
        "your own response (e.g., 'Here is my take'). Just express the ideas."
    ),
    "reasoning": (
        "Show every step of your reasoning explicitly. Do NOT skip intermediate calculations "
        "or logical steps. Conciseness is NOT desired — thoroughness is. "
        "Remove only preamble like 'Here is my reasoning' — then show all steps."
    ),
    "document": (
        "Provide a thorough, detailed review. Do NOT shorten your response. "
        "Include specific line-level or section-level feedback. Cover: clarity, completeness, "
        "correctness, structure, and tone. Be specific and actionable."
    ),
    "general": (
        "KEEP IT CONCISE — 2-4 informative sentences. No preamble, no meta-commentary, "
        "no opening phrases like 'Here's my response' or 'Sure!'. Just answer directly."
    ),
}

# ---------------------------------------------------------------------------
# Pre-classifier rules (Section 1.3 of the plan)
# ---------------------------------------------------------------------------
SCENARIO_RULES = [
    # coding
    {
        "scenario": "coding",
        "patterns": [
            r"```",  # code fences
            r"\bwrite\s+a\s+function\b",
            r"\bimplement\b",
            r"\balgorithm\b",
            r"\bsort\s+a\s+list\b",
            r"\bprogramming\b",
            r"\bPython\b.*\b(function|class)\b",
            r"\bJavaScript\b.*\b(function|class)\b",
            r"\bdef\s+\w+\s*\(.*\)\s*:",
            r"\bclass\s+\w+.*:",
        ],
        "confidence": 0.95,
        "reason": "code-related keywords or fences detected",
    },
    # bugfix
    {
        "scenario": "bugfix",
        "patterns": [
            r"Traceback\b.*",
            r"\bat\s+line\s+\d+",
            r'File\s+"[^"]+"',
            r"\bbug\b",
            r"\bcrash\b",
            r"\berror\b",
            r"\bfix\b",
            r"\bdebug\b",
            r"\bwhy\s+is\s+my\s+code\b",
            r"stack\s+trace",
        ],
        "confidence": 0.90,
        "reason": "bug/error-related keywords or stack traces detected",
    },
    # qa
    {
        "scenario": "qa",
        "patterns": [
            r"^(what|who|where|when|how\s+many|how\s+much|which)\b",
        ],
        "qa_length_max": 120,  # only match if query is short
        "confidence": 0.85,
        "reason": "short factual question detected",
    },
    # plan_review
    {
        "scenario": "plan_review",
        "patterns": [
            r"\breview\b",
            r"\barchitecture\b",
            r"\bproposal\b",
            r"\bdesign\s+doc\b",
            r"\bimprove\s+this\s+plan\b",
            r"\bfeedback\s+on\b",
            r"\bplan\s+review\b",
        ],
        "confidence": 0.85,
        "reason": "plan/design review keywords detected",
    },
    # creative
    {
        "scenario": "creative",
        "patterns": [
            r"\bcreative\b",
            r"\bwrite\s+a\s+story\b",
            r"\bpoem\b",
            r"\bopinion\b",
            r"\bbest\s+approaches?\b",
            r"\bwhat\s+do\s+you\s+think\b",
            r"\bwhat\s+are\s+(some|the)\s+.*\b(ideas?|thoughts?|approaches?)\b",
        ],
        "confidence": 0.80,
        "reason": "creative/opinion keywords detected",
    },
    # reasoning
    {
        "scenario": "reasoning",
        "patterns": [
            r"\bprove\b",
            r"\bcalculate\b",
            r"\bsolve\s+for\b",
            r"\bif\.\.\.then\b",
            r"\bhow\s+many\s+steps\b",
            r"\blogic\s+puzzle\b",
            r"[\d\+\-\*\/\^=]{3,}",  # math notation
            r"\bstep\s+by\s+step\b",
            r"\bshow\s+your\s+work\b",
            r"\bchain\s+of\s+thought\b",
        ],
        "confidence": 0.85,
        "reason": "reasoning/math keywords detected",
    },
    # document
    {
        "scenario": "document",
        "patterns": [
            r"\bimprove\b",
            r"\breview\b",
            r"\bpolish\b",
            r"\brewrite\b",
            r"\bedit\b",
        ],
        "doc_length_min": 500,  # only match if query is long
        "confidence": 0.80,
        "reason": "document improvement keywords detected in long query",
    },
]


def classify_query(query, config=None):
    """Classify a user query into a scenario.

    Stage 1: Regex/keyword pre-classifier.
    If confidence is >= threshold (default 0.85) or no config is provided,
    returns the pre-classifier result directly (fast path).
    If confidence < threshold, triggers LLM second pass (Stage 2).

    Parameters
    ----------
    query : str
        The user query to classify.
    config : dict or None
        Full fusion config dict (from load_config). May contain
        classification settings for LLM second pass.

    Returns
    -------
    dict with keys:
        scenario : str
        confidence : float
        detection_method : str ('regex' or 'llm')
        reason : str
    Never raises.
    """
    if not query or not isinstance(query, str):
        query = ""

    query_lower = query.lower().strip()
    query_len = len(query_lower)

    # --- Stage 1: Regex pre-classifier ---
    best_match = None
    best_score = 0.0

    for rule in SCENARIO_RULES:
        scenario = rule["scenario"]
        score = 0.0
        matched_any = False
        match_reasons = []

        # Check patterns
        for pat in rule["patterns"]:
            if re.search(pat, query_lower, re.IGNORECASE):
                matched_any = True
                match_reasons.append(pat)

        if not matched_any:
            continue

        # Check length constraints
        qa_max = rule.get("qa_length_max")
        doc_min = rule.get("doc_length_min")

        if scenario == "qa" and qa_max and query_len > qa_max:
            # Too long for simple QA — skip entirely
            continue
        elif scenario == "document" and doc_min and query_len < doc_min:
            # Query too short for document review — skip entirely
            continue
        else:
            score = rule.get("confidence", 0.7)

        # Apply modifiers
        # If query has code fences, boost coding
        if scenario == "coding" and "```" in query:
            score = max(score, 0.98)

        # If query has stack traces, boost bugfix
        if scenario == "bugfix" and ("Traceback" in query or 'File "' in query):
            score = max(score, 0.95)

        # QA with question words AND code fences -> not QA
        if scenario == "qa" and "```" in query:
            score = min(score, 0.3)

        if score > best_score:
            best_score = score
            best_match = {
                "scenario": scenario,
                "confidence": round(score, 2),
                "detection_method": "regex",
                "reason": rule.get("reason", "keyword match"),
            }

    # Fallback to general
    if not best_match:
        best_match = {
            "scenario": "general",
            "confidence": 0.60,
            "detection_method": "regex",
            "reason": "no specific scenario matched, fallback to general",
        }

    # --- Stage 2: LLM second pass (if enabled and confidence low) ---
    threshold = 0.85
    if config and isinstance(config, dict):
        cls_config = config.get("classification", {})
        threshold = cls_config.get("confidence_threshold", 0.85)

    if best_match["confidence"] < threshold and config:
        llm_result = _llm_classifier(query, config)
        if llm_result:
            best_match = llm_result
            best_match["detection_method"] = "llm"

    return best_match


def _llm_classifier(query, config):
    """Perform LLM-based second-pass classification.

    Uses deepseek-v4-flash with temp=0.0, max_tokens=50.
    Prompt asks model to classify into one of the 8 scenarios with one word.

    Returns a classifier result dict, or None on failure.
    Never raises.
    """
    try:
        from llm_fusion.api_client import call_llm

        cls_config = config.get("classification", {})
        model = cls_config.get("llm_model", "deepseek-v4-flash")
        temperature = cls_config.get("llm_temp", 0.0)
        max_tokens = cls_config.get("llm_max_tokens", 50)

        prompt = (
            "Classify this query into one: coding|bugfix|qa|plan_review|creative|reasoning|document|general. "
            "Answer with one word only.\n\n"
            f"{query}"
        )

        result = call_llm(
            prompt=prompt,
            model=model,
            temperature=temperature,
            top_p=1.0,
            max_completion_tokens=max_tokens,
            timeout=30,
        )

        if result["success"] and result["content"]:
            response = result["content"].strip().lower().rstrip(".,!?;")
            valid_scenarios = {"coding", "bugfix", "qa", "plan_review",
                               "creative", "reasoning", "document", "general"}
            if response in valid_scenarios:
                return {
                    "scenario": response,
                    "confidence": 0.90,
                    "detection_method": "llm",
                    "reason": f"llm classifier identified as '{response}'",
                }

    except Exception:
        pass

    return None
