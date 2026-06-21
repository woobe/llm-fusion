"""Cleaning pipeline for llm-fusion.

Performs scenario-specific cleaning on each panel response:
- Preamble stripping
- Trailing meta stripping
- Code fence handling
- Minimum word length filtering
- Deduplication via SequenceMatcher

Never raises exceptions.
"""

import re
import sys
from difflib import SequenceMatcher


# ---------------------------------------------------------------------------
# Preamble patterns per scenario
# ---------------------------------------------------------------------------
PREAMBLE_PATTERNS = {
    "coding": [
        re.compile(r"^(here(?:'s| is) (?:the |my |a |)(?:code|solution|function|implementation)|sure(?:!|,)|ok(?:ay|)(?:,|!))", re.IGNORECASE),
    ],
    "bugfix": [
        re.compile(r"^(here(?:'s| is) (?:the |my |a |)(?:diagnosis|analysis|fix|solution)|sure(?:!|,)|looking at your|after analyzing)", re.IGNORECASE),
    ],
    "qa": [
        re.compile(r"^(here(?:'s| is)|sure(?:!|,)|ok(?:ay|)(?:,|!)|as an ai|based on|to answer|in response to|let me|i think)", re.IGNORECASE),
        re.compile(r"^(the answer (?:to|is)|regarding|with regards to)", re.IGNORECASE),
    ],
    "plan_review": [
        re.compile(r"^(here(?:'s| is) (?:my |a |)(?:review|analysis|feedback)|sure(?:!|,)|i(?:'ve| have) reviewed)", re.IGNORECASE),
    ],
    "creative": [
        re.compile(r"^(here(?:'s| is) (?:my |a |)(?:take|response|creative|story|idea)|sure(?:!|,)|ok(?:ay|)(?:,|!))", re.IGNORECASE),
    ],
    "reasoning": [
        re.compile(r"^(here(?:'s| is) (?:my |the |)(?:reasoning|solution|answer|calculation)|sure(?:!|,)|to solve|let me work)", re.IGNORECASE),
    ],
    "document": [
        re.compile(r"^(here(?:'s| is) (?:my |a |)(?:review|feedback|analysis|suggested rewrite)|sure(?:!|,)|i(?:'ve| have) (?:reviewed|read|looked at))", re.IGNORECASE),
    ],
    "general": [
        re.compile(r"^(here(?:'s| is)|sure(?:!|,)|ok(?:ay|)(?:,|!)|as an ai|based on|to answer|in response to|let me|i think)", re.IGNORECASE),
    ],
}

TRAILING_PATTERNS = {
    "coding": [
        re.compile(r"(let me know if|feel free to|i hope this|if you have any questions)", re.IGNORECASE),
    ],
    "bugfix": [
        re.compile(r"(let me know if|feel free to|i hope this|hope this helps)", re.IGNORECASE),
    ],
    "qa": [
        re.compile(r"(let me know if|feel free to|i hope this|if you have any (?:questions|other)|is there anything else)", re.IGNORECASE),
    ],
    "plan_review": [],  # KEEP trailing meta
    "creative": [],     # KEEP trailing meta
    "reasoning": [
        re.compile(r"(let me know if|feel free to|i hope this|hope this helps)", re.IGNORECASE),
    ],
    "document": [],     # KEEP trailing meta
    "general": [
        re.compile(r"(let me know if|feel free to|i hope this|if you have any)", re.IGNORECASE),
    ],
}


def clean_response(text, scenario_id, cleaning_profile=None):
    """Apply scenario-specific cleaning to a single response text.

    Steps:
    1. Strip preamble
    2. Strip trailing meta
    3. Handle code fences (strip or preserve)
    4. Strip leading/trailing whitespace

    Parameters
    ----------
    text : str
        The raw response text.
    scenario_id : str
        Scenario identifier (e.g. 'coding', 'qa').
    cleaning_profile : dict or None
        Cleaning profile with keys: strip_fences, strip_preamble, etc.
        If None, uses defaults for the scenario.

    Returns
    -------
    str (cleaned text) or '' if text is empty/invalid.
    Never raises.
    """
    if not text or not isinstance(text, str):
        return ""

    # Use profile or defaults
    if cleaning_profile is None:
        cleaning_profile = {}

    strip_fences = cleaning_profile.get("strip_fences", True)
    strip_preamble = cleaning_profile.get("strip_preamble", True)

    if scenario_id not in PREAMBLE_PATTERNS:
        scenario_id = "general"

    cleaned = text.strip()

    # Step 1: Strip preamble
    if strip_preamble:
        patterns = PREAMBLE_PATTERNS.get(scenario_id, PREAMBLE_PATTERNS["general"])
        for pat in patterns:
            match = pat.match(cleaned)
            if match:
                cleaned = cleaned[match.end():].strip()
                # If after stripping preamble we have a colon or comma, remove it too
                if cleaned.startswith(":") or cleaned.startswith(","):
                    cleaned = cleaned[1:].strip()
                break  # only apply first matched pattern

    # Step 2: Strip trailing meta
    trailing_patterns = TRAILING_PATTERNS.get(scenario_id, TRAILING_PATTERNS["general"])
    for pat in trailing_patterns:
        match = pat.search(cleaned)
        if match:
            cleaned = cleaned[:match.start()].strip()
            break  # only apply first matched pattern

    # Step 3: Handle code fences
    if strip_fences and "```" in cleaned:
        # Strip markdown code fences but keep the content
        cleaned = re.sub(r"```\w*\n?", "", cleaned).strip()

    # Step 4: Final whitespace cleanup
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned


def dedup_responses(responses, scenario_id, cleaning_profile=None):
    """Deduplicate a list of response dicts using pairwise SequenceMatcher.

    Responses are sorted by length (longest first). For each pair, if the
    similarity ratio exceeds the scenario's dedup_threshold, the shorter
    response is discarded (marked with 'discarded': True).

    Parameters
    ----------
    responses : list of dict
        Each dict must have at least 'content' (str) and 'label' (str).
        Other keys are preserved.
    scenario_id : str
        Scenario identifier.
    cleaning_profile : dict or None
        Cleaning profile with dedup_threshold.

    Returns
    -------
    list of dict (same structure, with 'discarded' bool added to each).
    Never raises.
    """
    if not responses:
        return []

    # Get threshold
    threshold = 0.85
    if cleaning_profile is not None:
        threshold = cleaning_profile.get("dedup_threshold", 0.85)
    else:
        # Fallback to defaults
        defaults = {
            "coding": 0.70, "bugfix": 0.75, "qa": 0.85, "plan_review": 0.80,
            "creative": 0.60, "reasoning": 0.75, "document": 0.80, "general": 0.85,
        }
        threshold = defaults.get(scenario_id, 0.85)

    # Add discarded flag to all
    for r in responses:
        r["discarded"] = False

    # Filter out failed responses
    valid = [r for r in responses if r.get("success") and r.get("content")]

    if len(valid) < 2:
        return responses

    # Sort by content length (longest first — keep longer responses)
    valid.sort(key=lambda r: len(r.get("content", "") or ""), reverse=True)

    # Pairwise comparison
    for i in range(len(valid)):
        if valid[i].get("discarded"):
            continue
        for j in range(i + 1, len(valid)):
            if valid[j].get("discarded"):
                continue
            a = valid[i].get("content", "") or ""
            b = valid[j].get("content", "") or ""
            if not a or not b:
                continue
            ratio = SequenceMatcher(None, a, b).ratio()
            if ratio > threshold:
                # Discard the shorter one
                if len(a) >= len(b):
                    valid[j]["discarded"] = True
                else:
                    valid[i]["discarded"] = True
                    break  # current 'i' was discarded, no need to compare more

    return responses


def clean_panel_responses(panel_result, scenario_id, config=None):
    """Full cleaning pipeline for panel responses.

    Steps:
    1. Clean each successful response via clean_response()
    2. Filter out responses below minimum word threshold
    3. Deduplicate via dedup_responses()

    Parameters
    ----------
    panel_result : dict
        Output from dispatch_panel() containing 'responses' list.
    scenario_id : str
        Scenario identifier.
    config : dict or None
        Full fusion config (for cleaning profiles).

    Returns
    -------
    dict with keys:
        cleaned_responses : list of dict (same as panel responses + 'cleaned_content')
        discarded_count : int
        survived_count : int
    Never raises.
    """
    result = {
        "cleaned_responses": [],
        "discarded_count": 0,
        "survived_count": 0,
    }

    from llm_fusion.config import get_cleaning_profile

    cleaning_profile = get_cleaning_profile(config, scenario_id)
    min_words = cleaning_profile.get("min_words", 10)

    raw_responses = panel_result.get("responses", [])

    # Step 1: Clean each response
    cleaned_list = []
    for resp in raw_responses:
        entry = dict(resp)
        if entry.get("success") and entry.get("content"):
            cleaned = clean_response(
                entry["content"], scenario_id, cleaning_profile
            )
            entry["cleaned_content"] = cleaned
            # Check minimum word threshold
            word_count = len(cleaned.split())
            entry["word_count"] = word_count
        else:
            entry["cleaned_content"] = ""
            entry["word_count"] = 0
        cleaned_list.append(entry)

    # Step 2: Filter by min_words
    for entry in cleaned_list:
        if entry.get("success") and entry.get("word_count", 0) < min_words:
            entry["success"] = False
            entry["error"] = (entry.get("error") or "") + (
                f" [discarded: below minimum word count ({entry.get('word_count', 0)} < {min_words})]"
            )

    # Step 3: Deduplicate
    cleaned_list = dedup_responses(cleaned_list, scenario_id, cleaning_profile)

    # Count
    discarded = sum(1 for r in cleaned_list if r.get("discarded") or not r.get("success"))
    survived = sum(1 for r in cleaned_list if r.get("success") and not r.get("discarded"))

    result["cleaned_responses"] = cleaned_list
    result["discarded_count"] = discarded
    result["survived_count"] = survived

    return result
