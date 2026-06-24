"""API client for llm-fusion.

Handles API key loading, making LLM API calls via urllib (stdlib only),
rate limiting, status-aware retries with exponential backoff, and optional
provider fallback. Never raises exceptions.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request


ENV_PATH = os.path.expanduser("~/.hermes/.env")
ENV_VAR = "OPENCODE_GO_API_KEY"
PRIMARY_ENDPOINT = "https://opencode.ai/zen/go/v1/chat/completions"
FALLBACK_ENDPOINT = "https://openrouter.ai/api/v1"
USER_AGENT = "Hermes-Agent/1.0"

# Default retryable/non-retryable HTTP status sets
_RETRYABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_NON_RETRYABLE_STATUSES = frozenset({400, 401, 403, 404})


# ---------------------------------------------------------------------------
# Config resolvers
# ---------------------------------------------------------------------------


def _resolve_rate_limit_config(config=None):
    """Resolve rate limit settings from config with safe defaults.

    Returns a dict with keys: ``enabled`` (bool), ``requests_per_second``
    (float), ``burst`` (int).  When *config* is None or missing the
    relevant section, defaults match the existing global ``RateLimiter``
    defaults (10 req/s, burst 20, enabled).
    """
    if not config or not isinstance(config, dict):
        return {"enabled": True, "requests_per_second": 10.0, "burst": 20}
    rl = config.get("api", {}).get("rate_limit", {})
    if not isinstance(rl, dict):
        rl = {}
    return {
        "enabled": bool(rl.get("enabled", True)),
        "requests_per_second": float(rl.get("requests_per_second", 10.0)),
        "burst": int(rl.get("burst", 20)),
    }


def _resolve_fallback_config(config=None):
    """Resolve fallback provider config with safe defaults.

    Returns a dict with keys: ``enabled`` (bool), ``provider`` (str),
    ``endpoint`` (str).  When *config* is None or the section is missing,
    fallback is disabled (``enabled=False``).
    """
    if not config or not isinstance(config, dict):
        return {"enabled": False}
    fb = config.get("api", {}).get("fallback", {})
    if not isinstance(fb, dict):
        return {"enabled": False}
    return {
        "enabled": bool(fb.get("enabled", False)),
        "provider": str(fb.get("provider", "openrouter")),
        "endpoint": str(
            fb.get(
                "endpoint",
                "https://openrouter.ai/api/v1/chat/completions",
            )
        ),
    }


def _resolve_retry_policy(config=None, retry_policy=None):
    """Resolve retry policy from config or explicit policy dict.

    When *retry_policy* is a dict it is returned as-is (caller override).
    Otherwise *config* is consulted for ``api.primary.retry`` and its
    ``backoff`` sub-dict.

    Returns a dict with keys: ``max_retries``, ``delays``,
    ``retryable_statuses``, ``non_retryable_statuses``,
    ``backoff_enabled``, ``base_delay_seconds``, ``max_delay_seconds``,
    ``jitter_ratio``.
    """
    if retry_policy and isinstance(retry_policy, dict):
        return dict(retry_policy)

    if config and isinstance(config, dict):
        api_cfg = config.get("api", {}).get("primary", {}).get("retry", {})
        if isinstance(api_cfg, dict):
            backoff = api_cfg.get("backoff", {})
            if not isinstance(backoff, dict):
                backoff = {}
            return {
                "max_retries": int(api_cfg.get("max_retries", 2)),
                "delays": tuple(api_cfg.get("delays_seconds", [1, 3])),
                "retryable_statuses": frozenset(
                    api_cfg.get(
                        "retryable_statuses",
                        _RETRYABLE_STATUSES,
                    )
                ),
                "non_retryable_statuses": frozenset(
                    api_cfg.get(
                        "non_retryable_statuses",
                        _NON_RETRYABLE_STATUSES,
                    )
                ),
                "backoff_enabled": bool(backoff.get("enabled", True)),
                "base_delay_seconds": float(
                    backoff.get("base_delay_seconds", 1.0)
                ),
                "max_delay_seconds": float(
                    backoff.get("max_delay_seconds", 30.0)
                ),
                "jitter_ratio": float(backoff.get("jitter_ratio", 0.25)),
            }

    return {
        "max_retries": 2,
        "delays": (1, 3),
        "retryable_statuses": frozenset(_RETRYABLE_STATUSES),
        "non_retryable_statuses": frozenset(_NON_RETRYABLE_STATUSES),
        "backoff_enabled": True,
        "base_delay_seconds": 1.0,
        "max_delay_seconds": 30.0,
        "jitter_ratio": 0.25,
    }


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------


def read_api_key(env_path=None, env_var=None):
    """Read the API key from .env file or environment variable.

    Tries, in order:
    1. Environment variable *env_var* (default: OPENCODE_GO_API_KEY)
    2. Parsing *env_path* (.env file, default: ~/.hermes/.env)

    Returns the key as str, or None if not found.
    Never raises.
    """
    env_var = env_var or ENV_VAR

    # 1. Try environment variable
    key = os.environ.get(env_var)
    if key:
        return key

    # 2. Try .env file
    path = env_path or ENV_PATH
    try:
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(env_var + "="):
                    return line.split("=", 1)[1]
    except (FileNotFoundError, PermissionError, OSError):
        pass

    # 3. Try HERMES_HOME/.env if no explicit path was given and ~ resolution was wrong
    if env_path is None:
        hermes_home = os.environ.get("HERMES_HOME")
        if hermes_home:
            hermes_path = os.path.join(hermes_home, ".env")
            if hermes_path != path:
                try:
                    with open(hermes_path, "r") as fh:
                        for line in fh:
                            line = line.strip()
                            if line.startswith(env_var + "="):
                                return line.split("=", 1)[1]
                except (FileNotFoundError, PermissionError, OSError):
                    pass

    return None


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------


def _is_retryable_result(result, retryable_statuses=None,
                         non_retryable_statuses=None):
    """Determine whether *result* should be retried.

    Decision order (first match wins):
    1. If ``success`` is true → ``False`` (stop).
    2. If ``http_status`` is in *non_retryable_statuses* → ``False``.
    3. If ``http_status`` is in *retryable_statuses* → ``True``.
    4. If ``http_status`` is ``None`` (transport failure) → ``True``.
    5. Otherwise → ``False``.

    Parameters
    ----------
    result : dict
        The result dict from :func:`call_llm`.
    retryable_statuses : set or frozenset, optional
        HTTP statuses that are retryable.
    non_retryable_statuses : set or frozenset, optional
        HTTP statuses that are not retryable.

    Returns
    -------
    bool
    """
    if result.get("success"):
        return False
    if non_retryable_statuses is None:
        non_retryable_statuses = _NON_RETRYABLE_STATUSES
    if retryable_statuses is None:
        retryable_statuses = _RETRYABLE_STATUSES

    status = result.get("http_status")
    if status in non_retryable_statuses:
        return False
    if status in retryable_statuses:
        return True
    if status is None:
        return True  # transport failure — retry
    return False


def _compute_retry_delay(attempt_index, delays, retry_policy, result,
                         random_func=None):
    """Compute the delay (in seconds) before the next retry attempt.

    Behaviour
    ---------
    - When *retry_policy* has ``backoff_enabled=True`` and the result's
      ``http_status`` is in the retryable set, computes
      ``base * 2 ** attempt_index`` capped at ``max_delay_seconds`` with
      uniform jitter applied.
    - Otherwise uses the legacy *delays* list (compatibility path).

    Parameters
    ----------
    attempt_index : int
        Zero-based attempt number (0 = first retry, 1 = second, ...).
    delays : tuple of float
        Legacy delay list (used when backoff is disabled or status is not
        retryable).
    retry_policy : dict or None
        Retry policy dict as returned by :func:`_resolve_retry_policy`.
    result : dict
        The result dict from :func:`call_llm`.
    random_func : callable or None
        For deterministic testing; ``random_func(a, b)`` should return a
        float in ``[a, b]``.  Defaults to ``random.uniform``.

    Returns
    -------
    float
        Delay in seconds (>= 0).
    """
    if (
        retry_policy
        and isinstance(retry_policy, dict)
        and retry_policy.get("backoff_enabled", True)
    ):
        status = result.get("http_status")
        retryable = retry_policy.get(
            "retryable_statuses", _RETRYABLE_STATUSES
        )
        if status in retryable:
            import random as _random
            rand = random_func if random_func else _random.uniform
            base = float(retry_policy.get("base_delay_seconds", 1.0))
            max_delay = float(retry_policy.get("max_delay_seconds", 30.0))
            jitter_ratio = float(retry_policy.get("jitter_ratio", 0.25))
            delay = base * (2 ** attempt_index)
            delay = min(delay, max_delay)
            delay *= 1.0 + rand(-jitter_ratio, jitter_ratio)
            return max(0.0, delay)
    # Compatibility path: use fixed delays list
    if delays:
        idx = min(attempt_index, len(delays) - 1)
        return float(delays[idx])
    return 1.0  # safe fallback


# ---------------------------------------------------------------------------
# Deadline helpers
# ---------------------------------------------------------------------------


def _deadline_remaining(deadline_timestamp):
    """Return seconds remaining until *deadline_timestamp*, or ``None``.

    Parameters
    ----------
    deadline_timestamp : float or None
        Absolute ``time.monotonic()`` deadline.  ``None`` means no
        deadline is active.

    Returns
    -------
    float or None
        ``max(0.0, deadline_timestamp - time.monotonic())``, or
        ``None`` when *deadline_timestamp* is ``None``.
    """
    if deadline_timestamp is None:
        return None
    return max(0.0, deadline_timestamp - time.monotonic())


def _deadline_allows_retry(deadline_timestamp, delay, buffer_seconds=0.25):
    """Check whether there is enough time before the deadline for a retry.

    Decision:
    - No deadline (``None``) → always allows.
    - Remaining time >= delay + buffer → allows.
    - Otherwise → does not allow.

    Parameters
    ----------
    deadline_timestamp : float or None
        Absolute ``time.monotonic()`` deadline.
    delay : float
        The computed delay before the next attempt.
    buffer_seconds : float
        Safety buffer (default 0.25 seconds).

    Returns
    -------
    (bool, float or None)
        ``(allows, remaining)`` where *allows* is ``True`` when the
        deadline allows the retry, and *remaining* is the seconds
        remaining (0.0 if already past deadline) or ``None`` when no
        deadline is set.
    """
    if deadline_timestamp is None:
        return True, None
    remaining = deadline_timestamp - time.monotonic()
    if remaining <= 0.0:
        return False, 0.0
    return remaining >= (delay + buffer_seconds), max(0.0, remaining)


# ---------------------------------------------------------------------------
# Observability helpers
# ---------------------------------------------------------------------------


def _categorize_error(error=None, http_status=None,
                      retry_stopped_reason=None):
    """Categorize an error into a compact vocabulary.

    Decision order:
    1. If ``http_status`` is set and success (2xx), return ``None``.
    2. If ``http_status`` is set, use the HTTP status code.
    3. If ``retry_stopped_reason`` contains ``deadline``, return
       ``"timeout"``.
    4. If ``error`` text matches key patterns, derive category from text.
    5. If ``http_status`` is ``None`` (transport failure), return
       ``"network_error"``.
    6. Otherwise return ``None`` (no error).

    Parameters
    ----------
    error : str or None
        The error string from a result dict.
    http_status : int or None
        HTTP status code.
    retry_stopped_reason : str or None
        Retry stop reason (e.g. ``"deadline_exceeded"``).

    Returns
    -------
    str or None
        Category from the vocabulary, or ``None`` when there is no error
        (success path).
    """
    if http_status is not None:
        if 200 <= http_status <= 299:
            return None  # success - no error
        if http_status in (401, 403):
            return "auth_error"
        if http_status == 429:
            return "rate_limited"
        if http_status in (408, 409, 425):
            return "conflict_retryable"
        if http_status in (400, 404):
            return "bad_request"
        if 500 <= http_status <= 599:
            return "server_error"
        if 400 <= http_status <= 499:
            return "bad_request"
        return None

    # No HTTP status - derive from error text or stop reason
    if retry_stopped_reason and "deadline" in retry_stopped_reason:
        return "timeout"

    if error:
        err_lower = error.lower()
        if "timeout" in err_lower or "timed out" in err_lower:
            return "timeout"
        if "json decode" in err_lower or "parse" in err_lower:
            return "parse_error"
        if "no api key" in err_lower or "api key" in err_lower:
            return "config_error"
        if "auth" in err_lower or "unauthorized" in err_lower:
            return "auth_error"
        return "unknown_error"

    # http_status is None with no error text - transport / network failure
    return "network_error"



def _normalize_usage_counters(usage):
    """Normalize usage dict to canonical ``input_tokens``, ``output_tokens``,
    ``total_tokens`` keys.

    Accepts OpenAI-style keys (``prompt_tokens``, ``completion_tokens``,
    ``total_tokens``) and provider-style keys (``input_tokens``,
    ``output_tokens``).  Never raises on malformed input.

    Returns
    -------
    dict with keys ``input_tokens``, ``output_tokens``, ``total_tokens``.
    Each value is ``int`` or ``None``.
    """
    result = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }

    if not usage or not isinstance(usage, dict):
        return result

    # input / prompt tokens
    input_t = usage.get("input_tokens") or usage.get("prompt_tokens")
    if input_t is not None:
        try:
            result["input_tokens"] = int(input_t)
        except (ValueError, TypeError):
            pass

    # output / completion tokens
    output_t = usage.get("output_tokens") or usage.get("completion_tokens")
    if output_t is not None:
        try:
            result["output_tokens"] = int(output_t)
        except (ValueError, TypeError):
            pass

    # total tokens (use explicit or compute)
    total_t = usage.get("total_tokens")
    if total_t is not None:
        try:
            result["total_tokens"] = int(total_t)
        except (ValueError, TypeError):
            pass

    if result["total_tokens"] is None:
        if result["input_tokens"] is not None and result["output_tokens"] is not None:
            try:
                result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
            except TypeError:
                pass

    return result


def _attach_call_observability(result, prompt=None, system_prompt=None,
                               retry_policy=None):
    """Attach safe observability fields to *result* dict in-place.

    Adds the following fields if not already set:
    - ``error_category``
    - ``attempt_count`` (default 1)
    - ``retryable``
    - ``final_http_status``
    - ``prompt_chars``, ``system_prompt_chars``, ``input_chars``
    - ``output_chars``, ``reasoning_output_chars``
    - ``input_tokens``, ``output_tokens``, ``total_tokens``
    - ``attempts`` (if not already present and attempt_count is set)

    Uses the result's existing ``error``, ``http_status``,
    ``retry_stopped_reason`` fields plus the optional char-based input
    arguments.

    Never raises and never mutates existing fields that are already of
    the correct type.
    """
    # --- error_category ---
    if "error_category" not in result:
        result["error_category"] = _categorize_error(
            error=result.get("error"),
            http_status=result.get("http_status"),
            retry_stopped_reason=result.get("retry_stopped_reason"),
        )

    # --- attempt_count ---
    if "attempt_count" not in result:
        # If attempts is already set (from retry wrapper), use it;
        # otherwise default to 1 for a single call.
        result["attempt_count"] = result.get("attempts", 1)

    # --- retryable ---
    if "retryable" not in result:
        if result.get("success"):
            result["retryable"] = False
        else:
            from scripts.api_client import _is_retryable_result
            result["retryable"] = _is_retryable_result(
                result,
                retryable_statuses=(
                    (retry_policy or {}).get("retryable_statuses")
                ) if retry_policy else None,
                non_retryable_statuses=(
                    (retry_policy or {}).get("non_retryable_statuses")
                ) if retry_policy else None,
            )
        # Override retryable for non-retryable error categories
        # (config errors, auth errors, bad requests, parse errors)
        cat = result.get("error_category")
        if cat in ("config_error", "auth_error", "bad_request",
                   "parse_error", "empty_response"):
            result["retryable"] = False

    # --- final_http_status ---
    if "final_http_status" not in result:
        result["final_http_status"] = result.get("http_status")

    # --- Char counters ---
    prompt_text = prompt or ""
    system_text = system_prompt or ""
    output_text = result.get("content") or ""
    reasoning_text = result.get("reasoning_content") or ""

    result["prompt_chars"] = len(prompt_text)
    result["system_prompt_chars"] = len(system_text)
    result["input_chars"] = len(prompt_text) + len(system_text)
    result["output_chars"] = len(output_text)
    result["reasoning_output_chars"] = len(reasoning_text)

    # --- Normalized token counters ---
    usage = result.get("usage")
    if usage and isinstance(usage, dict):
        normalized = _normalize_usage_counters(usage)
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if key not in result:
                result[key] = normalized[key]
    else:
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if key not in result:
                result[key] = None

    # --- Backward-compatible attempts alias ---
    if "attempts" not in result and "attempt_count" in result:
        result["attempts"] = result["attempt_count"]

    return result


# ---------------------------------------------------------------------------
# Provider fallback attempt
# ---------------------------------------------------------------------------


def _attempt_provider_fallback(prompt, fallback_config, **kwargs):
    """Make a single provider-fallback LLM call.

    This is the minimal fallback-attempt helper used by
    :func:`call_llm_with_retry` after primary retries are exhausted.
    It does NOT wrap retry logic — it is a single HTTP request to the
    fallback provider endpoint.

    Parameters
    ----------
    prompt : str
        The user message content.
    fallback_config : dict
        Fallback config from :func:`_resolve_fallback_config`.
    **kwargs
        Forwarded from the original call: ``model``, ``temperature``,
        ``top_p``, ``timeout``, ``system_prompt``, ``extra_params``, etc.

    Returns
    -------
    dict with the same shape as :func:`call_llm`, plus keys
    ``from_fallback`` (True) and ``fallback_provider``.
    On failure returns a dict with ``success=False``.
    Never raises.
    """
    start = time.monotonic()
    result = {
        "success": False,
        "data": None,
        "content": None,
        "reasoning_content": None,
        "usage": None,
        "error": None,
        "http_status": None,
        "elapsed": 0.0,
        "attempt_count": 1,
        "prompt_chars": 0,
        "system_prompt_chars": 0,
        "input_chars": 0,
        "output_chars": 0,
        "reasoning_output_chars": 0,
        "from_fallback": True,
        "fallback_provider": fallback_config.get("provider", "openrouter"),
    }

    try:
        from scripts.fallback import FALLBACK_CONFIGS, rate_limited_request
        from scripts.api_client import read_api_key

        provider = fallback_config.get("provider", "openrouter")
        endpoint = fallback_config.get("endpoint")
        provider_cfg = FALLBACK_CONFIGS.get(provider, {})

        if not endpoint:
            endpoint = provider_cfg.get(
                "endpoint",
                "https://openrouter.ai/api/v1/chat/completions",
            )

        api_key = read_api_key()
        if not api_key:
            result["error"] = "No API key for fallback"
            result["elapsed"] = time.monotonic() - start
            _attach_call_observability(result, prompt=prompt)
            return result

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        extra_headers = provider_cfg.get("headers", {})
        headers.update(extra_headers)

        # Build minimal payload (system prompt preserved if present)
        messages = []
        system_prompt = kwargs.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": kwargs.get("model", "deepseek-v4-flash"),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.75),
            "top_p": kwargs.get("top_p", 0.9),
        }
        # Token params
        if kwargs.get("max_tokens") is not None:
            payload["max_tokens"] = kwargs["max_tokens"]
        elif kwargs.get("max_completion_tokens") is not None:
            payload["max_completion_tokens"] = kwargs["max_completion_tokens"]
        else:
            payload["max_tokens"] = 2000

        if kwargs.get("reasoning_mode"):
            payload["reasoning_mode"] = kwargs["reasoning_mode"]

        extra = kwargs.get("extra_params")
        if extra and isinstance(extra, dict):
            payload.update(extra)

        data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data_bytes,
            headers=headers,
            method="POST",
        )

        timeout = kwargs.get("timeout", 60)
        http_status, raw_body, error = rate_limited_request(
            req, timeout=timeout, enabled=True,
        )

        elapsed = time.monotonic() - start
        result["http_status"] = http_status
        result["elapsed"] = elapsed

        if error or http_status is None or http_status >= 400:
            result["error"] = error or f"HTTP {http_status}"
            _attach_call_observability(result, prompt=prompt)
            return result

        parsed = json.loads(raw_body)  # type: ignore[arg-type]
        result["data"] = parsed
        result["success"] = True

        try:
            msg = parsed["choices"][0]["message"]
            result["content"] = msg.get("content")
            result["reasoning_content"] = msg.get("reasoning_content")
        except (KeyError, IndexError, TypeError):
            pass
        result["usage"] = parsed.get("usage")

    except json.JSONDecodeError as exc:
        result["error"] = f"Fallback JSON decode error: {exc}"
    except Exception as exc:
        result["error"] = f"Fallback exception: {exc}"

    result["elapsed"] = time.monotonic() - start
    _attach_call_observability(result, prompt=prompt)
    return result


# ---------------------------------------------------------------------------
# Primary LLM call  (single request, rate-limited)
# ---------------------------------------------------------------------------


def call_llm(
    prompt,
    system_prompt=None,
    model="deepseek-v4-flash",
    temperature=0.75,
    top_p=0.9,
    max_tokens=None,
    max_completion_tokens=None,
    reasoning_mode=None,
    timeout=60,
    endpoint=None,
    api_key=None,
    extra_params=None,
    config=None,
    rate_limiter=None,
    fallback_config=None,
    retry_policy=None,
):
    """Call an LLM chat completions endpoint and return the parsed response.

    Every outbound network request goes through rate limiting (unless
    rate limiting is explicitly disabled via config).  The result shape
    is the same regardless of success or failure — callers should check
    ``result["success"]``.

    Parameters
    ----------
    prompt : str
        The user message content.
    system_prompt : str or None
        Optional system message.
    model : str
        Model name (e.g. 'deepseek-v4-flash', 'mimo-v2.5').
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling parameter.
    max_tokens : int or None
        For models that use 'max_tokens' (mimo-v2.5).
    max_completion_tokens : int or None
        For models that use 'max_completion_tokens' (deepseek-v4-flash).
    reasoning_mode : str or None
        'high', 'max', or None.
    timeout : int
        Request timeout in seconds.
    endpoint : str or None
        API endpoint URL. Defaults to PRIMARY_ENDPOINT.
    api_key : str or None
        API key. If None, read via read_api_key().
    extra_params : dict or None
        Additional JSON payload keys. Values here are authoritative; for
        example, a caller may pass ``{"thinking": {"type": "enabled"}}``
        for Mimo judge calls.
    config : dict or None
        Full fusion config dict (from ``load_config``). Used to resolve
        rate limit, fallback, and retry policy settings.
    rate_limiter : RateLimiter or None
        Injected rate limiter for deterministic testing. When None, a
        global limiter is created from config via ``get_rate_limiter``.
    fallback_config : dict or None
        Pre-resolved fallback config dict. Only used by
        :func:`call_llm_with_retry`; ignored in the single-call path.
    retry_policy : dict or None
        Pre-resolved retry policy dict. Only used by
        :func:`call_llm_with_retry`; ignored in the single-call path.

    Returns
    -------
    dict with keys:
        success: bool
        data: dict or None (parsed API response)
        content: str or None (extracted assistant message)
        reasoning_content: str or None
        usage: dict or None
        error: str or None
        http_status: int or None
        elapsed: float
    Never raises.
    """
    start = time.monotonic()
    result = {
        "success": False,
        "data": None,
        "content": None,
        "reasoning_content": None,
        "usage": None,
        "error": None,
        "http_status": None,
        "elapsed": 0.0,
        "attempt_count": 1,
        "prompt_chars": 0,
        "system_prompt_chars": 0,
        "input_chars": 0,
        "output_chars": 0,
        "reasoning_output_chars": 0,
    }

    # Resolve API key
    if not api_key:
        api_key = read_api_key()
    if not api_key:
        result["error"] = (
            "No API key found. Set OPENCODE_GO_API_KEY env var "
            "or ensure ~/.hermes/.env exists."
        )
        result["elapsed"] = time.monotonic() - start
        _attach_call_observability(result, prompt=prompt,
                                   system_prompt=system_prompt)
        return result

    # Resolve endpoint
    if not endpoint:
        endpoint = PRIMARY_ENDPOINT

    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Build payload
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }

    # Different models use different token params
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        payload["max_completion_tokens"] = max_completion_tokens

    # reasoning_mode for deepseek models
    if reasoning_mode:
        payload["reasoning_mode"] = reasoning_mode

    # Extra params are caller-authoritative
    if extra_params:
        payload.update(extra_params)

    # Mimo panel calls default to thinking disabled
    if "mimo" in model.lower() and "thinking" not in payload:
        payload["thinking"] = {"type": "disabled"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }

    data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=data_bytes, headers=headers, method="POST",
    )

    # Resolve rate limiter
    rl_config = _resolve_rate_limit_config(config)
    if rate_limiter is None:
        from scripts.fallback import get_rate_limiter
        rate_limiter = get_rate_limiter(
            rate=rl_config["requests_per_second"],
            burst=rl_config["burst"],
        )

    # Make the request (rate-limited or direct)
    from scripts.fallback import rate_limited_request as _rl_req
    try:
        http_status, raw_body, error = _rl_req(
            req,
            timeout=timeout,
            rate_limiter=rate_limiter,
            enabled=rl_config["enabled"],
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        result["elapsed"] = elapsed
        result["error"] = f"Unexpected transport error: {exc}"
        _attach_call_observability(result, prompt=prompt,
                                   system_prompt=system_prompt)
        return result

    elapsed = time.monotonic() - start
    result["http_status"] = http_status
    result["elapsed"] = elapsed

    if error or http_status is None or http_status >= 400:
        result["error"] = error or f"HTTP {http_status}"
        _attach_call_observability(result, prompt=prompt,
                                   system_prompt=system_prompt)
        return result

    # Parse JSON body
    try:
        parsed = json.loads(raw_body)  # type: ignore[arg-type]
        result["data"] = parsed
        result["success"] = True

        # Extract content
        try:
            msg = parsed["choices"][0]["message"]
            result["content"] = msg.get("content")
            result["reasoning_content"] = msg.get("reasoning_content")
        except (KeyError, IndexError, TypeError):
            pass

        # Extract usage
        result["usage"] = parsed.get("usage")

    except json.JSONDecodeError as exc:
        result["error"] = f"JSON decode error: {exc}"

    _attach_call_observability(result, prompt=prompt,
                               system_prompt=system_prompt)
    return result


# ---------------------------------------------------------------------------
# Retrying + fallback orchestrator
# ---------------------------------------------------------------------------


def call_llm_with_retry(
    prompt,
    retries=2,
    delays=(1, 3),
    config=None,
    rate_limiter=None,
    fallback_config=None,
    retry_policy=None,
    deadline_timestamp=None,
    deadline_buffer_seconds=0.25,
    **kwargs,
):
    """Call ``call_llm`` with status-aware retry and optional provider fallback.

    Parameters
    ----------
    prompt : str
        The user message content.
    retries : int
        Maximum number of retries (additional attempts after the first
        call).  The total number of attempts is ``retries + 1``.
    delays : tuple of float
        Legacy delay list for compatibility path. Only used when backoff
        is disabled or the result status is not in the retryable set.
    config : dict or None
        Full fusion config dict.
    rate_limiter : RateLimiter or None
        Injected rate limiter for tests.
    fallback_config : dict or None
        Pre-resolved fallback config. When None, resolved from *config*.
    retry_policy : dict or None
        Pre-resolved retry policy. When None, resolved from *config*.
    deadline_timestamp : float or None
        Absolute ``time.monotonic()`` deadline.  When set, the function
        will not start a new attempt if the deadline has passed, and
        will not sleep for a retry delay that would exceed the deadline.
        ``None`` means no deadline (backward-compatible default).
    deadline_buffer_seconds : float
        Safety buffer (default 0.25 s) added when checking whether a
        retry delay fits within the remaining deadline time.  Only
        meaningful when *deadline_timestamp* is set.
    **kwargs
        Forwarded to ``call_llm``.

    Returns
    -------
    dict with the same shape as :func:`call_llm`, plus optional
    observability keys ``attempts``, ``retry_stopped_reason``,
    ``from_fallback``, ``fallback_provider``, ``fallback_error``,
    ``retry_deadline_remaining_seconds``, and ``retry_next_delay_seconds``.
    Never raises.
    """
    # Resolve config-based policies
    resolved_retry_policy = _resolve_retry_policy(config, retry_policy)
    resolved_fallback_config = _resolve_fallback_config(config)
    if fallback_config is not None:
        resolved_fallback_config = fallback_config

    effective_retries = resolved_retry_policy.get("max_retries", retries)
    effective_delays = resolved_retry_policy.get(
        "delays", delays
    ) or delays

    last_result = None

    for attempt in range(effective_retries + 1):
        # --- Deadline check before attempt ---
        if deadline_timestamp is not None:
            remaining = deadline_timestamp - time.monotonic()
            if remaining <= 0.0:
                # Deadline already passed — do not start this attempt
                if last_result is not None:
                    last_result["retry_stopped_reason"] = "deadline_exceeded"
                    last_result["retry_deadline_remaining_seconds"] = 0.0
                else:
                    # No prior result — return an explicit failure
                    result = {
                        "success": False,
                        "error": "Retry deadline exceeded before attempt",
                        "http_status": None,
                        "elapsed": 0.0,
                        "attempts": attempt + 1,
                        "attempt_count": attempt + 1,
                        "retry_stopped_reason": "deadline_exceeded",
                        "retry_deadline_remaining_seconds": 0.0,
                    }
                    _attach_call_observability(result)
                    return result
                break

        result = call_llm(
            prompt,
            config=config,
            rate_limiter=rate_limiter,
            **kwargs,
        )
        # Ensure observability fields are set with retry-policy awareness
        _attach_call_observability(
            result, prompt=prompt,
            retry_policy=resolved_retry_policy,
        )
        result["attempts"] = attempt + 1
        result["attempt_count"] = attempt + 1
        last_result = result

        if result.get("success"):
            result["retry_stopped_reason"] = "success"
            return result

        if attempt < effective_retries:
            if not _is_retryable_result(
                result,
                retryable_statuses=resolved_retry_policy.get(
                    "retryable_statuses"
                ),
                non_retryable_statuses=resolved_retry_policy.get(
                    "non_retryable_statuses"
                ),
            ):
                last_result["retry_stopped_reason"] = "non_retryable_status"
                break
            delay = _compute_retry_delay(
                attempt,
                effective_delays,
                resolved_retry_policy,
                result,
            )
            # --- Deadline check before sleep ---
            if deadline_timestamp is not None:
                allows, remaining = _deadline_allows_retry(
                    deadline_timestamp, delay, deadline_buffer_seconds,
                )
                if not allows:
                    last_result["retry_stopped_reason"] = "deadline_insufficient_budget"
                    last_result["retry_deadline_remaining_seconds"] = remaining
                    last_result["retry_next_delay_seconds"] = delay
                    break
            time.sleep(delay)
        else:
            last_result["retry_stopped_reason"] = "attempts_exhausted"
    else:
        # The for-else runs only if the loop completed without break
        if last_result and not last_result.get("retry_stopped_reason"):
            last_result["retry_stopped_reason"] = "attempts_exhausted"

    # -- Provider fallback (after primary retries exhausted) --
    if resolved_fallback_config.get("enabled"):
        last_status = (last_result or {}).get("http_status")
        # Skip fallback for auth failures (same key unlikely to help)
        if last_status in (401, 403):
            last_result["retry_stopped_reason"] = "non_retryable_status"
            if last_result:
                last_result["fallback_skipped"] = True
        else:
            fb_result = _attempt_provider_fallback(
                prompt,
                resolved_fallback_config,
                **kwargs,
            )
            if fb_result.get("success"):
                _attach_call_observability(
                    fb_result, prompt=prompt,
                )
                return fb_result
            # Attach fallback error to primary result
            if last_result is not None:
                last_result["fallback_error"] = fb_result.get("error")
                last_result["retry_stopped_reason"] = "fallback_failed"

    if last_result is not None:
        _attach_call_observability(
            last_result, prompt=prompt,
            retry_policy=resolved_retry_policy,
        )
        return last_result

    return {
        "success": False,
        "error": "No result from call_llm_with_retry",
        "http_status": None,
        "elapsed": 0.0,
        "attempt_count": 0,
        "retryable": False,
        "error_category": "unknown_error",
        "final_http_status": None,
        "prompt_chars": 0,
        "system_prompt_chars": 0,
        "input_chars": 0,
        "output_chars": 0,
        "reasoning_output_chars": 0,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
