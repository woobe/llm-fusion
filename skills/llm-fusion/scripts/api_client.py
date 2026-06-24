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
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
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
        return result

    elapsed = time.monotonic() - start
    result["http_status"] = http_status
    result["elapsed"] = elapsed

    if error or http_status is None or http_status >= 400:
        result["error"] = error or f"HTTP {http_status}"
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
    **kwargs
        Forwarded to ``call_llm``.

    Returns
    -------
    dict with the same shape as :func:`call_llm`, plus optional
    observability keys ``attempts``, ``retry_stopped_reason``,
    ``from_fallback``, ``fallback_provider``, and ``fallback_error``.
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
        result = call_llm(
            prompt,
            config=config,
            rate_limiter=rate_limiter,
            **kwargs,
        )
        result["attempts"] = attempt + 1
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
                return fb_result
            # Attach fallback error to primary result
            if last_result is not None:
                last_result["fallback_error"] = fb_result.get("error")
                last_result["retry_stopped_reason"] = "fallback_failed"

    return last_result if last_result is not None else {
        "success": False,
        "error": "No result from call_llm_with_retry",
        "http_status": None,
        "elapsed": 0.0,
    }
