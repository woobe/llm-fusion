"""Fallback provider and rate limiting for llm-fusion.

Supports OpenRouter as a fallback provider when the primary endpoint fails,
and provides simple rate limiting via token bucket.

Never raises exceptions.
"""

import sys
import time
import json
import urllib.error
import urllib.request


# ---------------------------------------------------------------------------
# Rate Limiter (Token Bucket)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple token bucket rate limiter.

    Parameters
    ----------
    rate : float
        Tokens per second (replenishment rate).
    burst : int
        Maximum burst size (bucket capacity).

    Example
    -------
    >>> limiter = RateLimiter(rate=10, burst=20)
    >>> limiter.acquire()  # Blocks until a token is available
    True
    """

    def __init__(self, rate=10.0, burst=20):
        self.rate = float(rate)
        self.burst = float(burst)
        self.tokens = float(burst)
        self.last_refill = time.monotonic()

    def acquire(self, tokens=1, block=True):
        """Acquire *tokens* from the bucket.

        If *block* is True (default), waits until tokens are available.
        If *block* is False, returns True if tokens were available.

        Returns True if tokens acquired, False immediately if not blocking.
        Never raises.
        """
        if tokens <= 0:
            return True

        while True:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            if not block:
                return False
            # Wait for more tokens
            wait_time = (tokens - self.tokens) / self.rate if self.rate > 0 else 0.1
            time.sleep(min(wait_time, 0.1))

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_refill = now


# Global rate limiter instance (10 req/s burst 20)
_rate_limiter = RateLimiter(rate=10.0, burst=20)


def rate_limited_request(req, timeout=60, rate_limiter=None):
    """Make a URL request with rate limiting.

    Parameters
    ----------
    req : urllib.request.Request
        Prepared request object.
    timeout : int
        Request timeout.
    rate_limiter : RateLimiter or None
        Rate limiter instance. Uses global default if None.

    Returns
    -------
    tuple of (http_status, raw_body, error)
    Never raises.
    """
    if rate_limiter is None:
        rate_limiter = _rate_limiter

    rate_limiter.acquire(1)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_body = resp.read().decode("utf-8")
            return resp.status, raw_body, None
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
            return exc.code, None, f"HTTP {exc.code}: {detail[:500]}"
        except Exception:
            return exc.code, None, f"HTTP {exc.code}: {exc.reason}"
    except urllib.error.URLError as exc:
        return None, None, f"URLError: {exc.reason}"
    except TimeoutError:
        return None, None, f"Timeout after {timeout}s"
    except Exception as exc:
        return None, None, f"Unexpected error: {exc}"


# ---------------------------------------------------------------------------
# Fallback Provider
# ---------------------------------------------------------------------------

FALLBACK_CONFIGS = {
    "openrouter": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "requires_key": True,
        "headers": {
            "HTTP-Referer": "https://github.com/nousresearch/hermes-agent",
            "X-Title": "llm-fusion",
        },
    },
}


def call_with_fallback(prompt, primary_func, fallback_config=None, **kwargs):
    """Call an LLM with automatic fallback to alternate providers.

    Parameters
    ----------
    prompt : str
        The user prompt.
    primary_func : callable
        Primary API call function (e.g. call_llm or call_llm_with_retry).
        Must accept params: prompt, **kwargs and return a result dict.
    fallback_config : dict or None
        Fallback provider config. If None, no fallback is attempted.
    **kwargs
        Additional keyword arguments passed to *primary_func*.

    Returns
    -------
    dict
        Result dict from primary or fallback call (whichever succeeds first).
    Never raises.
    """
    # Try primary
    result = primary_func(prompt, **kwargs)
    if result.get("success"):
        return result

    # Try fallback if configured
    if not fallback_config:
        return result

    try:
        fallback_endpoint = fallback_config.get("endpoint")
        fallback_provider = fallback_config.get("provider", "openrouter")

        provider_cfg = FALLBACK_CONFIGS.get(fallback_provider)
        if not provider_cfg:
            return result

        # Build fallback request
        api_key = None
        from scripts.api_client import read_api_key
        api_key = read_api_key()

        if not api_key:
            result["error"] = (result.get("error") or "") + "; No API key for fallback"
            return result

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Hermes-Agent/1.0",
        }
        extra = provider_cfg.get("headers", {})
        headers.update(extra)

        payload = {
            "model": kwargs.get("model", "deepseek-v4-flash"),
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": kwargs.get("temperature", 0.75),
            "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", kwargs.get("max_completion_tokens", 2000)),
        }

        if kwargs.get("reasoning_mode"):
            payload["reasoning_mode"] = kwargs["reasoning_mode"]

        data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            fallback_endpoint,
            data=data_bytes,
            headers=headers,
            method="POST",
        )

        status, raw_body, error = rate_limited_request(
            req, timeout=kwargs.get("timeout", 60)
        )

        if error or status is None:
            result["fallback_error"] = error
            return result

        parsed = json.loads(raw_body)
        fallback_result = {
            "success": True,
            "data": parsed,
            "content": None,
            "reasoning_content": None,
            "usage": parsed.get("usage"),
            "error": None,
            "http_status": status,
            "elapsed": 0.0,
        }

        try:
            msg = parsed["choices"][0]["message"]
            fallback_result["content"] = msg.get("content")
            fallback_result["reasoning_content"] = msg.get("reasoning_content")
        except (KeyError, IndexError, TypeError):
            pass

        fallback_result["from_fallback"] = True
        return fallback_result

    except json.JSONDecodeError as exc:
        result["fallback_error"] = f"Fallback JSON decode error: {exc}"
    except Exception as exc:
        result["fallback_error"] = f"Fallback exception: {exc}"

    return result
