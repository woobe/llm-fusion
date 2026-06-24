"""Fallback provider and rate limiting for llm-fusion.

Supports OpenRouter as a fallback provider when the primary endpoint fails,
and provides simple rate limiting via token bucket.

Never raises exceptions.
"""

import sys
import time
import json
import threading
import urllib.error
import urllib.request


# ---------------------------------------------------------------------------
# Rate Limiter (Token Bucket) — thread-safe
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple token bucket rate limiter (thread-safe).

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
        self._lock = threading.Lock()

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
            with self._lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                if not block:
                    return False
                wait_time = (tokens - self.tokens) / self.rate if self.rate > 0 else 0.1
            # Sleep outside the lock to avoid blocking other threads
            time.sleep(min(wait_time, 0.1))

    def _refill(self):
        """Refill tokens based on elapsed time.  Caller must hold _lock."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_refill = now


# Global rate limiter instance (10 req/s burst 20)
_rate_limiter = RateLimiter(rate=10.0, burst=20)

# Global rate limiter settings tracker for get_rate_limiter()
_global_rate_limiter = None
_global_rate_limiter_settings = (None, None)


def get_rate_limiter(rate=10.0, burst=20):
    """Return a process-global rate limiter, recreated only when settings change.

    Parameters
    ----------
    rate : float
        Tokens per second (replenishment rate).
    burst : int
        Maximum burst size (bucket capacity).

    Returns
    -------
    RateLimiter
        A global singleton that gets replaced only when *rate* or *burst*
        differ from the previously requested values.
    """
    global _global_rate_limiter, _global_rate_limiter_settings
    settings = (float(rate), float(burst))
    if _global_rate_limiter_settings != settings:
        _global_rate_limiter = RateLimiter(rate=rate, burst=burst)
        _global_rate_limiter_settings = settings
    return _global_rate_limiter


def rate_limited_request(req, timeout=60, rate_limiter=None, enabled=True):
    """Make a URL request, optionally rate-limited.

    Parameters
    ----------
    req : urllib.request.Request
        Prepared request object.
    timeout : int
        Request timeout.
    rate_limiter : RateLimiter or None
        Rate limiter instance. Uses global default if None.
    enabled : bool
        When True (default), acquire a token from the rate limiter before
        making the request. When False, skip rate limiting entirely.

    Returns
    -------
    tuple of (http_status, raw_body, error)
    Never raises.
    """
    if enabled:
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

