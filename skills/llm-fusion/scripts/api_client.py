"""API client for llm-fusion.

Handles API key loading, making LLM API calls via urllib (stdlib only),
and consuming the OpenCode Go / OpenRouter chat completions endpoint.
Never raises exceptions.
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
):
    """Call an LLM chat completions endpoint and return the parsed response.

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
        Additional JSON payload keys.

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
        result["error"] = "No API key found. Set OPENCODE_GO_API_KEY env var or ensure ~/.hermes/.env exists."
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

    # Mimo model needs thinking disabled
    if "mimo" in model.lower():
        payload["thinking"] = {"type": "disabled"}

    # Extra params (e.g. thinking config for specific models)
    if extra_params:
        payload.update(extra_params)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }

    data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data_bytes, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.monotonic() - start
            raw_body = resp.read().decode("utf-8")
            result["http_status"] = resp.status
            parsed = json.loads(raw_body)
            result["data"] = parsed
            result["elapsed"] = elapsed

            # Extract content
            try:
                msg = parsed["choices"][0]["message"]
                result["content"] = msg.get("content")
                result["reasoning_content"] = msg.get("reasoning_content")
            except (KeyError, IndexError, TypeError):
                pass

            # Extract usage
            result["usage"] = parsed.get("usage")

            result["success"] = True

    except urllib.error.HTTPError as exc:
        elapsed = time.monotonic() - start
        result["http_status"] = exc.code
        result["elapsed"] = elapsed
        try:
            detail = exc.read().decode("utf-8", errors="replace")
            result["error"] = f"HTTP {exc.code}: {detail[:500]}"
        except Exception:
            result["error"] = f"HTTP {exc.code}: {exc.reason}"

    except urllib.error.URLError as exc:
        elapsed = time.monotonic() - start
        result["elapsed"] = elapsed
        result["error"] = f"URLError: {exc.reason}"

    except TimeoutError:
        elapsed = time.monotonic() - start
        result["elapsed"] = elapsed
        result["error"] = f"Timeout after {timeout}s"

    except json.JSONDecodeError as exc:
        elapsed = time.monotonic() - start
        result["elapsed"] = elapsed
        result["error"] = f"JSON decode error: {exc}"

    except Exception as exc:
        elapsed = time.monotonic() - start
        result["elapsed"] = elapsed
        result["error"] = f"Unexpected error: {exc}"

    return result


def call_llm_with_retry(prompt, retries=2, delays=(1, 3), **kwargs):
    """Call call_llm with retry logic.

    Retries on failure (non-success) up to *retries* times with *delays*
    between attempts. Returns the first successful result.
    If all attempts fail, returns the last result.
    Never raises.
    """
    last_result = None
    for attempt in range(retries + 1):
        result = call_llm(prompt, **kwargs)
        last_result = result
        if result["success"]:
            return result
        if attempt < retries:
            delay = delays[min(attempt, len(delays) - 1)]
            time.sleep(delay)
    return last_result
