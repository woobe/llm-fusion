"""Config loader for llm-fusion.

Loads YAML configuration and provides access to scenario-specific,
cleaning, and API configuration values. Never raises exceptions.
"""

import os
import sys

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


def _discover_config_path():
    """Walk the portable config discovery order and return first path found.

    Returns the path as str, or None if nothing exists.
    Never raises.
    """
    # 1. LLM_FUSION_CONFIG env var
    env_path = os.environ.get("LLM_FUSION_CONFIG")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. cwd/fusion_config.yaml
    cwd_path = os.path.join(os.getcwd(), "fusion_config.yaml")
    if os.path.isfile(cwd_path):
        return cwd_path

    # 3. cwd/.llm-fusion/fusion_config.yaml
    hidden_path = os.path.join(os.getcwd(), ".llm-fusion", "fusion_config.yaml")
    if os.path.isfile(hidden_path):
        return hidden_path

    # 4. XDG_CONFIG_HOME/llm-fusion/fusion_config.yaml
    xdg_home = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    xdg_path = os.path.join(xdg_home, "llm-fusion", "fusion_config.yaml")
    if os.path.isfile(xdg_path):
        return xdg_path

    # 5. ~/.llm-fusion/fusion_config.yaml
    home_hidden = os.path.join(os.path.expanduser("~"), ".llm-fusion", "fusion_config.yaml")
    if os.path.isfile(home_hidden):
        return home_hidden

    # 6. Bundled example (last resort — for defaults only)
    bundled = _find_bundled_example()
    if bundled and os.path.isfile(bundled):
        return bundled

    return None


def _find_bundled_example():
    """Find the bundled example config relative to this package or skill dir."""
    candidates = [
        # When running from the skill wrapper
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", ".agents", "skills", "llm-fusion",
                     "assets", "fusion_config.yaml.example"),
        # When running from the package (editable install in repo root)
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "..", "..", ".agents", "skills", "llm-fusion",
                     "assets", "fusion_config.yaml.example"),
        # Direct relative from cwd
        os.path.join(os.getcwd(), ".agents", "skills", "llm-fusion",
                     "assets", "fusion_config.yaml.example"),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isfile(c):
            return c
    return None


def load_config(path=None):
    """Load fusion_config.yaml from *path*, or auto-discover.

    Returns a dict with the full fusion config, or an empty dict on failure.
    Never raises.
    """
    if path is None:
        # Auto-discover
        discovered = _discover_config_path()
        if discovered:
            candidates = [discovered]
        else:
            candidates = []
    else:
        candidates = [path]

    if yaml is None:
        print("[llm_fusion.config] ERROR: PyYAML is not installed. Install with: pip install pyyaml",
              file=sys.stderr)
        return {}

    for cand in candidates:
        cand = os.path.abspath(cand)
        if os.path.isfile(cand):
            try:
                with open(cand, "r") as fh:
                    data = yaml.safe_load(fh)
                if isinstance(data, dict) and "fusion" in data:
                    return data["fusion"]
                # If 'fusion' key missing but data is a dict, return data
                if isinstance(data, dict):
                    return data
            except (yaml.YAMLError, OSError, PermissionError) as exc:
                print(f"[llm_fusion.config] Error loading {cand}: {exc}", file=sys.stderr)
                return {}
    if not candidates:
        print("[llm_fusion.config] No config file found (auto-discover returned nothing).", file=sys.stderr)
    return {}


def get_scenario_config(config, scenario_id):
    """Get merged scenario config (scenario-specific over default).

    Returns a flat dict with keys: panel (dict), judge (dict), cleaning (dict),
    conciseness_suffix (str).
    Never raises — falls back to 'general' or a sensible empty dict.
    """
    if not config or not isinstance(config, dict):
        return _default_scenario_config()

    default_panel = config.get("default", {}).get("panel", {})
    default_judge = config.get("default", {}).get("judge", {})

    scenarios = config.get("scenarios", {})
    scenario = scenarios.get(scenario_id, scenarios.get("general", {}))

    if not scenario:
        return _default_scenario_config()

    # Build panel config
    panel_config = dict(default_panel)
    scenario_panel = scenario.get("panel", {})
    if scenario_panel:
        # Merge deepseek settings
        if "deepseek" in scenario_panel:
            if "models" not in panel_config:
                panel_config["models"] = []
            ds = scenario_panel["deepseek"]
            # Override the deepseek model entry
            ds_model = {
                "name": "deepseek-v4-flash",
                "count": 3,
                "temp": ds.get("temp", 0.75),
                "top_p": default_judge.get("top_p", 0.9),
                "max_completion_tokens": ds.get("max_completion_tokens", 2000),
            }
            # Replace or append
            found = False
            for m in panel_config.get("models", []):
                if m.get("name") == "deepseek-v4-flash":
                    m.update(ds_model)
                    found = True
                    break
            if not found:
                panel_config.setdefault("models", []).append(ds_model)

        # Merge mimo settings
        if "mimo" in scenario_panel:
            mimo = scenario_panel["mimo"]
            mimo_model = {
                "name": "mimo-v2.5",
                "count": 3,
                "temps": mimo.get("temps", [0.6, 0.7, 0.8]),
                "top_p": 0.95,
                "max_tokens": mimo.get("max_tokens", 600),
                "thinking": {"type": "disabled"},
            }
            found = False
            for m in panel_config.get("models", []):
                if m.get("name") == "mimo-v2.5":
                    m.update(mimo_model)
                    found = True
                    break
            if not found:
                panel_config.setdefault("models", []).append(mimo_model)

    # Build judge config
    judge_config = dict(default_judge)
    scenario_judge = scenario.get("judge", {})
    if scenario_judge:
        judge_config.update(scenario_judge)

    # Cleaning config
    cleaning_config = get_cleaning_profile(config, scenario_id)

    # Conciseness suffix
    from llm_fusion.classifier import CONCISENESS_SUFFIXES  # noqa: F811
    conciseness_suffix = CONCISENESS_SUFFIXES.get(scenario_id, CONCISENESS_SUFFIXES["general"])

    return {
        "panel": panel_config,
        "judge": judge_config,
        "cleaning": cleaning_config,
        "conciseness_suffix": conciseness_suffix,
    }


def get_cleaning_profile(config, scenario_id):
    """Get the cleaning profile for *scenario_id* from config.

    Returns a dict with keys: strip_fences, strip_preamble, min_words,
    dedup_threshold, and optional preamble_patterns/trailing_patterns.
    Falls back to 'general' if scenario is not found.
    Never raises.
    """
    if not config or not isinstance(config, dict):
        return _default_cleaning_profile()

    profiles = config.get("cleaning", {}).get("profiles", {})
    profile = profiles.get(scenario_id, profiles.get("general", {}))

    if not profile:
        return _default_cleaning_profile()

    return {
        "strip_fences": profile.get("strip_fences", True),
        "strip_preamble": profile.get("strip_preamble", True),
        "min_words": profile.get("min_words", 10),
        "dedup_threshold": profile.get("dedup_threshold", 0.85),
    }


def _default_scenario_config():
    """Return a safe default scenario config when everything fails."""
    return {
        "panel": {
            "models": [
                {"name": "deepseek-v4-flash", "count": 3, "temp": 0.75, "top_p": 0.9,
                 "max_completion_tokens": 800},
                {"name": "mimo-v2.5", "count": 3, "temps": [0.6, 0.7, 0.8], "top_p": 0.95,
                 "max_tokens": 600, "thinking": {"type": "disabled"}},
            ],
        },
        "judge": {
            "model": "deepseek-v4-flash",
            "temp": 0.0,
            "top_p": 1.0,
            "stages": "single",
            "reasoning_mode": "high",
            "max_completion_tokens": 8000,
        },
        "cleaning": _default_cleaning_profile(),
        "conciseness_suffix": (
            "KEEP IT CONCISE — 2-4 informative sentences. No preamble, no meta-commentary, "
            "no opening phrases like 'Here's my response' or 'Sure!'. Just answer directly."
        ),
    }


def _default_cleaning_profile():
    """Safe default cleaning profile."""
    return {
        "strip_fences": True,
        "strip_preamble": True,
        "min_words": 10,
        "dedup_threshold": 0.85,
    }
