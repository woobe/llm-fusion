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

# ---------------------------------------------------------------------------
# Panel tier system — hardcoded fallback when config has no tiers key
# ---------------------------------------------------------------------------

TIER_MAP = {
    "min": {"deepseek-v4-flash": 2, "mimo-v2.5": 1},
    "low": {"deepseek-v4-flash": 2, "mimo-v2.5": 2},
    "medium": {"deepseek-v4-flash": 1, "mimo-v2.5": 1, "deepseek-v4-pro": 1},
    "high": {"deepseek-v4-pro": 1, "minimax-m3": 1, "qwen3.7-plus": 1},
}

MINIMAX_DEFAULTS = {
    "name": "minimax-m3",
    "count": 0,
    "temp": 0.85,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 2048,
    "thinking": {"type": "adaptive"},
}

QWEN_DEFAULTS = {
    "name": "qwen3.7-plus",
    "count": 0,
    "temp": 0.8,
    "top_p": 0.92,
    "top_k": 20,
    "reasoning_effort": "high",
    "max_tokens": 2048,
}

DEEPSEEK_V4_PRO_DEFAULTS = {
    "name": "deepseek-v4-pro",
    "count": 0,
    "temp": 0.9,
    "top_p": 0.95,
    "reasoning_mode": "high",
    "max_completion_tokens": 2048,
}

TIER_MODEL_DEFAULTS = {
    "minimax-m3": MINIMAX_DEFAULTS,
    "qwen3.7-plus": QWEN_DEFAULTS,
    "deepseek-v4-pro": DEEPSEEK_V4_PRO_DEFAULTS,
}

TIER_CONTROLLED_MODELS = frozenset(
    name for tier_counts in TIER_MAP.values() for name in tier_counts
) | frozenset(TIER_MODEL_DEFAULTS)


def _default_model_entry(name, count):
    defaults = TIER_MODEL_DEFAULTS.get(name)
    if defaults is not None:
        entry = dict(defaults)
        entry["count"] = count
        return entry
    return {"name": name, "count": count}

VALID_TIERS = frozenset(TIER_MAP.keys())


def normalize_tier(tier):
    """Normalize a tier value; defaults to 'low' on None or invalid input.

    Never raises — falls back to ``low`` for any unrecognised value.
    """
    if tier is not None and isinstance(tier, str) and tier in VALID_TIERS:
        return tier
    return "low"


def _apply_tier_counts(models, tier):
    """Apply *tier* model counts to an existing models list.

    Known tier-controlled models absent from the selected tier are disabled
    with count=0. Unknown custom models keep their original count.
    Never raises.
    """
    tier_map = TIER_MAP.get(tier, TIER_MAP["low"])
    updated = []
    seen = set()

    for entry in models:
        name = entry.get("name", "")
        if name in tier_map:
            entry["count"] = tier_map[name]
            seen.add(name)
        elif name in TIER_CONTROLLED_MODELS:
            entry["count"] = 0
        updated.append(entry)

    for name, count in tier_map.items():
        if name not in seen:
            updated.append(_default_model_entry(name, count))

    return updated

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
    """Find the bundled example config in this scripts/assets layout."""
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        # Repository/skill layout: scripts/ next to assets/.
        os.path.join(scripts_dir, "..", "assets", "fusion_config.yaml.example"),
        # Direct relative from cwd when launched from the repo root.
        os.path.join(os.getcwd(), "assets", "fusion_config.yaml.example"),
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
        print("[scripts.config] ERROR: PyYAML is not installed. Install with: pip install pyyaml",
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
                print(f"[scripts.config] Error loading {cand}: {exc}", file=sys.stderr)
                return {}
    if not candidates:
        print("[scripts.config] No config file found (auto-discover returned nothing).", file=sys.stderr)
    return {}


# Map of scenario panel keys → full model names
_MODEL_NAME_MAP = {
    "deepseek": "deepseek-v4-flash",
    "mimo": "mimo-v2.5",
    "minimax": "minimax-m3",
    "qwen": "qwen3.7-plus",
    "deepseek-v4-pro": "deepseek-v4-pro",
}


def resolve_tier_models(panel_cfg, tier="low"):
    """Resolve a tier-based model list from panel config.

    Supports two formats:

    1. **New format** — *panel_cfg* has ``tiers`` and ``model_defaults``
       keys.  Builds model entries by combining the tier definition (model
       name + count) with base parameters from ``model_defaults``.
    2. **Legacy format** — *panel_cfg* has a plain ``models`` list.  Uses
       the canonical :data:`TIER_MAP` to override counts.

    If neither key exists, falls back to :data:`TIER_MAP` for the
    requested tier.

    Parameters
    ----------
    panel_cfg : dict
        The ``default.panel`` section of the fusion config (or a sub-dict
        containing keys ``tiers``, ``model_defaults``, and/or ``models``).
    tier : str
        One of ``"min"``, ``"low"``, ``"medium"``, or ``"high"``.  Falls back to
        ``"low"`` when the requested tier is missing from the config.

    Returns
    -------
    list[dict]
        Model entries with full parameters merged. Never returns None.
    """
    base_tier = normalize_tier(tier)
    tier_defs = None

    # 1. Try new ``tiers`` key
    if "tiers" in panel_cfg:
        tiers = panel_cfg["tiers"]
        tier_defs = tiers.get(base_tier, tiers.get("low", []))

    if tier_defs is not None:
        # New format: merge with model_defaults
        model_defaults = panel_cfg.get("model_defaults", {})
        models = []
        for entry in tier_defs:
            name = entry["name"]
            count = entry.get("count", 1)
            defaults = model_defaults.get(name, {})
            model_entry = {"name": name, "count": count}
            model_entry.update(defaults)
            models.append(model_entry)
        return models

    # 2. Legacy ``models`` list – apply TIER_MAP counts
    legacy_models = panel_cfg.get("models")
    if legacy_models is not None:
        return _apply_tier_counts(list(legacy_models), base_tier)

    # 3. Nothing at all – return TIER_MAP defaults
    tier_map = TIER_MAP.get(base_tier, TIER_MAP["low"])
    return [_default_model_entry(name, count) for name, count in tier_map.items()]


def _apply_scenario_overrides(models, scenario_panel):
    """Apply scenario-specific parameter overrides on a resolved model list.

    Iterates over the resolved *models* list and overrides parameters from
    *scenario_panel* entries (keyed by short names such as ``deepseek`` or
    ``mimo``). Preserves ``count`` and ``name``. Mutates entries in place.

    Parameters
    ----------
    models : list[dict]
        Resolved model entries (from ``resolve_tier_models`` or legacy).
    scenario_panel : dict
        The ``panel`` sub-dict from a scenario config (may be empty).

    Returns
    -------
    list[dict]
        The same list, updated in place.
    """
    if not scenario_panel or not models:
        return models

    for alias, full_name in _MODEL_NAME_MAP.items():
        if alias not in scenario_panel:
            continue
        overrides = scenario_panel[alias]
        for m in models:
            if m.get("name") == full_name:
                # Apply overrides but keep structural fields intact
                for k, v in overrides.items():
                    if k not in ("name", "count"):
                        m[k] = v
                break
    return models


def get_scenario_config(config, scenario_id, tier=None):
    """Get merged scenario config (scenario-specific over default), with tier.

    Parameters
    ----------
    config : dict
        Full fusion config.
    scenario_id : str
        One of the known scenario identifiers.
    tier : str or None
        Panel tier (``min``, ``low``, ``medium``, or ``None`` for default).

    Returns a flat dict with keys: panel (dict), judge (dict), cleaning (dict),
    conciseness_suffix (str).
    Never raises — falls back to 'general' or a sensible empty dict.
    """
    if not config or not isinstance(config, dict):
        cfg = _default_scenario_config()
        cfg["panel"]["models"] = _apply_tier_counts(cfg["panel"]["models"], tier)
        return cfg

    default_panel = config.get("default", {}).get("panel", {})
    default_judge = config.get("default", {}).get("judge", {})

    scenarios = config.get("scenarios", {})
    scenario = scenarios.get(scenario_id, scenarios.get("general", {}))

    if not scenario:
        cfg = _default_scenario_config()
        cfg["panel"]["models"] = _apply_tier_counts(cfg["panel"]["models"], tier)
        return cfg

    # Build panel config with tier resolution
    panel_config = dict(default_panel)
    panel_config["models"] = resolve_tier_models(panel_config, tier)

    # Apply scenario overrides on top (preserves count and name)
    scenario_panel = scenario.get("panel", {})
    _apply_scenario_overrides(panel_config.get("models", []), scenario_panel)

    # Build judge config
    judge_config = dict(default_judge)
    scenario_judge = scenario.get("judge", {})
    if scenario_judge:
        judge_config.update(scenario_judge)

    # Cleaning config
    cleaning_config = get_cleaning_profile(config, scenario_id)

    # Conciseness suffix
    from scripts.classifier import CONCISENESS_SUFFIXES  # noqa: F811
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
