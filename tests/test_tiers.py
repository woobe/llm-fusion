"""Tests for the panel tier system.

Tests cover:
  - normalize_tier() helper
  - TIER_MAP counts
  - get_scenario_config(..., tier=...) producing correct counts
  - Judge config unchanged across tiers
  - Panel dispatch with tier (count<=0 skip, top_k passthrough)
  - CLI --tier argument in dry-run
"""

import unittest
from unittest import mock


class TestTierConfig(unittest.TestCase):
    """Test the tier map and normalize_tier helper."""

    def setUp(self):
        self.minimal_config = {
            "version": "2.0.0",
            "default": {
                "panel": {
                    "models": [
                        {"name": "deepseek-v4-flash", "count": 3, "temp": 0.75, "top_p": 0.9},
                        {"name": "mimo-v2.5", "count": 3, "temps": [0.6, 0.7, 0.8], "top_p": 0.95},
                    ],
                },
                "judge": {"model": "deepseek-v4-flash", "temp": 0.0, "top_p": 1.0},
            },
            "scenarios": {
                "coding": {
                    "panel": {"deepseek": {"temp": 0.5, "max_completion_tokens": 2000}},
                    "judge": {"stages": "single", "reasoning_mode": "max", "max_completion_tokens": 16000},
                },
                "general": {
                    "panel": {"deepseek": {"temp": 0.75, "max_completion_tokens": 800}},
                    "judge": {"stages": "single", "reasoning_mode": "high", "max_completion_tokens": 8000},
                },
            },
            "pipeline": {
                "max_panel_workers": 6,
                "min_survivors": 2,
                "graceful_degradation": True,
            },
        }

    def test_normalize_tier_default(self):
        """normalize_tier(None) returns 'low2'."""
        from scripts.config import normalize_tier
        self.assertEqual(normalize_tier(None), "low2")

    def test_normalize_tier_low1(self):
        """normalize_tier('low1') returns 'low1'."""
        from scripts.config import normalize_tier
        self.assertEqual(normalize_tier("low1"), "low1")

    def test_normalize_tier_low2(self):
        """normalize_tier('low2') returns 'low2'."""
        from scripts.config import normalize_tier
        self.assertEqual(normalize_tier("low2"), "low2")

    def test_normalize_tier_medium(self):
        """normalize_tier('medium') returns 'medium'."""
        from scripts.config import normalize_tier
        self.assertEqual(normalize_tier("medium"), "medium")

    def test_normalize_tier_invalid_falls_back(self):
        """normalize_tier('invalid') falls back to 'low2'."""
        from scripts.config import normalize_tier
        self.assertEqual(normalize_tier("invalid"), "low2")

    def test_normalize_tier_empty_string(self):
        """normalize_tier('') falls back to 'low2'."""
        from scripts.config import normalize_tier
        self.assertEqual(normalize_tier(""), "low2")

    def test_tier_map_has_expected_tiers(self):
        """TIER_MAP has low1, low2, low3, medium, high."""
        from scripts.config import TIER_MAP
        self.assertIn("low1", TIER_MAP)
        self.assertIn("low2", TIER_MAP)
        self.assertIn("low3", TIER_MAP)
        self.assertIn("medium", TIER_MAP)
        self.assertIn("high", TIER_MAP)

    def test_tier_map_low1_counts(self):
        """low1 tier: 1 deepseek + 1 mimo = 2 total calls."""
        from scripts.config import TIER_MAP
        counts = TIER_MAP["low1"]
        self.assertEqual(counts.get("deepseek-v4-flash"), 1)
        self.assertEqual(counts.get("mimo-v2.5"), 1)
        self.assertEqual(sum(counts.values()), 2)

    def test_tier_map_low2_counts(self):
        """low2 tier: 2 deepseek + 2 mimo = 4 total calls."""
        from scripts.config import TIER_MAP
        counts = TIER_MAP["low2"]
        self.assertEqual(counts.get("deepseek-v4-flash"), 2)
        self.assertEqual(counts.get("mimo-v2.5"), 2)
        self.assertEqual(sum(counts.values()), 4)

    def test_tier_map_low3_counts(self):
        """low3 tier: 3 deepseek + 3 mimo = 6 total calls."""
        from scripts.config import TIER_MAP
        counts = TIER_MAP["low3"]
        self.assertEqual(counts.get("deepseek-v4-flash"), 3)
        self.assertEqual(counts.get("mimo-v2.5"), 3)
        self.assertEqual(sum(counts.values()), 6)

    def test_tier_map_medium_counts(self):
        """medium tier: 1 deepseek + 1 mimo + 1 deepseek-v4-pro = 3 total calls."""
        from scripts.config import TIER_MAP
        counts = TIER_MAP["medium"]
        self.assertEqual(counts.get("deepseek-v4-flash"), 1)
        self.assertEqual(counts.get("mimo-v2.5"), 1)
        self.assertEqual(counts.get("deepseek-v4-pro"), 1)
        self.assertNotIn("minimax-m3", counts)
        self.assertNotIn("qwen3.7-plus", counts)
        self.assertEqual(sum(counts.values()), 3)

    def test_tier_map_high_counts(self):
        """high tier: 1 deepseek-v4-pro + 1 minimax-m3 + 1 qwen3.7-plus = 3 total calls."""
        from scripts.config import TIER_MAP
        counts = TIER_MAP["high"]
        self.assertEqual(counts.get("deepseek-v4-pro"), 1)
        self.assertEqual(counts.get("minimax-m3"), 1)
        self.assertEqual(counts.get("qwen3.7-plus"), 1)
        self.assertNotIn("deepseek-v4-flash", counts)
        self.assertNotIn("mimo-v2.5", counts)
        self.assertEqual(sum(counts.values()), 3)


class TestScenarioConfigWithTier(unittest.TestCase):
    """Test get_scenario_config with tier parameter."""

    def setUp(self):
        self.minimal_config = {
            "version": "2.0.0",
            "default": {
                "panel": {
                    "models": [
                        {"name": "deepseek-v4-flash", "count": 3, "temp": 0.75, "top_p": 0.9},
                        {"name": "mimo-v2.5", "count": 3, "temps": [0.6, 0.7, 0.8], "top_p": 0.95},
                    ],
                },
                "judge": {"model": "deepseek-v4-flash", "temp": 0.0, "top_p": 1.0},
            },
            "scenarios": {
                "coding": {
                    "panel": {"deepseek": {"temp": 0.5, "max_completion_tokens": 2000}},
                    "judge": {"stages": "single", "reasoning_mode": "max", "max_completion_tokens": 16000},
                },
                "general": {
                    "panel": {"deepseek": {"temp": 0.75, "max_completion_tokens": 800}},
                    "judge": {"stages": "single", "reasoning_mode": "high", "max_completion_tokens": 8000},
                },
            },
        }

    def _total_panel_count(self, models):
        """Sum count field across all models."""
        return sum(m.get("count", 0) for m in models)

    def test_low1_tier_total_calls(self):
        """low1 tier produces 2 total panel calls."""
        from scripts.config import get_scenario_config
        cfg = get_scenario_config(self.minimal_config, "general", tier="low1")
        models = cfg["panel"]["models"]
        self.assertEqual(self._total_panel_count(models), 2)

    def test_low2_tier_total_calls(self):
        """low2 tier produces 4 total panel calls."""
        from scripts.config import get_scenario_config
        cfg = get_scenario_config(self.minimal_config, "general", tier="low2")
        models = cfg["panel"]["models"]
        self.assertEqual(self._total_panel_count(models), 4)

    def test_medium_tier_total_calls(self):
        """medium tier produces 3 total panel calls (1+1+1)."""
        from scripts.config import get_scenario_config
        cfg = get_scenario_config(self.minimal_config, "general", tier="medium")
        models = cfg["panel"]["models"]
        self.assertEqual(self._total_panel_count(models), 3)

    def test_default_tier_low2(self):
        """No tier argument defaults to low2 (4 calls)."""
        from scripts.config import get_scenario_config
        cfg = get_scenario_config(self.minimal_config, "general")
        models = cfg["panel"]["models"]
        self.assertEqual(self._total_panel_count(models), 4)

    def test_low1_tier_deepseek_count(self):
        """low1 tier has deepseek count=1."""
        from scripts.config import get_scenario_config
        cfg = get_scenario_config(self.minimal_config, "general", tier="low1")
        for m in cfg["panel"]["models"]:
            if m["name"] == "deepseek-v4-flash":
                self.assertEqual(m["count"], 1)
                break
        else:
            self.fail("deepseek-v4-flash not found in models")

    def test_low1_tier_mimo_count(self):
        """low1 tier has mimo count=1."""
        from scripts.config import get_scenario_config
        cfg = get_scenario_config(self.minimal_config, "general", tier="low1")
        for m in cfg["panel"]["models"]:
            if m["name"] == "mimo-v2.5":
                self.assertEqual(m["count"], 1)
                break
        else:
            self.fail("mimo-v2.5 not found in models")

    def test_medium_tier_includes_deepseek_v4_pro_and_mimo(self):
        """medium tier includes deepseek-v4-pro and mimo-v2.5 (count=1 each)."""
        from scripts.config import get_scenario_config
        cfg = get_scenario_config(self.minimal_config, "general", tier="medium")
        models = cfg["panel"]["models"]
        names = [m["name"] for m in models]
        self.assertIn("deepseek-v4-pro", names)
        self.assertIn("mimo-v2.5", names)
        self.assertIn("deepseek-v4-flash", names)
        for m in models:
            if m["name"] == "deepseek-v4-pro":
                self.assertEqual(m.get("count"), 1)
            if m["name"] == "mimo-v2.5":
                self.assertEqual(m.get("count"), 1)

    def test_minimax_not_in_low1_low2_low3_or_medium(self):
        """minimax-m3 should NOT appear in low1, low2, low3, or medium tiers."""
        from scripts.config import get_scenario_config
        for tier in ("low1", "low2", "low3", "medium"):
            cfg = get_scenario_config(self.minimal_config, "general", tier=tier)
            names = [m["name"] for m in cfg["panel"]["models"]]
            self.assertNotIn("minimax-m3", names, f"minimax-m3 should not be in {tier} tier")

    def test_high_tier_includes_minimax_and_qwen(self):
        """high tier includes minimax-m3, qwen3.7-plus, and deepseek-v4-pro (deepseek disabled)."""
        from scripts.config import get_scenario_config
        cfg = get_scenario_config(self.minimal_config, "general", tier="high")
        names = [m["name"] for m in cfg["panel"]["models"]]
        self.assertIn("minimax-m3", names)
        self.assertIn("qwen3.7-plus", names)
        self.assertIn("deepseek-v4-pro", names)
        # deepseek-v4-flash and mimo-v2.5 are in the list but disabled
        for m in cfg["panel"]["models"]:
            if m["name"] == "deepseek-v4-flash":
                self.assertEqual(m.get("count"), 0)
            if m["name"] == "mimo-v2.5":
                self.assertEqual(m.get("count"), 0)

    def test_minimax_defaults_sensible(self):
        """minimax-m3 gets sensible default parameters (in high tier)."""
        from scripts.config import get_scenario_config
        cfg = get_scenario_config(self.minimal_config, "general", tier="high")
        for m in cfg["panel"]["models"]:
            if m["name"] == "minimax-m3":
                self.assertIn("temp", m)
                self.assertIn("top_p", m)
                self.assertIn("top_k", m)
                self.assertIn("max_tokens", m)
                self.assertIn("thinking", m)
                break
        else:
            self.fail("minimax-m3 not found")

    def test_qwen_defaults_sensible(self):
        """qwen3.7-plus gets validated default parameters (in high tier)."""
        from scripts.config import get_scenario_config
        cfg = get_scenario_config(self.minimal_config, "general", tier="high")
        for m in cfg["panel"]["models"]:
            if m["name"] == "qwen3.7-plus":
                self.assertEqual(m["temp"], 0.8)
                self.assertEqual(m["top_p"], 0.92)
                self.assertEqual(m["top_k"], 20)
                self.assertEqual(m["reasoning_effort"], "high")
                self.assertEqual(m["max_tokens"], 4096)
                self.assertNotIn("max_completion_tokens", m)
                break
        else:
            self.fail("qwen3.7-plus not found")

    def test_deepseek_v4_pro_defaults(self):
        """deepseek-v4-pro gets validated default parameters (in medium tier)."""
        from scripts.config import get_scenario_config
        cfg = get_scenario_config(self.minimal_config, "general", tier="medium")
        for m in cfg["panel"]["models"]:
            if m["name"] == "deepseek-v4-pro":
                self.assertEqual(m["temp"], 0.9)
                self.assertEqual(m["top_p"], 0.95)
                self.assertEqual(m["reasoning_mode"], "high")
                self.assertEqual(m["max_completion_tokens"], 4096)
                break
        else:
            self.fail("deepseek-v4-pro not found")

    def test_judge_config_unchanged_by_tier(self):
        """Judge config is identical regardless of tier."""
        from scripts.config import get_scenario_config
        cfg_no_tier = get_scenario_config(self.minimal_config, "coding")
        for tier in ("low1", "low2", "low3", "medium", "high"):
            cfg_tier = get_scenario_config(self.minimal_config, "coding", tier=tier)
            self.assertEqual(cfg_tier["judge"], cfg_no_tier["judge"],
                             f"Judge config changed for tier={tier}")

    def test_judge_config_unchanged_general(self):
        """Judge config is identical for general scenario too."""
        from scripts.config import get_scenario_config
        cfg_no_tier = get_scenario_config(self.minimal_config, "general")
        for tier in ("low1", "low2", "low3", "medium", "high"):
            cfg_tier = get_scenario_config(self.minimal_config, "general", tier=tier)
            self.assertEqual(cfg_tier["judge"], cfg_no_tier["judge"],
                             f"Judge config changed for tier={tier}")


class TestPanelTier(unittest.TestCase):
    """Test panel dispatch respects tier configs without making network calls."""

    def test_medium_tier_builds_expected_three_specs(self):
        """medium tier builds deepseek, mimo, and deepseek-v4-pro specs only."""
        from scripts.panel import dispatch_panel

        config = {
            "default": {
                "panel": {
                    "models": [
                        {"name": "deepseek-v4-flash", "count": 3, "temp": 0.75, "top_p": 0.9,
                         "max_completion_tokens": 800},
                        {"name": "mimo-v2.5", "count": 3, "temps": [0.6, 0.7, 0.8], "top_p": 0.95,
                         "max_tokens": 600, "thinking": {"type": "disabled"}},
                    ],
                },
                "judge": {"model": "deepseek-v4-flash", "temp": 0.0, "top_p": 1.0},
            },
            "scenarios": {
                "general": {
                    "panel": {"deepseek": {"temp": 0.75, "max_completion_tokens": 800}},
                    "judge": {"stages": "single", "reasoning_mode": "high", "max_completion_tokens": 8000},
                },
            },
            "api": {
                "primary": {
                    "endpoint": "http://127.0.0.1:1/nonexistent",
                    "timeout": {
                        "panel_floor": 2,
                        "judge_floor": 2,
                        "panel_throughput": 9999,
                        "judge_throughput": 9999,
                        "overhead_seconds": 0,
                        "max_timeout": 300,
                    },
                    "retry": {"max_retries": 0, "delays_seconds": [0.1]},
                },
            },
            "pipeline": {
                "max_panel_workers": 6,
                "min_survivors": 1,
                "graceful_degradation": True,
            },
        }

        # Mock the API call so we don't make network requests
        with mock.patch("scripts.panel.call_llm_with_retry") as mock_call:
            mock_call.return_value = {"success": True, "content": "test",
                                      "reasoning_content": None, "usage": {},
                                      "elapsed": 0.01}
            result = dispatch_panel("test query", "general", config=config, tier="medium",
                                    max_workers=3)

        labels = [r.get("label", "") for r in result["responses"]]
        self.assertEqual(len(labels), 3)
        self.assertIn("deepseek-v4-flash #1", labels)
        self.assertIn("deepseek-v4-pro #1", labels)
        self.assertIn("mimo-v2.5 #1", labels)
        self.assertFalse(any("minimax-m3" in label for label in labels), labels)
        self.assertFalse(any("qwen3.7-plus" in label for label in labels), labels)

    def test_min_tier_only_low1_specs(self):
        """low1 tier should produce 2 call specs (1 deepseek, 1 mimo)."""
        from scripts.panel import dispatch_panel

        config = {
            "default": {"panel": {"models": []}, "judge": {}},
            "scenarios": {
                "general": {
                    "panel": {
                        "deepseek": {"temp": 0.75, "max_completion_tokens": 800},
                        "mimo": {"temps": [0.7], "max_tokens": 600},
                    },
                    "judge": {"stages": "single"},
                },
            },
            "api": {
                "primary": {
                    "endpoint": "http://127.0.0.1:1/nonexistent",
                    "timeout": {
                        "panel_floor": 2,
                        "judge_floor": 2,
                        "panel_throughput": 9999,
                        "judge_throughput": 9999,
                        "overhead_seconds": 0,
                        "max_timeout": 300,
                    },
                    "retry": {"max_retries": 0, "delays_seconds": [0.1]},
                },
            },
            "pipeline": {"max_panel_workers": 3, "min_survivors": 1, "graceful_degradation": True},
        }

        with mock.patch("scripts.panel.call_llm_with_retry") as mock_call:
            mock_call.return_value = {"success": True, "content": "test",
                                      "reasoning_content": None, "usage": {},
                                      "elapsed": 0.01}
            result = dispatch_panel("test", "general", config=config, tier="low1", max_workers=3)

        self.assertEqual(len(result["responses"]), 2)

    def test_top_k_passed_in_extra_params(self):
        """top_k from model config should be passed via extra_params."""
        from scripts.panel import dispatch_panel

        config = {
            "default": {
                "panel": {
                    "models": [
                        {"name": "minimax-m3", "count": 1, "temp": 0.85, "top_p": 0.9,
                         "top_k": 40, "max_tokens": 2048, "thinking": {"type": "adaptive"}},
                    ],
                },
                "judge": {"model": "deepseek-v4-flash", "temp": 0.0, "top_p": 1.0},
            },
            "scenarios": {
                "general": {"panel": {}, "judge": {"stages": "single"}},
            },
            "api": {
                "primary": {
                    "endpoint": "http://127.0.0.1:1/nonexistent",
                    "timeout": {
                        "panel_floor": 2,
                        "judge_floor": 2,
                        "panel_throughput": 9999,
                        "judge_throughput": 9999,
                        "overhead_seconds": 0,
                        "max_timeout": 300,
                    },
                    "retry": {"max_retries": 0, "delays_seconds": [0.1]},
                },
            },
            "pipeline": {"max_panel_workers": 1, "min_survivors": 1, "graceful_degradation": True},
        }

        with mock.patch("scripts.panel.call_llm_with_retry") as mock_call:
            mock_call.return_value = {"success": True, "content": "test",
                                      "reasoning_content": None, "usage": {},
                                      "elapsed": 0.01}
            dispatch_panel("test", "general", config=config, tier="high", max_workers=1)

        # Check that call_llm_with_retry was called with extra_params containing top_k
        found_top_k = False
        for call_args in mock_call.call_args_list:
            kwargs = call_args[1]
            extra = kwargs.get("extra_params") or {}
            if "top_k" in extra:
                found_top_k = True
                break
        self.assertTrue(found_top_k, "No call had top_k in extra_params")

    def test_qwen_reasoning_effort_passed_in_extra_params(self):
        """qwen reasoning_effort and top_k should pass through extra_params."""
        from scripts.panel import _build_call_specs

        models_list = [
            {"name": "qwen3.7-plus", "count": 1, "temp": 0.8, "top_p": 0.92,
             "top_k": 20, "reasoning_effort": "high", "max_tokens": 2048},
        ]
        specs = _build_call_specs(models_list, "test query", {})
        self.assertEqual(len(specs), 1)
        spec = specs[0]
        self.assertEqual(spec["model"], "qwen3.7-plus")
        self.assertEqual(spec["max_tokens"], 2048)
        self.assertIsNone(spec["max_completion_tokens"])
        self.assertEqual(spec["extra_params"]["top_k"], 20)
        self.assertEqual(spec["extra_params"]["reasoning_effort"], "high")

    def test_reasoning_effort_applies_timeout_multiplier(self):
        """reasoning_effort should apply the same 1.5x timeout multiplier as thinking."""
        from scripts.panel import _derive_timeout
        timeout_cfg = {
            "panel_floor": 2,
            "panel_throughput": 100,
            "overhead_seconds": 0,
            "max_timeout": 300,
        }
        model_entry = {"max_tokens": 1000, "reasoning_effort": "high"}
        self.assertEqual(_derive_timeout(model_entry, timeout_cfg), 15)

    def test_count_zero_skipped(self):
        """Model entry with count<=0 should not produce call specs."""
        from scripts.panel import _build_call_specs

        models_list = [
            {"name": "deepseek-v4-flash", "count": 0, "temp": 0.75, "top_p": 0.9},
            {"name": "mimo-v2.5", "count": 2, "temps": [0.6, 0.7], "top_p": 0.95},
        ]
        specs = _build_call_specs(models_list, "test query", {})
        self.assertEqual(len(specs), 2)
        for spec in specs:
            self.assertNotEqual(spec["model"], "deepseek-v4-flash")


class TestCLITier(unittest.TestCase):
    """Test CLI --tier argument."""

    def _run(self, *args):
        """Run scripts.cli.main with args and return (returncode, stdout, stderr)."""
        from io import StringIO
        from scripts.cli import main
        import sys

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        try:
            rc = main(list(args))
            return rc, sys.stdout.getvalue(), sys.stderr.getvalue()
        except SystemExit as e:
            return e.code if e.code is not None else 0, sys.stdout.getvalue(), sys.stderr.getvalue()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def test_dry_run_includes_tier_default(self):
        """Dry-run JSON includes tier field (default low2)."""
        import json
        rc, out, _ = self._run("--dry-run", "--query", "test")
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertIn("tier", data)

    def test_dry_run_tier_low1(self):
        """Dry-run with --tier low1 shows tier=low1."""
        import json
        rc, out, _ = self._run("--dry-run", "--query", "test", "--tier", "low1")
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual(data["tier"], "low1")

    def test_dry_run_tier_medium(self):
        """Dry-run with --tier medium shows tier=medium."""
        import json
        rc, out, _ = self._run("--dry-run", "--query", "test", "--tier", "medium")
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual(data["tier"], "medium")

    def test_dry_run_tier_high(self):
        """Dry-run with --tier high shows tier=high."""
        import json
        rc, out, _ = self._run("--dry-run", "--query", "test", "--tier", "high")
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual(data["tier"], "high")

    def test_dry_run_tier_invalid_accepted(self):
        """Invalid --tier value is accepted at parse time (validation happens in pipeline)."""
        import json
        rc, out, _ = self._run("--dry-run", "--query", "test", "--tier", "ultra")
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual(data["tier"], "ultra")


class TestPipelineTier(unittest.TestCase):
    """Test pipeline tier integration."""

    def test_pipeline_accepts_tier(self):
        """run_pipeline accepts tier parameter and includes it in metadata."""
        from scripts.pipeline import run_pipeline
        result = run_pipeline("What is 2+2?", tier="low1")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"].get("tier"), "low1")

    def test_pipeline_default_tier(self):
        """run_pipeline defaults tier to low2 in metadata."""
        from scripts.pipeline import run_pipeline
        result = run_pipeline("What is 2+2?")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"].get("tier"), "low2")


if __name__ == "__main__":
    unittest.main()
