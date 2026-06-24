"""Test suite for llm-fusion (migrated from root-level test_fusion.py).

Tests are organized by module and run without external API dependencies
for unit tests. Integration tests make real API calls.
"""

import os
import sys
import json
import time
import unittest
import tempfile

import scripts


class TestConfig(unittest.TestCase):
    """Test llm_fusion/config.py"""

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
            "cleaning": {
                "profiles": {
                    "coding": {"strip_fences": False, "strip_preamble": True, "min_words": 15, "dedup_threshold": 0.70},
                    "general": {"strip_fences": True, "strip_preamble": True, "min_words": 10, "dedup_threshold": 0.85},
                },
            },
            "api": {
                "primary": {
                    "endpoint": "https://opencode.ai/zen/go/v1/chat/completions",
                    "timeout": {
                        "panel_floor": 30,
                        "judge_floor": 60,
                        "panel_throughput": 25,
                        "judge_throughput": 20,
                        "overhead_seconds": 10,
                        "max_timeout": 300,
                    },
                },
            },
            "pipeline": {
                "max_panel_workers": 6,
                "min_survivors": 2,
                "graceful_degradation": True,
            },
        }

    def test_get_scenario_config_coding(self):
        from scripts.config import get_scenario_config
        cfg = get_scenario_config(self.minimal_config, "coding")
        self.assertIn("panel", cfg)
        self.assertIn("judge", cfg)
        self.assertIn("cleaning", cfg)
        self.assertIn("conciseness_suffix", cfg)
        judge = cfg["judge"]
        self.assertEqual(judge.get("stages"), "single")
        self.assertEqual(judge.get("reasoning_mode"), "max")

    def test_get_scenario_config_fallback(self):
        from scripts.config import get_scenario_config
        cfg = get_scenario_config(self.minimal_config, "nonexistent_scenario")
        self.assertIn("panel", cfg)

    def test_get_scenario_config_empty(self):
        from scripts.config import get_scenario_config
        cfg = get_scenario_config({}, "coding")
        self.assertIn("panel", cfg)

    def test_default_scenario_config_uses_mimo_judge(self):
        from scripts.config import get_scenario_config

        cfg = get_scenario_config({}, "general")
        judge = cfg["judge"]

        self.assertEqual(judge["model"], "mimo-v2.5")
        self.assertEqual(judge["temp"], 1.0)
        self.assertEqual(judge["top_p"], 0.95)
        self.assertEqual(judge["max_tokens"], 4096)
        self.assertEqual(judge["thinking"], {"type": "enabled"})
        self.assertNotIn("max_completion_tokens", judge)
        self.assertNotIn("reasoning_mode", judge)

        mimo_panel = next(m for m in cfg["panel"]["models"] if m["name"] == "mimo-v2.5")
        self.assertEqual(mimo_panel["max_tokens"], 1000)

    def test_bundled_configs_mirror_timeout_and_deadline_policy(self):
        from scripts.config import load_config

        active = load_config("skills/llm-fusion/assets/fusion_config.yaml")
        example = load_config("skills/llm-fusion/assets/fusion_config.yaml.example")

        expected_timeout = {
            "panel_floor": 60,
            "judge_floor": 90,
            "panel_throughput": 25,
            "judge_throughput": 20,
            "overhead_seconds": 15,
            "max_timeout": 360,
        }

        self.assertEqual(active["api"]["primary"]["timeout"], expected_timeout)
        self.assertEqual(example["api"]["primary"]["timeout"], expected_timeout)
        self.assertEqual(active["pipeline"]["soft_deadline_seconds"], 300)
        self.assertEqual(example["pipeline"]["soft_deadline_seconds"], 300)

    def test_bundled_configs_have_scenario_mimo_panel_budgets(self):
        from scripts.config import get_scenario_config, load_config

        expected = {
            "qa": 800,
            "general": 1200,
            "coding": 2000,
            "bugfix": 2000,
            "reasoning": 2000,
            "plan_review": 2500,
            "creative": 2500,
            "document": 3000,
        }
        paths = [
            "skills/llm-fusion/assets/fusion_config.yaml",
            "skills/llm-fusion/assets/fusion_config.yaml.example",
        ]
        for path in paths:
            cfg = load_config(path)
            for scenario, max_tokens in expected.items():
                scenario_cfg = get_scenario_config(cfg, scenario, tier="low1")
                mimo = next(m for m in scenario_cfg["panel"]["models"] if m["name"] == "mimo-v2.5")
                self.assertEqual(mimo["max_tokens"], max_tokens, f"{path}:{scenario}")
                self.assertEqual(mimo["thinking"], {"type": "disabled"}, f"{path}:{scenario}")

    def test_bundled_configs_have_4096_high_tier_defaults(self):
        from scripts.config import load_config

        paths = [
            "skills/llm-fusion/assets/fusion_config.yaml",
            "skills/llm-fusion/assets/fusion_config.yaml.example",
        ]
        for path in paths:
            cfg = load_config(path)
            models = {m["name"]: m for m in cfg["default"]["panel"]["models"]}
            self.assertEqual(models["minimax-m3"]["max_tokens"], 4096, path)
            self.assertEqual(models["qwen3.7-plus"]["max_tokens"], 4096, path)
            self.assertEqual(models["deepseek-v4-pro"]["max_completion_tokens"], 4096, path)

    def test_bundled_configs_mirror_panel_response_caps(self):
        from scripts.config import load_config

        active = load_config("skills/llm-fusion/assets/fusion_config.yaml")
        example = load_config("skills/llm-fusion/assets/fusion_config.yaml.example")

        for scenario in ("qa", "general"):
            active_cap = active["scenarios"][scenario]["judge"].get("max_panel_response_chars")
            example_cap = example["scenarios"][scenario]["judge"].get("max_panel_response_chars")
            self.assertEqual(example_cap, active_cap, scenario)

        self.assertEqual(active["scenarios"]["qa"]["judge"]["max_panel_response_chars"], 1200)
        self.assertEqual(active["scenarios"]["general"]["judge"]["max_panel_response_chars"], 1800)

    def test_bundled_configs_use_mimo_judge_params(self):
        from scripts.config import load_config

        paths = [
            "skills/llm-fusion/assets/fusion_config.yaml",
            "skills/llm-fusion/assets/fusion_config.yaml.example",
        ]
        for path in paths:
            cfg = load_config(path)
            judge = cfg["default"]["judge"]
            self.assertEqual(judge["model"], "mimo-v2.5", path)
            self.assertEqual(judge["temp"], 1.0, path)
            self.assertEqual(judge["top_p"], 0.95, path)
            self.assertEqual(judge["max_tokens"], 4096, path)
            self.assertEqual(judge["thinking"], {"type": "enabled"}, path)
            self.assertNotIn("max_completion_tokens", judge, path)
            self.assertNotIn("reasoning_mode", judge, path)

            for scenario, scenario_cfg in cfg.get("scenarios", {}).items():
                scenario_judge = scenario_cfg.get("judge", {})
                self.assertNotIn("reasoning_mode", scenario_judge, f"{path}:{scenario}")
                self.assertNotIn("max_completion_tokens", scenario_judge, f"{path}:{scenario}")
                for stage_key in ("stage1", "stage2"):
                    stage = scenario_judge.get(stage_key, {})
                    self.assertNotIn("reasoning_mode", stage, f"{path}:{scenario}:{stage_key}")
                    self.assertNotIn("max_completion_tokens", stage, f"{path}:{scenario}:{stage_key}")
                    if stage:
                        self.assertEqual(stage.get("max_tokens"), 6144, f"{path}:{scenario}:{stage_key}")

    def test_bundled_two_stage_judges_use_6144_tokens(self):
        from scripts.config import load_config

        two_stage_scenarios = ("bugfix", "plan_review", "reasoning", "document")
        paths = [
            "skills/llm-fusion/assets/fusion_config.yaml",
            "skills/llm-fusion/assets/fusion_config.yaml.example",
        ]
        for path in paths:
            cfg = load_config(path)
            for scenario in two_stage_scenarios:
                judge = cfg["scenarios"][scenario]["judge"]
                self.assertEqual(judge["stages"], "two", f"{path}:{scenario}")
                self.assertEqual(judge["stage1"]["max_tokens"], 6144, f"{path}:{scenario}:stage1")
                self.assertEqual(judge["stage2"]["max_tokens"], 6144, f"{path}:{scenario}:stage2")

    def test_get_cleaning_profile(self):
        from scripts.config import get_cleaning_profile
        prof = get_cleaning_profile(self.minimal_config, "coding")
        self.assertFalse(prof["strip_fences"])
        self.assertEqual(prof["min_words"], 15)

    def test_get_cleaning_profile_fallback(self):
        from scripts.config import get_cleaning_profile
        prof = get_cleaning_profile(self.minimal_config, "missing")
        self.assertEqual(prof["min_words"], 10)

    def test_get_cleaning_profile_empty(self):
        from scripts.config import get_cleaning_profile
        prof = get_cleaning_profile({}, "coding")
        self.assertEqual(prof["min_words"], 10)

    def test_load_config_missing_file(self):
        from scripts.config import load_config
        cfg = load_config("/tmp/nonexistent_fusion_config_xyz.yaml")
        self.assertEqual(cfg, {})


class TestAPIClient(unittest.TestCase):
    """Test llm_fusion/api_client.py"""

    def test_read_api_key_missing(self):
        from scripts.api_client import read_api_key
        old_val = os.environ.pop("OPENCODE_GO_API_KEY", None)
        try:
            key = read_api_key(env_path="/tmp/nonexistent.env.xyz")
            self.assertIsNone(key)
        finally:
            if old_val is not None:
                os.environ["OPENCODE_GO_API_KEY"] = old_val

    def test_call_llm_no_key(self):
        from scripts.api_client import call_llm
        old_val = os.environ.pop("OPENCODE_GO_API_KEY", None)
        try:
            result = call_llm("test prompt", endpoint="https://api.example.com/v1/nonexistent")
            self.assertFalse(result["success"])
            self.assertIsNotNone(result["error"])
        finally:
            if old_val is not None:
                os.environ["OPENCODE_GO_API_KEY"] = old_val

    def test_call_llm_empty_prompt(self):
        from scripts.api_client import call_llm
        result = call_llm("", endpoint="https://api.example.com/v1/nonexistent")
        self.assertIn("success", result)

    def test_call_llm_with_retry_no_key(self):
        from scripts.api_client import call_llm_with_retry
        old_val = os.environ.pop("OPENCODE_GO_API_KEY", None)
        try:
            result = call_llm_with_retry(
                "test", retries=0, delays=(0.1,),
                endpoint="http://127.0.0.1:1/nonexistent",
                api_key="invalid-but-not-none",
            )
            self.assertFalse(result["success"])
        finally:
            if old_val is not None:
                os.environ["OPENCODE_GO_API_KEY"] = old_val

    def test_call_llm_mimo_defaults_thinking_disabled_when_not_provided(self):
        from unittest import mock
        from scripts.api_client import call_llm

        captured = {}

        class _Resp:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def read(self):
                return b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}'

        def _fake_urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _Resp()

        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            result = call_llm(
                "test",
                model="mimo-v2.5",
                endpoint="https://api.example.test/v1/chat/completions",
                api_key="test-key",
                max_tokens=2048,
            )

        self.assertTrue(result["success"])
        self.assertEqual(captured["payload"]["thinking"], {"type": "disabled"})
        self.assertEqual(captured["payload"]["max_tokens"], 2048)
        self.assertNotIn("max_completion_tokens", captured["payload"])

    def test_call_llm_mimo_preserves_explicit_thinking_enabled(self):
        from unittest import mock
        from scripts.api_client import call_llm

        captured = {}

        class _Resp:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def read(self):
                return b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}'

        def _fake_urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _Resp()

        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            result = call_llm(
                "test",
                model="mimo-v2.5",
                endpoint="https://api.example.test/v1/chat/completions",
                api_key="test-key",
                max_tokens=2048,
                extra_params={"thinking": {"type": "enabled"}},
            )

        self.assertTrue(result["success"])
        self.assertEqual(captured["payload"]["thinking"], {"type": "enabled"})

    def test_call_llm_mimo_panel_thinking_disabled_when_extra_params_not_provided(self):
        """Panel Mimo calls (without extra_params) still get thinking.disabled default."""
        from unittest import mock
        from scripts.api_client import call_llm

        captured = {}

        class _Resp:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def read(self):
                return b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}'

        def _fake_urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _Resp()

        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            result = call_llm(
                "test",
                model="mimo-v2.5",
                endpoint="https://api.example.test/v1/chat/completions",
                api_key="test-key",
                max_tokens=600,
            )

        self.assertTrue(result["success"])
        self.assertEqual(captured["payload"]["thinking"], {"type": "disabled"})


class TestClassifier(unittest.TestCase):
    """Test llm_fusion/classifier.py"""

    def test_classify_coding(self):
        from scripts.classifier import classify_query
        result = classify_query("Write a Python function that sorts a list")
        self.assertEqual(result["scenario"], "coding")
        self.assertEqual(result["detection_method"], "regex")
        self.assertGreaterEqual(result["confidence"], 0.8)

    def test_classify_coding_with_fences(self):
        from scripts.classifier import classify_query
        result = classify_query("```python\ndef hello():\n    pass\n```\nWhat does this do?")
        self.assertEqual(result["scenario"], "coding")

    def test_classify_qa_short(self):
        from scripts.classifier import classify_query
        result = classify_query("What is the capital of France?")
        self.assertEqual(result["scenario"], "qa")

    def test_classify_qa_long(self):
        from scripts.classifier import classify_query
        long_q = "What " + ("x" * 200)
        result = classify_query(long_q)
        self.assertNotEqual(result["scenario"], "qa")

    def test_classify_bugfix(self):
        from scripts.classifier import classify_query
        result = classify_query("Why is my code crashing? Traceback (most recent call last):")
        self.assertEqual(result["scenario"], "bugfix")

    def test_classify_reasoning(self):
        from scripts.classifier import classify_query
        result = classify_query("Prove that the square root of 2 is irrational")
        self.assertEqual(result["scenario"], "reasoning")

    def test_classify_creative(self):
        from scripts.classifier import classify_query
        result = classify_query("Write a poem about artificial intelligence")
        self.assertEqual(result["scenario"], "creative")

    def test_classify_plan_review(self):
        from scripts.classifier import classify_query
        result = classify_query("Review this architecture proposal for the new system")
        self.assertEqual(result["scenario"], "plan_review")

    def test_classify_document(self):
        from scripts.classifier import classify_query
        long_query = "Please improve " + ("x" * 500)
        result = classify_query(long_query)
        self.assertEqual(result["scenario"], "document")

    def test_classify_general(self):
        from scripts.classifier import classify_query
        result = classify_query("Tell me something interesting")
        self.assertEqual(result["scenario"], "general")

    def test_classify_empty(self):
        from scripts.classifier import classify_query
        result = classify_query("")
        self.assertEqual(result["scenario"], "general")

    def test_classify_none(self):
        from scripts.classifier import classify_query
        result = classify_query(None)
        self.assertEqual(result["scenario"], "general")

    def test_conciseness_suffixes_exist(self):
        from scripts.classifier import CONCISENESS_SUFFIXES
        expected = {"coding", "bugfix", "qa", "plan_review", "creative", "reasoning", "document", "general"}
        self.assertEqual(set(CONCISENESS_SUFFIXES.keys()), expected)

    def test_classify_default_disabled_no_llm_call(self):
        """classification.enabled defaults to false; LLM classifier not called with empty classification config."""
        from unittest import mock
        from scripts.classifier import classify_query

        with mock.patch("scripts.classifier._llm_classifier") as mock_llm:
            result = classify_query(
                "Tell me something interesting",
                {"classification": {}},
            )
        self.assertEqual(result["scenario"], "general")
        self.assertEqual(result["detection_method"], "regex")
        mock_llm.assert_not_called()

    def test_classify_explicit_disabled_no_llm_call(self):
        """classification.enabled: false explicitly; LLM classifier not called."""
        from unittest import mock
        from scripts.classifier import classify_query

        with mock.patch("scripts.classifier._llm_classifier") as mock_llm:
            result = classify_query(
                "Tell me something interesting",
                {"classification": {"enabled": False, "confidence_threshold": 0.85}},
            )
        self.assertEqual(result["scenario"], "general")
        self.assertEqual(result["detection_method"], "regex")
        mock_llm.assert_not_called()

    def test_classify_enabled_calls_llm_for_low_confidence(self):
        """classification.enabled: true triggers LLM classifier for low-confidence general fallback."""
        from unittest import mock
        from scripts.classifier import classify_query

        mock_result = {"scenario": "creative", "confidence": 0.9, "reason": "mock"}
        with mock.patch("scripts.classifier._llm_classifier", return_value=mock_result) as mock_llm:
            result = classify_query(
                "Tell me something interesting",
                {"classification": {"enabled": True, "confidence_threshold": 0.85}},
            )
        mock_llm.assert_called_once()
        self.assertEqual(result["detection_method"], "llm")
        self.assertEqual(result["scenario"], "creative")

    def test_classify_high_confidence_bypasses_llm_even_when_enabled(self):
        """High-confidence regex match bypasses LLM even when classification.enabled is true."""
        from unittest import mock
        from scripts.classifier import classify_query

        with mock.patch("scripts.classifier._llm_classifier") as mock_llm:
            result = classify_query(
                "Write a Python function that sorts a list",
                {"classification": {"enabled": True, "confidence_threshold": 0.85}},
            )
        self.assertEqual(result["scenario"], "coding")
        self.assertEqual(result["detection_method"], "regex")
        mock_llm.assert_not_called()


class TestCleaner(unittest.TestCase):
    """Test llm_fusion/cleaner.py"""

    def test_clean_response_strip_preamble(self):
        from scripts.cleaner import clean_response
        text = "Here is the code:\n\ndef foo():\n    pass"
        cleaned = clean_response(text, "coding", {"strip_fences": False, "strip_preamble": True})
        self.assertNotIn("Here is the code", cleaned)
        self.assertIn("def foo():", cleaned)

    def test_clean_response_qa_preamble(self):
        from scripts.cleaner import clean_response
        text = "Sure! The capital of France is Paris."
        cleaned = clean_response(text, "qa", {"strip_fences": True, "strip_preamble": True})
        self.assertNotIn("Sure!", cleaned)
        self.assertIn("Paris", cleaned)

    def test_clean_response_strip_trailing(self):
        from scripts.cleaner import clean_response
        text = "The answer is 42. Let me know if you have any questions."
        cleaned = clean_response(text, "qa", {"strip_fences": True, "strip_preamble": False})
        self.assertIn("42", cleaned)
        self.assertNotIn("Let me know", cleaned)

    def test_clean_response_preserve_trailing_creative(self):
        from scripts.cleaner import clean_response
        text = "The poem ends here. I hope you enjoyed it!"
        cleaned = clean_response(text, "creative", {"strip_fences": True, "strip_preamble": False})
        self.assertIn("I hope you enjoyed it!", cleaned)

    def test_clean_response_code_fences(self):
        from scripts.cleaner import clean_response
        text = "```python\nprint('hello')\n```"
        cleaned = clean_response(text, "qa", {"strip_fences": True, "strip_preamble": False})
        self.assertNotIn("```", cleaned)

    def test_clean_response_preserve_fences_coding(self):
        from scripts.cleaner import clean_response
        text = "```python\nprint('hello')\n```"
        cleaned = clean_response(text, "coding", {"strip_fences": False, "strip_preamble": False})
        self.assertIn("```", cleaned)

    def test_clean_response_empty(self):
        from scripts.cleaner import clean_response
        self.assertEqual(clean_response("", "general"), "")
        self.assertEqual(clean_response(None, "general"), "")

    def test_dedup_responses(self):
        from scripts.cleaner import dedup_responses
        responses = [
            {"label": "A", "success": True, "content": "The capital of France is Paris. It is a beautiful city and a major European hub for culture."},
            {"label": "B", "success": True, "content": "The capital of France is Paris. It is a beautiful city and a major European hub for culture, art, and fashion."},
        ]
        result = dedup_responses(responses, "general")
        surviving = [r for r in result if not r.get("discarded") and r.get("success")]
        self.assertEqual(len(surviving), 1)

    def test_dedup_responses_empty(self):
        from scripts.cleaner import dedup_responses
        result = dedup_responses([], "general")
        self.assertEqual(result, [])

    def test_dedup_responses_diverse(self):
        from scripts.cleaner import dedup_responses
        responses = [
            {"label": "A", "success": True, "content": "The capital of France is Paris."},
            {"label": "B", "success": True, "content": "Python is a programming language."},
        ]
        result = dedup_responses(responses, "qa")
        surviving = [r for r in result if not r.get("discarded") and r.get("success")]
        self.assertEqual(len(surviving), 2)

    def test_clean_panel_responses(self):
        from scripts.cleaner import clean_panel_responses
        panel_result = {
            "responses": [
                {"label": "A", "success": True,
                 "content": "Sure! The answer is 42."},
                {"label": "B", "success": True,
                 "content": "The answer is 42. Let me know if you need more help."},
                {"label": "C", "success": False,
                 "content": "", "error": "API error"},
            ],
        }
        result = clean_panel_responses(panel_result, "qa", self._make_minimal_config())
        self.assertGreaterEqual(result["survived_count"], 1)

    def _make_minimal_config(self):
        return {
            "cleaning": {
                "profiles": {
                    "qa": {"strip_fences": True, "strip_preamble": True, "min_words": 3, "dedup_threshold": 0.85},
                    "general": {"strip_fences": True, "strip_preamble": True, "min_words": 10, "dedup_threshold": 0.85},
                },
            },
        }


class TestOutput(unittest.TestCase):
    """Test llm_fusion/output.py"""

    def test_format_for_chat_success(self):
        from scripts.output import format_for_chat
        result = {
            "success": True,
            "answer": "42",
            "scenario": "qa",
            "reasoning_content": None,
            "metadata": {
                "classification": {"scenario": "qa", "confidence": 0.85, "method": "regex"},
                "panel": {"models_attempted": 6, "models_succeeded": 6, "models_discarded": 0},
                "judge": {"config": {"stages": "single"}},
                "timing_ms": {"total": 5000},
            },
        }
        formatted = format_for_chat(result)
        self.assertIn("42", formatted)
        self.assertIn("qa", formatted)

    def test_format_for_chat_error(self):
        from scripts.output import format_for_chat
        result = {"success": False, "error": "Something went wrong", "scenario": "general"}
        formatted = format_for_chat(result)
        self.assertIn("Error", formatted)
        self.assertIn("Something went wrong", formatted)

    def test_format_for_chat_none(self):
        from scripts.output import format_for_chat
        formatted = format_for_chat(None)
        self.assertIn("Error", formatted)

    def test_save_output(self):
        from scripts.output import save_output
        with tempfile.TemporaryDirectory() as tmpdir:
            result = {"success": True, "answer": "test", "scenario": "qa", "metadata": {}}
            path = save_output(result, output_dir=tmpdir, filename="test_output.json")
            self.assertIsNotNone(path)
            self.assertTrue(os.path.exists(path))
            with open(path) as fh:
                data = json.load(fh)
            self.assertEqual(data["answer"], "test")

    def test_save_output_invalid_dir(self):
        from scripts.output import save_output
        result = save_output({"success": True}, output_dir="/nonexistent_dir_xyz")
        self.assertIsNone(result)


class TestFallback(unittest.TestCase):
    """Test llm_fusion/fallback.py"""

    def test_rate_limiter_acquire(self):
        from scripts.fallback import RateLimiter
        limiter = RateLimiter(rate=100, burst=100)
        self.assertTrue(limiter.acquire(1, block=False))
        self.assertTrue(limiter.acquire(50, block=False))

    def test_rate_limiter_exhaustion(self):
        from scripts.fallback import RateLimiter
        limiter = RateLimiter(rate=100, burst=5)
        self.assertTrue(limiter.acquire(5, block=False))
        self.assertFalse(limiter.acquire(1, block=False))

    def test_rate_limiter_refill(self):
        from scripts.fallback import RateLimiter
        limiter = RateLimiter(rate=1000, burst=5)
        limiter.acquire(5, block=False)
        time.sleep(0.01)
        self.assertTrue(limiter.acquire(1, block=False))


class TestPipeline(unittest.TestCase):
    """Test llm_fusion/pipeline.py (basic structure, no API calls)."""

    def test_run_pipeline_empty(self):
        from scripts.pipeline import run_pipeline
        result = run_pipeline("")
        self.assertFalse(result["success"])

    def test_run_pipeline_none(self):
        from scripts.pipeline import run_pipeline
        result = run_pipeline(None)
        self.assertFalse(result["success"])

    def test_pipeline_returns_metadata(self):
        from scripts.pipeline import run_pipeline
        result = run_pipeline("What is 2+2?")
        self.assertIn("success", result)
        self.assertIn("scenario", result)
        self.assertIn("metadata", result)
        self.assertIn("timing_ms", result["metadata"])

    def test_direct_fallback(self):
        from scripts.pipeline import _direct_fallback
        result = _direct_fallback("test", {})
        self.assertIn("success", result)

    def test_pipeline_soft_deadline_disabled_by_default(self):
        """Pipeline runs normally when soft_deadline_seconds=0 (disabled)."""
        from scripts.pipeline import run_pipeline
        result = run_pipeline("What is 2+2?")
        self.assertIn("success", result)
        # No deadline_exceeded flag when disabled
        self.assertNotIn("deadline_exceeded", result.get("metadata", {}))

    def test_pipeline_soft_deadline_triggers_fallback(self):
        """Pipeline falls back to direct call when soft deadline is exceeded."""
        from unittest import mock
        from scripts.pipeline import run_pipeline

        # A config with a very short deadline (0.001s) so panel dispatch
        # will exceed it. Mock the API calls to avoid network requests.
        config_with_deadline = {
            "default": {
                "panel": {
                    "models": [
                        {"name": "deepseek-v4-flash", "count": 2, "temp": 0.75,
                         "top_p": 0.9, "max_completion_tokens": 100},
                    ],
                },
                "judge": {"model": "deepseek-v4-flash", "temp": 0.0, "top_p": 1.0},
            },
            "scenarios": {
                "general": {
                    "panel": {"deepseek": {"temp": 0.75, "max_completion_tokens": 100}},
                    "judge": {"stages": "single", "reasoning_mode": "low", "max_completion_tokens": 100},
                },
            },
            "api": {
                "primary": {
                    "endpoint": "http://127.0.0.1:1/nonexistent",
                    "timeout": {"panel_floor": 2, "judge_floor": 2,
                                "panel_throughput": 9999, "judge_throughput": 9999,
                                "overhead_seconds": 0, "max_timeout": 300},
                    "retry": {"max_retries": 0, "delays_seconds": [0.01]},
                },
            },
            "pipeline": {
                "soft_deadline_seconds": 0.001,  # very short
                "max_panel_workers": 2,
                "min_survivors": 1,
                "graceful_degradation": True,
            },
        }

        with mock.patch("scripts.pipeline.load_config") as mock_load:
            mock_load.return_value = config_with_deadline
            with mock.patch("scripts.pipeline.dispatch_panel") as mock_panel:
                # Simulate a slow panel that takes 0.1s
                import time as _time
                def _slow_panel(*args, **kwargs):
                    _time.sleep(0.1)
                    return {"success": True, "responses": [
                        {"label": "A", "success": True, "content": "test",
                         "reasoning_content": None, "usage": {}, "elapsed": 0.1},
                    ]}
                mock_panel.side_effect = _slow_panel

                with mock.patch("scripts.pipeline._direct_fallback") as mock_fallback:
                    mock_fallback.return_value = {
                        "success": True, "content": "fallback answer",
                        "reasoning_content": None, "usage": {}, "elapsed": 0.01,
                    }
                    result = run_pipeline("test query")

        # Should have triggered fallback due to deadline
        self.assertTrue(result["success"], "Should succeed via fallback")
        self.assertEqual(result["answer"], "fallback answer")
        self.assertTrue(result["metadata"].get("deadline_exceeded", False),
                        "Should have deadline_exceeded flag")
        self.assertEqual(result["metadata"].get("judge", {}).get("mode"),
                         "direct_fallback")

    def test_pipeline_soft_deadline_no_graceful(self):
        """Pipeline returns error on deadline when graceful_degradation=False."""
        from unittest import mock
        from scripts.pipeline import run_pipeline

        config_no_grace = {
            "default": {
                "panel": {
                    "models": [
                        {"name": "deepseek-v4-flash", "count": 1, "temp": 0.75,
                         "top_p": 0.9, "max_completion_tokens": 100},
                    ],
                },
                "judge": {"model": "deepseek-v4-flash", "temp": 0.0, "top_p": 1.0},
            },
            "scenarios": {
                "general": {
                    "panel": {"deepseek": {"temp": 0.75, "max_completion_tokens": 100}},
                    "judge": {"stages": "single", "reasoning_mode": "low", "max_completion_tokens": 100},
                },
            },
            "api": {
                "primary": {
                    "endpoint": "http://127.0.0.1:1/nonexistent",
                    "timeout": {"panel_floor": 2, "judge_floor": 2,
                                "panel_throughput": 9999, "judge_throughput": 9999,
                                "overhead_seconds": 0, "max_timeout": 300},
                    "retry": {"max_retries": 0, "delays_seconds": [0.01]},
                },
            },
            "pipeline": {
                "soft_deadline_seconds": 0.001,
                "max_panel_workers": 1,
                "min_survivors": 1,
                "graceful_degradation": False,
            },
        }

        with mock.patch("scripts.pipeline.load_config") as mock_load:
            mock_load.return_value = config_no_grace
            with mock.patch("scripts.pipeline.dispatch_panel") as mock_panel:
                import time as _time
                def _slow_panel(*args, **kwargs):
                    _time.sleep(0.1)
                    return {"success": True, "responses": [
                        {"label": "A", "success": True, "content": "test",
                         "reasoning_content": None, "usage": {}, "elapsed": 0.1},
                    ]}
                mock_panel.side_effect = _slow_panel

                result = run_pipeline("test query")

        # Should fail due to deadline with no graceful degradation
        self.assertFalse(result["success"])
        self.assertIn("deadline_exceeded", result.get("metadata", {}))
        self.assertIn("exceeded", result.get("error", ""))

    def test_pipeline_metadata_reports_mimo_judge_config(self):
        from unittest import mock
        from scripts.pipeline import run_pipeline

        config = {
            "default": {
                "panel": {
                    "models": [
                        {"name": "mimo-v2.5", "count": 1, "temp": 0.7,
                         "top_p": 0.95, "max_tokens": 100, "thinking": {"type": "disabled"}},
                    ],
                },
                "judge": {
                    "model": "mimo-v2.5",
                    "temp": 1.0,
                    "top_p": 0.95,
                    "stages": "single",
                    "max_tokens": 2048,
                    "thinking": {"type": "enabled"},
                },
            },
            "scenarios": {"general": {"judge": {"stages": "single"}}},
            "cleaning": {"profiles": {"general": {"strip_fences": True, "strip_preamble": True,
                                                    "min_words": 1, "dedup_threshold": 0.85}}},
            "api": {"primary": {"timeout": {"panel_floor": 2, "judge_floor": 2,
                                               "panel_throughput": 9999, "judge_throughput": 9999,
                                               "overhead_seconds": 0, "max_timeout": 300},
                                "retry": {"max_retries": 0, "delays_seconds": []}}},
            "pipeline": {"max_panel_workers": 1, "min_survivors": 1, "graceful_degradation": False},
        }

        with mock.patch("scripts.pipeline.load_config", return_value=config), \
             mock.patch("scripts.pipeline.dispatch_panel") as mock_panel, \
             mock.patch("scripts.pipeline.judge_single_stage") as mock_judge:
            mock_panel.return_value = {"success": True, "responses": [
                {"label": "A", "success": True, "content": "panel answer",
                 "reasoning_content": None, "usage": {}, "elapsed": 0.01},
            ]}
            mock_judge.return_value = {"success": True, "content": "final answer",
                                       "reasoning_content": None, "usage": {}, "elapsed": 0.02}
            result = run_pipeline("test query")

        self.assertTrue(result["success"])
        judge_config = result["metadata"]["judge"]["config"]
        self.assertEqual(judge_config["model"], "mimo-v2.5")
        self.assertEqual(judge_config["max_tokens"], 2048)
        self.assertEqual(judge_config["thinking"], {"type": "enabled"})
        self.assertIsNone(judge_config["reasoning_mode"])
        self.assertIsNone(judge_config["max_completion_tokens"])

    # ------------------------------------------------------------------
    # Config-Driven Direct Fallback tests
    # ------------------------------------------------------------------

    def test_direct_fallback_uses_config_values(self):
        """_direct_fallback passes config-driven values to call_llm_with_retry."""
        from unittest import mock
        from scripts.pipeline import _direct_fallback

        config = {
            "pipeline": {
                "direct_fallback": {
                    "model": "custom-model",
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "max_tokens": 1234,
                    "timeout": 7,
                    "retries": 3,
                    "delays_seconds": [0.1, 0.2, 0.3],
                },
            },
            "api": {
                "primary": {
                    "endpoint": "http://example.invalid",
                    "timeout": {"judge_floor": 99},
                },
            },
        }

        captured = {}

        def _fake_call(**kwargs):
            captured.update(kwargs)
            return {"success": True, "content": "ok", "elapsed": 0.01}

        # Patch call_llm_with_retry at its definition site
        with mock.patch("scripts.api_client.call_llm_with_retry", side_effect=_fake_call):
            result = _direct_fallback("test query", config)

        self.assertEqual(captured["model"], "custom-model")
        self.assertEqual(captured["temperature"], 0.2)
        self.assertEqual(captured["top_p"], 0.8)
        # max_tokens config key maps to max_completion_tokens
        self.assertEqual(captured["max_completion_tokens"], 1234)
        self.assertNotIn("max_tokens", captured)
        self.assertEqual(captured["timeout"], 7)
        self.assertEqual(captured["retries"], 3)
        self.assertEqual(captured["delays"], (0.1, 0.2, 0.3))
        self.assertEqual(captured["endpoint"], "http://example.invalid")

    def test_direct_fallback_preserves_defaults_with_empty_config(self):
        """_direct_fallback uses safe defaults when config has no direct_fallback block."""
        from unittest import mock
        from scripts.pipeline import _direct_fallback

        captured = {}

        def _fake_call(**kwargs):
            captured.update(kwargs)
            return {"success": True, "content": "ok", "elapsed": 0.01}

        with mock.patch("scripts.api_client.call_llm_with_retry", side_effect=_fake_call):
            result = _direct_fallback("test", {})

        self.assertEqual(captured["model"], "deepseek-v4-flash")
        self.assertEqual(captured["temperature"], 0.75)
        self.assertEqual(captured["top_p"], 0.9)
        self.assertEqual(captured["max_completion_tokens"], 2000)
        self.assertEqual(captured["timeout"], 60)
        self.assertEqual(captured["retries"], 1)
        self.assertEqual(captured["delays"], (2,))

    def test_apply_direct_fallback_writes_success_metadata(self):
        """_apply_direct_fallback sets fallback metadata on success."""
        from unittest import mock
        from scripts.pipeline import _apply_direct_fallback

        result = {
            "success": False,
            "answer": None,
            "reasoning_content": None,
            "metadata": {
                "level": "low",
                "judge": {},
                "timing_ms": {},
            },
        }

        config = {
            "pipeline": {
                "direct_fallback": {
                    "model": "fallback-model",
                },
            },
            "api": {
                "primary": {
                    "endpoint": "http://example.invalid",
                    "timeout": {"judge_floor": 5},
                },
            },
        }

        def _fake_fallback(query, cfg):
            return {
                "success": True,
                "content": "fallback answer",
                "reasoning_content": "some reasoning",
                "elapsed": 0.05,
            }

        with mock.patch("scripts.pipeline._direct_fallback", side_effect=_fake_fallback):
            direct_result = _apply_direct_fallback(result, "test_reason", "query", config)

        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "fallback answer")
        self.assertEqual(result["reasoning_content"], "some reasoning")
        self.assertEqual(result["metadata"]["level"], "low")
        self.assertEqual(result["metadata"]["judge"], {"mode": "direct_fallback", "model": "fallback-model"})
        self.assertEqual(result["metadata"]["fallback_reason"], "test_reason")
        self.assertEqual(result["metadata"]["fallback_model"], "fallback-model")
        self.assertIsInstance(result["metadata"]["fallback_elapsed_ms"], int)
        self.assertGreaterEqual(result["metadata"]["fallback_elapsed_ms"], 0)
        self.assertIsNone(result["metadata"]["fallback_error"])

    def test_apply_direct_fallback_writes_failure_metadata(self):
        """_apply_direct_fallback sets fallback_error and leaves success=False on failure."""
        from unittest import mock
        from scripts.pipeline import _apply_direct_fallback

        result = {
            "success": False,
            "answer": None,
            "metadata": {
                "level": "low",
                "judge": {},
                "timing_ms": {},
            },
        }

        config = {
            "pipeline": {"direct_fallback": {"model": "fb-model"}},
            "api": {"primary": {"timeout": {"judge_floor": 5}}},
        }

        def _fake_fallback(query, cfg):
            return {"success": False, "error": "boom", "elapsed": 0.05}

        with mock.patch("scripts.pipeline._direct_fallback", side_effect=_fake_fallback):
            direct_result = _apply_direct_fallback(result, "failure_reason", "query", config)

        self.assertFalse(result["success"])
        self.assertEqual(result["metadata"]["fallback_reason"], "failure_reason")
        self.assertEqual(result["metadata"]["fallback_error"], "boom")
        self.assertIsInstance(result["metadata"]["fallback_elapsed_ms"], int)
        self.assertGreaterEqual(result["metadata"]["fallback_elapsed_ms"], 0)

    def test_pipeline_panel_failure_branch_uses_fallback_metadata(self):
        """Pipeline panel failure uses _apply_direct_fallback and records fallback metadata."""
        from unittest import mock
        from scripts.pipeline import run_pipeline

        config = {
            "default": {
                "panel": {
                    "models": [
                        {"name": "deepseek-v4-flash", "count": 1, "temp": 0.75,
                         "top_p": 0.9, "max_completion_tokens": 100},
                    ],
                },
                "judge": {"model": "deepseek-v4-flash", "temp": 0.0, "top_p": 1.0},
            },
            "scenarios": {"general": {}},
            "cleaning": {"profiles": {"general": {}}},
            "api": {
                "primary": {
                    "endpoint": "http://127.0.0.1:1/nonexistent",
                    "timeout": {"panel_floor": 2, "judge_floor": 2,
                                "panel_throughput": 9999, "judge_throughput": 9999,
                                "overhead_seconds": 0, "max_timeout": 300},
                    "retry": {"max_retries": 0, "delays_seconds": []},
                },
            },
            "pipeline": {
                "soft_deadline_seconds": 0,
                "max_panel_workers": 1,
                "min_survivors": 1,
                "graceful_degradation": True,
            },
        }

        def _fake_fallback(query, cfg):
            return {
                "success": True,
                "content": "fallback answer via panel failure",
                "reasoning_content": None,
                "elapsed": 0.01,
            }

        with mock.patch("scripts.pipeline.load_config", return_value=config), \
             mock.patch("scripts.pipeline.dispatch_panel",
                        return_value={"success": False, "responses": []}), \
             mock.patch("scripts.pipeline._direct_fallback", side_effect=_fake_fallback):
            result = run_pipeline("test query")

        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "fallback answer via panel failure")
        self.assertEqual(result["metadata"]["fallback_reason"], "panel_failure")
        self.assertIn("fallback_model", result["metadata"])
        self.assertIn("fallback_elapsed_ms", result["metadata"])
        self.assertIsNone(result["metadata"].get("fallback_error"))

    def test_pipeline_insufficient_survivors_handles_fallback_failure(self):
        """Insufficient survivors returns early with metadata when fallback also fails."""
        from unittest import mock
        from scripts.pipeline import run_pipeline

        config = {
            "default": {
                "panel": {
                    "models": [
                        {"name": "deepseek-v4-flash", "count": 1, "temp": 0.75,
                         "top_p": 0.9, "max_completion_tokens": 100},
                    ],
                },
                "judge": {"model": "deepseek-v4-flash", "temp": 0.0, "top_p": 1.0},
            },
            "scenarios": {"general": {}},
            "cleaning": {"profiles": {"general": {}}},
            "api": {
                "primary": {
                    "endpoint": "http://127.0.0.1:1/nonexistent",
                    "timeout": {"panel_floor": 2, "judge_floor": 2,
                                "panel_throughput": 9999, "judge_throughput": 9999,
                                "overhead_seconds": 0, "max_timeout": 300},
                    "retry": {"max_retries": 0, "delays_seconds": []},
                },
            },
            "pipeline": {
                "soft_deadline_seconds": 0,
                "max_panel_workers": 1,
                "min_survivors": 2,
                "graceful_degradation": True,
            },
        }

        def _fake_fallback(query, cfg):
            return {"success": False, "error": "fallback failed", "content": None, "elapsed": 0.01}

        with mock.patch("scripts.pipeline.load_config", return_value=config), \
             mock.patch("scripts.pipeline.dispatch_panel") as mock_panel, \
             mock.patch("scripts.pipeline.clean_panel_responses") as mock_clean, \
             mock.patch("scripts.pipeline._direct_fallback", side_effect=_fake_fallback), \
             mock.patch("scripts.pipeline.judge_single_stage") as mock_judge:
            mock_panel.return_value = {
                "success": True,
                "responses": [
                    {"label": "A", "success": True, "content": "panel answer",
                     "reasoning_content": None, "usage": {}, "elapsed": 0.01},
                ],
            }
            mock_clean.return_value = {
                "cleaned_responses": [],
                "survived_count": 1,
                "discarded_count": 0,
            }
            result = run_pipeline("test query")

        # Should fail early with fallback metadata, NOT fall through to judge
        self.assertFalse(result["success"])
        self.assertEqual(result["metadata"].get("fallback_reason"), "insufficient_survivors")
        self.assertEqual(result["metadata"].get("fallback_error"), "fallback failed")
        self.assertIn("fallback_elapsed_ms", result["metadata"])
        mock_judge.assert_not_called()

    def test_pipeline_judge_failure_writes_fallback_metadata(self):
        """Judge failure branch writes fallback metadata via _apply_direct_fallback."""
        from unittest import mock
        from scripts.pipeline import run_pipeline

        config = {
            "default": {
                "panel": {
                    "models": [
                        {"name": "deepseek-v4-flash", "count": 1, "temp": 0.75,
                         "top_p": 0.9, "max_completion_tokens": 100},
                    ],
                },
                "judge": {"model": "deepseek-v4-flash", "temp": 0.0, "top_p": 1.0},
            },
            "scenarios": {"general": {}},
            "cleaning": {"profiles": {"general": {}}},
            "api": {
                "primary": {
                    "endpoint": "http://127.0.0.1:1/nonexistent",
                    "timeout": {"panel_floor": 2, "judge_floor": 2,
                                "panel_throughput": 9999, "judge_throughput": 9999,
                                "overhead_seconds": 0, "max_timeout": 300},
                    "retry": {"max_retries": 0, "delays_seconds": []},
                },
            },
            "pipeline": {
                "soft_deadline_seconds": 0,
                "max_panel_workers": 1,
                "min_survivors": 1,
                "graceful_degradation": True,
            },
        }

        def _fake_fallback(query, cfg):
            return {
                "success": True,
                "content": "judge fallback answer",
                "reasoning_content": None,
                "elapsed": 0.01,
            }

        with mock.patch("scripts.pipeline.load_config", return_value=config), \
             mock.patch("scripts.pipeline.dispatch_panel") as mock_panel, \
             mock.patch("scripts.pipeline.clean_panel_responses") as mock_clean, \
             mock.patch("scripts.pipeline._direct_fallback", side_effect=_fake_fallback):
            mock_panel.return_value = {
                "success": True,
                "responses": [
                    {"label": "A", "success": True, "content": "panel answer",
                     "reasoning_content": None, "usage": {}, "elapsed": 0.01},
                ],
            }
            mock_clean.return_value = {
                "cleaned_responses": [{"label": "A", "cleaned_content": "panel answer"}],
                "survived_count": 1,
                "discarded_count": 0,
            }
            # Make judge fail by not patching judge — the fake endpoint will fail
            result = run_pipeline("test query")

        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "judge fallback answer")
        self.assertEqual(result["metadata"]["fallback_reason"], "judge_failure")
        self.assertIn("fallback_model", result["metadata"])
        self.assertIn("fallback_elapsed_ms", result["metadata"])
        self.assertIsNone(result["metadata"].get("fallback_error"))

    def test_pipeline_soft_deadline_writes_fallback_metadata(self):
        """Soft deadline fallback records fallback_reason starting with 'soft_deadline:'."""
        from unittest import mock
        from scripts.pipeline import run_pipeline

        config = {
            "default": {
                "panel": {
                    "models": [
                        {"name": "deepseek-v4-flash", "count": 1, "temp": 0.75,
                         "top_p": 0.9, "max_completion_tokens": 100},
                    ],
                },
                "judge": {"model": "deepseek-v4-flash", "temp": 0.0, "top_p": 1.0},
            },
            "scenarios": {"general": {}},
            "cleaning": {"profiles": {"general": {}}},
            "api": {
                "primary": {
                    "endpoint": "http://127.0.0.1:1/nonexistent",
                    "timeout": {"panel_floor": 2, "judge_floor": 2,
                                "panel_throughput": 9999, "judge_throughput": 9999,
                                "overhead_seconds": 0, "max_timeout": 300},
                    "retry": {"max_retries": 0, "delays_seconds": []},
                },
            },
            "pipeline": {
                "soft_deadline_seconds": 0.001,
                "max_panel_workers": 1,
                "min_survivors": 1,
                "graceful_degradation": True,
            },
        }

        def _fake_fallback(query, cfg):
            return {"success": True, "content": "deadline fallback", "reasoning_content": None, "elapsed": 0.01}

        with mock.patch("scripts.pipeline.load_config", return_value=config), \
             mock.patch("scripts.pipeline.dispatch_panel") as mock_panel, \
             mock.patch("scripts.pipeline._direct_fallback", side_effect=_fake_fallback):
            import time as _time
            def _slow_panel(*args, **kwargs):
                _time.sleep(0.1)
                return {"success": True, "responses": [
                    {"label": "A", "success": True, "content": "test",
                     "reasoning_content": None, "usage": {}, "elapsed": 0.1},
                ]}
            mock_panel.side_effect = _slow_panel
            result = run_pipeline("test query")

        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "deadline fallback")
        self.assertTrue(result["metadata"].get("deadline_exceeded", False))
        self.assertIn("fallback_reason", result["metadata"])
        self.assertTrue(
            result["metadata"]["fallback_reason"].startswith("soft_deadline:"),
            f"Expected fallback_reason to start with 'soft_deadline:', got {result['metadata']['fallback_reason']!r}",
        )
        self.assertIn("fallback_model", result["metadata"])
        self.assertIn("fallback_elapsed_ms", result["metadata"])

    def test_bundled_configs_have_direct_fallback_block(self):
        """Both bundled config files define pipeline.direct_fallback with expected keys."""
        from scripts.config import load_config

        expected_keys = {"model", "temperature", "top_p", "max_tokens",
                         "timeout", "retries", "delays_seconds"}

        for path in ("skills/llm-fusion/assets/fusion_config.yaml",
                     "skills/llm-fusion/assets/fusion_config.yaml.example"):
            cfg = load_config(path)
            fb = cfg.get("pipeline", {}).get("direct_fallback", {})
            self.assertTrue(fb, f"{path} is missing pipeline.direct_fallback")
            self.assertEqual(
                set(fb.keys()),
                expected_keys,
                f"{path} direct_fallback keys mismatch: expected {expected_keys}, got {set(fb.keys())}",
            )

    def test_pipeline_metadata_includes_quorum_fields(self):
        """Pipeline metadata includes early quorum fields when panel provides them."""
        from unittest import mock
        from scripts.pipeline import run_pipeline

        config = {
            "default": {
                "panel": {
                    "models": [
                        {"name": "deepseek-v4-flash", "count": 1, "temp": 0.75,
                         "top_p": 0.9, "max_completion_tokens": 100},
                    ],
                },
                "judge": {"model": "deepseek-v4-flash", "temp": 0.0, "top_p": 1.0},
            },
            "scenarios": {"general": {}},
            "cleaning": {"profiles": {"general": {}}},
            "api": {
                "primary": {
                    "endpoint": "http://127.0.0.1:1/nonexistent",
                    "timeout": {"panel_floor": 2, "judge_floor": 2,
                                "panel_throughput": 9999, "judge_throughput": 9999,
                                "overhead_seconds": 0, "max_timeout": 300},
                    "retry": {"max_retries": 0, "delays_seconds": []},
                },
            },
            "pipeline": {
                "soft_deadline_seconds": 0,
                "max_panel_workers": 1,
                "min_survivors": 2,
                "graceful_degradation": False,
            },
        }

        panel_result = {
            "success": True,
            "responses": [
                {"label": "model #1", "success": True, "content": "a",
                 "reasoning_content": None, "usage": {}, "elapsed": 0.01},
                {"label": "model #2", "success": True, "content": "b",
                 "reasoning_content": None, "usage": {}, "elapsed": 0.01},
            ],
            "config_used": {},
            "elapsed": 0.05,
            "total_calls": 3,
            "quorum": 2,
            "quorum_reached": True,
            "quorum_at_ms": 123,
            "cancelled_count": 1,
            "late_completed_count": 0,
            "panel_calls_early_exit": True,
        }

        with mock.patch("scripts.pipeline.load_config", return_value=config), \
             mock.patch("scripts.pipeline.dispatch_panel",
                        return_value=panel_result), \
             mock.patch("scripts.pipeline.clean_panel_responses") as mock_clean, \
             mock.patch("scripts.pipeline.judge_single_stage") as mock_judge:
            mock_clean.return_value = {
                "cleaned_responses": [
                    {"label": "model #1", "cleaned_content": "a"},
                    {"label": "model #2", "cleaned_content": "b"},
                ],
                "survived_count": 2,
                "discarded_count": 0,
            }
            mock_judge.return_value = {
                "success": True, "content": "final answer",
                "reasoning_content": None, "usage": {}, "elapsed": 0.02,
            }
            result = run_pipeline("test query")

        self.assertTrue(result["success"])
        panel_meta = result["metadata"]["panel"]
        self.assertEqual(panel_meta["models_submitted"], 3)
        self.assertEqual(panel_meta["quorum"], 2)
        self.assertTrue(panel_meta["quorum_reached"])
        self.assertEqual(panel_meta["quorum_at_ms"], 123)
        self.assertEqual(panel_meta["cancelled_count"], 1)
        self.assertEqual(panel_meta["late_completed_count"], 0)
        self.assertTrue(panel_meta["panel_calls_early_exit"])


class TestJudge(unittest.TestCase):
    """Test llm_fusion/judge.py"""

    def test_build_responses_section(self):
        from scripts.judge import _build_responses_section
        responses = [
            {"label": "A", "cleaned_content": "Response A"},
            {"label": "B", "cleaned_content": "Response B"},
        ]
        section = _build_responses_section(responses)
        self.assertIn("=== A ===", section)
        self.assertIn("Response A", section)
        self.assertIn("=== B ===", section)
        self.assertIn("Response B", section)

    def test_build_responses_section_fallback(self):
        from scripts.judge import _build_responses_section
        responses = [
            {"label": "A", "content": "Raw Content"},
        ]
        section = _build_responses_section(responses)
        self.assertIn("Raw Content", section)

    def test_build_responses_section_empty(self):
        from scripts.judge import _build_responses_section
        section = _build_responses_section([])
        self.assertIn("no valid responses", section)

    def test_judge_single_stage_no_api(self):
        from scripts.judge import judge_single_stage
        fail_fast_config = {"api": {"primary": {"endpoint": "http://127.0.0.1:1/nonexistent", "timeout": {"judge_floor": 2, "panel_floor": 2, "judge_throughput": 9999, "panel_throughput": 9999, "overhead_seconds": 0, "max_timeout": 300}}}}
        result = judge_single_stage(
            "test", [{"label": "A", "content": "hello"}],
            "general", config=fail_fast_config,
            judge_config={
                "model": "mimo-v2.5",
                "temp": 1.0,
                "top_p": 0.95,
                "max_tokens": 2048,
                "thinking": {"type": "enabled"},
            },
        )
        self.assertIn("success", result)

    def test_judge_two_stage_no_api(self):
        from scripts.judge import judge_two_stage
        fail_fast_config = {"api": {"primary": {"endpoint": "http://127.0.0.1:1/nonexistent", "timeout": {"judge_floor": 2, "panel_floor": 2, "judge_throughput": 9999, "panel_throughput": 9999, "overhead_seconds": 0, "max_timeout": 300}}}}
        result = judge_two_stage(
            "test", [{"label": "A", "content": "hello"}],
            "general", config=fail_fast_config,
            judge_config={
                "model": "mimo-v2.5",
                "temp": 1.0,
                "top_p": 0.95,
                "max_tokens": 2048,
                "thinking": {"type": "enabled"},
                "stage1": {},
                "stage2": {},
            },
        )
        self.assertIn("success", result)
        self.assertIn("stage1", result)
        self.assertIn("stage2", result)

    # ------------------------------------------------------------------
    # _derive_judge_timeout tests
    # ------------------------------------------------------------------

    def test_derive_judge_timeout_from_tokens(self):
        """_derive_judge_timeout computes timeout from max_completion_tokens."""
        from scripts.judge import _derive_judge_timeout
        api_cfg = {
            "timeout": {
                "judge_floor": 30,
                "judge_throughput": 20,
                "overhead_seconds": 10,
                "max_timeout": 300,
            },
        }
        # budget=2000 → 2000/20 + 10 = 110
        judge_cfg = {"max_completion_tokens": 2000}
        timeout = _derive_judge_timeout(judge_cfg, api_cfg)
        self.assertEqual(timeout, 110)

    def test_derive_judge_timeout_floor(self):
        """_derive_judge_timeout falls back to judge_floor when no tokens."""
        from scripts.judge import _derive_judge_timeout
        api_cfg = {
            "timeout": {
                "judge_floor": 60,
                "judge_throughput": 20,
                "overhead_seconds": 10,
                "max_timeout": 300,
            },
        }
        timeout = _derive_judge_timeout({}, api_cfg)
        self.assertEqual(timeout, 60)

    def test_derive_judge_timeout_reasoning_multiplier(self):
        """_derive_judge_timeout multiplies by 1.5 when reasoning_mode is high or max."""
        from scripts.judge import _derive_judge_timeout
        api_cfg = {
            "timeout": {
                "judge_floor": 30,
                "judge_throughput": 20,
                "overhead_seconds": 10,
                "max_timeout": 300,
            },
        }
        # budget=2000 → 2000/20 + 10 = 110, *1.5 = 165
        judge_cfg = {"max_completion_tokens": 2000, "reasoning_mode": "high"}
        timeout = _derive_judge_timeout(judge_cfg, api_cfg)
        self.assertEqual(timeout, 165)

    def test_derive_judge_timeout_no_multiplier_low_reasoning(self):
        """_derive_judge_timeout does NOT multiply for reasoning_mode=low."""
        from scripts.judge import _derive_judge_timeout
        api_cfg = {
            "timeout": {
                "judge_floor": 30,
                "judge_throughput": 20,
                "overhead_seconds": 10,
                "max_timeout": 300,
            },
        }
        judge_cfg = {"max_completion_tokens": 2000, "reasoning_mode": "low"}
        timeout = _derive_judge_timeout(judge_cfg, api_cfg)
        self.assertEqual(timeout, 110)

    def test_derive_judge_timeout_two_stage(self):
        """_derive_judge_timeout considers stage2 max_completion_tokens for two-stage judges."""
        from scripts.judge import _derive_judge_timeout
        api_cfg = {
            "timeout": {
                "judge_floor": 30,
                "judge_throughput": 20,
                "overhead_seconds": 10,
                "max_timeout": 300,
            },
        }
        # stage1=4000 → 4000/20+10=210
        # stage2=8000, reasoning=max → (8000/20+10)*1.5 = 615
        # max(210, 615) = 615, clamped to 300
        judge_cfg = {
            "max_completion_tokens": 4000,
            "stage1": {"max_completion_tokens": 4000},
            "stage2": {"max_completion_tokens": 8000, "reasoning_mode": "max"},
        }
        timeout = _derive_judge_timeout(judge_cfg, api_cfg)
        self.assertEqual(timeout, 300)

    def test_derive_judge_timeout_two_stage_uses_stage2(self):
        """_derive_judge_timeout uses stage2 config when it demands a longer timeout."""
        from scripts.judge import _derive_judge_timeout
        api_cfg = {
            "timeout": {
                "judge_floor": 30,
                "judge_throughput": 20,
                "overhead_seconds": 10,
                "max_timeout": 300,
            },
        }
        # stage1=2000 → 2000/20+10=110
        # stage2=4000 → 4000/20+10=210
        judge_cfg = {
            "max_completion_tokens": 2000,
            "stage1": {"max_completion_tokens": 2000},
            "stage2": {"max_completion_tokens": 4000},
        }
        timeout = _derive_judge_timeout(judge_cfg, api_cfg)
        self.assertEqual(timeout, 210)

    def test_derive_judge_timeout_capped(self):
        """_derive_judge_timeout caps at max_timeout."""
        from scripts.judge import _derive_judge_timeout
        api_cfg = {
            "timeout": {
                "judge_floor": 30,
                "judge_throughput": 1,
                "overhead_seconds": 10,
                "max_timeout": 120,
            },
        }
        # budget=20000 → 20000/1+10 = 20010, capped to 120
        judge_cfg = {"max_completion_tokens": 20000}
        timeout = _derive_judge_timeout(judge_cfg, api_cfg)
        self.assertEqual(timeout, 120)

    def test_derive_judge_timeout_missing_timeout_config(self):
        """_derive_judge_timeout uses defaults when api timeout config is empty."""
        from scripts.judge import _derive_judge_timeout
        api_cfg = {"timeout": {}}
        judge_cfg = {"max_completion_tokens": 2000}
        # floor=90 (default), throughput=20, overhead=15 → 2000/20+15=115
        timeout = _derive_judge_timeout(judge_cfg, api_cfg)
        self.assertEqual(timeout, 115)

    def test_derive_judge_timeout_with_flat_timeout_judge(self):
        """_derive_judge_timeout ignores legacy timeout_judge in favor of derived."""
        from scripts.judge import _derive_judge_timeout
        api_cfg = {
            "timeout": {
                "judge_floor": 30,
                "judge_throughput": 20,
                "overhead_seconds": 10,
                "max_timeout": 300,
            },
            "timeout_judge": 999,  # legacy key, should be ignored
        }
        judge_cfg = {"max_completion_tokens": 2000}
        timeout = _derive_judge_timeout(judge_cfg, api_cfg)
        # derived: 2000/20+10=110, not 999
        self.assertEqual(timeout, 110)

    def test_build_judge_llm_kwargs_mimo_uses_max_tokens_and_thinking(self):
        from scripts.judge import _build_judge_llm_kwargs

        kwargs = _build_judge_llm_kwargs({
            "model": "mimo-v2.5",
            "temp": 1.0,
            "top_p": 0.95,
            "max_tokens": 2048,
            "thinking": {"type": "enabled"},
        })

        self.assertEqual(kwargs["model"], "mimo-v2.5")
        self.assertEqual(kwargs["temperature"], 1.0)
        self.assertEqual(kwargs["top_p"], 0.95)
        self.assertEqual(kwargs["max_tokens"], 2048)
        self.assertNotIn("max_completion_tokens", kwargs)
        self.assertNotIn("reasoning_mode", kwargs)
        self.assertEqual(kwargs["extra_params"], {"thinking": {"type": "enabled"}})

    def test_build_judge_llm_kwargs_mimo_missing_budget_defaults_to_4096(self):
        from scripts.judge import _build_judge_llm_kwargs

        kwargs = _build_judge_llm_kwargs({
            "model": "mimo-v2.5",
            "thinking": {"type": "enabled"},
        })

        self.assertEqual(kwargs["max_tokens"], 4096)
        self.assertNotIn("max_completion_tokens", kwargs)
        self.assertEqual(kwargs["extra_params"], {"thinking": {"type": "enabled"}})

    def test_build_judge_llm_kwargs_deepseek_preserves_legacy_params(self):
        from scripts.judge import _build_judge_llm_kwargs

        kwargs = _build_judge_llm_kwargs({
            "model": "deepseek-v4-flash",
            "temp": 0.0,
            "top_p": 1.0,
            "max_completion_tokens": 8000,
            "reasoning_mode": "high",
        })

        self.assertEqual(kwargs["model"], "deepseek-v4-flash")
        self.assertEqual(kwargs["max_completion_tokens"], 8000)
        self.assertNotIn("max_tokens", kwargs)
        self.assertEqual(kwargs["reasoning_mode"], "high")
        self.assertNotIn("extra_params", kwargs)

    def test_merge_judge_call_config_stage_inherits_mimo_defaults(self):
        from scripts.judge import _merge_judge_call_config

        merged = _merge_judge_call_config(
            {
                "model": "mimo-v2.5",
                "temp": 1.0,
                "top_p": 0.95,
                "max_tokens": 2048,
                "thinking": {"type": "enabled"},
                "stage1": {"temp": 0.9},
                "stage2": {"max_tokens": 1024},
            },
            {"temp": 0.9},
        )

        self.assertEqual(merged["model"], "mimo-v2.5")
        self.assertEqual(merged["temp"], 0.9)
        self.assertEqual(merged["top_p"], 0.95)
        self.assertEqual(merged["max_tokens"], 2048)
        self.assertEqual(merged["thinking"], {"type": "enabled"})
        self.assertNotIn("stage1", merged)
        self.assertNotIn("stage2", merged)

    def test_derive_judge_timeout_uses_max_tokens_and_thinking_multiplier(self):
        from scripts.judge import _derive_judge_timeout

        api_cfg = {
            "timeout": {
                "judge_floor": 30,
                "judge_throughput": 20,
                "overhead_seconds": 10,
                "max_timeout": 300,
            },
        }
        judge_cfg = {"max_tokens": 2048, "thinking": {"type": "enabled"}}
        timeout = _derive_judge_timeout(judge_cfg, api_cfg)

        # int((2048 / 20 + 10) * 1.5) == 168
        self.assertEqual(timeout, 168)

    def test_judge_single_stage_passes_mimo_params_to_api_client(self):
        from unittest import mock
        from scripts.judge import judge_single_stage

        captured = {}

        def _fake_call(**kwargs):
            captured.update(kwargs)
            return {"success": True, "content": "ok", "reasoning_content": None,
                    "usage": {}, "error": None, "elapsed": 0.01}

        with mock.patch("scripts.judge.call_llm_with_retry", side_effect=_fake_call):
            result = judge_single_stage(
                "query",
                [{"label": "A", "content": "answer"}],
                "general",
                config={"api": {"primary": {"timeout": {"judge_floor": 2}}}},
                judge_config={
                    "model": "mimo-v2.5",
                    "temp": 1.0,
                    "top_p": 0.95,
                    "max_tokens": 2048,
                    "thinking": {"type": "enabled"},
                },
            )

        self.assertTrue(result["success"])
        self.assertEqual(captured["model"], "mimo-v2.5")
        self.assertEqual(captured["temperature"], 1.0)
        self.assertEqual(captured["top_p"], 0.95)
        self.assertEqual(captured["max_tokens"], 2048)
        self.assertEqual(captured["extra_params"], {"thinking": {"type": "enabled"}})
        self.assertNotIn("max_completion_tokens", captured)
        self.assertNotIn("reasoning_mode", captured)

    def test_judge_single_stage_preserves_deepseek_params_to_api_client(self):
        from unittest import mock
        from scripts.judge import judge_single_stage

        captured = {}

        def _fake_call(**kwargs):
            captured.update(kwargs)
            return {"success": True, "content": "ok", "reasoning_content": "reasoning",
                    "usage": {}, "error": None, "elapsed": 0.01}

        with mock.patch("scripts.judge.call_llm_with_retry", side_effect=_fake_call):
            result = judge_single_stage(
                "query",
                [{"label": "A", "content": "answer"}],
                "general",
                config={"api": {"primary": {"timeout": {"judge_floor": 2}}}},
                judge_config={
                    "model": "deepseek-v4-flash",
                    "temp": 0.0,
                    "top_p": 1.0,
                    "max_completion_tokens": 8000,
                    "reasoning_mode": "high",
                },
            )

        self.assertTrue(result["success"])
        self.assertEqual(captured["model"], "deepseek-v4-flash")
        self.assertEqual(captured["max_completion_tokens"], 8000)
        self.assertEqual(captured["reasoning_mode"], "high")
        self.assertNotIn("max_tokens", captured)
        self.assertNotIn("extra_params", captured)

    # ------------------------------------------------------------------ #
    # Two-stage judge token reduction tests
    # ------------------------------------------------------------------ #

    def test_two_stage_no_raw_responses_omits_responses_section(self):
        """stage2_include_raw_responses=false omits responses from stage2 prompt."""
        from unittest import mock
        from scripts.judge import judge_two_stage

        captured_prompts = []
        call_count = [0]

        def _fake_call(**kwargs):
            call_count[0] += 1
            prompt = kwargs.get("prompt", "")
            captured_prompts.append(prompt)
            if call_count[0] == 1:
                return {"success": True, "content": "Stage 1 analysis here",
                        "reasoning_content": None, "usage": {}, "error": None}
            return {"success": True, "content": "Final answer",
                    "reasoning_content": None, "usage": {}, "error": None}

        responses = [
            {"label": "Model-A", "cleaned_content": "Answer from model A"},
            {"label": "Model-B", "cleaned_content": "Answer from model B"},
        ]

        with mock.patch("scripts.judge.call_llm_with_retry", side_effect=_fake_call):
            result = judge_two_stage(
                "What is 2+2?",
                responses,
                "bugfix",
                config={"api": {"primary": {"timeout": {"judge_floor": 2}}}},
                judge_config={
                    "model": "mimo-v2.5",
                    "temp": 1.0,
                    "top_p": 0.95,
                    "max_tokens": 2048,
                    "thinking": {"type": "enabled"},
                    "stage1": {},
                    "stage2": {},
                    "stage2_include_raw_responses": False,
                },
            )

        self.assertTrue(result["success"])
        self.assertEqual(len(captured_prompts), 2)
        # Stage 1 prompt should contain responses
        self.assertIn("Model-A", captured_prompts[0])
        self.assertIn("Model-B", captured_prompts[0])
        # Stage 2 prompt should NOT contain raw responses
        self.assertNotIn("Model-A", captured_prompts[1])
        self.assertNotIn("Model-B", captured_prompts[1])
        # Stage 2 prompt should contain the stage 1 analysis
        self.assertIn("Stage 1 analysis here", captured_prompts[1])

    def test_two_stage_with_raw_responses_includes_when_explicit(self):
        """stage2_include_raw_responses=true includes responses in stage2; absent (default false) excludes them."""
        from unittest import mock
        from scripts.judge import judge_two_stage

        captured_prompts = []
        call_count = [0]

        def _fake_call(**kwargs):
            call_count[0] += 1
            prompt = kwargs.get("prompt", "")
            captured_prompts.append(prompt)
            if call_count[0] == 1:
                return {"success": True, "content": "analysis",
                        "reasoning_content": None, "usage": {}, "error": None}
            return {"success": True, "content": "final",
                    "reasoning_content": None, "usage": {}, "error": None}

        responses = [{"label": "X", "cleaned_content": "content X"}]

        # Test with explicit true
        with mock.patch("scripts.judge.call_llm_with_retry", side_effect=_fake_call):
            result = judge_two_stage(
                "test query", responses, "bugfix",
                config={"api": {"primary": {"timeout": {"judge_floor": 2}}}},
                judge_config={
                    "model": "mimo-v2.5", "temp": 1.0, "top_p": 0.95,
                    "max_tokens": 2048, "thinking": {"type": "enabled"},
                    "stage1": {}, "stage2": {},
                    "stage2_include_raw_responses": True,
                },
            )

        self.assertTrue(result["success"])
        # Both prompts should contain responses
        self.assertIn("content X", captured_prompts[0])
        self.assertIn("content X", captured_prompts[1])

        # Also test with key absent (defaults to false — optimization)
        captured_prompts.clear()
        call_count[0] = 0

        with mock.patch("scripts.judge.call_llm_with_retry", side_effect=_fake_call):
            result = judge_two_stage(
                "test query", responses, "bugfix",
                config={"api": {"primary": {"timeout": {"judge_floor": 2}}}},
                judge_config={
                    "model": "mimo-v2.5", "temp": 1.0, "top_p": 0.95,
                    "max_tokens": 2048, "thinking": {"type": "enabled"},
                    "stage1": {}, "stage2": {},
                },
            )

        self.assertTrue(result["success"])
        # Stage 1 should have responses
        self.assertIn("content X", captured_prompts[0])
        # Stage 2 should NOT have responses (default false)
        self.assertNotIn("content X", captured_prompts[1])

    def test_two_stage_metadata_fields_populated(self):
        """Stage input char metadata is populated in result dict."""
        from unittest import mock
        from scripts.judge import judge_two_stage

        def _fake_call(**kwargs):
            return {"success": True, "content": "analysis result",
                    "reasoning_content": None, "usage": {}, "error": None}

        responses = [
            {"label": "A", "cleaned_content": "Short answer"},
            {"label": "B", "cleaned_content": "Another short answer"},
        ]

        with mock.patch("scripts.judge.call_llm_with_retry", side_effect=_fake_call):
            result = judge_two_stage(
                "test query", responses, "bugfix",
                config={"api": {"primary": {"timeout": {"judge_floor": 2}}}},
                judge_config={
                    "model": "mimo-v2.5", "temp": 1.0, "top_p": 0.95,
                    "max_tokens": 2048, "thinking": {"type": "enabled"},
                    "stage1": {}, "stage2": {},
                },
            )

        self.assertIn("stage1_input_chars", result)
        self.assertIn("stage2_input_chars", result)
        self.assertIsInstance(result["stage1_input_chars"], int)
        self.assertIsInstance(result["stage2_input_chars"], int)
        self.assertGreater(result["stage1_input_chars"], 0)
        self.assertGreater(result["stage2_input_chars"], 0)
        # Check truncation metadata (no truncation in this test)
        self.assertIn("panel_response_truncated_count", result)
        self.assertIn("panel_response_truncated_chars", result)
        self.assertIn("max_panel_response_chars", result)
        self.assertIn("stage2_include_raw_responses", result)
        self.assertEqual(result["panel_response_truncated_count"], 0)
        self.assertEqual(result["panel_response_truncated_chars"], 0)
        self.assertIsNone(result["max_panel_response_chars"])
        self.assertFalse(result["stage2_include_raw_responses"])

    def test_max_panel_response_chars_truncation_two_stage(self):
        """max_panel_response_chars truncates responses in both stage prompts."""
        from unittest import mock
        from scripts.judge import judge_two_stage

        captured_prompts = []
        call_count = [0]

        def _fake_call(**kwargs):
            call_count[0] += 1
            captured_prompts.append(kwargs.get("prompt", ""))
            if call_count[0] == 1:
                return {"success": True, "content": "analysis",
                        "reasoning_content": None, "usage": {}, "error": None}
            return {"success": True, "content": "final",
                    "reasoning_content": None, "usage": {}, "error": None}

        long_content = "A" * 5000
        responses = [{"label": "Long", "cleaned_content": long_content}]

        with mock.patch("scripts.judge.call_llm_with_retry", side_effect=_fake_call):
            result = judge_two_stage(
                "test", responses, "bugfix",
                config={"api": {"primary": {"timeout": {"judge_floor": 2}}}},
                judge_config={
                    "model": "mimo-v2.5", "temp": 1.0, "top_p": 0.95,
                    "max_tokens": 2048, "thinking": {"type": "enabled"},
                    "stage1": {}, "stage2": {},
                    "max_panel_response_chars": 1000,
                },
            )

        self.assertTrue(result["success"])
        # Stage 1 prompt should have truncated content
        self.assertIn("[truncated to first 1000 chars]", captured_prompts[0])
        self.assertNotIn("A" * 1001, captured_prompts[0])
        # Check truncation metadata
        self.assertEqual(result["panel_response_truncated_count"], 1)
        self.assertGreaterEqual(result["panel_response_truncated_chars"], 4000)
        self.assertEqual(result["max_panel_response_chars"], 1000)

    def test_two_stage_truncation_stats_no_truncation(self):
        """When no truncation occurs, metadata shows zero truncation."""
        from unittest import mock
        from scripts.judge import judge_two_stage

        def _fake_call(**kwargs):
            return {"success": True, "content": "analysis",
                    "reasoning_content": None, "usage": {}, "error": None}

        with mock.patch("scripts.judge.call_llm_with_retry", side_effect=_fake_call):
            result = judge_two_stage(
                "query", [{"label": "A", "cleaned_content": "short"}], "bugfix",
                config={"api": {"primary": {"timeout": {"judge_floor": 2}}}},
                judge_config={
                    "model": "mimo-v2.5", "temp": 1.0, "top_p": 0.95,
                    "max_tokens": 2048, "thinking": {"type": "enabled"},
                    "stage1": {}, "stage2": {},
                    "max_panel_response_chars": 5000,
                },
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["panel_response_truncated_count"], 0)
        self.assertEqual(result["panel_response_truncated_chars"], 0)
        self.assertEqual(result["max_panel_response_chars"], 5000)

    def test_two_stage_evidence_bundle_in_stage1_prompt(self):
        """Stage 1 prompt includes evidence bundle instructions in the user prompt."""
        from unittest import mock
        from scripts.judge import judge_two_stage

        captured_prompts = []

        def _fake_call(**kwargs):
            captured_prompts.append(kwargs.get("prompt", ""))
            return {"success": True, "content": "analysis",
                    "reasoning_content": None, "usage": {}, "error": None}

        with mock.patch("scripts.judge.call_llm_with_retry", side_effect=_fake_call):
            judge_two_stage(
                "test", [{"label": "A", "cleaned_content": "test content"}], "bugfix",
                config={"api": {"primary": {"timeout": {"judge_floor": 2}}}},
                judge_config={
                    "model": "mimo-v2.5", "temp": 1.0, "top_p": 0.95,
                    "max_tokens": 2048, "thinking": {"type": "enabled"},
                    "stage1": {}, "stage2": {},
                },
            )

        self.assertGreater(len(captured_prompts), 0)
        stage1_prompt = captured_prompts[0]
        # Evidence bundle instructions should be in the stage 1 user prompt
        self.assertIn("compact evidence bundle", stage1_prompt.lower())
        self.assertIn("produce a compact evidence bundle", stage1_prompt.lower())

    def test_two_stage_stage2_excludes_responses_by_default(self):
        """Without stage2_include_raw_responses, stage2 prompt omits raw responses."""
        from unittest import mock
        from scripts.judge import judge_two_stage

        captured_prompts = []
        call_count = [0]

        def _fake_call(**kwargs):
            call_count[0] += 1
            captured_prompts.append(kwargs.get("prompt", ""))
            if call_count[0] == 1:
                return {"success": True, "content": "analysis",
                        "reasoning_content": None, "usage": {}, "error": None}
            return {"success": True, "content": "final",
                    "reasoning_content": None, "usage": {}, "error": None}

        responses = [
            {"label": "A", "cleaned_content": "Some long analysis response here"},
            {"label": "B", "cleaned_content": "Another model analysis here"},
        ]

        with mock.patch("scripts.judge.call_llm_with_retry", side_effect=_fake_call):
            result = judge_two_stage(
                "test query", responses, "bugfix",
                config={"api": {"primary": {"timeout": {"judge_floor": 2}}}},
                judge_config={
                    "model": "mimo-v2.5", "temp": 1.0, "top_p": 0.95,
                    "max_tokens": 2048, "thinking": {"type": "enabled"},
                    "stage1": {}, "stage2": {},
                },
            )

        self.assertTrue(result["success"])
        self.assertEqual(len(captured_prompts), 2)
        # Stage 1 has raw responses
        self.assertIn("Some long analysis response here", captured_prompts[0])
        # Stage 2 does NOT have raw responses by default
        self.assertNotIn("Some long analysis response here", captured_prompts[1])
        # Stage 2 should still have the analysis
        self.assertIn("analysis", captured_prompts[1])
        # Metadata confirms default
        self.assertFalse(result["stage2_include_raw_responses"])


class TestSkillHandler(unittest.TestCase):
    """Test llm_fusion/skill_handler.py"""

    def test_handle_fusion_trigger(self):
        from scripts.skill_handler import handle_fusion_trigger
        result = handle_fusion_trigger("What is 2+2?", verbose=False)
        self.assertIsInstance(result, str)

    def test_get_skill_manifest(self):
        from scripts.skill_handler import get_skill_manifest
        manifest = get_skill_manifest()
        self.assertEqual(manifest["name"], "llm-fusion")
        self.assertIn("triggers", manifest)


class TestTierResolution(unittest.TestCase):
    """Test resolve_tier_models and tier-aware get_scenario_config."""

    def setUp(self):
        self.tiered_config = {
            "version": "2.0.0",
            "default": {
                "panel": {
                    "model_defaults": {
                        "deepseek-v4-flash": {
                            "temp": 0.75, "top_p": 0.9, "max_completion_tokens": 800,
                        },
                        "mimo-v2.5": {
                            "temps": [0.6, 0.7, 0.8], "top_p": 0.95,
                            "max_tokens": 600, "thinking": {"type": "disabled"},
                        },
                        "minimax-m3": {
                            "temp": 0.85, "top_p": 0.9, "top_k": 40,
                            "max_tokens": 800, "thinking": {"type": "adaptive"},
                        },
                        "qwen3.7-plus": {
                            "temp": 0.8, "top_p": 0.92, "top_k": 20,
                            "reasoning_effort": "high", "max_tokens": 2048,
                        },
                        "deepseek-v4-pro": {
                            "temp": 0.9, "top_p": 0.95,
                            "reasoning_mode": "high", "max_completion_tokens": 2048,
                        },
                    },
                    "tiers": {
                        "low1": [
                            {"name": "deepseek-v4-flash", "count": 1},
                            {"name": "mimo-v2.5", "count": 1},
                        ],
                        "low2": [
                            {"name": "deepseek-v4-flash", "count": 2},
                            {"name": "mimo-v2.5", "count": 2},
                        ],
                        "low3": [
                            {"name": "deepseek-v4-flash", "count": 3},
                            {"name": "mimo-v2.5", "count": 3},
                        ],
                        "medium": [
                            {"name": "deepseek-v4-flash", "count": 1},
                            {"name": "minimax-m3", "count": 1},
                            {"name": "qwen3.7-plus", "count": 1},
                        ],
                        "high": [
                            {"name": "deepseek-v4-pro", "count": 1},
                            {"name": "minimax-m3", "count": 1},
                            {"name": "qwen3.7-plus", "count": 1},
                        ],
                    },
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
            "cleaning": {
                "profiles": {
                    "coding": {"strip_fences": False, "strip_preamble": True, "min_words": 15, "dedup_threshold": 0.70},
                    "general": {"strip_fences": True, "strip_preamble": True, "min_words": 10, "dedup_threshold": 0.85},
                },
            },
            "api": {
                "primary": {
                    "endpoint": "https://opencode.ai/zen/go/v1/chat/completions",
                    "timeout": {
                        "panel_floor": 30,
                        "judge_floor": 60,
                        "panel_throughput": 25,
                        "judge_throughput": 20,
                        "overhead_seconds": 10,
                        "max_timeout": 300,
                    },
                },
            },
            "pipeline": {
                "max_panel_workers": 6, "min_survivors": 2, "graceful_degradation": True,
            },
        }

        self.legacy_config = {
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
            "scenarios": {},
            "cleaning": {"profiles": {}},
            "api": {"primary": {}},
            "pipeline": {},
        }

    # --- resolve_tier_models tests ---

    def test_resolve_low1_tier(self):
        """low1 tier returns 2 models (1 deepseek + 1 mimo)."""
        from scripts.config import resolve_tier_models
        panel_cfg = self.tiered_config["default"]["panel"]
        models = resolve_tier_models(panel_cfg, "low1")
        self.assertEqual(len(models), 2)
        names = [m["name"] for m in models]
        self.assertEqual(names, ["deepseek-v4-flash", "mimo-v2.5"])
        # Counts
        self.assertEqual(models[0]["count"], 1)
        self.assertEqual(models[1]["count"], 1)
        # Defaults merged
        self.assertEqual(models[0]["temp"], 0.75)
        self.assertEqual(models[0]["top_p"], 0.9)
        self.assertEqual(models[1]["top_p"], 0.95)

    def test_resolve_low2_tier(self):
        """low2 tier returns 2 model entries (counts 2 each)."""
        from scripts.config import resolve_tier_models
        panel_cfg = self.tiered_config["default"]["panel"]
        models = resolve_tier_models(panel_cfg, "low2")
        self.assertEqual(len(models), 2)
        # low2 has 2 entries, each with count=2 → expands in dispatch
        names = [m["name"] for m in models]
        self.assertEqual(names, ["deepseek-v4-flash", "mimo-v2.5"])
        self.assertEqual(models[0]["count"], 2)
        self.assertEqual(models[1]["count"], 2)

    def test_resolve_medium_tier(self):
        """medium tier returns 3 entries (1 deepseek + 1 minimax + 1 qwen)."""
        from scripts.config import resolve_tier_models
        panel_cfg = self.tiered_config["default"]["panel"]
        models = resolve_tier_models(panel_cfg, "medium")
        self.assertEqual(len(models), 3)
        names = [m["name"] for m in models]
        self.assertEqual(names, ["deepseek-v4-flash", "minimax-m3", "qwen3.7-plus"])
        self.assertEqual([m["count"] for m in models], [1, 1, 1])
        self.assertEqual(models[1]["top_k"], 40)
        self.assertEqual(models[1]["thinking"]["type"], "adaptive")
        self.assertEqual(models[2]["temp"], 0.8)
        self.assertEqual(models[2]["top_k"], 20)
        self.assertEqual(models[2]["reasoning_effort"], "high")
        self.assertEqual(models[2]["max_tokens"], 2048)

    def test_resolve_high_tier(self):
        """high tier returns 3 entries (1 deepseek-v4-pro + 1 minimax + 1 qwen)."""
        from scripts.config import resolve_tier_models
        panel_cfg = self.tiered_config["default"]["panel"]
        models = resolve_tier_models(panel_cfg, "high")
        self.assertEqual(len(models), 3)
        names = [m["name"] for m in models]
        self.assertEqual(names, ["deepseek-v4-pro", "minimax-m3", "qwen3.7-plus"])
        self.assertEqual([m["count"] for m in models], [1, 1, 1])
        self.assertEqual(models[0]["temp"], 0.9)
        self.assertEqual(models[0]["top_p"], 0.95)
        self.assertEqual(models[0]["reasoning_mode"], "high")
        self.assertEqual(models[0]["max_completion_tokens"], 2048)
        self.assertEqual(models[1]["top_k"], 40)
        self.assertEqual(models[1]["thinking"]["type"], "adaptive")
        self.assertEqual(models[2]["temp"], 0.8)
        self.assertEqual(models[2]["top_k"], 20)
        self.assertEqual(models[2]["reasoning_effort"], "high")

    def test_resolve_unknown_tier_falls_back_to_default(self):
        """Unknown tier name falls back to default tier (medium)."""
        from scripts.config import resolve_tier_models
        panel_cfg = self.tiered_config["default"]["panel"]
        models = resolve_tier_models(panel_cfg, "nonexistent")
        self.assertEqual(len(models), 3)
        self.assertEqual([m["count"] for m in models], [1, 1, 1])
        names = [m["name"] for m in models]
        self.assertEqual(names, ["deepseek-v4-flash", "minimax-m3", "qwen3.7-plus"])

    def test_resolve_legacy_models_fallback(self):
        """No tiers key → TIER_MAP overrides counts on legacy models list."""
        from scripts.config import resolve_tier_models
        panel_cfg = self.legacy_config["default"]["panel"]
        models = resolve_tier_models(panel_cfg, "low2")
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["name"], "deepseek-v4-flash")
        # Count overridden by TIER_MAP["low2"] (2 each)
        self.assertEqual(models[0]["count"], 2)
        self.assertEqual(models[1]["name"], "mimo-v2.5")
        self.assertEqual(models[1]["count"], 2)

    def test_resolve_legacy_models_low1_tier(self):
        """low1 tier with legacy config yields count=1 deepseek + 1 mimo (2 models, 2 calls)."""
        from scripts.config import resolve_tier_models
        panel_cfg = self.legacy_config["default"]["panel"]
        models = resolve_tier_models(panel_cfg, "low1")
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["name"], "deepseek-v4-flash")
        self.assertEqual(models[0]["count"], 1)
        self.assertEqual(models[1]["name"], "mimo-v2.5")
        self.assertEqual(models[1]["count"], 1)

    def test_resolve_empty_panel(self):
        """Empty panel config returns TIER_MAP low2 defaults."""
        from scripts.config import resolve_tier_models
        models = resolve_tier_models({}, "low2")
        # Falls back to TIER_MAP["low2"] defaults
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["name"], "deepseek-v4-flash")
        self.assertEqual(models[0]["count"], 2)

    # --- get_scenario_config with tier ---

    def test_get_scenario_config_with_tier(self):
        """get_scenario_config with tier param returns correct model count."""
        from scripts.config import get_scenario_config
        cfg = get_scenario_config(self.tiered_config, "coding", tier="low1")
        models = cfg["panel"]["models"]
        self.assertEqual(len(models), 2)
        # Scenario override applied
        ds = [m for m in models if m["name"] == "deepseek-v4-flash"][0]
        self.assertEqual(ds["temp"], 0.5)  # overridden by scenario
        self.assertEqual(ds["max_completion_tokens"], 2000)
        self.assertEqual(ds["count"], 1)  # count preserved from tier

    def test_get_scenario_config_legacy(self):
        """Legacy config (no tiers) applies TIER_MAP counts with tier param."""
        from scripts.config import get_scenario_config
        cfg = get_scenario_config(self.legacy_config, "general", tier="low2")
        self.assertIn("panel", cfg)
        models = cfg["panel"]["models"]
        self.assertEqual(len(models), 2)
        # Count overridden by TIER_MAP["low2"] (2 each)
        self.assertEqual(models[0]["count"], 2)
        self.assertEqual(models[1]["count"], 2)

    def test_get_scenario_config_medium_tier(self):
        """Medium tier yields deepseek, minimax, and qwen model entries."""
        from scripts.config import get_scenario_config
        cfg = get_scenario_config(self.tiered_config, "general", tier="medium")
        models = cfg["panel"]["models"]
        names = [m["name"] for m in models]
        self.assertEqual(names, ["deepseek-v4-flash", "minimax-m3", "qwen3.7-plus"])
        self.assertEqual(sum(m["count"] for m in models), 3)

    def test_get_scenario_config_high_tier(self):
        """High tier yields deepseek-v4-pro, minimax, and qwen model entries."""
        from scripts.config import get_scenario_config
        cfg = get_scenario_config(self.tiered_config, "general", tier="high")
        models = cfg["panel"]["models"]
        names = [m["name"] for m in models]
        self.assertEqual(names, ["deepseek-v4-pro", "minimax-m3", "qwen3.7-plus"])
        self.assertEqual(sum(m["count"] for m in models), 3)
        # Check scenario overrides still work via new deepseek-v4-pro alias
        ds = [m for m in models if m["name"] == "deepseek-v4-pro"][0]
        self.assertIsNotNone(ds)


# ---------------------------------------------------------------------------
# Fallback Provider + Rate Limiter tests
# ---------------------------------------------------------------------------


class TestRateLimiterConcurrency(unittest.TestCase):
    """Thread-safety tests for RateLimiter."""

    def test_rate_limiter_concurrent_acquire(self):
        """RateLimiter is safe under concurrent acquisition."""
        from scripts.fallback import RateLimiter
        import threading

        limiter = RateLimiter(rate=1000, burst=5)
        errors = []

        def _worker():
            for _ in range(10):
                try:
                    limiter.acquire(1, block=False)
                except Exception as exc:
                    errors.append(str(exc))

        threads = [threading.Thread(target=_worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Concurrent acquire errors: {errors}")

    def test_rate_limiter_lock_held_during_refill_only(self):
        """RateLimiter does not hold lock while sleeping."""
        from scripts.fallback import RateLimiter

        limiter = RateLimiter(rate=100, burst=1)
        # Exhaust the bucket
        self.assertTrue(limiter.acquire(1, block=False))
        self.assertFalse(limiter.acquire(1, block=False))
        # After refill, should get a token (rate=100 → 100 tokens/s)
        import time
        time.sleep(0.02)  # ~2 tokens should have refilled
        self.assertTrue(limiter.acquire(1, block=False))

    def test_get_rate_limiter_reuses_instance(self):
        """get_rate_limiter returns same instance for same settings."""
        from scripts.fallback import get_rate_limiter

        rl1 = get_rate_limiter(rate=5.0, burst=10)
        rl2 = get_rate_limiter(rate=5.0, burst=10)
        self.assertIs(rl1, rl2, "Same settings should return same instance")

    def test_get_rate_limiter_recreates_on_change(self):
        """get_rate_limiter creates new instance when settings change."""
        from scripts.fallback import get_rate_limiter

        rl1 = get_rate_limiter(rate=5.0, burst=10)
        rl2 = get_rate_limiter(rate=10.0, burst=20)
        self.assertIsNot(rl1, rl2, "Different settings should return new instance")


class TestRetryHelpers(unittest.TestCase):
    """Test _is_retryable_result and _compute_retry_delay."""

    def test_is_retryable_success(self):
        """Success result is not retryable."""
        from scripts.api_client import _is_retryable_result
        self.assertFalse(_is_retryable_result({"success": True, "http_status": 200}))

    def test_is_retryable_401(self):
        """401 is not retryable."""
        from scripts.api_client import _is_retryable_result
        self.assertFalse(_is_retryable_result(
            {"success": False, "http_status": 401}
        ))

    def test_is_retryable_403(self):
        """403 is not retryable."""
        from scripts.api_client import _is_retryable_result
        self.assertFalse(_is_retryable_result(
            {"success": False, "http_status": 403}
        ))

    def test_is_retryable_429(self):
        """429 is retryable."""
        from scripts.api_client import _is_retryable_result
        self.assertTrue(_is_retryable_result(
            {"success": False, "http_status": 429}
        ))

    def test_is_retryable_500(self):
        """500 is retryable."""
        from scripts.api_client import _is_retryable_result
        self.assertTrue(_is_retryable_result(
            {"success": False, "http_status": 500}
        ))

    def test_is_retryable_502(self):
        """502 is retryable."""
        from scripts.api_client import _is_retryable_result
        self.assertTrue(_is_retryable_result(
            {"success": False, "http_status": 502}
        ))

    def test_is_retryable_503(self):
        """503 is retryable."""
        from scripts.api_client import _is_retryable_result
        self.assertTrue(_is_retryable_result(
            {"success": False, "http_status": 503}
        ))

    def test_is_retryable_504(self):
        """504 is retryable."""
        from scripts.api_client import _is_retryable_result
        self.assertTrue(_is_retryable_result(
            {"success": False, "http_status": 504}
        ))

    def test_is_retryable_400(self):
        """400 is not retryable by default."""
        from scripts.api_client import _is_retryable_result
        self.assertFalse(_is_retryable_result(
            {"success": False, "http_status": 400}
        ))

    def test_is_retryable_404(self):
        """404 is not retryable by default."""
        from scripts.api_client import _is_retryable_result
        self.assertFalse(_is_retryable_result(
            {"success": False, "http_status": 404}
        ))

    def test_is_retryable_none_status(self):
        """None http_status (transport failure) is retryable."""
        from scripts.api_client import _is_retryable_result
        self.assertTrue(_is_retryable_result(
            {"success": False, "http_status": None, "error": "Timeout"}
        ))

    def test_is_retryable_custom_sets(self):
        """Custom retryable/non-retryable sets override defaults."""
        from scripts.api_client import _is_retryable_result
        # 418 not in defaults, but added to retryable → is retryable
        self.assertTrue(_is_retryable_result(
            {"success": False, "http_status": 418},
            retryable_statuses={418},
            non_retryable_statuses=set(),
        ))

    def test_compute_retry_delay_backoff(self):
        """_compute_retry_delay computes exponential backoff for retryable statuses."""
        from scripts.api_client import _compute_retry_delay
        retry_policy = {
            "backoff_enabled": True,
            "base_delay_seconds": 1.0,
            "max_delay_seconds": 30.0,
            "jitter_ratio": 0.0,  # no jitter for deterministic test
            "retryable_statuses": {429, 500},
        }
        result = {"http_status": 429}
        # attempt 0: 1 * 2**0 = 1.0
        delay = _compute_retry_delay(0, (1, 3), retry_policy, result)
        self.assertAlmostEqual(delay, 1.0, places=5)
        # attempt 1: 1 * 2**1 = 2.0
        delay = _compute_retry_delay(1, (1, 3), retry_policy, result)
        self.assertAlmostEqual(delay, 2.0, places=5)
        # attempt 2: 1 * 2**2 = 4.0
        delay = _compute_retry_delay(2, (1, 3), retry_policy, result)
        self.assertAlmostEqual(delay, 4.0, places=5)

    def test_compute_retry_delay_jitter(self):
        """_compute_retry_delay applies jitter when random_func is injected."""
        from scripts.api_client import _compute_retry_delay
        retry_policy = {
            "backoff_enabled": True,
            "base_delay_seconds": 2.0,
            "max_delay_seconds": 30.0,
            "jitter_ratio": 0.25,
            "retryable_statuses": {429},
        }
        # Inject a deterministic random that always returns +0.1
        delay = _compute_retry_delay(
            0, (1, 3), retry_policy, {"http_status": 429},
            random_func=lambda a, b: 0.1,
        )
        # base=2, attempt=0 → 2.0 * (1 + 0.1) = 2.2
        self.assertAlmostEqual(delay, 2.2, places=5)

    def test_compute_retry_delay_capped(self):
        """_compute_retry_delay caps at max_delay_seconds."""
        from scripts.api_client import _compute_retry_delay
        retry_policy = {
            "backoff_enabled": True,
            "base_delay_seconds": 10.0,
            "max_delay_seconds": 15.0,
            "jitter_ratio": 0.0,
            "retryable_statuses": {429},
        }
        # attempt 3: 10 * 2**3 = 80, capped at 15
        delay = _compute_retry_delay(
            3, (1, 3), retry_policy, {"http_status": 429},
        )
        self.assertAlmostEqual(delay, 15.0, places=5)

    def test_compute_retry_delay_compatibility(self):
        """_compute_retry_delay uses delays list when backoff is disabled."""
        from scripts.api_client import _compute_retry_delay
        retry_policy = {
            "backoff_enabled": False,
            "base_delay_seconds": 1.0,
            "max_delay_seconds": 30.0,
            "jitter_ratio": 0.0,
            "retryable_statuses": {429},
        }
        # Should use delays list
        delay = _compute_retry_delay(0, (2, 4), retry_policy, {"http_status": 429})
        self.assertAlmostEqual(delay, 2.0, places=5)
        delay = _compute_retry_delay(1, (2, 4), retry_policy, {"http_status": 429})
        self.assertAlmostEqual(delay, 4.0, places=5)

    def test_compute_retry_delay_non_retryable_path(self):
        """_compute_retry_delay uses delays list for non-retryable statuses."""
        from scripts.api_client import _compute_retry_delay
        retry_policy = {
            "backoff_enabled": True,
            "base_delay_seconds": 1.0,
            "max_delay_seconds": 30.0,
            "jitter_ratio": 0.0,
            "retryable_statuses": {429, 500},
        }
        # 400 is not in retryable set → use delays list
        delay = _compute_retry_delay(
            0, (3, 5), retry_policy, {"http_status": 400},
        )
        self.assertAlmostEqual(delay, 3.0, places=5)


class TestConfigResolvers(unittest.TestCase):
    """Test config resolver helpers."""

    def test_resolve_rate_limit_config_none(self):
        """_resolve_rate_limit_config returns defaults when config is None."""
        from scripts.api_client import _resolve_rate_limit_config
        rl = _resolve_rate_limit_config(None)
        self.assertTrue(rl["enabled"])
        self.assertEqual(rl["requests_per_second"], 10.0)
        self.assertEqual(rl["burst"], 20)

    def test_resolve_rate_limit_config_full(self):
        """_resolve_rate_limit_config reads values from config."""
        from scripts.api_client import _resolve_rate_limit_config
        config = {
            "api": {
                "rate_limit": {
                    "enabled": False,
                    "requests_per_second": 2.5,
                    "burst": 3,
                },
            },
        }
        rl = _resolve_rate_limit_config(config)
        self.assertFalse(rl["enabled"])
        self.assertEqual(rl["requests_per_second"], 2.5)
        self.assertEqual(rl["burst"], 3)

    def test_resolve_fallback_config_none(self):
        """_resolve_fallback_config returns disabled when config is None."""
        from scripts.api_client import _resolve_fallback_config
        fb = _resolve_fallback_config(None)
        self.assertFalse(fb["enabled"])

    def test_resolve_fallback_config_enabled(self):
        """_resolve_fallback_config reads values from config."""
        from scripts.api_client import _resolve_fallback_config
        config = {
            "api": {
                "fallback": {
                    "enabled": True,
                    "provider": "openrouter",
                    "endpoint": "https://test.example.com/v1",
                },
            },
        }
        fb = _resolve_fallback_config(config)
        self.assertTrue(fb["enabled"])
        self.assertEqual(fb["provider"], "openrouter")
        self.assertEqual(fb["endpoint"], "https://test.example.com/v1")

    def test_resolve_retry_policy_defaults(self):
        """_resolve_retry_policy returns defaults with no config."""
        from scripts.api_client import _resolve_retry_policy
        rp = _resolve_retry_policy(None, None)
        self.assertEqual(rp["max_retries"], 2)
        self.assertTrue(rp["backoff_enabled"])

    def test_resolve_retry_policy_from_config(self):
        """_resolve_retry_policy reads from config."""
        from scripts.api_client import _resolve_retry_policy
        config = {
            "api": {
                "primary": {
                    "retry": {
                        "max_retries": 3,
                        "delays_seconds": [1, 2, 4],
                        "retryable_statuses": [429, 500],
                        "non_retryable_statuses": [401, 403],
                        "backoff": {
                            "enabled": False,
                        },
                    },
                },
            },
        }
        rp = _resolve_retry_policy(config, None)
        self.assertEqual(rp["max_retries"], 3)
        self.assertFalse(rp["backoff_enabled"])
        self.assertEqual(rp["delays"], (1, 2, 4))

    def test_resolve_retry_policy_explicit_override(self):
        """_resolve_retry_policy returns explicit retry_policy when provided."""
        from scripts.api_client import _resolve_retry_policy
        explicit = {"max_retries": 0, "backoff_enabled": False}
        rp = _resolve_retry_policy({}, explicit)
        self.assertEqual(rp["max_retries"], 0)
        self.assertEqual(rp["backoff_enabled"], False)


class TestRateLimitedCall(unittest.TestCase):
    """Tests that call_llm applies rate limiting correctly."""

    def test_call_llm_rate_limited(self):
        """call_llm goes through rate limiting by default with a mock."""
        from unittest import mock
        from scripts.api_client import call_llm

        call_count = 0

        def _fake_rl_req(req, timeout=60, rate_limiter=None, enabled=True):
            nonlocal call_count
            call_count += 1
            return 200, b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}', None

        with mock.patch(
            "scripts.fallback.rate_limited_request",
            side_effect=_fake_rl_req,
        ):
            result = call_llm(
                "test",
                endpoint="https://api.example.test/v1/chat/completions",
                api_key="test-key",
                max_tokens=100,
            )

        self.assertTrue(result["success"])
        self.assertEqual(call_count, 1)


class TestStatusAwareRetry(unittest.TestCase):
    """Integration tests for call_llm_with_retry status-aware behavior.

    These tests mock call_llm at the module level to simulate status codes
    without network calls.
    """

    def _make_config(self, retry_cfg=None, fallback_cfg=None):
        """Build a minimal config for testing."""
        config = {
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
                    "retry": {
                        "max_retries": 2,
                        "delays_seconds": [0.01, 0.02],
                        "retryable_statuses": [429, 500, 502, 503, 504],
                        "non_retryable_statuses": [400, 401, 403, 404],
                        "backoff": {"enabled": False},
                    },
                },
            },
        }
        if retry_cfg:
            config["api"]["primary"]["retry"].update(retry_cfg)
        if fallback_cfg:
            config["api"]["fallback"] = fallback_cfg
        return config

    def test_does_not_retry_401(self):
        """call_llm_with_retry does not retry 401."""
        from unittest import mock
        from scripts.api_client import call_llm_with_retry

        call_count = 0

        def _fake_call(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"success": False, "http_status": 401, "error": "HTTP 401"}

        config = self._make_config()
        with mock.patch("scripts.api_client.call_llm", side_effect=_fake_call):
            result = call_llm_with_retry(
                "test", retries=3, delays=(0.01, 0.02), config=config,
            )

        self.assertEqual(call_count, 1, "Should only attempt once for 401")
        self.assertFalse(result["success"])
        self.assertEqual(result.get("retry_stopped_reason"), "non_retryable_status")

    def test_does_not_retry_403(self):
        """call_llm_with_retry does not retry 403."""
        from unittest import mock
        from scripts.api_client import call_llm_with_retry

        call_count = 0

        def _fake_call(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"success": False, "http_status": 403, "error": "HTTP 403"}

        config = self._make_config()
        with mock.patch("scripts.api_client.call_llm", side_effect=_fake_call):
            result = call_llm_with_retry(
                "test", retries=3, delays=(0.01, 0.02), config=config,
            )

        self.assertEqual(call_count, 1)
        self.assertEqual(result.get("retry_stopped_reason"), "non_retryable_status")

    def test_does_not_retry_400(self):
        """call_llm_with_retry does not retry 400 by default."""
        from unittest import mock
        from scripts.api_client import call_llm_with_retry

        call_count = 0

        def _fake_call(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"success": False, "http_status": 400, "error": "HTTP 400"}

        config = self._make_config()
        with mock.patch("scripts.api_client.call_llm", side_effect=_fake_call):
            result = call_llm_with_retry(
                "test", retries=3, delays=(0.01, 0.02), config=config,
            )

        self.assertEqual(call_count, 1)

    def test_does_not_retry_404(self):
        """call_llm_with_retry does not retry 404 by default."""
        from unittest import mock
        from scripts.api_client import call_llm_with_retry

        call_count = 0

        def _fake_call(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"success": False, "http_status": 404, "error": "HTTP 404"}

        config = self._make_config()
        with mock.patch("scripts.api_client.call_llm", side_effect=_fake_call):
            result = call_llm_with_retry(
                "test", retries=3, delays=(0.01, 0.02), config=config,
            )

        self.assertEqual(call_count, 1)

    def test_retries_429_then_succeeds(self):
        """call_llm_with_retry retries 429 then succeeds."""
        from unittest import mock
        from scripts.api_client import call_llm_with_retry

        call_count = 0
        responses = [
            {"success": False, "http_status": 429, "error": "HTTP 429"},
            {"success": False, "http_status": 429, "error": "HTTP 429"},
            {"success": True, "http_status": 200, "content": "ok"},
        ]

        def _fake_call(prompt, **kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            return responses[idx] if idx < len(responses) else responses[-1]

        config = self._make_config()
        with mock.patch("scripts.api_client.call_llm", side_effect=_fake_call), \
             mock.patch("scripts.api_client.time.sleep"):
            result = call_llm_with_retry(
                "test", retries=3, delays=(0.01, 0.02), config=config,
            )

        self.assertEqual(call_count, 3)
        self.assertTrue(result["success"])
        self.assertEqual(result.get("retry_stopped_reason"), "success")

    def test_retries_500(self):
        """call_llm_with_retry retries 500."""
        from unittest import mock
        from scripts.api_client import call_llm_with_retry

        call_count = 0

        def _fake_call(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"success": False, "http_status": 500, "error": "HTTP 500"}
            return {"success": True, "http_status": 200, "content": "ok"}

        config = self._make_config()
        with mock.patch("scripts.api_client.call_llm", side_effect=_fake_call), \
             mock.patch("scripts.api_client.time.sleep"):
            result = call_llm_with_retry(
                "test", retries=3, delays=(0.01, 0.02), config=config,
            )

        self.assertEqual(call_count, 2)
        self.assertTrue(result["success"])

    def test_retries_transport_failure(self):
        """call_llm_with_retry retries transport failures with http_status=None."""
        from unittest import mock
        from scripts.api_client import call_llm_with_retry

        call_count = 0

        def _fake_call(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"success": False, "http_status": None, "error": "URLError: timeout"}
            return {"success": True, "http_status": 200, "content": "ok"}

        config = self._make_config()
        with mock.patch("scripts.api_client.call_llm", side_effect=_fake_call), \
             mock.patch("scripts.api_client.time.sleep"):
            result = call_llm_with_retry(
                "test", retries=3, delays=(0.01, 0.02), config=config,
            )

        self.assertEqual(call_count, 2)
        self.assertTrue(result["success"])

    def test_fallback_disabled(self):
        """Provider fallback is not attempted when disabled."""
        from unittest import mock
        from scripts.api_client import call_llm_with_retry

        call_count = 0

        def _fake_call(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"success": False, "http_status": 429, "error": "HTTP 429"}

        config = self._make_config(
            retry_cfg={"max_retries": 1},
            fallback_cfg={"enabled": False},
        )
        with mock.patch("scripts.api_client.call_llm", side_effect=_fake_call), \
             mock.patch("scripts.api_client.time.sleep"), \
             mock.patch("scripts.api_client._attempt_provider_fallback") as mock_fb:
            result = call_llm_with_retry(
                "test", retries=2, delays=(0.01,), config=config,
            )

        mock_fb.assert_not_called()
        self.assertEqual(call_count, 2)

    def test_fallback_enabled_attempts_after_exhaustion(self):
        """Provider fallback is attempted after primary retries exhausted."""
        from unittest import mock
        from scripts.api_client import call_llm_with_retry

        def _fake_call(prompt, **kwargs):
            return {"success": False, "http_status": 500, "error": "HTTP 500"}

        def _fake_fallback(*args, **kwargs):
            return {
                "success": True,
                "content": "fallback answer",
                "http_status": 200,
                "from_fallback": True,
                "fallback_provider": "openrouter",
            }

        config = self._make_config(
            retry_cfg={"max_retries": 1},
            fallback_cfg={
                "enabled": True,
                "provider": "openrouter",
                "endpoint": "https://test.example.com/v1",
            },
        )
        with mock.patch("scripts.api_client.call_llm", side_effect=_fake_call), \
             mock.patch("scripts.api_client.time.sleep"), \
             mock.patch(
                 "scripts.api_client._attempt_provider_fallback",
                 side_effect=_fake_fallback,
             ) as mock_fb:
            result = call_llm_with_retry(
                "test", retries=2, delays=(0.01,), config=config,
            )

        mock_fb.assert_called_once()
        self.assertTrue(result["success"])
        self.assertTrue(result.get("from_fallback"))
        self.assertEqual(result.get("fallback_provider"), "openrouter")

    def test_fallback_skipped_for_401(self):
        """Provider fallback is not attempted after 401."""
        from unittest import mock
        from scripts.api_client import call_llm_with_retry

        def _fake_call(prompt, **kwargs):
            return {"success": False, "http_status": 401, "error": "HTTP 401"}

        config = self._make_config(
            retry_cfg={"max_retries": 1},
            fallback_cfg={"enabled": True, "provider": "openrouter"},
        )
        with mock.patch("scripts.api_client.call_llm", side_effect=_fake_call), \
             mock.patch("scripts.api_client.time.sleep"), \
             mock.patch(
                 "scripts.api_client._attempt_provider_fallback",
             ) as mock_fb:
            result = call_llm_with_retry(
                "test", retries=2, delays=(0.01,), config=config,
            )

        mock_fb.assert_not_called()
        self.assertFalse(result["success"])
        self.assertTrue(result.get("fallback_skipped"))

    def test_fallback_skipped_for_403(self):
        """Provider fallback is not attempted after 403."""
        from unittest import mock
        from scripts.api_client import call_llm_with_retry

        def _fake_call(prompt, **kwargs):
            return {"success": False, "http_status": 403, "error": "HTTP 403"}

        config = self._make_config(
            retry_cfg={"max_retries": 1},
            fallback_cfg={"enabled": True, "provider": "openrouter"},
        )
        with mock.patch("scripts.api_client.call_llm", side_effect=_fake_call), \
             mock.patch("scripts.api_client.time.sleep"), \
             mock.patch(
                 "scripts.api_client._attempt_provider_fallback",
             ) as mock_fb:
            result = call_llm_with_retry(
                "test", retries=2, delays=(0.01,), config=config,
            )

        mock_fb.assert_not_called()
        self.assertFalse(result["success"])
        self.assertTrue(result.get("fallback_skipped"))


class TestConfigSmokeFallbackRateLimit(unittest.TestCase):
    """Verify the bundled configs contain the expected new keys."""

    def test_bundled_configs_have_rate_limit_keys(self):
        """Both bundled configs have api.rate_limit with expected keys."""
        from scripts.config import load_config

        expected_keys = {"enabled", "requests_per_second", "burst"}

        for path in (
            "skills/llm-fusion/assets/fusion_config.yaml",
            "skills/llm-fusion/assets/fusion_config.yaml.example",
        ):
            cfg = load_config(path)
            rl = cfg.get("api", {}).get("rate_limit", {})
            self.assertTrue(rl, f"{path} is missing api.rate_limit")
            self.assertIn("enabled", rl, f"{path}")
            self.assertIn("requests_per_second", rl, f"{path}")
            self.assertIn("burst", rl, f"{path}")
            self.assertEqual(set(rl.keys()), expected_keys,
                             f"{path} rate_limit keys mismatch")

    def test_bundled_configs_have_fallback_keys(self):
        """Both bundled configs have api.fallback with expected keys."""
        from scripts.config import load_config

        expected_keys = {"enabled", "provider", "endpoint"}

        for path in (
            "skills/llm-fusion/assets/fusion_config.yaml",
            "skills/llm-fusion/assets/fusion_config.yaml.example",
        ):
            cfg = load_config(path)
            fb = cfg.get("api", {}).get("fallback", {})
            self.assertTrue(fb, f"{path} is missing api.fallback")
            self.assertEqual(set(fb.keys()), expected_keys,
                             f"{path} fallback keys mismatch")
            self.assertFalse(fb["enabled"],
                             f"{path} fallback should be disabled by default")

    def test_bundled_configs_have_retry_status_keys(self):
        """Both bundled configs have retryable/non-retryable status lists."""
        from scripts.config import load_config

        for path in (
            "skills/llm-fusion/assets/fusion_config.yaml",
            "skills/llm-fusion/assets/fusion_config.yaml.example",
        ):
            cfg = load_config(path)
            retry = cfg.get("api", {}).get("primary", {}).get("retry", {})
            self.assertIn("retryable_statuses", retry, f"{path}")
            self.assertIn("non_retryable_statuses", retry, f"{path}")
            self.assertEqual(set(retry["retryable_statuses"]), {429, 500, 502, 503, 504},
                             f"{path}")
            self.assertEqual(set(retry["non_retryable_statuses"]), {400, 401, 403, 404},
                             f"{path}")

    def test_bundled_configs_have_backoff_keys(self):
        """Both bundled configs have api.primary.retry.backoff sub-section."""
        from scripts.config import load_config

        expected_keys = {"enabled", "base_delay_seconds",
                         "max_delay_seconds", "jitter_ratio"}

        for path in (
            "skills/llm-fusion/assets/fusion_config.yaml",
            "skills/llm-fusion/assets/fusion_config.yaml.example",
        ):
            cfg = load_config(path)
            backoff = cfg.get("api", {}).get("primary", {}).get("retry", {}).get("backoff", {})
            self.assertTrue(backoff, f"{path} is missing backoff block")
            self.assertEqual(set(backoff.keys()), expected_keys,
                             f"{path} backoff keys mismatch")
            self.assertTrue(backoff["enabled"],
                            f"{path} backoff should be enabled by default")


if __name__ == "__main__":
    unittest.main()
