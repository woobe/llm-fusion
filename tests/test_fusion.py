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
                    "timeout_panel": 30,
                    "timeout_judge": 60,
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
        fail_fast_config = {"api": {"primary": {"endpoint": "http://127.0.0.1:1/nonexistent", "timeout_judge": 2}}}
        result = judge_single_stage(
            "test", [{"label": "A", "content": "hello"}],
            "general", config=fail_fast_config,
            judge_config={"model": "deepseek-v4-flash"},
        )
        self.assertIn("success", result)

    def test_judge_two_stage_no_api(self):
        from scripts.judge import judge_two_stage
        fail_fast_config = {"api": {"primary": {"endpoint": "http://127.0.0.1:1/nonexistent", "timeout_judge": 2}}}
        result = judge_two_stage(
            "test", [{"label": "A", "content": "hello"}],
            "general", config=fail_fast_config,
            judge_config={"model": "deepseek-v4-flash", "stage1": {}, "stage2": {}},
        )
        self.assertIn("success", result)
        self.assertIn("stage1", result)
        self.assertIn("stage2", result)


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


if __name__ == "__main__":
    unittest.main()
