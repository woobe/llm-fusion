"""Package and import tests for llm-fusion."""

import unittest


class TestPackageImports(unittest.TestCase):
    """Verify all expected imports work."""

    def test_package_import(self):
        """Top-level package import should work."""
        import llm_fusion
        self.assertTrue(hasattr(llm_fusion, "__version__"))
        self.assertTrue(hasattr(llm_fusion, "__author__"))

    def test_pipeline_import(self):
        """from llm_fusion.pipeline import run_pipeline should work."""
        from llm_fusion.pipeline import run_pipeline
        self.assertTrue(callable(run_pipeline))

    def test_config_import(self):
        """Config functions should be importable."""
        from llm_fusion.config import load_config, get_scenario_config
        self.assertTrue(callable(load_config))
        self.assertTrue(callable(get_scenario_config))

    def test_cli_import(self):
        """CLI module should be importable and have main()."""
        from llm_fusion.cli import main, build_parser
        self.assertTrue(callable(main))
        parser = build_parser()
        self.assertIsNotNone(parser)

    def test_classifier_import(self):
        """Classifier functions should be importable."""
        from llm_fusion.classifier import classify_query, CONCISENESS_SUFFIXES
        self.assertTrue(callable(classify_query))
        self.assertIn("general", CONCISENESS_SUFFIXES)

    def test_api_client_import(self):
        """API client functions should be importable."""
        from llm_fusion.api_client import call_llm, call_llm_with_retry, read_api_key
        self.assertTrue(callable(call_llm))
        self.assertTrue(callable(call_llm_with_retry))
        self.assertTrue(callable(read_api_key))

    def test_cleaner_import(self):
        """Cleaner functions should be importable."""
        from llm_fusion.cleaner import clean_response, dedup_responses, clean_panel_responses
        self.assertTrue(callable(clean_response))
        self.assertTrue(callable(dedup_responses))

    def test_output_import(self):
        """Output functions should be importable."""
        from llm_fusion.output import format_for_chat, save_output
        self.assertTrue(callable(format_for_chat))
        self.assertTrue(callable(save_output))

    def test_fallback_import(self):
        """Fallback module should be importable."""
        from llm_fusion.fallback import RateLimiter, call_with_fallback
        self.assertTrue(callable(call_with_fallback))

    def test_judge_import(self):
        """Judge functions should be importable."""
        from llm_fusion.judge import judge_single_stage, judge_two_stage
        self.assertTrue(callable(judge_single_stage))
        self.assertTrue(callable(judge_two_stage))

    def test_panel_import(self):
        """Panel dispatch should be importable."""
        from llm_fusion.panel import dispatch_panel
        self.assertTrue(callable(dispatch_panel))

    def test_skill_handler_import(self):
        """Skill handler functions should be importable."""
        from llm_fusion.skill_handler import handle_fusion_trigger, get_skill_manifest
        self.assertTrue(callable(handle_fusion_trigger))
        self.assertTrue(callable(get_skill_manifest))
