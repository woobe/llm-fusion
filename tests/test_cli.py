"""Tests for the llm-fusion CLI."""

import os
import sys
import json
import unittest


class TestCLI(unittest.TestCase):
    """Test the llm-fusion CLI via direct calls."""

    def _run(self, *args):
        """Run scripts.cli.main with args and return (returncode, stdout, stderr)."""
        from io import StringIO
        from scripts.cli import main

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

    def test_version(self):
        """--version should print version and exit 0."""
        rc, out, _ = self._run("--version")
        self.assertEqual(rc, 0)
        self.assertIn("llm-fusion", out.lower())

    def test_dry_run_basic(self):
        """--dry-run --query should print JSON and exit 0."""
        rc, out, _ = self._run("--dry-run", "--query", "What is 2+2?")
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertTrue(data["ok"])
        self.assertEqual(data["query"], "What is 2+2?")

    def test_dry_run_with_config(self):
        """--dry-run --config should show the config path in JSON."""
        rc, out, _ = self._run("--dry-run", "--config", "/tmp/fake.yaml", "--query", "test")
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual(data["config"], "/tmp/fake.yaml")

    def test_dry_run_output_dir(self):
        """--output-dir should appear in dry-run JSON."""
        rc, out, _ = self._run("--dry-run", "--output-dir", "/tmp/out", "--query", "test")
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual(data["output_dir"], "/tmp/out")

    def test_dry_run_verbose(self):
        """--verbose should appear in dry-run JSON."""
        rc, out, _ = self._run("--dry-run", "--verbose", "--query", "test")
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertTrue(data["verbose"])

    def test_query_required(self):
        """--query is required when not in dry-run mode."""
        rc, out, err = self._run()
        self.assertNotEqual(rc, 0)
        # Should get a parse error about --query
        self.assertTrue(rc != 0 or "query" in err.lower() or "query" in out.lower())

    def test_import(self):
        """Can import the package and cli module."""
        import scripts
        import scripts.cli
        self.assertTrue(callable(scripts.cli.main))
