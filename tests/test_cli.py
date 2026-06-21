"""Tests for the llm-fusion CLI."""

import os
import sys
import json
import subprocess
import unittest


class TestCLI(unittest.TestCase):
    """Test the llm-fusion CLI via subprocess."""

    def _run(self, *args):
        """Run llm_fusion.cli as a module and return (returncode, stdout, stderr)."""
        cmd = [sys.executable, "-m", "llm_fusion"] + list(args)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return proc.returncode, proc.stdout, proc.stderr

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
        import llm_fusion
        import llm_fusion.cli
        self.assertTrue(callable(llm_fusion.cli.main))
