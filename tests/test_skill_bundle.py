"""Tests for the llm-fusion source structure.

Validates that the tap-discoverable llm-fusion/ subdirectory satisfies
the Agent Skills spec when installed as a skill bundle.
"""

import os
import sys
import json
import subprocess
import tempfile
import unittest


SKILL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "skills", "llm-fusion")
)


class TestSkillBundle(unittest.TestCase):
    """Validate the source structure as a skill bundle."""

    def test_tap_discovery_layout(self):
        """Hermes GitHub taps discover skills as top-level dirs containing SKILL.md."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        tap_skill_path = os.path.join(repo_root, "skills", "llm-fusion", "SKILL.md")
        self.assertTrue(os.path.isfile(tap_skill_path),
                        "tap installs require llm-fusion/SKILL.md, not only root SKILL.md")

    def test_skill_dir_exists(self):
        """The skill directory must exist."""
        self.assertTrue(os.path.isdir(SKILL_DIR), f"Skill dir not found: {SKILL_DIR}")

    def test_skill_dir_name_matches(self):
        """Directory name should be 'llm-fusion'."""
        self.assertEqual(os.path.basename(SKILL_DIR), "llm-fusion")

    def test_skill_md_exists(self):
        """SKILL.md must exist."""
        path = os.path.join(SKILL_DIR, "SKILL.md")
        self.assertTrue(os.path.isfile(path), f"SKILL.md not found at {path}")

    def test_skill_md_frontmatter(self):
        """Validate SKILL.md YAML frontmatter."""
        path = os.path.join(SKILL_DIR, "SKILL.md")
        with open(path) as fh:
            text = fh.read()
        self.assertTrue(text.startswith("---"), "SKILL.md must start with YAML frontmatter")
        parts = text.split("---", 2)
        self.assertGreaterEqual(len(parts), 3, "SKILL.md must have closing ---")
        data = {}
        for line in parts[1].splitlines():
            if ":" in line and not line.startswith(" "):
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()
        self.assertEqual(data.get("name"), "llm-fusion")
        self.assertTrue(data.get("description"), "description must be non-empty")
        # No secrets in frontmatter
        front_str = parts[1].lower()
        self.assertNotIn("api_key:", front_str)
        self.assertNotIn("password:", front_str)
        self.assertNotIn("secret:", front_str)

    def test_scripts_package_exists(self):
        """The scripts/ package must exist with __init__.py."""
        path = os.path.join(SKILL_DIR, "scripts", "__init__.py")
        self.assertTrue(os.path.isfile(path), f"scripts/__init__.py not found at {path}")

    def test_cli_module_runnable(self):
        """The CLI module must be runnable from a temp cwd."""
        env = os.environ.copy()
        env["PYTHONPATH"] = SKILL_DIR + os.pathsep + env.get("PYTHONPATH", "")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, "-m", "scripts", "--dry-run", "--query", "hello"],
                capture_output=True, text=True, timeout=10,
                cwd=tmpdir, env=env,
            )
            self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")
            data = json.loads(result.stdout)
            self.assertTrue(data["ok"])
            self.assertEqual(data["query"], "hello")

    def test_assets_exist(self):
        """The assets directory should have the example config."""
        assets_dir = os.path.join(SKILL_DIR, "assets")
        self.assertTrue(os.path.isdir(assets_dir))
        self.assertTrue(os.path.isfile(os.path.join(assets_dir, "fusion_config.yaml.example")))
