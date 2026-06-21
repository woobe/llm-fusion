"""Tests for the Agent Skills bundle structure.

Validates that .agents/skills/llm-fusion/ satisfies the Agent Skills spec.
"""

import os
import sys
import json
import subprocess
import tempfile
import unittest


SKILL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".agents", "skills", "llm-fusion")
)


class TestSkillBundle(unittest.TestCase):
    """Validate the Agent Skills bundle."""

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
        text = open(path).read()
        self.assertTrue(text.startswith("---"), "SKILL.md must start with YAML frontmatter")
        parts = text.split("---", 2)
        self.assertGreaterEqual(len(parts), 3, "SKILL.md must have closing ---")
        import yaml
        data = yaml.safe_load(parts[1])
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("name"), "llm-fusion")
        self.assertTrue(data.get("description"), "description must be non-empty")
        # No secrets in frontmatter
        front_str = parts[1].lower()
        self.assertNotIn("api_key:", front_str)
        self.assertNotIn("password:", front_str)
        self.assertNotIn("secret:", front_str)

    def test_wrapper_script_exists(self):
        """The wrapper script must exist."""
        path = os.path.join(SKILL_DIR, "scripts", "llm_fusion.py")
        self.assertTrue(os.path.isfile(path), f"Wrapper script not found at {path}")

    def test_wrapper_script_runable(self):
        """The wrapper script must be runnable with python3 from a temp cwd."""
        script = os.path.join(SKILL_DIR, "scripts", "llm_fusion.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, script, "--dry-run", "--query", "hello"],
                capture_output=True, text=True, timeout=10,
                cwd=tmpdir,
            )
            self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")
            data = json.loads(result.stdout)
            self.assertTrue(data["ok"])
            self.assertEqual(data["query"], "hello")

    def test_references_exist(self):
        """The references directory should have the expected files."""
        ref_dir = os.path.join(SKILL_DIR, "references")
        self.assertTrue(os.path.isdir(ref_dir))
        self.assertTrue(os.path.isfile(os.path.join(ref_dir, "configuration.md")))
        self.assertTrue(os.path.isfile(os.path.join(ref_dir, "troubleshooting.md")))
        self.assertTrue(os.path.isfile(os.path.join(ref_dir, "pipeline.md")))

    def test_assets_exist(self):
        """The assets directory should have the example config."""
        assets_dir = os.path.join(SKILL_DIR, "assets")
        self.assertTrue(os.path.isdir(assets_dir))
        self.assertTrue(os.path.isfile(os.path.join(assets_dir, "fusion_config.yaml.example")))
