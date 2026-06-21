"""Pytest configuration for llm-fusion tests."""
import os
import sys

# Add the skill bundle root so scripts/ package is importable from tests.
_skill_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills", "llm-fusion"))
if _skill_root not in sys.path:
    sys.path.insert(0, _skill_root)
