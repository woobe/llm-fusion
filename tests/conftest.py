"""Pytest configuration for llm-fusion tests."""
import os
import sys

# Add project root so scripts/ package is importable
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
