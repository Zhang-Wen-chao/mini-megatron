"""Shared pytest fixtures for mini-megatron tests."""
import os
import sys
import pytest

# Add repo root to sys.path so `import config`, `import model.*` etc. work
# when pytest is run from any directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
