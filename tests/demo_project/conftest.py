"""
Demo project conftest: add the repo root to sys.path so that
``from app.checkout import ...`` works regardless of how pytest is
invoked (from the main project root or from this directory).
"""
import sys
from pathlib import Path

# Ensure that 'app/' is importable as a top-level package.
_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
