"""
tests/conftest.py — Pytest configuration for Graphite.

Adds the project root to sys.path so tests can import core/ and domains/.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
