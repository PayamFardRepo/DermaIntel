"""
Integration tests conftest - imports all fixtures from parent conftest.

This file ensures that all fixtures defined in backend/tests/conftest.py
are available to integration tests.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all fixtures from parent conftest
from conftest import *
