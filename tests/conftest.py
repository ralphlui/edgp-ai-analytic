"""
Pytest configuration file.

This file is automatically loaded by pytest before any tests run.
It sets up the test environment configuration.
"""
import os
import sys
from pathlib import Path

# Set APP_ENV to test before any other imports
os.environ['APP_ENV'] = 'test'

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
