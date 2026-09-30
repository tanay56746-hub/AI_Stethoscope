#!/usr/bin/env python3
# ==============================================================================
# [LAYER: ROOT CLI & RENDER ENTRYPOINT] | File: manage.py
# Purpose: Top-level entrypoint for Render and CLI management
# ==============================================================================
"""
AI Stethoscope - Root Management Entrypoint
===========================================
Allows Render to start the Python backend using:
  python manage.py runserver
or directly from repository root.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.manage import main

if __name__ == "__main__":
    main()
