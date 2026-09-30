#!/usr/bin/env python3
# ==============================================================================
# [LAYER: BACKEND CLI ENTRYPOINT] | File: backend/manage.py
# Purpose: Convenient CLI Launcher (`python manage.py runserver` or `predict`)
# ==============================================================================
"""
AI Stethoscope - Backend Management & Server Entrypoint
=======================================================
Supports standard commands when working inside `backend/` or project root:
  python manage.py runserver [port]
  python backend/manage.py runserver [port]
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.demo_server import run_server


def main():
    args = sys.argv[1:]
    port = int(os.environ.get("BACKEND_PORT", "8000"))
    host = os.environ.get("BACKEND_HOST", "127.0.0.1")

    for arg in args:
        if arg == "runserver":
            continue
        elif ":" in arg:
            h, p = arg.split(":", 1)
            if h:
                host = h
            if p.isdigit():
                port = int(p)
        elif arg.isdigit():
            port = int(arg)

    run_server(port=port, host=host)


if __name__ == "__main__":
    main()
