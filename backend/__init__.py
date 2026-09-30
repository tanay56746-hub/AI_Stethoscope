# ==============================================================================
# [LAYER: BACKEND PACKAGE & ENV LOADER] | File: backend/__init__.py
# Purpose: Package Initializer & Zero-Dependency `.env` Loader for Python Backend
# ==============================================================================
"""AI Stethoscope Backend Runtime & Inference Package."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env_file(env_path: Path = None) -> None:
    """
    Zero-dependency loader for root `.env` variables into `os.environ`.
    Does not overwrite existing system environment variables.
    """
    target = env_path or (PROJECT_ROOT / ".env")
    if not target.exists():
        return

    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Automatically load `.env` whenever `backend` is imported
load_env_file()
