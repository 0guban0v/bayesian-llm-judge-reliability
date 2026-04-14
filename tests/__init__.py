"""Test package bootstrap for writable local cache directories."""

from __future__ import annotations

import os
from pathlib import Path


def _ensure_pytensor_flags() -> None:
    """Force PyTensor to use a writable compiledir during tests."""

    existing = os.environ.get("PYTENSOR_FLAGS", "")
    if "compiledir=" in existing:
        return
    cache_root = Path(os.environ.get("UV_CACHE_DIR", ".uv-cache")).resolve()
    compiledir = cache_root / "pytensor"
    compiledir.mkdir(parents=True, exist_ok=True)
    prefix = f"{existing}," if existing else ""
    os.environ["PYTENSOR_FLAGS"] = f"{prefix}compiledir={compiledir}"


_ensure_pytensor_flags()
