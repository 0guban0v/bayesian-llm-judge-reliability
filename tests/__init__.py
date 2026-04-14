"""Test package bootstrap for writable local cache directories."""

from __future__ import annotations

import os


def _ensure_pytensor_flags() -> None:
    """Force PyTensor to use a writable compiledir during tests."""

    compiledir = os.path.abspath(".uv-cache/pytensor")
    existing = os.environ.get("PYTENSOR_FLAGS", "")
    if "compiledir=" in existing:
        return
    prefix = f"{existing}," if existing else ""
    os.environ["PYTENSOR_FLAGS"] = f"{prefix}compiledir={compiledir}"


_ensure_pytensor_flags()
