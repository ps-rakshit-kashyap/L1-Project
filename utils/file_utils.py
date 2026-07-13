"""General filesystem helpers.

This file collects tiny path and hashing helpers that are shared across the
project. Keeping them here avoids repeating the same path normalization logic
in multiple services.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def sha1_text(text: str) -> str:
    """Create a stable hash for cache keys or file fingerprints."""
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def safe_relpath(path: Path, base: Path) -> str:
    """Return a normalized relative path that is safe to display in the UI."""
    return str(path.resolve().relative_to(base.resolve())).replace(os.sep, "/")


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return the same path for chaining."""
    path.mkdir(parents=True, exist_ok=True)
    return path
