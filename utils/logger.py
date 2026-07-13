"""Logging utilities for the application.

The project uses this module to set up a single shared logging configuration
for console output and the on-disk application log. That keeps Streamlit reruns
from attaching duplicate handlers and makes debugging the retrieval pipeline a
lot easier.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import settings


def configure_logging() -> Path:
    """Configure file and console logging."""
    # Make sure the log folder exists before we create the file handler.
    settings.logs_path.mkdir(parents=True, exist_ok=True)
    log_file = settings.logs_path / "app.log"

    # Streamlit reruns the script often, so we only attach handlers once.
    root = logging.getLogger()
    if root.handlers:
        return log_file

    # INFO is a practical default: enough to trace the pipeline without
    # flooding the log with debug noise.
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    return log_file
