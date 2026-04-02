"""Structured JSON logging helpers for the FastAPI surface."""

from __future__ import annotations

import logging

try:
    import structlog
except ImportError:  # pragma: no cover - optional runtime dependency
    structlog = None


def configure_logging() -> None:
    """Configure JSON-friendly structlog output with a standard-library fallback."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if structlog is None:
        return
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
