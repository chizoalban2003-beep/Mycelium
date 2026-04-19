"""Central logger for physml.

All physml modules should use::

    from physml._log import get_logger
    _logger = get_logger(__name__)

Structured JSON logging can be enabled by setting the environment variable::

    MYCO_LOG_FORMAT=json

This emits newline-delimited JSON records instead of the default text format,
which is easier to ingest in log aggregators (Loki, Datadog, CloudWatch, etc.).

Example record::

    {"ts": 1714500000.123, "level": "INFO", "logger": "physml.goal_engine",
     "msg": "GoalEngine: goal a1b2c3d4 COMPLETED in 2.3s"}
"""
from __future__ import annotations

import json
import logging
import os
import time


class _JSONFormatter(logging.Formatter):
    """Emit log records as newline-delimited JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _build_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    if os.environ.get("MYCO_LOG_FORMAT", "").lower() == "json":
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s")
        )
    return handler


def configure_logging(level: str = "INFO") -> None:
    """Install a handler on the root physml logger.

    Call this once at application startup (the worker script and server do
    this automatically). Library users who manage their own logging can ignore
    it — physml is silent by default until a handler is attached.

    Parameters
    ----------
    level : str
        Logging level name, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
        The ``MYCO_VERBOSITY`` env var overrides this:
        ``concise`` → WARNING, ``normal`` → INFO, ``verbose`` → DEBUG.
    """
    verbosity = os.environ.get("MYCO_VERBOSITY", "").lower()
    level_map = {"concise": "WARNING", "normal": "INFO", "verbose": "DEBUG"}
    effective = level_map.get(verbosity, level).upper()

    root = logging.getLogger("physml")
    if root.handlers:
        return  # already configured
    root.setLevel(getattr(logging, effective, logging.INFO))
    root.addHandler(_build_handler())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
