"""Central logger for physml.

All physml modules should use::

    from physml._log import get_logger
    _logger = get_logger(__name__)

This keeps logging opt-in for library users (no handlers installed by default)
while giving them a standard way to enable debug output::

    import logging
    logging.getLogger("physml").setLevel(logging.DEBUG)
"""
from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
