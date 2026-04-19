"""Shared pytest configuration and fixtures."""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as CPU-intensive; excluded from default CI run "
        "(select with -m slow, deselect with -m 'not slow')",
    )
