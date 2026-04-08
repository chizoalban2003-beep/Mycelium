"""Compatibility wrapper for older deployment notes.

Gemini-style instructions often reference `scripts/observe_daemon.py`.
In this repo the actual implementation is `scripts/passive_telemetry_daemon.py`.

This wrapper simply forwards args to the real script.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "scripts" / "passive_telemetry_daemon.py"
    if not target.exists():
        raise SystemExit("passive_telemetry_daemon.py not found")

    # Replace argv[0] so help/usage is clearer.
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
