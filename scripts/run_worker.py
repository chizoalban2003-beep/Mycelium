"""Mycelium background worker.

Starts the companion in worker mode: goal execution loop + scheduled goals
+ file watcher. Runs indefinitely. Designed for the 'worker' Docker target
and for headless server deployments.

Environment variables (all optional):
    ANTHROPIC_API_KEY   — enables Claude for intelligent planning
    MYCO_NAME           — companion name (default: Myco)
    MYCO_DATA_DIR       — data directory (default: ~/.mycelium)
    MYCO_VERBOSITY      — concise | normal | verbose
    MYCO_WATCH_DIRS     — comma-separated dirs to watch for new files

Usage:
    python scripts/run_worker.py
    ANTHROPIC_API_KEY=sk-... python scripts/run_worker.py
"""

from __future__ import annotations

import os
import signal
import time

# Try to load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main() -> None:
    name = os.environ.get("MYCO_NAME", "Myco")
    data_dir = os.environ.get("MYCO_DATA_DIR", "~/.mycelium")
    verbosity = os.environ.get("MYCO_VERBOSITY", "normal")
    watch_dirs_raw = os.environ.get("MYCO_WATCH_DIRS", "")
    watch_dirs = [d.strip() for d in watch_dirs_raw.split(",") if d.strip()]

    from physml.companion import MyceliumCompanion

    companion = MyceliumCompanion(
        name=name,
        data_dir=data_dir,
        verbosity=verbosity,
    )
    companion.start()

    if watch_dirs:
        companion.personalise("watch_dirs", watch_dirs)

    # Start autonomous loops
    companion.goal_engine.start_loop()
    companion.scheduler.start()

    print(f"[Mycelium worker] {name} started — data_dir={data_dir}")
    print(f"[Mycelium worker] LLM: {'active' if companion.llm and companion.llm.available else 'offline (no API key)'}")
    print(f"[Mycelium worker] Goal loop: active (interval={companion.goal_engine.loop_interval:.0f}s)")
    print(f"[Mycelium worker] Scheduler: active ({companion.scheduler.status()['total']} schedules)")
    if watch_dirs:
        print(f"[Mycelium worker] Watching: {watch_dirs}")
    print("[Mycelium worker] Running. Press Ctrl+C to stop.")

    stop = False

    def _handle_signal(sig, frame):
        nonlocal stop
        print("\n[Mycelium worker] Shutdown signal received...")
        stop = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        while not stop:
            time.sleep(5)
    finally:
        companion.stop()
        print("[Mycelium worker] Stopped cleanly.")


if __name__ == "__main__":
    main()
