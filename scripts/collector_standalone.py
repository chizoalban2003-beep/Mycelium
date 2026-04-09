#!/usr/bin/env python3
"""Standalone signal collector — runs independently of the web server.

Usage:
    python3 scripts/collector_standalone.py [--interval 15] [--user-id 1]

This is Process 2 in the 3-process architecture:
    Process 1: Web server (uvicorn)
    Process 2: Signal collector (this script) ← runs on boot via systemd
    Process 3: Learning engine (scripts/learner_standalone.py)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mycelium_app.learning_daemon import run_signal_collection_tick
from mycelium_app.signal_collector import CollectorState


def main():
    parser = argparse.ArgumentParser(description="Myco standalone signal collector")
    parser.add_argument("--interval", type=int, default=15, help="Seconds between collection ticks")
    parser.add_argument("--user-id", type=int, default=1, help="User ID to collect for")
    parser.add_argument("--device-id", type=str, default="local", help="Device identifier")
    args = parser.parse_args()

    print(f"🌱 Myco signal collector | user={args.user_id} interval={args.interval}s device={args.device_id}")
    state = CollectorState()

    while True:
        try:
            n = run_signal_collection_tick(state, user_id=args.user_id, device_id=args.device_id)
            if n > 0:
                print(f"  tick {state.tick_count}: {n} signals")
        except Exception as e:
            print(f"  error: {e}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
