#!/usr/bin/env python3
"""Standalone learning engine — runs independently of the web server.

Usage:
    python3 scripts/learner_standalone.py [--interval 120] [--user-id 1]

This is Process 3 in the 3-process architecture:
    Process 1: Web server (uvicorn)
    Process 2: Signal collector (scripts/collector_standalone.py)
    Process 3: Learning engine (this script) ← runs on boot or via cron
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mycelium_app.learning_daemon import run_learning_tick


def main():
    parser = argparse.ArgumentParser(description="Myco standalone learning engine")
    parser.add_argument("--interval", type=int, default=120, help="Seconds between learning cycles")
    parser.add_argument("--user-id", type=int, default=1, help="User ID to learn for")
    parser.add_argument("--device-id", type=str, default="local", help="Device identifier")
    parser.add_argument("--window-hours", type=int, default=6, help="Signal lookback window")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    args = parser.parse_args()

    print(f"🧠 Myco learning engine | user={args.user_id} interval={args.interval}s window={args.window_hours}h")

    while True:
        try:
            result = run_learning_tick(
                user_id=args.user_id,
                device_id=args.device_id,
                window_hours=args.window_hours,
            )
            stage = result.get("stage", "?")
            actions = result.get("actions", [])
            ff = result.get("force_field", {})
            tune = result.get("auto_tune", {})

            print(f"  stage={stage} actions={actions}")
            if ff:
                print(f"  field: particles={ff.get('n_particles')} coherence={ff.get('agent_coherence'):.3f}")
            if tune:
                print(f"  tune: gen={tune.get('generation')} mae={tune.get('last_mae')}")

        except Exception as e:
            print(f"  error: {e}")

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
