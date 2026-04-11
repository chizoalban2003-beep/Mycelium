#!/usr/bin/env python3
"""Persistent thermal runner for cloud handoff mission control.

Designed for long-running Cursor cloud sessions using state persistence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
THERMAL_SCRIPT = ROOT / "scripts" / "thermal_awakening_cycle.py"
RUNNER_LOG = ROOT / "agent_metabolism" / "cloud_handoff_runner_log.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_runner_log(record: dict) -> None:
    RUNNER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNNER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_once(spike_id: str, dry_run: bool) -> dict:
    cmd = ["python3", str(THERMAL_SCRIPT), "--spike-id", spike_id]
    if dry_run:
        cmd.append("--dry-run")
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    record = {
        "at": _now_iso(),
        "spike_id": spike_id,
        "dry_run": dry_run,
        "returncode": int(proc.returncode),
    }
    if proc.returncode != 0:
        record["error"] = (proc.stderr or "").strip()[:400]
        _append_runner_log(record)
        return {"ok": False, **record}
    try:
        payload = json.loads(proc.stdout or "{}")
    except Exception:
        payload = {"ok": False, "detail": "invalid_json_output"}
    if isinstance(payload, dict):
        record["summary"] = {
            "ok": bool(payload.get("ok", False)),
            "retired_count": int(payload.get("retired_count", 0) or 0),
            "mutated_count": int(payload.get("mutated_count", 0) or 0),
            "promoted_count": int(payload.get("promoted_count", 0) or 0),
        }
    _append_runner_log(record)
    return payload if isinstance(payload, dict) else {"ok": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent thermal cycle mission-control runner.")
    parser.add_argument("--interval-seconds", type=int, default=3600, help="Seconds between cycles.")
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--spike-prefix", type=str, default="cloud-handoff", help="Spike ID prefix.")
    parser.add_argument("--dry-run", action="store_true", help="Run cycle in dry-run mode.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    interval = max(10, int(args.interval_seconds))
    max_cycles = max(0, int(args.max_cycles))
    i = 0
    while True:
        i += 1
        spike_id = f"{args.spike_prefix}-{i}"
        result = run_once(spike_id=spike_id, dry_run=bool(args.dry_run))
        print(json.dumps({"cycle": i, "result": result}, indent=2, sort_keys=True))
        if max_cycles and i >= max_cycles:
            break
        time.sleep(interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

