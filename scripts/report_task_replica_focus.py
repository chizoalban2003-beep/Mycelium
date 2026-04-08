#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def _post(url: str, token: str, body: dict[str, object]) -> tuple[int, str]:
    req = urllib.request.Request(
        url=url,
        method="POST",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
        return int(e.code), msg


def main() -> int:
    p = argparse.ArgumentParser(description="Report post-execution focus outcome for a task replica.")
    p.add_argument("--base-url", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--replica-id", type=int, required=True)
    p.add_argument("--planned-minutes", type=int, default=45)
    p.add_argument("--focused-minutes", type=int, required=True)
    p.add_argument("--completed", action="store_true")
    p.add_argument("--closed-early", action="store_true")
    p.add_argument("--interruptions", type=int, default=0)
    p.add_argument("--notes", default="")
    args = p.parse_args()

    url = f"{args.base_url.rstrip('/')}/api/nexus/tasks/replicas/{int(args.replica_id)}/verify"
    body = {
        "planned_minutes": int(args.planned_minutes),
        "focused_minutes": int(args.focused_minutes),
        "completed": bool(args.completed),
        "closed_early": bool(args.closed_early),
        "interruption_count": int(args.interruptions),
        "notes": str(args.notes or ""),
    }
    status, text = _post(url, str(args.token), body)
    print(text)
    return 0 if 200 <= status < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())
