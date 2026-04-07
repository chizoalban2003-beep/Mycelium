#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _request(method: str, url: str, token: str, body: dict[str, object] | None = None) -> tuple[int, str]:
    data = None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, method=method.upper(), data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
        return int(e.code), msg


def _post(url: str, token: str, body: dict[str, object]) -> tuple[int, str]:
    return _request("POST", url, token, body)


def _json_or_raise(status: int, text: str, step: str) -> dict[str, object]:
    if not (200 <= status < 300):
        raise RuntimeError(f"{step} failed: HTTP {status} :: {text}")
    try:
        obj = json.loads(text or "{}")
    except Exception as e:
        raise RuntimeError(f"{step} returned invalid JSON: {e}\n{text}")
    if not isinstance(obj, dict):
        raise RuntimeError(f"{step} returned non-object JSON: {obj!r}")
    return obj


def main() -> int:
    p = argparse.ArgumentParser(description="Run E2E smoke test for auto-handoff launch -> confirm -> ack -> verify flow.")
    p.add_argument("--base-url", required=True, help="API base URL, e.g. https://your-domain")
    p.add_argument("--token", required=True, help="Bearer token")
    p.add_argument("--project-id", type=int, default=None)
    p.add_argument("--window-minutes", type=int, default=120)
    p.add_argument("--base-duration", type=int, default=45)
    p.add_argument("--current-device-id", default="phone")
    p.add_argument("--candidate-device-ids", default="phone,laptop,desktop")
    p.add_argument("--focus-app", default="mycelium")
    p.add_argument("--ack-status", default="executed", choices=["executed", "failed"])
    p.add_argument("--planned-minutes", type=int, default=45)
    p.add_argument("--focused-minutes", type=int, default=45)
    p.add_argument("--completed", action="store_true")
    p.add_argument("--closed-early", action="store_true")
    p.add_argument("--interruptions", type=int, default=0)
    p.add_argument("--allow-recovery", action="store_true", help="Treat recovery launch mode as non-failure")
    args = p.parse_args()

    base = args.base_url.rstrip("/")
    token = str(args.token)

    candidate_ids = [x.strip() for x in str(args.candidate_device_ids).split(",") if x.strip()]

    launch_url = f"{base}/api/nexus/hybrid/directive/work-session/auto-handoff-launch"
    launch_body: dict[str, object] = {
        "project_id": args.project_id,
        "window_minutes": int(args.window_minutes),
        "base_duration_minutes": int(args.base_duration),
        "current_device_id": str(args.current_device_id),
        "candidate_device_ids": candidate_ids,
        "focus_app": str(args.focus_app),
    }

    s1, t1 = _post(launch_url, token, launch_body)
    launch = _json_or_raise(s1, t1, "auto-handoff-launch")

    launch_mode = str(launch.get("launch_mode") or "")
    replica_id = int(launch.get("replica_id") or 0)
    recommended_device_id = str(launch.get("recommended_device_id") or "")
    queued_action_id = int(launch.get("queued_device_action_id") or 0)

    print(f"launch_mode={launch_mode}")
    print(f"recommended_device_id={recommended_device_id or 'n/a'}")
    print(f"replica_id={replica_id or 0}")

    if launch_mode == "recovery":
        print("recovery_mode=true")
        print(f"reason={launch.get('reason')}")
        return 0 if args.allow_recovery else 2

    if replica_id <= 0:
        raise RuntimeError(f"launch did not return replica_id: {launch}")

    if launch_mode != "approved":
        confirm_url = f"{base}/api/nexus/hybrid/directive/work-session/auto-handoff-confirm"
        confirm_body = {
            "replica_id": int(replica_id),
            "device_id": (recommended_device_id or None),
        }
        s2, t2 = _post(confirm_url, token, confirm_body)
        confirm = _json_or_raise(s2, t2, "auto-handoff-confirm")
        queued_action_id = int(confirm.get("queued_device_action_id") or 0)
        print(f"confirm_ok={bool(confirm.get('ok', False))}")

    print(f"queued_device_action_id={queued_action_id}")

    ack_url = f"{base}/api/nexus/tasks/replicas/{replica_id}/ack"
    ack_body = {
        "status": str(args.ack_status),
        "notes": "smoke_autonomy_handoff_flow",
    }
    s3, t3 = _post(ack_url, token, ack_body)
    ack = _json_or_raise(s3, t3, "replica-ack")
    print(f"ack_status={ack.get('status')}")

    verify_url = f"{base}/api/nexus/tasks/replicas/{replica_id}/verify"
    verify_body = {
        "planned_minutes": int(args.planned_minutes),
        "focused_minutes": int(args.focused_minutes),
        "completed": bool(args.completed),
        "closed_early": bool(args.closed_early),
        "interruption_count": int(args.interruptions),
        "notes": "smoke_autonomy_handoff_flow",
    }
    s4, t4 = _post(verify_url, token, verify_body)
    verify = _json_or_raise(s4, t4, "replica-verify")

    print(f"verify_ok={bool(verify.get('ok', False))}")
    print(f"adherence={verify.get('adherence')}")
    print(f"accepted={verify.get('accepted')}")
    print(f"updated_species_confidence={verify.get('updated_species_confidence')}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
