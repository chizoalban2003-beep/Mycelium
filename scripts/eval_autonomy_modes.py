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


def _json_obj(status: int, text: str, step: str) -> dict[str, object]:
    if not (200 <= status < 300):
        raise RuntimeError(f"{step} failed: HTTP {status} :: {text}")
    try:
        obj = json.loads(text or "{}")
    except Exception as e:
        raise RuntimeError(f"{step} invalid JSON: {e}")
    if not isinstance(obj, dict):
        raise RuntimeError(f"{step} expected JSON object")
    return obj


def _set_policy(base: str, token: str, *, mode: str) -> None:
    url = f"{base}/api/nexus/policy"
    body = {
        "policy": {
            "actions": {
                "enabled": True,
                "notify_only": False,
                "require_confirm": False,
                "autonomy_mode": mode,
                "device_control_enabled": True,
                "min_confidence": 0.0,
                "allowed_capabilities": ["start_focus_session"],
            }
        }
    }
    s, t = _request("POST", url, token, body)
    _json_obj(s, t, f"set-policy-{mode}")


def _launch(base: str, token: str, *, current_device_id: str, candidate_device_ids: list[str]) -> dict[str, object]:
    url = f"{base}/api/nexus/hybrid/directive/work-session/auto-handoff-launch"
    body = {
        "window_minutes": 120,
        "base_duration_minutes": 45,
        "current_device_id": current_device_id,
        "candidate_device_ids": candidate_device_ids,
        "focus_app": "mycelium",
    }
    s, t = _request("POST", url, token, body)
    return _json_obj(s, t, "auto-handoff-launch")


def main() -> int:
    p = argparse.ArgumentParser(description="Scenario eval for strict|balanced|auto autonomy modes.")
    p.add_argument("--base-url", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--current-device-id", default="phone")
    p.add_argument("--candidate-device-ids", default="phone,laptop,desktop")
    args = p.parse_args()

    base = str(args.base_url).rstrip("/")
    token = str(args.token)
    candidates = [x.strip() for x in str(args.candidate_device_ids).split(",") if x.strip()]

    results: list[dict[str, object]] = []
    for mode in ("strict", "balanced", "auto"):
        _set_policy(base, token, mode=mode)
        launch = _launch(
            base,
            token,
            current_device_id=str(args.current_device_id),
            candidate_device_ids=candidates,
        )
        results.append(
            {
                "mode": mode,
                "launch_mode": launch.get("launch_mode"),
                "handoff_recommended": launch.get("handoff_recommended"),
                "recommended_device_id": launch.get("recommended_device_id"),
                "replica_id": launch.get("replica_id"),
                "queued_device_action_id": launch.get("queued_device_action_id"),
                "reason": launch.get("reason"),
            }
        )

    print(json.dumps({"ok": True, "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
