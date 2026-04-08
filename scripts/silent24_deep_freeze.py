#!/usr/bin/env python3
from __future__ import annotations

"""Silent-24 helper: run a Deep Freeze sweep and print 'First Truth'.

This script is intentionally stdlib-only. It:
- logs in (optional) to fetch a bearer token
- calls /api/nexus/telemetry/summary to show current observation state
- calls /api/nexus/telemetry/deep-freeze-sweep to record a GrowthLedger entry
- calls /api/nexus/growth/status to show stage + unlocked features

You can run it anytime; it’s safe and deterministic.
"""

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class HttpResult:
    status: int
    body_text: str


def _request(
    method: str,
    url: str,
    *,
    token: str | None = None,
    json_body: dict[str, object] | None = None,
    form_body: dict[str, str] | None = None,
    timeout_s: float = 20.0,
) -> HttpResult:
    headers = {"Accept": "application/json"}
    data: bytes | None = None

    if json_body is not None and form_body is not None:
        raise ValueError("Provide either json_body or form_body")

    if json_body is not None:
        data = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    if form_body is not None:
        data = urllib.parse.urlencode(form_body).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"

    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url=url, method=method.upper(), headers=headers, data=data)

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return HttpResult(status=int(resp.status), body_text=body)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(e)
        return HttpResult(status=int(e.code), body_text=body)


def _parse_json(text: str) -> object:
    try:
        return json.loads(text or "{}")
    except Exception:
        return {"raw": text}


def login_for_token(base_url: str, *, email: str, password: str) -> str:
    url = base_url.rstrip("/") + "/api/auth/login"
    res = _request("POST", url, form_body={"username": email, "password": password})
    if res.status != 200:
        raise SystemExit(f"Login failed ({res.status}): {_parse_json(res.body_text)}")

    parsed = _parse_json(res.body_text)
    if isinstance(parsed, dict) and isinstance(parsed.get("access_token"), str) and parsed["access_token"].strip():
        return str(parsed["access_token"]).strip()

    raise SystemExit(f"Login response missing access_token: {parsed}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Silent-24 Deep Freeze sweep and print growth status")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--project-id", type=int, default=None)
    parser.add_argument("--device-id", default=None)

    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--min-pairs", type=int, default=30)
    parser.add_argument("--accept-r2", type=float, default=0.90)

    auth = parser.add_argument_group("auth")
    auth.add_argument("--token", default=None)
    auth.add_argument("--email", default=None)
    auth.add_argument("--password", default=None)

    args = parser.parse_args()

    token = str(args.token).strip() if args.token else None
    if not token and args.email and args.password:
        token = login_for_token(args.base_url, email=str(args.email), password=str(args.password))

    if not token:
        raise SystemExit("Missing auth: pass --token or --email + --password")

    # 1) Show observation summary (includes 'first_word' when confidence is high).
    summary_url = args.base_url.rstrip("/") + "/api/nexus/telemetry/summary" + (
        f"?window_hours={int(args.window_hours)}" + (f"&project_id={int(args.project_id)}" if args.project_id is not None else "")
    )
    summary_res = _request("GET", summary_url, token=token)
    summary_json = _parse_json(summary_res.body_text)

    # 2) Run the Deep Freeze sweep (records a GrowthLedgerEntry).
    sweep_url = args.base_url.rstrip("/") + "/api/nexus/telemetry/deep-freeze-sweep"
    sweep_body: dict[str, object] = {
        "project_id": args.project_id,
        "device_id": args.device_id,
        "window_hours": int(args.window_hours),
        "min_pairs": int(args.min_pairs),
        "accept_r2_threshold": float(args.accept_r2),
    }
    sweep_res = _request("POST", sweep_url, token=token, json_body=sweep_body)
    sweep_json = _parse_json(sweep_res.body_text)

    # 3) Growth status (stage + unlocks + motto).
    status_url = args.base_url.rstrip("/") + "/api/nexus/growth/status" + (
        f"?project_id={int(args.project_id)}" if args.project_id is not None else ""
    )
    status_res = _request("GET", status_url, token=token)
    status_json = _parse_json(status_res.body_text)

    out = {
        "telemetry_summary_status": int(summary_res.status),
        "telemetry_summary": summary_json,
        "deep_freeze_sweep_status": int(sweep_res.status),
        "deep_freeze_sweep": sweep_json,
        "growth_status_code": int(status_res.status),
        "growth_status": status_json,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))

    # Provide a short, human-friendly line too.
    if isinstance(summary_json, dict) and summary_json.get("first_word"):
        print("\nfirst_truth:")
        print(str(summary_json.get("first_word")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
