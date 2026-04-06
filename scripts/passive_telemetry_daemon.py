#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class HttpResult:
    status: int
    body_text: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _request(
    method: str,
    url: str,
    *,
    token: str | None = None,
    json_body: dict[str, object] | None = None,
    form_body: dict[str, str] | None = None,
    timeout_s: float = 15.0,
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


def _parse_json_maybe(text: str) -> object:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


def login_for_token(base_url: str, *, email: str, password: str) -> str:
    url = base_url.rstrip("/") + "/api/auth/login"
    # FastAPI OAuth2PasswordRequestForm expects fields: username, password
    res = _request(
        "POST",
        url,
        form_body={"username": email, "password": password},
    )
    if res.status != 200:
        parsed = _parse_json_maybe(res.body_text)
        raise SystemExit(f"Login failed ({res.status}): {parsed}")

    parsed = _parse_json_maybe(res.body_text)
    if isinstance(parsed, dict) and isinstance(parsed.get("access_token"), str) and parsed["access_token"].strip():
        return str(parsed["access_token"]).strip()

    raise SystemExit(f"Login response missing access_token: {parsed}")


def _which(cmd: str) -> str | None:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(p) / cmd
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _run(cmd: list[str], *, timeout_s: float = 1.5) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=timeout_s)
        return out.decode("utf-8", errors="replace")
    except Exception:
        return None


def _xprop(args: list[str]) -> str | None:
    if _which("xprop") is None:
        return None
    return _run(["xprop", *args])


_ACTIVE_WIN_RE = re.compile(r"window id # (0x[0-9a-fA-F]+)")


def _active_window_id_x11() -> str | None:
    out = _xprop(["-root", "_NET_ACTIVE_WINDOW"])
    if not out:
        return None
    m = _ACTIVE_WIN_RE.search(out)
    if not m:
        return None
    wid = m.group(1).lower()
    if wid in ("0x0", "0x00000000"):
        return None
    return wid


def _parse_xprop_quoted_list(out: str) -> list[str]:
    # Typical: WM_CLASS(STRING) = "google-chrome", "Google-chrome"
    return re.findall(r"\"([^\"]+)\"", out or "")


def _active_app_token_x11(wid_hex: str) -> str | None:
    out = _xprop(["-id", wid_hex, "WM_CLASS"])
    if not out:
        return None
    parts = _parse_xprop_quoted_list(out)
    if not parts:
        return None

    # Heuristic: prefer the last token (often human-facing class)
    token = str(parts[-1]).strip()
    if not token:
        return None
    return token[:128]


def _post_app_open(
    *,
    base_url: str,
    token: str,
    project_id: int | None,
    device_id: str | None,
    app_token: str,
    backend: str,
    dry_run: bool,
) -> None:
    body: dict[str, object] = {
        "project_id": project_id,
        "device_id": device_id,
        "signal_type": "app_open",
        "payload": {"app": app_token, "backend": backend, "observed_at": _utc_now_iso()},
    }

    if dry_run:
        print(json.dumps({"would_post": body}, ensure_ascii=False))
        return

    url = base_url.rstrip("/") + "/api/nexus/telemetry/ingest"
    res = _request("POST", url, token=token, json_body=body)
    if res.status != 200:
        parsed = _parse_json_maybe(res.body_text)
        raise RuntimeError(f"telemetry ingest failed ({res.status}): {parsed}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "PassiveTelemetry daemon (Linux, best-effort). Polls active app/window and posts app_open events into "
            "/api/nexus/telemetry/ingest. X11 backend only; Wayland support is not implemented yet."
        )
    )

    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--project-id", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true", help="Print events but do not POST")

    auth = parser.add_argument_group("auth")
    auth.add_argument("--token", default=None, help="Bearer token (preferred)")
    auth.add_argument("--email", default=None, help="If set with --password, logs in to fetch a token")
    auth.add_argument("--password", default=None, help="Password for --email login")

    args = parser.parse_args()

    token = str(args.token).strip() if args.token else None
    if not token and args.email and args.password:
        token = login_for_token(args.base_url, email=str(args.email), password=str(args.password))
        print("login_ok token_issued")

    if not token and not args.dry_run:
        raise SystemExit("Missing auth: pass --token or --email + --password (or use --dry-run)")

    # Backend selection
    if not os.environ.get("DISPLAY"):
        raise SystemExit("DISPLAY is not set; X11 backend requires an X11 session")
    if _which("xprop") is None:
        raise SystemExit("Missing dependency: xprop (install package 'x11-utils' on Debian/Ubuntu)")

    poll = max(0.25, min(float(args.poll_seconds), 30.0))

    stop = {"value": False}

    def _handle(_sig, _frame):
        stop["value"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    last_app: str | None = None
    last_window: str | None = None

    print(
        json.dumps(
            {
                "started": True,
                "at": _utc_now_iso(),
                "base_url": args.base_url,
                "project_id": args.project_id,
                "device_id": args.device_id,
                "backend": "x11",
                "poll_seconds": poll,
                "dry_run": bool(args.dry_run),
                "privacy": "Only app identifiers are captured (no window titles, no keystrokes).",
            },
            ensure_ascii=False,
        )
    )

    while not stop["value"]:
        wid = _active_window_id_x11()
        if wid and wid != last_window:
            app = _active_app_token_x11(wid)
            last_window = wid

            if app and app != last_app:
                last_app = app
                try:
                    _post_app_open(
                        base_url=str(args.base_url),
                        token=str(token or ""),
                        project_id=args.project_id,
                        device_id=args.device_id,
                        app_token=app,
                        backend="x11",
                        dry_run=bool(args.dry_run),
                    )
                    print(json.dumps({"event": "app_open", "app": app, "at": _utc_now_iso()}, ensure_ascii=False))
                except Exception as e:
                    print(json.dumps({"error": str(e), "at": _utc_now_iso()}, ensure_ascii=False), file=sys.stderr)

        time.sleep(poll)

    print(json.dumps({"stopped": True, "at": _utc_now_iso()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
