#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def _parse_json(text: str) -> object:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


def login_for_token(base_url: str, *, email: str, password: str) -> str:
    url = base_url.rstrip("/") + "/api/auth/login"
    res = _request("POST", url, form_body={"username": email, "password": password})
    if res.status != 200:
        raise SystemExit(f"Login failed ({res.status}): {_parse_json(res.body_text)}")

    parsed = _parse_json(res.body_text)
    if isinstance(parsed, dict) and isinstance(parsed.get("access_token"), str) and str(parsed["access_token"]).strip():
        return str(parsed["access_token"]).strip()
    raise SystemExit(f"Login response missing access_token: {parsed}")


def _which(cmd: str) -> str | None:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        c = Path(p) / cmd
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def _notify(title: str, message: str) -> None:
    # Prefer Termux (Android) if available.
    if _which("termux-notification") is not None:
        try:
            subprocess.run(
                [
                    "termux-notification",
                    "--title",
                    str(title)[:64],
                    "--content",
                    str(message)[:256],
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except Exception:
            pass

    # Fallback: Linux desktops.
    if _which("notify-send") is not None:
        try:
            subprocess.run(["notify-send", str(title)[:64], str(message)[:256]], check=False)
            return
        except Exception:
            pass

    print(f"NOTIFY: {title} :: {message}")


def _state_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".mycelium_last_nudge_id"


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Poll /api/nexus/nudges/recent and trigger a device notification (Termux on Android supported). "
            "This is an optional phone-side helper so the assistant can surface nudges even when the web UI isn't open."
        )
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--poll-seconds", type=float, default=20.0)
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--unseen-only", action="store_true", default=True)
    p.add_argument("--no-unseen-only", dest="unseen_only", action="store_false")
    p.add_argument("--auto-ack", action="store_true", help="Mark the nudge as seen after notifying")
    p.add_argument("--state-file", default=None, help="Where to store last-seen nudge id")

    filt = p.add_argument_group("filters")
    filt.add_argument("--kinds", default=None, help="Comma-separated list of nudge kinds to notify on")
    filt.add_argument("--min-confidence", type=float, default=0.0, help="If payload.confidence exists, require >= this")

    auth = p.add_argument_group("auth")
    auth.add_argument("--token", default=None, help="Bearer token (preferred)")
    auth.add_argument("--email", default=None)
    auth.add_argument("--password", default=None)

    args = p.parse_args()

    token = args.token
    if not token:
        if not args.email or not args.password:
            raise SystemExit("Provide --token OR (--email and --password)")
        token = login_for_token(args.base_url, email=str(args.email), password=str(args.password))

    kinds: set[str] | None = None
    if args.kinds:
        kinds = {k.strip() for k in str(args.kinds).split(",") if k.strip()}

    state_file = _state_path(args.state_file)
    state_file.parent.mkdir(parents=True, exist_ok=True)

    last_id = ""
    if state_file.exists():
        try:
            last_id = state_file.read_text(encoding="utf-8").strip()
        except Exception:
            last_id = ""

    poll_s = max(5.0, min(float(args.poll_seconds), 3600.0))
    limit = max(1, min(int(args.limit), 5))

    while True:
        try:
            url = (
                args.base_url.rstrip("/")
                + f"/api/nexus/nudges/recent?limit={limit}&unseen_only={'true' if args.unseen_only else 'false'}"
            )
            res = _request("GET", url, token=token)
            if res.status != 200:
                time.sleep(poll_s)
                continue

            parsed = _parse_json(res.body_text)
            n = None
            if isinstance(parsed, dict) and isinstance(parsed.get("nudges"), list) and parsed["nudges"]:
                n = parsed["nudges"][0]

            if not isinstance(n, dict):
                time.sleep(poll_s)
                continue

            nid = str(n.get("id") or "")
            if not nid or nid == last_id:
                time.sleep(poll_s)
                continue

            kind = str(n.get("kind") or "")
            if kinds is not None and kind not in kinds:
                time.sleep(poll_s)
                continue

            payload = n.get("payload") if isinstance(n.get("payload"), dict) else {}
            conf = payload.get("confidence")
            if isinstance(conf, (int, float)) and float(conf) < float(args.min_confidence):
                time.sleep(poll_s)
                continue

            title = str(n.get("title") or "Nudge")
            message = str(n.get("message") or "")
            _notify(title, message)

            # Persist local state so we don't re-notify.
            try:
                state_file.write_text(nid, encoding="utf-8")
                last_id = nid
            except Exception:
                pass

            if bool(args.auto_ack):
                try:
                    ack_url = args.base_url.rstrip("/") + "/api/nexus/nudges/ack"
                    _request("POST", ack_url, token=token, json_body={"nudge_id": int(nid)})
                except Exception:
                    pass

        except KeyboardInterrupt:
            return 0
        except Exception:
            pass

        time.sleep(poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
