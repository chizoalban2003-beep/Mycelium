#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request


def _request(url: str, token: str) -> tuple[int, str]:
    req = urllib.request.Request(
        url=url,
        method="GET",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
        return int(e.code), body


def _tg_send(bot_token: str, chat_id: str, text: str) -> tuple[bool, str]:
    if not bot_token.strip() or not chat_id.strip():
        return False, "telegram credentials missing"
    url = f"https://api.telegram.org/bot{bot_token.strip()}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id.strip(), "text": text[:4096]}).encode("utf-8")
    req = urllib.request.Request(url=url, method="POST", data=data)
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310
            return (200 <= int(getattr(resp, "status", 200)) < 300), ""
    except Exception as e:
        return False, str(e)


def main() -> int:
    p = argparse.ArgumentParser(description="Check handoff SLO and optionally alert Telegram.")
    p.add_argument("--base-url", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--window-hours", type=int, default=24)
    p.add_argument("--error-budget-failure-rate", type=float, default=0.10)
    p.add_argument("--telegram-bot-token", default="")
    p.add_argument("--telegram-chat-id", default="")
    args = p.parse_args()

    q = urllib.parse.urlencode(
        {
            "window_hours": int(args.window_hours),
            "error_budget_failure_rate": float(args.error_budget_failure_rate),
        }
    )
    url = f"{args.base_url.rstrip('/')}/api/nexus/hybrid/handoff/slo?{q}"

    status, text = _request(url, str(args.token))
    if not (200 <= status < 300):
        print(f"handoff_slo failed: HTTP {status} :: {text}")
        return 2

    obj = json.loads(text or "{}") if text else {}
    print(json.dumps(obj, ensure_ascii=False))

    ok = bool(obj.get("error_budget_ok", True))
    if not ok and str(args.telegram_bot_token).strip() and str(args.telegram_chat_id).strip():
        msg = (
            "SynapseHive SLO alert\n"
            f"failure_rate={obj.get('failure_rate')} timeout_rate={obj.get('timeout_rate')}\n"
            f"total={obj.get('total')} failed={obj.get('failed')} timed_out={obj.get('timed_out')}"
        )
        _tg_send(str(args.telegram_bot_token), str(args.telegram_chat_id), msg)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
