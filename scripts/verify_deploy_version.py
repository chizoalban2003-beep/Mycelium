#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
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


def main() -> int:
    p = argparse.ArgumentParser(description="Verify running deploy SHA from /api/nexus/deploy/version.")
    p.add_argument("--base-url", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--expected-git-sha", default=os.getenv("EXPECTED_GIT_SHA", ""))
    args = p.parse_args()

    status, text = _request(f"{args.base_url.rstrip('/')}/api/nexus/deploy/version", str(args.token))
    if not (200 <= status < 300):
        print(f"verify_deploy_version failed: HTTP {status} :: {text}", file=sys.stderr)
        return 2

    try:
        obj = json.loads(text or "{}")
    except Exception as e:
        print(f"verify_deploy_version invalid JSON: {e}", file=sys.stderr)
        return 3

    live_sha = str((obj or {}).get("git_sha") or "")
    expected = str(args.expected_git_sha or "").strip()

    print(json.dumps({"ok": True, "live_git_sha": live_sha, "expected_git_sha": expected}, ensure_ascii=False))
    if expected and expected != live_sha:
        print("deploy SHA mismatch", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
