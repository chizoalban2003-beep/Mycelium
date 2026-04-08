#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def _fetch(url: str, timeout: float = 5.0) -> tuple[int, str]:
    req = urllib.request.Request(url=url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return int(resp.status), body


def main() -> int:
    base_url = str(os.getenv("BASE_URL", "http://127.0.0.1:8000") or "http://127.0.0.1:8000").rstrip("/")
    health_url = f"{base_url}/health"
    docs_url = f"{base_url}/docs"

    try:
        health_status, health_body = _fetch(health_url)
        if health_status != 200:
            print(f"health check failed: {health_status} {health_body}", file=sys.stderr)
            return 2

        try:
            parsed = json.loads(health_body)
        except Exception:
            parsed = {}
        if isinstance(parsed, dict) and str(parsed.get("status", "")).lower() != "ok":
            print(f"health payload is not ok: {health_body}", file=sys.stderr)
            return 2

        docs_status, _ = _fetch(docs_url)
        if docs_status != 200:
            print(f"docs check failed: {docs_status}", file=sys.stderr)
            return 2

        print(json.dumps({"ok": True, "base_url": base_url, "health": health_status, "docs": docs_status}))
        return 0
    except urllib.error.URLError as exc:
        print(f"unable to reach {base_url}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"smoke test failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
