from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone


def _env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or "")


def main() -> int:
    parent_url = _env("PARENT_HUB_URL", "http://127.0.0.1:8000").rstrip("/")
    token = _env("HIVE_INGEST_TOKEN", "").strip()
    device_id = _env("NEXUS_DEVICE_ID", "child-smoketest").strip() or "child-smoketest"

    if not token:
        print("Missing HIVE_INGEST_TOKEN env var.", file=sys.stderr)
        return 2

    endpoint = f"{parent_url}/api/hive/curiosity/concept/import"

    payload = {
        "source": "child_smoketest",
        "version": "concept_v1",
        "concept": {
            "meta": {
                "kind": "curiosity_concept",
                "device_id": device_id,
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
            "tag": "smoketest",
            "verdict": "confirm",
            "note": "Child connected to Parent Hub via X-Hive-Token.",
        },
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Hive-Token": token,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(body)
            return 0 if 200 <= resp.status < 300 else 1
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        print(f"HTTP {e.code}: {body}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
