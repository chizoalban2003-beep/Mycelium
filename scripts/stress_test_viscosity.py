from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import sys
from urllib import parse, request


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Mycelium synthetic causal stress test")
    parser.add_argument("--base-url", default=os.getenv("MYCELIUM_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--email", default=os.getenv("MYCELIUM_EMAIL", ""))
    parser.add_argument("--password", default=os.getenv("MYCELIUM_PASSWORD", ""))
    parser.add_argument("--project-id", type=int, default=None)
    parser.add_argument("--node-id", default="edge-node-0")
    parser.add_argument("--baseline-cpu-temp-c", type=float, default=58.0)
    parser.add_argument("--trial-cpu-temp-c", type=float, default=92.0)
    parser.add_argument("--baseline-battery-level", type=float, default=78.0)
    parser.add_argument("--trial-battery-level", type=float, default=70.0)
    parser.add_argument("--baseline-interruptions", type=int, default=1)
    parser.add_argument("--trial-interruptions", type=int, default=6)
    args = parser.parse_args()

    if not args.email or not args.password:
        print("Provide --email and --password (or MYCELIUM_EMAIL / MYCELIUM_PASSWORD).", file=sys.stderr)
        return 2

    base_url = str(args.base_url).rstrip("/")

    cookie_jar = http.cookiejar.CookieJar()
    opener = request.build_opener(request.HTTPCookieProcessor(cookie_jar))

    login_payload = parse.urlencode({"username": args.email, "password": args.password}).encode("utf-8")
    login_req = request.Request(
        url=f"{base_url}/api/auth/login",
        data=login_payload,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with opener.open(login_req, timeout=30) as resp:
        if int(getattr(resp, "status", 200)) >= 400:
            raise RuntimeError(f"login failed: {getattr(resp, 'status', 'unknown')}")

    stress_payload = {
        "project_id": args.project_id,
        "node_id": args.node_id,
        "spike_label": "cpu_temp_spike",
        "baseline_cpu_temp_c": args.baseline_cpu_temp_c,
        "trial_cpu_temp_c": args.trial_cpu_temp_c,
        "baseline_battery_level": args.baseline_battery_level,
        "trial_battery_level": args.trial_battery_level,
        "baseline_interruptions": args.baseline_interruptions,
        "trial_interruptions": args.trial_interruptions,
        "metric_name": "thermal_headroom",
        "target_col": "cpu_temp_c",
    }
    stress_req = request.Request(
        url=f"{base_url}/api/nexus/diagnostics/stress-test",
        data=json.dumps(stress_payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with opener.open(stress_req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    audit_url = f"{base_url}/api/nexus/knowledge/audit?limit=5"
    audit_req = request.Request(url=audit_url, method="GET")
    with opener.open(audit_req, timeout=60) as resp:
        audit_json = json.loads(resp.read().decode("utf-8"))

    print(json.dumps(
        {
            "stress": result,
            "reasoning": audit_json.get("reasoning", {}),
            "recent_trace_count": len(audit_json.get("validation", {}).get("recent_traces", [])),
        },
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())