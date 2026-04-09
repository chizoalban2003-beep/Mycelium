#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request


def _request_json(url: str, token: str = "") -> tuple[int, str, object]:
    headers = {"Accept": "application/json"}
    if token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"

    req = urllib.request.Request(url=url, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body) if body else {}
            return int(resp.status), body, data
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}
        return int(e.code), body, data


def _csv_values(v: str) -> list[str]:
    return [x.strip() for x in (v or "").split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description="Run public alpha readiness checks against a deployed Myco base URL.")
    p.add_argument("--base-url", required=True)
    p.add_argument("--token", default="")
    p.add_argument("--expected-git-sha", default="")
    p.add_argument("--window-hours", type=int, default=24)
    p.add_argument("--error-budget-failure-rate", type=float, default=0.10)
    p.add_argument("--expected-android-package", default="")
    p.add_argument("--expected-android-fingerprints-csv", default="")
    p.add_argument("--require-assetlinks", action="store_true")
    args = p.parse_args()

    base = args.base_url.rstrip("/")
    checks: list[dict[str, object]] = []

    def add_check(name: str, ok: bool, detail: str, extra: dict[str, object] | None = None) -> None:
        payload = {"name": name, "ok": bool(ok), "detail": detail}
        if extra:
            payload.update(extra)
        checks.append(payload)

    # 1) health
    status, _, health = _request_json(f"{base}/health")
    health_ok = 200 <= status < 300 and str((health or {}).get("status", "")).lower() == "ok"
    add_check("health", health_ok, f"http_status={status}", {"response_status": (health or {}).get("status")})

    # 2) deploy version
    if args.token.strip():
        status, _, version_obj = _request_json(f"{base}/api/nexus/deploy/version", token=args.token)
        live_sha = str((version_obj or {}).get("git_sha") or "")
        expected_sha = str(args.expected_git_sha or "").strip()
        sha_ok = (200 <= status < 300) and (not expected_sha or expected_sha == live_sha)
        detail = f"http_status={status}"
        if expected_sha:
            detail += f", expected_sha={expected_sha}, live_sha={live_sha}"
        add_check("deploy_version", sha_ok, detail, {"live_git_sha": live_sha, "expected_git_sha": expected_sha})

        # 3) SLO
        q = urllib.parse.urlencode(
            {
                "window_hours": int(args.window_hours),
                "error_budget_failure_rate": float(args.error_budget_failure_rate),
            }
        )
        status, _, slo_obj = _request_json(f"{base}/api/nexus/hybrid/handoff/slo?{q}", token=args.token)
        slo_ok = 200 <= status < 300 and bool((slo_obj or {}).get("error_budget_ok", True))
        add_check(
            "handoff_slo",
            slo_ok,
            f"http_status={status}, error_budget_ok={(slo_obj or {}).get('error_budget_ok', None)}",
            {
                "failure_rate": (slo_obj or {}).get("failure_rate"),
                "timeout_rate": (slo_obj or {}).get("timeout_rate"),
                "total": (slo_obj or {}).get("total"),
            },
        )
    else:
        add_check("deploy_version", False, "missing --token")
        add_check("handoff_slo", False, "missing --token")

    # 4) assetlinks
    status, _, assetlinks = _request_json(f"{base}/.well-known/assetlinks.json")
    assetlinks_list = assetlinks if isinstance(assetlinks, list) else []
    assetlinks_ok = 200 <= status < 300 and (bool(assetlinks_list) or not args.require_assetlinks)

    pkg_expected = str(args.expected_android_package or "").strip()
    fps_expected = {x.upper() for x in _csv_values(str(args.expected_android_fingerprints_csv or ""))}

    found_pkg = False
    found_fps: set[str] = set()
    if assetlinks_list:
        for stmt in assetlinks_list:
            target = (stmt or {}).get("target") if isinstance(stmt, dict) else {}
            if not isinstance(target, dict):
                continue
            pkg = str(target.get("package_name") or "")
            ns = str(target.get("namespace") or "")
            fps = target.get("sha256_cert_fingerprints") or []
            fps_set = {str(x).strip().upper() for x in fps if str(x).strip()}
            if ns == "android_app":
                if pkg_expected and pkg == pkg_expected:
                    found_pkg = True
                    found_fps |= fps_set
                elif not pkg_expected:
                    found_pkg = True
                    found_fps |= fps_set

    if pkg_expected and not found_pkg:
        assetlinks_ok = False
    if fps_expected and not fps_expected.issubset(found_fps):
        assetlinks_ok = False

    add_check(
        "assetlinks",
        assetlinks_ok,
        f"http_status={status}, statements={len(assetlinks_list)}",
        {
            "package_expected": pkg_expected,
            "package_found": found_pkg,
            "fingerprints_expected_count": len(fps_expected),
            "fingerprints_found_count": len(found_fps),
        },
    )

    ok = all(bool(c.get("ok")) for c in checks)
    failed_checks = [str(c.get("name")) for c in checks if not bool(c.get("ok"))]
    print(
        json.dumps(
            {
                "ok": ok,
                "base_url": base,
                "checks": checks,
                "failed_checks": failed_checks,
            },
            ensure_ascii=False,
        )
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
