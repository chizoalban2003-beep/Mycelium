#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
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


def _get_policy(base: str, token: str) -> dict[str, object]:
    s, t = _request("GET", f"{base}/api/nexus/policy", token)
    obj = _json_obj(s, t, "get-policy")
    p = obj.get("policy") if isinstance(obj.get("policy"), dict) else {}
    return p


def _set_policy(base: str, token: str, policy: dict[str, object]) -> None:
    s, t = _request("POST", f"{base}/api/nexus/policy", token, {"policy": policy})
    _json_obj(s, t, "set-policy")


def _launch_probe(base: str, token: str, *, current_device_id: str, candidate_device_ids: list[str]) -> dict[str, object]:
    body = {
        "window_minutes": 120,
        "base_duration_minutes": 45,
        "current_device_id": current_device_id,
        "candidate_device_ids": candidate_device_ids,
        "focus_app": "mycelium",
    }
    s, t = _request("POST", f"{base}/api/nexus/hybrid/directive/work-session/auto-handoff-launch", token, body)
    return _json_obj(s, t, "launch-probe")


def _fetch_audit_timeline(base: str, token: str, limit: int) -> list[dict[str, object]]:
    url = f"{base}/api/nexus/tasks/actions/audit/timeline?" + urllib.parse.urlencode({"limit": int(limit)})
    s, t = _request("GET", url, token)
    obj = _json_obj(s, t, "audit-timeline")
    items = obj.get("items") if isinstance(obj.get("items"), list) else []
    return [x for x in items if isinstance(x, dict)]


def _fetch_feedback_summary(base: str, token: str, window_hours: int) -> dict[str, object]:
    url = f"{base}/api/nexus/tasks/replicas/feedback/summary?" + urllib.parse.urlencode({"window_hours": int(window_hours)})
    s, t = _request("GET", url, token)
    return _json_obj(s, t, "feedback-summary")


def _tg_send(bot_token: str, chat_id: str, text: str) -> tuple[bool, str]:
    if not bot_token.strip() or not chat_id.strip():
        return False, "telegram credentials missing"
    url = f"https://api.telegram.org/bot{bot_token.strip()}/sendMessage"
    data = urllib.parse.urlencode(
        {
            "chat_id": chat_id.strip(),
            "text": text[:4096],
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    req = urllib.request.Request(url=url, method="POST", data=data)
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310
            ok = 200 <= int(getattr(resp, "status", 200)) < 300
            return bool(ok), ""
    except Exception as e:
        return False, str(e)


def _build_exec_summary(report: dict[str, object]) -> str:
    audit = report.get("audit") if isinstance(report.get("audit"), dict) else {}
    evals = report.get("mode_eval") if isinstance(report.get("mode_eval"), dict) else {}
    fb = report.get("feedback") if isinstance(report.get("feedback"), dict) else {}

    lines = [
        "Myco Global Audit",
        f"audited_actions={int(audit.get('total_actions', 0) or 0)}",
        f"would_pass_now={int(audit.get('would_pass_now', 0) or 0)}",
        f"would_fail_now={int(audit.get('would_fail_now', 0) or 0)}",
        f"acceptance_rate={float(fb.get('acceptance_rate', 0.0) or 0.0):.3f}",
    ]
    for mode in ("strict", "balanced", "auto"):
        m = evals.get(mode) if isinstance(evals.get(mode), dict) else {}
        lines.append(f"{mode}: launch_mode={m.get('launch_mode')} handoff={m.get('handoff_recommended')}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Global governance + autonomy maturity audit report.")
    p.add_argument("--base-url", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--current-device-id", default="phone")
    p.add_argument("--candidate-device-ids", default="phone,laptop,desktop")
    p.add_argument("--audit-limit", type=int, default=1000)
    p.add_argument("--feedback-window-hours", type=int, default=168)
    p.add_argument("--telegram-bot-token", default=os.getenv("NOTIFICATIONS_TELEGRAM_BOT_TOKEN", ""))
    p.add_argument("--telegram-chat-id", default=os.getenv("AUDIT_TELEGRAM_CHAT_ID", ""))
    args = p.parse_args()

    base = str(args.base_url).rstrip("/")
    token = str(args.token)
    candidates = [x.strip() for x in str(args.candidate_device_ids).split(",") if x.strip()]

    original_policy = _get_policy(base, token)

    # Gather audit + feedback first.
    items = _fetch_audit_timeline(base, token, limit=int(args.audit_limit))
    total = len(items)
    pass_now = sum(1 for i in items if bool(i.get("would_pass_now", False)))
    fail_now = max(0, total - pass_now)

    gate_counts: dict[str, int] = {}
    for i in items:
        gates = i.get("gates") if isinstance(i.get("gates"), list) else []
        for g in gates:
            k = str(g)
            gate_counts[k] = int(gate_counts.get(k, 0) + 1)

    fb = _fetch_feedback_summary(base, token, window_hours=int(args.feedback_window_hours))
    total_verified = int(fb.get("total_verified", 0) or 0)
    label_accept = fb.get("label_acceptance") if isinstance(fb.get("label_acceptance"), dict) else {}
    acceptance_rate = 0.0
    if total_verified > 0 and label_accept:
        acceptance_rate = float(sum(float(v) for v in label_accept.values()) / max(1, len(label_accept)))

    # Evaluate all modes then restore policy.
    mode_eval: dict[str, dict[str, object]] = {}
    try:
        for mode in ("strict", "balanced", "auto"):
            p2 = dict(original_policy)
            actions = p2.get("actions") if isinstance(p2.get("actions"), dict) else {}
            p2["actions"] = {
                **actions,
                "autonomy_mode": mode,
                "enabled": bool(actions.get("enabled", True)),
                "device_control_enabled": bool(actions.get("device_control_enabled", True)),
            }
            _set_policy(base, token, p2)
            launch = _launch_probe(
                base,
                token,
                current_device_id=str(args.current_device_id),
                candidate_device_ids=candidates,
            )
            mode_eval[mode] = {
                "launch_mode": launch.get("launch_mode"),
                "handoff_recommended": launch.get("handoff_recommended"),
                "recommended_device_id": launch.get("recommended_device_id"),
                "reason": launch.get("reason"),
            }
    finally:
        _set_policy(base, token, original_policy)

    report: dict[str, object] = {
        "ok": True,
        "audit": {
            "total_actions": total,
            "would_pass_now": pass_now,
            "would_fail_now": fail_now,
            "pass_rate": (float(pass_now) / float(total) if total > 0 else 0.0),
            "gate_counts": gate_counts,
        },
        "feedback": {
            "window_hours": int(args.feedback_window_hours),
            "total_verified": total_verified,
            "acceptance_rate": float(round(acceptance_rate, 6)),
            "label_counts": fb.get("label_counts") if isinstance(fb.get("label_counts"), dict) else {},
            "label_acceptance": label_accept,
        },
        "mode_eval": mode_eval,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if str(args.telegram_bot_token).strip() and str(args.telegram_chat_id).strip():
        text = _build_exec_summary(report)
        ok, err = _tg_send(str(args.telegram_bot_token), str(args.telegram_chat_id), text)
        if not ok:
            print(json.dumps({"telegram_sent": False, "error": err}, ensure_ascii=False))
        else:
            print(json.dumps({"telegram_sent": True}, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
