from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta

from sqlmodel import Session, select

from mycelium_app.models import NexusNudge, ProjectMember, ProjectRole, SignalLedgerEvent
from mycelium_app.parental_policy import get_policy
from mycelium_app.settings import settings


def _dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _simple_confidence(n_events: int, signal_counts: dict[str, int]) -> float:
    unique = len([k for k, v in signal_counts.items() if v > 0])
    density = min(1.0, n_events / 500.0)
    diversity = min(1.0, unique / 6.0)
    conf = 0.25 + 0.55 * density + 0.20 * diversity
    return float(max(0.0, min(1.0, conf)))


def maybe_queue_telemetry_assistant_nudge(
    session: Session,
    *,
    user_id: int,
    project_id: int | None = None,
    device_id: str | None = None,
    window_hours: int | None = None,
) -> bool:
    """Create a telemetry-derived nudge (opt-in, throttled).

    This is intentionally conservative:
    - respects parental policy allow_modalities
    - only speaks when confidence is high
    - throttles to avoid spam

    Returns True if a nudge was queued.
    """

    policy = get_policy(session, int(user_id))
    allow_modalities = policy.get("allow_modalities") if isinstance(policy.get("allow_modalities"), list) else []
    if allow_modalities and "telemetry" not in set(str(m).lower() for m in allow_modalities):
        return False

    if window_hours is None:
        window_hours = int(getattr(settings, "nexus_telemetry_assistant_window_hours", 6))
    tick_window_hours = max(1, min(int(window_hours), 168))
    since = datetime.utcnow() - timedelta(hours=tick_window_hours)

    q = select(SignalLedgerEvent).where(
        SignalLedgerEvent.created_by_user_id == int(user_id),
        SignalLedgerEvent.created_at >= since,
    )
    if project_id is not None:
        q = q.where(SignalLedgerEvent.project_id == int(project_id))
    if device_id:
        q = q.where(SignalLedgerEvent.device_id == str(device_id)[:64])

    rows = session.exec(q).all()
    if not rows:
        return False

    counts: Counter[str] = Counter()
    for r in rows:
        counts[str(r.signal_type or "").lower()] += 1
    signal_counts = {k: int(v) for k, v in counts.most_common(50)}

    conf = _simple_confidence(len(rows), signal_counts)
    conf_threshold = float(getattr(settings, "nexus_telemetry_assistant_confidence_threshold", 0.85) or 0.0)
    if conf < conf_threshold:
        return False

    throttle_min = max(5, min(int(getattr(settings, "nexus_telemetry_assistant_throttle_minutes", 120)), 7 * 24 * 60))
    throttled_since = datetime.utcnow() - timedelta(minutes=int(throttle_min))

    qn = (
        select(NexusNudge)
        .where(NexusNudge.created_by_user_id == int(user_id))
        .where(NexusNudge.kind == "telemetry_assistant")
        .where(NexusNudge.created_at >= throttled_since)
        .order_by(NexusNudge.created_at.desc())
    )
    if project_id is None:
        qn = qn.where(NexusNudge.project_id.is_(None))
    else:
        qn = qn.where(NexusNudge.project_id == int(project_id))

    recent = session.exec(qn.limit(1)).first()
    if recent is not None:
        return False

    patterns: list[str] = []
    if signal_counts.get("app_open", 0) >= 12:
        patterns.append("app_viscosity")
    if (signal_counts.get("screen_on", 0) + signal_counts.get("screen_off", 0)) >= 12:
        patterns.append("temporal_pulse")
    if signal_counts.get("network", 0) >= 6:
        patterns.append("connectivity_flow")

    actions_cfg = policy.get("actions") if isinstance(policy.get("actions"), dict) else {}
    actions_enabled = bool(actions_cfg.get("enabled", False))

    project_role: str | None = None
    can_execute_project_actions = True
    if project_id is not None:
        member = session.exec(
            select(ProjectMember).where(
                ProjectMember.project_id == int(project_id),
                ProjectMember.user_id == int(user_id),
            )
        ).first()
        if member is not None:
            try:
                project_role = str(member.role.value if isinstance(member.role, ProjectRole) else member.role)
            except Exception:
                project_role = str(member.role or "")
        if project_role == str(ProjectRole.viewer.value):
            can_execute_project_actions = False

    proposed_actions: list[dict[str, object]] = []
    if actions_enabled and can_execute_project_actions:
        # Proposals only; execution is always user-confirmed.
        if "app_viscosity" in patterns:
            proposed_actions.append(
                {
                    "action_id": "suggest_focus_zone",
                    "title": "Suggest a Work-Zone",
                    "detail": "Propose a daily focus window and learn from accept/reject feedback.",
                    "requires_confirm": True,
                }
            )
        proposed_actions.append(
            {
                "action_id": "run_deep_freeze_sweep",
                "title": "Run Deep Freeze sweep",
                "detail": "Run a transparent next-app transition sweep over recent telemetry.",
                "requires_confirm": True,
                "endpoint": "/api/nexus/telemetry/deep-freeze-sweep",
            }
        )

    title = "Telemetry insight"
    if actions_enabled and proposed_actions:
        title = "Assistant proposal"

    msg = (
        "I’ve observed a stable pattern in your recent signals. "
        "If you want, I can propose a small, reversible optimization and learn from your feedback."
    )

    payload = {
        "confidence": float(round(conf, 4)),
        "window_hours": int(tick_window_hours),
        "n_events": int(len(rows)),
        "signal_counts": signal_counts,
        "patterns": patterns,
        "actions": proposed_actions,
        "policy": {
            "actions_enabled": actions_enabled,
            "notify_only": bool(actions_cfg.get("notify_only", True)),
            "require_confirm": bool(actions_cfg.get("require_confirm", True)),
        },
        "project_role": project_role,
        "can_execute_actions": bool(actions_enabled and can_execute_project_actions),
    }

    n = NexusNudge(
        created_by_user_id=int(user_id),
        project_id=(None if project_id is None else int(project_id)),
        kind="telemetry_assistant",
        title=title,
        message=msg,
        payload_json=_dumps(payload),
    )
    session.add(n)
    return True
