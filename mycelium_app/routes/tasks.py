from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import (
    AdaptiveMemoryEntry,
    GrowthLedgerEntry,
    HiveOutboxMessage,
    ProjectMember,
    ProjectRole,
    TaskReplica,
    TaskTrajectory,
    User,
)
from mycelium_app.parental_policy import get_policy, set_policy
from mycelium_app.schemas import (
    TaskBootstrapWorkSessionRequest,
    TaskBootstrapWorkSessionResponse,
    TaskReplicaAckRequest,
    TaskReplicaAckResponse,
    TaskReplicaDecisionRequest,
    TaskReplicaDecisionResponse,
    TaskReplicaListResponse,
    TaskReplicaProposeRequest,
    TaskReplicaProposeResponse,
    TaskReplicaPublic,
    TaskReplicaVerifyRequest,
    TaskReplicaFeedbackSummaryResponse,
    TaskReplicaExplainResponse,
    TaskActionKillSwitchRequest,
    TaskActionKillSwitchResponse,
    TaskActionReplayRequest,
    TaskActionReplayResponse,
    TaskActionAuditItem,
    TaskActionAuditTimelineResponse,
    TaskReplicaVerifyResponse,
    TaskTrajectoryRecordRequest,
    TaskTrajectoryRecordResponse,
)
from mycelium_app.settings import settings


router = APIRouter(prefix="/api/nexus/tasks", tags=["tasks"])


_ALLOWED_FEEDBACK_LABELS = {
    "helpful",
    "timely",
    "annoying",
    "wrong_device",
    "too_early",
    "too_late",
    "too_long",
    "too_short",
    "interruptive",
}


def _dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads_dict(s: str | None) -> dict:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _ensure_project_access(session: Session, user_id: int, project_id: int | None) -> ProjectRole | None:
    if project_id is None:
        return None
    member = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == int(project_id), ProjectMember.user_id == int(user_id))
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a project member")
    try:
        return member.role if isinstance(member.role, ProjectRole) else ProjectRole(str(member.role))
    except Exception:
        return None


def _normalize_feedback_labels(labels: list[str] | None) -> list[str]:
    if not labels:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in labels:
        label = str(raw or "").strip().lower()[:64]
        if not label:
            continue
        if label not in _ALLOWED_FEEDBACK_LABELS:
            continue
        if label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out[:12]


def _trajectory_key_from_sequence(sequence: list[str]) -> str:
    normalized = [str(x).strip().lower()[:96] for x in sequence if str(x).strip()]
    raw = "|".join(normalized)
    if not raw:
        raw = "empty"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _clamp01(x: float) -> float:
    return max(0.0, min(float(x), 1.0))


def _permission_tier_for_capability(actions_cfg: dict[str, object], capability: str) -> str:
    caps = actions_cfg.get("permission_tiers") if isinstance(actions_cfg.get("permission_tiers"), dict) else {}
    key = str(capability or "").strip().lower()
    raw = caps.get(key, actions_cfg.get("default_permission_tier", "execute"))
    tier = str(raw or "execute").strip().lower()
    if tier not in {"suggest", "queue", "execute"}:
        tier = "execute"
    return tier


def _capability_from_action_payload(payload: dict[str, object]) -> str:
    cap = str(payload.get("capability") or "").strip().lower()[:64]
    if cap:
        return cap
    action_id = str(payload.get("action_id") or "").strip().lower()[:64]
    if action_id == "device_start_focus_session":
        return "start_focus_session"
    return ""


def _audit_gates_for_action(
    *,
    actions_cfg: dict[str, object],
    capability: str,
    confidence: float,
) -> list[str]:
    gates: list[str] = []
    if bool(actions_cfg.get("kill_switch", False)):
        gates.append("kill_switch_enabled")
    if not bool(actions_cfg.get("enabled", False)):
        gates.append("actions_disabled")
    if not bool(actions_cfg.get("device_control_enabled", False)):
        gates.append("device_control_disabled")

    caps = actions_cfg.get("allowed_capabilities") if isinstance(actions_cfg.get("allowed_capabilities"), list) else []
    allowed = {str(c).strip().lower() for c in caps if str(c).strip()}
    if allowed and capability and capability not in allowed:
        gates.append("capability_not_allowed")

    tier = _permission_tier_for_capability(actions_cfg, capability)
    if tier == "suggest":
        gates.append("permission_tier_suggest")

    try:
        min_conf = float(actions_cfg.get("min_confidence", 0.90))
    except Exception:
        min_conf = 0.90
    min_conf = max(0.0, min(min_conf, 1.0))
    if float(confidence) < min_conf:
        gates.append("confidence_below_minimum")
    return gates


def _memory_upsert(
    session: Session,
    *,
    user_id: int,
    project_id: int | None,
    lane: str,
    memory_key: str,
    source: str,
    content: dict[str, object],
    tags: list[str],
    strength_delta: float,
    decay_half_life_hours: float,
    device_id: str,
) -> None:
    q = (
        select(AdaptiveMemoryEntry)
        .where(AdaptiveMemoryEntry.created_by_user_id == int(user_id))
        .where(AdaptiveMemoryEntry.project_id == project_id)
        .where(AdaptiveMemoryEntry.lane == str(lane))
        .where(AdaptiveMemoryEntry.memory_key == str(memory_key))
    )
    row = session.exec(q.order_by(AdaptiveMemoryEntry.updated_at.desc())).first()

    now = datetime.utcnow()
    tags_norm = [str(t).strip().lower()[:64] for t in tags if str(t).strip()][:20]
    half_life = max(1.0, min(float(decay_half_life_hours), 24.0 * 365.0))

    if row is None:
        row = AdaptiveMemoryEntry(
            created_by_user_id=int(user_id),
            project_id=project_id,
            device_id=str(device_id or settings.nexus_device_id or "local")[:128],
            lane=str(lane),
            memory_key=str(memory_key)[:128],
            source=str(source)[:64],
            content_json=_dumps(content),
            tags_json=_dumps(tags_norm),
            strength=_clamp01(0.5 + float(strength_delta)),
            decay_half_life_hours=float(half_life),
            last_reinforced_at=now,
            last_accessed_at=now,
            updated_at=now,
        )
        session.add(row)
        return

    row.source = str(source)[:64]
    row.content_json = _dumps(content)
    row.tags_json = _dumps(tags_norm)
    row.decay_half_life_hours = float(half_life)
    row.strength = _clamp01(float(row.strength or 0.0) + float(strength_delta))
    row.last_reinforced_at = now
    row.last_accessed_at = now
    row.updated_at = now
    session.add(row)


def _to_replica_public(row: TaskReplica) -> TaskReplicaPublic:
    return TaskReplicaPublic(
        id=int(row.id or 0),
        created_at=row.created_at,
        updated_at=row.updated_at,
        project_id=row.project_id,
        device_id=str(row.device_id or ""),
        title=str(row.title or ""),
        trajectory_key=str(row.trajectory_key or ""),
        consensus_fraction=float(row.consensus_fraction or 0.0),
        species_confidence=float(row.species_confidence or 0.0),
        capability=str(row.capability or ""),
        status=str(row.status or ""),
        command=_loads_dict(row.command_json),
        notes=str(row.notes or ""),
    )


def approve_replica_and_queue(
    session: Session,
    *,
    user_id: int,
    row: TaskReplica,
    device_id: str | None = None,
    auto_execute: bool = False,
) -> tuple[int, str, bool]:
    """Approve a replica and queue execution with idempotent outbox behavior.

    Returns: (message_id, detail, reused_existing_message)
    """

    policy = get_policy(session, user_id)
    actions_cfg = policy.get("actions") if isinstance(policy.get("actions"), dict) else {}
    assistant_cfg = policy.get("assistant") if isinstance(policy.get("assistant"), dict) else {}
    persona_mode = str(assistant_cfg.get("persona_mode", "calm")).strip().lower()
    if persona_mode not in {"coach", "calm", "briefing"}:
        persona_mode = "calm"
    if bool(actions_cfg.get("kill_switch", False)):
        raise HTTPException(status_code=423, detail="Action kill-switch is enabled")
    if not bool(actions_cfg.get("enabled", False)):
        raise HTTPException(status_code=403, detail="Actions disabled by user policy")
    if not bool(actions_cfg.get("device_control_enabled", False)):
        raise HTTPException(status_code=403, detail="Device control not granted by user policy")

    permission_tier = _permission_tier_for_capability(actions_cfg, str(row.capability or ""))
    if permission_tier == "suggest":
        raise HTTPException(status_code=403, detail="Capability is configured as suggest-only")
    if bool(auto_execute) and permission_tier != "execute":
        raise HTTPException(status_code=403, detail="Capability tier blocks autonomous execution")

    caps = actions_cfg.get("allowed_capabilities") if isinstance(actions_cfg.get("allowed_capabilities"), list) else []
    allowed = {str(c).strip().lower() for c in caps if str(c).strip()}
    if allowed and str(row.capability or "").lower() not in allowed:
        raise HTTPException(status_code=403, detail="Capability not allowed by user policy")

    try:
        min_conf_raw = float(actions_cfg.get("min_confidence", 0.90))
    except Exception:
        min_conf_raw = 0.90
    min_conf = max(0.0, min(min_conf_raw, 1.0))
    if float(row.species_confidence or 0.0) < min_conf:
        raise HTTPException(
            status_code=409,
            detail=f"Species confidence below policy minimum ({float(row.species_confidence):.3f} < {min_conf:.3f})",
        )

    # Idempotency guard: if this replica already has a pending device_action in outbox,
    # reuse it instead of enqueueing a duplicate.
    existing_q = (
        select(HiveOutboxMessage)
        .where(HiveOutboxMessage.created_by_user_id == int(user_id))
        .where(HiveOutboxMessage.kind == "device_action")
        .where(HiveOutboxMessage.submitted_at.is_(None))
        .order_by(HiveOutboxMessage.created_at.desc())
        .limit(200)
    )
    existing_rows = session.exec(existing_q).all()
    action_id = f"task_replica:{int(row.id or 0)}"
    for m in existing_rows:
        payload = _loads_dict(m.payload_json)
        if str(payload.get("action_id") or "") == action_id:
            row.status = "approved"
            if row.approved_at is None:
                row.approved_at = datetime.utcnow()
            row.updated_at = datetime.utcnow()
            session.add(row)
            session.commit()
            return int(m.id or 0), "Replica already queued; reused existing device action", True

    row.status = "approved"
    row.approved_at = datetime.utcnow()
    row.updated_at = datetime.utcnow()
    session.add(row)

    message = HiveOutboxMessage(
        created_by_user_id=user_id,
        project_id=row.project_id,
        device_id=str(device_id or row.device_id or settings.nexus_device_id or "local")[:128],
        kind="device_action",
        payload_json=_dumps(
            {
                "action_id": action_id,
                "confidence": float(round(float(row.species_confidence or 0.0), 4)),
                "command": _loads_dict(row.command_json),
                "capability": str(row.capability or ""),
                "title": str(row.title or ""),
                "trajectory_key": str(row.trajectory_key or ""),
                "autonomy_mode": _loads_dict(row.command_json).get("autonomy", {}).get("mode"),
                "auto_confirmed": bool(_loads_dict(row.command_json).get("autonomy", {}).get("auto_confirmed", False)),
                "gate_reason": _loads_dict(row.command_json).get("autonomy", {}).get("gate_reason"),
                "permission_tier": permission_tier,
                "explain": {
                    "why_now": _loads_dict(row.command_json).get("autonomy", {}).get("gate_reason"),
                    "why_device": (
                        _loads_dict(row.command_json).get("handoff", {}).get("recommended_device_id")
                        or str(device_id or row.device_id or settings.nexus_device_id or "local")[:128]
                    ),
                },
                "persona_mode": persona_mode,
            }
        ),
        submitted_at=None,
    )
    session.add(message)
    session.commit()
    session.refresh(message)
    return int(message.id or 0), "Replica approved and queued for local execution", False


@router.post("/bootstrap/work-session", response_model=TaskBootstrapWorkSessionResponse)
def bootstrap_work_session(
    payload: TaskBootstrapWorkSessionRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Seed the first recommended directive: prepare a focused work session.

    This creates:
    - one trajectory template (behavioral pattern)
    - one task replica proposal (executable command)
    """

    user_id = int(current_user.id or 0)
    role = _ensure_project_access(session, user_id, payload.project_id)
    if payload.project_id is not None and role not in {ProjectRole.owner, ProjectRole.editor}:
        raise HTTPException(status_code=403, detail="Owner or editor role required")

    duration = max(10, min(int(payload.duration_minutes), 180))
    focus_app = str(payload.focus_app or "mycelium").strip().lower()[:64] or "mycelium"

    sequence = [
        "session_start_detected",
        "open_mycelium_dashboard",
        "open_focus_app",
        "enable_dnd",
        "set_focus_timer",
    ]
    trajectory_key = _trajectory_key_from_sequence(sequence)

    trajectory = TaskTrajectory(
        created_by_user_id=user_id,
        project_id=payload.project_id,
        device_id=str(payload.device_id or settings.nexus_device_id or "local")[:64],
        trajectory_key=trajectory_key,
        sequence_json=_dumps(sequence),
        app_state_json=_dumps({"mode": "work_session", "focus_app": focus_app}),
        input_vector_json=_dumps({"trigger": "manual_bootstrap", "time_block_minutes": duration}),
        confidence=max(0.0, min(float(payload.species_confidence), 1.0)),
        support_count=1,
    )
    session.add(trajectory)
    session.flush()

    replica = TaskReplica(
        created_by_user_id=user_id,
        project_id=payload.project_id,
        device_id=str(payload.device_id or settings.nexus_device_id or "local")[:64],
        title="Prepare Focus Work Session",
        trajectory_key=trajectory_key,
        consensus_fraction=max(0.0, min(float(payload.consensus_fraction), 1.0)),
        species_confidence=max(0.0, min(float(payload.species_confidence), 1.0)),
        capability="start_focus_session",
        command_json=_dumps(
            {
                "op": "focus_session",
                "duration_minutes": duration,
                "enable_dnd": True,
                "open_app": focus_app,
                "open_dashboard": True,
            }
        ),
        status="proposed",
        notes="Bootstrapped starter directive for first-time behavioral mirroring.",
    )
    session.add(replica)
    session.commit()
    session.refresh(trajectory)
    session.refresh(replica)

    return TaskBootstrapWorkSessionResponse(
        ok=True,
        trajectory_id=int(trajectory.id or 0),
        trajectory_key=trajectory_key,
        replica=_to_replica_public(replica),
    )


@router.post("/trajectory/record", response_model=TaskTrajectoryRecordResponse)
def record_trajectory(
    payload: TaskTrajectoryRecordRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, payload.project_id)

    seq = [str(x).strip()[:96] for x in (payload.sequence or []) if str(x).strip()][:200]
    if not seq:
        raise HTTPException(status_code=400, detail="sequence is required")

    trajectory_key = str(payload.trajectory_key or "").strip()[:64] or _trajectory_key_from_sequence(seq)
    conf = max(0.0, min(float(payload.confidence or 0.0), 1.0))
    support = max(1, min(int(payload.support_count or 1), 100_000))

    row = TaskTrajectory(
        created_by_user_id=user_id,
        project_id=payload.project_id,
        device_id=str(payload.device_id or settings.nexus_device_id or "local")[:64],
        trajectory_key=trajectory_key,
        sequence_json=_dumps(seq),
        app_state_json=_dumps(payload.app_state or {}),
        input_vector_json=_dumps(payload.input_vector or {}),
        confidence=conf,
        support_count=support,
    )
    session.add(row)
    session.commit()
    session.refresh(row)

    return TaskTrajectoryRecordResponse(ok=True, trajectory_id=int(row.id or 0), trajectory_key=trajectory_key)


@router.post("/replicas/propose", response_model=TaskReplicaProposeResponse)
def propose_replica(
    payload: TaskReplicaProposeRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    role = _ensure_project_access(session, user_id, payload.project_id)
    if payload.project_id is not None and role not in {ProjectRole.owner, ProjectRole.editor}:
        raise HTTPException(status_code=403, detail="Owner or editor role required")

    title = str(payload.title or "").strip()[:120]
    if not title:
        raise HTTPException(status_code=400, detail="title is required")

    trajectory_key = str(payload.trajectory_key or "").strip()[:64]
    if not trajectory_key:
        raise HTTPException(status_code=400, detail="trajectory_key is required")

    capability = str(payload.capability or "").strip().lower()[:64]
    if not capability:
        raise HTTPException(status_code=400, detail="capability is required")

    replica = TaskReplica(
        created_by_user_id=user_id,
        project_id=payload.project_id,
        device_id=str(payload.device_id or settings.nexus_device_id or "local")[:64],
        title=title,
        trajectory_key=trajectory_key,
        consensus_fraction=max(0.0, min(float(payload.consensus_fraction or 0.0), 1.0)),
        species_confidence=max(0.0, min(float(payload.species_confidence or 0.0), 1.0)),
        capability=capability,
        command_json=_dumps(payload.command or {}),
        status="proposed",
        notes=str(payload.notes or "")[:1000],
    )
    session.add(replica)
    session.commit()
    session.refresh(replica)

    return TaskReplicaProposeResponse(ok=True, replica=_to_replica_public(replica))


@router.get("/replicas/recent", response_model=TaskReplicaListResponse)
def list_replicas(
    limit: int = 50,
    status: str | None = None,
    project_id: int | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, project_id)

    lim = max(1, min(int(limit), 500))
    q = select(TaskReplica).where(TaskReplica.created_by_user_id == user_id)
    if project_id is not None:
        q = q.where(TaskReplica.project_id == int(project_id))
    if status:
        q = q.where(TaskReplica.status == str(status).strip().lower()[:32])
    q = q.order_by(TaskReplica.created_at.desc()).limit(lim)

    rows = session.exec(q).all()
    return TaskReplicaListResponse(replicas=[_to_replica_public(r) for r in rows])


@router.post("/replicas/{replica_id}/decision", response_model=TaskReplicaDecisionResponse)
def replica_decision(
    replica_id: int,
    payload: TaskReplicaDecisionRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    row = session.exec(
        select(TaskReplica).where(TaskReplica.id == int(replica_id), TaskReplica.created_by_user_id == user_id)
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Replica not found")

    role = _ensure_project_access(session, user_id, row.project_id)
    if row.project_id is not None and role not in {ProjectRole.owner, ProjectRole.editor}:
        raise HTTPException(status_code=403, detail="Owner or editor role required")

    decision = str(payload.decision or "approve").strip().lower()
    if decision not in {"approve", "reject"}:
        raise HTTPException(status_code=400, detail="decision must be approve|reject")

    row.updated_at = datetime.utcnow()
    if decision == "reject":
        row.status = "rejected"
        row.updated_at = datetime.utcnow()
        session.add(row)
        session.commit()
        return TaskReplicaDecisionResponse(
            ok=True,
            replica_id=int(row.id or 0),
            decision=decision,
            queued_device_action_id=None,
            detail="Replica rejected",
        )

    message_id, detail, _reused = approve_replica_and_queue(
        session,
        user_id=user_id,
        row=row,
        device_id=payload.device_id,
        auto_execute=False,
    )

    return TaskReplicaDecisionResponse(
        ok=True,
        replica_id=int(row.id or 0),
        decision=decision,
        queued_device_action_id=int(message_id or 0),
        detail=detail,
    )


@router.get("/replicas/{replica_id}/explain", response_model=TaskReplicaExplainResponse)
def replica_explain(
    replica_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    row = session.exec(
        select(TaskReplica).where(TaskReplica.id == int(replica_id), TaskReplica.created_by_user_id == user_id)
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Replica not found")

    _ensure_project_access(session, user_id, row.project_id)

    policy = get_policy(session, user_id)
    actions_cfg = policy.get("actions") if isinstance(policy.get("actions"), dict) else {}

    caps = actions_cfg.get("allowed_capabilities") if isinstance(actions_cfg.get("allowed_capabilities"), list) else []
    allowed = {str(c).strip().lower() for c in caps if str(c).strip()}

    try:
        min_conf = float(actions_cfg.get("min_confidence", 0.90))
    except Exception:
        min_conf = 0.90
    min_conf = max(0.0, min(min_conf, 1.0))

    capability = str(row.capability or "").strip().lower()
    tier = _permission_tier_for_capability(actions_cfg, capability)
    gates: list[str] = []
    if bool(actions_cfg.get("kill_switch", False)):
        gates.append("kill_switch_enabled")
    if not bool(actions_cfg.get("enabled", False)):
        gates.append("actions_disabled")
    if not bool(actions_cfg.get("device_control_enabled", False)):
        gates.append("device_control_disabled")
    if allowed and capability not in allowed:
        gates.append("capability_not_allowed")
    if float(row.species_confidence or 0.0) < min_conf:
        gates.append("confidence_below_minimum")
    if tier == "suggest":
        gates.append("permission_tier_suggest")

    cmd = _loads_dict(row.command_json)
    autonomy = cmd.get("autonomy") if isinstance(cmd.get("autonomy"), dict) else {}
    handoff = cmd.get("handoff") if isinstance(cmd.get("handoff"), dict) else {}

    return TaskReplicaExplainResponse(
        ok=True,
        replica_id=int(row.id or 0),
        status=str(row.status or ""),
        capability=capability,
        species_confidence=float(round(float(row.species_confidence or 0.0), 6)),
        policy_min_confidence=float(round(min_conf, 6)),
        permission_tier=tier,
        kill_switch=bool(actions_cfg.get("kill_switch", False)),
        autonomy={
            "mode": autonomy.get("mode"),
            "auto_confirmed": bool(autonomy.get("auto_confirmed", False)),
            "gate_reason": autonomy.get("gate_reason"),
            "recommended_device_id": handoff.get("recommended_device_id"),
            "handoff_recommended": bool(handoff.get("handoff_recommended", False)),
        },
        gates=gates,
        recommended_decision=("approve" if not gates else "reject"),
    )


@router.post("/actions/kill-switch", response_model=TaskActionKillSwitchResponse)
def set_action_kill_switch(
    payload: TaskActionKillSwitchRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    if payload.project_id is not None:
        role = _ensure_project_access(session, user_id, payload.project_id)
        if role not in {ProjectRole.owner, ProjectRole.editor}:
            raise HTTPException(status_code=403, detail="Owner or editor role required")

    policy = get_policy(session, user_id)
    actions_cfg = policy.get("actions") if isinstance(policy.get("actions"), dict) else {}
    updated_policy = {
        **policy,
        "actions": {
            **actions_cfg,
            "kill_switch": bool(payload.enabled),
        },
    }
    set_policy(session, user_id, updated_policy)

    cleared = 0
    if bool(payload.clear_pending):
        q = (
            select(HiveOutboxMessage)
            .where(HiveOutboxMessage.created_by_user_id == user_id)
            .where(HiveOutboxMessage.kind == "device_action")
            .where(HiveOutboxMessage.submitted_at.is_(None))
        )
        if payload.project_id is None:
            q = q.where(HiveOutboxMessage.project_id.is_(None))
        else:
            q = q.where(HiveOutboxMessage.project_id == int(payload.project_id))

        rows = session.exec(q.order_by(HiveOutboxMessage.created_at.asc()).limit(2000)).all()
        now = datetime.utcnow()
        for m in rows:
            p = _loads_dict(m.payload_json)
            p["ack"] = {
                "status": "rejected",
                "notes": "cleared_by_kill_switch",
                "acked_at": now.isoformat() + "Z",
            }
            m.payload_json = _dumps(p)
            m.submitted_at = now
            session.add(m)
            cleared += 1
        session.commit()

    return TaskActionKillSwitchResponse(ok=True, enabled=bool(payload.enabled), cleared_pending=int(cleared))


@router.post("/actions/{message_id}/replay", response_model=TaskActionReplayResponse)
def replay_device_action(
    message_id: int,
    payload: TaskActionReplayRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    msg = session.exec(
        select(HiveOutboxMessage).where(
            HiveOutboxMessage.id == int(message_id),
            HiveOutboxMessage.created_by_user_id == int(user_id),
            HiveOutboxMessage.kind == "device_action",
        )
    ).first()
    if not msg:
        raise HTTPException(status_code=404, detail="Device action not found")

    role = _ensure_project_access(session, user_id, msg.project_id)
    if msg.project_id is not None and role not in {ProjectRole.owner, ProjectRole.editor}:
        raise HTTPException(status_code=403, detail="Owner or editor role required")

    policy = get_policy(session, user_id)
    actions_cfg = policy.get("actions") if isinstance(policy.get("actions"), dict) else {}
    if bool(actions_cfg.get("kill_switch", False)):
        raise HTTPException(status_code=423, detail="Action kill-switch is enabled")

    old_payload = _loads_dict(msg.payload_json)
    capability = _capability_from_action_payload(old_payload)
    if capability:
        tier = _permission_tier_for_capability(actions_cfg, capability)
        if tier == "suggest":
            raise HTTPException(status_code=403, detail="Capability is configured as suggest-only")

    replay_payload = dict(old_payload)
    replay_payload.pop("ack", None)
    replay_payload["replay"] = {
        "from_message_id": int(msg.id or 0),
        "requested_at": datetime.utcnow().isoformat() + "Z",
        "reason": str(payload.reason or "manual_replay")[:120],
    }

    new_msg = HiveOutboxMessage(
        created_by_user_id=int(user_id),
        project_id=msg.project_id,
        device_id=str(payload.device_id or msg.device_id or settings.nexus_device_id or "local")[:128],
        kind="device_action",
        payload_json=_dumps(replay_payload),
        submitted_at=None,
    )
    session.add(new_msg)
    session.commit()
    session.refresh(new_msg)

    return TaskActionReplayResponse(
        ok=True,
        original_message_id=int(msg.id or 0),
        replay_message_id=int(new_msg.id or 0),
        detail="Device action replay queued",
    )


@router.get("/actions/audit/timeline", response_model=TaskActionAuditTimelineResponse)
def action_audit_timeline(
    limit: int = 100,
    project_id: int | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, project_id)

    lim = max(1, min(int(limit), 1000))
    q = (
        select(HiveOutboxMessage)
        .where(HiveOutboxMessage.created_by_user_id == int(user_id))
        .where(HiveOutboxMessage.kind == "device_action")
    )
    if project_id is None:
        q = q.where(HiveOutboxMessage.project_id.is_(None))
    else:
        q = q.where(HiveOutboxMessage.project_id == int(project_id))
    rows = session.exec(q.order_by(HiveOutboxMessage.created_at.desc()).limit(lim)).all()

    policy = get_policy(session, user_id)
    actions_cfg = policy.get("actions") if isinstance(policy.get("actions"), dict) else {}
    try:
        min_conf = float(actions_cfg.get("min_confidence", 0.90))
    except Exception:
        min_conf = 0.90
    min_conf = max(0.0, min(min_conf, 1.0))

    items: list[TaskActionAuditItem] = []
    for r in rows:
        payload = _loads_dict(r.payload_json)
        capability = _capability_from_action_payload(payload)
        confidence = _clamp01(float(payload.get("confidence", 0.0) or 0.0))
        tier = _permission_tier_for_capability(actions_cfg, capability)
        gates = _audit_gates_for_action(actions_cfg=actions_cfg, capability=capability, confidence=confidence)

        ack = payload.get("ack") if isinstance(payload.get("ack"), dict) else {}
        status = "pending"
        if r.submitted_at is not None:
            status = str(ack.get("status") or "submitted")[:32]

        items.append(
            TaskActionAuditItem(
                message_id=int(r.id or 0),
                created_at=r.created_at,
                project_id=r.project_id,
                device_id=str(r.device_id or ""),
                action_id=str(payload.get("action_id") or "")[:64],
                capability=capability,
                confidence=float(round(confidence, 6)),
                permission_tier=tier,
                kill_switch=bool(actions_cfg.get("kill_switch", False)),
                min_confidence=float(round(min_conf, 6)),
                status=status,
                gates=gates,
                would_pass_now=(len(gates) == 0),
            )
        )

    return TaskActionAuditTimelineResponse(ok=True, items=items)


@router.post("/replicas/{replica_id}/ack", response_model=TaskReplicaAckResponse)
def replica_ack(
    replica_id: int,
    payload: TaskReplicaAckRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    row = session.exec(
        select(TaskReplica).where(TaskReplica.id == int(replica_id), TaskReplica.created_by_user_id == user_id)
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Replica not found")

    status = str(payload.status or "executed").strip().lower()
    if status not in {"executed", "failed"}:
        raise HTTPException(status_code=400, detail="status must be executed|failed")

    row.status = status
    row.executed_at = datetime.utcnow()
    row.updated_at = datetime.utcnow()
    if str(payload.notes or "").strip():
        row.notes = str(payload.notes or "")[:1000]
    session.add(row)
    session.commit()

    return TaskReplicaAckResponse(ok=True, replica_id=int(row.id or 0), status=status)


@router.post("/replicas/{replica_id}/verify", response_model=TaskReplicaVerifyResponse)
def replica_verify(
    replica_id: int,
    payload: TaskReplicaVerifyRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Report post-execution outcomes so directives can self-tune over time."""

    user_id = int(current_user.id or 0)
    row = session.exec(
        select(TaskReplica).where(TaskReplica.id == int(replica_id), TaskReplica.created_by_user_id == user_id)
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Replica not found")

    planned = max(1, min(int(payload.planned_minutes), 24 * 60))
    focused = max(0, min(int(payload.focused_minutes), 24 * 60))
    interruptions = max(0, min(int(payload.interruption_count), 10_000))
    feedback_labels = _normalize_feedback_labels(payload.feedback_labels)
    adherence = max(0.0, min(float(focused) / float(planned), 1.0))

    accepted = bool(payload.completed) and (adherence >= 0.80) and (not bool(payload.closed_early))

    # Conservative reward shaping for confidence updates.
    base_delta = 0.05 if accepted else -0.05
    scale = (0.50 + (0.50 * adherence)) if accepted else (0.50 + (0.50 * (1.0 - adherence)))
    reward_delta = float(base_delta * scale)

    old_conf = max(0.0, min(float(row.species_confidence or 0.0), 1.0))
    new_conf = max(0.0, min(old_conf + reward_delta, 1.0))

    row.species_confidence = float(new_conf)
    row.updated_at = datetime.utcnow()
    if accepted and str(row.status or "") in {"approved", "executed"}:
        row.status = "executed"
    elif not accepted and str(row.status or "") not in {"executed", "failed", "rejected"}:
        row.status = "failed"

    verify_note = (
        f"verify: planned={planned} focused={focused} adherence={adherence:.3f} "
        f"completed={bool(payload.completed)} closed_early={bool(payload.closed_early)} interruptions={interruptions}"
    )
    if feedback_labels:
        verify_note = f"{verify_note} labels={','.join(feedback_labels)}"
    note_extra = str(payload.notes or "").strip()[:500]
    if note_extra:
        verify_note = f"{verify_note}; note={note_extra}"
    prev = str(row.notes or "").strip()
    row.notes = (f"{prev}\n{verify_note}" if prev else verify_note)[:1000]
    session.add(row)

    growth = GrowthLedgerEntry(
        created_by_user_id=user_id,
        project_id=row.project_id,
        device_id=str(row.device_id or settings.nexus_device_id or "local")[:64],
        domain="task_replica_focus",
        metric="adherence",
        score=float(adherence),
        accepted=bool(accepted),
        proposal_json=_dumps(
            {
                "replica_id": int(row.id or 0),
                "trajectory_key": str(row.trajectory_key or ""),
                "capability": str(row.capability or ""),
            }
        ),
        outcome_json=_dumps(
            {
                "planned_minutes": planned,
                "focused_minutes": focused,
                "completed": bool(payload.completed),
                "closed_early": bool(payload.closed_early),
                "interruption_count": interruptions,
                "feedback_labels": feedback_labels,
                "old_species_confidence": old_conf,
                "new_species_confidence": new_conf,
                "reward_delta": reward_delta,
            }
        ),
        notes="Task replica verification feedback",
    )
    session.add(growth)

    # Adaptive Memory auto-sync (Grow with Data):
    # - episodic: store this concrete verification event
    # - semantic: reinforce durable preference pattern when accepted
    # - procedural: reinforce successful execution recipe
    pattern_bucket = int(round((float(planned) / 5.0))) * 5
    device = str(row.device_id or settings.nexus_device_id or "local")[:64]

    _memory_upsert(
        session,
        user_id=user_id,
        project_id=row.project_id,
        lane="episodic",
        memory_key=f"replica:{int(row.id or 0)}",
        source="task_verify",
        content={
            "replica_id": int(row.id or 0),
            "trajectory_key": str(row.trajectory_key or ""),
            "device_id": device,
            "planned_minutes": planned,
            "focused_minutes": focused,
            "adherence": float(round(adherence, 6)),
            "accepted": bool(accepted),
            "feedback_labels": feedback_labels,
        },
        tags=["task_replica_focus", "episodic"] + feedback_labels,
        strength_delta=(0.10 if accepted else -0.03),
        decay_half_life_hours=72.0,
        device_id=device,
    )

    _memory_upsert(
        session,
        user_id=user_id,
        project_id=row.project_id,
        lane="semantic",
        memory_key=f"focus_pattern:{device}:planned_{pattern_bucket}",
        source="task_verify",
        content={
            "preferred_device_id": device,
            "preferred_minutes": int(pattern_bucket),
            "latest_adherence": float(round(adherence, 6)),
            "latest_feedback_labels": feedback_labels,
            "accepted": bool(accepted),
        },
        tags=["focus", "semantic"] + feedback_labels,
        strength_delta=(0.12 if accepted else -0.05),
        decay_half_life_hours=24.0 * 14.0,
        device_id=device,
    )

    _memory_upsert(
        session,
        user_id=user_id,
        project_id=row.project_id,
        lane="procedural",
        memory_key=f"procedure:{str(row.capability or 'task')}:work_session",
        source="task_verify",
        content={
            "capability": str(row.capability or ""),
            "trajectory_key": str(row.trajectory_key or ""),
            "accepted": bool(accepted),
            "adherence": float(round(adherence, 6)),
            "feedback_labels": feedback_labels,
        },
        tags=["procedure", "work_session"] + feedback_labels,
        strength_delta=(0.10 if accepted else -0.04),
        decay_half_life_hours=24.0 * 30.0,
        device_id=device,
    )

    session.commit()
    session.refresh(growth)

    return TaskReplicaVerifyResponse(
        ok=True,
        replica_id=int(row.id or 0),
        adherence=float(round(adherence, 6)),
        accepted=bool(accepted),
        reward_delta=float(round(reward_delta, 6)),
        updated_species_confidence=float(round(new_conf, 6)),
        feedback_labels=feedback_labels,
        growth_entry_id=int(growth.id or 0),
    )


@router.get("/replicas/feedback/summary", response_model=TaskReplicaFeedbackSummaryResponse)
def replica_feedback_summary(
    window_hours: int = 168,
    project_id: int | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, project_id)

    hours = max(1, min(int(window_hours), 24 * 60))
    since = datetime.utcnow() - timedelta(hours=hours)

    q = (
        select(GrowthLedgerEntry)
        .where(GrowthLedgerEntry.created_by_user_id == user_id)
        .where(GrowthLedgerEntry.domain == "task_replica_focus")
        .where(GrowthLedgerEntry.metric == "adherence")
        .where(GrowthLedgerEntry.created_at >= since)
    )
    if project_id is None:
        q = q.where(GrowthLedgerEntry.project_id.is_(None))
    else:
        q = q.where(GrowthLedgerEntry.project_id == int(project_id))

    rows = session.exec(q.order_by(GrowthLedgerEntry.created_at.desc()).limit(2000)).all()

    label_counts: dict[str, int] = {}
    label_accept_counts: dict[str, int] = {}
    for r in rows:
        outcome = _loads_dict(r.outcome_json)
        labels = outcome.get("feedback_labels") if isinstance(outcome.get("feedback_labels"), list) else []
        accepted = bool(r.accepted)
        for raw in labels:
            label = str(raw or "").strip().lower()[:64]
            if not label:
                continue
            label_counts[label] = int(label_counts.get(label, 0) + 1)
            if accepted:
                label_accept_counts[label] = int(label_accept_counts.get(label, 0) + 1)

    label_acceptance: dict[str, float] = {}
    for label, count in label_counts.items():
        if count <= 0:
            continue
        accepted_n = int(label_accept_counts.get(label, 0))
        label_acceptance[label] = float(round(float(accepted_n) / float(count), 4))

    return TaskReplicaFeedbackSummaryResponse(
        ok=True,
        window_hours=hours,
        total_verified=len(rows),
        label_counts=label_counts,
        label_acceptance=label_acceptance,
    )
