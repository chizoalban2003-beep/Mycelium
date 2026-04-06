from __future__ import annotations

import hashlib
import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import HiveOutboxMessage, ProjectMember, ProjectRole, TaskReplica, TaskTrajectory, User
from mycelium_app.parental_policy import get_policy
from mycelium_app.schemas import (
    TaskReplicaAckRequest,
    TaskReplicaAckResponse,
    TaskReplicaDecisionRequest,
    TaskReplicaDecisionResponse,
    TaskReplicaListResponse,
    TaskReplicaProposeRequest,
    TaskReplicaProposeResponse,
    TaskReplicaPublic,
    TaskTrajectoryRecordRequest,
    TaskTrajectoryRecordResponse,
)
from mycelium_app.settings import settings


router = APIRouter(prefix="/api/nexus/tasks", tags=["tasks"])


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


def _trajectory_key_from_sequence(sequence: list[str]) -> str:
    normalized = [str(x).strip().lower()[:96] for x in sequence if str(x).strip()]
    raw = "|".join(normalized)
    if not raw:
        raw = "empty"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


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
        session.add(row)
        session.commit()
        return TaskReplicaDecisionResponse(
            ok=True,
            replica_id=int(row.id or 0),
            decision=decision,
            queued_device_action_id=None,
            detail="Replica rejected",
        )

    policy = get_policy(session, user_id)
    actions_cfg = policy.get("actions") if isinstance(policy.get("actions"), dict) else {}
    if not bool(actions_cfg.get("enabled", False)):
        raise HTTPException(status_code=403, detail="Actions disabled by user policy")
    if not bool(actions_cfg.get("device_control_enabled", False)):
        raise HTTPException(status_code=403, detail="Device control not granted by user policy")

    caps = actions_cfg.get("allowed_capabilities") if isinstance(actions_cfg.get("allowed_capabilities"), list) else []
    allowed = {str(c).strip().lower() for c in caps if str(c).strip()}
    if allowed and str(row.capability or "").lower() not in allowed:
        raise HTTPException(status_code=403, detail="Capability not allowed by user policy")

    min_conf = max(0.0, min(float(actions_cfg.get("min_confidence", 0.90) or 0.90), 1.0))
    if float(row.species_confidence or 0.0) < min_conf:
        raise HTTPException(
            status_code=409,
            detail=f"Species confidence below policy minimum ({float(row.species_confidence):.3f} < {min_conf:.3f})",
        )

    row.status = "approved"
    row.approved_at = datetime.utcnow()
    session.add(row)

    message = HiveOutboxMessage(
        created_by_user_id=user_id,
        project_id=row.project_id,
        device_id=str(payload.device_id or row.device_id or settings.nexus_device_id or "local")[:128],
        kind="device_action",
        payload_json=_dumps(
            {
                "action_id": f"task_replica:{int(row.id or 0)}",
                "confidence": float(round(float(row.species_confidence or 0.0), 4)),
                "command": _loads_dict(row.command_json),
                "capability": str(row.capability or ""),
                "title": str(row.title or ""),
                "trajectory_key": str(row.trajectory_key or ""),
            }
        ),
        submitted_at=None,
    )
    session.add(message)
    session.commit()
    session.refresh(message)

    return TaskReplicaDecisionResponse(
        ok=True,
        replica_id=int(row.id or 0),
        decision=decision,
        queued_device_action_id=int(message.id or 0),
        detail="Replica approved and queued for local execution",
    )


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
