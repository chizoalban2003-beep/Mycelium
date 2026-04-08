from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.growth import compute_growth_stage
from mycelium_app.models import GrowthLedgerEntry, ProjectMember, User
from mycelium_app.stimulus import record_stimulus_event
from mycelium_app.schemas import (
    GrowthRecentResponse,
    GrowthStatusResponse,
    GrowthSweepPublic,
    GrowthSweepRecordRequest,
    GrowthSweepRecordResponse,
)
from mycelium_app.settings import settings


router = APIRouter(prefix="/api/nexus/growth", tags=["growth"])


def _ensure_project_access(session: Session, user_id: int, project_id: int | None) -> None:
    if project_id is None:
        return
    member = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == project_id, ProjectMember.user_id == user_id)
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a project member")


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


def _to_public(row: GrowthLedgerEntry) -> GrowthSweepPublic:
    return GrowthSweepPublic(
        created_at=row.created_at,
        device_id=row.device_id,
        project_id=row.project_id,
        domain=row.domain,
        metric=row.metric,
        score=float(row.score),
        accepted=bool(row.accepted),
        notes=row.notes,
        proposal=_loads_dict(row.proposal_json),
        outcome=_loads_dict(row.outcome_json),
    )


@router.post("/sweep", response_model=GrowthSweepRecordResponse)
def record_sweep(
    payload: GrowthSweepRecordRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, payload.project_id)

    domain = (payload.domain or "").strip().lower()[:64]
    metric = (payload.metric or "").strip().lower()[:32]
    if not domain or not metric:
        raise HTTPException(status_code=400, detail="domain and metric are required")

    try:
        score = float(payload.score)
    except Exception:
        raise HTTPException(status_code=400, detail="score must be a number")

    device_id = (payload.device_id or settings.nexus_device_id or "local").strip()[:64]

    row = GrowthLedgerEntry(
        created_by_user_id=user_id,
        project_id=payload.project_id,
        device_id=device_id,
        domain=domain,
        metric=metric,
        score=score,
        accepted=bool(payload.accepted),
        proposal_json=_dumps(payload.proposal or {}),
        outcome_json=_dumps(payload.outcome or {}),
        notes=(payload.notes or "")[:500],
    )
    session.add(row)
    session.commit()
    session.refresh(row)

    try:
        record_stimulus_event(
            session,
            user_id=user_id,
            project_id=payload.project_id,
            device_id=device_id,
            source="growth_api",
            modality="growth",
            signal_type="growth_sweep_record",
            stimulus={"domain": domain, "metric": metric, "score": score, "accepted": bool(payload.accepted)},
            occurred_at=row.created_at,
        )
    except Exception:
        pass

    return GrowthSweepRecordResponse(ok=True, entry_id=int(row.id or 0))


@router.get("/recent", response_model=GrowthRecentResponse)
def recent(
    limit: int = 50,
    project_id: int | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, project_id)

    limit = max(1, min(int(limit), 500))
    q = select(GrowthLedgerEntry).where(GrowthLedgerEntry.created_by_user_id == user_id)
    if project_id is not None:
        q = q.where(GrowthLedgerEntry.project_id == project_id)
    q = q.order_by(GrowthLedgerEntry.created_at.desc()).limit(limit)

    rows = session.exec(q).all()

    try:
        record_stimulus_event(
            session,
            user_id=user_id,
            project_id=project_id,
            device_id=str(settings.nexus_device_id or "local"),
            source="growth_api",
            modality="growth",
            signal_type="growth_recent_view",
            stimulus={"limit": limit, "entries_count": len(rows)},
            occurred_at=datetime.utcnow(),
        )
    except Exception:
        pass

    return GrowthRecentResponse(entries=[_to_public(r) for r in rows])


@router.get("/status", response_model=GrowthStatusResponse)
def status(
    project_id: int | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, project_id)

    stage, unlocked, stats = compute_growth_stage(session, user_id=user_id, project_id=project_id)

    try:
        record_stimulus_event(
            session,
            user_id=user_id,
            project_id=project_id,
            device_id=str(settings.nexus_device_id or "local"),
            source="growth_api",
            modality="growth",
            signal_type="growth_status_view",
            stimulus={"stage": stage, "unlocked_features_count": len(unlocked)},
            occurred_at=datetime.utcnow(),
        )
    except Exception:
        pass

    return GrowthStatusResponse(
        ok=True,
        stage=stage,
        unlocked_features=unlocked,
        stats=stats,
        motto=str(settings.system_motto),
    )
