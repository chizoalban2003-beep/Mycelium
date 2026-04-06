from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import NexusNudge, User
from mycelium_app.schemas import NexusNudgeAckRequest, NexusNudgeAckResponse, NexusNudgeListResponse, NexusNudgePublic


router = APIRouter(prefix="/api/nexus/nudges", tags=["nudges"])


def _loads_dict(s: str | None) -> dict:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _to_public(row: NexusNudge) -> NexusNudgePublic:
    return NexusNudgePublic(
        id=int(row.id or 0),
        created_at=row.created_at,
        project_id=row.project_id,
        kind=str(row.kind or ""),
        title=str(row.title or ""),
        message=str(row.message or ""),
        payload=_loads_dict(row.payload_json),
        seen_at=row.seen_at,
    )


@router.get("/recent", response_model=NexusNudgeListResponse)
def list_recent(
    limit: int = 10,
    unseen_only: bool = True,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    limit = max(1, min(int(limit), 50))

    q = select(NexusNudge).where(NexusNudge.created_by_user_id == user_id)
    if bool(unseen_only):
        q = q.where(NexusNudge.seen_at.is_(None))
    q = q.order_by(NexusNudge.created_at.desc()).limit(limit)

    rows = session.exec(q).all()
    return NexusNudgeListResponse(nudges=[_to_public(r) for r in rows])


@router.post("/ack", response_model=NexusNudgeAckResponse)
def ack(
    payload: NexusNudgeAckRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    nudge_id = int(payload.nudge_id)
    if nudge_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid nudge_id")

    row = session.exec(
        select(NexusNudge).where(NexusNudge.id == nudge_id, NexusNudge.created_by_user_id == user_id)
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")

    if row.seen_at is None:
        row.seen_at = datetime.utcnow()
        session.add(row)
        session.commit()

    return NexusNudgeAckResponse(ok=True)
