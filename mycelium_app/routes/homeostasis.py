from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.homeostasis import tick_homeostasis
from mycelium_app.models import HomeostasisState, ProjectMember, User
from mycelium_app.schemas import HomeostasisStatusResponse, HomeostasisTickResponse


router = APIRouter(prefix="/api/nexus/homeostasis", tags=["homeostasis"])


def _ensure_project_access(session: Session, user_id: int, project_id: int | None) -> None:
    if project_id is None:
        return
    member = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == project_id, ProjectMember.user_id == user_id)
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a project member")


def _loads_dict(s: str | None) -> dict:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


@router.get("/status", response_model=HomeostasisStatusResponse)
def status(
    project_id: int | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, project_id)

    q = select(HomeostasisState).where(HomeostasisState.user_id == user_id)
    if project_id is None:
        q = q.where(HomeostasisState.project_id.is_(None))
    else:
        q = q.where(HomeostasisState.project_id == project_id)

    row = session.exec(q).first()
    if not row:
        return HomeostasisStatusResponse(ok=True, state=None)

    return HomeostasisStatusResponse(
        ok=True,
        state={
            "updated_at": row.updated_at,
            "project_id": row.project_id,
            "mood": row.mood,
            "mood_signal": _loads_dict(row.mood_signal_json),
            "identity_hash": row.identity_hash,
            "agitated_cycles": int(row.agitated_cycles),
            "last_deep_breath_at": row.last_deep_breath_at,
            "last_identity_backup_at": row.last_identity_backup_at,
            "disk_total_bytes": int(row.disk_total_bytes),
            "disk_free_bytes": int(row.disk_free_bytes),
            "venv_present": bool(row.venv_present),
            "notes": row.notes,
        },
    )


@router.post("/tick", response_model=HomeostasisTickResponse)
def tick(
    project_id: int | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, project_id)

    res = tick_homeostasis(session, user_id=user_id, project_id=project_id)

    return HomeostasisTickResponse(
        ok=True,
        mood=str(res.state.mood),
        identity_hash=str(res.state.identity_hash),
        actions=res.actions,
    )
