from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.assistant_profile import get_assistant_profile_effective, set_assistant_profile
from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import ProjectMember, ProjectRole, User
from mycelium_app.schemas import AssistantProfilePublic, AssistantProfileUpdateRequest
from mycelium_app.stimulus import record_stimulus_event


router = APIRouter(prefix="/api/nexus/assistant", tags=["assistant"])


def _ensure_project_owner(session: Session, user_id: int, project_id: int | None) -> None:
    if project_id is None:
        return
    member = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == int(project_id), ProjectMember.user_id == int(user_id))
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a project member")
    try:
        role = member.role if isinstance(member.role, ProjectRole) else ProjectRole(str(member.role))
    except Exception:
        raise HTTPException(status_code=403, detail="Owner role required")
    if role != ProjectRole.owner:
        raise HTTPException(status_code=403, detail="Owner role required")


@router.get("/configure", response_model=AssistantProfilePublic)
def get_config(
    project_id: int | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_owner(session, user_id, project_id)

    p = get_assistant_profile_effective(session, user_id=user_id, project_id=project_id)
    return AssistantProfilePublic(
        ok=True,
        project_id=p.get("project_id"),
        given_name=str(p.get("given_name", "Myco")),
        gender_identity=str(p.get("gender_identity", "neutral")),
        vocal_preset=str(p.get("vocal_preset", "alloy")),
        assistant_avatar_url=str(p.get("assistant_avatar_url", "")),
        created_at=p.get("created_at"),
        updated_at=p.get("updated_at"),
        is_default=bool(p.get("is_default", True)),
    )


@router.post("/configure", response_model=AssistantProfilePublic)
def configure_assistant(
    payload: AssistantProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_owner(session, user_id, payload.project_id)

    row = set_assistant_profile(
        session,
        user_id=user_id,
        project_id=payload.project_id,
        given_name=payload.given_name,
        gender_identity=payload.gender_identity,
        vocal_preset=payload.vocal_preset,
        assistant_avatar_url=payload.assistant_avatar_url,
    )
    try:
        record_stimulus_event(
            session,
            user_id=user_id,
            project_id=payload.project_id,
            device_id="local",
            source="assistant_api",
            modality="identity",
            signal_type="assistant_profile_update",
            stimulus={
                "given_name_len": len(str(payload.given_name or "")),
                "gender_identity": str(payload.gender_identity or ""),
                "vocal_preset": str(payload.vocal_preset or ""),
                "has_avatar": bool(str(payload.assistant_avatar_url or "").strip()),
            },
            occurred_at=row.updated_at,
        )
    except Exception:
        pass
    p = get_assistant_profile_effective(session, user_id=user_id, project_id=payload.project_id)
    return AssistantProfilePublic(
        ok=True,
        project_id=row.project_id,
        given_name=str(row.given_name or "Myco"),
        gender_identity=str(row.gender_identity or "neutral"),
        vocal_preset=str(row.vocal_preset or "alloy"),
        assistant_avatar_url=str(p.get("assistant_avatar_url", "")),
        created_at=row.created_at,
        updated_at=row.updated_at,
        is_default=False,
    )
