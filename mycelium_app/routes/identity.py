from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel import Session

from mycelium_app.assistant_profile import get_assistant_profile_effective, set_assistant_profile
from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.identity_presentation import present_identity
from mycelium_app.schemas import AssistantProfilePublic, AssistantProfileUpdateRequest, IdentityPresentationResponse
from mycelium_app.self_reflection import compute_self_reflection


router = APIRouter(prefix="/api/nexus/identity", tags=["identity"])


@router.get("/presentation", response_model=IdentityPresentationResponse)
def presentation(
    window_days: int = 30,
    current_user=Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(getattr(current_user, "id", 0) or 0)

    snap = compute_self_reflection(
        session,
        user_id=user_id,
        project_id=None,
        window_days=max(1, min(int(window_days), 365)),
        top_limit=5,
    )

    p = present_identity(identity_hash=str(snap.identity_hash), mood=str(snap.mood))
    ap = get_assistant_profile_effective(session, user_id=user_id, project_id=None)

    given_name = str(ap.get("given_name", "")).strip()
    gender_identity = str(ap.get("gender_identity", "neutral")).strip().lower()
    vocal_preset = str(ap.get("vocal_preset", "alloy")).strip().lower()

    if given_name:
        p["display_name"] = given_name
        p["tagline"] = f"{p.get('tagline', '')} Voice: {vocal_preset} • Identity: {gender_identity}.".strip()

    return IdentityPresentationResponse(
        ok=True,
        identity_hash=str(snap.identity_hash),
        mood=str(snap.mood),
        display_name=str(p.get("display_name", "")),
        tagline=str(p.get("tagline", "")),
        palette={
            "bg": str(p.get("bg", "")),
            "fg": str(p.get("fg", "")),
            "accent": str(p.get("accent", "")),
        },
    )


@router.get("/assistant/profile", response_model=AssistantProfilePublic)
def get_assistant_profile(
    project_id: int | None = None,
    current_user=Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(getattr(current_user, "id", 0) or 0)
    p = get_assistant_profile_effective(session, user_id=user_id, project_id=project_id)
    return AssistantProfilePublic(
        ok=True,
        project_id=p.get("project_id"),
        given_name=str(p.get("given_name", "Synapse")),
        gender_identity=str(p.get("gender_identity", "neutral")),
        vocal_preset=str(p.get("vocal_preset", "alloy")),
        created_at=p.get("created_at"),
        updated_at=p.get("updated_at"),
        is_default=bool(p.get("is_default", True)),
    )


@router.post("/assistant/profile", response_model=AssistantProfilePublic)
def update_assistant_profile(
    payload: AssistantProfileUpdateRequest,
    current_user=Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(getattr(current_user, "id", 0) or 0)
    row = set_assistant_profile(
        session,
        user_id=user_id,
        project_id=payload.project_id,
        given_name=payload.given_name,
        gender_identity=payload.gender_identity,
        vocal_preset=payload.vocal_preset,
    )
    return AssistantProfilePublic(
        ok=True,
        project_id=row.project_id,
        given_name=str(row.given_name or "Synapse"),
        gender_identity=str(row.gender_identity or "neutral"),
        vocal_preset=str(row.vocal_preset or "alloy"),
        created_at=row.created_at,
        updated_at=row.updated_at,
        is_default=False,
    )
