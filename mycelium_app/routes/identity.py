from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel import Session

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.identity_presentation import present_identity
from mycelium_app.schemas import IdentityPresentationResponse
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
