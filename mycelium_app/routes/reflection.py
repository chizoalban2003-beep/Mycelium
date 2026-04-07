from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import ProjectMember, User
from mycelium_app.parental_policy import get_policy
from mycelium_app.schemas import DailyConsolidationResponse, SelfReflectionResponse
from mycelium_app.self_reflection import compute_daily_consolidation, compute_self_reflection


router = APIRouter(prefix="/api/nexus/reflection", tags=["reflection"])


def _ensure_project_access(session: Session, user_id: int, project_id: int | None) -> None:
    if project_id is None:
        return
    member = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == project_id, ProjectMember.user_id == user_id)
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a project member")


@router.get("", response_model=SelfReflectionResponse)
def reflect(
    window_days: int = 30,
    top_limit: int = 5,
    project_id: int | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """SelfReflection: analyze the GrowthLedger as if it were "the body".

    This endpoint turns sweep history into:
    - identity: stable hash of best accepted sweeps
    - mood: a transparent translation of stability/acceptance into a label
    - preferences: the best sweeps and their stable knobs

    It is intentionally non-mystical: the goal is introspection and UX.
    """

    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, project_id)

    policy = get_policy(session, user_id)
    allow_modalities = policy.get("allow_modalities") if isinstance(policy.get("allow_modalities"), list) else []
    if allow_modalities and "telemetry" not in set(str(m).lower() for m in allow_modalities):
        raise HTTPException(status_code=403, detail="Telemetry/reflection blocked by parental policy")

    snapshot = compute_self_reflection(
        session,
        user_id=user_id,
        project_id=project_id,
        window_days=window_days,
        top_limit=top_limit,
    )

    return SelfReflectionResponse(
        ok=True,
        mood=snapshot.mood,
        mood_signal=snapshot.mood_signal,
        identity_hash=snapshot.identity_hash,
        top_preferences=snapshot.top_preferences,
        causal_hints=snapshot.causal_hints,
        stats=snapshot.stats,
    )


@router.get("/daily-summary", response_model=DailyConsolidationResponse)
def daily_summary(
    window_hours: int = 24,
    project_id: int | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, project_id)

    policy = get_policy(session, user_id)
    allow_modalities = policy.get("allow_modalities") if isinstance(policy.get("allow_modalities"), list) else []
    if allow_modalities and "telemetry" not in set(str(m).lower() for m in allow_modalities):
        raise HTTPException(status_code=403, detail="Telemetry/reflection blocked by parental policy")

    out = compute_daily_consolidation(
        session,
        user_id=user_id,
        project_id=project_id,
        window_hours=window_hours,
    )
    return DailyConsolidationResponse(ok=True, **out)
