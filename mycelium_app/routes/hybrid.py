from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.hybrid_predictor import predict_next_work_session
from mycelium_app.models import ProjectMember, User
from mycelium_app.schemas import HybridWorkSessionPredictRequest, HybridWorkSessionPredictResponse
from mycelium_app.settings import settings


router = APIRouter(prefix="/api/nexus/hybrid", tags=["hybrid"])


def _ensure_project_access(session: Session, user_id: int, project_id: int | None) -> None:
    if project_id is None:
        return
    member = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == int(project_id), ProjectMember.user_id == int(user_id))
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a project member")


@router.post("/work-session/next", response_model=HybridWorkSessionPredictResponse)
def predict_work_session(
    payload: HybridWorkSessionPredictRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    if not bool(getattr(settings, "hybrid_predictor_enabled", True)):
        raise HTTPException(status_code=404, detail="Hybrid predictor is disabled")

    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, payload.project_id)

    wm = int(payload.window_minutes or int(getattr(settings, "hybrid_predictor_window_minutes", 120) or 120))

    out = predict_next_work_session(
        session,
        user_id=user_id,
        project_id=payload.project_id,
        window_minutes=wm,
    )

    return HybridWorkSessionPredictResponse(ok=True, **out)
