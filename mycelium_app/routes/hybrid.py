from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.hybrid_predictor import predict_next_work_session
from mycelium_app.models import ProjectMember, SignalLedgerEvent, User
from mycelium_app.schemas import (
    AdaptiveDirectiveRequest,
    AdaptiveDirectiveResponse,
    HybridWorkSessionPredictRequest,
    HybridWorkSessionPredictResponse,
)
from mycelium_app.settings import settings
from mycelium_app.viscosity import calculate_live_viscosity


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


@router.post("/directive/work-session/adaptive", response_model=AdaptiveDirectiveResponse)
def adaptive_work_session_directive(
    payload: AdaptiveDirectiveRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    if not bool(getattr(settings, "hybrid_predictor_enabled", True)):
        raise HTTPException(status_code=404, detail="Hybrid predictor is disabled")

    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, payload.project_id)

    wm = int(payload.window_minutes or int(getattr(settings, "hybrid_predictor_window_minutes", 120) or 120))
    wm = max(15, min(wm, 24 * 60))

    out = predict_next_work_session(
        session,
        user_id=user_id,
        project_id=payload.project_id,
        window_minutes=wm,
    )
    hybrid = HybridWorkSessionPredictResponse(ok=True, **out)

    since = datetime.utcnow() - timedelta(minutes=wm)
    q = select(SignalLedgerEvent).where(
        SignalLedgerEvent.created_by_user_id == user_id,
        SignalLedgerEvent.created_at >= since,
    )
    if payload.project_id is None:
        q = q.where(SignalLedgerEvent.project_id.is_(None))
    else:
        q = q.where(SignalLedgerEvent.project_id == int(payload.project_id))
    signals = session.exec(q).all()

    viscosity = calculate_live_viscosity(signals)

    base = max(15, min(int(payload.base_duration_minutes), 180))
    suggested = base
    strategy = "hold"
    reason = "Maintaining baseline directive."

    if viscosity.prediction_state == "gated" or float(viscosity.score or 0.0) >= 0.75:
        suggested = min(base, 25)
        strategy = "shorten"
        reason = "High environmental resistance detected; suggest a short session."
    elif viscosity.prediction_state == "observe" or float(viscosity.score or 0.0) >= 0.35:
        suggested = min(base, 35)
        strategy = "shorten"
        reason = "Moderate resistance detected; suggest a medium session."
    elif bool(hybrid.recommend) and float(hybrid.timing_score or 0.0) >= 0.8:
        suggested = max(base, 50)
        strategy = "normalize"
        reason = "Low resistance and strong timing score; full session recommended."

    return AdaptiveDirectiveResponse(
        ok=True,
        project_id=payload.project_id,
        base_duration_minutes=base,
        suggested_duration_minutes=int(suggested),
        strategy=strategy,
        reason=reason,
        hybrid=hybrid,
        viscosity=viscosity,
    )
