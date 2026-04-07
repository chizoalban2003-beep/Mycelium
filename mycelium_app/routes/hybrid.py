from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.hybrid_predictor import predict_next_work_session
from mycelium_app.models import ProjectMember, SignalLedgerEvent, User
from mycelium_app.schemas import (
    AdaptiveMultiNodeDirectiveRequest,
    AdaptiveMultiNodeDirectiveResponse,
    AdaptiveNodeRecommendation,
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


def _duration_strategy(
    *,
    base_minutes: int,
    viscosity_score: float,
    viscosity_state: str,
    hybrid_recommend: bool,
    hybrid_timing_score: float,
) -> tuple[int, str, str]:
    suggested = int(base_minutes)
    strategy = "hold"
    reason = "Maintaining baseline directive."

    if viscosity_state == "gated" or float(viscosity_score) >= 0.75:
        suggested = min(int(base_minutes), 15)
        strategy = "shorten"
        reason = "High resistance detected; recommend short sprint or recovery mode."
    elif viscosity_state == "observe" or float(viscosity_score) >= 0.35:
        suggested = min(int(base_minutes), 25)
        strategy = "shorten"
        reason = "Moderate resistance detected; recommend adaptive sprint."
    elif bool(hybrid_recommend) and float(hybrid_timing_score) >= 0.8:
        suggested = max(int(base_minutes), 50)
        strategy = "normalize"
        reason = "Low resistance and high momentum; full session recommended."

    suggested = max(0, min(int(suggested), 180))
    return int(suggested), strategy, reason


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
    suggested, strategy, reason = _duration_strategy(
        base_minutes=base,
        viscosity_score=float(viscosity.score or 0.0),
        viscosity_state=str(viscosity.prediction_state or "observe"),
        hybrid_recommend=bool(hybrid.recommend),
        hybrid_timing_score=float(hybrid.timing_score or 0.0),
    )

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


@router.post("/directive/work-session/multinode", response_model=AdaptiveMultiNodeDirectiveResponse)
def adaptive_multinode_directive(
    payload: AdaptiveMultiNodeDirectiveRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    if not bool(getattr(settings, "hybrid_predictor_enabled", True)):
        raise HTTPException(status_code=404, detail="Hybrid predictor is disabled")

    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, payload.project_id)

    wm = int(payload.window_minutes or int(getattr(settings, "hybrid_predictor_window_minutes", 120) or 120))
    wm = max(15, min(wm, 24 * 60))
    base = max(0, min(int(payload.base_duration_minutes), 180))

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

    grouped: dict[str, list[SignalLedgerEvent]] = defaultdict(list)
    for s in signals:
        did = str(s.device_id or "").strip()[:64] or "local"
        grouped[did].append(s)

    requested_ids = [str(x).strip()[:64] for x in (payload.candidate_device_ids or []) if str(x).strip()]
    current_device_id = str(payload.current_device_id or "").strip()[:64] or None

    candidate_ids: list[str]
    if requested_ids:
        candidate_ids = list(dict.fromkeys(requested_ids))
    else:
        candidate_ids = list(grouped.keys())

    if current_device_id and current_device_id not in candidate_ids:
        candidate_ids.append(current_device_id)

    if not candidate_ids:
        candidate_ids = ["local"]

    recs: list[AdaptiveNodeRecommendation] = []
    for did in candidate_ids[:20]:
        node_signals = grouped.get(did, [])
        vis = calculate_live_viscosity(node_signals)
        suggested, strategy, reason = _duration_strategy(
            base_minutes=base,
            viscosity_score=float(vis.score or 0.0),
            viscosity_state=str(vis.prediction_state or "observe"),
            hybrid_recommend=bool(hybrid.recommend),
            hybrid_timing_score=float(hybrid.timing_score or 0.0),
        )
        recs.append(
            AdaptiveNodeRecommendation(
                device_id=did,
                n_signals=int(len(node_signals)),
                suggested_duration_minutes=int(suggested),
                strategy=strategy,
                reason=reason,
                viscosity=vis,
            )
        )

    recs.sort(key=lambda r: (float(r.viscosity.score), -int(r.n_signals), str(r.device_id)))
    best = recs[0] if recs else None
    recommended_device_id = str(best.device_id) if best else None

    handoff_recommended = False
    reason = "No handoff needed."

    if best is not None and current_device_id:
        cur = next((r for r in recs if str(r.device_id) == current_device_id), None)
        if cur is None:
            handoff_recommended = True
            reason = f"Current device '{current_device_id}' lacks telemetry; best target is '{best.device_id}'."
        else:
            delta = float(cur.viscosity.score) - float(best.viscosity.score)
            if best.device_id != current_device_id and (float(cur.viscosity.score) >= 0.70 or delta >= 0.15):
                handoff_recommended = True
                reason = (
                    f"Switch from '{current_device_id}' (η={float(cur.viscosity.score):.2f}) "
                    f"to '{best.device_id}' (η={float(best.viscosity.score):.2f}) for better flow."
                )
            else:
                reason = f"Current device '{current_device_id}' is within acceptable resistance."
    elif best is not None:
        reason = f"Best available device is '{best.device_id}' (η={float(best.viscosity.score):.2f})."

    return AdaptiveMultiNodeDirectiveResponse(
        ok=True,
        project_id=payload.project_id,
        current_device_id=current_device_id,
        recommended_device_id=recommended_device_id,
        handoff_recommended=bool(handoff_recommended),
        reason=reason,
        hybrid=hybrid,
        recommendations=recs,
    )
