from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.hybrid_predictor import predict_next_work_session
from mycelium_app.models import HandoffSession, ProjectMember, SignalLedgerEvent, TaskReplica, TaskTrajectory, User
from mycelium_app.parental_policy import get_policy
from mycelium_app.routes.tasks import approve_replica_and_queue
from mycelium_app.schemas import (
    AutoHandoffConfirmRequest,
    AutoHandoffConfirmResponse,
    AutoHandoffLaunchRequest,
    AutoHandoffLaunchResponse,
    AdaptiveMultiNodeDirectiveRequest,
    AdaptiveMultiNodeDirectiveResponse,
    AdaptiveNodeRecommendation,
    AdaptiveDirectiveRequest,
    AdaptiveDirectiveResponse,
    HybridWorkSessionPredictRequest,
    HybridWorkSessionPredictResponse,
    HandoffSessionPublic,
    HandoffSessionStartRequest,
    HandoffSessionStartResponse,
    HandoffSessionTickRequest,
    HandoffSessionTickResponse,
)
from mycelium_app.settings import settings
from mycelium_app.viscosity import calculate_live_viscosity


router = APIRouter(prefix="/api/nexus/hybrid", tags=["hybrid"])


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


def _trajectory_key_from_sequence(sequence: list[str]) -> str:
    normalized = [str(x).strip().lower()[:96] for x in sequence if str(x).strip()]
    raw = "|".join(normalized) if normalized else "empty"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _analyze_multinode(
    session: Session,
    *,
    user_id: int,
    project_id: int | None,
    window_minutes: int,
    base_duration_minutes: int,
    current_device_id: str | None,
    candidate_device_ids: list[str],
) -> tuple[HybridWorkSessionPredictResponse, list[AdaptiveNodeRecommendation], str | None, bool, str]:
    out = predict_next_work_session(
        session,
        user_id=user_id,
        project_id=project_id,
        window_minutes=window_minutes,
    )
    hybrid = HybridWorkSessionPredictResponse(ok=True, **out)

    since = datetime.utcnow() - timedelta(minutes=window_minutes)
    q = select(SignalLedgerEvent).where(
        SignalLedgerEvent.created_by_user_id == int(user_id),
        SignalLedgerEvent.created_at >= since,
    )
    if project_id is None:
        q = q.where(SignalLedgerEvent.project_id.is_(None))
    else:
        q = q.where(SignalLedgerEvent.project_id == int(project_id))
    signals = session.exec(q).all()

    grouped: dict[str, list[SignalLedgerEvent]] = defaultdict(list)
    for s in signals:
        did = str(s.device_id or "").strip()[:64] or "local"
        grouped[did].append(s)

    requested_ids = [str(x).strip()[:64] for x in (candidate_device_ids or []) if str(x).strip()]
    current_id = str(current_device_id or "").strip()[:64] or None

    if requested_ids:
        ids = list(dict.fromkeys(requested_ids))
    else:
        ids = list(grouped.keys())
    if current_id and current_id not in ids:
        ids.append(current_id)
    if not ids:
        ids = ["local"]

    recs: list[AdaptiveNodeRecommendation] = []
    for did in ids[:20]:
        node_signals = grouped.get(did, [])
        vis = calculate_live_viscosity(node_signals)
        suggested, strategy, reason = _duration_strategy(
            base_minutes=base_duration_minutes,
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
    if best is not None and current_id:
        cur = next((r for r in recs if str(r.device_id) == current_id), None)
        if cur is None:
            handoff_recommended = True
            reason = f"Current device '{current_id}' lacks telemetry; best target is '{best.device_id}'."
        else:
            delta = float(cur.viscosity.score) - float(best.viscosity.score)
            if best.device_id != current_id and (float(cur.viscosity.score) >= 0.70 or delta >= 0.15):
                handoff_recommended = True
                reason = (
                    f"Switch from '{current_id}' (η={float(cur.viscosity.score):.2f}) "
                    f"to '{best.device_id}' (η={float(best.viscosity.score):.2f}) for better flow."
                )
            else:
                reason = f"Current device '{current_id}' is within acceptable resistance."
    elif best is not None:
        reason = f"Best available device is '{best.device_id}' (η={float(best.viscosity.score):.2f})."

    return hybrid, recs, recommended_device_id, bool(handoff_recommended), reason


def _handoff_to_public(row: HandoffSession) -> HandoffSessionPublic:
    return HandoffSessionPublic(
        id=int(row.id or 0),
        created_at=row.created_at,
        updated_at=row.updated_at,
        project_id=row.project_id,
        current_device_id=str(row.current_device_id or ""),
        target_device_id=str(row.target_device_id or ""),
        replica_id=(int(row.replica_id) if row.replica_id is not None else None),
        status=str(row.status or ""),
        launch_mode=str(row.launch_mode or ""),
        attempt_count=int(row.attempt_count or 0),
        max_attempts=int(row.max_attempts or 0),
        timeout_at=row.timeout_at,
        next_retry_at=row.next_retry_at,
        last_error=str(row.last_error or ""),
        details=_loads_dict(row.details_json),
    )


def _should_auto_confirm(
    *,
    policy: dict[str, object],
    handoff_recommended: bool,
    best: AdaptiveNodeRecommendation | None,
    hybrid: HybridWorkSessionPredictResponse,
) -> tuple[bool, str, str]:
    actions = policy.get("actions") if isinstance(policy.get("actions"), dict) else {}
    autonomy_mode = str(actions.get("autonomy_mode", "strict")).strip().lower() or "strict"
    if autonomy_mode not in {"strict", "balanced", "auto"}:
        autonomy_mode = "strict"

    if not bool(actions.get("enabled", False)) or not bool(actions.get("device_control_enabled", False)):
        return False, autonomy_mode, "Actions/device-control policy is disabled."
    if bool(actions.get("require_confirm", True)):
        return False, autonomy_mode, "Policy requires explicit confirmation."
    if best is None:
        return False, autonomy_mode, "No candidate node available."

    score = float(best.viscosity.score or 0.0)
    state = str(best.viscosity.prediction_state or "observe")

    if autonomy_mode == "strict":
        return False, autonomy_mode, "Strict mode keeps proposals manual."

    if autonomy_mode == "balanced":
        if handoff_recommended:
            return False, autonomy_mode, "Balanced mode requires manual confirm for handoff."
        if state != "flow" or score > 0.35:
            return False, autonomy_mode, "Balanced mode requires low viscosity flow state."
        if not bool(hybrid.governor_ok):
            return False, autonomy_mode, "Balanced mode requires governor approval."
        return True, autonomy_mode, "Balanced mode auto-confirmed under low resistance."

    # autonomy_mode == auto
    if state == "gated" or score >= 0.75:
        return False, autonomy_mode, "Auto mode blocked by gated/high-resistance node."
    if not bool(hybrid.governor_ok):
        return False, autonomy_mode, "Auto mode blocked by governor confidence gate."
    return True, autonomy_mode, "Auto mode confirmed under policy and governor gates."


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

    hybrid, recs, recommended_device_id, handoff_recommended, reason = _analyze_multinode(
        session,
        user_id=user_id,
        project_id=payload.project_id,
        window_minutes=wm,
        base_duration_minutes=base,
        current_device_id=payload.current_device_id,
        candidate_device_ids=payload.candidate_device_ids,
    )

    return AdaptiveMultiNodeDirectiveResponse(
        ok=True,
        project_id=payload.project_id,
        current_device_id=(str(payload.current_device_id or "").strip()[:64] or None),
        recommended_device_id=recommended_device_id,
        handoff_recommended=bool(handoff_recommended),
        reason=reason,
        hybrid=hybrid,
        recommendations=recs,
    )


@router.post("/directive/work-session/auto-handoff-launch", response_model=AutoHandoffLaunchResponse)
def auto_handoff_launch(
    payload: AutoHandoffLaunchRequest,
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

    hybrid, recs, recommended_device_id, handoff_recommended, handoff_reason = _analyze_multinode(
        session,
        user_id=user_id,
        project_id=payload.project_id,
        window_minutes=wm,
        base_duration_minutes=base,
        current_device_id=payload.current_device_id,
        candidate_device_ids=payload.candidate_device_ids,
    )

    if not recs or all(str(r.viscosity.prediction_state or "") == "gated" for r in recs):
        return AutoHandoffLaunchResponse(
            ok=True,
            project_id=payload.project_id,
            handoff_recommended=bool(handoff_recommended),
            recommended_device_id=recommended_device_id,
            launch_mode="recovery",
            suggested_duration_minutes=0,
            reason="All candidate nodes are gated. Recovery mode recommended.",
            trajectory_id=None,
            replica_id=None,
            queued_device_action_id=None,
            hybrid=hybrid,
            recommendations=recs,
        )

    best = recs[0]
    policy = get_policy(session, user_id)
    auto_confirm, autonomy_mode, gate_reason = _should_auto_confirm(
        policy=policy,
        handoff_recommended=bool(handoff_recommended),
        best=best,
        hybrid=hybrid,
    )

    selected_device_id = str(recommended_device_id or best.device_id or "local")[:64] or "local"
    duration = max(10, min(int(best.suggested_duration_minutes or base or 25), 180))
    focus_app = str(payload.focus_app or "mycelium").strip().lower()[:64] or "mycelium"

    # Dedup guard: reuse very recent equivalent proposed launch.
    recent_since = datetime.utcnow() - timedelta(minutes=2)
    rq = select(TaskReplica).where(TaskReplica.created_by_user_id == int(user_id))
    if payload.project_id is None:
        rq = rq.where(TaskReplica.project_id.is_(None))
    else:
        rq = rq.where(TaskReplica.project_id == int(payload.project_id))
    rq = (
        rq.where(TaskReplica.status == "proposed")
        .where(TaskReplica.capability == "start_focus_session")
        .where(TaskReplica.device_id == selected_device_id)
        .where(TaskReplica.created_at >= recent_since)
        .order_by(TaskReplica.created_at.desc())
        .limit(5)
    )
    recent_rows = session.exec(rq).all()
    for r in recent_rows:
        cmd = _loads_dict(r.command_json)
        if int(cmd.get("duration_minutes") or -1) == int(duration) and str(cmd.get("open_app") or "") == focus_app:
            # Try to find trajectory with same key created around same time.
            tr = session.exec(
                select(TaskTrajectory)
                .where(TaskTrajectory.created_by_user_id == int(user_id))
                .where(TaskTrajectory.trajectory_key == str(r.trajectory_key or ""))
                .order_by(TaskTrajectory.created_at.desc())
                .limit(1)
            ).first()
            return AutoHandoffLaunchResponse(
                ok=True,
                project_id=payload.project_id,
                handoff_recommended=bool(handoff_recommended),
                recommended_device_id=selected_device_id,
                launch_mode="proposed",
                suggested_duration_minutes=int(duration),
                reason=f"Reused recent equivalent launch proposal (dedupe guard). {gate_reason}",
                trajectory_id=(int(tr.id or 0) if tr else None),
                replica_id=int(r.id or 0),
                queued_device_action_id=None,
                hybrid=hybrid,
                recommendations=recs,
            )

    sequence = [
        "session_start_detected",
        "multinode_handoff_analyzed",
        f"handoff_target:{selected_device_id}",
        "open_mycelium_dashboard",
        "open_focus_app",
        "enable_dnd",
        "set_focus_timer",
    ]
    trajectory_key = _trajectory_key_from_sequence(sequence)

    trajectory = TaskTrajectory(
        created_by_user_id=user_id,
        project_id=payload.project_id,
        device_id=selected_device_id,
        trajectory_key=trajectory_key,
        sequence_json=_dumps(sequence),
        app_state_json=_dumps({"mode": "work_session", "focus_app": focus_app, "handoff": True}),
        input_vector_json=_dumps(
            {
                "trigger": "auto_handoff_launch",
                "time_block_minutes": duration,
                "current_device_id": str(payload.current_device_id or "").strip()[:64] or None,
                "recommended_device_id": selected_device_id,
            }
        ),
        confidence=max(0.0, min(float(hybrid.timing_score or 0.0), 1.0)),
        support_count=max(1, int(best.n_signals or 1)),
    )
    session.add(trajectory)
    session.flush()

    replica = TaskReplica(
        created_by_user_id=user_id,
        project_id=payload.project_id,
        device_id=selected_device_id,
        title="Auto-Handoff Focus Session",
        trajectory_key=trajectory_key,
        consensus_fraction=max(0.0, min(float(hybrid.governor_confidence or 0.0), 1.0)),
        species_confidence=max(0.0, min(float(hybrid.timing_score or 0.0), 1.0)),
        capability="start_focus_session",
        command_json=_dumps(
            {
                "op": "focus_session",
                "duration_minutes": duration,
                "enable_dnd": True,
                "open_app": focus_app,
                "open_dashboard": True,
                "handoff": {
                    "current_device_id": str(payload.current_device_id or "").strip()[:64] or None,
                    "recommended_device_id": selected_device_id,
                    "handoff_recommended": bool(handoff_recommended),
                },
                "autonomy": {
                    "mode": autonomy_mode,
                    "auto_confirmed": bool(auto_confirm),
                    "gate_reason": gate_reason,
                },
            }
        ),
        status="proposed",
        notes=f"{handoff_reason} | autonomy_mode={autonomy_mode} auto_confirmed={bool(auto_confirm)} gate_reason={gate_reason}",
    )
    session.add(replica)
    session.commit()
    session.refresh(trajectory)
    session.refresh(replica)

    launch_mode = "proposed"
    queued_device_action_id: int | None = None
    reason = f"{handoff_reason} {gate_reason}".strip()
    if auto_confirm:
        try:
            message_id, detail, _reused = approve_replica_and_queue(
                session,
                user_id=user_id,
                row=replica,
                device_id=selected_device_id,
                auto_execute=True,
            )
            launch_mode = "approved"
            queued_device_action_id = int(message_id or 0)
            reason = detail
        except Exception as e:
            launch_mode = "proposed"
            reason = f"Auto-confirm attempted but failed ({type(e).__name__}). {gate_reason}"

    return AutoHandoffLaunchResponse(
        ok=True,
        project_id=payload.project_id,
        handoff_recommended=bool(handoff_recommended),
        recommended_device_id=selected_device_id,
        launch_mode=launch_mode,
        suggested_duration_minutes=int(duration),
        reason=reason,
        trajectory_id=int(trajectory.id or 0),
        replica_id=int(replica.id or 0),
        queued_device_action_id=queued_device_action_id,
        hybrid=hybrid,
        recommendations=recs,
    )


@router.post("/directive/work-session/auto-handoff-confirm", response_model=AutoHandoffConfirmResponse)
def auto_handoff_confirm(
    payload: AutoHandoffConfirmRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)

    row = session.exec(
        select(TaskReplica).where(TaskReplica.id == int(payload.replica_id), TaskReplica.created_by_user_id == user_id)
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Replica not found")

    _ensure_project_access(session, user_id, row.project_id)

    message_id, detail, _reused = approve_replica_and_queue(
        session,
        user_id=user_id,
        row=row,
        device_id=payload.device_id,
        auto_execute=False,
    )

    return AutoHandoffConfirmResponse(
        ok=True,
        replica_id=int(row.id or 0),
        queued_device_action_id=int(message_id or 0),
        detail=detail,
    )


@router.post("/handoff/session/start", response_model=HandoffSessionStartResponse)
def handoff_session_start(
    payload: HandoffSessionStartRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, payload.project_id)

    launch = auto_handoff_launch(
        AutoHandoffLaunchRequest(
            project_id=payload.project_id,
            window_minutes=payload.window_minutes,
            base_duration_minutes=payload.base_duration_minutes,
            current_device_id=payload.current_device_id,
            candidate_device_ids=payload.candidate_device_ids,
            focus_app=payload.focus_app,
        ),
        current_user=current_user,
        session=session,
    )

    max_attempts = max(1, min(int(payload.max_attempts), 10))
    timeout_seconds = max(30, min(int(payload.timeout_seconds), 3600))
    now = datetime.utcnow()

    status = "proposed"
    if str(launch.launch_mode) == "approved":
        status = "queued"
    elif str(launch.launch_mode) == "recovery":
        status = "recovery"

    hs = HandoffSession(
        created_by_user_id=user_id,
        project_id=payload.project_id,
        current_device_id=str(payload.current_device_id or "")[:64],
        target_device_id=str(launch.recommended_device_id or "")[:64],
        replica_id=(int(launch.replica_id) if launch.replica_id is not None else None),
        status=status,
        launch_mode=str(launch.launch_mode or "proposed")[:32],
        attempt_count=0,
        max_attempts=max_attempts,
        timeout_at=now + timedelta(seconds=timeout_seconds),
        details_json=_dumps(
            {
                "reason": str(launch.reason or ""),
                "handoff_recommended": bool(launch.handoff_recommended),
                "suggested_duration_minutes": int(launch.suggested_duration_minutes or 0),
            }
        ),
    )
    session.add(hs)
    session.commit()
    session.refresh(hs)
    return HandoffSessionStartResponse(ok=True, session=_handoff_to_public(hs))


@router.post("/handoff/session/{session_id}/tick", response_model=HandoffSessionTickResponse)
def handoff_session_tick(
    session_id: int,
    payload: HandoffSessionTickRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    hs = session.exec(
        select(HandoffSession).where(
            HandoffSession.id == int(session_id),
            HandoffSession.created_by_user_id == user_id,
        )
    ).first()
    if not hs:
        raise HTTPException(status_code=404, detail="Handoff session not found")

    _ensure_project_access(session, user_id, hs.project_id)

    terminal = {"completed", "failed", "timed_out", "recovery"}
    now = datetime.utcnow()
    if str(hs.status or "") in terminal:
        return HandoffSessionTickResponse(ok=True, session=_handoff_to_public(hs))

    if hs.timeout_at is not None and now >= hs.timeout_at:
        hs.status = "timed_out"
        hs.last_error = "handoff_session_timeout"
        hs.updated_at = now
        session.add(hs)
        session.commit()
        session.refresh(hs)
        return HandoffSessionTickResponse(ok=True, session=_handoff_to_public(hs))

    replica: TaskReplica | None = None
    if hs.replica_id is not None:
        replica = session.exec(
            select(TaskReplica).where(TaskReplica.id == int(hs.replica_id), TaskReplica.created_by_user_id == user_id)
        ).first()

    if replica is not None and str(replica.status or "") in {"executed"}:
        hs.status = "completed"
        hs.updated_at = now
    elif replica is not None and str(replica.status or "") in {"failed", "rejected"}:
        hs.status = "failed"
        hs.updated_at = now
        hs.last_error = f"replica_{str(replica.status or '').lower()}"
    elif str(hs.status or "") in {"proposed", "waiting_retry", "launched"} and hs.replica_id is not None:
        if hs.next_retry_at is not None and now < hs.next_retry_at:
            return HandoffSessionTickResponse(ok=True, session=_handoff_to_public(hs))

        try:
            auto_handoff_confirm(
                AutoHandoffConfirmRequest(replica_id=int(hs.replica_id), device_id=(hs.target_device_id or None)),
                current_user=current_user,
                session=session,
            )
            hs.status = "queued"
            hs.next_retry_at = None
            hs.last_error = ""
            hs.updated_at = now
        except Exception as e:
            hs.attempt_count = int(hs.attempt_count or 0) + 1
            hs.last_error = type(e).__name__
            hs.updated_at = now
            if int(hs.attempt_count or 0) >= int(hs.max_attempts or 1):
                hs.status = "failed"
            else:
                hs.status = "waiting_retry"
                wait_s = max(5, min(int(payload.retry_wait_seconds), 300))
                hs.next_retry_at = now + timedelta(seconds=wait_s)

    session.add(hs)
    session.commit()
    session.refresh(hs)
    return HandoffSessionTickResponse(ok=True, session=_handoff_to_public(hs))
