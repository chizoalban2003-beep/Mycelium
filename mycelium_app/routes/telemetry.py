from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.growth import compute_growth_stage
from mycelium_app.models import GrowthLedgerEntry, HiveOutboxMessage, NexusNudge, ProjectMember, ProjectRole, SignalLedgerEvent, User
from mycelium_app.parental_policy import get_policy
from mycelium_app.stimulus import record_stimulus_event
from mycelium_app.telemetry_assistant import maybe_queue_telemetry_assistant_nudge
from mycelium_app.schemas import (
    TelemetryAssistantActionRequest,
    TelemetryAssistantActionResponse,
    TelemetryDeviceActionAckRequest,
    TelemetryDeviceActionAckResponse,
    TelemetryDeviceActionPendingResponse,
    TelemetryDeviceActionPublic,
    TelemetryDeepFreezeSweepRequest,
    TelemetryDeepFreezeSweepResponse,
    TelemetryAssistantTickResponse,
    TelemetryIngestRequest,
    TelemetryIngestResponse,
    TelemetrySummaryResponse,
)
from mycelium_app.settings import settings


router = APIRouter(prefix="/api/nexus/telemetry", tags=["telemetry"])


def _ensure_project_access(session: Session, user_id: int, project_id: int | None) -> None:
    if project_id is None:
        return
    member = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == project_id, ProjectMember.user_id == user_id)
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a project member")


def _project_role(session: Session, user_id: int, project_id: int | None) -> ProjectRole | None:
    if project_id is None:
        return None
    member = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == int(project_id), ProjectMember.user_id == int(user_id))
    ).first()
    if not member:
        return None
    try:
        return member.role if isinstance(member.role, ProjectRole) else ProjectRole(str(member.role))
    except Exception:
        return None


def _dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _simple_confidence(n_events: int, signal_counts: dict[str, int]) -> float:
    # Heuristic: enough events + diversity implies higher confidence.
    unique = len([k for k, v in signal_counts.items() if v > 0])
    density = min(1.0, n_events / 500.0)
    diversity = min(1.0, unique / 6.0)
    conf = 0.25 + 0.55 * density + 0.20 * diversity
    return float(max(0.0, min(1.0, conf)))


def _extract_app_token(payload_json: str) -> str | None:
    # We accept a few common key names so multiple collectors can coexist.
    # Only app identifiers are used; no titles/URLs/text content.
    try:
        payload = json.loads(payload_json or "{}")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    for k in ("app", "app_name", "bundle_id", "process", "exe"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()[:128]
    return None


def _r2_score(y_true: list[int], y_pred: list[int]) -> float:
    # NOTE: This is used as a *proxy* quality score to match the existing
    # growth-stage trigger design (telemetry_next_app:r2). Here, apps are
    # encoded into stable integers to make the score deterministic.
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    n = len(y_true)
    mean = sum(y_true) / float(n)
    ss_tot = sum((yt - mean) ** 2 for yt in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    return float(1.0 - (ss_res / (ss_tot + 1e-12)))


@router.post("/deep-freeze-sweep", response_model=TelemetryDeepFreezeSweepResponse)
def deep_freeze_sweep(
    payload: TelemetryDeepFreezeSweepRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Run a deterministic 'Deep Freeze' sweep over recent telemetry.

    Model:
    - Build a transition table from app A -> next app B counts.
    - Predict next app as argmax_B count(A->B) (a simple Markov-1 baseline).

    Output:
    - Computes accuracy (exact-match next-app rate) and an R²-style proxy over a
      stable integer encoding, then records the result in GrowthLedgerEntry as:
      domain=telemetry_next_app, metric=r2.

    Design goal:
    - Keep this sweep transparent, deterministic, and cheap to run locally.
    """

    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, payload.project_id)

    policy = get_policy(session, user_id)
    allow_modalities = policy.get("allow_modalities") if isinstance(policy.get("allow_modalities"), list) else []
    if allow_modalities and "telemetry" not in set(str(m).lower() for m in allow_modalities):
        raise HTTPException(status_code=403, detail="Telemetry blocked by parental policy")

    window_hours = max(1, min(int(payload.window_hours), 168))
    since = datetime.utcnow() - timedelta(hours=window_hours)

    device_id = (payload.device_id or settings.nexus_device_id or "local").strip()[:64]

    q = select(SignalLedgerEvent).where(
        SignalLedgerEvent.created_by_user_id == user_id,
        SignalLedgerEvent.created_at >= since,
        SignalLedgerEvent.signal_type == "app_open",
    )
    if payload.project_id is not None:
        q = q.where(SignalLedgerEvent.project_id == payload.project_id)
    if device_id:
        q = q.where(SignalLedgerEvent.device_id == device_id)
    q = q.order_by(SignalLedgerEvent.created_at.asc())

    rows = session.exec(q).all()

    apps: list[str] = []
    for r in rows:
        token = _extract_app_token(r.payload_json)
        if token:
            apps.append(token)

    # Build (current -> next) pairs.
    # We drop immediate repeats (A->A) to avoid inflating trivial transitions.
    pairs: list[tuple[str, str]] = []
    for i in range(len(apps) - 1):
        a = apps[i]
        b = apps[i + 1]
        if a and b and a != b:
            pairs.append((a, b))

    min_pairs = max(5, min(int(payload.min_pairs), 50_000))
    if len(pairs) < min_pairs:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough app_open transitions for sweep (have {len(pairs)}, need {min_pairs}).",
        )

    # Transition counts per current app.
    next_counts: dict[str, Counter[str]] = {}
    for a, b in pairs:
        bucket = next_counts.get(a)
        if bucket is None:
            bucket = Counter()
            next_counts[a] = bucket
        bucket[b] += 1

    # Deterministic predictor: argmax next app per current app.
    predictor: dict[str, str] = {}
    for a, c in next_counts.items():
        predictor[a] = c.most_common(1)[0][0]

    # Stable encoding (sorted) so the R² proxy is reproducible.
    vocab = sorted({x for ab in pairs for x in ab})
    to_int = {app: i for i, app in enumerate(vocab)}

    y_true: list[int] = []
    y_pred: list[int] = []
    correct = 0

    for a, b in pairs:
        pred = predictor.get(a)
        if pred is None:
            continue
        y_true.append(int(to_int[b]))
        y_pred.append(int(to_int.get(pred, 0)))
        if pred == b:
            correct += 1

    n_pairs = len(y_true)
    if n_pairs < min_pairs:
        raise HTTPException(status_code=400, detail="Not enough usable pairs for sweep")

    accuracy = float(correct / float(n_pairs))
    r2 = float(_r2_score(y_true, y_pred))

    accept_r2 = float(payload.accept_r2_threshold)
    accepted = bool(r2 >= accept_r2)

    ledger = GrowthLedgerEntry(
        created_by_user_id=user_id,
        project_id=payload.project_id,
        device_id=device_id,
        domain="telemetry_next_app",
        metric="r2",
        score=float(r2),
        accepted=accepted,
        proposal_json=_dumps(
            {
                "sweep": "deep_freeze_telemetry_next_app",
                "window_hours": window_hours,
                "model": "markov_argmax",
                "min_pairs": min_pairs,
            }
        ),
        outcome_json=_dumps(
            {
                "n_events": len(rows),
                "n_pairs": n_pairs,
                "accuracy": accuracy,
                "vocab_size": len(vocab),
                "accepted": accepted,
                "accept_r2_threshold": accept_r2,
            }
        ),
        notes=("Deep Freeze telemetry sweep (next-app prediction)" if accepted else "Telemetry sweep recorded"),
    )
    session.add(ledger)
    session.commit()
    session.refresh(ledger)

    return TelemetryDeepFreezeSweepResponse(
        ok=True,
        entry_id=int(ledger.id or 0),
        domain=str(ledger.domain),
        metric=str(ledger.metric),
        r2=float(round(r2, 6)),
        accuracy=float(round(accuracy, 6)),
        n_pairs=int(n_pairs),
        accepted=bool(accepted),
    )


@router.post("/ingest", response_model=TelemetryIngestResponse)
def ingest(
    payload: TelemetryIngestRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, payload.project_id)

    policy = get_policy(session, user_id)
    allow_modalities = policy.get("allow_modalities") if isinstance(policy.get("allow_modalities"), list) else []
    if allow_modalities and "telemetry" not in set(str(m).lower() for m in allow_modalities):
        raise HTTPException(status_code=403, detail="Telemetry blocked by parental policy")

    signal_type = (payload.signal_type or "").strip().lower()[:64]
    if not signal_type:
        raise HTTPException(status_code=400, detail="signal_type is required")

    device_id = (payload.device_id or settings.nexus_device_id or "local").strip()[:64]

    # Payload limits: keep events small.
    raw = payload.payload or {}
    dumped = _dumps(raw)
    if len(dumped) > 20_000:
        raise HTTPException(status_code=413, detail="payload too large")

    occurred_at = payload.occurred_at or datetime.utcnow()
    if occurred_at > datetime.utcnow() + timedelta(minutes=5):
        occurred_at = datetime.utcnow()

    row, _ = record_stimulus_event(
        session,
        user_id=user_id,
        project_id=payload.project_id,
        device_id=device_id,
        source="telemetry",
        modality="event",
        signal_type=signal_type,
        stimulus=raw,
        occurred_at=occurred_at,
    )

    return TelemetryIngestResponse(ok=True, event_id=int(row.id or 0))


@router.get("/summary", response_model=TelemetrySummaryResponse)
def summary(
    window_hours: int = 24,
    project_id: int | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, project_id)

    window_hours = max(1, min(int(window_hours), 168))
    since = datetime.utcnow() - timedelta(hours=window_hours)

    q = select(SignalLedgerEvent).where(
        SignalLedgerEvent.created_by_user_id == user_id,
        SignalLedgerEvent.created_at >= since,
    )
    if project_id is not None:
        q = q.where(SignalLedgerEvent.project_id == project_id)

    rows = session.exec(q).all()

    counts: Counter[str] = Counter()
    for r in rows:
        counts[str(r.signal_type or "").lower()] += 1

    signal_counts = {k: int(v) for k, v in counts.most_common(50)}
    conf = _simple_confidence(len(rows), signal_counts)

    stage, unlocked, _stats = compute_growth_stage(session, user_id=user_id, project_id=project_id)

    patterns: list[dict[str, object]] = []
    if signal_counts.get("screen_on", 0) + signal_counts.get("screen_off", 0) >= 10:
        patterns.append({"pattern": "temporal_pulse", "detail": "Screen on/off rhythm detected"})
    if signal_counts.get("app_open", 0) >= 10:
        patterns.append({"pattern": "app_viscosity", "detail": "App-open session signals detected"})
    if signal_counts.get("network", 0) >= 5:
        patterns.append({"pattern": "connectivity_flow", "detail": "Network change signals detected"})
    if signal_counts.get("text_sample", 0) >= 5:
        patterns.append({"pattern": "language_sampling", "detail": "Language samples observed"})

    first_word = None
    if conf >= 0.85:
        if stage == "infant":
            first_word = (
                "I've observed a stable routine pattern. Should I pre-warm your Work-Zone viscosity during your "
                "high-focus window to silence non-essential notifications?"
            )
        elif stage == "toddler":
            first_word = (
                "Pattern locked with high confidence. Want me to run a small sweep: auto-silence distractions in your "
                "Work-Zone window and learn from your accept/reject feedback?"
            )
        else:
            first_word = (
                "Your routine looks stable. I can lock a Deep-Work zone (pre-warm viscosity + block known distractions) "
                "and periodically propose macro-optimizations. Approve?"
            )

    return TelemetrySummaryResponse(
        ok=True,
        window_hours=window_hours,
        n_events=int(len(rows)),
        signal_counts=signal_counts,
        confidence=float(round(conf, 4)),
        patterns=patterns
        + ([{"pattern": "growth_stage", "detail": stage, "unlocked": unlocked}] if stage else []),
        first_word=first_word,
    )


@router.post("/assistant/tick", response_model=TelemetryAssistantTickResponse)
def assistant_tick(
    window_hours: int | None = None,
    project_id: int | None = None,
    device_id: str | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Manually run one telemetry-assistant pass for the current user.

    Useful for testing from a phone/UI without waiting for the background loop.
    """

    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, project_id)

    created = False
    try:
        created = bool(
            maybe_queue_telemetry_assistant_nudge(
                session,
                user_id=user_id,
                project_id=project_id,
                device_id=device_id,
                window_hours=window_hours,
            )
        )
        if created:
            session.commit()
    except Exception:
        created = False

    return TelemetryAssistantTickResponse(ok=True, created=bool(created))


def _loads_dict(s: str | None) -> dict:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _permission_tier_for_capability(actions_cfg: dict[str, object], capability: str) -> str:
    caps = actions_cfg.get("permission_tiers") if isinstance(actions_cfg.get("permission_tiers"), dict) else {}
    key = str(capability or "").strip().lower()
    raw = caps.get(key, actions_cfg.get("default_permission_tier", "execute"))
    tier = str(raw or "execute").strip().lower()
    if tier not in {"suggest", "queue", "execute"}:
        tier = "execute"
    return tier


def _queue_device_action(
    *,
    session: Session,
    user_id: int,
    project_id: int | None,
    device_id: str,
    action_id: str,
    confidence: float,
    command: dict[str, object],
) -> int:
    row = HiveOutboxMessage(
        created_by_user_id=int(user_id),
        project_id=project_id,
        device_id=str(device_id or "local")[:128],
        kind="device_action",
        payload_json=_dumps(
            {
                "action_id": str(action_id or "")[:64],
                "confidence": float(round(confidence, 4)),
                "command": command if isinstance(command, dict) else {},
                "requested_at": datetime.utcnow().isoformat() + "Z",
            }
        ),
        submitted_at=None,
    )
    session.add(row)
    session.flush()
    return int(row.id or 0)


@router.post("/assistant/action", response_model=TelemetryAssistantActionResponse)
def assistant_action(
    payload: TelemetryAssistantActionRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Approve/reject a telemetry assistant proposed action.

    Safety: actions execute only after explicit user approval.
    """

    user_id = int(current_user.id or 0)
    nudge = session.exec(
        select(NexusNudge).where(
            NexusNudge.id == int(payload.nudge_id),
            NexusNudge.created_by_user_id == user_id,
        )
    ).first()
    if not nudge:
        raise HTTPException(status_code=404, detail="Nudge not found")

    decision = str(payload.decision or "approve").strip().lower()
    if decision not in {"approve", "reject"}:
        raise HTTPException(status_code=400, detail="decision must be approve|reject")

    action_id = str(payload.action_id or "").strip()[:64]
    if not action_id:
        raise HTTPException(status_code=400, detail="action_id is required")

    policy = get_policy(session, user_id)
    actions_cfg = policy.get("actions") if isinstance(policy.get("actions"), dict) else {}
    assistant_cfg = policy.get("assistant") if isinstance(policy.get("assistant"), dict) else {}
    persona_mode = str(assistant_cfg.get("persona_mode", "calm")).strip().lower()
    if persona_mode not in {"coach", "calm", "briefing"}:
        persona_mode = "calm"
    if bool(actions_cfg.get("kill_switch", False)):
        raise HTTPException(status_code=423, detail="Action kill-switch is enabled")
    actions_enabled = bool(actions_cfg.get("enabled", False))
    require_confirm = bool(actions_cfg.get("require_confirm", True))
    if not actions_enabled:
        raise HTTPException(status_code=403, detail="Device actions disabled by parental policy")

    notify_only = bool(actions_cfg.get("notify_only", True))
    device_control_enabled = bool(actions_cfg.get("device_control_enabled", False))
    try:
        min_confidence = float(actions_cfg.get("min_confidence", 0.90))
    except Exception:
        min_confidence = 0.90
    min_confidence = max(0.0, min(min_confidence, 1.0))

    caps_raw = actions_cfg.get("allowed_capabilities") if isinstance(actions_cfg.get("allowed_capabilities"), list) else []
    capabilities = {str(x).strip().lower() for x in caps_raw if str(x).strip()}

    # Project membrane: in project-scoped actions, only owner/editor can execute.
    if nudge.project_id is not None:
        role = _project_role(session, user_id, int(nudge.project_id))
        if role is None:
            raise HTTPException(status_code=403, detail="Not a project member")
        if role not in {ProjectRole.owner, ProjectRole.editor}:
            raise HTTPException(status_code=403, detail="Owner or editor role required for project actions")

    # Confirm this action is one of the proposed actions in nudge payload.
    nudge_payload = _loads_dict(nudge.payload_json)
    actions = nudge_payload.get("actions") if isinstance(nudge_payload.get("actions"), list) else []
    allowed_ids = {
        str(a.get("action_id") or "").strip()
        for a in actions
        if isinstance(a, dict) and str(a.get("action_id") or "").strip()
    }
    if action_id not in allowed_ids:
        raise HTTPException(status_code=400, detail="Action not proposed by this nudge")

    if decision == "reject":
        if nudge.seen_at is None:
            nudge.seen_at = datetime.utcnow()
            session.add(nudge)
            session.commit()
        return TelemetryAssistantActionResponse(
            ok=True,
            executed=False,
            action_id=action_id,
            decision=decision,
            detail="Action rejected by user.",
            sweep_entry_id=None,
        )

    if require_confirm is False:
        # For safety we still require explicit call to this endpoint; this branch simply documents policy state.
        pass

    sweep_entry_id: int | None = None
    queued_device_action_id: int | None = None
    detail = ""

    if action_id == "run_deep_freeze_sweep":
        req = TelemetryDeepFreezeSweepRequest(
            project_id=nudge.project_id,
            device_id=(None if not isinstance(nudge_payload.get("device_id"), str) else str(nudge_payload.get("device_id"))),
            window_hours=int(nudge_payload.get("window_hours", 24) or 24),
            min_pairs=30,
            accept_r2_threshold=0.90,
        )
        res = deep_freeze_sweep(payload=req, current_user=current_user, session=session)
        sweep_entry_id = int(res.entry_id)
        detail = f"Deep Freeze sweep executed (r2={res.r2}, accuracy={res.accuracy})."
    elif action_id == "suggest_focus_zone":
        followup = NexusNudge(
            created_by_user_id=user_id,
            project_id=nudge.project_id,
            kind="telemetry_focus_zone",
            title="Focus Zone ready",
            message=(
                "I prepared a Focus-Zone suggestion from your recent signals. "
                "Try a 45-minute deep-work block and then confirm/correct this recommendation."
            ),
            payload_json=_dumps({"source_nudge_id": int(nudge.id or 0), "action_id": action_id}),
        )
        session.add(followup)
        detail = "Focus-Zone suggestion queued."
    elif action_id == "device_start_focus_session":
        if notify_only:
            return TelemetryAssistantActionResponse(
                ok=True,
                executed=False,
                action_id=action_id,
                decision=decision,
                detail="Policy is notify-only; no device operation was queued.",
                sweep_entry_id=None,
                queued_device_action_id=None,
            )
        if not device_control_enabled:
            raise HTTPException(status_code=403, detail="Device control not granted by user policy")
        if "start_focus_session" not in capabilities:
            raise HTTPException(status_code=403, detail="Capability not allowed by user policy")
        if _permission_tier_for_capability(actions_cfg, "start_focus_session") == "suggest":
            raise HTTPException(status_code=403, detail="Capability is configured as suggest-only")

        conf = _safe_float(nudge_payload.get("confidence"), 0.0)
        if conf < min_confidence:
            raise HTTPException(
                status_code=409,
                detail=f"Confidence {conf:.3f} below policy minimum {min_confidence:.3f}",
            )

        target_device_id = str(nudge_payload.get("device_id") or settings.nexus_device_id or "local").strip()[:64]
        queued_device_action_id = _queue_device_action(
            session=session,
            user_id=user_id,
            project_id=nudge.project_id,
            device_id=target_device_id,
            action_id=action_id,
            confidence=conf,
            command={
                "op": "focus_session",
                "duration_minutes": 45,
                "enable_dnd": True,
                "open_app": "focus",
                "persona_mode": persona_mode,
            },
        )
        detail = f"Device action queued for companion agent ({target_device_id})."
    else:
        raise HTTPException(status_code=400, detail="Unsupported action_id")

    if nudge.seen_at is None:
        nudge.seen_at = datetime.utcnow()
        session.add(nudge)
    session.commit()

    return TelemetryAssistantActionResponse(
        ok=True,
        executed=True,
        action_id=action_id,
        decision=decision,
        detail=detail,
        sweep_entry_id=sweep_entry_id,
        queued_device_action_id=queued_device_action_id,
    )


@router.get("/device-actions/pending", response_model=TelemetryDeviceActionPendingResponse)
def device_actions_pending(
    device_id: str | None = None,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Poll pending user-approved device actions for a local companion agent."""

    user_id = int(current_user.id or 0)
    limit = max(1, min(int(limit), 200))
    target_device = (device_id or settings.nexus_device_id or "local").strip()[:64]

    q = (
        select(HiveOutboxMessage)
        .where(HiveOutboxMessage.created_by_user_id == user_id)
        .where(HiveOutboxMessage.kind == "device_action")
        .where(HiveOutboxMessage.submitted_at.is_(None))
        .where(HiveOutboxMessage.device_id == target_device)
        .order_by(HiveOutboxMessage.created_at.asc())
        .limit(limit)
    )
    rows = session.exec(q).all()

    out: list[TelemetryDeviceActionPublic] = []
    for r in rows:
        payload = _loads_dict(r.payload_json)
        out.append(
            TelemetryDeviceActionPublic(
                message_id=int(r.id or 0),
                created_at=r.created_at,
                device_id=str(r.device_id or ""),
                project_id=r.project_id,
                action_id=str(payload.get("action_id") or ""),
                confidence=_safe_float(payload.get("confidence"), 0.0),
                command=payload.get("command") if isinstance(payload.get("command"), dict) else {},
            )
        )

    return TelemetryDeviceActionPendingResponse(ok=True, actions=out)


@router.post("/device-actions/{message_id}/ack", response_model=TelemetryDeviceActionAckResponse)
def device_action_ack(
    message_id: int,
    payload: TelemetryDeviceActionAckRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Acknowledge execution result for a queued device action."""

    status = str(payload.status or "executed").strip().lower()
    if status not in {"executed", "failed", "rejected"}:
        raise HTTPException(status_code=400, detail="status must be executed|failed|rejected")

    user_id = int(current_user.id or 0)
    row = session.exec(
        select(HiveOutboxMessage).where(
            HiveOutboxMessage.id == int(message_id),
            HiveOutboxMessage.created_by_user_id == user_id,
            HiveOutboxMessage.kind == "device_action",
        )
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Device action not found")

    wanted_device = (payload.device_id or "").strip()
    if wanted_device and wanted_device != str(row.device_id or ""):
        raise HTTPException(status_code=403, detail="device_id does not match queued action")

    p = _loads_dict(row.payload_json)
    p["ack"] = {
        "status": status,
        "notes": str(payload.notes or "")[:500],
        "acked_at": datetime.utcnow().isoformat() + "Z",
    }
    row.payload_json = _dumps(p)
    row.submitted_at = datetime.utcnow()
    session.add(row)
    session.commit()

    return TelemetryDeviceActionAckResponse(
        ok=True,
        message_id=int(row.id or 0),
        status=status,
        executed=bool(status == "executed"),
    )
