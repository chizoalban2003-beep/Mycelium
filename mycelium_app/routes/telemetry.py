from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import ProjectMember, SignalLedgerEvent, User
from mycelium_app.parental_policy import get_policy
from mycelium_app.schemas import TelemetryIngestRequest, TelemetryIngestResponse, TelemetrySummaryResponse
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


def _dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _simple_confidence(n_events: int, signal_counts: dict[str, int]) -> float:
    # Heuristic: enough events + diversity implies higher confidence.
    unique = len([k for k, v in signal_counts.items() if v > 0])
    density = min(1.0, n_events / 500.0)
    diversity = min(1.0, unique / 6.0)
    conf = 0.25 + 0.55 * density + 0.20 * diversity
    return float(max(0.0, min(1.0, conf)))


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

    row = SignalLedgerEvent(
        created_at=occurred_at,
        created_by_user_id=user_id,
        project_id=payload.project_id,
        device_id=device_id,
        signal_type=signal_type,
        payload_json=dumped,
    )
    session.add(row)
    session.commit()
    session.refresh(row)

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
        first_word = (
            "I've observed a stable routine pattern. Should I pre-warm your Work-Zone viscosity during your "
            "high-focus window to silence non-essential notifications?"
        )

    return TelemetrySummaryResponse(
        ok=True,
        window_hours=window_hours,
        n_events=int(len(rows)),
        signal_counts=signal_counts,
        confidence=float(round(conf, 4)),
        patterns=patterns,
        first_word=first_word,
    )
