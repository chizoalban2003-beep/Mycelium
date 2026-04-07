from __future__ import annotations

import json
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import ProjectMember, SignalLedgerEvent, User
from mycelium_app.parental_policy import get_policy
from mycelium_app.schemas import DigitalStimulusIngestRequest, DigitalStimulusIngestResponse
from mycelium_app.settings import settings
from mycelium_app.stimulus import build_stimulus_tabular_payload, recommend_learning_profile


router = APIRouter(prefix="/api/nexus/stimulus", tags=["stimulus"])


def _dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ensure_project_access(session: Session, user_id: int, project_id: int | None) -> None:
    if project_id is None:
        return
    member = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == int(project_id), ProjectMember.user_id == int(user_id))
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a project member")


@router.post("/ingest", response_model=DigitalStimulusIngestResponse)
def ingest_stimulus(
    payload: DigitalStimulusIngestRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, payload.project_id)

    policy = get_policy(session, user_id)
    deny_sources = policy.get("deny_sources") if isinstance(policy.get("deny_sources"), list) else []
    allow_modalities = policy.get("allow_modalities") if isinstance(policy.get("allow_modalities"), list) else []

    source = (payload.source or "stimulus").strip().lower()[:32]
    modality = (payload.modality or "auto").strip().lower()[:32]
    signal_type = (payload.signal_type or source or modality or "stimulus").strip().lower()[:64]

    if source and source in {str(s).strip().lower() for s in deny_sources}:
        raise HTTPException(status_code=403, detail="Source blocked by parental policy")
    if allow_modalities and modality not in {str(m).strip().lower() for m in allow_modalities}:
        raise HTTPException(status_code=403, detail="Modality blocked by parental policy")

    device_id = (payload.device_id or settings.nexus_device_id or "local").strip()[:64]
    occurred_at = payload.occurred_at or datetime.utcnow()
    if occurred_at > datetime.utcnow() + timedelta(minutes=5):
        occurred_at = datetime.utcnow()

    envelope = build_stimulus_tabular_payload(
        stimulus=payload.stimulus,
        source=source,
        modality=modality,
        signal_type=signal_type,
        device_id=device_id,
        project_id=payload.project_id,
        occurred_at=occurred_at,
    )

    stored_payload = {"kind": "digital_stimulus", **envelope}
    learning_profile = recommend_learning_profile(stimulus=payload.stimulus, signal_type=signal_type, modality=modality)
    stored_payload["learning_profile"] = learning_profile
    dumped = _dumps(stored_payload)
    if len(dumped) > 50_000:
        raise HTTPException(status_code=413, detail="stimulus payload too large")

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

    return DigitalStimulusIngestResponse(
        ok=True,
        event_id=int(row.id or 0),
        signal_type=signal_type,
        payload_kind=str(envelope.get("meta", {}).get("payload_kind", "")),
        learning_profile=learning_profile,
        tabular=envelope.get("tabular", {}),
    )
