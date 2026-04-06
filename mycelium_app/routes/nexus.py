from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.feedback_ionizer import ionize_user_feedback
from mycelium_app.models import ExperienceBufferEntry, ProjectMember, User
from mycelium_app.nexus_ionizer import grammar_suggest, ionize_finance, style_profile
from mycelium_app.parental_policy import get_policy, set_policy
from mycelium_app.schemas import (
    NexusEntryPublic,
    NexusExportResponse,
    NexusImportRequest,
    NexusImportResponse,
    NexusIngestTextRequest,
    NexusIngestTextResponse,
    NexusIntroResponse,
    NexusListResponse,
    NexusFeedbackIonizeRequest,
    NexusFeedbackIonizeResponse,
    NexusPolicyPublic,
    NexusPolicyUpdateRequest,
)
from mycelium_app.settings import settings


router = APIRouter(prefix="/api/nexus", tags=["nexus"])


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


def _loads_dict(s: str | None) -> dict:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _loads_list(s: str | None) -> list:
    if not s:
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _to_public(entry: ExperienceBufferEntry) -> NexusEntryPublic:
    return NexusEntryPublic(
        entry_uuid=entry.entry_uuid,
        created_at=entry.created_at,
        project_id=entry.project_id,
        device_id=entry.device_id,
        source=entry.source,
        modality=entry.modality,
        raw_text=entry.raw_text,
        extracted=_loads_dict(entry.extracted_json),
        physics_used=_loads_dict(entry.physics_used_json),
        confidence=entry.confidence,
        feedback=entry.feedback,
        tags=_loads_list(entry.tags_json),
    )


@router.post("/ingest/text", response_model=NexusIngestTextResponse)
def ingest_text(
    payload: NexusIngestTextRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, payload.project_id)

    raw_text = (payload.text or "").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(raw_text) > 200_000:
        raise HTTPException(status_code=413, detail="text too large")

    modality = (payload.modality or "auto").strip().lower()[:32]
    source = (payload.source or "text").strip().lower()[:32]

    policy = get_policy(session, user_id)
    deny_sources = policy.get("deny_sources") if isinstance(policy.get("deny_sources"), list) else []
    allow_modalities = (
        policy.get("allow_modalities") if isinstance(policy.get("allow_modalities"), list) else []
    )
    if str(source).lower() in set(str(s).lower() for s in deny_sources):
        raise HTTPException(status_code=403, detail="Source blocked by parental policy")
    if allow_modalities and str(modality).lower() not in set(str(m).lower() for m in allow_modalities):
        raise HTTPException(status_code=403, detail="Modality blocked by parental policy")

    extracted: dict = {}
    confidence: float | None = None

    if modality in ("finance", "money"):
        events = ionize_finance(raw_text)
        extracted = {
            "kind": "finance",
            "events": [
                {"kind": e.kind, "payload": e.payload, "confidence": e.confidence} for e in events
            ],
        }
        if events:
            confidence = sum(e.confidence for e in events) / float(len(events))
        else:
            confidence = 0.25
    elif modality in ("style", "fingerprint"):
        extracted = {"kind": "style", "profile": style_profile(raw_text)}
        confidence = 0.6
    elif modality in ("grammar", "rewrite"):
        g = grammar_suggest(raw_text)
        extracted = {"kind": "grammar", **g}
        confidence = 0.7 if bool(g.get("changed")) else 0.4
    else:
        # auto: compute all deterministic views
        events = ionize_finance(raw_text)
        extracted = {
            "kind": "auto",
            "finance": {
                "events": [
                    {"kind": e.kind, "payload": e.payload, "confidence": e.confidence} for e in events
                ]
            },
            "style": {"profile": style_profile(raw_text)},
            "grammar": grammar_suggest(raw_text),
        }
        if events:
            confidence = sum(e.confidence for e in events) / float(len(events))
        else:
            confidence = 0.5

    tags = payload.tags or []
    tags = [str(t).strip()[:64] for t in tags if t and str(t).strip()]
    tags = list(dict.fromkeys(tags))[:50]

    entry = ExperienceBufferEntry(
        created_by_user_id=user_id,
        project_id=payload.project_id,
        device_id=str(settings.nexus_device_id or "local"),
        source=source,
        modality=modality,
        raw_text=raw_text,
        extracted_json=_dumps(extracted),
        physics_used_json=_dumps(payload.physics_used or {}),
        confidence=confidence,
        feedback=(payload.feedback or "").strip(),
        tags_json=_dumps(tags),
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)

    return NexusIngestTextResponse(ok=True, entry=_to_public(entry))


@router.get("/policy", response_model=NexusPolicyPublic)
def get_parental_policy(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    return NexusPolicyPublic(policy=get_policy(session, user_id))


@router.post("/policy", response_model=NexusPolicyPublic)
def update_parental_policy(
    payload: NexusPolicyUpdateRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    updated = set_policy(session, user_id, payload.policy)
    return NexusPolicyPublic(policy=updated)


@router.get("/intro", response_model=NexusIntroResponse)
def intro(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    policy = get_policy(session, user_id)
    intro = policy.get("intro") if isinstance(policy.get("intro"), dict) else {}
    mode = str(intro.get("mode", "ask")).strip().lower()
    observe_hours = int(intro.get("observe_hours", 24))

    if mode == "observe":
        msg = (
            f"I will silently observe for {observe_hours}h (only what you explicitly send me) "
            "and then ask a few calibration questions. You can change this in /api/nexus/policy."
        )
    else:
        msg = (
            "Quick calibration: what are your top 1–2 goals this week (e.g., budgeting, writing clarity, "
            "prediction projects)? You can switch to silent-observe in /api/nexus/policy."
        )

    return NexusIntroResponse(mode=mode, observe_hours=observe_hours, message=msg)


@router.get("/experience/recent", response_model=NexusListResponse)
def list_recent(
    limit: int = 50,
    project_id: int | None = None,
    modality: str | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, project_id)

    limit = max(1, min(int(limit), 500))
    q = select(ExperienceBufferEntry).where(ExperienceBufferEntry.created_by_user_id == user_id)
    if project_id is not None:
        q = q.where(ExperienceBufferEntry.project_id == project_id)
    if modality:
        q = q.where(ExperienceBufferEntry.modality == modality)
    q = q.order_by(ExperienceBufferEntry.created_at.desc()).limit(limit)

    rows = session.exec(q).all()
    return NexusListResponse(entries=[_to_public(r) for r in rows])


@router.post("/sync/export", response_model=NexusExportResponse)
def export_entries(
    limit: int = 500,
    project_id: int | None = None,
    since: datetime | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, project_id)

    limit = max(1, min(int(limit), 5000))
    q = select(ExperienceBufferEntry).where(ExperienceBufferEntry.created_by_user_id == user_id)
    if project_id is not None:
        q = q.where(ExperienceBufferEntry.project_id == project_id)
    if since is not None:
        q = q.where(ExperienceBufferEntry.created_at >= since)
    q = q.order_by(ExperienceBufferEntry.created_at.asc()).limit(limit)

    rows = session.exec(q).all()
    return NexusExportResponse(
        device_id=str(settings.nexus_device_id or "local"),
        exported_at=datetime.utcnow(),
        entries=[_to_public(r) for r in rows],
    )


@router.post("/feedback/ionize", response_model=NexusFeedbackIonizeResponse)
def ionize_feedback(
    payload: NexusFeedbackIonizeRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, payload.project_id)

    text = (payload.concept_text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="concept_text is required")
    if len(text) > 10_000:
        raise HTTPException(status_code=413, detail="concept_text too large")

    try:
        res = ionize_user_feedback(
            session,
            user_id=user_id,
            project_id=payload.project_id,
            nudge_id=payload.nudge_id,
            hint_tag=payload.hint_tag,
            concept_text=payload.concept_text,
            action=payload.action,
            export_to_hive=bool(payload.export_to_hive),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return NexusFeedbackIonizeResponse(**res)


@router.post("/sync/import", response_model=NexusImportResponse)
def import_entries(
    payload: NexusImportRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    imported = 0
    skipped = 0

    user_id = int(current_user.id or 0)

    for e in payload.entries[:5000]:
        _ensure_project_access(session, user_id, e.project_id)

        existing = session.exec(
            select(ExperienceBufferEntry).where(
                ExperienceBufferEntry.entry_uuid == e.entry_uuid,
                ExperienceBufferEntry.created_by_user_id == user_id,
            )
        ).first()
        if existing:
            skipped += 1
            continue

        entry = ExperienceBufferEntry(
            entry_uuid=e.entry_uuid,
            created_at=e.created_at,
            created_by_user_id=user_id,
            project_id=e.project_id,
            device_id=(e.device_id or "")[:64],
            source=(e.source or "text")[:32],
            modality=(e.modality or "auto")[:32],
            raw_text=(e.raw_text or "")[:200_000],
            extracted_json=_dumps(e.extracted or {}),
            physics_used_json=_dumps(e.physics_used or {}),
            confidence=e.confidence,
            feedback=e.feedback or "",
            tags_json=_dumps([str(t).strip()[:64] for t in (e.tags or []) if str(t).strip()][:50]),
        )
        session.add(entry)
        imported += 1

    session.commit()
    return NexusImportResponse(ok=True, imported=imported, skipped=skipped)
