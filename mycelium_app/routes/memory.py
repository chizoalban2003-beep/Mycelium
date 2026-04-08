from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import AdaptiveMemoryEntry, ProjectMember, User
from mycelium_app.schemas import (
    AdaptiveMemoryDecayRunRequest,
    AdaptiveMemoryDecayRunResponse,
    AdaptiveMemoryEntryPublic,
    AdaptiveMemoryListResponse,
    AdaptiveMemoryReinforceRequest,
    AdaptiveMemoryReinforceResponse,
    AdaptiveMemoryUpsertRequest,
    AdaptiveMemoryUpsertResponse,
)
from mycelium_app.settings import settings
from mycelium_app.stimulus import record_stimulus_event


router = APIRouter(prefix="/api/nexus/memory", tags=["memory"])

_ALLOWED_LANES = {"episodic", "semantic", "procedural"}


def _dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads_dict(s: str | None) -> dict[str, object]:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _loads_list(s: str | None) -> list[str]:
    if not s:
        return []
    try:
        v = json.loads(s)
        if not isinstance(v, list):
            return []
    except Exception:
        return []
    out: list[str] = []
    for item in v:
        val = str(item or "").strip().lower()[:64]
        if val and val not in out:
            out.append(val)
    return out


def _clamp01(x: float) -> float:
    return max(0.0, min(float(x), 1.0))


def _normalize_lane(raw: str) -> str:
    lane = str(raw or "").strip().lower()
    if lane not in _ALLOWED_LANES:
        raise HTTPException(status_code=400, detail="lane must be episodic|semantic|procedural")
    return lane


def _ensure_project_access(session: Session, user_id: int, project_id: int | None) -> None:
    if project_id is None:
        return
    member = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == int(project_id), ProjectMember.user_id == int(user_id))
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a project member")


def _to_public(row: AdaptiveMemoryEntry) -> AdaptiveMemoryEntryPublic:
    return AdaptiveMemoryEntryPublic(
        id=int(row.id or 0),
        created_at=row.created_at,
        updated_at=row.updated_at,
        project_id=row.project_id,
        device_id=str(row.device_id or ""),
        lane=str(row.lane or "episodic"),
        memory_key=str(row.memory_key or ""),
        source=str(row.source or "manual"),
        content=_loads_dict(row.content_json),
        tags=_loads_list(row.tags_json),
        strength=float(row.strength or 0.0),
        decay_half_life_hours=float(row.decay_half_life_hours or 168.0),
        last_reinforced_at=row.last_reinforced_at,
        last_accessed_at=row.last_accessed_at,
    )


@router.post("/upsert", response_model=AdaptiveMemoryUpsertResponse)
def memory_upsert(
    payload: AdaptiveMemoryUpsertRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, payload.project_id)

    lane = _normalize_lane(payload.lane)
    key = str(payload.memory_key or "").strip().lower()[:128]
    if not key:
        raise HTTPException(status_code=400, detail="memory_key is required")

    q = (
        select(AdaptiveMemoryEntry)
        .where(AdaptiveMemoryEntry.created_by_user_id == user_id)
        .where(AdaptiveMemoryEntry.lane == lane)
        .where(AdaptiveMemoryEntry.memory_key == key)
    )
    if payload.project_id is None:
        q = q.where(AdaptiveMemoryEntry.project_id.is_(None))
    else:
        q = q.where(AdaptiveMemoryEntry.project_id == int(payload.project_id))

    row = session.exec(q.order_by(AdaptiveMemoryEntry.updated_at.desc())).first()

    now = datetime.utcnow()
    tags = []
    for t in payload.tags:
        v = str(t or "").strip().lower()[:64]
        if v and v not in tags:
            tags.append(v)

    half_life = max(1.0, min(float(payload.decay_half_life_hours or 168.0), 24.0 * 365.0))
    delta = max(-1.0, min(float(payload.strength_delta), 1.0))

    if row is None:
        row = AdaptiveMemoryEntry(
            created_by_user_id=user_id,
            project_id=payload.project_id,
            device_id=str(settings.nexus_device_id or "local")[:128],
            lane=lane,
            memory_key=key,
            source=str(payload.source or "manual")[:64],
            content_json=_dumps(payload.content),
            tags_json=_dumps(tags),
            strength=_clamp01(0.5 + delta),
            decay_half_life_hours=float(half_life),
            last_reinforced_at=now,
            last_accessed_at=now,
            updated_at=now,
        )
        session.add(row)
    else:
        row.source = str(payload.source or row.source or "manual")[:64]
        row.content_json = _dumps(payload.content)
        row.tags_json = _dumps(tags)
        row.decay_half_life_hours = float(half_life)
        row.strength = _clamp01(float(row.strength or 0.0) + delta)
        row.last_reinforced_at = now
        row.last_accessed_at = now
        row.updated_at = now
        session.add(row)

    session.commit()
    session.refresh(row)

    try:
        record_stimulus_event(
            session,
            user_id=user_id,
            project_id=payload.project_id,
            device_id=str(settings.nexus_device_id or "local"),
            source="memory_api",
            modality="memory",
            signal_type="memory_upsert",
            stimulus={"lane": lane, "memory_key_len": len(key), "strength_delta": float(delta)},
            occurred_at=row.updated_at,
        )
    except Exception:
        pass

    return AdaptiveMemoryUpsertResponse(ok=True, memory=_to_public(row))


@router.get("/list", response_model=AdaptiveMemoryListResponse)
def memory_list(
    lane: str | None = None,
    project_id: int | None = None,
    min_strength: float = 0.0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, project_id)

    lim = max(1, min(int(limit), 500))
    min_s = _clamp01(float(min_strength))

    q = select(AdaptiveMemoryEntry).where(AdaptiveMemoryEntry.created_by_user_id == user_id)
    if project_id is None:
        q = q.where(AdaptiveMemoryEntry.project_id.is_(None))
    else:
        q = q.where(AdaptiveMemoryEntry.project_id == int(project_id))

    if lane is not None:
        q = q.where(AdaptiveMemoryEntry.lane == _normalize_lane(lane))

    q = q.where(AdaptiveMemoryEntry.strength >= float(min_s))
    q = q.order_by(AdaptiveMemoryEntry.strength.desc(), AdaptiveMemoryEntry.updated_at.desc()).limit(lim)

    rows = session.exec(q).all()

    now = datetime.utcnow()
    for row in rows:
        row.last_accessed_at = now
        session.add(row)
    session.commit()

    return AdaptiveMemoryListResponse(memories=[_to_public(r) for r in rows])


@router.post("/{memory_id}/reinforce", response_model=AdaptiveMemoryReinforceResponse)
def memory_reinforce(
    memory_id: int,
    payload: AdaptiveMemoryReinforceRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    row = session.exec(
        select(AdaptiveMemoryEntry).where(
            AdaptiveMemoryEntry.id == int(memory_id), AdaptiveMemoryEntry.created_by_user_id == user_id
        )
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Memory not found")

    _ensure_project_access(session, user_id, row.project_id)

    delta = max(-1.0, min(float(payload.delta), 1.0))
    now = datetime.utcnow()

    row.strength = _clamp01(float(row.strength or 0.0) + delta)
    row.last_reinforced_at = now
    row.last_accessed_at = now
    row.updated_at = now
    session.add(row)
    session.commit()

    try:
        record_stimulus_event(
            session,
            user_id=user_id,
            project_id=payload.project_id,
            device_id=str(settings.nexus_device_id or "local"),
            source="memory_api",
            modality="memory",
            signal_type="memory_reinforce",
            stimulus={"memory_id": int(memory_id), "lane": str(row.lane or ""), "delta": float(delta)},
            occurred_at=now,
        )
    except Exception:
        pass

    session.refresh(row)

    return AdaptiveMemoryReinforceResponse(ok=True, memory=_to_public(row))


@router.post("/decay/run", response_model=AdaptiveMemoryDecayRunResponse)
def memory_decay_run(
    payload: AdaptiveMemoryDecayRunRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, payload.project_id)

    q = select(AdaptiveMemoryEntry).where(AdaptiveMemoryEntry.created_by_user_id == user_id)
    if payload.project_id is None:
        q = q.where(AdaptiveMemoryEntry.project_id.is_(None))
    else:
        q = q.where(AdaptiveMemoryEntry.project_id == int(payload.project_id))

    lane = payload.lane
    if lane is not None:
        q = q.where(AdaptiveMemoryEntry.lane == _normalize_lane(lane))

    rows = session.exec(q.order_by(AdaptiveMemoryEntry.updated_at.desc()).limit(5000)).all()

    min_elapsed = max(0.0, min(float(payload.min_elapsed_hours), 24.0 * 365.0))
    now = datetime.utcnow()

    updated = 0
    total_before = 0.0
    total_after = 0.0

    for row in rows:
        anchor = row.last_reinforced_at or row.updated_at or row.created_at
        elapsed_h = max(0.0, (now - anchor).total_seconds() / 3600.0)
        if elapsed_h < min_elapsed:
            continue

        half_life = max(1.0, float(row.decay_half_life_hours or 168.0))
        before = _clamp01(float(row.strength or 0.0))
        after = _clamp01(before * (0.5 ** (elapsed_h / half_life)))

        if abs(after - before) < 1e-9:
            continue

        row.strength = float(after)
        row.updated_at = now
        session.add(row)

        updated += 1
        total_before += before
        total_after += after

    if updated > 0:
        session.commit()

    mean_before = float(total_before / updated) if updated else 0.0
    mean_after = float(total_after / updated) if updated else 0.0

    try:
        record_stimulus_event(
            session,
            user_id=user_id,
            project_id=payload.project_id,
            device_id=str(settings.nexus_device_id or "local"),
            source="memory_api",
            modality="memory",
            signal_type="memory_decay_run",
            stimulus={"updated": int(updated), "mean_before": float(mean_before), "mean_after": float(mean_after)},
            occurred_at=now,
        )
    except Exception:
        pass

    return AdaptiveMemoryDecayRunResponse(
        ok=True,
        updated=int(updated),
        mean_strength_before=float(round(mean_before, 6)),
        mean_strength_after=float(round(mean_after, 6)),
    )
