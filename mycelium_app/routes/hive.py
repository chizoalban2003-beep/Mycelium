from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.hive_sync import build_anonymized_report
from mycelium_app.models import ExperienceBufferEntry, HiveGlobalUpdate, HiveOutboxReport, ProjectMember, User
from mycelium_app.parental_policy import get_policy
from mycelium_app.schemas import (
    HiveGlobalUpdateImportRequest,
    HiveGlobalUpdateImportResponse,
    HiveGlobalUpdateListResponse,
    HiveGlobalUpdatePublic,
    HiveOutboxListResponse,
    HiveOutboxStoreResponse,
    HiveReportBuildRequest,
    HiveReportBuildResponse,
    HiveReportPublic,
)
from mycelium_app.settings import settings


router = APIRouter(prefix="/api/hive", tags=["hive"])


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


def _to_report_public(report: dict[str, object], *, project_id: int | None) -> HiveReportPublic:
    meta = report.get("meta") if isinstance(report.get("meta"), dict) else {}
    created_at = datetime.utcnow()
    try:
        # best-effort parse
        raw = meta.get("created_at")
        if isinstance(raw, str):
            created_at = datetime.fromisoformat(raw.replace("Z", ""))
    except Exception:
        pass
    device_id = str(meta.get("device_id", ""))

    return HiveReportPublic(
        created_at=created_at,
        device_id=device_id,
        project_id=project_id,
        report=report,
    )


def _to_update_public(row: HiveGlobalUpdate) -> HiveGlobalUpdatePublic:
    return HiveGlobalUpdatePublic(
        update_uuid=row.update_uuid,
        created_at=row.created_at,
        source=row.source,
        version=row.version,
        update=_loads_dict(row.update_json),
    )


@router.post("/report/build", response_model=HiveReportBuildResponse)
def build_report(
    payload: HiveReportBuildRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    if not bool(settings.hive_enabled):
        raise HTTPException(status_code=403, detail="HiveSync disabled")

    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, payload.project_id)

    policy = get_policy(session, user_id)
    privacy = policy.get("privacy") if isinstance(policy.get("privacy"), dict) else {}
    export_enabled = bool(privacy.get("export_enabled"))
    if not export_enabled:
        raise HTTPException(status_code=403, detail="Export disabled by parental policy")

    limit = max(1, min(int(payload.limit or 500), 5000))

    q = select(ExperienceBufferEntry).where(ExperienceBufferEntry.created_by_user_id == user_id)
    if payload.project_id is not None:
        q = q.where(ExperienceBufferEntry.project_id == payload.project_id)
    if payload.since is not None:
        q = q.where(ExperienceBufferEntry.created_at >= payload.since)
    q = q.order_by(ExperienceBufferEntry.created_at.desc()).limit(limit)

    rows = session.exec(q).all()

    report = build_anonymized_report(
        rows,
        device_id=str(settings.nexus_device_id or "local"),
        project_id=payload.project_id,
    )

    return HiveReportBuildResponse(
        ok=True,
        report=_to_report_public(report, project_id=payload.project_id),
    )


@router.post("/outbox/store", response_model=HiveOutboxStoreResponse)
def store_outbox(
    payload: HiveReportBuildRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    # Store is just a convenience wrapper around build_report + DB insert.
    res = build_report(payload, current_user=current_user, session=session)
    user_id = int(current_user.id or 0)

    row = HiveOutboxReport(
        created_by_user_id=user_id,
        project_id=payload.project_id,
        device_id=str(settings.nexus_device_id or "local"),
        report_json=_dumps(res.report.report),
        submitted_at=None,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return HiveOutboxStoreResponse(ok=True, outbox_id=int(row.id or 0))


@router.get("/outbox/recent", response_model=HiveOutboxListResponse)
def list_outbox(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    if not bool(settings.hive_enabled):
        raise HTTPException(status_code=403, detail="HiveSync disabled")

    user_id = int(current_user.id or 0)
    limit = max(1, min(int(limit), 500))

    q = (
        select(HiveOutboxReport)
        .where(HiveOutboxReport.created_by_user_id == user_id)
        .order_by(HiveOutboxReport.created_at.desc())
        .limit(limit)
    )

    rows = session.exec(q).all()
    reports: list[HiveReportPublic] = []
    for r in rows:
        rep = _loads_dict(r.report_json)
        reports.append(
            HiveReportPublic(
                created_at=r.created_at,
                device_id=r.device_id,
                project_id=r.project_id,
                report=rep,
            )
        )

    return HiveOutboxListResponse(reports=reports)


@router.post("/updates/import", response_model=HiveGlobalUpdateImportResponse)
def import_global_update(
    payload: HiveGlobalUpdateImportRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    if not bool(settings.hive_enabled):
        raise HTTPException(status_code=403, detail="HiveSync disabled")

    _ = current_user

    update_uuid = (payload.update_uuid or "").strip() or None
    if update_uuid:
        existing = session.exec(select(HiveGlobalUpdate).where(HiveGlobalUpdate.update_uuid == update_uuid)).first()
        if existing:
            # Idempotent import
            return HiveGlobalUpdateImportResponse(ok=True, update_uuid=existing.update_uuid)

    row = HiveGlobalUpdate(
        source=(payload.source or "manual_import")[:64],
        version=(payload.version or "")[:64],
        update_json=_dumps(payload.update or {}),
    )
    if update_uuid:
        row.update_uuid = update_uuid
    session.add(row)
    session.commit()
    session.refresh(row)

    return HiveGlobalUpdateImportResponse(ok=True, update_uuid=row.update_uuid)


@router.get("/updates/recent", response_model=HiveGlobalUpdateListResponse)
def list_global_updates(
    limit: int = 25,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    if not bool(settings.hive_enabled):
        raise HTTPException(status_code=403, detail="HiveSync disabled")

    _ = current_user

    limit = max(1, min(int(limit), 200))
    rows = session.exec(select(HiveGlobalUpdate).order_by(HiveGlobalUpdate.created_at.desc()).limit(limit)).all()
    return HiveGlobalUpdateListResponse(updates=[_to_update_public(r) for r in rows])
