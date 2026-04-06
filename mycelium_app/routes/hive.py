from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.hive_empathy import compute_wisdom_latest, queue_wisdom_whisper
from mycelium_app.hive_sync import build_anonymized_report
from mycelium_app.models import (
    ExperienceBufferEntry,
    HiveGlobalUpdate,
    HiveOutboxMessage,
    HiveOutboxReport,
    MetricSnapshot,
    ProjectMember,
    User,
)
from mycelium_app.parental_policy import get_policy
from mycelium_app.schemas import (
    HiveCuriosityConceptImportRequest,
    HiveCuriosityConceptImportResponse,
    HiveCuriosityFeedbackImportRequest,
    HiveCuriosityFeedbackImportResponse,
    HiveHealthPoint,
    HiveHealthResponse,
    HiveGlobalUpdateImportRequest,
    HiveGlobalUpdateImportResponse,
    HiveGlobalUpdateListResponse,
    HiveGlobalUpdatePublic,
    HiveMetricTrend,
    HiveMetricTrendPoint,
    HiveOutboxListResponse,
    HiveOutboxMessageListResponse,
    HiveOutboxMessagePublic,
    HiveOutboxMessageStoreResponse,
    HiveOutboxStoreResponse,
    HiveReportBuildRequest,
    HiveReportBuildResponse,
    HiveReportPublic,
    HiveWisdomLatestResponse,
    HiveWhisperImportRequest,
    HiveWhisperImportResponse,
)
from mycelium_app.settings import settings


router = APIRouter(prefix="/api/hive", tags=["hive"])


def _day_key(dt: datetime) -> str:
    try:
        return dt.date().isoformat()
    except Exception:
        return "unknown"


def _extract_update_kind(update_obj: dict) -> str:
    if not isinstance(update_obj, dict):
        return ""
    k = update_obj.get("kind")
    if isinstance(k, str) and k.strip():
        return k.strip()
    meta = update_obj.get("meta") if isinstance(update_obj.get("meta"), dict) else {}
    mk = meta.get("kind")
    return str(mk or "").strip()


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


def _stable_uuid_from_obj(obj: object) -> str:
    """Compute deterministic UUID-like hex string for idempotent imports."""

    b = _dumps(obj).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


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


def _to_message_public(row: HiveOutboxMessage) -> HiveOutboxMessagePublic:
    return HiveOutboxMessagePublic(
        created_at=row.created_at,
        device_id=row.device_id,
        project_id=row.project_id,
        kind=str(row.kind or ""),
        payload=_loads_dict(row.payload_json),
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


@router.post("/whisper/queue", response_model=HiveOutboxMessageStoreResponse)
def queue_whisper(
    payload: HiveReportBuildRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Queue a wisdom whisper into the Hive outbox.

    We reuse HiveReportBuildRequest shape for convenience:
    - project_id selects scope
    - limit controls how many PhysicsLedger entries to consider
    """

    if not bool(settings.hive_enabled):
        raise HTTPException(status_code=403, detail="HiveSync disabled")

    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, payload.project_id)

    message_id, reason = queue_wisdom_whisper(
        session,
        user_id=user_id,
        project_id=payload.project_id,
        device_id=str(settings.nexus_device_id or "local"),
        limit=int(payload.limit or 200),
    )

    if message_id is None:
        raise HTTPException(status_code=403, detail=f"Whisper not queued: {reason or 'unknown'}")

    return HiveOutboxMessageStoreResponse(ok=True, message_id=int(message_id))


@router.post("/whisper/import", response_model=HiveWhisperImportResponse)
def import_whisper_as_global_update(
    payload: HiveWhisperImportRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Import a wisdom whisper as a HiveGlobalUpdate.

    This is the 'parent-side' ingest path: it stores the whisper into the
    Global Update table so other devices can fetch it as recommended baseline
    settings.
    """

    if not bool(settings.hive_enabled):
        raise HTTPException(status_code=403, detail="HiveSync disabled")

    _ = current_user

    whisper = payload.whisper or {}
    if not isinstance(whisper, dict):
        raise HTTPException(status_code=400, detail="Invalid whisper")

    meta = whisper.get("meta") if isinstance(whisper.get("meta"), dict) else {}
    if str(meta.get("kind", "")) != "wisdom_whisper":
        raise HTTPException(status_code=400, detail="Not a wisdom_whisper payload")

    update_uuid = (payload.update_uuid or "").strip() or _stable_uuid_from_obj(whisper)

    existing = session.exec(select(HiveGlobalUpdate).where(HiveGlobalUpdate.update_uuid == update_uuid)).first()
    if existing:
        return HiveWhisperImportResponse(ok=True, update_uuid=existing.update_uuid, imported=False)

    update_obj = {
        "kind": "wisdom_whisper",
        "whisper": whisper,
    }

    row = HiveGlobalUpdate(
        source=(payload.source or "hive_empathy")[:64],
        version=(payload.version or "whisper_v1")[:64],
        update_json=_dumps(update_obj),
    )
    row.update_uuid = update_uuid
    session.add(row)
    session.commit()
    session.refresh(row)

    return HiveWhisperImportResponse(ok=True, update_uuid=row.update_uuid, imported=True)


@router.post("/curiosity/import", response_model=HiveCuriosityFeedbackImportResponse)
def import_curiosity_feedback_as_global_update(
    payload: HiveCuriosityFeedbackImportRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Import Active Curiosity feedback as a HiveGlobalUpdate.

    This stores privacy-safe feedback (tags + coarse meta) so it can be
    broadcast back to children as aggregated "hints" via /wisdom/latest.
    """

    if not bool(settings.hive_enabled):
        raise HTTPException(status_code=403, detail="HiveSync disabled")

    _ = current_user

    fb = payload.feedback or {}
    if not isinstance(fb, dict):
        raise HTTPException(status_code=400, detail="Invalid feedback")

    meta = fb.get("meta") if isinstance(fb.get("meta"), dict) else {}
    if str(meta.get("kind", "")) != "curiosity_feedback":
        raise HTTPException(status_code=400, detail="Not a curiosity_feedback payload")

    update_uuid = (payload.update_uuid or "").strip() or _stable_uuid_from_obj(fb)

    existing = session.exec(select(HiveGlobalUpdate).where(HiveGlobalUpdate.update_uuid == update_uuid)).first()
    if existing:
        return HiveCuriosityFeedbackImportResponse(ok=True, update_uuid=existing.update_uuid, imported=False)

    update_obj = {
        "kind": "curiosity_feedback",
        "feedback": fb,
    }

    row = HiveGlobalUpdate(
        source=(payload.source or "active_curiosity")[:64],
        version=(payload.version or "curiosity_v1")[:64],
        update_json=_dumps(update_obj),
    )
    row.update_uuid = update_uuid
    session.add(row)
    session.commit()
    session.refresh(row)

    return HiveCuriosityFeedbackImportResponse(ok=True, update_uuid=row.update_uuid, imported=True)


@router.post("/curiosity/concept/import", response_model=HiveCuriosityConceptImportResponse)
def import_curiosity_concept_as_global_update(
    payload: HiveCuriosityConceptImportRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Import UserFeedbackIonizer concept messages as a HiveGlobalUpdate.

    This stores short parent-provided concepts (confirm/correct) so they can be
    aggregated into WisdomBroadcast and reflected back as hints.
    """

    if not bool(settings.hive_enabled):
        raise HTTPException(status_code=403, detail="HiveSync disabled")

    _ = current_user

    concept = payload.concept or {}
    if not isinstance(concept, dict):
        raise HTTPException(status_code=400, detail="Invalid concept")

    meta = concept.get("meta") if isinstance(concept.get("meta"), dict) else {}
    if str(meta.get("kind", "")) != "curiosity_concept":
        raise HTTPException(status_code=400, detail="Not a curiosity_concept payload")

    update_uuid = (payload.update_uuid or "").strip() or _stable_uuid_from_obj(concept)

    existing = session.exec(select(HiveGlobalUpdate).where(HiveGlobalUpdate.update_uuid == update_uuid)).first()
    if existing:
        return HiveCuriosityConceptImportResponse(ok=True, update_uuid=existing.update_uuid, imported=False)

    update_obj = {
        "kind": "curiosity_concept",
        "concept": concept,
    }

    row = HiveGlobalUpdate(
        source=(payload.source or "user_feedback_ionizer")[:64],
        version=(payload.version or "concept_v1")[:64],
        update_json=_dumps(update_obj),
    )
    row.update_uuid = update_uuid
    session.add(row)
    session.commit()
    session.refresh(row)

    return HiveCuriosityConceptImportResponse(ok=True, update_uuid=row.update_uuid, imported=True)


@router.get("/outbox/messages/recent", response_model=HiveOutboxMessageListResponse)
def list_outbox_messages(
    limit: int = 50,
    kind: str | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    if not bool(settings.hive_enabled):
        raise HTTPException(status_code=403, detail="HiveSync disabled")

    user_id = int(current_user.id or 0)
    limit = max(1, min(int(limit), 500))
    kind = (kind or "").strip() or None

    q = select(HiveOutboxMessage).where(HiveOutboxMessage.created_by_user_id == user_id)
    if kind is not None:
        q = q.where(HiveOutboxMessage.kind == kind)
    q = q.order_by(HiveOutboxMessage.created_at.desc()).limit(limit)

    rows = session.exec(q).all()
    return HiveOutboxMessageListResponse(messages=[_to_message_public(r) for r in rows])


@router.get("/wisdom/latest", response_model=HiveWisdomLatestResponse)
def wisdom_latest(
    project_id: int | None = None,
    include_project_scoped: bool = False,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Broadcast aggregated Hive wisdom back to children.

    Reads recent HiveGlobalUpdate rows, extracts wisdom_whisper payloads,
    re-filters allowlisted knobs, and aggregates into a single recommended
    baseline kwargs dict.
    """

    if not bool(settings.hive_enabled):
        raise HTTPException(status_code=403, detail="HiveSync disabled")

    _ = current_user

    res = compute_wisdom_latest(
        session,
        project_id=project_id,
        include_project_scoped=bool(include_project_scoped),
        limit=int(limit),
    )
    return HiveWisdomLatestResponse(
        ok=True,
        as_of=res.as_of,
        project_id=project_id,
        n_updates_considered=int(res.n_updates_considered),
        n_whispers_used=int(res.n_whispers_used),
        recommended_kwargs=dict(res.recommended_kwargs),
        evidence=dict(res.evidence),
    )


@router.get("/health", response_model=HiveHealthResponse)
def hive_health(
    window_days: int = 30,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Hive health: global growth curve + message counts + metric trend.

    Intended as a parent-side dashboard endpoint.
    """

    if not bool(settings.hive_enabled):
        raise HTTPException(status_code=403, detail="Hive disabled")

    _ = int(current_user.id or 0)  # auth only; endpoint is global

    window_days = max(1, min(int(window_days), 180))
    since = datetime.utcnow() - timedelta(days=int(window_days))
    as_of = datetime.utcnow()

    updates = session.exec(
        select(HiveGlobalUpdate).where(HiveGlobalUpdate.created_at >= since).order_by(HiveGlobalUpdate.created_at.asc())
    ).all()

    by_day: dict[str, Counter[str]] = defaultdict(Counter)
    kinds_total: Counter[str] = Counter()

    for r in updates:
        day = _day_key(r.created_at)
        obj = _loads_dict(r.update_json)
        kind = _extract_update_kind(obj)
        if not kind:
            continue
        kinds_total[kind] += 1
        by_day[day]["global_updates"] += 1
        if kind == "wisdom_whisper":
            by_day[day]["wisdom_whisper"] += 1
        if kind == "curiosity_concept":
            by_day[day]["curiosity_concept"] += 1

    msgs = session.exec(
        select(HiveOutboxMessage).where(HiveOutboxMessage.created_at >= since).order_by(HiveOutboxMessage.created_at.asc())
    ).all()
    messages_by_kind: Counter[str] = Counter()
    device_ids: set[str] = set()
    user_ids: set[int] = set()
    for m in msgs:
        messages_by_kind[str(m.kind or "").strip() or "unknown"] += 1
        if m.device_id:
            device_ids.add(str(m.device_id))
        if m.created_by_user_id is not None:
            user_ids.add(int(m.created_by_user_id))

    snaps = session.exec(
        select(MetricSnapshot)
        .where(MetricSnapshot.created_at >= since)
        .where(MetricSnapshot.phase == "trial")
        .order_by(MetricSnapshot.created_at.asc())
    ).all()

    metric_points: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in snaps:
        day = _day_key(s.created_at)
        name = str(s.metric_name or "").strip() or "unknown"
        metric_points[name][day].append(float(s.metric_value or 0.0))

    growth_curve: list[HiveHealthPoint] = []
    for day in sorted(by_day.keys()):
        c = by_day[day]
        growth_curve.append(
            HiveHealthPoint(
                date=day,
                n_global_updates=int(c.get("global_updates", 0)),
                n_wisdom_whispers=int(c.get("wisdom_whisper", 0)),
                n_curiosity_concepts=int(c.get("curiosity_concept", 0)),
            )
        )

    metric_trends: list[HiveMetricTrend] = []
    for metric_name in sorted(metric_points.keys()):
        days = metric_points[metric_name]
        pts: list[HiveMetricTrendPoint] = []
        for day in sorted(days.keys()):
            xs = days[day]
            if not xs:
                continue
            pts.append(HiveMetricTrendPoint(date=day, n=int(len(xs)), avg=float(sum(xs) / float(len(xs)))))
        metric_trends.append(HiveMetricTrend(metric_name=metric_name, points=pts))

    totals = {
        "n_global_updates": int(len(updates)),
        "n_update_kinds": int(len(kinds_total)),
        "n_outbox_messages": int(len(msgs)),
        "n_devices_seen": int(len(device_ids)),
        "n_users_seen": int(len(user_ids)),
        "n_metric_snapshots_trial": int(len(snaps)),
        "n_curiosity_concepts": int(kinds_total.get("curiosity_concept", 0)),
        "n_wisdom_whispers": int(kinds_total.get("wisdom_whisper", 0)),
    }

    return HiveHealthResponse(
        ok=True,
        as_of=as_of,
        window_days=int(window_days),
        totals=totals,
        messages_by_kind=dict(messages_by_kind),
        growth_curve=growth_curve,
        metric_trends=metric_trends,
    )


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
