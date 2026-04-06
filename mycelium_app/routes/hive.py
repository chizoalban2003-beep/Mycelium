from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user, require_hive_ingest_principal
from mycelium_app.hive_empathy import compute_wisdom_latest, queue_wisdom_whisper
from mycelium_app.hive_sync import build_anonymized_report
from mycelium_app.models import (
    ExperienceBufferEntry,
    HiveDevice,
    HiveGlobalUpdate,
    HiveOutboxMessage,
    HiveOutboxReport,
    MetricSnapshot,
    NexusNudge,
    ProjectMember,
    User,
)
from mycelium_app.parental_policy import get_policy
from mycelium_app.schemas import (
    HiveCuriosityConceptImportRequest,
    HiveCuriosityConceptImportResponse,
    HiveCuriosityFeedbackImportRequest,
    HiveCuriosityFeedbackImportResponse,
    HiveBroadcastImpactEvent,
    HiveHealthPoint,
    HiveHealthSmoothedPoint,
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
    HiveRegressionAlert,
    HiveReportBuildRequest,
    HiveReportBuildResponse,
    HiveReportPublic,
    HiveWisdomLatestResponse,
    HiveWhisperImportRequest,
    HiveWhisperImportResponse,
)
from mycelium_app.settings import settings


router = APIRouter(prefix="/api/hive", tags=["hive"])


def _csv_set(s: str | None) -> set[str]:
    if not s:
        return set()
    parts = [p.strip().lower() for p in str(s).split(",")]
    return {p for p in parts if p}


def _require_health_access(user: User) -> None:
    allow = _csv_set(getattr(settings, "hive_health_allowlist_emails_csv", ""))
    if not allow:
        return
    email = str(getattr(user, "email", "") or "").strip().lower()
    if email and (email in allow):
        return
    raise HTTPException(status_code=403, detail="Hive Health restricted")


def _metric_higher_is_better(metric_name: str) -> bool:
    m = (metric_name or "").strip().lower()
    # Default stance: most error metrics (mae/rmse/mape) are lower-is-better.
    return any(k in m for k in ("accuracy", "acc", "f1", "precision", "recall", "auc", "r2"))


def _safe_pct(delta: float, baseline: float) -> float:
    denom = abs(float(baseline))
    if denom <= 1e-12:
        return 0.0
    return float(delta) / denom


def _rolling_mean(xs: list[float], window: int) -> list[float]:
    if window <= 1:
        return [float(x) for x in xs]
    out: list[float] = []
    buf: list[float] = []
    s = 0.0
    for x in xs:
        xf = float(x)
        buf.append(xf)
        s += xf
        if len(buf) > window:
            s -= buf.pop(0)
        out.append(s / float(len(buf)))
    return out


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


def _enforce_whisper_import_rate_limit(
    session: Session,
    *,
    source: str,
    device_id: str,
) -> None:
    if not bool(getattr(settings, "hive_whisper_import_rate_limit_enabled", True)):
        return

    window_s = max(1, min(int(getattr(settings, "hive_whisper_import_rate_limit_window_seconds", 60) or 60), 86_400))
    max_per_source = max(1, min(int(getattr(settings, "hive_whisper_import_rate_limit_max_per_source", 60) or 60), 50_000))
    max_per_device = max(1, min(int(getattr(settings, "hive_whisper_import_rate_limit_max_per_device", 20) or 20), 50_000))

    since = datetime.utcnow() - timedelta(seconds=window_s)

    q_source = select(HiveGlobalUpdate).where(
        HiveGlobalUpdate.created_at >= since,
        HiveGlobalUpdate.source == str(source or "hive_empathy")[:64],
    )
    n_source = len(session.exec(q_source).all())
    if n_source >= max_per_source:
        raise HTTPException(
            status_code=429,
            detail=(
                "Whisper import throttled (source rate limit). "
                f"Try again later. window={window_s}s limit={max_per_source}"
            ),
        )

    did = str(device_id or "").strip()
    if did:
        needle = f'"device_id":"{did}"'
        q_device = select(HiveGlobalUpdate).where(
            HiveGlobalUpdate.created_at >= since,
            HiveGlobalUpdate.source == str(source or "hive_empathy")[:64],
            HiveGlobalUpdate.update_json.contains(needle),
        )
        n_device = len(session.exec(q_device).all())
        if n_device >= max_per_device:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Whisper import throttled (device rate limit). "
                    f"Try again later. window={window_s}s limit={max_per_device}"
                ),
            )


def _loads_dict(s: str | None) -> dict:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _extract_device_id_any(obj: object, *, depth: int = 0, max_depth: int = 6) -> str:
    """Best-effort device_id extraction from nested Hive update payloads.

    Children include `meta.device_id` inside whisper/concept/feedback objects.
    Hive Health uses this to estimate how many distinct nodes have reported in.
    """

    if depth > max_depth:
        return ""

    if isinstance(obj, dict):
        meta = obj.get("meta")
        if isinstance(meta, dict):
            did = meta.get("device_id")
            if isinstance(did, str) and did.strip():
                return did.strip()

        for v in obj.values():
            did = _extract_device_id_any(v, depth=depth + 1, max_depth=max_depth)
            if did:
                return did
        return ""

    if isinstance(obj, list):
        for v in obj:
            did = _extract_device_id_any(v, depth=depth + 1, max_depth=max_depth)
            if did:
                return did
        return ""

    return ""


def _operator_user_ids(session: Session) -> list[int]:
    """Resolve operator recipients for system nudges.

    If Hive Health is restricted via allowlist, we nudge only those accounts.
    Otherwise, we nudge all active users.
    """

    allow = _csv_set(getattr(settings, "hive_health_allowlist_emails_csv", ""))
    q = select(User).where(User.is_active.is_(True))
    if allow:
        q = q.where(User.email.in_(sorted(allow)))
    rows = session.exec(q).all()
    out: list[int] = []
    for u in rows:
        uid = int(getattr(u, "id", 0) or 0)
        if uid > 0:
            out.append(uid)
    return out


def _maybe_record_device_seen(session: Session, *, device_id: str, source: str) -> bool:
    """Upsert a HiveDevice row. Returns True if this is the first time seen."""

    did = (device_id or "").strip()
    if not did:
        return False

    now = datetime.utcnow()

    row = session.exec(select(HiveDevice).where(HiveDevice.device_id == did)).first()
    if row:
        row.last_seen_at = now
        row.last_source = (source or "")[:64]
        session.add(row)
        session.commit()
        return False

    row = HiveDevice(device_id=did[:128], first_seen_at=now, last_seen_at=now, last_source=(source or "")[:64])
    session.add(row)
    try:
        session.commit()
        return True
    except IntegrityError:
        # Concurrent first-seen: another request inserted it.
        session.rollback()
        return False


def _maybe_nudge_child_connected(
    session: Session,
    *,
    device_id: str,
    source: str,
    update_uuid: str,
    kind: str,
) -> None:
    did = (device_id or "").strip()
    if not did:
        return

    try:
        first_seen = _maybe_record_device_seen(session, device_id=did, source=source)
    except Exception:
        return

    if not first_seen:
        return

    payload = {
        "device_id": did,
        "source": str(source or ""),
        "update_uuid": str(update_uuid or ""),
        "kind": str(kind or ""),
        "first_seen_at": datetime.utcnow().isoformat() + "Z",
    }

    for uid in _operator_user_ids(session):
        n = NexusNudge(
            created_by_user_id=int(uid),
            project_id=None,
            kind="child_connected",
            title="New child connected",
            message=f"First whisper received from device '{did}'.",
            payload_json=_dumps(payload),
        )
        session.add(n)

    try:
        session.commit()
    except Exception:
        session.rollback()


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
    principal: User | None = Depends(require_hive_ingest_principal),
    session: Session = Depends(get_session),
):
    """Import a wisdom whisper as a HiveGlobalUpdate.

    This is the 'parent-side' ingest path: it stores the whisper into the
    Global Update table so other devices can fetch it as recommended baseline
    settings.
    """

    if not bool(settings.hive_enabled):
        raise HTTPException(status_code=403, detail="HiveSync disabled")

    _ = principal

    whisper = payload.whisper or {}
    if not isinstance(whisper, dict):
        raise HTTPException(status_code=400, detail="Invalid whisper")

    meta = whisper.get("meta") if isinstance(whisper.get("meta"), dict) else {}
    if str(meta.get("kind", "")) != "wisdom_whisper":
        raise HTTPException(status_code=400, detail="Not a wisdom_whisper payload")

    source = (payload.source or "hive_empathy")[:64]
    device_id = str(meta.get("device_id") or "").strip()[:128]
    _enforce_whisper_import_rate_limit(session, source=source, device_id=device_id)

    project_id = meta.get("project_id")
    if project_id is not None:
        try:
            pid = int(project_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid meta.project_id")
        if principal is None:
            raise HTTPException(status_code=403, detail="Project-scoped whisper requires user authentication")
        _ensure_project_access(session, int(principal.id or 0), pid)

    update_uuid = (payload.update_uuid or "").strip() or _stable_uuid_from_obj(whisper)

    existing = session.exec(select(HiveGlobalUpdate).where(HiveGlobalUpdate.update_uuid == update_uuid)).first()
    if existing:
        return HiveWhisperImportResponse(ok=True, update_uuid=existing.update_uuid, imported=False)

    update_obj = {
        "kind": "wisdom_whisper",
        "whisper": whisper,
    }

    row = HiveGlobalUpdate(
        source=source,
        version=(payload.version or "whisper_v1")[:64],
        update_json=_dumps(update_obj),
    )
    row.update_uuid = update_uuid
    session.add(row)
    session.commit()
    session.refresh(row)

    if device_id:
        _maybe_nudge_child_connected(
            session,
            device_id=device_id,
            source=source,
            update_uuid=row.update_uuid,
            kind="wisdom_whisper",
        )

    return HiveWhisperImportResponse(ok=True, update_uuid=row.update_uuid, imported=True)


@router.post("/curiosity/import", response_model=HiveCuriosityFeedbackImportResponse)
def import_curiosity_feedback_as_global_update(
    payload: HiveCuriosityFeedbackImportRequest,
    principal: User | None = Depends(require_hive_ingest_principal),
    session: Session = Depends(get_session),
):
    """Import Active Curiosity feedback as a HiveGlobalUpdate.

    This stores privacy-safe feedback (tags + coarse meta) so it can be
    broadcast back to children as aggregated "hints" via /wisdom/latest.
    """

    if not bool(settings.hive_enabled):
        raise HTTPException(status_code=403, detail="HiveSync disabled")

    _ = principal

    fb = payload.feedback or {}
    if not isinstance(fb, dict):
        raise HTTPException(status_code=400, detail="Invalid feedback")

    meta = fb.get("meta") if isinstance(fb.get("meta"), dict) else {}
    if str(meta.get("kind", "")) != "curiosity_feedback":
        raise HTTPException(status_code=400, detail="Not a curiosity_feedback payload")

    project_id = meta.get("project_id")
    if project_id is not None:
        try:
            pid = int(project_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid meta.project_id")
        if principal is None:
            raise HTTPException(status_code=403, detail="Project-scoped curiosity import requires user authentication")
        _ensure_project_access(session, int(principal.id or 0), pid)

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

    device_id = ""
    try:
        device_id = str(meta.get("device_id") or "").strip()
    except Exception:
        device_id = ""
    if device_id:
        _maybe_nudge_child_connected(
            session,
            device_id=device_id,
            source=(payload.source or "active_curiosity")[:64],
            update_uuid=row.update_uuid,
            kind="curiosity_feedback",
        )

    return HiveCuriosityFeedbackImportResponse(ok=True, update_uuid=row.update_uuid, imported=True)


@router.post("/curiosity/concept/import", response_model=HiveCuriosityConceptImportResponse)
def import_curiosity_concept_as_global_update(
    payload: HiveCuriosityConceptImportRequest,
    principal: User | None = Depends(require_hive_ingest_principal),
    session: Session = Depends(get_session),
):
    """Import UserFeedbackIonizer concept messages as a HiveGlobalUpdate.

    This stores short parent-provided concepts (confirm/correct) so they can be
    aggregated into WisdomBroadcast and reflected back as hints.
    """

    if not bool(settings.hive_enabled):
        raise HTTPException(status_code=403, detail="HiveSync disabled")

    _ = principal

    concept = payload.concept or {}
    if not isinstance(concept, dict):
        raise HTTPException(status_code=400, detail="Invalid concept")

    meta = concept.get("meta") if isinstance(concept.get("meta"), dict) else {}
    if str(meta.get("kind", "")) != "curiosity_concept":
        raise HTTPException(status_code=400, detail="Not a curiosity_concept payload")

    project_id = meta.get("project_id")
    if project_id is not None:
        try:
            pid = int(project_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid meta.project_id")
        if principal is None:
            raise HTTPException(status_code=403, detail="Project-scoped concept import requires user authentication")
        _ensure_project_access(session, int(principal.id or 0), pid)

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

    device_id = ""
    try:
        device_id = str(meta.get("device_id") or "").strip()
    except Exception:
        device_id = ""
    if device_id:
        _maybe_nudge_child_connected(
            session,
            device_id=device_id,
            source=(payload.source or "user_feedback_ionizer")[:64],
            update_uuid=row.update_uuid,
            kind="curiosity_concept",
        )

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
    principal: User | None = Depends(require_hive_ingest_principal),
    session: Session = Depends(get_session),
):
    """Broadcast aggregated Hive wisdom back to children.

    Reads recent HiveGlobalUpdate rows, extracts wisdom_whisper payloads,
    re-filters allowlisted knobs, and aggregates into a single recommended
    baseline kwargs dict.
    """

    if not bool(settings.hive_enabled):
        raise HTTPException(status_code=403, detail="HiveSync disabled")

    # If called headlessly (X-Hive-Token), only allow global wisdom.
    if principal is None:
        project_id = None
        include_project_scoped = False
    else:
        _ensure_project_access(session, int(principal.id or 0), project_id)

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
    include_smoothing: bool = False,
    smoothing_window: int = 7,
    include_regression: bool = True,
    regression_baseline_days: int = 14,
    regression_min_last_n: int = 5,
    regression_min_baseline_n: int = 30,
    regression_min_delta: float = 0.01,
    include_broadcast_impact: bool = False,
    broadcast_pre_days: int = 3,
    broadcast_post_days: int = 3,
    broadcast_limit: int = 25,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Hive health: global growth curve + message counts + metric trend.

    Intended as a parent-side dashboard endpoint.
    """

    if not bool(settings.hive_enabled):
        raise HTTPException(status_code=403, detail="Hive disabled")

    _ = int(current_user.id or 0)  # auth only; endpoint is global
    _require_health_access(current_user)

    window_days = max(1, min(int(window_days), 180))
    smoothing_window = max(1, min(int(smoothing_window), 30))
    regression_baseline_days = max(1, min(int(regression_baseline_days), 60))
    regression_min_last_n = max(1, min(int(regression_min_last_n), 10_000))
    regression_min_baseline_n = max(1, min(int(regression_min_baseline_n), 100_000))
    regression_min_delta = float(regression_min_delta)
    broadcast_pre_days = max(1, min(int(broadcast_pre_days), 30))
    broadcast_post_days = max(1, min(int(broadcast_post_days), 30))
    broadcast_limit = max(1, min(int(broadcast_limit), 200))
    since = datetime.utcnow() - timedelta(days=int(window_days))
    as_of = datetime.utcnow()

    device_ids: set[str] = set()
    user_ids: set[int] = set()

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

        # Count nodes that have reported in (even if they only do parent-side imports).
        did = _extract_device_id_any(obj)
        if did:
            device_ids.add(str(did))

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

    growth_curve_smoothed: list[HiveHealthSmoothedPoint] = []
    if include_smoothing and growth_curve:
        xs_global = [float(p.n_global_updates) for p in growth_curve]
        xs_whisper = [float(p.n_wisdom_whispers) for p in growth_curve]
        xs_concept = [float(p.n_curiosity_concepts) for p in growth_curve]
        ma_global = _rolling_mean(xs_global, window=int(smoothing_window))
        ma_whisper = _rolling_mean(xs_whisper, window=int(smoothing_window))
        ma_concept = _rolling_mean(xs_concept, window=int(smoothing_window))
        for p, g, w, c in zip(growth_curve, ma_global, ma_whisper, ma_concept, strict=False):
            growth_curve_smoothed.append(
                HiveHealthSmoothedPoint(
                    date=p.date,
                    global_updates_ma=float(g),
                    wisdom_whispers_ma=float(w),
                    curiosity_concepts_ma=float(c),
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

    regression_alerts: list[HiveRegressionAlert] = []
    if include_regression and metric_trends:
        for trend in metric_trends:
            pts = list(trend.points or [])
            if len(pts) < 2:
                continue
            last = pts[-1]
            if int(last.n) < int(regression_min_last_n):
                continue

            baseline_pts = pts[-(int(regression_baseline_days) + 1) : -1]
            baseline_n = sum(int(p.n) for p in baseline_pts)
            if baseline_n < int(regression_min_baseline_n):
                continue
            baseline_sum = sum(float(p.avg) * float(p.n) for p in baseline_pts)
            baseline_avg = baseline_sum / float(baseline_n)

            last_avg = float(last.avg)
            delta = float(last_avg - baseline_avg)
            delta_pct = float(_safe_pct(delta, baseline_avg))

            higher_better = _metric_higher_is_better(trend.metric_name)
            direction = "higher_better" if higher_better else "lower_better"
            is_regression = (delta < -abs(regression_min_delta)) if higher_better else (delta > abs(regression_min_delta))
            if not is_regression:
                continue

            severity = "critical" if abs(delta) >= (2.0 * abs(regression_min_delta)) else "warn"
            regression_alerts.append(
                HiveRegressionAlert(
                    metric_name=str(trend.metric_name or "unknown"),
                    date=str(last.date),
                    direction=direction,
                    baseline_days=int(regression_baseline_days),
                    baseline_n=int(baseline_n),
                    baseline_avg=float(baseline_avg),
                    last_n=int(last.n),
                    last_avg=float(last_avg),
                    delta=float(delta),
                    delta_pct=float(delta_pct),
                    severity=severity,
                )
            )

    broadcast_impact: list[HiveBroadcastImpactEvent] = []
    if include_broadcast_impact and growth_curve and metric_trends:
        broadcast_days = [p.date for p in growth_curve if int(p.n_wisdom_whispers) > 0]
        if broadcast_days:
            # Build quick index for per-metric day->(avg,n)
            metric_day: dict[str, dict[str, tuple[float, int]]] = {}
            for t in metric_trends:
                d: dict[str, tuple[float, int]] = {}
                for p in (t.points or []):
                    d[str(p.date)] = (float(p.avg), int(p.n))
                metric_day[str(t.metric_name or "unknown")] = d

            # Need a stable ordered list of days for window slicing.
            all_days = sorted({p.date for p in growth_curve})
            day_pos = {d: i for i, d in enumerate(all_days)}

            events: list[HiveBroadcastImpactEvent] = []
            for bday in broadcast_days:
                i = day_pos.get(bday)
                if i is None:
                    continue
                pre_slice = all_days[max(0, i - int(broadcast_pre_days)) : i]
                post_slice = all_days[i + 1 : min(len(all_days), i + 1 + int(broadcast_post_days))]
                if not pre_slice or not post_slice:
                    continue

                for metric_name, dmap in metric_day.items():
                    pre_vals = [(dmap[d][0], dmap[d][1]) for d in pre_slice if d in dmap]
                    post_vals = [(dmap[d][0], dmap[d][1]) for d in post_slice if d in dmap]
                    pre_n = sum(n for _, n in pre_vals)
                    post_n = sum(n for _, n in post_vals)
                    if pre_n <= 0 or post_n <= 0:
                        continue
                    pre_avg = sum(avg * n for avg, n in pre_vals) / float(pre_n)
                    post_avg = sum(avg * n for avg, n in post_vals) / float(post_n)
                    delta = float(post_avg - pre_avg)
                    events.append(
                        HiveBroadcastImpactEvent(
                            broadcast_date=str(bday),
                            metric_name=str(metric_name),
                            pre_days=int(broadcast_pre_days),
                            post_days=int(broadcast_post_days),
                            pre_n=int(pre_n),
                            post_n=int(post_n),
                            pre_avg=float(pre_avg),
                            post_avg=float(post_avg),
                            delta=float(delta),
                            delta_pct=float(_safe_pct(delta, pre_avg)),
                        )
                    )

            # Keep the most informative events (largest absolute delta).
            events.sort(key=lambda e: abs(float(e.delta)), reverse=True)
            broadcast_impact = events[: int(broadcast_limit)]

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
        growth_curve_smoothed=growth_curve_smoothed,
        metric_trends=metric_trends,
        regression_alerts=regression_alerts,
        broadcast_impact=broadcast_impact,
    )


@router.post("/updates/import", response_model=HiveGlobalUpdateImportResponse)
def import_global_update(
    payload: HiveGlobalUpdateImportRequest,
    principal: User | None = Depends(require_hive_ingest_principal),
    session: Session = Depends(get_session),
):
    if not bool(settings.hive_enabled):
        raise HTTPException(status_code=403, detail="HiveSync disabled")

    _ = principal

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
