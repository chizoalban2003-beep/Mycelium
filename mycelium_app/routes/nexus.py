from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.causal_trace import dumps_top_shifts, extract_causal_trace
from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.feedback_ionizer import ionize_user_feedback
from mycelium_app.hive_empathy import compute_wisdom_latest
from mycelium_app.models import ExperienceBufferEntry, MetricCausalTrace, MetricSnapshot, ProjectMember, User
from mycelium_app.physics_predictor import (
    BondInfo,
    EquilibriumZone,
    IterationInfo,
    PhysicsPlane,
    PredictionMetrics,
    PredictionResult,
    WeightInfo,
)
from mycelium_app.nexus_ionizer import grammar_suggest, ionize_finance, style_profile
from mycelium_app.parental_policy import get_policy, set_policy
from mycelium_app.stimulus import record_stimulus_event
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
    NexusKnowledgeAuditResponse,
    DeployVersionResponse,
    NexusPolicyPublic,
    NexusPolicyUpdateRequest,
    NexusPrivacyExportStatus,
    NexusPrivacyExportUpdateRequest,
    NexusSyntheticStressTestRequest,
    NexusSyntheticStressTestResponse,
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


def _synthetic_prediction_result(
    *,
    label: str,
    cpu_temp_c: float,
    battery_level: float,
    interruptions: int,
    random_seed: int,
) -> PredictionResult:
    thermal_weight = max(0.05, float(cpu_temp_c - 50.0) / 100.0)
    stability_weight = max(0.02, float(max(0.0, battery_level)) / 100.0)
    interruption_weight = max(0.02, float(max(0, interruptions)) / 10.0)
    weights = [
        WeightInfo(
            feature="cpu_temp_c",
            weight=thermal_weight,
            method="synthetic_telemetry",
            feature_kind="numeric",
            signed=True,
        ),
        WeightInfo(
            feature="battery_level",
            weight=stability_weight,
            method="synthetic_telemetry",
            feature_kind="numeric",
            signed=True,
        ),
        WeightInfo(
            feature="recent_interruptions",
            weight=interruption_weight,
            method="synthetic_telemetry",
            feature_kind="numeric",
            signed=True,
        ),
    ]
    if cpu_temp_c >= 80.0:
        weights.append(
            WeightInfo(
                feature="thermal_load",
                weight=max(0.05, float(cpu_temp_c - 70.0) / 10.0),
                method="synthetic_telemetry",
                feature_kind="numeric",
                signed=True,
            )
        )

    metrics = PredictionMetrics(
        target_kind="numeric",
        n_rows=64,
        n_train=48,
        n_test=16,
        train_fraction=0.75,
        random_seed=int(random_seed),
        n_features_used=len(weights),
        mae=max(0.05, float(cpu_temp_c) / 100.0),
        rmse=max(0.05, float(cpu_temp_c) / 90.0),
    )
    return PredictionResult(
        target=str(label),
        target_kind="numeric",
        plane=PhysicsPlane.solid,
        weights=weights,
        migration_map=[],
        bonding_map=[BondInfo(feature_a="cpu_temp_c", feature_b="battery_level", affinity=0.44, bonding_factor=0.36)],
        iteration_gains=[IterationInfo(cycle=1, test_mae=metrics.mae, test_rmse=metrics.rmse)],
        equilibrium_zones=[
            EquilibriumZone(zone_id=1, features=["cpu_temp_c", "battery_level"], avg_pI=0.4, avg_momentum=0.6, strength=0.5)
        ],
        metrics=metrics,
        preview_rows=[
            {
                "node_id": label,
                "cpu_temp_c": float(cpu_temp_c),
                "battery_level": float(battery_level),
                "interruptions": int(interruptions),
            }
        ],
        diagnostics={"synthetic": True, "random_seed": int(random_seed)},
        test_row_indices=[0],
        test_actual=[float(cpu_temp_c)],
        test_predicted=[float(cpu_temp_c)],
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

    try:
        record_stimulus_event(
            session,
            user_id=user_id,
            project_id=payload.project_id,
            device_id=str(settings.nexus_device_id or "local"),
            source="nexus_api",
            modality=modality,
            signal_type="nexus_ingest_text",
            stimulus={"source": source, "modality": modality, "text_len": len(raw_text), "tags_count": len(tags)},
            occurred_at=entry.created_at,
        )
    except Exception:
        pass

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


@router.get("/privacy/export", response_model=NexusPrivacyExportStatus)
def get_export_status(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    policy = get_policy(session, user_id)
    privacy = policy.get("privacy") if isinstance(policy.get("privacy"), dict) else {}
    return NexusPrivacyExportStatus(
        hive_enabled=bool(getattr(settings, "hive_enabled", False)),
        export_enabled=bool(privacy.get("export_enabled")),
    )


@router.post("/privacy/export", response_model=NexusPrivacyExportStatus)
def update_export_status(
    payload: NexusPrivacyExportUpdateRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    policy = get_policy(session, user_id)
    privacy = policy.get("privacy") if isinstance(policy.get("privacy"), dict) else {}
    updated_policy = {
        **policy,
        "privacy": {
            **privacy,
            "export_enabled": bool(payload.export_enabled),
        },
    }
    updated = set_policy(session, user_id, updated_policy)
    privacy2 = updated.get("privacy") if isinstance(updated.get("privacy"), dict) else {}

    try:
        record_stimulus_event(
            session,
            user_id=user_id,
            project_id=None,
            device_id=str(settings.nexus_device_id or "local"),
            source="nexus_api",
            modality="policy",
            signal_type="nexus_privacy_export_toggle",
            stimulus={"export_enabled": bool(payload.export_enabled)},
            occurred_at=datetime.utcnow(),
        )
    except Exception:
        pass

    return NexusPrivacyExportStatus(
        hive_enabled=bool(getattr(settings, "hive_enabled", False)),
        export_enabled=bool(privacy2.get("export_enabled")),
    )


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


@router.get("/deploy/version", response_model=DeployVersionResponse)
def deploy_version(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ = get_policy(session, user_id)

    app_version = str(
        os.getenv("APP_VERSION")
        or os.getenv("RAILWAY_GIT_COMMIT_MESSAGE")
        or "unknown"
    )[:128]
    git_sha = str(
        os.getenv("GIT_SHA")
        or os.getenv("RAILWAY_GIT_COMMIT_SHA")
        or "unknown"
    )[:128]
    build_id = str(
        os.getenv("BUILD_ID")
        or os.getenv("RAILWAY_DEPLOYMENT_ID")
        or os.getenv("RAILWAY_PROJECT_ID")
        or "unknown"
    )[:128]
    railway_environment = str(
        os.getenv("RAILWAY_ENVIRONMENT")
        or os.getenv("RAILWAY_ENVIRONMENT_NAME")
        or "unknown"
    )[:128]

    return DeployVersionResponse(
        ok=True,
        app_name=str(getattr(settings, "app_name", "Mycelium")),
        app_version=app_version,
        git_sha=git_sha,
        build_id=build_id,
        railway_environment=railway_environment,
    )


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

    try:
        record_stimulus_event(
            session,
            user_id=user_id,
            project_id=project_id,
            device_id=str(settings.nexus_device_id or "local"),
            source="nexus_api",
            modality="sync",
            signal_type="nexus_sync_export",
            stimulus={"limit": limit, "since_provided": bool(since is not None)},
            occurred_at=datetime.utcnow(),
        )
    except Exception:
        pass

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

    try:
        record_stimulus_event(
            session,
            user_id=user_id,
            project_id=payload.project_id,
            device_id=str(settings.nexus_device_id or "local"),
            source="nexus_api",
            modality="feedback",
            signal_type="nexus_feedback_ionize",
            stimulus={"nudge_id": payload.nudge_id, "hint_tag": payload.hint_tag, "export_to_hive": bool(payload.export_to_hive)},
            occurred_at=datetime.utcnow(),
        )
    except Exception:
        pass

    return NexusFeedbackIonizeResponse(**res)


@router.get("/knowledge/audit", response_model=NexusKnowledgeAuditResponse)
def knowledge_audit(
    project_id: int | None = None,
    include_project_scoped: bool = False,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Summarize what the child learned locally vs from the Hive.

    Returns:
    - local: recent user ionized concepts (confirm/correct)
    - hive: current WisdomBroadcast evidence (including top_concepts)
    - validation: recent MetricSnapshots and CausalTraces (Validation Shadow)
    """

    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, project_id)

    limit = max(1, min(int(limit), 200))

    def _loads(s: str | None) -> dict:
        if not s:
            return {}
        try:
            v = json.loads(s)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}

    # Local ionized concepts (stored in ExperienceBufferEntry).
    q = (
        select(ExperienceBufferEntry)
        .where(ExperienceBufferEntry.created_by_user_id == user_id)
        .where(ExperienceBufferEntry.modality == "curiosity_feedback")
        .order_by(ExperienceBufferEntry.created_at.desc())
        .limit(500)
    )
    if project_id is not None:
        q = q.where(ExperienceBufferEntry.project_id == project_id)

    rows = session.exec(q).all()
    local_recent: list[dict[str, object]] = []
    n_confirm = 0
    n_correct = 0

    for r in rows:
        ex = _loads(r.extracted_json)
        if str(ex.get("kind", "")) != "user_feedback_ionized":
            continue
        action = str(ex.get("action", "confirm")).strip().lower()
        hint_tag = str(ex.get("hint_tag", "")).strip()
        concept = str(ex.get("concept", "")).strip()
        if action == "correct":
            n_correct += 1
        else:
            n_confirm += 1

        local_recent.append(
            {
                "created_at": r.created_at.isoformat() + "Z",
                "action": action,
                "hint_tag": hint_tag,
                "concept": concept,
                "nudge_id": ex.get("nudge_id"),
                "digest": ex.get("digest"),
            }
        )
        if len(local_recent) >= limit:
            break

    local_obj: dict[str, object] = {
        "n_entries_scanned": int(len(rows)),
        "n_confirm": int(n_confirm),
        "n_correct": int(n_correct),
        "recent": local_recent,
    }

    # Hive knowledge: reuse compute_wisdom_latest aggregation evidence.
    hive_latest = compute_wisdom_latest(
        session,
        project_id=project_id,
        include_project_scoped=bool(include_project_scoped),
        limit=50,
    )
    hive_obj: dict[str, object] = {
        "as_of": (None if hive_latest.as_of is None else hive_latest.as_of.isoformat() + "Z"),
        "evidence": hive_latest.evidence,
    }

    # Validation Shadow artifacts.
    snap_q = (
        select(MetricSnapshot)
        .where(MetricSnapshot.created_by_user_id == user_id)
        .order_by(MetricSnapshot.created_at.desc())
        .limit(30)
    )
    trace_q = (
        select(MetricCausalTrace)
        .where(MetricCausalTrace.created_by_user_id == user_id)
        .order_by(MetricCausalTrace.created_at.desc())
        .limit(20)
    )
    if project_id is not None:
        snap_q = snap_q.where(MetricSnapshot.project_id == project_id)
        trace_q = trace_q.where(MetricCausalTrace.project_id == project_id)

    snaps = session.exec(snap_q).all()
    traces = session.exec(trace_q).all()

    latest_trace = traces[0] if traces else None
    reasoning_obj: dict[str, object] = {
        "empty": not bool(latest_trace),
        "title": "No recent causal trace" if latest_trace is None else str(latest_trace.metric_name or "Causal trace"),
        "narrative": None if latest_trace is None else str(latest_trace.narrative or "").strip() or None,
        "metric_name": None if latest_trace is None else str(latest_trace.metric_name or ""),
        "improvement_frac": None if latest_trace is None or latest_trace.improvement_frac is None else float(latest_trace.improvement_frac),
        "method": None if latest_trace is None else str(latest_trace.method or ""),
        "top_shifts": [] if latest_trace is None else _loads_list(latest_trace.top_shifts_json)[:3],
        "trace_count": int(len(traces)),
        "snapshot_count": int(len(snaps)),
    }

    validation_obj: dict[str, object] = {
        "recent_snapshots": [
            {
                "id": int(s.id or 0),
                "created_at": s.created_at.isoformat() + "Z",
                "phase": str(s.phase or ""),
                "metric_name": str(s.metric_name or ""),
                "metric_value": float(s.metric_value or 0.0),
                "target_kind": str(s.target_kind or ""),
                "target_col": str(s.target_col or ""),
                "dataset_digest": str(s.dataset_digest or ""),
                "wisdom_digest": str(s.wisdom_digest or ""),
            }
            for s in snaps
        ],
        "recent_traces": [
            {
                "id": int(t.id or 0),
                "created_at": t.created_at.isoformat() + "Z",
                "metric_name": str(t.metric_name or ""),
                "improvement_frac": (None if t.improvement_frac is None else float(t.improvement_frac)),
                "method": str(t.method or ""),
                "narrative": str(t.narrative or ""),
                "top_shifts": _loads_list(t.top_shifts_json),
                "baseline_snapshot_id": int(t.baseline_snapshot_id),
                "trial_snapshot_id": int(t.trial_snapshot_id),
                "dataset_digest": str(t.dataset_digest or ""),
                "wisdom_digest": str(t.wisdom_digest or ""),
            }
            for t in traces
        ],
    }

    return NexusKnowledgeAuditResponse(
        ok=True,
        as_of=datetime.utcnow(),
        project_id=project_id,
        local=local_obj,
        hive=hive_obj,
        validation=validation_obj,
        reasoning=reasoning_obj,
    )


@router.post("/diagnostics/stress-test", response_model=NexusSyntheticStressTestResponse)
def diagnostics_stress_test(
    payload: NexusSyntheticStressTestRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Run a synthetic CPU-temperature spike through the causal-trace path."""

    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, payload.project_id)

    policy = get_policy(session, user_id)
    actions_cfg = policy.get("actions") if isinstance(policy.get("actions"), dict) else {}
    if bool(actions_cfg.get("kill_switch", False)):
        raise HTTPException(status_code=423, detail="Action kill-switch is enabled")

    spike_label = str(payload.spike_label or "cpu_temp_spike").strip()[:64] or "cpu_temp_spike"
    node_id = str(payload.node_id or "node-0").strip()[:64] or "node-0"
    metric_name = str(payload.metric_name or "thermal_headroom").strip()[:64] or "thermal_headroom"
    target_col = str(payload.target_col or "cpu_temp_c").strip()[:64] or "cpu_temp_c"

    baseline = _synthetic_prediction_result(
        label=f"{node_id}:baseline",
        cpu_temp_c=float(payload.baseline_cpu_temp_c),
        battery_level=float(payload.baseline_battery_level),
        interruptions=int(payload.baseline_interruptions),
        random_seed=17,
    )
    trial = _synthetic_prediction_result(
        label=f"{node_id}:spike",
        cpu_temp_c=float(payload.trial_cpu_temp_c),
        battery_level=float(payload.trial_battery_level),
        interruptions=int(payload.trial_interruptions),
        random_seed=17,
    )
    trace = extract_causal_trace(baseline, trial, top_k=5)

    now = datetime.utcnow()
    config_digest = hashlib.sha256(
        _dumps(
            {
                "node_id": node_id,
                "spike_label": spike_label,
                "metric_name": metric_name,
                "target_col": target_col,
                "baseline": {
                    "cpu_temp_c": float(payload.baseline_cpu_temp_c),
                    "battery_level": float(payload.baseline_battery_level),
                    "interruptions": int(payload.baseline_interruptions),
                },
                "trial": {
                    "cpu_temp_c": float(payload.trial_cpu_temp_c),
                    "battery_level": float(payload.trial_battery_level),
                    "interruptions": int(payload.trial_interruptions),
                },
            }
        ).encode("utf-8")
    ).hexdigest()

    baseline_value = max(0.0, min(1.0, 1.0 - (float(payload.baseline_cpu_temp_c) / 120.0)))
    trial_value = max(0.0, min(1.0, 1.0 - (float(payload.trial_cpu_temp_c) / 120.0)))

    baseline_row = MetricSnapshot(
        created_by_user_id=user_id,
        project_id=payload.project_id,
        dataset_digest=config_digest,
        wisdom_digest=config_digest,
        phase="baseline",
        target_col=target_col,
        target_kind="numeric",
        metric_name=metric_name,
        metric_value=float(baseline_value),
        kwargs_json=_dumps({"synthetic": True, "node_id": node_id, "spike_label": spike_label, "state": "baseline"}),
        notes=f"synthetic_stress_test;cpu_temp_c={float(payload.baseline_cpu_temp_c):.1f}",
    )
    trial_row = MetricSnapshot(
        created_by_user_id=user_id,
        project_id=payload.project_id,
        dataset_digest=config_digest,
        wisdom_digest=config_digest,
        phase="trial",
        target_col=target_col,
        target_kind="numeric",
        metric_name=metric_name,
        metric_value=float(trial_value),
        kwargs_json=_dumps({"synthetic": True, "node_id": node_id, "spike_label": spike_label, "state": "trial"}),
        notes=f"synthetic_stress_test;cpu_temp_c={float(payload.trial_cpu_temp_c):.1f}",
    )
    session.add(baseline_row)
    session.add(trial_row)
    session.commit()
    session.refresh(baseline_row)
    session.refresh(trial_row)

    causal_trace_id: int | None = None
    narrative = None
    improvement_frac: float | None = None
    top_shifts: list[dict[str, object]] = []
    if trace.ok and baseline_row.id is not None and trial_row.id is not None:
        narrative = str(trace.narrative or "").strip() or None
        top_shifts = trace.top_shifts or []
        improvement_frac = float((baseline_value - trial_value) / abs(baseline_value)) if baseline_value else None
        trace_row = MetricCausalTrace(
            created_by_user_id=user_id,
            project_id=payload.project_id,
            baseline_snapshot_id=int(baseline_row.id),
            trial_snapshot_id=int(trial_row.id),
            dataset_digest=config_digest,
            wisdom_digest=config_digest,
            metric_name=metric_name,
            improvement_frac=improvement_frac,
            method="synthetic_stress_test",
            narrative=narrative or "Synthetic stress test generated a causal trace.",
            top_shifts_json=dumps_top_shifts(trace),
        )
        session.add(trace_row)
        session.commit()
        session.refresh(trace_row)
        causal_trace_id = int(trace_row.id or 0) or None

    try:
        record_stimulus_event(
            session,
            user_id=user_id,
            project_id=payload.project_id,
            device_id=str(settings.nexus_device_id or "local"),
            source="nexus_api",
            modality="diagnostic",
            signal_type="synthetic_causal_stress_test",
            stimulus={
                "node_id": node_id,
                "spike_label": spike_label,
                "baseline_cpu_temp_c": float(payload.baseline_cpu_temp_c),
                "trial_cpu_temp_c": float(payload.trial_cpu_temp_c),
                "metric_name": metric_name,
                "causal_trace_id": causal_trace_id,
            },
            occurred_at=now,
        )
    except Exception:
        pass

    message = (
        f"Synthetic stress test recorded for {node_id}: {spike_label} "
        f"({float(payload.baseline_cpu_temp_c):.1f}°C → {float(payload.trial_cpu_temp_c):.1f}°C)."
    )
    if narrative:
        message = f"{message} {narrative}"

    return NexusSyntheticStressTestResponse(
        ok=True,
        message=message,
        metric_name=metric_name,
        improvement_frac=improvement_frac,
        narrative=narrative,
        top_shifts=top_shifts,
        baseline_snapshot_id=int(baseline_row.id or 0) or None,
        trial_snapshot_id=int(trial_row.id or 0) or None,
        causal_trace_id=causal_trace_id,
        as_of=now,
    )


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

    try:
        record_stimulus_event(
            session,
            user_id=user_id,
            project_id=None,
            device_id=str(settings.nexus_device_id or "local"),
            source="nexus_api",
            modality="sync",
            signal_type="nexus_sync_import",
            stimulus={"imported": imported, "skipped": skipped, "entries_count": len(payload.entries[:5000])},
            occurred_at=datetime.utcnow(),
        )
    except Exception:
        pass

    return NexusImportResponse(ok=True, imported=imported, skipped=skipped)
