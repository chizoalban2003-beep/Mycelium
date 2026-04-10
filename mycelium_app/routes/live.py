from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import json
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import GrowthLedgerEntry, MissionLogLedgerEntry, NexusNudge, ProjectMember, ProjectRole, SignalLedgerEvent, User
from mycelium_app.settings import settings
from mycelium_app.stimulus import record_stimulus_event
from mycelium_app.schemas import LiveHiveEdge, LiveHiveNode, LiveHiveStateResponse, MissionLogEntry, MissionLogPruneRequest, MissionLogPruneResponse
from mycelium_app.viscosity import calculate_live_viscosity


router = APIRouter(prefix="/api/nexus/live", tags=["live"])


@dataclass(frozen=True)
class _MissionLogDraft:
    created_at: datetime
    source_kind: str
    source_ref: str
    mode: str
    tier: str
    title: str
    detail: str
    delta: float | None
    delta_text: str


def _mission_log_key(*, user_id: int, project_id: int | None, source_kind: str, source_ref: str) -> str:
    return f"{int(user_id)}:{'' if project_id is None else int(project_id)}:{str(source_kind).strip().lower()}:{str(source_ref).strip()}"


def _mission_log_row_to_public(row: MissionLogLedgerEntry) -> MissionLogEntry:
    return MissionLogEntry(
        at=row.created_at,
        mode=row.mode,
        tier=row.tier,
        title=row.title,
        detail=row.detail,
        delta=row.delta,
        delta_text=row.delta_text,
        source_kind=row.source_kind,
    )


def _ensure_project_owner(session: Session, user_id: int, project_id: int | None) -> None:
    if project_id is None:
        return
    member = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == int(project_id), ProjectMember.user_id == int(user_id))
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a project member")
    try:
        role = member.role if isinstance(member.role, ProjectRole) else ProjectRole(str(member.role))
    except Exception:
        raise HTTPException(status_code=403, detail="Owner role required")
    if role != ProjectRole.owner:
        raise HTTPException(status_code=403, detail="Owner role required")


def _persist_mission_log_drafts(
    session: Session,
    *,
    user_id: int,
    project_id: int | None,
    device_id: str,
    drafts: list[_MissionLogDraft],
) -> None:
    if not drafts:
        return

    for draft in drafts:
        key = _mission_log_key(
            user_id=user_id,
            project_id=project_id,
            source_kind=draft.source_kind,
            source_ref=draft.source_ref,
        )
        existing = session.exec(
            select(MissionLogLedgerEntry).where(MissionLogLedgerEntry.source_ref == key)
        ).first()
        if existing:
            continue
        session.add(
            MissionLogLedgerEntry(
                created_at=draft.created_at,
                created_by_user_id=int(user_id),
                project_id=project_id,
                device_id=str(device_id or settings.nexus_device_id or "local")[:64],
                source_kind=draft.source_kind,
                source_ref=key,
                mode=draft.mode,
                tier=draft.tier,
                title=draft.title[:160],
                detail=draft.detail[:500],
                delta=draft.delta,
                delta_text=draft.delta_text[:64],
            )
        )

    session.commit()


def _prune_mission_log(session: Session, *, user_id: int, project_id: int | None) -> None:
    retention_days = max(1, min(int(getattr(settings, "nexus_mission_log_retention_days", 14)), 3650))
    since = datetime.utcnow() - timedelta(days=retention_days)
    rows = session.exec(
        select(MissionLogLedgerEntry)
        .where(MissionLogLedgerEntry.created_by_user_id == int(user_id))
        .where(MissionLogLedgerEntry.created_at < since)
        .where(MissionLogLedgerEntry.project_id == project_id)
    ).all()
    for row in rows:
        session.delete(row)
    if rows:
        session.commit()


def _count_mission_log(session: Session, *, user_id: int, project_id: int | None) -> int:
    return len(
        session.exec(
            select(MissionLogLedgerEntry)
            .where(MissionLogLedgerEntry.created_by_user_id == int(user_id))
            .where(MissionLogLedgerEntry.project_id == project_id)
        ).all()
    )


def _load_mission_log(session: Session, *, user_id: int, project_id: int | None, limit: int = 8) -> list[MissionLogEntry]:
    rows = session.exec(
        select(MissionLogLedgerEntry)
        .where(MissionLogLedgerEntry.created_by_user_id == int(user_id))
        .where(MissionLogLedgerEntry.project_id == project_id)
        .order_by(MissionLogLedgerEntry.created_at.desc(), MissionLogLedgerEntry.id.desc())
        .limit(max(1, int(limit)))
    ).all()
    return [_mission_log_row_to_public(row) for row in rows]


def _loads_dict(s: str | None) -> dict:
    if not s:
        return {}
    try:
        value = json.loads(s)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _clamp_delta(value: float, limit: float = 0.25) -> float:
    return max(-float(limit), min(float(limit), float(value)))


def _delta_text(delta: float | None) -> str:
    if delta is None:
        return ""
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
    return f"Δη: {delta:+.2f} {arrow}"


def _float_from_payload(payload: dict, *keys: str) -> float | None:
    for key in keys:
        try:
            value = payload.get(key)
        except Exception:
            value = None
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def build_mission_log(
    *,
    session: Session | None = None,
    user_id: int | None = None,
    project_id: int | None = None,
    device_id: str = "",
    signals: list[SignalLedgerEvent],
    growth: list[GrowthLedgerEntry],
    nudges: list[NexusNudge],
    window_minutes: int,
) -> list[MissionLogEntry]:
    drafts: list[_MissionLogDraft] = []
    window_minutes = max(1, int(window_minutes))
    signal_load = len(signals) / float(window_minutes)
    nudge_pressure = len(nudges) / float(window_minutes)

    for row in signals[:3]:
        payload = _loads_dict(row.payload_json)
        if str(row.signal_type or "").strip().lower() == "synthetic_causal_stress_test":
            surface = payload.get("surface") if isinstance(payload.get("surface"), dict) else {}
            tabular = payload.get("tabular") if isinstance(payload.get("tabular"), dict) else {}
            baseline_temp = _float_from_payload(surface, "baseline_cpu_temp_c", "cpu_temp_c")
            trial_temp = _float_from_payload(surface, "trial_cpu_temp_c", "cpu_temp_c")
            if trial_temp is None:
                trial_temp = _float_from_payload(tabular, "trial_cpu_temp_c", "cpu_temp_c")
            if baseline_temp is None:
                baseline_temp = _float_from_payload(tabular, "baseline_cpu_temp_c", "cpu_temp_c")
            delta_temp = None if baseline_temp is None or trial_temp is None else float(trial_temp - baseline_temp)
            delta = _clamp_delta(0.65 if delta_temp is None else (delta_temp / 50.0), limit=1.0)
            detail_bits = []
            if baseline_temp is not None and trial_temp is not None:
                detail_bits.append(f"{baseline_temp:.1f}°C → {trial_temp:.1f}°C")
            interruptions = _float_from_payload(surface, "trial_interruptions", "interruption_count")
            if interruptions is None:
                interruptions = _float_from_payload(tabular, "trial_interruptions", "interruption_count")
            if interruptions is not None:
                detail_bits.append(f"interruptions {int(interruptions)}")
            detail = " • ".join(detail_bits) if detail_bits else f"{row.device_id or 'local'} • {row.signal_type or 'signal'}"
            drafts.append(
                _MissionLogDraft(
                    created_at=row.created_at,
                    source_kind="diagnostic",
                    source_ref=f"signal:{int(row.id or 0)}",
                    mode="[GATED]",
                    tier="E",
                    title="Thermal spike detected",
                    detail=detail,
                    delta=delta,
                    delta_text=_delta_text(delta),
                )
            )
            continue

        signal_name = str(row.signal_type or "signal").replace("_", " ").strip() or "signal"
        delta = _clamp_delta(-0.08 - (signal_load * 0.6))
        drafts.append(
            _MissionLogDraft(
                created_at=row.created_at,
                source_kind="signal",
                source_ref=f"signal:{int(row.id or 0)}",
                mode="[FLOW]",
                tier="S",
                title=f"{signal_name} observed",
                detail=f"{row.device_id or 'local'} • {row.signal_type or 'signal'}",
                delta=delta,
                delta_text=_delta_text(delta),
            )
        )

    for row in growth[:3]:
        score = float(row.score or 0.0)
        delta = _clamp_delta((score - 0.5) / 2.0)
        tier = "E" if bool(row.accepted) else "Q"
        mode = "[LEARN]" if bool(row.accepted) else "[QUEUE]"
        drafts.append(
            _MissionLogDraft(
                created_at=row.created_at,
                source_kind="growth",
                source_ref=f"growth:{int(row.id or 0)}",
                mode=mode,
                tier=tier,
                title=f"{row.domain or 'growth'} · {row.metric or 'outcome'}",
                detail=f"score {score:.3f} • accepted={str(bool(row.accepted)).lower()}",
                delta=delta,
                delta_text=_delta_text(delta),
            )
        )

    for row in nudges[:2]:
        delta = _clamp_delta(0.04 + (nudge_pressure * 0.5))
        drafts.append(
            _MissionLogDraft(
                created_at=row.created_at,
                source_kind="nudge",
                source_ref=f"nudge:{int(row.id or 0)}",
                mode="[GUARD]",
                tier="Q",
                title=str(row.title or row.kind or "nudge"),
                detail=str(row.message or row.kind or "queued guidance"),
                delta=delta,
                delta_text=_delta_text(delta),
            )
        )

    drafts.sort(key=lambda draft: draft.created_at, reverse=True)

    if session is not None and user_id is not None:
        _persist_mission_log_drafts(session, user_id=int(user_id), project_id=project_id, device_id=device_id, drafts=drafts)
        _prune_mission_log(session, user_id=int(user_id), project_id=project_id)
        return _load_mission_log(session, user_id=int(user_id), project_id=project_id, limit=8)

    return [
        MissionLogEntry(
            at=draft.created_at,
            mode=draft.mode,
            tier=draft.tier,
            title=draft.title,
            detail=draft.detail,
            delta=draft.delta,
            delta_text=draft.delta_text,
            source_kind=draft.source_kind,
        )
        for draft in drafts[:8]
    ]


@router.get("/state", response_model=LiveHiveStateResponse)
def live_state(
    window_minutes: int = 30,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    wm = max(1, min(int(window_minutes), 24 * 60))
    since = datetime.utcnow() - timedelta(minutes=wm)

    signals = session.exec(
        select(SignalLedgerEvent).where(
            SignalLedgerEvent.created_by_user_id == user_id,
            SignalLedgerEvent.created_at >= since,
        ).order_by(SignalLedgerEvent.created_at.desc())
    ).all()
    growth = session.exec(
        select(GrowthLedgerEntry).where(
            GrowthLedgerEntry.created_by_user_id == user_id,
            GrowthLedgerEntry.created_at >= since,
        ).order_by(GrowthLedgerEntry.created_at.desc())
    ).all()
    unseen_nudges = session.exec(
        select(NexusNudge).where(
            NexusNudge.created_by_user_id == user_id,
            NexusNudge.created_at >= since,
            NexusNudge.seen_at.is_(None),
        ).order_by(NexusNudge.created_at.desc())
    ).all()

    sig_counts: Counter[str] = Counter()
    for s in signals:
        sig_counts[str(s.signal_type or "unknown").lower()] += 1

    accepted_growth = sum(1 for g in growth if bool(g.accepted))
    viscosity = calculate_live_viscosity(signals)

    counters = {
        "signals": int(len(signals)),
        "growth_entries": int(len(growth)),
        "accepted_growth": int(accepted_growth),
        "unseen_nudges": int(len(unseen_nudges)),
        "interruptions": int(viscosity.recent_interruptions),
    }

    nodes = [
        LiveHiveNode(id="user", kind="actor", label="You", weight=max(1.0, float(len(signals) / 10.0))),
        LiveHiveNode(id="telemetry", kind="stream", label="Telemetry", weight=max(1.0, float(len(signals) / 20.0))),
        LiveHiveNode(id="growth", kind="memory", label="Growth", weight=max(1.0, float(len(growth) / 10.0))),
        LiveHiveNode(id="nudges", kind="voice", label="Nudges", weight=max(1.0, float(len(unseen_nudges)))),
        LiveHiveNode(id="assistant", kind="agent", label="Assistant", weight=max(1.0, float(accepted_growth + 1))),
    ]

    edges = [
        LiveHiveEdge(source="user", target="telemetry", flow=float(len(signals)), kind="signals"),
        LiveHiveEdge(source="telemetry", target="growth", flow=float(len(growth)), kind="learning"),
        LiveHiveEdge(source="growth", target="assistant", flow=float(accepted_growth), kind="confidence"),
        LiveHiveEdge(source="assistant", target="nudges", flow=float(len(unseen_nudges)), kind="nudge"),
    ]

    # Add top signal types as extra edges for richer animation.
    for name, ct in sig_counts.most_common(4):
        nid = f"sig:{name}"
        nodes.append(LiveHiveNode(id=nid, kind="signal", label=name, weight=max(1.0, float(ct))))
        edges.append(LiveHiveEdge(source=nid, target="telemetry", flow=float(ct), kind="signal_type"))

    try:
        record_stimulus_event(
            session,
            user_id=user_id,
            project_id=None,
            device_id=str(settings.nexus_device_id or "local"),
            source="live_api",
            modality="state",
            signal_type="live_state_view",
            stimulus={"window_minutes": wm, "signals_count": len(signals), "growth_count": len(growth), "unseen_nudges": len(unseen_nudges)},
            occurred_at=datetime.utcnow(),
        )
    except Exception:
        pass

    mission_log = build_mission_log(
        session=session,
        user_id=user_id,
        project_id=None,
        device_id=str(settings.nexus_device_id or "local"),
        signals=signals,
        growth=growth,
        nudges=unseen_nudges,
        window_minutes=wm,
    )

    return LiveHiveStateResponse(
        ok=True,
        as_of=datetime.utcnow(),
        window_minutes=wm,
        counters=counters,
        nodes=nodes,
        edges=edges,
        mission_log=mission_log,
        viscosity=viscosity,
    )


@router.post("/prune", response_model=MissionLogPruneResponse)
def prune_mission_log(
    payload: MissionLogPruneRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    project_id = payload.project_id
    _ensure_project_owner(session, user_id, project_id)

    clear_all = bool(payload.clear_all)
    older_than_hours = payload.older_than_hours

    total_before = _count_mission_log(session, user_id=user_id, project_id=project_id)
    pruned_count = 0
    retention_hours: int | None = None

    if clear_all:
        rows = session.exec(
            select(MissionLogLedgerEntry)
            .where(MissionLogLedgerEntry.created_by_user_id == int(user_id))
            .where(MissionLogLedgerEntry.project_id == project_id)
        ).all()
        for row in rows:
            session.delete(row)
        pruned_count = len(rows)
        retention_hours = None
    else:
        hours = 24 if older_than_hours is None else int(older_than_hours)
        hours = max(1, min(hours, 24 * 3650))
        retention_hours = hours
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        rows = session.exec(
            select(MissionLogLedgerEntry)
            .where(MissionLogLedgerEntry.created_by_user_id == int(user_id))
            .where(MissionLogLedgerEntry.project_id == project_id)
            .where(MissionLogLedgerEntry.created_at < cutoff)
        ).all()
        for row in rows:
            session.delete(row)
        pruned_count = len(rows)

    session.commit()

    remaining_count = max(0, total_before - pruned_count)
    try:
        record_stimulus_event(
            session,
            user_id=user_id,
            project_id=project_id,
            device_id=str(settings.nexus_device_id or "local"),
            source="live_api",
            modality="state",
            signal_type="mission_log_pruned",
            stimulus={
                "project_id": project_id,
                "cleared": clear_all,
                "pruned_count": pruned_count,
                "remaining_count": remaining_count,
                "retention_hours": retention_hours,
            },
            occurred_at=datetime.utcnow(),
        )
    except Exception:
        pass

    return MissionLogPruneResponse(
        ok=True,
        project_id=project_id,
        cleared=clear_all,
        pruned_count=pruned_count,
        remaining_count=remaining_count,
        retention_hours=retention_hours,
    )
