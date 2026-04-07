from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import GrowthLedgerEntry, NexusNudge, SignalLedgerEvent, User
from mycelium_app.settings import settings
from mycelium_app.stimulus import record_stimulus_event
from mycelium_app.schemas import LiveHiveEdge, LiveHiveNode, LiveHiveStateResponse, MissionLogEntry
from mycelium_app.viscosity import calculate_live_viscosity


router = APIRouter(prefix="/api/nexus/live", tags=["live"])


def _clamp_delta(value: float, limit: float = 0.25) -> float:
    return max(-float(limit), min(float(limit), float(value)))


def _delta_text(delta: float | None) -> str:
    if delta is None:
        return ""
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
    return f"Δη: {delta:+.2f} {arrow}"


def build_mission_log(
    *,
    signals: list[SignalLedgerEvent],
    growth: list[GrowthLedgerEntry],
    nudges: list[NexusNudge],
    window_minutes: int,
) -> list[MissionLogEntry]:
    entries: list[MissionLogEntry] = []
    window_minutes = max(1, int(window_minutes))
    signal_load = len(signals) / float(window_minutes)
    nudge_pressure = len(nudges) / float(window_minutes)

    for row in signals[:3]:
        signal_name = str(row.signal_type or "signal").replace("_", " ").strip() or "signal"
        delta = _clamp_delta(-0.08 - (signal_load * 0.6))
        entries.append(
            MissionLogEntry(
                at=row.created_at,
                mode="[FLOW]",
                tier="S",
                title=f"{signal_name} observed",
                detail=f"{row.device_id or 'local'} • {row.source or 'signal'}",
                delta=delta,
                delta_text=_delta_text(delta),
                source_kind="signal",
            )
        )

    for row in growth[:3]:
        score = float(row.score or 0.0)
        delta = _clamp_delta((score - 0.5) / 2.0)
        tier = "E" if bool(row.accepted) else "Q"
        mode = "[LEARN]" if bool(row.accepted) else "[QUEUE]"
        entries.append(
            MissionLogEntry(
                at=row.created_at,
                mode=mode,
                tier=tier,
                title=f"{row.domain or 'growth'} · {row.metric or 'outcome'}",
                detail=f"score {score:.3f} • accepted={str(bool(row.accepted)).lower()}",
                delta=delta,
                delta_text=_delta_text(delta),
                source_kind="growth",
            )
        )

    for row in nudges[:2]:
        delta = _clamp_delta(0.04 + (nudge_pressure * 0.5))
        entries.append(
            MissionLogEntry(
                at=row.created_at,
                mode="[GUARD]",
                tier="Q",
                title=str(row.title or row.kind or "nudge"),
                detail=str(row.message or row.kind or "queued guidance"),
                delta=delta,
                delta_text=_delta_text(delta),
                source_kind="nudge",
            )
        )

    entries.sort(key=lambda entry: entry.at, reverse=True)
    return entries[:8]


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

    mission_log = build_mission_log(signals=signals, growth=growth, nudges=unseen_nudges, window_minutes=wm)

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
