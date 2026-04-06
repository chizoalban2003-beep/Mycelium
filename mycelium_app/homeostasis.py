from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from sqlmodel import Session, select

from mycelium_app.growth import compute_growth_stage
from mycelium_app.models import (
    ExperienceBufferEntry,
    GrowthLedgerEntry,
    HiveOutboxReport,
    HomeostasisState,
    SignalLedgerEvent,
)
from mycelium_app.parental_policy import get_policy
from mycelium_app.self_reflection import compute_self_reflection
from mycelium_app.settings import settings


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


@dataclass(frozen=True)
class HomeostasisTick:
    state: HomeostasisState
    actions: list[str]


def _disk_health() -> tuple[int, int]:
    """Return (total_bytes, free_bytes) for the filesystem containing storage/."""

    # Prefer the DB/storage directory, since that’s what grows.
    p = Path("storage")
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        p = Path(".")

    usage = shutil.disk_usage(str(p))
    return int(usage.total), int(usage.free)


def _venv_present() -> bool:
    """Best-effort check for a local venv.

    This is not required for production (system python is fine), but it’s a
    useful self-repair signal for dev environments.
    """

    venv_python = Path(".venv/bin/python")
    return bool(venv_python.exists())


def _get_or_create_state(
    session: Session,
    *,
    user_id: int,
    project_id: int | None,
) -> HomeostasisState:
    q = select(HomeostasisState).where(HomeostasisState.user_id == user_id)
    if project_id is None:
        q = q.where(HomeostasisState.project_id.is_(None))
    else:
        q = q.where(HomeostasisState.project_id == project_id)

    row = session.exec(q).first()
    if row:
        return row

    row = HomeostasisState(user_id=user_id, project_id=project_id)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def _prune_if_needed(session: Session, *, user_id: int, project_id: int | None, free_bytes: int) -> list[str]:
    """Prune low-value memory when disk space is under pressure."""

    actions: list[str] = []
    min_free = int(settings.nexus_homeostasis_min_free_mb) * 1024 * 1024
    if free_bytes >= min_free:
        return actions

    now = datetime.utcnow()

    # 1) Telemetry signals: keep only a limited horizon.
    sig_days = max(1, min(int(settings.nexus_homeostasis_prune_signal_retention_days), 365))
    sig_since = now - timedelta(days=sig_days)
    q_sig = select(SignalLedgerEvent).where(
        SignalLedgerEvent.created_by_user_id == user_id,
        SignalLedgerEvent.created_at < sig_since,
    )
    if project_id is not None:
        q_sig = q_sig.where(SignalLedgerEvent.project_id == project_id)
    old_sigs = session.exec(q_sig).all()
    for r in old_sigs:
        session.delete(r)
    if old_sigs:
        actions.append(f"pruned_signal_events={len(old_sigs)}")

    # 2) Growth ledger: prune old *rejected* sweeps first (keep the successes as identity).
    growth_days = max(7, min(int(settings.nexus_homeostasis_prune_growth_retention_days), 3650))
    growth_since = now - timedelta(days=growth_days)
    q_g = select(GrowthLedgerEntry).where(
        GrowthLedgerEntry.created_by_user_id == user_id,
        GrowthLedgerEntry.created_at < growth_since,
        GrowthLedgerEntry.accepted == False,  # noqa: E712
    )
    if project_id is not None:
        q_g = q_g.where(GrowthLedgerEntry.project_id == project_id)
    old_growth = session.exec(q_g).all()
    for r in old_growth:
        session.delete(r)
    if old_growth:
        actions.append(f"pruned_rejected_growth={len(old_growth)}")

    # 3) Experience buffer: prune low-confidence items beyond horizon.
    exp_days = max(7, min(int(settings.nexus_homeostasis_prune_experience_retention_days), 3650))
    exp_since = now - timedelta(days=exp_days)
    conf_lt = float(settings.nexus_homeostasis_prune_experience_confidence_lt)

    q_e = select(ExperienceBufferEntry).where(
        ExperienceBufferEntry.created_by_user_id == user_id,
        ExperienceBufferEntry.created_at < exp_since,
    )
    if project_id is not None:
        q_e = q_e.where(ExperienceBufferEntry.project_id == project_id)

    rows = session.exec(q_e).all()
    to_delete: list[ExperienceBufferEntry] = []
    for r in rows:
        # Prefer to keep high-confidence or explicitly tagged entries.
        c = r.confidence
        if c is None:
            continue
        if float(c) < conf_lt:
            to_delete.append(r)

    for r in to_delete:
        session.delete(r)

    if to_delete:
        actions.append(f"pruned_low_conf_experience={len(to_delete)}")

    if actions:
        session.commit()

    return actions


def _maybe_backup_identity_to_hive(
    session: Session,
    *,
    user_id: int,
    project_id: int | None,
    device_id: str,
    identity_hash: str,
    mood: str,
    mood_signal: dict[str, float],
    last_backup_at: datetime | None,
) -> tuple[bool, str | None]:
    """Store an identity backup into Hive outbox.

    This does NOT transmit data. It writes to the local Hive outbox so a
    separate process (or future feature) can submit it.
    """

    if not bool(settings.hive_enabled):
        return False, "hive_disabled"

    policy = get_policy(session, user_id)
    privacy = policy.get("privacy") if isinstance(policy.get("privacy"), dict) else {}
    if not bool(privacy.get("export_enabled")):
        return False, "export_disabled_by_policy"

    interval_h = max(1, min(int(settings.nexus_homeostasis_identity_backup_hours), 24 * 30))
    now = datetime.utcnow()
    if last_backup_at is not None and now - last_backup_at < timedelta(hours=interval_h):
        return False, "backup_not_due"

    stage, unlocked, _stats = compute_growth_stage(session, user_id=user_id, project_id=project_id)

    report = {
        "meta": {
            "created_at": now.isoformat() + "Z",
            "device_id": device_id,
            "project_id": project_id,
            "kind": "identity_backup",
        },
        "identity": {
            "identity_hash": identity_hash,
            "mood": mood,
            "mood_signal": mood_signal,
            "stage": stage,
            "unlocked": unlocked,
        },
    }

    row = HiveOutboxReport(
        created_by_user_id=user_id,
        project_id=project_id,
        device_id=device_id,
        report_json=_dumps(report),
        submitted_at=None,
    )
    session.add(row)
    session.commit()

    return True, None


def tick_homeostasis(
    session: Session,
    *,
    user_id: int,
    project_id: int | None,
) -> HomeostasisTick:
    """Run one homeostasis cycle.

    Actions:
    - Compute self-reflection snapshot (mood + identity)
    - Track agitation persistence and trigger a "deep breath" ledger event
    - Check disk health; prune low-value memory if needed
    - Backup identity hash to local Hive outbox (if enabled) on interval

    This is the first place Nexus "acts" based on how it "feels".
    """

    actions: list[str] = []

    # 1) Broadcast reflection into state.
    window_days = max(1, min(int(settings.nexus_homeostasis_window_days), 365))
    snapshot = compute_self_reflection(
        session,
        user_id=user_id,
        project_id=project_id,
        window_days=window_days,
        top_limit=5,
    )

    state = _get_or_create_state(session, user_id=user_id, project_id=project_id)

    prev_mood = str(state.mood or "")
    state.mood = str(snapshot.mood)
    state.mood_signal_json = _dumps(snapshot.mood_signal)
    state.identity_hash = str(snapshot.identity_hash)

    # 2) Agitation persistence -> deep breath.
    if state.mood == "agitated" and prev_mood == "agitated":
        state.agitated_cycles = int(state.agitated_cycles) + 1
    elif state.mood == "agitated":
        state.agitated_cycles = 1
    else:
        state.agitated_cycles = 0

    trigger_n = max(1, min(int(settings.nexus_homeostasis_agitated_cycles_trigger), 100))
    cooldown_min = max(1, min(int(settings.nexus_homeostasis_deep_breath_cooldown_minutes), 24 * 60))

    now = datetime.utcnow()
    deep_breath_due = (
        state.agitated_cycles >= trigger_n
        and (state.last_deep_breath_at is None or now - state.last_deep_breath_at >= timedelta(minutes=cooldown_min))
    )

    if deep_breath_due:
        tension = float(_loads_dict(state.mood_signal_json).get("tension", 1.0))
        ledger = GrowthLedgerEntry(
            created_by_user_id=user_id,
            project_id=project_id,
            device_id=str(getattr(settings, "nexus_device_id", "local")),
            domain="homeostasis",
            metric="deep_breath",
            score=float(tension),
            accepted=True,
            proposal_json=_dumps({"trigger": "agitated_cycles", "cycles": int(state.agitated_cycles)}),
            outcome_json=_dumps({"action": "deep_breath", "cooldown_minutes": cooldown_min}),
            notes="Deep Breath: agitation persisted; stabilizing internal state.",
        )
        session.add(ledger)
        state.last_deep_breath_at = now
        state.agitated_cycles = 0
        actions.append("deep_breath")

    # 3) Resource health -> prune.
    total_b, free_b = _disk_health()
    state.disk_total_bytes = int(total_b)
    state.disk_free_bytes = int(free_b)

    pruned = _prune_if_needed(session, user_id=user_id, project_id=project_id, free_bytes=free_b)
    actions.extend(pruned)

    # 4) Identity backup -> outbox.
    did_backup, backup_reason = _maybe_backup_identity_to_hive(
        session,
        user_id=user_id,
        project_id=project_id,
        device_id=str(getattr(settings, "nexus_device_id", "local")),
        identity_hash=str(snapshot.identity_hash),
        mood=str(snapshot.mood),
        mood_signal=dict(snapshot.mood_signal),
        last_backup_at=state.last_identity_backup_at,
    )
    if did_backup:
        state.last_identity_backup_at = now
        actions.append("identity_backup_queued")
    elif backup_reason:
        # Only keep a short note.
        state.notes = f"backup: {backup_reason}"[:200]

    # 5) Self-repair signal.
    state.venv_present = bool(_venv_present())

    state.updated_at = now
    session.add(state)
    session.commit()
    session.refresh(state)

    return HomeostasisTick(state=state, actions=actions)


def list_recent_user_ids(session: Session, *, window_hours: int = 24) -> list[int]:
    """Return user ids seen in recent telemetry or growth ledger.

    Used by the background daemon to know who to update.
    """

    window_hours = max(1, min(int(window_hours), 168))
    since = datetime.utcnow() - timedelta(hours=window_hours)

    # SQLModel doesn't have a perfect distinct helper; we keep it simple.
    users: set[int] = set()

    q1 = select(SignalLedgerEvent).where(SignalLedgerEvent.created_at >= since).limit(2000)
    for r in session.exec(q1).all():
        users.add(int(r.created_by_user_id))

    q2 = select(GrowthLedgerEntry).where(GrowthLedgerEntry.created_at >= since).limit(2000)
    for r in session.exec(q2).all():
        users.add(int(r.created_by_user_id))

    return sorted(users)
