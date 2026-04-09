"""Hive readiness check — determines when the ecosystem is mature enough
to join the Hive network.

After ~7 days of signal collection, if the agent has reached toddler+
stage and coherence > 0.4, the system nudges the user to enable Hive
sharing. This ensures the first wisdom whisper is meaningful.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlmodel import Session, select

from mycelium_app.models import EcosystemTimeSeries, NexusNudge, SignalLedgerEvent


def check_hive_readiness(
    session: Session,
    *,
    user_id: int,
    min_days: int = 7,
    min_coherence: float = 0.4,
    min_signals: int = 1000,
) -> dict:
    """Check if the ecosystem is ready to join the Hive.

    Returns a dict with readiness status and reasons.
    """
    now = datetime.utcnow()

    # Count total signals
    total_signals = len(session.exec(
        select(SignalLedgerEvent).where(SignalLedgerEvent.created_by_user_id == int(user_id))
    ).all())

    # Find earliest signal
    earliest = session.exec(
        select(SignalLedgerEvent)
        .where(SignalLedgerEvent.created_by_user_id == int(user_id))
        .order_by(SignalLedgerEvent.created_at.asc())
        .limit(1)
    ).first()

    days_active = 0.0
    if earliest and earliest.created_at:
        days_active = (now - earliest.created_at).total_seconds() / 86400

    # Get latest coherence from time series
    latest_ts = session.exec(
        select(EcosystemTimeSeries)
        .where(EcosystemTimeSeries.user_id == int(user_id))
        .order_by(EcosystemTimeSeries.created_at.desc())
        .limit(1)
    ).first()

    coherence = float(latest_ts.coherence) if latest_ts else 0.0
    stage = str(latest_ts.agent_stage) if latest_ts else "infant"

    # Check readiness criteria
    reasons = []
    ready = True

    if days_active < min_days:
        ready = False
        reasons.append(f"Need {min_days} days of data (currently {days_active:.1f} days)")

    if total_signals < min_signals:
        ready = False
        reasons.append(f"Need {min_signals}+ signals (currently {total_signals})")

    if coherence < min_coherence:
        ready = False
        reasons.append(f"Need coherence > {min_coherence} (currently {coherence:.3f})")

    if stage == "infant":
        ready = False
        reasons.append(f"Need toddler+ stage (currently {stage})")

    return {
        "ready": ready,
        "days_active": round(days_active, 1),
        "total_signals": total_signals,
        "coherence": round(coherence, 4),
        "stage": stage,
        "reasons": reasons if not ready else ["All criteria met"],
    }


def maybe_nudge_hive_readiness(
    session: Session,
    *,
    user_id: int,
) -> bool:
    """Check if it's time to nudge the user about Hive and create the nudge.

    Only nudges once. Returns True if a nudge was created.
    """
    # Check if we already nudged about Hive
    existing = session.exec(
        select(NexusNudge)
        .where(NexusNudge.created_by_user_id == int(user_id))
        .where(NexusNudge.kind == "hive_readiness")
        .limit(1)
    ).first()

    if existing:
        return False

    readiness = check_hive_readiness(session, user_id=user_id)
    if not readiness["ready"]:
        return False

    nudge = NexusNudge(
        created_by_user_id=int(user_id),
        project_id=None,
        kind="hive_readiness",
        title="Your ecosystem is ready for the Hive",
        message=(
            f"After {readiness['days_active']:.0f} days and {readiness['total_signals']} signals, "
            f"your companion has reached the {readiness['stage']} stage with "
            f"{readiness['coherence']*100:.0f}% coherence. "
            "Your ecosystem is mature enough to share anonymized wisdom with other Myco instances. "
            "Enable Hive in your settings to connect with the collective."
        ),
    )
    session.add(nudge)
    session.commit()
    return True
