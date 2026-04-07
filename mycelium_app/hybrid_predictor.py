from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

from sqlmodel import Session, select

from mycelium_app.models import GrowthLedgerEntry, SignalLedgerEvent
from mycelium_app.settings import settings


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def predict_next_work_session(
    session: Session,
    *,
    user_id: int,
    project_id: int | None,
    window_minutes: int,
) -> dict[str, object]:
    wm = max(15, min(int(window_minutes), 24 * 60))
    now = datetime.utcnow()
    since = now - timedelta(minutes=wm)

    q = select(SignalLedgerEvent).where(
        SignalLedgerEvent.created_by_user_id == int(user_id),
        SignalLedgerEvent.created_at >= since,
    )
    if project_id is None:
        q = q.where(SignalLedgerEvent.project_id.is_(None))
    else:
        q = q.where(SignalLedgerEvent.project_id == int(project_id))

    signals = session.exec(q).all()
    n_signals = int(len(signals))

    min_events = max(1, int(getattr(settings, "hybrid_predictor_min_signal_events", 10) or 10))
    density = _clamp01(float(n_signals) / float(min_events * 2.0))

    recent_cutoff = now - timedelta(minutes=max(10, min(45, wm // 4)))
    n_recent = sum(1 for s in signals if s.created_at >= recent_cutoff)
    recency = _clamp01(float(n_recent) / float(max(n_signals, 1)))

    hour_counts: Counter[int] = Counter()
    type_counts: Counter[str] = Counter()
    for s in signals:
        try:
            hour_counts[int(s.created_at.hour)] += 1
        except Exception:
            pass
        type_counts[str(s.signal_type or "").strip().lower()] += 1

    now_hour = int(now.hour)
    circadian_hits = int(hour_counts.get(now_hour, 0) + hour_counts.get((now_hour - 1) % 24, 0) + hour_counts.get((now_hour + 1) % 24, 0))
    rhythm = _clamp01(float(circadian_hits) / float(max(n_signals, 1)))

    diversity = _clamp01(float(len([k for k, v in type_counts.items() if v > 0])) / 6.0)

    timing_score = _clamp01(0.45 * density + 0.30 * recency + 0.15 * rhythm + 0.10 * diversity)

    gq = select(GrowthLedgerEntry).where(
        GrowthLedgerEntry.created_by_user_id == int(user_id),
        GrowthLedgerEntry.created_at >= since,
    )
    if project_id is None:
        gq = gq.where(GrowthLedgerEntry.project_id.is_(None))
    else:
        gq = gq.where(GrowthLedgerEntry.project_id == int(project_id))
    growth_rows = session.exec(gq).all()

    if growth_rows:
        accepted = sum(1 for r in growth_rows if bool(r.accepted))
        accept_ratio = float(accepted) / float(max(1, len(growth_rows)))
    else:
        accept_ratio = 0.5

    governor_confidence = _clamp01(0.60 * accept_ratio + 0.40 * density)
    confidence_floor = _clamp01(float(getattr(settings, "hybrid_predictor_governor_min_confidence", 0.90) or 0.90))
    governor_ok = bool(governor_confidence >= confidence_floor and n_signals >= min_events)

    reasons: list[str] = []
    if n_signals < min_events:
        reasons.append(f"Need at least {min_events} signal events in the window.")
    if governor_confidence < confidence_floor:
        reasons.append(
            f"Governor confidence {governor_confidence:.2f} is below floor {confidence_floor:.2f}."
        )
    if timing_score < 0.65:
        reasons.append("Timing score is not yet strong enough for auto recommendation.")
    if not reasons:
        reasons.append("Pattern timing and governor checks are aligned.")

    recommend = bool(governor_ok and timing_score >= 0.65)

    suggested_minutes = 25
    if timing_score >= 0.80:
        suggested_minutes = 60
    elif timing_score >= 0.65:
        suggested_minutes = 45

    return {
        "project_id": project_id,
        "recommend": recommend,
        "timing_score": float(round(timing_score, 4)),
        "governor_ok": governor_ok,
        "governor_confidence": float(round(governor_confidence, 4)),
        "confidence_floor": float(round(confidence_floor, 4)),
        "n_signals": n_signals,
        "reasons": reasons[:4],
        "suggested_minutes": int(suggested_minutes),
    }
