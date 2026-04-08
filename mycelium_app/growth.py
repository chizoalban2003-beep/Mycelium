from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta

from sqlmodel import Session, select

from mycelium_app.models import GrowthLedgerEntry


def _loads_dict(s: str | None) -> dict:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def compute_growth_stage(
    session: Session,
    *,
    user_id: int,
    project_id: int | None = None,
    window_days: int = 30,
) -> tuple[str, list[str], dict[str, object]]:
    """Compute a simple growth stage from recorded sweep outcomes.

    Stages:
    - infant: mostly observing; few accepted sweeps
    - toddler: some accepted sweeps; begins experimentation
    - adolescent: consistent accepted sweeps; can suggest macro optimizations

    This is intentionally deterministic and transparent.
    """

    window_days = max(1, min(int(window_days), 365))
    since = datetime.utcnow() - timedelta(days=window_days)

    q = select(GrowthLedgerEntry).where(
        GrowthLedgerEntry.created_by_user_id == user_id,
        GrowthLedgerEntry.created_at >= since,
    )
    if project_id is not None:
        q = q.where(GrowthLedgerEntry.project_id == project_id)

    rows = session.exec(q).all()

    accepted = [r for r in rows if bool(r.accepted)]
    accepted_count = len(accepted)

    # Telemetry predictive quality proxy: max R2 in telemetry_next_app sweeps.
    telemetry_r2 = [r.score for r in accepted if r.domain == "telemetry_next_app" and r.metric == "r2"]
    best_telemetry_r2 = max(telemetry_r2) if telemetry_r2 else None

    # Grammar: acceptance rate proxy.
    grammar_accept = [r for r in accepted if r.domain.startswith("grammar")]

    stage = "infant"

    predictive_silencing_unlocked = bool(best_telemetry_r2 is not None and float(best_telemetry_r2) >= 0.90)

    # Toddler: active experimentation starts after multiple accepted sweeps.
    if accepted_count >= 3 and (predictive_silencing_unlocked or accepted_count >= 6):
        stage = "toddler"

    # Promote to adolescent if consistent success.
    if accepted_count >= 12 and (best_telemetry_r2 is None or float(best_telemetry_r2) >= 0.92):
        stage = "adolescent"

    unlocked: list[str] = []
    if predictive_silencing_unlocked:
        unlocked.append("predictive_silencing")
    if stage in ("toddler", "adolescent"):
        unlocked.append("mini_field_sweeps")
    if stage == "adolescent":
        unlocked.append("macro_optimizations")
        unlocked.append("multi_buffer_zones")

    counts = Counter((r.domain, r.metric, bool(r.accepted)) for r in rows)
    stats: dict[str, object] = {
        "window_days": window_days,
        "n_total": len(rows),
        "n_accepted": accepted_count,
        "best_telemetry_r2": best_telemetry_r2,
        "accepted_domains": dict(Counter(r.domain for r in accepted).most_common(20)),
        "accepted_grammar": len(grammar_accept),
        "raw_counts": {f"{k[0]}:{k[1]}:{'accepted' if k[2] else 'rejected'}": int(v) for k, v in counts.items()},
    }

    return stage, unlocked, stats
