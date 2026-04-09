"""Ecosystem trend analysis — tracks how the digital world evolves over time.

Reads EcosystemTimeSeries rows and computes:
    - Coherence trend (improving / stable / declining)
    - Energy trend
    - Attention entropy trend (diversifying / focusing)
    - Stage progression timeline
    - Force balance evolution
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sqlmodel import Session, select

from mycelium_app.models import EcosystemTimeSeries


def record_ecosystem_tick(
    session: Session,
    *,
    user_id: int,
    field_state: Any = None,
    sedimentation: dict | None = None,
    n_signals: int = 0,
    mae: float | None = None,
) -> None:
    """Record one time series data point from the current learning cycle."""
    row = EcosystemTimeSeries(user_id=user_id, n_signals_window=n_signals)

    if field_state:
        row.total_energy = round(field_state.total_energy, 4)
        row.coherence = round(field_state.agent.coherence, 4) if field_state.agent else 0
        row.n_particles = len(field_state.particles)
        row.n_bonds = field_state.n_bonds
        row.attention_entropy = round(field_state.conservation.entropy, 4) if field_state.conservation else 0
        row.agent_stage = field_state.agent.stage if field_state.agent else "infant"

        forces = field_state.forces_applied or {}
        row.dominant_force = max(forces, key=forces.get, default="") if forces else ""
        row.force_g = round(forces.get("gravity", 0.3), 4)
        row.force_em = round(forces.get("electromagnetic", 0.5), 4)
        row.force_strong = round(forces.get("strong_nuclear", 0.8), 4)
        row.force_weak = round(forces.get("weak_nuclear", 0.02), 4)

        layers = Counter(p.layer for p in field_state.particles)
        row.bedrock_count = layers.get("bedrock", 0)
        row.suspension_count = layers.get("suspension", 0)
        row.turbulent_count = layers.get("turbulent", 0)

    if sedimentation:
        layers = sedimentation.get("layers", {})
        if layers:
            row.bedrock_count = row.bedrock_count or layers.get("bedrock", 0)
            row.suspension_count = row.suspension_count or layers.get("suspension", 0)
            row.turbulent_count = row.turbulent_count or layers.get("turbulent", 0)

    if mae is not None:
        row.mae = round(mae, 6)

    session.add(row)
    session.commit()


def _trend_direction(values: list[float]) -> str:
    """Simple linear trend: improving / stable / declining."""
    if len(values) < 3:
        return "insufficient_data"
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    if abs(slope) < 0.001:
        return "stable"
    return "improving" if slope > 0 else "declining"


def compute_trends(
    session: Session,
    *,
    user_id: int,
    window_hours: int = 168,
) -> dict[str, Any]:
    """Compute ecosystem trends from time series data."""
    since = datetime.utcnow() - timedelta(hours=window_hours)

    rows = session.exec(
        select(EcosystemTimeSeries)
        .where(EcosystemTimeSeries.user_id == int(user_id), EcosystemTimeSeries.created_at >= since)
        .order_by(EcosystemTimeSeries.created_at)
    ).all()

    if len(rows) < 3:
        return {
            "ok": True, "n_points": len(rows),
            "message": "Not enough data points for trend analysis (need 3+)",
            "trends": {},
        }

    coherences = [r.coherence for r in rows]
    energies = [r.total_energy for r in rows]
    entropies = [r.attention_entropy for r in rows]
    particles = [r.n_particles for r in rows]
    bonds = [r.n_bonds for r in rows]
    bedrocks = [r.bedrock_count for r in rows]
    maes = [r.mae for r in rows if r.mae is not None]

    stages = [r.agent_stage for r in rows]
    stage_rank = {"infant": 0, "toddler": 1, "adolescent": 2, "adult": 3}
    stage_values = [stage_rank.get(s, 0) for s in stages]

    # Force balance over time
    force_ratios = []
    for r in rows:
        total = r.force_g + r.force_em + r.force_strong + r.force_weak
        if total > 0:
            force_ratios.append({
                "gravity": round(r.force_g / total, 3),
                "electromagnetic": round(r.force_em / total, 3),
                "strong_nuclear": round(r.force_strong / total, 3),
                "weak_nuclear": round(r.force_weak / total, 3),
            })

    return {
        "ok": True,
        "n_points": len(rows),
        "window_hours": window_hours,
        "period": {
            "from": rows[0].created_at.isoformat() if rows else None,
            "to": rows[-1].created_at.isoformat() if rows else None,
        },
        "trends": {
            "coherence": {
                "direction": _trend_direction(coherences),
                "current": round(coherences[-1], 4) if coherences else 0,
                "min": round(min(coherences), 4),
                "max": round(max(coherences), 4),
                "mean": round(float(np.mean(coherences)), 4),
            },
            "energy": {
                "direction": _trend_direction(energies),
                "current": round(energies[-1], 4) if energies else 0,
                "mean": round(float(np.mean(energies)), 4),
            },
            "attention_entropy": {
                "direction": _trend_direction(entropies),
                "current": round(entropies[-1], 4) if entropies else 0,
                "interpretation": "diversifying" if _trend_direction(entropies) == "improving" else "focusing" if _trend_direction(entropies) == "declining" else "stable",
            },
            "growth_stage": {
                "direction": _trend_direction(stage_values),
                "current": stages[-1] if stages else "infant",
                "history": stages[-10:],
            },
            "prediction_accuracy": {
                "direction": _trend_direction(maes) if maes else "insufficient_data",
                "current_mae": round(maes[-1], 6) if maes else None,
                "best_mae": round(min(maes), 6) if maes else None,
                "interpretation": "Note: lower MAE = better predictions",
            },
            "ecosystem_size": {
                "particles": _trend_direction([float(p) for p in particles]),
                "bonds": _trend_direction([float(b) for b in bonds]),
                "current_particles": particles[-1] if particles else 0,
                "current_bonds": bonds[-1] if bonds else 0,
            },
            "layers": {
                "bedrock": _trend_direction([float(b) for b in bedrocks]),
                "current_bedrock": bedrocks[-1] if bedrocks else 0,
            },
        },
        "force_balance_latest": force_ratios[-1] if force_ratios else None,
        "time_series": {
            "coherence": [round(c, 4) for c in coherences[-50:]],
            "energy": [round(e, 4) for e in energies[-50:]],
            "entropy": [round(e, 4) for e in entropies[-50:]],
        },
    }
