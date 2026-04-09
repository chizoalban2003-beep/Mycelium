"""Fast path predictor — real-time predictions from force field bonds.

Uses the force field's bond structure as a Markov transition matrix
for instant behavioral predictions (<10ms vs 200ms+ for full electrophoresis).

Use cases:
    - "What app will I open next?" (from app_focus transition bonds)
    - "Is this a focus session or browsing?" (from current particle cluster)
    - "Am I in my morning routine?" (from temporal pattern matching)
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from mycelium_app.force_field import ForceFieldState
from mycelium_app.humanizer import humanize_app


def predict_from_bonds(
    field: ForceFieldState,
    current_signal: str,
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    """Predict next signal from force field bond structure.

    Uses bond strengths as transition probabilities — instant, no
    physics computation needed.
    """
    current = current_signal.lower().strip()
    particle = None
    for p in field.particles:
        if p.name == current:
            particle = p
            break

    if not particle or not particle.bonds:
        return {"prediction": None, "confidence": 0.0, "alternatives": []}

    # Bond strengths as weights
    weighted: dict[str, float] = {}
    for bond_name in particle.bonds:
        strength = particle.bond_strengths.get(bond_name, 0.0)
        if bond_name != current and strength > 0:
            weighted[bond_name] = strength

    if not weighted:
        return {"prediction": None, "confidence": 0.0, "alternatives": []}

    total = sum(weighted.values())
    top = sorted(weighted.items(), key=lambda x: x[1], reverse=True)[:top_k]

    prediction = top[0][0]
    confidence = top[0][1] / total

    return {
        "prediction": prediction,
        "prediction_label": humanize_app(prediction),
        "confidence": round(confidence, 3),
        "alternatives": [
            {"signal": s, "label": humanize_app(s), "probability": round(w / total, 3)}
            for s, w in top
        ],
    }


def classify_session_type(
    field: ForceFieldState,
    active_signals: list[str],
) -> dict[str, Any]:
    """Classify the current session based on which cluster the active
    signals belong to.

    Returns: focus / browsing / communication / mixed / idle
    """
    if not active_signals:
        return {"type": "idle", "confidence": 0.0, "signals": []}

    active_set = {s.lower() for s in active_signals}

    # Classify based on which particles are active and their layers
    active_particles = [p for p in field.particles if p.name in active_set]
    if not active_particles:
        return {"type": "unknown", "confidence": 0.0, "signals": list(active_set)}

    layer_counts = Counter(p.layer for p in active_particles)
    dominant_layer = layer_counts.most_common(1)[0][0]

    # Heuristic classification from app types
    focus_apps = {"code", "vim", "nvim", "terminal", "bash", "zsh", "cursor", "idea", "pycharm"}
    browse_apps = {"chrome", "firefox", "safari", "msedge", "opera", "brave"}
    comm_apps = {"slack", "discord", "telegram", "teams", "zoom", "whatsapp"}

    focus_count = sum(1 for s in active_set if any(f in s for f in focus_apps))
    browse_count = sum(1 for s in active_set if any(b in s for b in browse_apps))
    comm_count = sum(1 for s in active_set if any(c in s for c in comm_apps))

    total = focus_count + browse_count + comm_count
    if total == 0:
        session_type = "mixed"
        confidence = 0.3
    elif focus_count >= browse_count and focus_count >= comm_count:
        session_type = "focus"
        confidence = focus_count / max(1, total)
    elif browse_count >= comm_count:
        session_type = "browsing"
        confidence = browse_count / max(1, total)
    else:
        session_type = "communication"
        confidence = comm_count / max(1, total)

    # Layer enriches classification
    if dominant_layer == "bedrock":
        confidence = min(1.0, confidence + 0.1)

    return {
        "type": session_type,
        "confidence": round(confidence, 3),
        "dominant_layer": dominant_layer,
        "signals": [humanize_app(s) for s in active_set],
        "breakdown": {
            "focus": focus_count,
            "browsing": browse_count,
            "communication": comm_count,
        },
    }


def detect_routine(
    field: ForceFieldState,
    recent_signals: list[str],
    *,
    min_sequence_length: int = 3,
) -> dict[str, Any]:
    """Check if the recent signal sequence matches a known routine
    (a strongly-bonded chain in the force field).
    """
    if len(recent_signals) < min_sequence_length:
        return {"in_routine": False, "confidence": 0.0}

    recent = [s.lower() for s in recent_signals[-10:]]
    particle_map = {p.name: p for p in field.particles}

    # Check bond chain strength
    chain_strengths = []
    for i in range(len(recent) - 1):
        p = particle_map.get(recent[i])
        if p and recent[i + 1] in p.bond_strengths:
            chain_strengths.append(p.bond_strengths[recent[i + 1]])
        else:
            chain_strengths.append(0.0)

    if not chain_strengths:
        return {"in_routine": False, "confidence": 0.0}

    avg_strength = sum(chain_strengths) / len(chain_strengths)
    strong_links = sum(1 for s in chain_strengths if s > 0.3)
    ratio = strong_links / len(chain_strengths)

    in_routine = ratio > 0.5 and avg_strength > 0.2

    return {
        "in_routine": in_routine,
        "confidence": round(avg_strength, 3),
        "chain_length": len(chain_strengths),
        "strong_links": strong_links,
        "sequence": [humanize_app(s) for s in recent],
    }
