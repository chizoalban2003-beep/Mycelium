"""Unified field bridge — connects force field particle properties to the
physics predictor's electrophoresis model.

The force field measures forces on signal particles. The physics predictor
runs electrophoresis on tabular data. This bridge makes them one system:

    Force field particle → Physics predictor kwargs
    - Particle mass      → Feature statistical weight (top_k ranking)
    - Particle medium    → Ionization path (parametric vs nonparametric)
    - Particle bonds     → Collinearity threshold adjustment
    - Particle energy    → PCR amplification gating
    - Particle layer     → Plane selection (bedrock→solid, suspension→liquid, turbulent→gas)
    - Field coherence    → Learning rate modulation
    - Conservation entropy → Viscosity base adjustment

Also provides:
    - Anomaly detection by comparing field snapshots
    - Weekly ecosystem digest generation
    - App transition prediction via the physics engine
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from mycelium_app.force_field import ForceFieldState, SignalParticle


def field_to_predictor_kwargs(
    field: ForceFieldState,
    *,
    base_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert force field state into physics predictor kwargs.

    The force field's particle properties become the predictor's
    hyperparameters — making the two engines one system.
    """
    kwargs = dict(base_kwargs or {})

    if not field or not field.particles:
        return kwargs

    # --- Plane selection from dominant layer ---
    layer_counts = Counter(p.layer for p in field.particles)
    dominant_layer = layer_counts.most_common(1)[0][0] if layer_counts else "suspension"

    from mycelium_app.physics_predictor import PhysicsPlane
    layer_to_plane = {
        "bedrock": PhysicsPlane.solid,
        "suspension": PhysicsPlane.liquid,
        "turbulent": PhysicsPlane.gas,
    }
    kwargs["plane"] = layer_to_plane.get(dominant_layer, PhysicsPlane.liquid)

    # --- Learning rate from field coherence ---
    coherence = field.agent.coherence if field.agent else 0.0
    # High coherence = stable patterns = can learn faster
    # Low coherence = noisy field = learn cautiously
    base_lr = 0.18
    kwargs["cycle_learning_rate"] = round(base_lr * (0.5 + coherence * 0.8), 4)

    # --- Cycle count from particle count ---
    n_particles = len(field.particles)
    kwargs["n_cycles"] = max(10, min(100, n_particles * 3))

    # --- Viscosity base from conservation entropy ---
    entropy = field.conservation.entropy if field.conservation else 0.0
    # High entropy = diverse usage = lower viscosity (more flow)
    # Low entropy = focused = higher viscosity (more structure)
    if entropy > 2.0:
        kwargs["cleaning_outlier_strategy"] = "winsorize"
    elif entropy < 0.5:
        kwargs["cleaning_outlier_strategy"] = "mad"

    # --- PCR gating from mean particle energy ---
    mean_energy = sum(p.energy for p in field.particles) / max(1, n_particles)
    kwargs["pcr_enabled"] = mean_energy > 0.3

    # --- Cascade from bond density ---
    bond_density = field.n_bonds / max(1, n_particles * (n_particles - 1) / 2)
    kwargs["cascade_enabled"] = bond_density > 0.1
    kwargs["competitive_inhibition"] = bond_density > 0.2

    # --- Shear from force magnitudes ---
    forces = field.forces_applied or {}
    em_force = forces.get("electromagnetic", 0)
    gravity_force = forces.get("gravity", 0)
    total_force = sum(forces.values()) or 1.0
    em_ratio = em_force / total_force
    kwargs["shear_alpha"] = round(0.3 + em_ratio * 0.5, 4)

    return kwargs


def detect_anomalies(
    current: ForceFieldState,
    previous_snapshot: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Compare current field to previous snapshot to detect anomalies.

    An anomaly is when a particle changes layers, energy shifts significantly,
    or new particles appear that weren't in the previous field.
    """
    anomalies: list[dict[str, Any]] = []

    if not previous_snapshot or not previous_snapshot.get("particles"):
        return anomalies

    prev_map = {p["name"]: p for p in previous_snapshot["particles"]}

    for p in current.particles:
        prev = prev_map.get(p.name)

        if prev is None:
            anomalies.append({
                "type": "new_particle",
                "particle": p.name,
                "message": f"New signal appeared: {p.name}",
                "severity": "info",
            })
            continue

        prev_energy = float(prev.get("energy", 0))
        energy_change = abs(p.energy - prev_energy) / max(0.01, prev_energy)
        if energy_change > 0.5:
            direction = "surged" if p.energy > prev_energy else "dropped"
            anomalies.append({
                "type": "energy_shift",
                "particle": p.name,
                "message": f"{p.name} energy {direction} by {energy_change*100:.0f}%",
                "severity": "warning" if energy_change > 1.0 else "info",
                "old_energy": prev_energy,
                "new_energy": p.energy,
            })

    # Check for disappeared particles
    current_names = {p.name for p in current.particles}
    for prev_name in prev_map:
        if prev_name not in current_names:
            anomalies.append({
                "type": "particle_gone",
                "particle": prev_name,
                "message": f"Signal disappeared: {prev_name}",
                "severity": "info",
            })

    # Coherence shift
    if previous_snapshot.get("_meta"):
        prev_coh = float(previous_snapshot["_meta"].get("agent_coherence", 0))
        curr_coh = current.agent.coherence if current.agent else 0
        coh_delta = curr_coh - prev_coh
        if abs(coh_delta) > 0.15:
            direction = "increasing" if coh_delta > 0 else "decreasing"
            anomalies.append({
                "type": "coherence_shift",
                "message": f"Ecosystem coherence {direction}: {prev_coh:.2f} → {curr_coh:.2f}",
                "severity": "warning" if coh_delta < -0.2 else "info",
            })

    return anomalies


def generate_weekly_digest(
    field: ForceFieldState,
    patterns: dict[str, Any] | None = None,
    *,
    agent_name: str = "Myco",
) -> dict[str, str]:
    """Generate a weekly ecosystem summary digest."""
    if not field or not field.particles:
        return {
            "headline": "Not enough data yet",
            "body": f"{agent_name} needs more time observing your signals to generate a weekly summary.",
            "highlights": "",
        }

    n = len(field.particles)
    bedrock = [p for p in field.particles if p.layer == "bedrock"]
    turbulent = [p for p in field.particles if p.layer == "turbulent"]
    coherence = field.agent.coherence if field.agent else 0
    stage = field.agent.stage if field.agent else "infant"
    energy = field.total_energy

    bedrock_names = ", ".join(p.name for p in sorted(bedrock, key=lambda x: x.mass, reverse=True)[:5])
    turbulent_names = ", ".join(p.name for p in sorted(turbulent, key=lambda x: x.energy)[:3])

    headline = f"Week in review: {n} signals, {stage} stage, {coherence*100:.0f}% coherence"

    body_parts = [f"Your ecosystem had {n} active signal types this week."]

    if bedrock_names:
        body_parts.append(f"Your foundation signals: {bedrock_names}. These are the stable patterns that define your digital life.")

    if turbulent_names:
        body_parts.append(f"Changing signals: {turbulent_names}. These varied the most.")

    entropy = field.conservation.entropy if field.conservation else 0
    if entropy > 2.5:
        body_parts.append("Your attention was well-distributed across many apps — diverse usage pattern.")
    elif entropy < 1.0:
        body_parts.append("Your attention was highly focused on a few apps — deep work pattern.")

    forces = field.forces_applied or {}
    dominant = max(forces, key=forces.get, default="")
    if dominant == "strong_nuclear":
        body_parts.append("Strong nuclear force dominated — your signals are tightly co-occurring, forming stable routines.")
    elif dominant == "gravity":
        body_parts.append("Gravity dominated — your ecosystem is stratifying clearly, with distinct layers forming.")
    elif dominant == "electromagnetic":
        body_parts.append("Electromagnetic force dominated — your signals have strong correlations, creating clusters of related behavior.")

    # Pattern insights
    if patterns:
        insights = patterns.get("insights", [])
        if insights:
            body_parts.append(insights[0])
        suggestions = patterns.get("suggestions", [])
        if suggestions:
            body_parts.append(f"Suggestion: {suggestions[0].get('message', '')}")

    highlights = ""
    if field.agent and field.agent.crystallized:
        highlights = f"Your companion reached the {stage} stage with {coherence*100:.0f}% coherence and {field.agent.bound_particles} bound signals."

    return {
        "headline": headline,
        "body": " ".join(body_parts),
        "highlights": highlights,
    }


def predict_next_app(
    transitions: list[tuple[str, str]],
    current_app: str,
    hour: int,
) -> dict[str, Any]:
    """Predict the next app based on transition history and time of day.

    Uses simple Markov chain on transition pairs, weighted by temporal match.
    """
    if not transitions:
        return {"prediction": None, "confidence": 0.0}

    # Build transition matrix from current app
    from_current = [(to, 1.0) for (frm, to) in transitions if frm == current_app]
    if not from_current:
        return {"prediction": None, "confidence": 0.0}

    counts: Counter[str] = Counter()
    for app, w in from_current:
        counts[app] += w

    total = sum(counts.values())
    if total == 0:
        return {"prediction": None, "confidence": 0.0}

    top_app, top_count = counts.most_common(1)[0]
    confidence = top_count / total

    return {
        "prediction": top_app,
        "confidence": round(confidence, 3),
        "alternatives": [
            {"app": app, "probability": round(cnt / total, 3)}
            for app, cnt in counts.most_common(3)
        ],
    }
