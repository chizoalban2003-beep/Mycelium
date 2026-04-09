"""Unified force-field engine — the grand unification of digital physics.

Every digital signal is a particle with mass, charge, velocity, spin, and
position. The ecosystem is governed by four fundamental forces:

    Gravity (G)         — statistical weight pulls signals toward dense regions
    Electromagnetism (E) — correlation creates attraction/repulsion between signals
    Strong Nuclear (S)   — temporal co-occurrence binds signals that always appear together
    Weak Nuclear (W)     — decay force; signals that don't persist lose energy over time

The agent emerges as a standing wave pattern when force equilibrium is reached.
Time is a fundamental dimension — signals decay, the field evolves continuously.
Attention is conserved — the user has 24 hours; app time is a zero-sum flow.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from mycelium_app.particle_stats import compute_fingerprint, fingerprint_to_particle_props


# ---------------------------------------------------------------------------
# Particle model
# ---------------------------------------------------------------------------

@dataclass
class SignalParticle:
    """A single signal particle in the force field."""

    id: str
    name: str
    signal_type: str

    # Intrinsic properties
    mass: float = 0.0           # statistical weight / significance
    charge: float = 0.0         # correlation polarity (-1 to +1)
    velocity: float = 0.0       # rate of change (how fast pattern shifts)
    spin: float = 0.0           # periodicity (0 = aperiodic, 1 = perfectly periodic)
    energy: float = 1.0         # signal strength (decays over time)

    # Position in the ecosystem (from force equilibrium)
    x: float = 0.0
    y: float = 0.0              # depth (negative = foundation, positive = surface)
    z: float = 0.0

    # Force accumulator
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0

    # Bonds
    bonds: list[str] = field(default_factory=list)
    bond_strengths: dict[str, float] = field(default_factory=dict)

    # Statistical fingerprint
    normality_p: float | None = None
    medium: str = "unknown"     # fluid / crystalline / gaseous / frozen
    autocorrelation: float = 0.0
    stationarity: float = 0.5
    entropy_stat: float = 0.0
    skewness: float = 0.0
    kurtosis_stat: float = 0.0

    # Force vectors (for visualization)
    gravity_vec: tuple[float, float, float] = (0.0, 0.0, 0.0)
    em_vec: tuple[float, float, float] = (0.0, 0.0, 0.0)
    strong_vec: tuple[float, float, float] = (0.0, 0.0, 0.0)
    weak_vec: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Momentum (for time evolution)
    vx: float = 0.0
    vy_vel: float = 0.0
    vz: float = 0.0

    # Metadata
    layer: str = "suspension"   # bedrock / suspension / turbulent
    age_hours: float = 0.0
    occurrences: int = 1
    last_seen: str = ""
    value_history: list[float] = field(default_factory=list)


@dataclass
class ForceVector:
    """A single force acting on a particle."""

    fx: float
    fy: float
    fz: float
    source: str                 # gravity / electromagnetic / strong / weak
    magnitude: float = 0.0


@dataclass(frozen=True, slots=True)
class AgentWaveform:
    """The agent's emergent state — a standing wave in the force field."""

    coherence: float            # 0–1, how stable the pattern is
    energy: float               # total bound energy
    center_x: float
    center_y: float
    center_z: float
    bound_particles: int        # number of particles in the standing wave
    dominant_frequency: float   # strongest periodicity
    stage: str                  # infant / toddler / adolescent / adult
    crystallized: bool          # has the agent emerged as a coherent entity


@dataclass
class ConservationState:
    """Attention conservation — the user's 24-hour budget."""

    total_minutes: float = 1440.0       # 24 hours
    allocated: dict[str, float] = field(default_factory=dict)  # app → minutes
    flow_rate: dict[str, float] = field(default_factory=dict)  # app → minutes/hour trend
    entropy: float = 0.0                # how evenly distributed attention is


@dataclass
class ForceFieldState:
    """Complete state of the ecosystem force field at a point in time."""

    particles: list[SignalParticle]
    agent: AgentWaveform
    conservation: ConservationState
    total_energy: float
    mean_coherence: float
    field_age_hours: float
    n_bonds: int
    forces_applied: dict[str, float]    # force_type → total magnitude


# ---------------------------------------------------------------------------
# Force calculations
# ---------------------------------------------------------------------------

_G = 0.3       # gravitational constant
_K_E = 0.5     # electromagnetic constant
_K_S = 0.8     # strong nuclear constant
_K_W = 0.02    # weak nuclear decay rate


def _gravity_force(particle: SignalParticle, all_particles: list[SignalParticle]) -> ForceVector:
    """Heavy (high-mass) particles pull lighter particles toward them.
    Creates vertical stratification — dense signals settle to the bottom.
    """
    fx, fy, fz = 0.0, 0.0, 0.0

    # Self-gravity: mass pulls particle downward (toward bedrock)
    fy -= _G * particle.mass * 2.0

    # Mutual attraction to heavy neighbors
    for other in all_particles:
        if other.id == particle.id:
            continue
        dx = other.x - particle.x
        dy = other.y - particle.y
        dz = other.z - particle.z
        dist_sq = dx * dx + dy * dy + dz * dz + 1.0
        dist = math.sqrt(dist_sq)

        # Only significant attraction from heavy particles
        if other.mass < 0.3:
            continue

        force_mag = _G * particle.mass * other.mass / dist_sq
        fx += (dx / dist) * force_mag * 0.1
        fy += (dy / dist) * force_mag * 0.3
        fz += (dz / dist) * force_mag * 0.1

    magnitude = math.sqrt(fx * fx + fy * fy + fz * fz)
    return ForceVector(fx=fx, fy=fy, fz=fz, source="gravity", magnitude=magnitude)


def _electromagnetic_force(
    particle: SignalParticle,
    all_particles: list[SignalParticle],
    correlation_matrix: dict[str, dict[str, float]],
) -> ForceVector:
    """Correlated signals attract (bonding). Anti-correlated signals repel.
    Creates horizontal clustering — similar signals group together.
    """
    fx, fy, fz = 0.0, 0.0, 0.0
    p_corrs = correlation_matrix.get(particle.name, {})

    for other in all_particles:
        if other.id == particle.id:
            continue

        corr = p_corrs.get(other.name, 0.0)
        if abs(corr) < 0.1:
            continue

        dx = other.x - particle.x
        dy = other.y - particle.y
        dz = other.z - particle.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz) + 0.1

        # Positive correlation = attraction, negative = repulsion
        force_mag = _K_E * corr * particle.energy * other.energy / (dist + 1.0)

        fx += (dx / dist) * force_mag
        fy += (dy / dist) * force_mag * 0.3
        fz += (dz / dist) * force_mag

    magnitude = math.sqrt(fx * fx + fy * fy + fz * fz)
    return ForceVector(fx=fx, fy=fy, fz=fz, source="electromagnetic", magnitude=magnitude)


def _strong_nuclear_force(
    particle: SignalParticle,
    cooccurrence: dict[str, set[str]],
) -> ForceVector:
    """Signals that always appear together are bound — like quarks in a proton.
    Short-range, very strong when close, zero when far.
    """
    fx, fy, fz = 0.0, 0.0, 0.0

    bound_partners = cooccurrence.get(particle.name, set())
    for bond_name, strength in particle.bond_strengths.items():
        if bond_name not in bound_partners:
            continue
        # Strong force pulls toward bond center (creates tight clusters)
        # This is a simplification — in reality it acts on the bond itself
        fy -= _K_S * strength * 0.5  # pull downward (stability)

    magnitude = abs(fy) * _K_S
    return ForceVector(fx=fx, fy=fy, fz=fz, source="strong_nuclear", magnitude=magnitude)


def _weak_nuclear_force(particle: SignalParticle) -> ForceVector:
    """Signals that don't persist lose energy over time.
    Old, infrequent signals decay — they float upward and eventually disappear.
    """
    # Decay pushes particles upward (away from foundation)
    decay = _K_W * particle.age_hours / max(1, particle.occurrences)
    fy = decay * 2.0  # upward push

    # Reduce energy
    energy_loss = _K_W * particle.age_hours * 0.01
    particle.energy = max(0.01, particle.energy - energy_loss)

    return ForceVector(fx=0.0, fy=fy, fz=0.0, source="weak_nuclear", magnitude=abs(fy))


# ---------------------------------------------------------------------------
# Agent emergence (standing wave detection)
# ---------------------------------------------------------------------------

def _detect_agent_emergence(particles: list[SignalParticle]) -> AgentWaveform:
    """Detect if the agent has crystallized as a standing wave.

    The agent emerges when enough particles are bound together with
    sufficient cohesion. It doesn't get created — it crystallizes
    from the force equilibrium.
    """
    if not particles:
        return AgentWaveform(
            coherence=0.0, energy=0.0, center_x=0.0, center_y=0.0, center_z=0.0,
            bound_particles=0, dominant_frequency=0.0, stage="infant", crystallized=False,
        )

    # Find bound cluster (particles with bonds)
    bound = [p for p in particles if len(p.bonds) >= 2 and p.energy > 0.3]
    total_energy = sum(p.energy for p in particles)
    bound_energy = sum(p.energy for p in bound)

    # Coherence = fraction of energy in bound state
    coherence = bound_energy / max(0.01, total_energy)

    # Center of mass of the bound cluster
    if bound:
        cx = sum(p.x * p.energy for p in bound) / max(0.01, bound_energy)
        cy = sum(p.y * p.energy for p in bound) / max(0.01, bound_energy)
        cz = sum(p.z * p.energy for p in bound) / max(0.01, bound_energy)
    else:
        cx, cy, cz = 0.0, 0.0, 0.0

    # Dominant frequency from spin values
    spins = [p.spin for p in bound if p.spin > 0]
    dom_freq = float(np.median(spins)) if spins else 0.0

    # Stage determination from coherence + bound count
    n_bound = len(bound)
    if coherence >= 0.7 and n_bound >= 15:
        stage = "adult"
    elif coherence >= 0.5 and n_bound >= 10:
        stage = "adolescent"
    elif coherence >= 0.3 and n_bound >= 5:
        stage = "toddler"
    else:
        stage = "infant"

    crystallized = coherence >= 0.4 and n_bound >= 5

    return AgentWaveform(
        coherence=round(coherence, 4),
        energy=round(bound_energy, 4),
        center_x=round(cx, 2),
        center_y=round(cy, 2),
        center_z=round(cz, 2),
        bound_particles=n_bound,
        dominant_frequency=round(dom_freq, 4),
        stage=stage,
        crystallized=crystallized,
    )


# ---------------------------------------------------------------------------
# Conservation laws
# ---------------------------------------------------------------------------

def _compute_conservation(
    app_durations: dict[str, float],
    window_hours: float,
) -> ConservationState:
    """Compute the attention conservation state.

    The user has a finite attention budget (24h). App time is zero-sum —
    when one app gains, another loses. Entropy measures how evenly
    distributed the attention is.
    """
    total_minutes = window_hours * 60
    allocated = {app: round(mins / 60, 2) for app, mins in app_durations.items()}

    # Flow rate: minutes per hour of the window
    flow_rate = {}
    for app, mins in app_durations.items():
        flow_rate[app] = round(mins / max(1, window_hours), 4)

    # Shannon entropy of the attention distribution
    total_time = sum(app_durations.values())
    entropy = 0.0
    if total_time > 0:
        for mins in app_durations.values():
            p = mins / total_time
            if p > 0:
                entropy -= p * math.log2(p)

    return ConservationState(
        total_minutes=total_minutes,
        allocated=allocated,
        flow_rate=flow_rate,
        entropy=round(entropy, 4),
    )


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def compute_force_field(
    signals: list[dict[str, Any]],
    *,
    window_hours: float = 24.0,
    n_iterations: int = 30,
    damping: float = 0.85,
) -> ForceFieldState:
    """Compute the complete force field state from raw signal data.

    Parameters
    ----------
    signals : list of signal dicts with keys: signal_type, app_name (optional),
              created_at (ISO string), payload (dict)
    window_hours : how far back the field considers
    n_iterations : force relaxation steps
    damping : velocity damping per iteration

    Returns
    -------
    ForceFieldState with particles, agent waveform, conservation, and metrics.
    """
    if not signals:
        return ForceFieldState(
            particles=[], agent=_detect_agent_emergence([]),
            conservation=ConservationState(), total_energy=0.0,
            mean_coherence=0.0, field_age_hours=0.0, n_bonds=0,
            forces_applied={},
        )

    now = datetime.utcnow()

    # --- Build particles from signals ---
    particle_map: dict[str, SignalParticle] = {}
    app_durations: dict[str, float] = defaultdict(float)
    cooccurrence_window: dict[str, set[str]] = defaultdict(set)
    time_buckets: dict[int, list[str]] = defaultdict(list)

    for sig in signals:
        sig_type = str(sig.get("signal_type", sig.get("type", "unknown"))).lower()
        app_name = str(sig.get("app_name", sig_type)).lower()[:32]
        name = app_name if app_name and app_name != sig_type else sig_type

        created_at_str = str(sig.get("created_at", sig.get("at", "")))
        try:
            created_at = datetime.fromisoformat(created_at_str.replace("Z", ""))
        except Exception:
            created_at = now

        age_hours = max(0.0, (now - created_at).total_seconds() / 3600)
        bucket = int(age_hours * 4)  # 15-minute buckets
        time_buckets[bucket].append(name)

        # Collect numeric values for fingerprinting
        numeric_val = None
        payload = sig.get("payload", {})
        if isinstance(payload, dict):
            for key in ("cpu_percent", "memory_percent", "battery_percent",
                        "bytes_sent_delta", "bytes_recv_delta", "session_seconds"):
                v = payload.get(key)
                if v is not None:
                    try:
                        numeric_val = float(v)
                    except Exception:
                        pass
                    break

        if name in particle_map:
            p = particle_map[name]
            p.occurrences += 1
            p.age_hours = min(p.age_hours, age_hours)
            p.last_seen = created_at.isoformat()
            if numeric_val is not None:
                p.value_history.append(numeric_val)
        else:
            pid = hashlib.md5(name.encode()).hexdigest()[:8]
            particle_map[name] = SignalParticle(
                id=pid, name=name, signal_type=sig_type,
                x=(hash(name) % 100 - 50) * 0.5,
                y=0.0,
                z=(hash(name + "z") % 100 - 50) * 0.5,
                age_hours=age_hours,
                last_seen=created_at.isoformat(),
            )

        # Track app durations for conservation
        session_secs = float(sig.get("session_seconds", 0) or 0)
        if session_secs > 0:
            app_durations[name] += session_secs / 60

    particles = list(particle_map.values())
    if not particles:
        return ForceFieldState(
            particles=[], agent=_detect_agent_emergence([]),
            conservation=ConservationState(), total_energy=0.0,
            mean_coherence=0.0, field_age_hours=0.0, n_bonds=0,
            forces_applied={},
        )

    # --- Compute intrinsic properties with statistical fingerprinting ---
    max_occurrences = max(p.occurrences for p in particles)
    for p in particles:
        # Base mass from occurrence frequency
        p.mass = p.occurrences / max(1, max_occurrences)

        # Velocity = inverse of age (recent signals are "fast")
        p.velocity = 1.0 / (1.0 + p.age_hours)

        # Spin = periodicity detection (appears in multiple time buckets?)
        buckets_present = sum(1 for b, names in time_buckets.items() if p.name in names)
        total_buckets = max(1, len(time_buckets))
        p.spin = buckets_present / total_buckets

        # Statistical fingerprint — the particle's full physical characterization
        if p.value_history and len(p.value_history) >= 3:
            fp = compute_fingerprint(p.value_history)
            props = fingerprint_to_particle_props(fp)

            p.mass *= props["mass_amplifier"]
            p.spin = p.spin * 0.5 + props["spin"] * 0.5
            p.normality_p = fp.get("normality_p")
            p.medium = props["medium"]
            p.autocorrelation = fp.get("autocorrelation", 0.0)
            p.stationarity = props["stability"]
            p.entropy_stat = fp.get("entropy", 0.0)
            p.skewness = props["skewness"]
            p.kurtosis_stat = props["kurtosis"]

        # Energy = mass * velocity (decays via weak force later)
        p.energy = p.mass * p.velocity

    # --- Build co-occurrence and correlation ---
    for bucket, names in time_buckets.items():
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                cooccurrence_window[a].add(b)
                cooccurrence_window[b].add(a)

    # Correlation matrix from co-occurrence strength
    correlation_matrix: dict[str, dict[str, float]] = {}
    for p in particles:
        corrs: dict[str, float] = {}
        partners = cooccurrence_window.get(p.name, set())
        for partner_name in partners:
            partner = particle_map.get(partner_name)
            if partner:
                # Strength based on co-occurrence frequency
                shared_buckets = sum(
                    1 for b, names in time_buckets.items()
                    if p.name in names and partner_name in names
                )
                corr = shared_buckets / max(1, total_buckets) * 2.0
                corr = min(1.0, corr)
                corrs[partner_name] = corr
        correlation_matrix[p.name] = corrs

    # --- Build bonds (strong nuclear) ---
    n_bonds = 0
    for p in particles:
        partners = cooccurrence_window.get(p.name, set())
        for partner_name in partners:
            corr = correlation_matrix.get(p.name, {}).get(partner_name, 0.0)
            if corr >= 0.3:
                p.bonds.append(partner_name)
                p.bond_strengths[partner_name] = corr
                n_bonds += 1

    # --- Load previous field state for momentum continuity ---
    # (requires user_id passed via kwargs if available)

    # --- Force relaxation iterations ---
    force_totals: dict[str, float] = {"gravity": 0.0, "electromagnetic": 0.0, "strong_nuclear": 0.0, "weak_nuclear": 0.0}

    for iteration in range(n_iterations):
        for p in particles:
            p.fx, p.fy, p.fz = 0.0, 0.0, 0.0

            # Apply four fundamental forces
            fg = _gravity_force(p, particles)
            fe = _electromagnetic_force(p, particles, correlation_matrix)
            fs = _strong_nuclear_force(p, cooccurrence_window)
            fw = _weak_nuclear_force(p)

            p.fx += fg.fx + fe.fx + fs.fx + fw.fx
            p.fy += fg.fy + fe.fy + fs.fy + fw.fy
            p.fz += fg.fz + fe.fz + fs.fz + fw.fz

            if iteration == n_iterations - 1:
                force_totals["gravity"] += fg.magnitude
                force_totals["electromagnetic"] += fe.magnitude
                force_totals["strong_nuclear"] += fs.magnitude
                force_totals["weak_nuclear"] += fw.magnitude
                # Store individual force vectors for visualization
                p.gravity_vec = (round(fg.fx, 3), round(fg.fy, 3), round(fg.fz, 3))
                p.em_vec = (round(fe.fx, 3), round(fe.fy, 3), round(fe.fz, 3))
                p.strong_vec = (round(fs.fx, 3), round(fs.fy, 3), round(fs.fz, 3))
                p.weak_vec = (round(fw.fx, 3), round(fw.fy, 3), round(fw.fz, 3))

        # Integrate positions with momentum
        for p in particles:
            p.vx = p.vx * damping + p.fx * 0.1
            p.vy_vel = p.vy_vel * damping + p.fy * 0.1
            p.vz = p.vz * damping + p.fz * 0.1
            p.x += p.vx
            p.y += p.vy_vel
            p.z += p.vz

    # --- Normalize positions and assign layers ---
    ys = [p.y for p in particles]
    y_min, y_max = min(ys), max(ys)
    y_range = y_max - y_min if y_max > y_min else 1.0

    for p in particles:
        normalized_y = (p.y - y_min) / y_range  # 0 = bottom, 1 = top
        if normalized_y < 0.33:
            p.layer = "bedrock"
        elif normalized_y < 0.66:
            p.layer = "suspension"
        else:
            p.layer = "turbulent"

    # --- Detect agent emergence ---
    agent = _detect_agent_emergence(particles)

    # --- Conservation ---
    conservation = _compute_conservation(app_durations, window_hours)

    # --- Metrics ---
    total_energy = sum(p.energy for p in particles)
    coherences = [len(p.bonds) / max(1, len(particles)) for p in particles]
    mean_coherence = float(np.mean(coherences)) if coherences else 0.0

    oldest = max((p.age_hours for p in particles), default=0.0)

    # Round force totals
    force_totals = {k: round(v, 4) for k, v in force_totals.items()}

    return ForceFieldState(
        particles=particles,
        agent=agent,
        conservation=conservation,
        total_energy=round(total_energy, 4),
        mean_coherence=round(mean_coherence, 6),
        field_age_hours=round(oldest, 2),
        n_bonds=n_bonds,
        forces_applied=force_totals,
    )


def save_field_snapshot(state: ForceFieldState, *, user_id: int) -> None:
    """Persist the force field state for time evolution across sessions."""
    from sqlmodel import Session as DBSession
    from mycelium_app.db import engine
    from mycelium_app.models import ForceFieldSnapshot

    forces = state.forces_applied or {}
    dominant = max(forces, key=forces.get, default="") if forces else ""

    snapshot = ForceFieldSnapshot(
        user_id=user_id,
        n_particles=len(state.particles),
        n_bonds=state.n_bonds,
        total_energy=state.total_energy,
        mean_coherence=state.mean_coherence,
        agent_stage=state.agent.stage,
        agent_coherence=state.agent.coherence,
        agent_crystallized=state.agent.crystallized,
        agent_bound_particles=state.agent.bound_particles,
        attention_entropy=state.conservation.entropy,
        dominant_force=dominant,
        field_json=json.dumps({
            "particles": [
                {"name": p.name, "x": round(p.x, 2), "y": round(p.y, 2), "z": round(p.z, 2),
                 "vx": round(p.vx, 3), "vy": round(p.vy_vel, 3), "vz": round(p.vz, 3),
                 "energy": round(p.energy, 4), "mass": round(p.mass, 4)}
                for p in state.particles
            ],
        }, separators=(",", ":")),
    )

    try:
        with DBSession(engine) as session:
            session.add(snapshot)
            session.commit()
    except Exception:
        pass


def load_previous_field(*, user_id: int) -> dict[str, Any] | None:
    """Load the most recent field snapshot for time evolution."""
    from sqlmodel import Session as DBSession, select
    from mycelium_app.db import engine
    from mycelium_app.models import ForceFieldSnapshot

    try:
        with DBSession(engine) as session:
            row = session.exec(
                select(ForceFieldSnapshot)
                .where(ForceFieldSnapshot.user_id == user_id)
                .order_by(ForceFieldSnapshot.created_at.desc())
                .limit(1)
            ).first()
            if row and row.field_json:
                data = json.loads(row.field_json)
                data["_meta"] = {
                    "agent_stage": row.agent_stage,
                    "agent_coherence": row.agent_coherence,
                    "total_energy": row.total_energy,
                    "created_at": row.created_at.isoformat() if row.created_at else "",
                }
                return data
    except Exception:
        pass
    return None


def serialize_force_field(state: ForceFieldState) -> dict[str, Any]:
    """Serialize force field state to JSON-safe dict for API and 3D rendering."""
    return {
        "particles": [
            {
                "id": p.id, "name": p.name, "type": p.signal_type,
                "mass": round(p.mass, 4), "charge": round(p.charge, 4),
                "velocity": round(p.velocity, 4), "spin": round(p.spin, 4),
                "energy": round(p.energy, 4),
                "x": round(p.x, 2), "y": round(p.y, 2), "z": round(p.z, 2),
                "layer": p.layer,
                "bonds": p.bonds[:10],
                "bond_count": len(p.bonds),
                "occurrences": p.occurrences,
                "age_hours": round(p.age_hours, 2),
                "medium": p.medium,
                "stationarity": round(p.stationarity, 4),
                "autocorrelation": round(p.autocorrelation, 4),
                "entropy": round(p.entropy_stat, 4),
                "forces": {
                    "gravity": list(p.gravity_vec),
                    "electromagnetic": list(p.em_vec),
                    "strong_nuclear": list(p.strong_vec),
                    "weak_nuclear": list(p.weak_vec),
                },
                "momentum": [round(p.vx, 3), round(p.vy_vel, 3), round(p.vz, 3)],
            }
            for p in state.particles
        ],
        "agent": {
            "coherence": state.agent.coherence,
            "energy": state.agent.energy,
            "center": [state.agent.center_x, state.agent.center_y, state.agent.center_z],
            "bound_particles": state.agent.bound_particles,
            "dominant_frequency": state.agent.dominant_frequency,
            "stage": state.agent.stage,
            "crystallized": state.agent.crystallized,
        },
        "conservation": {
            "total_minutes": state.conservation.total_minutes,
            "allocated": state.conservation.allocated,
            "flow_rate": state.conservation.flow_rate,
            "entropy": state.conservation.entropy,
        },
        "metrics": {
            "total_energy": state.total_energy,
            "mean_coherence": state.mean_coherence,
            "field_age_hours": state.field_age_hours,
            "n_particles": len(state.particles),
            "n_bonds": state.n_bonds,
            "forces": state.forces_applied,
        },
        "bonds": [
            {"source": p.name, "target": b, "strength": round(p.bond_strengths.get(b, 0), 4)}
            for p in state.particles
            for b in p.bonds[:5]
        ],
    }
