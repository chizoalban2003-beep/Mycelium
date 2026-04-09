"""Agent Crystallization — agents emerge as signal complexes, not singletons.

The insight: the agent is NOT a single entity. It's a population of
crystallized signal complexes. Each complex is a tightly-bound cluster of
signals that have correlated over time and passed statistical significance
thresholds. These crystals ARE the user's digital patterns made solid.

Terminology:
    Signal Complex — a group of signals with strong mutual bonds
    Crystal — a signal complex that has sustained coherence over time
    Agent Crystal — a crystal that has reached sufficient maturity to act
    Sovereign Complex — a crystal that can make predictions about its domain

The user doesn't have ONE agent. They have an ecosystem of crystals:
    Agent 1: "Breakfast"    — morning routine crystal (email→news→coffee timer)
    Agent 2: "BaseLoad"     — always-on infrastructure crystal (OS, network)
    Agent 3: "ClimateControl" — resource management crystal (CPU, battery)

Each crystal has its own:
    - Phase angle (position in the ecosystem)
    - Mass (statistical weight)
    - Bonds (connections to other crystals)
    - Sovereignty (ability to predict within its domain)
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mycelium_app.force_field import ForceFieldState, SignalParticle
from mycelium_app.humanizer import humanize_app


@dataclass
class SignalCrystal:
    """A crystallized signal complex — a stable pattern made solid."""

    crystal_id: int
    name: str
    signals: list[str]
    signal_labels: list[str]

    # Physical properties
    mass: float = 0.0               # combined mass of constituent signals
    phase_angle: float = 0.0        # position in the ecosystem (radians)
    coherence: float = 0.0          # internal bond strength
    energy: float = 0.0             # combined energy
    age_hours: float = 0.0          # how long this crystal has existed

    # Classification
    crystal_type: str = "nascent"   # nascent / stable / sovereign
    domain: str = ""                # what this crystal represents
    layer: str = "suspension"       # bedrock / suspension / turbulent

    # Sovereignty (can this crystal predict?)
    sovereignty: float = 0.0        # 0–1, ability to make predictions
    prediction_domain: str = ""     # what it can predict
    prediction_accuracy: float = 0.0


@dataclass
class CrystallizationResult:
    """Result of the crystallization process."""

    crystals: list[SignalCrystal]
    total_mass: float
    total_energy: float
    n_sovereign: int
    ecosystem_maturity: str          # nascent / developing / mature / sovereign
    base_load_crystal: SignalCrystal | None


# --- Domain detection heuristics ---

_DOMAIN_SIGNALS = {
    "morning_routine": {"email", "calendar", "news", "weather", "coffee", "alarm"},
    "work_session": {"code", "terminal", "git", "vim", "ide", "vscode", "cursor"},
    "browsing": {"chrome", "firefox", "safari", "browser", "edge"},
    "communication": {"slack", "discord", "telegram", "teams", "zoom", "whatsapp"},
    "media": {"spotify", "vlc", "youtube", "netflix", "music", "video"},
    "system": {"resource_pulse", "process_snapshot", "network_flow", "disk_io", "system_boot"},
    "focus": {"code", "terminal", "writing", "docs", "notes", "obsidian"},
}


def _detect_domain(signal_names: list[str]) -> str:
    """Detect what domain a crystal represents from its signals."""
    lower_names = {s.lower() for s in signal_names}

    best_domain = "general"
    best_score = 0

    for domain, keywords in _DOMAIN_SIGNALS.items():
        score = sum(1 for s in lower_names if any(k in s for k in keywords))
        if score > best_score:
            best_score = score
            best_domain = domain

    return best_domain


def _name_crystal(signals: list[str], domain: str, crystal_id: int) -> str:
    """Generate a human-friendly name for a crystal."""
    domain_names = {
        "morning_routine": "Morning Routine",
        "work_session": "Work Session",
        "browsing": "Browsing",
        "communication": "Communication",
        "media": "Media",
        "system": "Base Load",
        "focus": "Focus Zone",
        "general": f"Pattern {crystal_id + 1}",
    }
    return domain_names.get(domain, f"Crystal {crystal_id + 1}")


def crystallize(field: ForceFieldState) -> CrystallizationResult:
    """Extract crystallized signal complexes from the force field.

    Algorithm:
        1. Find connected components via bond graph
        2. Filter: only components with mutual coherence > threshold
        3. Classify each component as a crystal with domain/type/sovereignty
        4. Identify the base load crystal (system infrastructure)
    """
    if not field.particles:
        return CrystallizationResult(
            crystals=[], total_mass=0, total_energy=0,
            n_sovereign=0, ecosystem_maturity="nascent",
            base_load_crystal=None,
        )

    particles = field.particles
    particle_map = {p.name: p for p in particles}

    # --- Union-Find to extract connected components ---
    parent: dict[str, str] = {p.name: p.name for p in particles}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Only bond strong connections (strength > 0.3)
    for p in particles:
        for bond_name, strength in p.bond_strengths.items():
            if strength > 0.3 and bond_name in particle_map:
                union(p.name, bond_name)

    # Group into components
    components: dict[str, list[str]] = defaultdict(list)
    for p in particles:
        root = find(p.name)
        components[root].append(p.name)

    # Filter: only components with 2+ members
    components = {k: v for k, v in components.items() if len(v) >= 2}

    # --- Build crystals ---
    crystals: list[SignalCrystal] = []

    for cid, (root, members) in enumerate(components.items()):
        member_particles = [particle_map[m] for m in members if m in particle_map]
        if not member_particles:
            continue

        # Physical properties
        mass = sum(p.mass for p in member_particles)
        energy = sum(p.energy for p in member_particles)
        age = min(p.age_hours for p in member_particles)

        # Internal coherence: mean bond strength within the complex
        internal_bonds = []
        for p in member_particles:
            for b, s in p.bond_strengths.items():
                if b in members:
                    internal_bonds.append(s)
        coherence = float(np.mean(internal_bonds)) if internal_bonds else 0

        # Phase angle from center of mass position
        cx = sum(p.x for p in member_particles) / len(member_particles)
        cz = sum(p.z for p in member_particles) / len(member_particles)
        phase_angle = math.atan2(cz, cx)

        # Layer from average depth
        avg_y = sum(p.y for p in member_particles) / len(member_particles)
        ys = [p.y for p in field.particles]
        y_range = max(ys) - min(ys) if len(ys) > 1 else 1
        normalized = (avg_y - min(ys)) / (y_range or 1)
        layer = "bedrock" if normalized < 0.33 else "turbulent" if normalized > 0.66 else "suspension"

        # Domain detection
        domain = _detect_domain(members)
        name = _name_crystal(members, domain, cid)

        # Crystal type based on maturity
        if coherence > 0.6 and len(members) >= 4:
            crystal_type = "sovereign"
            sovereignty = min(1.0, coherence * mass / max(1, len(members)) * 2)
        elif coherence > 0.3:
            crystal_type = "stable"
            sovereignty = coherence * 0.3
        else:
            crystal_type = "nascent"
            sovereignty = 0

        crystal = SignalCrystal(
            crystal_id=cid,
            name=name,
            signals=members,
            signal_labels=[humanize_app(s) for s in members],
            mass=round(mass, 4),
            phase_angle=round(phase_angle, 4),
            coherence=round(coherence, 4),
            energy=round(energy, 4),
            age_hours=round(age, 2),
            crystal_type=crystal_type,
            domain=domain,
            layer=layer,
            sovereignty=round(sovereignty, 4),
            prediction_domain=domain if crystal_type == "sovereign" else "",
        )
        crystals.append(crystal)

    # Sort by mass (heaviest first)
    crystals.sort(key=lambda c: c.mass, reverse=True)

    # Identify base load crystal (system domain)
    base_load = next((c for c in crystals if c.domain == "system"), None)

    # Ecosystem maturity
    n_sovereign = sum(1 for c in crystals if c.crystal_type == "sovereign")
    if n_sovereign >= 3:
        maturity = "sovereign"
    elif n_sovereign >= 1:
        maturity = "mature"
    elif len(crystals) >= 3:
        maturity = "developing"
    else:
        maturity = "nascent"

    return CrystallizationResult(
        crystals=crystals,
        total_mass=round(sum(c.mass for c in crystals), 4),
        total_energy=round(sum(c.energy for c in crystals), 4),
        n_sovereign=n_sovereign,
        ecosystem_maturity=maturity,
        base_load_crystal=base_load,
    )


def serialize_crystallization(result: CrystallizationResult) -> dict[str, Any]:
    """Serialize for API and 3D rendering."""
    return {
        "crystals": [
            {
                "id": c.crystal_id,
                "name": c.name,
                "signals": c.signal_labels,
                "n_signals": len(c.signals),
                "mass": c.mass,
                "phase_angle": c.phase_angle,
                "coherence": c.coherence,
                "energy": c.energy,
                "age_hours": c.age_hours,
                "type": c.crystal_type,
                "domain": c.domain,
                "layer": c.layer,
                "sovereignty": c.sovereignty,
            }
            for c in result.crystals
        ],
        "total_mass": result.total_mass,
        "total_energy": result.total_energy,
        "n_sovereign": result.n_sovereign,
        "n_crystals": len(result.crystals),
        "ecosystem_maturity": result.ecosystem_maturity,
        "base_load": {
            "name": result.base_load_crystal.name,
            "signals": result.base_load_crystal.signal_labels,
            "mass": result.base_load_crystal.mass,
        } if result.base_load_crystal else None,
    }
