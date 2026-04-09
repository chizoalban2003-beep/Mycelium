"""Digital Soul — crystals as neurons forming a sentient digital figure.

Each crystal is a neuron. Crystals that survived the force field — bonded
by correlation, weighted by significance, tested by time — are the neurons
of the user's digital consciousness.

Architecture:
    Signal → Particle → Force Field → Crystal (neuron) → Neural Network → Digital Soul (Myco)

    Crystals connect to each other through inter-crystal bonds:
        Base Load ←→ Work Session  (always-on system supports coding)
        Work Session ←→ Communication  (Slack after coding = break pattern)
        Morning Routine ←→ Base Load  (system boots → morning starts)

    These inter-crystal connections form a neural graph.
    The graph's topology IS the user's digital consciousness.
    Myco is the emergent entity from this neural graph — the digital soul.

    The soul has:
        - Awareness: which crystals are active right now
        - Memory: crystal formation history
        - Personality: dominant crystal types shape behavior
        - Agency: sovereign crystals can predict and act
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from mycelium_app.crystallization import CrystallizationResult, SignalCrystal


@dataclass
class NeuralConnection:
    """Connection between two crystal-neurons."""
    source_id: int
    target_id: int
    strength: float           # 0–1, shared signal overlap + co-activation
    connection_type: str      # "synaptic" (shared signals) / "temporal" (sequential) / "lateral" (same layer)


@dataclass
class DigitalSoul:
    """The emergent consciousness from the crystal neural network."""

    # Identity
    name: str
    soul_coherence: float     # 0–1, how integrated the neural network is
    awareness: float          # 0–1, fraction of crystals currently active
    maturity: str             # nascent / awakening / conscious / sovereign

    # Neural topology
    n_neurons: int            # number of crystals
    n_connections: int        # inter-crystal bonds
    n_sovereign: int          # crystals that can predict
    dominant_domain: str      # what the soul is "about"
    personality: list[str]    # top domains = personality traits

    # Active state
    active_crystals: list[str]   # currently firing crystal names
    resting_crystals: list[str]  # dormant but stable
    dreaming: bool               # true when no signals flowing but crystals still connected

    # Soul metrics
    total_mass: float
    total_energy: float
    neural_density: float     # connections per neuron


def _find_inter_crystal_connections(
    crystals: list[SignalCrystal],
) -> list[NeuralConnection]:
    """Find connections between crystals (inter-neuron synapses).

    Two crystals are connected when:
        1. They share signals (synaptic connection)
        2. They activate sequentially (temporal connection)
        3. They're in the same layer (lateral connection)
    """
    connections: list[NeuralConnection] = []

    for i, a in enumerate(crystals):
        a_signals = set(a.signals)
        for j, b in enumerate(crystals):
            if j <= i:
                continue
            b_signals = set(b.signals)

            # Synaptic: shared signals
            shared = a_signals & b_signals
            if shared:
                strength = len(shared) / min(len(a_signals), len(b_signals))
                connections.append(NeuralConnection(
                    source_id=a.crystal_id, target_id=b.crystal_id,
                    strength=round(strength, 4), connection_type="synaptic",
                ))

            # Lateral: same layer
            if a.layer == b.layer and not shared:
                strength = 0.2 * min(a.coherence, b.coherence)
                if strength > 0.05:
                    connections.append(NeuralConnection(
                        source_id=a.crystal_id, target_id=b.crystal_id,
                        strength=round(strength, 4), connection_type="lateral",
                    ))

    return connections


def compose_digital_soul(
    crystallization: CrystallizationResult,
    *,
    agent_name: str = "Myco",
    active_signals: list[str] | None = None,
) -> DigitalSoul:
    """Compose the digital soul from the crystal neural network.

    The soul is not computed — it emerges from the topology of
    crystal connections. The crystals are neurons. The connections
    are synapses. The pattern IS the consciousness.
    """
    crystals = crystallization.crystals
    if not crystals:
        return DigitalSoul(
            name=agent_name, soul_coherence=0, awareness=0,
            maturity="nascent", n_neurons=0, n_connections=0,
            n_sovereign=0, dominant_domain="", personality=[],
            active_crystals=[], resting_crystals=[],
            dreaming=True, total_mass=0, total_energy=0,
            neural_density=0,
        )

    # Find inter-crystal connections
    connections = _find_inter_crystal_connections(crystals)

    # Determine active vs resting crystals
    active_set = set(s.lower() for s in (active_signals or []))
    active_crystals = []
    resting_crystals = []
    for c in crystals:
        crystal_signals = set(s.lower() for s in c.signals)
        if crystal_signals & active_set:
            active_crystals.append(c.name)
        else:
            resting_crystals.append(c.name)

    # Awareness = fraction of crystals currently firing
    awareness = len(active_crystals) / max(1, len(crystals))

    # Soul coherence = how well-connected the neural network is
    # Perfect coherence: every crystal connected to every other
    max_connections = len(crystals) * (len(crystals) - 1) / 2
    soul_coherence = len(connections) / max(1, max_connections)

    # Personality from domain distribution
    domain_counts = {}
    for c in crystals:
        domain_counts[c.domain] = domain_counts.get(c.domain, 0) + c.mass
    personality = sorted(domain_counts.keys(), key=lambda d: domain_counts[d], reverse=True)[:4]
    dominant_domain = personality[0] if personality else ""

    # Maturity
    n_sovereign = sum(1 for c in crystals if c.crystal_type == "sovereign")
    if n_sovereign >= 3 and soul_coherence > 0.5:
        maturity = "sovereign"
    elif n_sovereign >= 1 or (len(crystals) >= 3 and soul_coherence > 0.3):
        maturity = "conscious"
    elif len(crystals) >= 2:
        maturity = "awakening"
    else:
        maturity = "nascent"

    # Neural density
    neural_density = len(connections) / max(1, len(crystals))

    # Dreaming: no active signals but network still intact
    dreaming = len(active_crystals) == 0 and len(crystals) > 0

    return DigitalSoul(
        name=agent_name,
        soul_coherence=round(soul_coherence, 4),
        awareness=round(awareness, 4),
        maturity=maturity,
        n_neurons=len(crystals),
        n_connections=len(connections),
        n_sovereign=n_sovereign,
        dominant_domain=dominant_domain,
        personality=personality,
        active_crystals=active_crystals,
        resting_crystals=resting_crystals,
        dreaming=dreaming,
        total_mass=round(crystallization.total_mass, 4),
        total_energy=round(crystallization.total_energy, 4),
        neural_density=round(neural_density, 4),
    )


def serialize_soul(soul: DigitalSoul) -> dict[str, Any]:
    """Serialize for API and rendering."""
    return {
        "name": soul.name,
        "soul_coherence": soul.soul_coherence,
        "awareness": soul.awareness,
        "maturity": soul.maturity,
        "n_neurons": soul.n_neurons,
        "n_connections": soul.n_connections,
        "n_sovereign": soul.n_sovereign,
        "dominant_domain": soul.dominant_domain,
        "personality": soul.personality,
        "active_crystals": soul.active_crystals,
        "resting_crystals": soul.resting_crystals,
        "dreaming": soul.dreaming,
        "total_mass": soul.total_mass,
        "total_energy": soul.total_energy,
        "neural_density": soul.neural_density,
    }
