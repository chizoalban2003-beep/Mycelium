"""Tests for the unified force field engine."""
from mycelium_app.force_field import compute_force_field, serialize_force_field


def _make_signals(n_types=5, n_per_type=10):
    """Generate synthetic signal data for testing."""
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    signals = []
    types = ["resource_pulse", "process_snapshot", "network_flow", "app_focus", "disk_io"]
    for i in range(min(n_types, len(types))):
        for j in range(n_per_type):
            signals.append({
                "signal_type": types[i],
                "app_name": types[i],
                "created_at": (now - timedelta(minutes=j * 5)).isoformat(),
                "payload": {"cpu_percent": 20 + j * 2, "bytes_sent_delta": 1000 * j},
            })
    return signals


def test_force_field_produces_particles():
    """Force field should create particles from signals."""
    signals = _make_signals(3, 5)
    ff = compute_force_field(signals, window_hours=1, n_iterations=10)

    assert len(ff.particles) >= 3
    assert ff.total_energy > 0
    assert ff.n_bonds >= 0


def test_force_field_stratification():
    """Heavy particles should settle lower than light ones."""
    signals = _make_signals(5, 20)
    ff = compute_force_field(signals, window_hours=2, n_iterations=20)

    heavy = [p for p in ff.particles if p.mass > 0.5]
    light = [p for p in ff.particles if p.mass <= 0.5]

    if heavy and light:
        heavy_avg_y = sum(p.y for p in heavy) / len(heavy)
        light_avg_y = sum(p.y for p in light) / len(light)
        assert heavy_avg_y < light_avg_y, "Heavy particles should settle lower"


def test_force_field_bonds_from_cooccurrence():
    """Signals in the same time bucket should form bonds."""
    from datetime import datetime
    now = datetime.utcnow()
    signals = [
        {"signal_type": "app_a", "app_name": "app_a", "created_at": now.isoformat(), "payload": {}},
        {"signal_type": "app_b", "app_name": "app_b", "created_at": now.isoformat(), "payload": {}},
        {"signal_type": "app_a", "app_name": "app_a", "created_at": now.isoformat(), "payload": {}},
        {"signal_type": "app_b", "app_name": "app_b", "created_at": now.isoformat(), "payload": {}},
    ]
    ff = compute_force_field(signals, window_hours=1, n_iterations=5)
    assert ff.n_bonds > 0, "Co-occurring signals should form bonds"


def test_agent_emergence_with_enough_signals():
    """Agent should crystallize when enough bound particles exist."""
    signals = _make_signals(5, 30)
    ff = compute_force_field(signals, window_hours=6, n_iterations=25)

    assert ff.agent is not None
    assert ff.agent.stage in ("infant", "toddler", "adolescent", "adult")
    if ff.agent.bound_particles >= 5:
        assert ff.agent.coherence > 0


def test_agent_does_not_emerge_from_noise():
    """Single signals with no co-occurrence should not form an agent."""
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    signals = [
        {"signal_type": f"unique_{i}", "app_name": f"unique_{i}",
         "created_at": (now - timedelta(hours=i)).isoformat(), "payload": {}}
        for i in range(3)
    ]
    ff = compute_force_field(signals, window_hours=24, n_iterations=10)

    assert ff.agent.crystallized is False or ff.agent.bound_particles < 5


def test_four_forces_computed():
    """All four force types should have non-negative magnitudes."""
    signals = _make_signals(4, 15)
    ff = compute_force_field(signals, window_hours=2, n_iterations=15)

    forces = ff.forces_applied
    assert "gravity" in forces
    assert "electromagnetic" in forces
    assert "strong_nuclear" in forces
    assert "weak_nuclear" in forces
    assert all(v >= 0 for v in forces.values())


def test_serialization_round_trip():
    """Serialized force field should be JSON-safe and contain all sections."""
    signals = _make_signals(3, 10)
    ff = compute_force_field(signals, window_hours=1, n_iterations=10)
    data = serialize_force_field(ff)

    assert "particles" in data
    assert "agent" in data
    assert "conservation" in data
    assert "metrics" in data
    assert "bonds" in data

    # Check particle has force vectors and momentum
    if data["particles"]:
        p = data["particles"][0]
        assert "forces" in p
        assert "momentum" in p
        assert "medium" in p
        assert "gravity" in p["forces"]


def test_force_field_with_previous_state():
    """Previous field positions should seed the new computation."""
    signals = _make_signals(3, 10)

    ff1 = compute_force_field(signals, window_hours=1, n_iterations=15)
    serialized = serialize_force_field(ff1)

    # Use serialized as previous state
    prev = {"particles": [
        {"name": p["name"], "x": p["x"], "y": p["y"], "z": p["z"],
         "vx": p["momentum"][0], "vy": p["momentum"][1], "vz": p["momentum"][2]}
        for p in serialized["particles"]
    ]}

    ff2 = compute_force_field(signals, window_hours=1, n_iterations=15, previous_field=prev)

    # Both should produce valid fields
    assert len(ff2.particles) == len(ff1.particles)
    assert ff2.total_energy > 0


def test_empty_signals():
    """Empty signal list should return empty field state."""
    ff = compute_force_field([], window_hours=1)
    assert len(ff.particles) == 0
    assert ff.total_energy == 0
    assert ff.agent.crystallized is False
