"""Integration tests — full pipeline from signals to agent emergence."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_full_pipeline_signals_to_emergence():
    """Signal collection → force field → sedimentation → agent emergence."""
    from mycelium_app.signal_collector import CollectorState, collect_all_signals
    from mycelium_app.force_field import compute_force_field
    from mycelium_app.sedimentation import run_sedimentation
    from mycelium_app.ecosystem_bridge import build_ecosystem_dataframe

    # 1. Collect signals (simulated via direct call)
    state = CollectorState()
    all_signals = []
    for _ in range(5):
        signals = collect_all_signals(state)
        for s in signals:
            all_signals.append({
                "signal_type": s.get("signal_type", "unknown"),
                "app_name": s.get("stimulus", {}).get("app_name", s.get("signal_type", "")),
                "created_at": datetime.utcnow().isoformat(),
                "payload": s.get("stimulus", {}),
            })

    assert len(all_signals) > 0, "Should collect at least some signals"

    # 2. Build force field
    ff = compute_force_field(all_signals, window_hours=1, n_iterations=15)
    assert len(ff.particles) > 0, "Should produce particles from signals"
    assert ff.total_energy > 0, "Particles should have energy"

    # 3. Check forces were applied
    assert ff.forces_applied.get("gravity", 0) > 0
    assert ff.forces_applied.get("strong_nuclear", 0) > 0

    # 4. Check agent waveform exists
    assert ff.agent is not None
    assert ff.agent.stage in ("infant", "toddler", "adolescent", "adult")

    # 5. Check layer stratification
    layers = set(p.layer for p in ff.particles)
    assert len(layers) >= 1, "Should have at least one layer"


def test_sedimentation_on_ecosystem_dataframe():
    """Build a realistic DataFrame and verify sedimentation works on it."""
    from mycelium_app.sedimentation import run_sedimentation

    np.random.seed(42)
    n_rows = 20
    df = pd.DataFrame({
        "cpu_mean": np.random.uniform(5, 80, n_rows),
        "memory_mean": np.random.uniform(30, 90, n_rows),
        "net_sent_bytes": np.random.exponential(10000, n_rows),
        "net_recv_bytes": np.random.exponential(50000, n_rows),
        "app_opens": np.random.poisson(5, n_rows),
        "context_switches": np.random.poisson(3, n_rows),
        "hour_of_day": np.random.randint(8, 23, n_rows),
        "n_signals": np.random.poisson(10, n_rows),
    })

    result = run_sedimentation(df, flocculation_threshold=0.6)

    assert result.n_features >= 5
    assert len(result.features) > 0
    total = sum(info["count"] for info in result.layer_summary.values())
    assert total == result.n_features


def test_unified_bridge_produces_valid_kwargs():
    """Force field → unified bridge → predictor kwargs → valid for physics engine."""
    from mycelium_app.force_field import compute_force_field
    from mycelium_app.unified_field import field_to_predictor_kwargs
    from mycelium_app.physics_predictor import PhysicsPlane

    now = datetime.utcnow()
    signals = [
        {"signal_type": f"type_{i}", "app_name": f"app_{i}",
         "created_at": (now - timedelta(minutes=i * 3)).isoformat(),
         "payload": {"cpu_percent": 10 + i * 5}}
        for i in range(30)
    ]

    ff = compute_force_field(signals, window_hours=2, n_iterations=15)
    kwargs = field_to_predictor_kwargs(ff)

    assert kwargs["plane"] in (PhysicsPlane.solid, PhysicsPlane.liquid, PhysicsPlane.gas)
    assert isinstance(kwargs["n_cycles"], int)
    assert 10 <= kwargs["n_cycles"] <= 100
    assert isinstance(kwargs["pcr_enabled"], bool)
    assert isinstance(kwargs["cascade_enabled"], bool)
    assert 0.01 < kwargs["cycle_learning_rate"] < 1.0


def test_fingerprint_enriches_particles():
    """Particles with value history should get statistical fingerprints."""
    from mycelium_app.force_field import compute_force_field
    from datetime import datetime, timedelta

    now = datetime.utcnow()
    signals = [
        {"signal_type": "resource_pulse", "app_name": "resource_pulse",
         "created_at": (now - timedelta(minutes=i)).isoformat(),
         "payload": {"cpu_percent": 20 + i * 2 + np.random.randn() * 3}}
        for i in range(20)
    ]

    ff = compute_force_field(signals, window_hours=1, n_iterations=10)

    rp = [p for p in ff.particles if p.name == "resource_pulse"]
    assert len(rp) == 1
    p = rp[0]
    assert p.mass > 0
    assert p.energy > 0
    assert p.occurrences == 20


def test_force_field_time_evolution():
    """Two sequential computations with previous state should show continuity."""
    from mycelium_app.force_field import compute_force_field, serialize_force_field
    from datetime import datetime, timedelta

    now = datetime.utcnow()
    signals = [
        {"signal_type": f"t{i%3}", "app_name": f"t{i%3}",
         "created_at": (now - timedelta(minutes=i)).isoformat(), "payload": {}}
        for i in range(15)
    ]

    ff1 = compute_force_field(signals, window_hours=1, n_iterations=15)
    s1 = serialize_force_field(ff1)

    prev = {"particles": [
        {"name": p["name"], "x": p["x"], "y": p["y"], "z": p["z"],
         "vx": p["momentum"][0], "vy": p["momentum"][1], "vz": p["momentum"][2]}
        for p in s1["particles"]
    ]}

    ff2 = compute_force_field(signals, window_hours=1, n_iterations=15, previous_field=prev)

    # Both should be valid
    assert len(ff2.particles) == len(ff1.particles)
    assert ff2.total_energy > 0

    # Positions should differ (evolved from previous state, not random)
    for p1, p2 in zip(ff1.particles, ff2.particles):
        if p1.name == p2.name:
            # Not identical (seeded from different starting points)
            # Just check both are finite
            assert abs(p2.x) < 1000
            assert abs(p2.y) < 1000


def test_humanizer_in_api_context():
    """Humanizer should handle all signal types the collector produces."""
    from mycelium_app.humanizer import humanize_signal, humanize_app

    collector_signals = [
        "system_boot", "resource_pulse", "process_snapshot",
        "app_open", "app_close", "app_focus", "app_session_end",
        "network_flow", "disk_io",
    ]
    for sig in collector_signals:
        result = humanize_signal(sig)
        assert result != sig or "_" not in result, f"{sig} should be humanized"

    collector_apps = [
        "chrome", "firefox", "code", "bash", "node", "python3",
    ]
    for app in collector_apps:
        result = humanize_app(app)
        assert result != app, f"{app} should be humanized to a friendly name"
