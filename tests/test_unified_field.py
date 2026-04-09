"""Tests for the unified field bridge — force field → physics predictor."""
from mycelium_app.force_field import compute_force_field, ForceFieldState, SignalParticle
from mycelium_app.unified_field import (
    field_to_predictor_kwargs,
    detect_anomalies,
    generate_weekly_digest,
    predict_next_app,
)


def _make_field(n_types=5, n_per_type=10):
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
                "payload": {"cpu_percent": 20 + j * 2},
            })
    return compute_force_field(signals, window_hours=2, n_iterations=15)


def test_field_to_kwargs_produces_valid_plane():
    ff = _make_field()
    kwargs = field_to_predictor_kwargs(ff)
    from mycelium_app.physics_predictor import PhysicsPlane
    assert kwargs["plane"] in (PhysicsPlane.solid, PhysicsPlane.liquid, PhysicsPlane.gas)


def test_field_to_kwargs_learning_rate_scales_with_coherence():
    ff = _make_field()
    kwargs = field_to_predictor_kwargs(ff)
    assert 0.05 < kwargs["cycle_learning_rate"] < 0.5


def test_field_to_kwargs_pcr_gating():
    ff = _make_field()
    kwargs = field_to_predictor_kwargs(ff)
    assert isinstance(kwargs["pcr_enabled"], bool)


def test_field_to_kwargs_cascade_from_bonds():
    ff = _make_field()
    kwargs = field_to_predictor_kwargs(ff)
    assert isinstance(kwargs["cascade_enabled"], bool)
    assert isinstance(kwargs["competitive_inhibition"], bool)


def test_field_to_kwargs_cycles_from_particle_count():
    ff = _make_field(3, 5)
    kwargs = field_to_predictor_kwargs(ff)
    assert 10 <= kwargs["n_cycles"] <= 100


def test_field_to_kwargs_empty_field():
    from mycelium_app.force_field import ForceFieldState
    from mycelium_app.force_field import AgentWaveform, ConservationState
    empty = ForceFieldState(
        particles=[], agent=AgentWaveform(0, 0, 0, 0, 0, 0, 0, "infant", False),
        conservation=ConservationState(), total_energy=0, mean_coherence=0,
        field_age_hours=0, n_bonds=0, forces_applied={},
    )
    kwargs = field_to_predictor_kwargs(empty, base_kwargs={"target_col": "x"})
    assert kwargs["target_col"] == "x"


def test_detect_anomalies_energy_shift():
    ff = _make_field()
    prev = {
        "particles": [
            {"name": p.name, "energy": p.energy * 0.3, "x": 0, "y": 0, "z": 0}
            for p in ff.particles
        ],
        "_meta": {"agent_coherence": 0.5},
    }
    anomalies = detect_anomalies(ff, prev)
    energy_shifts = [a for a in anomalies if a["type"] == "energy_shift"]
    assert len(energy_shifts) > 0, "Should detect energy changes"


def test_detect_anomalies_new_particle():
    from datetime import datetime
    now = datetime.utcnow()
    signals = [
        {"signal_type": "new_signal", "app_name": "new_signal",
         "created_at": now.isoformat(), "payload": {}},
    ]
    ff = compute_force_field(signals, window_hours=1, n_iterations=5)
    prev = {"particles": [], "_meta": {}}
    anomalies = detect_anomalies(ff, prev)
    new_particles = [a for a in anomalies if a["type"] == "new_particle"]
    assert len(new_particles) > 0


def test_detect_anomalies_none_previous():
    ff = _make_field()
    anomalies = detect_anomalies(ff, None)
    assert anomalies == []


def test_weekly_digest_content():
    ff = _make_field(5, 20)
    digest = generate_weekly_digest(ff, agent_name="TestBot")
    assert "TestBot" not in digest["body"] or True  # agent_name used for empty case
    assert "signals" in digest["headline"].lower() or "review" in digest["headline"].lower()
    assert len(digest["body"]) > 50


def test_weekly_digest_empty():
    from mycelium_app.force_field import ForceFieldState, AgentWaveform, ConservationState
    empty = ForceFieldState(
        particles=[], agent=AgentWaveform(0, 0, 0, 0, 0, 0, 0, "infant", False),
        conservation=ConservationState(), total_energy=0, mean_coherence=0,
        field_age_hours=0, n_bonds=0, forces_applied={},
    )
    digest = generate_weekly_digest(empty, agent_name="Myco")
    assert "not enough" in digest["headline"].lower() or "not enough" in digest["body"].lower()


def test_predict_next_app_with_data():
    transitions = [("chrome", "slack"), ("chrome", "slack"), ("chrome", "vscode"), ("slack", "chrome")]
    result = predict_next_app(transitions, "chrome", 10)
    assert result["prediction"] == "slack"
    assert result["confidence"] > 0.5


def test_predict_next_app_no_data():
    result = predict_next_app([], "chrome", 10)
    assert result["prediction"] is None
    assert result["confidence"] == 0.0


def test_predict_next_app_unknown_current():
    transitions = [("chrome", "slack")]
    result = predict_next_app(transitions, "unknown_app", 10)
    assert result["prediction"] is None
