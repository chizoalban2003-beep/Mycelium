"""Tests for statistical fingerprinting engine."""
import math
from mycelium_app.particle_stats import compute_fingerprint, fingerprint_to_particle_props


def test_fingerprint_normal_distribution():
    """Gaussian data should be classified as 'fluid' medium."""
    import numpy as np
    np.random.seed(42)
    values = list(np.random.normal(50, 10, 100))
    fp = compute_fingerprint(values)

    assert fp["n_observations"] == 100
    assert fp["normality_p"] is not None
    assert fp["normality_p"] > 0.05, "Normal data should have high normality p-value"
    assert fp["medium"] == "fluid"
    assert 45 < fp["mean"] < 55
    assert fp["std"] > 5
    assert fp["entropy"] > 0


def test_fingerprint_uniform_distribution():
    """Uniform data should have high entropy."""
    values = list(range(100))
    fp = compute_fingerprint(values)

    assert fp["entropy"] > 3.0, "Uniform data should have high entropy"
    assert fp["n_observations"] == 100


def test_fingerprint_constant_values():
    """Constant values should have zero variance and low entropy."""
    values = [5.0] * 50
    fp = compute_fingerprint(values)

    assert fp["std"] == 0.0 or fp["std"] < 0.001
    assert fp["entropy"] == 0.0
    assert fp["medium"] == "frozen"


def test_fingerprint_periodic_signal():
    """Periodic signal should have high autocorrelation."""
    import numpy as np
    t = np.linspace(0, 10 * math.pi, 200)
    values = list(np.sin(t))
    fp = compute_fingerprint(values)

    assert abs(fp["autocorrelation"]) > 0.5, "Periodic signal should have high autocorrelation"


def test_fingerprint_too_few_values():
    """Less than 3 values should return defaults."""
    fp = compute_fingerprint([1.0, 2.0])
    assert fp["n_observations"] == 2
    assert fp["normality_p"] is None
    assert fp["medium"] == "gaseous"


def test_fingerprint_skewed_data():
    """Heavily skewed data should show non-zero skewness."""
    import numpy as np
    np.random.seed(42)
    values = list(np.random.exponential(2, 200))
    fp = compute_fingerprint(values)

    assert fp["skewness"] > 0.5, "Exponential data should be positively skewed"


def test_fingerprint_to_props_mapping():
    """Fingerprint should map to valid particle properties."""
    fp = {
        "effect_size": 2.0,
        "autocorrelation": 0.8,
        "entropy": 3.5,
        "stationarity": 0.9,
        "medium": "fluid",
        "skewness": 0.5,
        "kurtosis": 1.0,
    }
    props = fingerprint_to_particle_props(fp)

    assert props["mass_amplifier"] > 1.0
    assert props["spin"] > 0.5
    assert props["viscosity_contribution"] > 0
    assert props["stability"] == 0.9
    assert props["ionization"] == "parametric"
    assert props["medium"] == "fluid"


def test_stationarity_stable_signal():
    """A signal fluctuating around a constant mean should have high stationarity."""
    import numpy as np
    np.random.seed(42)
    values = list(10.0 + np.random.randn(50) * 0.5)
    fp = compute_fingerprint(values)
    assert fp["stationarity"] > 0.6, f"Stable signal should have high stationarity, got {fp['stationarity']}"


def test_stationarity_drifting_signal():
    """A signal with a big shift should have lower stationarity."""
    values = [10.0] * 25 + [100.0] * 25
    fp = compute_fingerprint(values)
    assert fp["stationarity"] < 0.5
