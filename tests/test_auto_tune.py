"""Tests for force constant auto-tuning."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from mycelium_app.auto_tune import auto_tune_constants, TunedConstants


def _make_signals(n=30):
    now = datetime.utcnow()
    return [
        {"signal_type": f"type_{i%5}", "app_name": f"app_{i%5}",
         "created_at": (now - timedelta(minutes=i * 3)).isoformat(),
         "payload": {"cpu_percent": 10 + i * 2 + np.random.randn() * 3}}
        for i in range(n)
    ]


def _make_df(n_rows=30):
    np.random.seed(42)
    base = np.random.randn(n_rows)
    return pd.DataFrame({
        "cpu_mean": np.random.uniform(5, 80, n_rows),
        "memory_mean": np.random.uniform(30, 90, n_rows),
        "net_recv": np.random.exponential(10000, n_rows),
        "net_sent": np.random.exponential(5000, n_rows),
        "disk_write": np.random.exponential(50000, n_rows),
        "hour_of_day": np.random.randint(8, 23, n_rows),
        "n_signals": np.random.poisson(10, n_rows),
        "app_opens": np.random.poisson(5, n_rows),
        "context_switches": np.random.poisson(3, n_rows),
        "process_count": 50 + np.random.poisson(10, n_rows),
    })


def test_auto_tune_runs_without_error():
    df = _make_df()
    signals = _make_signals()
    tc = auto_tune_constants(df, signals, "cpu_mean")
    assert tc.generation == 1
    assert tc.G > 0
    assert tc.K_E > 0
    assert tc.K_S > 0
    assert tc.K_W > 0


def test_auto_tune_preserves_positive_constants():
    df = _make_df()
    signals = _make_signals()
    tc = TunedConstants(G=0.01, K_E=0.01, K_S=0.01, K_W=0.001)
    tc = auto_tune_constants(df, signals, "cpu_mean", tc)
    assert tc.G > 0
    assert tc.K_E > 0
    assert tc.K_S > 0
    assert tc.K_W > 0


def test_auto_tune_records_history():
    df = _make_df()
    signals = _make_signals()
    tc = auto_tune_constants(df, signals, "cpu_mean")
    assert len(tc.history) >= 1
    entry = tc.history[-1]
    assert "generation" in entry
    assert "constants" in entry
    assert "action" in entry


def test_auto_tune_multiple_generations():
    df = _make_df()
    signals = _make_signals()
    tc = TunedConstants()
    for _ in range(3):
        tc = auto_tune_constants(df, signals, "cpu_mean", tc, delta=0.1)
    assert tc.generation == 3
    assert len(tc.history) == 3


def test_auto_tune_delta_decays():
    df = _make_df()
    signals = _make_signals()
    tc = TunedConstants(generation=20)
    tc = auto_tune_constants(df, signals, "cpu_mean", tc, delta=0.1, min_delta=0.005)
    entry = tc.history[-1]
    assert entry["delta"] < 0.1, "Delta should decay with generation"


def test_auto_tune_empty_signals():
    df = _make_df()
    tc = auto_tune_constants(df, [], "cpu_mean")
    assert tc.generation >= 0
    assert tc.G > 0 and tc.K_E > 0
