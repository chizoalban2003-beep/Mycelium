"""Tests for the PhysML physics predictor core engine."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from physml import (
    PhysicsPlane,
    PredictionResult,
    PredictorRuntimeState,
    infer_feature_kind,
    infer_target_kind,
    run_physics_prediction,
    serialize_predictor_state,
    deserialize_predictor_state,
)
from physml.predictor import PredictorError


# ── Fixtures ─────────────────────────────────────────────────────────────

def _make_classification_df(n_rows: int = 80, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X1 = rng.normal(0, 1, n_rows)
    X2 = rng.normal(0, 1, n_rows)
    X3 = rng.uniform(-1, 1, n_rows)
    y_raw = (X1 + 0.5 * X2 > 0).astype(int)
    return pd.DataFrame({"x1": X1, "x2": X2, "x3": X3, "target": y_raw})


def _make_regression_df(n_rows: int = 80, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X1 = rng.normal(0, 1, n_rows)
    X2 = rng.uniform(0, 5, n_rows)
    y = 2.0 * X1 + 0.5 * X2 + rng.normal(0, 0.5, n_rows)
    return pd.DataFrame({"x1": X1, "x2": X2, "target": y})


# ── Target kind inference ────────────────────────────────────────────────

def test_infer_target_kind_numeric():
    rng = np.random.default_rng(42)
    s = pd.Series(rng.uniform(0, 100, 200))  # 200 continuous values → numeric
    assert infer_target_kind(s) == "numeric"


def test_infer_target_kind_categorical_int():
    s = pd.Series([0, 1, 2, 0, 1, 2] * 15)
    assert infer_target_kind(s) == "categorical"


def test_infer_target_kind_categorical_str():
    s = pd.Series(["cat", "dog", "cat", "bird"] * 20)
    assert infer_target_kind(s) == "categorical"


def test_infer_feature_kind_numeric():
    s = pd.Series([1.0, 2.0, 3.5])
    assert infer_feature_kind(s) == "numeric"


def test_infer_feature_kind_categorical():
    s = pd.Series(["a", "b", "c"])
    assert infer_feature_kind(s) == "categorical"


# ── run_physics_prediction — basic operation ──────────────────────────────

def test_run_physics_prediction_classification_returns_result():
    df = _make_classification_df()
    result = run_physics_prediction(
        df, target_col="target",
        plane=PhysicsPlane.liquid, n_cycles=5, train_fraction=0.75,
        return_predictions=True,
    )
    assert result is not None
    assert isinstance(result, PredictionResult)
    assert result.target_kind == "categorical"
    assert result.metrics.accuracy is not None
    assert 0.0 <= result.metrics.accuracy <= 1.0


def test_run_physics_prediction_regression_returns_result():
    df = _make_regression_df()
    result = run_physics_prediction(
        df, target_col="target",
        plane=PhysicsPlane.solid, n_cycles=5, train_fraction=0.75,
        return_predictions=True,
    )
    assert result is not None
    assert result.target_kind == "numeric"
    assert result.metrics.mae is not None
    assert result.metrics.rmse is not None
    assert math.isfinite(result.metrics.mae)
    assert math.isfinite(result.metrics.rmse)


def test_run_physics_prediction_all_planes():
    df = _make_regression_df()
    for plane in (PhysicsPlane.solid, PhysicsPlane.liquid, PhysicsPlane.gas):
        result = run_physics_prediction(
            df, target_col="target", plane=plane,
            n_cycles=3, return_predictions=True,
        )
        assert result is not None, f"Failed for plane {plane}"


def test_run_physics_prediction_missing_target_column_raises():
    df = _make_classification_df()
    with pytest.raises(PredictorError, match="not found"):
        run_physics_prediction(df, target_col="nonexistent", n_cycles=3)


def test_run_physics_prediction_too_few_rows_raises():
    df = pd.DataFrame({"x": [1.0, 2.0], "target": [0, 1]})
    with pytest.raises(PredictorError):
        run_physics_prediction(df, target_col="target", n_cycles=3)


# ── explicit_train_mask ───────────────────────────────────────────────────

def test_explicit_train_mask_produces_correct_test_rows():
    df = _make_classification_df(n_rows=60)
    n_train = 40
    mask = np.zeros(60, dtype=bool)
    mask[:n_train] = True

    result = run_physics_prediction(
        df, target_col="target",
        plane=PhysicsPlane.liquid, n_cycles=5,
        explicit_train_mask=mask, return_predictions=True,
    )
    assert result is not None
    # Test rows are indices 40..59 in the combined df
    if result.test_row_indices:
        assert all(idx >= n_train for idx in result.test_row_indices)
    assert result.metrics.n_test == 20


def test_explicit_train_mask_wrong_length_raises():
    df = _make_classification_df(n_rows=60)
    bad_mask = np.zeros(50, dtype=bool)  # wrong length
    with pytest.raises(PredictorError, match="explicit_train_mask"):
        run_physics_prediction(df, target_col="target", explicit_train_mask=bad_mask)


# ── PredictionResult content ─────────────────────────────────────────────

def test_prediction_result_has_weights():
    df = _make_regression_df()
    result = run_physics_prediction(df, target_col="target", n_cycles=5)
    assert len(result.weights) > 0
    assert all(math.isfinite(w.weight) for w in result.weights)


def test_prediction_result_has_migration_map():
    df = _make_regression_df()
    result = run_physics_prediction(df, target_col="target", n_cycles=5)
    assert len(result.migration_map) > 0
    for m in result.migration_map:
        assert math.isfinite(m.viscosity)
        assert math.isfinite(m.terminal_velocity)


def test_prediction_result_test_predictions_returned():
    df = _make_classification_df()
    result = run_physics_prediction(
        df, target_col="target", n_cycles=5,
        return_predictions=True,
    )
    assert result.test_predicted is not None
    assert result.test_actual is not None
    assert len(result.test_predicted) == len(result.test_actual)


# ── Runtime state serialization ───────────────────────────────────────────

def test_runtime_state_serialize_deserialize_roundtrip():
    state = PredictorRuntimeState(
        cycle_index=7,
        adaptive_gain=0.9,
        homeostasis_score=0.6,
        preferred_plane=PhysicsPlane.solid,
        metadata={"source": "test"},
    )
    payload = serialize_predictor_state(state)
    restored = deserialize_predictor_state(payload)
    assert restored.cycle_index == 7
    assert abs(restored.adaptive_gain - 0.9) < 1e-9
    assert restored.preferred_plane == PhysicsPlane.solid


def test_runtime_state_updated_after_prediction():
    df = _make_regression_df()
    runtime = PredictorRuntimeState()
    assert runtime.cycle_index == 0

    run_physics_prediction(
        df, target_col="target", n_cycles=5,
        runtime_state=runtime,
    )
    # After one run the predictor should have incremented the cycle counter
    # (update_predictor_state_from_result bumps it).
    assert runtime.cycle_index >= 1


def test_clean_tabular_dataframe_removes_duplicates():
    from physml import clean_tabular_dataframe
    df = pd.DataFrame({
        "x": [1.0, 2.0, 1.0, 3.0],
        "y": [10.0, 20.0, 10.0, 30.0],
    })
    cleaned, _ = clean_tabular_dataframe(df, target_col="y")
    assert cleaned.shape[0] == 3  # one duplicate removed


def test_clean_tabular_dataframe_imputes_missing():
    from physml import clean_tabular_dataframe
    df = pd.DataFrame({
        "x": [1.0, float("nan"), 3.0, 4.0],
        "y": [10.0, 20.0, 30.0, 40.0],
    })
    cleaned, _ = clean_tabular_dataframe(df, target_col="y")
    assert cleaned["x"].isna().sum() == 0
