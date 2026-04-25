"""Tests for the NeuralPhysicsEngine and PhysicsPredictor(backend="neural")."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from physml import NeuralPhysicsEngine, PhysicsPredictor, run_neural_prediction
from physml.neural_engine import _FeatureAttentionBlock, _encode_dataframe


# ── Helpers ────────────────────────────────────────────────────────────────

def _clf_data(seed: int = 42, n: int = 120):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 5))
    y = ((X[:, 0] + 0.5 * X[:, 1]) > 0).astype(int)
    return X, y


def _reg_data(seed: int = 42, n: int = 120):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 4))
    y = 3.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, 0.2, n)
    return X, y


def _split(X, y, test_frac: float = 0.25, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_te = max(1, int(n * test_frac))
    te, tr = idx[:n_te], idx[n_te:]
    return X[tr], X[te], y[tr], y[te]


def _make_combined_df(X_train, X_test, y_train, n_features):
    """Build the combined DataFrame the estimator passes to _engine_predict."""
    cols = [f"f{i}" for i in range(n_features)]
    df_tr = pd.DataFrame(X_train, columns=cols)
    df_tr["__target__"] = y_train
    df_te = pd.DataFrame(X_test, columns=cols)
    df_te["__target__"] = 0.0
    combined = pd.concat([df_tr, df_te], axis=0, ignore_index=True)
    mask = np.zeros(len(combined), dtype=bool)
    mask[: len(df_tr)] = True
    return combined, mask


# ── _FeatureAttentionBlock ─────────────────────────────────────────────────

def test_attention_fit_produces_matrix():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (80, 6))
    attn = _FeatureAttentionBlock()
    attn.fit(X)
    assert attn.attn_matrix_ is not None
    assert attn.attn_matrix_.shape == (6, 6)


def test_attention_matrix_rows_sum_to_one():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (60, 4))
    attn = _FeatureAttentionBlock()
    attn.fit(X)
    row_sums = attn.attn_matrix_.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-6)


def test_attention_transform_shape():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (50, 5))
    attn = _FeatureAttentionBlock(max_attend_features=5)
    attn.fit(X)
    out = attn.transform(X)
    assert out.shape == (50, 5)


def test_attention_feature_importance_shape():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (40, 8))
    attn = _FeatureAttentionBlock(max_attend_features=8)
    attn.fit(X)
    assert attn.feature_importance_ is not None
    assert attn.feature_importance_.shape == (8,)


def test_attention_max_attend_clip():
    """Features beyond max_attend_features should have zero importance."""
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (40, 20))
    attn = _FeatureAttentionBlock(max_attend_features=5)
    attn.fit(X)
    assert attn.feature_importance_ is not None
    # Features beyond cap should be zero
    assert np.all(attn.feature_importance_[5:] == 0.0)


# ── _encode_dataframe ──────────────────────────────────────────────────────

def test_encode_numeric_df():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "a": rng.normal(0, 1, 50),
        "b": rng.normal(0, 1, 50),
        "__target__": rng.normal(0, 1, 50),
    })
    X, y, names, kinds, label_enc = _encode_dataframe(df, "__target__")
    assert X.shape == (50, 2)
    assert y.shape == (50,)
    assert label_enc is None
    assert "a" in names and "b" in names


def test_encode_categorical_target():
    rng = np.random.default_rng(6)
    n = 60
    df = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "__target__": rng.choice(["cat", "dog", "bird"], n),
    })
    X, y, names, kinds, label_enc = _encode_dataframe(df, "__target__")
    assert label_enc is not None
    assert set(np.unique(y)) <= {0, 1, 2}


def test_encode_categorical_feature_onehotencoded():
    rng = np.random.default_rng(7)
    n = 60
    df = pd.DataFrame({
        "num": rng.normal(0, 1, n),
        "cat": rng.choice(["a", "b", "c"], n),
        "__target__": rng.normal(0, 1, n),
    })
    X, y, names, kinds, label_enc = _encode_dataframe(df, "__target__")
    # "cat" should be one-hot expanded → > 2 columns total
    assert X.shape[1] > 2
    assert X.shape[0] == n


# ── NeuralPhysicsEngine.run ────────────────────────────────────────────────

def test_neural_engine_run_returns_result_regression():
    X, y = _reg_data(n=80)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    combined, mask = _make_combined_df(X_tr, X_te, y_tr, X.shape[1])
    engine = NeuralPhysicsEngine()
    result = engine.run(
        combined,
        target_col="__target__",
        explicit_train_mask=mask,
        random_seed=42,
        n_cycles=10,
    )
    assert result is not None


def test_neural_engine_run_returns_result_classification():
    X, y = _clf_data(n=80)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    combined, mask = _make_combined_df(X_tr, X_te, y_tr, X.shape[1])
    engine = NeuralPhysicsEngine()
    result = engine.run(
        combined,
        target_col="__target__",
        explicit_train_mask=mask,
        random_seed=42,
        n_cycles=10,
    )
    assert result is not None


def test_neural_engine_test_predicted_correct_length():
    X, y = _reg_data(n=80)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    combined, mask = _make_combined_df(X_tr, X_te, y_tr, X.shape[1])
    result = NeuralPhysicsEngine().run(
        combined,
        target_col="__target__",
        explicit_train_mask=mask,
        n_cycles=10,
    )
    assert result is not None
    assert result.test_predicted is not None
    assert len(result.test_predicted) == int((~mask).sum())


def test_neural_engine_predictions_finite():
    X, y = _reg_data(n=80)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    combined, mask = _make_combined_df(X_tr, X_te, y_tr, X.shape[1])
    result = NeuralPhysicsEngine().run(
        combined,
        target_col="__target__",
        explicit_train_mask=mask,
        n_cycles=10,
    )
    assert result is not None
    assert result.test_predicted is not None
    assert all(math.isfinite(float(v)) for v in result.test_predicted)


def test_neural_engine_weights_populated():
    X, y = _reg_data(n=80)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    combined, mask = _make_combined_df(X_tr, X_te, y_tr, X.shape[1])
    result = NeuralPhysicsEngine().run(
        combined, target_col="__target__", explicit_train_mask=mask, n_cycles=10
    )
    assert result is not None
    assert len(result.weights) > 0
    assert all(w.method == "neural_attention" for w in result.weights)


def test_neural_engine_migration_map_populated():
    X, y = _reg_data(n=80)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    combined, mask = _make_combined_df(X_tr, X_te, y_tr, X.shape[1])
    result = NeuralPhysicsEngine().run(
        combined, target_col="__target__", explicit_train_mask=mask, n_cycles=10
    )
    assert result is not None
    assert len(result.migration_map) > 0


def test_neural_engine_metrics_target_kind():
    X, y = _reg_data(n=80)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    combined, mask = _make_combined_df(X_tr, X_te, y_tr, X.shape[1])
    result = NeuralPhysicsEngine().run(
        combined, target_col="__target__", explicit_train_mask=mask, n_cycles=10
    )
    assert result is not None
    assert result.metrics.target_kind == "numeric"
    assert result.metrics.n_train + result.metrics.n_test == len(combined)


def test_neural_engine_diagnostics_backend_key():
    X, y = _reg_data(n=80)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    combined, mask = _make_combined_df(X_tr, X_te, y_tr, X.shape[1])
    result = NeuralPhysicsEngine().run(
        combined, target_col="__target__", explicit_train_mask=mask, n_cycles=10
    )
    assert result is not None
    assert result.diagnostics is not None
    assert result.diagnostics.get("backend") == "neural"


# ── run_neural_prediction functional interface ─────────────────────────────

def test_run_neural_prediction_regression():
    X, y = _reg_data(n=80)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    combined, mask = _make_combined_df(X_tr, X_te, y_tr, X.shape[1])
    result = run_neural_prediction(
        combined,
        target_col="__target__",
        explicit_train_mask=mask,
        n_cycles=10,
        random_seed=0,
    )
    assert result is not None
    assert result.test_predicted is not None


# ── PhysicsPredictor(backend="neural") — sklearn API ──────────────────────

def test_neural_backend_fit_returns_self():
    X, y = _clf_data()
    clf = PhysicsPredictor(n_cycles=5, backend="neural")
    assert clf.fit(X, y) is clf


def test_neural_backend_predict_shape_clf():
    X, y = _clf_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    clf = PhysicsPredictor(n_cycles=5, backend="neural")
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    assert preds.shape == (len(y_te),)


def test_neural_backend_predict_valid_classes():
    X, y = _clf_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    clf = PhysicsPredictor(n_cycles=5, backend="neural")
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    unique_classes = set(np.unique(y_tr).tolist())
    for p in preds:
        assert p in unique_classes, f"Unexpected prediction: {p}"


def test_neural_backend_score_returns_float_clf():
    X, y = _clf_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    clf = PhysicsPredictor(n_cycles=5, backend="neural")
    clf.fit(X_tr, y_tr)
    score = clf.score(X_te, y_te)
    assert 0.0 <= score <= 1.0


def test_neural_backend_predict_shape_reg():
    X, y = _reg_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    reg = PhysicsPredictor(plane="solid", n_cycles=5, backend="neural")
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert preds.shape == (len(y_te),)


def test_neural_backend_predict_finite_reg():
    X, y = _reg_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    reg = PhysicsPredictor(plane="solid", n_cycles=5, backend="neural")
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert all(math.isfinite(float(p)) for p in preds)


def test_neural_backend_score_finite_reg():
    X, y = _reg_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    reg = PhysicsPredictor(plane="solid", n_cycles=5, backend="neural")
    reg.fit(X_tr, y_tr)
    score = reg.score(X_te, y_te)
    assert math.isfinite(score)


@pytest.mark.slow
def test_neural_backend_cross_val_score_clf():
    from sklearn.model_selection import cross_val_score
    X, y = _clf_data()
    clf = PhysicsPredictor(n_cycles=5, backend="neural")
    scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


@pytest.mark.slow
def test_neural_backend_cross_val_score_reg():
    from sklearn.model_selection import cross_val_score
    X, y = _reg_data()
    reg = PhysicsPredictor(plane="solid", n_cycles=5, backend="neural")
    scores = cross_val_score(reg, X, y, cv=3, scoring="r2")
    assert len(scores) == 3


def test_neural_backend_get_params_has_backend():
    clf = PhysicsPredictor(backend="neural")
    params = clf.get_params()
    assert "backend" in params
    assert params["backend"] == "neural"


def test_neural_backend_set_params():
    clf = PhysicsPredictor()
    clf.set_params(backend="neural")
    assert clf.backend == "neural"


def test_default_backend_is_physics():
    clf = PhysicsPredictor()
    assert clf.backend == "physics"


def test_physics_backend_unchanged():
    """Ensure default backend still works exactly as before."""
    X, y = _clf_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    clf = PhysicsPredictor(n_cycles=5, backend="physics")
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    assert preds.shape == (len(y_te),)


def test_neural_backend_with_quantile_transform():
    X, y = _reg_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    reg = PhysicsPredictor(
        plane="solid", n_cycles=5, backend="neural", quantile_transform=True
    )
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert all(math.isfinite(float(p)) for p in preds)


def test_neural_backend_with_poly_degree2():
    X, y = _reg_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    reg = PhysicsPredictor(
        plane="solid", n_cycles=5, backend="neural", poly_degree=2
    )
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert all(math.isfinite(float(p)) for p in preds)


def test_neural_backend_with_categorical_features():
    rng = np.random.default_rng(99)
    n = 100
    X = pd.DataFrame({
        "num1": rng.normal(0, 1, n),
        "cat1": rng.choice(["a", "b", "c"], n),
    })
    y = (X["num1"] > 0).astype(int).to_numpy()
    X_tr, X_te = X.iloc[:75].reset_index(drop=True), X.iloc[75:].reset_index(drop=True)
    y_tr, y_te = y[:75], y[75:]
    clf = PhysicsPredictor(n_cycles=5, backend="neural")
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    assert preds.shape == (len(y_te),)


def test_neural_backend_predict_before_fit_raises():
    clf = PhysicsPredictor(backend="neural")
    X = np.ones((5, 3))
    with pytest.raises(Exception):
        clf.predict(X)


def test_neural_backend_equilibrium_zones_populated():
    X, y = _reg_data(n=80)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    combined, mask = _make_combined_df(X_tr, X_te, y_tr, X.shape[1])
    result = NeuralPhysicsEngine().run(
        combined, target_col="__target__", explicit_train_mask=mask, n_cycles=10
    )
    assert result is not None
    assert len(result.equilibrium_zones) > 0
