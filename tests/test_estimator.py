"""Tests for the PhysicsPredictor scikit-learn estimator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physml import PhysicsPredictor, PhysicsPlane
from physml.estimator import _to_dataframe


# ── Helpers ───────────────────────────────────────────────────────────────

def _clf_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (100, 4))
    y = ((X[:, 0] + 0.5 * X[:, 1]) > 0).astype(int)
    return X, y


def _reg_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (100, 3))
    y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(0, 0.3, 100)
    return X, y


def _split(X, y, test_frac: float = 0.25, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_te = max(1, int(n * test_frac))
    te, tr = idx[:n_te], idx[n_te:]
    return X[tr], X[te], y[tr], y[te]


# ── to_dataframe helper ───────────────────────────────────────────────────

def test_to_dataframe_from_numpy():
    arr = np.ones((10, 3))
    df = _to_dataframe(arr)
    assert df.shape == (10, 3)
    assert list(df.columns) == ["f0", "f1", "f2"]


def test_to_dataframe_preserves_dataframe():
    import pandas as pd
    df_in = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df_out = _to_dataframe(df_in)
    assert list(df_out.columns) == ["a", "b"]


# ── fit / predict ─────────────────────────────────────────────────────────

def test_classifier_fit_returns_self():
    X, y = _clf_data()
    clf = PhysicsPredictor(n_cycles=5)
    result = clf.fit(X, y)
    assert result is clf


def test_classifier_predict_correct_shape():
    X, y = _clf_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    clf = PhysicsPredictor(n_cycles=5)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    assert preds.shape == (len(y_te),)


def test_classifier_predict_valid_classes():
    X, y = _clf_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    clf = PhysicsPredictor(n_cycles=5)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    unique_classes = set(np.unique(y_tr).tolist())
    for p in preds:
        assert p in unique_classes, f"Unexpected prediction: {p}"


def test_classifier_score_returns_float_in_01():
    X, y = _clf_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    clf = PhysicsPredictor(n_cycles=8)
    clf.fit(X_tr, y_tr)
    score = clf.score(X_te, y_te)
    assert 0.0 <= score <= 1.0


def test_regressor_predict_correct_shape():
    X, y = _reg_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    reg = PhysicsPredictor(plane="solid", n_cycles=5)
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert preds.shape == (len(y_te),)


def test_regressor_predict_finite_values():
    X, y = _reg_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    reg = PhysicsPredictor(plane="solid", n_cycles=5)
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert all(math.isfinite(float(p)) for p in preds)


def test_regressor_score_returns_float():
    X, y = _reg_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    reg = PhysicsPredictor(plane="solid", n_cycles=8)
    reg.fit(X_tr, y_tr)
    score = reg.score(X_te, y_te)
    assert isinstance(score, float)


# ── Plane variants ────────────────────────────────────────────────────────

@pytest.mark.parametrize("plane", ["solid", "liquid", "gas"])
def test_all_planes_work_for_regression(plane: str):
    X, y = _reg_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    reg = PhysicsPredictor(plane=plane, n_cycles=4)
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert preds.shape == (len(y_te),)


# ── get_params / set_params ───────────────────────────────────────────────

def test_get_params_returns_expected_keys():
    clf = PhysicsPredictor(n_cycles=15, plane="solid")
    params = clf.get_params()
    assert "n_cycles" in params
    assert "plane" in params
    assert params["n_cycles"] == 15
    assert params["plane"] == "solid"


def test_set_params_updates_attributes():
    clf = PhysicsPredictor(n_cycles=10)
    clf.set_params(n_cycles=20, plane="gas")
    assert clf.n_cycles == 20
    assert clf.plane == "gas"


# ── sklearn cross_val_score compatibility ─────────────────────────────────

def test_sklearn_cross_val_score_classification():
    from sklearn.model_selection import cross_val_score
    X, y = _clf_data()
    clf = PhysicsPredictor(n_cycles=5)
    scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_sklearn_cross_val_score_regression():
    from sklearn.model_selection import cross_val_score
    X, y = _reg_data()
    reg = PhysicsPredictor(plane="solid", n_cycles=5)
    scores = cross_val_score(reg, X, y, cv=3, scoring="r2")
    assert len(scores) == 3


# ── is_classifier detection ───────────────────────────────────────────────

def test_is_classifier_detected_for_int_labels():
    X, y = _clf_data()
    clf = PhysicsPredictor(n_cycles=3)
    clf.fit(X, y)
    assert clf.is_classifier_ is True


def test_is_regressor_detected_for_continuous():
    X, y = _reg_data()
    reg = PhysicsPredictor(n_cycles=3)
    reg.fit(X, y)
    assert reg.is_classifier_ is False


# ── predict before fit raises ─────────────────────────────────────────────

def test_predict_before_fit_raises():
    clf = PhysicsPredictor()
    X = np.ones((5, 3))
    with pytest.raises(Exception):
        clf.predict(X)
