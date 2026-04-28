"""Tests for the PhysicsPredictor scikit-learn estimator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physml import PhysicsPredictor
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

@pytest.mark.slow
def test_sklearn_cross_val_score_classification():
    from sklearn.model_selection import cross_val_score
    X, y = _clf_data()
    clf = PhysicsPredictor(n_cycles=5)
    scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


@pytest.mark.slow
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


# ── A: QuantileTransformer ────────────────────────────────────────────────

def test_quantile_transform_clf_shape():
    X, y = _clf_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    clf = PhysicsPredictor(n_cycles=5, quantile_transform=True)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    assert preds.shape == (len(y_te),)


def test_quantile_transform_stores_transformer():
    X, y = _clf_data()
    clf = PhysicsPredictor(n_cycles=3, quantile_transform=True)
    clf.fit(X, y)
    assert clf.qt_ is not None
    assert len(clf.qt_numeric_cols_) > 0


def test_quantile_transform_disabled_leaves_qt_none():
    X, y = _clf_data()
    clf = PhysicsPredictor(n_cycles=3, quantile_transform=False)
    clf.fit(X, y)
    assert clf.qt_ is None


def test_quantile_transform_reg_finite_predictions():
    X, y = _reg_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    reg = PhysicsPredictor(plane="solid", n_cycles=5, quantile_transform=True)
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert all(math.isfinite(float(p)) for p in preds)


# ── C: PolynomialFeatures ─────────────────────────────────────────────────

def test_poly_degree2_clf_shape():
    X, y = _clf_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    clf = PhysicsPredictor(n_cycles=5, poly_degree=2)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    assert preds.shape == (len(y_te),)


def test_poly_degree2_stores_transformer():
    X, y = _clf_data()
    clf = PhysicsPredictor(n_cycles=3, poly_degree=2)
    clf.fit(X, y)
    assert clf.poly_ is not None
    assert len(clf.poly_top_features_) >= 2


def test_poly_degree1_leaves_poly_none():
    X, y = _clf_data()
    clf = PhysicsPredictor(n_cycles=3, poly_degree=1)
    clf.fit(X, y)
    assert clf.poly_ is None


def test_poly_adds_columns_to_train_df():
    X, y = _clf_data()
    clf = PhysicsPredictor(n_cycles=3, poly_degree=2)
    clf.fit(X, y)
    poly_cols = [c for c in clf.train_df_.columns if c.startswith("__poly__")]
    assert len(poly_cols) > 0


def test_poly_degree2_reg_finite():
    X, y = _reg_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    reg = PhysicsPredictor(plane="solid", n_cycles=5, poly_degree=2)
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert all(math.isfinite(float(p)) for p in preds)


# ── B: Isotope fix ────────────────────────────────────────────────────────

def test_isotope_recipes_stored_on_fit_with_categorical():
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({
        "num1": rng.normal(0, 1, n),
        "cat1": rng.choice(["a", "b", "c"], n),
    })
    y = (X["num1"] > 0).astype(int)
    clf = PhysicsPredictor(n_cycles=3, enable_isotopes=True)
    clf.fit(X, y)
    # Recipes are stored (may be empty if no categorical col detected as feature)
    assert isinstance(clf.isotope_recipes_, list)
    assert isinstance(clf.isotope_train_means_, dict)


def test_isotope_recipes_empty_without_categorical():
    """Pure numeric datasets should produce no isotope recipes."""
    X, y = _clf_data()
    clf = PhysicsPredictor(n_cycles=3, enable_isotopes=True)
    clf.fit(X, y)
    assert clf.isotope_recipes_ == []


def test_isotope_predict_shape_with_categorical():
    import pandas as pd
    rng = np.random.default_rng(1)
    n = 80
    X = pd.DataFrame({
        "num1": rng.normal(0, 1, n),
        "cat1": rng.choice(["a", "b", "c"], n),
    })
    y = (X["num1"] > 0).astype(int)
    X_tr, X_te = X.iloc[:60].reset_index(drop=True), X.iloc[60:].reset_index(drop=True)
    y_tr, y_te = y.iloc[:60].to_numpy(), y.iloc[60:].to_numpy()
    clf = PhysicsPredictor(n_cycles=5)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    assert preds.shape == (len(y_te),)


# ── D: Ensemble / bagging ─────────────────────────────────────────────────

def test_ensemble_clf_shape():
    X, y = _clf_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    clf = PhysicsPredictor(n_cycles=4, n_estimators=3)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    assert preds.shape == (len(y_te),)


def test_ensemble_clf_valid_classes():
    X, y = _clf_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    clf = PhysicsPredictor(n_cycles=4, n_estimators=3)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    unique_classes = set(np.unique(y_tr).tolist())
    for p in preds:
        assert p in unique_classes


def test_ensemble_reg_finite():
    X, y = _reg_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    reg = PhysicsPredictor(plane="solid", n_cycles=4, n_estimators=3)
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert all(math.isfinite(float(p)) for p in preds)


def test_bootstrap_ensemble_stores_estimators():
    X, y = _clf_data()
    clf = PhysicsPredictor(n_cycles=3, n_estimators=3, bootstrap=True)
    clf.fit(X, y)
    assert len(clf.estimator_train_dfs_) == 3


def test_ensemble_n1_same_as_single():
    """n_estimators=1 should behave identically to the default."""
    X, y = _clf_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    clf1 = PhysicsPredictor(n_cycles=5, n_estimators=1, random_seed=7)
    clf1.fit(X_tr, y_tr)
    p1 = clf1.predict(X_te)

    clf_default = PhysicsPredictor(n_cycles=5, random_seed=7)
    clf_default.fit(X_tr, y_tr)
    p_def = clf_default.predict(X_te)

    np.testing.assert_array_equal(p1, p_def)


# ── E: Residual stacking ──────────────────────────────────────────────────

def test_residual_ridge_reg_finite():
    X, y = _reg_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    reg = PhysicsPredictor(plane="solid", n_cycles=5, residual_model="ridge")
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert preds.shape == (len(y_te),)
    assert all(math.isfinite(float(p)) for p in preds)


def test_residual_logistic_clf_valid_classes():
    X, y = _clf_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    clf = PhysicsPredictor(n_cycles=5, residual_model="logistic")
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    unique_classes = set(np.unique(y_tr).tolist())
    for p in preds:
        assert p in unique_classes


def test_residual_none_leaves_estimator_none():
    X, y = _clf_data()
    clf = PhysicsPredictor(n_cycles=3, residual_model=None)
    clf.fit(X, y)
    assert clf.residual_estimator_ is None


def test_residual_stores_estimator_when_enabled():
    X, y = _clf_data(seed=10)
    clf = PhysicsPredictor(n_cycles=3, residual_model="logistic")
    clf.fit(X, y)
    assert clf.residual_estimator_ is not None


def test_residual_ridge_reg_score_finite():
    X, y = _reg_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    reg = PhysicsPredictor(plane="solid", n_cycles=5, residual_model="ridge")
    reg.fit(X_tr, y_tr)
    s = reg.score(X_te, y_te)
    assert math.isfinite(s)


# ── Combined improvements ─────────────────────────────────────────────────

@pytest.mark.slow
def test_all_improvements_combined_clf():
    """All five improvements active simultaneously."""
    X, y = _clf_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    clf = PhysicsPredictor(
        n_cycles=5,
        quantile_transform=True,
        poly_degree=2,
        n_estimators=2,
        residual_model="logistic",
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    assert preds.shape == (len(y_te),)
    unique_classes = set(np.unique(y_tr).tolist())
    for p in preds:
        assert p in unique_classes


@pytest.mark.slow
def test_all_improvements_combined_reg():
    """All five improvements active simultaneously for regression."""
    X, y = _reg_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    reg = PhysicsPredictor(
        plane="solid",
        n_cycles=5,
        quantile_transform=True,
        poly_degree=2,
        n_estimators=2,
        residual_model="ridge",
    )
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert all(math.isfinite(float(p)) for p in preds)


def test_get_params_contains_new_keys():
    clf = PhysicsPredictor()
    params = clf.get_params()
    for key in ("quantile_transform", "poly_degree", "poly_top_k",
                "n_estimators", "bootstrap", "residual_model"):
        assert key in params, f"Missing param: {key}"


def test_set_params_new_keys():
    clf = PhysicsPredictor()
    clf.set_params(quantile_transform=True, poly_degree=2, n_estimators=3,
                   bootstrap=True, residual_model="ridge")
    assert clf.quantile_transform is True
    assert clf.poly_degree == 2
    assert clf.n_estimators == 3
    assert clf.bootstrap is True
    assert clf.residual_model == "ridge"


@pytest.mark.slow
def test_sklearn_cross_val_with_improvements():
    from sklearn.model_selection import cross_val_score
    X, y = _clf_data()
    clf = PhysicsPredictor(n_cycles=4, quantile_transform=True, poly_degree=2)
    scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)
