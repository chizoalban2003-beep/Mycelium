"""scikit-learn compatible estimator wrapper for the PhysML physics predictor.

Usage
-----
    from physml import PhysicsPredictor

    # Classification
    clf = PhysicsPredictor(plane="liquid", n_cycles=20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))

    # Regression
    reg = PhysicsPredictor(plane="solid", n_cycles=30)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

Competitive improvements
------------------------
Five techniques narrow the gap to ensemble methods:

A. ``quantile_transform=True``
    Applies a QuantileTransformer (rank-normalization) to every numeric
    feature before the physics pass, exposing non-linear structure to the
    linear electrophoresis equations.

B. Isotope fix in predict
    Interaction ("isotope") columns created during ``fit`` are stored as
    named recipes and reconstructed from test rows in ``predict``, so the
    engine sees the same feature space during training and inference.

C. ``poly_degree=2``
    Adds all pairwise interaction terms of the top-k most-charged features
    via ``PolynomialFeatures``, giving the linear physics engine approximate
    non-linear capacity.

D. ``n_estimators > 1`` / ``bootstrap=True``
    Runs multiple independent physics passes on (bootstrap) subsamples and
    combines them by majority vote (classification) or mean (regression),
    reducing variance the same way a Random Forest does.

E. ``residual_model="ridge"`` / ``"logistic"``
    After the physics engine, a lightweight Ridge or Logistic Regression is
    fit on out-of-fold physics predictions stacked with raw features.  This
    second stage corrects systematic residuals the linear engine cannot
    model.

Notes
-----
The physics engine runs a coupled train+test electrophoresis pass each time
`predict` (or `score`) is called, using the stored training data as the
"train rows" and new rows as "test rows" via an explicit train mask.
This transductive design preserves the full physics simulation fidelity.
"""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.base import BaseEstimator
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.preprocessing import PolynomialFeatures, QuantileTransformer
    from sklearn.utils.multiclass import unique_labels
    from sklearn.utils.validation import check_is_fitted
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False
    BaseEstimator = object  # type: ignore[assignment,misc]

from physml.predictor import (
    PhysicsPlane,
    PredictionResult,
    PredictorRuntimeState,
    infer_target_kind,
    run_physics_prediction,
)
from physml.neural_engine import NeuralPhysicsEngine, _encode_dataframe


def _to_dataframe(X: Any, feature_names: list[str] | None = None) -> pd.DataFrame:
    """Convert array-like or DataFrame input to a pandas DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cols = feature_names if feature_names is not None else [f"f{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


def _resolve_plane(plane: str | PhysicsPlane) -> PhysicsPlane:
    if isinstance(plane, PhysicsPlane):
        return plane
    try:
        return PhysicsPlane(str(plane).lower())
    except ValueError:
        return PhysicsPlane.liquid


def _majority_vote(preds_2d: list[np.ndarray]) -> np.ndarray:
    """Column-wise majority vote over a list of equal-length prediction arrays."""
    arr = np.array([[str(p) for p in row] for row in preds_2d])  # (n_est, n_samples)
    return np.array([Counter(arr[:, j]).most_common(1)[0][0] for j in range(arr.shape[1])])


class PhysicsPredictor(BaseEstimator):
    """Physics-inspired predictor for structured tabular data.

    Implements the electrophoresis prediction engine as a scikit-learn
    compatible estimator.  The estimator auto-detects whether the target
    is categorical (classification) or numeric (regression) and adjusts
    scoring accordingly.

    Parameters
    ----------
    plane : {"solid", "liquid", "gas"}, default "liquid"
        Electrophoresis medium preset.  "solid" → low viscosity, high
        structure; "gas" → high viscosity, exploratory.
    n_cycles : int, default 30
        Number of electrophoresis migration cycles.
    cycle_learning_rate : float, default 0.18
        Per-cycle learning rate for charge updates.
    random_seed : int, default 42
        Random seed for reproducibility.
    enable_isotopes : bool, default True
        Generate interaction isotope features automatically.
    pcr_enabled : bool, default False
        Enable PCR-style amplification of statistically significant
        features.
    cascade_enabled : bool, default True
        Enable collinearity complex suppression.
    competitive_inhibition : bool, default True
        Enable cross-feature competitive inhibition.
    train_fraction : float, default 0.8
        Used only during ``fit`` for internal diagnostics; ``predict``
        always uses an explicit train mask based on stored training data.
    n_cycles_fit : int or None, default None
        Override ``n_cycles`` when calling ``fit`` (leave None to use
        the same ``n_cycles``).
    quantile_transform : bool, default False
        (Improvement A) Rank-normalize numeric features before the physics
        pass so non-linear structure is linearised.
    poly_degree : int, default 1
        (Improvement C) Polynomial interaction degree applied to the top-k
        features by physics weight.  1 = disabled; 2 = pairwise products.
    poly_top_k : int, default 10
        Maximum number of features selected for polynomial expansion.
    n_estimators : int, default 1
        (Improvement D) Number of independent physics passes to average.
        Values > 1 activate ensemble / bagging mode.
    bootstrap : bool, default False
        When ``n_estimators > 1``, sample training rows with replacement
        for each base estimator (bagging).
    residual_model : {"ridge", "logistic"} or None, default None
        (Improvement E) Second-stage corrector fit on out-of-fold physics
        predictions stacked with the original features.
    extra_kwargs : dict or None, default None
        Any additional keyword arguments forwarded verbatim to
        ``run_physics_prediction`` (``backend="physics"``) or
        ``run_neural_prediction`` (``backend="neural"``).
    backend : {"physics", "neural"}, default "physics"
        Prediction backend to use.

        * ``"physics"`` — original electrophoresis engine (default, no
          breaking changes).
        * ``"neural"`` — neural engine: 3-layer MLP (256 → 128) with a
          single-head feature-attention block (Stage 1 + 2).  The sklearn
          API (fit / predict / score / cross_val_score) is identical; only
          the inner prediction loop changes.  ``QuantileTransformer`` (A)
          and ``PolynomialFeatures`` (C) are still applied as a front-end.
          Supports ``partial_fit`` for continual / online learning.
    replay_size : int, default 500
        Maximum number of rows kept in the replay buffer used by
        ``partial_fit`` (neural backend only).  Older rows are discarded
        when the buffer is full.
    """

    def __init__(
        self,
        plane: str | PhysicsPlane = "liquid",
        n_cycles: int = 30,
        cycle_learning_rate: float = 0.18,
        random_seed: int = 42,
        enable_isotopes: bool = True,
        pcr_enabled: bool = False,
        cascade_enabled: bool = True,
        competitive_inhibition: bool = True,
        train_fraction: float = 0.8,
        n_cycles_fit: int | None = None,
        # A – rank normalization
        quantile_transform: bool = False,
        # C – polynomial interactions
        poly_degree: int = 1,
        poly_top_k: int = 10,
        # D – ensemble / bagging
        n_estimators: int = 1,
        bootstrap: bool = False,
        # E – residual stacking
        residual_model: str | None = None,
        extra_kwargs: dict[str, Any] | None = None,
        # Neural backend (Stage 1 + 2)
        backend: str = "physics",
        # Stage 3 — continual learning
        replay_size: int = 500,
    ) -> None:
        self.plane = plane
        self.n_cycles = n_cycles
        self.cycle_learning_rate = cycle_learning_rate
        self.random_seed = random_seed
        self.enable_isotopes = enable_isotopes
        self.pcr_enabled = pcr_enabled
        self.cascade_enabled = cascade_enabled
        self.competitive_inhibition = competitive_inhibition
        self.train_fraction = train_fraction
        self.n_cycles_fit = n_cycles_fit
        self.quantile_transform = quantile_transform
        self.poly_degree = poly_degree
        self.poly_top_k = poly_top_k
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.residual_model = residual_model
        self.extra_kwargs = extra_kwargs
        self.backend = backend
        self.replay_size = replay_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_kwargs(self) -> dict[str, Any]:
        """Return base kwargs for run_physics_prediction."""
        return {
            "plane": _resolve_plane(self.plane),
            "n_cycles": int(self.n_cycles),
            "cycle_learning_rate": float(self.cycle_learning_rate),
            "random_seed": int(self.random_seed),
            "enable_isotopes": bool(self.enable_isotopes),
            "pcr_enabled": bool(self.pcr_enabled),
            "cascade_enabled": bool(self.cascade_enabled),
            "competitive_inhibition": bool(self.competitive_inhibition),
            "return_predictions": True,
            **(dict(self.extra_kwargs or {})),
        }

    def _is_classifier(self) -> bool:
        check_is_fitted(self)
        return bool(getattr(self, "is_classifier_", False))

    # ── A: QuantileTransformer ─────────────────────────────────────────

    def _apply_qt(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Apply stored QuantileTransformer to numeric columns (in-place copy)."""
        qt = getattr(self, "qt_", None)
        if qt is None:
            return X_df
        num_cols = [c for c in getattr(self, "qt_numeric_cols_", []) if c in X_df.columns]
        if not num_cols:
            return X_df
        result = X_df.copy()
        try:
            result[num_cols] = qt.transform(result[num_cols].astype(float).values)
        except Exception:
            pass
        return result

    # ── B: Isotope reconstruction ──────────────────────────────────────

    def _apply_isotopes(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct isotope interaction columns from stored recipes."""
        import warnings
        recipes: list[dict] = getattr(self, "isotope_recipes_", [])
        if not recipes:
            return X_df
        result = X_df.copy()
        train_means: dict[str, float] = getattr(self, "isotope_train_means_", {})
        for recipe in recipes:
            col_name: str = recipe["column"]
            if col_name in result.columns:
                continue
            num_col: str = recipe["numeric"]
            cat_col: str = recipe["categorical"]
            level: str = recipe["level"]
            if num_col not in result.columns or cat_col not in result.columns:
                missing = [c for c in (num_col, cat_col) if c not in result.columns]
                warnings.warn(
                    f"Isotope column {col_name!r} cannot be reconstructed: "
                    f"source column(s) {missing} absent at predict time. "
                    "Filling with 0.0.",
                    UserWarning,
                    stacklevel=3,
                )
                result[col_name] = 0.0
                continue
            x = pd.to_numeric(result[num_col], errors="coerce").fillna(0.0)
            x_centered = x - float(train_means.get(num_col, 0.0))
            # Use object dtype comparison to avoid pandas categorical/string mismatch
            cat_series = result[cat_col].astype(object).fillna("__MISSING__")
            level_mask = (cat_series == level).astype(float)
            result[col_name] = x_centered.values * level_mask.values
        return result

    # ── C: Polynomial features ─────────────────────────────────────────

    def _apply_poly(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial interaction columns for stored top-k features."""
        poly = getattr(self, "poly_", None)
        if poly is None:
            return X_df
        top_feats = [f for f in getattr(self, "poly_top_features_", []) if f in X_df.columns]
        if len(top_feats) < 2:
            return X_df
        result = X_df.copy()
        try:
            X_sub = result[top_feats].astype(float).values
            X_poly = poly.transform(X_sub)
            for i, name in enumerate(poly.get_feature_names_out(top_feats)):
                safe = f"__poly__{name}"
                if safe not in result.columns:
                    result[safe] = X_poly[:, i]
        except Exception:
            pass
        return result

    # ── Combined preprocessing pipeline ───────────────────────────────

    def _preprocess(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Apply QT → isotope reconstruction → polynomial expansion."""
        X_df = self._apply_qt(X_df)
        X_df = self._apply_isotopes(X_df)
        X_df = self._apply_poly(X_df)
        return X_df

    # ── Core engine predict pass ───────────────────────────────────────

    def _engine_predict(
        self,
        train_df_with_target: pd.DataFrame,
        X_test_df: pd.DataFrame,
        seed: int,
    ) -> list[Any]:
        """Run one physics engine predict pass and return test-row predictions.

        Parameters
        ----------
        train_df_with_target : DataFrame
            Training rows that already include the ``__target__`` column.
        X_test_df : DataFrame
            Test feature rows (no ``__target__`` column).
        seed : int
            Random seed forwarded to the engine.
        """
        n_train = int(train_df_with_target.shape[0])
        n_test = int(X_test_df.shape[0])
        train_targets = train_df_with_target["__target__"].to_numpy()
        dummy_target: Any = (
            train_targets[0]
            if self.is_classifier_
            else float(np.nanmean(train_targets.astype(float)))
        )

        test_part = X_test_df.copy()
        test_part["__target__"] = dummy_target
        # Align columns: test may be missing columns that only appear in train.
        for col in train_df_with_target.columns:
            if col not in test_part.columns:
                test_part[col] = 0.0 if col != "__target__" else dummy_target
        test_part = test_part[train_df_with_target.columns]

        combined = pd.concat([train_df_with_target, test_part], axis=0, ignore_index=True)
        explicit_mask = np.zeros(n_train + n_test, dtype=bool)
        explicit_mask[:n_train] = True

        kwargs = self._build_kwargs()
        kwargs["enable_isotopes"] = False  # isotopes already reconstructed before this call
        kwargs["explicit_train_mask"] = explicit_mask
        kwargs["random_seed"] = seed

        result: PredictionResult | None = None
        backend = str(getattr(self, "backend", "physics")).lower().strip()
        try:
            if backend == "neural":
                result = NeuralPhysicsEngine().run(
                    combined,
                    target_col="__target__",
                    runtime_state=None,
                    n_cycles=kwargs.get("n_cycles", int(self.n_cycles)),
                    cycle_learning_rate=kwargs.get("cycle_learning_rate", float(self.cycle_learning_rate)),
                    random_seed=seed,
                    explicit_train_mask=explicit_mask,
                    return_predictions=True,
                    plane=kwargs.get("plane", _resolve_plane(self.plane)),
                )
            else:
                result = run_physics_prediction(combined, target_col="__target__", runtime_state=None, **kwargs)
        except Exception:
            pass

        if result is not None and result.test_predicted:
            preds = result.test_predicted
            if result.test_row_indices is not None:
                idx_map = {int(i): preds[k] for k, i in enumerate(result.test_row_indices)}
                return [idx_map.get(n_train + j, preds[0] if preds else dummy_target) for j in range(n_test)]
            out = list(preds[:n_test])
            if len(out) < n_test:
                out += [dummy_target] * (n_test - len(out))
            return out
        return [dummy_target] * n_test

    # ── Stacking feature builder ───────────────────────────────────────

    def _stacking_X(self, physics_preds: np.ndarray, X_df: pd.DataFrame) -> np.ndarray:
        """Build stacking feature matrix: [encoded_physics_pred, numeric_X]."""
        num_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
        X_num = X_df[num_cols].astype(float).values if num_cols else np.empty((len(X_df), 0))
        if self.is_classifier_:
            cls_map = {str(c): float(i) for i, c in enumerate(self.classes_)}
            pred_enc = np.array([cls_map.get(str(p), 0.0) for p in physics_preds]).reshape(-1, 1)
        else:
            pred_enc = np.array([float(p) for p in physics_preds]).reshape(-1, 1)
        return np.hstack([pred_enc, X_num])

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: Any, y: Any) -> "PhysicsPredictor":
        """Fit the physics predictor on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X_df = _to_dataframe(X)
        self.feature_names_in_: list[str] = list(X_df.columns)
        self.n_features_in_: int = int(X_df.shape[1])

        y_arr = np.asarray(y)
        y_series = pd.Series(y_arr).reset_index(drop=True)
        target_kind = infer_target_kind(y_series)
        self.is_classifier_: bool = target_kind == "categorical"
        self.target_dtype_ = y_series.dtype

        if self.is_classifier_:
            self.classes_: np.ndarray = unique_labels(y) if _SKLEARN_AVAILABLE else np.unique(y)

        # ── A: Fit QuantileTransformer ────────────────────────────────
        self.qt_: QuantileTransformer | None = None
        self.qt_numeric_cols_: list[str] = []
        if bool(self.quantile_transform) and _SKLEARN_AVAILABLE:
            num_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
            if num_cols:
                n_q = min(int(X_df.shape[0]), 1000)
                qt = QuantileTransformer(
                    n_quantiles=n_q,
                    output_distribution="normal",
                    random_state=int(self.random_seed),
                )
                try:
                    qt.fit(X_df[num_cols].astype(float).values)
                    self.qt_ = qt
                    self.qt_numeric_cols_ = list(num_cols)
                except Exception:
                    pass

        X_df_t = self._apply_qt(X_df)

        # ── Initial physics fit pass (for weights + isotope diagnostics) ──
        n_cycles_fit = int(self.n_cycles_fit) if self.n_cycles_fit is not None else int(self.n_cycles)
        fit_input = X_df_t.copy()
        fit_input["__target__"] = y_series.to_numpy()
        fit_input = fit_input.reset_index(drop=True)

        kwargs_fit = self._build_kwargs()
        kwargs_fit["n_cycles"] = n_cycles_fit
        kwargs_fit["train_fraction"] = float(self.train_fraction)
        kwargs_fit["enable_isotopes"] = bool(self.enable_isotopes)

        runtime = PredictorRuntimeState(metadata={"source": "PhysicsPredictor.fit"})
        fit_result: PredictionResult | None = None
        backend = str(getattr(self, "backend", "physics")).lower().strip()
        try:
            if backend == "neural":
                fit_result = NeuralPhysicsEngine().run(
                    fit_input.copy(),
                    target_col="__target__",
                    runtime_state=runtime,
                    n_cycles=n_cycles_fit,
                    cycle_learning_rate=float(self.cycle_learning_rate),
                    random_seed=int(self.random_seed),
                    train_fraction=float(self.train_fraction),
                )
            else:
                fit_result = run_physics_prediction(
                    fit_input.copy(),
                    target_col="__target__",
                    runtime_state=runtime,
                    **kwargs_fit,
                )
        except Exception:
            pass
        self.runtime_state_: PredictorRuntimeState = runtime

        # ── B: Store isotope recipes and reconstruct columns ──────────
        self.isotope_recipes_: list[dict[str, str]] = []
        self.isotope_train_means_: dict[str, float] = {}
        if fit_result is not None and fit_result.diagnostics:
            iso_diag: dict = fit_result.diagnostics.get("isotopes") or {}
            self.isotope_recipes_ = list(iso_diag.get("pairs", []))
            for recipe in self.isotope_recipes_:
                num_col = recipe.get("numeric", "")
                if num_col and num_col in X_df_t.columns and num_col not in self.isotope_train_means_:
                    vals = pd.to_numeric(X_df_t[num_col], errors="coerce")
                    self.isotope_train_means_[num_col] = float(vals.mean())

        X_df_t = self._apply_isotopes(X_df_t)

        # ── C: Fit PolynomialFeatures on top-k physics-weighted features ─
        self.poly_: PolynomialFeatures | None = None
        self.poly_top_features_: list[str] = []
        if int(self.poly_degree) >= 2 and _SKLEARN_AVAILABLE:
            top_k = max(2, int(self.poly_top_k))
            if fit_result is not None and fit_result.weights:
                sorted_weights = sorted(fit_result.weights, key=lambda w: abs(w.weight), reverse=True)
                top_features = [
                    wi.feature
                    for wi in sorted_weights
                    if wi.feature in X_df_t.columns
                    and wi.feature_kind in ("numeric", "datetime", "bool")
                    and not wi.feature.startswith("__iso__")
                ][:top_k]
            else:
                top_features = [
                    c for c in X_df_t.columns
                    if c != "__target__"
                    and not c.startswith("__iso__")
                    and pd.api.types.is_numeric_dtype(X_df_t[c])
                ][:top_k]

            if len(top_features) >= 2:
                poly = PolynomialFeatures(
                    degree=int(self.poly_degree),
                    interaction_only=True,
                    include_bias=False,
                )
                try:
                    poly.fit(X_df_t[top_features].astype(float).values)
                    self.poly_ = poly
                    self.poly_top_features_ = list(top_features)
                    X_df_t = self._apply_poly(X_df_t)
                except Exception:
                    pass

        # Store final (fully transformed) training dataframe.
        train_final = X_df_t.copy()
        train_final["__target__"] = y_series.to_numpy()
        self.train_df_: pd.DataFrame = train_final.reset_index(drop=True)

        # ── Neural: fit and store the inductive engine ────────────────
        # When backend="neural", fit a persistent NeuralPhysicsEngine that
        # can be reused across predict() calls and updated via partial_fit().
        self._neural_engine_: NeuralPhysicsEngine | None = None
        if str(getattr(self, "backend", "physics")).lower().strip() == "neural":
            try:
                n_epochs = int(np.clip(
                    (int(self.n_cycles_fit) if self.n_cycles_fit else int(self.n_cycles)) * 10,
                    100, 2000,
                ))
                lr = float(np.clip(float(self.cycle_learning_rate) * 0.01, 1e-4, 0.01))
                tmp_df = self.train_df_.copy()
                X_enc, y_enc, feat_names, feat_kinds, lbl_enc = _encode_dataframe(tmp_df, "__target__")
                engine = NeuralPhysicsEngine(alpha=1e-4)
                engine.fit_model(
                    X_enc, y_enc,
                    is_classifier=bool(self.is_classifier_),
                    n_epochs=n_epochs,
                    lr=lr,
                    random_state=int(self.random_seed),
                    encoded_feature_names=feat_names,
                    encoded_feature_kinds=feat_kinds,
                    label_enc=lbl_enc,
                )
                self._neural_engine_ = engine
            except Exception:
                pass
        # Initialise the replay buffer (neural backend only)
        self._replay_buffer_: deque = deque(maxlen=max(1, int(self.replay_size)))

        # ── D: Store per-estimator training sets (ensemble / bagging) ─
        n_est = max(1, int(self.n_estimators))
        self.estimator_train_dfs_: list[pd.DataFrame] = []
        if n_est > 1:
            rng_boot = np.random.default_rng(int(self.random_seed))
            for _ in range(n_est):
                if bool(self.bootstrap):
                    idx = rng_boot.integers(0, len(y_arr), size=len(y_arr))
                    boot_df = X_df_t.iloc[idx].reset_index(drop=True).copy()
                    boot_df["__target__"] = y_arr[idx]
                else:
                    boot_df = train_final.copy()
                self.estimator_train_dfs_.append(boot_df)

        # ── E: Fit residual stacking corrector ────────────────────────
        self.residual_estimator_: Any = None
        res_name = str(self.residual_model or "").lower().strip()
        if res_name in ("ridge", "logistic") and _SKLEARN_AVAILABLE:
            n_splits = min(3, max(2, len(y_arr) // 10))
            try:
                if self.is_classifier_:
                    kf: KFold | StratifiedKFold = StratifiedKFold(
                        n_splits=n_splits, shuffle=True, random_state=int(self.random_seed)
                    )
                else:
                    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(self.random_seed))

                oof_preds: list[Any] = [None] * len(y_arr)
                for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_df_t, y_arr)):
                    fold_train = X_df_t.iloc[tr_idx].reset_index(drop=True).copy()
                    fold_train["__target__"] = y_arr[tr_idx]
                    fold_test = X_df_t.iloc[val_idx].reset_index(drop=True)
                    fold_p = self._engine_predict(
                        fold_train, fold_test,
                        seed=int(self.random_seed) + fold_i * 1000,
                    )
                    for k, vi in enumerate(val_idx):
                        oof_preds[vi] = fold_p[k]

                oof_arr = np.array(oof_preds, dtype=object)
                X_stack = self._stacking_X(oof_arr, X_df_t)
                if self.is_classifier_:
                    est: Any = LogisticRegression(max_iter=1000, random_state=int(self.random_seed))
                else:
                    est = Ridge(alpha=1.0)
                est.fit(X_stack, y_arr)
                self.residual_estimator_ = est
            except Exception:
                self.residual_estimator_ = None

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values for new samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        X_df = _to_dataframe(X, feature_names=self.feature_names_in_)
        # Apply the same preprocessing pipeline used during fit.
        X_df_t = self._preprocess(X_df)

        # ── Neural inductive path: use stored engine directly ─────────
        neural_engine: NeuralPhysicsEngine | None = getattr(self, "_neural_engine_", None)
        if (
            str(getattr(self, "backend", "physics")).lower().strip() == "neural"
            and neural_engine is not None
            and hasattr(neural_engine, "mlp_")
        ):
            output = self._neural_predict_inductive(neural_engine, X_df_t)
        else:
            # ── D: Ensemble / bagging predict (physics or fallback) ───
            n_est = max(1, int(self.n_estimators))
            estimator_train_dfs = getattr(self, "estimator_train_dfs_", [])
            if n_est > 1 and estimator_train_dfs:
                all_preds = [
                    self._engine_predict(est_df, X_df_t, seed=int(self.random_seed) + i * 137)
                    for i, est_df in enumerate(estimator_train_dfs)
                ]
                if self.is_classifier_:
                    output = _majority_vote(all_preds)
                else:
                    output = np.mean([np.array(p, dtype=float) for p in all_preds], axis=0)
            else:
                raw = self._engine_predict(self.train_df_, X_df_t, seed=int(self.random_seed))
                output = np.array(raw)

        # ── E: Residual stacking correction ───────────────────────────
        if self.residual_estimator_ is not None:
            try:
                X_stack = self._stacking_X(output, X_df_t)
                output = self.residual_estimator_.predict(X_stack)
            except Exception:
                pass

        if self.is_classifier_:
            try:
                return np.array(output).astype(self.target_dtype_)
            except Exception:
                return np.array(output)
        return np.array(output, dtype=float)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return class probability estimates (neural backend, classification only).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities ordered by ``self.classes_``.

        Raises
        ------
        ValueError
            If called on a regression task or when ``backend != "neural"``.
        """
        check_is_fitted(self)
        if not self.is_classifier_:
            raise ValueError("predict_proba is only available for classifiers.")
        neural_engine: NeuralPhysicsEngine | None = getattr(self, "_neural_engine_", None)
        if (
            str(getattr(self, "backend", "physics")).lower().strip() != "neural"
            or neural_engine is None
            or not hasattr(neural_engine, "mlp_")
        ):
            raise ValueError(
                "predict_proba is only supported when backend='neural'. "
                "For the physics backend use predict() directly."
            )
        X_df = _to_dataframe(X, feature_names=self.feature_names_in_)
        X_df_t = self._preprocess(X_df)
        tmp = X_df_t.copy()
        tmp["__target__"] = 0
        for col in self.train_df_.columns:
            if col not in tmp.columns:
                tmp[col] = 0.0 if col != "__target__" else 0
        tmp = tmp[self.train_df_.columns]
        X_aligned, _ = neural_engine.encode_aligned(tmp, "__target__")
        return neural_engine.predict_proba_model(X_aligned)

    def _neural_predict_inductive(
        self,
        engine: NeuralPhysicsEngine,
        X_df_t: pd.DataFrame,
    ) -> np.ndarray:
        """Use the stored NeuralPhysicsEngine directly (inductive mode).

        Encodes ``X_df_t`` to match the training feature schema, runs the
        stored MLP, and decodes classification labels back to original
        values.
        """
        try:
            # Build a temporary df with a dummy target for _encode_dataframe
            tmp = X_df_t.copy()
            tmp["__target__"] = 0
            # Align to train columns (same strategy as _engine_predict)
            for col in self.train_df_.columns:
                if col not in tmp.columns:
                    tmp[col] = 0.0 if col != "__target__" else 0
            tmp = tmp[self.train_df_.columns]

            X_enc, _, _, _, _ = _encode_dataframe(tmp, "__target__")
            X_aligned, _ = engine.encode_aligned(tmp, "__target__")
            raw_pred = engine.predict_model(X_aligned)

            if self.is_classifier_ and engine.label_enc_ is not None:
                try:
                    raw_int = raw_pred.astype(int)
                    raw_int = np.clip(raw_int, 0, len(engine.label_enc_.classes_) - 1)
                    decoded = engine.label_enc_.inverse_transform(raw_int)
                    # Cast back to the original target dtype (e.g. int64 for sklearn
                    # integer-label datasets) so that accuracy metrics compare correctly.
                    try:
                        return decoded.astype(getattr(self, "target_dtype_", decoded.dtype))
                    except (ValueError, TypeError):
                        return decoded
                except Exception:
                    pass
            return raw_pred
        except Exception:
            # Fall back to transductive path on any encoding error
            raw = self._engine_predict(self.train_df_, X_df_t, seed=int(self.random_seed))
            return np.array(raw)

    # ── Stage 3: partial_fit / continual learning ──────────────────────

    def partial_fit(self, X: Any, y: Any, *, ewc_lambda: float = 0.4) -> "PhysicsPredictor":
        """Incrementally update the neural engine with new labelled data.

        Only supported when ``backend="neural"``.  Internally:

        1. Applies the same preprocessing pipeline as ``fit`` (QT, isotopes,
           polynomial features).
        2. Appends the new rows to the replay buffer.
        3. Draws a replay sample from the buffer (excluding the newest rows).
        4. Calls ``NeuralPhysicsEngine.partial_fit_model`` with the combined
           new + replay batch.
        5. Applies EWC weight consolidation (pull weights toward the anchor
           set during ``fit``).
        6. Appends new rows to ``train_df_`` for future reference.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        ewc_lambda : float, default 0.4
            EWC consolidation strength forwarded to ``partial_fit_model``.
            0 = no EWC, 1 = weights frozen.

        Returns
        -------
        self
        """
        check_is_fitted(self)
        if str(getattr(self, "backend", "physics")).lower().strip() != "neural":
            raise ValueError(
                "partial_fit is only supported for backend='neural'. "
                "For the physics backend re-call fit() with augmented data."
            )
        engine: NeuralPhysicsEngine | None = getattr(self, "_neural_engine_", None)
        if engine is None or not hasattr(engine, "mlp_"):
            raise RuntimeError(
                "No fitted neural engine found. Call fit() before partial_fit()."
            )

        X_df = _to_dataframe(X, feature_names=self.feature_names_in_)
        X_df_t = self._preprocess(X_df)
        y_arr = np.asarray(y)

        # Align to training schema
        tmp = X_df_t.copy()
        tmp["__target__"] = y_arr
        for col in self.train_df_.columns:
            if col not in tmp.columns:
                tmp[col] = 0.0 if col != "__target__" else 0
        tmp = tmp[self.train_df_.columns]

        X_new_enc, y_new_enc = engine.encode_aligned(tmp, "__target__")

        # Update replay buffer
        buf: deque = getattr(self, "_replay_buffer_", deque(maxlen=max(1, int(self.replay_size))))
        self._replay_buffer_ = buf
        for i in range(len(y_arr)):
            buf.append((X_new_enc[i], y_new_enc[i]))

        # Sample replay (all buffer items except the newest len(y_arr) rows)
        X_replay: np.ndarray | None = None
        y_replay: np.ndarray | None = None
        n_new = len(y_arr)
        if len(buf) > n_new:
            replay_items = list(buf)[:-n_new]
            X_replay = np.array([r[0] for r in replay_items])
            y_replay = np.array([r[1] for r in replay_items])

        engine.partial_fit_model(
            X_new_enc, y_new_enc,
            X_replay=X_replay,
            y_replay=y_replay,
            ewc_lambda=ewc_lambda,
        )

        # Extend train_df_ so future _engine_predict calls see new data
        new_rows = tmp.reset_index(drop=True)
        self.train_df_ = pd.concat(
            [self.train_df_, new_rows], axis=0, ignore_index=True
        )
        return self

    # ── Stage 6: save / load ───────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist the fitted predictor (and neural engine) to disk.

        Uses ``joblib`` for serialisation, which handles numpy arrays and
        sklearn objects efficiently.

        Parameters
        ----------
        path : str or Path
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("joblib is required for save/load") from exc
        joblib.dump(self, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "PhysicsPredictor":
        """Load a previously saved predictor.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        PhysicsPredictor
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("joblib is required for save/load") from exc
        obj = joblib.load(str(path))
        if not isinstance(obj, cls):
            raise TypeError(f"Expected PhysicsPredictor, got {type(obj)}")
        return obj

    def score(self, X: Any, y: Any) -> float:
        """Return accuracy (classification) or R² (regression).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        score : float
        """
        check_is_fitted(self)
        y_pred = self.predict(X)
        y_true = np.asarray(y)
        if self.is_classifier_:
            return float(np.mean(y_pred == y_true))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return float(1.0 - ss_res / ss_tot)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "plane": self.plane,
            "n_cycles": self.n_cycles,
            "cycle_learning_rate": self.cycle_learning_rate,
            "random_seed": self.random_seed,
            "enable_isotopes": self.enable_isotopes,
            "pcr_enabled": self.pcr_enabled,
            "cascade_enabled": self.cascade_enabled,
            "competitive_inhibition": self.competitive_inhibition,
            "train_fraction": self.train_fraction,
            "n_cycles_fit": self.n_cycles_fit,
            "quantile_transform": self.quantile_transform,
            "poly_degree": self.poly_degree,
            "poly_top_k": self.poly_top_k,
            "n_estimators": self.n_estimators,
            "bootstrap": self.bootstrap,
            "residual_model": self.residual_model,
            "extra_kwargs": self.extra_kwargs,
            "backend": self.backend,
            "replay_size": self.replay_size,
        }

    def set_params(self, **params: Any) -> "PhysicsPredictor":
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ---------------------------------------------------------------------------
# Convenience subclasses — task-optimised defaults
# ---------------------------------------------------------------------------

class PhysicsRegressor(PhysicsPredictor):
    """PhysicsPredictor pre-configured for regression tasks.

    Enables ``quantile_transform`` and ``residual_model="ridge"`` by default,
    which consistently improves R² on tabular regression benchmarks.  All
    other parameters are unchanged and can still be overridden.

    Naming convention
    -----------------
    ``PhysicsRegressor`` follows the scikit-learn naming convention
    (``<Algorithm><Task>``) used by ``LinearRegression``, ``SVR``,
    ``RandomForestRegressor``, etc.

    Parameters
    ----------
    plane : str or PhysicsPlane, default "solid"
        Solid medium is the recommended preset for regression tasks.
    n_cycles : int, default 30
    quantile_transform : bool, default True
        Rank-normalise numeric features before the physics pass.
    residual_model : str or None, default "ridge"
        Ridge second-stage corrector for systematic residuals.
    **kwargs
        All remaining ``PhysicsPredictor`` parameters.

    Examples
    --------
    >>> from physml import PhysicsRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> reg = PhysicsRegressor()
    >>> reg.fit(X, y)
    PhysicsRegressor(...)
    >>> reg.score(X, y)        # R²
    """

    def __init__(
        self,
        plane: str | PhysicsPlane = "solid",
        n_cycles: int = 30,
        *,
        quantile_transform: bool = True,
        residual_model: str | None = "ridge",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            plane=plane,
            n_cycles=n_cycles,
            quantile_transform=quantile_transform,
            residual_model=residual_model,
            **kwargs,
        )


class PhysicsClassifier(PhysicsPredictor):
    """PhysicsPredictor pre-configured for classification tasks.

    Enables ``quantile_transform`` and ``residual_model="logistic"`` by
    default, which narrows the accuracy gap to ensemble methods on tabular
    classification benchmarks.

    Naming convention
    -----------------
    ``PhysicsClassifier`` follows the scikit-learn naming convention used by
    ``LogisticRegression``, ``SVC``, ``RandomForestClassifier``, etc.

    Parameters
    ----------
    plane : str or PhysicsPlane, default "liquid"
        Liquid medium is the recommended preset for classification tasks.
    n_cycles : int, default 20
    quantile_transform : bool, default True
        Rank-normalise numeric features before the physics pass.
    residual_model : str or None, default "logistic"
        Logistic second-stage corrector for systematic residuals.
    **kwargs
        All remaining ``PhysicsPredictor`` parameters.

    Examples
    --------
    >>> from physml import PhysicsClassifier
    >>> from sklearn.datasets import load_wine
    >>> X, y = load_wine(return_X_y=True)
    >>> clf = PhysicsClassifier()
    >>> clf.fit(X, y)
    PhysicsClassifier(...)
    >>> clf.score(X, y)        # accuracy
    """

    def __init__(
        self,
        plane: str | PhysicsPlane = "liquid",
        n_cycles: int = 20,
        *,
        quantile_transform: bool = True,
        residual_model: str | None = "logistic",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            plane=plane,
            n_cycles=n_cycles,
            quantile_transform=quantile_transform,
            residual_model=residual_model,
            **kwargs,
        )
