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

Notes
-----
The physics engine runs a coupled train+test electrophoresis pass each time
`predict` (or `score`) is called, using the stored training data as the
"train rows" and new rows as "test rows" via an explicit train mask.
This transductive design preserves the full physics simulation fidelity.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.utils.multiclass import unique_labels
    from sklearn.utils.validation import check_is_fitted
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    BaseEstimator = object
    ClassifierMixin = object
    RegressorMixin = object

from physml.predictor import (
    PhysicsPlane,
    PredictionResult,
    PredictorRuntimeState,
    infer_target_kind,
    run_physics_prediction,
)


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
    extra_kwargs : dict or None, default None
        Any additional keyword arguments forwarded verbatim to
        ``run_physics_prediction``.
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
        extra_kwargs: dict[str, Any] | None = None,
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
        self.extra_kwargs = extra_kwargs

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
        n_features = int(X_df.shape[1])

        y_arr = np.asarray(y)
        # Preserve the original dtype so infer_target_kind correctly detects
        # integer class labels as "categorical" (few unique numeric values).
        y_series = pd.Series(y_arr).reset_index(drop=True)
        target_kind = infer_target_kind(y_series)
        self.is_classifier_: bool = target_kind == "categorical"
        self.target_dtype_ = y_series.dtype

        if self.is_classifier_:
            self.classes_: np.ndarray = unique_labels(y) if _SKLEARN_AVAILABLE else np.unique(y)

        # Store training data (reset index for safe concatenation later)
        train_df = X_df.copy()
        train_df["__target__"] = y_series.to_numpy()
        self.train_df_: pd.DataFrame = train_df.reset_index(drop=True)
        self.n_features_in_: int = n_features

        # Run an in-sample diagnostics pass to warm up the runtime state.
        n_cycles_fit = int(self.n_cycles_fit) if self.n_cycles_fit is not None else int(self.n_cycles)
        kwargs = self._build_kwargs()
        kwargs["n_cycles"] = n_cycles_fit
        kwargs["train_fraction"] = float(self.train_fraction)

        runtime = PredictorRuntimeState(metadata={"source": "PhysicsPredictor.fit"})
        try:
            run_physics_prediction(
                self.train_df_.copy(),
                target_col="__target__",
                runtime_state=runtime,
                **kwargs,
            )
        except Exception:
            pass
        self.runtime_state_: PredictorRuntimeState = runtime
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

        n_train = int(self.train_df_.shape[0])
        n_test = int(X_df.shape[0])

        # Build a dummy target for test rows using the same dtype as training targets
        # so infer_target_kind sees a consistent column type.
        train_targets = self.train_df_["__target__"].to_numpy()
        if self.is_classifier_:
            dummy_target = train_targets[0]
        else:
            dummy_target = float(np.nanmean(train_targets.astype(float)))

        test_part = X_df.copy()
        test_part["__target__"] = dummy_target

        # Align columns: test may lack isotope-derived columns added during fit.
        extra_cols = [c for c in self.train_df_.columns if c not in test_part.columns]
        for col in extra_cols:
            fill = 0.0 if col != "__target__" else dummy_target
            test_part[col] = fill
        test_part = test_part[self.train_df_.columns]

        combined = pd.concat(
            [self.train_df_, test_part],
            axis=0,
            ignore_index=True,
        )

        # Explicit train mask: first n_train rows are training rows.
        explicit_mask = np.zeros(n_train + n_test, dtype=bool)
        explicit_mask[:n_train] = True

        kwargs = self._build_kwargs()
        kwargs["enable_isotopes"] = False  # avoid adding new isotope cols in combined df
        kwargs["explicit_train_mask"] = explicit_mask

        result: PredictionResult | None = None
        try:
            result = run_physics_prediction(
                combined,
                target_col="__target__",
                runtime_state=None,
                **kwargs,
            )
        except Exception:
            pass

        # Extract test row predictions from result.
        if result is not None and result.test_predicted:
            preds = result.test_predicted
            # test_row_indices contains the indices (in combined) that are test rows.
            if result.test_row_indices is not None:
                idx_map = {int(i): preds[k] for k, i in enumerate(result.test_row_indices)}
                output = [idx_map.get(n_train + j, preds[0] if preds else dummy_target) for j in range(n_test)]
            else:
                # Fall back to first n_test predictions
                output = list(preds[:n_test])
                if len(output) < n_test:
                    output += [dummy_target] * (n_test - len(output))
        else:
            output = [dummy_target] * n_test

        if self.is_classifier_:
            # Cast predictions to the same dtype as the original training labels so that
            # integer class labels (0, 1, 2) compare correctly with str predictions.
            try:
                return np.array(output).astype(self.target_dtype_)
            except Exception:
                return np.array(output)
        return np.array(output, dtype=float)

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
        # R²
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
            "extra_kwargs": self.extra_kwargs,
        }

    def set_params(self, **params: Any) -> "PhysicsPredictor":
        for key, value in params.items():
            setattr(self, key, value)
        return self
