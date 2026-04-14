"""Stage 49 — Explainability module.

Provides model-agnostic feature-attribution using permutation importance
(Breiman, 2001) — no SHAP dependency required.

For estimators that expose ``feature_importances_`` (tree-based) or
``coef_`` (linear), those are used directly; otherwise permutation
importance is computed automatically.

Integration
-----------
* :meth:`~physml.mycelium_agent.MyceliumAgent.introspect` appends a
  ``"feature_importance"`` key when an ``Explainer`` is attached.
* :func:`explain_agent` is a convenience function that creates an
  ``Explainer`` from a fitted ``MyceliumAgent``.

Usage
-----
::

    from physml.explainability import Explainer

    exp = Explainer(n_repeats=5, random_state=0)
    exp.fit(estimator, X_val, y_val, feature_names=["a", "b", "c"])

    print(exp.top_features(k=3))
    # [("b", 0.42), ("a", 0.18), ("c", 0.07)]

    report = exp.report()
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Explainer:
    """Model-agnostic feature-importance estimator.

    Priority order for importance source:
    1. ``feature_importances_`` attribute (tree-based models)
    2. Absolute value of ``coef_`` (linear models)
    3. Permutation importance (model-agnostic fallback)

    Parameters
    ----------
    n_repeats : int, default 5
        Number of permutation shuffles per feature (used only in fallback).
    random_state : int | None, default None
        RNG seed for reproducibility.
    scoring : callable | None, default None
        ``scoring(y_true, y_pred) → float``.  If *None*, accuracy is used
        for classifiers (``predict_proba`` present) and MAE-derived score
        for regressors.
    """

    def __init__(
        self,
        n_repeats: int = 5,
        random_state: int | None = None,
        scoring: Any = None,
    ) -> None:
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.scoring = scoring

        self.importances_: np.ndarray | None = None
        self.importances_std_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None
        self._n_features: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "Explainer":
        """Compute feature importances.

        Parameters
        ----------
        estimator : fitted sklearn-compatible estimator
        X : array-like of shape (n_samples, n_features)
            Validation data used to estimate importance.
        y : array-like of shape (n_samples,)
        feature_names : list of str, optional

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_features = X.shape[1]
        self._n_features = n_features

        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = [f"x{i}" for i in range(n_features)]

        # Try direct model attributes first
        imps, stds = self._try_model_importances(estimator, n_features)
        if imps is None:
            imps, stds = self._permutation_importance(estimator, X, y)

        self.importances_ = imps
        self.importances_std_ = stds
        return self

    def top_features(self, k: int | None = None) -> list[tuple[str, float]]:
        """Return feature names sorted by importance (descending).

        Parameters
        ----------
        k : int | None
            Return only the top-k features.  All features if *None*.

        Returns
        -------
        list of (name, importance) tuples
        """
        self._require_fitted()
        pairs = list(zip(self.feature_names_, self.importances_))
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs if k is None else pairs[:k]

    def report(self) -> dict[str, Any]:
        """Return a summary dict with all feature importances."""
        self._require_fitted()
        return {
            "feature_importances": dict(zip(self.feature_names_, self.importances_.tolist())),
            "top_5": self.top_features(k=5),
            "n_features": self._n_features,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_model_importances(
        self, estimator: Any, n_features: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Try to extract importances directly from the model."""
        # Tree-based: feature_importances_
        if hasattr(estimator, "feature_importances_"):
            fi = np.asarray(estimator.feature_importances_, dtype=float)
            if fi.shape == (n_features,):
                return fi, np.zeros_like(fi)

        # Linear: coef_
        if hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_)
            if coef.ndim == 2:
                coef = np.mean(np.abs(coef), axis=0)
            else:
                coef = np.abs(coef)
            if coef.shape == (n_features,):
                total = coef.sum() or 1.0
                return coef / total, np.zeros_like(coef)

        return None, None

    def _permutation_importance(
        self, estimator: Any, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Model-agnostic permutation importance."""
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]
        is_classifier = hasattr(estimator, "predict_proba")

        # Baseline score
        baseline = self._score(estimator, X, y, is_classifier)

        all_deltas: list[list[float]] = [[] for _ in range(n_features)]
        for _ in range(self.n_repeats):
            for j in range(n_features):
                X_perm = X.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])
                delta = baseline - self._score(estimator, X_perm, y, is_classifier)
                all_deltas[j].append(delta)

        imps = np.array([np.mean(d) for d in all_deltas])
        stds = np.array([np.std(d) for d in all_deltas])
        # Clip negatives to 0 (shuffling improves score ↔ feature hurts)
        imps = np.clip(imps, 0.0, None)
        total = imps.sum() or 1.0
        return imps / total, stds

    def _score(
        self, estimator: Any, X: np.ndarray, y: np.ndarray, is_classifier: bool
    ) -> float:
        if self.scoring is not None:
            y_pred = estimator.predict(X)
            return float(self.scoring(y, y_pred))
        y_pred = estimator.predict(X)
        if is_classifier:
            return float(np.mean(y_pred == y))
        mae = float(np.mean(np.abs(y.astype(float) - y_pred.astype(float))))
        scale = float(np.std(y.astype(float))) or 1.0
        return 1.0 - mae / scale  # normalised score, higher = better

    def _require_fitted(self) -> None:
        if self.importances_ is None:
            raise RuntimeError("Call fit() before using the Explainer.")


def explain_agent(
    agent: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str] | None = None,
    n_repeats: int = 5,
    random_state: int | None = None,
) -> Explainer:
    """Convenience: create and fit an :class:`Explainer` from a fitted agent.

    Parameters
    ----------
    agent : MyceliumAgent (or any object with a ``_agent`` attribute that has
            a ``predictor`` or a ``predict`` method)
    X_val, y_val : validation data
    feature_names : optional column names
    n_repeats : permutation repeats
    random_state : RNG seed

    Returns
    -------
    Explainer
        Already fitted explainer.
    """
    # Resolve the underlying sklearn estimator
    estimator = _resolve_estimator(agent)
    exp = Explainer(n_repeats=n_repeats, random_state=random_state)
    exp.fit(estimator, X_val, y_val, feature_names=feature_names)
    return exp


def _resolve_estimator(agent: Any) -> Any:
    """Walk the agent wrapper chain to find a predict-capable object."""
    # MyceliumAgent wraps PhysicsAgent in ._agent
    if hasattr(agent, "_agent"):
        agent = agent._agent
    # PhysicsAgent exposes .predictor
    if hasattr(agent, "predictor") and agent.predictor is not None:
        return agent.predictor
    # Fall through: hope the agent itself is predict-capable
    return agent
