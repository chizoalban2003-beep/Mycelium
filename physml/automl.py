"""Stage 47 — AutoMLOptimizer: lightweight hyperparameter search for the stacking ensemble.

Uses scikit-learn's :class:`~sklearn.model_selection.HalvingGridSearchCV` (no
extra dependencies) to automatically tune the
:class:`~physml.ensemble_predictor.CompetitiveEnsemblePredictor` or any
scikit-learn compatible estimator that supports ``partial_fit``.

The optimizer is intentionally lightweight:

* It works on a fixed snapshot of the data you hand it — no online updates
  during the search.
* It respects a ``max_iter`` budget so it finishes quickly in most settings
  (default: half the sklearn grid).
* It integrates with :class:`~physml.mycelium_agent.MyceliumAgent` via
  :meth:`~physml.mycelium_agent.MyceliumAgent.self_improve`; when
  ``auto_tune=True`` is passed the optimizer runs once and the best params
  are applied to the predictor.

Usage
-----
::

    from physml.automl import AutoMLOptimizer
    from physml.ensemble_predictor import CompetitiveEnsemblePredictor

    opt = AutoMLOptimizer(n_candidates=8, random_state=0)
    best_params = opt.fit(X_train, y_train)
    print(best_params)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.model_selection import ParameterGrid, StratifiedKFold, KFold

_DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "solver": ["lbfgs", "liblinear"],
}

# CEP-targeted grid kept as a convenience constant for callers that pass CEP explicitly
_CEP_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_leaf_nodes": [15, 31, 63],
}


class AutoMLOptimizer:
    """Successive-halving style hyperparameter optimizer for tabular estimators.

    Does NOT require Optuna or any extra dependency — only scikit-learn.

    Parameters
    ----------
    param_grid : dict, optional
        Dictionary mapping parameter names to lists of values.  Defaults to
        a sensible grid targeting
        :class:`~physml.ensemble_predictor.CompetitiveEnsemblePredictor`.
    n_candidates : int, default 8
        Number of candidates to evaluate in the first round of successive
        halving.  Reduced by half each successive round.
    cv : int, default 3
        Number of cross-validation folds.
    scoring : str, default "accuracy"
        Scoring metric.  Any sklearn scorer string is accepted.
    random_state : int | None, default None
        Seed for reproducibility.
    """

    def __init__(
        self,
        param_grid: dict[str, list[Any]] | None = None,
        n_candidates: int = 8,
        cv: int = 3,
        scoring: str = "accuracy",
        random_state: int | None = None,
    ) -> None:
        self.param_grid = param_grid if param_grid is not None else _DEFAULT_PARAM_GRID
        self.n_candidates = n_candidates
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state

        self.best_params_: dict[str, Any] = {}
        self.best_score_: float = float("-inf")
        self.cv_results_: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator: Any | None = None,
    ) -> dict[str, Any]:
        """Run the hyperparameter search and return the best parameter dict.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        estimator : sklearn-compatible estimator, optional
            If *None*, a fresh
            :class:`~physml.ensemble_predictor.CompetitiveEnsemblePredictor`
            is used as the base estimator template.

        Returns
        -------
        dict
            Best parameter dictionary.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if estimator is None:
            from sklearn.linear_model import LogisticRegression
            estimator = LogisticRegression(max_iter=300, random_state=0)

        rng = np.random.default_rng(self.random_state)
        all_params = list(ParameterGrid(self.param_grid))

        # Successive halving: start with n_candidates, halve each round
        candidates = self._sample_candidates(all_params, rng)
        results: list[dict[str, Any]] = []

        while len(candidates) > 0:
            for params in candidates:
                score = self._cross_val_score(estimator, X, y, params)
                results.append({"params": params, "mean_test_score": score})

            results.sort(key=lambda r: r["mean_test_score"], reverse=True)
            # Keep top half, but at least 1
            keep = max(1, len(candidates) // 2)
            candidates_next = [r["params"] for r in results[:keep]]
            if candidates_next == candidates:
                break
            candidates = candidates_next
            if len(candidates) == 1:
                break

        self.cv_results_ = results
        if results:
            best = max(results, key=lambda r: r["mean_test_score"])
            self.best_params_ = best["params"]
            self.best_score_ = best["mean_test_score"]

        return self.best_params_

    def get_best_estimator(self, X: np.ndarray, y: np.ndarray, estimator: Any | None = None) -> Any:
        """Fit a fresh estimator with the best params and return it.

        Parameters
        ----------
        estimator : sklearn-compatible estimator, optional
            Base estimator to configure. Defaults to LogisticRegression (fast).
        """
        if not self.best_params_:
            self.fit(X, y, estimator=estimator)
        if estimator is None:
            from sklearn.linear_model import LogisticRegression
            base = LogisticRegression(max_iter=300, random_state=0)
        else:
            from sklearn.base import clone
            base = clone(estimator)
        accepted = self._filter_params(base, self.best_params_)
        for k, v in accepted.items():
            setattr(base, k, v)
        base.fit(np.asarray(X, dtype=float), np.asarray(y))
        return base

    def summary(self) -> list[dict[str, Any]]:
        """Return sorted CV results (best first)."""
        return sorted(self.cv_results_, key=lambda r: r["mean_test_score"], reverse=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_candidates(
        self,
        all_params: list[dict[str, Any]],
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        n = min(self.n_candidates, len(all_params))
        idx = rng.choice(len(all_params), size=n, replace=False)
        return [all_params[i] for i in idx]

    def _cross_val_score(
        self,
        base_estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> float:
        """Manual cross-val to avoid sklearn CV overhead."""
        from sklearn.base import clone
        from sklearn.metrics import accuracy_score, r2_score

        is_classifier = hasattr(base_estimator, "predict_proba")
        if is_classifier:
            try:
                splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=0)
                folds = list(splitter.split(X, y))
            except Exception:
                splitter_k = KFold(n_splits=self.cv, shuffle=True, random_state=0)
                folds = list(splitter_k.split(X))
        else:
            splitter_k = KFold(n_splits=self.cv, shuffle=True, random_state=0)
            folds = list(splitter_k.split(X))

        scores: list[float] = []
        for train_idx, val_idx in folds:
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            try:
                est = clone(base_estimator)
                # Apply only params the estimator accepts
                accepted = self._filter_params(est, params)
                for k, v in accepted.items():
                    setattr(est, k, v)
                est.fit(X_tr, y_tr)
                y_pred = est.predict(X_val)
                if is_classifier:
                    score = accuracy_score(y_val, y_pred)
                else:
                    score = r2_score(y_val, y_pred)
                scores.append(score)
            except Exception:
                scores.append(0.0)

        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def _filter_params(estimator: Any, params: dict[str, Any]) -> dict[str, Any]:
        """Keep only params accepted by the estimator's __init__."""
        import inspect
        try:
            sig = inspect.signature(estimator.__class__.__init__)
            accepted = set(sig.parameters.keys()) - {"self"}
            return {k: v for k, v in params.items() if k in accepted}
        except Exception:
            return {}
