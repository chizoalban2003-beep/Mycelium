"""Stage 48 — Conformal Prediction wrappers.

Split-conformal prediction produces *valid* prediction sets (classifiers)
or *valid* prediction intervals (regressors) with a user-specified coverage
guarantee — without any distributional assumptions.

References
----------
Angelopoulos & Bates (2022), "A Gentle Introduction to Conformal Prediction
and Distribution-Free Uncertainty Quantification", arXiv:2107.07511.

Usage
-----
::

    from physml.conformal import ConformalClassifier, ConformalRegressor

    # Classification
    clf = ConformalClassifier(base_estimator, alpha=0.1)   # 90 % coverage
    clf.fit(X_train, y_train)
    clf.calibrate(X_cal, y_cal)
    prediction_sets = clf.predict_set(X_test)   # list[set]

    # Regression
    reg = ConformalRegressor(base_estimator, alpha=0.1)
    reg.fit(X_train, y_train)
    reg.calibrate(X_cal, y_cal)
    intervals = reg.predict_interval(X_test)    # ndarray (n, 2)
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ConformalClassifier:
    """Split-conformal classifier with guaranteed coverage (1 − alpha).

    Requires the base estimator to have a ``predict_proba`` method.

    Parameters
    ----------
    base_estimator : sklearn-compatible classifier
        Must implement ``fit(X, y)`` and ``predict_proba(X)``.
    alpha : float, default 0.1
        Miscoverage level.  Coverage ≥ 1 − alpha is guaranteed on
        exchangeable data.
    """

    def __init__(self, base_estimator: Any, alpha: float = 0.1) -> None:
        self.base_estimator = base_estimator
        self.alpha = alpha

        self._classes: np.ndarray | None = None
        self._qhat: float | None = None          # calibration quantile
        self._is_calibrated: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalClassifier":
        """Fit the base estimator."""
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.base_estimator.fit(X, y)
        self._classes = np.unique(y)
        return self

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalClassifier":
        """Compute the calibration quantile from a held-out calibration set.

        This is the *split* step in split-conformal prediction.
        """
        X_cal = np.asarray(X_cal, dtype=float)
        y_cal = np.asarray(y_cal)

        proba = self.base_estimator.predict_proba(X_cal)  # (n, k)
        scores = self._nonconformity_scores(proba, y_cal)
        n = len(scores)
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(level, 1.0)
        self._qhat = float(np.quantile(scores, level))
        self._is_calibrated = True
        return self

    def predict_set(self, X: np.ndarray) -> list[np.ndarray]:
        """Return a prediction *set* for each sample.

        Each set is a 1-D array of class labels included in the conformal
        prediction set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        list of ndarray
            Length = n_samples; each element is an array of class labels.
        """
        self._require_calibrated()
        X = np.asarray(X, dtype=float)
        proba = self.base_estimator.predict_proba(X)  # (n, k)
        q = self._qhat
        sets: list[np.ndarray] = []
        for row in proba:
            # Include class if 1 - p(class) <= q  ↔  p(class) >= 1 - q
            included = self._classes[row >= (1.0 - q)]
            if len(included) == 0:
                # Fallback: always include most likely class
                included = self._classes[[np.argmax(row)]]
            sets.append(included)
        return sets

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point prediction (argmax of predict_proba)."""
        X = np.asarray(X, dtype=float)
        return self.base_estimator.predict(X)

    def coverage(self, X: np.ndarray, y: np.ndarray) -> float:
        """Empirical coverage fraction on a test set."""
        y = np.asarray(y)
        sets = self.predict_set(X)
        hits = sum(y[i] in sets[i] for i in range(len(y)))
        return hits / len(y)

    def set_sizes(self, X: np.ndarray) -> np.ndarray:
        """Return the size of each prediction set."""
        return np.array([len(s) for s in self.predict_set(X)])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _nonconformity_scores(
        self, proba: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """1 − P(true class | x) for each calibration sample."""
        classes_list = list(self._classes)
        scores = np.zeros(len(y))
        for i, yi in enumerate(y):
            try:
                j = classes_list.index(yi)
                scores[i] = 1.0 - proba[i, j]
            except (ValueError, IndexError):
                scores[i] = 1.0
        return scores

    def _require_calibrated(self) -> None:
        if not self._is_calibrated:
            raise RuntimeError("Call calibrate() before predict_set().")


class ConformalRegressor:
    """Split-conformal regressor producing valid prediction intervals.

    Parameters
    ----------
    base_estimator : sklearn-compatible regressor
        Must implement ``fit(X, y)`` and ``predict(X)``.
    alpha : float, default 0.1
        Miscoverage level.  Coverage ≥ 1 − alpha is guaranteed.
    """

    def __init__(self, base_estimator: Any, alpha: float = 0.1) -> None:
        self.base_estimator = base_estimator
        self.alpha = alpha

        self._qhat: float | None = None
        self._is_calibrated: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalRegressor":
        """Fit the base estimator."""
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.base_estimator.fit(X, y)
        return self

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalRegressor":
        """Compute the calibration interval half-width."""
        X_cal = np.asarray(X_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=float)

        y_pred = self.base_estimator.predict(X_cal)
        scores = np.abs(y_cal - y_pred)
        n = len(scores)
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(level, 1.0)
        self._qhat = float(np.quantile(scores, level))
        self._is_calibrated = True
        return self

    def predict_interval(self, X: np.ndarray) -> np.ndarray:
        """Return prediction intervals of shape (n_samples, 2).

        Column 0 = lower bound, column 1 = upper bound.
        """
        self._require_calibrated()
        X = np.asarray(X, dtype=float)
        y_pred = self.base_estimator.predict(X)
        q = self._qhat
        return np.column_stack([y_pred - q, y_pred + q])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point prediction."""
        X = np.asarray(X, dtype=float)
        return self.base_estimator.predict(X)

    def coverage(self, X: np.ndarray, y: np.ndarray) -> float:
        """Empirical coverage fraction on a test set."""
        y = np.asarray(y, dtype=float)
        intervals = self.predict_interval(X)
        hits = np.sum((y >= intervals[:, 0]) & (y <= intervals[:, 1]))
        return float(hits) / len(y)

    def interval_widths(self, X: np.ndarray) -> np.ndarray:
        """Return the width of each prediction interval."""
        intervals = self.predict_interval(X)
        return intervals[:, 1] - intervals[:, 0]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_calibrated(self) -> None:
        if not self._is_calibrated:
            raise RuntimeError("Call calibrate() before predict_interval().")
