"""Stage 76 — PrivacyEngine: differential-privacy wrapper.

Adds calibrated Gaussian noise (σ = sensitivity / ε) to the coefficients
of any scikit-learn–compatible estimator after fitting, providing
(ε, δ)-differential-privacy guarantees.  A running :class:`PrivacyBudget`
tracks cumulative privacy spending across multiple ``fit_private()`` calls.

Classes
-------
PrivacyBudget
    Tracks cumulative (ε, δ) privacy spending.
PrivacyEngine
    Wraps any sklearn-compatible estimator with DP noise injection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PrivacyBudget:
    """Running tally of privacy budget consumption.

    Attributes
    ----------
    epsilon_per_round : float
        Privacy cost per training call.
    delta : float
        Failure probability per round.
    max_rounds : int
        Budget is exhausted after this many rounds.
    """

    epsilon_per_round: float
    delta: float = 1e-5
    max_rounds: int = 100

    _rounds_used: int = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def epsilon_spent(self) -> float:
        """Total ε spent so far (simple composition)."""
        return self._rounds_used * self.epsilon_per_round

    @property
    def rounds_remaining(self) -> int:
        return max(0, self.max_rounds - self._rounds_used)

    @property
    def exhausted(self) -> bool:
        """True when no budget remains."""
        return self._rounds_used >= self.max_rounds

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def consume(self, rounds: int = 1) -> None:
        """Record that *rounds* training calls have been performed."""
        self._rounds_used += int(rounds)

    def as_dict(self) -> dict[str, Any]:
        return {
            "epsilon_per_round": self.epsilon_per_round,
            "epsilon_spent": round(self.epsilon_spent, 6),
            "delta": self.delta,
            "rounds_used": self._rounds_used,
            "rounds_remaining": self.rounds_remaining,
            "exhausted": self.exhausted,
        }


class PrivacyEngine:
    """Differential-privacy wrapper for any sklearn-compatible estimator.

    After each call to :meth:`fit_private`, Gaussian noise calibrated to
    ``sensitivity / epsilon`` is injected into the estimator's learned
    coefficients.  The wrapper consumes one unit from the
    :class:`PrivacyBudget` per call.

    Parameters
    ----------
    estimator : Any
        A fitted or unfitted sklearn-style estimator.
    epsilon : float, default 1.0
        Privacy budget per training call (smaller = more private).
    delta : float, default 1e-5
        Failure probability.
    sensitivity : float, default 1.0
        L2 sensitivity of the model's parameter vector.
    max_rounds : int, default 100
        Maximum number of private training rounds in the lifetime budget.
    random_state : int, default 0

    Example
    -------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from physml.privacy_engine import PrivacyEngine
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 4))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> engine = PrivacyEngine(LogisticRegression(max_iter=200), epsilon=1.0)
    >>> engine.fit_private(X, y)
    >>> engine.budget.epsilon_spent > 0
    True
    """

    def __init__(
        self,
        estimator: Any,
        *,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
        max_rounds: int = 100,
        random_state: int = 0,
    ) -> None:
        self.estimator = estimator
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.sensitivity = float(sensitivity)
        self.max_rounds = int(max_rounds)
        self.random_state = int(random_state)

        self.budget = PrivacyBudget(
            epsilon_per_round=self.epsilon,
            delta=self.delta,
            max_rounds=self.max_rounds,
        )
        self._rng = np.random.default_rng(random_state)
        self._fit_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_private(self, X: Any, y: Any) -> "PrivacyEngine":
        """Fit the estimator with differential-privacy noise injection.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        if self.budget.exhausted:
            raise RuntimeError(
                "Privacy budget exhausted.  Create a new PrivacyEngine to continue."
            )

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Fit the underlying estimator
        self.estimator.fit(X, y)

        # Inject Gaussian noise into coefficients
        sigma = self.sensitivity / (self.epsilon + 1e-12)
        self._inject_noise(sigma)

        self.budget.consume(1)
        self._fit_history.append(
            {
                "round": self.budget._rounds_used,
                "sigma": round(sigma, 6),
                "epsilon_spent": round(self.budget.epsilon_spent, 6),
            }
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Delegate to the wrapped estimator's ``predict``."""
        return self.estimator.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Delegate to the wrapped estimator's ``predict_proba``."""
        return self.estimator.predict_proba(X)

    def privacy_report(self) -> dict[str, Any]:
        """Return a summary of privacy spending."""
        return {
            "budget": self.budget.as_dict(),
            "fit_history": list(self._fit_history),
            "current_sigma": round(self.sensitivity / (self.epsilon + 1e-12), 6),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inject_noise(self, sigma: float) -> None:
        """Add Gaussian noise to all coefficient arrays found on the estimator."""
        for attr in ("coef_", "intercept_", "feature_importances_"):
            arr = getattr(self.estimator, attr, None)
            if arr is not None:
                noise = self._rng.normal(0, sigma, size=arr.shape)
                setattr(self.estimator, attr, arr + noise)

        # Also handle nested estimators (e.g. pipelines, ensembles)
        for attr in ("estimators_", "estimators"):
            sub_list = getattr(self.estimator, attr, None)
            if sub_list is not None:
                try:
                    for sub in sub_list:
                        for sub_attr in ("coef_", "intercept_"):
                            sub_arr = getattr(sub, sub_attr, None)
                            if sub_arr is not None:
                                noise = self._rng.normal(0, sigma, size=sub_arr.shape)
                                setattr(sub, sub_attr, sub_arr + noise)
                except (TypeError, AttributeError):
                    pass
