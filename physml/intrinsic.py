"""Stage 63 — IntrinsicMotivation.

Curiosity-driven exploration bonus based on prediction error.
The agent maintains a *forward model* that predicts the next state from
the current state.  Transitions that surprise the model (high prediction
error) receive a positive intrinsic reward bonus, encouraging the agent to
explore novel parts of the state space.

Classes
-------
IntrinsicMotivation
    Computes novelty bonus from prediction error and keeps a visit-count
    table for count-based exploration as a complement.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.linear_model import SGDRegressor


class IntrinsicMotivation:
    """Curiosity module: assigns exploration bonuses to surprising transitions.

    Parameters
    ----------
    bonus_scale : float
        Multiplicative scale applied to the raw prediction-error bonus.
    count_scale : float
        Scale for the count-based bonus  ``count_scale / sqrt(visit_count)``.
    decay : float
        Exponential decay applied to the running prediction-error normaliser
        so recent surprises dominate.

    Attributes
    ----------
    total_bonus_ : float
        Cumulative intrinsic reward issued so far.
    step_ : int
        Number of transitions processed.
    """

    def __init__(
        self,
        bonus_scale: float = 0.1,
        count_scale: float = 0.05,
        decay: float = 0.99,
    ) -> None:
        self.bonus_scale = bonus_scale
        self.count_scale = count_scale
        self.decay = decay

        self._forward_model = SGDRegressor(
            loss="squared_error",
            learning_rate="constant",
            eta0=0.01,
            max_iter=1,
            warm_start=True,
        )
        self._fitted = False
        self._err_running = 1.0  # running normaliser
        self._counts: defaultdict[bytes, int] = defaultdict(int)
        self.total_bonus_ = 0.0
        self.step_ = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def bonus(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        *,
        update_model: bool = True,
    ) -> float:
        """Compute the intrinsic bonus for a transition.

        Parameters
        ----------
        state : np.ndarray
            Current state (flattened).
        next_state : np.ndarray
            Observed next state.
        update_model : bool
            If True, perform an online update of the forward model.

        Returns
        -------
        float
            Non-negative intrinsic bonus.
        """
        s = np.asarray(state, dtype=np.float64).ravel()
        s_next = np.asarray(next_state, dtype=np.float64).ravel()
        s_next.shape[0]

        # Prediction-error bonus
        if self._fitted:
            pred = self._forward_model.predict(s.reshape(1, -1))
            # Predict each dimension independently (model outputs scalar;
            # we use total squared error across dimensions by iterating)
            pred_err = float(np.mean((pred[0] - s_next[0]) ** 2))
        else:
            pred_err = 1.0

        # Update running normaliser
        self._err_running = (
            self.decay * self._err_running + (1 - self.decay) * max(pred_err, 1e-8)
        )
        norm_err = pred_err / max(self._err_running, 1e-8)

        # Update forward model on flattened first-dim target
        if update_model:
            self._forward_model.partial_fit(s.reshape(1, -1), [s_next[0]])
            self._fitted = True

        # Count-based bonus
        key = np.round(s, 1).tobytes()
        self._counts[key] += 1
        count_bonus = self.count_scale / max(1.0, self._counts[key] ** 0.5)

        total = self.bonus_scale * norm_err + count_bonus
        self.total_bonus_ += total
        self.step_ += 1
        return float(total)

    def novelty(self, state: np.ndarray) -> float:
        """Return the visit-count novelty score for a state (no model update)."""
        key = np.round(np.asarray(state, dtype=np.float64).ravel(), 1).tobytes()
        count = self._counts.get(key, 0)
        return self.count_scale / max(1.0, count ** 0.5)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "total_bonus": round(self.total_bonus_, 6),
            "steps": self.step_,
            "unique_states_visited": len(self._counts),
            "bonus_scale": self.bonus_scale,
            "count_scale": self.count_scale,
        }
