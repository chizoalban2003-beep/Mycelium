"""Stage 62 — WorldModel.

A lightweight model-based planning component.  The WorldModel learns to
predict the *next state* and *expected reward* given the current state and
an action index, enabling the agent to evaluate candidate actions without
consulting the real environment.

Classes
-------
WorldModel
    Learns transition (s, a) → s' and reward (s, a) → r.
    Exposes ``plan(state, actions)`` to score candidate actions via
    multi-step rollout imagination.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class WorldModel:
    """Model-based component: learns environment dynamics from experience.

    Parameters
    ----------
    horizon : int
        Number of imagined rollout steps used in ``plan()``.
    n_actions : int
        Cardinality of the discrete action space.
    discount : float
        Discount factor γ applied to future imagined rewards.

    Attributes
    ----------
    fitted_ : bool
        True after the first call to ``update()``.
    """

    def __init__(
        self,
        horizon: int = 3,
        n_actions: int = 2,
        discount: float = 0.95,
    ) -> None:
        self.horizon = horizon
        self.n_actions = n_actions
        self.discount = discount

        # One transition model per action
        self._transition: list[Ridge] = [Ridge(alpha=1.0) for _ in range(n_actions)]
        self._reward: list[Ridge] = [Ridge(alpha=1.0) for _ in range(n_actions)]
        self._scaler = StandardScaler()
        self.fitted_ = False

        # Replay store
        self._states: list[np.ndarray] = []
        self._actions: list[int] = []
        self._next_states: list[np.ndarray] = []
        self._rewards: list[float] = []

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def record(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
    ) -> "WorldModel":
        """Store a (s, a, s', r) transition for subsequent learning."""
        self._states.append(np.asarray(state, dtype=np.float64).ravel())
        self._actions.append(int(action))
        self._next_states.append(np.asarray(next_state, dtype=np.float64).ravel())
        self._rewards.append(float(reward))
        return self

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def update(self, min_samples: int = 5) -> "WorldModel":
        """Fit transition and reward models from collected experience.

        Parameters
        ----------
        min_samples : int
            Minimum per-action samples required to fit that action's model.
        """
        if len(self._states) < min_samples:
            return self

        S = np.array(self._states)
        S_scaled = self._scaler.fit_transform(S)
        A = np.array(self._actions)
        S_next = np.array(self._next_states)
        R = np.array(self._rewards)

        for a in range(self.n_actions):
            mask = A == a
            if mask.sum() < min_samples:
                continue
            Xa = S_scaled[mask]
            self._transition[a].fit(Xa, S_next[mask])
            self._reward[a].fit(Xa, R[mask])

        self.fitted_ = True
        return self

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(self, state: np.ndarray, actions: list[int] | None = None) -> int:
        """Return the best action index via multi-step imagined rollout.

        If the model has not been fitted yet, returns action 0.

        Parameters
        ----------
        state : np.ndarray
            Current observable state vector.
        actions : list[int] or None
            Candidate actions to evaluate.  Defaults to all n_actions.

        Returns
        -------
        int
            Index of the action with the highest discounted imagined return.
        """
        if not self.fitted_:
            return 0

        if actions is None:
            actions = list(range(self.n_actions))

        best_action = actions[0]
        best_return = -np.inf

        for a in actions:
            s = np.asarray(state, dtype=np.float64).ravel()
            total = 0.0
            gamma = 1.0
            for _ in range(self.horizon):
                s_scaled = self._scaler.transform(s.reshape(1, -1))
                r = float(self._reward[a].predict(s_scaled)[0])
                total += gamma * r
                gamma *= self.discount
                s = self._transition[a].predict(s_scaled)[0]
            if total > best_return:
                best_return = total
                best_action = a

        return best_action

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a short diagnostic summary."""
        return {
            "fitted": self.fitted_,
            "n_transitions": len(self._states),
            "horizon": self.horizon,
            "n_actions": self.n_actions,
            "discount": self.discount,
        }
