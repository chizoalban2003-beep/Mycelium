"""Stage 96 — EnvironmentModel: track and predict external environment state.

Maintains a lightweight model of the agent's environment — recording
observations, estimating transition dynamics, and producing a predicted
next state from any current state/action pair.

Classes
-------
EnvState
    A snapshot of the environment at one timestep.
EnvironmentModel
    Tracks observations and fits a transition model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class EnvState:
    """One observed environment snapshot.

    Attributes
    ----------
    obs : list[float]
        Observation vector.
    action : float or None
        Action taken from this state (if known).
    reward : float or None
        Reward received after the action (if known).
    timestamp : float
        Unix time of the observation.
    metadata : dict
        Extra information.
    """

    obs: List[float]
    action: Optional[float] = None
    reward: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnvironmentModel:
    """Tracks environment observations and fits a predictive transition model.

    The model learns ``next_obs = f(obs, action)`` from historical
    (obs, action, next_obs) tuples using a ridge-regression estimator by
    default.

    Parameters
    ----------
    obs_dim : int, optional
        Expected dimensionality of observation vectors.  Used only for
        validation when provided.
    model : sklearn-compatible regressor, optional
        Custom estimator with ``fit(X, y)`` and ``predict(X)`` interface.
        Defaults to ``Ridge(alpha=1.0)``.
    max_history : int
        Maximum number of :class:`EnvState` snapshots to store.  Oldest
        entries are evicted when the limit is reached.  ``-1`` = unlimited.

    Attributes
    ----------
    history_ : list[EnvState]
        Recorded environment states.
    fitted_ : bool
        Whether the transition model has been trained.
    """

    def __init__(
        self,
        obs_dim: Optional[int] = None,
        model=None,
        max_history: int = 1000,
    ) -> None:
        self.obs_dim = obs_dim
        self.max_history = max_history
        if model is None:
            from sklearn.linear_model import Ridge

            model = Ridge(alpha=1.0)
        self._model = model
        self.history_: List[EnvState] = []
        self.fitted_: bool = False

    # ------------------------------------------------------------------
    def record(self, state: EnvState) -> None:
        """Append *state* to the internal history."""
        self.history_.append(state)
        if self.max_history > 0 and len(self.history_) > self.max_history:
            self.history_.pop(0)

    def record_transition(
        self,
        obs: Sequence[float],
        action: float,
        reward: float = 0.0,
    ) -> EnvState:
        """Convenience wrapper — create and record a state.

        Returns
        -------
        EnvState
        """
        state = EnvState(obs=list(obs), action=action, reward=reward)
        self.record(state)
        return state

    # ------------------------------------------------------------------
    def fit(self) -> "EnvironmentModel":
        """Fit the transition model on recorded (obs, action) → next_obs pairs.

        Requires at least two recorded states.

        Returns
        -------
        EnvironmentModel
            *self* for chaining.

        Raises
        ------
        ValueError
            If fewer than two transitions have been recorded.
        """
        if len(self.history_) < 2:
            raise ValueError("Need at least 2 recorded states to fit.")

        X, y = [], []
        for i in range(len(self.history_) - 1):
            s = self.history_[i]
            action = s.action if s.action is not None else 0.0
            x_row = list(s.obs) + [action]
            X.append(x_row)
            y.append(self.history_[i + 1].obs)

        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=float)
        self._model.fit(X_arr, y_arr)
        self.fitted_ = True
        return self

    # ------------------------------------------------------------------
    def predict_next(
        self,
        obs: Sequence[float],
        action: float = 0.0,
    ) -> List[float]:
        """Predict the next observation from *obs* and *action*.

        Parameters
        ----------
        obs : sequence of float
        action : float

        Returns
        -------
        list[float]

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict_next().")
        x = np.array(list(obs) + [action], dtype=float).reshape(1, -1)
        pred = self._model.predict(x)
        return pred[0].tolist()

    # ------------------------------------------------------------------
    def avg_reward(self) -> float:
        """Return average reward across all recorded states that have one."""
        rewards = [s.reward for s in self.history_ if s.reward is not None]
        if not rewards:
            return 0.0
        return float(np.mean(rewards))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EnvironmentModel(obs_dim={self.obs_dim}, "
            f"history={len(self.history_)}, fitted={self.fitted_})"
        )
