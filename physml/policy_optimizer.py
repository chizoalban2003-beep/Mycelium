"""Stage 103 — PolicyOptimizer: REINFORCE-style policy gradient optimizer.

Maintains a simple tabular or vector policy and updates it via a
Monte-Carlo policy-gradient estimate (REINFORCE) computed from episode
returns.  Designed to be lightweight and dependency-free.

Classes
-------
PolicyUpdate
    Record of one gradient-update step.
PolicyOptimizer
    Accumulates episode trajectories and applies policy-gradient updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# PolicyUpdate
# ---------------------------------------------------------------------------


@dataclass
class PolicyUpdate:
    """Record of one policy-gradient update.

    Attributes
    ----------
    episode : int
        Episode index at which the update was applied.
    mean_return : float
        Mean discounted return over the episode.
    policy_norm : float
        L2 norm of the policy parameter vector after the update.
    loss : float
        Estimated REINFORCE loss (negative log-likelihood × return).
    metadata : dict
        Optional extra information.
    """

    episode: int
    mean_return: float
    policy_norm: float
    loss: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PolicyOptimizer
# ---------------------------------------------------------------------------


class PolicyOptimizer:
    """REINFORCE policy-gradient optimizer.

    Maintains a softmax policy over a finite action space and updates it
    using the Monte-Carlo gradient estimate.

    Parameters
    ----------
    n_actions : int
        Number of discrete actions.
    state_dim : int, default 1
        Dimensionality of the state representation.  For a tabular policy
        pass ``state_dim=1`` and use scalar state indices as observations.
    learning_rate : float, default 0.01
        Step size for gradient updates.
    gamma : float, default 0.99
        Discount factor for computing returns.
    random_state : int or None, default None
        Seed for weight initialisation.
    """

    def __init__(
        self,
        n_actions: int,
        state_dim: int = 1,
        learning_rate: float = 0.01,
        gamma: float = 0.99,
        random_state: Optional[int] = None,
    ) -> None:
        if n_actions < 1:
            raise ValueError("n_actions must be >= 1")
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        rng = np.random.default_rng(random_state)
        self._theta: np.ndarray = rng.standard_normal((state_dim, n_actions)) * 0.01
        self._episode_count: int = 0
        self._update_history: List[PolicyUpdate] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        return exp / (exp.sum() + 1e-9)

    def _compute_returns(self, rewards: Sequence[float]) -> np.ndarray:
        """Compute discounted returns G_t for each time step."""
        rewards_arr = np.asarray(rewards, dtype=float)
        returns = np.zeros_like(rewards_arr)
        g = 0.0
        for t in reversed(range(len(rewards_arr))):
            g = rewards_arr[t] + self.gamma * g
            returns[t] = g
        # Normalise to reduce variance
        std = returns.std()
        if std > 1e-8:
            returns = (returns - returns.mean()) / std
        return returns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """Sample an action from the current policy.

        Parameters
        ----------
        state : array-like, shape (state_dim,)

        Returns
        -------
        int
            Sampled action index.
        """
        s = np.asarray(state, dtype=float).reshape(1, -1)
        logits = (s @ self._theta).ravel()
        probs = self._softmax(logits)
        return int(np.random.choice(self.n_actions, p=probs))

    def action_probs(self, state: np.ndarray) -> np.ndarray:
        """Return action probabilities for *state*."""
        s = np.asarray(state, dtype=float).reshape(1, -1)
        logits = (s @ self._theta).ravel()
        return self._softmax(logits)

    def update(
        self,
        states: Sequence,
        actions: Sequence[int],
        rewards: Sequence[float],
    ) -> PolicyUpdate:
        """Apply a REINFORCE gradient update from one episode.

        Parameters
        ----------
        states : sequence of array-like
            State observations at each time step.
        actions : sequence of int
            Actions taken at each time step.
        rewards : sequence of float
            Rewards received at each time step.

        Returns
        -------
        PolicyUpdate
        """
        returns = self._compute_returns(rewards)
        total_loss = 0.0
        for state, action, ret in zip(states, actions, returns):
            s = np.asarray(state, dtype=float).reshape(1, -1)
            probs = self._softmax((s @ self._theta).ravel())
            # Gradient of log π(a|s) w.r.t. θ
            grad = -np.outer(s.ravel(), (np.eye(self.n_actions)[action] - probs))
            self._theta -= self.learning_rate * ret * grad
            total_loss += float(-np.log(probs[action] + 1e-9) * ret)

        update = PolicyUpdate(
            episode=self._episode_count,
            mean_return=float(returns.mean()),
            policy_norm=float(np.linalg.norm(self._theta)),
            loss=total_loss / max(len(rewards), 1),
        )
        self._update_history.append(update)
        self._episode_count += 1
        return update

    @property
    def update_history(self) -> List[PolicyUpdate]:
        """All recorded update steps."""
        return list(self._update_history)

    def reset(self) -> None:
        """Re-initialise policy parameters and clear history."""
        rng = np.random.default_rng(None)
        self._theta = rng.standard_normal(self._theta.shape) * 0.01
        self._episode_count = 0
        self._update_history = []
