"""Stage 104 — ValueEstimator: state-value function approximation.

Approximates V(s) — the expected discounted return from state *s* — using
a linear function approximator with TD(0) updates.  Serves as the critic
component in actor-critic architectures and as a baseline for policy
gradient methods.

Classes
-------
ValueEstimate
    A single value-function query result.
ValueEstimator
    Linear TD(0) value-function approximator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# ValueEstimate
# ---------------------------------------------------------------------------


@dataclass
class ValueEstimate:
    """Result of a value-function query.

    Attributes
    ----------
    state_id : str
        Optional identifier for the queried state.
    value : float
        Estimated V(s).
    td_error : float or None
        TD error from the most recent update for this state, if available.
    metadata : dict
        Optional supplementary information.
    """

    state_id: str
    value: float
    td_error: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ValueEstimator
# ---------------------------------------------------------------------------


class ValueEstimator:
    """Linear TD(0) value-function approximator.

    Represents V(s) ≈ wᵀ φ(s) where φ(s) is a feature vector of the
    state.  Parameters are updated online with the semi-gradient TD(0) rule:

        w ← w + α · δ · φ(s)

    where δ = r + γ V(s') − V(s) is the TD error.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state feature vector.
    learning_rate : float, default 0.01
        Step size α for TD updates.
    gamma : float, default 0.99
        Discount factor γ.
    random_state : int or None, default None
        Seed for weight initialisation.
    """

    def __init__(
        self,
        state_dim: int,
        learning_rate: float = 0.01,
        gamma: float = 0.99,
        random_state: Optional[int] = None,
    ) -> None:
        if state_dim < 1:
            raise ValueError("state_dim must be >= 1")
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        rng = np.random.default_rng(random_state)
        self._w: np.ndarray = rng.standard_normal(state_dim) * 0.01
        self._n_updates: int = 0
        self._td_errors: List[float] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _phi(self, state: np.ndarray) -> np.ndarray:
        """Return the feature vector for *state* (identity by default)."""
        return np.asarray(state, dtype=float).ravel()[: self.state_dim]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        state: np.ndarray,
        state_id: str = "",
    ) -> ValueEstimate:
        """Estimate V(s) for the given state.

        Parameters
        ----------
        state : array-like, shape (state_dim,)
        state_id : str
            Optional human-readable label.

        Returns
        -------
        ValueEstimate
        """
        phi = self._phi(state)
        value = float(self._w @ phi)
        return ValueEstimate(state_id=state_id or "", value=value)

    def update(
        self,
        state: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        terminal: bool = False,
        state_id: str = "",
    ) -> ValueEstimate:
        """Perform one TD(0) update and return the updated value estimate.

        Parameters
        ----------
        state : array-like, shape (state_dim,)
            Current state s.
        reward : float
            Observed reward r.
        next_state : array-like, shape (state_dim,)
            Next state s'.
        terminal : bool, default False
            Whether s' is a terminal state (V(s') treated as 0).
        state_id : str
            Optional label for the returned :class:`ValueEstimate`.

        Returns
        -------
        ValueEstimate
            Value estimate for *state* after the update.
        """
        phi_s = self._phi(state)
        v_s = float(self._w @ phi_s)

        if terminal:
            v_sp = 0.0
        else:
            phi_sp = self._phi(next_state)
            v_sp = float(self._w @ phi_sp)

        td_error = reward + self.gamma * v_sp - v_s
        self._w += self.learning_rate * td_error * phi_s
        self._n_updates += 1
        self._td_errors.append(td_error)

        return ValueEstimate(
            state_id=state_id or "",
            value=float(self._w @ phi_s),
            td_error=td_error,
        )

    def batch_update(
        self,
        states: Sequence,
        rewards: Sequence[float],
        next_states: Sequence,
        terminals: Optional[Sequence[bool]] = None,
    ) -> List[ValueEstimate]:
        """Apply TD(0) updates for a batch of transitions.

        Returns
        -------
        list of ValueEstimate
        """
        if terminals is None:
            terminals = [False] * len(rewards)
        results = []
        for s, r, sp, done in zip(states, rewards, next_states, terminals):
            results.append(self.update(s, r, sp, terminal=done))
        return results

    @property
    def weights(self) -> np.ndarray:
        """Current parameter vector (copy)."""
        return self._w.copy()

    @property
    def n_updates(self) -> int:
        """Total number of TD update steps applied."""
        return self._n_updates

    def mean_td_error(self) -> float:
        """Mean absolute TD error over all updates."""
        if not self._td_errors:
            return 0.0
        return float(np.mean(np.abs(self._td_errors)))

    def reset(self) -> None:
        """Re-initialise weights and clear history."""
        rng = np.random.default_rng(None)
        self._w = rng.standard_normal(self.state_dim) * 0.01
        self._n_updates = 0
        self._td_errors = []
