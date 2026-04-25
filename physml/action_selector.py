"""Stage 105 — ActionSelector: intelligent multi-strategy action selection.

Wraps a :class:`~physml.policy_optimizer.PolicyOptimizer` and a
:class:`~physml.value_estimator.ValueEstimator` to provide a unified
action-selection interface supporting ε-greedy, softmax (Boltzmann), and
UCB1 exploration strategies.

Classes
-------
SelectionResult
    Outcome of one action-selection call.
ActionSelector
    Multi-strategy action-selection module for autonomous agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# SelectionResult
# ---------------------------------------------------------------------------


@dataclass
class SelectionResult:
    """Result of one action-selection call.

    Attributes
    ----------
    action : int
        Selected action index.
    strategy : str
        Exploration strategy used (``"greedy"``, ``"epsilon_greedy"``,
        ``"softmax"``, or ``"ucb"``).
    action_probs : list of float
        Probability distribution over all actions at decision time.
    step : int
        Global selection step counter at the time of selection.
    metadata : dict
        Optional supplementary information.
    """

    action: int
    strategy: str
    action_probs: List[float]
    step: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ActionSelector
# ---------------------------------------------------------------------------


class ActionSelector:
    """Multi-strategy action selector for autonomous agents.

    Maintains per-action visit counts and estimated values for UCB1,
    and delegates softmax / ε-greedy selection to raw logit scores.

    Parameters
    ----------
    n_actions : int
        Number of discrete actions.
    strategy : {"epsilon_greedy", "softmax", "ucb", "greedy"}, default "epsilon_greedy"
        Default exploration strategy.
    epsilon : float, default 0.1
        Exploration probability for ε-greedy.
    temperature : float, default 1.0
        Softmax temperature (lower → more greedy).
    ucb_c : float, default 1.4
        Exploration constant for UCB1 (√(2 ln N / n_a)).
    random_state : int or None, default None
        Seed for reproducibility.
    """

    _STRATEGIES = frozenset({"epsilon_greedy", "softmax", "ucb", "greedy"})

    def __init__(
        self,
        n_actions: int,
        strategy: str = "epsilon_greedy",
        epsilon: float = 0.1,
        temperature: float = 1.0,
        ucb_c: float = 1.4,
        random_state: Optional[int] = None,
    ) -> None:
        if strategy not in self._STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(self._STRATEGIES)}")
        if n_actions < 1:
            raise ValueError("n_actions must be >= 1")
        self.n_actions = n_actions
        self.strategy = strategy
        self.epsilon = epsilon
        self.temperature = temperature
        self.ucb_c = ucb_c
        self._rng = np.random.default_rng(random_state)
        self._counts: np.ndarray = np.zeros(n_actions, dtype=int)
        self._values: np.ndarray = np.zeros(n_actions, dtype=float)
        self._total_steps: int = 0
        self._history: List[SelectionResult] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _softmax(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        scaled = logits / max(temperature, 1e-9)
        shifted = scaled - scaled.max()
        exp = np.exp(shifted)
        return exp / (exp.sum() + 1e-9)

    def _ucb_scores(self) -> np.ndarray:
        n_total = self._total_steps + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            bonus = self.ucb_c * np.sqrt(np.log(n_total) / np.where(self._counts > 0, self._counts, 1))
        return self._values + bonus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        logits: Optional[np.ndarray] = None,
        strategy: Optional[str] = None,
    ) -> SelectionResult:
        """Select an action.

        Parameters
        ----------
        logits : array-like of shape (n_actions,), optional
            Raw action scores from a policy network.  Defaults to the
            internal UCB value estimates when not provided.
        strategy : str, optional
            Override the instance-level strategy for this call.

        Returns
        -------
        SelectionResult
        """
        strat = strategy or self.strategy
        if logits is not None:
            q = np.asarray(logits, dtype=float).ravel()[: self.n_actions]
        else:
            q = self._values.copy()

        probs: np.ndarray

        if strat == "greedy":
            probs = np.zeros(self.n_actions)
            probs[int(np.argmax(q))] = 1.0
            action = int(np.argmax(q))

        elif strat == "epsilon_greedy":
            if self._rng.random() < self.epsilon:
                action = int(self._rng.integers(0, self.n_actions))
            else:
                action = int(np.argmax(q))
            probs = np.full(self.n_actions, self.epsilon / self.n_actions)
            probs[int(np.argmax(q))] += 1.0 - self.epsilon

        elif strat == "softmax":
            probs = self._softmax(q, self.temperature)
            action = int(self._rng.choice(self.n_actions, p=probs))

        else:  # ucb
            ucb = self._ucb_scores()
            action = int(np.argmax(ucb))
            probs = self._softmax(ucb, 1.0)

        self._counts[action] += 1
        self._total_steps += 1

        result = SelectionResult(
            action=action,
            strategy=strat,
            action_probs=probs.tolist(),
            step=self._total_steps,
        )
        self._history.append(result)
        return result

    def update_value(self, action: int, reward: float) -> None:
        """Update the running mean value estimate for *action*.

        Uses an incremental mean update: V(a) ← V(a) + (r − V(a)) / N(a).
        """
        self._counts[action] = max(self._counts[action], 1)
        self._values[action] += (reward - self._values[action]) / self._counts[action]

    @property
    def selection_history(self) -> List[SelectionResult]:
        """All recorded selection results."""
        return list(self._history)

    @property
    def action_counts(self) -> np.ndarray:
        """Per-action visit counts (copy)."""
        return self._counts.copy()

    def reset(self) -> None:
        """Reset counts, values, and history."""
        self._counts = np.zeros(self.n_actions, dtype=int)
        self._values = np.zeros(self.n_actions, dtype=float)
        self._total_steps = 0
        self._history = []
