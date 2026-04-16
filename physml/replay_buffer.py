"""Stage 52 — Prioritized Experience Replay Buffer.

Implements a ring-buffer backed replay store with TD-error based priority
weights so the online-RLHF loop can revisit high-loss transitions more often.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Transition:
    """A single SARS' experience tuple stored in the buffer."""

    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool = False
    priority: float = 1.0
    metadata: dict = field(default_factory=dict)


class ReplayBuffer:
    """Fixed-capacity ring buffer for experience replay.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, capacity: int = 10_000, seed: Optional[int] = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = int(capacity)
        self._buffer: List[Transition] = []
        self._index: int = 0
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def push(self, transition: Transition) -> None:
        """Add a transition, evicting the oldest if full."""
        if len(self._buffer) < self.capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._index] = transition
        self._index = (self._index + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        """Uniform random sample of *batch_size* transitions."""
        n = min(batch_size, len(self._buffer))
        return self._rng.sample(self._buffer, n)

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer = []
        self._index = 0

    @property
    def is_ready(self) -> bool:
        """True once buffer has at least one transition."""
        return len(self._buffer) > 0

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def push_many(self, transitions: Sequence[Transition]) -> None:
        for t in transitions:
            self.push(t)

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (rewards, priorities, dones) arrays over the full buffer."""
        rewards = np.array([t.reward for t in self._buffer], dtype=float)
        priorities = np.array([t.priority for t in self._buffer], dtype=float)
        dones = np.array([t.done for t in self._buffer], dtype=bool)
        return rewards, priorities, dones


class PrioritizedReplay(ReplayBuffer):
    """Replay buffer where sampling is proportional to |TD-error| + epsilon.

    After sampling a batch the caller should update priorities using
    :meth:`update_priorities`.

    Parameters
    ----------
    capacity : int
        Maximum transitions kept.
    alpha : float
        Priority exponent (0 = uniform, 1 = fully prioritised).
    epsilon : float
        Small constant added to avoid zero-probability transitions.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        capacity: int = 10_000,
        alpha: float = 0.6,
        epsilon: float = 1e-4,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(capacity=capacity, seed=seed)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def push(self, transition: Transition) -> None:
        """Add transition, assigning max current priority to new entries."""
        if self._buffer:
            max_p = max(t.priority for t in self._buffer)
        else:
            max_p = 1.0
        transition.priority = max_p
        super().push(transition)

    def sample(self, batch_size: int) -> List[Transition]:  # type: ignore[override]
        """Priority-weighted sampling without replacement."""
        n = min(batch_size, len(self._buffer))
        priorities = np.array(
            [(t.priority + self.epsilon) ** self.alpha for t in self._buffer],
            dtype=float,
        )
        probs = priorities / priorities.sum()
        indices = self._rng.choices(range(len(self._buffer)), weights=probs.tolist(), k=n)
        return [self._buffer[i] for i in indices]

    def update_priorities(
        self, transitions: List[Transition], td_errors: Sequence[float]
    ) -> None:
        """Update priority for each transition to |td_error|."""
        for t, err in zip(transitions, td_errors):
            t.priority = float(abs(err)) + self.epsilon

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def priority_stats(self) -> dict:
        """Return mean/max/min priority across the buffer."""
        if not self._buffer:
            return {"mean": 0.0, "max": 0.0, "min": 0.0, "size": 0}
        ps = [t.priority for t in self._buffer]
        return {
            "mean": float(np.mean(ps)),
            "max": float(np.max(ps)),
            "min": float(np.min(ps)),
            "size": len(ps),
        }
