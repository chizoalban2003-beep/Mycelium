"""Stage 58 — RewardShaper: transforms raw environment rewards into
richer training signals for the active-learning agent.

Supports:
* **Clipping** — bound rewards to [min_r, max_r].
* **Normalisation** — running mean/std Z-normalisation.
* **Potential-based shaping** — Φ(s') − γ·Φ(s) additive bonus.
* **Curiosity bonus** — small exploration reward based on prediction error.

Key class
---------
:class:`RewardShaper`

Usage
-----
::

    from physml.reward_shaper import RewardShaper

    shaper = RewardShaper(clip=(-1.0, 1.0), normalise=True, gamma=0.99)
    r_shaped = shaper.shape(raw_reward=0.7, state=obs, next_state=obs2)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class RewardShaper:
    """Transform raw scalar rewards into richer training signals.

    Parameters
    ----------
    clip : tuple[float, float] | None, default None
        If given, rewards are clipped to ``[clip[0], clip[1]]`` **after**
        all other transformations.
    normalise : bool, default False
        If *True*, rewards are Z-normalised using a running mean/variance
        estimate (Welford online algorithm).
    gamma : float, default 0.99
        Discount factor used by potential-based shaping.
    curiosity_weight : float, default 0.0
        Weight of the curiosity bonus added to the reward.  Set > 0 to
        encourage exploration.
    potential_fn : callable | None, default None
        Φ(state) function used for potential-based shaping.  Receives a
        state array and must return a float.
    """

    def __init__(
        self,
        clip: tuple[float, float] | None = None,
        normalise: bool = False,
        gamma: float = 0.99,
        curiosity_weight: float = 0.0,
        potential_fn: Any | None = None,
    ) -> None:
        self.clip = clip
        self.normalise = normalise
        self.gamma = gamma
        self.curiosity_weight = curiosity_weight
        self.potential_fn = potential_fn

        # Welford running stats
        self._n: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0   # sum of squared deviations

        # History for diagnostics
        self._raw_rewards: list[float] = []
        self._shaped_rewards: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def shape(
        self,
        raw_reward: float,
        state: Any | None = None,
        next_state: Any | None = None,
        error: float = 0.0,
    ) -> float:
        """Transform *raw_reward* and return the shaped reward.

        Parameters
        ----------
        raw_reward : float
            Original scalar reward signal.
        state : array-like | None
            Current observation (used for potential-based shaping).
        next_state : array-like | None
            Next observation (used for potential-based shaping).
        error : float, default 0.0
            Prediction error magnitude for the curiosity bonus.

        Returns
        -------
        float
            Shaped reward.
        """
        r = float(raw_reward)
        self._raw_rewards.append(r)

        # 1. Potential-based shaping  F = γ·Φ(s') − Φ(s)
        if self.potential_fn is not None:
            phi_s = float(self.potential_fn(state)) if state is not None else 0.0
            phi_s2 = float(self.potential_fn(next_state)) if next_state is not None else 0.0
            r += self.gamma * phi_s2 - phi_s

        # 2. Curiosity bonus
        if self.curiosity_weight > 0.0:
            r += self.curiosity_weight * float(error)

        # 3. Running normalisation (before clipping so stats are meaningful)
        if self.normalise:
            self._update_stats(r)
            r = self._normalise(r)

        # 4. Clipping
        if self.clip is not None:
            r = float(np.clip(r, self.clip[0], self.clip[1]))

        self._shaped_rewards.append(r)
        return r

    def reset_stats(self) -> None:
        """Reset running normalisation statistics."""
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def clear_history(self) -> None:
        """Erase stored raw/shaped reward history."""
        self._raw_rewards.clear()
        self._shaped_rewards.clear()

    @property
    def running_mean(self) -> float:
        return self._mean

    @property
    def running_std(self) -> float:
        if self._n < 2:
            return 1.0
        return math.sqrt(self._m2 / (self._n - 1))

    @property
    def n_samples(self) -> int:
        return self._n

    def history(self) -> dict[str, list[float]]:
        """Return a copy of raw and shaped reward history."""
        return {
            "raw": list(self._raw_rewards),
            "shaped": list(self._shaped_rewards),
        }

    def summary(self) -> dict[str, float]:
        """Return summary statistics over all shaped rewards seen so far."""
        shaped = np.asarray(self._shaped_rewards, dtype=float)
        if shaped.size == 0:
            return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "n": int(shaped.size),
            "mean": float(shaped.mean()),
            "std": float(shaped.std()),
            "min": float(shaped.min()),
            "max": float(shaped.max()),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_stats(self, value: float) -> None:
        """Welford online mean/variance update."""
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._m2 += delta * delta2

    def _normalise(self, value: float) -> float:
        std = self.running_std
        if std < 1e-8:
            return 0.0
        return (value - self._mean) / std

    def __repr__(self) -> str:
        return (
            f"RewardShaper(clip={self.clip}, normalise={self.normalise}, "
            f"gamma={self.gamma}, curiosity_weight={self.curiosity_weight})"
        )
