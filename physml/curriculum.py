"""Stage 59 — CurriculumScheduler: orders training tasks from easy to hard.

A curriculum progressively exposes the learner to more difficult examples,
preventing early over-fitting to hard cases and accelerating convergence.

Supported strategies
--------------------
* ``"linear"``   — difficulty grows linearly over steps.
* ``"cosine"``   — slow start, fast middle, slow end (cosine annealing).
* ``"step"``     — difficulty jumps at fixed milestones.
* ``"adaptive"`` — difficulty increases only when the agent's running
                   accuracy exceeds a threshold.

Key class
---------
:class:`CurriculumScheduler`

Usage
-----
::

    from physml.curriculum import CurriculumScheduler

    sched = CurriculumScheduler(strategy="linear", max_difficulty=1.0,
                                total_steps=1000)
    for step in range(1000):
        diff = sched.step()
        batch = dataset.sample(difficulty_threshold=diff)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any


class CurriculumScheduler:
    """Schedule training difficulty from easy to hard.

    Parameters
    ----------
    strategy : str, default "linear"
        One of ``"linear"``, ``"cosine"``, ``"step"``, ``"adaptive"``.
    min_difficulty : float, default 0.0
        Starting difficulty level (0 = easiest).
    max_difficulty : float, default 1.0
        Maximum difficulty level (1 = hardest).
    total_steps : int, default 1000
        Total number of training steps (used by linear/cosine schedules).
    milestones : list[int] | None, default None
        Step indices at which difficulty increases (``"step"`` strategy).
    milestone_factor : float, default 0.25
        Fraction of the difficulty range added at each milestone.
    adaptive_threshold : float, default 0.80
        Accuracy threshold above which difficulty is advanced
        (``"adaptive"`` strategy).
    adaptive_window : int, default 10
        Window size for the running accuracy average (``"adaptive"``).
    adaptive_increment : float, default 0.05
        Amount to increase difficulty per advancement (``"adaptive"``).
    """

    _VALID_STRATEGIES = {"linear", "cosine", "step", "adaptive"}

    def __init__(
        self,
        strategy: str = "linear",
        min_difficulty: float = 0.0,
        max_difficulty: float = 1.0,
        total_steps: int = 1000,
        milestones: list[int] | None = None,
        milestone_factor: float = 0.25,
        adaptive_threshold: float = 0.80,
        adaptive_window: int = 10,
        adaptive_increment: float = 0.05,
    ) -> None:
        if strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(self._VALID_STRATEGIES)}, got {strategy!r}"
            )
        if min_difficulty > max_difficulty:
            raise ValueError("min_difficulty must be <= max_difficulty")

        self.strategy = strategy
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.total_steps = max(1, total_steps)
        self.milestones = sorted(milestones or [])
        self.milestone_factor = milestone_factor
        self.adaptive_threshold = adaptive_threshold
        self.adaptive_window = max(1, adaptive_window)
        self.adaptive_increment = adaptive_increment

        self._current_step: int = 0
        self._difficulty: float = min_difficulty
        self._milestone_idx: int = 0
        self._acc_window: deque[float] = deque(maxlen=self.adaptive_window)
        self._history: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, accuracy: float | None = None) -> float:
        """Advance the curriculum by one step and return the current difficulty.

        Parameters
        ----------
        accuracy : float | None
            Current model accuracy; required only for ``"adaptive"`` strategy.

        Returns
        -------
        float
            Current difficulty in ``[min_difficulty, max_difficulty]``.
        """
        self._current_step += 1
        t = min(self._current_step, self.total_steps)
        span = self.max_difficulty - self.min_difficulty

        if self.strategy == "linear":
            frac = t / self.total_steps
            self._difficulty = self.min_difficulty + span * frac

        elif self.strategy == "cosine":
            frac = 1.0 - math.cos(math.pi * t / self.total_steps)
            frac /= 2.0
            self._difficulty = self.min_difficulty + span * frac

        elif self.strategy == "step":
            while (
                self._milestone_idx < len(self.milestones)
                and t > self.milestones[self._milestone_idx]
            ):
                self._difficulty = min(
                    self.max_difficulty,
                    self._difficulty + span * self.milestone_factor,
                )
                self._milestone_idx += 1

        elif self.strategy == "adaptive":
            if accuracy is not None:
                self._acc_window.append(float(accuracy))
            if len(self._acc_window) >= self.adaptive_window:
                avg = sum(self._acc_window) / len(self._acc_window)
                if avg >= self.adaptive_threshold:
                    self._difficulty = min(
                        self.max_difficulty,
                        self._difficulty + self.adaptive_increment,
                    )
                    self._acc_window.clear()

        self._difficulty = float(
            max(self.min_difficulty, min(self.max_difficulty, self._difficulty))
        )
        self._history.append(self._difficulty)
        return self._difficulty

    def reset(self) -> None:
        """Reset to initial state."""
        self._current_step = 0
        self._difficulty = self.min_difficulty
        self._milestone_idx = 0
        self._acc_window.clear()
        self._history.clear()

    @property
    def current_difficulty(self) -> float:
        """Current difficulty level without advancing the schedule."""
        return self._difficulty

    @property
    def current_step(self) -> int:
        return self._current_step

    def progress(self) -> float:
        """Fraction of schedule completed (0.0 → 1.0)."""
        return min(1.0, self._current_step / self.total_steps)

    def history(self) -> list[float]:
        """Difficulty value at every step taken so far."""
        return list(self._history)

    def filter_by_difficulty(
        self,
        difficulties: Any,
        *,
        threshold: float | None = None,
    ) -> Any:
        """Return a boolean mask selecting samples at or below the threshold.

        Parameters
        ----------
        difficulties : array-like of float
            Per-sample difficulty scores.
        threshold : float | None
            Maximum difficulty to include.  Defaults to
            ``self.current_difficulty``.

        Returns
        -------
        numpy ndarray of bool
        """
        import numpy as np
        diff_arr = np.asarray(difficulties, dtype=float)
        thr = threshold if threshold is not None else self._difficulty
        return diff_arr <= thr

    def __repr__(self) -> str:
        return (
            f"CurriculumScheduler(strategy={self.strategy!r}, "
            f"difficulty={self._difficulty:.3f}/{self.max_difficulty}, "
            f"step={self._current_step}/{self.total_steps})"
        )
