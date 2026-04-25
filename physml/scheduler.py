"""Stage 53 — HyperScheduler: parameter scheduling for online training.

Provides step, cosine-annealing, and exponential decay schedules that can
be attached to any numeric hyperparameter (learning rate, regularisation,
exploration epsilon, etc.).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List


class _BaseSchedule:
    """Abstract base for all schedules."""

    def __init__(self, initial_value: float) -> None:
        self.initial_value = float(initial_value)
        self._step: int = 0

    def step(self) -> float:
        """Advance the schedule by one step and return the new value."""
        self._step += 1
        return self.get_value()

    def get_value(self) -> float:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:
        self._step = 0

    @property
    def current_step(self) -> int:
        return self._step


class StepSchedule(_BaseSchedule):
    """Multiply the parameter by *gamma* every *step_size* steps.

    Parameters
    ----------
    initial_value : float
    step_size : int
        Number of steps between each decay.
    gamma : float
        Multiplicative factor (e.g. 0.5 halves the value every step_size steps).
    min_value : float
        Lower bound; value never drops below this.
    """

    def __init__(
        self,
        initial_value: float,
        step_size: int = 10,
        gamma: float = 0.5,
        min_value: float = 1e-8,
    ) -> None:
        super().__init__(initial_value)
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self.min_value = float(min_value)

    def get_value(self) -> float:
        factor = self.gamma ** (self._step // self.step_size)
        return max(self.min_value, self.initial_value * factor)


class CosineSchedule(_BaseSchedule):
    """Cosine annealing schedule between *eta_max* and *eta_min*.

    Parameters
    ----------
    initial_value : float
        Maximum value (*eta_max*).
    T_max : int
        Half-period of the cosine wave in steps.
    eta_min : float
        Minimum value.
    """

    def __init__(
        self,
        initial_value: float,
        T_max: int = 100,
        eta_min: float = 0.0,
    ) -> None:
        super().__init__(initial_value)
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)

    def get_value(self) -> float:
        cos = math.cos(math.pi * self._step / self.T_max)
        return self.eta_min + (self.initial_value - self.eta_min) * (1 + cos) / 2


class ExponentialSchedule(_BaseSchedule):
    """Exponential decay: value = initial * gamma^step.

    Parameters
    ----------
    initial_value : float
    gamma : float
        Decay rate per step.
    min_value : float
        Lower clamp.
    """

    def __init__(
        self,
        initial_value: float,
        gamma: float = 0.99,
        min_value: float = 1e-8,
    ) -> None:
        super().__init__(initial_value)
        self.gamma = float(gamma)
        self.min_value = float(min_value)

    def get_value(self) -> float:
        return max(self.min_value, self.initial_value * (self.gamma ** self._step))


class LinearSchedule(_BaseSchedule):
    """Linear interpolation from *start* to *end* over *n_steps* steps.

    After *n_steps* the value remains at *end_value*.
    """

    def __init__(
        self,
        initial_value: float,
        end_value: float = 0.0,
        n_steps: int = 100,
    ) -> None:
        super().__init__(initial_value)
        self.end_value = float(end_value)
        self.n_steps = int(n_steps)

    def get_value(self) -> float:
        t = min(self._step, self.n_steps) / self.n_steps
        return self.initial_value + t * (self.end_value - self.initial_value)


class HyperScheduler:
    """Manages one or more named schedules and applies them to an object.

    Example
    -------
    >>> sched = HyperScheduler()
    >>> sched.register("lr", StepSchedule(0.1, step_size=5, gamma=0.5))
    >>> sched.step()   # advances all schedules, returns current param dict
    {'lr': 0.1}
    >>> for _ in range(5): sched.step()
    >>> sched["lr"]
    0.05
    """

    def __init__(self) -> None:
        self._schedules: Dict[str, _BaseSchedule] = {}
        self._callbacks: List[Callable[[str, float], None]] = []

    def register(self, name: str, schedule: _BaseSchedule) -> "HyperScheduler":
        """Register a schedule under *name*."""
        self._schedules[name] = schedule
        return self

    def add_callback(self, fn: Callable[[str, float], None]) -> None:
        """Add a callback called with (name, new_value) on each step."""
        self._callbacks.append(fn)

    def step(self) -> Dict[str, float]:
        """Advance all schedules by one step. Returns current values."""
        values: Dict[str, float] = {}
        for name, sched in self._schedules.items():
            val = sched.step()
            values[name] = val
            for cb in self._callbacks:
                cb(name, val)
        return values

    def get_all(self) -> Dict[str, float]:
        """Return current values without advancing."""
        return {name: s.get_value() for name, s in self._schedules.items()}

    def __getitem__(self, name: str) -> float:
        return self._schedules[name].get_value()

    def reset_all(self) -> None:
        for s in self._schedules.values():
            s.reset()

    def history_summary(self) -> Dict[str, Any]:
        return {
            name: {"step": s.current_step, "value": s.get_value()}
            for name, s in self._schedules.items()
        }
