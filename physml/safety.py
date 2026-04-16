"""Stage 66 — SafetyMonitor.

Enforces safety constraints and alignment guardrails on agent actions and
predictions.  Any action or output that violates a registered constraint is
blocked and an alternative safe action is returned.

Classes
-------
SafetyConstraint
    A named predicate over (state, action) pairs.
SafetyViolation
    Record of a constraint violation.
SafetyMonitor
    Registers constraints, screens candidate actions, and logs violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class SafetyConstraint:
    """A named safety predicate.

    Parameters
    ----------
    name : str
        Human-readable constraint name.
    predicate : Callable[[np.ndarray, int], bool]
        Returns ``True`` when the (state, action) pair is **safe**.
    penalty : float
        Penalty to apply to the reward when this constraint is violated.
    """

    name: str
    predicate: Callable[[np.ndarray, int], bool]
    penalty: float = 1.0


@dataclass
class SafetyViolation:
    """Record of a safety constraint violation."""

    constraint_name: str
    state_hash: int
    action: int
    step: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "constraint": self.constraint_name,
            "action": self.action,
            "step": self.step,
        }


class SafetyMonitor:
    """Screens agent actions against registered safety constraints.

    Parameters
    ----------
    safe_action : int
        Fallback action returned when all candidates are unsafe.
    max_violations : int or None
        If set, raises ``RuntimeError`` when violation count exceeds this.

    Attributes
    ----------
    n_violations_ : int
        Total number of violations recorded since creation.
    violations_ : list[SafetyViolation]
        Full violation log.
    """

    def __init__(
        self,
        safe_action: int = 0,
        max_violations: int | None = None,
    ) -> None:
        self.safe_action = safe_action
        self.max_violations = max_violations
        self._constraints: list[SafetyConstraint] = []
        self.n_violations_ = 0
        self.violations_: list[SafetyViolation] = []
        self._step = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_constraint(self, constraint: SafetyConstraint) -> "SafetyMonitor":
        """Register a safety constraint."""
        self._constraints.append(constraint)
        return self

    def add_bound_constraint(
        self,
        name: str,
        feature_idx: int,
        low: float = -np.inf,
        high: float = np.inf,
        penalty: float = 1.0,
    ) -> "SafetyMonitor":
        """Convenience: add a feature-value bound constraint.

        Blocks any action when ``state[feature_idx]`` is outside ``[low, high]``.
        """

        def predicate(state: np.ndarray, action: int) -> bool:
            s = np.asarray(state).ravel()
            if feature_idx >= len(s):
                return True
            return bool(low <= s[feature_idx] <= high)

        self.add_constraint(
            SafetyConstraint(name=name, predicate=predicate, penalty=penalty)
        )
        return self

    # ------------------------------------------------------------------
    # Screening
    # ------------------------------------------------------------------

    def is_safe(self, state: np.ndarray, action: int) -> bool:
        """Return True if *action* satisfies all constraints in *state*."""
        s = np.asarray(state, dtype=np.float64).ravel()
        return all(c.predicate(s, action) for c in self._constraints)

    def screen(
        self,
        state: np.ndarray,
        candidate_action: int,
        alternatives: list[int] | None = None,
    ) -> int:
        """Return a safe action, logging any violation.

        Parameters
        ----------
        state : np.ndarray
        candidate_action : int
            The action the agent wants to take.
        alternatives : list[int] or None
            Other actions to consider if the candidate is unsafe.

        Returns
        -------
        int
            The candidate action if safe, otherwise the first safe
            alternative, otherwise ``self.safe_action``.
        """
        self._step += 1
        s = np.asarray(state, dtype=np.float64).ravel()

        if self.is_safe(s, candidate_action):
            return candidate_action

        # Log violation
        for c in self._constraints:
            if not c.predicate(s, candidate_action):
                v = SafetyViolation(
                    constraint_name=c.name,
                    state_hash=hash(s.tobytes()),
                    action=candidate_action,
                    step=self._step,
                )
                self.violations_.append(v)
                self.n_violations_ += 1

        if self.max_violations is not None and self.n_violations_ > self.max_violations:
            raise RuntimeError(
                f"SafetyMonitor: exceeded max_violations={self.max_violations}"
            )

        # Try alternatives
        if alternatives:
            for alt in alternatives:
                if self.is_safe(s, alt):
                    return alt

        return self.safe_action

    def penalty_for(self, state: np.ndarray, action: int) -> float:
        """Return sum of penalties from violated constraints."""
        s = np.asarray(state, dtype=np.float64).ravel()
        total = 0.0
        for c in self._constraints:
            if not c.predicate(s, action):
                total += c.penalty
        return total

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def report(self) -> dict[str, Any]:
        return {
            "n_constraints": len(self._constraints),
            "n_violations": self.n_violations_,
            "steps": self._step,
            "violation_rate": (
                round(self.n_violations_ / max(1, self._step), 4)
            ),
            "recent_violations": [
                v.as_dict() for v in self.violations_[-5:]
            ],
        }
