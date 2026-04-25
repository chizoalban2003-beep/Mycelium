"""Stage 71 — SelfHealer: anomaly-triggered checkpoint rollback and curriculum reset.

When :class:`~physml.anomaly.AnomalyGuard` flags too many corrupted inputs
or detects model collapse (accuracy near chance), the agent automatically:

1. Rolls back to the most recent :class:`~physml.checkpoint.AgentCheckpoint`.
2. Logs the incident with timestamp, anomaly rate, and trigger reason.
3. Optionally resets a :class:`~physml.curriculum.CurriculumScheduler` to an
   easier difficulty level so the recovered agent can rebuild competence
   gradually.

Classes
-------
SelfHealer
    Wraps an agent with anomaly-triggered self-healing.
HealingIncident
    Record of a single healing event.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class HealingIncident:
    """Record of a single autonomous healing event.

    Attributes
    ----------
    timestamp : float
        Unix timestamp of the incident.
    trigger : str
        Reason the heal was triggered (``"anomaly"`` or ``"collapse"``).
    anomaly_rate : float
        Fraction of inputs flagged as anomalous.
    accuracy_before : float
        Accuracy estimate that caused the trigger (NaN if unavailable).
    checkpoint_path : str or None
        Path of the checkpoint the agent was rolled back to.
    curriculum_reset : bool
        Whether the curriculum scheduler was reset to easier difficulty.
    """

    timestamp: float
    trigger: str
    anomaly_rate: float
    accuracy_before: float
    checkpoint_path: str | None
    curriculum_reset: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "trigger": self.trigger,
            "anomaly_rate": round(self.anomaly_rate, 4),
            "accuracy_before": (
                round(self.accuracy_before, 4)
                if not np.isnan(self.accuracy_before)
                else None
            ),
            "checkpoint_path": self.checkpoint_path,
            "curriculum_reset": self.curriculum_reset,
        }


class SelfHealer:
    """Autonomous self-healing wrapper around an agent.

    Monitors incoming data with an :class:`~physml.anomaly.AnomalyGuard`.
    If the anomaly rate exceeds *anomaly_threshold* or accuracy dips below
    *collapse_threshold* (model collapse), the agent is rolled back to the
    last saved :class:`~physml.checkpoint.AgentCheckpoint` and the optional
    curriculum scheduler is reset to difficulty ``reset_difficulty``.

    Parameters
    ----------
    agent : Any
        The agent to protect.  Must expose ``fit(X, y)`` and ``predict(X)``.
    checkpoint_path : str or Path
        Where to save / restore checkpoints.
    anomaly_threshold : float, default 0.3
        Fraction of anomalous inputs that triggers healing.
    collapse_threshold : float, default 0.55
        Accuracy below this level is considered "model collapse".
    anomaly_contamination : float, default 0.1
        Expected anomaly contamination for the AnomalyGuard.
    curriculum : CurriculumScheduler or None
        Optional scheduler to reset on healing.
    reset_difficulty : float, default 0.2
        Difficulty the curriculum is reset to after healing.
    auto_checkpoint : bool, default True
        If True, the healer saves a fresh checkpoint after every successful
        ``protect()`` call (when the data is clean).

    Example
    -------
    >>> from sklearn.datasets import make_classification
    >>> from physml import MyceliumAgent
    >>> from physml.self_healer import SelfHealer
    >>> import tempfile, os
    >>> X, y = make_classification(n_samples=200, n_features=8, random_state=0)
    >>> agent = MyceliumAgent()
    >>> agent.fit(X[:100], y[:100])
    >>> with tempfile.TemporaryDirectory() as d:
    ...     healer = SelfHealer(agent, os.path.join(d, "agent.ckpt"))
    ...     healer.checkpoint()
    ...     result = healer.protect(X[100:], y[100:])
    """

    def __init__(
        self,
        agent: Any,
        checkpoint_path: str | Path,
        *,
        anomaly_threshold: float = 0.3,
        collapse_threshold: float = 0.55,
        anomaly_contamination: float = 0.1,
        curriculum: Any | None = None,
        reset_difficulty: float = 0.2,
        auto_checkpoint: bool = True,
    ) -> None:
        self.agent = agent
        self.checkpoint_path = Path(checkpoint_path)
        self.anomaly_threshold = float(anomaly_threshold)
        self.collapse_threshold = float(collapse_threshold)
        self.anomaly_contamination = float(anomaly_contamination)
        self.curriculum = curriculum
        self.reset_difficulty = float(reset_difficulty)
        self.auto_checkpoint = auto_checkpoint

        self._incidents: list[HealingIncident] = []
        self._guard: Any = None
        self._guard_fitted: bool = False
        self._checkpoint_exists: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def checkpoint(self) -> Path:
        """Save the current agent state as the rollback checkpoint.

        Returns
        -------
        Path
            Path of the written checkpoint file.
        """
        from physml.checkpoint import AgentCheckpoint

        path = AgentCheckpoint.save(self.agent, self.checkpoint_path)
        self._checkpoint_exists = True
        return path

    def fit_guard(self, X_clean: Any) -> "SelfHealer":
        """Fit the AnomalyGuard on known-clean data.

        Parameters
        ----------
        X_clean : array-like
            Representative inlier data (no anomalies).

        Returns
        -------
        self
        """
        from physml.anomaly import AnomalyGuard

        X = np.asarray(X_clean, dtype=float)
        self._guard = AnomalyGuard(contamination=self.anomaly_contamination)
        self._guard.fit(X)
        self._guard_fitted = True
        return self

    def protect(
        self,
        X: Any,
        y: Any | None = None,
    ) -> dict[str, Any]:
        """Screen *X* for anomalies; heal if thresholds are breached.

        Parameters
        ----------
        X : array-like
            Input data to screen.
        y : array-like or None
            Labels (used to estimate accuracy for collapse detection).

        Returns
        -------
        dict
            ``{"healed": bool, "anomaly_rate": float, "accuracy": float|None,
               "incident": HealingIncident|None}``
        """
        X_arr = np.asarray(X, dtype=float)

        # Compute anomaly rate
        anomaly_rate = 0.0
        if self._guard_fitted and self._guard is not None:
            try:
                results = self._guard.predict(X_arr)
                anomaly_rate = sum(r.is_anomaly for r in results) / max(1, len(results))
            except Exception:
                anomaly_rate = 0.0

        # Compute accuracy
        accuracy: float = float("nan")
        if y is not None:
            try:
                from sklearn.metrics import accuracy_score

                preds = self._predict(X_arr)
                accuracy = float(accuracy_score(np.asarray(y), preds))
            except Exception:
                pass

        # Decide whether to heal
        anomaly_trigger = anomaly_rate >= self.anomaly_threshold
        collapse_trigger = (
            not np.isnan(accuracy) and accuracy < self.collapse_threshold
        )
        trigger = None
        if anomaly_trigger:
            trigger = "anomaly"
        elif collapse_trigger:
            trigger = "collapse"

        incident = None
        if trigger is not None:
            incident = self._heal(anomaly_rate, accuracy, trigger)

        # Auto-checkpoint when data is clean
        elif self.auto_checkpoint and self._checkpoint_exists:
            try:
                self.checkpoint()
            except Exception:
                pass

        return {
            "healed": trigger is not None,
            "anomaly_rate": anomaly_rate,
            "accuracy": accuracy if not np.isnan(accuracy) else None,
            "incident": incident,
        }

    def rollback(self) -> bool:
        """Manually roll back the agent to the last saved checkpoint.

        Returns
        -------
        bool
            True if rollback succeeded, False otherwise.
        """
        if not self._checkpoint_exists or not self.checkpoint_path.exists():
            return False
        try:
            from physml.checkpoint import AgentCheckpoint

            recovered = AgentCheckpoint.load(self.checkpoint_path)
            # Restore internal state of the agent in-place
            self.agent.__dict__.update(recovered.__dict__)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def incidents(self) -> list[HealingIncident]:
        """List of all healing incidents recorded."""
        return list(self._incidents)

    @property
    def n_heals(self) -> int:
        """Number of healing events triggered."""
        return len(self._incidents)

    def summary(self) -> dict[str, Any]:
        """High-level summary of the self-healer's activity."""
        return {
            "n_heals": self.n_heals,
            "guard_fitted": self._guard_fitted,
            "checkpoint_exists": self._checkpoint_exists,
            "anomaly_threshold": self.anomaly_threshold,
            "collapse_threshold": self.collapse_threshold,
            "incidents": [i.as_dict() for i in self._incidents],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _heal(
        self, anomaly_rate: float, accuracy: float, trigger: str
    ) -> HealingIncident:
        """Execute healing: rollback + optional curriculum reset."""
        ckpt_path_str: str | None = None
        rolled_back = False

        if self._checkpoint_exists:
            rolled_back = self.rollback()
            if rolled_back:
                ckpt_path_str = str(self.checkpoint_path)

        # Reset curriculum to easier difficulty
        curriculum_reset = False
        if self.curriculum is not None:
            try:
                self.curriculum.difficulty = self.reset_difficulty
                curriculum_reset = True
            except Exception:
                pass

        incident = HealingIncident(
            timestamp=time.time(),
            trigger=trigger,
            anomaly_rate=anomaly_rate,
            accuracy_before=accuracy,
            checkpoint_path=ckpt_path_str,
            curriculum_reset=curriculum_reset,
        )
        self._incidents.append(incident)
        return incident

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Unified predict helper."""
        if hasattr(self.agent, "predict"):
            return np.asarray(self.agent.predict(X))
        if hasattr(self.agent, "observe"):
            preds = []
            for row in X:
                result = self.agent.observe(row)
                pred = getattr(result, "prediction", result)
                preds.append(pred)
            return np.asarray(preds)
        raise AttributeError("Agent has neither predict() nor observe()")
