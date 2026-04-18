"""Stage 118 — ProactiveAdvisor: drift/accuracy monitoring + proactive alerts.

Proactively monitors agent metrics (drift, accuracy drops, anomalies) and
generates unsolicited advice to the user.  Integrates with
:class:`~physml.drift.DriftDetector`, :class:`~physml.anomaly.AnomalyGuard`,
and :class:`~physml.reflection_engine.ReflectionEngine`.

Advice objects contain a severity level, human-readable message, and a
recommended action.

Usage
-----
::

    from physml.proactive_advisor import ProactiveAdvisor

    advisor = ProactiveAdvisor(agent=mycelium_agent)
    advices = advisor.check()
    for a in advices:
        print(a.severity, a.message, a.action)

    advisor.enable_background(interval=300)   # check every 5 min
    advisor.disable_background()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


@dataclass
class Advice:
    """A single piece of proactive advice.

    Attributes
    ----------
    severity : str
        ``"info"``, ``"warning"``, or ``"critical"``.
    message : str
        Human-readable description.
    action : str
        Recommended action.
    timestamp : float
        Unix time when the advice was generated.
    source : str
        Which sub-system generated the advice.
    """

    severity: str
    message: str
    action: str
    timestamp: float = field(default_factory=time.time)
    source: str = "advisor"


class ProactiveAdvisor:
    """Proactive recommendation engine.

    Parameters
    ----------
    agent : any, optional
        The Mycelium agent to monitor.  Should expose:
        ``drift_detector``, ``anomaly_guard``, ``reflection_engine``,
        ``self_eval_result``, ``feedback_buffer``.
    accuracy_drop_threshold : float, default 0.1
        Relative accuracy drop that triggers a warning (10 %).
    callbacks : list of callable, optional
        Additional check functions: ``fn() → list[Advice]``.
    """

    def __init__(
        self,
        agent: Any = None,
        accuracy_drop_threshold: float = 0.10,
        callbacks: Optional[List[Callable[[], List[Advice]]]] = None,
    ) -> None:
        self.agent = agent
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self._callbacks: List[Callable[[], List[Advice]]] = list(callbacks or [])
        self._history: List[Advice] = []
        self._last_accuracy: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> List[Advice]:
        """Run all checks and return a list of :class:`Advice` objects.

        Returns
        -------
        list[Advice]
        """
        advices: List[Advice] = []
        advices.extend(self._check_drift())
        advices.extend(self._check_anomaly())
        advices.extend(self._check_accuracy())
        advices.extend(self._check_reflection())
        for cb in self._callbacks:
            try:
                extra = cb()
                if extra:
                    advices.extend(extra)
            except Exception as e:
                _logger.warning("ProactiveAdvisor: callback error: %s", e)

        self._history.extend(advices)
        # Keep last 200
        if len(self._history) > 200:
            self._history = self._history[-200:]

        return advices

    def add_callback(self, fn: Callable[[], List[Advice]]) -> None:
        """Register an extra check callback.

        Parameters
        ----------
        fn : callable
            Zero-argument function returning a list of :class:`Advice`.
        """
        self._callbacks.append(fn)

    def enable_background(self, interval: float = 300.0) -> None:
        """Start background check loop.

        Parameters
        ----------
        interval : float, default 300
            Seconds between checks.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, kwargs={"interval": interval}, daemon=True
        )
        self._thread.start()
        _logger.info("ProactiveAdvisor: background enabled (interval=%.0fs)", interval)

    def disable_background(self) -> None:
        """Stop the background check loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        _logger.info("ProactiveAdvisor: background disabled")

    @property
    def history(self) -> List[Advice]:
        """All advice generated so far."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Check implementations
    # ------------------------------------------------------------------

    def _check_drift(self) -> List[Advice]:
        if self.agent is None:
            return []
        dd = getattr(self.agent, "drift_detector", None)
        if dd is None:
            return []
        try:
            # DriftDetector exposes n_drifts or drift_detected
            n_drifts = getattr(dd, "n_drifts", 0)
            if n_drifts > 0:
                return [
                    Advice(
                        severity="warning",
                        message=f"Concept drift detected ({n_drifts} events).",
                        action="retrain",
                        source="drift_detector",
                    )
                ]
        except Exception as e:
            _logger.warning("ProactiveAdvisor: drift check failed: %s", e)
        return []

    def _check_anomaly(self) -> List[Advice]:
        if self.agent is None:
            return []
        ag = getattr(self.agent, "anomaly_guard", None)
        if ag is None:
            return []
        try:
            recent = getattr(ag, "last_result", None)
            if recent is not None and getattr(recent, "is_anomaly", False):
                return [
                    Advice(
                        severity="warning",
                        message="Anomaly detected in recent input data.",
                        action="inspect_data",
                        source="anomaly_guard",
                    )
                ]
        except Exception as e:
            _logger.warning("ProactiveAdvisor: anomaly check failed: %s", e)
        return []

    def _check_accuracy(self) -> List[Advice]:
        if self.agent is None:
            return []
        # Look for self_eval_result or recent accuracy
        sr = getattr(self.agent, "self_eval_result", None)
        if sr is None:
            return []
        try:
            acc = getattr(sr, "accuracy", None) or getattr(sr, "score", None)
            if acc is None:
                return []
            advices: List[Advice] = []
            if self._last_accuracy is not None:
                drop = self._last_accuracy - acc
                rel_drop = drop / max(self._last_accuracy, 1e-9)
                if rel_drop > self.accuracy_drop_threshold:
                    pct = rel_drop * 100
                    advices.append(
                        Advice(
                            severity="warning",
                            message=f"Model accuracy dropped {pct:.1f}% (from {self._last_accuracy:.3f} to {acc:.3f}).",
                            action="retrain",
                            source="accuracy_monitor",
                        )
                    )
            self._last_accuracy = float(acc)
            return advices
        except Exception as e:
            _logger.warning("ProactiveAdvisor: accuracy check failed: %s", e)
        return []

    def _check_reflection(self) -> List[Advice]:
        if self.agent is None:
            return []
        re_ = getattr(self.agent, "reflection_engine", None)
        if re_ is None:
            return []
        try:
            reflections = getattr(re_, "reflections", [])
            if reflections:
                last = reflections[-1]
                trend = getattr(last, "trend", None)
                if trend == "declining":
                    return [
                        Advice(
                            severity="warning",
                            message="Agent performance trend is declining.",
                            action="self_improve",
                            source="reflection_engine",
                        )
                    ]
        except Exception as e:
            _logger.warning("ProactiveAdvisor: reflection check failed: %s", e)
        return []

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _loop(self, interval: float) -> None:
        while not self._stop_event.is_set():
            try:
                self.check()
            except Exception as e:
                _logger.warning("ProactiveAdvisor: background check error: %s", e)
            self._stop_event.wait(timeout=interval)

    def __repr__(self) -> str:
        return (
            f"ProactiveAdvisor("
            f"n_history={len(self._history)}, "
            f"background={'running' if self._thread and self._thread.is_alive() else 'off'})"
        )
