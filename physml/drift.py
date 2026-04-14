"""Stage 17 — Concept drift detection for streaming settings.

:class:`DriftDetector` monitors a rolling stream of prediction errors and
fires a drift event when a significant shift in the error rate is detected.
Two algorithms are provided:

* ``"page_hinkley"`` (default) — Page-Hinkley test on the cumulative error
  sum.  Lightweight, single-pass, low memory.
* ``"adwin"`` — Adaptive Windowing (simplified).  Maintains a growing window
  and splits it when the error rate in the two halves diverges significantly.

When a drift event fires, :class:`~physml.mycelium_agent.MyceliumAgent`
(via the agent) automatically resets the homeostasis state and temporarily
lowers the ask-threshold to re-explore the new distribution.

Usage
-----
::

    from physml.drift import DriftDetector

    detector = DriftDetector(algorithm="page_hinkley", threshold=50.0)
    for error in error_stream:
        if detector.update(error):
            print(f"Drift detected at step {detector.n_updates}!")
            detector.reset()

Integration in MyceliumAgent
-----------------------------
Pass ``drift_detection=True`` to :class:`~physml.mycelium_agent.MyceliumAgent`
and drift is handled automatically inside :meth:`~physml.agent.PhysicsAgent.reward`.
"""

from __future__ import annotations

from typing import Literal


class DriftDetector:
    """Online concept-drift detector.

    Parameters
    ----------
    algorithm : {"page_hinkley", "adwin"}, default "page_hinkley"
        Detection algorithm.
    threshold : float, default 50.0
        Detection threshold for the Page-Hinkley test.
        Larger values mean slower but more robust detection.
    delta : float, default 0.005
        Allowable mean increase per step (Page-Hinkley).
    adwin_delta : float, default 0.002
        Confidence parameter for the ADWIN test (smaller → more sensitive).
    min_samples : int, default 30
        Minimum number of samples before drift can be declared.
    """

    def __init__(
        self,
        algorithm: Literal["page_hinkley", "adwin"] = "page_hinkley",
        *,
        threshold: float = 50.0,
        delta: float = 0.005,
        adwin_delta: float = 0.002,
        min_samples: int = 30,
    ) -> None:
        self.algorithm = str(algorithm)
        self.threshold = float(threshold)
        self.delta = float(delta)
        self.adwin_delta = float(adwin_delta)
        self.min_samples = int(min_samples)
        self.n_updates: int = 0
        self.n_drifts: int = 0

        # Page-Hinkley state
        self._ph_sum: float = 0.0
        self._ph_mean: float = 0.0
        self._ph_min: float = 0.0

        # ADWIN state
        self._adwin_window: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, error: float) -> bool:
        """Add a new error observation and return True if drift is detected.

        Parameters
        ----------
        error : float in [0, 1]
            Prediction error for the latest sample (0 = correct, 1 = wrong).

        Returns
        -------
        bool — True if a drift event was detected on this update.
        """
        self.n_updates += 1
        drift = False
        if self.algorithm == "adwin":
            drift = self._update_adwin(float(error))
        else:
            drift = self._update_page_hinkley(float(error))

        if drift and self.n_updates >= self.min_samples:
            self.n_drifts += 1
            return True
        return False

    def reset(self) -> None:
        """Reset the detector state (call after a drift event has been handled)."""
        self._ph_sum = 0.0
        self._ph_mean = 0.0
        self._ph_min = 0.0
        self._adwin_window.clear()
        self.n_updates = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_page_hinkley(self, error: float) -> bool:
        """Page-Hinkley cumulative sum test."""
        n = self.n_updates
        # Incremental mean
        self._ph_mean += (error - self._ph_mean) / n
        # Cumulative sum
        self._ph_sum += error - self._ph_mean - self.delta
        # Track running minimum
        if self._ph_sum < self._ph_min:
            self._ph_min = self._ph_sum
        # Drift if the cumulative deviation exceeds the threshold
        return (self._ph_sum - self._ph_min) > self.threshold

    def _update_adwin(self, error: float) -> bool:
        """Simplified ADWIN: two-window cut test."""
        self._adwin_window.append(error)
        n = len(self._adwin_window)
        if n < self.min_samples:
            return False

        # Try a split at the mid-point and compare the two halves
        mid = n // 2
        w0 = self._adwin_window[:mid]
        w1 = self._adwin_window[mid:]
        if not w0 or not w1:
            return False

        mean0 = sum(w0) / len(w0)
        mean1 = sum(w1) / len(w1)

        # Hoeffding bound — uses the smaller half-size for the bound
        n_min = min(len(w0), len(w1))
        bound = (2.0 / n_min * (1.0 / self.adwin_delta)) ** 0.5

        if abs(mean0 - mean1) >= bound:
            # Drift detected — keep only the more recent window
            self._adwin_window = list(w1)
            return True
        return False
