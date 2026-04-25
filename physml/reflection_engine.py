"""Stage 94 — ReflectionEngine: self-evaluation and improvement loop.

Enables the agent to periodically review its recent performance,
identify weaknesses, generate corrective insights, and log them as
structured reflections.

Classes
-------
Reflection
    A single self-evaluation record.
ReflectionEngine
    Analyses recent episode metrics and produces actionable reflections.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any, Dict, List


@dataclass
class Reflection:
    """One self-evaluation entry.

    Attributes
    ----------
    timestamp : float
        Unix time when the reflection was generated.
    window : int
        Number of episodes that were analysed.
    avg_reward : float
        Mean reward over the analysis window.
    std_reward : float
        Reward standard deviation (0.0 for single-episode windows).
    trend : str
        ``"improving"``, ``"declining"``, or ``"stable"``.
    insights : list[str]
        Actionable observations produced by the engine.
    metadata : dict
        Extra engine-specific data.
    """

    timestamp: float
    window: int
    avg_reward: float
    std_reward: float
    trend: str
    insights: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReflectionEngine:
    """Produces self-evaluation reflections from episode reward history.

    Parameters
    ----------
    window : int
        Number of most-recent reward samples to analyse per call.
    improve_threshold : float
        Minimum relative improvement (fraction) needed to flag trend as
        ``"improving"``.
    decline_threshold : float
        Maximum relative decline (fraction) before trend becomes
        ``"declining"``.

    Attributes
    ----------
    history_ : list[float]
        All reward values appended so far.
    reflections_ : list[Reflection]
        All reflections generated so far.
    """

    def __init__(
        self,
        window: int = 10,
        improve_threshold: float = 0.05,
        decline_threshold: float = -0.05,
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.improve_threshold = improve_threshold
        self.decline_threshold = decline_threshold
        self.history_: List[float] = []
        self.reflections_: List[Reflection] = []

    # ------------------------------------------------------------------
    def log_reward(self, reward: float) -> None:
        """Append a reward sample from a completed episode."""
        self.history_.append(reward)

    def log_rewards(self, rewards: List[float]) -> None:
        """Append multiple reward samples."""
        self.history_.extend(rewards)

    # ------------------------------------------------------------------
    def reflect(self) -> Reflection:
        """Analyse the latest *window* rewards and generate a reflection.

        Returns
        -------
        Reflection
            The newly created reflection, also appended to
            :attr:`reflections_`.

        Raises
        ------
        RuntimeError
            If no rewards have been logged.
        """
        if not self.history_:
            raise RuntimeError("No reward history to reflect on.")

        recent = self.history_[-self.window :]
        avg = mean(recent)
        std = stdev(recent) if len(recent) > 1 else 0.0

        trend = self._compute_trend(recent)
        insights = self._generate_insights(recent, avg, std, trend)

        r = Reflection(
            timestamp=time.time(),
            window=len(recent),
            avg_reward=avg,
            std_reward=std,
            trend=trend,
            insights=insights,
        )
        self.reflections_.append(r)
        return r

    # ------------------------------------------------------------------
    def _compute_trend(self, recent: List[float]) -> str:
        if len(recent) < 2:
            return "stable"
        half = max(1, len(recent) // 2)
        first_half = mean(recent[:half])
        second_half = mean(recent[half:])
        if first_half == 0.0:
            return "stable"
        change = (second_half - first_half) / abs(first_half)
        if change >= self.improve_threshold:
            return "improving"
        if change <= self.decline_threshold:
            return "declining"
        return "stable"

    def _generate_insights(
        self, recent: List[float], avg: float, std: float, trend: str
    ) -> List[str]:
        insights: List[str] = []
        insights.append(f"Average reward over last {len(recent)} episodes: {avg:.4f}.")
        if trend == "improving":
            insights.append("Performance is improving — maintain current strategy.")
        elif trend == "declining":
            insights.append(
                "Performance is declining — consider exploration or hyperparameter tuning."
            )
        else:
            insights.append("Performance is stable.")
        if std > abs(avg) * 0.5 and avg != 0.0:
            insights.append(
                "High reward variance detected — environment may be stochastic."
            )
        min_r = min(recent)
        if min_r < 0:
            insights.append(
                f"Negative rewards observed (min={min_r:.4f}) — check penalty conditions."
            )
        return insights

    def summary(self) -> Dict[str, Any]:
        """Return a dict summary of all reflections."""
        if not self.reflections_:
            return {"total_reflections": 0}
        latest = self.reflections_[-1]
        return {
            "total_reflections": len(self.reflections_),
            "latest_avg_reward": latest.avg_reward,
            "latest_trend": latest.trend,
            "total_episodes_logged": len(self.history_),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ReflectionEngine(window={self.window}, "
            f"reflections={len(self.reflections_)})"
        )
