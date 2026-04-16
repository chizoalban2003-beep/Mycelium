"""Stage 73 — EvalScheduler: autonomous evaluation and reporting.

Schedules automated :class:`~physml.competitive_report.CompetitiveReport`
runs that compare the current model against baselines, append results to a
:class:`~physml.knowledge_graph.KnowledgeGraph`, and flag an alert condition
when the agent's competitive rank exceeds a configurable threshold.

Classes
-------
EvalScheduler
    Manages periodic competitive benchmark runs and knowledge-graph logging.
ScheduledReport
    Snapshot of one scheduled evaluation run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ScheduledReport:
    """Snapshot of one autonomous evaluation run.

    Attributes
    ----------
    report_idx : int
        Zero-based index of this scheduled report.
    timestamp : float
        Unix timestamp when the report was generated.
    mycelium_rank : int
        Rank of the Mycelium agent in the competitive leaderboard.
    mycelium_accuracy : float
        Accuracy of the Mycelium agent in this run.
    n_competitors : int
        Total number of models in the leaderboard.
    alert : bool
        True when the agent's rank exceeds *alert_rank_threshold*.
    winner : str
        Name of the top-ranked model.
    stored_in_graph : bool
        Whether this report was written to the KnowledgeGraph.
    """

    report_idx: int
    timestamp: float
    mycelium_rank: int
    mycelium_accuracy: float
    n_competitors: int
    alert: bool
    winner: str
    stored_in_graph: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "report_idx": self.report_idx,
            "timestamp": self.timestamp,
            "mycelium_rank": self.mycelium_rank,
            "mycelium_accuracy": round(self.mycelium_accuracy, 4),
            "n_competitors": self.n_competitors,
            "alert": self.alert,
            "winner": self.winner,
            "stored_in_graph": self.stored_in_graph,
        }


class EvalScheduler:
    """Autonomous evaluation and reporting scheduler.

    Runs :class:`~physml.competitive_report.CompetitiveReport` on demand
    or on a configurable cadence, appends results to a
    :class:`~physml.knowledge_graph.KnowledgeGraph`, and exposes an
    ``alert`` flag when the agent slips below *alert_rank_threshold*.

    Parameters
    ----------
    agent : Any
        The agent to benchmark.
    eval_every : int, default 1
        Number of :meth:`maybe_run` calls between actual benchmark runs.
    alert_rank_threshold : int, default 2
        Raise an alert when the agent's rank exceeds this value.
    knowledge_graph : KnowledgeGraph or None
        When provided, each report is stored as a KnowledgeNode.
    extra_baselines : dict or None
        Additional competing models for each report.
    n_samples : int, default 500
        Synthetic dataset size when no real test data is provided.
    n_features : int, default 10
        Feature count for the synthetic dataset.
    random_state : int, default 42

    Example
    -------
    >>> from physml import MyceliumAgent
    >>> from physml.eval_scheduler import EvalScheduler
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=300, n_features=8, random_state=0)
    >>> agent = MyceliumAgent()
    >>> agent.fit(X[:200], y[:200])
    >>> scheduler = EvalScheduler(agent, alert_rank_threshold=2)
    >>> report = scheduler.run(X[200:], y[200:])
    >>> print(report.alert)
    """

    def __init__(
        self,
        agent: Any,
        *,
        eval_every: int = 1,
        alert_rank_threshold: int = 2,
        knowledge_graph: Any | None = None,
        extra_baselines: dict[str, Any] | None = None,
        n_samples: int = 500,
        n_features: int = 10,
        random_state: int = 42,
    ) -> None:
        self.agent = agent
        self.eval_every = max(1, int(eval_every))
        self.alert_rank_threshold = int(alert_rank_threshold)
        self.knowledge_graph = knowledge_graph
        self.extra_baselines = extra_baselines
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

        self._history: list[ScheduledReport] = []
        self._call_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        X_test: Any | None = None,
        y_test: Any | None = None,
        *,
        dataset_name: str = "scheduled_eval",
    ) -> ScheduledReport:
        """Run a competitive benchmark report immediately.

        Parameters
        ----------
        X_test, y_test : array-like or None
            Hold-out test data.  If None, a synthetic dataset is generated.
        dataset_name : str
            Label used in the report.

        Returns
        -------
        ScheduledReport
        """
        from physml.competitive_report import CompetitiveReport

        reporter = CompetitiveReport(
            n_samples=self.n_samples,
            n_features=self.n_features,
            random_state=self.random_state,
        )

        raw = reporter.run(
            self.agent,
            X=X_test,
            y=y_test,
            dataset_name=dataset_name,
            extra_baselines=self.extra_baselines,
        )

        summary = raw["summary"]
        rank = int(summary["mycelium_rank"])
        accuracy = float(summary["mycelium_accuracy"])
        n_comp = int(summary["n_competitors"])
        winner = str(summary["winner"])
        alert = rank > self.alert_rank_threshold

        stored = False
        if self.knowledge_graph is not None:
            stored = self._store_to_graph(raw, rank, accuracy, alert)

        report = ScheduledReport(
            report_idx=len(self._history),
            timestamp=time.time(),
            mycelium_rank=rank,
            mycelium_accuracy=accuracy,
            n_competitors=n_comp,
            alert=alert,
            winner=winner,
            stored_in_graph=stored,
        )
        self._history.append(report)
        return report

    def maybe_run(
        self,
        X_test: Any | None = None,
        y_test: Any | None = None,
        *,
        dataset_name: str = "scheduled_eval",
    ) -> ScheduledReport | None:
        """Run a report only every ``eval_every`` calls; otherwise return None."""
        self._call_count += 1
        if self._call_count % self.eval_every == 0:
            return self.run(X_test, y_test, dataset_name=dataset_name)
        return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[ScheduledReport]:
        """Ordered list of all completed scheduled reports."""
        return list(self._history)

    @property
    def alert(self) -> bool:
        """True if the most recent report raised an alert."""
        if not self._history:
            return False
        return self._history[-1].alert

    def summary(self) -> dict[str, Any]:
        """High-level summary across all evaluation runs."""
        ranks = [r.mycelium_rank for r in self._history]
        accs = [r.mycelium_accuracy for r in self._history]
        n_alerts = sum(r.alert for r in self._history)
        return {
            "n_reports": len(self._history),
            "n_alerts": n_alerts,
            "best_rank_ever": min(ranks) if ranks else None,
            "latest_rank": ranks[-1] if ranks else None,
            "latest_accuracy": round(accs[-1], 4) if accs else None,
            "alert_rank_threshold": self.alert_rank_threshold,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store_to_graph(
        self, raw: dict, rank: int, accuracy: float, alert: bool
    ) -> bool:
        """Write report summary to the KnowledgeGraph."""
        try:
            import time as _time

            node_name = f"eval_report_{len(self._history)}"
            self.knowledge_graph.add_node(
                node_name,
                node_type="eval_report",
                mycelium_rank=rank,
                mycelium_accuracy=accuracy,
                alert=alert,
                timestamp=_time.time(),
            )
            return True
        except Exception:
            return False
