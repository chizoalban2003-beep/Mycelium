"""Stage 64 — CompetitiveArena.

Head-to-head evaluation harness.  The arena pits two or more agents against
each other on the same dataset splits and returns a structured leaderboard.

Classes
-------
ArenaResult
    Leaderboard row for a single competitor.
CompetitiveArena
    Runs multiple agents on the same benchmark and declares a winner.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class _Competitor(Protocol):
    """Minimal interface expected from each competitor."""

    def fit(self, X: Any, y: Any) -> Any: ...
    def predict(self, X: Any) -> Any: ...


@dataclass
class ArenaResult:
    """Performance summary for one competitor in the arena."""

    name: str
    accuracy: float
    f1: float
    roc_auc: float
    fit_time_s: float
    predict_time_s: float
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "accuracy": round(self.accuracy, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "fit_time_s": round(self.fit_time_s, 4),
            "predict_time_s": round(self.predict_time_s, 4),
            "rank": self.rank,
        }


class CompetitiveArena:
    """Benchmark harness for head-to-head agent comparison.

    Parameters
    ----------
    metric : str
        Primary ranking metric: ``"accuracy"`` | ``"f1"`` | ``"roc_auc"``.

    Example
    -------
    >>> arena = CompetitiveArena()
    >>> arena.register("agent_a", agent_a)
    >>> arena.register("agent_b", agent_b)
    >>> results = arena.run(X_train, y_train, X_test, y_test)
    >>> print(results[0].name)  # winner
    """

    def __init__(self, metric: str = "accuracy") -> None:
        self.metric = metric
        self._competitors: list[tuple[str, _Competitor]] = []

    def register(self, name: str, agent: _Competitor) -> "CompetitiveArena":
        """Register a competitor agent."""
        self._competitors.append((name, agent))
        return self

    def run(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        *,
        average: str = "macro",
    ) -> list[ArenaResult]:
        """Run all registered competitors on the given split.

        Parameters
        ----------
        X_train, y_train : training data.
        X_test, y_test : evaluation data.
        average : str
            Averaging strategy for multi-class F1 / ROC-AUC.

        Returns
        -------
        list[ArenaResult]
            Ranked list (best first) of competitor results.
        """
        y_test_arr = np.asarray(y_test)
        classes = np.unique(y_test_arr)
        n_classes = len(classes)
        results: list[ArenaResult] = []

        for name, agent in self._competitors:
            # Fit
            t0 = time.perf_counter()
            agent.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0

            # Predict
            t0 = time.perf_counter()
            y_pred = np.asarray(agent.predict(X_test))
            pred_time = time.perf_counter() - t0

            acc = float(accuracy_score(y_test_arr, y_pred))
            f1 = float(f1_score(y_test_arr, y_pred, average=average, zero_division=0))

            # ROC-AUC requires probability scores; fall back if unavailable
            roc = 0.0
            try:
                if hasattr(agent, "predict_proba"):
                    proba = np.asarray(agent.predict_proba(X_test))
                    if n_classes == 2:
                        roc = float(roc_auc_score(y_test_arr, proba[:, 1]))
                    else:
                        roc = float(
                            roc_auc_score(
                                y_test_arr,
                                proba,
                                multi_class="ovr",
                                average=average,
                            )
                        )
            except Exception:
                roc = acc  # degrade gracefully

            results.append(
                ArenaResult(
                    name=name,
                    accuracy=acc,
                    f1=f1,
                    roc_auc=roc,
                    fit_time_s=fit_time,
                    predict_time_s=pred_time,
                )
            )

        # Rank by primary metric (descending)
        results.sort(key=lambda r: getattr(r, self.metric), reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    def leaderboard(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
    ) -> list[dict[str, Any]]:
        """Convenience wrapper returning plain dicts."""
        return [r.as_dict() for r in self.run(X_train, y_train, X_test, y_test)]
