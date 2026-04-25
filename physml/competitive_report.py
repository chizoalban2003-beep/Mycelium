"""Stage 68 — CompetitiveReport.

Automated competitive benchmark report that evaluates ``AutonomousAgent``
(backed by ``MyceliumAgent`` / CompetitiveEnsemblePredictor) against
standard sklearn baselines on a user-supplied or synthetic dataset and
produces a JSON-serialisable report.

Classes
-------
CompetitiveReport
    Runs the full benchmark suite and produces a structured report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from physml.arena import CompetitiveArena


@dataclass
class ReportEntry:
    """One benchmark entry in the competitive report."""

    dataset: str
    results: list[dict[str, Any]]
    winner: str
    mycelium_rank: int
    mycelium_accuracy: float


class CompetitiveReport:
    """Runs competitive benchmarks and produces a structured report.

    Parameters
    ----------
    test_size : float
        Fraction of data held out for evaluation.
    random_state : int
        Reproducibility seed.
    n_samples : int
        Number of samples in the auto-generated synthetic dataset (when no
        real dataset is provided).
    n_features : int
        Number of features in the auto-generated dataset.

    Example
    -------
    >>> from physml import MyceliumAgent
    >>> reporter = CompetitiveReport()
    >>> report = reporter.run(agent=MyceliumAgent())
    >>> print(report["summary"]["mycelium_wins"])
    """

    _BASELINES: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(max_iter=300, random_state=0),
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=0),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=50, random_state=0
        ),
    }

    def __init__(
        self,
        test_size: float = 0.3,
        random_state: int = 42,
        n_samples: int = 600,
        n_features: int = 10,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.n_samples = n_samples
        self.n_features = n_features

    # ------------------------------------------------------------------
    # Benchmark runner
    # ------------------------------------------------------------------

    def run(
        self,
        agent: Any,
        *,
        X: Any = None,
        y: Any = None,
        dataset_name: str = "synthetic",
        extra_baselines: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the full competitive benchmark.

        Parameters
        ----------
        agent : Any
            The Mycelium agent (or AutonomousAgent) to benchmark.
        X, y : array-like or None
            Dataset to use.  If None, a synthetic classification dataset
            is generated automatically.
        dataset_name : str
            Label used in the report.
        extra_baselines : dict or None
            Additional competing models beyond the defaults.

        Returns
        -------
        dict
            Full structured report with per-dataset results and a summary.
        """
        # Build dataset
        if X is None or y is None:
            X, y = make_classification(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_informative=max(2, self.n_features // 2),
                random_state=self.random_state,
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        baselines = dict(self._BASELINES)
        if extra_baselines:
            baselines.update(extra_baselines)

        # Run arena
        arena = CompetitiveArena(metric="accuracy")
        arena.register("MyceliumAgent", agent)
        for name, bl in baselines.items():
            arena.register(name, bl)

        results = arena.run(X_train, y_train, X_test, y_test)

        # Find Mycelium rank
        mycelium_result = next(r for r in results if r.name == "MyceliumAgent")
        entries = [r.as_dict() for r in results]

        is_competitive = mycelium_result.rank <= max(2, len(results) // 2)

        report = {
            "dataset": dataset_name,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": np.asarray(X).shape[1],
            "leaderboard": entries,
            "summary": {
                "winner": results[0].name,
                "mycelium_rank": mycelium_result.rank,
                "mycelium_accuracy": round(mycelium_result.accuracy, 4),
                "mycelium_f1": round(mycelium_result.f1, 4),
                "mycelium_roc_auc": round(mycelium_result.roc_auc, 4),
                "n_competitors": len(results),
                "is_competitive": is_competitive,
                "competitive_threshold": "top 50%",
            },
            "verdict": (
                "✅ MyceliumAgent is COMPETITIVE — ranks in the top half."
                if is_competitive
                else "⚠️  MyceliumAgent is below top-50% — further tuning recommended."
            ),
        }

        return report

    def print_report(self, report: dict[str, Any]) -> None:
        """Pretty-print the competitive report to stdout."""
        print(f"\n{'='*60}")
        print(f"  COMPETITIVE BENCHMARK REPORT — {report['dataset']}")
        print(f"{'='*60}")
        print(
            f"  Dataset: {report['n_train']} train / {report['n_test']} test  "
            f"({report['n_features']} features)"
        )
        print()
        print(f"  {'Rank':<6} {'Name':<30} {'Acc':>7} {'F1':>7} {'AUC':>7}")
        print(f"  {'-'*6} {'-'*30} {'-'*7} {'-'*7} {'-'*7}")
        for row in report["leaderboard"]:
            print(
                f"  {row['rank']:<6} {row['name']:<30} "
                f"{row['accuracy']:>7.4f} {row['f1']:>7.4f} {row['roc_auc']:>7.4f}"
            )
        print()
        print(f"  {report['verdict']}")
        print(f"{'='*60}\n")
