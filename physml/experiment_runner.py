"""physml.experiment_runner — Benchmark the physics engine on synthetic tabular data.

This module provides :class:`ExperimentRunner`, a lightweight harness that:

1. Generates synthetic regression and classification datasets.
2. Fits :class:`~physml.estimator.PhysicsPredictor` (physics backend) with
   configurable hyperparameter grids.
3. Records accuracy / R² scores, fit times, and per-experiment metadata.
4. Persists results as JSON-lines (``experiments.jsonl``) so they survive
   restarts and accumulate over time.
5. Returns a :class:`BenchmarkSummary` with the best configuration found.

Usage::

    from physml.experiment_runner import ExperimentRunner

    runner = ExperimentRunner(results_dir="~/.mycelium/experiments")
    summary = runner.run(task="regression", n_samples=200, n_features=5)
    print(summary.best_config)
    print(summary.best_score)
    print(summary.n_experiments)

Command-line (calls :func:`cli_main`)::

    python -m physml.experiment_runner --task classification --samples 150
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from physml._log import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """A single experiment run.

    Attributes
    ----------
    task : str
        ``"regression"`` or ``"classification"``.
    config : dict
        Hyperparameter configuration used.
    score : float
        R² (regression) or accuracy (classification) on the held-out set.
    fit_time_s : float
        Seconds to fit the model.
    predict_time_s : float
        Seconds to predict on the test set.
    n_train : int
        Number of training samples.
    n_test : int
        Number of test samples.
    n_features : int
        Number of input features.
    timestamp : float
        Unix timestamp of the run.
    error : str or None
        Error message if the run failed.
    """

    task: str
    config: Dict[str, Any]
    score: float
    fit_time_s: float
    predict_time_s: float
    n_train: int
    n_test: int
    n_features: int
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentResult":
        return cls(**d)


@dataclass
class BenchmarkSummary:
    """Summary of a benchmark run.

    Attributes
    ----------
    task : str
    n_experiments : int
        Number of configurations tried.
    best_score : float
        Best score across all experiments.
    best_config : dict
        Hyperparameters of the best run.
    mean_score : float
    std_score : float
    total_time_s : float
    results : list of ExperimentResult
    """

    task: str
    n_experiments: int
    best_score: float
    best_config: Dict[str, Any]
    mean_score: float
    std_score: float
    total_time_s: float
    results: List[ExperimentResult] = field(default_factory=list)

    def __str__(self) -> str:
        metric = "R²" if self.task == "regression" else "Accuracy"
        return (
            f"BenchmarkSummary(task={self.task}, experiments={self.n_experiments}, "
            f"best_{metric}={self.best_score:.4f}, "
            f"mean_{metric}={self.mean_score:.4f}±{self.std_score:.4f}, "
            f"time={self.total_time_s:.1f}s)"
        )


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------


_DEFAULT_CONFIGS: List[Dict[str, Any]] = [
    {"plane": "liquid", "n_cycles": 5},
    {"plane": "liquid", "n_cycles": 10},
    {"plane": "solid", "n_cycles": 5},
    {"plane": "solid", "n_cycles": 10},
    {"plane": "gas", "n_cycles": 5},
]


class ExperimentRunner:
    """Benchmark :class:`~physml.estimator.PhysicsPredictor` on synthetic data.

    Parameters
    ----------
    results_dir : str
        Directory where ``experiments.jsonl`` is written.
    configs : list of dict, optional
        Hyperparameter configurations to try.  Each dict is passed as kwargs
        to :class:`~physml.estimator.PhysicsPredictor`.  Defaults to a small
        built-in grid.
    random_seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        results_dir: str = "~/.mycelium/experiments",
        configs: Optional[List[Dict[str, Any]]] = None,
        random_seed: int = 42,
    ) -> None:
        self.results_dir = Path(results_dir).expanduser()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.configs = configs or _DEFAULT_CONFIGS
        self.random_seed = random_seed
        self._log_path = self.results_dir / "experiments.jsonl"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        task: str = "regression",
        n_samples: int = 200,
        n_features: int = 5,
        test_fraction: float = 0.25,
    ) -> BenchmarkSummary:
        """Run all configurations and return a :class:`BenchmarkSummary`.

        Parameters
        ----------
        task : str
            ``"regression"`` or ``"classification"``.
        n_samples : int
            Total number of synthetic samples (train + test).
        n_features : int
            Number of input features.
        test_fraction : float
            Fraction of data used for evaluation.

        Returns
        -------
        BenchmarkSummary
        """
        if task not in ("regression", "classification"):
            raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")

        _logger.info(
            "ExperimentRunner: running %d configs on %s (n=%d, f=%d)",
            len(self.configs), task, n_samples, n_features,
        )

        X_train, X_test, y_train, y_test = self._make_data(
            task, n_samples, n_features, test_fraction
        )

        results: List[ExperimentResult] = []
        t_total_start = time.time()

        for cfg in self.configs:
            result = self._run_one(cfg, task, X_train, X_test, y_train, y_test)
            results.append(result)
            self._append_log(result)
            score_str = f"{result.score:.4f}" if result.error is None else f"ERROR:{result.error}"
            _logger.info("  config=%s  score=%s  t=%.2fs", cfg, score_str, result.fit_time_s)

        total_time = time.time() - t_total_start

        # Pick best
        valid = [r for r in results if r.error is None and math.isfinite(r.score)]
        if not valid:
            return BenchmarkSummary(
                task=task,
                n_experiments=len(results),
                best_score=float("nan"),
                best_config={},
                mean_score=float("nan"),
                std_score=float("nan"),
                total_time_s=total_time,
                results=results,
            )

        scores = [r.score for r in valid]
        best = max(valid, key=lambda r: r.score)

        return BenchmarkSummary(
            task=task,
            n_experiments=len(results),
            best_score=best.score,
            best_config=best.config,
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            total_time_s=total_time,
            results=results,
        )

    def load_history(self) -> List[ExperimentResult]:
        """Load all past experiment results from the log file."""
        if not self._log_path.exists():
            return []
        results = []
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(ExperimentResult.from_dict(json.loads(line)))
                    except Exception:
                        pass
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_data(
        self,
        task: str,
        n_samples: int,
        n_features: int,
        test_fraction: float,
    ):
        rng = np.random.default_rng(self.random_seed)
        X = rng.normal(0, 1, (n_samples, n_features))

        if task == "regression":
            coef = rng.normal(0, 1, n_features)
            y = X @ coef + rng.normal(0, 0.2, n_samples)
        else:  # classification
            coef = rng.normal(0, 1, n_features)
            logit = X @ coef
            y = (logit > 0).astype(int)

        n_test = max(1, int(n_samples * test_fraction))
        idx = rng.permutation(n_samples)
        te_idx, tr_idx = idx[:n_test], idx[n_test:]
        return X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

    def _run_one(
        self,
        cfg: Dict[str, Any],
        task: str,
        X_train,
        X_test,
        y_train,
        y_test,
    ) -> ExperimentResult:
        try:
            from physml.estimator import PhysicsPredictor

            predictor = PhysicsPredictor(**cfg)

            t0 = time.time()
            predictor.fit(X_train, y_train)
            fit_time = time.time() - t0

            t0 = time.time()
            score = float(predictor.score(X_test, y_test))
            pred_time = time.time() - t0

            return ExperimentResult(
                task=task,
                config=cfg,
                score=score,
                fit_time_s=fit_time,
                predict_time_s=pred_time,
                n_train=len(y_train),
                n_test=len(y_test),
                n_features=X_train.shape[1],
            )
        except Exception as exc:
            _logger.warning("ExperimentRunner._run_one failed: %s", exc)
            return ExperimentResult(
                task=task,
                config=cfg,
                score=float("nan"),
                fit_time_s=0.0,
                predict_time_s=0.0,
                n_train=len(y_train),
                n_test=len(y_test),
                n_features=X_train.shape[1],
                error=str(exc),
            )

    def _append_log(self, result: ExperimentResult) -> None:
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_dict()) + "\n")
        except Exception as exc:
            _logger.warning("ExperimentRunner: could not write log: %s", exc)

    def __repr__(self) -> str:
        return (
            f"ExperimentRunner(configs={len(self.configs)}, "
            f"results_dir={str(self.results_dir)!r})"
        )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def cli_main(argv: Optional[List[str]] = None) -> None:
    """Command-line runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark the PhysML physics engine on synthetic data."
    )
    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        default="regression",
        help="Prediction task type.",
    )
    parser.add_argument(
        "--samples", type=int, default=200, help="Number of synthetic samples."
    )
    parser.add_argument(
        "--features", type=int, default=5, help="Number of input features."
    )
    parser.add_argument(
        "--results-dir",
        default="~/.mycelium/experiments",
        help="Directory for experiment logs.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )

    args = parser.parse_args(argv)
    runner = ExperimentRunner(
        results_dir=args.results_dir,
        random_seed=args.seed,
    )
    summary = runner.run(task=args.task, n_samples=args.samples, n_features=args.features)
    print(summary)
    print(f"\nBest config: {summary.best_config}")
    metric = "R²" if args.task == "regression" else "Accuracy"
    print(f"Best {metric}: {summary.best_score:.4f}")
    print(f"Mean {metric}: {summary.mean_score:.4f} ± {summary.std_score:.4f}")
    print(f"Results saved to: {runner._log_path}")


if __name__ == "__main__":
    cli_main()
