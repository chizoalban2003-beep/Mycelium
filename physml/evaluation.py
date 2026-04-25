"""Stage 14 — Evaluation harness for myco / PhysicsAgent.

:func:`benchmark_agent` runs a standard active-learning evaluation loop
and records accuracy, oracle call rate, and ask-rate trajectory.

The loop simulates an oracle (the known ground-truth labels) and a fixed
budget of labelling operations:

1. The agent observes each unlabelled sample.
2. When the agent says "ask", the oracle label is provided (as long as the
   budget allows).
3. Accuracy, oracle calls, and ask-rate are recorded at each step.

The function returns a :class:`BenchmarkResult` with per-step history and
summary statistics, suitable for plotting learning curves or publishing
benchmark numbers.

Usage
-----
::

    from physml import myco
    from physml.evaluation import benchmark_agent
    import numpy as np

    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] > 0).astype(int)

    agent = myco()
    result = benchmark_agent(agent, X, y, oracle_budget=40, seed_size=20)

    print(result.summary())
    # Plot: result.accuracy_curve, result.ask_rate_curve
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Results returned by :func:`benchmark_agent`.

    Attributes
    ----------
    accuracy_curve : list[float]
        Rolling accuracy after each evaluation step (computed on the *test*
        portion of the data).
    ask_rate_curve : list[float]
        Fraction of steps where the agent issued an "ask" action, measured
        in a trailing window of ``window`` steps.
    oracle_calls : int
        Total number of oracle (label) calls made during the benchmark.
    total_steps : int
        Total number of evaluation steps (= len(X_test)).
    budget_exhausted_at : int or None
        Step index at which the oracle budget was exhausted, or ``None`` if
        the full budget was not consumed.
    history : list[dict]
        Per-step records: ``{step, action, confidence, correct, cumulative_acc}``.
    """

    accuracy_curve: list[float] = field(default_factory=list)
    ask_rate_curve: list[float] = field(default_factory=list)
    oracle_calls: int = 0
    total_steps: int = 0
    budget_exhausted_at: int | None = None
    history: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        final_acc = self.accuracy_curve[-1] if self.accuracy_curve else float("nan")
        final_ask = self.ask_rate_curve[-1] if self.ask_rate_curve else float("nan")
        lines = [
            "BenchmarkResult",
            f"  total_steps          : {self.total_steps}",
            f"  oracle_calls         : {self.oracle_calls}",
            f"  budget_exhausted_at  : {self.budget_exhausted_at}",
            f"  final accuracy       : {final_acc:.3f}",
            f"  final ask-rate       : {final_ask:.3f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main benchmark function
# ---------------------------------------------------------------------------

def benchmark_agent(
    agent: Any,
    X: np.ndarray,
    y: np.ndarray,
    oracle_budget: int = 50,
    seed_size: int = 20,
    window: int = 20,
    shuffle: bool = True,
    random_state: int | None = 42,
) -> BenchmarkResult:
    """Evaluate an agent over a labelled dataset using a simulated oracle.

    Parameters
    ----------
    agent : MyceliumAgent or PhysicsAgent (unfitted)
        Will be reset and fitted on the seed split before evaluation.
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    oracle_budget : int, default 50
        Maximum number of oracle (label) queries allowed during the
        evaluation phase.
    seed_size : int, default 20
        Number of labelled samples used to seed the agent before evaluation.
        Must be < ``len(y)``.
    window : int, default 20
        Window size for the trailing ask-rate in ``ask_rate_curve``.
    shuffle : bool, default True
        Shuffle *X* and *y* before splitting into seed / eval sets.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    BenchmarkResult
    """
    X_arr = np.atleast_2d(X)
    y_arr = np.atleast_1d(y)
    n = len(y_arr)

    if seed_size >= n:
        raise ValueError(
            f"seed_size ({seed_size}) must be less than the dataset size ({n})."
        )

    rng = np.random.default_rng(random_state)

    if shuffle:
        perm = rng.permutation(n)
        X_arr = X_arr[perm]
        y_arr = y_arr[perm]

    X_seed, y_seed = X_arr[:seed_size], y_arr[:seed_size]
    X_eval, y_eval = X_arr[seed_size:], y_arr[seed_size:]

    # Seed the agent
    agent.fit(X_seed, y_seed)

    result = BenchmarkResult(total_steps=len(y_eval))
    oracle_calls = 0
    n_correct = 0
    budget_exhausted_at: int | None = None
    ask_window: list[int] = []

    for step, (x_i, y_i) in enumerate(zip(X_eval, y_eval)):
        x_row = x_i.reshape(1, -1)
        action = agent.observe(x_row)

        asked = action.action == "ask"
        ask_window.append(1 if asked else 0)
        if len(ask_window) > window:
            ask_window.pop(0)

        # Provide oracle label if budget remains
        if asked:
            if oracle_calls < oracle_budget:
                oracle_calls += 1
                agent.reward(x_row, np.array([y_i]))
            elif budget_exhausted_at is None:
                budget_exhausted_at = step

        # Evaluate accuracy: count correct *predictions* (ignore asks)
        if action.prediction is not None:
            try:
                pred_val = action.prediction
                if hasattr(pred_val, "__len__"):
                    pred_val = pred_val[0]
                n_correct += int(pred_val == y_i)
            except Exception:
                pass

        cum_acc = n_correct / (step + 1)
        ask_rate = float(np.mean(ask_window))

        result.accuracy_curve.append(cum_acc)
        result.ask_rate_curve.append(ask_rate)
        result.history.append(
            {
                "step": step,
                "action": action.action,
                "confidence": action.confidence,
                "correct": action.prediction == y_i if action.prediction is not None else None,
                "cumulative_acc": cum_acc,
                "ask_rate": ask_rate,
            }
        )

    result.oracle_calls = oracle_calls
    result.budget_exhausted_at = budget_exhausted_at
    return result
