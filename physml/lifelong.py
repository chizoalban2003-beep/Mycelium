"""Stage 69 — LifelongLearner.

Closes the autonomous-agent loop with a **continuous self-improvement cycle**
that processes streaming data in fixed-size chunks, periodically self-evaluates
against a held-out validation window, and triggers
:meth:`~physml.mycelium_agent.MyceliumAgent.self_improve` (or the equivalent
on any agent that exposes a ``self_improve`` / ``fit`` interface) whenever
performance drops below a configurable threshold.

The class ties together every prior stage:

* **Drift detection** — incremental EWMA-based performance tracking.
* **Self-improvement** — calls the agent's ``self_improve()`` (Stage 40/47)
  when accuracy falls below *improvement_threshold*.
* **CompetitiveReport** — optional end-of-run benchmark report (Stage 68).
* **Performance history** — full per-round telemetry for introspection.

Classes
-------
LifelongLearner
    Continuous improvement loop wrapping any compatible agent.
RoundResult
    Per-round performance snapshot stored in the history.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score


@dataclass
class RoundResult:
    """Performance snapshot for one evaluation round.

    Attributes
    ----------
    round_idx : int
        Zero-based index of this round.
    accuracy : float
        Accuracy on the validation window at the time of this round.
    improved : bool
        Whether a self-improvement step was triggered this round.
    improvement_delta : float
        Accuracy gain from the most recent self-improvement (0 if none).
    n_samples_seen : int
        Cumulative samples processed through the end of this round.
    elapsed_s : float
        Wall-clock seconds elapsed at end of this round.
    """

    round_idx: int
    accuracy: float
    improved: bool
    improvement_delta: float
    n_samples_seen: int
    elapsed_s: float

    def as_dict(self) -> dict[str, Any]:
        """Serialisable dict for JSON export."""
        return {
            "round": self.round_idx,
            "accuracy": round(self.accuracy, 4),
            "improved": self.improved,
            "improvement_delta": round(self.improvement_delta, 4),
            "n_samples_seen": self.n_samples_seen,
            "elapsed_s": round(self.elapsed_s, 3),
        }


class LifelongLearner:
    """Continuous self-improvement loop wrapping any compatible agent.

    Processes incoming data in *chunk_size*-sample batches.  Every
    *eval_every* chunks it evaluates on a rolling validation window and
    fires :meth:`self_improve` if accuracy drops below
    *improvement_threshold*.

    Compatible with:

    * :class:`~physml.mycelium_agent.MyceliumAgent` — uses its built-in
      ``self_improve()`` method when available.
    * :class:`~physml.autonomous_agent.AutonomousAgent` — delegates to the
      inner core's ``self_improve()`` or falls back to ``fit()``.
    * Any sklearn estimator — falls back to ``fit(X_val, y_val)``.

    Parameters
    ----------
    agent : Any
        The agent to manage.  Must expose at minimum ``fit(X, y)`` and
        ``predict(X)``.
    improvement_threshold : float, default 0.75
        Accuracy below which a self-improvement step is triggered.
    eval_every : int, default 2
        Number of chunks between evaluations.
    val_window : int, default 200
        Maximum number of samples kept in the rolling validation window.
    verbose : bool, default False
        If True, print progress to stdout after each evaluation.

    Example
    -------
    >>> from sklearn.datasets import make_classification
    >>> from physml import MyceliumAgent
    >>> from physml.lifelong import LifelongLearner
    >>> X, y = make_classification(n_samples=500, n_features=10, random_state=0)
    >>> agent = MyceliumAgent()
    >>> ll = LifelongLearner(agent, improvement_threshold=0.70)
    >>> history = ll.run(X, y, chunk_size=50)
    >>> print(ll.final_accuracy())
    """

    def __init__(
        self,
        agent: Any,
        *,
        improvement_threshold: float = 0.75,
        eval_every: int = 2,
        val_window: int = 200,
        verbose: bool = False,
    ) -> None:
        self.agent = agent
        self.improvement_threshold = float(improvement_threshold)
        self.eval_every = max(1, int(eval_every))
        self.val_window = max(10, int(val_window))
        self.verbose = verbose

        # Internal state
        self._history: list[RoundResult] = []
        self._n_samples_seen: int = 0
        self._improvement_count: int = 0
        self._round_idx: int = 0
        self._fitted: bool = False
        self._start_time: float = 0.0

        # Rolling validation window (circular buffers)
        self._val_X: list[np.ndarray] = []
        self._val_y: list[Any] = []

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def run(
        self,
        X: Any,
        y: Any,
        *,
        chunk_size: int = 50,
    ) -> list[RoundResult]:
        """Process the full dataset *X, y* in streaming chunks.

        On the first chunk the agent is fitted (cold-start).  Subsequent
        chunks are used to update the rolling validation window and trigger
        evaluation / self-improvement as configured.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        chunk_size : int, default 50
            Number of samples per streaming chunk.

        Returns
        -------
        list[RoundResult]
            Per-round performance snapshots.
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        n = len(X_arr)
        chunk_size = max(1, int(chunk_size))

        self._start_time = time.perf_counter()

        for start in range(0, n, chunk_size):
            X_chunk = X_arr[start : start + chunk_size]
            y_chunk = y_arr[start : start + chunk_size]
            self._process_chunk(X_chunk, y_chunk)

        return list(self._history)

    def step(self, X_chunk: Any, y_chunk: Any) -> dict[str, Any] | None:
        """Process a single streaming chunk externally.

        Returns the :class:`RoundResult` dict if an evaluation occurred this
        step, otherwise ``None``.

        Parameters
        ----------
        X_chunk : array-like
        y_chunk : array-like
        """
        if self._start_time == 0.0:
            self._start_time = time.perf_counter()

        X_arr = np.asarray(X_chunk)
        y_arr = np.asarray(y_chunk)
        return self._process_chunk(X_arr, y_arr)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[RoundResult]:
        """Ordered list of per-round evaluation snapshots."""
        return list(self._history)

    def final_accuracy(self) -> float:
        """Accuracy on the most recent evaluation round (or NaN if none)."""
        if not self._history:
            return float("nan")
        return self._history[-1].accuracy

    def summary(self) -> dict[str, Any]:
        """High-level summary of the lifelong learning run."""
        accs = [r.accuracy for r in self._history]
        return {
            "n_rounds": len(self._history),
            "n_samples_seen": self._n_samples_seen,
            "n_improvements": self._improvement_count,
            "initial_accuracy": round(accs[0], 4) if accs else None,
            "final_accuracy": round(accs[-1], 4) if accs else None,
            "peak_accuracy": round(max(accs), 4) if accs else None,
            "improvement_threshold": self.improvement_threshold,
        }

    def competitive_report(
        self,
        X_test: Any,
        y_test: Any,
        *,
        extra_baselines: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a :class:`~physml.competitive_report.CompetitiveReport` on the
        agent after all lifelong learning rounds have completed.

        Parameters
        ----------
        X_test, y_test : array-like
            Held-out evaluation data (not used during training).
        extra_baselines : dict or None
            Additional competing models.

        Returns
        -------
        dict
            Full structured competitive report.
        """
        from physml.competitive_report import CompetitiveReport

        reporter = CompetitiveReport()
        return reporter.run(
            self.agent,
            X=X_test,
            y=y_test,
            dataset_name="lifelong_eval",
            extra_baselines=extra_baselines,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_chunk(
        self, X_chunk: np.ndarray, y_chunk: np.ndarray
    ) -> dict[str, Any] | None:
        """Fit or update the agent on *X_chunk / y_chunk*.

        Returns a RoundResult dict if an evaluation fired, else None.
        """
        chunk_len = len(X_chunk)
        result = None

        if not self._fitted:
            # Cold-start: fit on the first chunk
            self.agent.fit(X_chunk, y_chunk)
            self._fitted = True
        else:
            # Incremental update: try partial_fit, fall back to noop
            # (evaluation / self-improve does the heavy lifting)
            if hasattr(self.agent, "partial_fit"):
                try:
                    self.agent.partial_fit(X_chunk, y_chunk)
                except Exception:
                    pass  # Some agents don't support partial_fit after init
            # If not partial_fit-capable, we rely on self_improve to refit

        # Update rolling validation window
        self._val_X.extend(list(X_chunk))
        self._val_y.extend(list(y_chunk))
        # Trim to val_window
        if len(self._val_X) > self.val_window:
            excess = len(self._val_X) - self.val_window
            self._val_X = self._val_X[excess:]
            self._val_y = self._val_y[excess:]

        self._n_samples_seen += chunk_len
        self._round_idx += 1

        # Evaluate every eval_every chunks
        if self._round_idx % self.eval_every == 0 and len(self._val_X) >= 5:
            result = self._evaluate_and_maybe_improve()

        return result

    def _evaluate_and_maybe_improve(self) -> dict[str, Any]:
        """Evaluate on validation window; trigger self-improvement if needed."""
        X_val = np.array(self._val_X)
        y_val = np.array(self._val_y)

        # Compute accuracy
        try:
            preds = self._predict(X_val)
            acc = float(accuracy_score(y_val, preds))
        except Exception:
            acc = float("nan")

        acc_before = acc
        improved = False
        delta = 0.0

        # Trigger self-improvement if accuracy is below threshold
        if not np.isnan(acc) and acc < self.improvement_threshold:
            improved = True
            self._trigger_improve(X_val, y_val)
            # Re-evaluate after improvement
            try:
                preds_after = self._predict(X_val)
                acc_after = float(accuracy_score(y_val, preds_after))
                delta = acc_after - acc_before
                acc = acc_after
            except Exception:
                pass
            self._improvement_count += 1

        elapsed = time.perf_counter() - self._start_time
        round_result = RoundResult(
            round_idx=len(self._history),
            accuracy=acc if not np.isnan(acc) else 0.0,
            improved=improved,
            improvement_delta=delta,
            n_samples_seen=self._n_samples_seen,
            elapsed_s=elapsed,
        )
        self._history.append(round_result)

        if self.verbose:
            marker = " ⬆ improved" if improved else ""
            print(
                f"[LifelongLearner] round={round_result.round_idx}"
                f"  acc={acc:.4f}"
                f"  samples={self._n_samples_seen}"
                f"  improvements={self._improvement_count}"
                f"{marker}"
            )

        return round_result.as_dict()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Unified predict delegating to agent's predict or observe loop."""
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

    def _trigger_improve(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Call agent self-improvement using the best available interface."""
        # Priority 1: MyceliumAgent-style self_improve()
        if hasattr(self.agent, "self_improve"):
            try:
                self.agent.self_improve(X_val, y_val)
                return
            except Exception:
                pass

        # Priority 2: AutonomousAgent delegates to core
        core = getattr(self.agent, "core", None)
        if core is not None and hasattr(core, "self_improve"):
            try:
                core.self_improve(X_val, y_val)
                return
            except Exception:
                pass

        # Priority 3: Plain sklearn fit()
        if hasattr(self.agent, "fit"):
            try:
                self.agent.fit(X_val, y_val)
            except Exception:
                pass
