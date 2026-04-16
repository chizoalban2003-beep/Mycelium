"""Stage 83 — OnlineEvaluator: sliding-window incremental model evaluation.

Tracks prediction performance in real time over a streaming sequence of
(prediction, label) batches using a configurable sliding window.  Supports
both classification (accuracy, F1) and regression (MAE, RMSE) metrics.

Classes
-------
EvalWindow
    Snapshot of accuracy/loss metrics for one evaluation window.
OnlineEvaluator
    Accumulates streamed (y_pred, y_true) pairs and computes rolling metrics.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EvalWindow:
    """Metrics snapshot for one sliding window.

    Attributes
    ----------
    window_id : int
        Sequential window index (0-based).
    n_samples : int
        Number of samples in this window.
    accuracy : float or None
        Classification accuracy (None for regression task).
    f1_macro : float or None
        Macro-averaged F1 score (None for regression task).
    mae : float or None
        Mean absolute error (None for classification task).
    rmse : float or None
        Root mean squared error (None for classification task).
    timestamp : float
        Unix timestamp when this window was computed.
    """

    window_id: int
    n_samples: int
    accuracy: float | None
    f1_macro: float | None
    mae: float | None
    rmse: float | None
    timestamp: float

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "window_id": self.window_id,
            "n_samples": self.n_samples,
            "timestamp": round(self.timestamp, 4),
        }
        if self.accuracy is not None:
            d["accuracy"] = round(self.accuracy, 4)
        if self.f1_macro is not None:
            d["f1_macro"] = round(self.f1_macro, 4)
        if self.mae is not None:
            d["mae"] = round(self.mae, 4)
        if self.rmse is not None:
            d["rmse"] = round(self.rmse, 4)
        return d


class OnlineEvaluator:
    """Accumulate streaming predictions and compute rolling window metrics.

    Parameters
    ----------
    task : str, default ``"classification"``
        ``"classification"`` or ``"regression"``.
    window_size : int, default 100
        Number of samples in each sliding window.
    step_size : int, default 50
        Number of new samples to accumulate before emitting a new
        :class:`EvalWindow`.  ``step_size=window_size`` gives tumbling
        (non-overlapping) windows.

    Example
    -------
    >>> import numpy as np
    >>> from physml.online_evaluator import OnlineEvaluator
    >>> ev = OnlineEvaluator(task="classification", window_size=50, step_size=25)
    >>> rng = np.random.default_rng(0)
    >>> for _ in range(4):
    ...     y_pred = rng.integers(0, 2, 25)
    ...     y_true = rng.integers(0, 2, 25)
    ...     ev.update(y_pred, y_true)
    >>> len(ev.windows) >= 1
    True
    """

    def __init__(
        self,
        *,
        task: str = "classification",
        window_size: int = 100,
        step_size: int = 50,
    ) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError(
                f"task must be 'classification' or 'regression', got {task!r}"
            )
        self.task = task
        self.window_size = int(window_size)
        self.step_size = int(step_size)

        self._pred_buf: deque[float | int] = deque()
        self._true_buf: deque[float | int] = deque()
        self._n_since_last: int = 0
        self._windows: list[EvalWindow] = []
        self._window_id: int = 0
        self._total_pred: list[float | int] = []
        self._total_true: list[float | int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, y_pred: Any, y_true: Any) -> list[EvalWindow]:
        """Add a batch of predictions and return any newly completed windows.

        Parameters
        ----------
        y_pred : array-like of shape (n,)
        y_true : array-like of shape (n,)

        Returns
        -------
        list[EvalWindow]
            Newly emitted windows (may be empty if step_size not reached).
        """
        y_pred = np.asarray(y_pred).ravel()
        y_true = np.asarray(y_true).ravel()
        assert len(y_pred) == len(y_true), "y_pred and y_true must have the same length"

        new_windows: list[EvalWindow] = []

        for p, t in zip(y_pred.tolist(), y_true.tolist()):
            self._pred_buf.append(p)
            self._true_buf.append(t)
            self._total_pred.append(p)
            self._total_true.append(t)
            self._n_since_last += 1

            # Trim buffer to window_size
            while len(self._pred_buf) > self.window_size:
                self._pred_buf.popleft()
                self._true_buf.popleft()

            if self._n_since_last >= self.step_size and len(self._pred_buf) >= min(
                self.step_size, self.window_size
            ):
                win = self._compute_window(
                    np.array(list(self._pred_buf)),
                    np.array(list(self._true_buf)),
                )
                new_windows.append(win)
                self._windows.append(win)
                self._n_since_last = 0
                self._window_id += 1

        return new_windows

    def flush(self) -> EvalWindow | None:
        """Force-emit a window from whatever data is currently buffered.

        Returns ``None`` if the buffer is empty.
        """
        if not self._pred_buf:
            return None
        win = self._compute_window(
            np.array(list(self._pred_buf)),
            np.array(list(self._true_buf)),
        )
        self._windows.append(win)
        self._window_id += 1
        self._n_since_last = 0
        return win

    def global_metrics(self) -> dict[str, float]:
        """Return metrics computed over *all* samples seen so far."""
        if not self._total_pred:
            return {}
        return self._metrics(
            np.array(self._total_pred), np.array(self._total_true)
        )

    @property
    def windows(self) -> list[EvalWindow]:
        """All emitted windows in chronological order."""
        return list(self._windows)

    @property
    def n_total(self) -> int:
        """Total number of (pred, true) pairs seen so far."""
        return len(self._total_pred)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_window(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> EvalWindow:
        m = self._metrics(y_pred, y_true)
        return EvalWindow(
            window_id=self._window_id,
            n_samples=len(y_pred),
            accuracy=m.get("accuracy"),
            f1_macro=m.get("f1_macro"),
            mae=m.get("mae"),
            rmse=m.get("rmse"),
            timestamp=time.time(),
        )

    def _metrics(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> dict[str, float]:
        if self.task == "classification":
            acc = float(np.mean(y_pred == y_true))
            f1 = self._f1_macro(y_pred, y_true)
            return {"accuracy": acc, "f1_macro": f1}
        else:
            errors = y_pred.astype(float) - y_true.astype(float)
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            return {"mae": mae, "rmse": rmse}

    @staticmethod
    def _f1_macro(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute macro-averaged F1 without sklearn dependency."""
        classes = np.unique(np.concatenate([y_pred, y_true]))
        f1_scores = []
        for cls in classes:
            tp = float(np.sum((y_pred == cls) & (y_true == cls)))
            fp = float(np.sum((y_pred == cls) & (y_true != cls)))
            fn = float(np.sum((y_pred != cls) & (y_true == cls)))
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2.0 * precision * recall / (precision + recall + 1e-9)
            f1_scores.append(f1)
        return float(np.mean(f1_scores)) if f1_scores else 0.0
