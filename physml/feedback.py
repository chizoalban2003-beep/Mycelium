"""Stage 45 — FeedbackBuffer and online RLHF loop.

Provides:
* :class:`FeedbackBuffer` — accumulates ``(features, label, weight)`` triples
  from human or automated feedback.  Supports sampling, de-duplication, and
  priority-weighted batches.
* :class:`OnlineRLHF` — orchestrates continuous improvement of a
  :class:`~physml.mycelium_agent.MyceliumAgent` from a :class:`FeedbackBuffer`:
  - Accumulates labelled feedback.
  - Triggers ``partial_fit`` when the buffer reaches a configurable threshold.
  - Optionally reweights examples by recency and confidence.

Design rationale
----------------
Current ``self_improve()`` only adjusts the ask-threshold.  Stage 45 closes the
RLHF loop properly:

1. **FeedbackBuffer** separates concerns — anything that generates labels
   (human annotation, tool output, oracle) pushes to the buffer.
2. **OnlineRLHF.step()** is called periodically.  When the buffer has
   ``min_batch_size`` or more examples it calls ``partial_fit`` on the
   underlying predictor and resets the buffer.  This makes the predictor
   genuinely improve from deployment feedback rather than only adjusting
   thresholds.
3. The ``weight`` field allows importance-weighting — e.g. high-confidence
   oracle labels get weight 1.0 while automatically-generated pseudo-labels
   get weight 0.5.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from physml.mycelium_agent import MyceliumAgent


@dataclass
class FeedbackItem:
    """A single labelled example with an importance weight.

    Attributes
    ----------
    features : np.ndarray
        1-D float32 feature vector.
    label : int or float
        Ground-truth label (classification or regression).
    weight : float
        Importance weight in (0, 1].  Default 1.0.
    source : str
        Where this label came from (e.g. ``"oracle"``, ``"tool"``,
        ``"self_label"``).  For audit trails.
    """

    features: np.ndarray
    label: Any
    weight: float = 1.0
    source: str = "oracle"


class FeedbackBuffer:
    """Bounded FIFO buffer of labelled feedback items.

    Parameters
    ----------
    capacity : int, default 2000
        Maximum items retained.  Older items are dropped when full.
    dedup_window : int, default 50
        Approximate deduplication window — identical feature vectors
        within the last *dedup_window* entries are skipped.
    """

    def __init__(self, capacity: int = 2000, dedup_window: int = 50) -> None:
        self.capacity = int(capacity)
        self.dedup_window = int(dedup_window)
        self._buffer: deque[FeedbackItem] = deque(maxlen=self.capacity)
        self._recent_hashes: deque[int] = deque(maxlen=self.dedup_window)

    def push(self, item: FeedbackItem) -> bool:
        """Add *item* to the buffer.

        Duplicate detection is performed by hashing the feature vector
        bytes within the recent window.

        Parameters
        ----------
        item : FeedbackItem

        Returns
        -------
        bool — ``True`` if the item was added, ``False`` if it was skipped
        as a duplicate.
        """
        h = hash(item.features.tobytes())
        if h in self._recent_hashes:
            return False
        self._recent_hashes.append(h)
        self._buffer.append(item)
        return True

    def push_raw(
        self,
        X: Any,
        y: Any,
        weight: float = 1.0,
        source: str = "oracle",
    ) -> int:
        """Convenience wrapper — push a batch of raw arrays.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (n_features,)
        y : array-like, shape (n_samples,) or scalar
        weight : float, default 1.0
        source : str, default "oracle"

        Returns
        -------
        int — number of items successfully added (after dedup).
        """
        X_arr = np.atleast_2d(np.array(X, dtype=np.float32))
        y_arr = np.atleast_1d(np.array(y))
        if y_arr.shape[0] == 1 and X_arr.shape[0] > 1:
            y_arr = np.broadcast_to(y_arr, (X_arr.shape[0],))

        added = 0
        for xi, yi in zip(X_arr, y_arr):
            item = FeedbackItem(
                features=xi.copy(),
                label=yi,
                weight=float(weight),
                source=source,
            )
            if self.push(item):
                added += 1
        return added

    def sample_batch(
        self,
        n: int | None = None,
        *,
        recency_weight: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a batch of ``(X, y, weights)`` from the buffer.

        Parameters
        ----------
        n : int or None
            Number of samples.  ``None`` returns the full buffer.
        recency_weight : bool, default True
            When ``True``, items are weighted by both their stored weight
            and a recency factor (newer items get higher probability).

        Returns
        -------
        (X, y, weights) — float32 arrays.
        """
        items = list(self._buffer)
        if not items:
            empty = np.empty((0,), dtype=np.float32)
            return empty, empty, empty

        m = len(items)
        if recency_weight:
            base_weights = np.array([it.weight for it in items], dtype=np.float64)
            recency = np.linspace(0.5, 1.0, m)
            probs = base_weights * recency
            probs /= probs.sum()
        else:
            probs = np.array([it.weight for it in items], dtype=np.float64)
            probs /= probs.sum()

        if n is None or n >= m:
            indices = np.arange(m)
        else:
            rng = np.random.default_rng()
            indices = rng.choice(m, size=n, replace=False, p=probs)

        selected = [items[i] for i in indices]
        X = np.array([it.features for it in selected], dtype=np.float32)
        y = np.array([it.label for it in selected])
        w = np.array([it.weight for it in selected], dtype=np.float32)
        return X, y, w

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        """Empty the buffer."""
        self._buffer.clear()
        self._recent_hashes.clear()

    def stats(self) -> dict:
        """Return summary statistics about the buffer contents."""
        items = list(self._buffer)
        if not items:
            return {"size": 0, "sources": {}, "mean_weight": 0.0}
        sources: dict[str, int] = {}
        weights = []
        for it in items:
            sources[it.source] = sources.get(it.source, 0) + 1
            weights.append(it.weight)
        return {
            "size": len(items),
            "sources": sources,
            "mean_weight": float(np.mean(weights)),
        }


class OnlineRLHF:
    """Continuously improve a :class:`~physml.mycelium_agent.MyceliumAgent`
    from a :class:`FeedbackBuffer`.

    Parameters
    ----------
    agent : MyceliumAgent
        The agent to improve. Must already be fitted.
    buffer : FeedbackBuffer
        Buffer from which training batches are drawn.
    min_batch_size : int, default 32
        Minimum items in buffer before a ``partial_fit`` is triggered.
    max_batch_size : int, default 256
        Maximum batch size per update step.
    clear_after_fit : bool, default False
        When ``True``, the buffer is cleared after each update.
        When ``False`` (default), items are kept for replay.
    """

    def __init__(
        self,
        agent: "MyceliumAgent",
        buffer: FeedbackBuffer,
        min_batch_size: int = 32,
        max_batch_size: int = 256,
        clear_after_fit: bool = False,
    ) -> None:
        self.agent = agent
        self.buffer = buffer
        self.min_batch_size = int(min_batch_size)
        self.max_batch_size = int(max_batch_size)
        self.clear_after_fit = bool(clear_after_fit)

        self._n_updates: int = 0
        self._n_samples_seen: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_feedback(
        self,
        X: Any,
        y: Any,
        weight: float = 1.0,
        source: str = "oracle",
    ) -> int:
        """Push labelled feedback into the buffer.

        Parameters
        ----------
        X : array-like
        y : array-like
        weight : float, default 1.0
        source : str, default "oracle"

        Returns
        -------
        int — number of unique items added.
        """
        return self.buffer.push_raw(X, y, weight=weight, source=source)

    def step(self) -> dict:
        """Trigger a ``partial_fit`` update if the buffer is large enough.

        Returns
        -------
        dict — ``{"updated": bool, "n_samples": int, "n_updates": int}``
        """
        if len(self.buffer) < self.min_batch_size:
            return {"updated": False, "n_samples": 0, "n_updates": self._n_updates}

        X, y, _w = self.buffer.sample_batch(n=self.max_batch_size, recency_weight=True)
        if len(X) == 0 or len(np.unique(y)) < 2:
            return {"updated": False, "n_samples": 0, "n_updates": self._n_updates}

        predictor = self.agent._predictor
        if predictor is None:
            return {"updated": False, "n_samples": 0, "n_updates": self._n_updates}

        try:
            if hasattr(predictor, "partial_fit"):
                # Support both sklearn-style (classes=) and CEP-style (no classes=)
                import inspect
                sig = inspect.signature(predictor.partial_fit)
                if "classes" in sig.parameters:
                    classes = np.unique(y)
                    predictor.partial_fit(X, y, classes=classes)
                else:
                    predictor.partial_fit(X, y)
            elif hasattr(predictor, "fit"):
                predictor.fit(X, y)
            else:
                return {"updated": False, "n_samples": 0, "n_updates": self._n_updates}
        except Exception:
            return {"updated": False, "n_samples": 0, "n_updates": self._n_updates}

        self._n_updates += 1
        self._n_samples_seen += len(X)
        if self.clear_after_fit:
            self.buffer.clear()

        return {"updated": True, "n_samples": len(X), "n_updates": self._n_updates}

    def report(self) -> dict:
        """Summary of RLHF loop activity."""
        return {
            "n_updates": self._n_updates,
            "n_samples_seen": self._n_samples_seen,
            "buffer_stats": self.buffer.stats(),
        }
