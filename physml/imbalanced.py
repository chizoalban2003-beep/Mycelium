"""Stage 82 — ImbalancedHandler: class-imbalance mitigation.

Provides three complementary strategies for dealing with imbalanced
classification datasets:

* **oversample** — randomly duplicate minority-class rows until each class
  reaches ``target_ratio`` of the majority class.
* **undersample** — randomly drop majority-class rows until each class is
  no larger than ``target_ratio`` × minority-class size.
* **weights** — compute per-sample inverse-frequency weights for use with
  estimators that accept ``sample_weight`` (no data is added or removed).

Classes
-------
ImbalanceReport
    Statistics describing the original and resampled dataset.
ImbalancedHandler
    Applies oversampling, undersampling, or weight computation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ImbalanceReport:
    """Statistics about an imbalance-mitigation pass.

    Attributes
    ----------
    strategy : str
        Strategy applied: ``"oversample"``, ``"undersample"``, or
        ``"weights"``.
    class_counts_before : dict[int, int]
        Class → sample count *before* resampling.
    class_counts_after : dict[int, int]
        Class → sample count *after* resampling (same as before for
        ``"weights"``).
    imbalance_ratio_before : float
        Majority / minority count ratio before resampling.
    imbalance_ratio_after : float
        Majority / minority count ratio after resampling.
    n_added : int
        Samples added (oversampling only, else 0).
    n_removed : int
        Samples removed (undersampling only, else 0).
    elapsed_s : float
        Wall-clock duration.
    """

    strategy: str
    class_counts_before: dict[int, int]
    class_counts_after: dict[int, int]
    imbalance_ratio_before: float
    imbalance_ratio_after: float
    n_added: int
    n_removed: int
    elapsed_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "class_counts_before": dict(self.class_counts_before),
            "class_counts_after": dict(self.class_counts_after),
            "imbalance_ratio_before": round(self.imbalance_ratio_before, 4),
            "imbalance_ratio_after": round(self.imbalance_ratio_after, 4),
            "n_added": self.n_added,
            "n_removed": self.n_removed,
            "elapsed_s": round(self.elapsed_s, 4),
        }


class ImbalancedHandler:
    """Apply class-imbalance mitigation to tabular datasets.

    Parameters
    ----------
    strategy : str, default ``"oversample"``
        One of ``"oversample"``, ``"undersample"``, ``"weights"``.
    target_ratio : float, default 1.0
        Desired majority-to-minority ratio after resampling (1.0 = balanced).
        Ignored for the ``"weights"`` strategy.
    random_state : int, default 0

    Example
    -------
    >>> import numpy as np
    >>> from physml.imbalanced import ImbalancedHandler
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((150, 4))
    >>> y = np.array([0] * 100 + [1] * 50)
    >>> handler = ImbalancedHandler(strategy="oversample")
    >>> X_res, y_res, report = handler.resample(X, y)
    >>> report.imbalance_ratio_after <= 1.05
    True
    """

    _STRATEGIES = frozenset({"oversample", "undersample", "weights"})

    def __init__(
        self,
        *,
        strategy: str = "oversample",
        target_ratio: float = 1.0,
        random_state: int = 0,
    ) -> None:
        if strategy not in self._STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(self._STRATEGIES)}, got {strategy!r}"
            )
        self.strategy = strategy
        self.target_ratio = float(target_ratio)
        self.random_state = int(random_state)
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resample(
        self,
        X: Any,
        y: Any,
    ) -> tuple[np.ndarray, np.ndarray, ImbalanceReport]:
        """Apply the configured strategy and return resampled data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Integer class labels.

        Returns
        -------
        X_resampled : np.ndarray
        y_resampled : np.ndarray
        report : ImbalanceReport
        """
        t0 = time.time()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        counts_before = dict(zip(classes.tolist(), counts.tolist()))

        if self.strategy == "oversample":
            X_res, y_res = self._oversample(X, y, classes, counts)
        elif self.strategy == "undersample":
            X_res, y_res = self._undersample(X, y, classes, counts)
        else:
            X_res, y_res = X.copy(), y.copy()

        _, counts_after = np.unique(y_res, return_counts=True)
        ratio_before = float(counts.max()) / float(counts.min()) if counts.min() > 0 else 1.0
        ratio_after = float(counts_after.max()) / float(counts_after.min()) if counts_after.min() > 0 else 1.0

        classes_after, counts_after_arr = np.unique(y_res, return_counts=True)
        counts_after_dict = dict(zip(classes_after.tolist(), counts_after_arr.tolist()))

        n_added = max(0, len(X_res) - len(X))
        n_removed = max(0, len(X) - len(X_res))

        report = ImbalanceReport(
            strategy=self.strategy,
            class_counts_before=counts_before,
            class_counts_after=counts_after_dict,
            imbalance_ratio_before=ratio_before,
            imbalance_ratio_after=ratio_after,
            n_added=n_added,
            n_removed=n_removed,
            elapsed_s=time.time() - t0,
        )
        return X_res, y_res, report

    def compute_weights(self, y: Any) -> np.ndarray:
        """Compute inverse-frequency sample weights without resampling.

        Parameters
        ----------
        y : array-like of shape (n_samples,)

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Weight for each sample proportional to 1 / class_frequency.
        """
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        freq = dict(zip(classes.tolist(), (counts / len(y)).tolist()))
        weights = np.array([1.0 / freq[int(c)] for c in y])
        # Normalise so mean weight = 1
        weights /= weights.mean()
        return weights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _oversample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray,
        counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        majority_count = int(counts.max())
        target_count = max(1, int(majority_count * self.target_ratio))

        X_parts = [X]
        y_parts = [y]

        for cls, cnt in zip(classes, counts):
            if int(cnt) < target_count:
                n_add = target_count - int(cnt)
                mask = y == cls
                X_cls = X[mask]
                idx = self._rng.integers(0, len(X_cls), size=n_add)
                X_parts.append(X_cls[idx])
                y_parts.append(np.full(n_add, cls, dtype=y.dtype))

        return np.vstack(X_parts), np.concatenate(y_parts)

    def _undersample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray,
        counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        minority_count = int(counts.min())
        target_count = max(1, int(minority_count / self.target_ratio))

        X_parts = []
        y_parts = []

        for cls, cnt in zip(classes, counts):
            mask = y == cls
            X_cls = X[mask]
            if int(cnt) > target_count:
                idx = self._rng.choice(len(X_cls), size=target_count, replace=False)
                X_parts.append(X_cls[idx])
                y_parts.append(np.full(target_count, cls, dtype=y.dtype))
            else:
                X_parts.append(X_cls)
                y_parts.append(np.full(len(X_cls), cls, dtype=y.dtype))

        return np.vstack(X_parts), np.concatenate(y_parts)
