"""Stage 87 — BandpassFilter: variance-based feature filtering.

Keeps only features whose variance falls within a configurable
``[low_var, high_var]`` band, discarding both near-constant and
extremely noisy features.

Classes
-------
FilterResult
    Summary of which features were kept after filtering.
BandpassFilter
    Fit-transform variance band filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FilterResult:
    """Summary of a BandpassFilter transform.

    Attributes
    ----------
    n_original : int
        Number of features before filtering.
    n_kept : int
        Number of features retained.
    kept_indices : list[int]
        Column indices (relative to the original array) that were kept.
    """

    n_original: int
    n_kept: int
    kept_indices: list[int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_original": self.n_original,
            "n_kept": self.n_kept,
            "kept_indices": list(self.kept_indices),
        }

    def __repr__(self) -> str:
        return (
            f"FilterResult(n_original={self.n_original}, "
            f"n_kept={self.n_kept}, "
            f"kept_indices={self.kept_indices!r})"
        )


class BandpassFilter:
    """Filters features whose variance lies in ``[low_var, high_var]``.

    Features with variance below ``low_var`` are considered near-constant
    (low information).  Features with variance above ``high_var`` are
    treated as excessively noisy.  Setting ``high_var=None`` disables the
    upper bound.

    Parameters
    ----------
    low_var : float, default 0.0
        Minimum variance (inclusive) required to keep a feature.
    high_var : float or None, default None
        Maximum variance (inclusive) allowed.  ``None`` means no upper limit.
    """

    def __init__(
        self,
        low_var: float = 0.0,
        high_var: float | None = None,
    ) -> None:
        if low_var < 0:
            raise ValueError("low_var must be >= 0.")
        if high_var is not None and high_var < low_var:
            raise ValueError("high_var must be >= low_var.")
        self.low_var = low_var
        self.high_var = high_var
        self._kept_indices: list[int] = []
        self._variances: np.ndarray | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, X: Any) -> "BandpassFilter":
        """Compute per-feature variances and determine which features to keep.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2-D.")
        variances = np.var(X_arr, axis=0)
        self._variances = variances
        mask = variances >= self.low_var
        if self.high_var is not None:
            mask &= variances <= self.high_var
        self._kept_indices = [int(i) for i in np.where(mask)[0]]
        self._fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Return only the kept columns of *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_filtered : ndarray of shape (n_samples, n_kept)
        """
        self._require_fitted()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2-D.")
        if not self._kept_indices:
            return np.empty((X_arr.shape[0], 0), dtype=float)
        return X_arr[:, self._kept_indices]

    def fit_transform(self, X: Any) -> np.ndarray:
        """Fit and immediately transform *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_filtered : ndarray of shape (n_samples, n_kept)
        """
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # Result summary
    # ------------------------------------------------------------------

    def result(self) -> FilterResult:
        """Return a :class:`FilterResult` describing this filter's outcome.

        Returns
        -------
        FilterResult
        """
        self._require_fitted()
        return FilterResult(
            n_original=len(self._variances),  # type: ignore[arg-type]
            n_kept=len(self._kept_indices),
            kept_indices=list(self._kept_indices),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def kept_indices_(self) -> list[int]:
        """Indices of the retained features."""
        self._require_fitted()
        return list(self._kept_indices)

    @property
    def variances_(self) -> np.ndarray:
        """Per-feature variances computed during ``fit``."""
        self._require_fitted()
        return self._variances  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("BandpassFilter is not fitted yet. Call fit() first.")

    def __repr__(self) -> str:
        return (
            f"BandpassFilter(low_var={self.low_var}, high_var={self.high_var})"
        )
