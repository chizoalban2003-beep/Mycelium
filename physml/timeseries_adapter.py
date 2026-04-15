"""Stage 77 — TimeSeriesAdapter: time-series → tabular feature engineering.

Converts univariate or multivariate time-series data into a flat tabular
feature matrix suitable for the PhysML / MyceliumAgent pipeline by generating
lag features, rolling-window statistics, and first differences.

Classes
-------
AdapterResult
    Wrapper returned by :meth:`TimeSeriesAdapter.transform`.
TimeSeriesAdapter
    Transforms sequential data into a lagged tabular feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AdapterResult:
    """Wrapper for the output of :class:`TimeSeriesAdapter`.

    Attributes
    ----------
    X_transformed : np.ndarray
        Lagged feature matrix of shape ``(n_valid, n_features_out)``.
    y_aligned : np.ndarray or None
        Target values aligned to *X_transformed* (same length), or None if
        no target was provided.
    feature_names : list[str]
        Names of the generated feature columns.
    n_dropped : int
        Number of leading rows dropped to accommodate the maximum lag.
    """

    X_transformed: np.ndarray
    y_aligned: "np.ndarray | None"
    feature_names: list[str]
    n_dropped: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.X_transformed.shape),
            "n_features_out": len(self.feature_names),
            "n_dropped": self.n_dropped,
            "feature_names": self.feature_names,
        }


class TimeSeriesAdapter:
    """Convert sequential data into a tabular feature matrix.

    For each input column the adapter generates:

    * **Lag features**: ``col_lag_1, col_lag_2, …, col_lag_n_lags``
    * **Rolling mean**: ``col_roll_mean_w`` for window *w* in *windows*
    * **Rolling std**: ``col_roll_std_w``
    * **First difference**: ``col_diff_1``

    The first ``max_lag`` rows are dropped because they cannot be fully
    populated; the returned ``AdapterResult`` records how many were removed.

    Parameters
    ----------
    n_lags : int, default 3
        Number of lag features to generate per column.
    windows : list[int] or None
        Rolling window sizes.  Defaults to ``[3]``.
    include_diff : bool, default True
        Whether to add a first-difference feature per column.
    feature_names : list[str] or None
        Optional names for the input columns.

    Example
    -------
    >>> import numpy as np
    >>> from physml.timeseries_adapter import TimeSeriesAdapter
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 2))
    >>> adapter = TimeSeriesAdapter(n_lags=2, windows=[3])
    >>> result = adapter.transform(X)
    >>> result.X_transformed.shape[0] == 100 - adapter._max_drop
    True
    """

    def __init__(
        self,
        *,
        n_lags: int = 3,
        windows: list[int] | None = None,
        include_diff: bool = True,
        feature_names: list[str] | None = None,
    ) -> None:
        self.n_lags = max(1, int(n_lags))
        self.windows: list[int] = [int(w) for w in (windows or [3])]
        self.include_diff = bool(include_diff)
        self.feature_names = feature_names

        self._max_drop: int = 0  # set during transform()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(
        self,
        X: Any,
        y: Any | None = None,
    ) -> AdapterResult:
        """Build the lagged feature matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Time-series data (rows = time steps).
        y : array-like of shape (n_samples,) or None

        Returns
        -------
        AdapterResult
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Resolve column names
        if self.feature_names is not None and len(self.feature_names) == n_features:
            col_names = list(self.feature_names)
        else:
            col_names = [f"col{i}" for i in range(n_features)]

        # Determine maximum drop (to align all features)
        max_window = max(self.windows) if self.windows else 1
        max_drop = max(self.n_lags, max_window - 1)
        self._max_drop = max_drop

        features: list[np.ndarray] = []
        names: list[str] = []

        for col_idx, col_name in enumerate(col_names):
            series = X[:, col_idx]

            # Lag features
            for lag in range(1, self.n_lags + 1):
                lagged = np.full(n_samples, np.nan)
                lagged[lag:] = series[:-lag]
                features.append(lagged)
                names.append(f"{col_name}_lag_{lag}")

            # Rolling mean & std
            for w in self.windows:
                roll_mean = np.full(n_samples, np.nan)
                roll_std = np.full(n_samples, np.nan)
                for t in range(w - 1, n_samples):
                    window_vals = series[t - w + 1 : t + 1]
                    roll_mean[t] = window_vals.mean()
                    roll_std[t] = window_vals.std(ddof=0)
                features.append(roll_mean)
                names.append(f"{col_name}_roll_mean_{w}")
                features.append(roll_std)
                names.append(f"{col_name}_roll_std_{w}")

            # First difference
            if self.include_diff:
                diff = np.full(n_samples, np.nan)
                diff[1:] = np.diff(series)
                features.append(diff)
                names.append(f"{col_name}_diff_1")

        # Stack and drop leading NaN rows
        mat = np.column_stack(features)
        valid_start = max_drop
        X_out = mat[valid_start:, :]

        y_out = None
        if y is not None:
            y_arr = np.asarray(y)
            y_out = y_arr[valid_start:]

        return AdapterResult(
            X_transformed=X_out,
            y_aligned=y_out,
            feature_names=names,
            n_dropped=valid_start,
        )

    def fit_transform(self, X: Any, y: Any | None = None) -> AdapterResult:
        """Convenience alias for :meth:`transform` (stateless adapter)."""
        return self.transform(X, y)

    def n_features_out(self, n_features_in: int) -> int:
        """Compute the number of output features for *n_features_in* input columns."""
        per_col = self.n_lags + 2 * len(self.windows) + int(self.include_diff)
        return n_features_in * per_col
