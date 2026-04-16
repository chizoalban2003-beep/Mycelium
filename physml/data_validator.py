"""Stage 88 — DataValidator: input data quality checks.

Inspects a feature matrix (and optional label vector) for common data
quality issues: missing values, constant features, duplicate rows, and
infinite values.

Classes
-------
ValidationReport
    Summary of all quality checks performed.
DataValidator
    Runs a configurable set of data quality checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ValidationReport:
    """Summary of data quality checks.

    Attributes
    ----------
    n_rows : int
        Number of samples in the validated array.
    n_cols : int
        Number of features in the validated array.
    missing_count : int
        Total number of NaN values detected.
    constant_features : list[int]
        Column indices whose variance is exactly zero.
    duplicate_rows : int
        Number of rows that are exact duplicates of an earlier row.
    infinite_count : int
        Total number of ±inf values detected.
    is_valid : bool
        ``True`` when ``missing_count == 0`` and ``infinite_count == 0``.
    """

    n_rows: int
    n_cols: int
    missing_count: int
    constant_features: list[int]
    duplicate_rows: int
    infinite_count: int
    is_valid: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "missing_count": self.missing_count,
            "constant_features": list(self.constant_features),
            "duplicate_rows": self.duplicate_rows,
            "infinite_count": self.infinite_count,
            "is_valid": self.is_valid,
        }

    def __repr__(self) -> str:
        return (
            f"ValidationReport(n_rows={self.n_rows}, n_cols={self.n_cols}, "
            f"missing_count={self.missing_count}, "
            f"constant_features={self.constant_features}, "
            f"duplicate_rows={self.duplicate_rows}, "
            f"infinite_count={self.infinite_count}, "
            f"is_valid={self.is_valid})"
        )


class DataValidator:
    """Validates input data quality.

    Parameters
    ----------
    check_missing : bool, default True
        Whether to count NaN values.
    check_constant : bool, default True
        Whether to detect constant (zero-variance) features.
    check_duplicates : bool, default True
        Whether to count duplicate rows.
    check_infinite : bool, default True
        Whether to count infinite values.
    """

    def __init__(
        self,
        check_missing: bool = True,
        check_constant: bool = True,
        check_duplicates: bool = True,
        check_infinite: bool = True,
    ) -> None:
        self.check_missing = check_missing
        self.check_constant = check_constant
        self.check_duplicates = check_duplicates
        self.check_infinite = check_infinite

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, X: Any, y: Any = None) -> ValidationReport:
        """Run all enabled checks on *X* (and optionally *y*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix to validate.
        y : array-like of shape (n_samples,), optional
            Labels (currently unused in checks, reserved for future use).

        Returns
        -------
        ValidationReport
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2:
            raise ValueError("X must be 1-D or 2-D.")

        n_rows, n_cols = X_arr.shape

        # Missing values
        missing_count = int(np.sum(np.isnan(X_arr))) if self.check_missing else 0

        # Constant features (mask out NaN/inf for variance computation)
        constant_features: list[int] = []
        if self.check_constant:
            finite_mask = np.isfinite(X_arr)
            for j in range(n_cols):
                col_vals = X_arr[:, j][finite_mask[:, j]]
                if len(col_vals) == 0 or np.var(col_vals) == 0.0:
                    constant_features.append(j)

        # Duplicate rows (compare only finite rows to avoid NaN != NaN issues)
        duplicate_rows = 0
        if self.check_duplicates and n_rows > 1:
            # Use a row-hash approach that handles NaNs by treating them as a
            # sentinel value
            sentinel = np.nanmax(np.abs(X_arr[np.isfinite(X_arr)]), initial=0) + 1e9
            X_no_nan = np.where(np.isnan(X_arr), sentinel, X_arr)
            X_no_nan = np.where(np.isinf(X_no_nan), sentinel * 2, X_no_nan)
            seen: set[bytes] = set()
            for row in X_no_nan:
                key = row.tobytes()
                if key in seen:
                    duplicate_rows += 1
                else:
                    seen.add(key)

        # Infinite values
        infinite_count = int(np.sum(np.isinf(X_arr))) if self.check_infinite else 0

        is_valid = (missing_count == 0) and (infinite_count == 0)

        return ValidationReport(
            n_rows=n_rows,
            n_cols=n_cols,
            missing_count=missing_count,
            constant_features=constant_features,
            duplicate_rows=duplicate_rows,
            infinite_count=infinite_count,
            is_valid=is_valid,
        )

    def __repr__(self) -> str:
        return (
            f"DataValidator(check_missing={self.check_missing}, "
            f"check_constant={self.check_constant}, "
            f"check_duplicates={self.check_duplicates}, "
            f"check_infinite={self.check_infinite})"
        )
