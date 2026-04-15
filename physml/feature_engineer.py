"""Stage 81 — FeatureEngineer: automated feature engineering.

Generates new features from raw tabular data through four complementary
transformations:

* **polynomial** — degree-2 terms ``x_i²`` for every column.
* **interaction** — pairwise products ``x_i * x_j`` (optionally capped).
* **ratio** — safe ratios ``x_i / (x_j + ε)`` for selected column pairs.
* **log** — ``sign(x) * log(|x| + 1)`` for each column (handles negatives).

After generation, the top-*k* features are selected by mutual information
with the target (classification MI or regression MI depending on task).

Classes
-------
EngineeredFeatures
    Result of one feature-engineering pass.
FeatureEngineer
    Automated feature generator and selector.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EngineeredFeatures:
    """Result of one feature-engineering pass.

    Attributes
    ----------
    feature_names : list[str]
        Names of the *selected* features (original + engineered).
    n_original : int
        Number of original input features.
    n_generated : int
        Total number of candidate features generated before selection.
    n_selected : int
        Number of features retained after MI-based selection.
    mi_scores : list[float]
        Mutual-information scores for the selected features.
    elapsed_s : float
        Wall-clock duration of the engineering call.
    """

    feature_names: list[str]
    n_original: int
    n_generated: int
    n_selected: int
    mi_scores: list[float]
    elapsed_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_original": self.n_original,
            "n_generated": self.n_generated,
            "n_selected": self.n_selected,
            "feature_names": self.feature_names,
            "elapsed_s": round(self.elapsed_s, 4),
        }


class FeatureEngineer:
    """Automated feature generator and MI-based selector.

    Parameters
    ----------
    polynomial : bool, default True
        Add squared terms for each column.
    interactions : bool, default True
        Add pairwise interaction products.
    ratios : bool, default False
        Add pairwise ratio features.
    log_transform : bool, default True
        Add signed log-transformed columns.
    max_interactions : int, default 50
        Maximum number of interaction pairs to generate (prevents explosion).
    top_k : int or None, default 20
        Keep only the *top_k* features ranked by mutual information.
        If ``None`` all generated features are retained.
    task : str, default ``"classification"``
        ``"classification"`` or ``"regression"`` — controls MI estimator.
    random_state : int, default 0

    Example
    -------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from physml.feature_engineer import FeatureEngineer
    >>> X, y = make_classification(n_samples=200, n_features=6, random_state=0)
    >>> fe = FeatureEngineer(top_k=15)
    >>> X_new, result = fe.fit_transform(X, y)
    >>> X_new.shape[1] == result.n_selected
    True
    """

    def __init__(
        self,
        *,
        polynomial: bool = True,
        interactions: bool = True,
        ratios: bool = False,
        log_transform: bool = True,
        max_interactions: int = 50,
        top_k: int | None = 20,
        task: str = "classification",
        random_state: int = 0,
    ) -> None:
        self.polynomial = polynomial
        self.interactions = interactions
        self.ratios = ratios
        self.log_transform = log_transform
        self.max_interactions = int(max_interactions)
        self.top_k = int(top_k) if top_k is not None else None
        self.task = task
        self.random_state = int(random_state)

        self._selected_indices: list[int] = []
        self._all_names: list[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X: Any,
        y: Any,
        *,
        feature_names: list[str] | None = None,
    ) -> tuple[np.ndarray, EngineeredFeatures]:
        """Generate, rank, and select features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        feature_names : list[str] or None
            Optional names for the original columns.

        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_selected)
        result : EngineeredFeatures
        """
        t0 = time.time()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        base_names = (
            list(feature_names)
            if feature_names is not None and len(feature_names) == n_features
            else [f"x{i}" for i in range(n_features)]
        )

        # Build candidate feature matrix
        cols: list[np.ndarray] = [X]
        names: list[str] = list(base_names)

        if self.polynomial:
            cols.append(X ** 2)
            names += [f"{n}^2" for n in base_names]

        if self.log_transform:
            cols.append(np.sign(X) * np.log(np.abs(X) + 1.0))
            names += [f"log({n})" for n in base_names]

        if self.interactions:
            pairs = [
                (i, j)
                for i in range(n_features)
                for j in range(i + 1, n_features)
            ]
            pairs = pairs[: self.max_interactions]
            for i, j in pairs:
                cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
                names.append(f"{base_names[i]}*{base_names[j]}")

        if self.ratios:
            pairs = [
                (i, j)
                for i in range(n_features)
                for j in range(n_features)
                if i != j
            ]
            pairs = pairs[: self.max_interactions]
            for i, j in pairs:
                col = X[:, i] / (X[:, j] + 1e-8)
                cols.append(col.reshape(-1, 1))
                names.append(f"{base_names[i]}/{base_names[j]}")

        X_all = np.hstack([c if c.ndim == 2 else c.reshape(n_samples, -1) for c in cols])
        n_generated = X_all.shape[1]
        self._all_names = names

        # MI-based selection
        mi = self._mutual_info(X_all, y)
        if self.top_k is not None:
            k = min(self.top_k, n_generated)
            selected = np.argsort(mi)[::-1][:k]
        else:
            selected = np.arange(n_generated)

        self._selected_indices = selected.tolist()
        self._fitted = True

        X_new = X_all[:, selected]
        sel_names = [names[i] for i in selected]
        sel_mi = [round(float(mi[i]), 4) for i in selected]

        result = EngineeredFeatures(
            feature_names=sel_names,
            n_original=n_features,
            n_generated=n_generated,
            n_selected=len(selected),
            mi_scores=sel_mi,
            elapsed_s=time.time() - t0,
        )
        return X_new, result

    def transform(self, X: Any) -> np.ndarray:
        """Apply previously learned feature selection to new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_selected)
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform() first.")

        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        base_names_count = n_features
        cols: list[np.ndarray] = [X]

        if self.polynomial:
            cols.append(X ** 2)
        if self.log_transform:
            cols.append(np.sign(X) * np.log(np.abs(X) + 1.0))
        if self.interactions:
            pairs = [
                (i, j)
                for i in range(n_features)
                for j in range(i + 1, n_features)
            ]
            pairs = pairs[: self.max_interactions]
            for i, j in pairs:
                cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
        if self.ratios:
            pairs = [
                (i, j)
                for i in range(n_features)
                for j in range(n_features)
                if i != j
            ]
            pairs = pairs[: self.max_interactions]
            for i, j in pairs:
                col = X[:, i] / (X[:, j] + 1e-8)
                cols.append(col.reshape(-1, 1))

        X_all = np.hstack([c if c.ndim == 2 else c.reshape(n_samples, -1) for c in cols])
        return X_all[:, self._selected_indices]

    @property
    def selected_feature_names(self) -> list[str]:
        """Names of the selected features after fit_transform()."""
        if not self._fitted:
            return []
        return [self._all_names[i] for i in self._selected_indices]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mutual_info(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Approximate MI using sklearn estimators if available, else fallback."""
        try:
            if self.task == "classification":
                from sklearn.feature_selection import mutual_info_classif

                return mutual_info_classif(
                    X, y, random_state=self.random_state, discrete_features=False
                )
            else:
                from sklearn.feature_selection import mutual_info_regression

                return mutual_info_regression(
                    X, y, random_state=self.random_state
                )
        except Exception:
            # Fallback: absolute Pearson correlation
            y_f = y.astype(float)
            mi = np.zeros(X.shape[1])
            for j in range(X.shape[1]):
                corr = np.corrcoef(X[:, j], y_f)[0, 1]
                mi[j] = abs(float(corr)) if np.isfinite(corr) else 0.0
            return mi
