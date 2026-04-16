"""Stage 86 — ClusterEngine: KMeans-based unsupervised clustering.

Wraps scikit-learn's KMeans with a convenience ``report()`` method that
computes inertia and silhouette score in one call.

Classes
-------
ClusterReport
    Summary statistics for a fitted clustering.
ClusterEngine
    KMeans clustering with one-call reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ClusterReport:
    """Summary of a fitted KMeans clustering.

    Attributes
    ----------
    n_clusters : int
        Number of clusters requested.
    inertia : float
        Sum of squared distances of samples to their closest centroid.
    silhouette_score : float
        Mean silhouette coefficient across all samples (in [-1, 1]).
        Set to ``float("nan")`` when the score cannot be computed
        (e.g., only one cluster or fewer samples than clusters).
    """

    n_clusters: int
    inertia: float
    silhouette_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "inertia": self.inertia,
            "silhouette_score": self.silhouette_score,
        }

    def __repr__(self) -> str:
        return (
            f"ClusterReport(n_clusters={self.n_clusters}, "
            f"inertia={self.inertia:.4f}, "
            f"silhouette_score={self.silhouette_score:.4f})"
        )


class ClusterEngine:
    """Unsupervised KMeans clustering engine.

    Parameters
    ----------
    n_clusters : int, default 3
        Number of clusters to form.
    random_state : int or None, default 0
        Random seed for reproducibility.
    max_iter : int, default 300
        Maximum KMeans iterations per run.
    n_init : int, default 10
        Number of times KMeans is run with different centroid seeds.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int | None = 0,
        max_iter: int = 300,
        n_init: int = 10,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self._kmeans: Any = None
        self._labels: np.ndarray | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting / prediction
    # ------------------------------------------------------------------

    def fit(self, X: Any) -> "ClusterEngine":
        """Fit KMeans on *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        from sklearn.cluster import KMeans

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2-D.")

        n_clusters = min(self.n_clusters, X_arr.shape[0])
        self._kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=self.n_init,
        )
        self._kmeans.fit(X_arr)
        self._labels = self._kmeans.labels_
        self._X_fit = X_arr
        self._fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Assign cluster labels to new samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        self._require_fitted()
        X_arr = np.asarray(X, dtype=float)
        return self._kmeans.predict(X_arr)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> ClusterReport:
        """Compute and return a :class:`ClusterReport` for the fitted clustering.

        Returns
        -------
        ClusterReport
        """
        self._require_fitted()
        from sklearn.metrics import silhouette_score

        inertia = float(self._kmeans.inertia_)
        n_unique = len(np.unique(self._labels))
        if n_unique < 2 or len(self._labels) <= n_unique:
            sil = float("nan")
        else:
            try:
                sil = float(silhouette_score(self._X_fit, self._labels))
            except Exception:
                sil = float("nan")

        return ClusterReport(
            n_clusters=self._kmeans.n_clusters,
            inertia=inertia,
            silhouette_score=sil,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def labels_(self) -> np.ndarray:
        """Cluster label for each training sample."""
        self._require_fitted()
        return self._labels  # type: ignore[return-value]

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Coordinates of cluster centroids."""
        self._require_fitted()
        return self._kmeans.cluster_centers_

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("ClusterEngine is not fitted yet. Call fit() first.")

    def __repr__(self) -> str:
        return (
            f"ClusterEngine(n_clusters={self.n_clusters}, "
            f"random_state={self.random_state})"
        )
