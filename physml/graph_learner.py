"""Stage 85 — GraphLearner: sparse feature-correlation graph.

Learns a sparse undirected graph where nodes are features and edges
represent Spearman rank correlations whose absolute value exceeds a
configurable threshold.

Classes
-------
GraphResult
    A single edge in the learned graph.
GraphLearner
    Learns and queries the feature-correlation graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class GraphResult:
    """A single weighted edge in the feature-correlation graph.

    Attributes
    ----------
    feature_a : int
        Index of the first feature (node).
    feature_b : int
        Index of the second feature (node).
    weight : float
        Spearman correlation coefficient for this edge.
    """

    feature_a: int
    feature_b: int
    weight: float

    def as_dict(self) -> dict[str, Any]:
        return {"feature_a": self.feature_a, "feature_b": self.feature_b, "weight": self.weight}

    def __repr__(self) -> str:
        return (
            f"GraphResult(feature_a={self.feature_a}, "
            f"feature_b={self.feature_b}, weight={self.weight:.4f})"
        )


class GraphLearner:
    """Learns a sparse feature-correlation graph from data.

    Parameters
    ----------
    threshold : float, default 0.3
        Minimum absolute Spearman correlation to include an edge.
    """

    def __init__(self, threshold: float = 0.3) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1].")
        self.threshold = threshold
        self._edges: list[GraphResult] = []
        self._n_features: int = 0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: Any, y: Any = None) -> "GraphLearner":
        """Compute pairwise Spearman correlations and build the edge list.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        from scipy.stats import spearmanr

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2-D.")
        n_samples, n_features = X_arr.shape
        self._n_features = n_features
        self._edges = []

        if n_samples < 3 or n_features < 2:
            self._fitted = True
            return self

        corr, _ = spearmanr(X_arr)
        # spearmanr returns a scalar when n_features == 2
        if n_features == 2:
            corr = np.array([[1.0, float(corr)], [float(corr), 1.0]])
        else:
            corr = np.asarray(corr)

        for i in range(n_features):
            for j in range(i + 1, n_features):
                w = float(corr[i, j])
                if abs(w) >= self.threshold:
                    self._edges.append(GraphResult(feature_a=i, feature_b=j, weight=w))

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_graph(self) -> list[GraphResult]:
        """Return all edges in the learned graph.

        Returns
        -------
        list[GraphResult]
            Edges sorted by descending absolute weight.
        """
        self._require_fitted()
        return sorted(self._edges, key=lambda e: abs(e.weight), reverse=True)

    def most_connected(self, n: int = 5) -> list[int]:
        """Return the *n* features with the highest degree (edge count).

        Parameters
        ----------
        n : int, default 5

        Returns
        -------
        list[int]
            Feature indices ordered by degree (descending).
        """
        self._require_fitted()
        degree: dict[int, int] = {}
        for edge in self._edges:
            degree[edge.feature_a] = degree.get(edge.feature_a, 0) + 1
            degree[edge.feature_b] = degree.get(edge.feature_b, 0) + 1
        ranked = sorted(degree.items(), key=lambda kv: kv[1], reverse=True)
        return [feat for feat, _ in ranked[:n]]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_edges(self) -> int:
        """Number of edges in the learned graph."""
        return len(self._edges)

    @property
    def n_features(self) -> int:
        """Number of features (nodes) in the graph."""
        return self._n_features

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("GraphLearner is not fitted yet. Call fit() first.")

    def __repr__(self) -> str:
        return (
            f"GraphLearner(threshold={self.threshold}, "
            f"n_edges={self.n_edges if self._fitted else 'unfitted'})"
        )
