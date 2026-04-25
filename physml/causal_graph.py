"""Stage 75 — CausalGraph: correlation-based causal discovery.

Learns a directed causal skeleton from observational data using pairwise
partial-correlation thresholds (skeleton discovery) and variance-asymmetry
heuristics for edge orientation (à la LiNGAM) — no heavy dependencies
required.

Classes
-------
CausalEdge
    A directed or undirected edge in the discovered causal graph.
CausalGraph
    Discovers and stores a causal skeleton; supports counterfactual queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CausalEdge:
    """One edge in the discovered causal graph.

    Attributes
    ----------
    source : str
        Name of the source (cause) variable.
    target : str
        Name of the target (effect) variable.
    weight : float
        Absolute correlation strength of this link.
    directed : bool
        True when the edge orientation has been determined.
    """

    source: str
    target: str
    weight: float
    directed: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": round(self.weight, 4),
            "directed": self.directed,
        }

    def __repr__(self) -> str:
        arrow = "→" if self.directed else "—"
        return f"CausalEdge({self.source!r} {arrow} {self.target!r}, w={self.weight:.3f})"


class CausalGraph:
    """Discover a simple directed causal graph from tabular data.

    Uses pairwise Pearson-correlation thresholds to build an undirected
    skeleton, then orients each edge by comparing the variance of the
    post-nonlinear residuals in both directions: the *causal* direction
    typically yields a smaller residual variance (ANM heuristic).

    Parameters
    ----------
    threshold : float, default 0.1
        Minimum absolute correlation to include an edge in the skeleton.
    feature_names : list[str] or None
        Human-readable names for the columns.  If None, ``"x0", "x1", …``
        are used.
    include_target : bool, default True
        When *y* is provided to :meth:`discover`, treat it as a named node
        (``"y"``).
    random_state : int, default 0

    Example
    -------
    >>> import numpy as np
    >>> from physml.causal_graph import CausalGraph
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 4))
    >>> y = X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(200) * 0.1
    >>> cg = CausalGraph(threshold=0.2)
    >>> edges = cg.discover(X, y)
    >>> len(edges) >= 1
    True
    """

    def __init__(
        self,
        *,
        threshold: float = 0.1,
        feature_names: list[str] | None = None,
        include_target: bool = True,
        random_state: int = 0,
    ) -> None:
        self.threshold = float(threshold)
        self.feature_names = feature_names
        self.include_target = include_target
        self.random_state = int(random_state)

        self._edges: list[CausalEdge] = []
        self._node_names: list[str] = []
        self._adjacency: dict[str, list[str]] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(
        self,
        X: Any,
        y: Any | None = None,
    ) -> list[CausalEdge]:
        """Discover the causal skeleton and orient edges.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or None
            Optional target variable included as a node named ``"y"``.

        Returns
        -------
        list[CausalEdge]
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        # Build column names
        if self.feature_names is not None and len(self.feature_names) == n_features:
            names = list(self.feature_names)
        else:
            names = [f"x{i}" for i in range(n_features)]

        # Optionally append target
        if y is not None and self.include_target:
            y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
            data = np.hstack([X, y_arr])
            all_names = names + ["y"]
        else:
            data = X
            all_names = names

        self._node_names = all_names
        n_nodes = len(all_names)

        # Step 1 — correlation skeleton
        corr = np.corrcoef(data, rowvar=False)
        np.nan_to_num(corr, copy=False, nan=0.0)

        skeleton: list[tuple[int, int, float]] = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                w = abs(float(corr[i, j]))
                if w >= self.threshold:
                    skeleton.append((i, j, w))

        # Step 2 — orient edges via residual-variance heuristic (ANM)
        self._edges = []
        for i, j, w in skeleton:
            src, tgt = self._orient(data[:, i], data[:, j], i, j, all_names)
            self._edges.append(CausalEdge(source=src, target=tgt, weight=w))

        # Build adjacency map
        self._adjacency = {n: [] for n in all_names}
        for e in self._edges:
            self._adjacency[e.source].append(e.target)

        self._fitted = True
        return list(self._edges)

    def parents(self, node: str) -> list[str]:
        """Return all direct causes of *node*."""
        return [e.source for e in self._edges if e.target == node]

    def children(self, node: str) -> list[str]:
        """Return all direct effects of *node*."""
        return self._adjacency.get(node, [])

    def counterfactual(
        self,
        X: Any,
        interventions: dict[str, float],
    ) -> np.ndarray:
        """Estimate interventional mean for all other nodes.

        Performs a *do-calculus*-style intervention by fixing the named
        features to constant values and propagating their linear effects
        through the discovered edges (linearised structural equations).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Observational data.
        interventions : dict[str, float]
            Mapping ``{feature_name: value}`` for intervened variables.

        Returns
        -------
        np.ndarray of shape (n_nodes,)
            Post-intervention column means (original + delta from
            interventions).
        """
        if not self._fitted:
            raise RuntimeError("Call discover() before counterfactual().")

        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]

        if self.feature_names is not None and len(self.feature_names) == n_features:
            names = list(self.feature_names)
        else:
            names = [f"x{i}" for i in range(n_features)]

        # Build full data array (without y, since it may not be available)
        means = X.mean(axis=0)

        # Apply interventions
        result = dict(zip(names, means.tolist()))
        for name, val in interventions.items():
            if name in result:
                result[name] = float(val)

        # Linear propagation through edges (one pass in topological order)
        for e in self._edges:
            if e.source in interventions and e.target in result:
                # Shift target mean by correlation-weighted delta
                original = X.mean(axis=0)
                src_idx = self._node_names.index(e.source)
                tgt_idx = self._node_names.index(e.target)

                if tgt_idx < n_features and src_idx < n_features:
                    beta = (
                        np.cov(X[:, src_idx], X[:, tgt_idx])[0, 1]
                        / (np.var(X[:, src_idx]) + 1e-9)
                    )
                    delta = (interventions[e.source] - original[src_idx]) * beta
                    result[e.target] = result[e.target] + delta

        return np.array(list(result.values()))

    def summary(self) -> dict[str, Any]:
        """Return a human-readable summary dict."""
        return {
            "n_nodes": len(self._node_names),
            "n_edges": len(self._edges),
            "nodes": self._node_names,
            "edges": [e.as_dict() for e in self._edges],
            "threshold": self.threshold,
        }

    @property
    def edges(self) -> list[CausalEdge]:
        """List of discovered causal edges."""
        return list(self._edges)

    @property
    def nodes(self) -> list[str]:
        """List of node names."""
        return list(self._node_names)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _orient(
        self,
        xi: np.ndarray,
        xj: np.ndarray,
        i: int,
        j: int,
        names: list[str],
    ) -> tuple[str, str]:
        """Orient edge i–j using residual-variance asymmetry (ANM heuristic).

        Fit a linear model in each direction and compare residual variances.
        The direction with *smaller* residual variance is chosen as causal.
        """
        # xi → xj
        beta_ij = np.cov(xi, xj)[0, 1] / (np.var(xi) + 1e-9)
        resid_ij = np.var(xj - beta_ij * xi)

        # xj → xi
        beta_ji = np.cov(xj, xi)[0, 1] / (np.var(xj) + 1e-9)
        resid_ji = np.var(xi - beta_ji * xj)

        if resid_ij <= resid_ji:
            return names[i], names[j]
        return names[j], names[i]
