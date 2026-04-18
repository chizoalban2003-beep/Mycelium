"""Stage 19 — Federated / privacy-preserving learning.

:class:`FederatedMyceliumAgent` coordinates multiple local
:class:`~physml.mycelium_agent.MyceliumAgent` nodes using **FedAvg** —
federated averaging of MLP weights.  Each node trains on its own private
data; only the flat model weight *deltas* (difference from the global model)
are aggregated across nodes.

.. note::
    **Privacy caveat:** each :class:`~physml.mycelium_agent.MyceliumAgent`
    stores its training data in memory by default (for transductive prediction).
    Raw features therefore *do* remain in process memory on the node unless you
    pass ``store_training_data=False`` in ``predictor_kwargs`` — in which case
    transductive re-scoring is disabled.  Weight deltas alone do not guarantee
    differential privacy; combine with :class:`~physml.privacy.DifferentialPrivacyEngine`
    for ε,δ-DP guarantees.

This mirrors the fungal network metaphor beautifully: each mycelial node
grows autonomously but periodically shares chemical signals (weight deltas)
with the broader network to collectively improve.

Architecture
------------
::

    FederatedMyceliumAgent
    ├── node "site_A" ← MyceliumAgent (private data A)
    ├── node "site_B" ← MyceliumAgent (private data B)
    └── node "site_C" ← MyceliumAgent (private data C)

    federation.aggregate()
        → averages weight deltas from all nodes
        → broadcasts updated global weights to all nodes

Usage
-----
::

    from physml.federated import FederatedMyceliumAgent

    fed = FederatedMyceliumAgent()

    # Each node trains on its own data
    fed.add_node("hospital_A", X_A, y_A)
    fed.add_node("hospital_B", X_B, y_B)

    # Federated averaging round
    fed.aggregate()

    # Get the global model for inference
    global_agent = fed.global_agent()
    action = global_agent.observe(X_new)
"""

from __future__ import annotations

from typing import Any

import numpy as np


class FederatedMyceliumAgent:
    """Coordinator for federated learning across multiple myco nodes.

    Parameters
    ----------
    n_rounds : int, default 1
        Number of FedAvg rounds performed on each call to
        :meth:`aggregate`.
    predictor_kwargs : dict or None
        Passed to each new :class:`~physml.mycelium_agent.MyceliumAgent`
        when a node is created without a pre-built predictor.
    calibrate : bool, default False
        Whether to run temperature calibration on each node's local fit.
        Disabled by default to keep federation rounds fast.
    """

    def __init__(
        self,
        *,
        n_rounds: int = 1,
        predictor_kwargs: dict[str, Any] | None = None,
        calibrate: bool = False,
    ) -> None:
        self.n_rounds = int(n_rounds)
        self._predictor_kwargs = dict(predictor_kwargs or {})
        self.calibrate = bool(calibrate)
        self._nodes: dict[str, Any] = {}  # name → MyceliumAgent
        self._global_weights: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(
        self,
        name: str,
        X: Any,
        y: Any,
        agent: Any = None,
    ) -> "FederatedMyceliumAgent":
        """Create and fit a federated node.

        Parameters
        ----------
        name : str
            Unique node identifier.
        X : array-like of shape (n_samples, n_features)
            Local training data (stays on this node — never shared).
        y : array-like of shape (n_samples,)
            Local labels.
        agent : MyceliumAgent or None
            Pre-built agent.  When ``None``, a fresh agent is created
            with ``calibrate=self.calibrate`` and ``predictor_kwargs``.

        Returns
        -------
        self
        """
        from physml.mycelium_agent import MyceliumAgent

        if agent is None:
            agent = MyceliumAgent(
                calibrate=self.calibrate,
                predictor_kwargs=self._predictor_kwargs or None,
            )
        X_arr = np.atleast_2d(X)
        y_arr = np.atleast_1d(y)
        agent.fit(X_arr, y_arr)
        self._nodes[name] = agent

        # Initialise global weights from the first node
        if self._global_weights is None:
            self._global_weights = self._extract_weights(agent)

        return self

    def remove_node(self, name: str) -> "FederatedMyceliumAgent":
        """Remove a node from the federation."""
        self._nodes.pop(name, None)
        return self

    # ------------------------------------------------------------------
    # Aggregation (FedAvg)
    # ------------------------------------------------------------------

    def aggregate(self) -> "FederatedMyceliumAgent":
        """Run one round of FedAvg across all registered nodes.

        Each node's weight delta (local_weights - global_weights) is
        computed and averaged.  The averaged delta is added to the global
        model and broadcast back to all nodes.

        Returns
        -------
        self
        """
        if not self._nodes:
            return self
        if self._global_weights is None:
            node = next(iter(self._nodes.values()))
            self._global_weights = self._extract_weights(node)

        for _ in range(self.n_rounds):
            deltas: list[dict[str, np.ndarray]] = []
            for agent in self._nodes.values():
                local_w = self._extract_weights(agent)
                if local_w is None:
                    continue
                delta = {}
                for key in local_w:
                    if key in self._global_weights:
                        try:
                            delta[key] = local_w[key] - self._global_weights[key]
                        except Exception:
                            pass
                if delta:
                    deltas.append(delta)

            if not deltas:
                break

            # Average the deltas
            avg_delta: dict[str, np.ndarray] = {}
            for key in deltas[0]:
                try:
                    avg_delta[key] = np.mean(
                        [d[key] for d in deltas if key in d], axis=0
                    )
                except Exception:
                    pass

            # Update global weights
            for key, delta in avg_delta.items():
                self._global_weights[key] = self._global_weights[key] + delta

            # Broadcast updated weights to all nodes
            for agent in self._nodes.values():
                self._apply_weights(agent, self._global_weights)

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def global_agent(self) -> Any:
        """Return the first node's agent (with globally-averaged weights) for inference.

        If no nodes have been registered, raises ``RuntimeError``.

        Returns
        -------
        MyceliumAgent
        """
        if not self._nodes:
            raise RuntimeError(
                "No nodes registered.  Call add_node() first."
            )
        return next(iter(self._nodes.values()))

    def node_agent(self, name: str) -> Any:
        """Return the agent for a specific node.

        Parameters
        ----------
        name : str

        Returns
        -------
        MyceliumAgent
        """
        if name not in self._nodes:
            raise KeyError(f"Node {name!r} not found.")
        return self._nodes[name]

    def list_nodes(self) -> list[str]:
        """Return the names of all registered nodes."""
        return list(self._nodes.keys())

    # ------------------------------------------------------------------
    # Weight extraction / application helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_weights(agent: Any) -> dict[str, np.ndarray] | None:
        """Extract flat MLP weight arrays from an agent's predictor."""
        predictor = getattr(agent, "_predictor", None)
        if predictor is None:
            return None

        # Try neural engine directly
        mlp = _find_mlp(predictor)
        if mlp is None:
            return None

        weights: dict[str, np.ndarray] = {}
        coefs = getattr(mlp, "coefs_", None)
        intercepts = getattr(mlp, "intercepts_", None)
        if coefs is None:
            return None
        for i, (c, b) in enumerate(zip(coefs, intercepts or [])):
            weights[f"coef_{i}"] = c.copy()
            weights[f"intercept_{i}"] = b.copy()
        return weights

    @staticmethod
    def _apply_weights(agent: Any, weights: dict[str, np.ndarray]) -> None:
        """Apply flat weight arrays back to an agent's MLP."""
        predictor = getattr(agent, "_predictor", None)
        if predictor is None:
            return
        mlp = _find_mlp(predictor)
        if mlp is None:
            return
        coefs = getattr(mlp, "coefs_", None)
        intercepts = getattr(mlp, "intercepts_", None)
        if coefs is None:
            return
        for i in range(len(coefs)):
            key_c = f"coef_{i}"
            key_b = f"intercept_{i}"
            if key_c in weights and weights[key_c].shape == coefs[i].shape:
                mlp.coefs_[i] = weights[key_c].copy()
            if (
                intercepts is not None
                and key_b in weights
                and weights[key_b].shape == intercepts[i].shape
            ):
                mlp.intercepts_[i] = weights[key_b].copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_mlp(predictor: Any) -> Any:
    """Recursively locate a fitted sklearn MLP inside a predictor/engine."""
    # Direct MLP
    for attr in ("coefs_",):
        if hasattr(predictor, attr):
            return predictor

    # NeuralPhysicsEngine stores the MLP at ._clf or ._reg
    for attr in ("_clf", "_reg", "_mlp", "estimator_"):
        sub = getattr(predictor, attr, None)
        if sub is not None and hasattr(sub, "coefs_"):
            return sub

    # PhysicsPredictor wraps NeuralPhysicsEngine at ._engine or ._neural_engine
    for attr in ("_engine", "_neural_engine", "engine_"):
        sub = getattr(predictor, attr, None)
        if sub is not None:
            result = _find_mlp(sub)
            if result is not None:
                return result

    return None
