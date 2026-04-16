"""Stage 34 — Self-supervised masked-feature pretraining for NeuralPhysicsEngine.

Provides two public entry-points:

* :func:`pretrain_neural_engine` — pretrain a :class:`~physml.neural_engine.NeuralPhysicsEngine`
  on unlabelled data using a masked-feature reconstruction objective.
* :func:`pretrain_mycelium` — convenience wrapper that finds the engine inside a
  :class:`~physml.mycelium_agent.MyceliumAgent` and calls :func:`pretrain_neural_engine`.

No labels are required: the network learns to reconstruct randomly-masked
features, producing useful initial weights before labelled data arrives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.neural_network import MLPRegressor

if TYPE_CHECKING:  # pragma: no cover
    from physml.mycelium_agent import MyceliumAgent
    from physml.neural_engine import NeuralPhysicsEngine


def pretrain_neural_engine(
    engine: "NeuralPhysicsEngine",
    X_unlabelled: np.ndarray,
    mask_fraction: float = 0.15,
    n_epochs: int = 10,
    batch_size: int = 32,
    random_state: int | None = 42,
) -> "NeuralPhysicsEngine":
    """Self-supervised masked-feature pretraining for *engine*.

    Randomly masks ``mask_fraction`` of features per sample and trains a
    reconstruction MLP to predict the original values.  The first-layer
    weights of the reconstruction network are then transferred to *engine*
    if its MLP is already fitted and the shapes are compatible.

    Parameters
    ----------
    engine : NeuralPhysicsEngine
    X_unlabelled : np.ndarray, shape (n_samples, n_features)
    mask_fraction : float, default 0.15
        Fraction of features to zero-out per sample during training.
    n_epochs : int, default 10
        Number of training epochs for the reconstruction MLP.
    batch_size : int, default 32
        Mini-batch size (used to derive ``max_iter`` for MLPRegressor).
    random_state : int or None, default 42

    Returns
    -------
    NeuralPhysicsEngine
        The same *engine* object, with ``pretrained_coefs_`` attribute set.
    """
    rng = np.random.default_rng(random_state)
    X = np.atleast_2d(np.asarray(X_unlabelled, dtype=np.float32))
    n, d = X.shape

    # Build corrupted input / target pairs
    mask = rng.random(X.shape) < mask_fraction
    X_corrupted = X.copy()
    X_corrupted[mask] = 0.0
    X_target = X.copy()  # reconstruct original from corrupted

    # Compute max_iter so total gradient steps ≈ n_epochs * ceil(n / batch_size)
    steps_per_epoch = max(1, int(np.ceil(n / batch_size)))
    max_iter = max(10, n_epochs * steps_per_epoch)

    hidden1 = min(256, d * 2)
    hidden2 = min(128, d)
    recon_mlp = MLPRegressor(
        hidden_layer_sizes=(hidden1, hidden2),
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=False,
    )
    recon_mlp.fit(X_corrupted, X_target)

    # Store pretrained weights
    engine.pretrained_coefs_ = recon_mlp.coefs_

    # Transfer first-layer weights into engine's MLP if shapes match
    mlp = getattr(engine, "mlp_", None)
    if mlp is not None and hasattr(mlp, "coefs_") and len(mlp.coefs_) > 0:
        if mlp.coefs_[0].shape == recon_mlp.coefs_[0].shape:
            mlp.coefs_[0] = recon_mlp.coefs_[0].copy()

    return engine


def pretrain_mycelium(
    agent: "MyceliumAgent",
    X_unlabelled: np.ndarray,
    **kwargs: Any,
) -> "MyceliumAgent":
    """Pretrain the :class:`~physml.neural_engine.NeuralPhysicsEngine` inside *agent*.

    Locates the engine, calls :func:`pretrain_neural_engine`, and sets
    ``agent._pretrained = True`` and ``agent._pretrain_coefs_``.

    Parameters
    ----------
    agent : MyceliumAgent
    X_unlabelled : np.ndarray, shape (n_samples, n_features)
    **kwargs
        Forwarded to :func:`pretrain_neural_engine`.

    Returns
    -------
    MyceliumAgent
    """
    from physml.neural_engine import NeuralPhysicsEngine

    engine = _find_engine(agent)

    if engine is None:
        # Create a standalone engine for pretraining and attach it
        engine = NeuralPhysicsEngine()
        agent._pretrain_engine = engine

    pretrain_neural_engine(engine, X_unlabelled, **kwargs)
    agent._pretrained = True
    agent._pretrain_coefs_ = engine.pretrained_coefs_

    return agent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_engine(agent: "MyceliumAgent") -> "NeuralPhysicsEngine | None":
    """Search for a NeuralPhysicsEngine inside *agent*."""
    from physml.neural_engine import NeuralPhysicsEngine

    predictor = getattr(agent, "_predictor", None)
    if predictor is None:
        return None

    if isinstance(predictor, NeuralPhysicsEngine):
        return predictor

    for attr in ("_engine", "engine", "_neural", "neural"):
        candidate = getattr(predictor, attr, None)
        if isinstance(candidate, NeuralPhysicsEngine):
            return candidate

    return None
