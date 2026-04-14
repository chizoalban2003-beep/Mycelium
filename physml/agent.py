"""Stage 4 + 5 — Autonomous agent loop and big-data streaming.

``PhysicsAgent`` wraps a :class:`PhysicsPredictor` with an uncertainty gate,
a decide/abstain policy, and an online-learning loop.  Users observe
predictions, provide feedback (true labels), and the agent adapts in
real-time via :meth:`~PhysicsPredictor.partial_fit`.

``DataStream`` (Stage 5) provides mini-batch streaming over any iterable of
``(X_chunk, y_chunk)`` pairs, enabling training on datasets that do not fit
in memory.

Usage — basic agent loop
------------------------
::

    from physml import PhysicsPredictor
    from physml.agent import PhysicsAgent, AgentAction

    predictor = PhysicsPredictor(backend="neural", n_cycles=20)
    predictor.fit(X_seed, y_seed)

    agent = PhysicsAgent(predictor, uncertainty_threshold=0.35)

    for X_new in stream_of_samples:
        action: AgentAction = agent.observe(X_new)
        if action.action == "ask":
            y_true = oracle(X_new)       # request human label
            agent.reward(X_new, y_true)  # teach the agent
        else:
            use_prediction(action.prediction)

Usage — streaming big-data fit
-------------------------------
::

    from physml.agent import DataStream

    chunks = [(X_1, y_1), (X_2, y_2), ...]   # generator or list
    stream = DataStream(chunks)
    predictor = stream.fit_stream(predictor)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# AgentAction — result of a single observe() call
# ---------------------------------------------------------------------------

@dataclass
class AgentAction:
    """Result returned by :meth:`PhysicsAgent.observe`.

    Attributes
    ----------
    prediction : any or None
        Model prediction.  ``None`` when ``action == "ask"``.
    confidence : float
        Confidence score in [0, 1].  Derived from ``homeostasis_score``
        and / or label probability for classifiers.
    action : {"predict", "abstain", "ask"}
        ``"predict"`` — prediction is reliable, return it.
        ``"abstain"`` — uncertain but not requesting a label.
        ``"ask"``     — uncertain; agent requests a ground-truth label.
    needs_label : bool
        Shortcut: ``True`` when ``action == "ask"``.
    input_X : any
        Echo of the input passed to :meth:`observe`.
    metadata : dict
        Extra diagnostics (homeostasis score, explore mode flag, etc.).
    """

    prediction: Any
    confidence: float
    action: str
    needs_label: bool
    input_X: Any
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PhysicsAgent
# ---------------------------------------------------------------------------

class PhysicsAgent:
    """Autonomous prediction agent wrapping a :class:`PhysicsPredictor`.

    The agent implements a **predict / abstain / ask** policy controlled by
    a confidence threshold that adapts with the predictor's
    ``homeostasis_score``:

    * **High homeostasis** (stable, well-trained model) → raise the threshold
      so the agent exploits its knowledge and abstains less.
    * **Low homeostasis** (uncertain model) → lower the threshold so the
      agent explores by asking for labels more often.

    Confidence for **classifiers** is derived from the maximum class
    probability (``predict_proba`` when available, otherwise 1.0 if the
    predicted label matches the most common training class, else 0.5).

    Confidence for **regressors** is derived entirely from
    ``homeostasis_score``.

    Parameters
    ----------
    predictor : PhysicsPredictor
        A *fitted* predictor with ``backend="neural"`` for continual
        learning.  A physics-backend predictor works too, but ``reward()``
        will not update the model (no partial_fit support).
    uncertainty_threshold : float, default 0.35
        Base threshold below which the agent asks for a label.  Actual
        threshold is modulated by homeostasis (see above).
    homeostasis_weight : float, default 0.3
        How strongly homeostasis modulates the threshold.
        Final threshold = base ± homeostasis_weight * (homeostasis - 0.5).
    ewc_lambda : float, default 0.4
        EWC consolidation strength forwarded to ``partial_fit``.
    n_observations : int
        Counter of total observations processed.
    n_asks : int
        Counter of times the agent requested a human label.
    n_rewards : int
        Counter of labelled examples the agent has learned from.
    """

    def __init__(
        self,
        predictor: Any,
        uncertainty_threshold: float = 0.35,
        homeostasis_weight: float = 0.3,
        ewc_lambda: float = 0.4,
    ) -> None:
        self.predictor = predictor
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.homeostasis_weight = float(homeostasis_weight)
        self.ewc_lambda = float(ewc_lambda)
        self.n_observations: int = 0
        self.n_asks: int = 0
        self.n_rewards: int = 0
        # Buffer of (X_sample, y_true) pairs waiting to be batch-learned
        self._pending_labels: list[tuple[Any, Any]] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def observe(self, X: Any) -> AgentAction:
        """Process a new sample and decide whether to predict or ask.

        Parameters
        ----------
        X : array-like of shape (1, n_features) or (n_features,)
            A single sample or a small batch.

        Returns
        -------
        AgentAction
        """
        import numpy as np

        X_arr = np.atleast_2d(X)
        self.n_observations += 1

        # Get homeostasis score from the runtime state
        homeostasis = self._homeostasis()

        # Effective threshold: modulated by homeostasis
        # High homeostasis → agent trusts itself → raise threshold → ask less
        # Low homeostasis  → agent uncertain    → lower threshold → ask more
        eff_threshold = float(np.clip(
            self.uncertainty_threshold
            - self.homeostasis_weight * (homeostasis - 0.5),
            0.05, 0.95,
        ))

        # Get prediction and confidence
        try:
            prediction = self.predictor.predict(X_arr)
            confidence = self._estimate_confidence(X_arr, homeostasis)
        except Exception:
            prediction = None
            confidence = 0.0

        # Policy
        if confidence >= eff_threshold:
            action = "predict"
            needs_label = False
        else:
            action = "ask"
            needs_label = True
            self.n_asks += 1
            prediction = None

        return AgentAction(
            prediction=prediction,
            confidence=confidence,
            action=action,
            needs_label=needs_label,
            input_X=X,
            metadata={
                "homeostasis": homeostasis,
                "effective_threshold": eff_threshold,
                "explore_mode": homeostasis < 0.4,
            },
        )

    def reward(self, X: Any, y_true: Any, *, immediate: bool = True) -> None:
        """Provide a ground-truth label so the agent can learn from it.

        Parameters
        ----------
        X : array-like
            Input features (same format as passed to ``observe``).
        y_true : array-like
            True target label(s).
        immediate : bool, default True
            If True, immediately call ``partial_fit`` (neural backend) or
            buffer the sample for the next ``adapt()`` call.
        """
        import numpy as np

        X_arr = np.atleast_2d(X)
        y_arr = np.atleast_1d(y_true)
        self._pending_labels.append((X_arr, y_arr))
        self.n_rewards += 1

        if immediate:
            self.adapt()

    def adapt(self) -> None:
        """Flush the pending label buffer and update the model.

        For ``backend="neural"``, calls ``partial_fit``.  For the physics
        backend, all pending samples are re-fitted from scratch (via ``fit``
        on the augmented training set) — this is a slower but correct
        fallback.
        """
        if not self._pending_labels:
            return

        import numpy as np
        import pandas as pd

        Xs = [item[0] for item in self._pending_labels]
        ys = [item[1] for item in self._pending_labels]
        X_batch = np.vstack(Xs)
        y_batch = np.concatenate(ys)
        self._pending_labels.clear()

        backend = str(getattr(self.predictor, "backend", "physics")).lower().strip()
        if backend == "neural":
            try:
                self.predictor.partial_fit(X_batch, y_batch, ewc_lambda=self.ewc_lambda)
            except Exception:
                pass
        else:
            # Physics backend: append to train_df_ and re-fit
            try:
                from physml.estimator import _to_dataframe
                existing = self.predictor.train_df_.drop(columns=["__target__"], errors="ignore")
                y_existing = self.predictor.train_df_["__target__"].to_numpy()
                X_new_df = _to_dataframe(X_batch, feature_names=self.predictor.feature_names_in_)
                X_combined = pd.concat([existing, X_new_df], axis=0, ignore_index=True)
                y_combined = np.concatenate([y_existing, y_batch])
                self.predictor.fit(X_combined, y_combined)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _homeostasis(self) -> float:
        """Return the predictor's current homeostasis score (0–1)."""
        state = getattr(self.predictor, "runtime_state_", None)
        if state is None:
            return 0.5
        score = getattr(state, "homeostasis_score", 0.5)
        return float(np.clip(score, 0.0, 1.0))

    def _estimate_confidence(self, X_arr: np.ndarray, homeostasis: float) -> float:
        """Estimate prediction confidence as a scalar in [0, 1].

        For classifiers: use max class probability if ``predict_proba`` is
        available, otherwise fall back to ``homeostasis``.
        For regressors: use ``homeostasis`` directly.
        """
        try:
            is_clf = bool(getattr(self.predictor, "is_classifier_", False))
            if is_clf:
                try:
                    proba = self.predictor.predict_proba(X_arr)
                    return float(proba.max(axis=1).mean())
                except Exception:
                    pass
            return float(np.clip(homeostasis, 0.0, 1.0))
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """Return a summary of agent activity.

        Returns
        -------
        dict with keys:
            n_observations, n_asks, n_rewards, ask_rate,
            homeostasis, effective_threshold.
        """
        homeostasis = self._homeostasis()
        eff_threshold = float(np.clip(
            self.uncertainty_threshold
            - self.homeostasis_weight * (homeostasis - 0.5),
            0.05, 0.95,
        ))
        ask_rate = (
            float(self.n_asks) / max(self.n_observations, 1)
        )
        return {
            "n_observations": self.n_observations,
            "n_asks": self.n_asks,
            "n_rewards": self.n_rewards,
            "ask_rate": ask_rate,
            "homeostasis": homeostasis,
            "effective_threshold": eff_threshold,
        }


# ---------------------------------------------------------------------------
# Stage 5 — DataStream (mini-batch streaming)
# ---------------------------------------------------------------------------

class DataStream:
    """Mini-batch streaming wrapper for big-data / out-of-memory training.

    Wraps any iterable of ``(X_chunk, y_chunk)`` pairs and calls
    :meth:`~PhysicsPredictor.partial_fit` on each chunk sequentially
    (or ``fit`` for the very first chunk when no existing model is present).

    Parameters
    ----------
    chunks : iterable of (X, y) tuples
        Any iterable; each element is ``(X_chunk, y_chunk)`` where both are
        array-like with the same number of rows.

    Examples
    --------
    ::

        stream = DataStream(iter_csv_chunks("big_data.csv", chunksize=1000))
        predictor = PhysicsPredictor(backend="neural", n_cycles=10)
        predictor = stream.fit_stream(predictor, seed_X=X_seed, seed_y=y_seed)
    """

    def __init__(self, chunks: Iterable[tuple[Any, Any]]) -> None:
        self._chunks = chunks

    def fit_stream(
        self,
        predictor: Any,
        *,
        seed_X: Any = None,
        seed_y: Any = None,
        ewc_lambda: float = 0.4,
    ) -> Any:
        """Stream all chunks through the predictor using ``partial_fit``.

        If the predictor is not yet fitted (``seed_X`` is provided OR the
        first chunk is used as the seed), ``fit`` is called on the seed data
        first to initialise the model.  All subsequent chunks use
        ``partial_fit``.

        Parameters
        ----------
        predictor : PhysicsPredictor
        seed_X, seed_y : array-like or None
            Optional explicit seed data for the initial ``fit``.  If not
            supplied, the first chunk is used as the seed.
        ewc_lambda : float, default 0.4
            EWC consolidation strength forwarded to ``partial_fit``.

        Returns
        -------
        predictor : PhysicsPredictor
            The same predictor, updated in-place.
        """
        import numpy as np

        fitted = False
        chunks_iter = iter(self._chunks)

        # Seed fit
        if seed_X is not None and seed_y is not None:
            predictor.fit(np.atleast_2d(seed_X), np.atleast_1d(seed_y))
            fitted = True

        for X_chunk, y_chunk in chunks_iter:
            X_c = np.atleast_2d(X_chunk)
            y_c = np.atleast_1d(y_chunk)
            if not fitted:
                predictor.fit(X_c, y_c)
                fitted = True
            else:
                backend = str(getattr(predictor, "backend", "physics")).lower().strip()
                if backend == "neural":
                    try:
                        predictor.partial_fit(X_c, y_c, ewc_lambda=ewc_lambda)
                    except Exception:
                        pass
                # Physics backend: skip incremental update (re-fit not done
                # in streaming mode to avoid O(n²) cost).

        return predictor
