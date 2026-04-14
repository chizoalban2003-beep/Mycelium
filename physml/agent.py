"""Stage 4 + 5 + 8 + 10 — Autonomous agent loop and big-data streaming.

``PhysicsAgent`` wraps a :class:`PhysicsPredictor` with an uncertainty gate,
a decide/abstain policy, and an online-learning loop.  Users observe
predictions, provide feedback (true labels), and the agent adapts in
real-time via :meth:`~PhysicsPredictor.partial_fit`.

Stage 8 adds **active learning** via ``query_strategy="entropy"``:
``select_informative(X_pool)`` scores a pool of candidates by prediction
entropy and returns the index of the highest-entropy sample, reducing oracle
calls while maintaining accuracy.

Stage 10 adds **reward shaping** via ``policy="adaptive"``: the agent keeps a
sliding window of recent prediction errors and adjusts its ask-threshold
automatically — asking more when it has been wrong recently and relying on
itself more when it has been accurate.

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

Usage — active learning (Stage 8)
----------------------------------
::

    agent = PhysicsAgent(predictor, query_strategy="entropy")

    # Given a pool of unlabelled candidates, find the most informative one
    best_idx = agent.select_informative(X_pool)
    y_true = oracle(X_pool[best_idx])
    agent.reward(X_pool[best_idx], y_true)

Usage — streaming big-data fit
-------------------------------
::

    from physml.agent import DataStream

    chunks = [(X_1, y_1), (X_2, y_2), ...]   # generator or list
    stream = DataStream(chunks)
    predictor = stream.fit_stream(predictor)
"""

from __future__ import annotations

from collections import deque
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

    Stage 8 — Active learning
    -------------------------
    Set ``query_strategy="entropy"`` to enable entropy-based query selection.
    The :meth:`select_informative` method accepts a pool of unlabelled
    candidates and returns the index of the sample with the highest prediction
    entropy, concentrating oracle queries where they teach the most.

    Stage 9 — Multi-task support
    ----------------------------
    Pass a :class:`~physml.multitask_engine.MultiTaskPhysicsEngine` as the
    ``predictor`` together with a ``task_id`` string.  The agent will route
    ``predict`` / ``adapt`` calls through the engine's per-task head.

    Stage 10 — Adaptive threshold (reward shaping)
    -----------------------------------------------
    Set ``policy="adaptive"`` to enable reward-shaped threshold adjustment.
    The agent maintains a sliding window of recent prediction errors; the
    effective threshold rises when the model has been accurate and falls when
    it has been wrong, replacing the fixed heuristic with a self-calibrating
    policy.

    Parameters
    ----------
    predictor : PhysicsPredictor or MultiTaskPhysicsEngine
        A *fitted* predictor.  Use ``backend="neural"`` for continual
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
    query_strategy : {"threshold", "entropy"}, default "threshold"
        Active-learning query strategy used by :meth:`select_informative`.
        ``"threshold"`` selects the sample with the lowest confidence;
        ``"entropy"`` selects the sample with the highest prediction entropy
        (requires ``predict_proba`` on the underlying predictor).
    policy : {"fixed", "adaptive"}, default "fixed"
        Threshold policy.  ``"fixed"`` uses the static
        ``uncertainty_threshold``; ``"adaptive"`` adjusts the threshold based
        on the rolling prediction error rate tracked via :meth:`reward`.
    error_window_size : int, default 20
        Size of the sliding window used by the ``"adaptive"`` policy.
    task_id : str or None, default None
        When set, routes predict / adapt calls through a
        :class:`~physml.multitask_engine.MultiTaskPhysicsEngine` using this
        task identifier.
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
        query_strategy: str = "threshold",
        policy: str = "fixed",
        error_window_size: int = 20,
        task_id: str | None = None,
    ) -> None:
        self.predictor = predictor
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.homeostasis_weight = float(homeostasis_weight)
        self.ewc_lambda = float(ewc_lambda)
        self.query_strategy = str(query_strategy)
        self.policy = str(policy)
        self.error_window_size = int(error_window_size)
        self.task_id = task_id
        self.n_observations: int = 0
        self.n_asks: int = 0
        self.n_rewards: int = 0
        # Buffer of (X_sample, y_true) pairs waiting to be batch-learned
        self._pending_labels: list[tuple[Any, Any]] = []
        # Stage 10 — sliding window of recent prediction errors (0.0–1.0)
        self._error_window: deque[float] = deque(maxlen=self.error_window_size)
        # Stage 10 — last prediction made (used to compute error in reward())
        self._last_prediction: Any = None
        self._last_X: Any = None

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

        # Effective threshold: modulated by homeostasis and (optionally) the
        # adaptive policy.
        eff_threshold = self._adaptive_threshold(homeostasis)

        # Get prediction and confidence
        try:
            prediction = self._predict(X_arr)
            confidence = self._estimate_confidence(X_arr, homeostasis)
        except Exception:
            prediction = None
            confidence = 0.0

        # Stage 10 — store last prediction for error tracking in reward()
        self._last_prediction = prediction
        self._last_X = X_arr

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
                "policy": self.policy,
                "error_rate": self._error_rate(),
            },
        )

    def reward(self, X: Any, y_true: Any, *, immediate: bool = True) -> None:
        """Provide a ground-truth label so the agent can learn from it.

        When ``policy="adaptive"``, the agent also computes the prediction
        error and logs it to the sliding window used by
        :meth:`_adaptive_threshold`.

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

        # Stage 10 — record prediction error for adaptive threshold
        if self.policy == "adaptive":
            self._log_error(X_arr, y_arr)

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

        When ``task_id`` is set and the predictor is a
        :class:`~physml.multitask_engine.MultiTaskPhysicsEngine`, the pending
        samples are used to update the task-specific head via
        ``fit_task(task_id, X, y)``.
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

        # Stage 9 — multi-task routing
        if self.task_id is not None:
            try:
                self.predictor.fit_task(self.task_id, X_batch, y_batch)
            except Exception:
                pass
            return

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
    # Stage 8 — Active learning
    # ------------------------------------------------------------------

    def select_informative(self, X_pool: Any) -> int:
        """Return the index of the most informative sample in *X_pool*.

        Uses the configured ``query_strategy``:

        * ``"entropy"`` — selects the sample with the highest prediction
          entropy (``-Σ p log p``), which requires ``predict_proba`` on the
          underlying predictor.  Falls back to confidence inversion if
          ``predict_proba`` is unavailable.
        * ``"threshold"`` — selects the sample with the lowest confidence
          (equivalent to the most uncertain sample under the threshold
          policy).

        Parameters
        ----------
        X_pool : array-like of shape (n_candidates, n_features)
            Pool of unlabelled candidates.

        Returns
        -------
        int
            Index into ``X_pool`` of the most informative sample.
        """
        X_arr = np.atleast_2d(X_pool)
        n = len(X_arr)

        if n == 1:
            return 0

        if self.query_strategy == "entropy":
            try:
                is_clf = bool(getattr(self.predictor, "is_classifier_", False))
                if is_clf:
                    proba = self.predictor.predict_proba(X_arr)
                    eps = 1e-12
                    entropy = -np.sum(proba * np.log(proba + eps), axis=1)
                    return int(np.argmax(entropy))
            except Exception:
                pass

        # Fallback / "threshold" strategy: lowest confidence = most uncertain
        homeostasis = self._homeostasis()
        confidences = np.array([
            self._estimate_confidence(X_arr[i: i + 1], homeostasis)
            for i in range(n)
        ])
        return int(np.argmin(confidences))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict(self, X_arr: np.ndarray) -> Any:
        """Route a predict call through the right interface.

        Supports plain ``PhysicsPredictor`` and, when ``task_id`` is set,
        :class:`~physml.multitask_engine.MultiTaskPhysicsEngine`.
        """
        if self.task_id is not None:
            return self.predictor.predict_task(self.task_id, X_arr)
        return self.predictor.predict(X_arr)

    def _homeostasis(self) -> float:
        """Return the predictor's current homeostasis score (0–1)."""
        state = getattr(self.predictor, "runtime_state_", None)
        if state is None:
            return 0.5
        score = getattr(state, "homeostasis_score", 0.5)
        return float(np.clip(score, 0.0, 1.0))

    # Stage 10 — adaptive threshold helpers

    def _error_rate(self) -> float:
        """Rolling prediction error rate from the sliding window."""
        if not self._error_window:
            return 0.5  # unknown — start in the middle
        return float(np.mean(list(self._error_window)))

    def _adaptive_threshold(self, homeostasis: float) -> float:
        """Compute the effective ask-threshold for the current step.

        When ``policy="fixed"``:
            threshold = base ± homeostasis_weight * (homeostasis - 0.5)

        When ``policy="adaptive"``:
            threshold = (fixed threshold) + adjustment from rolling error rate
            The error rate shifts the threshold by up to ±0.2:
            * error_rate → 1.0  ⇒ threshold − 0.2  (ask more)
            * error_rate → 0.0  ⇒ threshold + 0.2  (ask less)
        """
        base = float(np.clip(
            self.uncertainty_threshold
            - self.homeostasis_weight * (homeostasis - 0.5),
            0.05, 0.95,
        ))
        if self.policy == "adaptive" and self._error_window:
            error_rate = self._error_rate()
            # High error_rate → raise threshold (confidence needs to clear a
            # higher bar → agent asks more labels).
            # Low error_rate  → lower threshold (agent relies on itself more).
            # adjustment is negative when error_rate > 0.5, positive otherwise.
            adjustment = 0.2 * (0.5 - error_rate)  # in [-0.1, +0.1]
            # Subtracting a negative adjustment RAISES the threshold:
            #   error_rate=1.0 → adjustment=-0.1 → base += 0.1 → asks more
            #   error_rate=0.0 → adjustment=+0.1 → base -= 0.1 → asks less
            base = float(np.clip(base - adjustment, 0.05, 0.95))
        return base

    def _log_error(self, X_arr: np.ndarray, y_true: np.ndarray) -> None:
        """Compute prediction error on X_arr vs y_true and log to window."""
        try:
            y_pred = self._predict(X_arr)
            y_pred = np.atleast_1d(y_pred)
            y_t = np.atleast_1d(y_true)
            is_clf = bool(getattr(self.predictor, "is_classifier_", False))
            if is_clf:
                # Error = fraction of incorrect predictions
                error = float(np.mean(y_pred != y_t))
            else:
                # Normalised MAE relative to target range (clipped to [0, 1])
                target_range = float(np.ptp(y_t)) if len(y_t) > 1 else 1.0
                if target_range == 0:
                    target_range = 1.0
                error = float(np.clip(
                    np.mean(np.abs(y_pred.astype(float) - y_t.astype(float))) / target_range,
                    0.0, 1.0,
                ))
            self._error_window.append(error)
        except Exception:
            pass

    def _estimate_confidence(self, X_arr: np.ndarray, homeostasis: float) -> float:
        """Estimate prediction confidence as a scalar in [0, 1].

        For classifiers: use max class probability from ``predict_proba``
        (when available) blended with the runtime ``homeostasis_score``.
        Blending prevents an overconfident (saturated) MLP from reporting
        1.0 confidence regardless of the threshold setting.
        For regressors: use ``homeostasis`` directly.
        """
        try:
            is_clf = bool(getattr(self.predictor, "is_classifier_", False))
            if is_clf:
                try:
                    proba = self.predictor.predict_proba(X_arr)
                    max_p = float(proba.max(axis=1).mean())
                    # Blend MLP confidence with homeostasis: neither alone
                    # is a reliable signal.  Softmax saturation can push
                    # max_p to ~1.0 even for poorly-trained models.
                    return float((max_p + homeostasis) / 2.0)
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
            homeostasis, effective_threshold, policy, error_rate.
        """
        homeostasis = self._homeostasis()
        eff_threshold = self._adaptive_threshold(homeostasis)
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
            "policy": self.policy,
            "error_rate": self._error_rate(),
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
