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

from physml._log import get_logger

_logger = get_logger(__name__)


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
        drift_detection: bool = False,
        drift_algorithm: str = "page_hinkley",
        n_ensemble: int = 5,
    ) -> None:
        self.predictor = predictor
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.homeostasis_weight = float(homeostasis_weight)
        self.ewc_lambda = float(ewc_lambda)
        self.query_strategy = str(query_strategy)
        self.policy = str(policy)
        self.error_window_size = int(error_window_size)
        self.task_id = task_id
        self.drift_detection = bool(drift_detection)
        self.drift_algorithm = str(drift_algorithm)
        self.n_ensemble = max(2, int(n_ensemble))
        self.n_observations: int = 0
        self.n_asks: int = 0
        self.n_rewards: int = 0
        self.n_drifts_detected: int = 0
        # cumulative oracle cost (Stage 25)
        self._oracle_cost: float = 0.0
        # Buffer of (X_sample, y_true) pairs waiting to be batch-learned
        self._pending_labels: list[tuple[Any, Any]] = []
        # Stage 10 — sliding window of recent prediction errors (0.0–1.0)
        self._error_window: deque[float] = deque(maxlen=self.error_window_size)
        # Stage 10 — last prediction made (used to compute error in reward())
        self._last_prediction: Any = None
        self._last_X: Any = None
        # Stage 15 — contextual bandit (created lazily on first observe)
        self._bandit: Any = None
        # Stage 17 — drift detector (created lazily when drift_detection=True)
        self._drift_detector: Any = None
        if self.drift_detection:
            from physml.drift import DriftDetector
            self._drift_detector = DriftDetector(algorithm=self.drift_algorithm)
        # Stage 24 — GP surrogate for "gp" query strategy (fitted lazily)
        self._gp: Any = None
        self._gp_X: list[Any] = []
        self._gp_y: list[Any] = []
        # Stage 26 — ensemble members for "ensemble" policy (fitted lazily)
        self._ensemble_members: list[Any] = []

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
        if self.policy == "bandit":
            # Stage 15 — contextual bandit
            ask_prob = self._bandit_ask_probability(X_arr, homeostasis)
            # Convert ask-probability to an effective threshold comparison:
            # if ask_prob > 0.5 the bandit wants to ask, irrespective of confidence
            if ask_prob > 0.5:
                action = "ask"
                needs_label = True
                self.n_asks += 1
                prediction = None
            elif confidence >= eff_threshold:
                action = "predict"
                needs_label = False
            else:
                action = "ask"
                needs_label = True
                self.n_asks += 1
                prediction = None
        elif self.policy == "ensemble":
            # Stage 26 — query-by-committee: use ensemble disagreement as signal
            disagreement = self._ensemble_disagreement(X_arr)
            # disagreement is in [0, 1]; treat it like an inverted confidence
            ensemble_confidence = float(1.0 - disagreement)
            if ensemble_confidence >= eff_threshold:
                action = "predict"
                needs_label = False
                confidence = ensemble_confidence
            else:
                action = "ask"
                needs_label = True
                self.n_asks += 1
                prediction = None
                confidence = ensemble_confidence
        elif confidence >= eff_threshold:
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

    def reward(self, X: Any, y_true: Any, *, immediate: bool = True, cost: float = 1.0) -> None:
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
        cost : float, default 1.0
            Oracle annotation cost for this sample (Stage 25).  Used by the
            bandit to optimise *accuracy per unit cost* rather than raw
            accuracy.  Tracked in ``total_oracle_cost`` via :meth:`report`.
        """
        import numpy as np

        X_arr = np.atleast_2d(X)
        y_arr = np.atleast_1d(y_true)
        self._oracle_cost += float(cost)

        # Stage 10 — record prediction error for adaptive threshold
        if self.policy == "adaptive":
            self._log_error(X_arr, y_arr)

        # Stage 15 — update contextual bandit with cost-adjusted reward signal
        if self.policy == "bandit":
            self._bandit_update(X_arr, y_arr, cost=float(cost))

        # Stage 24 — accumulate GP training data
        self._gp_X.append(X_arr)
        self._gp_y.append(y_arr)
        self._gp = None  # invalidate cached GP so it is re-fitted next query

        # Stage 26 — accumulate ensemble training data & trigger re-fit
        self._ensemble_members = []  # invalidate so it is re-fitted next observe

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
            except Exception as _exc:
                _logger.warning("fit_task(%r) failed: %s", self.task_id, _exc)
            return

        backend = str(getattr(self.predictor, "backend", "physics")).lower().strip()
        if backend in ("neural", "ensemble") or callable(getattr(self.predictor, "partial_fit", None)):
            # Any predictor that supports partial_fit (neural, ensemble, etc.)
            try:
                self.predictor.partial_fit(X_batch, y_batch, ewc_lambda=self.ewc_lambda)
            except TypeError:
                # partial_fit doesn't accept ewc_lambda
                try:
                    self.predictor.partial_fit(X_batch, y_batch)
                except Exception as _exc:
                    _logger.warning("partial_fit failed: %s", _exc)
            except Exception as _exc:
                _logger.warning("partial_fit failed: %s", _exc)
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

        if self.query_strategy == "gp":
            gp_idx = self._gp_select(X_arr)
            if gp_idx is not None:
                return gp_idx

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

    def select_batch(self, X_pool: Any, k: int) -> list[int]:
        """Return the indices of the *k* most informative samples in *X_pool*.

        Uses **coreset greedy selection** (Stage 16): iteratively pick the
        sample that is furthest from all already-selected samples in feature
        space, maximising coverage while minimising redundancy.  This is
        significantly better than selecting the top-*k* by entropy alone,
        which tends to cluster picks in the same region of the feature space.

        Parameters
        ----------
        X_pool : array-like of shape (n_candidates, n_features)
        k : int
            Number of samples to select.  Clamped to ``len(X_pool)``.

        Returns
        -------
        list[int]
            Indices into ``X_pool`` of the selected samples, in selection order.
        """
        X_arr = np.atleast_2d(X_pool)
        n = len(X_arr)
        k = min(k, n)

        if k <= 0:
            return []
        if k == 1:
            return [self.select_informative(X_arr)]

        # Start with the single most informative sample (entropy / confidence)
        selected = [self.select_informative(X_arr)]
        remaining = list(range(n))
        remaining.remove(selected[0])

        # Greedy coreset: add the sample maximally distant from the current set
        for _ in range(k - 1):
            if not remaining:
                break
            selected_arr = X_arr[selected]  # (|selected|, n_feat)
            # For each remaining candidate: min distance to any selected sample
            min_dists = np.array([
                float(np.min(np.linalg.norm(X_arr[i] - selected_arr, axis=1)))
                for i in remaining
            ])
            best_local = int(np.argmax(min_dists))
            best_global = remaining[best_local]
            selected.append(best_global)
            remaining.pop(best_local)

        return selected

    def _predict(self, X_arr: np.ndarray) -> Any:
        """Route a predict call through the right interface.

        Supports plain ``PhysicsPredictor`` and, when ``task_id`` is set,
        :class:`~physml.multitask_engine.MultiTaskPhysicsEngine`.

        Returns a scalar (or 1-element array squeezed to scalar) for
        single-sample inputs so that ``int(prediction)`` works with NumPy 2.x.
        """
        if self.task_id is not None:
            raw = self.predictor.predict_task(self.task_id, X_arr)
        else:
            raw = self.predictor.predict(X_arr)
        # Squeeze 1-element arrays to a plain Python scalar so that
        # ``int(prediction)`` works with NumPy 2.x (which no longer allows
        # implicit conversion of 0-dim arrays to Python scalars).
        try:
            arr = np.asarray(raw)
            if arr.ndim == 1 and arr.shape[0] == 1:
                return arr[0].item()
            if arr.ndim == 0:
                return arr.item()
        except Exception:
            pass
        return raw

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
            # Stage 17 — drift detection
            if self._drift_detector is not None:
                if self._drift_detector.update(error):
                    self.n_drifts_detected += 1
                    self._handle_drift()
        except Exception:
            pass

    def _handle_drift(self) -> None:
        """React to a detected concept drift event.

        * Reset the homeostasis state so the model re-explores.
        * Temporarily lower the ask-threshold to collect more labels.
        * Clear the rolling error window so the adaptive policy starts fresh.
        * Reset the drift detector.
        """
        # Reset homeostasis on the predictor's runtime state
        state = getattr(self.predictor, "runtime_state_", None)
        if state is not None:
            try:
                state.homeostasis_score = 0.1  # force re-exploration
            except Exception:
                pass
        # Lower ask-threshold burst: set it to a generous asking level
        # The adaptive policy will naturally raise it again as the model recovers.
        self._error_window.clear()
        for _ in range(self.error_window_size // 2):
            self._error_window.append(1.0)  # pretend many recent errors
        # Reset the detector so it can detect future drifts
        if self._drift_detector is not None:
            self._drift_detector.reset()

    # Stage 15 — contextual bandit helpers

    def _bandit_ask_probability(self, X_arr: np.ndarray, homeostasis: float) -> float:
        """Return the bandit's ask-probability for the current sample."""
        if self._bandit is None:
            n_feat = X_arr.shape[1] if X_arr.ndim == 2 else len(X_arr.ravel())
            from physml.bandit import ContextualBandit
            self._bandit = ContextualBandit(n_features=n_feat)
        x_flat = X_arr.ravel()
        return self._bandit.ask_probability(x_flat, homeostasis)

    def _bandit_update(self, X_arr: np.ndarray, y_true: np.ndarray, cost: float = 1.0) -> None:
        """Update the bandit with the cost-adjusted reward signal."""
        if self._bandit is None:
            return
        homeostasis = self._homeostasis()
        # Compute reward as accuracy improvement: reward > 0 means asking was good
        try:
            y_pred_now = self._predict(X_arr)
            y_pred_now = np.atleast_1d(y_pred_now)
            y_t = np.atleast_1d(y_true)
            is_clf = bool(getattr(self.predictor, "is_classifier_", False))
            if is_clf:
                accuracy = float(np.mean(y_pred_now == y_t))
            else:
                target_range = float(np.ptp(y_t)) or 1.0
                mae = float(np.mean(np.abs(y_pred_now.astype(float) - y_t.astype(float))))
                accuracy = 1.0 - float(np.clip(mae / target_range, 0.0, 1.0))
            # Stage 25: scale reward by inverse cost (accuracy per unit cost)
            reward = (accuracy - 0.5) / max(cost, 1e-6)
        except Exception:
            reward = 0.0
        x_flat = X_arr.ravel()
        asked = (self._last_prediction is None)  # True if last action was "ask"
        self._bandit.update(x_flat, homeostasis, reward=reward, asked=asked)

    # Stage 24 — GP uncertainty helpers

    def _gp_select(self, X_pool: np.ndarray) -> int | None:
        """Select the pool sample with the highest GP predictive variance.

        Falls back to ``None`` (which causes the caller to use entropy/threshold)
        when fewer than 3 labelled examples are available or sklearn's GP
        is unavailable.
        """
        if len(self._gp_X) < 3:
            return None
        try:
            from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel

            X_train = np.vstack(self._gp_X)
            y_train = np.concatenate(self._gp_y)

            if self._gp is None:
                kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
                is_clf = bool(getattr(self.predictor, "is_classifier_", False))
                if is_clf:
                    gp = GaussianProcessClassifier(kernel=kernel, max_iter_predict=20)
                    gp.fit(X_train, y_train)
                else:
                    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
                    gp.fit(X_train, y_train)
                self._gp = (gp, is_clf)

            gp, is_clf = self._gp
            if is_clf:
                proba = gp.predict_proba(X_pool)
                eps = 1e-12
                scores = -np.sum(proba * np.log(proba + eps), axis=1)
            else:
                _, std = gp.predict(X_pool, return_std=True)
                scores = std
            return int(np.argmax(scores))
        except Exception:
            return None

    # Stage 26 — ensemble helpers

    def _build_ensemble(self) -> None:
        """Build bootstrap ensemble from accumulated labelled data."""
        if len(self._gp_X) < 2:
            return
        try:
            from sklearn.neural_network import MLPClassifier, MLPRegressor

            X_train = np.vstack(self._gp_X)
            y_train = np.concatenate(self._gp_y)
            n = len(y_train)
            is_clf = bool(getattr(self.predictor, "is_classifier_", False))
            rng = np.random.default_rng(42)
            members = []
            for _ in range(self.n_ensemble):
                idx = rng.choice(n, size=n, replace=True)
                X_b = X_train[idx]
                y_b = y_train[idx]
                if is_clf:
                    m = MLPClassifier(
                        hidden_layer_sizes=(32,),
                        max_iter=100,
                        random_state=int(rng.integers(10000)),
                    )
                else:
                    m = MLPRegressor(
                        hidden_layer_sizes=(32,),
                        max_iter=100,
                        random_state=int(rng.integers(10000)),
                    )
                try:
                    m.fit(X_b, y_b)
                    members.append((m, is_clf))
                except Exception:
                    pass
            self._ensemble_members = members
        except Exception:
            self._ensemble_members = []

    def _ensemble_disagreement(self, X_arr: np.ndarray) -> float:
        """Return committee disagreement score in [0, 1] for *X_arr*.

        Uses vote entropy for classifiers and coefficient of variation of
        predictions for regressors.  Returns 0.5 (maximum uncertainty) if the
        ensemble has not been built yet.
        """
        if not self._ensemble_members:
            self._build_ensemble()
        if not self._ensemble_members:
            return 0.5
        try:
            preds = []
            for m, is_clf in self._ensemble_members:
                p = m.predict(X_arr)
                preds.append(p)
            preds = np.array(preds)  # (n_members, n_samples)
            if self._ensemble_members[0][1]:  # classifier
                n_members = len(preds)
                n_samples = preds.shape[1]
                disagreements = []
                for j in range(n_samples):
                    votes = preds[:, j]
                    classes, counts = np.unique(votes, return_counts=True)
                    probs = counts / n_members
                    eps = 1e-12
                    ent = -np.sum(probs * np.log(probs + eps))
                    max_ent = np.log(max(len(classes), 2))
                    disagreements.append(float(ent / max_ent) if max_ent > 0 else 0.0)
                return float(np.mean(disagreements))
            else:
                # Regressor: use coefficient of variation
                std = float(np.std(preds))
                mean = float(np.abs(np.mean(preds))) + 1e-6
                return float(np.clip(std / mean, 0.0, 1.0))
        except Exception:
            return 0.5

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
            "n_drifts_detected": self.n_drifts_detected,
            "drift_detection": self.drift_detection,
            "total_oracle_cost": self._oracle_cost,
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
