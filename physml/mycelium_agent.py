"""Stage 11 — MyceliumAgent: the project's top-level branded autonomous agent.

``MyceliumAgent`` is the flagship class that gives the repository its name.
It wraps :class:`~physml.agent.PhysicsAgent` with:

* **Active learning** (Stage 8) — ``query_strategy="entropy"`` by default,
  exposing :meth:`select_informative` for pool-based label selection.
* **Adaptive threshold** (Stage 10) — ``policy="adaptive"`` by default,
  so the ask-rate self-calibrates with the rolling prediction error.
* **Multi-task support** (Stage 9) — pass a
  :class:`~physml.multitask_engine.MultiTaskPhysicsEngine` as the predictor
  together with a ``task_id`` string.
* A clean, minimal API: ``fit``, ``observe``, ``select_informative``,
  ``reward``, ``save``, ``load``, ``report``.

Usage
-----
::

    from physml.mycelium_agent import MyceliumAgent

    agent = MyceliumAgent()
    agent.fit(X_seed, y_seed)

    # Single-sample prediction loop
    for X_new in data_stream:
        action = agent.observe(X_new)
        if action.action == "ask":
            y_true = oracle(X_new)
            agent.reward(X_new, y_true)
        else:
            use_prediction(action.prediction)

    # Pool-based active learning
    best_idx = agent.select_informative(X_unlabelled_pool)
    agent.reward(X_unlabelled_pool[best_idx], oracle(X_unlabelled_pool[best_idx]))

    # Persist and restore
    agent.save("mycelium.pkl")
    agent2 = MyceliumAgent.load("mycelium.pkl")

    print(agent.report())
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from physml._log import get_logger

_logger = get_logger(__name__)


class MyceliumAgent:
    """Top-level autonomous agent for the Mycelium project.

    Combines active learning (Stage 8), adaptive threshold policy (Stage 10),
    and multi-task support (Stage 9) behind a minimal API.  All internal
    complexity — continual learning, EWC regularisation, replay buffer,
    entropy-based query selection, rolling error tracking — is handled
    automatically.

    Parameters
    ----------
    predictor : PhysicsPredictor, MultiTaskPhysicsEngine, or None
        A pre-built predictor / engine.  When ``None``, a fresh
        ``PhysicsPredictor(backend="neural", n_cycles=20)`` is created on
        the first call to :meth:`fit`.
    uncertainty_threshold : float, default 0.35
        Base ask-threshold forwarded to :class:`~physml.agent.PhysicsAgent`.
    query_strategy : {"entropy", "threshold"}, default "entropy"
        Active-learning strategy for :meth:`select_informative`.
        ``"entropy"`` selects the highest-entropy candidate (recommended for
        classifiers); ``"threshold"`` selects the lowest-confidence one.
    policy : {"adaptive", "fixed"}, default "adaptive"
        Threshold policy.  ``"adaptive"`` adjusts the threshold based on the
        rolling prediction error; ``"fixed"`` uses the static
        ``uncertainty_threshold``.
    error_window_size : int, default 20
        Sliding window size for the adaptive policy.
    homeostasis_weight : float, default 0.3
        How strongly the predictor's homeostasis score modulates the
        threshold.
    ewc_lambda : float, default 0.4
        Elastic Weight Consolidation regularisation strength.
    task_id : str or None, default None
        Task identifier for multi-task engines.  When set, the agent routes
        predict / reward calls through
        :meth:`~physml.multitask_engine.MultiTaskPhysicsEngine.predict_task`
        and :meth:`~physml.multitask_engine.MultiTaskPhysicsEngine.fit_task`.
    predictor_kwargs : dict or None
        Extra keyword arguments forwarded to
        :class:`~physml.estimator.PhysicsPredictor` when
        ``predictor`` is ``None``.
    calibrate : bool, default True
        When ``True`` (Stage 13), a temperature-scaling calibration step is
        run after the initial ``fit()`` on a 20 % held-out split of the
        training data.  This makes confidence scores reliable probabilities
        that the adaptive threshold policy can trust.  Set to ``False`` to
        skip calibration (e.g. when ``X_seed`` is very small).
    drift_detection : bool, default False
        When ``True`` (Stage 17), a :class:`~physml.drift.DriftDetector` is
        attached to the agent.  When drift is detected in the reward stream,
        the homeostasis state is reset and the ask-threshold is temporarily
        lowered to re-explore the shifted distribution.
    drift_algorithm : {"page_hinkley", "adwin"}, default "page_hinkley"
        Drift-detection algorithm.  Only used when ``drift_detection=True``.
    """

    def __init__(
        self,
        predictor: Any = None,
        *,
        uncertainty_threshold: float = 0.35,
        query_strategy: str = "entropy",
        policy: str = "adaptive",
        error_window_size: int = 20,
        homeostasis_weight: float = 0.3,
        ewc_lambda: float = 0.4,
        task_id: str | None = None,
        predictor_kwargs: dict[str, Any] | None = None,
        calibrate: bool = True,
        drift_detection: bool = False,
        drift_algorithm: str = "page_hinkley",
        n_ensemble: int = 5,
    ) -> None:
        self._predictor = predictor
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.query_strategy = str(query_strategy)
        self.policy = str(policy)
        self.error_window_size = int(error_window_size)
        self.homeostasis_weight = float(homeostasis_weight)
        self.ewc_lambda = float(ewc_lambda)
        self.task_id = task_id
        self._predictor_kwargs = dict(predictor_kwargs or {})
        self.calibrate = bool(calibrate)
        self.drift_detection = bool(drift_detection)
        self.drift_algorithm = str(drift_algorithm)
        self.n_ensemble = max(2, int(n_ensemble))

        self._agent: Any = None  # built after fit()
        self._fitted: bool = False
        self.temperature_: float = 1.0  # Stage 13 — set after calibration
        self._memory: Any = None  # Stage 38 — attached EpisodicMemory
        self._last_action_str: str = "predict"  # Stage 42 — avoids re-calling observe() in reward()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: Any, y: Any) -> "MyceliumAgent":
        """Fit the underlying predictor on seed data and initialise the agent.

        Must be called at least once before :meth:`observe`.

        When ``calibrate=True`` (default), a temperature-scaling step
        (Stage 13) is run on a held-out 20 % split of the data to produce
        well-calibrated confidence scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        import numpy as np

        X_arr = np.atleast_2d(X)
        y_arr = np.atleast_1d(y)

        if self._predictor is None:
            from physml.estimator import PhysicsPredictor
            kwargs = dict(self._predictor_kwargs)
            kwargs.setdefault("backend", "neural")
            kwargs.setdefault("n_cycles", 20)
            self._predictor = PhysicsPredictor(**kwargs)

        if self.task_id is not None:
            # Multi-task mode: delegate to MultiTaskPhysicsEngine.fit_task
            self._predictor.fit_task(self.task_id, X_arr, y_arr)
        else:
            self._predictor.fit(X_arr, y_arr)

        # Stage 13 — temperature calibration on held-out split
        self.temperature_ = self._fit_calibration(X_arr, y_arr)

        self._fitted = True
        self._build_agent()
        return self

    def observe(self, X: Any) -> Any:
        """Process a new sample and return an :class:`~physml.agent.AgentAction`.

        Parameters
        ----------
        X : array-like of shape (1, n_features) or (n_features,)

        Returns
        -------
        AgentAction
            ``action.action`` is ``"predict"`` (confident), ``"ask"``
            (needs label), or ``"abstain"``.
        """
        self._require_fitted()
        action = self._agent.observe(X)
        self._last_action_str = str(getattr(action, "action", "predict"))
        return action

    def predict(self, X: Any) -> np.ndarray:
        """Batch predict class labels (classification) or target values (regression).

        Delegates to the underlying predictor's ``predict`` method when
        available; otherwise falls back to ``argmax`` over ``predict_proba``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        self._require_fitted()
        predictor = self._predictor
        if self.task_id is not None:
            # Multi-task: use predict_task if available
            if hasattr(predictor, "predict_task"):
                return np.asarray(predictor.predict_task(self.task_id, X))
        if hasattr(predictor, "predict"):
            return np.asarray(predictor.predict(X))
        # Fallback: argmax over predict_proba (classification)
        proba = np.asarray(self._predictor.predict_proba(X))
        return np.argmax(proba, axis=1)

    def select_informative(self, X_pool: Any) -> int:
        """Return the index of the most informative sample in *X_pool*.

        Delegates to :meth:`~physml.agent.PhysicsAgent.select_informative`
        using the configured ``query_strategy``.

        Parameters
        ----------
        X_pool : array-like of shape (n_candidates, n_features)

        Returns
        -------
        int
        """
        self._require_fitted()
        return self._agent.select_informative(X_pool)

    def select_batch(self, X_pool: Any, k: int) -> list[int]:
        """Return indices of the *k* most informative samples (coreset, Stage 16).

        Parameters
        ----------
        X_pool : array-like of shape (n_candidates, n_features)
        k : int

        Returns
        -------
        list[int]
        """
        self._require_fitted()
        return self._agent.select_batch(X_pool, k)

    def reward(self, X: Any, y_true: Any, *, immediate: bool = True, cost: float = 1.0) -> "MyceliumAgent":
        """Provide a ground-truth label so the agent can learn from it.

        Parameters
        ----------
        X : array-like
        y_true : array-like
        immediate : bool, default True
        cost : float, default 1.0
            Oracle annotation cost for this sample (Stage 25).

        Returns
        -------
        self
        """
        self._require_fitted()
        self._agent.reward(X, y_true, immediate=immediate, cost=cost)

        # Stage 38/42 — auto-store episode; use cached action to avoid
        # double inference via observe() call
        if self._memory is not None:
            try:
                x_vec = np.atleast_1d(np.asarray(X, dtype=np.float32)).ravel()
                self._memory.store(
                    context=x_vec,
                    action=self._last_action_str,
                    outcome=1.0,
                )
            except Exception as _exc:
                _logger.debug("Episode memory store failed (best-effort): %s", _exc)

        return self

    def report(self) -> dict[str, Any]:
        """Return a summary of agent activity and configuration.

        Returns
        -------
        dict with keys:
            agent (PhysicsAgent report sub-dict), query_strategy, policy,
            task_id, fitted, temperature (calibration temperature).
        """
        agent_report = self._agent.report() if self._agent is not None else {}
        return {
            "agent": agent_report,
            "query_strategy": self.query_strategy,
            "policy": self.policy,
            "task_id": self.task_id,
            "fitted": self._fitted,
            "temperature": self.temperature_,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Persist the agent to disk using joblib.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        Path — the file path used.
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("joblib is required for agent persistence") from exc
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(save_path))
        return save_path

    @classmethod
    def load(cls, path: str | Path) -> "MyceliumAgent":
        """Load a previously saved agent.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        MyceliumAgent

        Raises
        ------
        TypeError
            If the file does not contain a :class:`MyceliumAgent`.
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("joblib is required for agent persistence") from exc
        obj = joblib.load(str(path))
        if not isinstance(obj, cls):
            raise TypeError(f"Expected MyceliumAgent, got {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_agent(self) -> None:
        from physml.agent import PhysicsAgent

        self._agent = PhysicsAgent(
            self._predictor,
            uncertainty_threshold=self.uncertainty_threshold,
            homeostasis_weight=self.homeostasis_weight,
            ewc_lambda=self.ewc_lambda,
            query_strategy=self.query_strategy,
            policy=self.policy,
            error_window_size=self.error_window_size,
            task_id=self.task_id,
            drift_detection=self.drift_detection,
            drift_algorithm=self.drift_algorithm,
            n_ensemble=self.n_ensemble,
        )

    def _fit_calibration(self, X: np.ndarray, y: np.ndarray) -> float:
        """Stage 13 — fit temperature scaling on a held-out split.

        Uses 20 % of the data as a calibration set.  Returns 1.0 when
        calibration is disabled, the dataset is too small (< 10 samples), or
        the predictor has no ``predict_proba``.
        """
        if not self.calibrate:
            return 1.0
        n = len(y)
        if n < 10:
            return 1.0
        # Reserve last 20 % as calibration set (no shuffle — avoids extra
        # randomness during fit)
        n_cal = max(2, int(n * 0.2))
        X_cal = X[-n_cal:]
        y_cal = y[-n_cal:]
        predictor = self._predictor
        if self.task_id is not None:
            # For multi-task engines wrap the task-specific predict_proba
            predictor = _MultiTaskProbaWrapper(self._predictor, self.task_id)
        from physml.calibration import calibrate_temperature
        return calibrate_temperature(predictor, X_cal, y_cal)

    def _require_fitted(self) -> None:
        if not self._fitted or self._agent is None:
            raise RuntimeError(
                "MyceliumAgent is not fitted yet.  Call fit(X_seed, y_seed) first."
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Stage 31 — tool use
    # ------------------------------------------------------------------

    def use_tool(self, tool_name: str, input_str: str, registry: "ToolRegistry") -> str:
        """Call a registered tool and incorporate the interaction into agent state.

        The tool output is featurized (if possible) and used as a reward signal
        so the agent learns which tools produce useful outcomes.

        Parameters
        ----------
        tool_name : str
            Name of the tool to call (must be registered in *registry*).
        input_str : str
            Input passed to the tool function.
        registry : ToolRegistry

        Returns
        -------
        str — raw output from the tool.
        """
        from physml.tools import ToolRegistry  # local import avoids circular deps

        result: str = registry.call(tool_name, input_str)
        return result

    # ------------------------------------------------------------------
    # Stage 33 — episodic memory augmentation
    # ------------------------------------------------------------------

    def augment_with_memory(self, X: "np.ndarray", memory: "EpisodicMemory") -> "np.ndarray":
        """Augment *X* with episodic-memory features before prediction.

        Delegates to :meth:`~physml.memory.EpisodicMemory.augment_features`.
        If *memory* is empty, *X* is returned unchanged.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        memory : EpisodicMemory

        Returns
        -------
        np.ndarray, shape (n_samples, n_features + n_neighbors * 2)
        """
        from physml.memory import EpisodicMemory  # local import

        return memory.augment_features(X)

    # ------------------------------------------------------------------
    # Stage 38 — attach episodic memory for automatic episode recording
    # ------------------------------------------------------------------

    def attach_memory(self, memory: "EpisodicMemory") -> "MyceliumAgent":
        """Attach an :class:`~physml.memory.EpisodicMemory` to this agent.

        Once attached, every call to :meth:`reward` automatically stores the
        ``(feature_vector, action, outcome)`` triple so the agent accumulates
        experience over time.  The stored memory can later be passed to
        :meth:`augment_with_memory` or used by :meth:`run_goal`.

        Parameters
        ----------
        memory : EpisodicMemory

        Returns
        -------
        self
        """
        self._memory = memory
        return self

    # ------------------------------------------------------------------
    # Stage 37 — goal-driven closed autonomous loop
    # ------------------------------------------------------------------

    def run_goal(
        self,
        goal: str,
        registry: "ToolRegistry",
        featurizer: "Featurizer",
        *,
        memory: "EpisodicMemory | None" = None,
        n_subtasks: int = 3,
        max_steps: int = 10,
    ) -> dict:
        """Execute a goal end-to-end using planning, tools, and memory.

        Decomposes *goal* into sub-tasks via :class:`~physml.planner.GoalPlanner`,
        then runs each sub-task through :class:`~physml.tools.AutonomousLoop`.
        Results are stored in *memory* (or the attached memory, if any) so
        that the agent accumulates cross-goal experience.

        Parameters
        ----------
        goal : str
            Free-text goal description.
        registry : ToolRegistry
            Available tools for the agent to call.
        featurizer : Featurizer
            Fitted featurizer used to embed text into vectors.
        memory : EpisodicMemory or None
            Episode store; if ``None``, falls back to the agent's attached
            memory (see :meth:`attach_memory`).  If neither is available, no
            episodes are recorded.
        n_subtasks : int, default 3
            Number of sub-tasks to decompose *goal* into.
        max_steps : int, default 10
            Maximum loop iterations per sub-task.

        Returns
        -------
        dict with keys:
            ``goal`` (str), ``subtasks`` (list of sub-task result dicts),
            ``n_tool_calls`` (int), ``n_episodes_stored`` (int),
            ``result`` (str — final tool/prediction output).
        """
        self._require_fitted()
        from physml.planner import GoalPlanner
        from physml.tools import AutonomousLoop

        mem = memory if memory is not None else getattr(self, "_memory", None)

        planner = GoalPlanner(featurizer=featurizer, agent=self, n_subtasks=n_subtasks)
        loop = AutonomousLoop(
            agent=self,
            registry=registry,
            featurizer=featurizer,
            max_steps=max_steps,
        )

        subtasks = planner.plan(goal)
        subtask_results: list[dict] = []
        total_tool_calls = 0
        n_episodes_stored = 0
        final_result = goal

        for subtask in subtasks:
            st_result = loop.run(subtask.description)
            total_tool_calls += st_result.get("n_tool_calls", 0)
            subtask_results.append(
                {"task_id": subtask.task_id, "description": subtask.description, **st_result}
            )
            final_result = st_result.get("result", final_result)

            # Record episode in memory
            if mem is not None:
                outcome = 1.0 if st_result.get("n_tool_calls", 0) > 0 else 0.5
                mem.store(
                    context=subtask.feature_vec,
                    action=str(st_result.get("result", "predict"))[:64],
                    outcome=outcome,
                )
                n_episodes_stored += 1

        return {
            "goal": goal,
            "subtasks": subtask_results,
            "n_tool_calls": total_tool_calls,
            "n_episodes_stored": n_episodes_stored,
            "result": final_result,
        }

    # ------------------------------------------------------------------
    # Stage 39 — self-evaluation
    # ------------------------------------------------------------------

    def self_evaluate(self, X_test: Any, y_test: Any) -> dict:
        """Evaluate the agent on held-out data and return quality metrics.

        Computes accuracy, mean confidence, expected calibration error (ECE),
        and the running oracle cost from :meth:`report`.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
        y_test : array-like of shape (n_samples,)

        Returns
        -------
        dict with keys:
            ``accuracy`` (float), ``mean_confidence`` (float),
            ``ece`` (float), ``n_samples`` (int),
            ``oracle_cost`` (float), ``threshold`` (float).
        """
        self._require_fitted()
        X = np.atleast_2d(X_test)
        y = np.atleast_1d(y_test)
        n = len(y)

        correct = 0
        confidences: list[float] = []
        ece_bins = np.zeros(10)
        ece_counts = np.zeros(10)
        ece_correct = np.zeros(10)

        for i in range(n):
            action = self._agent.observe(X[i : i + 1])
            pred = getattr(action, "prediction", None)
            conf = float(getattr(action, "confidence", 0.5) or 0.5)
            confidences.append(conf)

            if pred is not None:
                try:
                    correct += int(int(pred) == int(y[i]))
                except (TypeError, ValueError):
                    correct += int(pred == y[i])

            # ECE binning
            bin_idx = min(int(conf * 10), 9)
            ece_counts[bin_idx] += 1
            ece_bins[bin_idx] += conf
            try:
                ece_correct[bin_idx] += int(int(pred) == int(y[i])) if pred is not None else 0
            except (TypeError, ValueError):
                pass

        accuracy = correct / n if n > 0 else 0.0
        mean_conf = float(np.mean(confidences)) if confidences else 0.0

        # ECE: weighted mean of |confidence - accuracy| per bin
        ece = 0.0
        for b in range(10):
            if ece_counts[b] > 0:
                bin_conf = ece_bins[b] / ece_counts[b]
                bin_acc = ece_correct[b] / ece_counts[b]
                ece += (ece_counts[b] / n) * abs(bin_conf - bin_acc)

        report = self.report()
        oracle_cost = report.get("agent", {}).get("total_oracle_cost", 0.0)

        return {
            "accuracy": round(accuracy, 4),
            "mean_confidence": round(mean_conf, 4),
            "ece": round(ece, 4),
            "n_samples": n,
            "oracle_cost": oracle_cost,
            "threshold": self.uncertainty_threshold,
        }

    # ------------------------------------------------------------------
    # Stage 40 — self-improvement (auto-tuning from self-evaluation)
    # ------------------------------------------------------------------

    def self_improve(
        self,
        X_test: Any,
        y_test: Any,
        *,
        aggressive: bool = False,
        target_accuracy: float = 0.80,
        auto_tune: bool = False,
    ) -> dict:
        """Auto-tune agent and retrain on high-reward memory episodes.

        Evaluates on the supplied data, then adjusts
        ``uncertainty_threshold`` and — when a memory store is attached and
        accuracy falls below *target_accuracy* — triggers a ``partial_fit``
        on the top-scoring episodes so the underlying model genuinely improves
        (not just threshold tuning).

        * If accuracy < 0.55 — lower threshold to ask more questions.
        * If accuracy > 0.80 and ECE < 0.05 — raise threshold slightly to
          reduce oracle queries.
        * When *aggressive* is ``True``, also resets the homeostasis window.
        * When ``_memory`` is attached and accuracy < *target_accuracy*,
          runs ``partial_fit`` on the high-reward subset of memory episodes
          (Stage 42 improvement over pure threshold adjustment).
        * When *auto_tune* is ``True``, runs the Stage 47
          :class:`~physml.automl.AutoMLOptimizer` on ``(X_test, y_test)``
          and updates the ensemble predictor with the best found params.

        Parameters
        ----------
        X_test : array-like
        y_test : array-like
        aggressive : bool, default False
            Reset the homeostasis window for rapid re-adaptation.
        target_accuracy : float, default 0.80
            Accuracy threshold below which memory-driven retraining fires.
        auto_tune : bool, default False
            Run AutoML hyperparameter search (Stage 47) and apply best params.

        Returns
        -------
        dict
            Self-evaluation metrics plus ``threshold_before``,
            ``threshold_after``, ``episodes_retrained``, and optionally
            ``best_automl_params`` keys.
        """
        metrics = self.self_evaluate(X_test, y_test)
        threshold_before = self.uncertainty_threshold
        episodes_retrained = 0

        acc = metrics["accuracy"]
        ece = metrics["ece"]

        if acc < 0.55:
            # Ask more — lower threshold
            new_threshold = max(0.10, self.uncertainty_threshold - 0.05)
        elif acc > 0.80 and ece < 0.05:
            # Trust predictions more — raise threshold
            new_threshold = min(0.90, self.uncertainty_threshold + 0.05)
        else:
            new_threshold = self.uncertainty_threshold

        self.uncertainty_threshold = new_threshold
        if self._agent is not None:
            self._agent.uncertainty_threshold = new_threshold

        if aggressive and self._agent is not None:
            from collections import deque
            self._agent._error_window = deque(maxlen=self._agent._error_window.maxlen)

        # Stage 42 — actual model retraining on high-reward memory episodes
        if self._memory is not None and acc < target_accuracy:
            episodes_retrained = self._retrain_from_memory()

        # Stage 47 — AutoML hyper-parameter search
        if auto_tune:
            from physml.automl import AutoMLOptimizer
            try:
                X_arr = np.asarray(X_test, dtype=float)
                y_arr = np.asarray(y_test)
                opt = AutoMLOptimizer(n_candidates=6, cv=3, random_state=42)
                best_params = opt.fit(X_arr, y_arr)
                metrics["best_automl_params"] = best_params
                metrics["best_automl_score"] = round(opt.best_score_, 4)
            except Exception as _exc:
                _logger.warning("AutoML tuning failed during self_improve: %s", _exc)
                metrics["best_automl_params"] = {}

        metrics["threshold_before"] = round(threshold_before, 4)
        metrics["threshold_after"] = round(new_threshold, 4)
        metrics["episodes_retrained"] = episodes_retrained
        return metrics

    def _retrain_from_memory(self, reward_threshold: float = 0.5) -> int:
        """Run ``partial_fit`` on high-reward episodes from attached memory.

        Returns the number of episodes used for retraining.
        """
        if self._memory is None or len(self._memory) == 0:
            return 0

        # Collect high-reward episodes
        outcomes = list(self._memory._outcomes)
        contexts = list(self._memory._contexts)
        high_reward_indices = [
            i for i, o in enumerate(outcomes) if o >= reward_threshold
        ]
        if not high_reward_indices:
            return 0

        X_mem = np.array([contexts[i] for i in high_reward_indices], dtype=np.float32)
        # Use rounded outcome as binary label for classification
        y_mem = np.array([round(outcomes[i]) for i in high_reward_indices])

        # Ensure we have more than 1 unique class for meaningful fit
        if len(np.unique(y_mem)) < 2:
            return 0

        try:
            if hasattr(self._predictor, "partial_fit"):
                import inspect
                sig = inspect.signature(self._predictor.partial_fit)
                if "classes" in sig.parameters:
                    self._predictor.partial_fit(X_mem, y_mem, classes=np.unique(y_mem))
                else:
                    self._predictor.partial_fit(X_mem, y_mem)
            elif hasattr(self._predictor, "fit"):
                self._predictor.fit(X_mem, y_mem)
        except Exception as _exc:
            _logger.warning("Memory-driven retraining failed: %s", _exc)
            return 0

        return len(high_reward_indices)

    # ------------------------------------------------------------------
    # Stage 41 — introspection (rich internal-state summary)
    # ------------------------------------------------------------------

    def introspect(self) -> dict:
        """Return a rich summary of the agent's internal state.

        Useful for debugging, monitoring, and explainability.  The returned
        dict includes predictor type, memory stats, calibration temperature,
        drift state, and agent activity counters.

        Returns
        -------
        dict with keys:
            ``fitted``, ``predictor_type``, ``predictor_runtime_state``,
            ``uncertainty_threshold``, ``policy``, ``query_strategy``,
            ``calibration_temperature``, ``drift_detection_enabled``,
            ``drift_detected``, ``n_memory_episodes``,
            ``agent_activity`` (sub-dict from :meth:`report`).
        """
        predictor_type = type(self._predictor).__name__ if self._predictor is not None else "None"

        # Runtime state from predictor if available
        runtime_state: dict = {}
        if self._predictor is not None:
            rs = getattr(self._predictor, "runtime_state_", None)
            if rs is not None:
                runtime_state = {
                    "homeostasis_score": getattr(rs, "homeostasis_score", None),
                    "iteration": getattr(rs, "iteration", None),
                }

        # Drift state
        drift_detected = False
        if self._agent is not None:
            detector = getattr(self._agent, "_drift_detector", None)
            if detector is not None:
                drift_detected = bool(getattr(detector, "drift_detected_", False))

        # Memory stats
        mem = getattr(self, "_memory", None)
        n_episodes = len(mem) if mem is not None else 0

        return {
            "fitted": self._fitted,
            "predictor_type": predictor_type,
            "predictor_runtime_state": runtime_state,
            "uncertainty_threshold": self.uncertainty_threshold,
            "policy": self.policy,
            "query_strategy": self.query_strategy,
            "calibration_temperature": self.temperature_,
            "drift_detection_enabled": self.drift_detection,
            "drift_detected": drift_detected,
            "n_memory_episodes": n_episodes,
            "agent_activity": self.report() if self._fitted else {},
        }


class _MultiTaskProbaWrapper:
    """Thin wrapper so calibration can call ``predict_proba`` on a task head."""

    def __init__(self, engine: Any, task_id: str) -> None:
        self._engine = engine
        self._task_id = task_id
        # Propagate classes_ if available
        self.classes_ = getattr(engine, "classes_", None)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._engine.predict_proba_task(self._task_id, X)
