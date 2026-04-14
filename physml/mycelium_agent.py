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

        self._agent: Any = None  # built after fit()
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: Any, y: Any) -> "MyceliumAgent":
        """Fit the underlying predictor on seed data and initialise the agent.

        Must be called at least once before :meth:`observe`.

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
        return self._agent.observe(X)

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

    def reward(self, X: Any, y_true: Any, *, immediate: bool = True) -> "MyceliumAgent":
        """Provide a ground-truth label so the agent can learn from it.

        Parameters
        ----------
        X : array-like
        y_true : array-like
        immediate : bool, default True

        Returns
        -------
        self
        """
        self._require_fitted()
        self._agent.reward(X, y_true, immediate=immediate)
        return self

    def report(self) -> dict[str, Any]:
        """Return a summary of agent activity and configuration.

        Returns
        -------
        dict with keys:
            agent (PhysicsAgent report sub-dict), query_strategy, policy,
            task_id, fitted.
        """
        agent_report = self._agent.report() if self._agent is not None else {}
        return {
            "agent": agent_report,
            "query_strategy": self.query_strategy,
            "policy": self.policy,
            "task_id": self.task_id,
            "fitted": self._fitted,
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
        )

    def _require_fitted(self) -> None:
        if not self._fitted or self._agent is None:
            raise RuntimeError(
                "MyceliumAgent is not fitted yet.  Call fit(X_seed, y_seed) first."
            )
