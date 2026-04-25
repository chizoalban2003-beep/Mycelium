"""Stage 67 — AutonomousIntegration.

Full end-to-end integration helper that wires all Stage 62-66 components
(WorldModel, IntrinsicMotivation, GoalConditionedPolicy, SafetyMonitor) into
a unified ``AutonomousAgent`` façade that delegates prediction to a
``MyceliumAgent`` or any compatible predictor.

Classes
-------
AutonomousAgent
    Top-level autonomous agent integrating world model, curiosity,
    goal conditioning, and safety screening around a MyceliumAgent core.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from physml.world_model import WorldModel
from physml.intrinsic import IntrinsicMotivation
from physml.arena import CompetitiveArena, ArenaResult
from physml.goal_policy import GoalConditionedPolicy, GoalSpec
from physml.safety import SafetyMonitor


class AutonomousAgent:
    """Fully integrated competitive autonomous agent.

    Wraps a ``MyceliumAgent`` (or any sklearn-compatible estimator) with:
    - **WorldModel** for model-based action planning
    - **IntrinsicMotivation** for curiosity-driven exploration
    - **GoalConditionedPolicy** for structured goal pursuit
    - **SafetyMonitor** for constraint enforcement

    Parameters
    ----------
    core : Any
        A fitted or unfitted ``MyceliumAgent`` (or sklearn estimator) that
        provides ``fit(X, y)`` and ``predict(X)`` / ``predict_proba(X)``.
    n_actions : int
        Number of discrete actions available in the environment.
    horizon : int
        Rollout horizon for the world model planner.
    bonus_scale : float
        Intrinsic motivation bonus scale factor.
    embedding_dim : int
        Goal embedding dimension for the goal-conditioned policy.
    safe_action : int
        Fallback action when safety screening rejects all candidates.
    """

    def __init__(
        self,
        core: Any,
        *,
        n_actions: int = 2,
        horizon: int = 3,
        bonus_scale: float = 0.1,
        embedding_dim: int = 16,
        safe_action: int = 0,
    ) -> None:
        self.core = core
        self.n_actions = n_actions

        self.world_model = WorldModel(horizon=horizon, n_actions=n_actions)
        self.curiosity = IntrinsicMotivation(bonus_scale=bonus_scale)
        self.goal_policy = GoalConditionedPolicy(
            n_actions=n_actions, embedding_dim=embedding_dim
        )
        self.safety = SafetyMonitor(safe_action=safe_action)

        self._step = 0
        self._total_reward = 0.0

    # ------------------------------------------------------------------
    # Delegation to core predictor
    # ------------------------------------------------------------------

    def fit(self, X: Any, y: Any) -> "AutonomousAgent":
        """Fit the core predictor."""
        self.core.fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        """Predict using the core predictor (supports both observe() and predict())."""
        import numpy as np

        if hasattr(self.core, "predict"):
            return self.core.predict(X)
        if hasattr(self.core, "observe"):
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                result = self.core.observe(X_arr)
                # Unwrap AgentAction if needed
                if hasattr(result, "prediction"):
                    return result.prediction
                return result
            preds = []
            for row in X_arr:
                result = self.core.observe(row)
                if hasattr(result, "prediction"):
                    preds.append(result.prediction)
                else:
                    preds.append(result)
            return np.array(preds)
        raise AttributeError("Core has neither predict() nor observe()")

    def predict_proba(self, X: Any) -> Any:
        """Predict class probabilities using the core predictor (if supported)."""
        if hasattr(self.core, "predict_proba"):
            return self.core.predict_proba(X)
        # MyceliumAgent wraps a predictor with predict_proba via _predictor
        if hasattr(self.core, "_predictor") and hasattr(self.core._predictor, "predict_proba"):
            import numpy as np
            X_arr = np.asarray(X)
            return self.core._predictor.predict_proba(X_arr)
        raise AttributeError("Core predictor does not support predict_proba")

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------

    def act(
        self,
        state: np.ndarray,
        goal: GoalSpec | str | None = None,
    ) -> int:
        """Select an action via integrated planning + goal + safety pipeline.

        Order of priority:
        1. Goal-conditioned policy (if goal provided and policy fitted)
        2. World-model planner (if model fitted)
        3. Fallback: action 0

        The chosen action is then screened by the SafetyMonitor.
        """
        s = np.asarray(state, dtype=np.float64).ravel()

        if goal is not None and self.goal_policy._fitted:
            candidate = self.goal_policy.act(s, goal)
        elif self.world_model.fitted_:
            candidate = self.world_model.plan(s)
        else:
            candidate = 0

        return self.safety.screen(s, candidate, list(range(self.n_actions)))

    def step(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action: int,
        extrinsic_reward: float,
        goal: GoalSpec | str | None = None,
    ) -> float:
        """Record a transition and return the shaped total reward.

        Parameters
        ----------
        state, next_state : np.ndarray
            Pre/post-transition environment state.
        action : int
            Action taken.
        extrinsic_reward : float
            Reward from the environment.
        goal : GoalSpec or str or None
            Current goal (used to update goal policy).

        Returns
        -------
        float
            Total reward = extrinsic + intrinsic bonus - safety penalty.
        """
        s = np.asarray(state, dtype=np.float64).ravel()
        s_next = np.asarray(next_state, dtype=np.float64).ravel()

        # Intrinsic curiosity bonus
        bonus = self.curiosity.bonus(s, s_next)

        # Safety penalty
        penalty = self.safety.penalty_for(s, action)

        # World model update
        self.world_model.record(s, action, s_next, extrinsic_reward)
        if self._step % 10 == 0:
            self.world_model.update()

        # Goal policy update
        if goal is not None:
            self.goal_policy.update(s, goal, action)

        total = extrinsic_reward + bonus - penalty
        self._total_reward += total
        self._step += 1
        return total

    # ------------------------------------------------------------------
    # Competitive benchmark
    # ------------------------------------------------------------------

    def compete(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        baselines: dict[str, Any] | None = None,
    ) -> list[ArenaResult]:
        """Run a competitive arena benchmark vs. provided baselines.

        Parameters
        ----------
        baselines : dict mapping name → estimator
            Competitors to include.  If None, competes alone.

        Returns
        -------
        list[ArenaResult]
            Ranked results (best first).
        """
        arena = CompetitiveArena(metric="accuracy")
        arena.register("AutonomousAgent(Mycelium)", self)
        if baselines:
            for name, agent in baselines.items():
                arena.register(name, agent)
        return arena.run(X_train, y_train, X_test, y_test)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Full agent status report."""
        return {
            "steps": self._step,
            "total_reward": round(self._total_reward, 4),
            "world_model": self.world_model.summary(),
            "curiosity": self.curiosity.summary(),
            "goal_policy": self.goal_policy.summary(),
            "safety": self.safety.report(),
        }
