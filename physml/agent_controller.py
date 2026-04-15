"""Stage 99 — AgentController: top-level autonomous-agent control loop.

Ties together the agent's core subsystems — memory, task decomposer,
plan executor, skill library, belief updater, and reflection engine —
into a single ``step()`` / ``run()`` interface.

Classes
-------
ControlStep
    Record of one control-loop iteration.
AgentController
    Orchestrates perception → belief → planning → execution → reflection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class ControlStep:
    """Record produced by one control-loop iteration.

    Attributes
    ----------
    step_id : int
        Zero-based iteration index.
    goal : str
        The goal string that was processed.
    plan_success : bool
        Whether all plan subtasks completed without error.
    reward : float
        Reward signal for this step.
    reflection_trend : str or None
        Trend string from the reflection engine (if reflection was run).
    elapsed : float
        Wall-clock seconds for the step.
    metadata : dict
        Any extra information from subsystems.
    """

    step_id: int
    goal: str
    plan_success: bool
    reward: float
    reflection_trend: Optional[str]
    elapsed: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentController:
    """Orchestrates the autonomous-agent control loop.

    The controller wires together optional subsystems.  Any subsystem can
    be omitted (pass ``None``); the controller degrades gracefully.

    Parameters
    ----------
    memory : AgentMemory, optional
    task_decomposer : TaskDecomposer, optional
    plan_executor : PlanExecutor, optional
    skill_library : SkillLibrary, optional
    belief_updater : BeliefUpdater, optional
    reflection_engine : ReflectionEngine, optional
    reward_fn : Callable[[ControlStep], float], optional
        Called after each step to compute the reward signal.
        Defaults to ``lambda step: 1.0 if step.plan_success else -1.0``.
    reflect_every : int
        Number of steps between reflection calls.  Default ``5``.

    Attributes
    ----------
    steps_ : list[ControlStep]
        All completed control steps.
    step_count_ : int
        Total steps executed.
    """

    def __init__(
        self,
        memory=None,
        task_decomposer=None,
        plan_executor=None,
        skill_library=None,
        belief_updater=None,
        reflection_engine=None,
        reward_fn: Optional[Callable] = None,
        reflect_every: int = 5,
    ) -> None:
        self.memory = memory
        self.task_decomposer = task_decomposer
        self.plan_executor = plan_executor
        self.skill_library = skill_library
        self.belief_updater = belief_updater
        self.reflection_engine = reflection_engine
        self.reward_fn = reward_fn or (lambda s: 1.0 if s.plan_success else -1.0)
        self.reflect_every = max(1, reflect_every)
        self.steps_: List[ControlStep] = []
        self.step_count_: int = 0

    # ------------------------------------------------------------------
    def step(
        self,
        goal: str,
        observation: Optional[Any] = None,
        evidence: Optional[str] = None,
    ) -> ControlStep:
        """Run one iteration of the perception → planning → execution loop.

        Parameters
        ----------
        goal : str
            High-level goal for this step.
        observation : Any, optional
            Observation to record in memory.
        evidence : str, optional
            Evidence to feed to the belief updater.

        Returns
        -------
        ControlStep
        """
        start = time.time()
        meta: Dict[str, Any] = {}

        # 1. Perception — store observation
        if self.memory is not None and observation is not None:
            self.memory.record(observation=observation)

        # 2. Belief update
        if self.belief_updater is not None and evidence is not None:
            belief = self.belief_updater.update(evidence)
            meta["most_likely"] = belief.most_likely

        # 3. Plan
        subtasks: list = []
        if self.task_decomposer is not None:
            subtasks = self.task_decomposer.decompose(goal)
            meta["subtask_count"] = len(subtasks)

        # 4. Execute
        plan_success = True
        if subtasks and self.plan_executor is not None:
            result = self.plan_executor.execute(subtasks, plan_id=f"step_{self.step_count_}")
            plan_success = result.success
            meta["plan_completed"] = result.completed
            meta["plan_failed"] = result.failed

        # 5. Compute reward
        # Build a provisional ControlStep to pass to reward_fn
        provisional = ControlStep(
            step_id=self.step_count_,
            goal=goal,
            plan_success=plan_success,
            reward=0.0,
            reflection_trend=None,
            elapsed=0.0,
            metadata=meta,
        )
        reward = float(self.reward_fn(provisional))

        # 6. Log to reflection engine
        reflection_trend: Optional[str] = None
        if self.reflection_engine is not None:
            self.reflection_engine.log_reward(reward)
            if (self.step_count_ + 1) % self.reflect_every == 0:
                try:
                    ref = self.reflection_engine.reflect()
                    reflection_trend = ref.trend
                    meta["reflection_insights"] = ref.insights
                except RuntimeError:
                    pass

        elapsed = time.time() - start
        cs = ControlStep(
            step_id=self.step_count_,
            goal=goal,
            plan_success=plan_success,
            reward=reward,
            reflection_trend=reflection_trend,
            elapsed=elapsed,
            metadata=meta,
        )
        self.steps_.append(cs)
        self.step_count_ += 1
        return cs

    # ------------------------------------------------------------------
    def run(
        self,
        goals: Sequence[str],
        observations: Optional[Sequence[Any]] = None,
        evidences: Optional[Sequence[Optional[str]]] = None,
    ) -> List[ControlStep]:
        """Run multiple control steps sequentially.

        Parameters
        ----------
        goals : sequence of str
        observations : sequence, optional
        evidences : sequence of str or None, optional

        Returns
        -------
        list[ControlStep]
        """
        results = []
        for i, goal in enumerate(goals):
            obs = observations[i] if observations is not None else None
            ev = evidences[i] if evidences is not None else None
            results.append(self.step(goal, observation=obs, evidence=ev))
        return results

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Return a high-level summary of all executed steps."""
        if not self.steps_:
            return {"total_steps": 0}
        rewards = [s.reward for s in self.steps_]
        successes = sum(1 for s in self.steps_ if s.plan_success)
        return {
            "total_steps": len(self.steps_),
            "success_rate": successes / len(self.steps_),
            "avg_reward": sum(rewards) / len(rewards),
            "total_reward": sum(rewards),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AgentController(steps={self.step_count_}, "
            f"reflect_every={self.reflect_every})"
        )
