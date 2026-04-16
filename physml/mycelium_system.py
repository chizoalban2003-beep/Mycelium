"""Stage 100 — MyceliumSystem: grand-finale integration layer.

Ties every prior subsystem into a single production-ready autonomous agent:

* **Perception & belief** — :class:`~physml.belief_updater.BeliefUpdater`
* **Planning** — :class:`~physml.task_decomposer.TaskDecomposer` +
  :class:`~physml.plan_executor.PlanExecutor`
* **Skill execution** — :class:`~physml.skill_library.SkillLibrary`
* **Memory** — :class:`~physml.agent_memory.AgentMemory`
* **Communication** — :class:`~physml.agent_comms.AgentComms`
* **Reflection** — :class:`~physml.reflection_engine.ReflectionEngine`
* **Reward** — :class:`~physml.reward_model.RewardModel`
* **Control** — :class:`~physml.agent_controller.AgentController`
* **Environment model** — :class:`~physml.environment_model.EnvironmentModel`

Classes
-------
SystemMetrics
    Aggregate statistics collected over all system steps.
MyceliumSystem
    Top-level orchestrator that wires all subsystems together and exposes
    ``step()``, ``run()``, ``save()``, and ``load()`` interfaces.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# SystemMetrics
# ---------------------------------------------------------------------------


@dataclass
class SystemMetrics:
    """Aggregate statistics produced by :class:`MyceliumSystem`.

    Attributes
    ----------
    total_steps : int
        Number of control-loop iterations executed.
    successful_steps : int
        Steps where the execution plan succeeded.
    failed_steps : int
        Steps where one or more plan tasks failed.
    total_reward : float
        Cumulative reward across all steps.
    avg_reward : float
        Mean reward per step (0.0 when *total_steps* is 0).
    avg_step_time : float
        Average wall-clock seconds per step.
    uptime : float
        Total wall-clock seconds since the system was created.
    """

    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    avg_step_time: float = 0.0
    uptime: float = 0.0

    def update(self, success: bool, reward: float, elapsed: float) -> None:
        """Record the outcome of one step."""
        self.total_steps += 1
        if success:
            self.successful_steps += 1
        else:
            self.failed_steps += 1
        self.total_reward += reward
        self.avg_reward = self.total_reward / self.total_steps
        # Running average for step time
        self.avg_step_time = (
            (self.avg_step_time * (self.total_steps - 1) + elapsed)
            / self.total_steps
        )

    @property
    def success_rate(self) -> float:
        """Fraction of successful steps (0.0–1.0)."""
        if self.total_steps == 0:
            return 0.0
        return self.successful_steps / self.total_steps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "successful_steps": self.successful_steps,
            "failed_steps": self.failed_steps,
            "total_reward": self.total_reward,
            "avg_reward": self.avg_reward,
            "avg_step_time": self.avg_step_time,
            "uptime": self.uptime,
            "success_rate": self.success_rate,
        }


# ---------------------------------------------------------------------------
# MyceliumSystem
# ---------------------------------------------------------------------------


class MyceliumSystem:
    """Grand-finale integration: a fully autonomous, self-improving agent.

    Parameters
    ----------
    agent_id : str, optional
        Stable identifier for this system instance.  Defaults to
        ``"mycelium-1"``.
    max_memory : int, optional
        Maximum number of entries kept in the agent memory store.
    skill_handlers : dict, optional
        Mapping of skill name → callable passed to the
        :class:`~physml.skill_library.SkillLibrary`.
    plan_handler : callable, optional
        Default handler forwarded to
        :class:`~physml.plan_executor.PlanExecutor`.
    stop_on_error : bool, optional
        Whether the plan executor stops when a task fails.
    reflect_every : int, optional
        Run the reflection engine every *N* steps (0 = never).
    verbose : bool, optional
        Print a one-line summary after each step.
    """

    def __init__(
        self,
        agent_id: str = "mycelium-1",
        max_memory: int = 1000,
        skill_handlers: Optional[Dict[str, Callable]] = None,
        plan_handler: Optional[Callable] = None,
        stop_on_error: bool = False,
        reflect_every: int = 5,
        verbose: bool = False,
    ) -> None:
        from physml.agent_comms import AgentComms
        from physml.agent_controller import AgentController
        from physml.agent_memory import AgentMemory
        from physml.belief_updater import BeliefUpdater
        from physml.environment_model import EnvironmentModel
        from physml.plan_executor import PlanExecutor
        from physml.reflection_engine import ReflectionEngine
        from physml.reward_model import RewardModel
        from physml.skill_library import SkillLibrary
        from physml.task_decomposer import TaskDecomposer

        self.agent_id = agent_id
        self.reflect_every = reflect_every
        self.verbose = verbose
        self._created_at = time.time()

        # Subsystems
        self.memory = AgentMemory(max_episodic=max_memory)
        self.comms = AgentComms()
        self.env_model = EnvironmentModel()
        self.belief_updater = BeliefUpdater(hypotheses=["success", "failure"])
        self.task_decomposer = TaskDecomposer()
        self.plan_executor = PlanExecutor(
            default_handler=plan_handler,
            stop_on_error=stop_on_error,
        )
        self.skill_library = SkillLibrary()
        self.reward_model = RewardModel()
        self.reflection_engine = ReflectionEngine()
        self.controller = AgentController(
            memory=self.memory,
            task_decomposer=self.task_decomposer,
            plan_executor=self.plan_executor,
            skill_library=self.skill_library,
            belief_updater=self.belief_updater,
            reflection_engine=self.reflection_engine,
            reflect_every=reflect_every,
        )

        # Register skill handlers
        if skill_handlers:
            for name, fn in skill_handlers.items():
                self.skill_library.register(name=name, fn=fn)

        # Metrics
        self.metrics = SystemMetrics()
        self._step_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(
        self,
        goal: str,
        observation: Optional[Dict[str, Any]] = None,
        reward_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute one autonomous control-loop iteration.

        Parameters
        ----------
        goal : str
            High-level goal string for this step.
        observation : dict, optional
            Environmental observations forwarded to the belief updater
            and environment model.
        reward_override : float, optional
            If provided, bypasses the reward model and uses this value.

        Returns
        -------
        dict
            Summary of the step including ``goal``, ``success``,
            ``reward``, ``elapsed``, and ``reflection_trend``.
        """
        t0 = time.time()

        # Update environment model if observation given
        if observation:
            self.env_model.record(observation)

        # Delegate to AgentController
        ctrl_step = self.controller.step(
            goal=goal,
        )

        # Apply reward override after the fact
        if reward_override is not None:
            from physml.agent_controller import ControlStep as _CS
            ctrl_step = _CS(
                step_id=ctrl_step.step_id,
                goal=ctrl_step.goal,
                plan_success=ctrl_step.plan_success,
                reward=float(reward_override),
                reflection_trend=ctrl_step.reflection_trend,
                elapsed=ctrl_step.elapsed,
                metadata=ctrl_step.metadata,
            )

        elapsed = time.time() - t0
        self.metrics.update(ctrl_step.plan_success, ctrl_step.reward, elapsed)
        self.metrics.uptime = time.time() - self._created_at

        record: Dict[str, Any] = {
            "step_id": ctrl_step.step_id,
            "goal": goal,
            "success": ctrl_step.plan_success,
            "reward": ctrl_step.reward,
            "elapsed": elapsed,
            "reflection_trend": ctrl_step.reflection_trend,
        }
        self._step_log.append(record)

        if self.verbose:
            status = "✓" if ctrl_step.plan_success else "✗"
            print(
                f"[{self.agent_id}] step={ctrl_step.step_id} "
                f"{status} reward={ctrl_step.reward:.3f} "
                f"({elapsed*1000:.1f} ms)"
            )

        return record

    def run(
        self,
        goals: Sequence[str],
        observations: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """Run the system over a sequence of goals.

        Parameters
        ----------
        goals : sequence of str
            Ordered list of goal strings.
        observations : sequence of dict or None, optional
            Per-goal observation dicts (``None`` entries are skipped).

        Returns
        -------
        list of dict
            One record per goal (same format as :meth:`step`).
        """
        results = []
        for i, goal in enumerate(goals):
            obs = (observations[i] if observations and i < len(observations) else None)
            results.append(self.step(goal, observation=obs))
        return results

    def broadcast(self, content: str, recipients: Optional[List[str]] = None) -> None:
        """Publish a message via AgentComms."""
        from physml.agent_comms import Message

        targets = recipients or ["*"]
        for recipient in targets:
            msg = Message(
                sender=self.agent_id,
                topic="broadcast",
                content=content,
                recipient=recipient,
            )
            self.comms.publish(msg)

    def report(self) -> Dict[str, Any]:
        """Return a snapshot of system metrics and state."""
        self.metrics.uptime = time.time() - self._created_at
        return {
            "agent_id": self.agent_id,
            "metrics": self.metrics.to_dict(),
            "memory_size": len(self.memory.episodic),
            "skill_count": len(self.skill_library.list_names()),
            "step_count": len(self._step_log),
        }

    def save(self, path: str) -> None:
        """Persist system report and step log to *path* as JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "agent_id": self.agent_id,
            "metrics": self.metrics.to_dict(),
            "step_log": self._step_log,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def load(self, path: str) -> None:
        """Restore step log and metrics from a previously saved JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        m = data.get("metrics", {})
        self.metrics.total_steps = m.get("total_steps", 0)
        self.metrics.successful_steps = m.get("successful_steps", 0)
        self.metrics.failed_steps = m.get("failed_steps", 0)
        self.metrics.total_reward = m.get("total_reward", 0.0)
        self.metrics.avg_reward = m.get("avg_reward", 0.0)
        self.metrics.avg_step_time = m.get("avg_step_time", 0.0)
        self.metrics.uptime = m.get("uptime", 0.0)
        self._step_log = data.get("step_log", [])

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        """Total number of steps executed."""
        return self.metrics.total_steps

    @property
    def success_rate(self) -> float:
        """Fraction of successful steps."""
        return self.metrics.success_rate

    def reset_metrics(self) -> None:
        """Zero-out all accumulated metrics (useful for benchmarking)."""
        self.metrics = SystemMetrics()
        self._step_log.clear()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MyceliumSystem(agent_id={self.agent_id!r}, "
            f"steps={self.step_count}, "
            f"success_rate={self.success_rate:.1%})"
        )
