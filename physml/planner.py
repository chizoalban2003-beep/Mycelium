"""Stage 32 — GoalPlanner: multi-step task decomposition and execution.

Provides:
* :class:`SubTask` — a single sub-goal with featurized representation and
  dependency tracking.
* :class:`GoalPlanner` — decomposes a free-text goal into a sequence of
  :class:`SubTask` objects and executes them via a
  :class:`~physml.mycelium_agent.MyceliumAgent`.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from physml.featurizer import Featurizer
    from physml.mycelium_agent import MyceliumAgent


@dataclass
class SubTask:
    """A single decomposed sub-goal.

    Parameters
    ----------
    task_id : str
        Unique identifier.
    description : str
        Human-readable description of the sub-task.
    feature_vec : np.ndarray
        Featurized representation used for agent inference.
    depends_on : list[str]
        ``task_id`` values of tasks that must complete before this one.
    """

    task_id: str
    description: str
    feature_vec: np.ndarray
    depends_on: list[str] = field(default_factory=list)


class GoalPlanner:
    """Decompose and execute multi-step goals.

    Parameters
    ----------
    featurizer : Featurizer
        A fitted Featurizer used to embed sub-task descriptions.
    agent : MyceliumAgent
        A fitted MyceliumAgent used to process each sub-task.
    n_subtasks : int, default 3
        Target number of sub-tasks to decompose a goal into.
    """

    def __init__(
        self,
        featurizer: "Featurizer",
        agent: "MyceliumAgent",
        n_subtasks: int = 3,
    ) -> None:
        self.featurizer = featurizer
        self.agent = agent
        self.n_subtasks = max(1, int(n_subtasks))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, goal: str) -> list[SubTask]:
        """Decompose *goal* into :attr:`n_subtasks` :class:`SubTask` objects.

        Uses sentence/clause splitting followed by word-chunk fallback to
        produce exactly ``n_subtasks`` parts.  Each task depends on the
        previous one (linear chain).

        Parameters
        ----------
        goal : str

        Returns
        -------
        list[SubTask]
        """
        parts = self._split_goal(goal, self.n_subtasks)
        subtasks: list[SubTask] = []
        prev_ids: list[str] = []

        for i, part in enumerate(parts):
            task_id = f"task_{i}_{uuid.uuid4().hex[:6]}"
            feat_vec = self.featurizer.transform([part])[0]
            subtask = SubTask(
                task_id=task_id,
                description=part,
                feature_vec=feat_vec,
                depends_on=list(prev_ids),
            )
            subtasks.append(subtask)
            prev_ids = [task_id]  # linear chain: each depends on previous

        return subtasks

    def execute(self, goal: str) -> dict:
        """Plan and execute each sub-task via :meth:`~physml.mycelium_agent.MyceliumAgent.observe`.

        Returns
        -------
        dict with keys:
            ``plan`` (list of dicts), ``results`` (list of dicts),
            ``n_steps`` (int).
        """
        subtasks = self.plan(goal)
        sorted_tasks = self._topo_sort(subtasks)

        results: list[dict] = []
        completed: set[str] = set()

        for task in sorted_tasks:
            # Skip if dependencies not met (should not happen with linear chain)
            if not all(dep in completed for dep in task.depends_on):
                continue

            try:
                action = self.agent.observe(task.feature_vec.reshape(1, -1))
                results.append(
                    {
                        "task_id": task.task_id,
                        "description": task.description,
                        "action": str(getattr(action, "action", action)),
                        "prediction": str(getattr(action, "prediction", None)),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "task_id": task.task_id,
                        "description": task.description,
                        "action": "error",
                        "error": str(exc),
                    }
                )

            completed.add(task.task_id)

        return {
            "plan": [
                {
                    "task_id": t.task_id,
                    "description": t.description,
                    "depends_on": t.depends_on,
                }
                for t in subtasks
            ],
            "results": results,
            "n_steps": len(sorted_tasks),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_goal(self, goal: str, n: int) -> list[str]:
        """Split *goal* into *n* non-empty string parts."""
        separators = [
            r"\.\s+",
            r";\s*",
            r",\s+(?:and|then|next|after|first|second|third)\s+",
            r"\s+and\s+",
            r"\s+then\s+",
        ]

        parts: list[str] = [goal]
        for sep in separators:
            new_parts: list[str] = []
            for p in parts:
                split = re.split(sep, p, flags=re.IGNORECASE)
                new_parts.extend(s.strip() for s in split if s.strip())
            parts = new_parts
            if len(parts) >= n:
                break

        # Word-chunk fallback if we still don't have enough parts
        if len(parts) < n:
            words = goal.split()
            chunk_size = max(1, len(words) // n)
            parts = []
            for i in range(n):
                chunk = words[i * chunk_size : (i + 1) * chunk_size]
                if chunk:
                    parts.append(" ".join(chunk))
            remaining = words[n * chunk_size :]
            if remaining and parts:
                parts[-1] += " " + " ".join(remaining)

        # Ensure exactly n parts
        if len(parts) > n:
            parts = parts[:n]
        while len(parts) < n:
            parts.append(goal)

        return parts

    def _topo_sort(self, subtasks: list[SubTask]) -> list[SubTask]:
        """Return subtasks in dependency order (Kahn's algorithm)."""
        in_degree: dict[str, int] = {t.task_id: len(t.depends_on) for t in subtasks}
        ready = [t for t in subtasks if in_degree[t.task_id] == 0]
        sorted_tasks: list[SubTask] = []

        while ready:
            task = ready.pop(0)
            sorted_tasks.append(task)
            for other in subtasks:
                if task.task_id in other.depends_on:
                    in_degree[other.task_id] -= 1
                    if in_degree[other.task_id] == 0:
                        ready.append(other)

        return sorted_tasks
