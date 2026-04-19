"""Stage 92 — TaskDecomposer: break high-level goals into subtasks.

Provides a rule-based and model-agnostic planner that splits a complex
goal string into an ordered list of concrete subtasks.

Classes
-------
SubTask
    A single decomposed work item.
TaskDecomposer
    Decomposes high-level goals using registered decomposition rules
    or a simple heuristic fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class SubTask:
    """One step produced by task decomposition.

    Attributes
    ----------
    index : int
        Zero-based position in the plan.
    description : str
        Human-readable description of what must be done.
    done : bool
        Whether this subtask has been completed.
    metadata : dict
        Arbitrary extra information (e.g., tool hints).
    """

    index: int
    description: str
    done: bool = False
    metadata: Dict = field(default_factory=dict)

    def complete(self) -> None:
        """Mark this subtask as done."""
        self.done = True


class TaskDecomposer:
    """Decomposes a high-level goal into an ordered list of subtasks.

    Users may register custom *rules* — callable(goal) → list[str] —
    keyed on a keyword present in the goal.  If no rule matches, the
    decomposer splits the goal on commas/semicolons or wraps it in a
    single subtask.

    Parameters
    ----------
    default_steps : list[str], optional
        Fallback subtask descriptions used when no rule fires *and*
        the goal cannot be split automatically.

    Attributes
    ----------
    rules_ : dict[str, Callable]
        Registered decomposition rules.
    """

    def __init__(
        self,
        default_steps: Optional[List[str]] = None,
        llm: Any = None,
    ) -> None:
        self.rules_: Dict[str, Callable[[str], List[str]]] = {}
        self._default_steps = default_steps or []
        self._llm = llm

    # ------------------------------------------------------------------
    def register_rule(self, keyword: str, fn: Callable[[str], List[str]]) -> None:
        """Register a decomposition rule triggered by *keyword*.

        Parameters
        ----------
        keyword : str
            If this string appears (case-insensitive) in the goal,
            *fn* is called to produce subtask descriptions.
        fn : Callable[[str], list[str]]
            Takes the full goal string and returns a list of step
            description strings.
        """
        self.rules_[keyword.lower()] = fn

    # ------------------------------------------------------------------
    def decompose(self, goal: str) -> List[SubTask]:
        """Decompose *goal* into an ordered list of :class:`SubTask` objects.

        Parameters
        ----------
        goal : str
            High-level goal description.

        Returns
        -------
        list[SubTask]
        """
        descriptions = self._apply_rules(goal)
        if not descriptions and self._llm is not None:
            descriptions = self._llm_decompose(goal)
        if not descriptions:
            descriptions = self._heuristic_split(goal)
        return [SubTask(index=i, description=d) for i, d in enumerate(descriptions)]

    def decompose_and_summarise(self, goal: str) -> str:
        """Decompose *goal* and return a numbered plan string."""
        tasks = self.decompose(goal)
        lines = [f"Plan for: {goal}"]
        for t in tasks:
            lines.append(f"  {t.index + 1}. {t.description}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _apply_rules(self, goal: str) -> List[str]:
        lower = goal.lower()
        for keyword, fn in self.rules_.items():
            if keyword in lower:
                return fn(goal)
        return []

    def _llm_decompose(self, goal: str) -> List[str]:
        """Ask Claude to break the goal into numbered steps."""
        try:
            prompt = (
                f"Break the following goal into 3-7 concrete, actionable steps.\n"
                f"Reply with a numbered list only (no extra text).\n\n"
                f"Goal: {goal}"
            )
            result = self._llm.complete(
                prompt,
                system="You are a task planning assistant. Return a numbered list of steps only.",
            )
            if result.available and result.text:
                import re
                lines = result.text.strip().splitlines()
                steps = []
                for line in lines:
                    line = line.strip()
                    m = re.match(r"^\d+[\.\)]\s*(.+)$", line)
                    if m:
                        steps.append(m.group(1).strip())
                    elif line and not line.startswith("#"):
                        steps.append(line)
                if steps:
                    return steps
        except Exception:
            pass
        return []

    def _heuristic_split(self, goal: str) -> List[str]:
        import re

        parts = [p.strip() for p in re.split(r"[,;]+", goal) if p.strip()]
        if len(parts) > 1:
            return parts
        if self._default_steps:
            return list(self._default_steps)
        return [goal.strip()]

    def __repr__(self) -> str:  # pragma: no cover
        return f"TaskDecomposer(rules={list(self.rules_.keys())})"
