"""Stage 95 — PlanExecutor: execute ordered plans of subtasks.

Runs a sequence of :class:`~physml.task_decomposer.SubTask` objects by
dispatching each one to a registered *action handler*.  Tracks outcomes,
supports retry logic, and records a structured :class:`ExecutionResult`.

Classes
-------
ExecutionResult
    Summary of one plan execution run.
PlanExecutor
    Executes plans produced by :class:`~physml.task_decomposer.TaskDecomposer`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ExecutionResult:
    """Summary of one plan-execution attempt.

    Attributes
    ----------
    plan_id : str
        Identifier of the executed plan.
    total : int
        Total number of subtasks in the plan.
    completed : int
        Number of subtasks that finished successfully.
    failed : int
        Number of subtasks that raised an error.
    skipped : int
        Subtasks skipped due to a prior failure when
        ``stop_on_error=True``.
    elapsed : float
        Wall-clock seconds taken for the entire run.
    outcomes : list[dict]
        Per-subtask outcome records.
    success : bool
        ``True`` iff all subtasks completed without error.
    """

    plan_id: str
    total: int
    completed: int
    failed: int
    skipped: int
    elapsed: float
    outcomes: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.failed == 0 and self.skipped == 0


class PlanExecutor:
    """Executes an ordered list of subtasks via registered handlers.

    Parameters
    ----------
    stop_on_error : bool
        If ``True`` (default), remaining subtasks are skipped after the
        first failure.
    max_retries : int
        Number of extra attempts for a failing subtask before marking it
        as failed.  Default is 0 (no retries).
    default_handler : Callable[[SubTask], Any], optional
        Fallback handler used when no handler is registered for a
        subtask description keyword.  If *None* and no handler matches,
        a ``RuntimeError`` is raised.

    Attributes
    ----------
    handlers_ : dict[str, Callable]
        Registered keyword-to-handler mappings.
    history_ : list[ExecutionResult]
        All execution results produced so far.
    """

    def __init__(
        self,
        stop_on_error: bool = True,
        max_retries: int = 0,
        default_handler: Optional[Callable] = None,
    ) -> None:
        self.stop_on_error = stop_on_error
        self.max_retries = max_retries
        self.default_handler = default_handler
        self.handlers_: Dict[str, Callable] = {}
        self.history_: List[ExecutionResult] = []

    # ------------------------------------------------------------------
    def register(self, keyword: str, fn: Callable) -> None:
        """Register *fn* as the handler for subtasks whose description
        contains *keyword* (case-insensitive).

        Parameters
        ----------
        keyword : str
        fn : Callable[[SubTask], Any]
        """
        self.handlers_[keyword.lower()] = fn

    # ------------------------------------------------------------------
    def execute(self, subtasks: list, plan_id: str = "plan") -> ExecutionResult:
        """Execute *subtasks* in order.

        Parameters
        ----------
        subtasks : list[SubTask]
            Ordered subtask list (e.g., from
            :meth:`~physml.task_decomposer.TaskDecomposer.decompose`).
        plan_id : str
            Identifier stored in the returned result.

        Returns
        -------
        ExecutionResult
        """
        start = time.time()
        completed = failed = skipped = 0
        outcomes: List[Dict[str, Any]] = []
        abort = False

        for task in subtasks:
            if abort:
                skipped += 1
                outcomes.append(
                    {"index": task.index, "description": task.description, "status": "skipped"}
                )
                continue

            handler = self._resolve_handler(task)
            outcome = self._run_with_retry(task, handler)
            outcomes.append(outcome)

            if outcome["status"] == "ok":
                task.complete()
                completed += 1
            else:
                failed += 1
                if self.stop_on_error:
                    abort = True

        result = ExecutionResult(
            plan_id=plan_id,
            total=len(subtasks),
            completed=completed,
            failed=failed,
            skipped=skipped,
            elapsed=time.time() - start,
            outcomes=outcomes,
        )
        self.history_.append(result)
        return result

    # ------------------------------------------------------------------
    def _resolve_handler(self, task) -> Optional[Callable]:
        desc_lower = task.description.lower()
        for keyword, fn in self.handlers_.items():
            if keyword in desc_lower:
                return fn
        if self.default_handler is not None:
            return self.default_handler
        return None

    def _run_with_retry(self, task, handler: Optional[Callable]) -> Dict[str, Any]:
        if handler is None:
            return {
                "index": task.index,
                "description": task.description,
                "status": "error",
                "error": "No handler registered",
            }

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                result = handler(task)
                return {
                    "index": task.index,
                    "description": task.description,
                    "status": "ok",
                    "result": result,
                    "attempts": attempt + 1,
                }
            except Exception as exc:  # noqa: BLE001
                last_exc = exc

        return {
            "index": task.index,
            "description": task.description,
            "status": "error",
            "error": str(last_exc),
            "attempts": self.max_retries + 1,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PlanExecutor(stop_on_error={self.stop_on_error}, "
            f"max_retries={self.max_retries}, "
            f"handlers={list(self.handlers_.keys())})"
        )
