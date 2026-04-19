"""Stage 137 — GoalEngine: autonomous long-horizon goal execution loop.

The missing bridge between *planning* and *doing*.  Previous stages could
decompose a goal into subtasks (Stage 92) and execute a flat list (Stage 95),
but there was no persistent, tool-backed, background-capable loop that could
actually *drive* those subtasks to completion autonomously.

GoalEngine provides:

* **Persistent goal queue** — goals survive restarts (JSON on disk).
* **Lifecycle tracking** — pending → active → completed / failed / blocked.
* **Tool dispatch** — subtask descriptions are matched to real tool handlers
  (document read, model train, predict, web browse, notify, etc.).
* **Background loop** — processes pending goals while the companion is idle.
* **Retry + escalation** — retries failures up to ``max_retries`` then marks
  the goal blocked and invokes an optional escalation callback.
* **Permission gating** — checks ``PermissionManager`` before destructive
  actions.
* **Completion notifications** — fires ``Notifier`` on goal complete/fail.

Usage
-----
::

    from physml.goal_engine import GoalEngine

    engine = GoalEngine(
        task_decomposer=td,
        companion=companion,
        notifier=notifier,
        permission_manager=pm,
        state_dir="~/.mycelium/goals",
    )

    goal_id = engine.add_goal("Read sales.csv and tell me if revenue is falling")
    engine.start_loop()          # runs in background

    # or synchronously:
    record = engine.run_now(goal_id)
    print(record.status)         # GoalStatus.COMPLETED
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class GoalStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


@dataclass
class StepResult:
    """Outcome of executing one subtask step."""

    index: int
    description: str
    status: str           # "ok" | "error" | "skipped"
    output: str = ""
    error: str = ""
    elapsed: float = 0.0


@dataclass
class GoalRecord:
    """Full lifecycle record for one goal."""

    id: str
    description: str
    status: GoalStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    steps: List[Dict] = field(default_factory=list)
    retries: int = 0
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.completed_at or time.time()
        return end - self.started_at

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "steps": self.steps,
            "retries": self.retries,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GoalRecord":
        return cls(
            id=d["id"],
            description=d["description"],
            status=GoalStatus(d.get("status", "pending")),
            created_at=d.get("created_at", time.time()),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            steps=d.get("steps", []),
            retries=d.get("retries", 0),
            error=d.get("error"),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# GoalEngine
# ---------------------------------------------------------------------------


class GoalEngine:
    """Autonomous goal queue with background execution loop.

    Parameters
    ----------
    task_decomposer : TaskDecomposer or None
        Breaks goal descriptions into subtask steps.
    companion : MyceliumCompanion or None
        Provides access to all tool subsystems.
    notifier : Notifier or None
        Desktop notifications on completion/failure.
    permission_manager : PermissionManager or None
        Action gating before destructive steps.
    llm : LLMIntegration or None
        Used for open-ended "analyse / summarise" steps.
    state_dir : str
        Directory where goal state JSON is persisted.
    max_retries : int
        How many times to retry a failing goal before marking it blocked.
    loop_interval : float
        Seconds between background loop ticks.
    escalation_callback : callable or None
        Called with (GoalRecord) when a goal becomes blocked after all retries.
    """

    def __init__(
        self,
        task_decomposer: Any = None,
        companion: Any = None,
        notifier: Any = None,
        permission_manager: Any = None,
        llm: Any = None,
        state_dir: str = "~/.mycelium/goals",
        max_retries: int = 2,
        loop_interval: float = 30.0,
        escalation_callback: Optional[Callable] = None,
    ) -> None:
        self._td = task_decomposer
        self._companion = companion
        self._notifier = notifier
        self._pm = permission_manager
        self._llm = llm
        self._state_dir = Path(state_dir).expanduser()
        self.max_retries = max_retries
        self.loop_interval = loop_interval
        self._escalation = escalation_callback

        self._goals: Dict[str, GoalRecord] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._handlers: Dict[str, Callable] = {}   # instance-level, not shared

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_goal(
        self,
        description: str,
        metadata: Optional[Dict] = None,
        run_immediately: bool = False,
    ) -> str:
        """Add a goal to the queue and return its ID.

        Parameters
        ----------
        description : str
            Natural-language goal (e.g. "Read sales.csv and summarise trends").
        metadata : dict, optional
            Arbitrary extra info stored with the goal.
        run_immediately : bool
            When True, execute the goal synchronously before returning.

        Returns
        -------
        str
            The goal ID.
        """
        goal = GoalRecord(
            id=str(uuid.uuid4())[:8],
            description=description,
            status=GoalStatus.PENDING,
            created_at=time.time(),
            metadata=metadata or {},
        )
        with self._lock:
            self._goals[goal.id] = goal
        self._save_state()
        _logger.info("GoalEngine: added goal %s — %r", goal.id, description)
        if run_immediately:
            self.run_now(goal.id)
        return goal.id

    def run_now(self, goal_id: str) -> GoalRecord:
        """Execute a goal synchronously and return the updated record.

        Parameters
        ----------
        goal_id : str
            ID returned by :meth:`add_goal`.

        Returns
        -------
        GoalRecord
        """
        with self._lock:
            goal = self._goals.get(goal_id)
        if goal is None:
            raise KeyError(f"No goal with id={goal_id!r}")
        return self._execute_goal(goal)

    def cancel_goal(self, goal_id: str) -> bool:
        """Cancel a pending or active goal. Returns True if cancelled."""
        with self._lock:
            goal = self._goals.get(goal_id)
        if goal is None:
            return False
        if goal.status in (GoalStatus.COMPLETED, GoalStatus.CANCELLED):
            return False
        goal.status = GoalStatus.CANCELLED
        goal.completed_at = time.time()
        self._save_state()
        _logger.info("GoalEngine: goal %s cancelled", goal_id)
        return True

    def goals(self, status: Optional[GoalStatus] = None) -> List[GoalRecord]:
        """Return goals, optionally filtered by status."""
        with self._lock:
            all_goals = list(self._goals.values())
        if status is not None:
            all_goals = [g for g in all_goals if g.status == status]
        return sorted(all_goals, key=lambda g: g.created_at)

    def get(self, goal_id: str) -> Optional[GoalRecord]:
        """Retrieve a single goal by ID."""
        with self._lock:
            return self._goals.get(goal_id)

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def start_loop(self) -> None:
        """Start the background goal-processing loop."""
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="GoalEngineLoop")
        self._thread.start()
        self._running = True
        _logger.info("GoalEngine: background loop started (interval=%.0fs)", self.loop_interval)

    def stop_loop(self) -> None:
        """Stop the background loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._running = False
        _logger.info("GoalEngine: background loop stopped")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            pending = self.goals(GoalStatus.PENDING)
            for goal in pending:
                if self._stop_event.is_set():
                    break
                try:
                    self._execute_goal(goal)
                except Exception as exc:
                    _logger.warning("GoalEngine: loop error on %s: %s", goal.id, exc)
            self._stop_event.wait(self.loop_interval)

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _execute_goal(self, goal: GoalRecord) -> GoalRecord:
        """Decompose and execute one goal. Mutates and persists goal state."""
        goal.status = GoalStatus.ACTIVE
        goal.started_at = time.time()
        self._save_state()
        _logger.info("GoalEngine: executing goal %s — %r", goal.id, goal.description)

        try:
            subtasks = self._decompose(goal.description)
            goal.steps = [{"index": i, "description": t.description, "status": "pending"}
                          for i, t in enumerate(subtasks)]
            self._save_state()

            any_failed = False
            for i, task in enumerate(subtasks):
                step_start = time.time()
                try:
                    output = self._dispatch_step(task.description, goal)
                    goal.steps[i].update({"status": "ok", "output": output[:500]})
                    _logger.info("GoalEngine: step %d ok — %r", i, task.description[:60])
                except Exception as exc:
                    error_msg = str(exc)
                    goal.steps[i].update({"status": "error", "error": error_msg})
                    _logger.warning("GoalEngine: step %d failed — %s", i, error_msg)
                    any_failed = True
                finally:
                    goal.steps[i]["elapsed"] = round(time.time() - step_start, 2)
                self._save_state()

            if any_failed:
                goal.retries += 1
                if goal.retries > self.max_retries:
                    goal.status = GoalStatus.BLOCKED
                    goal.error = "Max retries exceeded; some steps failed"
                    _logger.warning("GoalEngine: goal %s BLOCKED after %d retries", goal.id, goal.retries)
                    self._notify(f"Goal blocked: {goal.description[:60]}", title="Myco — Blocked")
                    if self._escalation:
                        try:
                            self._escalation(goal)
                        except Exception:
                            pass
                else:
                    goal.status = GoalStatus.PENDING  # re-queue for retry
                    _logger.info("GoalEngine: goal %s queued for retry (%d/%d)", goal.id, goal.retries, self.max_retries)
            else:
                goal.status = GoalStatus.COMPLETED
                goal.completed_at = time.time()
                _logger.info("GoalEngine: goal %s COMPLETED in %.1fs", goal.id, goal.elapsed)
                self._notify(
                    f"Goal complete: {goal.description[:60]}",
                    title="✓ Myco",
                )

        except Exception as exc:
            goal.status = GoalStatus.FAILED
            goal.error = str(exc)
            goal.completed_at = time.time()
            _logger.error("GoalEngine: goal %s FAILED: %s", goal.id, exc)
            self._notify(f"Goal failed: {goal.description[:60]}", title="⚠ Myco")

        self._save_state()
        return goal

    # ------------------------------------------------------------------
    # Step dispatch — keyword-based tool routing
    # ------------------------------------------------------------------

    def _dispatch_step(self, description: str, goal: GoalRecord) -> str:
        """Route a subtask description to the appropriate tool handler.

        Routing priority:
        1. Custom handlers registered via :meth:`register_handler`
        2. Keyword-based built-in routing (file, train, predict, browse, etc.)
        3. LLM general reasoning
        4. No-op log
        """
        low = description.lower()

        # 1. Custom handlers
        for kw, fn in self._handlers.items():
            if kw in low:
                return str(fn(description, goal))

        # 2. Built-in keyword routing
        if any(k in low for k in ("read", "open", "load", "ingest", "parse", "process")):
            return self._step_read(description, goal)

        if any(k in low for k in ("train", "fit", "learn from", "learn on")):
            return self._step_train(description, goal)

        if any(k in low for k in ("predict", "forecast", "estimate", "score")):
            return self._step_predict(description, goal)

        if any(k in low for k in ("browse", "fetch", "navigate", "http", "https", "url", "web")):
            return self._step_browse(description, goal)

        if any(k in low for k in ("screenshot", "capture screen", "take screen")):
            return self._step_screenshot(description, goal)

        if any(k in low for k in ("notify", "alert", "send notification", "remind")):
            return self._step_notify(description, goal)

        if any(k in low for k in ("save", "persist", "store", "write")):
            return self._step_save(description, goal)

        if any(k in low for k in ("search", "find", "look up", "query")):
            return self._step_search(description, goal)

        # 3. LLM general reasoning
        if self._llm is not None and getattr(self._llm, "available", False):
            return self._step_llm(description, goal)

        # 4. Fallback
        _logger.info("GoalEngine: no handler for step %r — logged only", description)
        return f"Noted: {description}"

    def register_handler(self, keyword: str, fn: Callable) -> None:
        """Register a custom step handler.

        Parameters
        ----------
        keyword : str
            If this word appears in a step description, *fn* is called.
        fn : callable(description, goal) -> str
        """
        self._handlers[keyword.lower()] = fn

    # ------------------------------------------------------------------
    # Built-in step handlers
    # ------------------------------------------------------------------

    def _step_read(self, desc: str, goal: GoalRecord) -> str:
        import re
        paths = re.findall(r"[\w/~\.\-]+\.(?:csv|txt|pdf|json|xlsx|xls|md|parquet)", desc)
        urls = re.findall(r"https?://\S+", desc)
        companion = self._companion

        if urls and companion and hasattr(companion, "browse"):
            return companion.browse(urls[0])

        if paths and companion and hasattr(companion, "doc_processor"):
            result = companion.doc_processor.process(paths[0])
            if result.success:
                return result.text[:600] or f"Read {paths[0]}"
            return f"Could not read {paths[0]}: {result.error}"

        # Try reading from goal metadata
        meta_path = goal.metadata.get("path")
        if meta_path and companion and hasattr(companion, "doc_processor"):
            result = companion.doc_processor.process(meta_path)
            if result.success:
                return result.text[:600] or f"Read {meta_path}"

        return f"Step completed: {desc}"

    def _step_train(self, desc: str, goal: GoalRecord) -> str:
        import re
        paths = re.findall(r"[\w/~\.\-]+\.(?:csv|tsv|xlsx|xls|parquet)", desc)
        if not paths:
            paths = [goal.metadata.get("path", "")]
        path = paths[0] if paths else ""

        companion = self._companion
        if not path:
            return "No training file specified in this step."

        if self._pm and not self._pm.check("train"):
            return "Permission denied for model training."

        if companion and hasattr(companion, "model_manager"):
            result = companion.model_manager.train_from_csv(path)
            if result.success:
                companion.model_manager.save()
            return result.message
        return f"Training step noted: {desc}"

    def _step_predict(self, desc: str, goal: GoalRecord) -> str:
        import re
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", desc)]
        companion = self._companion
        if companion and hasattr(companion, "model_manager"):
            if not companion.model_manager.fitted:
                return "No trained model available. Train a model first."
            if nums:
                result = companion.model_manager.predict(nums)
                if result.error:
                    return f"Prediction error: {result.error}"
                return f"Prediction: {result.value:.4g} (confidence {result.confidence:.0%})"
            return "No feature values found in step description."
        return f"Prediction step noted: {desc}"

    def _step_browse(self, desc: str, goal: GoalRecord) -> str:
        import re
        urls = re.findall(r"https?://\S+", desc)
        companion = self._companion
        if not urls:
            return f"No URL found in step: {desc}"
        if self._pm and not self._pm.check("browser.navigate"):
            return "Permission denied for browser.navigate."
        if companion and hasattr(companion, "browse"):
            return companion.browse(urls[0])
        if companion and hasattr(companion, "browser_agent"):
            return companion.browser_agent.fetch_text(urls[0])[:600]
        return f"Browse step noted: {urls[0]}"

    def _step_screenshot(self, desc: str, goal: GoalRecord) -> str:
        if self._pm and not self._pm.check("screen.screenshot"):
            return "Permission denied for screen.screenshot."
        companion = self._companion
        if companion and hasattr(companion, "take_screenshot"):
            return companion.take_screenshot()
        return "Screenshot step noted."

    def _step_notify(self, desc: str, goal: GoalRecord) -> str:
        if self._notifier:
            self._notifier.send("Myco Goal Update", desc)
        return f"Notification sent: {desc}"

    def _step_save(self, desc: str, goal: GoalRecord) -> str:
        if self._pm and not self._pm.check("file.write"):
            return "Permission denied for file.write."
        companion = self._companion
        if companion and hasattr(companion, "_handle_save"):
            return companion._handle_save()
        return f"Save step completed: {desc}"

    def _step_search(self, desc: str, goal: GoalRecord) -> str:
        companion = self._companion
        if companion and hasattr(companion, "vector_memory") and companion.vector_memory:
            results = companion.vector_memory.search(desc, k=3)
            if results:
                snippets = "\n".join(f"- {r.text}" for r in results[:3])
                return f"Memory search results:\n{snippets}"
        return f"Search step: {desc} — no memory results found."

    def _step_llm(self, desc: str, goal: GoalRecord) -> str:
        """Delegate an open-ended step to the LLM."""
        try:
            context = f"Goal: {goal.description}\nCurrent step: {desc}"
            result = self._llm.complete(
                f"Execute this step as part of completing the goal below.\n\n"
                f"Goal: {goal.description}\n"
                f"Step: {desc}\n\n"
                f"Provide a concise result or action taken (1-3 sentences).",
                system="You are Myco, an autonomous AI assistant. Execute the given step and report the outcome concisely.",
            )
            if result.available and result.text:
                return result.text.strip()
        except Exception as exc:
            _logger.warning("GoalEngine LLM step failed: %s", exc)
        return f"Step: {desc}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _decompose(self, description: str):
        """Return a list of SubTask objects for *description*."""
        if self._td is not None:
            return self._td.decompose(description)
        from physml.task_decomposer import TaskDecomposer
        return TaskDecomposer().decompose(description)

    def _notify(self, message: str, title: str = "Myco") -> None:
        if self._notifier is not None:
            try:
                self._notifier.send(title, message)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            path = self._state_dir / "goals.json"
            with self._lock:
                data = {gid: g.to_dict() for gid, g in self._goals.items()}
            path.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            _logger.debug("GoalEngine: state save failed: %s", exc)

    def _load_state(self) -> None:
        path = self._state_dir / "goals.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            with self._lock:
                for gid, d in data.items():
                    self._goals[gid] = GoalRecord.from_dict(d)
            # Reset active→pending on reload (may have died mid-execution)
            with self._lock:
                for g in self._goals.values():
                    if g.status == GoalStatus.ACTIVE:
                        g.status = GoalStatus.PENDING
            _logger.info("GoalEngine: loaded %d goals from disk", len(self._goals))
        except Exception as exc:
            _logger.debug("GoalEngine: state load failed: %s", exc)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Summary of all goal counts by status."""
        with self._lock:
            all_goals = list(self._goals.values())
        counts: Dict[str, int] = {}
        for s in GoalStatus:
            counts[s.value] = sum(1 for g in all_goals if g.status == s)
        return {
            "total": len(all_goals),
            "running": self._running,
            "loop_interval": self.loop_interval,
            **counts,
        }

    def __repr__(self) -> str:  # pragma: no cover
        s = self.status()
        return (
            f"GoalEngine(total={s['total']}, pending={s['pending']}, "
            f"completed={s['completed']}, running={self._running})"
        )
