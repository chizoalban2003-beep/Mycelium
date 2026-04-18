"""Stage 109 — ScheduledTaskRunner: background task scheduler.

Register tasks with interval-based schedules (interval_seconds, or named
presets: daily/hourly/weekly).  Uses :class:`threading.Timer` for lightweight
scheduling without external dependencies.  Tasks are plain callables.
Schedule is persisted to JSON.

Usage
-----
::

    from physml.scheduler_tasks import ScheduledTaskRunner

    runner = ScheduledTaskRunner()
    runner.schedule("daily_report", fn=lambda: print("report"), interval_seconds=86400)
    runner.start()          # non-blocking
    runner.list_tasks()     # → [{"name": "daily_report", ...}]
    runner.cancel("daily_report")
    runner.stop()
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)

# Named interval presets
_PRESETS: Dict[str, int] = {
    "hourly": 3600,
    "daily": 86400,
    "weekly": 604800,
}


@dataclass
class ScheduledTask:
    """Metadata for a registered task.

    Attributes
    ----------
    name : str
        Unique task identifier.
    interval_seconds : int
        Seconds between executions.
    last_run : float or None
        Unix time of last execution.
    run_count : int
        Total number of times executed.
    enabled : bool
        Whether the task is active.
    """

    name: str
    interval_seconds: int
    last_run: Optional[float] = None
    run_count: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScheduledTaskRunner:
    """Background task scheduler using :class:`threading.Timer`.

    Parameters
    ----------
    persist_path : str or None
        JSON file to persist task metadata.  When ``None``, no persistence.
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self._tasks: Dict[str, ScheduledTask] = {}
        self._fns: Dict[str, Callable[[], Any]] = {}
        self._timers: Dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self._running = False
        self.persist_path = persist_path

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def schedule(
        self,
        name: str,
        fn: Callable[[], Any],
        interval_seconds: Optional[int] = None,
        preset: Optional[str] = None,
    ) -> "ScheduledTaskRunner":
        """Register a task.

        Parameters
        ----------
        name : str
            Unique task name.
        fn : callable
            Zero-argument callable to invoke.
        interval_seconds : int, optional
            Seconds between runs.  Required unless *preset* is given.
        preset : str, optional
            One of ``"hourly"``, ``"daily"``, ``"weekly"``.

        Returns
        -------
        self
        """
        if preset is not None:
            if preset not in _PRESETS:
                raise ValueError(f"Unknown preset {preset!r}; choose from {list(_PRESETS)}")
            interval_seconds = _PRESETS[preset]
        if interval_seconds is None:
            raise ValueError("Either interval_seconds or preset must be provided")

        with self._lock:
            task = ScheduledTask(name=name, interval_seconds=int(interval_seconds))
            self._tasks[name] = task
            self._fns[name] = fn

        if self._running:
            self._arm(name)

        _logger.info("ScheduledTaskRunner: registered %r every %ds", name, interval_seconds)
        return self

    def start(self) -> None:
        """Start the scheduler (non-blocking)."""
        if self._running:
            return
        self._running = True
        with self._lock:
            for name in self._tasks:
                self._arm(name)
        _logger.info("ScheduledTaskRunner: started with %d tasks", len(self._tasks))

    def stop(self) -> None:
        """Stop the scheduler and cancel all pending timers."""
        self._running = False
        with self._lock:
            for name, timer in list(self._timers.items()):
                timer.cancel()
            self._timers.clear()
        _logger.info("ScheduledTaskRunner: stopped")

    def cancel(self, name: str) -> bool:
        """Cancel a scheduled task.

        Parameters
        ----------
        name : str

        Returns
        -------
        bool
            ``True`` if the task existed, ``False`` otherwise.
        """
        with self._lock:
            if name not in self._tasks:
                return False
            self._tasks[name].enabled = False
            if name in self._timers:
                self._timers[name].cancel()
                del self._timers[name]
            del self._tasks[name]
            if name in self._fns:
                del self._fns[name]
        _logger.info("ScheduledTaskRunner: cancelled %r", name)
        return True

    def list_tasks(self) -> List[Dict[str, Any]]:
        """Return metadata for all registered tasks.

        Returns
        -------
        list of dict
        """
        with self._lock:
            return [asdict(t) for t in self._tasks.values()]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Persist task metadata (not callables) to JSON.

        Parameters
        ----------
        path : str, optional
            Overrides ``persist_path`` if given.
        """
        p = Path(path or self.persist_path or "scheduler_tasks.json").expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = [asdict(t) for t in self._tasks.values()]
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_meta(self, path: Optional[str] = None) -> None:
        """Load task metadata from JSON (callables must be re-registered).

        Parameters
        ----------
        path : str, optional
        """
        p = Path(path or self.persist_path or "scheduler_tasks.json").expanduser()
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        with self._lock:
            for item in data:
                name = item["name"]
                if name not in self._tasks:
                    self._tasks[name] = ScheduledTask(**item)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _arm(self, name: str) -> None:
        """Schedule the next fire for *name*."""
        task = self._tasks.get(name)
        if task is None or not task.enabled:
            return
        timer = threading.Timer(task.interval_seconds, self._fire, args=(name,))
        timer.daemon = True
        timer.start()
        self._timers[name] = timer

    def _fire(self, name: str) -> None:
        """Execute task *name* and re-arm."""
        with self._lock:
            task = self._tasks.get(name)
            fn = self._fns.get(name)
        if task is None or not task.enabled or fn is None:
            return
        try:
            fn()
            task.last_run = time.time()
            task.run_count += 1
        except Exception as e:
            _logger.warning("ScheduledTaskRunner: task %r raised: %s", name, e)
        if self._running:
            self._arm(name)

    def __repr__(self) -> str:
        return f"ScheduledTaskRunner(n_tasks={len(self._tasks)}, running={self._running})"
