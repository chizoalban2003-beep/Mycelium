"""Stage 138 — ScheduledGoals: recurring goal scheduler.

Turns the one-shot GoalEngine into a time-aware autonomous agent.
Users can register goals with a schedule — "every hour", "every morning",
"every Monday", or a raw interval in seconds — and ScheduledGoals will
spawn new GoalEngine runs at the right times, automatically.

This is the difference between "do this once" and "keep doing this for me".

Usage
-----
::

    from physml.scheduled_goals import ScheduledGoals, Schedule

    scheduler = ScheduledGoals(goal_engine=engine, notifier=notifier)

    # Run every morning at 08:00
    scheduler.add("Check my sales data and alert me to anomalies",
                  schedule=Schedule.daily(hour=8))

    # Run every 30 minutes
    scheduler.add("Monitor system health", schedule=Schedule.interval(1800))

    # Run every Monday at 09:00
    scheduler.add("Generate weekly report", schedule=Schedule.weekly(weekday=0, hour=9))

    scheduler.start()   # background thread; fires goals when due
    # ...
    scheduler.stop()

Schedule strings (human-readable shorthand)
-------------------------------------------
- ``"hourly"``              — every 3600 s
- ``"daily"``               — every 86400 s (from first run)
- ``"weekly"``              — every 604800 s
- ``"every N minutes"``     — every N*60 s
- ``"every N hours"``       — every N*3600 s
- ``"every N seconds"``     — every N s
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


@dataclass
class Schedule:
    """Defines when a recurring goal should fire.

    Parameters
    ----------
    interval_seconds : float
        How often to fire (in seconds).
    hour : int or None
        If set, only fire when ``datetime.now().hour == hour``.
        Coarse daily alignment — useful for "every morning at 8".
    weekday : int or None
        0=Monday … 6=Sunday.  Only fire on this day of the week.
        Combined with *hour* for weekly schedules.
    label : str
        Human-readable description of the schedule.
    """

    interval_seconds: float
    hour: Optional[int] = None
    weekday: Optional[int] = None
    label: str = ""

    # ------------------------------------------------------------------
    @classmethod
    def interval(cls, seconds: float, label: str = "") -> "Schedule":
        """Fire every *seconds* seconds."""
        return cls(interval_seconds=seconds, label=label or f"every {seconds:.0f}s")

    @classmethod
    def hourly(cls) -> "Schedule":
        return cls(interval_seconds=3600, label="hourly")

    @classmethod
    def daily(cls, hour: int = 8) -> "Schedule":
        return cls(interval_seconds=86400, hour=hour, label=f"daily at {hour:02d}:00")

    @classmethod
    def weekly(cls, weekday: int = 0, hour: int = 9) -> "Schedule":
        return cls(
            interval_seconds=604800,
            hour=hour,
            weekday=weekday,
            label=f"weekly on day {weekday} at {hour:02d}:00",
        )

    @classmethod
    def from_string(cls, s: str) -> "Schedule":
        """Parse a human-readable schedule string.

        Recognised forms::

            "hourly", "daily", "weekly"
            "every N minutes", "every N hours", "every N seconds"
            "every morning", "every evening"
        """
        s = s.strip().lower()
        if s == "hourly":
            return cls.hourly()
        if s in ("daily", "every day", "every morning"):
            return cls.daily(hour=8)
        if s == "every evening":
            return cls.daily(hour=18)
        if s == "weekly":
            return cls.weekly()

        m = re.match(r"every\s+(\d+(?:\.\d+)?)\s+(second|minute|hour)s?", s)
        if m:
            n, unit = float(m.group(1)), m.group(2)
            mult = {"second": 1, "minute": 60, "hour": 3600}[unit]
            return cls.interval(n * mult, label=s)

        raise ValueError(f"Cannot parse schedule string: {s!r}")

    def to_dict(self) -> dict:
        return {
            "interval_seconds": self.interval_seconds,
            "hour": self.hour,
            "weekday": self.weekday,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Schedule":
        return cls(
            interval_seconds=d.get("interval_seconds", 3600),
            hour=d.get("hour"),
            weekday=d.get("weekday"),
            label=d.get("label", ""),
        )


# ---------------------------------------------------------------------------
# ScheduledGoal record
# ---------------------------------------------------------------------------


@dataclass
class ScheduledGoal:
    """One registered recurring goal."""

    id: str
    description: str
    schedule: Schedule
    enabled: bool = True
    last_run_at: Optional[float] = None
    next_run_at: float = field(default_factory=time.time)
    run_count: int = 0
    last_goal_id: Optional[str] = None

    def is_due(self) -> bool:
        """Return True if this goal should fire right now."""
        if not self.enabled:
            return False
        now = time.time()
        if now < self.next_run_at:
            return False
        # Optional hour / weekday filters
        if self.schedule.hour is not None or self.schedule.weekday is not None:
            import datetime
            dt = datetime.datetime.now()
            if self.schedule.weekday is not None and dt.weekday() != self.schedule.weekday:
                # Not the right day — push next_run_at forward by 1 hour to re-check later
                self.next_run_at = now + 3600
                return False
            if self.schedule.hour is not None and dt.hour != self.schedule.hour:
                self.next_run_at = now + 3600
                return False
        return True

    def mark_ran(self, goal_id: str) -> None:
        self.last_run_at = time.time()
        self.next_run_at = time.time() + self.schedule.interval_seconds
        self.run_count += 1
        self.last_goal_id = goal_id

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "schedule": self.schedule.to_dict(),
            "enabled": self.enabled,
            "last_run_at": self.last_run_at,
            "next_run_at": self.next_run_at,
            "run_count": self.run_count,
            "last_goal_id": self.last_goal_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScheduledGoal":
        return cls(
            id=d["id"],
            description=d["description"],
            schedule=Schedule.from_dict(d.get("schedule", {})),
            enabled=d.get("enabled", True),
            last_run_at=d.get("last_run_at"),
            next_run_at=d.get("next_run_at", time.time()),
            run_count=d.get("run_count", 0),
            last_goal_id=d.get("last_goal_id"),
        )


# ---------------------------------------------------------------------------
# ScheduledGoals
# ---------------------------------------------------------------------------


class ScheduledGoals:
    """Background scheduler that spawns GoalEngine runs on a timed schedule.

    Parameters
    ----------
    goal_engine : GoalEngine or None
        Where to submit goals when they fire.
    notifier : Notifier or None
        Desktop notification when a scheduled goal fires.
    state_dir : str
        Where to persist the schedule state (JSON).
    tick_interval : float
        How often (in seconds) the scheduler checks for due goals.
    """

    def __init__(
        self,
        goal_engine: Any = None,
        notifier: Any = None,
        state_dir: str = "~/.mycelium/schedule",
        tick_interval: float = 60.0,
    ) -> None:
        self._engine = goal_engine
        self._notifier = notifier
        self._state_dir = Path(state_dir).expanduser()
        self.tick_interval = tick_interval

        self._schedules: Dict[str, ScheduledGoal] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        self._load()

    # ------------------------------------------------------------------
    # Schedule management
    # ------------------------------------------------------------------

    def add(
        self,
        description: str,
        schedule: "Schedule | str",
        goal_id: Optional[str] = None,
        enabled: bool = True,
    ) -> str:
        """Register a recurring goal.

        Parameters
        ----------
        description : str
            Natural-language goal text.
        schedule : Schedule or str
            When to run — pass a :class:`Schedule` object or a string like
            ``"every 30 minutes"``, ``"daily"``, ``"hourly"``.
        goal_id : str, optional
            Custom ID; auto-generated if omitted.

        Returns
        -------
        str
            The scheduled-goal ID.
        """
        if isinstance(schedule, str):
            schedule = Schedule.from_string(schedule)

        import uuid
        sid = goal_id or str(uuid.uuid4())[:8]
        sg = ScheduledGoal(
            id=sid,
            description=description,
            schedule=schedule,
            enabled=enabled,
        )
        with self._lock:
            self._schedules[sid] = sg
        self._save()
        _logger.info(
            "ScheduledGoals: registered %r (%s)", description[:50], schedule.label
        )
        return sid

    def remove(self, sid: str) -> bool:
        """Remove a scheduled goal. Returns True if found."""
        with self._lock:
            found = sid in self._schedules
            if found:
                del self._schedules[sid]
        if found:
            self._save()
        return found

    def enable(self, sid: str, enabled: bool = True) -> None:
        """Enable or disable a scheduled goal without removing it."""
        with self._lock:
            if sid in self._schedules:
                self._schedules[sid].enabled = enabled
        self._save()

    def list(self) -> List[ScheduledGoal]:
        """Return all registered schedules."""
        with self._lock:
            return list(self._schedules.values())

    def get(self, sid: str) -> Optional[ScheduledGoal]:
        with self._lock:
            return self._schedules.get(sid)

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the scheduler background loop."""
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="ScheduledGoalsLoop"
        )
        self._thread.start()
        self._running = True
        _logger.info(
            "ScheduledGoals: started (tick=%.0fs, %d schedules)",
            self.tick_interval, len(self._schedules),
        )

    def stop(self) -> None:
        """Stop the scheduler loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._running = False
        _logger.info("ScheduledGoals: stopped")

    def tick(self) -> List[str]:
        """Check for due goals and fire them. Returns list of fired goal IDs.

        Call this manually if you prefer synchronous control.
        """
        fired: List[str] = []
        with self._lock:
            due = [sg for sg in self._schedules.values() if sg.is_due()]

        for sg in due:
            try:
                goal_id = self._fire(sg)
                fired.append(goal_id)
            except Exception as exc:
                _logger.warning("ScheduledGoals: error firing %s: %s", sg.id, exc)

        return fired

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception as exc:
                _logger.warning("ScheduledGoals: loop error: %s", exc)
            self._stop_event.wait(self.tick_interval)

    def _fire(self, sg: ScheduledGoal) -> str:
        """Submit the scheduled goal to GoalEngine."""
        _logger.info(
            "ScheduledGoals: firing %r (run #%d)", sg.description[:50], sg.run_count + 1
        )
        if self._notifier:
            try:
                self._notifier.send(
                    "Myco Scheduled Task",
                    f"Starting: {sg.description[:60]}",
                )
            except Exception:
                pass

        if self._engine is not None:
            goal_id = self._engine.add_goal(
                sg.description,
                metadata={"scheduled": True, "schedule_id": sg.id},
            )
        else:
            import uuid
            goal_id = str(uuid.uuid4())[:8]
            _logger.info("ScheduledGoals: no engine — goal %s logged only", goal_id)

        sg.mark_ran(goal_id)
        self._save()
        return goal_id

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            path = self._state_dir / "schedules.json"
            with self._lock:
                data = {sid: sg.to_dict() for sid, sg in self._schedules.items()}
            path.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            _logger.debug("ScheduledGoals: save failed: %s", exc)

    def _load(self) -> None:
        path = self._state_dir / "schedules.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            with self._lock:
                for sid, d in data.items():
                    self._schedules[sid] = ScheduledGoal.from_dict(d)
            _logger.info("ScheduledGoals: loaded %d schedules", len(self._schedules))
        except Exception as exc:
            _logger.debug("ScheduledGoals: load failed: %s", exc)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            sched = list(self._schedules.values())
        return {
            "total": len(sched),
            "enabled": sum(1 for s in sched if s.enabled),
            "running": self._running,
            "tick_interval": self.tick_interval,
            "schedules": [
                {
                    "id": s.id,
                    "description": s.description[:50],
                    "schedule": s.schedule.label,
                    "enabled": s.enabled,
                    "run_count": s.run_count,
                    "next_run_at": s.next_run_at,
                }
                for s in sched
            ],
        }

    def __repr__(self) -> str:  # pragma: no cover
        s = self.status()
        return f"ScheduledGoals(total={s['total']}, running={self._running})"
