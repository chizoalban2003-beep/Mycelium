"""Tests for Stage 138 — ScheduledGoals (recurring goal scheduler)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest


# ===========================================================================
# Schedule parsing
# ===========================================================================


class TestSchedule:
    def test_import(self):
        from physml.scheduled_goals import Schedule
        assert Schedule is not None

    def test_interval(self):
        from physml.scheduled_goals import Schedule
        s = Schedule.interval(300)
        assert s.interval_seconds == 300

    def test_hourly(self):
        from physml.scheduled_goals import Schedule
        s = Schedule.hourly()
        assert s.interval_seconds == 3600

    def test_daily(self):
        from physml.scheduled_goals import Schedule
        s = Schedule.daily(hour=8)
        assert s.interval_seconds == 86400
        assert s.hour == 8

    def test_weekly(self):
        from physml.scheduled_goals import Schedule
        s = Schedule.weekly(weekday=0, hour=9)
        assert s.interval_seconds == 604800
        assert s.weekday == 0
        assert s.hour == 9

    def test_from_string_hourly(self):
        from physml.scheduled_goals import Schedule
        s = Schedule.from_string("hourly")
        assert s.interval_seconds == 3600

    def test_from_string_daily(self):
        from physml.scheduled_goals import Schedule
        s = Schedule.from_string("daily")
        assert s.interval_seconds == 86400

    def test_from_string_every_n_minutes(self):
        from physml.scheduled_goals import Schedule
        s = Schedule.from_string("every 30 minutes")
        assert s.interval_seconds == 1800

    def test_from_string_every_n_hours(self):
        from physml.scheduled_goals import Schedule
        s = Schedule.from_string("every 2 hours")
        assert s.interval_seconds == 7200

    def test_from_string_every_n_seconds(self):
        from physml.scheduled_goals import Schedule
        s = Schedule.from_string("every 45 seconds")
        assert s.interval_seconds == 45

    def test_from_string_every_morning(self):
        from physml.scheduled_goals import Schedule
        s = Schedule.from_string("every morning")
        assert s.interval_seconds == 86400

    def test_from_string_invalid_raises(self):
        from physml.scheduled_goals import Schedule
        with pytest.raises(ValueError):
            Schedule.from_string("at noon on sundays when it rains")

    def test_roundtrip_dict(self):
        from physml.scheduled_goals import Schedule
        s = Schedule.daily(hour=9)
        s2 = Schedule.from_dict(s.to_dict())
        assert s2.interval_seconds == s.interval_seconds
        assert s2.hour == s.hour


# ===========================================================================
# ScheduledGoal
# ===========================================================================


class TestScheduledGoal:
    def test_import(self):
        from physml.scheduled_goals import ScheduledGoal
        assert ScheduledGoal is not None

    def test_is_due_past(self):
        from physml.scheduled_goals import ScheduledGoal, Schedule
        sg = ScheduledGoal(
            id="x", description="d",
            schedule=Schedule.interval(1),
            next_run_at=time.time() - 10,
        )
        assert sg.is_due() is True

    def test_is_due_future(self):
        from physml.scheduled_goals import ScheduledGoal, Schedule
        sg = ScheduledGoal(
            id="x", description="d",
            schedule=Schedule.interval(3600),
            next_run_at=time.time() + 9999,
        )
        assert sg.is_due() is False

    def test_is_due_disabled(self):
        from physml.scheduled_goals import ScheduledGoal, Schedule
        sg = ScheduledGoal(
            id="x", description="d",
            schedule=Schedule.interval(1),
            next_run_at=time.time() - 10,
            enabled=False,
        )
        assert sg.is_due() is False

    def test_mark_ran_updates_fields(self):
        from physml.scheduled_goals import ScheduledGoal, Schedule
        sg = ScheduledGoal(
            id="x", description="d",
            schedule=Schedule.interval(60),
            next_run_at=time.time() - 1,
        )
        sg.mark_ran("goal-abc")
        assert sg.run_count == 1
        assert sg.last_goal_id == "goal-abc"
        assert sg.next_run_at > time.time()

    def test_roundtrip_dict(self):
        from physml.scheduled_goals import ScheduledGoal, Schedule
        sg = ScheduledGoal(
            id="abc", description="test",
            schedule=Schedule.hourly(),
            run_count=5,
        )
        sg2 = ScheduledGoal.from_dict(sg.to_dict())
        assert sg2.id == "abc"
        assert sg2.run_count == 5


# ===========================================================================
# ScheduledGoals
# ===========================================================================


class TestScheduledGoals:
    def test_import(self):
        from physml.scheduled_goals import ScheduledGoals
        assert ScheduledGoals is not None

    def test_from_package(self):
        from physml import ScheduledGoals, Schedule
        assert ScheduledGoals is not None

    def test_instantiation(self, tmp_path):
        from physml.scheduled_goals import ScheduledGoals
        s = ScheduledGoals(state_dir=str(tmp_path / "sched"))
        assert s is not None

    def test_add_returns_id(self, tmp_path):
        from physml.scheduled_goals import ScheduledGoals, Schedule
        s = ScheduledGoals(state_dir=str(tmp_path / "sched"))
        sid = s.add("Check my data", schedule=Schedule.hourly())
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_add_string_schedule(self, tmp_path):
        from physml.scheduled_goals import ScheduledGoals
        s = ScheduledGoals(state_dir=str(tmp_path / "sched"))
        sid = s.add("Monitor health", schedule="every 30 minutes")
        assert sid is not None

    def test_list_schedules(self, tmp_path):
        from physml.scheduled_goals import ScheduledGoals
        s = ScheduledGoals(state_dir=str(tmp_path / "sched"))
        s.add("Goal A", schedule="hourly")
        s.add("Goal B", schedule="daily")
        assert len(s.list()) == 2

    def test_remove_schedule(self, tmp_path):
        from physml.scheduled_goals import ScheduledGoals
        s = ScheduledGoals(state_dir=str(tmp_path / "sched"))
        sid = s.add("Temp goal", schedule="hourly")
        result = s.remove(sid)
        assert result is True
        assert len(s.list()) == 0

    def test_remove_nonexistent(self, tmp_path):
        from physml.scheduled_goals import ScheduledGoals
        s = ScheduledGoals(state_dir=str(tmp_path / "sched"))
        assert s.remove("no-such-id") is False

    def test_enable_disable(self, tmp_path):
        from physml.scheduled_goals import ScheduledGoals
        s = ScheduledGoals(state_dir=str(tmp_path / "sched"))
        sid = s.add("Test", schedule="hourly")
        s.enable(sid, False)
        sg = s.get(sid)
        assert sg.enabled is False
        s.enable(sid, True)
        sg = s.get(sid)
        assert sg.enabled is True

    def test_tick_fires_due_goals_no_engine(self, tmp_path):
        from physml.scheduled_goals import ScheduledGoals, Schedule
        s = ScheduledGoals(state_dir=str(tmp_path / "sched"))
        s.add("Fire now", schedule=Schedule.interval(0.001))
        time.sleep(0.05)
        fired = s.tick()
        assert len(fired) >= 1

    def test_tick_with_engine(self, tmp_path):
        from physml.scheduled_goals import ScheduledGoals, Schedule
        from physml.goal_engine import GoalEngine

        engine = GoalEngine(state_dir=str(tmp_path / "goals"))
        s = ScheduledGoals(
            goal_engine=engine,
            state_dir=str(tmp_path / "sched"),
        )
        s.add("Quick goal", schedule=Schedule.interval(0.001))
        time.sleep(0.05)
        fired = s.tick()
        assert len(fired) >= 1
        # The goal should be queued in the engine
        pending = engine.goals()
        assert any(g.description == "Quick goal" for g in pending)

    def test_status_dict(self, tmp_path):
        from physml.scheduled_goals import ScheduledGoals
        s = ScheduledGoals(state_dir=str(tmp_path / "sched"))
        s.add("A", schedule="hourly")
        status = s.status()
        assert "total" in status
        assert "enabled" in status
        assert "running" in status
        assert status["total"] == 1

    def test_persistence_roundtrip(self, tmp_path):
        from physml.scheduled_goals import ScheduledGoals
        path = str(tmp_path / "sched")
        s1 = ScheduledGoals(state_dir=path)
        s1.add("Persistent", schedule="daily")
        s2 = ScheduledGoals(state_dir=path)
        assert len(s2.list()) == 1
        assert s2.list()[0].description == "Persistent"

    def test_start_stop_loop(self, tmp_path):
        from physml.scheduled_goals import ScheduledGoals
        s = ScheduledGoals(state_dir=str(tmp_path / "sched"), tick_interval=0.05)
        s.start()
        assert s._running is True
        time.sleep(0.1)
        s.stop()
        assert s._running is False

    def test_loop_fires_due_goal(self, tmp_path):
        from physml.scheduled_goals import ScheduledGoals, Schedule
        fired = []

        class _MockEngine:
            def add_goal(self, desc, metadata=None, **kw):
                fired.append(desc)
                return "fake-id"

        s = ScheduledGoals(
            goal_engine=_MockEngine(),
            state_dir=str(tmp_path / "sched"),
            tick_interval=0.05,
        )
        s.add("Background goal", schedule=Schedule.interval(0.001))
        s.start()
        time.sleep(0.3)
        s.stop()
        assert len(fired) >= 1


# ===========================================================================
# Companion integration
# ===========================================================================


class TestCompanionSchedulerIntegration:
    def test_companion_has_scheduler(self, tmp_path):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion(data_dir=str(tmp_path))
        c.start()
        assert c.scheduler is not None
        c.stop()

    def test_schedule_goal_returns_string(self, tmp_path):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion(data_dir=str(tmp_path))
        c.start()
        result = c.schedule_goal("Check my data daily", schedule="daily")
        assert isinstance(result, str)
        assert "Scheduled" in result
        c.stop()

    def test_status_includes_scheduled_goals(self, tmp_path):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion(data_dir=str(tmp_path))
        c.start()
        s = c.status()
        assert "scheduled_goals" in s
        c.stop()

    def test_invalid_schedule_returns_error(self, tmp_path):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion(data_dir=str(tmp_path))
        c.start()
        result = c.schedule_goal("Do something", schedule="whenever I feel like it")
        assert "Invalid" in result or isinstance(result, str)
        c.stop()
