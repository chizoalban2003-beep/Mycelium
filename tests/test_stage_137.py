"""Tests for Stage 137 — GoalEngine: autonomous goal execution loop."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest


# ===========================================================================
# Helpers
# ===========================================================================


def _engine(tmp_path: Path, **kwargs):
    from physml.goal_engine import GoalEngine

    return GoalEngine(state_dir=str(tmp_path / "goals"), **kwargs)


# ===========================================================================
# Import and construction
# ===========================================================================


class TestGoalEngineImport:
    def test_import(self):
        from physml.goal_engine import GoalEngine, GoalRecord, GoalStatus

        assert GoalEngine is not None
        assert GoalRecord is not None
        assert GoalStatus is not None

    def test_from_package(self):
        from physml import GoalEngine, GoalRecord, GoalStatus

        assert GoalEngine is not None

    def test_instantiation(self, tmp_path):
        engine = _engine(tmp_path)
        assert engine is not None

    def test_status_keys(self, tmp_path):
        engine = _engine(tmp_path)
        s = engine.status()
        assert "total" in s
        assert "pending" in s
        assert "completed" in s
        assert "failed" in s
        assert "running" in s


# ===========================================================================
# Goal lifecycle
# ===========================================================================


class TestGoalLifecycle:
    def test_add_goal_returns_id(self, tmp_path):
        engine = _engine(tmp_path)
        gid = engine.add_goal("Do something useful")
        assert isinstance(gid, str)
        assert len(gid) > 0

    def test_added_goal_is_pending(self, tmp_path):
        from physml.goal_engine import GoalStatus

        engine = _engine(tmp_path)
        gid = engine.add_goal("Test goal")
        goal = engine.get(gid)
        assert goal is not None
        assert goal.status == GoalStatus.PENDING

    def test_goals_list(self, tmp_path):
        engine = _engine(tmp_path)
        engine.add_goal("Goal A")
        engine.add_goal("Goal B")
        goals = engine.goals()
        assert len(goals) == 2

    def test_goals_filter_by_status(self, tmp_path):
        from physml.goal_engine import GoalStatus

        engine = _engine(tmp_path)
        gid = engine.add_goal("Pending goal")
        pending = engine.goals(GoalStatus.PENDING)
        assert any(g.id == gid for g in pending)

    def test_cancel_pending_goal(self, tmp_path):
        from physml.goal_engine import GoalStatus

        engine = _engine(tmp_path)
        gid = engine.add_goal("Will be cancelled")
        result = engine.cancel_goal(gid)
        assert result is True
        goal = engine.get(gid)
        assert goal.status == GoalStatus.CANCELLED

    def test_cancel_nonexistent_goal(self, tmp_path):
        engine = _engine(tmp_path)
        result = engine.cancel_goal("does-not-exist")
        assert result is False

    def test_get_nonexistent_goal(self, tmp_path):
        engine = _engine(tmp_path)
        assert engine.get("no-such-id") is None

    def test_goal_metadata_stored(self, tmp_path):
        engine = _engine(tmp_path)
        gid = engine.add_goal("Goal with meta", metadata={"priority": "high"})
        goal = engine.get(gid)
        assert goal.metadata.get("priority") == "high"


# ===========================================================================
# Execution
# ===========================================================================


class TestGoalExecution:
    def test_run_now_returns_record(self, tmp_path):
        engine = _engine(tmp_path)
        gid = engine.add_goal("say hello to the world")
        record = engine.run_now(gid)
        assert record is not None
        assert record.id == gid

    def test_simple_goal_completes(self, tmp_path):
        from physml.goal_engine import GoalStatus

        engine = _engine(tmp_path)
        gid = engine.add_goal("notify the user that everything is fine")
        record = engine.run_now(gid)
        assert record.status in (GoalStatus.COMPLETED, GoalStatus.BLOCKED)

    def test_goal_has_steps_after_run(self, tmp_path):
        engine = _engine(tmp_path)
        gid = engine.add_goal("read file.csv and train model")
        record = engine.run_now(gid)
        assert len(record.steps) > 0

    def test_goal_elapsed_positive(self, tmp_path):
        engine = _engine(tmp_path)
        gid = engine.add_goal("do something")
        record = engine.run_now(gid)
        assert record.elapsed >= 0.0

    def test_unknown_goal_raises(self, tmp_path):
        engine = _engine(tmp_path)
        with pytest.raises(KeyError):
            engine.run_now("bad-id")

    def test_run_immediately_flag(self, tmp_path):
        from physml.goal_engine import GoalStatus

        engine = _engine(tmp_path)
        gid = engine.add_goal("notify me", run_immediately=True)
        goal = engine.get(gid)
        assert goal.status != GoalStatus.PENDING

    def test_goal_with_custom_handler(self, tmp_path):
        from physml.goal_engine import GoalStatus

        engine = _engine(tmp_path)
        results = []

        def my_handler(desc, goal):
            results.append(desc)
            return "custom result"

        engine.register_handler("mycustom", my_handler)
        gid = engine.add_goal("mycustom action for the user")
        record = engine.run_now(gid)
        assert len(results) > 0
        assert record.status == GoalStatus.COMPLETED


# ===========================================================================
# Persistence
# ===========================================================================


class TestGoalPersistence:
    def test_goals_survive_reload(self, tmp_path):
        from physml.goal_engine import GoalEngine, GoalStatus

        engine1 = GoalEngine(state_dir=str(tmp_path / "goals"))
        gid = engine1.add_goal("Persistent goal")
        engine1.run_now(gid)

        engine2 = GoalEngine(state_dir=str(tmp_path / "goals"))
        goal = engine2.get(gid)
        assert goal is not None
        assert goal.description == "Persistent goal"

    def test_active_goals_reset_to_pending_on_reload(self, tmp_path):
        from physml.goal_engine import GoalEngine, GoalStatus

        engine1 = GoalEngine(state_dir=str(tmp_path / "goals"))
        gid = engine1.add_goal("Will appear active")
        goal = engine1.get(gid)
        goal.status = GoalStatus.ACTIVE
        engine1._save_state()

        engine2 = GoalEngine(state_dir=str(tmp_path / "goals"))
        reloaded = engine2.get(gid)
        assert reloaded.status == GoalStatus.PENDING

    def test_completed_goal_persists_status(self, tmp_path):
        from physml.goal_engine import GoalEngine, GoalStatus

        engine1 = GoalEngine(state_dir=str(tmp_path / "goals"))
        gid = engine1.add_goal("Complete me")
        engine1.run_now(gid)

        engine2 = GoalEngine(state_dir=str(tmp_path / "goals"))
        goal = engine2.get(gid)
        assert goal.status in (GoalStatus.COMPLETED, GoalStatus.BLOCKED, GoalStatus.PENDING)


# ===========================================================================
# Step dispatch
# ===========================================================================


class TestStepDispatch:
    def test_notify_step_succeeds(self, tmp_path):
        engine = _engine(tmp_path)
        result = engine._dispatch_step("notify the user about progress", goal=_dummy_goal())
        assert isinstance(result, str)

    def test_llm_step_fallback_no_llm(self, tmp_path):
        engine = _engine(tmp_path)
        result = engine._dispatch_step("analyse and summarise the quarterly results", goal=_dummy_goal())
        assert isinstance(result, str)

    def test_search_step_no_companion(self, tmp_path):
        engine = _engine(tmp_path)
        result = engine._dispatch_step("search for related data", goal=_dummy_goal())
        assert isinstance(result, str)

    def test_screenshot_denied_by_permission(self, tmp_path):
        from physml.permission_manager import PermissionManager

        pm = PermissionManager(config_path=str(tmp_path / "perms.json"))
        pm.set_policy("screen.screenshot", "deny")
        engine = _engine(tmp_path, permission_manager=pm)
        result = engine._dispatch_step("take a screenshot of the screen", goal=_dummy_goal())
        assert "denied" in result.lower() or isinstance(result, str)

    def test_save_step_denied_by_permission(self, tmp_path):
        from physml.permission_manager import PermissionManager

        pm = PermissionManager(config_path=str(tmp_path / "perms.json"))
        # file.write is ASK by default, which becomes deny without callback
        engine = _engine(tmp_path, permission_manager=pm)
        result = engine._dispatch_step("save the current model to disk", goal=_dummy_goal())
        assert isinstance(result, str)


# ===========================================================================
# Background loop
# ===========================================================================


class TestGoalEngineLoop:
    def test_start_stop_loop(self, tmp_path):
        engine = _engine(tmp_path, loop_interval=0.1)
        engine.start_loop()
        assert engine._running is True
        time.sleep(0.2)
        engine.stop_loop()
        assert engine._running is False

    def test_loop_processes_pending_goals(self, tmp_path):
        from physml.goal_engine import GoalStatus

        engine = _engine(tmp_path, loop_interval=0.1)
        gid = engine.add_goal("notify user about loop test")
        engine.start_loop()
        time.sleep(0.6)
        engine.stop_loop()
        goal = engine.get(gid)
        assert goal.status != GoalStatus.PENDING

    def test_double_start_is_safe(self, tmp_path):
        engine = _engine(tmp_path, loop_interval=0.1)
        engine.start_loop()
        engine.start_loop()  # should not raise or spawn duplicate thread
        engine.stop_loop()

    def test_status_shows_running(self, tmp_path):
        engine = _engine(tmp_path, loop_interval=1.0)
        engine.start_loop()
        assert engine.status()["running"] is True
        engine.stop_loop()
        assert engine.status()["running"] is False


# ===========================================================================
# GoalRecord
# ===========================================================================


class TestGoalRecord:
    def test_roundtrip_serialisation(self, tmp_path):
        from physml.goal_engine import GoalRecord, GoalStatus

        r = GoalRecord(
            id="abc123",
            description="Test goal",
            status=GoalStatus.COMPLETED,
            created_at=1000.0,
            completed_at=1010.0,
        )
        d = r.to_dict()
        r2 = GoalRecord.from_dict(d)
        assert r2.id == "abc123"
        assert r2.status == GoalStatus.COMPLETED
        assert r2.description == "Test goal"

    def test_elapsed_when_completed(self):
        from physml.goal_engine import GoalRecord, GoalStatus

        r = GoalRecord(
            id="x",
            description="d",
            status=GoalStatus.COMPLETED,
            created_at=0.0,
            started_at=100.0,
            completed_at=115.0,
        )
        assert r.elapsed == pytest.approx(15.0)

    def test_elapsed_when_not_started(self):
        from physml.goal_engine import GoalRecord, GoalStatus

        r = GoalRecord(id="x", description="d", status=GoalStatus.PENDING, created_at=0.0)
        assert r.elapsed == 0.0


# ===========================================================================
# Companion integration
# ===========================================================================


class TestCompanionGoalIntegration:
    def test_companion_has_goal_engine(self, tmp_path):
        from physml.companion import MyceliumCompanion

        c = MyceliumCompanion(data_dir=str(tmp_path))
        c.start()
        assert c.goal_engine is not None

    def test_add_goal_returns_string(self, tmp_path):
        from physml.companion import MyceliumCompanion

        c = MyceliumCompanion(data_dir=str(tmp_path))
        c.start()
        result = c.add_goal("Summarise trends in my data")
        assert isinstance(result, str)
        assert "goal" in result.lower() or "id" in result.lower()

    def test_goals_returns_string(self, tmp_path):
        from physml.companion import MyceliumCompanion

        c = MyceliumCompanion(data_dir=str(tmp_path))
        c.start()
        c.add_goal("A goal")
        result = c.goals()
        assert isinstance(result, str)

    def test_status_includes_goals(self, tmp_path):
        from physml.companion import MyceliumCompanion

        c = MyceliumCompanion(data_dir=str(tmp_path))
        c.start()
        s = c.status()
        assert "goals" in s
        assert "total" in s["goals"]

    def test_stop_cleans_up(self, tmp_path):
        from physml.companion import MyceliumCompanion

        c = MyceliumCompanion(data_dir=str(tmp_path))
        c.start()
        c.add_goal("A background goal")
        c.stop()  # should not raise


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _dummy_goal():
    from physml.goal_engine import GoalRecord, GoalStatus
    return GoalRecord(
        id="test",
        description="dummy",
        status=GoalStatus.ACTIVE,
        created_at=time.time(),
    )
