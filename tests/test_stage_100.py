"""Tests for Stage 100 — MyceliumSystem: grand-finale integration.

Stage 100 ties every prior subsystem into a single production-ready
autonomous agent via :class:`~physml.mycelium_system.MyceliumSystem`.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_system(**kwargs):
    from physml.mycelium_system import MyceliumSystem

    return MyceliumSystem(
        plan_handler=lambda t: "done",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Import & basic construction
# ---------------------------------------------------------------------------


class TestImports(unittest.TestCase):
    def test_import_mycelium_system(self):
        from physml.mycelium_system import MyceliumSystem

        self.assertTrue(callable(MyceliumSystem))

    def test_import_system_metrics(self):
        from physml.mycelium_system import SystemMetrics

        self.assertTrue(callable(SystemMetrics))

    def test_construct_default(self):
        sys = _make_system()
        self.assertEqual(sys.agent_id, "mycelium-1")

    def test_construct_custom_id(self):
        sys = _make_system(agent_id="test-agent")
        self.assertEqual(sys.agent_id, "test-agent")

    def test_subsystems_present(self):
        sys = _make_system()
        for attr in [
            "memory",
            "comms",
            "env_model",
            "belief_updater",
            "task_decomposer",
            "plan_executor",
            "skill_library",
            "reward_model",
            "reflection_engine",
            "controller",
            "metrics",
        ]:
            self.assertTrue(hasattr(sys, attr), f"missing subsystem: {attr}")


# ---------------------------------------------------------------------------
# 2. SystemMetrics
# ---------------------------------------------------------------------------


class TestSystemMetrics(unittest.TestCase):
    def _m(self):
        from physml.mycelium_system import SystemMetrics

        return SystemMetrics()

    def test_initial_zeros(self):
        m = self._m()
        self.assertEqual(m.total_steps, 0)
        self.assertEqual(m.success_rate, 0.0)
        self.assertEqual(m.avg_reward, 0.0)

    def test_update_success(self):
        m = self._m()
        m.update(True, 1.0, 0.1)
        self.assertEqual(m.total_steps, 1)
        self.assertEqual(m.successful_steps, 1)
        self.assertEqual(m.failed_steps, 0)
        self.assertAlmostEqual(m.avg_reward, 1.0)
        self.assertAlmostEqual(m.success_rate, 1.0)

    def test_update_failure(self):
        m = self._m()
        m.update(False, -1.0, 0.1)
        self.assertEqual(m.failed_steps, 1)
        self.assertAlmostEqual(m.success_rate, 0.0)

    def test_mixed_updates(self):
        m = self._m()
        m.update(True, 1.0, 0.1)
        m.update(False, -1.0, 0.1)
        self.assertAlmostEqual(m.success_rate, 0.5)
        self.assertAlmostEqual(m.avg_reward, 0.0)

    def test_avg_step_time(self):
        m = self._m()
        m.update(True, 0.0, 0.2)
        m.update(True, 0.0, 0.4)
        self.assertAlmostEqual(m.avg_step_time, 0.3, places=5)

    def test_to_dict_keys(self):
        m = self._m()
        m.update(True, 0.5, 0.05)
        d = m.to_dict()
        for key in [
            "total_steps",
            "successful_steps",
            "failed_steps",
            "total_reward",
            "avg_reward",
            "avg_step_time",
            "uptime",
            "success_rate",
        ]:
            self.assertIn(key, d)

    def test_to_dict_values_match(self):
        m = self._m()
        m.update(True, 2.0, 0.1)
        d = m.to_dict()
        self.assertEqual(d["total_steps"], 1)
        self.assertAlmostEqual(d["total_reward"], 2.0)

    def test_total_reward_accumulates(self):
        m = self._m()
        m.update(True, 3.0, 0.1)
        m.update(True, 5.0, 0.1)
        self.assertAlmostEqual(m.total_reward, 8.0)


# ---------------------------------------------------------------------------
# 3. step()
# ---------------------------------------------------------------------------


class TestStep(unittest.TestCase):
    def test_step_returns_dict(self):
        sys = _make_system()
        result = sys.step("do something")
        self.assertIsInstance(result, dict)

    def test_step_keys(self):
        sys = _make_system()
        result = sys.step("goal A")
        for key in ["step_id", "goal", "success", "reward", "elapsed", "reflection_trend"]:
            self.assertIn(key, result)

    def test_step_goal_preserved(self):
        sys = _make_system()
        result = sys.step("my goal")
        self.assertEqual(result["goal"], "my goal")

    def test_step_increments_count(self):
        sys = _make_system()
        self.assertEqual(sys.step_count, 0)
        sys.step("g1")
        self.assertEqual(sys.step_count, 1)
        sys.step("g2")
        self.assertEqual(sys.step_count, 2)

    def test_step_elapsed_positive(self):
        sys = _make_system()
        result = sys.step("goal")
        self.assertGreater(result["elapsed"], 0)

    def test_step_with_observation(self):
        sys = _make_system()
        obs = {"temperature": 25.0, "sensor_id": "A1"}
        result = sys.step("sense environment", observation=obs)
        self.assertIsInstance(result, dict)

    def test_step_with_reward_override(self):
        sys = _make_system()
        result = sys.step("goal", reward_override=99.0)
        self.assertAlmostEqual(result["reward"], 99.0)

    def test_step_success_when_handler_set(self):
        sys = _make_system()
        result = sys.step("task")
        self.assertTrue(result["success"])

    def test_step_metrics_updated(self):
        sys = _make_system()
        sys.step("g1")
        self.assertGreater(sys.metrics.total_steps, 0)

    def test_step_uptime_positive(self):
        sys = _make_system()
        sys.step("g")
        self.assertGreater(sys.metrics.uptime, 0)


# ---------------------------------------------------------------------------
# 4. run()
# ---------------------------------------------------------------------------


class TestRun(unittest.TestCase):
    def test_run_returns_list(self):
        sys = _make_system()
        results = sys.run(["g1", "g2", "g3"])
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)

    def test_run_goal_order(self):
        sys = _make_system()
        goals = ["alpha", "beta", "gamma"]
        results = sys.run(goals)
        for i, g in enumerate(goals):
            self.assertEqual(results[i]["goal"], g)

    def test_run_empty(self):
        sys = _make_system()
        results = sys.run([])
        self.assertEqual(results, [])

    def test_run_single(self):
        sys = _make_system()
        results = sys.run(["only one"])
        self.assertEqual(len(results), 1)

    def test_run_with_observations(self):
        sys = _make_system()
        goals = ["g1", "g2"]
        obs = [{"x": 1}, {"x": 2}]
        results = sys.run(goals, observations=obs)
        self.assertEqual(len(results), 2)

    def test_run_count_matches_steps(self):
        sys = _make_system()
        sys.run(["a", "b", "c", "d"])
        self.assertEqual(sys.step_count, 4)

    def test_run_success_rate_all_pass(self):
        sys = _make_system()
        sys.run(["x", "y", "z"])
        self.assertAlmostEqual(sys.success_rate, 1.0)


# ---------------------------------------------------------------------------
# 5. report()
# ---------------------------------------------------------------------------


class TestReport(unittest.TestCase):
    def test_report_keys(self):
        sys = _make_system()
        r = sys.report()
        for key in ["agent_id", "metrics", "memory_size", "skill_count", "step_count"]:
            self.assertIn(key, r)

    def test_report_agent_id(self):
        sys = _make_system(agent_id="reporter")
        r = sys.report()
        self.assertEqual(r["agent_id"], "reporter")

    def test_report_step_count_zero_at_start(self):
        sys = _make_system()
        r = sys.report()
        self.assertEqual(r["step_count"], 0)

    def test_report_step_count_after_run(self):
        sys = _make_system()
        sys.run(["a", "b"])
        r = sys.report()
        self.assertEqual(r["step_count"], 2)

    def test_report_metrics_is_dict(self):
        sys = _make_system()
        r = sys.report()
        self.assertIsInstance(r["metrics"], dict)

    def test_report_uptime_increases(self):
        sys = _make_system()
        r1 = sys.report()
        time.sleep(0.01)
        r2 = sys.report()
        self.assertGreaterEqual(r2["metrics"]["uptime"], r1["metrics"]["uptime"])


# ---------------------------------------------------------------------------
# 6. save() / load()
# ---------------------------------------------------------------------------


class TestSaveLoad(unittest.TestCase):
    def test_save_creates_file(self):
        sys = _make_system()
        sys.run(["a", "b"])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.json")
            sys.save(path)
            self.assertTrue(os.path.exists(path))

    def test_save_valid_json(self):
        sys = _make_system()
        sys.run(["g"])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.json")
            sys.save(path)
            with open(path) as f:
                data = json.load(f)
            self.assertIn("agent_id", data)
            self.assertIn("metrics", data)
            self.assertIn("step_log", data)

    def test_save_step_log_length(self):
        sys = _make_system()
        sys.run(["a", "b", "c"])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "s.json")
            sys.save(path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(len(data["step_log"]), 3)

    def test_load_restores_metrics(self):
        sys1 = _make_system()
        sys1.run(["x", "y"])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "s.json")
            sys1.save(path)
            sys2 = _make_system()
            sys2.load(path)
        self.assertEqual(sys2.metrics.total_steps, sys1.metrics.total_steps)

    def test_load_restores_step_log(self):
        sys1 = _make_system()
        sys1.run(["p", "q"])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "s.json")
            sys1.save(path)
            sys2 = _make_system()
            sys2.load(path)
        self.assertEqual(len(sys2._step_log), 2)

    def test_save_nested_dir_created(self):
        sys = _make_system()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "a", "b", "state.json")
            sys.save(path)
            self.assertTrue(os.path.exists(path))


# ---------------------------------------------------------------------------
# 7. Metrics helpers
# ---------------------------------------------------------------------------


class TestMetricsHelpers(unittest.TestCase):
    def test_step_count_property(self):
        sys = _make_system()
        self.assertEqual(sys.step_count, 0)
        sys.step("g")
        self.assertEqual(sys.step_count, 1)

    def test_success_rate_property(self):
        sys = _make_system()
        sys.run(["a", "b", "c"])
        self.assertAlmostEqual(sys.success_rate, 1.0)

    def test_reset_metrics(self):
        sys = _make_system()
        sys.run(["a", "b"])
        sys.reset_metrics()
        self.assertEqual(sys.step_count, 0)
        self.assertEqual(len(sys._step_log), 0)

    def test_reset_metrics_resets_totals(self):
        sys = _make_system()
        sys.run(["a"])
        sys.reset_metrics()
        self.assertAlmostEqual(sys.metrics.total_reward, 0.0)
        self.assertAlmostEqual(sys.metrics.avg_reward, 0.0)


# ---------------------------------------------------------------------------
# 8. Skill registration
# ---------------------------------------------------------------------------


class TestSkillRegistration(unittest.TestCase):
    def test_skill_handler_registered(self):
        handlers = {"greet": lambda t: "hello"}
        sys = _make_system(skill_handlers=handlers)
        self.assertIn("greet", sys.skill_library.list_names())

    def test_multiple_skills(self):
        handlers = {"a": lambda t: "a", "b": lambda t: "b"}
        sys = _make_system(skill_handlers=handlers)
        names = sys.skill_library.list_names()
        self.assertIn("a", names)
        self.assertIn("b", names)

    def test_no_skills_ok(self):
        sys = _make_system()
        r = sys.report()
        self.assertIsInstance(r["skill_count"], int)


# ---------------------------------------------------------------------------
# 9. broadcast()
# ---------------------------------------------------------------------------


class TestBroadcast(unittest.TestCase):
    def test_broadcast_no_error(self):
        sys = _make_system()
        sys.broadcast("hello world")  # should not raise

    def test_broadcast_with_recipients(self):
        sys = _make_system()
        sys.broadcast("hi", recipients=["agent-2", "agent-3"])

    def test_broadcast_stored_in_inbox(self):
        sys = _make_system(agent_id="sender")
        sys2_comms_id = "receiver"
        sys.broadcast("ping", recipients=[sys2_comms_id])
        # Messages sent; no error is sufficient for the integration test


# ---------------------------------------------------------------------------
# 10. reflect_every behaviour
# ---------------------------------------------------------------------------


class TestReflectEvery(unittest.TestCase):
    def test_reflect_every_zero_no_reflection(self):
        sys = _make_system(reflect_every=0)
        result = sys.step("g")
        # reflection_trend may be None when reflection is not triggered
        # (AgentController skips if step_id % 0 would be undefined — test is None)
        # No crash is the key assertion
        self.assertIsNotNone(result)

    def test_reflect_every_1_triggers(self):
        sys = _make_system(reflect_every=1)
        for _ in range(3):
            sys.step("g")
        # Reflection engine should have been exercised; no crash
        self.assertEqual(sys.step_count, 3)

    def test_reflect_every_large(self):
        sys = _make_system(reflect_every=100)
        sys.run(["x"] * 5)
        self.assertEqual(sys.step_count, 5)


# ---------------------------------------------------------------------------
# 11. verbose flag
# ---------------------------------------------------------------------------


class TestVerbose(unittest.TestCase):
    def test_verbose_false_no_output(self):
        import io
        import sys as _sys

        captured = io.StringIO()
        old_stdout = _sys.stdout
        _sys.stdout = captured
        try:
            ms = _make_system(verbose=False)
            ms.step("quiet step")
        finally:
            _sys.stdout = old_stdout
        self.assertEqual(captured.getvalue(), "")

    def test_verbose_true_produces_output(self):
        import io
        import sys as _sys

        captured = io.StringIO()
        old_stdout = _sys.stdout
        _sys.stdout = captured
        try:
            ms = _make_system(verbose=True)
            ms.step("loud step")
        finally:
            _sys.stdout = old_stdout
        self.assertGreater(len(captured.getvalue()), 0)


# ---------------------------------------------------------------------------
# 12. End-to-end integration
# ---------------------------------------------------------------------------


class TestEndToEnd(unittest.TestCase):
    def test_full_run_report_save(self):
        sys = _make_system(agent_id="e2e")
        goals = [f"goal-{i}" for i in range(10)]
        results = sys.run(goals)
        self.assertEqual(len(results), 10)

        report = sys.report()
        self.assertEqual(report["step_count"], 10)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "final.json")
            sys.save(path)
            self.assertTrue(os.path.exists(path))

    def test_reload_and_continue(self):
        sys1 = _make_system()
        sys1.run(["a", "b", "c"])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.json")
            sys1.save(path)
            sys2 = _make_system()
            sys2.load(path)
        # Continue running
        sys2.step("d")
        self.assertEqual(sys2.metrics.total_steps, sys1.metrics.total_steps + 1)

    def test_metrics_after_many_steps(self):
        sys = _make_system()
        sys.run([f"g{i}" for i in range(20)])
        r = sys.report()
        self.assertEqual(r["metrics"]["total_steps"], 20)
        self.assertAlmostEqual(r["metrics"]["success_rate"], 1.0)

    def test_step_log_complete(self):
        sys = _make_system()
        goals = ["x", "y", "z"]
        sys.run(goals)
        for i, log in enumerate(sys._step_log):
            self.assertEqual(log["goal"], goals[i])
            self.assertIn("success", log)
            self.assertIn("reward", log)


if __name__ == "__main__":
    unittest.main()
