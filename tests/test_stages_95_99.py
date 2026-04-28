"""Tests for stages 95-99.

Stage 95 — PlanExecutor
Stage 96 — EnvironmentModel
Stage 97 — SkillLibrary
Stage 98 — BeliefUpdater
Stage 99 — AgentController
"""

from __future__ import annotations

import math
import time
import unittest

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stage 95 — PlanExecutor
# ---------------------------------------------------------------------------
class TestPlanExecutor(unittest.TestCase):
    def _tasks(self, n=3):
        from physml.task_decomposer import TaskDecomposer

        return TaskDecomposer().decompose(", ".join(f"step{i}" for i in range(n)))

    def test_import(self):

        self.assertTrue(True)

    def test_execute_all_success(self):
        from physml.plan_executor import PlanExecutor

        pe = PlanExecutor(default_handler=lambda t: "done")
        result = pe.execute(self._tasks(3))
        self.assertEqual(result.completed, 3)
        self.assertEqual(result.failed, 0)
        self.assertTrue(result.success)

    def test_execute_missing_handler_fails(self):
        from physml.plan_executor import PlanExecutor

        pe = PlanExecutor()  # no default handler
        tasks = self._tasks(2)
        result = pe.execute(tasks)
        self.assertGreater(result.failed, 0)

    def test_stop_on_error_skips_remaining(self):
        from physml.plan_executor import PlanExecutor

        pe = PlanExecutor(stop_on_error=True)
        tasks = self._tasks(3)
        result = pe.execute(tasks)
        self.assertEqual(result.failed + result.skipped, 3)

    def test_no_stop_on_error_continues(self):
        from physml.plan_executor import PlanExecutor

        pe = PlanExecutor(stop_on_error=False)
        tasks = self._tasks(3)
        result = pe.execute(tasks)
        self.assertEqual(result.skipped, 0)
        self.assertEqual(result.failed, 3)

    def test_register_handler_by_keyword(self):
        from physml.plan_executor import PlanExecutor
        from physml.task_decomposer import TaskDecomposer

        pe = PlanExecutor()
        pe.register("step", lambda t: "ok")
        tasks = TaskDecomposer().decompose("step1, step2")
        result = pe.execute(tasks)
        self.assertEqual(result.completed, 2)

    def test_max_retries(self):
        from physml.plan_executor import PlanExecutor

        calls = {"n": 0}

        def flaky(t):
            calls["n"] += 1
            raise ValueError("oops")

        pe = PlanExecutor(max_retries=2, default_handler=flaky)
        tasks = self._tasks(1)
        result = pe.execute(tasks)
        self.assertEqual(calls["n"], 3)  # 1 original + 2 retries
        self.assertEqual(result.failed, 1)

    def test_history_grows(self):
        from physml.plan_executor import PlanExecutor

        pe = PlanExecutor(default_handler=lambda t: None)
        pe.execute(self._tasks(1))
        pe.execute(self._tasks(1))
        self.assertEqual(len(pe.history_), 2)

    def test_execution_result_fields(self):
        from physml.plan_executor import PlanExecutor

        pe = PlanExecutor(default_handler=lambda t: 42)
        tasks = self._tasks(2)
        result = pe.execute(tasks, plan_id="test_plan")
        self.assertEqual(result.plan_id, "test_plan")
        self.assertEqual(result.total, 2)
        self.assertGreaterEqual(result.elapsed, 0.0)

    def test_subtasks_marked_done(self):
        from physml.plan_executor import PlanExecutor

        pe = PlanExecutor(default_handler=lambda t: None)
        tasks = self._tasks(3)
        pe.execute(tasks)
        for t in tasks:
            self.assertTrue(t.done)

    def test_outcomes_length_matches_tasks(self):
        from physml.plan_executor import PlanExecutor

        pe = PlanExecutor(default_handler=lambda t: None)
        tasks = self._tasks(4)
        result = pe.execute(tasks)
        self.assertEqual(len(result.outcomes), 4)

    def test_mixed_handlers(self):
        from physml.plan_executor import PlanExecutor
        from physml.task_decomposer import TaskDecomposer

        pe = PlanExecutor()
        pe.register("collect", lambda t: "collected")
        pe.register("train", lambda t: "trained")
        tasks = TaskDecomposer().decompose("collect data, train model")
        result = pe.execute(tasks)
        self.assertEqual(result.completed, 2)


# ---------------------------------------------------------------------------
# Stage 96 — EnvironmentModel
# ---------------------------------------------------------------------------
class TestEnvironmentModel(unittest.TestCase):
    def _populate(self, model, n=10):
        rng = np.random.default_rng(42)
        for i in range(n):
            model.record_transition(
                obs=rng.random(3).tolist(),
                action=float(i % 2),
                reward=float(i),
            )

    def test_import(self):

        self.assertTrue(True)

    @pytest.mark.slow
    def test_record_and_fit(self):
        from physml.environment_model import EnvironmentModel

        m = EnvironmentModel()
        self._populate(m, 10)
        m.fit()
        self.assertTrue(m.fitted_)

    @pytest.mark.slow
    def test_predict_next_shape(self):
        from physml.environment_model import EnvironmentModel

        m = EnvironmentModel()
        self._populate(m, 10)
        m.fit()
        nxt = m.predict_next([0.1, 0.2, 0.3], action=0.0)
        self.assertEqual(len(nxt), 3)

    def test_fit_too_few_raises(self):
        from physml.environment_model import EnvironmentModel

        m = EnvironmentModel()
        m.record_transition([1.0, 2.0], action=0.0)
        with self.assertRaises(ValueError):
            m.fit()

    def test_predict_before_fit_raises(self):
        from physml.environment_model import EnvironmentModel

        m = EnvironmentModel()
        self._populate(m)
        with self.assertRaises(RuntimeError):
            m.predict_next([0.0, 0.0, 0.0])

    def test_max_history_eviction(self):
        from physml.environment_model import EnvironmentModel

        m = EnvironmentModel(max_history=5)
        self._populate(m, 10)
        self.assertEqual(len(m.history_), 5)

    def test_unlimited_history(self):
        from physml.environment_model import EnvironmentModel

        m = EnvironmentModel(max_history=-1)
        self._populate(m, 20)
        self.assertEqual(len(m.history_), 20)

    def test_avg_reward(self):
        from physml.environment_model import EnvironmentModel

        m = EnvironmentModel()
        for r in [1.0, 2.0, 3.0]:
            m.record_transition([0.0], action=0.0, reward=r)
        self.assertAlmostEqual(m.avg_reward(), 2.0)

    @pytest.mark.slow
    def test_chaining(self):
        from physml.environment_model import EnvironmentModel

        m = EnvironmentModel()
        self._populate(m)
        self.assertIs(m.fit(), m)

    def test_env_state_timestamp(self):
        from physml.environment_model import EnvState

        before = time.time()
        s = EnvState(obs=[1.0, 2.0])
        after = time.time()
        self.assertTrue(before <= s.timestamp <= after)

    @pytest.mark.slow
    def test_custom_model(self):
        from sklearn.linear_model import Lasso

        from physml.environment_model import EnvironmentModel

        m = EnvironmentModel(model=Lasso())
        self._populate(m)
        m.fit()
        self.assertTrue(m.fitted_)


# ---------------------------------------------------------------------------
# Stage 97 — SkillLibrary
# ---------------------------------------------------------------------------
class TestSkillLibrary(unittest.TestCase):
    def test_import(self):

        self.assertTrue(True)

    def test_register_and_invoke(self):
        from physml.skill_library import SkillLibrary

        lib = SkillLibrary()
        lib.register("add", lambda a, b: a + b)
        result = lib.invoke("add", 2, 3)
        self.assertEqual(result, 5)

    def test_has(self):
        from physml.skill_library import SkillLibrary

        lib = SkillLibrary()
        lib.register("foo", lambda: None)
        self.assertTrue(lib.has("foo"))
        self.assertFalse(lib.has("bar"))

    def test_get_missing_raises(self):
        from physml.skill_library import SkillLibrary

        with self.assertRaises(KeyError):
            SkillLibrary().get("missing")

    def test_duplicate_register_raises(self):
        from physml.skill_library import SkillLibrary

        lib = SkillLibrary()
        lib.register("x", lambda: 1)
        with self.assertRaises(ValueError):
            lib.register("x", lambda: 2)

    def test_update_overwrites(self):
        from physml.skill_library import SkillLibrary

        lib = SkillLibrary()
        lib.register("x", lambda: 1)
        lib.update("x", lambda: 99)
        self.assertEqual(lib.invoke("x"), 99)

    def test_find_by_tag(self):
        from physml.skill_library import SkillLibrary

        lib = SkillLibrary()
        lib.register("skill_a", lambda: None, tags=["ml", "data"])
        lib.register("skill_b", lambda: None, tags=["ml"])
        lib.register("skill_c", lambda: None, tags=["ops"])
        results = lib.find_by_tag("ml")
        self.assertEqual(len(results), 2)

    def test_find_by_tag_case_insensitive(self):
        from physml.skill_library import SkillLibrary

        lib = SkillLibrary()
        lib.register("s", lambda: None, tags=["ML"])
        self.assertEqual(len(lib.find_by_tag("ml")), 1)

    def test_call_count(self):
        from physml.skill_library import SkillLibrary

        lib = SkillLibrary()
        lib.register("counter", lambda: None)
        lib.invoke("counter")
        lib.invoke("counter")
        self.assertEqual(lib.get("counter").call_count, 2)

    def test_remove(self):
        from physml.skill_library import SkillLibrary

        lib = SkillLibrary()
        lib.register("tmp", lambda: None)
        self.assertTrue(lib.remove("tmp"))
        self.assertFalse(lib.has("tmp"))

    def test_remove_missing(self):
        from physml.skill_library import SkillLibrary

        self.assertFalse(SkillLibrary().remove("ghost"))

    def test_list_names(self):
        from physml.skill_library import SkillLibrary

        lib = SkillLibrary()
        lib.register("b", lambda: None)
        lib.register("a", lambda: None)
        self.assertEqual(lib.list_names(), ["a", "b"])

    def test_len(self):
        from physml.skill_library import SkillLibrary

        lib = SkillLibrary()
        lib.register("one", lambda: None)
        lib.register("two", lambda: None)
        self.assertEqual(len(lib), 2)

    def test_last_called_timestamp(self):
        from physml.skill_library import SkillLibrary

        lib = SkillLibrary()
        lib.register("ts", lambda: None)
        before = time.time()
        lib.invoke("ts")
        after = time.time()
        skill = lib.get("ts")
        self.assertTrue(before <= skill.last_called <= after)


# ---------------------------------------------------------------------------
# Stage 98 — BeliefUpdater
# ---------------------------------------------------------------------------
class TestBeliefUpdater(unittest.TestCase):
    def test_import(self):

        self.assertTrue(True)

    def test_uniform_prior(self):
        from physml.belief_updater import BeliefUpdater

        bu = BeliefUpdater(["A", "B", "C"])
        for p in bu.distribution_.values():
            self.assertAlmostEqual(p, 1 / 3)

    def test_custom_prior_normalised(self):
        from physml.belief_updater import BeliefUpdater

        bu = BeliefUpdater(["A", "B"], prior={"A": 3.0, "B": 1.0})
        self.assertAlmostEqual(bu.distribution_["A"], 0.75)
        self.assertAlmostEqual(bu.distribution_["B"], 0.25)

    def test_update_shifts_distribution(self):
        from physml.belief_updater import BeliefUpdater

        bu = BeliefUpdater(["H1", "H2"])
        bu.set_likelihood("obs_a", {"H1": 0.9, "H2": 0.1})
        bu.update("obs_a")
        self.assertGreater(bu.distribution_["H1"], bu.distribution_["H2"])

    def test_update_normalised(self):
        from physml.belief_updater import BeliefUpdater

        bu = BeliefUpdater(["H1", "H2", "H3"])
        bu.set_likelihood("e", {"H1": 0.5, "H2": 0.3, "H3": 0.2})
        bu.update("e")
        self.assertAlmostEqual(sum(bu.distribution_.values()), 1.0)

    def test_most_likely(self):
        from physml.belief_updater import BeliefUpdater

        bu = BeliefUpdater(["A", "B"], prior={"A": 9.0, "B": 1.0})
        self.assertEqual(bu.most_likely(), "A")

    def test_belief_snapshot(self):
        from physml.belief_updater import BeliefUpdater

        bu = BeliefUpdater(["X", "Y"])
        bu.set_likelihood("ev", {"X": 0.8, "Y": 0.2})
        belief = bu.update("ev")
        self.assertEqual(belief.evidence, "ev")
        self.assertEqual(belief.most_likely, "X")
        self.assertIn("X", belief.distribution)

    def test_history_grows(self):
        from physml.belief_updater import BeliefUpdater

        bu = BeliefUpdater(["A", "B"])
        bu.update("unknown1")
        bu.update("unknown2")
        self.assertEqual(len(bu.history_), 2)

    def test_unknown_evidence_no_change(self):
        from physml.belief_updater import BeliefUpdater

        bu = BeliefUpdater(["A", "B"])
        before = dict(bu.distribution_)
        bu.update("mystery")
        self.assertEqual(bu.distribution_, before)

    def test_entropy_uniform(self):
        from physml.belief_updater import BeliefUpdater

        bu = BeliefUpdater(["A", "B"])
        ent = bu.entropy()
        self.assertAlmostEqual(ent, math.log(2), places=5)

    def test_entropy_certain(self):
        from physml.belief_updater import BeliefUpdater

        bu = BeliefUpdater(["A", "B"], prior={"A": 1.0, "B": 0.0})
        self.assertAlmostEqual(bu.entropy(), 0.0, places=5)

    def test_reset_uniform(self):
        from physml.belief_updater import BeliefUpdater

        bu = BeliefUpdater(["A", "B"])
        bu.update("x")
        bu.reset()
        self.assertEqual(len(bu.history_), 0)
        for p in bu.distribution_.values():
            self.assertAlmostEqual(p, 0.5)

    def test_empty_hypotheses_raises(self):
        from physml.belief_updater import BeliefUpdater

        with self.assertRaises(ValueError):
            BeliefUpdater([])

    def test_multiple_updates_converge(self):
        from physml.belief_updater import BeliefUpdater

        bu = BeliefUpdater(["correct", "wrong"])
        bu.set_likelihood("signal", {"correct": 0.9, "wrong": 0.1})
        for _ in range(10):
            bu.update("signal")
        self.assertGreater(bu.distribution_["correct"], 0.99)


# ---------------------------------------------------------------------------
# Stage 99 — AgentController
# ---------------------------------------------------------------------------
class TestAgentController(unittest.TestCase):
    def _make_controller(self):
        from physml.agent_controller import AgentController
        from physml.agent_memory import AgentMemory
        from physml.belief_updater import BeliefUpdater
        from physml.plan_executor import PlanExecutor
        from physml.reflection_engine import ReflectionEngine
        from physml.task_decomposer import TaskDecomposer

        mem = AgentMemory()
        td = TaskDecomposer()
        pe = PlanExecutor(default_handler=lambda t: "ok")
        bu = BeliefUpdater(["state_a", "state_b"])
        bu.set_likelihood("ev", {"state_a": 0.8, "state_b": 0.2})
        re = ReflectionEngine(window=3)
        return AgentController(
            memory=mem,
            task_decomposer=td,
            plan_executor=pe,
            belief_updater=bu,
            reflection_engine=re,
            reflect_every=3,
        )

    def test_import(self):

        self.assertTrue(True)

    def test_single_step(self):
        ctrl = self._make_controller()
        cs = ctrl.step("do something")
        self.assertEqual(cs.step_id, 0)
        self.assertIsInstance(cs.reward, float)

    def test_step_increments_count(self):
        ctrl = self._make_controller()
        ctrl.step("goal 1")
        ctrl.step("goal 2")
        self.assertEqual(ctrl.step_count_, 2)

    def test_step_stores_observation(self):
        ctrl = self._make_controller()
        ctrl.step("g", observation={"x": 1})
        self.assertEqual(len(ctrl.memory.episodic), 1)

    def test_belief_evidence_processed(self):
        ctrl = self._make_controller()
        cs = ctrl.step("g", evidence="ev")
        self.assertIn("most_likely", cs.metadata)

    def test_plan_success_true(self):
        ctrl = self._make_controller()
        cs = ctrl.step("step1, step2")
        self.assertTrue(cs.plan_success)

    def test_reflection_after_n_steps(self):
        ctrl = self._make_controller()
        for i in range(3):
            cs = ctrl.step(f"goal {i}")
        self.assertIsNotNone(cs.reflection_trend)

    def test_run_multiple_goals(self):
        ctrl = self._make_controller()
        goals = ["goal a", "goal b", "goal c"]
        results = ctrl.run(goals)
        self.assertEqual(len(results), 3)
        self.assertEqual(ctrl.step_count_, 3)

    def test_summary_fields(self):
        ctrl = self._make_controller()
        ctrl.run(["g1", "g2", "g3"])
        s = ctrl.summary()
        self.assertIn("total_steps", s)
        self.assertEqual(s["total_steps"], 3)
        self.assertIn("avg_reward", s)

    def test_summary_empty(self):
        from physml.agent_controller import AgentController

        s = AgentController().summary()
        self.assertEqual(s["total_steps"], 0)

    def test_custom_reward_fn(self):
        from physml.agent_controller import AgentController

        ctrl = AgentController(reward_fn=lambda s: 42.0)
        cs = ctrl.step("x")
        self.assertAlmostEqual(cs.reward, 42.0)

    def test_no_subsystems(self):
        from physml.agent_controller import AgentController

        ctrl = AgentController()
        cs = ctrl.step("bare goal")
        self.assertIsNotNone(cs)

    def test_control_step_fields(self):
        ctrl = self._make_controller()
        cs = ctrl.step("test")
        self.assertIsInstance(cs.elapsed, float)
        self.assertIsInstance(cs.goal, str)
        self.assertIsInstance(cs.metadata, dict)

    def test_run_with_observations(self):
        ctrl = self._make_controller()
        ctrl.run(["a", "b"], observations=[{"v": 1}, {"v": 2}])
        self.assertEqual(len(ctrl.memory.episodic), 2)


if __name__ == "__main__":
    unittest.main()
