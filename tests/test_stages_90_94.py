"""Tests for stages 90-94.

Stage 90 — RewardModel
Stage 91 — AgentMemory
Stage 92 — TaskDecomposer
Stage 93 — AgentComms
Stage 94 — ReflectionEngine
"""

from __future__ import annotations

import time
import unittest

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stage 90 — RewardModel
# ---------------------------------------------------------------------------
class TestRewardModel(unittest.TestCase):
    def _samples(self):
        from physml.reward_model import RewardSample

        return [
            RewardSample(state=[float(i), float(i * 2)], action=float(i % 3), reward=float(i))
            for i in range(10)
        ]

    def test_import(self):
        from physml.reward_model import RewardModel, RewardSample

        self.assertTrue(True)

    @pytest.mark.slow
    def test_add_and_fit(self):
        from physml.reward_model import RewardModel, RewardSample

        model = RewardModel()
        for s in self._samples():
            model.add_sample(s)
        model.fit()
        self.assertTrue(model.fitted_)

    def test_add_samples_bulk(self):
        from physml.reward_model import RewardModel

        model = RewardModel()
        model.add_samples(self._samples())
        self.assertEqual(len(model.samples_), 10)

    def test_fit_empty_raises(self):
        from physml.reward_model import RewardModel

        with self.assertRaises(ValueError):
            RewardModel().fit()

    def test_predict_before_fit_raises(self):
        from physml.reward_model import RewardModel

        m = RewardModel()
        with self.assertRaises(RuntimeError):
            m.predict([1.0, 2.0], 1.0)

    @pytest.mark.slow
    def test_predict_returns_float(self):
        from physml.reward_model import RewardModel

        model = RewardModel()
        model.add_samples(self._samples())
        model.fit()
        result = model.predict([1.0, 2.0], 1.0)
        self.assertIsInstance(result, float)

    @pytest.mark.slow
    def test_custom_model(self):
        from sklearn.linear_model import Lasso

        from physml.reward_model import RewardModel

        m = RewardModel(model=Lasso())
        m.add_samples(self._samples())
        m.fit()
        self.assertTrue(m.fitted_)

    @pytest.mark.slow
    def test_chaining(self):
        from physml.reward_model import RewardModel

        m = RewardModel()
        m.add_samples(self._samples())
        result = m.fit()
        self.assertIs(result, m)

    def test_reward_sample_fields(self):
        from physml.reward_model import RewardSample

        s = RewardSample(state=[1.0, 2.0], action=0.5, reward=3.14)
        self.assertEqual(s.action, 0.5)
        self.assertAlmostEqual(s.reward, 3.14)


# ---------------------------------------------------------------------------
# Stage 91 — AgentMemory
# ---------------------------------------------------------------------------
class TestAgentMemory(unittest.TestCase):
    def test_import(self):
        from physml.agent_memory import AgentMemory, MemoryEntry

        self.assertTrue(True)

    def test_record_and_recall(self):
        from physml.agent_memory import AgentMemory

        mem = AgentMemory()
        mem.record(observation=[1, 2, 3], action=0, reward=1.0, tag="test")
        entries = mem.recall(tag="test")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].reward, 1.0)

    def test_total_reward(self):
        from physml.agent_memory import AgentMemory

        mem = AgentMemory()
        for r in [1.0, 2.0, 3.0]:
            mem.record(observation={}, reward=r)
        self.assertAlmostEqual(mem.total_reward(), 6.0)

    def test_max_episodic_eviction(self):
        from physml.agent_memory import AgentMemory

        mem = AgentMemory(max_episodic=5)
        for i in range(10):
            mem.record(observation=i)
        self.assertEqual(len(mem.episodic), 5)

    def test_unlimited_episodic(self):
        from physml.agent_memory import AgentMemory

        mem = AgentMemory(max_episodic=-1)
        for i in range(20):
            mem.record(observation=i)
        self.assertEqual(len(mem.episodic), 20)

    def test_semantic_remember_retrieve(self):
        from physml.agent_memory import AgentMemory

        mem = AgentMemory()
        mem.remember("goal", "maximize_score")
        self.assertEqual(mem.retrieve("goal"), "maximize_score")

    def test_semantic_default(self):
        from physml.agent_memory import AgentMemory

        mem = AgentMemory()
        self.assertIsNone(mem.retrieve("missing"))
        self.assertEqual(mem.retrieve("missing", default=42), 42)

    def test_forget(self):
        from physml.agent_memory import AgentMemory

        mem = AgentMemory()
        mem.remember("x", 1)
        self.assertTrue(mem.forget("x"))
        self.assertFalse(mem.forget("x"))
        self.assertIsNone(mem.retrieve("x"))

    def test_clear_episodic(self):
        from physml.agent_memory import AgentMemory

        mem = AgentMemory()
        mem.record(observation=1)
        mem.clear_episodic()
        self.assertEqual(len(mem.episodic), 0)

    def test_recall_n_limit(self):
        from physml.agent_memory import AgentMemory

        mem = AgentMemory()
        for i in range(20):
            mem.record(observation=i)
        recent = mem.recall(n=5)
        self.assertEqual(len(recent), 5)

    def test_memory_entry_timestamp(self):
        from physml.agent_memory import AgentMemory

        before = time.time()
        mem = AgentMemory()
        e = mem.record(observation="obs")
        after = time.time()
        self.assertTrue(before <= e.timestamp <= after)


# ---------------------------------------------------------------------------
# Stage 92 — TaskDecomposer
# ---------------------------------------------------------------------------
class TestTaskDecomposer(unittest.TestCase):
    def test_import(self):
        from physml.task_decomposer import SubTask, TaskDecomposer

        self.assertTrue(True)

    def test_single_goal(self):
        from physml.task_decomposer import TaskDecomposer

        td = TaskDecomposer()
        tasks = td.decompose("Do something important")
        self.assertIsInstance(tasks, list)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].description, "Do something important")

    def test_comma_split(self):
        from physml.task_decomposer import TaskDecomposer

        td = TaskDecomposer()
        tasks = td.decompose("collect data, train model, evaluate")
        self.assertEqual(len(tasks), 3)

    def test_semicolon_split(self):
        from physml.task_decomposer import TaskDecomposer

        td = TaskDecomposer()
        tasks = td.decompose("step1; step2; step3")
        self.assertEqual(len(tasks), 3)

    def test_subtask_indices(self):
        from physml.task_decomposer import TaskDecomposer

        td = TaskDecomposer()
        tasks = td.decompose("a, b, c")
        for i, t in enumerate(tasks):
            self.assertEqual(t.index, i)

    def test_subtask_done_toggle(self):
        from physml.task_decomposer import TaskDecomposer

        td = TaskDecomposer()
        tasks = td.decompose("do x")
        self.assertFalse(tasks[0].done)
        tasks[0].complete()
        self.assertTrue(tasks[0].done)

    def test_custom_rule(self):
        from physml.task_decomposer import TaskDecomposer

        td = TaskDecomposer()
        td.register_rule("train", lambda g: ["prepare data", "fit model", "save"])
        tasks = td.decompose("train the neural net")
        self.assertEqual(len(tasks), 3)
        self.assertEqual(tasks[1].description, "fit model")

    def test_default_steps_fallback(self):
        from physml.task_decomposer import TaskDecomposer

        td = TaskDecomposer(default_steps=["alpha", "beta"])
        tasks = td.decompose("opaque goal")
        self.assertEqual([t.description for t in tasks], ["alpha", "beta"])

    def test_rule_priority_over_heuristic(self):
        from physml.task_decomposer import TaskDecomposer

        td = TaskDecomposer()
        td.register_rule("data", lambda g: ["A", "B"])
        tasks = td.decompose("collect data, also do stuff")
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].description, "A")


# ---------------------------------------------------------------------------
# Stage 93 — AgentComms
# ---------------------------------------------------------------------------
class TestAgentComms(unittest.TestCase):
    def test_import(self):
        from physml.agent_comms import AgentComms, Message

        self.assertTrue(True)

    def test_subscribe_and_broadcast(self):
        from physml.agent_comms import AgentComms, Message

        bus = AgentComms()
        bus.subscribe("agent_b", "updates")
        bus.subscribe("agent_c", "updates")
        msg = Message(sender="agent_a", topic="updates", content="hello")
        delivered = bus.publish(msg)
        self.assertEqual(delivered, 2)

    def test_sender_not_in_own_inbox(self):
        from physml.agent_comms import AgentComms, Message

        bus = AgentComms()
        bus.subscribe("agent_a", "topic")
        bus.subscribe("agent_b", "topic")
        bus.publish(Message(sender="agent_a", topic="topic", content="x"))
        # sender should not receive their own broadcast
        self.assertEqual(bus.pending("agent_a"), 0)
        self.assertEqual(bus.pending("agent_b"), 1)

    def test_direct_message(self):
        from physml.agent_comms import AgentComms, Message

        bus = AgentComms()
        bus.publish(Message(sender="a", topic="cmd", content="go", recipient="b"))
        msgs = bus.receive("b")
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].content, "go")

    def test_receive_clears_inbox(self):
        from physml.agent_comms import AgentComms, Message

        bus = AgentComms()
        bus.subscribe("bot", "t")
        bus.publish(Message(sender="x", topic="t", content=1))
        bus.receive("bot")
        self.assertEqual(bus.pending("bot"), 0)

    def test_receive_by_topic(self):
        from physml.agent_comms import AgentComms, Message

        bus = AgentComms()
        bus.subscribe("bot", "A")
        bus.subscribe("bot", "B")
        bus.publish(Message(sender="x", topic="A", content="alpha"))
        bus.publish(Message(sender="x", topic="B", content="beta"))
        only_a = bus.receive("bot", topic="A")
        self.assertEqual(len(only_a), 1)
        self.assertEqual(only_a[0].topic, "A")
        # B still pending
        self.assertEqual(bus.pending("bot"), 1)

    def test_unsubscribe(self):
        from physml.agent_comms import AgentComms, Message

        bus = AgentComms()
        bus.subscribe("bot", "t")
        bus.unsubscribe("bot", "t")
        bus.publish(Message(sender="x", topic="t", content="msg"))
        self.assertEqual(bus.pending("bot"), 0)

    def test_log_history(self):
        from physml.agent_comms import AgentComms, Message

        bus = AgentComms()
        for i in range(5):
            bus.publish(Message(sender="a", topic="t", content=i))
        self.assertEqual(len(bus.log_), 5)

    def test_no_subscribers_delivers_zero(self):
        from physml.agent_comms import AgentComms, Message

        bus = AgentComms()
        n = bus.publish(Message(sender="a", topic="empty", content="hi"))
        self.assertEqual(n, 0)

    def test_message_timestamp(self):
        from physml.agent_comms import Message

        before = time.time()
        m = Message(sender="a", topic="t", content="x")
        after = time.time()
        self.assertTrue(before <= m.timestamp <= after)


# ---------------------------------------------------------------------------
# Stage 94 — ReflectionEngine
# ---------------------------------------------------------------------------
class TestReflectionEngine(unittest.TestCase):
    def test_import(self):
        from physml.reflection_engine import Reflection, ReflectionEngine

        self.assertTrue(True)

    def test_reflect_basic(self):
        from physml.reflection_engine import ReflectionEngine

        eng = ReflectionEngine(window=5)
        eng.log_rewards([1.0, 2.0, 3.0, 4.0, 5.0])
        r = eng.reflect()
        self.assertAlmostEqual(r.avg_reward, 3.0)
        self.assertGreater(r.std_reward, 0.0)

    def test_reflect_empty_raises(self):
        from physml.reflection_engine import ReflectionEngine

        with self.assertRaises(RuntimeError):
            ReflectionEngine().reflect()

    def test_trend_improving(self):
        from physml.reflection_engine import ReflectionEngine

        eng = ReflectionEngine(window=10, improve_threshold=0.05)
        # first half low, second half high
        eng.log_rewards([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        r = eng.reflect()
        self.assertEqual(r.trend, "improving")

    def test_trend_declining(self):
        from physml.reflection_engine import ReflectionEngine

        eng = ReflectionEngine(window=10, decline_threshold=-0.05)
        eng.log_rewards([2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        r = eng.reflect()
        self.assertEqual(r.trend, "declining")

    def test_trend_stable(self):
        from physml.reflection_engine import ReflectionEngine

        eng = ReflectionEngine(window=6)
        eng.log_rewards([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        r = eng.reflect()
        self.assertEqual(r.trend, "stable")

    def test_insights_not_empty(self):
        from physml.reflection_engine import ReflectionEngine

        eng = ReflectionEngine()
        eng.log_reward(1.0)
        r = eng.reflect()
        self.assertGreater(len(r.insights), 0)

    def test_negative_reward_insight(self):
        from physml.reflection_engine import ReflectionEngine

        eng = ReflectionEngine()
        eng.log_rewards([-1.0, -2.0, -3.0])
        r = eng.reflect()
        combined = " ".join(r.insights)
        self.assertIn("Negative", combined)

    def test_multiple_reflections(self):
        from physml.reflection_engine import ReflectionEngine

        eng = ReflectionEngine(window=3)
        eng.log_rewards([1.0, 2.0, 3.0])
        eng.reflect()
        eng.log_reward(4.0)
        eng.reflect()
        self.assertEqual(len(eng.reflections_), 2)

    def test_window_constraint(self):
        from physml.reflection_engine import ReflectionEngine

        eng = ReflectionEngine(window=3)
        eng.log_rewards([10.0] * 20)
        r = eng.reflect()
        self.assertEqual(r.window, 3)

    def test_summary(self):
        from physml.reflection_engine import ReflectionEngine

        eng = ReflectionEngine()
        eng.log_rewards([1.0, 2.0, 3.0])
        eng.reflect()
        s = eng.summary()
        self.assertIn("total_reflections", s)
        self.assertEqual(s["total_reflections"], 1)

    def test_summary_empty(self):
        from physml.reflection_engine import ReflectionEngine

        s = ReflectionEngine().summary()
        self.assertEqual(s["total_reflections"], 0)

    def test_invalid_window(self):
        from physml.reflection_engine import ReflectionEngine

        with self.assertRaises(ValueError):
            ReflectionEngine(window=0)

    def test_single_episode_std_zero(self):
        from physml.reflection_engine import ReflectionEngine

        eng = ReflectionEngine(window=1)
        eng.log_reward(5.0)
        r = eng.reflect()
        self.assertEqual(r.std_reward, 0.0)

    def test_log_rewards_bulk(self):
        from physml.reflection_engine import ReflectionEngine

        eng = ReflectionEngine()
        eng.log_rewards([1.0, 2.0, 3.0])
        self.assertEqual(len(eng.history_), 3)


if __name__ == "__main__":
    unittest.main()
