"""Tests for stages 101-105.

Stage 101 — NeuralSearchEngine
Stage 102 — TraceRecorder
Stage 103 — PolicyOptimizer
Stage 104 — ValueEstimator
Stage 105 — ActionSelector
"""

from __future__ import annotations

import math
import time
import unittest

import numpy as np


# ---------------------------------------------------------------------------
# Stage 101 — NeuralSearchEngine
# ---------------------------------------------------------------------------
class TestNeuralSearchEngine(unittest.TestCase):
    def _make_data(self, n=80, n_features=4):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, n_features))
        y = (X[:, 0] > 0).astype(int)
        return X, y

    def test_import(self):
        from physml.neural_search import NeuralSearchEngine, SearchResult

        self.assertTrue(True)

    def test_search_returns_best_result(self):
        from physml.neural_search import NeuralSearchEngine, SearchResult

        X, y = self._make_data()
        eng = NeuralSearchEngine(
            search_space=[(32,), (64, 32)], cv=2, max_iter=20, random_state=42
        )
        best = eng.search(X, y)
        self.assertIsInstance(best, SearchResult)
        self.assertGreaterEqual(best.score, 0.0)
        self.assertLessEqual(best.score, 1.0)

    def test_all_results_populated(self):
        from physml.neural_search import NeuralSearchEngine

        X, y = self._make_data()
        eng = NeuralSearchEngine(search_space=[(32,), (64,)], cv=2, max_iter=15)
        eng.search(X, y)
        self.assertEqual(len(eng.all_results), 2)

    def test_best_result_is_highest_score(self):
        from physml.neural_search import NeuralSearchEngine

        X, y = self._make_data()
        eng = NeuralSearchEngine(search_space=[(32,), (64,), (128,)], cv=2, max_iter=15)
        best = eng.search(X, y)
        for r in eng.all_results:
            self.assertLessEqual(r.score, best.score + 1e-9)

    def test_reset_clears_results(self):
        from physml.neural_search import NeuralSearchEngine

        X, y = self._make_data()
        eng = NeuralSearchEngine(search_space=[(32,)], cv=2, max_iter=10)
        eng.search(X, y)
        eng.reset()
        self.assertEqual(len(eng.all_results), 0)
        self.assertIsNone(eng.best_result)

    def test_regression_task(self):
        from physml.neural_search import NeuralSearchEngine

        rng = np.random.default_rng(1)
        X = rng.standard_normal((60, 3))
        y = X[:, 0] * 2 + rng.standard_normal(60) * 0.1
        eng = NeuralSearchEngine(
            search_space=[(32,)], cv=2, max_iter=20, task="regression"
        )
        best = eng.search(X, y)
        self.assertIsNotNone(best)

    def test_search_result_hidden_layers(self):
        from physml.neural_search import NeuralSearchEngine

        X, y = self._make_data()
        space = [(16, 8), (32,)]
        eng = NeuralSearchEngine(search_space=space, cv=2, max_iter=10)
        eng.search(X, y)
        returned_layers = {r.hidden_layers for r in eng.all_results}
        self.assertEqual(returned_layers, {(16, 8), (32,)})

    def test_train_time_positive(self):
        from physml.neural_search import NeuralSearchEngine

        X, y = self._make_data()
        eng = NeuralSearchEngine(search_space=[(32,)], cv=2, max_iter=10)
        eng.search(X, y)
        for r in eng.all_results:
            self.assertGreaterEqual(r.train_time, 0.0)

    def test_default_search_space(self):
        from physml.neural_search import NeuralSearchEngine

        eng = NeuralSearchEngine()
        self.assertGreater(len(eng.search_space), 0)

    def test_no_results_before_search(self):
        from physml.neural_search import NeuralSearchEngine

        eng = NeuralSearchEngine()
        self.assertEqual(len(eng.all_results), 0)
        self.assertIsNone(eng.best_result)

    def test_override_search_space_in_call(self):
        from physml.neural_search import NeuralSearchEngine

        X, y = self._make_data()
        eng = NeuralSearchEngine(search_space=[(256, 128)], cv=2, max_iter=10)
        best = eng.search(X, y, search_space=[(16,)])
        self.assertEqual(best.hidden_layers, (16,))


# ---------------------------------------------------------------------------
# Stage 102 — TraceRecorder
# ---------------------------------------------------------------------------
class TestTraceRecorder(unittest.TestCase):
    def test_import(self):
        from physml.trace_recorder import ExecutionTrace, TraceRecorder

        self.assertTrue(True)

    def test_record_returns_trace(self):
        from physml.trace_recorder import ExecutionTrace, TraceRecorder

        rec = TraceRecorder()
        t = rec.record("observe", {"x": 1})
        self.assertIsInstance(t, ExecutionTrace)
        self.assertEqual(t.event_type, "observe")
        self.assertEqual(t.payload["x"], 1)

    def test_sequential_event_ids(self):
        from physml.trace_recorder import TraceRecorder

        rec = TraceRecorder()
        ids = [rec.record("action").event_id for _ in range(5)]
        self.assertEqual(ids, list(range(5)))

    def test_len(self):
        from physml.trace_recorder import TraceRecorder

        rec = TraceRecorder()
        for _ in range(7):
            rec.record("reward", {"r": 1.0})
        self.assertEqual(len(rec), 7)

    def test_filter_by_event_type(self):
        from physml.trace_recorder import TraceRecorder

        rec = TraceRecorder()
        rec.record("observe")
        rec.record("action")
        rec.record("observe")
        obs = rec.filter(event_type="observe")
        self.assertEqual(len(obs), 2)

    def test_filter_by_agent_id(self):
        from physml.trace_recorder import TraceRecorder

        rec = TraceRecorder(agent_id="alpha")
        rec.record("action", agent_id="alpha")
        rec.record("action", agent_id="beta")
        alpha = rec.filter(agent_id="alpha")
        self.assertEqual(len(alpha), 1)

    def test_summary_totals(self):
        from physml.trace_recorder import TraceRecorder

        rec = TraceRecorder()
        for _ in range(3):
            rec.record("observe")
        for _ in range(2):
            rec.record("action")
        s = rec.summary()
        self.assertEqual(s["total"], 5)
        self.assertEqual(s["by_type"]["observe"], 3)
        self.assertEqual(s["by_type"]["action"], 2)

    def test_to_dicts(self):
        from physml.trace_recorder import TraceRecorder

        rec = TraceRecorder()
        rec.record("reward", {"r": 0.5})
        dicts = rec.to_dicts()
        self.assertEqual(len(dicts), 1)
        self.assertIn("event_id", dicts[0])
        self.assertIn("payload", dicts[0])

    def test_max_size_enforced(self):
        from physml.trace_recorder import TraceRecorder

        rec = TraceRecorder(max_size=3)
        for _ in range(6):
            rec.record("ping")
        self.assertEqual(len(rec), 3)

    def test_clear(self):
        from physml.trace_recorder import TraceRecorder

        rec = TraceRecorder()
        rec.record("foo")
        rec.clear()
        self.assertEqual(len(rec), 0)

    def test_iter(self):
        from physml.trace_recorder import TraceRecorder

        rec = TraceRecorder()
        rec.record("a")
        rec.record("b")
        types = [t.event_type for t in rec]
        self.assertEqual(types, ["a", "b"])

    def test_timestamp_monotonic(self):
        from physml.trace_recorder import TraceRecorder

        rec = TraceRecorder()
        for _ in range(5):
            rec.record("tick")
        timestamps = [t.timestamp for t in rec]
        self.assertEqual(timestamps, sorted(timestamps))

    def test_custom_timestamp(self):
        from physml.trace_recorder import TraceRecorder

        rec = TraceRecorder()
        t = rec.record("event", timestamp=1_000_000.0)
        self.assertEqual(t.timestamp, 1_000_000.0)


# ---------------------------------------------------------------------------
# Stage 103 — PolicyOptimizer
# ---------------------------------------------------------------------------
class TestPolicyOptimizer(unittest.TestCase):
    def _simple_episode(self, n_steps=5, n_actions=3):
        rng = np.random.default_rng(0)
        states = [rng.standard_normal(4) for _ in range(n_steps)]
        actions = [int(rng.integers(0, n_actions)) for _ in range(n_steps)]
        rewards = rng.uniform(0, 1, n_steps).tolist()
        return states, actions, rewards

    def test_import(self):
        from physml.policy_optimizer import PolicyOptimizer, PolicyUpdate

        self.assertTrue(True)

    def test_select_action_in_range(self):
        from physml.policy_optimizer import PolicyOptimizer

        opt = PolicyOptimizer(n_actions=4, state_dim=4, random_state=0)
        state = np.zeros(4)
        a = opt.select_action(state)
        self.assertIn(a, range(4))

    def test_action_probs_sum_to_one(self):
        from physml.policy_optimizer import PolicyOptimizer

        opt = PolicyOptimizer(n_actions=3, state_dim=4)
        probs = opt.action_probs(np.ones(4))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_update_returns_policy_update(self):
        from physml.policy_optimizer import PolicyOptimizer, PolicyUpdate

        opt = PolicyOptimizer(n_actions=3, state_dim=4, random_state=1)
        states, actions, rewards = self._simple_episode()
        pu = opt.update(states, actions, rewards)
        self.assertIsInstance(pu, PolicyUpdate)

    def test_update_increments_episode_count(self):
        from physml.policy_optimizer import PolicyOptimizer

        opt = PolicyOptimizer(n_actions=3, state_dim=4)
        for _ in range(3):
            states, actions, rewards = self._simple_episode()
            opt.update(states, actions, rewards)
        self.assertEqual(len(opt.update_history), 3)

    def test_policy_norm_positive(self):
        from physml.policy_optimizer import PolicyOptimizer

        opt = PolicyOptimizer(n_actions=2, state_dim=4, random_state=7)
        states, actions, rewards = self._simple_episode(n_actions=2)
        pu = opt.update(states, actions, rewards)
        self.assertGreater(pu.policy_norm, 0.0)

    def test_reset_clears_history(self):
        from physml.policy_optimizer import PolicyOptimizer

        opt = PolicyOptimizer(n_actions=3, state_dim=4)
        states, actions, rewards = self._simple_episode()
        opt.update(states, actions, rewards)
        opt.reset()
        self.assertEqual(len(opt.update_history), 0)

    def test_invalid_n_actions_raises(self):
        from physml.policy_optimizer import PolicyOptimizer

        with self.assertRaises(ValueError):
            PolicyOptimizer(n_actions=0, state_dim=4)

    def test_multiple_updates_change_policy(self):
        from physml.policy_optimizer import PolicyOptimizer

        opt = PolicyOptimizer(n_actions=3, state_dim=4, random_state=2)
        probs_before = opt.action_probs(np.ones(4)).copy()
        for _ in range(10):
            states, actions, rewards = self._simple_episode()
            opt.update(states, actions, rewards)
        probs_after = opt.action_probs(np.ones(4))
        self.assertFalse(np.allclose(probs_before, probs_after))


# ---------------------------------------------------------------------------
# Stage 104 — ValueEstimator
# ---------------------------------------------------------------------------
class TestValueEstimator(unittest.TestCase):
    def test_import(self):
        from physml.value_estimator import ValueEstimate, ValueEstimator

        self.assertTrue(True)

    def test_estimate_returns_value_estimate(self):
        from physml.value_estimator import ValueEstimate, ValueEstimator

        ve = ValueEstimator(state_dim=4, random_state=0)
        est = ve.estimate(np.zeros(4))
        self.assertIsInstance(est, ValueEstimate)

    def test_initial_value_near_zero(self):
        from physml.value_estimator import ValueEstimator

        ve = ValueEstimator(state_dim=4, random_state=42)
        est = ve.estimate(np.zeros(4))
        self.assertAlmostEqual(est.value, 0.0, places=2)

    def test_update_returns_td_error(self):
        from physml.value_estimator import ValueEstimator

        ve = ValueEstimator(state_dim=3, random_state=0)
        est = ve.update(np.ones(3), reward=1.0, next_state=np.zeros(3))
        self.assertIsNotNone(est.td_error)

    def test_n_updates_increments(self):
        from physml.value_estimator import ValueEstimator

        ve = ValueEstimator(state_dim=3)
        for _ in range(5):
            ve.update(np.ones(3), 1.0, np.zeros(3))
        self.assertEqual(ve.n_updates, 5)

    def test_terminal_state_bootstrap_zero(self):
        from physml.value_estimator import ValueEstimator

        ve = ValueEstimator(state_dim=2, learning_rate=0.5, gamma=0.99)
        # V(s) should move toward r when terminal
        est = ve.update(np.array([1.0, 0.0]), reward=10.0, next_state=np.zeros(2), terminal=True)
        self.assertIsNotNone(est)

    def test_batch_update(self):
        from physml.value_estimator import ValueEstimator

        ve = ValueEstimator(state_dim=3)
        states = [np.ones(3) * i for i in range(4)]
        rewards = [1.0] * 4
        next_states = [np.zeros(3)] * 4
        results = ve.batch_update(states, rewards, next_states)
        self.assertEqual(len(results), 4)

    def test_weights_shape(self):
        from physml.value_estimator import ValueEstimator

        ve = ValueEstimator(state_dim=5)
        self.assertEqual(ve.weights.shape, (5,))

    def test_mean_td_error_zero_before_updates(self):
        from physml.value_estimator import ValueEstimator

        ve = ValueEstimator(state_dim=2)
        self.assertEqual(ve.mean_td_error(), 0.0)

    def test_mean_td_error_positive_after_updates(self):
        from physml.value_estimator import ValueEstimator

        ve = ValueEstimator(state_dim=3, random_state=0)
        ve.update(np.ones(3), 5.0, np.zeros(3))
        self.assertGreater(ve.mean_td_error(), 0.0)

    def test_reset_clears_state(self):
        from physml.value_estimator import ValueEstimator

        ve = ValueEstimator(state_dim=3)
        ve.update(np.ones(3), 1.0, np.zeros(3))
        ve.reset()
        self.assertEqual(ve.n_updates, 0)
        self.assertEqual(ve.mean_td_error(), 0.0)

    def test_invalid_state_dim_raises(self):
        from physml.value_estimator import ValueEstimator

        with self.assertRaises(ValueError):
            ValueEstimator(state_dim=0)


# ---------------------------------------------------------------------------
# Stage 105 — ActionSelector
# ---------------------------------------------------------------------------
class TestActionSelector(unittest.TestCase):
    def test_import(self):
        from physml.action_selector import ActionSelector, SelectionResult

        self.assertTrue(True)

    def test_select_returns_selection_result(self):
        from physml.action_selector import ActionSelector, SelectionResult

        sel = ActionSelector(n_actions=4)
        result = sel.select()
        self.assertIsInstance(result, SelectionResult)

    def test_action_in_range(self):
        from physml.action_selector import ActionSelector

        sel = ActionSelector(n_actions=5)
        for _ in range(20):
            r = sel.select()
            self.assertIn(r.action, range(5))

    def test_epsilon_greedy_strategy(self):
        from physml.action_selector import ActionSelector

        sel = ActionSelector(n_actions=3, strategy="epsilon_greedy", epsilon=0.0)
        r = sel.select(logits=np.array([0.1, 5.0, 0.2]))
        self.assertEqual(r.action, 1)  # greedy picks max

    def test_softmax_probs_sum_to_one(self):
        from physml.action_selector import ActionSelector

        sel = ActionSelector(n_actions=4, strategy="softmax", random_state=0)
        r = sel.select(logits=np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertAlmostEqual(sum(r.action_probs), 1.0, places=5)

    def test_greedy_strategy(self):
        from physml.action_selector import ActionSelector

        sel = ActionSelector(n_actions=3, strategy="greedy")
        r = sel.select(logits=np.array([0.5, 0.1, 3.0]))
        self.assertEqual(r.action, 2)

    def test_ucb_strategy(self):
        from physml.action_selector import ActionSelector

        sel = ActionSelector(n_actions=3, strategy="ucb", random_state=0)
        r = sel.select()
        self.assertIn(r.action, range(3))
        self.assertEqual(r.strategy, "ucb")

    def test_step_counter_increments(self):
        from physml.action_selector import ActionSelector

        sel = ActionSelector(n_actions=3)
        for i in range(5):
            r = sel.select()
        self.assertEqual(r.step, 5)

    def test_action_counts_tracked(self):
        from physml.action_selector import ActionSelector

        sel = ActionSelector(n_actions=2, strategy="greedy")
        for _ in range(10):
            sel.select(logits=np.array([0.0, 1.0]))
        counts = sel.action_counts
        self.assertEqual(counts[1], 10)
        self.assertEqual(counts[0], 0)

    def test_update_value(self):
        from physml.action_selector import ActionSelector

        sel = ActionSelector(n_actions=3)
        sel.select()  # ensures count[0] >= 1
        sel.update_value(0, reward=1.0)
        # No assertion on exact value; just verify no error

    def test_reset_clears_history(self):
        from physml.action_selector import ActionSelector

        sel = ActionSelector(n_actions=3)
        for _ in range(5):
            sel.select()
        sel.reset()
        self.assertEqual(len(sel.selection_history), 0)
        self.assertEqual(sel._total_steps, 0)

    def test_invalid_strategy_raises(self):
        from physml.action_selector import ActionSelector

        with self.assertRaises(ValueError):
            ActionSelector(n_actions=3, strategy="unknown")

    def test_invalid_n_actions_raises(self):
        from physml.action_selector import ActionSelector

        with self.assertRaises(ValueError):
            ActionSelector(n_actions=0)

    def test_override_strategy_per_call(self):
        from physml.action_selector import ActionSelector

        sel = ActionSelector(n_actions=3, strategy="softmax")
        r = sel.select(logits=np.array([0.0, 5.0, 0.0]), strategy="greedy")
        self.assertEqual(r.action, 1)
        self.assertEqual(r.strategy, "greedy")


if __name__ == "__main__":
    unittest.main()
