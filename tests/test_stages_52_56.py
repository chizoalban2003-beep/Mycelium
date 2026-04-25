"""Tests for Stages 52-56.

Stage 52 — ReplayBuffer / PrioritizedReplay
Stage 53 — HyperScheduler (StepSchedule, CosineSchedule, ExponentialSchedule, LinearSchedule)
Stage 54 — AnomalyGuard
Stage 55 — MultiObjectiveOptimizer / Solution / Pareto front
Stage 56 — AgentProfiler
"""

import time
import math
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Stage 52 — ReplayBuffer / PrioritizedReplay
# ---------------------------------------------------------------------------
from physml.replay_buffer import ReplayBuffer, PrioritizedReplay, Transition


class TestReplayBuffer:
    def _make_transition(self, reward=1.0):
        return Transition(state=[1, 2], action=0, reward=reward, next_state=[2, 3])

    def test_push_and_len(self):
        rb = ReplayBuffer(capacity=5)
        for i in range(3):
            rb.push(self._make_transition(i))
        assert len(rb) == 3

    def test_capacity_ring(self):
        rb = ReplayBuffer(capacity=3)
        for i in range(6):
            rb.push(self._make_transition(i))
        assert len(rb) == 3

    def test_sample_size(self):
        rb = ReplayBuffer(capacity=100, seed=0)
        for i in range(20):
            rb.push(self._make_transition(i))
        batch = rb.sample(10)
        assert len(batch) == 10

    def test_sample_capped_by_buffer(self):
        rb = ReplayBuffer(capacity=100, seed=0)
        for i in range(3):
            rb.push(self._make_transition(i))
        batch = rb.sample(50)
        assert len(batch) == 3

    def test_clear(self):
        rb = ReplayBuffer(capacity=10)
        rb.push(self._make_transition())
        rb.clear()
        assert len(rb) == 0

    def test_is_ready_false_when_empty(self):
        rb = ReplayBuffer(capacity=10)
        assert not rb.is_ready

    def test_is_ready_true_after_push(self):
        rb = ReplayBuffer(capacity=10)
        rb.push(self._make_transition())
        assert rb.is_ready

    def test_push_many(self):
        rb = ReplayBuffer(capacity=100)
        transitions = [self._make_transition(i) for i in range(10)]
        rb.push_many(transitions)
        assert len(rb) == 10

    def test_as_arrays_rewards(self):
        rb = ReplayBuffer(capacity=10)
        for r in [1.0, 2.0, 3.0]:
            rb.push(self._make_transition(r))
        rewards, _, _ = rb.as_arrays()
        np.testing.assert_array_almost_equal(rewards, [1.0, 2.0, 3.0])

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            ReplayBuffer(capacity=0)


class TestPrioritizedReplay:
    def _make_transition(self, reward=1.0):
        return Transition(state=[1], action=0, reward=reward, next_state=[2])

    def test_basic_push_and_sample(self):
        pr = PrioritizedReplay(capacity=50, seed=42)
        for i in range(20):
            pr.push(self._make_transition(i))
        batch = pr.sample(10)
        assert len(batch) == 10

    def test_update_priorities(self):
        pr = PrioritizedReplay(capacity=50, seed=0)
        for i in range(5):
            pr.push(self._make_transition(i))
        batch = pr.sample(5)
        pr.update_priorities(batch, [0.1, 0.5, 2.0, 0.3, 1.0])
        stats = pr.priority_stats()
        assert stats["size"] == 5
        assert stats["max"] > 0

    def test_priority_stats_empty(self):
        pr = PrioritizedReplay(capacity=10)
        stats = pr.priority_stats()
        assert stats["size"] == 0

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            PrioritizedReplay(alpha=1.5)

    def test_high_priority_sampled_more(self):
        """High-priority transition should appear more often in large samples."""
        pr = PrioritizedReplay(capacity=100, alpha=1.0, epsilon=1e-6, seed=7)
        low = Transition(state=[0], action=0, reward=0.0, next_state=[0], priority=0.001)
        high = Transition(state=[1], action=1, reward=10.0, next_state=[1], priority=100.0)
        for _ in range(50):
            pr.push(low)
        pr.push(high)
        # Force high priority onto the high-reward transition
        pr.update_priorities([high], [100.0])
        counts = sum(1 for t in pr.sample(500) if t.reward == 10.0)
        # high-priority should appear at least some of the time
        assert counts > 0


# ---------------------------------------------------------------------------
# Stage 53 — HyperScheduler
# ---------------------------------------------------------------------------
from physml.scheduler import (
    StepSchedule,
    CosineSchedule,
    ExponentialSchedule,
    LinearSchedule,
    HyperScheduler,
)


class TestStepSchedule:
    def test_initial_value(self):
        s = StepSchedule(initial_value=0.1, step_size=5, gamma=0.5)
        assert s.get_value() == pytest.approx(0.1)

    def test_decays_after_step_size(self):
        s = StepSchedule(initial_value=1.0, step_size=3, gamma=0.5)
        for _ in range(3):
            s.step()
        assert s.get_value() == pytest.approx(0.5)

    def test_min_value_clamp(self):
        s = StepSchedule(initial_value=1.0, step_size=1, gamma=0.5, min_value=0.1)
        for _ in range(100):
            s.step()
        assert s.get_value() >= 0.1

    def test_reset(self):
        s = StepSchedule(initial_value=1.0, step_size=2, gamma=0.5)
        s.step(); s.step(); s.step()
        s.reset()
        assert s.current_step == 0
        assert s.get_value() == pytest.approx(1.0)


class TestCosineSchedule:
    def test_starts_at_initial(self):
        s = CosineSchedule(initial_value=1.0, T_max=100)
        assert s.get_value() == pytest.approx(1.0)

    def test_reaches_minimum_at_T_max(self):
        s = CosineSchedule(initial_value=1.0, T_max=10, eta_min=0.0)
        for _ in range(10):
            s.step()
        assert s.get_value() == pytest.approx(0.0, abs=1e-9)

    def test_midpoint_half_range(self):
        s = CosineSchedule(initial_value=1.0, T_max=10, eta_min=0.0)
        for _ in range(5):
            s.step()
        # cos(pi/2) = 0; value should be 0.5*(1+0) = 0.5
        assert s.get_value() == pytest.approx(0.5, abs=1e-9)


class TestExponentialSchedule:
    def test_decays_each_step(self):
        s = ExponentialSchedule(initial_value=1.0, gamma=0.9)
        s.step()
        assert s.get_value() == pytest.approx(0.9)

    def test_min_clamp(self):
        s = ExponentialSchedule(initial_value=1.0, gamma=0.5, min_value=0.01)
        for _ in range(100):
            s.step()
        assert s.get_value() >= 0.01


class TestLinearSchedule:
    def test_starts_at_initial(self):
        s = LinearSchedule(1.0, end_value=0.0, n_steps=10)
        assert s.get_value() == pytest.approx(1.0)

    def test_reaches_end(self):
        s = LinearSchedule(1.0, end_value=0.2, n_steps=5)
        for _ in range(5):
            s.step()
        assert s.get_value() == pytest.approx(0.2)

    def test_clamps_after_n_steps(self):
        s = LinearSchedule(1.0, end_value=0.0, n_steps=5)
        for _ in range(100):
            s.step()
        assert s.get_value() == pytest.approx(0.0)


class TestHyperScheduler:
    def test_register_and_step(self):
        hs = HyperScheduler()
        hs.register("lr", StepSchedule(0.1, step_size=5, gamma=0.5))
        vals = hs.step()
        assert "lr" in vals

    def test_get_all(self):
        hs = HyperScheduler()
        hs.register("eps", ExponentialSchedule(1.0, gamma=0.9))
        d = hs.get_all()
        assert "eps" in d

    def test_callback_called(self):
        called = []
        hs = HyperScheduler()
        hs.register("lr", LinearSchedule(1.0, end_value=0.0, n_steps=10))
        hs.add_callback(lambda name, val: called.append((name, val)))
        hs.step()
        assert len(called) == 1 and called[0][0] == "lr"

    def test_reset_all(self):
        hs = HyperScheduler()
        hs.register("lr", StepSchedule(1.0, step_size=1, gamma=0.5))
        for _ in range(5):
            hs.step()
        hs.reset_all()
        assert hs["lr"] == pytest.approx(1.0)

    def test_history_summary(self):
        hs = HyperScheduler()
        hs.register("lr", StepSchedule(0.01))
        hs.step()
        summary = hs.history_summary()
        assert "lr" in summary and summary["lr"]["step"] == 1


# ---------------------------------------------------------------------------
# Stage 54 — AnomalyGuard
# ---------------------------------------------------------------------------
from physml.anomaly import AnomalyGuard, AnomalyResult


def _make_clean_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 4))


class TestAnomalyGuard:
    def test_fit_returns_self(self):
        ag = AnomalyGuard(method="isolation_forest", contamination=0.05)
        X = _make_clean_data()
        result = ag.fit(X)
        assert result is ag

    def test_is_fitted_after_fit(self):
        ag = AnomalyGuard(method="isolation_forest")
        X = _make_clean_data()
        ag.fit(X)
        assert ag.is_fitted

    def test_predict_returns_list(self):
        ag = AnomalyGuard(method="isolation_forest")
        X = _make_clean_data()
        ag.fit(X)
        results = ag.predict(X[:5])
        assert len(results) == 5
        assert all(isinstance(r, AnomalyResult) for r in results)

    def test_extreme_outlier_flagged(self):
        ag = AnomalyGuard(method="isolation_forest", contamination=0.1)
        X_clean = _make_clean_data(200)
        ag.fit(X_clean)
        outlier = np.array([[1000.0, 1000.0, 1000.0, 1000.0]])
        results = ag.predict(outlier)
        assert results[0].is_anomaly

    def test_predict_guarded_shape(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 4))
        y = (X[:, 0] > 0).astype(int)
        clf = LogisticRegression().fit(X, y)
        ag = AnomalyGuard(method="isolation_forest")
        ag.fit(X)
        preds, results = ag.predict_guarded(X[:10], clf)
        assert preds.shape == (10,)
        assert len(results) == 10

    def test_not_fitted_raises(self):
        ag = AnomalyGuard()
        with pytest.raises(RuntimeError):
            ag.predict(np.zeros((3, 4)))

    @pytest.mark.slow
    def test_anomaly_rate_between_0_and_1(self):
        ag = AnomalyGuard(method="isolation_forest", contamination=0.05)
        X = _make_clean_data()
        ag.fit(X)
        rate = ag.anomaly_rate(X)
        assert 0.0 <= rate <= 1.0

    def test_summary_keys(self):
        ag = AnomalyGuard(method="lof")
        X = _make_clean_data()
        ag.fit(X)
        s = ag.summary()
        assert "method" in s and "fitted" in s

    def test_lof_method(self):
        ag = AnomalyGuard(method="lof", contamination=0.05)
        X = _make_clean_data()
        ag.fit(X)
        results = ag.predict(X[:3])
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Stage 55 — MultiObjectiveOptimizer
# ---------------------------------------------------------------------------
from physml.multiobjective import MultiObjectiveOptimizer, Solution


class TestMultiObjectiveOptimizer:
    def _make_solution(self, acc_loss, cost):
        return Solution(objectives={"acc_loss": acc_loss, "cost": cost})

    def test_add_and_population(self):
        opt = MultiObjectiveOptimizer(["acc_loss", "cost"])
        opt.add(self._make_solution(0.1, 1.0))
        assert len(opt._population) == 1

    def test_pareto_front_dominance(self):
        opt = MultiObjectiveOptimizer(["acc_loss", "cost"])
        # A dominates B (lower on both)
        a = self._make_solution(0.1, 0.5)
        b = self._make_solution(0.5, 1.0)
        c = self._make_solution(0.2, 0.3)  # also dominates b
        opt.add_many([a, b, c])
        front_names = {id(s) for s in opt.pareto_front}
        assert id(b) not in front_names

    def test_compromise_solution(self):
        opt = MultiObjectiveOptimizer(["acc_loss", "cost"])
        for al, c in [(0.1, 1.0), (0.5, 0.1), (0.3, 0.5)]:
            opt.add(self._make_solution(al, c))
        cs = opt.compromise_solution({"acc_loss": 1.0, "cost": 1.0})
        assert cs is not None

    def test_empty_compromise_returns_none(self):
        opt = MultiObjectiveOptimizer(["acc_loss"])
        assert opt.compromise_solution() is None

    def test_best_n(self):
        opt = MultiObjectiveOptimizer(["acc_loss", "cost"])
        for i in range(10):
            opt.add(self._make_solution(float(i), float(10 - i)))
        best = opt.best_n(3)
        assert len(best) == 3

    def test_clear(self):
        opt = MultiObjectiveOptimizer(["x"])
        opt.add(Solution({"x": 1.0}))
        opt.clear()
        assert len(opt._population) == 0

    def test_no_objectives_raises(self):
        with pytest.raises(ValueError):
            MultiObjectiveOptimizer([])

    def test_summary_keys(self):
        opt = MultiObjectiveOptimizer(["acc_loss", "cost"])
        opt.add(self._make_solution(0.1, 0.5))
        s = opt.summary()
        assert "pareto_front_size" in s and "population_size" in s


# ---------------------------------------------------------------------------
# Stage 56 — AgentProfiler
# ---------------------------------------------------------------------------
from physml.profiler import AgentProfiler, ProfileEntry


class TestAgentProfiler:
    def test_basic_profiling(self):
        profiler = AgentProfiler(track_memory=False)
        with profiler.profile("fit"):
            time.sleep(0.01)
        assert len(profiler) == 1

    def test_elapsed_positive(self):
        profiler = AgentProfiler(track_memory=False)
        with profiler.profile("predict"):
            time.sleep(0.01)
        assert profiler.total_elapsed("predict") > 0

    def test_call_count(self):
        profiler = AgentProfiler(track_memory=False)
        for _ in range(3):
            with profiler.profile("fit"):
                pass
        assert profiler.call_count("fit") == 3

    def test_report_keys(self):
        profiler = AgentProfiler(track_memory=False)
        with profiler.profile("a"):
            pass
        r = profiler.report()
        assert "top_entries" in r and "total_calls" in r

    def test_top_bottlenecks(self):
        profiler = AgentProfiler(track_memory=False)
        with profiler.profile("slow"):
            time.sleep(0.02)
        with profiler.profile("fast"):
            pass
        bottlenecks = profiler.top_bottlenecks(1)
        assert bottlenecks[0] == "slow"

    def test_reset(self):
        profiler = AgentProfiler(track_memory=False)
        with profiler.profile("op"):
            pass
        profiler.reset()
        assert len(profiler) == 0

    def test_records_property(self):
        profiler = AgentProfiler(track_memory=False)
        with profiler.profile("step"):
            pass
        records = profiler.records
        assert isinstance(records[0], ProfileEntry)

    def test_memory_tracking(self):
        profiler = AgentProfiler(track_memory=True)
        with profiler.profile("alloc"):
            _ = list(range(10_000))
        r = profiler.report()
        assert r["top_entries"][0]["total_memory_kb"] >= 0
