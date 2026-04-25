"""Tests for Stages 62–68: WorldModel, IntrinsicMotivation, CompetitiveArena,
GoalConditionedPolicy, SafetyMonitor, AutonomousAgent, CompetitiveReport.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Stage 62 — WorldModel
# ---------------------------------------------------------------------------
from physml.world_model import WorldModel


class TestWorldModel:
    def _populate(self, wm: WorldModel, n: int = 20) -> WorldModel:
        rng = np.random.default_rng(0)
        for _ in range(n):
            s = rng.standard_normal(4)
            a = int(rng.integers(0, wm.n_actions))
            s_next = s + rng.standard_normal(4) * 0.1
            r = float(rng.uniform())
            wm.record(s, a, s_next, r)
        wm.update(min_samples=5)
        return wm

    def test_init(self):
        wm = WorldModel(horizon=2, n_actions=3, discount=0.9)
        assert not wm.fitted_
        assert wm.n_actions == 3

    def test_record_and_update(self):
        wm = WorldModel(n_actions=2)
        self._populate(wm)
        assert wm.fitted_
        assert len(wm._states) == 20

    def test_plan_returns_valid_action(self):
        wm = WorldModel(n_actions=2, horizon=2)
        self._populate(wm)
        state = np.zeros(4)
        action = wm.plan(state)
        assert action in (0, 1)

    def test_plan_unfitted_returns_zero(self):
        wm = WorldModel(n_actions=3)
        assert wm.plan(np.zeros(4)) == 0

    def test_plan_with_actions_subset(self):
        wm = WorldModel(n_actions=3)
        self._populate(wm, 30)
        action = wm.plan(np.zeros(4), actions=[1, 2])
        assert action in (1, 2)

    def test_summary(self):
        wm = WorldModel()
        s = wm.summary()
        assert "fitted" in s
        assert "n_transitions" in s

    def test_too_few_samples_not_fitted(self):
        wm = WorldModel(n_actions=2)
        wm.record(np.zeros(4), 0, np.ones(4), 1.0)
        wm.update(min_samples=10)
        assert not wm.fitted_


# ---------------------------------------------------------------------------
# Stage 63 — IntrinsicMotivation
# ---------------------------------------------------------------------------
from physml.intrinsic import IntrinsicMotivation


class TestIntrinsicMotivation:
    def test_bonus_positive(self):
        im = IntrinsicMotivation()
        b = im.bonus(np.zeros(4), np.ones(4))
        assert b >= 0.0

    def test_bonus_accumulates(self):
        im = IntrinsicMotivation(bonus_scale=0.5, count_scale=0.1)
        for i in range(10):
            im.bonus(np.random.randn(4), np.random.randn(4))
        assert im.total_bonus_ > 0
        assert im.step_ == 10

    def test_novelty_decreases_with_visits(self):
        im = IntrinsicMotivation(count_scale=1.0)
        s = np.array([1.0, 2.0, 3.0])
        n1 = im.novelty(s)
        im.bonus(s, s + 0.01)
        n2 = im.novelty(s)
        assert n2 <= n1

    def test_no_model_update(self):
        im = IntrinsicMotivation()
        b = im.bonus(np.zeros(4), np.ones(4), update_model=False)
        assert b >= 0.0
        assert not im._fitted

    def test_summary(self):
        im = IntrinsicMotivation()
        im.bonus(np.zeros(3), np.ones(3))
        s = im.summary()
        assert s["steps"] == 1
        assert "unique_states_visited" in s


# ---------------------------------------------------------------------------
# Stage 64 — CompetitiveArena
# ---------------------------------------------------------------------------
from physml.arena import CompetitiveArena, ArenaResult


class TestCompetitiveArena:
    def _data(self):
        X, y = make_classification(n_samples=200, n_features=8, random_state=0)
        return train_test_split(X, y, test_size=0.3, random_state=0)

    def test_single_competitor(self):
        Xtr, Xte, ytr, yte = self._data()
        arena = CompetitiveArena()
        arena.register("lr", LogisticRegression(max_iter=200))
        results = arena.run(Xtr, ytr, Xte, yte)
        assert len(results) == 1
        assert isinstance(results[0], ArenaResult)
        assert results[0].rank == 1

    def test_ranking(self):
        Xtr, Xte, ytr, yte = self._data()
        arena = CompetitiveArena(metric="accuracy")
        arena.register("lr", LogisticRegression(max_iter=200))
        arena.register("lr2", LogisticRegression(max_iter=100, C=0.01))
        results = arena.run(Xtr, ytr, Xte, yte)
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[0].accuracy >= results[1].accuracy

    def test_arena_result_has_all_fields(self):
        Xtr, Xte, ytr, yte = self._data()
        arena = CompetitiveArena()
        arena.register("lr", LogisticRegression(max_iter=200))
        r = arena.run(Xtr, ytr, Xte, yte)[0]
        assert r.fit_time_s >= 0
        assert r.predict_time_s >= 0
        assert 0.0 <= r.accuracy <= 1.0

    def test_leaderboard_returns_dicts(self):
        Xtr, Xte, ytr, yte = self._data()
        arena = CompetitiveArena()
        arena.register("lr", LogisticRegression(max_iter=200))
        lb = arena.leaderboard(Xtr, ytr, Xte, yte)
        assert isinstance(lb, list)
        assert isinstance(lb[0], dict)

    def test_multiple_competitors_ordering(self):
        Xtr, Xte, ytr, yte = self._data()
        arena = CompetitiveArena()
        for name in ["a", "b", "c"]:
            arena.register(name, LogisticRegression(max_iter=200, random_state=0))
        results = arena.run(Xtr, ytr, Xte, yte)
        assert len(results) == 3
        # Ranks should be ascending
        ranks = [r.rank for r in results]
        assert ranks == sorted(ranks)


# ---------------------------------------------------------------------------
# Stage 65 — GoalConditionedPolicy
# ---------------------------------------------------------------------------
from physml.goal_policy import GoalSpec, GoalConditionedPolicy


class TestGoalSpec:
    def test_achieved_true(self):
        g = GoalSpec("maximise accuracy", threshold=0.8)
        assert g.achieved({"accuracy": 0.9})

    def test_achieved_false(self):
        g = GoalSpec("maximise accuracy", threshold=0.8)
        assert not g.achieved({"accuracy": 0.7})

    def test_default_metric(self):
        g = GoalSpec("do something")
        assert g.target_metric == "accuracy"


class TestGoalConditionedPolicy:
    def test_act_unfitted_returns_zero(self):
        p = GoalConditionedPolicy(n_actions=3)
        action = p.act(np.zeros(5), "maximise accuracy")
        assert action == 0

    def test_update_and_act(self):
        p = GoalConditionedPolicy(n_actions=2)
        s = np.random.randn(5)
        g = GoalSpec("improve f1")
        for _ in range(20):
            p.update(s, g, 1)
        action = p.act(s, g)
        assert action in (0, 1)

    def test_encode_goal_length(self):
        p = GoalConditionedPolicy(embedding_dim=8)
        vec = p.encode_goal("maximize accuracy and reward")
        assert vec.shape == (8,)

    def test_encode_goal_spec(self):
        p = GoalConditionedPolicy(embedding_dim=16)
        g = GoalSpec("win", threshold=0.9)
        vec = p.encode_goal(g)
        assert vec.shape == (16,)

    def test_action_scores_unfitted(self):
        p = GoalConditionedPolicy(n_actions=3)
        scores = p.action_scores(np.zeros(4), "goal")
        assert scores.shape == (3,)

    def test_summary(self):
        p = GoalConditionedPolicy()
        s = p.summary()
        assert "fitted" in s


# ---------------------------------------------------------------------------
# Stage 66 — SafetyMonitor
# ---------------------------------------------------------------------------
from physml.safety import SafetyConstraint, SafetyViolation, SafetyMonitor


class TestSafetyMonitor:
    def test_safe_action_passes(self):
        sm = SafetyMonitor(safe_action=0)
        sm.add_constraint(
            SafetyConstraint("always_ok", predicate=lambda s, a: True)
        )
        result = sm.screen(np.zeros(4), candidate_action=1)
        assert result == 1

    def test_unsafe_falls_back(self):
        sm = SafetyMonitor(safe_action=0)
        sm.add_constraint(
            SafetyConstraint("block_1", predicate=lambda s, a: a != 1)
        )
        result = sm.screen(np.zeros(4), candidate_action=1)
        assert result == 0

    def test_alternative_action_selected(self):
        sm = SafetyMonitor(safe_action=0)
        sm.add_constraint(
            SafetyConstraint("block_1", predicate=lambda s, a: a != 1)
        )
        result = sm.screen(np.zeros(4), candidate_action=1, alternatives=[2, 3])
        assert result in (2, 3)

    def test_violation_logged(self):
        sm = SafetyMonitor()
        sm.add_constraint(
            SafetyConstraint("block_all", predicate=lambda s, a: False)
        )
        sm.screen(np.zeros(4), 1)
        assert sm.n_violations_ == 1
        assert len(sm.violations_) == 1

    def test_max_violations_raises(self):
        sm = SafetyMonitor(max_violations=2)
        sm.add_constraint(
            SafetyConstraint("block_all", predicate=lambda s, a: False)
        )
        sm.screen(np.zeros(4), 1)
        sm.screen(np.zeros(4), 1)
        with pytest.raises(RuntimeError, match="max_violations"):
            sm.screen(np.zeros(4), 1)

    def test_add_bound_constraint(self):
        sm = SafetyMonitor(safe_action=0)
        sm.add_bound_constraint("feature0_bound", 0, low=-1.0, high=1.0)
        # safe
        assert sm.screen(np.array([0.5, 0.0]), 1) == 1
        # unsafe (feature 0 = 2.0 out of bounds)
        result = sm.screen(np.array([2.0, 0.0]), 1)
        assert result == 0

    def test_penalty_for_violation(self):
        sm = SafetyMonitor()
        sm.add_constraint(
            SafetyConstraint("viol", predicate=lambda s, a: False, penalty=3.0)
        )
        p = sm.penalty_for(np.zeros(4), 0)
        assert p == pytest.approx(3.0)

    def test_is_safe(self):
        sm = SafetyMonitor()
        sm.add_constraint(
            SafetyConstraint("ok", predicate=lambda s, a: True)
        )
        assert sm.is_safe(np.zeros(4), 1)

    def test_report(self):
        sm = SafetyMonitor()
        r = sm.report()
        assert "n_violations" in r
        assert "violation_rate" in r


# ---------------------------------------------------------------------------
# Stage 67 — AutonomousAgent
# ---------------------------------------------------------------------------
from physml.autonomous_agent import AutonomousAgent


class TestAutonomousAgent:
    def _build_agent(self):
        core = LogisticRegression(max_iter=200)
        return AutonomousAgent(core, n_actions=2, horizon=2, bonus_scale=0.1)

    def _data(self):
        X, y = make_classification(n_samples=200, n_features=6, random_state=0)
        return train_test_split(X, y, test_size=0.3, random_state=0)

    def test_fit_and_predict(self):
        Xtr, Xte, ytr, yte = self._data()
        agent = self._build_agent()
        agent.fit(Xtr, ytr)
        preds = agent.predict(Xte)
        assert len(preds) == len(yte)

    def test_predict_proba(self):
        Xtr, Xte, ytr, yte = self._data()
        agent = self._build_agent()
        agent.fit(Xtr, ytr)
        proba = agent.predict_proba(Xte)
        assert proba.shape == (len(yte), 2)

    def test_act_unfitted_world_model(self):
        agent = self._build_agent()
        action = agent.act(np.zeros(6))
        assert action in (0, 1)

    def test_act_with_goal(self):
        agent = self._build_agent()
        g = GoalSpec("maximise accuracy", threshold=0.9)
        for i in range(20):
            agent.goal_policy.update(np.random.randn(6), g, i % 2)
        action = agent.act(np.random.randn(6), goal=g)
        assert action in (0, 1)

    def test_step_returns_shaped_reward(self):
        agent = self._build_agent()
        s = np.random.randn(4)
        s_next = s + 0.1
        total = agent.step(s, s_next, 0, 1.0)
        assert isinstance(total, float)

    def test_step_accumulates(self):
        agent = self._build_agent()
        for _ in range(5):
            s = np.random.randn(4)
            agent.step(s, s + 0.1, 0, 1.0)
        assert agent._step == 5

    def test_safety_screening_in_act(self):
        core = LogisticRegression(max_iter=200)
        agent = AutonomousAgent(core, n_actions=2, safe_action=0)
        agent.safety.add_constraint(
            SafetyConstraint("block_1", predicate=lambda s, a: a != 1)
        )
        # World model not fitted, goal policy not fitted → fallback 0
        action = agent.act(np.zeros(4))
        assert action == 0

    def test_compete(self):
        Xtr, Xte, ytr, yte = self._data()
        agent = self._build_agent()
        agent.fit(Xtr, ytr)
        results = agent.compete(
            Xtr, ytr, Xte, yte,
            baselines={"LR_baseline": LogisticRegression(max_iter=200)},
        )
        assert len(results) == 2
        assert results[0].rank == 1

    def test_status(self):
        agent = self._build_agent()
        st = agent.status()
        assert "steps" in st
        assert "world_model" in st
        assert "curiosity" in st
        assert "safety" in st


# ---------------------------------------------------------------------------
# Stage 68 — CompetitiveReport
# ---------------------------------------------------------------------------
from physml.competitive_report import CompetitiveReport


class TestCompetitiveReport:
    def _agent(self):
        return LogisticRegression(max_iter=200)

    def test_run_returns_report(self):
        reporter = CompetitiveReport(n_samples=300, n_features=6)
        report = reporter.run(self._agent())
        assert "leaderboard" in report
        assert "summary" in report
        assert "verdict" in report

    def test_mycelium_ranked(self):
        reporter = CompetitiveReport(n_samples=300, n_features=6)
        report = reporter.run(self._agent())
        assert report["summary"]["mycelium_rank"] >= 1

    def test_competitive_flag(self):
        reporter = CompetitiveReport(n_samples=300, n_features=6)
        report = reporter.run(self._agent())
        assert isinstance(report["summary"]["is_competitive"], bool)

    @pytest.mark.slow
    def test_custom_dataset(self):
        X, y = make_classification(n_samples=200, n_features=5, random_state=1)
        reporter = CompetitiveReport()
        report = reporter.run(self._agent(), X=X, y=y, dataset_name="custom")
        assert report["dataset"] == "custom"
        assert report["n_features"] == 5

    def test_extra_baselines(self):
        from sklearn.ensemble import RandomForestClassifier
        reporter = CompetitiveReport(n_samples=300, n_features=6)
        report = reporter.run(
            self._agent(),
            extra_baselines={"RF": RandomForestClassifier(n_estimators=10)},
        )
        names = [r["name"] for r in report["leaderboard"]]
        assert "RF" in names

    def test_leaderboard_sorted_by_accuracy(self):
        reporter = CompetitiveReport(n_samples=300, n_features=6)
        report = reporter.run(self._agent())
        accs = [r["accuracy"] for r in report["leaderboard"]]
        assert accs == sorted(accs, reverse=True)

    def test_print_report_runs(self, capsys):
        reporter = CompetitiveReport(n_samples=200, n_features=5)
        report = reporter.run(self._agent())
        reporter.print_report(report)
        captured = capsys.readouterr()
        assert "COMPETITIVE" in captured.out

    def test_autonomous_agent_in_report(self):
        """AutonomousAgent (with LR core) should appear in the report."""
        Xtr, Xte, ytr, yte = train_test_split(
            *make_classification(n_samples=300, n_features=6, random_state=0),
            test_size=0.3,
            random_state=0,
        )
        core = LogisticRegression(max_iter=200)
        agent = AutonomousAgent(core, n_actions=2)
        agent.fit(Xtr, ytr)
        reporter = CompetitiveReport(n_samples=300, n_features=6)
        report = reporter.run(agent, X=np.vstack([Xtr, Xte]), y=np.concatenate([ytr, yte]))
        assert any("MyceliumAgent" in r["name"] for r in report["leaderboard"])
