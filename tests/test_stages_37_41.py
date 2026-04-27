"""Tests for Stages 37–41: goal loop, memory attachment, self-eval,
self-improve, and introspection.

Stage 37 — MyceliumAgent.run_goal()
Stage 38 — MyceliumAgent.attach_memory() + auto episode recording in reward()
Stage 39 — MyceliumAgent.self_evaluate()
Stage 40 — MyceliumAgent.self_improve()
Stage 41 — MyceliumAgent.introspect()
"""

from __future__ import annotations

import numpy as np
import pytest

from physml.mycelium_agent import MyceliumAgent
from physml.memory import EpisodicMemory
from physml.featurizer import Featurizer
from physml.tools import Tool, ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_data(n: int = 60, n_features: int = 6):
    X = RNG.standard_normal((n, n_features)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def _fitted_agent(n: int = 60, n_features: int = 6) -> MyceliumAgent:
    X, y = _make_data(n=n, n_features=n_features)
    agent = MyceliumAgent(calibrate=False)
    agent.fit(X, y)
    return agent


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(name="echo", description="echo input", fn=lambda s: f"echo:{s}"))
    registry.register(Tool(name="upper", description="uppercase input", fn=str.upper))
    return registry


def _make_featurizer(output_dim: int = 6) -> Featurizer:
    texts = [
        "predict and classify", "analyse data features", "physics engine learning",
        "autonomous goal loop", "tool use episodic memory", "self evaluate improve",
        "entropy bandit policy", "multi-task reward stream",
    ]
    f = Featurizer(output_dim=output_dim)
    f.fit(texts)
    return f


# ===========================================================================
# Stage 37 — run_goal
# ===========================================================================


class TestStage37RunGoal:
    @pytest.mark.slow
    def test_run_goal_returns_dict(self):
        agent = _fitted_agent()
        registry = _make_registry()
        featurizer = _make_featurizer()
        result = agent.run_goal("Predict and analyse input data", registry, featurizer)
        assert isinstance(result, dict)

    @pytest.mark.slow
    def test_run_goal_has_expected_keys(self):
        agent = _fitted_agent()
        registry = _make_registry()
        featurizer = _make_featurizer()
        result = agent.run_goal("Learn and predict", registry, featurizer)
        for key in ("goal", "subtasks", "n_tool_calls", "n_episodes_stored", "result"):
            assert key in result, f"Missing key: {key}"

    @pytest.mark.slow
    def test_run_goal_goal_matches_input(self):
        agent = _fitted_agent()
        registry = _make_registry()
        featurizer = _make_featurizer()
        goal = "Classify this sample"
        result = agent.run_goal(goal, registry, featurizer)
        assert result["goal"] == goal

    @pytest.mark.slow
    def test_run_goal_subtasks_list(self):
        agent = _fitted_agent()
        registry = _make_registry()
        featurizer = _make_featurizer()
        result = agent.run_goal("Step one. Step two. Step three.", registry, featurizer, n_subtasks=3)
        assert isinstance(result["subtasks"], list)
        assert len(result["subtasks"]) == 3

    @pytest.mark.slow
    def test_run_goal_n_tool_calls_non_negative(self):
        agent = _fitted_agent()
        registry = _make_registry()
        featurizer = _make_featurizer()
        result = agent.run_goal("Predict and summarise", registry, featurizer)
        assert result["n_tool_calls"] >= 0

    @pytest.mark.slow
    def test_run_goal_records_episodes_with_memory(self):
        agent = _fitted_agent()
        registry = _make_registry()
        featurizer = _make_featurizer()
        memory = EpisodicMemory(capacity=100)
        result = agent.run_goal("Explore data", registry, featurizer, memory=memory, n_subtasks=2)
        assert result["n_episodes_stored"] == 2
        assert len(memory) == 2

    @pytest.mark.slow
    def test_run_goal_uses_attached_memory(self):
        agent = _fitted_agent()
        memory = EpisodicMemory(capacity=100)
        agent.attach_memory(memory)
        registry = _make_registry()
        featurizer = _make_featurizer()
        result = agent.run_goal("Adapt and predict", registry, featurizer, n_subtasks=2)
        assert result["n_episodes_stored"] == 2

    def test_run_goal_requires_fitted(self):
        agent = MyceliumAgent(calibrate=False)
        registry = _make_registry()
        featurizer = _make_featurizer()
        with pytest.raises(RuntimeError, match="not fitted"):
            agent.run_goal("goal", registry, featurizer)

    @pytest.mark.slow
    def test_run_goal_result_is_string(self):
        agent = _fitted_agent()
        registry = _make_registry()
        featurizer = _make_featurizer()
        result = agent.run_goal("Summarise features", registry, featurizer)
        assert isinstance(result["result"], str)


# ===========================================================================
# Stage 38 — attach_memory + auto episode recording
# ===========================================================================


class TestStage38AttachMemory:
    @pytest.mark.slow
    def test_attach_memory_returns_self(self):
        agent = _fitted_agent()
        memory = EpisodicMemory()
        returned = agent.attach_memory(memory)
        assert returned is agent

    @pytest.mark.slow
    def test_memory_attribute_set(self):
        agent = _fitted_agent()
        memory = EpisodicMemory()
        agent.attach_memory(memory)
        assert agent._memory is memory

    @pytest.mark.slow
    def test_reward_auto_stores_episode(self):
        X, y = _make_data(n=60)
        agent = MyceliumAgent(calibrate=False)
        agent.fit(X, y)
        memory = EpisodicMemory(capacity=200)
        agent.attach_memory(memory)
        # Call reward a few times
        for i in range(5):
            agent.reward(X[i : i + 1], y[i : i + 1])
        assert len(memory) > 0

    @pytest.mark.slow
    def test_reward_stores_up_to_capacity(self):
        X, y = _make_data(n=60)
        agent = MyceliumAgent(calibrate=False)
        agent.fit(X, y)
        memory = EpisodicMemory(capacity=3)
        agent.attach_memory(memory)
        for i in range(10):
            agent.reward(X[i % 60 : i % 60 + 1], y[i % 60 : i % 60 + 1])
        assert len(memory) == 3  # capacity is respected

    @pytest.mark.slow
    def test_no_memory_reward_still_works(self):
        X, y = _make_data(n=60)
        agent = MyceliumAgent(calibrate=False)
        agent.fit(X, y)
        # Without attaching memory — should not raise
        agent.reward(X[:1], y[:1])

    @pytest.mark.slow
    def test_augment_with_memory_manual(self):
        agent = _fitted_agent()
        memory = EpisodicMemory(n_neighbors=2)
        X, _ = _make_data(n=10)
        for i in range(5):
            memory.store(X[i], "predict", 1.0)
        X_aug = agent.augment_with_memory(X[:3], memory)
        assert X_aug.shape == (3, X.shape[1] + 4)  # 2 neighbors × 2 cols


# ===========================================================================
# Stage 39 — self_evaluate
# ===========================================================================


class TestStage39SelfEvaluate:
    @pytest.mark.slow
    def test_self_evaluate_returns_dict(self):
        X, y = _make_data(n=60)
        agent = MyceliumAgent(calibrate=False)
        agent.fit(X[:40], y[:40])
        metrics = agent.self_evaluate(X[40:], y[40:])
        assert isinstance(metrics, dict)

    @pytest.mark.slow
    def test_self_evaluate_keys(self):
        X, y = _make_data(n=60)
        agent = MyceliumAgent(calibrate=False)
        agent.fit(X[:40], y[:40])
        metrics = agent.self_evaluate(X[40:], y[40:])
        for key in ("accuracy", "mean_confidence", "ece", "n_samples", "oracle_cost", "threshold"):
            assert key in metrics, f"Missing key: {key}"

    @pytest.mark.slow
    def test_self_evaluate_accuracy_in_range(self):
        X, y = _make_data(n=60)
        agent = MyceliumAgent(calibrate=False)
        agent.fit(X[:40], y[:40])
        metrics = agent.self_evaluate(X[40:], y[40:])
        assert 0.0 <= metrics["accuracy"] <= 1.0

    @pytest.mark.slow
    def test_self_evaluate_mean_confidence_in_range(self):
        X, y = _make_data(n=60)
        agent = MyceliumAgent(calibrate=False)
        agent.fit(X[:40], y[:40])
        metrics = agent.self_evaluate(X[40:], y[40:])
        assert 0.0 <= metrics["mean_confidence"] <= 1.0

    @pytest.mark.slow
    def test_self_evaluate_ece_in_range(self):
        X, y = _make_data(n=60)
        agent = MyceliumAgent(calibrate=False)
        agent.fit(X[:40], y[:40])
        metrics = agent.self_evaluate(X[40:], y[40:])
        assert 0.0 <= metrics["ece"] <= 1.0

    @pytest.mark.slow
    def test_self_evaluate_n_samples_correct(self):
        X, y = _make_data(n=60)
        agent = MyceliumAgent(calibrate=False)
        agent.fit(X[:40], y[:40])
        metrics = agent.self_evaluate(X[40:], y[40:])
        assert metrics["n_samples"] == 20

    @pytest.mark.slow
    def test_self_evaluate_threshold_matches_agent(self):
        X, y = _make_data(n=60)
        agent = MyceliumAgent(calibrate=False, uncertainty_threshold=0.4)
        agent.fit(X[:40], y[:40])
        metrics = agent.self_evaluate(X[40:], y[40:])
        assert metrics["threshold"] == pytest.approx(0.4)

    def test_self_evaluate_requires_fitted(self):
        agent = MyceliumAgent(calibrate=False)
        X, y = _make_data(n=20)
        with pytest.raises(RuntimeError, match="not fitted"):
            agent.self_evaluate(X, y)


# ===========================================================================
# Stage 40 — self_improve
# ===========================================================================


class TestStage40SelfImprove:
    @pytest.mark.slow
    def test_self_improve_returns_dict(self):
        X, y = _make_data(n=60)
        agent = MyceliumAgent(calibrate=False)
        agent.fit(X[:40], y[:40])
        result = agent.self_improve(X[40:], y[40:])
        assert isinstance(result, dict)

    @pytest.mark.slow
    def test_self_improve_has_threshold_keys(self):
        X, y = _make_data(n=60)
        agent = MyceliumAgent(calibrate=False)
        agent.fit(X[:40], y[:40])
        result = agent.self_improve(X[40:], y[40:])
        assert "threshold_before" in result
        assert "threshold_after" in result

    @pytest.mark.slow
    def test_self_improve_threshold_changes_on_low_accuracy(self):
        """Force low accuracy by giving wrong labels; threshold should decrease."""
        X, y = _make_data(n=60)
        y_wrong = 1 - y  # invert labels to force low accuracy
        agent = MyceliumAgent(calibrate=False, uncertainty_threshold=0.35)
        agent.fit(X[:40], y[:40])
        result = agent.self_improve(X[40:], y_wrong[40:])
        # With inverted labels, accuracy should be low → threshold decreases
        assert result["threshold_after"] <= result["threshold_before"]

    @pytest.mark.slow
    def test_self_improve_threshold_in_valid_range(self):
        X, y = _make_data(n=60)
        agent = MyceliumAgent(calibrate=False)
        agent.fit(X[:40], y[:40])
        result = agent.self_improve(X[40:], y[40:])
        assert 0.0 < result["threshold_after"] < 1.0

    @pytest.mark.slow
    def test_self_improve_aggressive_no_error(self):
        X, y = _make_data(n=60)
        agent = MyceliumAgent(calibrate=False)
        agent.fit(X[:40], y[:40])
        result = agent.self_improve(X[40:], y[40:], aggressive=True)
        assert "threshold_after" in result

    @pytest.mark.slow
    def test_self_improve_updates_agent_threshold(self):
        X, y = _make_data(n=60)
        agent = MyceliumAgent(calibrate=False, uncertainty_threshold=0.35)
        agent.fit(X[:40], y[:40])
        result = agent.self_improve(X[40:], y[40:])
        assert agent.uncertainty_threshold == pytest.approx(result["threshold_after"])


# ===========================================================================
# Stage 41 — introspect
# ===========================================================================


class TestStage41Introspect:
    def test_introspect_unfitted_returns_dict(self):
        agent = MyceliumAgent(calibrate=False)
        info = agent.introspect()
        assert isinstance(info, dict)
        assert info["fitted"] is False

    @pytest.mark.slow
    def test_introspect_fitted_returns_dict(self):
        agent = _fitted_agent()
        info = agent.introspect()
        assert isinstance(info, dict)
        assert info["fitted"] is True

    @pytest.mark.slow
    def test_introspect_keys(self):
        agent = _fitted_agent()
        info = agent.introspect()
        expected = [
            "fitted",
            "predictor_type",
            "predictor_runtime_state",
            "uncertainty_threshold",
            "policy",
            "query_strategy",
            "calibration_temperature",
            "drift_detection_enabled",
            "drift_detected",
            "n_memory_episodes",
            "agent_activity",
        ]
        for key in expected:
            assert key in info, f"Missing key: {key}"

    @pytest.mark.slow
    def test_introspect_predictor_type_is_string(self):
        agent = _fitted_agent()
        info = agent.introspect()
        assert isinstance(info["predictor_type"], str)
        assert len(info["predictor_type"]) > 0

    @pytest.mark.slow
    def test_introspect_no_memory_zero_episodes(self):
        agent = _fitted_agent()
        info = agent.introspect()
        assert info["n_memory_episodes"] == 0

    @pytest.mark.slow
    def test_introspect_with_memory_shows_episode_count(self):
        X, y = _make_data(n=60)
        agent = MyceliumAgent(calibrate=False)
        agent.fit(X[:40], y[:40])
        memory = EpisodicMemory(capacity=100)
        agent.attach_memory(memory)
        for i in range(5):
            agent.reward(X[i : i + 1], y[i : i + 1])
        info = agent.introspect()
        assert info["n_memory_episodes"] > 0

    @pytest.mark.slow
    def test_introspect_threshold_matches_agent(self):
        agent = MyceliumAgent(calibrate=False, uncertainty_threshold=0.42)
        X, y = _make_data(n=60)
        agent.fit(X, y)
        info = agent.introspect()
        assert info["uncertainty_threshold"] == pytest.approx(0.42)

    @pytest.mark.slow
    def test_introspect_drift_detection_flag(self):
        X, y = _make_data(n=60)
        agent = MyceliumAgent(calibrate=False, drift_detection=True)
        agent.fit(X, y)
        info = agent.introspect()
        assert info["drift_detection_enabled"] is True

    @pytest.mark.slow
    def test_introspect_predictor_type_cep(self):
        agent = _fitted_agent()
        info = agent.introspect()
        assert "CompetitiveEnsemblePredictor" in info["predictor_type"]
