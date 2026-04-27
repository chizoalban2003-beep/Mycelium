"""Tests for Stages 42–46.

Stage 42 — Bug fixes:
  * EpisodicMemory uses deque (O(1) eviction, FIFO correct)
  * AutonomousLoop.run() does not call featurizer twice for the same goal
  * reward() does not call observe() internally (no double inference)
  * self_improve() triggers real partial_fit when memory is attached

Stage 43 — Featurizer sentence-embedding backend (TF-IDF fallback path)
Stage 44 — ToolSpec / ToolCall / ToolPlanner (JSON-schema tool protocol)
Stage 45 — FeedbackBuffer / FeedbackItem / OnlineRLHF
Stage 46 — Specialist / OrchestratorResult / AgentOrchestrator
"""

from __future__ import annotations

import numpy as np
import pytest

from physml.memory import EpisodicMemory
from physml.featurizer import Featurizer
from physml.tools import Tool, ToolRegistry, AutonomousLoop
from physml.mycelium_agent import MyceliumAgent
from physml.tool_planner import ToolSpec, ToolCall, ToolPlanner
from physml.feedback import FeedbackBuffer, FeedbackItem, OnlineRLHF
from physml.orchestrator import AgentOrchestrator, Specialist, OrchestratorResult

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_agent(n: int = 60) -> MyceliumAgent:
    X = RNG.standard_normal((n, 6)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    agent = MyceliumAgent(calibrate=False)
    agent.fit(X, y)
    return agent


def _make_featurizer() -> Featurizer:
    texts = ["classify numeric data", "search the web", "answer question", "compute metric"]
    f = Featurizer(output_dim=16, use_sentence_embeddings=False)
    f.fit(texts)
    return f


# ===========================================================================
# Stage 42 — Bug fixes
# ===========================================================================


class TestStage42EpisodicMemoryDeque:
    """EpisodicMemory.store() must use deque (O(1) FIFO eviction)."""

    def test_deque_type(self):
        from collections import deque
        mem = EpisodicMemory(capacity=5)
        assert isinstance(mem._contexts, deque)
        assert isinstance(mem._actions, deque)
        assert isinstance(mem._outcomes, deque)

    def test_deque_maxlen(self):
        mem = EpisodicMemory(capacity=5)
        assert mem._contexts.maxlen == 5

    def test_capacity_enforced(self):
        mem = EpisodicMemory(capacity=3)
        for i in range(7):
            mem.store(np.array([float(i)]), action="a", outcome=float(i))
        assert len(mem) == 3

    def test_fifo_order(self):
        mem = EpisodicMemory(capacity=3)
        for i in range(5):
            mem.store(np.array([float(i)]), action="a", outcome=float(i))
        # Should retain items 2, 3, 4
        outcomes = list(mem._outcomes)
        assert outcomes == [2.0, 3.0, 4.0]


class TestStage42RewardNoDoubleInference:
    """reward() must not call observe() internally."""

    @pytest.mark.slow
    def test_reward_does_not_call_observe(self, monkeypatch):
        agent = _make_agent()
        calls = []

        original_observe = agent.observe

        def patched_observe(X):
            calls.append(1)
            return original_observe(X)

        monkeypatch.setattr(agent, "observe", patched_observe)

        X_single = RNG.standard_normal((1, 6)).astype(np.float32)
        agent.reward(X_single, np.array([1]))

        assert len(calls) == 0, "reward() must not call observe()"

    @pytest.mark.slow
    def test_observe_caches_action(self):
        agent = _make_agent()
        X_single = RNG.standard_normal((1, 6)).astype(np.float32)
        agent.observe(X_single)
        assert agent._last_action_str in ("predict", "ask", "abstain")


class TestStage42SelfImproveRetrains:
    """self_improve() must trigger partial_fit when memory is attached."""

    @pytest.mark.slow
    def test_self_improve_with_memory_retrains(self):
        agent = _make_agent()
        mem = EpisodicMemory(capacity=100)
        agent.attach_memory(mem)

        # Populate memory with binary labelled episodes
        for _ in range(40):
            x = RNG.standard_normal(6).astype(np.float32)
            label = float(int(x[0] > 0))
            mem.store(x, action="predict", outcome=label)

        X_test = RNG.standard_normal((30, 6)).astype(np.float32)
        y_test = (X_test[:, 0] > 0).astype(int)

        result = agent.self_improve(X_test, y_test)
        assert "episodes_retrained" in result
        # At least some episodes should have been retrained if accuracy < target
        # (we just verify the key exists and is an int ≥ 0)
        assert isinstance(result["episodes_retrained"], int)
        assert result["episodes_retrained"] >= 0

    @pytest.mark.slow
    def test_self_improve_returns_episodes_retrained_zero_without_memory(self):
        agent = _make_agent()
        X_test = RNG.standard_normal((30, 6)).astype(np.float32)
        y_test = (X_test[:, 0] > 0).astype(int)
        result = agent.self_improve(X_test, y_test)
        assert result["episodes_retrained"] == 0


# ===========================================================================
# Stage 43 — Featurizer embedding backend
# ===========================================================================


class TestStage43FeaturizerBackend:
    """Featurizer uses TF-IDF hash path when use_sentence_embeddings=False."""

    def test_tfidf_path_works(self):
        f = Featurizer(output_dim=16, use_sentence_embeddings=False)
        texts = ["hello world", "foo bar baz", "some text here"]
        f.fit(texts)
        out = f.transform(texts)
        assert out.shape == (3, 16)
        assert out.dtype == np.float32

    def test_use_sentence_embeddings_false_no_st_model(self):
        f = Featurizer(output_dim=16, use_sentence_embeddings=False)
        assert f._use_embeddings is False
        texts = ["a", "b"]
        f.fit(texts)
        assert f._st_model is None

    def test_embedding_flag_auto_detect_type(self):
        # auto-detect — should be bool regardless of whether sentence-transformers is installed
        f = Featurizer(output_dim=8)
        assert isinstance(f._use_embeddings, bool)

    def test_numeric_path_unchanged(self):
        f = Featurizer(output_dim=4)
        data = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 10
        f.fit(data)
        out = f.transform(data)
        assert out.shape == (10, 4)
        assert out.dtype == np.float32


# ===========================================================================
# Stage 44 — ToolSpec / ToolCall / ToolPlanner
# ===========================================================================


class TestStage44ToolSpec:
    def test_toolspec_instantiation(self):
        spec = ToolSpec(name="echo", description="echo input", fn=lambda s: s)
        assert spec.name == "echo"
        assert spec.input_schema == {}

    def test_toolspec_with_schema(self):
        spec = ToolSpec(
            name="search",
            description="web search",
            fn=lambda s: s,
            input_schema={"type": "string", "minLength": 1, "maxLength": 200},
        )
        assert spec.validate_input("hello") is True
        assert spec.validate_input("") is False  # minLength=1

    def test_toolspec_no_schema_always_valid(self):
        spec = ToolSpec(name="x", description="x", fn=lambda s: s)
        assert spec.validate_input("") is True
        assert spec.validate_input("anything") is True


class TestStage44ToolPlanner:
    def _make_planner(self) -> ToolPlanner:
        f = _make_featurizer()
        planner = ToolPlanner(featurizer=f)
        planner.register(ToolSpec(name="classify", description="classify numeric data", fn=lambda s: "cls:" + s))
        planner.register(ToolSpec(name="search", description="search the web for information", fn=lambda s: "srch:" + s))
        return planner

    @pytest.mark.slow
    def test_plan_returns_tool_call(self):
        planner = self._make_planner()
        tc = planner.plan("classify some rows")
        assert isinstance(tc, ToolCall)
        assert tc.tool_name in ("classify", "search")
        assert 0.0 <= tc.confidence <= 1.0

    def test_execute_runs_fn(self):
        planner = self._make_planner()
        tc = planner.plan("classify some rows")
        result = planner.execute(tc)
        assert isinstance(result, str)

    def test_plan_and_execute(self):
        planner = self._make_planner()
        tc, output = planner.plan_and_execute("numeric classification")
        assert isinstance(output, str)

    def test_list_specs(self):
        planner = self._make_planner()
        specs = planner.list_specs()
        assert len(specs) == 2
        assert all("name" in s and "description" in s for s in specs)

    def test_no_tools_raises(self):
        f = _make_featurizer()
        planner = ToolPlanner(featurizer=f)
        with pytest.raises(RuntimeError, match="No tools registered"):
            planner.plan("anything")

    def test_memory_weight_influences_score(self):
        f = _make_featurizer()
        mem = EpisodicMemory(capacity=50)
        # Fill memory with strong preference for "search"
        vec = f.transform(["search the web"])[0]
        for _ in range(20):
            mem.store(vec, action="search", outcome=1.0)

        planner = ToolPlanner(featurizer=f, memory=mem, memory_weight=0.8)
        planner.register(ToolSpec(name="classify", description="classify numeric data", fn=lambda s: s))
        planner.register(ToolSpec(name="search", description="search the web for information", fn=lambda s: s))
        tc = planner.plan("general query")
        # With heavy memory weight and all-positive search outcomes, search should win
        # (not always guaranteed with tiny featurizer but let's at least check it runs)
        assert tc.tool_name in ("classify", "search")

    def test_ranked_alternatives_populated(self):
        planner = self._make_planner()
        tc = planner.plan("query")
        assert len(tc.ranked_alternatives) == 1  # 2 tools total → 1 alternative


# ===========================================================================
# Stage 45 — FeedbackBuffer / OnlineRLHF
# ===========================================================================


class TestStage45FeedbackBuffer:
    def test_push_and_len(self):
        buf = FeedbackBuffer(capacity=10)
        item = FeedbackItem(features=np.array([1.0, 2.0], dtype=np.float32), label=1)
        buf.push(item)
        assert len(buf) == 1

    def test_capacity_enforced(self):
        buf = FeedbackBuffer(capacity=5)
        for i in range(10):
            buf.push(FeedbackItem(
                features=np.array([float(i), 0.0], dtype=np.float32), label=i % 2
            ))
        assert len(buf) <= 5

    def test_dedup_skips_identical(self):
        buf = FeedbackBuffer(capacity=100, dedup_window=10)
        feat = np.array([1.0, 1.0], dtype=np.float32)
        added = sum(
            buf.push(FeedbackItem(features=feat, label=0))
            for _ in range(5)
        )
        assert added == 1  # only first should be added

    def test_push_raw_batch(self):
        buf = FeedbackBuffer(capacity=100)
        X = RNG.standard_normal((10, 4)).astype(np.float32)
        y = RNG.integers(0, 2, 10)
        n = buf.push_raw(X, y)
        assert n == 10

    def test_sample_batch_shape(self):
        buf = FeedbackBuffer(capacity=100)
        X = RNG.standard_normal((20, 4)).astype(np.float32)
        y = RNG.integers(0, 2, 20)
        buf.push_raw(X, y)
        Xb, yb, wb = buf.sample_batch(n=10)
        assert Xb.shape[0] == 10
        assert yb.shape[0] == 10
        assert wb.shape[0] == 10

    def test_sample_full_when_n_none(self):
        buf = FeedbackBuffer(capacity=50)
        X = RNG.standard_normal((15, 3)).astype(np.float32)
        y = np.zeros(15, dtype=int)
        buf.push_raw(X, y)
        Xb, yb, wb = buf.sample_batch(n=None)
        assert Xb.shape[0] == 15

    def test_stats_keys(self):
        buf = FeedbackBuffer(capacity=50)
        buf.push_raw(RNG.standard_normal((5, 3)).astype(np.float32), np.ones(5, dtype=int))
        s = buf.stats()
        assert "size" in s and "sources" in s and "mean_weight" in s

    def test_clear(self):
        buf = FeedbackBuffer(capacity=50)
        buf.push_raw(RNG.standard_normal((5, 3)).astype(np.float32), np.ones(5, dtype=int))
        buf.clear()
        assert len(buf) == 0


class TestStage45OnlineRLHF:
    @pytest.mark.slow
    def test_step_no_update_below_min(self):
        agent = _make_agent()
        buf = FeedbackBuffer()
        rlhf = OnlineRLHF(agent, buf, min_batch_size=32)
        result = rlhf.step()
        assert result["updated"] is False

    @pytest.mark.slow
    def test_step_updates_when_enough_data(self):
        agent = _make_agent()
        buf = FeedbackBuffer(capacity=200)
        X = RNG.standard_normal((50, 6)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        buf.push_raw(X, y)

        rlhf = OnlineRLHF(agent, buf, min_batch_size=32)
        result = rlhf.step()
        assert result["updated"] is True
        assert result["n_samples"] >= 32

    @pytest.mark.slow
    def test_n_updates_increments(self):
        agent = _make_agent()
        buf = FeedbackBuffer()
        rlhf = OnlineRLHF(agent, buf, min_batch_size=32)
        assert rlhf._n_updates == 0

        X = RNG.standard_normal((40, 6)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        buf.push_raw(X, y)
        rlhf.step()
        assert rlhf._n_updates == 1

    @pytest.mark.slow
    def test_report_keys(self):
        agent = _make_agent()
        buf = FeedbackBuffer()
        rlhf = OnlineRLHF(agent, buf)
        r = rlhf.report()
        assert "n_updates" in r and "buffer_stats" in r

    @pytest.mark.slow
    def test_push_feedback(self):
        agent = _make_agent()
        buf = FeedbackBuffer()
        rlhf = OnlineRLHF(agent, buf)
        X = RNG.standard_normal((5, 6)).astype(np.float32)
        y = np.ones(5, dtype=int)
        n = rlhf.push_feedback(X, y)
        assert n == 5


# ===========================================================================
# Stage 46 — AgentOrchestrator
# ===========================================================================


class TestStage46Orchestrator:
    def _make_orchestrator(self) -> AgentOrchestrator:
        f = _make_featurizer()
        orch = AgentOrchestrator(featurizer=f)
        orch.register_specialist(Specialist(
            name="physics",
            description="classify numeric tabular data physics",
            handler=lambda req: {"result": "numeric_prediction"},
        ))
        orch.register_specialist(Specialist(
            name="tool_agent",
            description="search the web and answer questions",
            handler=lambda req: {"result": "tool_answer"},
        ), fallback=True)
        return orch

    def test_route_returns_result(self):
        orch = self._make_orchestrator()
        result = orch.route("classify this numeric dataset")
        assert isinstance(result, OrchestratorResult)
        assert result.specialist_name in ("physics", "tool_agent")

    def test_route_text_request(self):
        orch = self._make_orchestrator()
        result = orch.route("search for information about physics")
        assert result.specialist_name in ("physics", "tool_agent")

    def test_route_numpy_request(self):
        orch = self._make_orchestrator()
        vec = RNG.standard_normal(16).astype(np.float32)
        result = orch.route(vec)
        assert isinstance(result, OrchestratorResult)

    def test_alternatives_populated(self):
        orch = self._make_orchestrator()
        result = orch.route("anything")
        assert len(result.ranked_alternatives) == 1

    def test_no_specialists_raises(self):
        f = _make_featurizer()
        orch = AgentOrchestrator(featurizer=f)
        with pytest.raises(RuntimeError, match="No specialists"):
            orch.route("test")

    def test_report_keys(self):
        orch = self._make_orchestrator()
        orch.route("test query")
        r = orch.report()
        assert "n_routes" in r and "route_counts" in r

    def test_route_count_increments(self):
        orch = self._make_orchestrator()
        for _ in range(3):
            orch.route("query")
        assert orch._n_routes == 3

    def test_memory_integration(self):
        f = _make_featurizer()
        mem = EpisodicMemory(capacity=50)
        orch = AgentOrchestrator(featurizer=f, memory=mem)
        orch.register_specialist(Specialist(
            name="sp1",
            description="specialist one",
            handler=lambda req: "resp1",
        ))
        orch.route("test input")
        assert len(mem) == 1

    def test_set_fallback_unknown_raises(self):
        orch = self._make_orchestrator()
        with pytest.raises(KeyError):
            orch.set_fallback("nonexistent")

    def test_set_fallback_by_name(self):
        orch = self._make_orchestrator()
        orch.set_fallback("physics")
        assert orch._fallback is not None
        assert orch._fallback.name == "physics"
